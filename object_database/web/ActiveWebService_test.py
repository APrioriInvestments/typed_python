#   Copyright 2018 Braxton Mckee
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import logging
import os
import requests
import subprocess
import sys
import tempfile
import time
import unittest
import websockets

from bs4 import BeautifulSoup
from object_database.service_manager.ServiceManager import ServiceManager
from object_database.web.ActiveWebService import (
    active_webservice_schema,
    ActiveWebService,
    User
)

from object_database import core_schema, connect, service_schema
from object_database.util import genToken, configureLogging

ownDir = os.path.dirname(os.path.abspath(__file__))
ownName = os.path.basename(os.path.abspath(__file__))

DATABASE_SERVER_PORT=8023

WEB_SERVER_PORT=8025


class ActiveWebServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://localhost:{port}".format(port=WEB_SERVER_PORT)
        configureLogging("aws_test")
        cls.logger = logging.getLogger(__name__)

    def configurableSetUp(self, auth_type="LDAP",
                          auth_hostname=None, authorized_groups=(),
                          ldap_base_dn=None, ldap_ntlm_domain=None,
                          company_name=None):
        self.token = genToken()
        self.tempDirObj = tempfile.TemporaryDirectory()
        self.tempDirectoryName = self.tempDirObj.__enter__()
        log_level = logging.getLogger(__name__).getEffectiveLevel()
        log_level_name = logging.getLevelName(log_level)

        self.server = subprocess.Popen(
            [sys.executable, os.path.join(ownDir, '..', 'frontends', 'service_manager.py'),
                'localhost', 'localhost', str(DATABASE_SERVER_PORT), '--run_db',
                '--source', os.path.join(self.tempDirectoryName,'source'),
                '--storage', os.path.join(self.tempDirectoryName,'storage'),
                '--service-token', self.token,
                '--shutdownTimeout', '.5',
                '--log-level', log_level_name
            ]
        )
        try:
            # this should throw a subprocess.TimeoutExpired exception if the service did not crash
            self.server.wait(0.7)
        except subprocess.TimeoutExpired:
            pass
        else:
            raise Exception("Failed to start service_manager (retcode:{})"
                .format(self.server.returncode)
            )

        try:
            self.database = connect("localhost", DATABASE_SERVER_PORT, self.token, retry=True)

            self.database.subscribeToSchema(core_schema, service_schema, active_webservice_schema)

            with self.database.transaction():
                service = ServiceManager.createOrUpdateService(ActiveWebService, "ActiveWebService", target_count=0)

            optional_args = []
            if len(authorized_groups) > 0:
                optional_args.extend(['--authorized-groups', *authorized_groups])

            if auth_hostname:
                optional_args.extend(['--auth-hostname', auth_hostname])

            if ldap_base_dn:
                optional_args.extend(['--ldap-base-dn', ldap_base_dn])

            if ldap_ntlm_domain:
                optional_args.extend(['--ldap-ntlm-domain', ldap_ntlm_domain])

            if company_name:
                optional_args.extend(['--company-name', company_name])

            ActiveWebService.configureFromCommandline(
                self.database,
                service,
                [
                    '--port', str(WEB_SERVER_PORT),
                    '--host', 'localhost',
                    '--log-level', log_level_name,
                    '--auth', auth_type
                ] + optional_args
            )

            with self.database.transaction():
                ServiceManager.startService("ActiveWebService", 1)

            self.waitUntilUp()
        except Exception:
            self.server.terminate()
            self.server.wait()
            raise

    def waitUntilUp(self, timeout = 2.0):
        t0 = time.time()

        while time.time() - t0 < timeout:
            try:
                res = requests.get(self.base_url + "/login")
                return
            except Exception:
                time.sleep(.5)

        raise Exception("Webservice never came up.")

    def tearDown(self):
        self.server.terminate()
        self.server.wait()
        self.tempDirObj.__exit__(None, None, None)

    def login(self, client, username='anonymous', password='bogus'):
        # Because of CSRF security we need to do the following to authenticate:
        # - Load the login page
        # - Extract the csrf token (using BeautifulSoup)
        # - Issue a POST request to the login endpoint that includes the CSRF token
        login_url = self.base_url + "/login"
        res = client.get(login_url)
        self.assertFalse(res.history)
        self.assertEqual(res.status_code, 200)

        soup = BeautifulSoup(res.text, 'html.parser')
        csrf_token = soup.find('input', dict(name='csrf_token'))['value']

        res = client.post(login_url, data=dict(username=username, password=password, csrf_token=csrf_token))
        self.assertTrue(res.history)
        self.assertEqual(res.status_code, 200)
        self.assertTrue('login' not in res.url)

    def test_web_service_no_auth(self):
        self.configurableSetUp(auth_type="NONE")
        url = self.base_url + "/content/object_database.css"
        client = requests.Session()

        res = client.get(url)
        self.assertTrue(res.history)  # first time around we WILL get redirects
        self.assertEqual(res.status_code, 200)

        res = client.get(url)
        self.assertFalse(res.history)  # second time around we will NOT get redirects
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.url, url)

    def test_web_service_login_and_access(self):
        self.configurableSetUp(auth_type="PERMISSIVE")
        url = self.base_url + "/content/object_database.css"
        client = requests.Session()
        username = 'anonymous'

        # 1. Cannot access without login
        res = requests.get(url)

        self.assertTrue(res.history)
        self.assertEqual(len(res.history), 1)
        self.assertEqual(res.status_code, 200)
        self.assertNotEqual(res.url, url)
        self.assertTrue('login' in res.url)

        # 2. login successfully
        self.login(client, username)

        # 3. now we can access our target page
        res = client.get(url)
        self.assertFalse(res.history)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.url, url)

        # 4. test that we get auto-logged-out by modifying the user in object DB
        with self.database.transaction():
            user = User.lookupAny(username=username)
            if user:
                user.logout()
        res = client.get(url)
        self.assertTrue(res.history)
        self.assertEqual(res.status_code, 200)
        self.assertTrue('login' in res.url)

