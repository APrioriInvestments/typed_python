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
import time
import unittest

from bs4 import BeautifulSoup
from object_database.service_manager.ServiceManager import ServiceManager
from object_database.web.AuthPlugin import PermissiveAuthPlugin
from object_database.web.LoginPlugin import LoginIpPlugin, User
from object_database.web.ActiveWebServiceSchema import active_webservice_schema
from object_database.web.ActiveWebService import ActiveWebService

from object_database import core_schema, connect, service_schema
from object_database.util import configureLogging, genToken
from object_database.test_util import autoconfigure_and_start_service_manager
from typed_python.Codebase import Codebase as TypedPythonCodebase

ownDir = os.path.dirname(os.path.abspath(__file__))
ownName = os.path.basename(os.path.abspath(__file__))

DATABASE_SERVER_PORT = 8023

WEB_SERVER_PORT = 8025


class ActiveWebServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cleanupFn = lambda error=None: None

        configureLogging("aws_test")
        cls._logger = logging.getLogger(__name__)
        cls.login_config = dict(company_name="Testing Company")

    def configurableSetUp(self, hostname='localhost',
                          login_plugin_factory=None,  # default: LoginIpPlugin,
                          login_config=None,
                          auth_plugins=(None), module=None,
                          db_init_fun=None):

        self.base_url = "http://{host}:{port}".format(
            host=hostname,
            port=WEB_SERVER_PORT
        )
        login_plugin_factory = login_plugin_factory or LoginIpPlugin
        self.token = genToken()
        log_level = self._logger.getEffectiveLevel()
        loglevel_name = logging.getLevelName(log_level)

        self.server, self.cleanupFn = autoconfigure_and_start_service_manager(
            port=DATABASE_SERVER_PORT,
            auth_token=self.token,
            loglevel_name=loglevel_name,
            own_hostname=hostname,
            db_hostname=hostname
        )

        try:
            self.database = connect(hostname, DATABASE_SERVER_PORT, self.token, retry=True)
            self.database.subscribeToSchema(core_schema, service_schema, active_webservice_schema)
            if db_init_fun is not None:
                db_init_fun(self.database)

            codebase = None
            if module is not None and not module.__name__.startswith("object_database."):
                self.database.serializeFromModule(module)

                root_path = TypedPythonCodebase.rootlevelPathFromModule(module)

                tpcodebase = TypedPythonCodebase.FromRootlevelPath(root_path)

                with self.database.transaction():
                    codebase = service_schema.Codebase.createFromCodebase(tpcodebase)

            with self.database.transaction():
                service = ServiceManager.createOrUpdateService(ActiveWebService, "ActiveWebService", target_count=0)

            ActiveWebService.configureFromCommandline(
                self.database,
                service,
                [
                    '--port', str(WEB_SERVER_PORT),
                    '--host', hostname,
                    '--log-level', loglevel_name,
                ]
            )

            if login_config is None:
                login_config = self.login_config

            ActiveWebService.setLoginPlugin(
                self.database,
                service,
                login_plugin_factory,
                auth_plugins,
                codebase=codebase,
                config=login_config
            )

            with self.database.transaction():
                ServiceManager.startService("ActiveWebService", 1)

            self.waitUntilUp()
        except Exception:
            self.cleanupFn(error=True)
            raise

    def waitUntilUp(self, timeout=4.0):
        t0 = time.time()

        while time.time() - t0 < timeout:
            try:
                requests.get(self.base_url + "/status")
                return
            except Exception:
                time.sleep(.5)

        raise Exception("Webservice failed to come up after {} seconds.".format(timeout))

    def tearDown(self):
        self.cleanupFn()

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
        self.configurableSetUp(auth_plugins=[None])
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
        self.configurableSetUp(auth_plugins=[PermissiveAuthPlugin()])
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

    def test_web_service_login_ip(self):
        self.configurableSetUp(auth_plugins=[PermissiveAuthPlugin()])
        url = self.base_url + "/content/object_database.css"
        client = requests.Session()
        username = 'anonymous'

        # we can access our target page after login
        self.login(client, username)
        res = client.get(url)
        self.assertFalse(res.history)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.url, url)

        # change the login_ip of the logged-in user, then check we have to re-login
        with self.database.transaction():
            users = User.lookupAll(username=username)
            self.assertEqual(len(users), 1)
            user = users[0]
            user.login_ip = ""

        # we get redirected to login because the login_ip does not match
        res = requests.get(url)
        self.assertTrue(res.history)
        self.assertEqual(len(res.history), 1)
        self.assertEqual(res.status_code, 200)
        self.assertNotEqual(res.url, url)
        self.assertTrue('login' in res.url)
