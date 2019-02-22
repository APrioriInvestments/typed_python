#   Copyright 2019 Nativepython Authors
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
#   limitations under the License.from collections import defaultdict

import unittest
from unittest.mock import patch

from object_database.web.AuthPlugin import LdapAuthPlugin


class LdapAuthPluginTest(unittest.TestCase):

    def test_ldap_plugin_authenticate_no_groups(self):
        ldap_auth = LdapAuthPlugin(
            "localhost", "bogus_base_dn", "bogus_ntlm_domain",
            authorized_groups=None
        )

        with patch.object(ldap_auth, 'getLdapConnection') as mock_getConn:
            with patch.object(ldap_auth, 'getUserAttribute') as mock_getUserAttr:
                error = ldap_auth.authenticate("McFly", "M@rty")

        self.assertEqual(error, '')
        mock_getConn.assert_called_once()
        self.assertEqual(mock_getUserAttr.call_count, 0)

    def test_ldap_plugin_authenticate_with_groups(self):
        ldap_auth = LdapAuthPlugin(
            "localhost", "bogus_base_dn", "bogus_ntlm_domain",
            authorized_groups=['Protagonists', 'Scientists']
        )

        # 1. test with users that belong to a valid group
        with patch.object(ldap_auth, 'getLdapConnection') as mock_getConn:
            with patch.object(ldap_auth,
                              'getUserAttribute',
                              return_value=['CN=Protagonists, OU=BackToDasFutuur',
                                            'CN=Stars, OU=BackToDasFutuur']) as mock_getUserAttr:
                error = ldap_auth.authenticate("McFly", "M@rty")

        self.assertEqual(error, '')
        mock_getConn.assert_called_once()
        mock_getUserAttr.assert_called_once()

        with patch.object(ldap_auth, 'getLdapConnection') as mock_getConn:
            with patch.object(ldap_auth,
                              'getUserAttribute',
                              return_value=['CN=SideKicks, OU=BackToDasFutuur',
                                            'CN=Scientists, OU=BackToDasFutuur']) as mock_getUserAttr:
                error = ldap_auth.authenticate("Brown", "Emmet, Dr.")

        self.assertEqual(error, '')
        mock_getConn.assert_called_once()
        mock_getUserAttr.assert_called_once()

        # 2. test with a user that does not belong to a valid group
        with patch.object(ldap_auth, 'getLdapConnection') as mock_getConn:
            with patch.object(ldap_auth,
                              'getUserAttribute',
                              return_value=['CN=Nemesis, OU=BackToDasFutuur',
                                            'CN=Brutes, OU=BackToDasFutuur']) as mock_getUserAttr:
                error = ldap_auth.authenticate("Tannen", "B1ff")

        self.assertNotEqual(error, '')
        mock_getConn.assert_called_once()
        mock_getUserAttr.assert_called_once()
