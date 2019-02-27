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
from unittest.mock import Mock

from object_database.web.AuthPlugin import PermissiveAuthPlugin
from object_database.web.LoginPlugin import LoginIpPlugin


class LoginIpPluginTest(unittest.TestCase):
    def test_login_plugin_init(self):
        db = Mock()
        login_config = dict(company_name="Testing Company")

        with self.assertRaisesRegex(Exception, "missing config"):
            lp = LoginIpPlugin(db, [None], {})

        with self.assertRaisesRegex(Exception, "LoginIpPlugin requires exactly 1"):
            LoginIpPlugin(db, [None, None], login_config)

        lp = LoginIpPlugin(db, [None], login_config)
        self.assertTrue(lp.bypassAuth)
        lp.authorized_groups_text
        self.assertIs(lp.authorized_groups, None)

        lp = LoginIpPlugin(db, [PermissiveAuthPlugin()], login_config)
        lp.authorized_groups_text
        self.assertIs(lp.authorized_groups, None)
