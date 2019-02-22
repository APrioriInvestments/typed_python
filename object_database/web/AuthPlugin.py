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

import ldap3
import logging


class AuthPluginBase:
    def authenticate(self, username, password) -> str:
        """ Tries to authenticate with given username and password.

            Returns:
            --------
            str
                "" if no error occurred, and an error message otherwise
        """
        raise NotImplementedError("derived class must implement this method")

    @property
    def authorized_groups(self):
        return None


class PermissiveAuthPlugin(AuthPluginBase):
    " An AuthPlugin that allows anyone to login (useful for testing)"

    def authenticate(self, username, password) -> str:
        return ''


class LdapAuthPlugin(AuthPluginBase):
    def __init__(self, hostname, base_dn, ntlm_domain=None, authorized_groups=None):
        self._hostname = hostname
        self._base_dn = base_dn
        self._ntlm_domain = ntlm_domain
        self._authorized_groups = authorized_groups

    @property
    def authorized_groups(self):
        return self._authorized_groups

    @property
    def _logger(self):
        """ _logger is a property rather than being set in __init__ to allow plugin
            objects to live in the Object Database
        """
        return logging.getLogger(__name__)

    @property
    def username_key(self):
        return 'sAMAccountName' if self._ntlm_domain else 'cn'

    def getLdapConnection(self, username, password):
        server = ldap3.Server(self._hostname, use_ssl=True, get_info=ldap3.ALL)
        if self._ntlm_domain:
            opts = dict(authentication=ldap3.NTLM)
            ldap_username = self._ntlm_domain + '\\' + username
        else:
            opts = dict()
            ldap_username = 'CN={},'.format(username) + self._base_dn

        return ldap3.Connection(server, ldap_username, password, **opts)

    def getUserAttribute(self, connection, username: str, attributeName: str):
        if not connection.search(
                self._base_dn,
                '({username_key}={username})'.format(username_key=self.username_key, username=username),
                attributes=attributeName
        ):
            return None

        if len(connection.response) != 1:
            raise Exception(
                "Non-unique LDAP username: {username}".format(username=username)
            )

        return connection.response[0]['attributes'][attributeName]

    def authenticate(self, username, password) -> str:
        with self.getLdapConnection(username, password) as conn:
            if not conn.bound:
                return "Invalid username or password"
            # else conn.bound==True -> auth to LDAP succeeded
            if self._authorized_groups is None:
                return ""
            # else check group membership
            memberOf = self.getUserAttribute(conn, username, 'memberOf')

            if memberOf is None:
                return "User '{}' is not a member of any group".format(username)

        # keep the first item of a comma-separated list and then keep the
        # last part after an equal-sign. This is because the memberOf attribute
        # returns a sequence of list like this "CN=GroupName, OU=..., ..."
        memberOfGroups = [group.split(',')[0].split('=')[-1] for group in memberOf]

        for group in memberOfGroups:
            if group in self._authorized_groups:
                return ''

        self._logger.debug(
            "User '{username}' authenticated successfully with LDAP "
            "but does not belong to an authorized group: {groups}"
            .format(username=username, groups=self._authorized_groups)
        )

        return "User '{}' does not belong to an authorized group".format(username)
