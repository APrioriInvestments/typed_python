import logging
from ldap3 import Server, Connection, ALL, NTLM


class AuthPluginBase:
    def authenticate(self, username, password) -> bool:
        return False


class PermissiveAuthPlugin(AuthPluginBase):
    def authenticate(self, username, password) -> bool:
        return True


class LdapAuthPlugin(AuthPluginBase):
    def __init__(self, hostname, base_dn, ntlm_domain=None, authorized_groups=None):
        self.hostname = hostname
        self.base_dn = base_dn
        self.ntlm_domain = ntlm_domain

        self.authorized_groups = ['CN={group}'.format(group=group) for group in authorized_groups]
        self._logger = logging.getLogger(__name__)

    def authenticate(self, username, password) -> bool:
        server = Server(self.hostname, use_ssl=True, get_info=ALL)
        if self.ntlm_domain:
            opts = dict(authentication=NTLM)
            ldap_username = self.ntlm_domain + '\\' + username
            username_key = 'sAMAccountName'
        else:
            opts = dict()
            ldap_username = 'CN={},'.format(username) + self.base_dn
            username_key = 'cn'

        with Connection(server, ldap_username, password, **opts) as conn:
            if not conn.bound:
                return False
            # else conn.bound==True
            if self.authorized_groups is None:
                return True
            # else check group membership
            if not conn.search(self.base_dn, '({username_key}={username})'.format(username_key=username_key, username=username), attributes='memberOf'):
                return False
            if len(conn.response) != 1:
                raise Exception(
                    "Non-unique LDAP username: {username}".format(username=username)
                )
            memberOfGroups = [group.split(',')[0] for group in conn.response[0]['attributes']['memberOf']]

            for group in memberOfGroups:
                if group in self.authorized_groups:
                    return True

            self._logger.debug(
                "User '{username}' authenticated successfully with LDAP "
                "but does not belong to an authorized group: {groups}"
                .format(username=username, groups=self.authorized_groups)
            )

            return False
