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

import logging
import time

from flask_login import current_user, login_user, logout_user
from flask import flash, redirect, render_template, url_for

from typed_python import Float64
from object_database.web.ActiveWebServiceSchema import active_webservice_schema
from object_database.web.flask_util import request_ip_address, next_url
from object_database.view import revisionConflictRetry
from object_database import Indexed

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired


class FlaskUser:
    """ User class that implements to the flask-login User API.

        We make these objects from our ObjectDB User classes so that Flask can
        handle them without having to hold a view into ObjectDB
    """
    @staticmethod
    def makeFromUser(user):
        if user is None:
            return None
        else:
            return FlaskUser(user.username, user.login_expiration, user.login_ip)

    def __init__(self, username, login_expiration, login_ip):
        self.username = username
        self.login_expiration = login_expiration
        self.login_ip = login_ip

    @property
    def is_authenticated(self) -> bool:
        if time.time() >= self.login_expiration:
            return False
        elif request_ip_address() != self.login_ip:
            return False
        else:
            return True

    @property
    def is_active(self) -> bool:
        return True

    @property
    def is_anonymous(self):
        return True if self.username.lower() == 'anonymous' else False

    def get_id(self) -> str:
        return self.username  # must return unicode by Python 3 strings are unicode


class LoginPluginInterface:
    """ Interface for a class that implements a login flow for a Flask app.

        The derived class will register `/login` and `/logout` endpoints with
        the Flask app and it will provide a load_user method.
    """
    REQUIRED_KEYS = None

    def __init__(self, object_db, auth_plugins, config=None):
        raise NotImplementedError("derived class must implement this method")

    def init_app(self, flask_app):
        raise NotImplementedError("derived class must implement this method")

    def getSerializationContext(self):
        raise NotImplementedError("derived class must implement this method")

    def load_user(self, username) -> FlaskUser:
        raise NotImplementedError("derived class must implement this method")

    @property
    def authorized_groups(self):
        raise NotImplementedError("derived class must implement this method")

    def init_config(self, config):
        if config is None:
            config = {}

        if self.REQUIRED_KEYS:
            for key in self.REQUIRED_KEYS:
                if key not in config:
                    raise Exception(
                        "{cls} missing configuration parameter '{key}'"
                        .format(cls=self.__class__.__name__, key=key)
                    )
                setattr(self, '_' + key, config[key])


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')


USER_LOGIN_DURATION = 24 * 60 * 60  # 24 hours


@active_webservice_schema.define
class User:
    username = Indexed(str)
    login_expiration = Float64
    login_ip = str

    def login(self, login_ip):
        self.login_expiration = time.time() + USER_LOGIN_DURATION
        self.login_ip = login_ip

    def logout(self):
        self.login_expiration = 0.0
        self.login_ip = ""


def authorized_groups_text(authorized_groups, default_text='All') -> str:
    """ Helper function that returns a string for display purposes. """
    res = default_text
    if authorized_groups:
        res = ', '.join(authorized_groups)
    return res


class LoginIpPlugin(LoginPluginInterface):
    REQUIRED_KEYS = ['company_name']

    def __init__(self, db, auth_plugins, config=None):
        self._logger = logging.getLogger(__name__)
        self._db = db
        self._db.subscribeToType(User)

        if len(auth_plugins) != 1:
            raise Exception(
                "LoginIpPlugin requires exactly 1 auth_plugin but {} were given."
                .format(len(auth_plugins))
            )

        self._auth_plugins = auth_plugins
        self._auth_plugin = auth_plugins[0]

        self._authorized_groups_text = authorized_groups_text(self.authorized_groups)

        self.init_config(config)

    def init_app(self, flask_app):
        flask_app.add_url_rule('/login', endpoint=None, view_func=self.login, methods=['GET', 'POST'])
        flask_app.add_url_rule('/logout', endpoint=None, view_func=self.logout)

    @property
    def authorized_groups_text(self):
        return self._authorized_groups_text

    @property
    def authorized_groups(self):
        return self._auth_plugin.authorized_groups if self._auth_plugin is not None else None

    def _authenticate(self, username, password) -> str:
        """ Attempts to authenticate with given username and password.

            Returns:
            --------
            str
                "" (empty string) if no error occurred and an error message otherwise
        """
        login_ip = request_ip_address()
        self._logger.info(f"User '{username}' trying to authenticate from IP {login_ip}")
        # error = self.authenticate(username, password, login_ip=login_ip)
        if not self.bypassAuth:
            error = self._auth_plugin.authenticate(username, password)
            if error:
                return error

        self._login_user(username, login_ip)
        return ''

    def login(self):
        if current_user.is_authenticated:
            return redirect(next_url())

        if self.bypassAuth:
            error = self._authenticate('anonymous', 'fake-pass')
            assert not error, error
            return redirect(next_url())

        form = LoginForm()

        def render(error=None):
            if error:
                flash(error, 'danger')
            return render_template(
                'login.html',
                form=form,
                title=self._company_name,
                authorized_groups_text=self.authorized_groups_text
            )

        if form.validate_on_submit():
            username = form.username.data
            password = form.password.data

            error = self._authenticate(username, password)
            if error:
                return render(error)
            else:
                return redirect(next_url())

        return render(error=form.errors)

    def logout(self):
        current_username = current_user.username
        self._logout_user(current_username)
        return redirect(url_for('index'))  # FIXME: we are assuming the app has defined the 'index' endpoint

    def load_user(self, username):
        with self._db.view():
            return FlaskUser.makeFromUser(User.lookupAny(username=username))

    @property
    def bypassAuth(self):
        return self._auth_plugin is None

    def _login_user(self, username, login_ip):
        self._login_objdb_user(username, login_ip)
        self._login_flask_user(username)

    @revisionConflictRetry
    def _login_objdb_user(self, username, login_ip):
        with self._db.transaction():
            users = User.lookupAll(username=username)

            if len(users) == 0:
                user = User(username=username)
            elif len(users) == 1:
                user = users[0]
            elif len(users) > 1:
                raise Exception("multiple users found with username={}".format(username))
            else:
                raise Exception("This should never happen: len(users)={}".format(len(users)))

            user.login(login_ip)

    def _login_flask_user(self, username):
        flask_user = self.load_user(username)
        login_user(flask_user)

    def _logout_user(self, username):
        self._logout_objdb_user(username)
        self._logout_flask_user()

    @revisionConflictRetry
    def _logout_objdb_user(self, username):
        with self._db.transaction():
            user = User.lookupAny(username=username)

            if user is not None:
                user.logout()

    def _logout_flask_user(self):
        logout_user()
