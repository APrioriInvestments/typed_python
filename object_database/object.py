#   Coyright 2017-2019 Nativepython Authors
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


class Indexed:
    def __init__(self, obj):
        assert isinstance(obj, type)
        self.obj = obj


class Index:
    def __init__(self, *names):
        self.names = names

    def __call__(self, instance):
        return tuple(getattr(instance, x) for x in self.names)
