#   Copyright 2017 Braxton Mckee
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

class Wrapper(object):
    is_pod = False

    def __init__(self, typeRepresentation):
        super().__init__()

        self.typeRepresentation = typeRepresentation

    def __eq__(self, other):
        if type(self) != type(other):
            return False

        return self.typeRepresentation == other.typeRepresentation

    def __hash__(self):
        return hash(self.typeRepresentation)

    def convert_assign(self, context, target, toStore):
        raise NotImplementedError()

    def convert_initialize_copy(self, context, target, toStore):
        raise NotImplementedError()

    def convert_destroy(self, context, instance):
        raise NotImplementedError()

    def convert_call(self, context, left, args):
        raise NotImplementedError()

    def toInt64(self, e):
        raise NotImplementedError()

    def toFloat64(self, e):
        raise NotImplementedError()

    def tooBool(self, e):
        raise NotImplementedError()
