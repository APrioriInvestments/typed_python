#   Copyright 2017-2019 Nativepython Authors
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

from typed_python import sha_hash


def index_value_to_hash(value, serializationContext=None):
    if isinstance(value, int):
        return b"int_" + str(value).encode("utf8")
    if isinstance(value, str) and len(value) < 37:
        return b"str_" + str(value).encode("utf8")
    return b"hash_" + sha_hash(value, serializationContext).digest
