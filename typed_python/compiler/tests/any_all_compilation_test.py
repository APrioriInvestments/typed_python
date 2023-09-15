#   Copyright 2017-2023 typed_python Authors
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

from typed_python import Entrypoint, TupleOf


@pytest.mark.group_one
def test_compiles_any_and_all():
    @Entrypoint
    def callAny(x):
        return any(x)

    @Entrypoint
    def callAll(x):
        return all(x)

    assert callAll.resultTypeFor(list).typeRepresentation is bool
    assert callAny.resultTypeFor(list).typeRepresentation is bool

    for T in [list, TupleOf(int)]:
        assert callAny(T([1, 2, 3])) == any(T([1, 2, 3]))
        assert callAll(T([1, 2, 3])) == all(T([1, 2, 3]))
        assert callAny(T([0, 0, 0])) == any(T([0, 0, 0]))
        assert callAll(T([0, 0, 0])) == all(T([0, 0, 0]))
