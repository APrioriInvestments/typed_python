#   Copyright 2017-2020 typed_python Authors
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

from typed_python import Entrypoint, ListOf, NotCompiled, _types


def test_string_to_int_error_catchable():
    @Entrypoint
    def f(x):
        try:
            int(x)
        except ValueError:
            return "CAUGHT"
        return "NOT CAUGHT"

    assert f(b"3.0") == "CAUGHT"
    assert f("3.0") == "CAUGHT"


def test_string_to_float_error_catchable():
    @Entrypoint
    def f(x):
        try:
            float(x)
        except ValueError:
            return "CAUGHT"
        return "NOT CAUGHT"

    assert f(b"asdf") == "CAUGHT"
    assert f("asdf") == "CAUGHT"


def test_mod_zero_catchable():
    @Entrypoint
    def f(x):
        try:
            1.0 % x
        except ZeroDivisionError:
            return "CAUGHT"
        return "NOT CAUGHT"

    assert f(0.0) == "CAUGHT"


def test_throwing_exceptions_from_C_code_triggers_destructors():
    @NotCompiled
    def throws():
        raise Exception("an exception")

    @Entrypoint
    def f(x):
        increfsIt = x[0] # noqa

        int("asdf")

    aList = ListOf(int)()
    aListOfList = ListOf(ListOf(int))([aList])

    assert _types.refcount(aList) == 2

    try:
        f(aListOfList)
    except Exception:
        pass

    assert _types.refcount(aList) == 2


def test_throwing_exceptions_from_uncompiled_code_triggers_destructors():
    @NotCompiled
    def throws():
        raise Exception("an exception")

    @Entrypoint
    def f(x):
        increfsIt = x[0] # noqa

        int("asdf")

    aList = ListOf(int)()
    aListOfList = ListOf(ListOf(int))([aList])

    assert _types.refcount(aList) == 2

    try:
        f(aListOfList)
    except Exception:
        pass

    assert _types.refcount(aList) == 2


def test_exceptions_from_not_compiled():
    @NotCompiled
    def g0(x):
        return 1/x

    def f0(x):
        try:
            g0(x)
        except ZeroDivisionError:
            return "caught"
        return "ok"

    assert f0(0.0) == "caught"
