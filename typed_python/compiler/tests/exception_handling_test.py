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

import pytest
import traceback

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

    @Entrypoint
    def f0(x):
        try:
            g0(x)
        except ZeroDivisionError:
            return "caught"
        return "ok"

    assert f0(0.0) == "caught"


def test_can_rethrow_in_compiled_code():
    def throws():
        raise Exception("something to throw")

    aList = ListOf(str)()

    @Entrypoint
    def f():
        try:
            throws()
        except Exception as e:
            aList.append(str(e))
            raise

    with pytest.raises(Exception):
        f()

    assert aList == ["something to throw"]


def test_raise_non_exception_in_compiled_code():
    @Entrypoint
    def throw(x):
        raise x

    with pytest.raises(TypeError):
        throw(10)

    with pytest.raises(TypeError):
        throw(None)


def test_can_capture_exception_and_rethrow():
    def throws():
        raise Exception("From 'throws'")

    def g():
        try:
            throws()
        except Exception as e:
            return e

    def rethrow(tup):
        raise tup

    def f():
        tup = g()
        rethrow(tup)

    def getStringTraceback(toCall):
        stringTb = None

        try:
            toCall()
        except Exception:
            stringTb = traceback.format_exc()

        return stringTb

    # make sure we can see 'g', which is where this came from. This is
    # just how python works - when you raise an Exception with an existing
    # traceback, python just keeps adding on to it
    stringTb = getStringTraceback(f)
    assert 'in g' in stringTb

    # verify the compiler is the same
    assert getStringTraceback(f) == getStringTraceback(Entrypoint(f))


def test_catch_and_return_none():
    def blah(x):
        if x:
            raise Exception("boo")
        return "a", "b"

    @Entrypoint
    def trySplit(x):
        try:
            a, b = x.split("_")
            return a + b
        except Exception:
            return None

    assert trySplit("a_b") == "ab"
    assert trySplit("a_b_c") is None
