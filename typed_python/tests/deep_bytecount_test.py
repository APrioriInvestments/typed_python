#   Copyright 2017-2021 typed_python Authors
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

from typed_python import deepBytecount

from typed_python import (
    TupleOf, ListOf, Dict, Class, Member, NamedTuple, ConstDict
)


def test_deep_bytecount_listof():
    l = ListOf(int)()

    for sz in [10, 100, 1000, 10000]:
        l.resize(sz)
        assert sz * 8 <= deepBytecount(l) <= 8 * sz + 112


def test_deep_bytecount_listof_aliasing():
    NT = NamedTuple(a=ListOf(int), b=ListOf(int))

    l = ListOf(int)()

    x = NT(a=l, b=l)

    l.resize(1000)

    assert 1000 * 8 <= deepBytecount(x) <= 1000 * 8 + 112


def test_deep_bytecount_sees_into_basic_python_objects():
    l = ListOf(int)()
    l.resize(1000)

    assert deepBytecount(()) < 100
    assert deepBytecount((l,)) > 8000
    assert deepBytecount([l]) > 8000
    assert deepBytecount({TupleOf(int)(l)}) > 8000
    assert deepBytecount({10: l}) > 8000

    class C:
        def __init__(self, x):
            self.x = x

    assert deepBytecount(C(l)) > 8000

    assert deepBytecount((l, l, l)) < 9000


def test_deep_bytecount_sees_into_Class_objects():
    class C(Class):
        x = Member(ListOf(int))

    assert deepBytecount(C(x=ListOf(int)(range(1000)))) > 1000


def test_deep_bytecount_sees_into_Dict_objects():
    assert deepBytecount(Dict(int, ListOf(int))({1: ListOf(int)(range(1000))}))


def test_deep_bytecount_sees_into_ConstDict_objects():
    assert deepBytecount(ConstDict(int, ListOf(int))({1: ListOf(int)(range(1000))}))


def test_deep_bytecount_of_empty_constDict():
    assert deepBytecount(ConstDict(int, int)()) == 0
