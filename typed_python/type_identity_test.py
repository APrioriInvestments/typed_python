#   Copyright 2020 typed_python Authors
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

from typed_python import (
    UInt64, UInt32,
    ListOf, TupleOf, Tuple, NamedTuple, Dict, OneOf, Forward, identityHash
)


def test_identity_of_register_types():
    assert isinstance(identityHash(UInt64), bytes)
    assert len(identityHash(UInt64)) == 20

    assert identityHash(UInt64) != identityHash(UInt32)


def test_identity_of_list_of():
    assert identityHash(ListOf(int)) != identityHash(ListOf(float))
    assert identityHash(ListOf(int)) == identityHash(ListOf(int))
    assert identityHash(ListOf(int)) != identityHash(TupleOf(int))


def test_identity_of_named_tuple_and_tuple():
    assert identityHash(NamedTuple(x=int)) != identityHash(NamedTuple(x=float))
    assert identityHash(NamedTuple(x=int)) == identityHash(NamedTuple(x=int))
    assert identityHash(NamedTuple(x=int)) != identityHash(Tuple(float))

    assert identityHash(NamedTuple(x=int)) != identityHash(NamedTuple(y=int))
    assert identityHash(NamedTuple(x=int, y=float)) != identityHash(NamedTuple(y=float, x=int))


def test_identity_of_dict():
    assert identityHash(Dict(int, float)) != identityHash(Dict(int, int))
    assert identityHash(Dict(int, float)) != identityHash(Dict(float, int))


def test_identity_of_oneof():
    assert identityHash(OneOf(None, int)) != identityHash(OneOf(None, float))


def test_identity_of_recursive_types():
    X = Forward("X")
    X = X.define(TupleOf(OneOf(int, X)))

    X2 = Forward("X")
    X2 = X2.define(TupleOf(OneOf(int, X2)))

    X3 = Forward("X")
    X3 = X3.define(TupleOf(OneOf(float, X3)))

    assert identityHash(X2) == identityHash(X)
    assert identityHash(X3) != identityHash(X)


def test_identity_of_recursive_types_2():
    X = Forward("X")
    X = X.define(TupleOf(OneOf(int, TupleOf(X))))

    identityHash(X)


def test_identity_of_recursive_types_produced_same_way():
    def make(name, T):
        X = Forward(name)
        return X.define(TupleOf(OneOf(T, X)))

    assert identityHash(make("X", int)) == identityHash(make("X", int))
    assert identityHash(make("X", int)) != identityHash(make("X", float))
    assert identityHash(make("X", int)) != identityHash(make("X2", int))
