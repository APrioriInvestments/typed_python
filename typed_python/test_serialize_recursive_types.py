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


from typed_python.test_util import callFunctionInFreshProcess
from typed_python import (
    Alternative, SerializationContext, Forward, resolveForwardDefinedType, Function
)


s = SerializationContext()


def test_can_serialize_function():
    @Function
    def f(x: int):
        return x + 1

    assert type(f) is type(s.deserialize(s.serialize(f)))


def test_can_serialize_recursive_alternative():
    def makeA():
        A = Forward("A")
        A.define(
            Alternative(
                "A",
                x=dict(a=A),
                y=dict(),
                f=lambda self: self
            )
        )

        A = A.resolve()

        return A

    assert makeA() is makeA()
    assert makeA() is s.deserialize(s.serialize(makeA()))


def test_can_serialize_recursive_alternative_out_of_proc():
    def makeA():
        A = Forward("A")
        A.define(
            Alternative(
                "A",
                x=dict(a=A),
                y=dict(),
                f=lambda self: self
            )
        )

        A = A.resolve()

        return A

    A_external = callFunctionInFreshProcess(makeA)

    assert makeA() is A_external


def test_can_serialize_recursive_alternative_out_of_proc_with_ref_to_self():
    def makeA():
        A = Forward("A")
        A.define(
            Alternative(
                "A",
                x=dict(a=A),
                y=dict(),
                f=lambda self: A
            )
        )

        A = A.resolve()

        return A

    A_external = callFunctionInFreshProcess(makeA)

    assert makeA() is A_external


def test_can_serialize_recursive_alternative_out_of_proc_with_ref_to_own_forward():
    def makeA():
        A = Forward("A")
        A_fwd = A
        A.define(
            Alternative(
                "A",
                x=dict(a=A),
                y=dict(),
                f=lambda self: A,
                forward=lambda self: A_fwd
            )
        )

        A = A.resolve()

        return A

    A_external = callFunctionInFreshProcess(makeA)

    assert makeA() is A_external
