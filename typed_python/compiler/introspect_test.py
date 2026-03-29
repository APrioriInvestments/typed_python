#   Copyright 2017-2019 typed_python Authors
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
import unittest

from typed_python import Entrypoint, ListOf
from typed_python.compiler import introspect


class TestRuntime(unittest.TestCase):

    # the compilation itself is handled in other tests.
    def naive_sum(someList, startingInt):
        for x in someList:
            startingInt += x
        return startingInt

    compiled = Entrypoint(naive_sum)

    def test_ir_compile_and_output_string(self):
        for test_func in [introspect.getNativeIRString, introspect.getLLVMString]:
            output_text = test_func(
                TestRuntime.compiled, args=[ListOf(int), int], kwargs=None
            )
            assert 'naive_sum' in output_text

    def test_ir_throw_error_if_uncompiled(self):
        for test_func in [introspect.getNativeIRString, introspect.getLLVMString]:
            with pytest.raises(AssertionError):
                _ = test_func(
                    TestRuntime.naive_sum, args=[ListOf(int), int], kwargs=None)

    def test_introspect_handles_tp_class(self):
        """Full TP class"""
        pass

    def test_introspect_distinguishes_overloads(self):
        """Check that we can obtain the correct IRs given multiple overloads."""
        for test_func in [introspect.getNativeIRString, introspect.getLLVMString]:
            overload_one = test_func(
                TestRuntime.compiled, args=[ListOf(int), int], kwargs=None)
            overload_two = test_func(
                TestRuntime.compiled, args=[ListOf(float), float], kwargs=None)
            assert overload_one != overload_two

    def test_introspect_handles_kwargs_correctly_(self):
        for test_func in [introspect.getNativeIRString, introspect.getLLVMString]:
            output_text = test_func(
                TestRuntime.compiled, args=[ListOf(int)], kwargs={'startingInt': int}
            )
            assert 'naive_sum' in output_text

    def test_introspect_rejects_invalid_args(self):
        for test_func in [introspect.getNativeIRString, introspect.getLLVMString]:
            with pytest.raises(ValueError):
                _ = test_func(TestRuntime.compiled, args=[ListOf(int), int, int])
            with pytest.raises(ValueError):
                _ = test_func(TestRuntime.compiled, args=[ListOf(int)], kwargs={'test': int})
