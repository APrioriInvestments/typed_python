#   Copyright 2019 Nativepython Authors
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

import unittest
import time
import textwrap

from typed_python.Codebase import Codebase

from nativepython.codebase_compiler import CodebaseCompiler


class TestCodebaseCompiler(unittest.TestCase):
    def test_basic(self):
        codebase = Codebase.Instantiate({"a.py": textwrap.dedent("""
            from typed_python import Function, ListOf, OneOf

            @Function
            def g(x):
                return x+x

            @Function
            def f(x: float):
                y = 0
                while x > 0:
                    x -= 1
                    y += g(x)
                return y
            """)})

        f = codebase.getClassByName("a.f")

        t0 = time.time()
        compiledCodebase = CodebaseCompiler.compile(codebase)
        compilation_time = time.time() - t0

        t0 = time.time()
        f(100000)
        f_time_first = time.time() - t0

        t0 = time.time()
        compiledCodebase.install()
        install_time = time.time() - t0

        t0 = time.time()
        f(100000)
        f_time_second = time.time() - t0

        self.assertTrue(f_time_second < f_time_first * .1)
        self.assertTrue(install_time < compilation_time * 0.1)
