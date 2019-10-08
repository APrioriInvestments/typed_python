#   Copyright 2019 typed_python Authors
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

import os
import unittest
import sys

from typed_python.Codebase import Codebase


class CodebaseTest(unittest.TestCase):
    def test_instantiated_codebase(self):
        codebase = Codebase.Instantiate({
            'test_module/__init__.py': '',
            'test_module/inner.py': 'f = lambda: 10',
        })
        self.assertEqual(codebase.getClassByName('test_module.inner.f')(), 10)

        codebase2 = Codebase._FromModule(codebase.getModuleByName("test_module"))
        self.assertEqual(codebase2.getClassByName('test_module.inner.f')(), 10)

        self.assertEqual(codebase.filesToContents, codebase2.filesToContents)

    def test_rootlevelPathFromModule(self):

        def check_module_name(mod_name):
            mod = sys.modules[mod_name]
            path = Codebase.rootlevelPathFromModule(mod)
            self.assertEqual(os.path.basename(path), 'typed_python')

        mod_name_parts = Codebase.__module__.split('.')

        for parts in range(len(mod_name_parts)):
            check_module_name(
                '.'.join(mod_name_parts[:parts+1])
            )
