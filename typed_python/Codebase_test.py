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

import unittest
import typed_python

from typed_python.Codebase import Codebase


class CodebaseTest(unittest.TestCase):
    def test_instantiated_codebase(self):
        codebase = Codebase.FromFileMap({
            'codebase_test_test_module/__init__.py': '',
            'codebase_test_test_module/inner.py': 'f = lambda: 10',
        })
        codebase.instantiate()
        codebase.instantiate()

        self.assertEqual(codebase.getClassByName('codebase_test_test_module.inner.f')(), 10)

        codebase2 = Codebase.FromRootlevelModule(codebase.getModuleByName("codebase_test_test_module"))

        self.assertTrue(codebase2.isInstantiated())
        self.assertEqual(codebase2.getClassByName('codebase_test_test_module.inner.f')(), 10)

        self.assertEqual(codebase.filesToContents, codebase2.filesToContents)

        vals = list(codebase.allModuleLevelValues())
        vals = [v[0] for v in vals if "__" not in v[0]]
        self.assertEqual(vals, ['codebase_test_test_module.inner', 'codebase_test_test_module.inner.f'])

        codebaseAlternativeCode = Codebase.FromFileMap(
            {'codebase_test_test_module/__init__.py': ""}
        )
        with self.assertRaisesRegex(Exception, "Module codebase_test_test_module is"):
            codebaseAlternativeCode.instantiate()

    def test_grab_native_codebase(self):
        codebase = Codebase.FromRootlevelModule(typed_python)

        assert codebase.isInstantiated()
