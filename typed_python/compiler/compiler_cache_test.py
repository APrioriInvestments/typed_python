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

import tempfile
import os
from typed_python.test_util import evaluateExprInFreshProcess

SOME_CODE = """
@Entrypoint
def f(x):
    return x + 1
"""


def test_compiler_cache_populates():
    with tempfile.TemporaryDirectory() as compilerCacheDir:
        assert evaluateExprInFreshProcess({'x.py': SOME_CODE}, 'x.f(10)', compilerCacheDir) == 11
        assert len(os.listdir(compilerCacheDir)) == 1

        assert evaluateExprInFreshProcess({'x.py': SOME_CODE}, 'x.f(10.5)', compilerCacheDir) == 11.5
        assert len(os.listdir(compilerCacheDir)) == 2

        assert evaluateExprInFreshProcess({'x.py': SOME_CODE}, 'x.f(11)', compilerCacheDir) == 12
        assert len(os.listdir(compilerCacheDir)) == 2
