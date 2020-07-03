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

MAIN_MODULE = """
@Entrypoint
def f(x):
    return x + 1
"""


def test_compiler_cache_populates():
    with tempfile.TemporaryDirectory() as compilerCacheDir:
        assert evaluateExprInFreshProcess({'x.py': MAIN_MODULE}, 'x.f(10)', compilerCacheDir) == 11
        assert len(os.listdir(compilerCacheDir)) == 1

        assert evaluateExprInFreshProcess({'x.py': MAIN_MODULE}, 'x.f(10.5)', compilerCacheDir) == 11.5
        assert len(os.listdir(compilerCacheDir)) == 2

        assert evaluateExprInFreshProcess({'x.py': MAIN_MODULE}, 'x.f(11)', compilerCacheDir) == 12
        assert len(os.listdir(compilerCacheDir)) == 2


def test_compiler_cache_can_handle_conflicting_versions_of_the_same_code():
    with tempfile.TemporaryDirectory() as compilerCacheDir:
        assert evaluateExprInFreshProcess({'x.py': MAIN_MODULE}, 'x.f(10)', compilerCacheDir) == 11
        assert len(os.listdir(compilerCacheDir)) == 1

        assert evaluateExprInFreshProcess({'x.py': MAIN_MODULE.replace('1', '2')}, 'x.f(10)', compilerCacheDir) == 12
        assert len(os.listdir(compilerCacheDir)) == 2

        assert evaluateExprInFreshProcess({'x.py': MAIN_MODULE}, 'x.f(10)', compilerCacheDir) == 11
        assert len(os.listdir(compilerCacheDir)) == 2


def test_compiler_cache_can_detect_invalidation_through_modules():
    xmodule = "\n".join([
        "def f(x):",
        "    return x + 1",
    ])
    ymodule = "\n".join([
        "from x import f",
        "@Entrypoint",
        "def g(x):",
        "    return f(x)",
    ])

    VERSION1 = {'x.py': xmodule, 'y.py': ymodule}
    VERSION2 = {'x.py': xmodule.replace('1', '2'), 'y.py': ymodule}

    with tempfile.TemporaryDirectory() as compilerCacheDir:
        assert evaluateExprInFreshProcess(VERSION1, 'y.g(10)', compilerCacheDir) == 11
        assert len(os.listdir(compilerCacheDir)) == 1

        assert evaluateExprInFreshProcess(VERSION2, 'y.g(10)', compilerCacheDir) == 12
        assert len(os.listdir(compilerCacheDir)) == 2

        assert evaluateExprInFreshProcess(VERSION1, 'y.g(10)', compilerCacheDir) == 11
        assert len(os.listdir(compilerCacheDir)) == 2


def test_compiler_cache_robust_to_irrelevant_module_changes():
    xmodule = "\n".join([
        "# this is a comment",
        "def f(x):",
        "    return x + 1",
    ])
    ymodule = "\n".join([
        "from x import f",
        "@Entrypoint",
        "def g(x):",
        "    return f(x)",
    ])

    VERSION1 = {'x.py': xmodule, 'y.py': ymodule}
    VERSION2 = {'x.py': xmodule.replace("this is a comment", "this comment is different"), 'y.py': ymodule}

    with tempfile.TemporaryDirectory() as compilerCacheDir:
        assert evaluateExprInFreshProcess(VERSION1, 'y.g(10)', compilerCacheDir) == 11
        assert len(os.listdir(compilerCacheDir)) == 1

        assert evaluateExprInFreshProcess(VERSION2, 'y.g(10)', compilerCacheDir) == 11
        assert len(os.listdir(compilerCacheDir)) == 1


def test_compiler_cache_understands_type_changes():
    xmodule = "\n".join([
        "G = Dict(int, int)({1: 2})",
        "def f(x):",
        "    return G[x]",
    ])
    ymodule = "\n".join([
        "from x import f",
        "@Entrypoint",
        "def g(x):",
        "    return f(x)",
    ])

    VERSION1 = {'x.py': xmodule, 'y.py': ymodule}
    VERSION2 = {'x.py': xmodule.replace("1: 2", "1: 3"), 'y.py': ymodule}
    VERSION3 = {'x.py': xmodule.replace("int, int", "int, float").replace('1: 2', '1: 2.5'), 'y.py': ymodule}

    assert '1: 3' in VERSION2['x.py']

    with tempfile.TemporaryDirectory() as compilerCacheDir:
        assert evaluateExprInFreshProcess(VERSION1, 'y.g(1)', compilerCacheDir) == 2
        assert len(os.listdir(compilerCacheDir)) == 1

        # no recompilation necessary
        assert evaluateExprInFreshProcess(VERSION2, 'y.g(1)', compilerCacheDir) == 3
        assert len(os.listdir(compilerCacheDir)) == 1

        # this forces a recompile
        assert evaluateExprInFreshProcess(VERSION3, 'y.g(1)', compilerCacheDir) == 2.5
        assert len(os.listdir(compilerCacheDir)) == 2
