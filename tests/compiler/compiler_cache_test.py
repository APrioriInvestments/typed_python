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
import pytest
from typed_python.test_util import evaluateExprInFreshProcess

MAIN_MODULE = """
@Entrypoint
def f(x):
    return x + 1
"""


@pytest.mark.skipif('sys.platform=="darwin"')
def test_compiler_cache_populates():
    with tempfile.TemporaryDirectory() as compilerCacheDir:
        assert evaluateExprInFreshProcess({'x.py': MAIN_MODULE}, 'x.f(10)', compilerCacheDir) == 11
        assert len(os.listdir(compilerCacheDir)) == 1

        assert evaluateExprInFreshProcess({'x.py': MAIN_MODULE}, 'x.f(10.5)', compilerCacheDir) == 11.5
        assert len(os.listdir(compilerCacheDir)) == 2

        assert evaluateExprInFreshProcess({'x.py': MAIN_MODULE}, 'x.f(11)', compilerCacheDir) == 12
        assert len(os.listdir(compilerCacheDir)) == 2


@pytest.mark.skipif('sys.platform=="darwin"')
def test_compiler_cache_can_handle_conflicting_versions_of_the_same_code():
    with tempfile.TemporaryDirectory() as compilerCacheDir:
        assert evaluateExprInFreshProcess({'x.py': MAIN_MODULE}, 'x.f(10)', compilerCacheDir) == 11
        assert len(os.listdir(compilerCacheDir)) == 1

        assert evaluateExprInFreshProcess({'x.py': MAIN_MODULE.replace('1', '2')}, 'x.f(10)', compilerCacheDir) == 12
        assert len(os.listdir(compilerCacheDir)) == 2

        assert evaluateExprInFreshProcess({'x.py': MAIN_MODULE}, 'x.f(10)', compilerCacheDir) == 11
        assert len(os.listdir(compilerCacheDir)) == 2


@pytest.mark.skipif('sys.platform=="darwin"')
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


@pytest.mark.skipif('sys.platform=="darwin"')
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


@pytest.mark.skipif('sys.platform=="darwin"')
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


@pytest.mark.skipif('sys.platform=="darwin"')
def test_compiler_cache_handles_exceptions_properly():
    xmodule = "\n".join([
        "@Entrypoint",
        "def f(x):",
        "    raise Exception('boo')",
        "def g(x):",
        "    try:",
        "        f(x)",
        "    except Exception:",
        "        import traceback",
        "        return traceback.format_exc()"
    ])

    VERSION1 = {'x.py': xmodule}

    with tempfile.TemporaryDirectory() as compilerCacheDir:
        assert 'boo' in (evaluateExprInFreshProcess(VERSION1, 'x.g(1)', compilerCacheDir))
        assert 'boo' in (evaluateExprInFreshProcess(VERSION1, 'x.g(1)', compilerCacheDir))


@pytest.mark.skipif('sys.platform=="darwin"')
def test_compiler_cache_understands_granular_module_accesses():
    xmodule = "\n".join([
        "@Entrypoint",
        "def f(x):",
        "    return x + 1",
        "@Entrypoint",
        "def g(x):",
        "    return x * 100",
    ])
    ymodule = "\n".join([
        "import x",
        "@Entrypoint",
        "def g(arg):",
        "    return x.f(arg)",
    ])

    VERSION1 = {'x.py': xmodule, 'y.py': ymodule}
    VERSION2 = {'x.py': xmodule.replace("100", "200"), 'y.py': ymodule}

    with tempfile.TemporaryDirectory() as compilerCacheDir:
        assert evaluateExprInFreshProcess(VERSION1, 'y.g(1)', compilerCacheDir) == 2
        assert len(os.listdir(compilerCacheDir)) == 1

        # no recompilation necessary
        assert evaluateExprInFreshProcess(VERSION2, 'y.g(1)', compilerCacheDir) == 2
        assert len(os.listdir(compilerCacheDir)) == 1


@pytest.mark.skipif('sys.platform=="darwin"')
def test_load_dependent_modules():
    xmodule = "\n".join([
        "@Entrypoint",
        "def f(x):",
        "    return x + 1"
    ])

    xmodule_cont = "\n".join([
        "@Entrypoint",
        "def g(x):",
        "    return f(x)"
    ])

    VERSION1 = {'x.py': xmodule}
    VERSION2 = {'x.py': xmodule + "\n" + xmodule_cont}

    with tempfile.TemporaryDirectory() as compilerCacheDir:
        # add an item to the cache
        assert evaluateExprInFreshProcess(VERSION1, 'x.f(1)', compilerCacheDir) == 2
        assert len(os.listdir(compilerCacheDir)) == 1

        # add a dependent function
        assert evaluateExprInFreshProcess(VERSION2, 'x.g(1)', compilerCacheDir) == 2
        assert len(os.listdir(compilerCacheDir)) == 2

        # we should be able to load correctly
        assert evaluateExprInFreshProcess(VERSION2, 'x.g(1)', compilerCacheDir) == 2
        assert len(os.listdir(compilerCacheDir)) == 2


@pytest.mark.skipif('sys.platform=="darwin"')
def test_reference_existing_function_twice():
    xmodule = "\n".join([
        "@Entrypoint",
        "def f(x):",
        "    return x + 1"
    ])

    xmodule_cont = "\n".join([
        "@Entrypoint",
        "def g1(x):",
        "    return f(x)",
        "@Entrypoint",
        "def g2(x):",
        "    return f(x)",
        "@Entrypoint",
        "def g(x):",
        "    return g1(x) + g2(x)"
    ])

    VERSION1 = {'x.py': xmodule}
    VERSION2 = {'x.py': xmodule + "\n" + xmodule_cont}

    with tempfile.TemporaryDirectory() as compilerCacheDir:
        assert evaluateExprInFreshProcess(VERSION1, 'x.f(1)', compilerCacheDir) == 2
        assert len(os.listdir(compilerCacheDir)) == 1

        # add some content and nothing recompiles
        assert evaluateExprInFreshProcess(VERSION2, 'x.f(1)', compilerCacheDir) == 2
        assert len(os.listdir(compilerCacheDir)) == 1

        # recompiles with 'g1' and 'g2' referencing 'f'
        assert evaluateExprInFreshProcess(VERSION2, 'x.g(1)', compilerCacheDir) == 4
        assert len(os.listdir(compilerCacheDir)) == 2

        # can load it
        assert evaluateExprInFreshProcess(VERSION2, 'x.g(1)', compilerCacheDir) == 4
        assert len(os.listdir(compilerCacheDir)) == 2


@pytest.mark.skipif('sys.platform=="darwin"')
def test_compiler_cache_handles_class_destructors_correctly():
    xmodule = "\n".join([
        "class C(Class):",
        "    def __init__(self, x):",
        "        self.x=x",
        "    x = Member(int)",
        "@Entrypoint",
        "def f(x):",
        "    return C(x).x",
        "@Entrypoint",
        "def g(x):",
        "    return C(x).x",
    ])

    VERSION = {'x.py': xmodule}

    with tempfile.TemporaryDirectory() as compilerCacheDir:
        assert evaluateExprInFreshProcess(VERSION, 'x.f(1)', compilerCacheDir) == 1
        assert len(os.listdir(compilerCacheDir)) == 2

        # we can reuse the class destructor from the first time around
        assert evaluateExprInFreshProcess(VERSION, 'x.g(1)', compilerCacheDir) == 1
        assert len(os.listdir(compilerCacheDir)) == 3


@pytest.mark.skipif('sys.platform=="darwin"')
def test_compiler_cache_handles_classes():
    xmodule = "\n".join([
        "class C(Class):",
        "    def __init__(self, x):",
        "        self.x=x",
        "    x = Member(int)",
        "@Entrypoint",
        "def f(x):",
        "    return C(x).x",
        "@Entrypoint",
        "def g(x):",
        "    return C(x).x",
    ])

    VERSION = {'x.py': xmodule}

    with tempfile.TemporaryDirectory() as compilerCacheDir:
        assert evaluateExprInFreshProcess(VERSION, 'x.f(1)', compilerCacheDir) == 1
        assert len(os.listdir(compilerCacheDir)) == 2

        # we can reuse the class destructor from the first time around
        assert evaluateExprInFreshProcess(VERSION, 'x.g(1)', compilerCacheDir) == 1
        assert len(os.listdir(compilerCacheDir)) == 3


@pytest.mark.skipif('sys.platform=="darwin"')
def test_compiler_cache_handles_references_to_globals():
    xmodule = "\n".join([
        "aList = []",
        "@Entrypoint",
        "def f(x):",
        "    aList.append(x)",
        "    return aList",
    ])

    VERSION = {'x.py': xmodule}

    with tempfile.TemporaryDirectory() as compilerCacheDir:
        assert evaluateExprInFreshProcess(VERSION, 'x.f(1)', compilerCacheDir) == [1]
        assert len(os.listdir(compilerCacheDir)) == 1

        # we can reuse the class destructor from the first time around
        assert evaluateExprInFreshProcess(VERSION, '(x.f(1), x.aList)', compilerCacheDir) == ([1], [1])
        assert len(os.listdir(compilerCacheDir)) == 1


@pytest.mark.skipif('sys.platform=="darwin"')
def test_compiler_cache_handles_changed_types():
    xmodule1 = "\n".join([
        "@Entrypoint",
        "def f(x):",
        "    return x",
        "aList=[]",
        "@Entrypoint",
        "def g1(x):",
        "    return len(aList) + f(x)",
    ])

    xmodule2 = "\n".join([
        "@Entrypoint",
        "def f(x):",
        "    return x",
        "@Entrypoint",
        "def g2(x):",
        "    return f(x)",
    ])

    VERSION1 = {'x.py': xmodule1}
    VERSION2 = {'x.py': xmodule2}

    with tempfile.TemporaryDirectory() as compilerCacheDir:
        assert evaluateExprInFreshProcess(VERSION1, 'x.g1(1)', compilerCacheDir) == 1
        assert len(os.listdir(compilerCacheDir)) == 1

        # if we try to use 'f', it should work even though we no longer have
        # a defniition for 'g2'
        assert evaluateExprInFreshProcess(VERSION2, 'x.f(1)', compilerCacheDir) == 1
        assert len(os.listdir(compilerCacheDir)) == 2

        badCt = 0
        for subdir in os.listdir(compilerCacheDir):
            if 'marked_invalid' in os.listdir(os.path.join(compilerCacheDir, subdir)):
                badCt += 1

        assert badCt == 1


@pytest.mark.skipif('sys.platform=="darwin"')
def test_ordering_is_stable_under_code_change():
    # check that the order of functions in a MutuallyRecursiveTypeGroup is
    # stable even if we change the code underneath it.
    moduleText = "\n".join([
        "def f1(x):",
        "    return f2(x) + 1",
        "def f2(x):",
        "    return f3(x) + 1",
        "def f3(x):",
        "    return f1(x) + 1",
    ])

    names = evaluateExprInFreshProcess(
        {'x.py': moduleText},
        '[f.__name__ for f in recursiveTypeGroup(x.f1)]'
    )

    for _ in range(5):
        names2 = evaluateExprInFreshProcess(
            {'x.py': moduleText.replace(" + 1", "+ 2")},
            '[f.__name__ for f in recursiveTypeGroup(x.f1)]'
        )

        assert names == names2
