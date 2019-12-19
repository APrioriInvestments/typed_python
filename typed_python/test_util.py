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

import psutil
import time
from typed_python import Entrypoint


def currentMemUsageMb(residentOnly=True):
    if residentOnly:
        return psutil.Process().memory_info().rss / 1024 ** 2
    else:
        return psutil.Process().memory_info().vms / 1024 ** 2


def compilerPerformanceComparison(f, *args, assertResultsEquivalent=True):
    """Call 'f' with args in entrypointed/unentrypointed form and benchmark

    If 'assertResultsEquivalent' check that the two results are '=='.

    Returns:
        (elapsedCompiled, elapsedUncompiled)
    """
    fEntrypointed = Entrypoint(f)
    fEntrypointed(*args)

    t0 = time.time()
    compiledRes = fEntrypointed(*args)
    t1 = time.time()
    uncompiledRes = f(*args)
    t2 = time.time()

    if assertResultsEquivalent:
        assert compiledRes == uncompiledRes, (compiledRes, uncompiledRes)

    return (t1 - t0, t2 - t1)
