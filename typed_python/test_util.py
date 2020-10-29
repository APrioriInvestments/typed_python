#   Copyright 2017-2020 typed_python Authors
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
import threading
import tempfile
import pickle
import subprocess
import sys
import os

from typed_python import Entrypoint, SerializationContext


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


def estimateFunctionMultithreadSlowdown(f, threadcount=2):
    t0 = time.time()
    f()
    t1 = time.time()

    threads = [threading.Thread(target=f) for _ in range(threadcount)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    t2 = time.time()

    return (t2 - t1) / (t1 - t0)


def instantiateFiles(filesToWrite, tf):
    """Write out a dict of files to a temporary directory.

    Args:
        filesToWrite - a dict from filename to file contents. Don't try to use
            subdirectories yet - it won't be cross platform.
        tf - the temporary directory to write into
    """
    for fname, contents in filesToWrite.items():
        fullname = os.path.join(tf, fname)
        dirname = os.path.dirname(fullname)

        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with open(fullname, "w") as f:
            f.write(
                "from typed_python import *\n"
                + contents
            )


def callFunctionInFreshProcess(func, argTup, compilerCacheDir=None):
    """Return the value of a function evaluated on some arguments in a subprocess.

    We use this to test the semantics of anonymous functions and classes in a process
    that didn't create those obects.

    Args:
        func - the function object to call
        argTup - a tuple of arguments

    Returns:
        the result of the expression.
    """
    with tempfile.TemporaryDirectory() as tf:
        env = dict(os.environ)

        if compilerCacheDir:
            env["TP_COMPILER_CACHE"] = compilerCacheDir

        sc = SerializationContext()

        with open(os.path.join(tf, "input"), "wb") as f:
            f.write(sc.serialize((func, argTup)))

        try:
            subprocess.check_output(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "from typed_python import SerializationContext\n"
                    "sc = SerializationContext()\n"
                    "with open('input', 'rb') as f:\n"
                    "    func, argTup = sc.deserialize(f.read())\n"
                    "with open('output', 'wb') as f:\n"
                    "    f.write(sc.serialize(func(*argTup)))\n"
                ],
                cwd=tf,
                env=env,
                stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            raise Exception("Subprocess failed:\n\n" + e.stdout.decode("ASCII") + "\n\nerr=\n" + e.stderr.decode("ASCII"))

        with open(os.path.join(tf, "output"), "rb") as f:
            result = sc.deserialize(f.read())

        return result


def evaluateExprInFreshProcess(filesToWrite, expression, compilerCacheDir=None, printComments=False):
    """Return the value of an expression evaluated in a subprocess.

    We use this to test using typed_python in codebases other than the main
    typed_python codebase, so that we can see what happens when some code itself
    changes underneath us.

    The value of the expression must be picklable, and shouldn't depend on
    any of the code in 'filesToWrite', since it won't make sense in the calling
    module.

    Args:
        filesToWrite = a dictionary from filename to the actual file contents to write.
        expression - the expression to evaluate. You should assume that we've imported
            all the modules given in 'filesToWrite', as well as everything
            from typed_python.

    Returns:
        the result of the expression.

    Example:
        evaluateExprInFreshProcess({'M.py': "x = 10"}, "M.x")
    """
    with tempfile.TemporaryDirectory() as tf:
        instantiateFiles(filesToWrite, tf)

        namesToImport = [
            fname[:-3].replace("/", ".") for fname in filesToWrite if '__init__' not in fname
        ]

        env = dict(os.environ)

        if compilerCacheDir:
            env["TP_COMPILER_CACHE"] = compilerCacheDir

        try:
            output = subprocess.check_output(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "".join(f"import {modname};" for modname in namesToImport) + (
                        f"import pickle;"
                        f"from typed_python._types import identityHash, recursiveTypeGroup;"
                        f"from typed_python import *;"
                        f"print(repr(pickle.dumps({expression})))"
                    )
                ],
                cwd=tf,
                env=env,
                stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            raise Exception("Subprocess failed:\n\n" + e.stdout.decode("ASCII") + "\n\nerr=\n" + e.stderr.decode("ASCII"))

        def isBytes(x):
            return x.startswith(b"b'") or x.startswith(b'b"')

        comments = [x for x in output.split(b"\n") if not isBytes(x) and x]
        result = b'\n'.join([x for x in output.split(b"\n") if isBytes(x)])

        if comments and printComments:
            print("GOT COMMENTS:\n", "\n".join(["\t" + x.decode("ASCII") for x in comments]))

        try:
            # we're returning a 'repr' of a bytes object. the 'eval'
            # turns it back into a python bytes object so we can compare it.
            return pickle.loads(eval(result))
        except Exception:
            raise Exception("Failed to understand output:\n" + output.decode("ASCII"))
