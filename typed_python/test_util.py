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

import random
import networkx as nx
import psutil
import time
import threading
import tempfile
import pickle
import subprocess
import sys
import os

from typed_python import sha_hash
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


def callFunctionInFreshProcess(func, argTup, compilerCacheDir=None, showStdout=False, extraEnvs={}):
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

        env.update(extraEnvs)

        sc = SerializationContext()

        with open(os.path.join(tf, "input"), "wb") as f:
            f.write(sc.serialize((func, argTup)))

        try:
            output = subprocess.check_output(
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
            raise Exception(
                "Subprocess failed:\n\n"
                + e.stdout.decode("ASCII")
                + "\n\nerr=\n"
                + e.stderr.decode("ASCII")
            )

        if showStdout:
            print(output.decode("ASCII"))

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
                        f"import typed_python;"
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


class CodeEvaluator:
    """Make a temporary directory and use it to evaluate code snippets.

    Usage:
        c = CodeEvaluator()
        m = {}

        c.evaluateInto("def f():\n\treturn 10", m)
        assert m['f']() == 10

    This is better than 'exec' because the code is backed by a file, so we
    can find the text and use various TP functions on it.
    """
    def __init__(self):
        self.dir = tempfile.TemporaryDirectory()

    def evaluateInto(self, code, moduleDict):
        filename = os.path.abspath(
            os.path.join(self.dir.name, sha_hash(code).hexdigest)
        )

        try:
            os.makedirs(os.path.dirname(filename))
        except OSError:
            pass

        with open(filename, "wb") as codeFile:
            codeFile.write(code.encode("utf8"))

        exec(compile(code, filename, "exec"), moduleDict)


class CodeGenerator:
    def __init__(self, num_functions: int, num_connections: int,
                 max_depth: int):
        self.num_functions = num_functions
        self.num_connections = num_connections
        self._graph = None
        self._python_code = None
        self.function_args = {}
        self.max_depth = max_depth
        self.supported_types = ['int', 'float', 'str', 'bool', 'None', 'list']

    def __str__(self):
        return self.python_code

    @property
    def python_code(self):
        if not self._python_code:
            self._python_code = self.generate_python_code()
        return self._python_code

    @property
    def graph(self):
        if not self._graph:
            self._graph = self.generate_graph()
        return self._graph

    def generate_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        nodes = list(range(self.num_functions))
        G.add_nodes_from(nodes)
        random.shuffle(nodes)

        loop_counter = 0
        added_edges = 0
        while added_edges <= self.num_connections:
            if loop_counter > 1000:
                raise RuntimeError(
                    'params for graph generation are too restrictive')
            src = random.choice(nodes)
            dst_candidates = list(
                filter(lambda x: x != src and not nx.has_path(G, x, src),
                       nodes))
            if dst_candidates:
                dst = random.choice(dst_candidates)
                G.add_edge(src, dst)
                if nx.dag_longest_path_length(G) > self.max_depth:
                    G.remove_edge(src, dst)
                else:
                    added_edges += 1
            else:
                break

            loop_counter += 1

        return G

    def generate_value(self, var_type):
        value = None
        if var_type not in self.supported_types:
            raise ValueError(f"var_type must be one of {self.supported_types}")
        if var_type == 'int':
            value = random.randint(1, 100)
        elif var_type == 'float':
            value = round(random.uniform(1, 100), 2)
        elif var_type == 'str':
            value = f"'{random.choice(['typed', 'python', 'fuzz'])}'"
        elif var_type == 'bool':
            value = random.choice(['True', 'False'])
        elif var_type == 'None':
            value = 'None'
        elif var_type == 'list':
            value = f"[{random.randint(1, 100)}, {random.randint(1, 100)}, {random.randint(1, 100)}]"
        assert value is not None
        return str(value)

    def generate_var_declaration(self, node, var_type):
        var_name = f"var_{node}"
        value = self.generate_value(var_type)
        return f"{var_name} = {value}"

    def generate_function_body(self, node, dependencies):
        args = [
            f"arg_{i}: {arg_type}"
            for i, arg_type in enumerate(self.function_args[node])
        ]
        signature = f"def func_{node}({', '.join(args)}):"
        var_declarations = [
            self.generate_var_declaration(i,
                                          random.choice(self.supported_types))
            for i in range(random.randint(1, 3))
        ]
        func_sigs = []
        for dep in dependencies:
            func_sig = f"func_{dep}({','.join(self.generate_value(var) for var in self.function_args[dep])})"
            func_sigs.append(func_sig)
        func_calls = [
            f"var_{i+len(var_declarations)} = {sig}"
            for i, sig in enumerate(func_sigs)
        ]
        statements = var_declarations + func_calls
        random.shuffle(statements)
        # return_var = f"var_{random.randrange(0, len(statements))}"
        var_list = '[' + ', '.join(f"var_{i}"
                                   for i in range(len(statements))) + ']'
        statements.append(f"return random.choice({var_list})")
        body = '\n    '.join(statements)
        return f"{signature}\n    {body}\n"

    def generate_python_code(self) -> str:
        G = self.graph
        # need to have specified all function signatures before any body can be generated.
        for node in G.nodes:
            args = [
                random.choice(self.supported_types)
                for _ in range(random.randint(0, 3))
            ]
            self.function_args[node] = args

        codebase = [
            self.generate_function_body(node, G.successors(node))
            for node in G.nodes
        ]
        # all functions depend on random for their return value
        codebase_with_import = 'import random\n' + '\n'.join(codebase)
        return codebase_with_import
