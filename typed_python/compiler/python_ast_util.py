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

import ast
import os
import linecache

_pathExistsOnDiskCache = {}
_linesCache = {}
_all_caches = [_linesCache, _pathExistsOnDiskCache]


def CachedByArgs(f):
    """Function decorator that adds a simple memo to 'f' on its arguments"""
    cache = {}
    _all_caches.append(cache)

    def inner(*args, **kwargs):
        keys = sorted(kwargs)
        all_args = args + tuple((k, kwargs[k]) for k in keys)
        if (all_args) not in cache:
            cache[all_args] = f(*args, **kwargs)
        return cache[all_args]

    return inner


def getLines(path):
    """return a list of lines for a given path"""
    if path in _linesCache:
        return _linesCache[path]

    if path not in _pathExistsOnDiskCache:
        _pathExistsOnDiskCache[path] = os.path.exists(path)

    if _pathExistsOnDiskCache[path]:
        with open(path, "r") as f:
            _linesCache[path] = f.readlines()
        return _linesCache[path]

    if path[:1] == '<' and path[-1:] == '>':
        return linecache.getlines(path)

    return None


def clearAllCaches():
    for a in _all_caches:
        a.clear()


@CachedByArgs
def pyAstFromText(text):
    return ast.parse(text)


def pyAstForCode(codeObject):
    """Given a code object that represents a function or lambda, get the ast for its code.

    This returns a python-native ast (from the 'ast' builtin module).

    Don't try to run this on code objects from classes or generators.
    """
    sourceFile = codeObject.co_filename
    sourceLines = getLines(sourceFile)
    if sourceLines is None:
        raise Exception(f"Couldn't get lines for source file {sourceFile} in {codeObject}")
    sourceText = "".join(sourceLines)

    wholeFileAst = pyAstFromText(sourceText)

    defs = functionDefsOrLambdaAtLineNumber(sourceText, wholeFileAst, codeObject.co_firstlineno)

    if len(defs) == 0:
        raise Exception(
            f"Error: can't convert {codeObject} back to a python_ast because when we"
            f" look, there are no defs or lambas at "
            f"{codeObject.co_filename}:{codeObject.co_firstlineno}"
        )

    if len(defs) == 1:
        return defs[0]

    if codeObject.co_name != "<lambda>":
        defs = [d for d in defs if not isinstance(d, ast.Lambda)]

        if len(defs) == 1:
            return defs[0]
    else:
        defs = [d for d in defs if isinstance(d, ast.Lambda)]

        if len(defs) == 1:
            return defs[0]

    for d in defs:
        assert isinstance(d, ast.Lambda), [type(x) for x in defs]

    # for each Lambda, convert it to an Expression object, compile it to a code
    # object that evaluates to that lambda, and pull out the constant 'code' object
    # which is what the lambda would bind. This is equivalent to the code object
    # we're interested in now.
    codes = [compile(ast.Expression(d), sourceFile, 'eval').co_consts[0] for d in defs]

    def unlikeObjectKey(x):
        return (type(x).__name__, x)

    def codeKey(code):
        return (
            code.co_code,
            namesInCodeObject(code),
            tuple(sorted([x for x in code.co_consts if x is not None], key=unlikeObjectKey))
        )

    # keep track of the first index of each unique piece of code we see
    # since we want to filter them out
    uniqueCodeIxs = {}
    for ix, c in enumerate(codes):
        if codeKey(c) not in uniqueCodeIxs:
            uniqueCodeIxs[codeKey(c)] = ix

    # try to disambiguate between the different asts by looking at their constants
    # and variables
    possibleDefs = [
        d for ix, d in enumerate(defs)
        if isPossibleAstFor(codes[ix], codeObject)
        and uniqueCodeIxs[codeKey(codes[ix])] == ix
    ]

    if not possibleDefs:
        raise Exception(
            f"Somehow, we filtered out all the possible function defs for "
            f"{codeObject.co_filename}:{codeObject.co_firstlineno}"
        )

    if len(possibleDefs) == 1:
        return possibleDefs[0]

    raise Exception(
        f"Error: can't convert {codeObject} back to an AST because when we"
        f" look, there are multiple code objects that have the same line number."
    )


def isPossibleAstFor(candidateCodeObject, codeObject):
    if set(candidateCodeObject.co_consts) - set([None]) != set(codeObject.co_consts) - set([None]):
        return False

    if namesInCodeObject(candidateCodeObject) != namesInCodeObject(codeObject):
        return False

    return True


def namesInCodeObject(codeObj):
    return tuple(sorted(
        list(codeObj.co_freevars)
        + list(codeObj.co_varnames)
        + list(codeObj.co_cellvars)
        + list(codeObj.co_names)
    ))


class AtLineNumberVisitor(ast.NodeVisitor):
    """Collects various types of nodes occurring at a given line number."""

    def __init__(self, lineNumber):
        self.funcDefSubnodesAtLineNumber = []
        self.lambdaSubnodesAtLineNumber = []
        self.lineNumber = lineNumber

    def visit_FunctionDef(self, node):
        if node.lineno == self.lineNumber:
            self.funcDefSubnodesAtLineNumber.append(node)
        else:
            for d in node.decorator_list:
                if d.lineno == self.lineNumber:
                    self.funcDefSubnodesAtLineNumber.append(node)
                    break

        ast.NodeVisitor.generic_visit(self, node)

    def visit_AsyncFunctionDef(self, node):
        if node.lineno == self.lineNumber:
            self.funcDefSubnodesAtLineNumber.append(node)
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Lambda(self, node):
        if node.lineno == self.lineNumber:
            self.lambdaSubnodesAtLineNumber.append(node)
        ast.NodeVisitor.generic_visit(self, node)


@CachedByArgs
def functionDefsOrLambdaAtLineNumber(sourceText, sourceAst, lineNumber):
    visitor = AtLineNumberVisitor(lineNumber)
    visitor.visit(sourceAst)

    return (
        visitor.funcDefSubnodesAtLineNumber +
        visitor.lambdaSubnodesAtLineNumber
    )
