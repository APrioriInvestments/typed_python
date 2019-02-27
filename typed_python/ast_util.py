#   Copyright 2017 Braxton Mckee
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

import typed_python.inspect_override as inspect

import ast
import os
import textwrap

LINENO_ATTRIBUTE_NAME = 'lineno'


class CantGetSourceTextError(Exception):
    pass


class VisitDone(Exception):
    """Raise this exception to short-circuit the visitor once we're done
    searching."""
    pass


def areAstsIdentical(ast1, ast2):
    return ast.dump(ast1) == ast.dump(ast2)


_all_caches = []


def clearAllCaches():
    inspect.pathExistsOnDiskCache_.clear()
    for a in _all_caches:
        a.clear()


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


def getSourceText(pyObject):
    source, lineno = getSourceLines(pyObject)
    # Create a prefix of (lineno - 1) blank lines to keep track of line numbers
    # for error reporting
    blankLines = os.linesep * (lineno - 1)
    # We don't know how to avoid the use of `textwrap.dedent to get the code
    # though `ast.parse, which means that the computed column_numbers may be
    # off and we shouldn't report them.
    return textwrap.dedent(blankLines + "".join(source))


sourceFileCache_ = {}


def getSourceFilenameAndText(pyObject):
    try:
        sourceFile = inspect.getsourcefile(pyObject)
    except TypeError as e:
        raise CantGetSourceTextError(e.message)

    if sourceFile is None:
        raise CantGetSourceTextError(
            "can't get source lines for file %s" % sourceFile
        )

    linesOrNone = inspect.getlines(sourceFile)

    if linesOrNone is None:
        raise CantGetSourceTextError(
            "can't get source lines for file %s" % sourceFile
        )

    if sourceFile not in sourceFileCache_:
        sourceFileCache_[sourceFile] = "".join(linesOrNone)

    return sourceFileCache_[sourceFile], sourceFile


def getSourceLines(pyObject):
    try:
        tr = inspect.getsourcelines(pyObject)
    except (TypeError, IOError) as e:
        raise CantGetSourceTextError(str(e))
    return tr


@CachedByArgs
def pyAstFromText(text):
    return ast.parse(text)


def pyAstFor(pyObject):
    return pyAstFromText(getSourceText(pyObject))


@CachedByArgs
def getAstFromFilePath(filename):
    linesOrNone = inspect.getlines(filename)
    if linesOrNone is not None:
        return pyAstFromText("".join(linesOrNone))

    return None


class FindEnclosingFunctionVisitor(ast.NodeVisitor):
    """"Visitor used to find the enclosing function at a given line of code.

    The class method 'find' is the preferred API entry point."""

    def __init__(self, line):
        self.targetLine = line
        self.enclosingFunction = None
        self._currentFunction = None
        self._stash = []

    def generic_visit(self, node):
        if hasattr(node, LINENO_ATTRIBUTE_NAME):
            if node.lineno >= self.targetLine:
                self.enclosingFunction = self._currentFunction
                raise VisitDone
        super(FindEnclosingFunctionVisitor, self).generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.lineno > self.targetLine:
            raise VisitDone
        self._stash.append(self._currentFunction)
        self._currentFunction = node.name
        self.generic_visit(node)
        self._currentFunction = self._stash.pop()

    def find(self, node):
        if hasattr(node, LINENO_ATTRIBUTE_NAME):
            if node.lineno > self.targetLine:
                return None
        try:
            self.visit(node)
        except VisitDone:
            return self.enclosingFunction
        return None


def findEnclosingFunctionName(astNode, lineno):
    vis = FindEnclosingFunctionVisitor(lineno)
    return vis.find(astNode)


class _AtLineNumberVisitor(ast.NodeVisitor):
    """Collects various types of nodes occurring at a given line number."""

    def __init__(self, lineNumber):
        self.funcDefSubnodesAtLineNumber = []
        self.lambdaSubnodesAtLineNumber = []
        self.lineNumber = lineNumber

    def visit_FunctionDef(self, node):
        if node.lineno == self.lineNumber:
            self.funcDefSubnodesAtLineNumber.append(node)
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Lambda(self, node):
        if node.lineno == self.lineNumber:
            self.lambdaSubnodesAtLineNumber.append(node)
        ast.NodeVisitor.generic_visit(self, node)


@CachedByArgs
def functionDefOrLambdaAtLineNumber(sourceAst, lineNumber):
    visitor = _AtLineNumberVisitor(lineNumber)
    visitor.visit(sourceAst)

    subnodesAtLineNumber = (
        visitor.funcDefSubnodesAtLineNumber +
        visitor.lambdaSubnodesAtLineNumber
    )

    if len(subnodesAtLineNumber) == 0:
        raise CantGetSourceTextError(
            "can't find a function definition at line %s." % lineNumber
        )
    if len(subnodesAtLineNumber) > 1:
        raise CantGetSourceTextError(
            "can't find a unique function definition at line %s. Do you " +
            "have two lambdas on the same line?" % lineNumber
        )

    return subnodesAtLineNumber[0]
