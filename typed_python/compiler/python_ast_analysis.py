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

"""utilities for analyzing python_ast objects."""

import collections

from typed_python.python_ast import (
    Module,
    Statement,
    Expr,
    Arg,
    NumericConstant,
    ExprContext,
    BooleanOp,
    BinaryOp,
    UnaryOp,
    ComparisonOp,
    Comprehension,
    ExceptionHandler,
    Arguments,
    Keyword,
    Alias,
    WithItem
)

nodeTypes = (
    Module,
    Statement,
    Expr,
    Arg,
    NumericConstant,
    ExprContext,
    BooleanOp,
    BinaryOp,
    UnaryOp,
    ComparisonOp,
    Comprehension,
    ExceptionHandler,
    Arguments,
    Keyword,
    Alias,
    WithItem
)


def visitPyAstChildren(node, callback):
    """Walk into 'node' calling 'callback' for every object.

    If callback returns 'False', stop recursing in this part of the tree.

    Args:
        node - any python_ast object, or a list/tuple of them, or a primitive.
        callback - a function accepting any kind of object and returning True
            if we wish to recurse more deeply into the object.
    """
    if isinstance(node, (int, float, str, bytes, bool, type(None))):
        return

    if isinstance(node, nodeTypes):
        if not callback(node):
            return

        if node.matches.Constant:
            return

        for name in node.ElementType.ElementNames:
            visitPyAstChildren(getattr(node, name), callback)
    else:
        for x in node:
            visitPyAstChildren(x, callback)


def computeAssignedVariables(astNode, includeImports=True):
    """Return a set of variable names that are assigned to by this ast node or children.

    Right now we don't handle the 'nonlocal' keyword correctly (or 'global' for that matter).
    We assume that interior function scopes can produce reads but not writes.

    Args:
        astNode - any python_ast object, or a list/tuple of them, or a primitive.
    """
    return set(computeVariablesAssignmentCounts(astNode, includeImports=includeImports).keys())


def computeVariablesAssignedOnlyOnce(astNode):
    assignmentCounts = computeVariablesAssignmentCounts(astNode)

    return set([varname for varname, count in assignmentCounts.items() if count == 1])


def computeVariablesAssignmentCounts(astNode, includeImports=True):
    """Count how many times each variable is assigned by this astNode in the current expression.

    Args:
        astNode - a Statement, Expression, or one of the types reachable from those.

    Returns:
        a Dict(str, int) with keys being variables and values being counts of assignments.
    """

    assignmentCounts = collections.defaultdict(int)

    def visit(x):
        if isinstance(x, Alias) and includeImports:
            if x.asname:
                assignmentCounts[x.asname] += 1
            else:
                assignmentCounts[x.name.split(".")[0]] += 1

        if isinstance(x, Expr):
            if x.matches.Name and (x.ctx.matches.Store or x.ctx.matches.Del or x.ctx.matches.AugStore):
                assignmentCounts[x.id] += 1

            if x.matches.Lambda:
                return False

            if (
                x.matches.ListComp
                or x.matches.SetComp
                or x.matches.DictComp
                or x.matches.GeneratorExp
            ):
                return False

        if isinstance(x, Statement):
            # we don't need to worry about all the individual operations because
            # we can catch the variable names from the Expr.Name context as they're used
            # in each place.

            if x.matches.FunctionDef:
                assignmentCounts[x.name] += 1
                # don't recurse into the body, since assignments there are not propagated
                # here. When we handle global/nonlocal, we'll need to modify this behavior
                return False

            if x.matches.ClassDef:
                assignmentCounts[x.name] += 1
                return False

            assertValidStatement(x)

        return True

    visitPyAstChildren(astNode, visit)

    return assignmentCounts


def computeVariablesReadByClosures(astNode, isInsideClassDef=False):
    """Determine the set of variables that are read from by deffed functions and lambdas."""
    closureVars = set()

    def visit(x):
        if isinstance(x, Statement):
            # we don't need to worry about all the individual operations because
            # we can catch the variable names from the Expr.Name context as they're used
            # in each place.

            if x.matches.FunctionDef:
                closureVars.update(computeReadVariables(x, isInsideClassDef))
                return False

            if x.matches.ClassDef:
                closureVars.update(computeVariablesReadByClosures(x.bases))
                closureVars.update(computeVariablesReadByClosures(x.keywords))
                closureVars.update(computeVariablesReadByClosures(x.decorator_list))
                for smt in x.body:
                    closureVars.update(computeReadVariables(x.body, True))
                return False

            assertValidStatement(x)

        if isinstance(x, Expr):
            if (
                x.matches.Lambda
                or x.matches.ListComp
                or x.matches.SetComp
                or x.matches.DictComp
                or x.matches.GeneratorExp
            ):
                closureVars.update(computeReadVariables(x))
                return False

        return True

    visitPyAstChildren(astNode, visit)

    return closureVars


def extractFunctionDefsInOrder(astNode):
    res = []

    def visit(x):
        if isinstance(x, Statement):
            if x.matches.FunctionDef or x.matches.ClassDef or x.matches.AsyncFunctionDef:
                res.append(x)
                return False

        if isinstance(x, Expr):
            if (
                x.matches.Lambda
                or x.matches.ListComp
                or x.matches.SetComp
                or x.matches.DictComp
                or x.matches.GeneratorExp
            ):
                res.append(x)
                return False

        return True

    visitPyAstChildren(astNode, visit)

    return res


def countYieldStatements(astNode):
    res = [0]

    def visit(x):
        if isinstance(x, Statement):
            if x.matches.FunctionDef or x.matches.ClassDef or x.matches.AsyncFunctionDef:
                return False

        if isinstance(x, Expr):
            if (
                x.matches.Lambda
                or x.matches.ListComp
                or x.matches.SetComp
                or x.matches.DictComp
                or x.matches.GeneratorExp
            ):
                return False

            if x.matches.Yield:
                res[0] += 1

        return True

    visitPyAstChildren(astNode, visit)

    return res[0]


def extractFunctionDefs(astNode):
    """Find all the direct 'def' operations that could produce closure elements.

    Returns:
        a list of Statement.FunctionDef instances
    """
    functionDefs = []
    assignedLambdas = []
    freeLambdas = []
    comprehensions = []
    generators = []

    def visit(x):
        if isinstance(x, Statement):
            # we don't need to worry about all the individual operations because
            # we can catch the variable names from the Expr.Name context as they're used
            # in each place.

            if x.matches.FunctionDef:
                functionDefs.append(x)
                return False

            if x.matches.Assign and len(x.targets) == 1 and x.targets[0].matches.Name and x.value.matches.Lambda:
                assignedLambdas.append((x.targets[0].id, x.value))
                return False

            if x.matches.ClassDef:
                return False

            assertValidStatement(x)

        if isinstance(x, Expr):
            if x.matches.Lambda:
                freeLambdas.append(x)
                return False

            if (
                x.matches.ListComp
                or x.matches.SetComp
                or x.matches.DictComp
            ):
                comprehensions.append(x)
                return False

            if x.matches.GeneratorExp:
                generators.append(x)
                return False

        return True

    visitPyAstChildren(astNode, visit)

    return (
        functionDefs,
        assignedLambdas,
        freeLambdas,
        comprehensions,
        generators
    )


def computeFunctionArgVariables(args: Arguments):
    """Compute the set of names bound by the given arguments.

    Returns:
        set(str)
    """
    names = set([arg.arg for arg in args.args])

    if args.vararg:
        names.add(args.vararg.arg)

    if args.kwarg:
        names.add(args.kwarg.arg)

    for arg in args.kwonlyargs:
        names.add(arg)

    return names


def computeMentionedConstants(astNode):
    """Return a set of inline constants that are mentioned by this code directly.

    These include inline strings, integers, floating point numbers, None, True, False,
    etc. This doesn't include constants within interior code objects.
    """
    constants = set()

    def visit(x):
        if isinstance(x, Expr):
            if x.matches.Str:
                constants.add(x.s)

            if x.matches.Bytes:
                constants.add(x.b)

            if x.matches.Constant:
                constants.add(x.value)

            if x.matches.Num:
                constants.add(x.n.value)

        if isinstance(x, Statement):
            if x.matches.FunctionDef:
                return False

            if x.matches.ClassDef:
                return False

        return True

    visitPyAstChildren(astNode, visit)

    return constants


def computeReadVariables(astNode, isInsideClassDef=False):
    """Return a set of variable names that are read from by this ast node or children.

    This doesn't have any masking behavior. E.g. something like

        x = 10
        return x

    reads 'x' despite the fact that it is not 'free' in 'x' because it reads from the slot
    that contains 'x'

    Args:
        astNode - the python_ast.Expr or Statement we're computing on.
        isInsideClassDef - true if this is a Statement within a 'class' definition
    """
    variables = set()

    def visit(x):
        if isinstance(x, Expr):
            if x.matches.Name and (x.ctx.matches.Load or x.ctx.matches.AugLoad):
                variables.add(x.id)

            if x.matches.Lambda:
                variables.update(
                    computeReadVariables(x.body).difference(
                        computeAssignedVariables(x.body).union(
                            computeFunctionArgVariables(x.args)
                        )
                    )
                )
                variables.update(computeReadVariables(x.args))
                return False

            if (
                x.matches.ListComp
                or x.matches.SetComp
                or x.matches.GeneratorExp
                or x.matches.DictComp
            ):
                masked = set()

                for comp in x.generators:
                    variables.update(computeReadVariables(comp.iter).difference(masked))

                    masked.update(
                        computeAssignedVariables(comp.target)
                    )

                    variables.update(computeReadVariables(comp.ifs).difference(masked))

                if x.matches.DictComp:
                    variables.update(computeReadVariables(x.key).difference(masked))
                    variables.update(computeReadVariables(x.value).difference(masked))
                else:
                    variables.update(computeReadVariables(x.elt).difference(masked))

                return False

        if isinstance(x, Statement):
            if x.matches.FunctionDef:
                variables.update(
                    # get the variables read by the body
                    computeReadVariables(x.body)
                    .union(computeReadVariables(x.returns))
                    # but remove anything bound by the function scope
                    .difference(
                        computeAssignedVariables(x.body)
                        .union(computeFunctionArgVariables(x.args))
                        # take out the function's name itself if we're defining
                        # a function anywhere where it might call itself recursively
                        # directly (e.g. not in a classdef)
                        .union(set([x.name] if not isInsideClassDef else []))
                    )
                )
                variables.update(computeReadVariables(x.decorator_list))

                variables.update(computeReadVariables(x.args.defaults))
                variables.update(computeReadVariables(x.args.kw_defaults))

                for a in (
                    list(x.args.args)
                    + list(x.args.kwonlyargs)
                    + ([x.args.vararg] if x.args.vararg else [])
                    + ([x.args.kwarg] if x.args.kwarg else [])
                ):
                    variables.update(computeReadVariables(a.annotation))

                return False
            if x.matches.ClassDef:
                variables.update(computeReadVariables(x.bases))
                variables.update(computeReadVariables(x.keywords))
                variables.update(computeReadVariables(x.decorator_list))
                for smt in x.body:
                    variables.update(computeReadVariables(smt, True))
                return False

            assertValidStatement(x)

        return True

    visitPyAstChildren(astNode, visit)

    return variables


def assertValidStatement(statement):
    """Complain about any python_ast.Statement objects we don't support."""

    if statement.matches.Nonlocal:
        raise NotImplementedError("Nonlocal keyword isn't supported yet.")
    if statement.matches.Global:
        raise NotImplementedError("Global keyword isn't supported yet.")
    if statement.matches.AsyncFunctionDef:
        raise NotImplementedError("AsyncFunctionDef isn't supported yet.")
    if statement.matches.AsyncWith:
        raise NotImplementedError("AsyncWith isn't supported yet.")
    if statement.matches.AsyncFor:
        raise NotImplementedError("AsyncFor isn't supported yet.")
