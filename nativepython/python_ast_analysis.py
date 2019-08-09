#   Coyright 2017-2019 Nativepython Authors
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

from typed_python.python_ast import (
    Module,
    Statement,
    Expr,
    Arg,
    NumericConstant,
    ExprContext,
    Slice,
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
    Slice,
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

        for name in node.ElementType.ElementNames:
            visitPyAstChildren(getattr(node, name), callback)
    else:
        for x in node:
            visitPyAstChildren(x, callback)


def computeAssignedVariables(astNode):
    """Return a set of variable names that are assigned to by this ast node or children.

    Right now we don't handle the 'nonlocal' keyword correctly (or 'global' for that matter).
    We assume that interior function scopes can produce reads but not writes.

    Args:
        astNode - any python_ast object, or a list/tuple of them, or a primitive.
    """
    variables = set()

    def visit(x):
        if isinstance(x, Alias):
            variables.add(x)
        if isinstance(x, Expr):
            if x.matches.Name and (x.ctx.matches.Store or x.ctx.matches.Del or x.ctx.matches.AugStore):
                variables.add(x.id)

            if x.matches.Lambda:
                return False

        if isinstance(x, Statement):
            # we don't need to worry about all the individual operations because
            # we can catch the variable names from the Expr.Name context as they're used
            # in each place.

            if x.matches.FunctionDef:
                variables.add(x.name)
                # don't recurse into the body, since assignments there are not propagated
                # here. When we handle global/nonlocal, we'll need to modify this behavior
                return False
            if x.matches.ClassDef:
                variables.add(x.name)
                return False
            if x.matches.Nonlocal:
                raise NotImplementedError("Nonlocal keyword isn't supported yet.")
            if x.matches.Global:
                raise NotImplementedError("Global keyword isn't supported yet.")
            if x.matches.AsyncFunctionDef:
                raise NotImplementedError("AsyncFunctionDef isn't supported yet.")
            if x.matches.AsyncWith:
                raise NotImplementedError("AsyncWith isn't supported yet.")
            if x.matches.AsyncFor:
                raise NotImplementedError("AsyncFor isn't supported yet.")

        return True

    visitPyAstChildren(astNode, visit)

    return variables


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


def computeReadVariables(astNode):
    """Return a set of variable names that are read from by this ast node or children."""
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
                        .union(set([x.name]))  # take out the function's name itself
                    )
                )
                variables.update(computeReadVariables(x.decorator_list))

                return False
            if x.matches.ClassDef:
                variables.update(computeReadVariables(x.bases))
                variables.update(computeReadVariables(x.keywords))
                variables.update(computeReadVariables(x.decorator_list))
                variables.update(computeReadVariables(x.body))
                return False
            if x.matches.Nonlocal:
                raise NotImplementedError("Nonlocal keyword isn't supported yet.")
            if x.matches.Global:
                raise NotImplementedError("Global keyword isn't supported yet.")
            if x.matches.AsyncFunctionDef:
                raise NotImplementedError("AsyncFunctionDef isn't supported yet.")
            if x.matches.AsyncWith:
                raise NotImplementedError("AsyncWith isn't supported yet.")
            if x.matches.AsyncFor:
                raise NotImplementedError("AsyncFor isn't supported yet.")

        return True

    visitPyAstChildren(astNode, visit)

    return variables
