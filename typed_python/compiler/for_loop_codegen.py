"""Utilities for rewriting for loops based on the types we've already inferred in a function.

basically

    for pat in expr:
        body
    else:
        orelse

becomes

    iterator = expr.__iter__()
    triggerElse = False

    while True:
        iterPtr = iterator.__fastnext__()
        if not iterPtr:
            triggerElse = True
            break

        pat = iterPtr.get()

        body

    if triggerElse:
        orelse

We rely on the main compiler to generate an appropriate '__fastnext__'
for objects that don't have __fastnext__ defined by calling __next__,
catching any StopIteration exceptions, and stashing the pointer
appropriately.

We also check the type of 'iterator', and if its a OneOf, we
introduce a sequence of isinstance branches (equivalent to a
checkOneOfType call)
"""
import typed_python.python_ast as python_ast
from typed_python.compiler.codegen_helpers import (
    const,
    branch,
    readVar,
    binOp,
    assign,
    compare,
    attr,
    makeCallExpr,
)


def rewriteForLoops(statement):
    """Walk over a set of statements, rewriting any For loops as while loops.

    We do this so that when we do type inference in function expansion we can get
    appropriate types on the iterator itself. This also simplifies rewriting
    the code for generators.

    Args:
        a list of python_ast.Statement objects

    Returns:
        a list of Statement objects
    """
    res = []

    for s in statement:
        for sub in rewriteForLoopStatement(s):
            res.append(sub)

    return res


def rewriteForLoopStatement(s):
    if (
        s.matches.Expr
        or s.matches.Assign
        or s.matches.AugAssign
        or s.matches.AnnAssign
        or s.matches.Pass
        or s.matches.Break
        or s.matches.Continue
        or s.matches.Assert
        or s.matches.Raise
        or s.matches.Return
        or s.matches.Import
        or s.matches.ImportFrom
        or s.matches.Delete
        or s.matches.FunctionDef
        or s.matches.ClassDef
        or s.matches.Global
        or s.matches.AsyncFunctionDef
        or s.matches.AsyncWith
        or s.matches.AsyncFor
        or s.matches.NonLocal
    ):
        yield s
        return

    if s.matches.If:
        yield python_ast.Statement.If(
            test=s.test,
            body=rewriteForLoops(s.body),
            orelse=rewriteForLoops(s.orelse),
        )
        return

    if s.matches.While:
        yield python_ast.Statement.While(
            test=s.test,
            body=rewriteForLoops(s.body),
            orelse=rewriteForLoops(s.orelse),
        )
        return

    if s.matches.Try:
        yield python_ast.Statement.Try(
            body=rewriteForLoops(s.body),
            handlers=[
                python_ast.ExceptionHandler.Item(
                    type=eh.type,
                    name=eh.name,
                    body=rewriteForLoops(eh.body)
                ) for eh in s.handlers
            ],
            orelse=rewriteForLoops(s.orelse),
            finalbody=rewriteForLoops(s.finalbody),
        )
        return

    if s.matches.With:
        yield python_ast.Statement.With(
            items=s.items,
            body=rewriteForLoops(s.body)
        )
        return

    assert s.matches.For, type(s)

    """Rewrite a generic for loop, making no assumptions about its type."""
    iteratorExpressionVarname = f".for.{s.line_number}.iteratorExpr"
    iteratorVarname = f".for.{s.line_number}.iterator"
    iteratorValuePtrVarname = f".for.{s.line_number}.iteratorValuePtr"
    iteratorTrigger = f".for.{s.line_number}.triggerElse"

    yield assign(iteratorExpressionVarname, s.iter)
    yield assign(iteratorVarname, makeCallExpr(attr(readVar(iteratorExpressionVarname), "__iter__")))
    yield assign(iteratorTrigger, const(False))

    yield python_ast.Statement.While(
        test=const(True),
        body=[
            assign(
                iteratorValuePtrVarname,
                makeCallExpr(attr(readVar(iteratorVarname), "__fastnext__"))
            ),
            branch(
                # check if the pointer is populated
                readVar(iteratorValuePtrVarname),
                [
                    # if so, get its value and assign it to our expression
                    python_ast.Statement.Assign(
                        targets=(s.target,),
                        value=makeCallExpr(attr(readVar(iteratorValuePtrVarname), "get"))
                    )
                ],
                [
                    # if not, then we exited the loop cleanly
                    # and we should trigger the 'orelse'
                    assign(iteratorTrigger, const(True)),
                    python_ast.Statement.Break()
                ],
            )
        ] + rewriteForLoops(s.body)
    )

    if s.orelse:
        yield python_ast.Statement.If(
            test=readVar(iteratorTrigger),
            body=rewriteForLoops(s.orelse)
        )


def rewriteIntiterForLoop(iterableVarname, target, body, orelse):
    """Rewrite a generic for loop, making no assumptions about its type."""
    iteratorMaxValue = f".for.{target.line_number}.iteratorMaxValue"
    iteratorValue = f".for.{target.line_number}.iteratorValue"
    iteratorTrigger = f".for.{target.line_number}.triggerElse"

    yield assign(
        iteratorMaxValue,
        makeCallExpr(attr(readVar(iterableVarname), "__typed_python_int_iter_size__"))
    )
    yield assign(iteratorValue, const(0))
    yield assign(iteratorTrigger, const(False))

    yield python_ast.Statement.While(
        test=compare(readVar(iteratorValue), readVar(iteratorMaxValue), "Lt"),
        body=[
            # if so, get its value and assign it to our expression
            python_ast.Statement.Assign(
                targets=(target,),
                value=makeCallExpr(
                    attr(readVar(iterableVarname), "__typed_python_int_iter_value__"),
                    readVar(iteratorValue)
                )
            ),
            assign(iteratorValue, binOp("Add", readVar(iteratorValue), const(1))),
        ] + body
    )

    if orelse:
        yield python_ast.Statement.If(
            test=readVar(iteratorTrigger),
            body=orelse
        )
