"""Utilities for building generators out of regular functions.

When we convert something that looks like

    def f(x):
        yield 1
        yield 2

we first turn it into something that looks (very roughly)

    class Generator:
        slot = Member(int)

        def __init__(self):
            self.slot = -1

        def __next__(self):
            if self.slot < 0:
                self.slot = 0
                return 1

            if self.slot < 1:
                self.slot = 1
                return 2

            raise StopIteration()

this allows us to wrap up normal-looking code as a generator.

The actual class encapsulates local variable and closure parameters,
and binds all variables with an extra '.' in front of them to avoid conflicting
with regular members.

The '..slot' member of the class encodes the state machine. a value of '-1' means
we are currently executing normally. A value of '-2' means we have ceased iteration
permanently, and a non-negative value means we are stopped at a 'yield' statement,
with '0' meaning the first yield statement, '1' meaning the second, etc.
"""

from typed_python import TupleOf
from typed_python.compiler.withblock_codegen import expandWithBlockIntoTryCatch
import typed_python.python_ast as python_ast
from typed_python.compiler.python_ast_analysis import (
    countYieldStatements,
)


def accessVar(varname, context=None):
    """Generate a 'self.(varname)' python expression"""
    return python_ast.Expr.Attribute(
        value=python_ast.Expr.Name(id="self", ctx=python_ast.ExprContext.Load()),
        attr=varname,
        ctx=python_ast.ExprContext.Load() if context is None else context
    )


def setVar(varname, val):
    """Generate a 'self.(varname) = val' python expression"""
    return python_ast.Statement.Assign(
        targets=(
            python_ast.Expr.Attribute(
                value=python_ast.Expr.Name(id="self", ctx=python_ast.ExprContext.Load()),
                attr=varname,
                ctx=python_ast.ExprContext.Store()
            ),
        ),
        value=val,
    )


def const(val):
    """Generate a python expression representing the constant 'val'"""
    if val is None:
        return python_ast.Expr.Num(n=python_ast.NumericConstant.None_())
    if isinstance(val, bool):
        return python_ast.Expr.Num(n=python_ast.NumericConstant.Boolean(value=val))
    if isinstance(val, int):
        return python_ast.Expr.Num(n=python_ast.NumericConstant.Int(value=val))
    if isinstance(val, str):
        return python_ast.Expr.Str(s=val)
    raise Exception("Don't know how to encode constant of type " + str(type(val)))


def compare(l, r, opcode):
    """Generate a pyexpression comparing python expressions l and r"""
    return python_ast.Expr.Compare(
        left=l,
        ops=(getattr(python_ast.ComparisonOp, opcode)(),),
        comparators=(r,)
    )


def boolOp(opcode, *values):
    """Generate 'or' or 'and' python expression.

    Args:
        opcode - one of 'And' or 'Or'
        values - python expressions to bool on
    """
    return python_ast.Expr.BoolOp(
        op=getattr(python_ast.BooleanOp, opcode)(),
        values=values,
    )


def branch(cond, l, r, isWhile=False):
    """Generate an If python statement"""
    Statement = python_ast.Statement

    return (Statement.If if not isWhile else Statement.While)(test=cond, body=l, orelse=r)


def genPrint(*exprs):
    """Generate a statement that prints all of 'exprs' (for debugging)"""
    return python_ast.Statement.Expr(
        value=python_ast.Expr.Call(
            func=python_ast.Expr.Name(id='print'),
            args=exprs
        )
    )


def raiseStopIteration(*args):
    return python_ast.Statement.Raise(
        exc=python_ast.Expr.Call(
            func=python_ast.Expr.Name(id='StopIteration'),
            args=tuple(args)
        ),
        cause=None
    )


def checkSlotBetween(low, high):
    if low > high:
        return const(False)

    if low == high:
        return compare(const(low), accessVar("..slot"), "Eq")

    return boolOp(
        "And",
        compare(const(low), accessVar("..slot"), "LtE"),
        compare(const(high), accessVar("..slot"), "GtE")
    )


class GeneratorCodegen:
    def __init__(self, localVars):
        self.localVars = localVars
        self.yieldsSeen = 0

    def changeExpr(self, expr: python_ast.Expr):
        """Change any expressions we're accessing to 'self' expressions."""
        return self._changeExpr(expr)

    def _changeExpr(self, expr):
        if expr.matches.Name:
            if expr.id in self.localVars:
                return accessVar("." + expr.id, expr.ctx)
            return expr

        args = {}

        for ix in range(len(expr.ElementType.ElementNames)):
            argT = expr.ElementType.ElementTypes[ix]
            name = expr.ElementType.ElementNames[ix]

            if argT is python_ast.Expr:
                args[name] = self._changeExpr(getattr(expr, name))
            elif argT is TupleOf(python_ast.Expr):
                args[name] = [self._changeExpr(subE) for subE in getattr(expr, name)]
            else:
                args[name] = getattr(expr, name)

        return type(expr)(**args)

    def changeStatement(self, s):
        yieldsInside = countYieldStatements(s)

        yieldUpperBound = self.yieldsSeen + yieldsInside

        yield branch(
            # if the target slot is lessthan or equal to the number of yields we'll have
            # _after_ we exit this code, then we need to go in
            compare(accessVar("..slot"), const(yieldUpperBound), "Lt"),
            # if there are no internal yields, then we don't need to do anything
            # but this external check, which guarantees we skip over the statement
            # if we're searching ahead for the next expression
            list(self.changeStatementInner(s)),
            []
        )

    def changeStatementInner(self, s):
        if s.matches.Expr:
            if s.value.matches.Yield:
                yield branch(
                    compare(accessVar("..slot"), const(self.yieldsSeen), "Eq"),
                    [setVar("..slot", const(-1))],
                    [
                        setVar("..slot", const(self.yieldsSeen)),
                        python_ast.Statement.Return(
                            value=self.changeExpr(s.value.value)
                        )
                    ]
                )
                self.yieldsSeen += 1
            else:
                yield self.changeExpr(s)

            return

        if s.matches.Assign:
            yield python_ast.Statement.Assign(
                targets=(self.changeExpr(x) for x in s.targets),
                value=self.changeExpr(s.value)
            )

            return

        if s.matches.AugAssign:
            yield python_ast.Statement.AugAssign(
                target=self.changeExpr(s.target),
                op=s.op,
                value=self.changeExpr(s.value)
            )

            return

        if s.matches.AnnAssign:
            yield python_ast.Statement.AnnAssign(
                target=self.changeExpr(s.target),
                annotation=const(0),
                value=self.changeExpr(s.value) if s.value is not None else None
            )

            return

        if s.matches.If or s.matches.While:
            # if the slot is -1, we're just running. But if the
            # slot is between [self.yieldsSeen, self.yieldsSeen + yieldsLeft) we want
            # to go directly to the 'left' branch without executing the condition,
            # and if greater, then we go to the right without executing the condition
            yieldsSeen = self.yieldsSeen
            yieldsLeft = countYieldStatements(s.body)

            yield branch(
                boolOp(
                    "Or",
                    checkSlotBetween(yieldsSeen, yieldsSeen + yieldsLeft - 1),
                    boolOp(
                        "And",
                        checkSlotBetween(-1, -1),
                        self.changeExpr(s.test)
                    )
                ),
                self.changeStatementSequence(s.body),
                self.changeStatementSequence(s.orelse),
                isWhile=s.matches.While
            )

            return

        if s.matches.Pass:
            yield python_ast.Statement.Pass()

            return

        if s.matches.Break:
            yield python_ast.Statement.Break()

            return

        if s.matches.Continue:
            yield python_ast.Statement.Continue()

            return

        if s.matches.Assert:
            yield python_ast.Statement.Assert(
                test=self.changeExpr(s.test),
                msg=None if s.msg is None else self.changeExpr(s.msg)
            )

            return

        if s.matches.Raise:
            yield python_ast.Statement.Raise(
                exc=self.changeExpr(s.exc) if s.exc is not None else None,
                cause=self.changeExpr(s.cause) if s.cause is not None else None,
            )

            return

        # to simply things, if we have exceptions and finally, break ourselves
        # apart into two
        if s.matches.Try and s.handlers and s.finalbody:
            for subexpr in self.changeStatement(
                python_ast.Statement.Try(
                    body=[
                        python_ast.Statement.Try(
                            body=s.body,
                            handlers=s.handlers,
                            orelse=s.orelse,
                            finalbody=[],
                            line_number=s.line_number,
                            col_offset=s.col_offset,
                            filename=s.filename
                        )
                    ],
                    handlers=[],
                    orelse=[],
                    finalbody=s.finalbody,
                    line_number=s.line_number,
                    col_offset=s.col_offset,
                    filename=s.filename
                )
            ):
                yield subexpr

            return

        if s.matches.Try:
            yieldsSeen = self.yieldsSeen
            yieldsBody = countYieldStatements(s.body)

            # if any of the exception handlers has a yield in it, we will
            # need a block allowing us to resume in it
            cleanupMatchers = []

            yieldsAtThisCleanupHandlerStart = yieldsSeen + yieldsBody

            # for each block of statements, we might have a resumption.
            for body in [eh.body for eh in s.handlers] + [s.orelse, s.finalbody]:
                thisEHYields = countYieldStatements(body)

                if thisEHYields:
                    # this is a stateful part of the conversion process,
                    # so we have to set it to the right value
                    self.yieldsSeen = yieldsAtThisCleanupHandlerStart

                    cleanupMatchers.append(
                        branch(
                            checkSlotBetween(
                                yieldsAtThisCleanupHandlerStart,
                                yieldsAtThisCleanupHandlerStart + thisEHYields - 1,
                            ),
                            self.changeStatementSequence(body),
                            []
                        )
                    )

                yieldsAtThisCleanupHandlerStart += thisEHYields

            # reset the counter so we can generate the main body
            self.yieldsSeen = yieldsSeen

            yield branch(
                checkSlotBetween(-1, yieldsSeen + yieldsBody - 1),
                # we're inside the body of the try block.
                [
                    python_ast.Statement.Try(
                        body=self.changeStatementSequence(s.body),
                        handlers=[
                            python_ast.ExceptionHandler.Item(
                                type=self.changeExpr(eh.type) if eh.type is not None else None,
                                name=eh.name,
                                body=self.changeStatementSequence(eh.body)
                            ) for eh in s.handlers
                        ],
                        orelse=self.changeStatementSequence(s.orelse),
                        finalbody=[
                            # only run the finally check if we are handling an
                            # the normal flow of execution. If the slot index is
                            # set to something other than -1, then we are paused
                            # and returning a value and shouldn't run.
                            branch(
                                checkSlotBetween(-1, -1),
                                self.changeStatementSequence(s.finalbody),
                                [],
                            )
                        ] if s.finalbody else []
                    )
                ],
                cleanupMatchers
            )

            return

        if s.matches.Return:
            if s.value is None:
                yield raiseStopIteration()
            else:
                yield raiseStopIteration(
                    self.changeExpr(s.value)
                )

            return

        if s.matches.With:
            for subexpr in self.changeStatementSequence(expandWithBlockIntoTryCatch(s)):
                yield subexpr

            return

        if s.matches.Import:
            yield s
            return

        if s.matches.ImportFrom:
            yield s
            return

        if s.matches.Delete:
            yield python_ast.Statement.Delete(
                targets=[self.changeExpr(e) for e in s.targets]
            )
            return

        if s.matches.FunctionDef:
            raise Exception("Not implemented")
        if s.matches.ClassDef:
            raise Exception("Not implemented")
        if s.matches.For:
            raise Exception("Not implemented")

        if s.matches.Global:
            raise Exception("Not implemented")
        if s.matches.AsyncFunctionDef:
            raise Exception("Not implemented")
        if s.matches.AsyncWith:
            raise Exception("Not implemented")
        if s.matches.AsyncFor:
            raise Exception("Not implemented")
        if s.matches.NonLocal:
            raise Exception("Not implemented")

        raise Exception("Unknown statement: " + str(type(s)))

    def changeStatementSequence(self, statements):
        return [subst for s in statements for subst in self.changeStatement(s)]

    def convertStatementsToFunctionDef(self, statements):
        return python_ast.Statement.FunctionDef(
            name="__next__",
            args=python_ast.Arguments.Item(
                args=[python_ast.Arg.Item(arg="self", annotation=None)],
                vararg=None,
                kwarg=None
            ),
            body=[
                python_ast.Statement.Try(
                    body=[
                        branch(
                            checkSlotBetween(-2, -2),
                            [raiseStopIteration()],
                            []
                        )
                    ] + self.changeStatementSequence(statements) + [
                        raiseStopIteration()
                    ],
                    finalbody=[
                        # if we exit during 'normal' execution (slot == -1)
                        # then we are unwinding an exception and we should never
                        # resume
                        branch(
                            checkSlotBetween(-1, -1),
                            [setVar("..slot", const(-2))],
                            []
                        )
                    ]
                )
            ],
            returns=None,
            filename=""
        )
