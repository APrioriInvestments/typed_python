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
"""
from typed_python import TupleOf
import typed_python.python_ast as python_ast
from typed_python.compiler.python_ast_analysis import (
    countYieldStatements,
)


def accessVar(varname):
    """Generate a 'self.(varname)' python expression"""
    return python_ast.Expr.Attribute(
        value=python_ast.Expr.Name(id="self", ctx=python_ast.ExprContext.Load()),
        attr=varname,
        ctx=python_ast.ExprContext.Load()
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


def branch(cond, l, r):
    """Generate an If python statement"""
    return python_ast.Statement.If(test=cond, body=l, orelse=r)


def genPrint(*exprs):
    """Generate a statement that prints all of 'exprs' (for debugging)"""
    return python_ast.Statement.Expr(
        value=python_ast.Expr.Call(
            func=python_ast.Expr.Name(id='print'),
            args=exprs
        )
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
    def __init__(self):
        self.yieldsSeen = 0

    def changeStatement(self, s):
        yieldsInside = countYieldStatements(s)

        yieldUpperBound = self.yieldsSeen + yieldsInside

        yield branch(
            # if the target slot is lessthan or equal to the number of yields we'll have
            # _after_ we exit this code, then we need to go in
            compare(accessVar("..slot"), const(yieldUpperBound), "Lt"),
            list(self.changeStatementInner(s)),
            []
        )

    def changeExpr(self, expr: python_ast.Expr):
        """Change any expressions we're accessing to 'self' expressions."""
        return self._changeExpr(expr)

    def _changeExpr(self, expr):
        if expr.matches.Name:
            return accessVar("." + expr.id)

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

        if s.matches.If:
            # if the slot is -1, we're just running. But if the
            # slot is between [self.yieldsSeen, self.yieldsSeen + yieldsLeft) we want
            # to go directly to the 'left' branch without executing the condition,
            # and if greater, then we go to the right without executing the condition
            yieldsSeen = self.yieldsSeen
            yieldsLeft = countYieldStatements(s.body)
            yieldsRight = countYieldStatements(s.orelse)

            yield branch(
                checkSlotBetween(-1, yieldsSeen + yieldsLeft + yieldsRight - 1),
                [
                    branch(
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
                    )
                ],
                []
            )

            return

        if s.matches.FunctionDef:
            raise Exception("Not implemented")
        if s.matches.ClassDef:
            raise Exception("Not implemented")
        if s.matches.Return:
            raise Exception("Not implemented")
        if s.matches.Delete:
            raise Exception("Not implemented")
        if s.matches.Assign:
            raise Exception("Not implemented")
        if s.matches.AugAssign:
            raise Exception("Not implemented")
        if s.matches.Print:
            raise Exception("Not implemented")
        if s.matches.For:
            raise Exception("Not implemented")
        if s.matches.While:
            raise Exception("Not implemented")
        if s.matches.If:
            raise Exception("Not implemented")
        if s.matches.With:
            raise Exception("Not implemented")
        if s.matches.Raise:
            raise Exception("Not implemented")
        if s.matches.Try:
            raise Exception("Not implemented")
        if s.matches.Assert:
            raise Exception("Not implemented")
        if s.matches.Import:
            raise Exception("Not implemented")
        if s.matches.ImportFrom:
            raise Exception("Not implemented")
        if s.matches.Global:
            raise Exception("Not implemented")
        if s.matches.Pass:
            raise Exception("Not implemented")
        if s.matches.Break:
            raise Exception("Not implemented")
        if s.matches.Continue:
            raise Exception("Not implemented")
        if s.matches.AsyncFunctionDef:
            raise Exception("Not implemented")
        if s.matches.AnnAssign:
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
            body=self.changeStatementSequence(statements) + [
                python_ast.Statement.Raise(
                    exc=python_ast.Expr.Call(
                        func=python_ast.Expr.Name(id='StopIteration'),
                        args=()
                    ),
                    cause=None
                )
            ],
            returns=None,
            filename=""
        )
