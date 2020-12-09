"""Common functions useful for generating python asts.

We do this because some language features are easier to implement by rewriting
certain parts of the code.
"""
import typed_python.python_ast as python_ast


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


def binOp(opcode, l, r):
    """Generate 'or' or 'and' python expression.

    Args:
        opcode - one of 'And' or 'Or'
        values - python expressions to bool on
    """
    return python_ast.Expr.BinOp(
        left=l,
        op=getattr(python_ast.BinaryOp, opcode)(),
        right=r
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


def attr(v, name):
    return python_ast.Expr.Attribute(
        value=v,
        attr=name,
        ctx=python_ast.ExprContext.Load()
    )


def readVar(n):
    """Generate a statement that prints all of 'exprs' (for debugging)"""
    return python_ast.Expr.Name(
        id=n,
        ctx=python_ast.ExprContext.Load()
    )


def assign(n, val):
    return python_ast.Statement.Assign(
        targets=(
            python_ast.Expr.Name(id=n, ctx=python_ast.ExprContext.Store()),
        ),
        value=val
    )


def makeCallExpr(func, *exprs):
    return python_ast.Expr.Call(
        func=func,
        args=exprs
    )


def makeCall(func, *exprs):
    return python_ast.Statement.Expr(makeCallExpr(func, *exprs))


def raiseStopIteration(*args):
    return python_ast.Statement.Raise(
        exc=python_ast.Expr.Call(
            func=python_ast.Expr.Name(id='StopIteration'),
            args=tuple(args)
        ),
        cause=None
    )
