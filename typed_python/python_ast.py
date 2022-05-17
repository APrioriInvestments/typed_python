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

"""
Convert python AST objects into a more explicit set of
Algebraic types. These are easier to work with than the
Python ast directly.
"""

import sys
import ast
import typed_python.compiler.python_ast_util as python_ast_util
import types
import traceback

from typed_python._types import Forward, Alternative, TupleOf, OneOf


# forward declarations.
Module = Forward("Module")
Statement = Forward("Statement")
Expr = Forward("Expr")
Arg = Forward("Arg")
NumericConstant = Forward("NumericConstant")
ExprContext = Forward("ExprContext")
BooleanOp = Forward("BooleanOp")
BinaryOp = Forward("BinaryOp")
UnaryOp = Forward("UnaryOp")
ComparisonOp = Forward("ComparisonOp")
Comprehension = Forward("Comprehension")
ExceptionHandler = Forward("ExceptionHandler")
Arguments = Forward("Arguments")
Keyword = Forward("Keyword")
Alias = Forward("Alias")
WithItem = Forward("WithItem")
TypeIgnore = Forward("TypeIgnore")

Module = Module.define(Alternative(
    "Module",
    Module={
        "body": TupleOf(Statement),
        **({"type_ignores": TupleOf(TypeIgnore)} if sys.version_info.minor >= 8 else {})
    },
    Expression={'body': Expr},
    Interactive={'body': TupleOf(Statement)},
    Suite={"body": TupleOf(Statement)}
))

TypeIgnore = TypeIgnore.define(Alternative(
    "TypeIgnore",
    Item={'lineno': int, 'tag': str}
))


def statementStrLines(self):
    if self.matches.FunctionDef:
        yield f"def {self.name}(...):"
        for s in self.body:
            for line in statementStrLines(s):
                yield "    " + line
        return

    elif self.matches.Expr:
        yield str(self.value)

    elif self.matches.If:
        yield f"if {self.test}:"
        for s in self.body:
            for line in statementStrLines(s):
                yield "    " + line
        if self.orelse:
            yield "else:"
            for s in self.orelse:
                for line in statementStrLines(s):
                    yield "    " + line

    elif self.matches.While:
        yield f"while {self.test}:"
        for s in self.body:
            for line in statementStrLines(s):
                yield "    " + line
        if self.orelse:
            yield "else:"
            for s in self.orelse:
                for line in statementStrLines(s):
                    yield "    " + line

    elif self.matches.Try:
        yield "try:"
        for s in self.body:
            for line in statementStrLines(s):
                yield "    " + line
        for eh in self.handlers:
            yield f"except {eh.type}" + (f" as {eh.name}")
            for s in eh.body:
                for line in statementStrLines(s):
                    yield "    " + line
        if self.orelse:
            yield "else:"
            for s in self.orelse:
                for line in statementStrLines(s):
                    yield "    " + line
        if self.finalbody:
            yield "finally:"
            for s in self.finalbody:
                for line in statementStrLines(s):
                    yield "    " + line

    elif self.matches.With:
        yield f"with {self.items}:"
        for s in self.body:
            for line in statementStrLines(s):
                yield "    " + line

    elif self.matches.Assign:
        yield f"{', '.join(str(x) for x in self.targets)} = {self.value}"

    elif self.matches.AugAssign:
        yield f"{self.target} {self.op}= {self.value}"

    elif self.matches.Raise:
        res = "raise"
        if self.exc is not None:
            res += " " + str(self.exc)

        if self.cause is not None:
            res += " from " + str(self.cause)

        yield res

    elif self.matches.Break:
        yield "break"

    elif self.matches.Continue:
        yield "continue"

    elif self.matches.Pass:
        yield "pass"

    elif self.matches.Return:
        if self.value is not None:
            yield f"return {self.value}"
        else:
            yield "return"
    else:
        yield str(type(self)) + "..."


def StatementStr(self):
    return "\n".join(list(statementStrLines(self)))


Statement = Statement.define(Alternative(
    "Statement",
    FunctionDef={
        "name": str,
        "args": Arguments,
        "body": TupleOf(Statement),
        "decorator_list": TupleOf(Expr),
        "returns": OneOf(Expr, None),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    ClassDef={
        "name": str,
        "bases": TupleOf(Expr),
        "keywords": TupleOf(Keyword),
        "body": TupleOf(Statement),
        "decorator_list": TupleOf(Expr),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Return={
        "value": OneOf(Expr, None),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Delete={
        "targets": TupleOf(Expr),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Assign={
        "targets": TupleOf(Expr),
        "value": Expr,
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    AugAssign={
        "target": Expr,
        "op": BinaryOp,
        "value": Expr,
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Print={
        "expr": OneOf(Expr, None),
        "values": TupleOf(Expr),
        "nl": int,
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    For={
        "target": Expr,
        "iter": Expr,
        "body": TupleOf(Statement),
        "orelse": TupleOf(Statement),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    While={
        "test": Expr,
        "body": TupleOf(Statement),
        "orelse": TupleOf(Statement),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    If={
        "test": Expr,
        "body": TupleOf(Statement),
        "orelse": TupleOf(Statement),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    With={
        "items": TupleOf(WithItem),
        "body": TupleOf(Statement),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Raise={
        "exc": OneOf(Expr, None),
        "cause": OneOf(Expr, None),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Try={
        "body": TupleOf(Statement),
        "handlers": TupleOf(ExceptionHandler),
        "orelse": TupleOf(Statement),
        "finalbody": TupleOf(Statement),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Assert={
        "test": Expr,
        "msg": OneOf(Expr, None),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Import={
        "names": TupleOf(Alias),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    ImportFrom={
        "module": OneOf(str, TupleOf(str)),
        "names": OneOf(Alias, TupleOf(Alias)),
        "level": OneOf(int, None),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Global={
        "names": TupleOf(str),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Expr={
        "value": Expr,
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Pass={
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Break={
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Continue={
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    AsyncFunctionDef={
        "name": str,
        "args": Arguments,
        "body": TupleOf(Statement),
        "decorator_list": TupleOf(Expr),
        "returns": OneOf(Expr, None),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    AnnAssign={
        "target": Expr,
        "annotation": Expr,
        'simple': int,
        "value": OneOf(Expr, None),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    AsyncWith={
        "items": TupleOf(WithItem),
        "body": TupleOf(Statement),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    AsyncFor={
        'target': Expr,
        'iter': Expr,
        'body': TupleOf(Statement),
        'orelse': TupleOf(Statement),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    NonLocal={
        "names": TupleOf(str),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    __str__=StatementStr
))


def ExpressionStr(self):
    if self.matches.ListComp:
        res = "[" + str(self.elt)
        for gen in self.generators:
            res += " for " + str(gen.target) + " in " + str(gen.iter)
            for ifS in gen.ifs:
                res += " if " + str(ifS)
        return res + "]"

    if self.matches.Lambda:
        return "(lambda ...: " + str(self.body) + ")"

    if self.matches.Subscript:
        return str(self.value) + "[" + str(self.slice) + "]"

    if self.matches.Num:
        return str(self.n)

    if self.matches.Call:
        return (
            f"({self.func})(" +
            ", ".join([str(x) for x in self.args] + [f"{kwd.arg}={kwd.value}" for kwd in self.keywords])
            + ")"
        )

    if self.matches.Str:
        return repr(self.s)

    if self.matches.Compare:
        res = str(self.left)
        for i in range(len(self.ops)):
            if self.ops[i].matches.Eq:
                sep = "=="
            if self.ops[i].matches.NotEq:
                sep = "!="
            if self.ops[i].matches.Lt:
                sep = "<"
            if self.ops[i].matches.LtE:
                sep = "<="
            if self.ops[i].matches.Gt:
                sep = ">"
            if self.ops[i].matches.GtE:
                sep = ">="
            if self.ops[i].matches.Is:
                sep = "is"
            if self.ops[i].matches.IsNot:
                sep = "is not"
            if self.ops[i].matches.In:
                sep = "in"
            if self.ops[i].matches.NotIn:
                sep = "not in"

            res += f" {sep} {self.comparators[i]}"
        return res

    if self.matches.BoolOp:
        sep = " and " if self.op.matches.And else " or "
        return sep.join([f"({x})" for x in self.values])

    if self.matches.BinOp:
        if self.op.matches.Add:
            sep = "+"
        if self.op.matches.Sub:
            sep = "-"
        if self.op.matches.Mult:
            sep = "*"
        if self.op.matches.Div:
            sep = "/"
        if self.op.matches.Mod:
            sep = "%"
        if self.op.matches.Pow:
            sep = "**"
        if self.op.matches.LShift:
            sep = "<<"
        if self.op.matches.RShift:
            sep = ">>"
        if self.op.matches.BitOr:
            sep = "|"
        if self.op.matches.BitXor:
            sep = "^"
        if self.op.matches.BitAnd:
            sep = "&"
        if self.op.matches.FloorDiv:
            sep = "//"
        if self.op.matches.MatMult:
            sep = "@"

        return f"({self.left}) {sep} ({self.right})"

    if self.matches.UnaryOp:
        if self.op.matches.Invert:
            sep = "~"
        if self.op.matches.Not:
            sep = "not "
        if self.op.matches.UAdd:
            sep = "+"
        if self.op.matches.USub:
            sep = "-"

        return f"{sep} ({self.operand})"

    if self.matches.Attribute:
        return f"({self.value}).{self.attr}"

    if self.matches.Yield:
        if self.value is None:
            return "yield"
        else:
            return f"yield {self.value}"

    if self.matches.Name:
        return self.id

    return str(type(self))


Expr = Expr.define(Alternative(
    "Expr",
    BoolOp={
        "op": BooleanOp,
        "values": TupleOf(Expr),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    BinOp={
        "left": Expr,
        "op": BinaryOp,
        "right": Expr,
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    UnaryOp={
        "op": UnaryOp,
        "operand": Expr,
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Lambda={
        "args": Arguments,
        "body": Expr,
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    IfExp={
        "test": Expr,
        "body": Expr,
        "orelse": Expr,
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Dict={
        "keys": TupleOf(OneOf(None, Expr)),
        "values": TupleOf(Expr),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Set={
        "elts": TupleOf(Expr),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    ListComp={
        "elt": Expr,
        "generators": TupleOf(Comprehension),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    SetComp={
        "elt": Expr,
        "generators": TupleOf(Comprehension),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    DictComp={
        "key": Expr,
        "value": Expr,
        "generators": TupleOf(Comprehension),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    GeneratorExp={
        "elt": Expr,
        "generators": TupleOf(Comprehension),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Yield={
        "value": OneOf(Expr, None),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Compare={
        "left": Expr,
        "ops": TupleOf(ComparisonOp),
        "comparators": TupleOf(Expr),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Call={
        "func": Expr,
        "args": TupleOf(Expr),
        "keywords": TupleOf(Keyword),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Num={
        "n": NumericConstant,
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Str={
        "s": str,
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Attribute={
        "value": Expr,
        "attr": str,
        "ctx": ExprContext,
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Subscript={
        "value": Expr,
        "slice": Expr,
        "ctx": ExprContext,
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Name={
        "id": str,
        "ctx": ExprContext,
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    List={
        "elts": TupleOf(Expr),
        "ctx": ExprContext,
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Tuple={
        "elts": TupleOf(Expr),
        "ctx": ExprContext,
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Starred={
        "value": Expr,
        "ctx": ExprContext,
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    YieldFrom={
        "value": Expr,
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Await={
        "value": Expr,
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    JoinedStr={
        "values": TupleOf(Expr),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Bytes={
        's': bytes,
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Constant={
        'value': OneOf(object, None),
        **({'kind': OneOf(None, str)} if sys.version_info.minor >= 8 else {}),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    FormattedValue={
        "value": Expr,
        "conversion": OneOf(int, None),
        "format_spec": OneOf(Expr, None),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    Slice={
        "lower": OneOf(Expr, None),
        "upper": OneOf(Expr, None),
        "step": OneOf(Expr, None),
        'line_number': int,
        'col_offset': int,
        'filename': str
    },
    __str__=ExpressionStr
))

NumericConstant = NumericConstant.define(Alternative(
    "NumericConstant",
    Int={"value": int},
    Long={"value": str},
    Boolean={"value": bool},
    None_={},
    Float={"value": float},
    Complex={"real": float, "imag": float},
    Unknown={},
    __str__=lambda self: (
        str(self.value) if (
            self.matches.Int or self.matches.Long
            or self.matches.Boolean or self.matches.Float
        ) else "None" if self.matches.None_ else
        f"{self.real} + {self.imag}j" if self.matches.Complex else "Unknown"
    )
))

ExprContext = ExprContext.define(Alternative(
    "ExprContext",
    Load={},
    Store={},
    Del={},
    AugLoad={},
    AugStore={},
    Param={}
))

BooleanOp = BooleanOp.define(Alternative(
    "BooleanOp",
    And={},
    Or={}
))

BinaryOp = BinaryOp.define(Alternative(
    "BinaryOp",
    Add={},
    Sub={},
    Mult={},
    Div={},
    Mod={},
    Pow={},
    LShift={},
    RShift={},
    BitOr={},
    BitXor={},
    BitAnd={},
    FloorDiv={},
    MatMult={}
))

UnaryOp = UnaryOp.define(Alternative(
    "UnaryOp",
    Invert={},
    Not={},
    UAdd={},
    USub={}
))

ComparisonOp = ComparisonOp.define(Alternative(
    "ComparisonOp",
    Eq={},
    NotEq={},
    Lt={},
    LtE={},
    Gt={},
    GtE={},
    Is={},
    IsNot={},
    In={},
    NotIn={}
))

Comprehension = Comprehension.define(Alternative(
    "Comprehension",
    Item={
        "target": Expr,
        "iter": Expr,
        "ifs": TupleOf(Expr),
        "is_async": bool
    }
))

ExceptionHandler = ExceptionHandler.define(Alternative(
    "ExceptionHandler",
    Item={
        "type": OneOf(Expr, None),
        "name": OneOf(str, None),
        "body": TupleOf(Statement),
        'line_number': int,
        'col_offset': int,
        'filename': str
    }
))

Arguments = Arguments.define(Alternative(
    "Arguments",
    Item={
        **({'posonlyargs': TupleOf(Arg)} if sys.version_info.minor >= 8 else {}),
        "args": TupleOf(Arg),
        "vararg": OneOf(Arg, None),
        "kwonlyargs": TupleOf(Arg),
        "kw_defaults": TupleOf(Expr),
        "kwarg": OneOf(Arg, None),
        "defaults": TupleOf(Expr),
    },
    totalArgCount=lambda self:
        len(self.args)
        + (1 if self.vararg else 0)
        + (1 if self.kwarg else 0)
        + len(self.kwonlyargs),
    argumentNames=lambda self:
        [a.arg for a in self.args]
        + ([self.vararg.arg] if self.vararg else [])
        + [a.arg for a in self.kwonlyargs]
        + ([self.kwarg.arg] if self.kwarg else [])
))

Arg = Arg.define(Alternative(
    "Arg",
    Item={
        'arg': str,
        'annotation': OneOf(Expr, None),
        'line_number': int,
        'col_offset': int,
        'filename': str
    }
))

Keyword = Keyword.define(Alternative(
    "Keyword",
    Item={
        "arg": OneOf(None, str),
        "value": Expr,
        **({'line_number': int, 'col_offset': int, 'filename': str} if sys.version_info.minor >= 9 else {})
    }
))

Alias = Alias.define(Alternative(
    "Alias",
    Item={
        "name": str,
        "asname": OneOf(str, None),
        **({
            'line_number': int,
            'col_offset': int,
            'filename': str
        } if sys.version_info.minor >= 10 else {})
    }
))

WithItem = WithItem.define(Alternative(
    "WithItem",
    Item={
        "context_expr": Expr,
        "optional_vars": OneOf(Expr, None),
    }
))

numericConverters = {
    int: lambda x: NumericConstant.Int(value=x),
    bool: lambda x: NumericConstant.Boolean(value=x),
    type(None): lambda x: NumericConstant.None_(),
    float: lambda x: NumericConstant.Float(value=x),
    complex: lambda x: NumericConstant.Complex(real=x.real, imag=x.imag)
}


def createPythonAstConstant(n, **kwds):
    if type(n) not in numericConverters:
        return Expr.Num(
            n=NumericConstant.Unknown(),
            **kwds
        )
    return Expr.Num(
        n=numericConverters[type(n)](n),
        **kwds
    )


def createPythonAstString(s, **kwds):
    try:
        return Expr.Str(s=str(s), **kwds)
    except Exception:
        return Expr.Num(
            n=NumericConstant.Unknown(),
            **kwds
        )


def makeNameConstant(value, **kwds):
    return Expr.Num(n=numericConverters[type(value)](value), **kwds)


def makeEllipsis(*args):
    return Expr.Constant(value=...)


def makeExtSlice(dims):
    return Expr.Tuple(elts=dims)


# map Python AST types to our syntax-tree types (defined `ove)
converters = {
    ast.Module: Module.Module,
    ast.Expression: Module.Expression,
    ast.Interactive: Module.Interactive,
    ast.Suite: Module.Suite,
    ast.FunctionDef: Statement.FunctionDef,
    ast.ClassDef: Statement.ClassDef,
    ast.Return: Statement.Return,
    ast.Delete: Statement.Delete,
    ast.Assign: Statement.Assign,
    ast.AugAssign: Statement.AugAssign,
    ast.AnnAssign: Statement.AnnAssign,
    ast.For: Statement.For,
    ast.While: Statement.While,
    ast.If: Statement.If,
    ast.With: Statement.With,
    ast.Raise: Statement.Raise,
    ast.Try: Statement.Try,
    ast.Assert: Statement.Assert,
    ast.Import: Statement.Import,
    ast.ImportFrom: Statement.ImportFrom,
    ast.Global: Statement.Global,
    ast.Nonlocal: Statement.NonLocal,
    ast.Expr: Statement.Expr,
    ast.Pass: Statement.Pass,
    ast.Break: Statement.Break,
    ast.Continue: Statement.Continue,
    ast.BoolOp: Expr.BoolOp,
    ast.BinOp: Expr.BinOp,
    ast.UnaryOp: Expr.UnaryOp,
    ast.Lambda: Expr.Lambda,
    ast.IfExp: Expr.IfExp,
    ast.Dict: Expr.Dict,
    ast.Set: Expr.Set,
    ast.JoinedStr: Expr.JoinedStr,
    ast.Bytes: Expr.Bytes,
    ast.Constant: Expr.Constant,
    ast.FormattedValue: Expr.FormattedValue,
    ast.ListComp: Expr.ListComp,
    ast.AsyncFunctionDef: Statement.AsyncFunctionDef,
    ast.AsyncWith: Statement.AsyncWith,
    ast.AsyncFor: Statement.AsyncFor,
    ast.Await: Expr.Await,
    ast.SetComp: Expr.SetComp,
    ast.DictComp: Expr.DictComp,
    ast.GeneratorExp: Expr.GeneratorExp,
    ast.Yield: Expr.Yield,
    ast.YieldFrom: Expr.YieldFrom,
    ast.Compare: Expr.Compare,
    ast.Call: Expr.Call,
    ast.Num: createPythonAstConstant,
    ast.Str: createPythonAstString,
    ast.Attribute: Expr.Attribute,
    ast.Subscript: Expr.Subscript,
    ast.Name: Expr.Name,
    ast.NameConstant: makeNameConstant,
    ast.List: Expr.List,
    ast.Tuple: Expr.Tuple,
    ast.Starred: Expr.Starred,
    ast.Load: ExprContext.Load,
    ast.Store: ExprContext.Store,
    ast.Del: ExprContext.Del,
    ast.AugLoad: ExprContext.AugLoad,
    ast.AugStore: ExprContext.AugStore,
    ast.Param: ExprContext.Param,
    ast.Ellipsis: makeEllipsis,
    ast.Slice: Expr.Slice,
    ast.ExtSlice: makeExtSlice,
    ast.Index: lambda value: value,
    ast.And: BooleanOp.And,
    ast.Or: BooleanOp.Or,
    ast.Add: BinaryOp.Add,
    ast.Sub: BinaryOp.Sub,
    ast.Mult: BinaryOp.Mult,
    ast.MatMult: BinaryOp.MatMult,
    ast.Div: BinaryOp.Div,
    ast.Mod: BinaryOp.Mod,
    ast.Pow: BinaryOp.Pow,
    ast.LShift: BinaryOp.LShift,
    ast.RShift: BinaryOp.RShift,
    ast.BitOr: BinaryOp.BitOr,
    ast.BitXor: BinaryOp.BitXor,
    ast.BitAnd: BinaryOp.BitAnd,
    ast.FloorDiv: BinaryOp.FloorDiv,
    ast.Invert: UnaryOp.Invert,
    ast.Not: UnaryOp.Not,
    ast.UAdd: UnaryOp.UAdd,
    ast.USub: UnaryOp.USub,
    ast.Eq: ComparisonOp.Eq,
    ast.NotEq: ComparisonOp.NotEq,
    ast.Lt: ComparisonOp.Lt,
    ast.LtE: ComparisonOp.LtE,
    ast.Gt: ComparisonOp.Gt,
    ast.GtE: ComparisonOp.GtE,
    ast.Is: ComparisonOp.Is,
    ast.IsNot: ComparisonOp.IsNot,
    ast.In: ComparisonOp.In,
    ast.NotIn: ComparisonOp.NotIn,
    ast.comprehension: Comprehension.Item,
    ast.excepthandler: lambda x: x,
    ast.ExceptHandler: ExceptionHandler.Item,
    ast.arguments: Arguments.Item,
    ast.arg: Arg.Item,
    ast.keyword: Keyword.Item,
    ast.alias: Alias.Item,
    ast.withitem: WithItem.Item,
    **({'ast.type_ignore': TypeIgnore.Item} if sys.version_info.minor >= 8 else {}),
}

# most converters map to an alternative type
reverseConverters = {
    t: v for v, t in converters.items()
    if hasattr(t, '__typed_python_category__') and t.__typed_python_category__ == "ConcreteAlternative"
}


def convertAlgebraicArgs(pyAst, *members):
    members = [x for x in members if x not in ['line_number', 'col_offset']]
    return {m: convertAlgebraicToPyAst(getattr(pyAst, m)) for m in members}


def convertAlgebraicToPyAst(pyAst):
    res = convertAlgebraicToPyAst_(pyAst)

    if hasattr(pyAst, "line_number"):
        res.lineno = pyAst.line_number
        res.col_offset = pyAst.col_offset

    return res


def convertAlgebraicToSlice(pyAst):
    if sys.version_info.minor >= 9:
        return convertAlgebraicToPyAst(pyAst)
    else:
        if pyAst.matches.Slice:
            args = {}

            if pyAst.lower is not None:
                args['lower'] = convertAlgebraicToPyAst(pyAst.lower)

            if pyAst.upper is not None:
                args['upper'] = convertAlgebraicToPyAst(pyAst.upper)

            if pyAst.step is not None:
                args['step'] = convertAlgebraicToPyAst(pyAst.step)

            return ast.Slice(**args)

        if pyAst.matches.Tuple:
            return ast.ExtSlice(dims=[convertAlgebraicToPyAst(x) for x in pyAst.elts])

        return ast.Index(convertAlgebraicToPyAst(pyAst))


def convertAlgebraicToPyAst_(pyAst):
    if pyAst is None:
        return None

    if isinstance(pyAst, (str, int, float, bool, bytes)):
        return pyAst

    if hasattr(pyAst, "__typed_python_category__") and pyAst.__typed_python_category__ == "TupleOf":
        return [convertAlgebraicToPyAst(x) for x in pyAst]

    if type(pyAst) is Expr.Str:
        return ast.Str(s=pyAst.s)

    if type(pyAst) is Expr.Num:
        if pyAst.n.matches.Boolean:
            return ast.NameConstant(value=True if pyAst.n.value else False)
        if pyAst.n.matches.None_:
            return ast.NameConstant(value=None)
        if pyAst.n.matches.Complex:
            return ast.Num(n=complex(pyAst.n.real, pyAst.n.imag))
        if pyAst.n.matches.Unknown:
            raise Exception(f"Unknown constant: {pyAst.filename}:{pyAst.line_number}")
        return ast.Num(n=pyAst.n.value)

    if type(pyAst) is Expr.Subscript:
        res = ast.Subscript(
            value=convertAlgebraicToPyAst(pyAst.value),
            slice=convertAlgebraicToSlice(pyAst.slice),
            ctx=convertAlgebraicToPyAst(pyAst.ctx),
        )

        res.lineno = pyAst.line_number
        res.col_offset = pyAst.col_offset

        return res

    if type(pyAst) is Expr.Constant:
        return reverseConverters[type(pyAst)](
            **{k: getattr(pyAst, k) for k in type(pyAst).ElementType.ElementNames if k not in ['line_number', 'col_offset']}
        )

    if type(pyAst) in reverseConverters:
        return reverseConverters[type(pyAst)](**convertAlgebraicArgs(pyAst, *type(pyAst).ElementType.ElementNames))

    assert False, type(pyAst)


def convertPyAstToAlgebraic(tree, fname, keepLineInformation=True):
    if issubclass(type(tree), ast.AST):
        converter = converters[type(tree)]
        args = {}

        for f in tree._fields:
            # type_comment was introduced in 3.8, but we don't need it
            if f != "type_comment":
                if hasattr(tree, f):
                    args[f] = convertPyAstToAlgebraic(getattr(tree, f), fname, keepLineInformation)
                else:
                    args[f] = None

        try:
            if keepLineInformation:
                args['line_number'] = tree.lineno
                args['col_offset'] = tree.col_offset
                args['filename'] = fname
            else:
                args['line_number'] = 0
                args['col_offset'] = 0
                args['filename'] = ''
        except AttributeError:
            pass

        try:
            # 'type_comment' is introduced in 3.8, but we don't need it
            # and don't do anything with it, and it's just a comment so it doesn't
            # affect execution semantics.
            if 'type_comment' in args:
                args.pop('type_comment')

            return converter(**args)
        except Exception:
            if 'line_number' in args:
                del args['line_number']
                del args['col_offset']
                del args['filename']

            try:
                return converter(**args)
            except Exception:
                raise UserWarning(
                    "Failed to construct %s from %s with arguments\n%s\n\n%s" % (
                        converter,
                        type(tree),
                        "\n".join([
                            "\t%s:%s (from %s)" % (
                                k, repr(v)[:50], getattr(tree, k) if hasattr(tree, k) else None
                            )
                            for k, v in args.items()
                        ]),
                        traceback.format_exc()
                    )
                )

    if isinstance(tree, list):
        return [convertPyAstToAlgebraic(x, fname, keepLineInformation) for x in tree]

    return tree


def stripDecoratorFromFuncDef(ast):
    """Strip any decorator_list elements from a Statement.FunctionDef.

    Args:
        ast - a Statement.FunctionDef or an Expr.Lambda

    Returns:
        same type as ast but without any decorator_list elements.
    """
    if not ast.matches.FunctionDef:
        return ast

    return Statement.FunctionDef(
        name=ast.name,
        args=ast.args,
        body=ast.body,
        decorator_list=(),  # strip decorators here
        returns=ast.returns,
        line_number=ast.line_number,
        col_offset=ast.col_offset,
        filename=ast.filename,
    )


# a map from (code) -> algebraic ast
_codeToAlgebraicAst = {}
_codeToAlgebraicAstWithoutLineInfo = {}


def convertFunctionToAlgebraicPyAst(f, keepLineInformation=True):
    # we really just care about the code itself
    if isinstance(f, types.FunctionType):
        fCode = f.__code__
    elif isinstance(f, types.CodeType):
        fCode = f
    else:
        raise Exception(
            "convertFunctionToAlgebraicPyAst requires a function object, or a code object."
        )

    if not keepLineInformation:
        if fCode in _codeToAlgebraicAstWithoutLineInfo:
            return _codeToAlgebraicAstWithoutLineInfo[fCode]

        algebraic = convertFunctionToAlgebraicPyAst(f)

        _codeToAlgebraicAstWithoutLineInfo[fCode] = convertPyAstToAlgebraic(
            convertAlgebraicToPyAst(algebraic),
            "",
            False
        )

        return _codeToAlgebraicAstWithoutLineInfo[fCode]

    # check if this is in the cache already
    if fCode in _codeToAlgebraicAst:
        return _codeToAlgebraicAst[fCode]

    # it's not. we'll have to build it
    try:
        pyast = python_ast_util.pyAstForCode(fCode)
    except Exception:
        raise Exception("Failed to get source for function %s:\n%s" % (fCode.co_name, traceback.format_exc()))

    try:
        algebraicAst = convertPyAstToAlgebraic(pyast, fCode.co_filename, True)

        # strip any decorators from the function def. They are not actually part of the
        # definition of the code object itself
        algebraicAst = stripDecoratorFromFuncDef(algebraicAst)

        cacheAstForCode(fCode, algebraicAst)
    except Exception as e:
        raise Exception(
            "Failed to convert function at %s:%s:\n%s"
            % (fCode.co_filename, fCode.co_firstlineno, repr(e))
        )

    return _codeToAlgebraicAst[fCode]


# a memo from pyAst to the 'code' object that we evaluate to def it.
# this is only relevant for the versions that do have line numbers
_pyAstToCodeObjectCache = {}


def stripAstArgAnnotations(arg: Arg):
    return Arg.Item(
        arg=arg.arg,
        annotation=None,
        line_number=arg.line_number,
        col_offset=arg.col_offset,
        filename=arg.filename
    )


def stripAstArgsAnnotations(args: Arguments):
    return Arguments.Item(
        args=[stripAstArgAnnotations(x) for x in args.args],
        vararg=stripAstArgAnnotations(args.vararg) if args.vararg is not None else None,
        kwonlyargs=[stripAstArgAnnotations(x) for x in args.kwonlyargs],
        kw_defaults=args.kw_defaults,
        kwarg=stripAstArgAnnotations(args.kwarg) if args.kwarg is not None else None,
        defaults=(),
    )


def evaluateFunctionPyAst(pyAst, globals=None, stripAnnotations=False):
    assert isinstance(pyAst, (Expr.Lambda, Statement.FunctionDef, Statement.AsyncFunctionDef))

    filename = pyAst.filename

    if isinstance(pyAst, Statement.FunctionDef):
        # strip out the decorator definitions. We just want the underlying function
        # object itself.
        pyAstModule = Statement.FunctionDef(
            name=pyAst.name,
            args=stripAstArgsAnnotations(pyAst.args) if stripAnnotations else pyAst.args,
            body=pyAst.body,
            decorator_list=(),
            returns=pyAst.returns if not stripAnnotations else None,
            line_number=pyAst.line_number,
            col_offset=pyAst.col_offset,
            filename=pyAst.filename,
        )
        pyAstModule = Module.Module(body=(pyAstModule,))
    elif isinstance(pyAst, Statement.AsyncFunctionDef):
        # strip out the decorator definitions. We just want the underlying function
        # object itself.
        pyAstModule = Statement.AsyncFunctionDef(
            name=pyAst.name,
            args=stripAstArgsAnnotations(pyAst.args) if stripAnnotations else pyAst.args,
            body=pyAst.body,
            decorator_list=(),
            returns=pyAst.returns if not stripAnnotations else None,
            line_number=pyAst.line_number,
            col_offset=pyAst.col_offset,
            filename=pyAst.filename,
        )
        pyAstModule = Module.Module(body=(pyAstModule,))
    elif isinstance(pyAst, Expr):
        pyAstModule = Module.Expression(body=pyAst)

    globals = dict(globals) if globals is not None else {}

    if pyAstModule.matches.Expression:
        if pyAst not in _pyAstToCodeObjectCache:
            _pyAstToCodeObjectCache[pyAst] = compile(
                convertAlgebraicToPyAst(pyAstModule), filename, 'eval'
            )

        res = eval(_pyAstToCodeObjectCache[pyAst], globals)
    else:
        if pyAst not in _pyAstToCodeObjectCache:
            _pyAstToCodeObjectCache[pyAst] = compile(
                convertAlgebraicToPyAst(pyAstModule), filename, 'exec'
            )

        exec(_pyAstToCodeObjectCache[pyAst], globals)

        res = globals[pyAstModule.body[0].name]

    # extract any inline code constants from the resulting closure and ensure
    # that we know their definitions as well.
    cacheAstForCode(res.__code__, pyAst)

    return res


def replaceFirstComprehensionArg(pyAst):
    """Replace the first expression in a comprehension with a '.0' varlookup.

    In general, when you write an expression like [x for x in EXPR] inside
    of a python function, you get an inner code object that represents the body
    of the list comprehension in the co_consts. This code object gets used to
    execute the inner stackframe of the list comprehension.

    That code object does not contain 'EXPR' - it assumes it gets passed that
    as a variable called '.0'. As a result, we need to make sure we don't embed
    that information in the code object itself.
    """
    def stripComprehension(c: Comprehension):
        return Comprehension.Item(
            target=c.target,
            iter=Expr.Name(id=".0"),
            ifs=c.ifs,
            is_async=c.is_async
        )

    if pyAst.matches.ListComp or pyAst.matches.SetComp or pyAst.matches.GeneratorExp:
        return type(pyAst)(
            elt=pyAst.elt,
            generators=[stripComprehension(pyAst.generators[0])] + list(pyAst.generators[1:]),
            line_number=pyAst.line_number,
            col_offset=pyAst.col_offset,
            filename=pyAst.filename,
        )

    if pyAst.matches.DictComp:
        return type(pyAst)(
            key=pyAst.key,
            value=pyAst.value,
            generators=[stripComprehension(pyAst.generators[0])] + list(pyAst.generators[1:]),
            line_number=pyAst.line_number,
            col_offset=pyAst.col_offset,
            filename=pyAst.filename,
        )

    return pyAst


def cacheAstForCode(code, pyAst):
    """Remember that 'code' is equivalent to pyAst, and also for contained code objects."""
    if code in _codeToAlgebraicAst:
        return

    # we have to import this within the function to break the import cycle
    from typed_python.compiler.python_ast_analysis import extractFunctionDefsInOrder

    codeConstants = [c for c in code.co_consts if isinstance(c, types.CodeType)]

    if isinstance(pyAst, (Statement.FunctionDef, Expr.Lambda, Statement.ClassDef, Statement.AsyncFunctionDef)):
        funcDefs = extractFunctionDefsInOrder(pyAst.body)
    else:
        funcDefs = extractFunctionDefsInOrder(pyAst.generators)

        if pyAst.matches.ListComp or pyAst.matches.SetComp or pyAst.matches.GeneratorExp:
            funcDefs = extractFunctionDefsInOrder(pyAst.elt) + funcDefs

        if pyAst.matches.DictComp:
            funcDefs = (
                extractFunctionDefsInOrder(pyAst.key)
                + extractFunctionDefsInOrder(pyAst.value)
                + funcDefs
            )

    _codeToAlgebraicAst[code] = replaceFirstComprehensionArg(
        stripDecoratorFromFuncDef(pyAst)
    )

    assert len(funcDefs) == len(codeConstants), (
        f"Expected {len(funcDefs)} func defs to cover the "
        f"{len(codeConstants)} code constants we found in "
        f"{code.co_name} in {code.co_filename}:{code.co_firstlineno}"
        f" of type {type(pyAst)}"
    )

    for i in range(len(funcDefs)):
        cacheAstForCode(codeConstants[i], funcDefs[i])


def evaluateFunctionDefWithLocalsInCells(pyAst, globals, locals, stripAnnotations=False):
    # make a new FunctionDef that defines a function
    # def f(l1, l2, ...):  #l1 ... lN in locals
    #   def pyAst():
    #       ...
    #   return pyAst
    #
    # and then call 'f' to get the closure out

    # strip out the decorator definitions. We just want the underlying function
    # object itself.
    if pyAst.matches.FunctionDef:
        statements = [
            Statement.FunctionDef(
                name=pyAst.name,
                args=stripAstArgsAnnotations(pyAst.args) if stripAnnotations else pyAst.args,
                body=pyAst.body,
                decorator_list=(),
                returns=pyAst.returns if not stripAnnotations else None,
                line_number=pyAst.line_number,
                col_offset=pyAst.col_offset,
                filename=pyAst.filename,
            ),
            Statement.Return(value=Expr.Name(id=pyAst.name, ctx=ExprContext.Load()))
        ]
    elif pyAst.matches.AsyncFunctionDef:
        statements = [
            Statement.AsyncFunctionDef(
                name=pyAst.name,
                args=stripAstArgsAnnotations(pyAst.args) if stripAnnotations else pyAst.args,
                body=pyAst.body,
                decorator_list=(),
                returns=pyAst.returns if not stripAnnotations else None,
                line_number=pyAst.line_number,
                col_offset=pyAst.col_offset,
                filename=pyAst.filename,
            ),
            Statement.Return(value=Expr.Name(id=pyAst.name, ctx=ExprContext.Load()))
        ]
    elif pyAst.matches.GeneratorExp or pyAst.matches.ListComp or pyAst.matches.SetComp or pyAst.matches.DictComp:
        # generators and list comprehensions always become functions that yield
        # the elements of the comprehension
        if pyAst.matches.DictComp:
            bodyExpr = Expr.Tuple(elts=(pyAst.key, pyAst.value), ctx=ExprContext.Load())
        else:
            bodyExpr = pyAst.elt

        body = Statement.Expr(value=Expr.Yield(value=bodyExpr))

        for comprehension in pyAst.generators:
            for ifExpr in comprehension.ifs:
                body = Statement.If(
                    test=ifExpr,
                    body=[body],
                    orelse=[]
                )

            body = Statement.For(
                target=comprehension.target,
                iter=comprehension.iter,
                body=[body]
            )

        statements = [
            Statement.FunctionDef(
                name="__typed_python_generator_builder__",
                args=Arguments.Item(
                    vararg=None,
                    kwarg=None
                ),
                body=[body],
                returns=None
            ),
            Statement.Return(value=Expr.Name(id="__typed_python_generator_builder__", ctx=ExprContext.Load()))
        ]
    elif pyAst.matches.Lambda:
        statements = [Statement.Return(value=pyAst)]
    else:
        raise Exception(f"Can't build a python AST out of {type(pyAst)}")

    pyAstBuilder = Statement.FunctionDef(
        name="__typed_python_func_builder__",
        args=Arguments.Item(
            args=[Arg.Item(arg=name, annotation=None) for name in locals],
            vararg=None,
            kwarg=None
        ),
        body=statements,
        returns=None,
        filename=pyAst.filename
    )

    func = evaluateFunctionPyAst(pyAstBuilder, globals)

    inner = func(*[val for name, val in locals.items()])

    cacheAstForCode(inner.__code__, pyAst)

    return inner
