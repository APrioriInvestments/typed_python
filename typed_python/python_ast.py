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

import ast
import typed_python.compiler.python_ast_util as python_ast_util
import weakref
import types
from typed_python._types import Forward, Alternative, TupleOf, OneOf


# forward declarations.
Module = Forward("Module")
Statement = Forward("Statement")
Expr = Forward("Expr")
Arg = Forward("Arg")
NumericConstant = Forward("NumericConstant")
ExprContext = Forward("ExprContext")
Slice = Forward("Slice")
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

Module = Module.define(Alternative(
    "Module",
    Module={"body": TupleOf(Statement)},
    Expression={'body': Expr},
    Interactive={'body': TupleOf(Statement)},
    Suite={"body": TupleOf(Statement)}
))

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
        "op": BinaryOp,
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
    }
))

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
        "keys": TupleOf(Expr),
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
        "slice": Slice,
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
    }
))

NumericConstant = NumericConstant.define(Alternative(
    "NumericConstant",
    Int={"value": int},
    Long={"value": str},
    Boolean={"value": bool},
    None_={},
    Float={"value": float},
    Unknown={}
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

Slice = Slice.define(Alternative(
    "Slice",
    Ellipsis={},
    Slice={
        "lower": OneOf(Expr, None),
        "upper": OneOf(Expr, None),
        "step": OneOf(Expr, None)
    },
    ExtSlice={"dims": TupleOf(Slice)},
    Index={"value": Expr}
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
        "args": TupleOf(Arg),
        "vararg": OneOf(Arg, None),
        "kwonlyargs": TupleOf(Arg),
        "kw_defaults": TupleOf(Expr),
        "kwarg": OneOf(Arg, None),
        "defaults": TupleOf(Expr)
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
        "value": Expr
    }
))

Alias = Alias.define(Alternative(
    "Alias",
    Item={
        "name": str,
        "asname": OneOf(str, None)
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
    float: lambda x: NumericConstant.Float(value=x)
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
    ast.Ellipsis: Slice.Ellipsis,
    ast.Slice: Slice.Slice,
    ast.ExtSlice: Slice.ExtSlice,
    ast.Index: Slice.Index,
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
        return ast.Num(n=pyAst.n.value)

    if type(pyAst) in reverseConverters:
        return reverseConverters[type(pyAst)](**convertAlgebraicArgs(pyAst, *type(pyAst).ElementType.ElementNames))

    assert False, type(pyAst)


def convertPyAstToAlgebraic(tree, fname, keepLineInformation=True):
    if issubclass(type(tree), ast.AST):
        converter = converters[type(tree)]
        args = {}

        for f in tree._fields:
            args[f] = convertPyAstToAlgebraic(getattr(tree, f), fname, keepLineInformation)

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
            return converter(**args)
        except Exception:
            del args['line_number']
            del args['col_offset']
            del args['filename']

            try:
                return converter(**args)
            except Exception:
                import traceback
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


# a nasty hack to allow us to find the Ast's of functions we have deserialized
# but for which we never had the source code.
_originalAstCache = weakref.WeakKeyDictionary()

# a memo for the pyast -> algebraic ast mapping
_algebraicAstCache = {}


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

    if fCode in _originalAstCache:
        return _originalAstCache[fCode]

    if fCode in _algebraicAstCache:
        return _algebraicAstCache[fCode, keepLineInformation]

    try:
        pyast = python_ast_util.pyAstForCode(fCode)
    except Exception:
        raise Exception("Failed to get source for function %s" % (fCode.co_name))

    try:
        algebraicAst = convertPyAstToAlgebraic(pyast, fCode.co_filename, keepLineInformation)

        _algebraicAstCache[fCode, keepLineInformation] = algebraicAst
        _originalAstCache[fCode] = algebraicAst

        return algebraicAst
    except Exception as e:
        raise Exception(
            "Failed to convert function at %s:%s:\n%s"
            % (fCode.co_filename, fCode.co_firstlineno, repr(e))
        )


# a memo from pyAst to the 'code' object that we evaluate to def it.
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


def cacheAstForCode(code, pyAst):
    """Remember that 'code' is equivalent to pyAst, and also for contained code objects."""

    # we have to import this within the function to break the import cycle
    from typed_python.compiler.python_ast_analysis import extractFunctionDefsInOrder

    codeConstants = [c for c in code.co_consts if isinstance(c, types.CodeType)]

    if isinstance(pyAst, (Statement.FunctionDef, Expr.Lambda, Statement.ClassDef, Statement.AsyncFunctionDef)):
        funcDefs = extractFunctionDefsInOrder(pyAst.body)
    else:
        funcDefs = extractFunctionDefsInOrder(pyAst.generators)

    _originalAstCache[code] = pyAst

    assert len(funcDefs) == len(codeConstants), (
        f"Expected {len(funcDefs)} func defs to cover the "
        f"{len(codeConstants)} code constants we found."
    )

    for i in range(len(funcDefs)):
        cacheAstForCode(codeConstants[i], funcDefs[i])


def evaluateFunctionDefWithLocalsInCells(pyAst, globals, locals, stripAnnotations=False):
    assert isinstance(pyAst, (Statement.FunctionDef, Statement.AsyncFunctionDef, Expr.Lambda))

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
    else:
        statements = [Statement.Return(value=pyAst)]

    pyAstBuilder = Statement.FunctionDef(
        name="__typed_python_func_builder__",
        args=Arguments.Item(
            args=[Arg.Item(arg=name, annotation=None) for name in locals],
            vararg=None,
            kwarg=None
        ),
        body=statements,
        returns=None
    )

    func = evaluateFunctionPyAst(pyAstBuilder, globals)
    inner = func(*[val for name, val in locals.items()])

    return inner
