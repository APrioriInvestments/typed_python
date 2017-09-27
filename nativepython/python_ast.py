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

"""
Convert python AST objects into a more explicit set of
Algebraic types. These are easier to work with than the
Python ast directly.
"""

import ast
from nativepython.algebraic import Alternative, AlternativeInstance, List, Nullable

Module = Alternative("Module")
Statement = Alternative("Statement")
Expr = Alternative("Expr")
NumericConstant = Alternative("NumericConstant")
ExprContext = Alternative("ExprContext")
Slice = Alternative("Slice")
BooleanOp = Alternative("BooleanOp")
BinaryOp = Alternative("BinaryOp")
UnaryOp = Alternative("UnaryOp")
ComparisonOp = Alternative("ComparisonOp")
Comprehension = Alternative("Comprehension")
ExceptionHandler = Alternative("ExceptionHandler")
Arguments = Alternative("Arguments")
Keyword = Alternative("Keyword")
Alias = Alternative("Alias")

Module.Module = {"body": List(Statement)}
Module.Expression = {'body': Expr}
Module.Interactive = {'body': List(Statement)}
Module.Suite = {"body": List(Statement)}

Statement.FunctionDef = {
   "name": str,
   "args": Arguments,
   "body": List(Statement),
   "decorator_list": List(Expr)
	}

Statement.ClassDef = {
   "name": str,
   "bases": List(Expr),
   "body": List(Statement),
   "decorator_list": List(Expr)
   }

Statement.Return = { "value": Nullable(Expr) }
Statement.Delete = {
   "value": List(Expr)
   }
Statement.Assign = {
   "targets": List(Expr),
   "value": Expr
   }
Statement.AugAssign = {
   "target": Expr,
   "op": BinaryOp,
   "value": Expr
   }
Statement.Print = {
   "expr": Nullable(Expr),
   "values": List(Expr),
   "nl": int
   }
Statement.For = {
   "target": Expr,
   "iter": Expr,
   "body": List(Statement),
   "orelse": List(Statement)
   }
Statement.While = {
   "test": Expr,
   "body": List(Statement),
   "orelse": List(Statement)
   }
Statement.If = {
   "test": Expr,
   "body": List(Statement),
   "orelse": List(Statement)
   }
Statement.With = {
   "context_expr": Expr,
   "optional_vars": Nullable(Expr),
   "body": List(Statement)
   }

Statement.Raise = {
   "type": Nullable(Expr),
   "inst": Nullable(Expr),
   "tback": Nullable(Expr)
   }
Statement.TryExcept = {
   "body": List(Statement),
   "handlers": List(ExceptionHandler),
   "orelse": List(Statement)
   }
Statement.TryFinally = {
   "body": List(Statement),
   "finalbody": List(Statement)
   }
Statement.Assert = {
   "test": Expr,
   "msg": Nullable(Expr)
   }
Statement.Import = {
   "names": List(Alias)
   }
Statement.ImportFrom = {
   "module": List(str),
   "names": List(Alias),
   "level": Nullable(int)
   }
Statement.Exec = {
   "body": Expr,
   "globals": Nullable(Expr),
   "locals": Nullable(Expr)
   }
Statement.Global = {
   "names": List(str)
   }
Statement.Expr = {
   "value": Expr
   }
Statement.Pass = {}
Statement.Break  = {}
Statement.Continue = {}

Expr.BoolOp = {
   "op": BooleanOp,
   "values": List(Expr)
   }
Expr.BinOp = {
   "left": Expr,
   "op": BinaryOp,
   "right": Expr
   }
Expr.UnaryOp = {
   "op": UnaryOp,
   "operand": Expr
   }
Expr.Lambda = {
   "args": Arguments,
   "body": Expr
   }
Expr.IfExp = {
   "test": Expr,
   "body": Expr,
   "orelse": Expr
   }
Expr.Dict = {
   "keys": List(Expr),
   "values": List(Expr)
   }
Expr.Set = {
   "elts": List(Expr)
   }
Expr.ListComp = {
   "elt": Expr,
   "generators": List(Comprehension)
   }
Expr.SetComp = {
   "elt": Expr,
   "generators": List(Comprehension)
   }
Expr.DictComp = {
   "key": Expr,
   "value": Expr,
   "generators": List(Comprehension)
   }
Expr.GeneratorExp = {
   "elt": Expr,
   "generators": List(Comprehension)
   }
Expr.Yield = {
   "value": Nullable(Expr)
   }
Expr.Compare = {
   "left": Expr,
   "ops": List(ComparisonOp),
   "comparators": List(Expr)
   }
Expr.Call = {
   "func": Expr,
   "args": List(Expr),
   "keywords": List(Keyword),
   "starargs": Nullable(Expr),
   "kwargs": Nullable(Expr)
   }
Expr.Repr = {
   "value": Expr
   }
Expr.Num = {"n": NumericConstant}
Expr.Str = {"s": str}
Expr.Attribute = {
   "value": Expr,
   "attr": str,
   "ctx": ExprContext
   }
Expr.Subscript = {
   "value": Expr,
   "slice": Slice,
   "ctx": ExprContext
   }
Expr.Name = {
   "id": str,
   "ctx": ExprContext
   }
Expr.List = {
   "elts": List(Expr),
   "ctx": ExprContext
   }
Expr.Tuple = {
   "elts": List(Expr),
   "ctx": ExprContext
   }

Expr.add_common_field("line_number", int)
Expr.add_common_field("col_offset", int)
Expr.add_common_field("filename", str)
Statement.add_common_field("line_number", int)
Statement.add_common_field("col_offset", int)
Statement.add_common_field("filename", str)

NumericConstant.Int = {"value": int}
NumericConstant.Long = {"value": str}
NumericConstant.Boolean = {"value": bool}
NumericConstant.__setattr__("None", {})
NumericConstant.Float = {"value": float}
NumericConstant.Unknown = {}

ExprContext.Load = {}
ExprContext.Store = {}
ExprContext.Del = {}
ExprContext.AugLoad = {}
ExprContext.AugStore = {}
ExprContext.Param = {}

Slice.Ellipsis = {}
Slice.Slice = {
       "lower": Nullable(Expr),
       "upper": Nullable(Expr),
       "step": Nullable(Expr)
       }
Slice.ExtSlice = {"dims": List(Slice)}

Slice.Index = {"value": Expr}

BooleanOp.And = {}
BooleanOp.Or = {}

BinaryOp.Add = {}
BinaryOp.Sub = {}
BinaryOp.Mult = {}
BinaryOp.Div = {}
BinaryOp.Mod = {}
BinaryOp.Pow = {}
BinaryOp.LShift = {}
BinaryOp.RShift = {}
BinaryOp.BitOr = {}
BinaryOp.BitXor = {}
BinaryOp.BitAnd = {}
BinaryOp.FloorDiv = {}

UnaryOp.Invert = {}
UnaryOp.Not = {}
UnaryOp.UAdd = {}
UnaryOp.USub = {}


ComparisonOp.Eq = {}
ComparisonOp.NotEq = {}
ComparisonOp.Lt = {}
ComparisonOp.LtE = {}
ComparisonOp.Gt = {}
ComparisonOp.GtE = {}
ComparisonOp.Is = {}
ComparisonOp.IsNot = {}
ComparisonOp.In = {}
ComparisonOp.NotIn = {}

Comprehension.Item = {
    "target": Expr,
    "iter": Expr,
    "conditions": List(Expr)
    }


ExceptionHandler.Item = {
    "type": Nullable(Expr),
    "name": Nullable(Expr),
    "body": List(Statement)
    }
ExceptionHandler.add_common_field("line_number", int)
ExceptionHandler.add_common_field("col_offset", int)
ExceptionHandler.add_common_field("filename", str)


Arguments.Item = {
    "args": List(Expr),
    "vararg": Nullable(str),
    "kwarg": Nullable(str),
    "defaults": List(Expr)
    }

Keyword.Item = {
    "arg": str,
    "value": Expr
    }

Alias = {
    "name": str,
    "asname": Nullable(str)
    }

numericConverters = {
    int: NumericConstant.Int,
    bool: NumericConstant.Boolean,
    long: lambda x: NumericConstant.Long(str(x)),
    type(None): lambda x: NumericConstant.None(),
    float: NumericConstant.Float
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
    #WARNING: this code deliberately discards unicode information
    try:
        return Expr.Str(s=str(s), **kwds)
    except:
        return Expr.Num(
            n=NumericConstant.Unknown(), 
            **kwds
            )

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
    ast.Print: Statement.Print,
    ast.For: Statement.For,
    ast.While: Statement.While,
    ast.If: Statement.If,
    ast.With: Statement.With,
    ast.Raise: Statement.Raise,
    ast.TryExcept: Statement.TryExcept,
    ast.TryFinally: Statement.TryFinally,
    ast.Assert: Statement.Assert,
    ast.Import: Statement.Import,
    ast.ImportFrom: Statement.ImportFrom,
    ast.Exec: Statement.Exec,
    ast.Global: Statement.Global,
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
    ast.ListComp: Expr.ListComp,
    ast.SetComp: Expr.SetComp,
    ast.DictComp: Expr.DictComp,
    ast.GeneratorExp: Expr.GeneratorExp,
    ast.Yield: Expr.Yield,
    ast.Compare: Expr.Compare,
    ast.Call: Expr.Call,
    ast.Repr: Expr.Repr,
    ast.Num: createPythonAstConstant,
    ast.Str: createPythonAstString,
    ast.Attribute: Expr.Attribute,
    ast.Subscript: Expr.Subscript,
    ast.Name: Expr.Name,
    ast.List: Expr.List,
    ast.Tuple: Expr.Tuple,
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
    ast.comprehension: Comprehension,
    ast.excepthandler: lambda x:x,
    ast.ExceptHandler: ExceptionHandler.Item,
    ast.arguments: Arguments,
    ast.keyword: Keyword,
    ast.alias: Alias
    }

def convertPyAstToAlgebraic(tree,fname):
    if issubclass(type(tree), ast.AST):
        converter = converters[type(tree)]
        args = {}

        for f in tree._fields:
            args[f] = convertPyAstToAlgebraic(getattr(tree, f), fname)

        try:
            args['line_number'] = tree.lineno
            args['col_offset'] = tree.col_offset
            args['filename'] = fname
        except AttributeError:
            pass

        try:
            return converter(**args)
        except Exception as e:
            import traceback
            raise UserWarning(
                "Failed to construct %s with arguments\n%s\nwithin %s:\n\n%s"
                % (type(tree), 
                   "\n".join(["\t%s:%s" % (k,repr(v)[:50]) for k,v in args.iteritems()]), 
                   converter, 
                   traceback.format_exc()
                   )
                )

    if isinstance(tree, list):
        return [convertPyAstToAlgebraic(x,fname) for x in tree]
    return tree
