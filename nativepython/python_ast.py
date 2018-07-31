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
from typed_python.algebraic import Alternative
from typed_python.types import TupleOf, Tuple, OneOf

Module = Alternative("Module")
Statement = Alternative("Statement")
Expr = Alternative("Expr")
Arg = Alternative("Arg")
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

Module.define(
  Module = {"body": TupleOf(Statement)},
  Expression = {'body': Expr},
  Interactive = {'body': TupleOf(Statement)},
  Suite = {"body": TupleOf(Statement)}
  )

Statement.define(
  FunctionDef = {
     "name": str,
     "args": Arguments,
     "body": TupleOf(Statement),
     "decorator_list": TupleOf(Expr),
     "returns": OneOf(Expr, None)
  	},

  ClassDef = {
     "name": str,
     "bases": TupleOf(Expr),
     "body": TupleOf(Statement),
     "decorator_list": TupleOf(Expr)
     },

  Return = { "value": OneOf(Expr, None) },
  Delete = {
     "value": TupleOf(Expr)
     },
  Assign = {
     "targets": TupleOf(Expr),
     "value": Expr
     },
  AugAssign = {
     "target": Expr,
     "op": BinaryOp,
     "value": Expr
     },
  Print = {
     "expr": OneOf(Expr, None),
     "values": TupleOf(Expr),
     "nl": int
     },
  For = {
     "target": Expr,
     "iter": Expr,
     "body": TupleOf(Statement),
     "orelse": TupleOf(Statement)
     },
  While = {
     "test": Expr,
     "body": TupleOf(Statement),
     "orelse": TupleOf(Statement)
     },
  If = {
     "test": Expr,
     "body": TupleOf(Statement),
     "orelse": TupleOf(Statement)
     },
  With = {
     "context_expr": Expr,
     "optional_vars": OneOf(Expr, None),
     "body": TupleOf(Statement)
     },

  Raise = {
     "exc": OneOf(Expr, None),
     "cause": OneOf(Expr, None)
     },
  Try = {
     "body": TupleOf(Statement),
     "handlers": TupleOf(ExceptionHandler),
     "orelse": TupleOf(Statement),
     "finalbody": TupleOf(Statement)
     },
  Assert = {
     "test": Expr,
     "msg": OneOf(Expr, None)
     },
  Import = {
     "names": TupleOf(Alias)
     },
  ImportFrom = {
     "module": TupleOf(str),
     "names": TupleOf(Alias),
     "level": OneOf(int, None)
     },
  Global = {
     "names": TupleOf(str)
     },
  Expr = {
     "value": Expr
     },
  Pass = {},
  Break  = {},
  Continue = {}
  )

Expr.define(
  BoolOp = {
   "op": BooleanOp,
   "values": TupleOf(Expr)
   },
  BinOp = {
     "left": Expr,
     "op": BinaryOp,
     "right": Expr
     },
  UnaryOp = {
     "op": UnaryOp,
     "operand": Expr
     },
  Lambda = {
     "args": Arguments,
     "body": Expr
     },
  IfExp = {
     "test": Expr,
     "body": Expr,
     "orelse": Expr
     },
  Dict = {
     "keys": TupleOf(Expr),
     "values": TupleOf(Expr)
     },
  Set = {
     "elts": TupleOf(Expr)
     },
  ListComp = {
     "elt": Expr,
     "generators": TupleOf(Comprehension)
     },
  SetComp = {
     "elt": Expr,
     "generators": TupleOf(Comprehension)
     },
  DictComp = {
     "key": Expr,
     "value": Expr,
     "generators": TupleOf(Comprehension)
     },
  GeneratorExp = {
     "elt": Expr,
     "generators": TupleOf(Comprehension)
     },
  Yield = {
     "value": OneOf(Expr, None)
     },
  Compare = {
     "left": Expr,
     "ops": TupleOf(ComparisonOp),
     "comparators": TupleOf(Expr)
     },
  Call = {
     "func": Expr,
     "args": TupleOf(Expr),
     "keywords": TupleOf(Keyword),
     },
  Num = {"n": NumericConstant},
  Str = {"s": str},
  Attribute = {
     "value": Expr,
     "attr": str,
     "ctx": ExprContext
     },
  Subscript = {
     "value": Expr,
     "slice": Slice,
     "ctx": ExprContext
     },
  Name = {
     "id": str,
     "ctx": ExprContext
     },
  List = {
     "elts": TupleOf(Expr),
     "ctx": ExprContext
     },
  Tuple = {
     "elts": TupleOf(Expr),
     "ctx": ExprContext
     },
  Starred = {
     "value": Expr,
     "ctx": ExprContext
     }
  )

Expr.add_common_field("line_number", int)
Expr.add_common_field("col_offset", int)
Expr.add_common_field("filename", str)
Statement.add_common_field("line_number", int)
Statement.add_common_field("col_offset", int)
Statement.add_common_field("filename", str)

NumericConstant.define(
  Int = {"value": int},
  Long = {"value": str},
  Boolean = {"value": bool},
  None_ = {},
  Float = {"value": float},
  Unknown = {}
  )

ExprContext.define(
  Load = {},
  Store = {},
  Del = {},
  AugLoad = {},
  AugStore = {},
  Param = {}
  )

Slice.define(
  Ellipsis = {},
  Slice = {
       "lower": OneOf(Expr, None),
       "upper": OneOf(Expr, None),
       "step": OneOf(Expr, None)
       },
  ExtSlice = {"dims": TupleOf(Slice)},
  Index = {"value": Expr}
  )

BooleanOp.define(
  And = {},
  Or = {}
  )

BinaryOp.define(
  Add = {},
  Sub = {},
  Mult = {},
  Div = {},
  Mod = {},
  Pow = {},
  LShift = {},
  RShift = {},
  BitOr = {},
  BitXor = {},
  BitAnd = {},
  FloorDiv = {}
  )

UnaryOp.define(
  Invert = {},
  Not = {},
  UAdd = {},
  USub = {}
  )

ComparisonOp.define(
  Eq = {},
  NotEq = {},
  Lt = {},
  LtE = {},
  Gt = {},
  GtE = {},
  Is = {},
  IsNot = {},
  In = {},
  NotIn = {}
  )

Comprehension.define(
  Item = {
    "target": Expr,
    "iter": Expr,
    "conditions": TupleOf(Expr)
    }
  )

ExceptionHandler.define(
  Item = {
    "type": OneOf(Expr, None),
    "name": OneOf(str, None),
    "body": TupleOf(Statement)
    }
  )
ExceptionHandler.add_common_field("line_number", int)
ExceptionHandler.add_common_field("col_offset", int)
ExceptionHandler.add_common_field("filename", str)


Arguments.define(
  Item = {
    "args": TupleOf(Arg),
    "vararg": OneOf(Arg, None),
    "kwonlyargs": TupleOf(Arg),
    "kw_defaults": TupleOf(Expr),
    "kwarg": OneOf(Arg, None),
    "defaults": TupleOf(Expr),
    }
  )

Arg.define(
  Item ={
  'arg': str,
  'annotation': OneOf(Expr, None),
  })
Arg.add_common_field("line_number", int)
Arg.add_common_field("col_offset", int)
Arg.add_common_field("filename", str)

Keyword.define(
  Item = {
    "arg": str,
    "value": Expr
    }
  )

Alias.define(Item = {
    "name": str,
    "asname": OneOf(str, None)
    }
    )

numericConverters = {
    int: NumericConstant.Int,
    bool: NumericConstant.Boolean,
    type(None): lambda x: NumericConstant.None_(),
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
    try:
        return Expr.Str(s=str(s), **kwds)
    except:
        return Expr.Num(
            n=NumericConstant.Unknown(), 
            **kwds
            )

def makeNameConstant(value, **kwds):
  return Expr.Num(n=numericConverters[type(value)](value), **kwds)

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
    ast.arguments: Arguments.Item,
    ast.arg: Arg.Item,
    ast.keyword: Keyword,
    ast.alias: Alias.Item
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
                "Failed to construct %s from %s with arguments\n%s\n\n%s"
                % (converter, type(tree), 
                   "\n".join(["\t%s:%s" % (k,repr(v)[:50]) for k,v in args.items()]), 
                   traceback.format_exc()
                   )
                )

    if isinstance(tree, list):
        return [convertPyAstToAlgebraic(x,fname) for x in tree]
    return tree
