#   Copyright 2023 typed_python Authors
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


from typed_python import TupleOf, ListOf, Tuple, Dict, ConstDict, NamedTuple

from typed_python.compiler.native_compiler.native_ast import (
    Constant,
    Type,
    UnaryOp,
    BinaryOp,
    NamedCallTarget,
    Expression,
    Teardown,
    ExpressionIntermediate,
    CallTarget,
    FunctionBody,
    Function,
    GlobalVariableMetadata
)


def visitAstChildren(node, callback):
    if not callback(node):
        return

    # don't look in these
    if isinstance(node, (UnaryOp, BinaryOp, Constant, Type, NamedCallTarget, GlobalVariableMetadata)):
        return

    if isinstance(node, (int, float, str, bytes, bool, type(None))):
        return

    if isinstance(node, Function):
        visitAstChildren(node.args, callback)
        visitAstChildren(node.body, callback)
        visitAstChildren(node.output_type, callback)
        return

    if isinstance(node, (Expression, Teardown, ExpressionIntermediate, CallTarget, FunctionBody)):
        for name in node.ElementType.ElementNames:
            visitAstChildren(getattr(node, name), callback)
        return

    if isinstance(node, (Dict, ConstDict, dict)):
        for k, v in node.items():
            visitAstChildren(k, callback)
            visitAstChildren(v, callback)
        return

    if isinstance(node, (TupleOf, ListOf, tuple, list, Tuple, NamedTuple)):
        for child in node:
            visitAstChildren(child, callback)
        return

    raise Exception(f"Unexpected AST node of type {type(node).__name__}")


def extractNamedCallTargets(ast):
    targets = set()

    def check(node):
        if isinstance(node, NamedCallTarget):
            targets.add(node)
        return True

    visitAstChildren(ast, check)

    return targets
