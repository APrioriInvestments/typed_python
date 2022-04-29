#   Copyright 2017-2020 typed_python Authors
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

from typed_python import Tuple, TupleOf, Alternative, NamedTuple, OneOf, Forward
import textwrap


def indent(x, indentBy="    "):
    return textwrap.indent(str(x), indentBy)


def type_attr_ix(t, attr):
    for i in range(len(t.element_types)):
        if t.element_types[i][0] == attr:
            return i
    return None


def type_str(c):
    if c.matches.Function:
        return "func((%s)->%s%s)" % (
            ",".join([str(x) for x in (c.args + tuple(["..."] if c.varargs else []))]),
            str(c.output),
            ",nothrow" if not c.can_throw else ""
        )
    if c.matches.Float:
        return "float" + str(c.bits)
    if c.matches.Int:
        return ("int" if c.signed else "uint") + str(c.bits)
    if c.matches.Struct:
        if c.name:
            return c.name
        return "(" + ",".join("%s=%s"%(k, v) for k, v in c.element_types) + ")"
    if c.matches.Pointer:
        return "*" + str(c.value_type)
    if c.matches.Void:
        return "void"
    if c.matches.Array:
        return str(c.element_type) + "[" + str(c.count) + "]"

    assert False, type(c)


def raising(e):
    raise e


Type = Forward("Type")
Type = Type.define(Alternative(
    "Type",
    Void={},
    Float={'bits': int},
    Int={'bits': int, 'signed': bool},
    Struct={'element_types': TupleOf(Tuple(str, Type)), 'name': str},
    Array={'element_type': Type, 'count': int},
    Function={'output': Type, 'args': TupleOf(Type), 'varargs': bool, 'can_throw': bool},
    Pointer={'value_type': Type},
    attr_ix=type_attr_ix,
    __str__=type_str,
    pointer=lambda self: Type.Pointer(value_type=self),
    zero=lambda self: Expression.Constant(self.zeroConstant()),
    zeroConstant=lambda self:
        Constant.Void() if self.matches.Void else
        Constant.Float(val=0.0, bits=self.bits) if self.matches.Float else
        Constant.Int(val=0, bits=self.bits, signed=self.signed) if self.matches.Int else
        Constant.Struct(elements=[(name, t.zeroConstant()) for name, t in self.element_types]) if self.matches.Struct else
        Constant.NullPointer(value_type=self.value_type) if self.matches.Pointer else
        Constant.Array(value_type=self, values=[self.element_type.zeroConstant()] * self.count) if self.matches.Array else
        raising(Exception("Can't make a zero value from %s" % self))
))


def const_truth_value(c):
    if c.matches.Int:
        return c.val != 0
    if c.matches.Float:
        return c.val != 0
    return False


def const_str(c):
    if c.matches.Float:
        if c.bits == 64:
            return str(c.val)
        else:
            return str(c.val) + "f32"
    if c.matches.ByteArray:
        return "ByteArray(%s)" % repr(c.val)
    if c.matches.Int:
        if c.bits == 64:
            return str(c.val)
        else:
            return str(c.val) + ("s" if c.signed else "u") + str(c.bits)
    if c.matches.Struct:
        return "(" + ",".join("%s=%s"%(k, v) for k, v in c.elements) + ")"
    if c.matches.NullPointer:
        return "nullptr"
    if c.matches.Void:
        return "void"
    if c.matches.Array:
        return "Array[" + ",".join(str(x) for x in c.values) + "]"

    assert False, type(c)


Constant = Forward("Constant")
Constant = Constant.define(Alternative(
    "Constant",
    Void={},
    Float={'val': float, 'bits': int},
    Int={'val': int, 'bits': int, 'signed': bool},
    Struct={'elements': TupleOf(Tuple(str, Constant))},
    ByteArray={'val': bytes},
    Array={'value_type': Type, 'values': TupleOf(Constant)},
    NullPointer={'value_type': Type},
    truth_value=const_truth_value,
    __str__=const_str
))

UnaryOp = Alternative(
    "UnaryOp",
    Add={},
    Negate={},
    LogicalNot={},
    BitwiseNot={},
    __str__=lambda o:
        "+" if o.matches.Add else
        "-" if o.matches.Negate else
        "!" if o.matches.LogicalNot else
        "~" if o.matches.BitwiseNot else "unknown unary op"
)

BinaryOp = Alternative(
    "BinaryOp",
    Add={}, Sub={}, Mul={}, Div={}, FloorDiv={}, Eq={},
    NotEq={}, Lt={}, LtE={}, Gt={}, GtE={},
    Mod={}, LShift={}, RShift={},
    BitOr={}, BitAnd={}, BitXor={},
    __str__=lambda o:
        "+" if o.matches.Add else
        "-" if o.matches.Sub else
        "*" if o.matches.Mul else
        "/" if o.matches.Div else
        "//" if o.matches.FloorDiv else
        "==" if o.matches.Eq else
        "!=" if o.matches.NotEq else
        "<" if o.matches.Lt else
        "<=" if o.matches.LtE else
        ">" if o.matches.Gt else
        ">=" if o.matches.GtE else
        "<<" if o.matches.LShift else
        ">>" if o.matches.RShift else
        "|" if o.matches.BitOr else
        "&" if o.matches.BitAnd else
        "^" if o.matches.BitXor else
        "unknown binary op"
)


# loads and stores - no assignments
Expression = Forward("Expression")
Teardown = Forward("Teardown")

NamedCallTarget = NamedTuple(
    name=str,
    arg_types=TupleOf(Type),
    output_type=Type,
    external=bool,
    varargs=bool,
    intrinsic=bool,
    can_throw=bool
)


def intm_return_targets(intm):
    if intm.matches.Effect:
        return intm.expr.returnTargets()

    if intm.matches.Terminal:
        return intm.expr.returnTargets()

    if intm.matches.Simple:
        return intm.expr.returnTargets()

    if intm.matches.StackSlot:
        return intm.expr.returnTargets()

    assert False, f"Unrecognized expression intermediate {intm}"


def intm_could_throw(intm):
    if intm.matches.Effect:
        return intm.expr.couldThrow()

    if intm.matches.Terminal:
        return intm.expr.couldThrow()

    if intm.matches.Simple:
        return intm.expr.couldThrow()

    if intm.matches.StackSlot:
        return intm.expr.couldThrow()

    assert False, f"Unrecognized expression intermediate {intm}"


ExpressionIntermediate = Alternative(
    "ExpressionIntermediate",
    Effect={"expr": Expression},
    Terminal={"expr": Expression},
    Simple={"name": str, "expr": Expression},
    StackSlot={"name": str, "expr": Expression},
    returnTargets=intm_return_targets,
    couldThrow=intm_could_throw,
)


def filterCallTargetArgs(args):
    """Given a list of native expressions or typed expressions, filter them down,
    dropping 'empty' arguments, as per our calling convention."""
    res = []
    for a in args:
        if isinstance(a, int):
            res.append(const_int_expr(a))
        elif isinstance(a, float):
            res.append(const_float_expr(a))
        elif isinstance(a, Expression):
            res.append(a)
        elif a.expr_type.is_empty:
            pass
        else:
            if a.expr_type.is_pass_by_ref:
                assert a.isReference
                res.append(a.expr)
            else:
                res.append(a.nonref_expr)
    return res


CallTarget = Alternative(
    "CallTarget",
    Named={'target': NamedCallTarget},
    Pointer={'expr': Expression},
    call=lambda self, *args: Expression.Call(target=self, args=filterCallTargetArgs(args))
)


def teardown_str(self):
    if self.matches.Always:
        return str(self.expr)
    if self.matches.ByTag:
        return "if slot_initialized(name=%s):\n" % self.tag + indent(str(self.expr), "    ")
    assert False, type(self)


Teardown = Teardown.define(Alternative(
    "Teardown",
    ByTag={'tag': str, 'expr': Expression},
    Always={'expr': Expression},
    __str__=teardown_str
))


def expr_concatenate(self, other):
    return makeSequence((self, other))


def makeSequence(elts):
    if len(elts) == 0:
        return nullExpr
    if len(elts) == 1:
        return elts[0]

    sequenceItems = []
    for e in elts:
        if e.matches.Sequence:
            sequenceItems.extend(e.vals)
        else:
            sequenceItems.append(e)

    sequenceItems = [e for e in sequenceItems[:-1] if not e.matches.Constant] + [sequenceItems[-1]]

    if len(sequenceItems) == 1:
        return sequenceItems[0]

    return Expression.Sequence(vals=sequenceItems)


def expr_str(self):
    if self.matches.Comment:
        e = str(self.expr)
        if "\n" in self.comment or "\n" in e:
            return "\n/*" + self.comment + "*/\n" + str(self.expr).rstrip()
        else:
            return "/*" + str(self.comment) + "*/" + str(self.expr)
    if self.matches.Constant:
        return str(self.val)
    if self.matches.Load:
        return "(" + str(self.ptr) + ").load"
    if self.matches.Store:
        return "(" + str(self.ptr) + ")[0]=" + str(self.val)
    if self.matches.AtomicAdd:
        return "atomic_add(" + str(self.ptr) + "," + str(self.val) + ")"
    if self.matches.Alloca:
        return "alloca(" + str(self.type) + ")"
    if self.matches.Cast:
        return "cast(%s,%s)" % (str(self.left), str(self.to_type))
    if self.matches.Binop:
        return "((%s)%s(%s))" % (str(self.left), str(self.op), str(self.right))
    if self.matches.Unaryop:
        return "(%s(%s))" % (str(self.op), str(self.operand))
    if self.matches.Variable:
        return self.name
    if self.matches.ElementPtr:
        return "(" + str(self.left) + ").gep" + \
            "(" + ",".join(str(r) for r in self.offsets) + ")"
    if self.matches.StructElementByIndex:
        return "(" + str(self.left) + ")[%s]" % self.index
    if self.matches.Call:
        if self.target.matches.Named:
            return "call(" + str(self.target.target.name) + "->"\
                + str(self.target.target.output_type) + ")"\
                + "(" + ",".join(str(r) for r in self.args) + ")"
        else:
            return "call(" + str(self.target.expr) + ")"\
                + "(" + ",".join(str(r) for r in self.args) + ")"

    if self.matches.Branch:
        t = str(self.true)
        f = str(self.false)
        c = str(self.cond)
        if "\n" in t or "\n" in f or "\n" in c:
            if "\n" in c:
                c = "(" + indent(indent(c)).strip() + ")"
            return "if " + c + ":\n"\
                + indent(t).rstrip() + ("\nelse:\n" + indent(f).rstrip() if self.false != nullExpr else "")
        else:
            return "((%s) if %s else (%s))" % (t, str(self.cond), f)
    if self.matches.MakeStruct:
        return "struct(" + \
            ",".join("%s=%s" % (k, str(v)) for k, v in self.args) + ")"
    if self.matches.While:
        t = str(self.while_true)
        f = str(self.orelse)
        c = str(self.cond)

        if "\n" in c:
            return "while (" + indent(indent(c)).strip() + "):\n"\
                   + indent(t).rstrip() + "\nelse:\n" + indent(f).rstrip()
        else:
            return "while " + c + ":\n"\
                   + indent(t).rstrip() + "\nelse:\n" + indent(f).rstrip()
    if self.matches.Return:
        if self.blockName is not None:
            return "return to " + self.blockName

        s = (str(self.arg) if self.arg is not None else "")

        if "\n" in s:
            return "return (" + s + ")"

        return "return " + s
    if self.matches.Let:
        if self.val.matches.Sequence and len(self.val.vals) > 1:
            return str(
                Expression.Sequence(
                    vals=(
                        self.val.vals[:-1] +
                        (Expression.Let(
                            var=self.var,
                            val=self.val.vals[-1],
                            within=self.within), )
                    )
                )
            )
        valstr = str(self.val)
        if "\n" in valstr:
            return self.var + " = (\n" + indent(valstr).rstrip() + ")\n"\
                + str(self.within)
        else:
            return self.var + " = " + str(self.val) + "\n" + str(self.within)
    if self.matches.Sequence:
        return "\n".join(str(x) for x in self.vals)
    if self.matches.Finally:
        label = " [label=" + self.name + "]" if self.name else ""
        return (
            "try:\n" + indent(str(self.expr)) + "\nfinally " + label + ":\n"
            + indent("\n".join(str(x) for x in self.teardowns))
        )
    if self.matches.TryCatch:
        return (
            "try:\n" + indent(str(self.expr)) + "\n" +
            "catch %s:\n" % self.varname + indent(str(self.handler)).rstrip()
        )
    if self.matches.FunctionPointer:
        return "&func(name=%s,(%s)->%s%s%s)" % (
            self.target.name,
            ",".join([str(x) for x in (self.target.arg_types + tuple(["..."] if self.target.varargs else []))]),
            str(self.target.output_type),
            ",nothrow" if not self.target.can_throw else "",
            ",intrinsic" if self.target.intrinsic else ""
        )
        return "&func(%s)" % str(self.expr)
    if self.matches.Throw:
        return "throw (%s)" % str(self.expr)
    if self.matches.ActivatesTeardown:
        return "mark slot %s initialized" % self.name
    if self.matches.StackSlot:
        return "slot(name=%s,t=%s)" % (self.name, str(self.type))
    if self.matches.GlobalVariable:
        return "global(name=%s,t=%s)" % (self.name, str(self.type))
    if self.matches.ExceptionPropagator:
        return (
            "try:\n" + indent(str(self.expr)) + "\n" +
            "rethrow with %s:\n" % self.varname + indent(str(self.handler)).rstrip()
        )
    if self.matches.ApplyIntermediates:
        return (
            "apply intermediates:\n" + "\n".join(str(x) for x in self.intermediates) + "\nto " + str(self.base)
        )

    assert False, type(self)


def expr_is_simple(expr):
    if expr.matches.StackSlot:
        return True
    if expr.matches.Constant:
        return True
    if expr.matches.Variable:
        return True
    if expr.matches.ElementPtr:
        return expr_is_simple(expr.left) and all(expr_is_simple(x) for x in expr.offsets)
    if expr.matches.Load:
        return expr_is_simple(expr.ptr)
    if expr.matches.Sequence and len(expr.vals) == 1:
        return expr_is_simple(expr.vals[0])
    if expr.matches.ApplyIntermediates and len(expr.intermediates) == 0:
        return expr_is_simple(expr.base)
    return False


def expr_return_targets(expr):
    if expr.matches.Finally:
        return expr.expr.returnTargets() - set([expr.name])

    if expr.matches.Return:
        if expr.blockName:
            return set([expr.blockName])
        else:
            return set()

    if expr.matches.MakeStruct:
        res = set()
        for _, e in expr.args:
            res |= e.returnTargets()
        return res

    res = set()

    for name in expr.ElementType.ElementNames:
        child = getattr(expr, name)

        if isinstance(child, Expression):
            res |= child.returnTargets()
        elif isinstance(child, TupleOf(Expression)):
            for c in child:
                res |= c.returnTargets()
        elif isinstance(child, TupleOf(ExpressionIntermediate)):
            for c in child:
                res |= c.returnTargets()

    return res


def expr_with_return_target_name(self, name):
    return Expression.Finally(
        expr=self,
        name=name
    )


def expr_could_throw(self):
    if self.matches.Throw:
        return True

    if self.matches.Call:
        return True

    if self.matches.MakeStruct:
        for _, e in self.args:
            if e.couldThrow():
                return True

    for name in self.ElementType.ElementNames:
        child = getattr(self, name)

        if isinstance(child, Expression):
            if child.couldThrow():
                return True
        elif isinstance(child, TupleOf(Expression)):
            for c in child:
                if c.couldThrow():
                    return True
        elif isinstance(child, TupleOf(ExpressionIntermediate)):
            for c in child:
                if c.couldThrow():
                    return True

    return False


Expression = Expression.define(Alternative(
    "Expression",
    Constant={'val': Constant},
    Comment={'comment': str, 'expr': Expression},
    Load={'ptr': Expression},
    Store={'ptr': Expression, 'val': Expression},
    AtomicAdd={'ptr': Expression, 'val': Expression},
    Alloca={'type': Type},
    Cast={'left': Expression, 'to_type': Type},
    Binop={'op': BinaryOp, 'left': Expression, 'right': Expression},
    Unaryop={'op': UnaryOp, 'operand': Expression},
    Variable={'name': str},
    Attribute={'left': Expression, 'attr': str},
    StructElementByIndex={'left': Expression, 'index': int},
    ElementPtr={'left': Expression, 'offsets': TupleOf(Expression)},
    Call={'target': CallTarget, 'args': TupleOf(Expression)},
    FunctionPointer={'target': NamedCallTarget},
    MakeStruct={'args': TupleOf(Tuple(str, Expression))},
    Branch={
        'cond': Expression,
        'true': Expression,
        'false': Expression
    },
    Throw={'expr': Expression },  # throw a pointer.
    # evaluate 'expr', which must have type 'void' if it returns. if it throws an exception,
    # evaluate 'handler', which must also have type 'void' if it returns, with the exception
    # bound to 'varname'
    TryCatch={
        'expr': Expression,
        'varname': str,  # varname is bound to a int8*
        'handler': Expression
    },
    # evaluate 'expr', which can have any type. If it throws, evaluate
    # 'handler', which must propagate the exception upward.
    ExceptionPropagator={
        'expr': Expression,
        'varname': str,  # varname is bound to a int8*
        'handler': Expression
    },
    While={
        'cond': Expression,
        'while_true': Expression,
        'orelse': Expression
    },
    # return control flow to a higher point in the stack. If 'name' is None, exit the function.
    # otherwise, search for the first 'finally' block above it with that name and return to that
    # scope. In that case 'arg' must be 'None'.
    Return={'arg': OneOf(None, Expression), 'blockName': OneOf(None, str)},
    Let={'var': str, 'val': Expression, 'within': Expression},
    # evaluate 'expr', and then call teardowns if we passed through a named 'ActivatesTeardown'
    # clause. If name is nonempty, then we can explicitly 'return' to it
    Finally={'expr': Expression, 'teardowns': TupleOf(Teardown), 'name': OneOf(None, str)},
    Sequence={'vals': TupleOf(Expression)},
    ActivatesTeardown={'name': str},
    StackSlot={'name': str, 'type': Type},
    GlobalVariable={'name': str, 'type': Type, 'metadata': object},
    ApplyIntermediates={'base': Expression, 'intermediates': TupleOf(ExpressionIntermediate)},
    ElementPtrIntegers=lambda self, *offsets:
        Expression.ElementPtr(
            left=self,
            offsets=tuple(
                Expression.Constant(
                    val=Constant.Int(bits=32, signed=True, val=index)
                )
                for index in offsets
            )
    ),
    __rshift__=expr_concatenate,
    __str__=expr_str,
    structElt=lambda self, ix: Expression.StructElementByIndex(left=self, index=ix),
    logical_not=lambda self: Expression.Unaryop(op=UnaryOp.LogicalNot(), operand=self),
    bitwise_not=lambda self: Expression.Unaryop(op=UnaryOp.BitwiseNot(), operand=self),
    negate=lambda self: Expression.Unaryop(op=UnaryOp.Negate(), operand=self),
    mod=lambda self, other: Expression.Binop(op=BinaryOp.Mod(), left=self, right=ensureExpr(other)),
    sub=lambda self, other: Expression.Binop(op=BinaryOp.Sub(), left=self, right=ensureExpr(other)),
    add=lambda self, other: Expression.Binop(op=BinaryOp.Add(), left=self, right=ensureExpr(other)),
    mul=lambda self, other: Expression.Binop(op=BinaryOp.Mul(), left=self, right=ensureExpr(other)),
    div=lambda self, other: Expression.Binop(op=BinaryOp.Div(), left=self, right=ensureExpr(other)),
    eq=lambda self, other: Expression.Binop(op=BinaryOp.Eq(), left=self, right=ensureExpr(other)),
    neq=lambda self, other: Expression.Binop(op=BinaryOp.NotEq(), left=self, right=ensureExpr(other)),
    lt=lambda self, other: Expression.Binop(op=BinaryOp.Lt(), left=self, right=ensureExpr(other)),
    gt=lambda self, other: Expression.Binop(op=BinaryOp.Gt(), left=self, right=ensureExpr(other)),
    lte=lambda self, other: Expression.Binop(op=BinaryOp.LtE(), left=self, right=ensureExpr(other)),
    gte=lambda self, other: Expression.Binop(op=BinaryOp.GtE(), left=self, right=ensureExpr(other)),
    lshift=lambda self, other: Expression.Binop(op=BinaryOp.LShift(), left=self, right=ensureExpr(other)),
    rshift=lambda self, other: Expression.Binop(op=BinaryOp.RShift(), left=self, right=ensureExpr(other)),
    bitand=lambda self, other: Expression.Binop(op=BinaryOp.BitAnd(), left=self, right=ensureExpr(other)),
    bitor=lambda self, other: Expression.Binop(op=BinaryOp.BitOr(), left=self, right=ensureExpr(other)),
    bitxor=lambda self, other: Expression.Binop(op=BinaryOp.BitXor(), left=self, right=ensureExpr(other)),
    load=lambda self: Expression.Load(ptr=self),
    store=lambda self, val: Expression.Store(ptr=self, val=ensureExpr(val)),
    atomic_add=lambda self, val: Expression.AtomicAdd(ptr=self, val=ensureExpr(val)),
    cast=lambda self, targetType: Expression.Cast(left=self, to_type=targetType),
    with_comment=lambda self, c: Expression.Comment(comment=c, expr=self),
    elemPtr=lambda self, *exprs: Expression.ElementPtr(left=self, offsets=[ensureExpr(e) for e in exprs]),
    is_simple=expr_is_simple,
    returnTargets=expr_return_targets,
    withReturnTargetName=expr_with_return_target_name,
    couldThrow=expr_could_throw
))


def ensureExpr(x):
    if isinstance(x, int):
        return const_int_expr(x)
    if isinstance(x, float):
        return const_float_expr(x)
    if isinstance(x, str):
        return const_utf8_cstr(x)
    if isinstance(x, Expression):
        return x
    return x.nonref_expr


nullExpr = Expression.Constant(val=Constant.Void())
emptyStructExpr = Expression.Constant(val=Constant.Struct(elements=[]))
trueExpr = Expression.Constant(val=Constant.Int(bits=1, val=1, signed=False))
falseExpr = Expression.Constant(val=Constant.Int(bits=1, val=0, signed=False))


def const_float_expr(f):
    return Expression.Constant(
        val=Constant.Float(bits=64, val=f)
    )


def const_float32_expr(f):
    return Expression.Constant(
        val=Constant.Float(bits=32, val=f)
    )


def const_int_expr(i):
    return Expression.Constant(
        val=Constant.Int(bits=64, val=i, signed=True)
    )


def const_uint64_expr(i):
    return Expression.Constant(
        val=Constant.Int(bits=64, val=i, signed=False)
    )


def const_uint32_expr(i):
    return Expression.Constant(
        val=Constant.Int(bits=32, val=i, signed=False)
    )


def const_uint16_expr(i):
    return Expression.Constant(
        val=Constant.Int(bits=16, val=i, signed=False)
    )


def const_uint8_expr(i):
    return Expression.Constant(
        val=Constant.Int(bits=8, val=i, signed=False)
    )


def const_int32_expr(i):
    return Expression.Constant(
        val=Constant.Int(bits=32, val=i, signed=True)
    )


def const_int16_expr(i):
    return Expression.Constant(
        val=Constant.Int(bits=16, val=i, signed=True)
    )


def const_int8_expr(i):
    return Expression.Constant(
        val=Constant.Int(bits=8, val=i, signed=True)
    )


def const_bool_expr(i):
    return Expression.Constant(
        val=Constant.Int(bits=1, val=i, signed=False)
    )


def const_utf8_cstr(i):
    return Expression.Constant(
        val=Constant.ByteArray(val=i.encode('utf-8'))
    )


def const_bytes_cstr(i):
    return Expression.Constant(
        val=Constant.ByteArray(val=i)
    )


FunctionBody = Alternative(
    "FunctionBody",
    Internal={'body': Expression},
    External={'name': str}
)

Function = NamedTuple(
    args=TupleOf(Tuple(str, Type)),
    body=FunctionBody,
    output_type=Type
)

Void = Type.Void()
VoidPtr = Void.pointer()
Bool = Type.Int(bits=1, signed=False)
UInt8 = Type.Int(bits=8, signed=False)
UInt8Ptr = UInt8.pointer()
Int8Ptr = Type.Pointer(value_type=Type.Int(bits=8, signed=True))
Float64 = Type.Float(bits=64)
Float32 = Type.Float(bits=32)
Int64 = Type.Int(bits=64, signed=True)
Int32 = Type.Int(bits=32, signed=True)
Int16 = Type.Int(bits=16, signed=True)
Int8 = Type.Int(bits=8, signed=True)
UInt64 = Type.Int(bits=64, signed=False)
UInt32 = Type.Int(bits=32, signed=False)
UInt16 = Type.Int(bits=16, signed=False)
UInt8 = Type.Int(bits=8, signed=False)
Int32Ptr = Int32.pointer()
Int64Ptr = Int64.pointer()


def var(name):
    return Expression.Variable(name=name)
