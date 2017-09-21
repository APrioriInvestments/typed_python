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

from nativepython.algebraic import Alternative, List, Nullable

def indent(text, amount=4, ch=' '):
    padding = amount * ch
    return ''.join(padding+line for line in text.splitlines(True))

Type = Alternative("Type")
Type.Void = {}
Type.Float = {'bits': int}
Type.Int = {'bits': int, 'signed': bool}
Type.Struct = {'element_types': List((str, Type))}
Type.Pointer = {'value_type': Type}

def type_attr_ix(t,attr):
    for i in xrange(len(t.element_types)):
        if t.element_types[i][0] == attr:
            return i
    return None
Type.attr_ix = type_attr_ix

def type_str(c):
    if c.matches.Float:
        return "float" + str(c.bits)
    if c.matches.Int:
        return ("int" if c.signed else "uint") + str(c.bits)
    if c.matches.Struct:
        return "(" + ",".join("%s=%s"%(k,v) for k,v in c.element_types) + ")"
    if c.matches.Pointer:
        return "*" + str(c.value_type)
    if c.matches.Void:
        return "void"

Type.__str__ = type_str

Constant = Alternative("Constant")
Constant.Void = {}
Constant.Float = {'val': float, 'bits': int}
Constant.Int = {'val': int, 'bits': int, 'signed': bool}
Constant.Struct = {'elements': List((str, Constant))}
Constant.ByteArray = {'val': bytes}
Constant.NullPointer = {'value_type': Type}

def const_truth_value(c):
    if c.matches.Float:
        return c.val != 0.0
    if c.matches.Int:
        return c.val != 0
    return False

Constant.truth_value = const_truth_value

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
        return "(" + ",".join("%s=%s"%(k,v) for k,v in c.elements) + ")"
    if c.matches.NullPointer:
        return "nullptr"
    if c.matches.Void:
        return "void"

Constant.__str__ = const_str

UnaryOp = Alternative("UnaryOp", Add={}, Negate={})
BinaryOp = Alternative("BinaryOp", 
                       Add={}, Sub={}, Mul={}, Div={}, Eq={}, 
                       NotEq={}, Lt={}, LtE={}, Gt={}, GtE={}
                       )

UnaryOp.__str__ = (lambda o:
    "+" if o.matches.Add else
    "-" if o.matches.Negate else None)
BinaryOp.__str__ = (lambda o:
    "+" if o.matches.Add else
    "-" if o.matches.Sub else 
    "*" if o.matches.Mul else 
    "/" if o.matches.Div else 
    "==" if o.matches.Eq else 
    "!=" if o.matches.NotEq else 
    "<" if o.matches.Lt else 
    "<=" if o.matches.LtE else 
    ">" if o.matches.Gt else 
    ">=" if o.matches.GtE else 
    None)

CallTarget = Alternative(
        "CallTarget", 
        Item = {'name': str, 
                'arg_types': List(Type), 
                'output_type': Type, 
                'external': bool, 
                'varargs': bool
                }
            )

#loads and stores - no assignments
Expression = Alternative("Expression")
Teardown = Alternative("Teardown")


Teardown.ByTag = {'tag': str, 'expr': Expression}
Teardown.Always = {'expr': Expression}

Expression.Constant = {'val': Constant}
Expression.Comment = {'comment': str, 'expr': Expression}
Expression.Load = {'ptr': Expression}
Expression.Store = {'ptr': Expression, 'val': Expression}
Expression.Alloca = {'type': Type}
Expression.Cast = {'left': Expression, 'to_type': Type}
Expression.Binop = {'op': BinaryOp, 'l': Expression, 'r': Expression}
Expression.Unaryop = {'op': UnaryOp, 'operand': Expression}
Expression.Variable = {'name': str}
Expression.Attribute = {'left': Expression, 'attr': str}
Expression.StructElementByIndex = {'left': Expression, 'index': int}
Expression.ElementPtr = {'left': Expression, 'offsets': List(Expression)}
Expression.Call = {'target': CallTarget, 'args': List(Expression)}
Expression.MakeStruct = {'args': List((str,Expression))}
Expression.Branch = {'cond': Expression, 
                     'true': Expression, 
                     'false': Expression
                     }
Expression.While = {'cond': Expression, 
                    'while_true': Expression, 
                    'orelse': Expression
                    }
Expression.Return = {'arg': Nullable(Expression)}
Expression.Let = {'var': str, 'val': Expression, 'within': Expression}

#evaluate 'expr', and then call teardowns if we passed through a named 'ActivatesTeardown'
#clause
Expression.Finally = {'expr': Expression, 'teardowns': List(Teardown)}
Expression.Sequence = {'vals': List(Expression)}

Expression.ActivatesTeardown = {'name': str}
Expression.StackSlot = {'name': str, 'type': Type}

def expr_add(self, other):
    if self.matches.Constant:
        return other

    if self.matches.Sequence and other.matches.Sequence:
        return Expression.Sequence(self.vals + other.vals)
    if self.matches.Sequence:
        return Expression.Sequence(self.vals + (other,))
    if other.matches.Sequence:
        return Expression.Sequence((self,) + other.vals)
    return Expression.Sequence((self,other))

def teardown_str(self):
    if self.matches.Always:
        return str(self.expr)
    if self.matches.ByTag:
        return "if slot_initialized(name=%s):\n" % self.tag + indent(str(self.expr))
Teardown.__str__ = teardown_str

def expr_str(self):
    if self.matches.Comment:
        e = str(self.expr)
        if "\n" in self.comment or "\n" in e:
            return "\n\t/*" + self.comment + "*/\n"\
                    + indent(str(self.expr).rstrip()) + "\n"
        else:
            return "/*" + str(self.comment) + "*/" + str(self.expr)
    if self.matches.Constant:
        return str(self.val)
    if self.matches.Load:
        return "(" + str(self.ptr) + ").load"
    if self.matches.Store:
        return "(" + str(self.ptr) + ")[0]=" + str(self.val)
    if self.matches.Alloca:
        return "alloca(" + str(self.type) + ")"
    if self.matches.Cast:
        return "cast(%s,%s)" % (str(self.left), str(self.to_type))
    if self.matches.Binop:
        return "((%s)%s(%s))" % (str(self.l), str(self.op), str(self.r))
    if self.matches.Unaryop:
        return "(%s(%s))" % (str(self.op), str(self.operand))
    if self.matches.Variable:
        return self.name
    if self.matches.Attribute:
        return "(" + str(self.left) + ")." + self.attr
    if self.matches.ElementPtr:
        return "(" + str(self.left) + ").gep" + \
            "(" + ",".join(str(r) for r in self.offsets) + ")"
    if self.matches.StructElementByIndex:
        return "(" + str(self.left) + ")[%s]" % self.index
    if self.matches.Call:
        return "call(" + str(self.target.name) + "->"\
            + str(self.target.output_type) + ")"\
            + "(" + ",".join(str(r) for r in self.args) + ")"

    if self.matches.Branch:
        t = str(self.true)
        f = str(self.false)
        if "\n" in t or "\n" in f:
            return "if " + str(self.cond) + ":\n"\
                    + indent(t).rstrip() + "\nelse:\n" + indent(f).rstrip()
        else:
            return "((%s) if %s else (%s))" % (t,str(self.cond),f)
    if self.matches.MakeStruct:
        return "struct(" + \
                ",".join("%s=%s" % (k,str(v)) for k,v in self.args)  + ")"
    if self.matches.While:
        t = str(self.while_true)
        f = str(self.orelse)
        return "while " + str(self.cond) + ":\n"\
               + indent(t).rstrip() + "\nelse:\n" + indent(f).rstrip()
    if self.matches.Return:
        s = (str(self.arg.val) if self.arg.matches.Value else "")
        if "\n" in s:
            return "return (" + s + ")"
        return "return " + s
    if self.matches.Let:
        if self.val.matches.Sequence and len(self.val.vals) > 1:
            return str(
                Expression.Sequence(
                    self.val.vals[:-1] + 
                    (Expression.Let(
                        var=self.var,
                        val=self.val.vals[-1],
                        within=self.within),)
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
        return "try:\n" + indent(str(self.expr)) + "\nfinally:\n"\
        + indent("\n".join(str(x) for x in self.teardowns))
    if self.matches.ActivatesTeardown:
        return "mark slot %s initialized" % self.name
    if self.matches.StackSlot:
        return "slot(name=%s,t=%s)" % (self.name, str(self.type))

    assert False

def expr_with_comment(self, c):
    return Expression.Comment(comment=c, expr=self)

def expr_load(self):
    return Expression.Load(self)

Expression.load = expr_load
Expression.__add__ = expr_add
Expression.__str__ = expr_str
Expression.with_comment = expr_with_comment
Expression.ElementPtrIntegers = (lambda self, *offsets: 
    Expression.ElementPtr(
    left=self,
    offsets=tuple(
        Expression.Constant(
                Constant.Int(bits=32,signed=True,val=index)
                )
            for index in offsets
            )
        )
    )


nullExpr = Expression.Constant(Constant.Void())

FunctionBody = Alternative("FunctionBody",
    Internal = {'body': Expression},
    External = {'name': str}
    )

Function = Alternative("Function")
Function.Definition = {'args': List((str, Type)), 'body': FunctionBody, 'output_type': Type}
