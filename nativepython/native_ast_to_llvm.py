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

import nativepython.native_ast as native_ast
import llvmlite.ir

class RETURNS_DISALLOWED:
    pass

def type_to_llvm_type(t):
    if t.matches.Void:
        return llvmlite.ir.VoidType()

    if t.matches.Struct:
        return llvmlite.ir.LiteralStructType(type_to_llvm_type(t[1]) for t in t.element_types)
        
    if t.matches.Pointer:
        #llvm won't allow a void*, so we model it as a pointer to an empty struct instead
        if t.value_type.matches.Void:
            return llvmlite.ir.PointerType(llvmlite.ir.LiteralStructType(()))

        return llvmlite.ir.PointerType(type_to_llvm_type(t.value_type))
        
    if t.matches.Float and t.bits == 64:
        return llvmlite.ir.DoubleType()
    
    if t.matches.Int:
        return llvmlite.ir.IntType(t.bits)

    assert False, "Can't handle %s yet" % t

strings_ever = [0]
def constant_to_typed_llvm_value(module, builder, c):
    if c.matches.Float and c.bits == 64:
        return TypedLLVMValue(
            llvmlite.ir.Constant(llvmlite.ir.DoubleType(), c.val),
            native_ast.Type.Float(bits=64)
            )
    if c.matches.Float and c.bits == 32:
        return TypedLLVMValue(
            llvmlite.ir.Constant(llvmlite.ir.FloatType(), c.val),
            native_ast.Type.Float(bits=32)
            )
    if c.matches.Int:
        return TypedLLVMValue(
            llvmlite.ir.Constant(llvmlite.ir.IntType(c.bits), c.val),
            native_ast.Type.Int(bits=c.bits, signed=c.signed)
            )
    
    if c.matches.NullPointer:
        nt = native_ast.Type.Pointer(c.value_type)
        t = type_to_llvm_type(nt)
        llvm_c = llvmlite.ir.Constant(t, None)
        return TypedLLVMValue(llvm_c, nt)

    if c.matches.Struct:
        vals = [constant_to_typed_llvm_value(module, builder, t) for _,t in c.elements]

        t = llvmlite.ir.LiteralStructType(type_to_llvm_type(t.llvm_value.type) for t in vals)
        llvm_c = llvmlite.ir.Constant(t, [t.llvm_value for t in vals])

        nt = native_ast.Type.Struct(
            [(c.elements[i][0],vals[i].native_type) for i in xrange(len(vals))]
            )

        return TypedLLVMValue(llvm_c, nt)

    if c.matches.ByteArray:
        byte_array = c.val

        t = llvmlite.ir.ArrayType(llvmlite.ir.IntType(8), len(byte_array) + 1)

        llvm_c = llvmlite.ir.Constant(t, bytearray(byte_array + bytes("\x00")))

        value = llvmlite.ir.GlobalVariable(module, t, "string_constant_%s" % strings_ever[0])
        strings_ever[0] += 1

        value.initializer = llvm_c

        nt = native_ast.Type.Pointer(native_ast.Type.Int(bits=8,signed=False))

        return TypedLLVMValue(
            builder.bitcast(value, llvmlite.ir.PointerType(llvmlite.ir.IntType(8))), 
            nt
            )

    if c.matches.Void:
        return TypedLLVMValue(None, native_ast.Type.Void())

    assert False, c

class TypedLLVMValue(object):
    def __init__(self, llvm_value, native_type):
        object.__init__(self)

        if native_type.matches.Void:
            assert llvm_value is None, llvm_value

        self.llvm_value = llvm_value
        self.native_type = native_type

class TeardownOnReturnBlock:
    def __init__(self, block, parent_teardowns):
        self.incoming_tags = {} #dict from block->name->(True or llvm_value)
        self.block = block
        self.incoming_blocks = set()
        self.parent_teardowns = parent_teardowns

    def accept_incoming(self, block, tags):
        self.incoming_blocks.add(block)
        self.incoming_tags[block] = dict(tags)

    def generate_tags(self, builder):
        assert builder.block == self.block

        tags = {}
        all_tags = set()
        for b in self.incoming_tags:
            for t in self.incoming_tags[b]:
                all_tags.add(t)

        for t in all_tags:
            tags[t] = {}

        for b in self.incoming_tags:
            for t in all_tags:
                if t not in self.incoming_tags[b]:
                    tags[t][b] = False
                else:
                    tags[t][b] = self.incoming_tags[b][t]

        def collapse_tags(t, incoming):
            if all(k is False for k in incoming.values()):
                return None
            if all(k is True for k in incoming.values()):
                return True
            phinode = builder.phi(llvmlite.ir.IntType(1), name='is_initialized.' + t)

            for b in incoming:
                if isinstance(incoming[b], bool):
                    val = llvmlite.ir.Constant(llvmlite.ir.IntType(1),incoming[b])
                else:
                    val = incoming[b]
                phinode.add_incoming(val, b)

            return phinode

        for t in all_tags:
            tags[t] = collapse_tags(t, tags[t])
            if tags[t] is None:
                del tags[t]

        return tags


def expression_to_llvm_ir(
            module, 
            converter, 
            builder, 
            arg_assignments, 
            output_type, 
            external_function_references
            ):
    if output_type.matches.Void:
        return_slot = None
    else:
        return_slot = builder.alloca(type_to_llvm_type(output_type))

    #if populated, we are expected to write our return value to 'return_slot' and jump here
    #on return
    teardowns_on_return_block = [None]

    stack_slots = {}

    #either 'True', or an llvmlite.ir.Value that's a phi node
    tags_initialized = [{}]

    def convert_teardown(teardown):
        if teardown.matches.Always:
            convert(teardown.expr)
        else:
            assert teardown.matches.ByTag

            if teardown.tag in tags_initialized[0]:
                if tags_initialized[0][teardown.tag] is True:
                    convert(teardown.expr)
                else:
                    llvm_value = tags_initialized[0][teardown.tag]

                    with builder.if_then(llvm_value):
                        convert(teardown.expr)

                del tags_initialized[0][teardown.tag]
            
    def convert(expr):
        if expr.matches.Let:
            l = convert(expr.val)

            prior = arg_assignments.get(expr.var,None)
            arg_assignments[expr.var] = l

            res = convert(expr.within)

            if prior is not None:
                arg_assignments[expr.var] = prior
            else:
                del arg_assignments[expr.var]

            return res

        if expr.matches.StackSlot:
            if expr.name not in stack_slots:
                if expr.type.matches.Void:
                    llvm_type = type_to_llvm_type(native_ast.Type.Struct(()))
                else:
                    llvm_type = type_to_llvm_type(expr.type)

                with builder.goto_entry_block():
                    stack_slots[expr.name] = \
                        TypedLLVMValue(
                            builder.alloca(llvm_type,name=expr.name), 
                            native_ast.Type.Pointer(expr.type)
                            )

            assert stack_slots[expr.name].native_type.value_type == expr.type

            return stack_slots[expr.name]

        if expr.matches.Alloca:
            if expr.type.matches.Void:
                llvm_type = type_to_llvm_type(native_ast.Type.Struct(()))
            else:
                llvm_type = type_to_llvm_type(expr.type)

            return TypedLLVMValue(builder.alloca(llvm_type), native_ast.Type.Pointer(expr.type))

        if expr.matches.MakeStruct:
            names_and_args = [(a[0], convert(a[1])) for a in expr.args]
            names_and_types = [(a[0], a[1].native_type) for a in names_and_args]
            exprs = [a[1].llvm_value for a in names_and_args]
            types = [a.type for a in exprs]

            value = llvmlite.ir.Constant(llvmlite.ir.LiteralStructType(types), None)

            for i in xrange(len(exprs)):
                value = builder.insert_value(value, exprs[i], i)

            return TypedLLVMValue(value, native_ast.Type.Struct(names_and_types))

        if expr.matches.Attribute:
            val = convert(expr.left)
            if val.native_type.matches.Struct:
                attr = expr.attr
                for i in xrange(len(val.native_type.element_types)):
                    if val.native_type.element_types[i][0] == attr:
                        return TypedLLVMValue(
                            builder.extract_value(val.llvm_value, i), 
                            val.native_type.element_types[i][1]
                            )
                assert False, "Type %s doesn't have attribute %s" % (val.native_type, attr)
            else:
                assert False, "Can't take attribute on something of type %s" % val.native_type

        if expr.matches.StructElementByIndex:
            val = convert(expr.left)
            if val.native_type.matches.Struct:
                i = expr.index
                return TypedLLVMValue(
                    builder.extract_value(val.llvm_value, i),
                    val.native_type.element_types[i][1]
                    )

        if expr.matches.Store:
            ptr = convert(expr.ptr)
            val = convert(expr.val)

            if not val.native_type.matches.Void:
                builder.store(val.llvm_value, ptr.llvm_value)

            return TypedLLVMValue(None, native_ast.Type.Void())

        if expr.matches.Load:
            ptr = convert(expr.ptr)

            assert ptr.native_type.matches.Pointer, ptr.native_type

            if ptr.native_type.value_type.matches.Void:
                return TypedLLVMValue(None, ptr.native_type.value_type)

            return TypedLLVMValue(builder.load(ptr.llvm_value), ptr.native_type.value_type)
        
        if expr.matches.Constant:
            return constant_to_typed_llvm_value(module, builder, expr.val)

        if expr.matches.Cast:
            l = convert(expr.left)

            if l is None:
                return

            target_type = type_to_llvm_type(expr.to_type)

            if l.native_type == expr.to_type:
                return l

            if l.native_type.matches.Pointer and expr.to_type.matches.Pointer:
                return TypedLLVMValue(builder.bitcast(l.llvm_value, target_type), expr.to_type)

            if l.native_type.matches.Pointer and expr.to_type.matches.Int:
                return TypedLLVMValue(builder.ptrtoint(l.llvm_value, target_type), expr.to_type)

            if l.native_type.matches.Int and expr.to_type.matches.Pointer:
                return TypedLLVMValue(builder.inttoptr(l.llvm_value, target_type), expr.to_type)

            if l.native_type.matches.Float and expr.to_type.matches.Int:
                if expr.to_type.signed:
                    return TypedLLVMValue(builder.fptosi(l.llvm_value, target_type), expr.to_type)
                else:
                    return TypedLLVMValue(builder.fptoui(l.llvm_value, target_type), expr.to_type)

            elif l.native_type.matches.Float and expr.to_type.matches.Float:
                if l.native_type.bits > expr.to_type.bits:
                    return TypedLLVMValue(builder.fptrunc(l.llvm_value, target_type), expr.to_type)
                else:
                    return TypedLLVMValue(builder.fpext(l.llvm_value, target_type), expr.to_type)

            elif l.native_type.matches.Int and expr.to_type.matches.Int:
                if l.native_type.bits < expr.to_type.bits:
                    if expr.to_type.signed:
                        return TypedLLVMValue(builder.sext(l.llvm_value, target_type), expr.to_type)
                    else:
                        return TypedLLVMValue(builder.zext(l.llvm_value, target_type), expr.to_type)
                else:
                    return TypedLLVMValue(builder.trunc(l.llvm_value, target_type), expr.to_type)

            elif l.native_type.matches.Int and expr.to_type.matches.Float:
                if l.native_type.signed:
                    return TypedLLVMValue(builder.sitofp(l.llvm_value, target_type), expr.to_type)
                else:
                    return TypedLLVMValue(builder.uitofp(l.llvm_value, target_type), expr.to_type)

        if expr.matches.Return:
            assert teardowns_on_return_block[0] is not RETURNS_DISALLOWED

            if expr.arg.matches.Null:
                if teardowns_on_return_block[0] is None:
                    builder.ret_void()
                else:
                    teardowns_on_return_block[0].accept_incoming(builder.block, tags_initialized[0])
                    builder.branch(teardowns_on_return_block[0].block)
            else:
                l = convert(expr.arg.val)

                if l is None:
                    return

                if teardowns_on_return_block[0] is None:
                    if l.llvm_value is None:
                        builder.ret_void()
                    else:
                        builder.ret(l.llvm_value)
                else:
                    teardowns_on_return_block[0].accept_incoming(builder.block, tags_initialized[0])

                    if output_type.matches.Void:
                        builder.branch(teardowns_on_return_block[0].block)
                    else:
                        assert return_slot is not None
                        builder.store(l.llvm_value, return_slot)
                        builder.branch(teardowns_on_return_block[0].block)

            return

        if expr.matches.Branch:
            cond = convert(expr.cond)

            cond_llvm = cond.llvm_value

            zero_like = llvmlite.ir.Constant(cond_llvm.type, 0)

            if cond.native_type.matches.Pointer:
                cond_llvm = builder.ptrtoint(cond_llvm, llvmlite.ir.IntType(64))
                cond_llvm = builder.icmp_signed("!=", cond_llvm, zero_like)
            elif cond.native_type.matches.Int:
                if cond_llvm.type.width != 1:
                    cond_llvm = builder.icmp_signed("!=", cond_llvm, zero_like)
            elif cond.native_type.matches.Float:
                cond_llvm = builder.fcmp_signed("!=", cond_llvm, zero_like)
            else:
                return convert(expr.false)
            
            orig_tags = dict(tags_initialized[0])
            true_tags = dict(orig_tags)
            false_tags = dict(orig_tags)

            with builder.if_else(cond_llvm) as (then, otherwise):
                with then:
                    tags_initialized[0] = true_tags
                    true = convert(expr.true)
                    true_tags = tags_initialized[0]
                    true_block = builder.block
                with otherwise:
                    tags_initialized[0] = false_tags
                    false = convert(expr.false)
                    false_tags = tags_initialized[0]
                    false_block = builder.block

            if true is None and false is None:
                tags_initialized[0] = orig_tags

                builder.unreachable()
                return None

            if true is None:
                tags_initialized[0] = false_tags
                return false

            if false is None:
                tags_initialized[0] = true_tags
                return true

            #we need to merge tags
            final_tags = {}
            for tag in set(true_tags.keys() + false_tags.keys()):
                true_val = true_tags.get(tag, False)
                false_val = false_tags.get(tag, False)

                if true_val is True and false_val is True:
                    final_tags[tag] = True
                else:
                    #it's not certain
                    tag_llvm_value = builder.phi(llvmlite.ir.IntType(1), 'is_initialized.' + tag)
                    if isinstance(true_val, bool):
                        true_val = llvmlite.ir.Constant(llvmlite.ir.IntType(1),true_val)
                    if isinstance(false_val, bool):
                        false_val = llvmlite.ir.Constant(llvmlite.ir.IntType(1),false_val)

                    tag_llvm_value.add_incoming(true_val, true_block)
                    tag_llvm_value.add_incoming(false_val, false_block)
                    final_tags[tag] = tag_llvm_value
                
            tags_initialized[0] = final_tags

            assert true.native_type == false.native_type, (true,false)

            if true.native_type.matches.Void:
                return TypedLLVMValue(None, native_ast.Type.Void())

            final = builder.phi(type_to_llvm_type(true.native_type))
            final.add_incoming(true.llvm_value, true_block)
            final.add_incoming(false.llvm_value, false_block)

            return TypedLLVMValue(final, true.native_type)

        if expr.matches.While:
            tags = dict(tags_initialized[0])

            loop_block = builder.append_basic_block()
            
            builder.branch(loop_block)
            builder.position_at_start(loop_block)

            cond = convert(expr.cond)

            with builder.if_else(cond.llvm_value) as (then, otherwise):
                with then:
                    true = convert(expr.while_true)
                    builder.branch(loop_block)
                    
                with otherwise:
                    false = convert(expr.orelse)

            #it's currently illegal to modify the initialized set in a while loop
            assert sorted(tags.keys()) == sorted(tags_initialized[0].keys())

            if false is None:
                builder.unreachable()
                return None

            return false

        if expr.matches.ElementPtr:
            arg = convert(expr.left)
            offsets = [convert(a) for a in expr.offsets]

            def gep_type(native_type, offsets):
                if len(offsets) == 1:
                    if native_type.matches.Pointer:
                        return native_type
                    if native_type.matches.Struct:
                        assert offsets[0].matches.Constant and offsets[0].val.matches.Int
                        i = offsets[0].val.val
                        return native_type.element_types[i][1]
                else:
                    assert native_type.matches.Pointer
                    return native_ast.Type.Pointer(gep_type(native_type.value_type, offsets[1:]))

            return TypedLLVMValue(
                builder.gep(arg.llvm_value, [o.llvm_value for o in offsets]),
                gep_type(arg.native_type, expr.offsets)
                )

        if expr.matches.Variable:
            assert expr.name in arg_assignments, (expr.name, list(arg_assignments.keys()))
            return arg_assignments[expr.name]
            
        if expr.matches.Unaryop:
            operand = convert(expr.operand)
            if expr.op.matches.Add:
                return operand
            if expr.op.matches.Negate:
                if operand.native_type.matches.Int:
                    return TypedLLVMValue(builder.neg(operand.llvm_value), operand.native_type)
                if operand.native_type.matches.Float:
                    return TypedLLVMValue(
                        builder.fmul(operand.llvm_value, llvmlite.ir.DoubleType()(-1.0)), 
                        operand.native_type
                        )

            assert False, "can't apply unary operand %s to %s" % (expr.op, str(operand.native_type))

        if expr.matches.Binop:
            l = convert(expr.l)
            if l is None:
                return
            r = convert(expr.r)
            if r is None:
                return

            for which, rep in [('Gt','>'),('Lt','<'),('GtE','>='),
                               ('LtE','<='),('Eq',"=="),("NotEq","!=")]:
                if getattr(expr.op.matches, which):
                    if l.native_type.matches.Float:
                        return TypedLLVMValue(
                            builder.fcmp_ordered(rep, l.llvm_value, r.llvm_value), 
                            native_ast.Type.Int(bits=1,signed=False)
                            )
                    elif l.native_type.matches.Int:
                        return TypedLLVMValue(
                            builder.icmp_signed(rep, l.llvm_value, r.llvm_value), 
                            native_ast.Type.Int(bits=1,signed=False)
                            )

            for py_op, floatop, intop in [('Add','fadd','add'), ('Mul','fmul','mul'), 
                                          ('Div','fdiv','div'), ('Sub','fsub','sub')]:
                if getattr(expr.op.matches, py_op):
                    assert l.native_type == r.native_type, \
                        "malformed types: expect l&r to be the same but got %s,%s,%s\n\nexpr=%s"\
                            % (py_op, l.native_type, r.native_type, expr)
                    if l.native_type.matches.Float:
                        return TypedLLVMValue(
                            getattr(builder, floatop)(l.llvm_value, r.llvm_value), 
                            l.native_type
                            )
                    elif l.native_type.matches.Int:
                        return TypedLLVMValue(
                            getattr(builder, intop)(l.llvm_value, r.llvm_value), 
                            l.native_type
                            )

        if expr.matches.Call:
            target = expr.target
            args = [convert(a) for a in expr.args]

            for i in xrange(len(args)):
                if args[i] is None:
                    return

            if target.external:
                if target.name not in external_function_references:
                    func_type = llvmlite.ir.FunctionType(
                        type_to_llvm_type(target.output_type),
                        [type_to_llvm_type(x) for x in target.arg_types],
                        var_arg=target.varargs
                        )
                    external_function_references[target.name] = \
                        llvmlite.ir.Function(module, func_type, target.name)

                func = external_function_references[target.name]

            else:
                func = converter._functions_by_name[target.name]
                if func.module is not module:
                    if target.name not in external_function_references:
                        external_function_references[target.name] = \
                            llvmlite.ir.Function(module, func.function_type, func.name)

                    func = external_function_references[target.name]

            try:
                llvm_call_result = builder.call(func, [a.llvm_value for a in args])
            except:
                print "failing while calling ", target.name
                for a in args:
                    print "\t", a.llvm_value, a.native_type
                raise

            if target.output_type.matches.Void:
                llvm_call_result = None

            return TypedLLVMValue(llvm_call_result, target.output_type)


        if expr.matches.Sequence:
            res = TypedLLVMValue(None, native_ast.Type.Void())

            for e in expr.vals:
                res = convert(e)
                if res is None:
                    return

            return res

        if expr.matches.Comment:
            return convert(expr.expr)

        if expr.matches.ActivatesTeardown:
            assert expr.name not in tags_initialized[0], "already initialized tag %s" % expr.name
            tags_initialized[0][expr.name] = True
            return TypedLLVMValue(None, native_ast.Type.Void())

        if expr.matches.Finally:
            teardown_block = TeardownOnReturnBlock(
                builder.append_basic_block(),
                teardowns_on_return_block[0]
                )

            teardowns_on_return_block[0] = teardown_block

            finally_result = convert(expr.expr)

            if finally_result is not None:
                for teardown in expr.teardowns:
                    convert_teardown(teardown)

            with builder.goto_block(teardowns_on_return_block[0].block):
                teardowns_on_return_block[0] = RETURNS_DISALLOWED

                old_tags = tags_initialized[0]
                tags_initialized[0] = teardown_block.generate_tags(builder)

                for teardown in expr.teardowns:
                    convert_teardown(teardown)
    
                if teardown_block.parent_teardowns is not None:
                    teardown_block.parent_teardowns.accept_incoming(builder.block, tags_initialized[0])
                    builder.branch(teardown_block.parent_teardowns.block)
                else:
                    if return_slot is None:
                        builder.ret_void()
                    else:
                        builder.ret(builder.load(return_slot))

                tags_initialized[0] = old_tags
                teardowns_on_return_block[0] = teardown_block.parent_teardowns

            return finally_result

        assert False, "can't handle %s" % repr(expr)

    return convert

class Converter(object):
    def __init__(self):
        object.__init__(self)
        self._modules = {}
        self._functions_by_name = {}
        self.verbose = False

    def add_functions(self, names_to_definitions):
        for name in names_to_definitions:
            assert name not in self._functions_by_name, "can't define %s twice" % name

        module_name = "module_%s" % len(self._modules)

        module = llvmlite.ir.Module(name=module_name)

        self._modules[module_name] = module

        external_function_references = {}

        for name, function in names_to_definitions.iteritems():
            func_type = llvmlite.ir.FunctionType(
                type_to_llvm_type(function.output_type),
                [type_to_llvm_type(x[1]) for x in function.args]
                )
            self._functions_by_name[name] = llvmlite.ir.Function(module, func_type, name)
            self._functions_by_name[name].linkage =  'external'


        if self.verbose:
            for name in names_to_definitions:
                definition = names_to_definitions[name]
                func = self._functions_by_name[name]

                print
                print "*************"
                print "def %s(%s): #->%s" % (name, 
                            ",".join(["%s=%s" % (k,str(t)) for k,t in definition.args]),
                            str(definition.output_type)
                            )
                print native_ast.indent(str(definition.body.body))
                print "*************"
                print

        for name in names_to_definitions:
            definition = names_to_definitions[name]
            func = self._functions_by_name[name]

            arg_assignments = {}
            for i in xrange(len(func.args)):
                arg_assignments[definition.args[i][0]] = \
                        TypedLLVMValue(func.args[i], definition.args[i][1])

            block = func.append_basic_block('entry')
            builder = llvmlite.ir.IRBuilder(block)

            try:
                res = expression_to_llvm_ir(
                    module, 
                    self, 
                    builder, 
                    arg_assignments, 
                    definition.output_type,
                    external_function_references
                    )(definition.body.body)


                if res is not None:
                    assert res.native_type == native_ast.Type.Struct(()) or\
                            res.native_type.matches.Void, res.native_type
                    builder.ret_void()

            except Exception as e:
                print "function failing = " + name
                raise


        return str(module)


