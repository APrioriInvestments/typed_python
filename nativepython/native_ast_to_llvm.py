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

llvm_i8ptr = llvmlite.ir.IntType(8).as_pointer()
llvm_i8 = llvmlite.ir.IntType(8)
llvm_i32 = llvmlite.ir.IntType(32)
llvm_i64 = llvmlite.ir.IntType(64)
llvm_i1 = llvmlite.ir.IntType(1)
llvm_void = llvmlite.ir.VoidType()

exception_type_llvm = llvmlite.ir.LiteralStructType([llvm_i8ptr, llvm_i32])

#just hardcoded for now. We check this in the compiler to ensure it's consistent.
pointer_size = 8

def llvm_bool(i):
    return llvmlite.ir.Constant(llvm_i1, i)

def assertTagDictsSame(left_tags, right_tags):
    for which in [left_tags,right_tags]:
        for tag in which:
            assert tag in left_tags and tag in right_tags and right_tags[tag] is left_tags[tag], "Tag %s is not the same" % tag

def type_to_llvm_type(t):
    if t.matches.Void:
        return llvmlite.ir.VoidType()

    if t.matches.Struct:
        return llvmlite.ir.LiteralStructType(type_to_llvm_type(t[1]) for t in t.element_types)

    if t.matches.Pointer:
        #llvm won't allow a void*, so we model it as a pointer to an empty struct instead
        if t.value_type.matches.Void:
            return llvmlite.ir.PointerType(llvmlite.ir.LiteralStructType(()))

        if t.value_type.matches.Function:
            return llvmlite.ir.PointerType(
                llvmlite.ir.FunctionType(
                    type_to_llvm_type(t.value_type.output),
                    [type_to_llvm_type(x) for x in t.value_type.args],
                    var_arg=t.value_type.varargs
                    )
                )

        return llvmlite.ir.PointerType(type_to_llvm_type(t.value_type))

    if t.matches.Array:
        return llvmlite.ir.ArrayType(type_to_llvm_type(t.element_type), t.count)

    if t.matches.Float and t.bits == 64:
        return llvmlite.ir.DoubleType()

    if t.matches.Float and t.bits == 32:
        return llvmlite.ir.FloatType()

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
        nt = native_ast.Type.Pointer(value_type=c.value_type)
        t = type_to_llvm_type(nt)
        llvm_c = llvmlite.ir.Constant(t, None)
        return TypedLLVMValue(llvm_c, nt)

    if c.matches.Struct:
        vals = [constant_to_typed_llvm_value(module, builder, t) for _,t in c.elements]

        t = llvmlite.ir.LiteralStructType(type_to_llvm_type(t.llvm_value.type) for t in vals)
        llvm_c = llvmlite.ir.Constant(t, [t.llvm_value for t in vals])

        nt = native_ast.Type.Struct(
            [(c.elements[i][0],vals[i].native_type) for i in range(len(vals))]
            )

        return TypedLLVMValue(llvm_c, nt)

    if c.matches.ByteArray:
        byte_array = c.val

        t = llvmlite.ir.ArrayType(llvm_i8, len(byte_array) + 1)

        llvm_c = llvmlite.ir.Constant(t, bytearray(byte_array + b"\x00"))

        value = llvmlite.ir.GlobalVariable(module, t, "string_constant_%s" % strings_ever[0])
        strings_ever[0] += 1

        value.initializer = llvm_c

        nt = native_ast.Type.Int(bits=8,signed=False).pointer()

        return TypedLLVMValue(
            builder.bitcast(value, llvm_i8ptr),
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
        else:
            assert llvm_value is not None

        self.llvm_value = llvm_value
        self.native_type = native_type

    def __str__(self):
        return "TypedLLVMValue(%s)" % self.native_type

    def __repr__(self):
        return str(self)

class TeardownOnScopeExit:
    def __init__(self, converter, parent_scope):
        self.converter = converter
        self.builder = converter.builder

        self.incoming_tags = {} #dict from block->name->(True or llvm_value)

        #is the incoming block from a scope trying to return (True)
        #or propagate an exception (False)
        self.incoming_is_return = {} #dict from block->(True/False or llvm_value)

        self._block = None

        self.incoming_blocks = set()
        self.parent_scope = parent_scope
        self.height = 0 if parent_scope is None else parent_scope.height+1
        self.return_is_generated = False

    def generate_is_return(self):
        assert self._block is not None

        assert self.incoming_is_return

        assert len(self.incoming_is_return) == len(self.incoming_blocks)

        assert not self.return_is_generated
        self.return_is_generated = True

        if len(self.incoming_is_return) == 1:
            return list(self.incoming_is_return.values())[0]

        if all([v is True for v in self.incoming_is_return.values()]):
            return True

        if all([v is False for v in self.incoming_is_return.values()]):
            return False

        with self.builder.goto_block(self._block):
            is_return = self.builder.phi(llvm_i1, name='is_return_flow_%s' % self.height)

            for b,val in self.incoming_is_return.items():
                if isinstance(val, bool):
                    val = llvm_bool(val)
                is_return.add_incoming(val, b)

            return is_return

    def accept_incoming(self, block, tags, is_return):
        assert block not in self.incoming_is_return
        assert not self.return_is_generated

        if isinstance(is_return, llvmlite.ir.Constant):
            is_return = is_return.constant

        self.incoming_is_return[block] = is_return
        self.incoming_blocks.add(block)
        self.incoming_tags[block] = dict(tags)

        if self._block is None:
            self._block = self.builder.append_basic_block("scope_exit_handler_%d" % self.height)
        return self._block

    def generate_tags(self):
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
            if all(not isinstance(k, bool) for k in incoming.values()):
                names = set(k.name for k in incoming.values())
                if len(names) == 1:
                    for i in incoming.values():
                        return i

            phinode = self.builder.phi(llvm_i1, name='is_initialized.' + t)

            for b in incoming:
                if isinstance(incoming[b], bool):
                    val = llvm_bool(incoming[b])
                else:
                    val = incoming[b]
                phinode.add_incoming(val, b)

            return phinode

        for t in all_tags:
            tags[t] = collapse_tags(t, tags[t])
            if tags[t] is None:
                del tags[t]

        return tags

    def generate_teardown(self, teardown_callback, return_slot = None, exception_slot = None):
        if not self.incoming_is_return:
            assert self._block is None
            return

        is_return = self.generate_is_return()

        with self.builder.goto_block(self._block):
            tags = self.generate_tags()

            teardown_callback(tags)

            if self.parent_scope is not None:
                block = self.parent_scope.accept_incoming(
                    self.builder.block,
                    tags,
                    is_return
                    )
                self.builder.branch(block)
            else:
                if is_return is True:
                    if return_slot is None:
                        self.builder.ret_void()
                    else:
                        self.builder.ret(self.builder.load(return_slot))
                elif is_return is False:
                    self.converter.generate_throw_expression(
                        self.builder.load(exception_slot)
                        )
                else:
                    assert isinstance(is_return, llvmlite.ir.Value)
                    with self.builder.if_else(is_return) as (then, otherwise):
                        with then:
                            if return_slot is None:
                                self.builder.ret_void()
                            else:
                                self.builder.ret(self.builder.load(return_slot))
                        with otherwise:
                            self.converter.generate_throw_expression(
                                self.builder.load(exception_slot)
                                )
                    self.builder.unreachable()

    def generate_trycatch_unwind(self, target_resume_block, generator):
        if not self.incoming_is_return:
            assert self._block is None
            return

        is_return = self.generate_is_return()

        with self.builder.goto_block(self._block):
            tags = self.generate_tags()

            if is_return is True:
                block = self.parent_scope.accept_incoming(
                    self.builder.block,
                    tags,
                    is_return
                    )
                self.builder.branch(block)
                return

            if is_return is False:
                generator(tags, target_resume_block)
                return

            assert isinstance(is_return, llvmlite.ir.Value)
            with self.builder.if_then(is_return):
                block = self.parent_scope.accept_incoming(
                    self.builder.block,
                    tags,
                    is_return
                    )
                self.builder.branch(block)

            generator(tags, target_resume_block)

class FunctionConverter:
    def __init__(self,
                module,
                function,
                converter,
                builder,
                arg_assignments,
                output_type,
                external_function_references
                ):
        self.function = function
        self.module = module
        self.converter = converter
        self.builder = builder
        self.arg_assignments = arg_assignments
        self.output_type = output_type
        self.external_function_references = external_function_references
        self.tags_initialized = {}
        self.stack_slots = {}

    def tags_as(self, new_tags):
        class scoper():
            def __init__(scoper_self):
                scoper_self.old_tags = None

            def __enter__(scoper_self, *args):
                scoper_self.old_tags = self.tags_initialized
                self.tags_initialized = new_tags

            def __exit__(scoper_self, *args):
                self.tags_initialized = scoper_self.old_tags

        return scoper()

    def setup(self):
        builder = self.builder

        if self.output_type.matches.Void:
            self.return_slot = None
        else:
            self.return_slot = builder.alloca(type_to_llvm_type(self.output_type))

        self.exception_slot = builder.alloca(llvm_i8ptr,name="exception_slot")

        #if populated, we are expected to write our return value to 'return_slot' and jump here
        #on return
        self.teardown_handler = TeardownOnScopeExit(self, None)

    def finalize(self):
        self.teardown_handler.generate_teardown(lambda tags: None, self.return_slot, self.exception_slot)

    def generate_exception_landing_pad(self, block):
        with self.builder.goto_block(block):
            res = self.builder.landingpad(exception_type_llvm)

            res.add_clause(
                llvmlite.ir.CatchClause(
                    llvmlite.ir.Constant(llvm_i8ptr,None)
                    )
                )

            actual_exception = self.builder.call(
                self.external_function_references["__cxa_begin_catch"],
                [self.builder.extract_value(res, 0)]
                )

            result = self.builder.load(
                self.builder.bitcast(
                    actual_exception,
                    llvm_i8ptr.as_pointer()
                    )
                )

            self.builder.store(result, self.exception_slot)

            self.builder.call(
                self.external_function_references["__cxa_end_catch"],
                [self.builder.extract_value(res, 0)]
                )

            block = self.teardown_handler.accept_incoming(
                self.builder.block,
                self.tags_initialized,
                is_return=False
                )

            self.builder.branch(block)

    def convert_teardown(self, teardown):
        orig_tags = dict(self.tags_initialized)

        if teardown.matches.Always:
            self.convert(teardown.expr)
        else:
            assert teardown.matches.ByTag

            if teardown.tag in self.tags_initialized:
                tagVal = self.tags_initialized[teardown.tag]

                #mark that the tag is no longer active
                del self.tags_initialized[teardown.tag]
                del orig_tags[teardown.tag]

                if tagVal is True:
                    self.convert(teardown.expr)
                else:
                    with self.builder.if_then(tagVal):
                        self.convert(teardown.expr)

        assertTagDictsSame(self.tags_initialized, orig_tags)

    def generate_exception_and_store_value(self, llvm_pointer_val):
        exception_ptr = self.builder.bitcast(
            self.builder.call(
                self.external_function_references["__cxa_allocate_exception"],
                [llvmlite.ir.Constant(llvm_i64,pointer_size)],
                name="alloc_e"
                ),
            llvm_i8ptr.as_pointer()
            )
        self.builder.store(
            self.builder.bitcast(llvm_pointer_val, llvm_i8ptr),
            exception_ptr
            )
        return self.builder.bitcast(exception_ptr, llvm_i8ptr)

    def namedCallTargetToLLVM(self, target):
        if target.external:
            if target.name not in self.external_function_references:
                func_type = llvmlite.ir.FunctionType(
                    type_to_llvm_type(target.output_type),
                    [type_to_llvm_type(x) for x in target.arg_types],
                    var_arg=target.varargs
                    )

                if target.intrinsic:
                    self.external_function_references[target.name] = \
                        self.module.declare_intrinsic(target.name, fnty=func_type)
                else:
                    self.external_function_references[target.name] = \
                        llvmlite.ir.Function(self.module, func_type, target.name)

            func = self.external_function_references[target.name]
        else:
            func = self.converter._functions_by_name[target.name]
            if func.module is not self.module:
                if target.name not in self.external_function_references:
                    self.external_function_references[target.name] = \
                        llvmlite.ir.Function(self.module, func.function_type, func.name)

                func = self.external_function_references[target.name]

        return TypedLLVMValue(
            func,
            native_ast.Type.Function(
                args=target.arg_types,
                output=target.output_type,
                varargs=target.varargs,
                can_throw=target.can_throw
                ).pointer()
            )

    def generate_throw_expression(self, llvm_pointer_val):
        exception_ptr = self.generate_exception_and_store_value(llvm_pointer_val)

        self.builder.call(
            self.external_function_references["__cxa_throw"],
            [exception_ptr] + [llvmlite.ir.Constant(llvm_i8ptr,None)] * 2
            )

        self.builder.unreachable()

    def convert(self, expr):
        if expr.matches.Let:
            l = self.convert(expr.val)

            prior = self.arg_assignments.get(expr.var,None)
            self.arg_assignments[expr.var] = l

            res = self.convert(expr.within)

            if prior is not None:
                self.arg_assignments[expr.var] = prior
            else:
                del self.arg_assignments[expr.var]

            return res

        if expr.matches.StackSlot:
            if expr.name not in self.stack_slots:
                if expr.type.matches.Void:
                    llvm_type = type_to_llvm_type(native_ast.Type.Struct(()))
                else:
                    llvm_type = type_to_llvm_type(expr.type)

                with self.builder.goto_entry_block():
                    self.stack_slots[expr.name] = \
                        TypedLLVMValue(
                            self.builder.alloca(llvm_type,name=expr.name),
                            native_ast.Type.Pointer(value_type=expr.type)
                            )

            assert self.stack_slots[expr.name].native_type.value_type == expr.type, \
                "StackSlot %s supposed to have value %s but got %s" % (
                    expr.name,
                    self.stack_slots[expr.name].native_type.value_type,
                    expr.type
                    )

            return self.stack_slots[expr.name]

        if expr.matches.Alloca:
            if expr.type.matches.Void:
                llvm_type = type_to_llvm_type(native_ast.Type.Struct(()))
            else:
                llvm_type = type_to_llvm_type(expr.type)

            return TypedLLVMValue(self.builder.alloca(llvm_type), native_ast.Type.Pointer(expr.type))

        if expr.matches.MakeStruct:
            names_and_args = [(a[0], self.convert(a[1])) for a in expr.args]
            names_and_types = [(a[0], a[1].native_type) for a in names_and_args]
            exprs = [a[1].llvm_value for a in names_and_args]
            types = [a.type for a in exprs]

            value = llvmlite.ir.Constant(llvmlite.ir.LiteralStructType(types), None)

            for i in range(len(exprs)):
                value = self.builder.insert_value(value, exprs[i], i)

            return TypedLLVMValue(value, native_ast.Type.Struct(names_and_types))

        if expr.matches.Attribute:
            val = self.convert(expr.left)
            if val.native_type.matches.Struct:
                attr = expr.attr
                for i in range(len(val.native_type.element_types)):
                    if val.native_type.element_types[i][0] == attr:
                        return TypedLLVMValue(
                            self.builder.extract_value(val.llvm_value, i),
                            val.native_type.element_types[i][1]
                            )
                assert False, "Type %s doesn't have attribute %s" % (val.native_type, attr)
            else:
                assert False, "Can't take attribute on something of type %s" % val.native_type

        if expr.matches.StructElementByIndex:
            val = self.convert(expr.left)
            if val.native_type.matches.Struct:
                i = expr.index
                return TypedLLVMValue(
                    self.builder.extract_value(val.llvm_value, i),
                    val.native_type.element_types[i][1]
                    )

        if expr.matches.Store:
            ptr = self.convert(expr.ptr)
            val = self.convert(expr.val)

            if not val.native_type.matches.Void:
                self.builder.store(val.llvm_value, ptr.llvm_value)

            return TypedLLVMValue(None, native_ast.Type.Void())

        if expr.matches.Load:
            ptr = self.convert(expr.ptr)

            assert ptr.native_type.matches.Pointer, ptr.native_type

            if ptr.native_type.value_type.matches.Void:
                return TypedLLVMValue(None, ptr.native_type.value_type)

            return TypedLLVMValue(self.builder.load(ptr.llvm_value), ptr.native_type.value_type)

        if expr.matches.Constant:
            return constant_to_typed_llvm_value(self.module, self.builder, expr.val)

        if expr.matches.Cast:
            l = self.convert(expr.left)

            if l is None:
                return

            target_type = type_to_llvm_type(expr.to_type)

            if l.native_type == expr.to_type:
                return l

            if l.native_type.matches.Pointer and expr.to_type.matches.Pointer:
                return TypedLLVMValue(self.builder.bitcast(l.llvm_value, target_type), expr.to_type)

            if l.native_type.matches.Pointer and expr.to_type.matches.Int:
                return TypedLLVMValue(self.builder.ptrtoint(l.llvm_value, target_type), expr.to_type)

            if l.native_type.matches.Int and expr.to_type.matches.Pointer:
                return TypedLLVMValue(self.builder.inttoptr(l.llvm_value, target_type), expr.to_type)

            if l.native_type.matches.Float and expr.to_type.matches.Int:
                if expr.to_type.signed:
                    return TypedLLVMValue(self.builder.fptosi(l.llvm_value, target_type), expr.to_type)
                else:
                    return TypedLLVMValue(self.builder.fptoui(l.llvm_value, target_type), expr.to_type)

            elif l.native_type.matches.Float and expr.to_type.matches.Float:
                if l.native_type.bits > expr.to_type.bits:
                    return TypedLLVMValue(self.builder.fptrunc(l.llvm_value, target_type), expr.to_type)
                else:
                    return TypedLLVMValue(self.builder.fpext(l.llvm_value, target_type), expr.to_type)

            elif l.native_type.matches.Int and expr.to_type.matches.Int:
                if l.native_type.bits < expr.to_type.bits:
                    if expr.to_type.signed:
                        return TypedLLVMValue(self.builder.sext(l.llvm_value, target_type), expr.to_type)
                    else:
                        return TypedLLVMValue(self.builder.zext(l.llvm_value, target_type), expr.to_type)
                else:
                    return TypedLLVMValue(self.builder.trunc(l.llvm_value, target_type), expr.to_type)

            elif l.native_type.matches.Int and expr.to_type.matches.Float:
                if l.native_type.signed:
                    return TypedLLVMValue(self.builder.sitofp(l.llvm_value, target_type), expr.to_type)
                else:
                    return TypedLLVMValue(self.builder.uitofp(l.llvm_value, target_type), expr.to_type)

        if expr.matches.Return:
            if expr.arg is not None:
                #write the value into the return slot
                l = self.convert(expr.arg)

                if l is None:
                    return

                if not self.output_type.matches.Void:
                    assert self.return_slot is not None
                    self.builder.store(l.llvm_value, self.return_slot)

            block = self.teardown_handler.accept_incoming(
                self.builder.block,
                self.tags_initialized,
                is_return=True
                )

            self.builder.branch(block)

            return

        if expr.matches.Branch:
            cond = self.convert(expr.cond)

            cond_llvm = cond.llvm_value

            zero_like = llvmlite.ir.Constant(cond_llvm.type, 0)

            if cond.native_type.matches.Pointer:
                cond_llvm = self.builder.ptrtoint(cond_llvm, llvm_i64)
                cond_llvm = self.builder.icmp_signed("!=", cond_llvm, zero_like)
            elif cond.native_type.matches.Int:
                if cond_llvm.type.width != 1:
                    cond_llvm = self.builder.icmp_signed("!=", cond_llvm, zero_like)
            elif cond.native_type.matches.Float:
                cond_llvm = self.builder.fcmp_unordered("!=", cond_llvm, zero_like)
            else:
                return self.convert(expr.false)

            orig_tags = dict(self.tags_initialized)
            true_tags = dict(orig_tags)
            false_tags = dict(orig_tags)

            with self.builder.if_else(cond_llvm) as (then, otherwise):
                with then:
                    self.tags_initialized = true_tags
                    true = self.convert(expr.true)
                    true_tags = self.tags_initialized
                    true_block = self.builder.block

                with otherwise:
                    self.tags_initialized = false_tags
                    false = self.convert(expr.false)
                    false_tags = self.tags_initialized
                    false_block = self.builder.block

            if true is None and false is None:
                self.tags_initialized = orig_tags

                self.builder.unreachable()
                return None

            if true is None:
                self.tags_initialized = false_tags
                return false

            if false is None:
                self.tags_initialized = true_tags
                return true

            #we need to merge tags
            final_tags = {}
            for tag in set(list(true_tags.keys()) + list(false_tags.keys())):
                true_val = true_tags.get(tag, False)
                false_val = false_tags.get(tag, False)

                if true_val is True and false_val is True:
                    final_tags[tag] = True
                else:
                    #it's not certain
                    if not isinstance(true_val, bool) and not isinstance(false_val, bool) and true_val.name == false_val.name:
                        #these are the same bit that's been passed between two different branches.
                        final_tags[tag] = true_val
                    else:
                        tag_llvm_value = self.builder.phi(llvm_i1, 'is_initialized.merge.' + tag)
                        if isinstance(true_val, bool):
                            true_val = llvm_bool(true_val)
                        if isinstance(false_val, bool):
                            false_val = llvm_bool(false_val)

                        tag_llvm_value.add_incoming(true_val, true_block)
                        tag_llvm_value.add_incoming(false_val, false_block)
                        final_tags[tag] = tag_llvm_value

            self.tags_initialized = final_tags

            if true.native_type != false.native_type:
                raise Exception("Expected left and right branches to have same type, but %s != %s\n\n%s" % (true,false,expr))

            if true.native_type.matches.Void:
                return TypedLLVMValue(None, native_ast.Type.Void())

            final = self.builder.phi(type_to_llvm_type(true.native_type))
            final.add_incoming(true.llvm_value, true_block)
            final.add_incoming(false.llvm_value, false_block)

            return TypedLLVMValue(final, true.native_type)

        if expr.matches.While:
            tags = dict(self.tags_initialized)

            loop_block = self.builder.append_basic_block()

            self.builder.branch(loop_block)
            self.builder.position_at_start(loop_block)

            cond = self.convert(expr.cond)

            cond_llvm = cond.llvm_value

            zero_like = llvmlite.ir.Constant(cond_llvm.type, 0)

            if cond.native_type.matches.Pointer:
                cond_llvm = self.builder.ptrtoint(cond_llvm, llvm_i64)
                cond_llvm = self.builder.icmp_signed("!=", cond_llvm, zero_like)
            elif cond.native_type.matches.Int:
                if cond_llvm.type.width != 1:
                    cond_llvm = self.builder.icmp_signed("!=", cond_llvm, zero_like)
            elif cond.native_type.matches.Float:
                cond_llvm = self.builder.fcmp_unordered("!=", cond_llvm, zero_like)
            else:
                cond_llvm = llvmlite.ir.Constant(llvm_i1, 0)

            with self.builder.if_else(cond_llvm) as (then, otherwise):
                with then:
                    true = self.convert(expr.while_true)
                    self.builder.branch(loop_block)

                with otherwise:
                    false = self.convert(expr.orelse)

            #it's currently illegal to modify the initialized set in a while loop
            assertTagDictsSame(tags, self.tags_initialized)

            if false is None:
                self.builder.unreachable()
                return None

            return false

        if expr.matches.ElementPtr:
            arg = self.convert(expr.left)
            offsets = [self.convert(a) for a in expr.offsets]

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
                    return gep_type(native_type.value_type, offsets[1:]).pointer()

            return TypedLLVMValue(
                self.builder.gep(arg.llvm_value, [o.llvm_value for o in offsets]),
                gep_type(arg.native_type, expr.offsets)
                )

        if expr.matches.Variable:
            assert expr.name in self.arg_assignments, (expr.name, list(self.arg_assignments.keys()))
            return self.arg_assignments[expr.name]

        if expr.matches.Unaryop:
            operand = self.convert(expr.operand)
            if operand.native_type == native_ast.Bool:
                if expr.op.matches.LogicalNot:
                    return TypedLLVMValue(
                        self.builder.not_(operand.llvm_value),
                        operand.native_type
                        )
            else:
                if expr.op.matches.Add:
                    return operand
                if expr.op.matches.LogicalNot:
                    zero_like = llvmlite.ir.Constant(operand.llvm_value.type, 0)

                    if operand.native_type.matches.Int:
                        return TypedLLVMValue(
                            self.builder.icmp_signed("==", operand.llvm_value, zero_like),
                            native_ast.Bool
                            )
                    if operand.native_type.matches.Float:
                        return TypedLLVMValue(
                            self.builder.fcmp_unordered("==", operand.llvm_value, zero_like),
                            native_ast.Bool
                            )

                if expr.op.matches.BitwiseNot and operand.native_type.matches.Int:
                    return TypedLLVMValue(self.builder.not_(operand.llvm_value), operand.native_type)

                if expr.op.matches.Negate:
                    if operand.native_type.matches.Int:
                        return TypedLLVMValue(self.builder.neg(operand.llvm_value), operand.native_type)
                    if operand.native_type.matches.Float:
                        return TypedLLVMValue(
                            self.builder.fmul(operand.llvm_value, llvmlite.ir.DoubleType()(-1.0)),
                            operand.native_type
                            )

            assert False, "can't apply unary operand %s to %s" % (expr.op, str(operand.native_type))

        if expr.matches.Binop:
            l = self.convert(expr.l)
            if l is None:
                return
            r = self.convert(expr.r)
            if r is None:
                return

            for which, rep in [('Gt','>'),('Lt','<'),('GtE','>='),
                               ('LtE','<='),('Eq',"=="),("NotEq","!=")]:
                if getattr(expr.op.matches, which):
                    if l.native_type.matches.Float:
                        return TypedLLVMValue(
                            self.builder.fcmp_ordered(rep, l.llvm_value, r.llvm_value),
                            native_ast.Bool
                            )
                    elif l.native_type.matches.Int:
                        return TypedLLVMValue(
                            self.builder.icmp_signed(rep, l.llvm_value, r.llvm_value),
                            native_ast.Bool
                            )

            for py_op, floatop, intop_s, intop_u in [
                        ('Add','fadd','add','add'),
                        ('Mul','fmul','mul','mul'),
                        ('Div','fdiv','sdiv', 'udiv'),
                        ('Mod','frem','srem', 'urem'),
                        ('Sub','fsub','sub','sub'),
                        ('LShift',None,'shl','shl'),
                        ('RShift',None,'ashr','lshr'),
                        ('BitOr',None,'or_','or_'),
                        ('BitXor',None,'xor','xor'),
                        ('BitAnd',None,'and_','and_')
                        ]:
                if getattr(expr.op.matches, py_op):
                    assert l.native_type == r.native_type, \
                        "malformed types: expect l&r to be the same but got %s,%s,%s\n\nexpr=%s"\
                            % (py_op, l.native_type, r.native_type, expr)
                    if l.native_type.matches.Float and floatop is not None:
                        return TypedLLVMValue(
                            getattr(self.builder, floatop)(l.llvm_value, r.llvm_value),
                            l.native_type
                            )
                    elif l.native_type.matches.Int:
                        llvm_op = intop_s if l.native_type.signed else intop_u

                        if llvm_op is not None:
                            return TypedLLVMValue(
                                getattr(self.builder, llvm_op)(l.llvm_value, r.llvm_value),
                                l.native_type
                                )


        if expr.matches.Call:
            target_or_ptr = expr.target
            args = [self.convert(a) for a in expr.args]

            for i in range(len(args)):
                if args[i] is None:
                    return

            if target_or_ptr.matches.Named:
                target = target_or_ptr.target

                func = self.namedCallTargetToLLVM(target)
            else:
                target = self.convert(target_or_ptr.expr)

                assert target.native_type.matches.Pointer
                assert target.native_type.value_type.matches.Function

                func = target
            try:
                if func.native_type.value_type.can_throw:
                    normal_target = self.builder.append_basic_block()
                    exception_target = self.builder.append_basic_block()

                    llvm_call_result = self.builder.invoke(
                        func.llvm_value,
                        [a.llvm_value for a in args],
                        normal_target,
                        exception_target
                        )

                    self.generate_exception_landing_pad(exception_target)

                    self.builder.position_at_start(normal_target)
                else:
                    llvm_call_result = self.builder.call(func.llvm_value, [a.llvm_value for a in args])
            except:
                print("failing while calling ", target.name)
                for a in args:
                    print("\t", a.llvm_value, a.native_type)
                raise

            output_type = func.native_type.value_type.output

            if output_type.matches.Void:
                llvm_call_result = None

            return TypedLLVMValue(llvm_call_result, output_type)


        if expr.matches.Sequence:
            res = TypedLLVMValue(None, native_ast.Type.Void())

            for e in expr.vals:
                res = self.convert(e)
                if res is None:
                    return

            return res

        if expr.matches.Comment:
            return self.convert(expr.expr)

        if expr.matches.ActivatesTeardown:
            assert expr.name not in self.tags_initialized, "already initialized tag %s" % expr.name
            self.tags_initialized[expr.name] = True
            return TypedLLVMValue(None, native_ast.Type.Void())

        if expr.matches.Throw:
            arg = self.convert(expr.expr)

            self.builder.store(
                self.builder.bitcast(arg.llvm_value, llvm_i8ptr),
                self.exception_slot
                )

            block = self.teardown_handler.accept_incoming(
                    self.builder.block,
                    self.tags_initialized,
                    False
                    )

            self.builder.branch(block)

            return

        if expr.matches.TryCatch:
            self.teardown_handler = TeardownOnScopeExit(
                self,
                self.teardown_handler
                )
            new_handler = self.teardown_handler

            result = self.convert(expr.expr)

            self.teardown_handler = new_handler.parent_scope

            def generator(tags, resume_normal_block):
                with self.tags_as(tags):
                    prior = self.arg_assignments.get(expr.varname,None)
                    self.arg_assignments[expr.varname] = \
                        TypedLLVMValue(
                            self.builder.load(
                                self.exception_slot
                                ),
                            native_ast.Int8Ptr
                            )

                    handler_res = self.convert(expr.handler)

                    if prior is None:
                        del self.arg_assignments[expr.varname]
                    else:
                        self.arg_assignments[expr.varname] = prior

                    if handler_res is not None:
                        self.builder.branch(resume_normal_block)

            target_resume_block = self.builder.append_basic_block()

            if result is not None:
                self.builder.branch(target_resume_block)

            new_handler.generate_trycatch_unwind(target_resume_block, generator)

            self.builder.position_at_start(target_resume_block)

            return result

        if expr.matches.FunctionPointer:
            return self.namedCallTargetToLLVM(expr.target)

        if expr.matches.Finally:
            self.teardown_handler = TeardownOnScopeExit(
                self,
                self.teardown_handler
                )

            new_handler = self.teardown_handler

            finally_result = self.convert(expr.expr)

            self.teardown_handler = self.teardown_handler.parent_scope

            #if we have a result, then we need to generate teardowns
            #in the normal course of execution
            if finally_result is not None:
                for teardown in expr.teardowns:
                    self.convert_teardown(teardown)

            def generate_teardowns(new_tags):
                with self.tags_as(new_tags):
                    for teardown in expr.teardowns:
                        self.convert_teardown(teardown)

            new_handler.generate_teardown(generate_teardowns)

            return finally_result

        assert False, "can't handle %s" % repr(expr)

def populate_needed_externals(external_function_references, module):
    def define(fname, output, inputs, vararg=False):
        external_function_references[fname] = \
            llvmlite.ir.Function(
                module,
                llvmlite.ir.FunctionType(
                    output,
                    inputs,
                    var_arg=vararg
                    ),
                fname
                )
    define("__cxa_allocate_exception", llvm_i8ptr, [llvm_i64])
    define("__cxa_throw", llvm_void, [llvm_i8ptr,llvm_i8ptr,llvm_i8ptr])
    define("__cxa_end_catch", llvm_i8ptr, [llvm_i8ptr])
    define("__cxa_begin_catch", llvm_i8ptr, [llvm_i8ptr])
    define("__gxx_personality_v0", llvm_i32, [], vararg=True)

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
        populate_needed_externals(external_function_references, module)

        for name, function in names_to_definitions.items():
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

                print()
                print("*************")
                print("def %s(%s): #->%s" % (name,
                            ",".join(["%s=%s" % (k,str(t)) for k,t in definition.args]),
                            str(definition.output_type)
                            )
                    )
                print(native_ast.indent(str(definition.body.body)))
                print("*************")
                print()

        for name in sorted(names_to_definitions):
            definition = names_to_definitions[name]
            func = self._functions_by_name[name]
            func.attributes.personality = external_function_references["__gxx_personality_v0"]

            arg_assignments = {}
            for i in range(len(func.args)):
                arg_assignments[definition.args[i][0]] = \
                        TypedLLVMValue(func.args[i], definition.args[i][1])

            block = func.append_basic_block('entry')
            builder = llvmlite.ir.IRBuilder(block)

            try:
                func_converter = FunctionConverter(
                    module,
                    func,
                    self,
                    builder,
                    arg_assignments,
                    definition.output_type,
                    external_function_references
                    )

                func_converter.setup()

                res = func_converter.convert(definition.body.body)

                func_converter.finalize()

                if res is not None:
                    assert res.llvm_value is None
                    assert definition.output_type == native_ast.Void
                    builder.ret_void()
                else:
                    if not builder.block.is_terminated:
                        builder.unreachable()

            except Exception as e:
                print("function failing = " + name)
                raise


        return str(module)


