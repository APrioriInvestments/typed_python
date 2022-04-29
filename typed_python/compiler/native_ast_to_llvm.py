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

import typed_python.compiler.native_ast as native_ast
from typed_python.compiler.module_definition import ModuleDefinition
from typed_python.compiler.global_variable_definition import GlobalVariableDefinition
import llvmlite.ir
import os

llvm_i8ptr = llvmlite.ir.IntType(8).as_pointer()
llvm_i8 = llvmlite.ir.IntType(8)
llvm_i32 = llvmlite.ir.IntType(32)
llvm_i64 = llvmlite.ir.IntType(64)
llvm_i1 = llvmlite.ir.IntType(1)
llvm_void = llvmlite.ir.VoidType()

exception_type_llvm = llvmlite.ir.LiteralStructType([llvm_i8ptr, llvm_i32])

# just hardcoded for now. We check this in the compiler to ensure it's consistent.
pointer_size = 8


CROSS_MODULE_INLINE_COMPLEXITY = 40


def llvmBool(i):
    return llvmlite.ir.Constant(llvm_i1, i)


def llvmI64(i):
    return llvmlite.ir.Constant(llvm_i64, i)


def assertTagDictsSame(left_tags, right_tags):
    for which in [left_tags, right_tags]:
        for tag in which:
            assert tag in left_tags and tag in right_tags and right_tags[tag] is left_tags[tag], (
                f"Tag {tag} is not the same: {left_tags} vs {right_tags}"
            )


def type_to_llvm_type(t):
    if t.matches.Void:
        return llvmlite.ir.VoidType()

    if t.matches.Struct:
        return llvmlite.ir.LiteralStructType(type_to_llvm_type(t[1]) for t in t.element_types)

    if t.matches.Pointer:
        # llvm won't allow a void*, so we model it as a pointer to an empty struct instead
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

    if t.matches.Function:
        return llvmlite.ir.FunctionType(
            type_to_llvm_type(t.output),
            [type_to_llvm_type(x) for x in t.args],
            var_arg=t.varargs
        )

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
        vals = [constant_to_typed_llvm_value(module, builder, t) for _, t in c.elements]

        t = llvmlite.ir.LiteralStructType(type_to_llvm_type(t.llvm_value.type) for t in vals)
        llvm_c = llvmlite.ir.Constant(t, [t.llvm_value for t in vals])

        nt = native_ast.Type.Struct(
            [(c.elements[i][0], vals[i].native_type) for i in range(len(vals))]
        )

        return TypedLLVMValue(llvm_c, nt)

    if c.matches.ByteArray:
        byte_array = c.val

        t = llvmlite.ir.ArrayType(llvm_i8, len(byte_array) + 1)

        llvm_c = llvmlite.ir.Constant(t, bytearray(byte_array + b"\x00"))

        value = llvmlite.ir.GlobalVariable(module, t, "string_constant_%s" % strings_ever[0])

        strings_ever[0] += 1

        value.linkage = "private"
        value.initializer = llvm_c

        nt = native_ast.Type.Int(bits=8, signed=False).pointer()

        return TypedLLVMValue(
            builder.bitcast(value, llvm_i8ptr),
            nt
        )

    if c.matches.Void:
        return TypedLLVMValue(None, native_ast.Type.Void())

    if c.matches.Array:
        vals = [constant_to_typed_llvm_value(module, builder, t) for t in c.values]

        t = llvmlite.ir.ArrayType(type_to_llvm_type(c.value_type.element_type), c.value_type.count)

        llvm_c = llvmlite.ir.Constant(t, [t.llvm_value for t in vals])

        return TypedLLVMValue(llvm_c, c.value_type)

    assert False, (c, type(c))


class TypedLLVMValue:
    def __init__(self, llvm_value, native_type):
        object.__init__(self)

        if native_type.matches.Void:
            if llvm_value is not None:
                assert llvm_value.type == llvm_void
                llvm_value = None
        else:
            assert llvm_value is not None

        self.llvm_value = llvm_value
        self.native_type = native_type

    def __str__(self):
        return "TypedLLVMValue(%s)" % self.native_type

    def __repr__(self):
        return str(self)


class TeardownHandler:
    """Machinery for generating code to call destructors when we exit blocks of code.

    We can exit a Finally block in four ways: through normal control flow (we generate
    the code for this case separately), through a 'return to target' where we
    return control flow to the post-teardown of the first Finally with the given name
    above us on the stack, through a exception handling, and through a regular
    'return' statement (that exits the function entirely).

    Each instance of this class has a pointer to its parent scope. The code converter
    instantiates a new TeardownHandler each time it wants to intercept destructor
    control flow (when we create a Finally or a TryCatch).

    Each TeardownHandler creates a block that it uses to invoke all the destructors
    and then resume control flow.  It initializes an empty block and then waits
    for the main code converter to send it incoming jumps from code below it
    that contains throw/return expressions.

    Once code below it has finished converting, we can resolve a teardown handler
    either as a try/catch handler or as a 'finally' handler. In either case, the
    handler populates the basic block with the relevant teardowns and then
    generates code to propagate control flow, either to a catch statement,
    to the parent stack, or to the resumption of the control flow at that moment.
    """

    def __init__(self, converter, parent_scope, name=None):
        self.converter = converter
        self.builder = converter.builder
        self.name = name

        self.incoming_tags = {}  # dict from block->name->(True or llvm_value)

        # is the incoming block from a scope trying to 'return' (integer number of stack frames)
        # or propagate an exception (-1)
        self.incomingControlFlowSwitch = {}  # dict from block->(an integer or an llvm_value)

        self._block = None

        self.incoming_blocks = set()
        self.parent_scope = parent_scope
        self.height = 0 if parent_scope is None else parent_scope.height + 1

        # track whether we've
        self._isConsumed = False

    def _generateControlFlowSwitchExpr(self):
        """Generate an expression in the body block indicating whether we are a 'return'.

        The expression will be an integer that contains a -1 if we are an exception,
        otherwise an integer giving the number of scopes we should unwind before
        resuming execution when entering the unwind handler.
        """
        assert self._block is not None

        assert self.incomingControlFlowSwitch

        assert len(self.incomingControlFlowSwitch) == len(self.incoming_blocks)

        assert not self._isConsumed
        self._isConsumed = True

        if len(self.incomingControlFlowSwitch) == 1:
            return list(self.incomingControlFlowSwitch.values())[0]

        if all([isinstance(v, int) for v in self.incomingControlFlowSwitch.values()]):
            vals = set(self.incomingControlFlowSwitch.values())
            if len(vals) == 1:
                return list(vals)[0]

        with self.builder.goto_block(self._block):
            switchVal = self.builder.phi(llvm_i64, name='control_flow_switch_%s' % self.height)

            for b, val in self.incomingControlFlowSwitch.items():
                if isinstance(val, int):
                    val = llvmI64(val)
                switchVal.add_incoming(val, b)

            return switchVal

    def controlFlowSwitchForReturn(self, name=None):
        if name is None:
            return self.height + 2

        if self.name == name:
            return 0

        if self.parent_scope is None:
            raise Exception(f"Couldn't find a 'Finally' with name '{name}'")

        return 1 + self.parent_scope.controlFlowSwitchForReturn(name)

    def controlFlowSwitchForException(self):
        return -1

    def acceptIncoming(self, block, tags, controlFlowIdentifier):
        assert block not in self.incomingControlFlowSwitch, "This block already jumped to us"
        assert not self._isConsumed, "This TeardownHandler was already consumed"

        if isinstance(controlFlowIdentifier, llvmlite.ir.Constant):
            controlFlowIdentifier = controlFlowIdentifier.constant

        self.incomingControlFlowSwitch[block] = controlFlowIdentifier
        self.incoming_blocks.add(block)
        self.incoming_tags[block] = dict(tags)

        if self._block is None:
            self._block = self.builder.append_basic_block(self.blockName())

        return self._block

    def blockName(self):
        name = "scope_exit_handler_%d" % self.height
        if self.name is not None:
            name += "_" + self.name
        return name

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
                    val = llvmBool(incoming[b])
                else:
                    val = incoming[b]
                phinode.add_incoming(val, b)

            return phinode

        for t in all_tags:
            tags[t] = collapse_tags(t, tags[t])
            if tags[t] is None:
                del tags[t]

        return tags

    def generate_teardown(self, teardown_callback, return_slot=None, exception_slot=None, normal_slot=None):
        if not self.incomingControlFlowSwitch:
            assert self._block is None
            return

        controlFlowSwitch = self._generateControlFlowSwitchExpr()

        with self.builder.goto_block(self._block):
            tags = self.generate_tags()

            teardown_callback(tags)

            if self.parent_scope is not None:
                if isinstance(controlFlowSwitch, int):
                    # we know exactly how control flow will go - just generate
                    # the relevant code and directly branch to the next block
                    if controlFlowSwitch == 0:
                        assert normal_slot is not None

                        self.builder.branch(normal_slot)
                        return

                    controlFlowSwitch -= 1

                    block = self.parent_scope.acceptIncoming(
                        self.builder.block,
                        tags,
                        controlFlowSwitch
                    )

                    self.builder.branch(block)
                else:
                    if normal_slot is not None:
                        assert self.name is not None

                        wantsPropagate = self.builder.icmp_signed("!=", controlFlowSwitch, llvmI64(0))

                        with self.builder.if_else(wantsPropagate) as (then, otherwise):
                            with then:
                                block = self.parent_scope.acceptIncoming(
                                    self.builder.block,
                                    tags,
                                    self.builder.sub(
                                        controlFlowSwitch,
                                        llvmI64(1)
                                    )
                                )

                                self.builder.branch(block)

                            with otherwise:
                                self.builder.branch(normal_slot)

                        self.builder.unreachable()
                    else:
                        assert self.name is None, self.name
                        block = self.parent_scope.acceptIncoming(
                            self.builder.block,
                            tags,
                            self.builder.sub(
                                controlFlowSwitch,
                                llvmI64(1)
                            )
                        )

                        self.builder.branch(block)
            else:
                assert self.name is None

                if isinstance(controlFlowSwitch, int):
                    if controlFlowSwitch >= 0:
                        if return_slot is None:
                            self.builder.ret_void()
                        else:
                            self.builder.ret(self.builder.load(return_slot))
                    else:
                        self.converter.generate_throw_expression(
                            self.builder.load(exception_slot)
                        )
                else:
                    assert isinstance(controlFlowSwitch, llvmlite.ir.Value)

                    isReturn = self.builder.icmp_signed(">=", controlFlowSwitch, llvmI64(0))

                    with self.builder.if_else(isReturn) as (then, otherwise):
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
        if not self.incomingControlFlowSwitch:
            assert self._block is None
            return

        controlFlowSwitch = self._generateControlFlowSwitchExpr()

        with self.builder.goto_block(self._block):
            tags = self.generate_tags()

            if isinstance(controlFlowSwitch, int):
                if controlFlowSwitch == 0:
                    raise Exception("Please don't jump to a try-catch")

                if controlFlowSwitch >= 0:
                    block = self.parent_scope.acceptIncoming(
                        self.builder.block,
                        tags,
                        controlFlowSwitch - 1
                    )
                    self.builder.branch(block)
                    return

                generator(tags, target_resume_block)
                return

            assert isinstance(controlFlowSwitch, llvmlite.ir.Value)

            isReturn = self.builder.icmp_signed(
                ">",
                controlFlowSwitch,
                llvmI64(0)
            )

            with self.builder.if_then(isReturn):
                block = self.parent_scope.acceptIncoming(
                    self.builder.block,
                    tags,
                    self.builder.sub(
                        controlFlowSwitch,
                        llvmI64(1)
                    )
                )
                self.builder.branch(block)

            generator(tags, target_resume_block)


class FunctionConverter:
    def __init__(self,
                 module,
                 globalDefinitions,
                 globalDefinitionLlvmValues,
                 function,
                 converter,
                 builder,
                 arg_assignments,
                 output_type,
                 external_function_references
                 ):
        self.function = function

        # dict from name to GlobalVariableDefinition
        self.globalDefinitions = globalDefinitions
        self.globalDefinitionLlvmValues = globalDefinitionLlvmValues

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

        self.exception_slot = builder.alloca(llvm_i8ptr, name="exception_slot")

        # if populated, we are expected to write our return value to 'return_slot' and jump here
        # on return
        self.teardown_handler = TeardownHandler(self, None)

    def finalize(self):
        self.teardown_handler.generate_teardown(lambda tags: None, self.return_slot, self.exception_slot)

    def generate_exception_landing_pad(self, block):
        with self.builder.goto_block(block):
            res = self.builder.landingpad(exception_type_llvm)

            res.add_clause(
                llvmlite.ir.CatchClause(
                    llvmlite.ir.Constant(llvm_i8ptr, None)
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

            block = self.teardown_handler.acceptIncoming(
                self.builder.block,
                self.tags_initialized,
                self.teardown_handler.controlFlowSwitchForException()
            )

            self.builder.branch(block)

    def convert_teardown(self, teardown, justClearTags=False):
        orig_tags = dict(self.tags_initialized)

        if teardown.matches.Always:
            if not justClearTags:
                self.convert(teardown.expr)
        else:
            assert teardown.matches.ByTag

            if teardown.tag in self.tags_initialized:
                tagVal = self.tags_initialized[teardown.tag]

                # mark that the tag is no longer active
                del self.tags_initialized[teardown.tag]
                del orig_tags[teardown.tag]

                if not justClearTags:
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
                [llvmI64(pointer_size)],
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
        elif target.name in self.converter._externallyDefinedFunctionTypes:
            # this function is defined in a shared object that we've loaded from a prior
            # invocation
            if target.name not in self.external_function_references:
                func_type = llvmlite.ir.FunctionType(
                    type_to_llvm_type(target.output_type),
                    [type_to_llvm_type(x) for x in target.arg_types],
                    var_arg=target.varargs
                )

                assert target.name not in self.converter._function_definitions, target.name

                self.external_function_references[target.name] = (
                    llvmlite.ir.Function(self.module, func_type, target.name)
                )

            func = self.external_function_references[target.name]
        else:
            func = self.converter._functions_by_name[target.name]

            if func.module is not self.module:
                # first, see if we'd like to inline this module
                if (
                    self.converter.totalFunctionComplexity(target.name) < CROSS_MODULE_INLINE_COMPLEXITY
                    and self.converter.canBeInlined(target.name)
                ):
                    func = self.converter.repeatFunctionInModule(target.name, self.module)
                else:
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
            [exception_ptr] + [llvmlite.ir.Constant(llvm_i8ptr, None)] * 2
        )

        self.builder.unreachable()

    def convert(self, expr):
        """Convert 'expr' into underlying llvm instructions.

        Also, verify that if we return a value, our control flow
        block is not terminated, and that if we don't return a value,
        we don't have a dangling block.
        """
        res = self._convert(expr)

        if res is not None:
            assert not self.builder.block.is_terminated, expr
        else:
            assert self.builder.block.is_terminated, expr

        return res

    def _convert(self, expr):
        """Actually convert 'expr' into underlying llvm instructions."""
        if expr.matches.ApplyIntermediates:
            res = TypedLLVMValue(None, native_ast.Type.Void())

            priorName = []
            priorRes = []

            for i in expr.intermediates:
                if i.matches.Terminal or i.matches.Effect:
                    res = self.convert(i.expr)

                    if res is None:
                        break

                elif i.matches.StackSlot:
                    res = self.convert(i.expr)

                    if res is None:
                        break

                elif i.matches.Simple:
                    lhs = self.convert(i.expr)

                    prior = self.arg_assignments.get(i.name, None)
                    self.arg_assignments[i.name] = lhs

                    priorName.append(i.name)
                    priorRes.append(prior)

            if res is not None:
                res = self.convert(expr.base)

            while priorName:
                name = priorName.pop()
                prior = priorRes.pop()

                if prior is not None:
                    self.arg_assignments[name] = prior
                else:
                    del self.arg_assignments[name]

            return res

        if expr.matches.Let:
            lhs = self.convert(expr.val)

            prior = self.arg_assignments.get(expr.var, None)
            self.arg_assignments[expr.var] = lhs

            res = self.convert(expr.within)

            if prior is not None:
                self.arg_assignments[expr.var] = prior
            else:
                del self.arg_assignments[expr.var]

            return res

        if expr.matches.StackSlot:
            if expr.name not in self.stack_slots:
                if expr.type.matches.Void:
                    llvm_type = type_to_llvm_type(native_ast.Type.Struct(element_types=(), name="void"))
                else:
                    llvm_type = type_to_llvm_type(expr.type)

                with self.builder.goto_entry_block():
                    self.stack_slots[expr.name] = \
                        TypedLLVMValue(
                            self.builder.alloca(llvm_type, name=expr.name),
                            native_ast.Type.Pointer(value_type=expr.type)
                    )

            assert self.stack_slots[expr.name].native_type.value_type == expr.type, \
                "StackSlot %s supposed to have value %s but got %s" % (
                    expr.name,
                    self.stack_slots[expr.name].native_type.value_type,
                    expr.type
            )

            return self.stack_slots[expr.name]

        if expr.matches.GlobalVariable:
            if expr.name in self.globalDefinitions:
                assert expr.metadata == self.globalDefinitions[expr.name].metadata
                assert expr.type == self.globalDefinitions[expr.name].type
            else:
                llvm_type = type_to_llvm_type(expr.type)

                self.globalDefinitions[expr.name] = GlobalVariableDefinition(
                    expr.name,
                    expr.type,
                    expr.metadata
                )
                self.globalDefinitionLlvmValues[expr.name] = TypedLLVMValue(
                    llvmlite.ir.GlobalVariable(self.module, llvm_type, expr.name),
                    native_ast.Type.Pointer(value_type=expr.type)
                )

                self.globalDefinitionLlvmValues[expr.name].llvm_value.linkage = "private"
                self.globalDefinitionLlvmValues[expr.name].llvm_value.initializer = (
                    constant_to_typed_llvm_value(
                        self.module,
                        self.builder,
                        expr.type.zero().val
                    ).llvm_value
                )

            return self.globalDefinitionLlvmValues[expr.name]

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

        if expr.matches.AtomicAdd:
            ptr = self.convert(expr.ptr)
            val = self.convert(expr.val)

            return TypedLLVMValue(
                self.builder.atomic_rmw("add", ptr.llvm_value, val.llvm_value, "monotonic"),
                val.native_type
            )

        if expr.matches.Load:
            ptr = self.convert(expr.ptr)

            assert ptr.native_type.matches.Pointer, ptr.native_type

            if ptr.native_type.value_type.matches.Void:
                return TypedLLVMValue(None, ptr.native_type.value_type)

            return TypedLLVMValue(self.builder.load(ptr.llvm_value), ptr.native_type.value_type)

        if expr.matches.Constant:
            return constant_to_typed_llvm_value(self.module, self.builder, expr.val)

        if expr.matches.Cast:
            lhs = self.convert(expr.left)

            if lhs is None:
                return

            target_type = type_to_llvm_type(expr.to_type)

            if lhs.native_type == expr.to_type:
                return lhs

            if lhs.native_type.matches.Pointer and expr.to_type.matches.Pointer:
                return TypedLLVMValue(self.builder.bitcast(lhs.llvm_value, target_type), expr.to_type)

            if lhs.native_type.matches.Pointer and expr.to_type.matches.Int:
                return TypedLLVMValue(self.builder.ptrtoint(lhs.llvm_value, target_type), expr.to_type)

            if lhs.native_type.matches.Int and expr.to_type.matches.Pointer:
                return TypedLLVMValue(self.builder.inttoptr(lhs.llvm_value, target_type), expr.to_type)

            if lhs.native_type.matches.Float and expr.to_type.matches.Int:
                if expr.to_type.signed:
                    return TypedLLVMValue(self.builder.fptosi(lhs.llvm_value, target_type), expr.to_type)
                else:
                    return TypedLLVMValue(self.builder.fptoui(lhs.llvm_value, target_type), expr.to_type)

            elif lhs.native_type.matches.Float and expr.to_type.matches.Float:
                if lhs.native_type.bits > expr.to_type.bits:
                    return TypedLLVMValue(self.builder.fptrunc(lhs.llvm_value, target_type), expr.to_type)
                else:
                    return TypedLLVMValue(self.builder.fpext(lhs.llvm_value, target_type), expr.to_type)

            elif lhs.native_type.matches.Int and expr.to_type.matches.Int:
                if lhs.native_type.bits < expr.to_type.bits:
                    if lhs.native_type.signed:
                        return TypedLLVMValue(self.builder.sext(lhs.llvm_value, target_type), expr.to_type)
                    else:
                        return TypedLLVMValue(self.builder.zext(lhs.llvm_value, target_type), expr.to_type)
                else:
                    return TypedLLVMValue(self.builder.trunc(lhs.llvm_value, target_type), expr.to_type)

            elif lhs.native_type.matches.Int and expr.to_type.matches.Float:
                if lhs.native_type.signed:
                    return TypedLLVMValue(self.builder.sitofp(lhs.llvm_value, target_type), expr.to_type)
                else:
                    return TypedLLVMValue(self.builder.uitofp(lhs.llvm_value, target_type), expr.to_type)

            else:
                raise Exception(f"Invalid cast: {lhs.native_type} to {expr.to_type}")

        if expr.matches.Return:
            if expr.blockName is not None:
                # assert expr.arg is None, expr.arg
                if expr.arg is not None:
                    # write the value into the return slot
                    arg = self.convert(expr.arg)

                    if arg is None:
                        # the expression threw an exception so we can't actually
                        # return
                        return

                    if not self.output_type.matches.Void:
                        assert self.return_slot is not None
                        self.builder.store(arg.llvm_value, self.return_slot)

                controlFlowSwitch = self.teardown_handler.controlFlowSwitchForReturn(name=expr.blockName)

                block = self.teardown_handler.acceptIncoming(
                    self.builder.block,
                    self.tags_initialized,
                    controlFlowSwitch
                )

                self.builder.branch(block)
                return
            else:
                # this is a naked 'return'
                if expr.arg is not None:
                    # write the value into the return slot
                    arg = self.convert(expr.arg)

                    if arg is None:
                        # the expression threw an exception so we can't actually
                        # return
                        return

                    if not self.output_type.matches.Void:
                        assert self.return_slot is not None
                        self.builder.store(arg.llvm_value, self.return_slot)

                controlFlowSwitch = self.teardown_handler.controlFlowSwitchForReturn(name=None)

                block = self.teardown_handler.acceptIncoming(
                    self.builder.block,
                    self.tags_initialized,
                    controlFlowSwitch
                )

                self.builder.branch(block)
                return

        if expr.matches.Branch:
            cond = self.convert(expr.cond)

            if cond is None:
                return None

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

            # we need to merge tags
            final_tags = {}
            for tag in set(list(true_tags.keys()) + list(false_tags.keys())):
                true_val = true_tags.get(tag, False)
                false_val = false_tags.get(tag, False)

                if true_val is True and false_val is True:
                    final_tags[tag] = True
                else:
                    # it's not certain
                    if not isinstance(true_val, bool) and not isinstance(false_val, bool) and true_val.name == false_val.name:
                        # these are the same bit that's been passed between two different branches.
                        final_tags[tag] = true_val
                    else:
                        tag_llvm_value = self.builder.phi(llvm_i1, 'is_initialized.merge.' + tag)
                        if isinstance(true_val, bool):
                            true_val = llvmBool(true_val)
                        if isinstance(false_val, bool):
                            false_val = llvmBool(false_val)

                        tag_llvm_value.add_incoming(true_val, true_block)
                        tag_llvm_value.add_incoming(false_val, false_block)
                        final_tags[tag] = tag_llvm_value

            self.tags_initialized = final_tags

            if true.native_type != false.native_type:
                raise Exception("Expected left and right branches to have same type, but %s != %s\n\n%s" % (true, false, expr))

            if true.native_type.matches.Void:
                return TypedLLVMValue(None, native_ast.Type.Void())

            final = self.builder.phi(type_to_llvm_type(true.native_type))
            final.add_incoming(true.llvm_value, true_block)
            final.add_incoming(false.llvm_value, false_block)

            return TypedLLVMValue(final, true.native_type)

        if expr.matches.While:
            tags = dict(self.tags_initialized)

            loop_block = self.builder.append_basic_block("while")

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
                    if true is not None:
                        self.builder.branch(loop_block)

                with otherwise:
                    false = self.convert(expr.orelse)

            # it's currently illegal to modify the initialized set in a while loop
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
                    assert native_type.matches.Pointer, f"Can't take element '{offsets[0]}' of {native_type}"
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
                        if operand.native_type.bits == 32:
                            return TypedLLVMValue(
                                self.builder.fmul(operand.llvm_value, llvmlite.ir.FloatType()(-1.0)),
                                operand.native_type
                            )
                        else:
                            return TypedLLVMValue(
                                self.builder.fmul(operand.llvm_value, llvmlite.ir.DoubleType()(-1.0)),
                                operand.native_type
                            )

            assert False, "can't apply unary operand %s to %s" % (expr.op, str(operand.native_type))

        if expr.matches.Binop:
            lhs = self.convert(expr.left)
            if lhs is None:
                return
            rhs = self.convert(expr.right)
            if rhs is None:
                return

            for which, rep in [('Gt', '>'), ('Lt', '<'), ('GtE', '>='),
                               ('LtE', '<='), ('Eq', "=="), ("NotEq", "!=")]:
                if getattr(expr.op.matches, which):
                    if lhs.native_type.matches.Float:
                        return TypedLLVMValue(
                            self.builder.fcmp_ordered(rep, lhs.llvm_value, rhs.llvm_value),
                            native_ast.Bool
                        )
                    elif lhs.native_type.matches.Int:
                        if lhs.native_type.signed:
                            return TypedLLVMValue(
                                self.builder.icmp_signed(rep, lhs.llvm_value, rhs.llvm_value),
                                native_ast.Bool
                            )
                        else:
                            return TypedLLVMValue(
                                self.builder.icmp_unsigned(rep, lhs.llvm_value, rhs.llvm_value),
                                native_ast.Bool
                            )

            for py_op, floatop, intop_s, intop_u in [('Add', 'fadd', 'add', 'add'),
                                                     ('Mul', 'fmul', 'mul', 'mul'),
                                                     ('Div', 'fdiv', 'sdiv', 'udiv'),
                                                     ('Mod', 'frem', 'srem', 'urem'),
                                                     ('Sub', 'fsub', 'sub', 'sub'),
                                                     ('LShift', None, 'shl', 'shl'),
                                                     ('RShift', None, 'ashr', 'lshr'),
                                                     ('BitOr', None, 'or_', 'or_'),
                                                     ('BitXor', None, 'xor', 'xor'),
                                                     ('BitAnd', None, 'and_', 'and_')]:
                if getattr(expr.op.matches, py_op):
                    assert lhs.native_type == rhs.native_type, \
                        "malformed types: expect lhs&rhs to be the same but got %s,%s,%s\n\nexpr=%s"\
                        % (py_op, lhs.native_type, rhs.native_type, expr)
                    if lhs.native_type.matches.Float and floatop is not None:
                        return TypedLLVMValue(
                            getattr(self.builder, floatop)(lhs.llvm_value, rhs.llvm_value),
                            lhs.native_type
                        )
                    elif lhs.native_type.matches.Int:
                        llvm_op = intop_s if lhs.native_type.signed else intop_u

                        if llvm_op is not None:
                            return TypedLLVMValue(
                                getattr(self.builder, llvm_op)(lhs.llvm_value, rhs.llvm_value),
                                lhs.native_type
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

                if self.converter._printAllNativeCalls:
                    self.builder.call(
                        self.namedCallTargetToLLVM(
                            native_ast.NamedCallTarget(
                                name="np_print_bytes",
                                arg_types=(native_ast.UInt8.pointer(),),
                                output_type=native_ast.Void,
                                external=True,
                                varargs=False,
                                intrinsic=False,
                                can_throw=False
                            )
                        ).llvm_value,
                        [constant_to_typed_llvm_value(
                            self.module,
                            self.builder, native_ast.Constant.ByteArray(
                                ("calling native fun " + target.name + "\n").encode("ASCII")
                            )
                        ).llvm_value]
                    )
            else:
                target = self.convert(target_or_ptr.expr)

                assert (target.native_type.matches.Pointer and target.native_type.value_type.matches.Function), \
                    f"{target.native_type} is not a Function pointer"

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
            except Exception:
                print("failing while calling ", target)
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

            block = self.teardown_handler.acceptIncoming(
                self.builder.block,
                self.tags_initialized,
                self.teardown_handler.controlFlowSwitchForException()
            )

            self.builder.branch(block)

            return

        if expr.matches.TryCatch or expr.matches.ExceptionPropagator:
            self.teardown_handler = TeardownHandler(
                self,
                self.teardown_handler
            )
            new_handler = self.teardown_handler

            result = self.convert(expr.expr)

            self.teardown_handler = new_handler.parent_scope

            def generator(tags, resume_normal_block):
                with self.tags_as(tags):
                    prior = self.arg_assignments.get(expr.varname, None)
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

            target_resume_block = self.builder.append_basic_block("try_catch_resume")

            if result is not None:
                self.builder.branch(target_resume_block)

            new_handler.generate_trycatch_unwind(target_resume_block, generator)

            self.builder.position_at_start(target_resume_block)

            # if we returned 'none', and we're a TryCatch, then by definition we return
            # 'void', which means we might return void ourselves. If we are an unwind
            # handler, we don't need to do this because we're just going to propgate
            # the exception anyways
            if result is None and expr.matches.TryCatch:
                result = TypedLLVMValue(None, native_ast.Type.Void())
            elif result is None and not self.builder.block.is_terminated:
                self.builder.unreachable()

            return result

        if expr.matches.FunctionPointer:
            return self.namedCallTargetToLLVM(expr.target)

        if expr.matches.Finally:
            self.teardown_handler = TeardownHandler(
                self,
                self.teardown_handler,
                expr.name
            )

            new_handler = self.teardown_handler

            finally_result = self.convert(expr.expr)

            self.teardown_handler = self.teardown_handler.parent_scope

            # if we have a result, then we need to generate teardowns
            # in the normal course of execution
            if finally_result is not None:
                for teardown in expr.teardowns:
                    self.convert_teardown(teardown)
            else:
                for teardown in expr.teardowns:
                    self.convert_teardown(teardown, justClearTags=True)

            def generate_teardowns(new_tags):
                with self.tags_as(new_tags):
                    for teardown in expr.teardowns:
                        self.convert_teardown(teardown)

            if expr.name is not None:
                finalBlock = self.builder.append_basic_block(self.teardown_handler.blockName() + "_resume")

                if finally_result is None:
                    # someone might be jumping here. just because we don't have
                    # a value doesn't mean this particular expression doesn't have
                    # a result, because we might be a 'finally' catching a result.
                    finally_result = TypedLLVMValue(None, native_ast.Type.Void())

                if not self.builder.block.is_terminated:
                    self.builder.branch(finalBlock)

                self.builder.position_at_start(finalBlock)
            else:
                finalBlock = None

            new_handler.generate_teardown(generate_teardowns, normal_slot=finalBlock)

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
    define("__cxa_throw", llvm_void, [llvm_i8ptr, llvm_i8ptr, llvm_i8ptr])
    define("__cxa_end_catch", llvm_i8ptr, [llvm_i8ptr])
    define("__cxa_begin_catch", llvm_i8ptr, [llvm_i8ptr])
    define("__gxx_personality_v0", llvm_i32, [], vararg=True)


class Converter:
    def __init__(self):
        object.__init__(self)
        self._modules = {}
        self._functions_by_name = {}
        self._function_definitions = {}

        # a map from function name to function type for functions that
        # are defined in external shared objects and linked in to this one.
        self._externallyDefinedFunctionTypes = {}

        # total number of instructions in each function, by name
        self._function_complexity = {}

        self._inlineRequests = []

        self._printAllNativeCalls = os.getenv("TP_COMPILER_LOG_NATIVE_CALLS")
        self.verbose = False

    def markExternal(self, functionNameToType):
        """Provide type signatures for a set of external functions."""
        self._externallyDefinedFunctionTypes.update(functionNameToType)

    def canBeInlined(self, name):
        return name not in self._externallyDefinedFunctionTypes

    def totalFunctionComplexity(self, name):
        """Return the total number of instructions contained in a function.

        The function must already have been defined in a prior parss. We use this
        information to decide which functions to repeat in new module definitions.
        """
        if name in self._function_complexity:
            return self._function_complexity[name]

        res = 0
        for block in self._functions_by_name[name].basic_blocks:
            res += len(block.instructions)

        self._function_complexity[name] = res

        return res

    def repeatFunctionInModule(self, name, module):
        """Request that the function given by 'name' be inlined into 'module'.

        It must already have been defined in another module.

        Returns:
            a fresh unpopulated function definition for the given function.
        """
        assert name in self._functions_by_name
        assert self._functions_by_name[name].module != module

        existingFunctionDef = self._functions_by_name[name]

        funcType = existingFunctionDef.type
        if funcType.is_pointer:
            funcType = funcType.pointee

        assert isinstance(funcType, llvmlite.ir.FunctionType)

        self._functions_by_name[name] = llvmlite.ir.Function(module, funcType, name)

        self._inlineRequests.append(name)

        return self._functions_by_name[name]

    def add_functions(self, names_to_definitions):
        names_to_definitions = dict(names_to_definitions)

        for name in names_to_definitions:
            assert name not in self._functions_by_name, "can't define %s twice" % name

        module_name = "module_%s" % len(self._modules)

        module = llvmlite.ir.Module(name=module_name)

        self._modules[module_name] = module

        external_function_references = {}
        populate_needed_externals(external_function_references, module)

        functionTypes = {}

        for name, function in names_to_definitions.items():
            functionTypes[name] = native_ast.Type.Function(
                output=function.output_type,
                args=[x[1] for x in function.args],
                varargs=False,
                can_throw=True
            )
            func_type = llvmlite.ir.FunctionType(
                type_to_llvm_type(function.output_type),
                [type_to_llvm_type(x[1]) for x in function.args]
            )
            self._functions_by_name[name] = llvmlite.ir.Function(module, func_type, name)

            self._functions_by_name[name].linkage = 'external'
            self._function_definitions[name] = function

        if self.verbose:
            for name in names_to_definitions:
                definition = names_to_definitions[name]
                func = self._functions_by_name[name]

                print()
                print("*************")
                print(
                    "def %s(%s): #->%s" % (
                        name,
                        ",".join(["%s=%s" % (k, str(t)) for k, t in definition.args]),
                        str(definition.output_type)
                    )
                )
                print(native_ast.indent(str(definition.body.body)))
                print("*************")
                print()

        globalDefinitions = {}
        globalDefinitionsLlvmValues = {}

        while names_to_definitions:
            for name in sorted(names_to_definitions):
                definition = names_to_definitions.pop(name)
                func = self._functions_by_name[name]
                func.attributes.personality = external_function_references["__gxx_personality_v0"]

                # for a in func.args:
                #     if a.type.is_pointer:
                #         a.add_attribute("noalias")

                arg_assignments = {}
                for i in range(len(func.args)):
                    arg_assignments[definition.args[i][0]] = \
                        TypedLLVMValue(func.args[i], definition.args[i][1])

                block = func.append_basic_block('entry')
                builder = llvmlite.ir.IRBuilder(block)

                try:
                    func_converter = FunctionConverter(
                        module,
                        globalDefinitions,
                        globalDefinitionsLlvmValues,
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
                        if definition.output_type != native_ast.Void:
                            if not builder.block.is_terminated:
                                builder.unreachable()
                        else:
                            builder.ret_void()
                    else:
                        if not builder.block.is_terminated:
                            builder.unreachable()

                except Exception:
                    print("function failing = " + name)
                    raise

            # each function listed here was deemed 'inlinable', which means that we
            # want to repeat its definition in this particular module.
            for name in self._inlineRequests:
                names_to_definitions[name] = self._function_definitions[name]
            self._inlineRequests.clear()

        # define a function that accepts a pointer and fills it out with a table of pointer values
        # so that we can link in any type objects that are defined within the source code.
        self.defineGlobalMetadataAccessor(module, globalDefinitions, globalDefinitionsLlvmValues)

        functionTypes[ModuleDefinition.GET_GLOBAL_VARIABLES_NAME] = native_ast.Type.Function(
            output=native_ast.Void,
            args=[native_ast.Void.pointer().pointer()]
        )

        return ModuleDefinition(
            str(module),
            functionTypes,
            globalDefinitions
        )

    def defineGlobalMetadataAccessor(self, module, globalDefinitions, globalDefinitionsLlvmValues):
        """Given a list of global variables, make a function to access them.

        The function will be named '.get_global_variables' and will accept
        a single argument that takes a PointerTo(PointerTo(None)) and fills
        it out with the values of the globalDefinitions in their lexical
        ordering.
        """
        accessorFunction = llvmlite.ir.Function(
            module,
            type_to_llvm_type(
                native_ast.Type.Function(
                    output=native_ast.Void,
                    args=[native_ast.Void.pointer().pointer()],
                    varargs=False,
                    can_throw=False
                )
            ),
            ModuleDefinition.GET_GLOBAL_VARIABLES_NAME
        )

        accessorFunction.linkage = "external"

        outPtr = accessorFunction.args[0]

        block = accessorFunction.append_basic_block('entry')
        builder = llvmlite.ir.IRBuilder(block)
        voidPtr = type_to_llvm_type(native_ast.Void.pointer())

        index = 0
        for name in sorted(globalDefinitions):
            builder.store(
                builder.bitcast(
                    globalDefinitionsLlvmValues[name].llvm_value,
                    voidPtr
                ),
                builder.gep(outPtr, [llvmI64(index)])
            )
            index += 1

        builder.ret_void()
