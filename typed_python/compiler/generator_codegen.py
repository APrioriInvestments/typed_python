"""Utilities for building generators out of regular functions.

When we convert something that looks like

    def f(x):
        yield 1
        yield 2

we first turn it into something that looks (very roughly)

    class Generator:
        slot = Member(int)

        def __init__(self):
            self.slot = -1

        def __next__(self):
            if self.slot < 0:
                self.slot = 0
                return 1

            if self.slot < 1:
                self.slot = 1
                return 2

            raise StopIteration()

this allows us to wrap up normal-looking code as a generator.

The actual class encapsulates local variable and closure parameters,
and binds all variables with an extra '.' in front of them to avoid conflicting
with regular members.

The '..slot' member of the class encodes the state machine. a value of '-1' means
we are currently executing normally. A value of '-2' means we have ceased iteration
permanently, and a non-negative value means we are stopped at a 'yield' statement,
with '0' meaning the first yield statement, '1' meaning the second, etc.
"""

from typed_python import TupleOf
from typed_python.compiler.withblock_codegen import expandWithBlockIntoTryCatch
import typed_python.python_ast as python_ast

from typed_python.compiler.python_ast_analysis import (
    countYieldStatements, computeFunctionArgVariables, computeAssignedVariables,
    computeReadVariables
)
from typed_python.compiler.codegen_helpers import (
    const,
    compare,
    boolOp,
    branch,
    makeCallExpr,
    attr,
    assign,
    readVar,
    raiseStopIteration
)


class GeneratorCodegen:
    def __init__(self, localVars, generatorLineNumber):
        self.localVars = localVars
        self.yieldsSeen = 0
        self.generatorLineNumber = generatorLineNumber
        self.selfName = f"__typed_python_generator_self_{self.generatorLineNumber}__"

    def returnNullPtr(self):
        return python_ast.Statement.Return(
            value=python_ast.Expr.Call(
                func=readVar(".PointerType"),
                args=()
            )
        )

    def accessVar(self, varname, context=None):
        """Generate a 'self.(varname)' python expression"""
        if varname == "..slot":
            return makeCallExpr(
                attr(readVar(".slotPtr"), "get")
            )

        return python_ast.Expr.Attribute(
            value=python_ast.Expr.Name(id=self.selfName, ctx=python_ast.ExprContext.Load()),
            attr=varname,
            ctx=python_ast.ExprContext.Load() if context is None else context
        )

    def setVar(self, varname, val):
        """Generate a 'self.(varname) = val' python expression"""
        return python_ast.Statement.Assign(
            targets=(
                python_ast.Expr.Attribute(
                    value=python_ast.Expr.Name(id=self.selfName, ctx=python_ast.ExprContext.Load()),
                    attr=varname,
                    ctx=python_ast.ExprContext.Store()
                ),
            ),
            value=val,
        )

    def checkSlotBetween(self, low, high):
        if low > high:
            return const(False)

        if low == high:
            return compare(const(low), self.accessVar("..slot"), "Eq")

        return boolOp(
            "And",
            compare(const(low), self.accessVar("..slot"), "LtE"),
            compare(const(high), self.accessVar("..slot"), "GtE")
        )

    def _changeInteriorObject(self, expr, varsToNotRebind=()):
        """Walk over (and return mapped versions of) "interior" ast objects.

        Generally, this means anything that's not part of the main function body
        including statements of interior closures, etc. In all of these cases,
        we're simply trying to re-map any accesses of variables, while being
        careful to track which variables are being bound within function bodies.
        """
        if isinstance(expr, (int, float, bool, str)) or expr is None:
            return expr

        if expr.matches.Name:
            if expr.id in self.localVars and expr.id not in varsToNotRebind:
                return self.accessVar("." + expr.id, expr.ctx)
            return expr

        if (
            expr.matches.GeneratorExp or
            expr.matches.ListComp or
            expr.matches.SetComp or
            expr.matches.DictComp
        ):
            generators = []
            boundVars = set()
            for g in expr.generators:
                postBoundVars = boundVars | set(computeReadVariables(g.target))

                generators.append(
                    python_ast.Comprehension.Item(
                        target=g.target,
                        iter=self._changeInteriorObject(g.iter, set(varsToNotRebind) | boundVars),
                        ifs=[
                            self._changeInteriorObject(x, set(varsToNotRebind) | postBoundVars)
                            for x in g.ifs
                        ],
                        is_async=g.is_async
                    )
                )

                boundVars = postBoundVars

            if expr.matches.DictComp:
                return type(expr)(
                    key=self._changeInteriorObject(expr.key, set(varsToNotRebind) | boundVars),
                    value=self._changeInteriorObject(expr.value, set(varsToNotRebind) | boundVars),
                    generators=generators,
                    line_number=expr.line_number,
                    col_offset=expr.col_offset,
                    filename=expr.filename
                )
            else:
                return type(expr)(
                    elt=self._changeInteriorObject(expr.elt, set(varsToNotRebind) | boundVars),
                    generators=generators,
                    line_number=expr.line_number,
                    col_offset=expr.col_offset,
                    filename=expr.filename
                )

        if expr.matches.Lambda:
            # so any variable in here that's bound by the function
            # is local to that function. any other variable should
            # be accessing '__typed_python_generator_self__'
            boundVars = computeFunctionArgVariables(expr.args)

            return python_ast.Expr.Lambda(
                args=self._changeInteriorObject(expr.args, varsToNotRebind),
                body=self._changeInteriorObject(expr.body, set(varsToNotRebind) | set(boundVars)),
                line_number=expr.line_number,
                col_offset=expr.col_offset,
                filename=expr.filename,
            )

        if expr.matches.FunctionDef:
            boundVars = set(computeFunctionArgVariables(expr.args)) | set(
                computeAssignedVariables(expr.body)
            )

            return python_ast.Statement.FunctionDef(
                name=expr.name,
                args=self._changeInteriorObject(expr.args),
                body=[self._changeInteriorObject(s, set(varsToNotRebind) | set(boundVars)) for s in expr.body],
                decorator_list=[self._changeInteriorObject(d) for d in expr.decorator_list],
                returns=self._changeInteriorObject(expr.returns) if expr.returns is not None else None,
                line_number=expr.line_number,
                col_offset=expr.col_offset,
                filename=expr.filename,
            )

        # a generic expression walker
        args = {}

        for ix in range(len(expr.ElementType.ElementNames)):
            name = expr.ElementType.ElementNames[ix]
            val = getattr(expr, name)
            argT = type(val)

            if issubclass(argT, (python_ast.Expr, python_ast.Arg, python_ast.Arguments, python_ast.Statement)):
                args[name] = self._changeInteriorObject(val, varsToNotRebind)
            elif argT in (
                TupleOf(python_ast.Expr),
                TupleOf(python_ast.Arg),
                TupleOf(python_ast.Statement)
            ):
                args[name] = [self._changeInteriorObject(subE, varsToNotRebind) for subE in val]
            else:
                args[name] = val

        return type(expr)(**args)

    def changeStatement(self, s):
        yieldsInside = countYieldStatements(s)

        yieldUpperBound = self.yieldsSeen + yieldsInside

        yield branch(
            # if the target slot is lessthan or equal to the number of yields we'll have
            # _after_ we exit this code, then we need to go in
            compare(self.accessVar("..slot"), const(yieldUpperBound), "Lt"),
            # if there are no internal yields, then we don't need to do anything
            # but this external check, which guarantees we skip over the statement
            # if we're searching ahead for the next expression
            list(self.changeStatementInner(s)),
            []
        )

    def changeStatementInner(self, s):
        if s.matches.Expr:
            if s.value.matches.Yield:
                yield branch(
                    compare(self.accessVar("..slot"), const(self.yieldsSeen), "Eq"),
                    [self.setVar("..slot", const(-1))],
                    [
                        self.setVar("..slot", const(self.yieldsSeen)),
                        self.setVar("..value", self._changeInteriorObject(s.value.value)),
                        python_ast.Statement.Return(
                            value=attr(
                                makeCallExpr(readVar(".pointerTo"), readVar(self.selfName)),
                                "..value"
                            )
                        )
                    ]
                )
                self.yieldsSeen += 1
            else:
                yield self._changeInteriorObject(s)

            return

        if s.matches.Assign:
            yield python_ast.Statement.Assign(
                targets=(self._changeInteriorObject(x) for x in s.targets),
                value=self._changeInteriorObject(s.value),
                line_number=s.line_number,
                col_offset=s.col_offset,
                filename=s.filename
            )

            return

        if s.matches.AugAssign:
            yield python_ast.Statement.AugAssign(
                target=self._changeInteriorObject(s.target),
                op=s.op,
                value=self._changeInteriorObject(s.value),
                line_number=s.line_number,
                col_offset=s.col_offset,
                filename=s.filename
            )

            return

        if s.matches.AnnAssign:
            yield python_ast.Statement.AnnAssign(
                target=self._changeInteriorObject(s.target),
                annotation=const(0),
                value=self._changeInteriorObject(s.value) if s.value is not None else None,
                line_number=s.line_number,
                col_offset=s.col_offset,
                filename=s.filename
            )

            return

        if s.matches.If or s.matches.While:
            # if the slot is -1, we're just running. But if the
            # slot is between [self.yieldsSeen, self.yieldsSeen + yieldsLeft) we want
            # to go directly to the 'left' branch without executing the condition,
            # and if greater, then we go to the right without executing the condition
            yieldsSeen = self.yieldsSeen
            yieldsLeft = countYieldStatements(s.body)

            yield branch(
                boolOp(
                    "Or",
                    self.checkSlotBetween(yieldsSeen, yieldsSeen + yieldsLeft - 1),
                    boolOp(
                        "And",
                        self.checkSlotBetween(-1, -1),
                        self._changeInteriorObject(s.test)
                    )
                ),
                self.changeStatementSequence(s.body),
                self.changeStatementSequence(s.orelse),
                isWhile=s.matches.While
            )

            return

        if s.matches.Pass:
            yield python_ast.Statement.Pass(
                line_number=s.line_number,
                col_offset=s.col_offset,
                filename=s.filename
            )

            return

        if s.matches.Break:
            yield python_ast.Statement.Break(
                line_number=s.line_number,
                col_offset=s.col_offset,
                filename=s.filename
            )

            return

        if s.matches.Continue:
            yield python_ast.Statement.Continue(
                line_number=s.line_number,
                col_offset=s.col_offset,
                filename=s.filename
            )

            return

        if s.matches.Assert:
            yield python_ast.Statement.Assert(
                test=self._changeInteriorObject(s.test),
                msg=None if s.msg is None else self._changeInteriorObject(s.msg),
                line_number=s.line_number,
                col_offset=s.col_offset,
                filename=s.filename,
            )

            return

        if s.matches.Raise:
            yield python_ast.Statement.Raise(
                exc=self._changeInteriorObject(s.exc) if s.exc is not None else None,
                cause=self._changeInteriorObject(s.cause) if s.cause is not None else None,
                line_number=s.line_number,
                col_offset=s.col_offset,
                filename=s.filename,
            )

            return

        # to simply things, if we have exceptions and finally, break ourselves
        # apart into two
        if s.matches.Try and s.handlers and s.finalbody:
            for subexpr in self.changeStatement(
                python_ast.Statement.Try(
                    body=[
                        python_ast.Statement.Try(
                            body=s.body,
                            handlers=s.handlers,
                            orelse=s.orelse,
                            finalbody=[],
                            line_number=s.line_number,
                            col_offset=s.col_offset,
                            filename=s.filename
                        )
                    ],
                    handlers=[],
                    orelse=[],
                    finalbody=s.finalbody,
                    line_number=s.line_number,
                    col_offset=s.col_offset,
                    filename=s.filename
                )
            ):
                yield subexpr

            return

        if s.matches.Try:
            yieldsSeen = self.yieldsSeen
            yieldsBody = countYieldStatements(s.body)

            # if any of the exception handlers has a yield in it, we will
            # need a block allowing us to resume in it
            cleanupMatchers = []

            yieldsAtThisCleanupHandlerStart = yieldsSeen + yieldsBody

            # for each block of statements, we might have a resumption.
            for body in [eh.body for eh in s.handlers] + [s.orelse, s.finalbody]:
                thisEHYields = countYieldStatements(body)

                if thisEHYields:
                    # this is a stateful part of the conversion process,
                    # so we have to set it to the right value
                    self.yieldsSeen = yieldsAtThisCleanupHandlerStart

                    cleanupMatchers.append(
                        branch(
                            self.checkSlotBetween(
                                yieldsAtThisCleanupHandlerStart,
                                yieldsAtThisCleanupHandlerStart + thisEHYields - 1,
                            ),
                            self.changeStatementSequence(body),
                            []
                        )
                    )

                yieldsAtThisCleanupHandlerStart += thisEHYields

            # reset the counter so we can generate the main body
            self.yieldsSeen = yieldsSeen

            yield branch(
                self.checkSlotBetween(-1, yieldsSeen + yieldsBody - 1),
                # we're inside the body of the try block.
                [
                    python_ast.Statement.Try(
                        body=self.changeStatementSequence(s.body),
                        handlers=[
                            python_ast.ExceptionHandler.Item(
                                type=self._changeInteriorObject(eh.type) if eh.type is not None else None,
                                name=eh.name,
                                body=self.changeStatementSequence(eh.body)
                            ) for eh in s.handlers
                        ],
                        orelse=self.changeStatementSequence(s.orelse),
                        finalbody=[
                            # only run the finally check if we are handling an
                            # the normal flow of execution. If the slot index is
                            # set to something other than -1, then we are paused
                            # and returning a value and shouldn't run.
                            branch(
                                self.checkSlotBetween(-1, -1),
                                self.changeStatementSequence(s.finalbody),
                                [],
                            )
                        ] if s.finalbody else []
                    )
                ],
                cleanupMatchers
            )

            return

        if s.matches.Return:
            if s.value is None:
                yield self.returnNullPtr()
            else:
                # this is a strange pathway that bypasses the normal
                # yield pathway
                yield self.setVar("..slot", const(-2))
                yield raiseStopIteration(
                    self._changeInteriorObject(s.value)
                )

            return

        if s.matches.With:
            for subexpr in self.changeStatementSequence(expandWithBlockIntoTryCatch(s)):
                yield subexpr

            return

        if s.matches.Import:
            yield s
            return

        if s.matches.ImportFrom:
            yield s
            return

        if s.matches.Delete:
            yield python_ast.Statement.Delete(
                targets=[self._changeInteriorObject(e) for e in s.targets],
                line_number=s.line_number,
                col_offset=s.col_offset,
                filename=s.filename,
            )
            return

        if s.matches.For:
            assert False, "this should already have been rewritten"

        if s.matches.FunctionDef:
            yield python_ast.Statement.FunctionDef(
                name=s.name,
                args=self._changeInteriorObject(s.args),
                body=[
                    self._changeInteriorObject(
                        stmt,
                        set(computeFunctionArgVariables(s.args)) |
                        set(computeAssignedVariables(s.body))
                    ) for stmt in s.body
                ],
                decorator_list=[self._changeInteriorObject(expr) for expr in s.decorator_list],
                returns=self._changeInteriorObject(s.returns),
                line_number=s.line_number,
                col_offset=s.col_offset,
                filename=s.filename,
            )

            return

        if s.matches.ClassDef:
            raise Exception("Not implemented")
        if s.matches.Global:
            raise Exception("Not implemented")
        if s.matches.AsyncFunctionDef:
            raise Exception("Not implemented")
        if s.matches.AsyncWith:
            raise Exception("Not implemented")
        if s.matches.AsyncFor:
            raise Exception("Not implemented")
        if s.matches.NonLocal:
            raise Exception("Not implemented")

        raise Exception("Unknown statement: " + str(type(s)))

    def changeStatementSequence(self, statements):
        return [subst for s in statements for subst in self.changeStatement(s)]

    def convertStatementsToFunctionDef(self, statements):
        return python_ast.Statement.FunctionDef(
            name="__fastnext__",
            args=python_ast.Arguments.Item(
                args=[python_ast.Arg.Item(arg=self.selfName, annotation=None)],
                vararg=None,
                kwarg=None
            ),
            body=[
                python_ast.Statement.Try(
                    body=[
                        assign(
                            ".slotPtr",
                            attr(
                                makeCallExpr(
                                    readVar(".pointerTo"),
                                    readVar(self.selfName)
                                ),
                                "..slot"
                            )
                        ),
                        branch(
                            self.checkSlotBetween(-2, -2),
                            [self.returnNullPtr()],
                            []
                        )
                    ] + self.changeStatementSequence(statements) + [
                        self.returnNullPtr()
                    ],
                    finalbody=[
                        # if we exit during 'normal' execution (slot == -1)
                        # then we are unwinding an exception and we should never
                        # resume
                        branch(
                            self.checkSlotBetween(-1, -1),
                            [self.setVar("..slot", const(-2))],
                            []
                        )
                    ]
                )
            ],
            returns=None,
            filename=""
        )
