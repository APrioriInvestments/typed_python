import typed_python.python_ast as python_ast


def expandWithBlockIntoTryCatch(ast):
    """Break a python_ast.Statement.With into lower-level python primitives."""
    if len(ast.items) > 1:
        # we can break 'with a, b:' down to
        # with a: with b:
        # and proceed recursively
        newBlock = python_ast.Statement.With(
            items=[ast.items[0]],
            body=[
                python_ast.Statement.With(
                    items=ast.items[1:],
                    body=ast.body,
                    line_number=ast.line_number,
                    col_offset=ast.items[1].context_expr.col_offset,
                    filename=ast.filename,
                )
            ],
            line_number=ast.line_number,
            col_offset=ast.col_offset,
            filename=ast.filename,
        )

        return expandWithBlockIntoTryCatch(newBlock)

    # directly expand the context manager code in terms of python primitives
    hasNoException = f".with_hit_except{ast.line_number}.{ast.col_offset}"
    withExceptionVar = f".with_exception{ast.line_number}.{ast.col_offset}"
    managerVar = f".with_cm_var{ast.line_number}.{ast.col_offset}"
    sysModuleVar = f".with_sys_var{ast.line_number}.{ast.col_offset}"

    # with EXPRESSION as TARGET:
    #     SUITE
    #
    # is semantically equivalent to:
    #
    # manager = (EXPRESSION)
    # value = enter(manager)
    # hit_except = False
    #
    # try:
    #     TARGET = value
    #     SUITE
    # except:
    #     hit_except = True
    #     if not exit(manager, *sys.exc_info()):
    #         raise
    # finally:
    #     if not hit_except:
    #         exit(manager, None, None, None)
    #             assert len(ast.items) == 1
    def makeStatement(ast, kind, **kwargs):
        """Helper function to make a Statement of type 'kind'

        Takes line/col from 'ast' and args from 'kwargs'
        """
        return getattr(python_ast.Statement, kind)(
            line_number=ast.line_number,
            col_offset=ast.col_offset,
            filename=ast.filename,
            **kwargs
        )

    def makeExpr(ast, kind, **kwargs):
        """Helper function to make an Expression of type 'kind'

        Takes line/col from 'ast' and args from 'kwargs'
        """
        return getattr(python_ast.Expr, kind)(
            line_number=ast.line_number,
            col_offset=ast.col_offset,
            filename=ast.filename,
            **kwargs
        )

    def makeStoreName(ast, name):
        """Make a 'Store' context name lookup Expression."""
        return makeExpr(ast, 'Name', id=name, ctx=python_ast.ExprContext.Store())

    def makeLoadName(ast, name):
        """Make a 'Load' context name lookup Expression."""
        return makeExpr(ast, 'Name', id=name, ctx=python_ast.ExprContext.Load())

    def makeCallAttribute(x, attributeName, *args):
        """Make getattr(x, attributeName)(*args) expression"""
        return makeExpr(
            x,
            "Call",
            func=makeExpr(x, 'Attribute', value=x, attr=attributeName, ctx=python_ast.ExprContext.Load()),
            args=args
        )

    def makeGetItem(x, index):
        """Make an x[index] expression"""
        return makeExpr(
            x,
            "Subscript",
            value=x,
            slice=makeExpr(ast, 'Num', n=python_ast.NumericConstant.Int(value=index)),
            ctx=python_ast.ExprContext.Load()
        )

    def makeNone():
        """Make an expression for 'None'"""
        return makeExpr(ast, 'Num', n=python_ast.NumericConstant.None_())

    statements = [
        # hasNoException = True
        makeStatement(
            ast,
            'Assign',
            targets=[
                makeStoreName(ast, hasNoException)
            ],
            value=makeExpr(ast, 'Num', n=python_ast.NumericConstant.Boolean(value=True))
        ),
        # managerVar = CONTEXT_MANAGER_EXPRESSION
        makeStatement(
            ast,
            'Assign',
            targets=[
                makeStoreName(ast, managerVar)
            ],
            value=ast.items[0].context_expr
        ),
    ]

    if ast.items[0].optional_vars is not None:
        statements.append(
            # CM_VAR_NAME = managerVar.__enter__()
            makeStatement(
                ast,
                'Assign',
                targets=[
                    ast.items[0].optional_vars
                ],
                value=makeCallAttribute(
                    makeLoadName(ast.items[0].optional_vars, managerVar),
                    "__enter__"
                )
            )
        )
    else:
        statements.append(
            makeStatement(
                ast,
                'Expr',
                value=makeCallAttribute(
                    makeLoadName(ast, managerVar),
                    "__enter__"
                )
            )
        )

    statements.append(
        makeStatement(
            ast,
            'Try',
            body=list(ast.body),
            handlers=[
                python_ast.ExceptionHandler.Item(
                    type=None,
                    name=None,
                    body=[
                        # hasNoException = False
                        makeStatement(
                            ast,
                            'Assign',
                            targets=[
                                makeStoreName(ast, hasNoException)
                            ],
                            value=makeExpr(ast, 'Num', n=python_ast.NumericConstant.Boolean(value=False))
                        ),
                        # withExceptionVar = sys.exc_info()
                        makeStatement(
                            ast,
                            "Import",
                            names=[
                                python_ast.Alias.Item(name="sys", asname=sysModuleVar)
                            ]
                        ),
                        makeStatement(
                            ast,
                            'Assign',
                            targets=[
                                makeStoreName(ast, withExceptionVar)
                            ],
                            value=makeCallAttribute(
                                makeLoadName(ast, sysModuleVar),
                                "exc_info"
                            )
                        ),
                        # if not manager.__exit__(withExceptionVar[0], ...):
                        #    raise
                        makeStatement(
                            ast,
                            'If',
                            test=makeCallAttribute(
                                makeLoadName(ast, managerVar),
                                "__exit__",
                                makeGetItem(makeLoadName(ast, withExceptionVar), 0),
                                makeGetItem(makeLoadName(ast, withExceptionVar), 1),
                                makeGetItem(makeLoadName(ast, withExceptionVar), 2),
                            ),
                            body=[
                                makeStatement(ast, "Pass")
                            ],
                            orelse=[
                                makeStatement(ast, 'Raise', exc=None, cause=None)
                            ]
                        )
                    ],
                )
            ],
            finalbody=[
                # if hasNoException:
                makeStatement(
                    ast,
                    'If',
                    test=makeLoadName(ast, hasNoException),
                    body=[
                        # manager.__exit__(None, None, None)
                        makeStatement(
                            ast,
                            'Expr',
                            value=makeCallAttribute(
                                makeLoadName(ast, managerVar),
                                "__exit__",
                                makeNone(),
                                makeNone(),
                                makeNone(),
                            )
                        )
                    ],
                    orelse=[]
                )
            ]
        )
    )

    return statements
