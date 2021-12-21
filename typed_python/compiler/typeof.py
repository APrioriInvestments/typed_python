import typed_python
from typed_python.compiler.type_wrappers.type_sets import SubclassOf, Either


typeWrapper = lambda t: typed_python.compiler.python_to_native_converter.typedPythonTypeToTypeWrapper(t)


class TypeOf:
    """A 'SignatureFunction' that infers the type of an expression.

    Clients can write something like

        def f(self, x) -> Typeof(lambda self, x: self.g(x)):
            ...

    to indicate that they want the type of evaluating an expression
    on the given types.
    """
    def __init__(self, F):
        self.F = typed_python.Function(F)

    def mapArg(self, arg):
        if isinstance(arg, SubclassOf):
            # the compiler assumes that if you pass it 'C', and 'C'
            # is not final, then you may have a subclass of C.
            return arg.T

        if isinstance(arg, Either):
            return typed_python.OneOf(*arg.Types)

        return arg

    def __call__(self, *args, **kwargs):
        args = [self.mapArg(a) for a in args]
        kwargs = {k: self.mapArg(v) for k, v in kwargs.items()}

        callTarget = self.resultTypeForCall(args, kwargs)

        converter = typed_python.compiler.runtime.Runtime.singleton().converter

        if callTarget is None and converter.isCurrentlyConverting():
            # return the 'empty' OneOf, which can't actually be constructed.
            return typed_python.OneOf()

        return callTarget.output_type.interpreterTypeRepresentation

    def resultTypeForCall(self, argTypes, kwargTypes):
        funcObj = typed_python._types.prepareArgumentToBePassedToCompiler(self.F)

        argTypes = [typeWrapper(a) for a in argTypes]
        kwargTypes = {k: typeWrapper(v) for k, v in kwargTypes.items()}

        overload = funcObj.overloads[0]

        ExpressionConversionContext = typed_python.compiler.expression_conversion_context.ExpressionConversionContext

        argumentSignature = ExpressionConversionContext.computeFunctionArgumentTypeSignature(overload, argTypes, kwargTypes)

        if argumentSignature is None:
            return None

        converter = typed_python.compiler.runtime.Runtime.singleton().converter

        callTarget = converter.convertTypedFunctionCall(
            type(funcObj),
            0,
            argumentSignature,
            assertIsRoot=False
        )

        return callTarget
