from typed_python import Entrypoint
from typed_python.compiler.type_wrappers.compilable_builtin import CompilableBuiltin
from typed_python.compiler.type_wrappers.runtime_functions import externalCallTarget, Float64


inlineLlvmFunc = externalCallTarget("inlineLlvmFunc", Float64, Float64, inlineLlvmDefinition="""
    define external double @"inlineLlvmFunc"(double %".1") {
    entry:
      %.4 = fadd double %.1, 1.000000e+00
      ret double %.4
    }
""")


class InlineLlvmFunc(CompilableBuiltin):
    def __eq__(self, other):
        return isinstance(other, inlineLlvmFunc)

    def __hash__(self):
        return hash("inlineLlvmFunc")

    def convert_call(self, context, instance, args, kwargs):
        return context.pushPod(
            float,
            inlineLlvmFunc.call(
                args[0],
            )
        )


def test_inline_llvm():
    @Entrypoint
    def f(x):
        return InlineLlvmFunc()(x)

    assert f(2.5) == 3.5
