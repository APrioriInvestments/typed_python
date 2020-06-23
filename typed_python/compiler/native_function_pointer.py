import ctypes
import typed_python.compiler.native_ast as native_ast
from typed_python import PointerTo


class NativeFunctionPointer:
    def __init__(self, fname, fp, input_types, output_type):
        self.fp = fp
        self.fname = fname
        self.input_types = input_types
        self.output_type = output_type

    def __repr__(self):
        return "NativeFunctionPointer(name=%s,addr=%x,in=%s,out=%s)" \
            % (self.fname, self.fp, [str(x) for x in self.input_types], str(self.output_type))

    def __call__(self, *args):
        """Attempt to call the function directly from python.

        We only allow very simple transformations and types - PointerTo, ints, and floats.
        """
        def mapToCtype(T):
            if T == native_ast.Void:
                return None

            if T == native_ast.Int64:
                return ctypes.c_long

            if T == native_ast.Float64:
                return ctypes.c_double

            if T.matches.Pointer:
                return ctypes.c_void_p

            raise Exception(f"Can't convert {T} to a ctypes type")

        def mapArg(a):
            if isinstance(a, (int, float)):
                return a

            if isinstance(a, PointerTo):
                return int(a)

            raise Exception(f"Can't convert {a} to a ctypes argument")

        # it should be initialized to zero
        func = ctypes.CFUNCTYPE(
            mapToCtype(self.output_type),
            *[mapToCtype(t) for t in self.input_types]
        )(self.fp)

        # get out the pointer table
        return func(*[mapArg(a) for a in args])
