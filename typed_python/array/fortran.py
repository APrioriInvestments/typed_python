import os
import ctypes

from typed_python import Int32, Float32, Entrypoint, PointerTo, ListOf, TupleOf, UInt8
from typed_python.compiler.conversion_level import ConversionLevel
from typed_python.compiler.type_wrappers.compilable_builtin import CompilableBuiltin
from typed_python.compiler.type_wrappers.runtime_functions import externalCallTarget
import typed_python.compiler.native_compiler.native_ast as native_ast


def _findScipyOpenBlas():
    """Find scipy's bundled OpenBLAS shared library.

    Scipy ships its own OpenBLAS in a '<site-packages>/scipy.libs/' directory.
    All BLAS symbols are prefixed with 'scipy_' (e.g. scipy_dgemm_).
    """
    import scipy
    libs_dir = os.path.join(
        os.path.dirname(os.path.dirname(scipy.__file__)),
        'scipy.libs'
    )
    if not os.path.isdir(libs_dir):
        return None

    for fname in os.listdir(libs_dir):
        if 'openblas' in fname and fname.endswith('.so'):
            return os.path.join(libs_dir, fname)

    return None


blasLibPath = _findScipyOpenBlas()

if blasLibPath is None:
    raise Exception(
        "Couldn't find scipy's bundled OpenBLAS. "
        "Make sure scipy is installed: pip install scipy"
    )

# Load BLAS as a global library so our LLVM-compiled code can resolve
# symbols at link time.
blas = ctypes.CDLL(blasLibPath, mode=ctypes.RTLD_GLOBAL)

try:
    blas.scipy_daxpy_
    blas.scipy_dgemm_
except Exception:
    raise Exception(
        f"scipy's OpenBLAS at {blasLibPath} doesn't export expected symbols."
    )

# Scipy's bundled OpenBLAS prefixes all symbols with 'scipy_'
BLAS_PREFIX = "scipy_"


def makePointer(e, viableOutputTypes):
    typ = e.expr_type.typeRepresentation

    if issubclass(typ, (ListOf, TupleOf)):
        e = e.convert_method_call("pointerUnsafe", (e.context.constant(0),), {})
        if e is None:
            return e
        typ = e.expr_type.typeRepresentation

    if not issubclass(typ, PointerTo):
        e.context.pushException(
            f"Can't convert a value of type {typ} to a pointer"
        )
        return None

    if typ.ElementType not in viableOutputTypes:
        e.context.pushException(
            f"Type {typ} doesn't point to one of "
            f"{','.join(repr(str(x)) for x in viableOutputTypes)}"
        )
        return None

    return e


def ensureOnStack(e, desiredType):
    e = e.convert_to_type(desiredType, ConversionLevel.Implicit)
    if e is None:
        return None

    if not e.isReference:
        e = e.context.pushMove(e)

    return e


class axpy_(CompilableBuiltin):
    def __eq__(self, other):
        return isinstance(other, axpy)

    def __hash__(self):
        return hash("axpy")

    def convert_call(self, context, instance, args, kwargs):
        # a very basic check
        if len(args) != 6:
            # this will just produce an exception in the generated code saying we can't
            # handle these arguments
            return super().convert_call(context, instance, args, kwargs)

        x = makePointer(args[2], (float, Float32))
        if not x:
            return

        T = x.expr_type.typeRepresentation.ElementType
        assert T in (Float32, float), T

        nativeT = native_ast.Float64 if T is float else native_ast.Float32

        y = makePointer(args[4], (T,))
        if not y:
            return

        n = ensureOnStack(args[0], Int32)
        a = ensureOnStack(args[1], T)
        inc1 = ensureOnStack(args[3], Int32)
        inc2 = ensureOnStack(args[5], Int32)

        if not n or not a or not inc1 or not inc2:
            return

        targetFun = externalCallTarget(
            BLAS_PREFIX + ("daxpy_" if T is float else "saxpy_"),
            native_ast.Void,
            native_ast.Int32.pointer(),
            nativeT.pointer(),
            nativeT.pointer(),
            native_ast.Int32.pointer(),
            nativeT.pointer(),
            native_ast.Int32.pointer()
        )

        context.pushEffect(
            targetFun.call(
                n.expr,
                a.expr,
                x.nonref_expr,
                inc1.expr,
                y.nonref_expr,
                inc2.expr
            )
        )

        return context.constant(None)


@Entrypoint
def axpy(n: int, a, x, xinc: int, y, yinc: int):
    """Implements y += a * x

    Args:
        n - the number of values
        a - the constant to multiply by. Must be float convertible.
        x - either a ListOf / TupleOf Float32 or float, or a pointer to same.
        xinc - an integer indicating the step size within 'x'
        y - like x, but must be the same type
        yinc - increment in 'y'
    """
    axpy_()(n, a, x, xinc, y, yinc)


class gemm_(CompilableBuiltin):
    def __eq__(self, other):
        return isinstance(other, gemm_)

    def __hash__(self):
        return hash("gemm")

    def convert_call(self, context, instance, args, kwargs):
        # the types for dgemm are (as given in the docs):

        # character   TRANSA,
        # character   TRANSB,
        # integer     M,
        # integer     N,
        # integer     K,
        # double precision    ALPHA,
        # double precision, dimension(lda,*)  A,
        # integer     LDA,
        # double precision, dimension(ldb,*)  B,
        # integer     LDB,
        # double precision    BETA,
        # double precision, dimension(ldc,*)  C,
        # integer     LDC

        if len(args) != 13:
            # this will just produce an exception in the generated code saying we can't
            # handle these arguments
            return super().convert_call(context, instance, args, kwargs)

        # determine the type from 'A'
        A = makePointer(args[6], (float, Float32))
        T = A.expr_type.typeRepresentation.ElementType
        assert T in (Float32, float), T
        nativeT = native_ast.Float64 if T is float else native_ast.Float32

        TRANSA = ensureOnStack(args[0], UInt8)
        TRANSB = ensureOnStack(args[1], UInt8)
        M = ensureOnStack(args[2], Int32)
        N = ensureOnStack(args[3], Int32)
        K = ensureOnStack(args[4], Int32)
        ALPHA = ensureOnStack(args[5], T)
        LDA = ensureOnStack(args[7], Int32)  # stride
        B = makePointer(args[8], (float, Float32))
        LDB = ensureOnStack(args[9], Int32)  # stride
        BETA = ensureOnStack(args[10], T)  # stride
        C = makePointer(args[11], (float, Float32))
        LDC = ensureOnStack(args[12], Int32)  # stride

        if not all([TRANSA, TRANSB, M, N, K, ALPHA, LDA, B, LDB, BETA, C, LDC]):
            return

        targetFun = externalCallTarget(
            BLAS_PREFIX + ("dgemm_" if T is float else "sgemm_"),
            native_ast.Void,
            native_ast.UInt8.pointer(),
            native_ast.UInt8.pointer(),
            native_ast.Int32.pointer(),
            native_ast.Int32.pointer(),
            native_ast.Int32.pointer(),
            nativeT.pointer(),
            nativeT.pointer(),
            native_ast.Int32.pointer(),
            nativeT.pointer(),
            native_ast.Int32.pointer(),
            nativeT.pointer(),
            nativeT.pointer(),
            native_ast.Int32.pointer(),
        )

        context.pushEffect(
            targetFun.call(
                TRANSA.expr,
                TRANSB.expr,
                M.expr,
                N.expr,
                K.expr,
                ALPHA.expr,
                A.expr,
                LDA.expr,
                B.expr,
                LDB.expr,
                BETA.expr,
                C.expr,
                LDC.expr,
            )
        )

        return context.constant(None)


@Entrypoint
def gemm(transa: str, transb: str, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """Matrix multiplication

    DGEMM  performs one of the matrix-matrix operations

        C := alpha*op( A )*op( B ) + beta*C,

    where  op( X ) is one of

        op( X ) = X   or   op( X ) = X**T,

    alpha and beta are scalars, and A, B and C are matrices, with op( A )
    an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
    """
    gemm_()(ord(transa[0]), ord(transb[0]), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


class gemv_(CompilableBuiltin):
    def __eq__(self, other):
        return isinstance(other, gemv_)

    def __hash__(self):
        return hash("gemv_")

    def convert_call(self, context, instance, args, kwargs):
        # the types for dgemm are (as given in the docs):

        # character   TRANS,
        # integer     M,
        # integer     N,
        # double precision    ALPHA,
        # double precision, dimension(lda,*)  A,
        # integer     LDA,
        # double precision, dimension(*)  X,
        # integer     INCX,
        # double precision    BETA,
        # double precision, dimension(*)  Y,
        # integer     INCY

        if len(args) != 11:
            # this will just produce an exception in the generated code saying we can't
            # handle these arguments
            return super().convert_call(context, instance, args, kwargs)

        # determine the type from 'A'
        A = makePointer(args[4], (float, Float32))
        T = A.expr_type.typeRepresentation.ElementType
        assert T in (Float32, float), T
        nativeT = native_ast.Float64 if T is float else native_ast.Float32

        TRANS = ensureOnStack(args[0], UInt8)
        M = ensureOnStack(args[1], Int32)
        N = ensureOnStack(args[2], Int32)
        ALPHA = ensureOnStack(args[3], T)
        LDA = ensureOnStack(args[5], Int32)  # stride
        X = makePointer(args[6], (float, Float32))
        INCX = ensureOnStack(args[7], Int32)  # stride
        BETA = ensureOnStack(args[8], T)  # stride
        Y = makePointer(args[9], (float, Float32))
        INCY = ensureOnStack(args[10], Int32)  # stride

        if not all([TRANS, M, N, ALPHA, LDA, X, INCX, BETA, Y, INCY]):
            return

        targetFun = externalCallTarget(
            BLAS_PREFIX + ("dgemv_" if T is float else "sgemv_"),
            native_ast.Void,
            native_ast.UInt8.pointer(),
            native_ast.Int32.pointer(),
            native_ast.Int32.pointer(),
            nativeT.pointer(),
            nativeT.pointer(),
            native_ast.Int32.pointer(),
            nativeT.pointer(),
            native_ast.Int32.pointer(),
            nativeT.pointer(),
            nativeT.pointer(),
            native_ast.Int32.pointer(),
        )

        context.pushEffect(
            targetFun.call(
                TRANS.expr,
                M.expr,
                N.expr,
                ALPHA.expr,
                A.expr,
                LDA.expr,
                X.expr,
                INCX.expr,
                BETA.expr,
                Y.expr,
                INCY.expr,
            )
        )

        return context.constant(None)


@Entrypoint
def gemv(trans: str, m, n, alpha, A, lda, X, incx, beta, Y, incy):
    """Matrix multiplication

     DGEMV  performs one of the matrix-vector operations

        y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,

    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.
    """
    gemv_()(ord(trans[0]), m, n, alpha, A, lda, X, incx, beta, Y, incy)


class getrf_(CompilableBuiltin):
    def __eq__(self, other):
        return isinstance(other, getrf_)

    def __hash__(self):
        return hash("getrf_")

    def convert_call(self, context, instance, args, kwargs):
        # integer     M,
        # integer     N,
        # double precision, dimension( lda, * )   A,
        # integer     LDA,
        # integer, dimension( * )     IPIV,
        # integer     INFO

        if len(args) != 6:
            # this will just produce an exception in the generated code saying we can't
            # handle these arguments
            return super().convert_call(context, instance, args, kwargs)

        # determine the type from 'A'
        A = makePointer(args[2], (float, Float32))
        T = A.expr_type.typeRepresentation.ElementType
        assert T in (Float32, float), T
        nativeT = native_ast.Float64 if T is float else native_ast.Float32

        M = ensureOnStack(args[0], Int32)
        N = ensureOnStack(args[1], Int32)
        LDA = ensureOnStack(args[3], Int32)
        IPIV = makePointer(args[4], (Int32,))
        INFO = ensureOnStack(args[5], Int32)

        if not all([M, N, A, LDA, IPIV, INFO]):
            return

        targetFun = externalCallTarget(
            BLAS_PREFIX + ("dgetrf_" if T is float else "sgetrf_"),
            native_ast.Void,
            native_ast.Int32.pointer(),
            native_ast.Int32.pointer(),
            nativeT.pointer(),
            native_ast.Int32.pointer(),
            native_ast.Int32.pointer(),
            native_ast.Int32.pointer(),
        )

        context.pushEffect(
            targetFun.call(
                M.expr,
                N.expr,
                A.expr,
                LDA.expr,
                IPIV.expr,
                INFO.expr,
            )
        )

        return INFO


@Entrypoint
def getrf(M, N, A, LDA, IPIV, INFO):
    """Matrix factorization

       DGETRF computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
       A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.
    """
    return getrf_()(M, N, A, LDA, IPIV, INFO)


class getri_(CompilableBuiltin):
    def __eq__(self, other):
        return isinstance(other, getri_)

    def __hash__(self):
        return hash("getri_")

    def convert_call(self, context, instance, args, kwargs):
        # integer     N,
        # double precision, dimension( lda, * )   A,
        # integer     LDA,
        # integer, dimension( * )     IPIV,
        # double precision, dimension( * )    WORK,
        # integer     LWORK,
        # integer     INFO

        if len(args) != 7:
            # this will just produce an exception in the generated code saying we can't
            # handle these arguments
            return super().convert_call(context, instance, args, kwargs)

        # determine the type from 'A'
        A = makePointer(args[1], (float, Float32))
        T = A.expr_type.typeRepresentation.ElementType
        assert T in (Float32, float), T
        nativeT = native_ast.Float64 if T is float else native_ast.Float32

        N = ensureOnStack(args[0], Int32)
        LDA = ensureOnStack(args[2], Int32)
        IPIV = makePointer(args[3], (Int32,))
        WORK = makePointer(args[4], (float, Float32))
        LWORK = makePointer(args[5], (Int32,))
        INFO = ensureOnStack(args[6], Int32)

        if not all([N, A, LDA, IPIV, WORK, LWORK, INFO]):
            return

        targetFun = externalCallTarget(
            BLAS_PREFIX + ("dgetri_" if T is float else "sgetri_"),
            native_ast.Void,
            native_ast.Int32.pointer(),
            nativeT.pointer(),
            native_ast.Int32.pointer(),
            native_ast.Int32.pointer(),
            nativeT.pointer(),
            native_ast.Int32.pointer(),
            native_ast.Int32.pointer(),
        )

        context.pushEffect(
            targetFun.call(
                N.expr,
                A.expr,
                LDA.expr,
                IPIV.expr,
                WORK.expr,
                LWORK.expr,
                INFO.expr,
            )
        )

        return INFO


@Entrypoint
def getri(N, A, LDA, IPIV, WORK, LWORK, INFO):
    """Matrix inversion

    DGETRI computes the inverse of a matrix using the LU factorization
    computed by DGETRF.

    This method inverts U and then computes inv(A) by solving the system
    inv(A)*L = inv(U) for inv(A).

    Returns:
        INFO, which is 0 on success, nonzero if there was an error.

    """
    return getri_()(N, A, LDA, IPIV, WORK, LWORK, INFO)
