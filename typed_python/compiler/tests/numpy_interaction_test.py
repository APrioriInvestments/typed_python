from typed_python import ListOf, Entrypoint
import numpy
import numpy.linalg


def test_convert_list_to_numpy_array():
    aList = ListOf(int)(range(20))
    assert isinstance(aList.toArray(), numpy.ndarray)


def test_can_add_numpy_arrays_in_compiled_code():
    @Entrypoint
    def add(x):
        return x + x

    assert isinstance(add(numpy.ones(10)), numpy.ndarray)


def test_can_call_numpy_builtins_from_compiled_code():
    @Entrypoint
    def callSin(x):
        return numpy.sin(x)

    assert isinstance(callSin(numpy.ones(10).cumsum()), numpy.ndarray)

    @Entrypoint
    def callF(f, x):
        return f(x)

    assert isinstance(callF(numpy.sin, numpy.ones(10).cumsum()), numpy.ndarray)


def test_can_call_numpy_matrix_funs():
    @Entrypoint
    def callDiagonal(x):
        a = numpy.identity(10)
        return numpy.diagonal(a)

    assert callDiagonal(10).tolist() == [1 for _ in range(10)]


def test_listof_from_sliced_numpy_array():
    x = numpy.array((0, 1, 2))
    y = x[::2]

    assert ListOf(int)(y) == [0, 2]
