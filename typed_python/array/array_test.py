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

import pytest
from flaky import flaky
import time
import numpy
import os

from typed_python.test_util import estimateFunctionMultithreadSlowdown
from typed_python.array.array import Array, Matrix
from typed_python import Entrypoint, Float32

TYPES_TO_CHECK = [float, Float32]


def test_float_array_addition():
    for typ in TYPES_TO_CHECK:
        x = Array(typ)([1, 2, 3])

        y = x + x

        assert y[0] == 2

        y += y

        assert y[0] == 4

        y += x

        assert y[0] == 5

        y = y + 1

        assert y[0] == 6

        y += 2

        assert y[0] == 8


def test_clone():
    """ Clone does not make an exact replica. It just makes the values the same."""
    stride = 2
    offset = 0
    for typ in TYPES_TO_CHECK:
        x = Array(typ)([1, 2, 3, 4, 5, 6], offset, stride, 3)
        y = x.clone()

        assert x._stride == 2
        assert y._stride == 1

        assert y[0] == 1.0
        assert y[1] == 3.0
        assert y[2] == 5.0

        assert x[0] == 1.0
        assert x[1] == 3.0
        assert x[2] == 5.0

        assert len(x) == 3
        assert len(y) == 3


def test_stride_and_offset():
    stride = 2
    offset = 0
    for typ in TYPES_TO_CHECK:
        x = Array(typ)([1, 2, 3, 4, 5, 6], offset, stride, 3)
        assert x.toList() == [1.0, 3.0, 5.0]

        stride = 2
        offset = 1
        x = Array(typ)([1, 2, 3, 4, 5, 6], offset, stride, 3)
        assert x.toList() == [2.0, 4.0, 6.0]


def test_float_array_subtraction():
    for typ in TYPES_TO_CHECK:
        x = Array(typ)([1, 2, 3])

        y = x - x

        assert y[0] == 0
        assert y[1] == 0
        assert y[2] == 0

        y -= x

        assert y[0] == -1
        assert y[1] == -2
        assert y[2] == -3

        y = y - 1

        assert y[0] == -2
        assert y[1] == -3
        assert y[2] == -4

        y -= 2

        assert y[0] == -4
        assert y[1] == -5
        assert y[2] == -6


def test_float_array_multiplication():
    for typ in TYPES_TO_CHECK:
        x = Array(typ)([1, 2, 3])

        y = x * x
        assert y[0] == 1
        assert y[1] == 4
        assert y[2] == 9

        y *= y
        assert y[0] == 1
        assert y[1] == 16
        assert y[2] == 81

        y *= x
        assert y[0] == 1
        assert y[1] == 32
        assert y[2] == 243

        y = y * 2
        assert y[0] == 2
        assert y[1] == 64
        assert y[2] == 486

        y *= 2
        assert y[0] == 4
        assert y[1] == 128
        assert y[2] == 972

        assert len(y) == 3


def test_float_array_addition_wrong_size():
    for typ in TYPES_TO_CHECK:
        x = Array(typ).ones(3)
        x2 = Array(typ).zeros(4)

        with pytest.raises(Exception):
            x + x2

        with pytest.raises(Exception):
            x += x2


def test_basic_matrix_ops():
    for typ in TYPES_TO_CHECK:
        m = Matrix(typ).identity(10)

        m = m + 1
        m[1][4] = 100

        m = m - m.transpose()

        assert m[1][4] == 99
        assert m[4][1] == -99


@flaky(max_runs=3, min_passes=1)
def test_matrix_speed():
    # this test requires parallelism, and the machines we're using in travis
    # don't always have free cores, so we get false negatives.
    if os.environ.get('TRAVIS_CI', None) is not None:
        return

    t0 = time.time()
    sz = 10
    for typ in TYPES_TO_CHECK:
        m = Matrix(typ).identity(sz)
        m_numpy = numpy.identity(sz)

        def manyAdds(m, count):
            # we actually need to get separate copies of these, or else
            # we end up not getting 2x parallelism because of the atomic refcounts!
            m = m * 1
            m2 = m * 0

            for i in range(count):
                m2 += m
                m2 += m.transpose()
                m2[1][2] += 2

            return m2

        manyAddsCompiled = Entrypoint(manyAdds)
        manyAddsCompiled(m, 1)

        t0 = time.time()
        manyAddsCompiled(m, 100000)
        t1 = time.time()
        manyAdds(m_numpy, 100000)
        t2 = time.time()

        print("typed_python took ", t1 - t0)
        print("numpy took ", t2 - t1)

        assert t1 - t0 < t2 - t1

        numpySlowdown = estimateFunctionMultithreadSlowdown(lambda: manyAdds(m_numpy, 100000))
        tpSlowdown = estimateFunctionMultithreadSlowdown(lambda: manyAddsCompiled(m, 100000))

        print("numpy slows down by ", numpySlowdown)
        print("tp slows down by ", tpSlowdown)

        # anything much less than 2 is a good test. But on the travis boxes
        # we don't actually get two full cores, so we have to have a wider
        # threshold. On my desktop I get ~ 1.0 for the tpSlowdown and 2.0
        # for the numpy slowdown because of the GIL.
        SLOWDOWN_THRESHOLD = 1.8 if os.environ.get('TRAVIS_CI', None) else 1.2

        assert numpySlowdown > 1.8, numpySlowdown
        assert tpSlowdown < SLOWDOWN_THRESHOLD, tpSlowdown


def test_square_matrix_vector_multiply():
    for typ in TYPES_TO_CHECK:
        m = Matrix(typ).identity(3)

        a = Array(typ)([1, 0, 0])

        for i in range(3):
            for j in range(3):
                m[i][j] = i * 10 + j

        # this picks out the first column
        assert (m @ a).toList() == [0, 10, 20]

        a[0] = 0
        a[1] = 1
        assert (m @ a).toList() == [1, 11, 21]


def test_rectangular_matrix_vector_multiply():
    for typ in TYPES_TO_CHECK:
        m = Matrix(typ).zeros(8, 4)

        for i in range(8):
            for j in range(4):
                m[i][j] = i * 10 + j

        a = Array(typ)([1, 0, 0, 0])

        # this picks out the first column
        assert (m @ a).toList() == [0, 10, 20, 30, 40, 50, 60, 70]

        a[0] = 0
        a[1] = 1

        # this picks out the second column
        assert (m @ a).toList() == [1, 11, 21, 31, 41, 51, 61, 71]

        a2 = Array(typ)([1, 0, 0, 0, 0, 0, 0, 0])
        # this picks out the first row

        assert (a2 @ m).toList() == m[0].toList()


def test_matrix_multiply():
    for typ in TYPES_TO_CHECK:
        m = Matrix(typ).identity(4)
        m2 = Matrix(typ).identity(4)

        m.diagonal()[2] = 2
        m2.diagonal()[2] = 3

        m.diagonal()[1] = 3
        m2.diagonal()[3] = 3

        assert (m @ m2).diagonal().toList() == [1, 3, 6, 3]

        m *= 0
        m2 *= 0

        m[1][0] = 1
        m2[0][3] = 1

        print(m @ m2)
        assert (m @ m2)[1][3] == 1


def test_rectangular_matrix_multiply():
    for typ in TYPES_TO_CHECK:
        m = Matrix(typ).zeros(5, 2)
        m2 = Matrix(typ).zeros(2, 4)

        assert (m @ m2).shape == (5, 4)

        m[0][1] = 1.0
        m2[1][3] = 1.0

        assert (m @ m2)[0][3] == 1.0


def l1norm(m):
    return m.flatten().abs().sum()


def test_invert():
    for typ in TYPES_TO_CHECK:
        m = Matrix(typ).identity(10) * 2

        assert l1norm((m @ ~m) - Matrix(typ).identity(10)) < 1e-10

        m += Matrix(typ).ones(10, 10)

        assert l1norm((m @ ~m) - Matrix(typ).identity(10)) < 1e-10


def test_create_matrix():
    for typ in TYPES_TO_CHECK:
        m = Matrix(typ).make(10, 10, lambda row, col: row * 20 - col)

        assert m[3][4] == 3 * 20 - 4


def test_negate_matrix():
    for typ in TYPES_TO_CHECK:
        m = Matrix(typ).identity(10)

        assert l1norm(+m + -m) < 1e-10


def test_matrix_cos_and_sin():
    for typ in TYPES_TO_CHECK:
        m = Matrix(typ).make(10, 10, lambda x, y: x**2 - y**2)

        assert l1norm(m.cos() ** 2 + m.sin() ** 2 - 1) < 1e-10


def test_increment_matrix_diagonal():
    for typ in TYPES_TO_CHECK:
        m = Matrix(typ).zeros(10, 10)
        d = m.diagonal()
        d += 1.0

    assert m.get(2, 2) == 1


def test_assign_matrix_row():
    m = Matrix(float).identity(10)

    m[4] = m[3]

    assert m.get(4, 3) == m.get(3, 3)

    m.transpose()[4] = m.transpose()[3]
    assert m.get(4, 4) == m.get(3, 4)
    assert m.get(2, 2) == 1
