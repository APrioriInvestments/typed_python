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

import math

from typed_python import (
    Class, Member, ListOf, Final, TypeFunction, Tuple, Float32, Int32, NotCompiled,
    Entrypoint
)

from typed_python.array.fortran import axpy, gemv, gemm, getri, getrf


def min(a, b):
    return a if a < b else b


@TypeFunction
def Array(T):
    """Implements a simple, strongly typed array."""
    class Array_(Class, Final):
        _vals = Member(ListOf(T))
        _offset = Member(int)
        _stride = Member(int)
        _shape = Member(int)

        dimensions = 1
        ElementType = T

        def __init__(self, vals):
            self._vals = ListOf(T)(vals)
            self._offset = 0
            self._stride = 1
            self._shape = len(vals)

        def __init__(self, vals, offset, stride, shape):  # noqa
            self._vals = vals
            self._offset = offset
            self._stride = stride
            self._shape = shape

        def __init__(self):  # noqa
            self._vals = ListOf(T)()
            self._offset = 0
            self._stride = 1
            self._shape = 0

        @property
        def shape(self):
            return Tuple(int)((self._shape,))

        def __len__(self):
            return self._shape

        def isCanonical(self):
            return self._offset == 0 and self._stride == 1 and self._shape == len(self._vals)

        ##################################################################
        # Operators
        # these are repeated below for 'matrix' because we don't have a
        # good way of doing class mixins yet.

        def __add__(self, other):
            self._inplaceBinopCheck(other)
            res = self.clone()
            res += other
            return res

        def __iadd__(self, other):
            self._inplaceBinopCheck(other)

            if (T is float or T is Float32) and isinstance(other, Array(T)):
                p = self._vals.pointerUnsafe(self._offset)
                p2 = other._vals.pointerUnsafe(other._offset)
                axpy(self._shape, 1.0, p2, self._stride, p, other._stride)
            else:
                self._inplaceBinop(other, lambda a, b: a + b)

            return self

        def __mul__(self, other):
            self._inplaceBinopCheck(other)
            res = self.clone()
            res._inplaceBinop(other, lambda a, b: a * b)
            return res

        def __imul__(self, other):
            self._inplaceBinopCheck(other)
            self._inplaceBinop(other, lambda a, b: a * b)
            return self

        def __truediv__(self, other):
            self._inplaceBinopCheck(other)
            res = self.clone()
            res._inplaceBinop(other, lambda a, b: a / b)
            return res

        def __itruediv__(self, other):
            self._inplaceBinopCheck(other)
            self._inplaceBinop(other, lambda a, b: a / b)
            return self

        def __floordiv__(self, other):
            self._inplaceBinopCheck(other)
            res = self.clone()
            res._inplaceBinop(other, lambda a, b: a // b)
            return res

        def __ifloordiv__(self, other):
            self._inplaceBinopCheck(other)
            self._inplaceBinop(other, lambda a, b: a // b)
            return self

        def __sub__(self, other):
            self._inplaceBinopCheck(other)
            res = self.clone()
            res._inplaceBinop(other, lambda a, b: a - b)
            return res

        def __isub__(self, other):
            self._inplaceBinopCheck(other)
            self._inplaceBinop(other, lambda a, b: a - b)
            return self

        def abs(self):
            self = self.clone()
            self._inplaceUnaryOp(lambda a: -a if a < 0 else a)
            return self

        def __pow__(self, p):
            self = self.clone()
            self._inplaceUnaryOp(lambda a: a ** p)
            return self

        def log(self):
            self = self.clone()
            self._inplaceUnaryOp(lambda a: math.log(a))
            return self

        def cos(self):
            self = self.clone()
            self._inplaceUnaryOp(lambda a: math.cos(a))
            return self

        def sin(self):
            self = self.clone()
            self._inplaceUnaryOp(lambda a: math.sin(a))
            return self

        def tanh(self):
            self = self.clone()
            self._inplaceUnaryOp(lambda a: math.tanh(a))
            return self

        def __neg__(self):
            self = self.clone()
            self._inplaceUnaryOp(lambda a: -a)
            return self

        def __pos__(self):
            return self.clone()

        # operators
        #########################################

        @Entrypoint
        def __matmul__(self, other: Array(T)) -> T:  # noqa
            if other.shape != self.shape:
                raise Exception(f"Mismatched array sizes: {self.shape} != {other.shape}")

            res = T()

            ownStride = self._stride
            otherStride = other._stride
            ownP = self._vals.pointerUnsafe(self._offset)
            otherP = other._vals.pointerUnsafe(other._offset)

            for i in range(self._shape):
                res += ownP.get() * otherP.get()
                ownP += ownStride
                otherP += otherStride

            return res

        def __matmul__(self, other: Matrix(T)) -> Array(T):  # noqa
            return other.__rmatmul__(self)

        @Entrypoint
        def _inplaceBinopCheck(self, other: T):
            pass

        @Entrypoint  # noqa
        def _inplaceBinopCheck(self, other: Array(T)):  # noqa
            if other._shape != self._shape:
                raise Exception("Mismatched array sizes.")

        @Entrypoint
        def _inplaceBinop(self, other: Array(T), binaryFunc):
            p = self._vals.pointerUnsafe(self._offset)
            p2 = other._vals.pointerUnsafe(other._offset)

            for i in range(self._shape):
                (p + i * self._stride).set(binaryFunc(
                    (p + i * self._stride).get(),
                    (p2 + i * other._stride).get()
                ))

            return self

        @Entrypoint
        def _inplaceUnaryOp(self, f):
            p = self._vals.pointerUnsafe(self._offset)

            for i in range(self._shape):
                p.set(f(p.get()))
                p += self._stride

        @Entrypoint  # noqa
        def _inplaceBinop(self, other: T, binaryFunc):  # noqa
            p = self._vals.pointerUnsafe(self._offset)

            for i in range(self._shape):
                (p + i * self._stride).set(binaryFunc((p + i * self._stride).get(), other))

            return self

        @Entrypoint
        def clone(self):
            return Array(T)(self.toList())

        @Entrypoint
        def toList(self):
            newVals = ListOf(T)()
            newVals.reserve(self._shape)
            pWrite = newVals.pointerUnsafe(0)
            pRead = self._vals.pointerUnsafe(self._offset)

            for i in range(self._shape):
                pWrite.set(pRead.get())
                pWrite += 1
                pRead += self._stride
            newVals.setSizeUnsafe(self._shape)

            return newVals

        @Entrypoint
        @staticmethod
        def full(count: int, value: T):
            if count < 0:
                raise Exception("Can't have a negative array size.")

            res = ListOf(T)()
            res.resize(count, value)
            return Array(T)(res)

        @Entrypoint
        def sum(self):
            res = T()

            pRead = self._vals.pointerUnsafe(self._offset)

            for i in range(self._shape):
                res += pRead.get()
                pRead += self._stride

            return res

        @staticmethod
        def ones(count):
            return Array_.full(count, 1.0)

        @staticmethod
        def zeros(count):
            return Array_.full(count, 0.0)

        def get(self, i):
            return self._vals[i * self._stride + self._offset]

        def set(self, i, value):
            self._vals[i * self._stride + self._offset] = value

        def __getitem__(self, i):
            if i < 0 or i >= self._shape:
                raise IndexError(f"Index {i} is out of bounds [0, {self._shape})")

            return self._vals[i * self._stride + self._offset]

        def __setitem__(self, i, val):
            if i < 0 or i >= self._shape:
                raise IndexError(f"Index {i} is out of bounds [0, {self._shape})")

            self._vals[i * self._stride + self._offset] = val

        def __repr__(self):
            items = ListOf(str)()
            for i in range(self._shape):
                items.append(str(self[i]))
                if i > 20:
                    items.append("...")
                    break

            return f"Array({T.__name__})([" + ", ".join(items) + "])"

        def __str__(self):
            return repr(self)

    return Array_


@TypeFunction
def Matrix(T):
    class Matrix_(Class, Final):
        _vals = Member(ListOf(T))

        # rows, then columns
        _shape = Member(Tuple(int, int))
        _offset = Member(int)
        _stride = Member(Tuple(int, int))

        dimensions = 2

        def __init__(self, vals, offset, stride, shape):
            self._vals = ListOf(T)(vals)
            self._offset = offset
            self._stride = stride
            self._shape = shape

        def __init__(self):  # noqa
            self._vals = ListOf(T)()
            self._shape = Tuple(int, int)((0, 0))
            self._stride = Tuple(int, int)((0, 1))
            self._offset = 0

        @property
        def _flatShape(self):
            return self._shape[0] * self._shape[1]

        @property
        def shape(self):
            return self._shape

        ##################################################################
        # Operators

        def __add__(self, other):
            self._inplaceBinopCheck(other)
            res = self.clone()
            res._inplaceBinop(other, lambda a, b: a + b)
            return res

        def __iadd__(self, other):
            self._inplaceBinopCheck(other)
            self._inplaceBinop(other, lambda a, b: a + b)

            return self

        def __mul__(self, other):
            self._inplaceBinopCheck(other)
            res = self.clone()
            res._inplaceBinop(other, lambda a, b: a * b)
            return res

        def __imul__(self, other):
            self._inplaceBinopCheck(other)
            self._inplaceBinop(other, lambda a, b: a * b)
            return self

        def __truediv__(self, other):
            self._inplaceBinopCheck(other)
            res = self.clone()
            res._inplaceBinop(other, lambda a, b: a / b)
            return res

        def __itruediv__(self, other):
            self._inplaceBinopCheck(other)
            self._inplaceBinop(other, lambda a, b: a / b)
            return self

        def __floordiv__(self, other):
            self._inplaceBinopCheck(other)
            res = self.clone()
            res._inplaceBinop(other, lambda a, b: a // b)
            return res

        def __ifloordiv__(self, other):
            self._inplaceBinopCheck(other)
            self._inplaceBinop(other, lambda a, b: a // b)
            return self

        def __sub__(self, other):
            self._inplaceBinopCheck(other)
            res = self.clone()
            res._inplaceBinop(other, lambda a, b: a - b)
            return res

        def __isub__(self, other):
            self._inplaceBinopCheck(other)
            self._inplaceBinop(other, lambda a, b: a - b)
            return self

        def abs(self):
            self = self.clone()
            self._inplaceUnaryOp(lambda a: -a if a < 0 else a)
            return self

        def __pow__(self, p):
            self = self.clone()
            self._inplaceUnaryOp(lambda a: a ** p)
            return self

        def log(self):
            self = self.clone()
            self._inplaceUnaryOp(lambda a: math.log(a))
            return self

        def cos(self):
            self = self.clone()
            self._inplaceUnaryOp(lambda a: math.cos(a))
            return self

        def sin(self):
            self = self.clone()
            self._inplaceUnaryOp(lambda a: math.sin(a))
            return self

        def tanh(self):
            self = self.clone()
            self._inplaceUnaryOp(lambda a: math.tanh(a))
            return self

        def __neg__(self):
            self = self.clone()
            self._inplaceUnaryOp(lambda a: -a)
            return self

        def __pos__(self):
            return self.clone()

        # operators
        #########################################

        @Entrypoint
        def _inplaceBinopCheck(self, other: T) -> None:
            pass

        @Entrypoint  # noqa
        def _inplaceBinopCheck(self, other: Matrix(T)) -> None:  # noqa
            if other.shape[0] != self.shape[0]:
                raise Exception("Mismatched array sizes.")

            if other.shape[1] != self.shape[1]:
                raise Exception("Mismatched array sizes.")

        @Entrypoint
        def _inplaceBinop(self, other: Matrix(T), binaryFunc):
            pSelf = self._vals.pointerUnsafe(self._offset)
            pOther = other._vals.pointerUnsafe(self._offset)

            for i0 in range(self._shape[0]):
                for i1 in range(self._shape[1]):
                    (pSelf + i1 * self._stride[1]).set(
                        binaryFunc(
                            (pSelf + i1 * self._stride[1]).get(),
                            (pOther + i1 * other._stride[1]).get()
                        )
                    )

                pSelf += self._stride[0]
                pOther += other._stride[0]

            return self

        @Entrypoint  # noqa
        def _inplaceBinop(self, other: T, binaryFunc):  # noqa
            pSelf = self._vals.pointerUnsafe(self._offset)

            for i0 in range(self._shape[0]):
                for i1 in range(self._shape[1]):
                    (pSelf + i1 * self._stride[1]).set(
                        binaryFunc(
                            (pSelf + i1 * self._stride[1]).get(),
                            other
                        )
                    )
                pSelf += self._stride[0]

            return self

        def _inplaceUnaryOp(self, f):
            p = self._vals.pointerUnsafe(self._offset)

            for i in range(self._shape[0]):
                pRow = p
                for j in range(self._shape[1]):
                    pRow.set(f(pRow.get()))
                    pRow += self._stride[1]

                p += self._stride[0]

        @Entrypoint
        def toList(self):
            newVals = ListOf(T)()
            newVals.reserve(self._flatShape)

            pWrite = newVals.pointerUnsafe(0)

            pSelf = self._vals.pointerUnsafe(self._offset)

            for i0 in range(self._shape[0]):
                for i1 in range(self._shape[1]):
                    pWrite.set((pSelf + i1 * self._stride[1]).get())
                    pWrite += 1

                pSelf += self._stride[0]

            newVals.setSizeUnsafe(self._flatShape)

            return newVals

        @Entrypoint
        def clone(self):
            newVals = self.toList()
            return Matrix(T)(newVals, 0, Tuple(int, int)((self._shape[1], 1)), self._shape)

        @staticmethod
        def full(rows: int, columns: int, value: T):
            if rows < 0 or columns < 0:
                raise Exception("Matrix dimensions can't be negative")

            res = ListOf(T)()
            res.resize(rows * columns, value)

            return Matrix(T)(res, 0, Tuple(int, int)((columns, 1)), Tuple(int, int)((rows, columns)))

        @Entrypoint
        @staticmethod
        def make(rows: int, columns: int, f):
            if rows < 0 or columns < 0:
                raise Exception("Can't have negative bounds")

            res = ListOf(T)()
            res.reserve(rows * columns)
            p = res.pointerUnsafe(0)

            for r in range(rows):
                for c in range(columns):
                    p.set(T(f(r, c)))
                    p += 1

            res.setSizeUnsafe(rows * columns)

            return Matrix(T)(res, 0, Tuple(int, int)((columns, 1)), Tuple(int, int)((rows, columns)))

        @staticmethod
        def ones(rows: int, columns: int):
            if rows < 0 or columns < 0:
                raise Exception("Can't have negative bounds")

            return Matrix_.full(rows, columns, 1.0)

        @staticmethod
        def zeros(rows: int, columns: int):
            if rows < 0 or columns < 0:
                raise Exception("Can't have negative bounds")

            return Matrix_.full(rows, columns, 0.0)

        @Entrypoint
        @staticmethod
        def identity(x: int):
            m = Matrix_.zeros(x, x)
            for i in range(x):
                m._vals[x * i + i] = 1
            return m

        def __getitem__(self, i: int):
            # grab a row of the data
            if i < 0 or i >= self._shape[0]:
                raise IndexError(f"Index {i} is out of bounds [0, {self._shape[0]})")

            return Array(T)(
                self._vals,
                self._offset + i * self._stride[0],
                self._stride[1],
                self._shape[1]
            )

        @Entrypoint
        def __setitem__(self, i: int, val: Array(T)):
            # set a row of the matrix
            if i < 0 or i >= self._shape[0]:
                raise IndexError(f"Index {i} is out of bounds [0, {self._shape[0]})")

            if len(val) != self._shape[1]:
                raise ValueError(
                    f"Target array is the wrong size: {len(val)} != {self._shape[1]}"
                )

            sourcePtr = val._vals.pointerUnsafe(val._offset)
            sourceStride = val._stride

            destPtr = self._vals.pointerUnsafe(self._offset + self._stride[0] * i)
            destStride = self._stride[1]

            for i in range(self._shape[1]):
                destPtr.set(sourcePtr.get())
                sourcePtr += sourceStride
                destPtr += destStride

        def transpose(self):
            return Matrix(T)(
                self._vals,
                self._offset,
                Tuple(int, int)((self._stride[1], self._stride[0])),
                Tuple(int, int)((self._shape[1], self._shape[0]))
            )

        def diagonal(self):
            return Array(T)(
                self._vals,
                self._offset,
                self._stride[0] + self._stride[1],
                min(self._shape[0], self._shape[1])
            )

        def get(self, i, j):
            return self._vals[i * self._stride[0] + j * self._stride[1] + self._offset]

        def set(self, i, j, value):
            self._vals[i * self._stride[0] + j * self._stride[1] + self._offset] = value

        def __matmul__(self, other: Matrix(T)):
            if self._shape[1] != other._shape[0]:
                raise Exception("Size mismatch")

            if self._stride[1] != 1:
                self = self.clone()

            if other._stride[1] != 1:
                other = other.clone()

            result = Matrix(T).zeros(self._shape[0], other._shape[1])

            gemm(
                'N',
                'N',
                other._shape[1],
                self._shape[0],
                self._shape[1],
                1.0,
                other._vals.pointerUnsafe(other._offset),
                other._stride[0],
                self._vals.pointerUnsafe(self._offset),
                self._stride[0],
                0.0,
                result._vals.pointerUnsafe(result._offset),
                result._stride[0],
            )

            return result

        def flatten(self):
            return Array(T)(self.toList())

        def __invert__(self):
            if self._shape[0] != self._shape[1]:
                raise Exception("Can't invert a non-square matrix")

            if self._shape[0] == 0:
                raise Exception("Can't invert an empty matrix")

            selfT = self.transpose().clone()

            ipiv = ListOf(Int32)()
            ipiv.resize(selfT._shape[1])

            getrf(
                selfT._shape[1],
                selfT._shape[0],
                selfT._vals,
                selfT._stride[0],
                ipiv,
                0
            )

            # do a workspace query
            work = ListOf(float)([1])
            lwork = ListOf(Int32)([-1])

            info = getri(
                selfT._shape[0], selfT._vals, selfT._stride[0], ipiv, work, lwork, 0
            )

            if info != 0:
                raise Exception(
                    "Malformed problem: couldn't even compute an LWORK: " + str(info)
                )

            work.resize(int(work[0]) * 2)
            lwork[0] = work[0]

            selfT._vals.resize(len(selfT._vals) * 2)
            ipiv.resize(len(ipiv) * 2)
            work.resize(len(work) * 2)

            info = getri(
                selfT._shape[0],
                selfT._vals.pointerUnsafe(selfT._offset),
                selfT._stride[0],
                ipiv.pointerUnsafe(0),
                work.pointerUnsafe(0),
                lwork.pointerUnsafe(0),
                0
            )

            if info != 0:
                # return an array of nan instead of throwing an exception,
                # as it's a data error (not a structural error)
                return selfT * math.nan

            return selfT.transpose()

        def __matmul__(self, other: Array(T)):  # noqa
            # ensures that this is a simply-strided (row-major) matrix
            if self._stride[1] != 1:
                self = self.clone()

            result = ListOf(T)()
            result.resize(self._shape[0])

            if self._shape[1] != other._shape:
                raise Exception("Size mismatch")

            gemv(
                'T',
                self._shape[1],
                self._shape[0],
                1.0,
                self._vals,
                self._stride[0],
                other._vals.pointerUnsafe(other._offset),
                other._stride,
                1.0,
                result,
                1
            )

            return Array(T)(result)

        def __rmatmul__(self, other: Array(T)):
            if self._stride[1] != 1:
                self = self.clone()

            result = ListOf(T)()
            result.resize(self._shape[1])

            if self._shape[0] != other._shape:
                raise Exception(f"Size mismatch: {self._shape[1]} != {other._shape}")

            gemv(
                'N',
                self._shape[1],
                self._shape[0],
                1.0,
                self._vals,
                self._stride[0],
                other._vals.pointerUnsafe(other._offset),
                other._stride,
                1.0,
                result,
                1
            )

            return Array(T)(result)

        @NotCompiled
        def __repr__(self):
            rows = ListOf(str)()

            for rowIx in range(self._shape[0]):
                items = ListOf(str)()
                row = self[rowIx]

                for j in range(self._shape[1]):
                    item = f"{row[j]:.6g}"

                    if len(item) < 8:
                        item = item + " " * (8 - len(item))

                    items.append(item)

                    if j > 20:
                        items.append("...")
                        break

                rows.append("[" + " ".join(items) + "]")

                if rowIx > 20:
                    rows.append("...")
                    break

            return f"Matrix({T.__name__})(\n    " + "\n    ".join(rows) + "\n])"

        def __str__(self):
            return repr(self)

    return Matrix_
