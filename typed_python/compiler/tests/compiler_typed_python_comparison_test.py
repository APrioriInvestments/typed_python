#   Copyright 2017-2019 typed_python Authors
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
import typed_python
import numpy
import unittest
import traceback

from typed_python.compiler.runtime import Runtime


class Operation:
    """Base class for all operations we want to test.

    This class models executing the operation in both the
    interpreter and the compiler, and describing which deviations
    between the interpreter and the compiler we'd expect.
    """
    def __init__(self):
        self._compiledFunctionCache = {}

    def arity(self):
        """Return the number of arguments we expect."""
        raise NotImplementedError(self)

    def name(self):
        """return our name."""
        raise NotImplementedError(self)

    def getLambda(self):
        """Return a lambda function representing the operation.

        The arguments must have names a1, a2, ...
        """
        raise NotImplementedError(self)

    def getCompiledLambda(self, types):
        if types not in self._compiledFunctionCache:
            self._compiledFunctionCache[types] = Runtime.singleton().compile(
                self.getLambda(),
                {f"a{i + 1}": types[i] for i in range(self.arity())}
            )

        return self._compiledFunctionCache[types]

    def subsetOfTypesWorthTesting(self, argIndex, typeList):
        """Return the subset of Type instances worth testing in the given argument."""
        return typeList

    def expectInterpreterDeviation(self, values):
        """Would we expect the interpreter and typed_python to be different?"""
        return False

    def expectCompilerDeviation(self, values, compilerTypes):
        """Would we expect typed_python and compiler to produce a deviation?"""
        return False


class Type:
    """Base class for all types we want to test.

    This class models the process of producing instances of the type, understanding which
    other types we can use to represent this type in a valid form, and understanding
    what more basic instances of a type we'd expect to behave like.
    """

    def name(self):
        """Return the name"""
        return self.pytype().__name__

    def pytype(self):
        """Return the actual type object we represent."""
        raise NotImplementedError(self)

    def instances(self):
        """Produce some instances of this type."""
        raise NotImplementedError(self)

    def isOutOfBounds(self, instance):
        return False

    def equivalentOwnInstance(self, interpInstance):
        """Given an interpreter instance, represent it as ourself.

        Returns (True, obj) or (False, None)
        """
        return (False, None)

    def containingTypes(self):
        """Produce a list of total types that the compiler should be willing to cast this to."""
        return [self.pytype(), object]

    def equivalentInterpreterInstance(self, instance):
        """Return a representation of the object as an interpreter object.

        If all values in an operation have an equivalent interpreter instance, then it should
        be the case that the operation produces the same result on the interpreter values.

        Returns:
            (True, obj) if there is an equivalent object or (False, None) if not.
        """
        return (False, None)

    def areEquivalent(self, instanceA, instanceB):
        return instanceA == instanceB


def isnan(x):
    if isinstance(x, float):
        return numpy.isnan(x)

    if isinstance(x, typed_python.Float32):
        return isnan(float(x))

    return False


def isinf(x):
    if isinstance(x, float):
        return numpy.isinf(x)

    if isinstance(x, typed_python.Float32):
        return isinf(float(x))

    return False


def isfinite(x):
    if isinstance(x, float):
        return numpy.isfinite(x)

    if isinstance(x, typed_python.Float32):
        return isfinite(float(x))

    return True


def isneginf(x):
    if isinstance(x, float):
        return numpy.isneginf(x)

    if isinstance(x, typed_python.Float32):
        return isneginf(float(x))

    return False


def isint(x):
    try:
        int(x)
        return True
    except Exception:
        return False


def isfloat(x):
    try:
        float(x)
        return True
    except Exception:
        return False


class ArithmeticOperation(Operation):
    def __init__(self, name):
        super().__init__()
        self._name = name

    def arity(self):
        return 2

    def name(self):
        return self._name

    def getLambda(self):
        name = self._name

        if name == "add":
            return lambda a1, a2: a1 + a2
        if name == "sub":
            return lambda a1, a2: a1 - a2
        if name == "mul":
            return lambda a1, a2: a1 * a2
        if name == "truediv":
            return lambda a1, a2: a1 / a2
        if name == "floordiv":
            return lambda a1, a2: a1 // a2
        if name == "and":
            return lambda a1, a2: a1 & a2
        if name == "or":
            return lambda a1, a2: a1 | a2
        if name == "xor":
            return lambda a1, a2: a1 ^ a2
        if name == "lshift":
            return lambda a1, a2: a1 << a2
        if name == "rshift":
            return lambda a1, a2: a1 >> a2
        if name == "mod":
            return lambda a1, a2: a1 % a2
        if name == "pow":
            return lambda a1, a2: a1 ** a2

        raise Exception(f"unknown op {self._name}")

    def expectInterpreterDeviation(self, values):
        if self._name in ("truediv", "floordiv"):
            if isinf(values[1]):
                return True

        if self._name in ("floordiv", 'mod', "pow", "rshift"):
            for v in values:
                if getattr(type(v), "IsUnsignedInt", False):
                    # unsigned integer values at the top of the range don't behave correctly
                    # because we cast them to signed values which wraps them around to negative
                    # numbers
                    if int(type(v)(v * 2)) != v:
                        return True

        if self._name == "mod" and type(values[0]) == typed_python.Int32 and type(values[1]) == typed_python.Float32:
            # int32 % float32 with integers close to the float cutoff can have odd roundoff errors
            if typed_python.Int32(typed_python.Float32(v)) != values[0]:
                return True

        if self._name in ("lshift", "pow"):
            if isint(values[0]) and isint(values[1]) and abs(int(values[1])) > 63:
                return True

        if self._name == "pow" and (isint(values[0]) or isfloat(values[0])) and values[0] < 0:
            return True

        if self._name == "pow" and isinf(values[1]):
            return True

        return False

    def expectCompilerDeviation(self, values, compilerTypes):
        if self.expectInterpreterDeviation(values):
            return True

        return False


class Float(Type):
    def pytype(self):
        return float

    def instances(self):
        return [-1e100, -1e10, -2.0, -1.0, -.5, 0.0, .5, 1.0, 2.0, 1e10, 1e100, numpy.nan, numpy.inf, -numpy.inf]

    def equivalentOwnInstance(self, interpInstance):
        return (True, interpInstance)

    def equivalentInterpreterInstance(self, instance):
        return (True, instance)

    def areEquivalent(self, instanceA, instanceB):
        if isfinite(instanceA) and isfinite(instanceB):
            if abs(instanceA) > 1.0:
                return (abs(instanceA) - abs(instanceB)) / (abs(instanceA) + abs(instanceB)) < 1e-10

            return abs(instanceA - instanceB) < 1e-15

        if isnan(instanceA) and isnan(instanceB):
            return True

        if isinf(instanceA) and isinf(instanceB):
            return True

        if isneginf(instanceA) and isneginf(instanceB):
            return True

        return False


class Float32(Type):
    def pytype(self):
        return typed_python.Float32

    def isOutOfBounds(self, instance):
        if isfinite(instance) and typed_python.Float32(instance + 1) == instance:
            return True
        return False

    def instances(self):
        return [typed_python.Float32(f) for f in Float().instances()]

    def equivalentOwnInstance(self, interpInstance):
        return (True, typed_python.Float32(interpInstance))

    def equivalentInterpreterInstance(self, instance):
        return (True, float(instance))

    def areEquivalent(self, instanceA, instanceB):
        return Float().areEquivalent(float(instanceA), float(instanceB))


class Int(Type):
    def pytype(self):
        return int

    def instances(self):
        return [
            -(1 << 63) + 1, -(1 << 31) + 1, -(1 << 15) + 1, -(1 << 8) + 1, -10, -1, -0,
            1, 10, 127, 255, (1 << 15) - 1, (1 << 31) - 1, (1 << 63) - 1
        ]

    def isOutOfBounds(self, instance):
        return (instance >= (1 << 63)) or (instance <= -(1 << 63))

    def equivalentOwnInstance(self, interpInstance):
        return (True, interpInstance)

    def equivalentInterpreterInstance(self, instance):
        return (True, instance)


class SmallInt(Type):
    def __init__(self, tpType):
        self.tpType = tpType
        super().__init__()

    def pytype(self):
        return self.tpType

    def instances(self):
        return [self.tpType(x) for x in Int().instances()]

    def isOutOfBounds(self, instance):
        maxSize = 1 << (self.tpType.Bits + (1 if self.tpType.IsUnsignedInt else 0))
        return (instance >= maxSize) or (instance <= -maxSize)

    def equivalentOwnInstance(self, interpInstance):
        return (True, self.tpType(interpInstance))

    def equivalentInterpreterInstance(self, instance):
        return (True, int(instance))


class Bool(Type):
    def pytype(self):
        return bool

    def instances(self):
        return [False, True]

    def equivalentOwnInstance(self, interpInstance):
        return (True, interpInstance)

    def equivalentInterpreterInstance(self, instance):
        return (True, instance)


allTypes = [
    Float(),
    Float32(),
    Int(),
    Bool(),
    SmallInt(typed_python.Int32),
    SmallInt(typed_python.Int16),
    SmallInt(typed_python.Int8),
    SmallInt(typed_python.UInt64),
    SmallInt(typed_python.UInt32),
    SmallInt(typed_python.UInt16),
    SmallInt(typed_python.UInt8)
]


def typeModelForValue(val):
    if isinstance(val, float):
        return Float()

    if isinstance(val, typed_python.Float32):
        return Float32()

    if isinstance(val, int):
        return Int()

    if isinstance(val, bool):
        return Bool()

    if getattr(type(val), "Bits", None) is not None:
        return SmallInt(type(val))


class Scenario:
    def __init__(self, op, argTypes, values, compileTypes):
        self.op = op
        self.argTypes = argTypes
        self.compileTypes = compileTypes
        self.values = values
        self.failureDesc = "didn't fail?"

    def check(self):
        # check interpreter against typed_python
        if not self.checkAgainstInterpreter():
            return False

        if not self.checkAgainstCompiler():
            return False

        return True

    def checkAgainstCompiler(self):
        if self.op.expectCompilerDeviation(self.values, self.compileTypes):
            return True

        try:
            compiledForm = self.op.getCompiledLambda(self.compileTypes)
        except Exception:
            self.failureDesc = "Failed to compile at all:\n" + traceback.format_exc()
            return False

        try:
            typedPythonVal = self.op.getLambda()(*self.values)
            typedPythonException = False
        except Exception:
            typedPythonException = traceback.format_exc()

        try:
            compiledVal = compiledForm(*self.values)
            compiledException = False
        except Exception:
            compiledException = traceback.format_exc()

        if compiledException and typedPythonException:
            return True

        if typedPythonException:
            self.failureDesc = "typed_python produced an exception but interpreter didn't:\n" + typedPythonException
            return False

        if compiledException:
            self.failureDesc = "compiler produced an exception but typed_python didn't:\n" + compiledException
            return False

        if type(typedPythonVal) != type(compiledVal):
            self.failureDesc = (
                f"Compiler produced {compiledVal} of type {type(compiledVal)}.\n"
                f"TypedPython produced {typedPythonVal} of type {type(typedPythonVal)}.\n"
                f"The types are not the same."
            )
            return False

        typedPythonValType = typeModelForValue(typedPythonVal)

        if typedPythonValType is None:
            self.failureDesc = (
                f"TypedPython produced {typedPythonVal} of type {type(typedPythonVal)}, and we don't have a type model for it."
            )
            return False

        if typedPythonValType.isOutOfBounds(typedPythonVal) or typedPythonValType.isOutOfBounds(compiledVal):
            # the interpreter's value is out of bounds for this value, so we can ignore it
            return True

        if not typedPythonValType.areEquivalent(typedPythonVal, compiledVal):
            self.failureDesc = (
                f"Compiler produced {compiledVal} of type {type(compiledVal)}.\n"
                f"TypedPython produced {typedPythonVal} of type {type(typedPythonVal)}.\n"
                f"These are not equivalent according to type model '{typedPythonValType.name()}'\n"
            )
            return False

        return True

    def checkAgainstInterpreter(self):
        """If all of the arguments here have a pure python form, check that against the typed-python form."""

        # don't generate an interpreter check for all the different variations of compile types.
        if not all(self.compileTypes[i] == self.argTypes[i].pytype() for i in range(len(self.compileTypes))):
            return True

        if self.op.expectInterpreterDeviation(self.values):
            return True

        # get a list of interpreter values that would be equivalent
        interpreterValues = []
        for i in range(len(self.argTypes)):
            hasInterpreterEquivalent, interpVal = self.argTypes[i].equivalentInterpreterInstance(self.values[i])
            if hasInterpreterEquivalent:
                interpreterValues.append(interpVal)

        if len(interpreterValues) != len(self.values):
            # we can't run this test because we don't have an equivalent for all values
            return True

        # check if the typedPython value is really the same as the interpreter value.
        try:
            interpVal = self.op.getLambda()(*interpreterValues)
            interpException = False
        except Exception:
            interpException = True

        try:
            typedPythonVal = self.op.getLambda()(*self.values)
            typedPythonException = False
        except Exception:
            typedPythonException = True

        if typedPythonException and interpException:
            return True

        if typedPythonException:
            self.failureDesc = "typed_python produced an exception but interpreter didn't."
            return False

        if interpException:
            self.failureDesc = "interpreter produced an exception but typed_python didn't."
            return False

        typedPythonValType = typeModelForValue(typedPythonVal)

        if typedPythonValType is None:
            self.failureDesc = (
                f"TypedPython produced {typedPythonVal} of type {type(typedPythonVal)}, and we don't have a type model for it."
            )
            return False

        hasInterpValRep, interpValAsTpVal = typedPythonValType.equivalentOwnInstance(interpVal)
        if not hasInterpValRep:
            self.failureDesc = (
                f"TypedPython produced {typedPythonVal} of type {type(typedPythonVal)}, which cannot represent the "
                f"interpreter's values of {interpVal} of type {type(interpVal)}."
            )
            return False

        if typedPythonValType.equivalentInterpreterInstance(interpValAsTpVal)[1] != interpVal:
            # the interpreter's value is out of bounds for this value, so we can ignore it
            return True

        if typedPythonValType.isOutOfBounds(interpValAsTpVal):
            return True

        if not typedPythonValType.areEquivalent(typedPythonVal, interpValAsTpVal):
            self.failureDesc = (
                f"Interpreter produced {interpVal} of type {type(interpVal)} using vals {interpreterValues}.\n"
                f"TypedPython produced {typedPythonVal} of type {type(typedPythonVal)}.\n"
                f"We cast the interpreter val to {interpValAsTpVal} of type {type(interpValAsTpVal)}.\n"
                f"These are not equivalent according to type model '{typedPythonValType.name()}'\n"
            )
            return False

        return True

    def failureDescription(self):
        res = f"Op '{self.op.name()}' failed on arguments:\n"
        for i in range(len(self.argTypes)):
            res += (
                f"    {str(self.values[i]):40s} of type {str(self.argTypes[i].name()):40s} "
                f"compiled as {str(self.compileTypes[i]):40s}\n"
            )
        res += "Failure:" + ("\n" + self.failureDesc).replace("\n", "\n    ") + "\n"

        return res


class TestTypedPythonAgainstCompiler(unittest.TestCase):
    """Systematically compare typed_python, the interpreter, and the compiler.

    We rely on two main invariants. First, the compiler should cause
    us to produce the same outputs as we would get when we run typed_python code
    in the interpreter.  Second, typed_python types are intended to work like their
    untyped counterparts: as long as all the datatypes are representable in equivalent
    forms between the untyped and typed versions, adding the typing shouldn't change
    the outcome.

    This test suite attempts to systematically verify that that is true. Because
    some functionality is not implemented, we provide functions to suppress errors
    when something doesn't work yet.
    """

    def callOrException(self, f, *args):
        try:
            return f(*args)
        except Exception:
            return 'Exception'

    def checkOperation(self, op):
        scenarios = self.scenariosWithArgs(op, ())

        print(f"Checking {op.name()} with {len(scenarios)} scenarios over {len(set(s.argTypes for s in scenarios))} signatures.")

        failed = []
        for s in scenarios:
            if not s.check():
                failed.append(s)

        if failed:
            print(f"Total failures: {len(failed)}")
            for f in failed:
                print(f.failureDescription())

        if failed:
            self.assertFalse(True, "Failures exist.")

    def scenariosWithArgs(self, op, argTypes):
        if len(argTypes) < op.arity():
            scenarios = []

            for possibleArgType in op.subsetOfTypesWorthTesting(len(argTypes), allTypes):
                scenarios.extend(self.scenariosWithArgs(op, argTypes + (possibleArgType,)))

            return scenarios

        return self.scenariosWithArgsAndValues(op, argTypes, ())

    def scenariosWithArgsAndValues(self, op, argTypes, values):
        if len(values) < op.arity():
            scenarios = []

            for possibleValue in argTypes[len(values)].instances():
                scenarios.extend(self.scenariosWithArgsAndValues(op, argTypes, values + (possibleValue,)))

            return scenarios

        return self.scenariosWithArgsAndValuesAndUptypes(op, argTypes, values, ())

    def scenariosWithArgsAndValuesAndUptypes(self, op, argTypes, values, actualTypes):
        if len(actualTypes) < op.arity():
            scenarios = []

            for uptype in argTypes[len(actualTypes)].containingTypes():
                scenarios.extend(self.scenariosWithArgsAndValuesAndUptypes(op, argTypes, values, actualTypes + (uptype,)))

            return scenarios

        return [Scenario(op, argTypes, values, actualTypes)]

    def test_add(self):
        self.checkOperation(ArithmeticOperation("add"))

    def test_mul(self):
        self.checkOperation(ArithmeticOperation("mul"))

    def test_sub(self):
        self.checkOperation(ArithmeticOperation("sub"))

    def test_truediv(self):
        self.checkOperation(ArithmeticOperation("truediv"))

    def test_floordiv(self):
        self.checkOperation(ArithmeticOperation("floordiv"))

    def test_mod(self):
        self.checkOperation(ArithmeticOperation("mod"))

    # we are not currently getting the types of 'pow' right. int ** int should be int.
    @pytest.mark.skip
    def test_pow(self):
        self.checkOperation(ArithmeticOperation("pow"))

    def test_lshift(self):
        self.checkOperation(ArithmeticOperation("lshift"))

    # we have numerous failures here
    @pytest.mark.skip
    def test_rshift(self):
        self.checkOperation(ArithmeticOperation("rshift"))

    def test_and(self):
        self.checkOperation(ArithmeticOperation("and"))

    def test_or(self):
        self.checkOperation(ArithmeticOperation("or"))

    def test_xor(self):
        self.checkOperation(ArithmeticOperation("xor"))
