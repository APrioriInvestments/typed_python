/******************************************************************************
   Copyright 2017-2020 typed_python Authors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

#pragma once

#include <numpy/arrayscalars.h>
#include "PyInstance.hpp"
#include "PromotesTo.hpp"
#include <cmath>

PyDoc_STRVAR(_complex_doc,
    "complex(n) -> complex\n"
    "\n"
    "Creates a complex number with real part n and zero imaginary part.\n"
    );
PyDoc_STRVAR(_round_doc,
    "round(n[, d=0]) -> number\n"
    "\n"
    "Without d, returns the integer closest to x, rounding 0.5 towards even.\n"
    "With d, rounds to precision of d decimal digits.\n"
    "d may be negative.\n"
    "Returned value has same type as n.\n"
    );
PyDoc_STRVAR(_trunc_doc,
    "trunc(n) -> number\n"
    "\n"
    "Returns n with decimal part truncated to the nearest integer toward 0.\n"
    "Returned value has same type as n.\n"
    );
PyDoc_STRVAR(_floor_doc,
    "floor(n) -> number\n"
    "\n"
    "Returns the largest integer <= n.\n"
    "Returned value has same type as n.\n"
    );
PyDoc_STRVAR(_ceil_doc,
    "ceil(n) -> number\n"
    "\n"
    "Returns the smallest integer >= n.\n"
    "Returned value has same type as n.\n"
    );

template<class T>
static PyObject* registerValueToPyValue(T val) {
    static Type* typeObj = GetRegisterType<T>()();

    return PyInstance::extractPythonObject((instance_ptr)&val, typeObj);
}

inline int64_t bitInvert(int64_t in) { return ~in; }
inline int32_t bitInvert(int32_t in) { return ~in; }
inline int16_t bitInvert(int16_t in) { return ~in; }
inline int8_t bitInvert(int8_t in) { return ~in; }
inline uint64_t bitInvert(uint64_t in) { return ~in; }
inline uint32_t bitInvert(uint32_t in) { return ~in; }
inline uint16_t bitInvert(uint16_t in) { return ~in; }
inline uint8_t bitInvert(uint8_t in) { return ~in; }
inline bool bitInvert(bool in) { return !in; }
//this never gets called, but we need it for the compiler to be happy
inline double bitInvert(double in) { return 0.0; }
inline float bitInvert(float in) { return 0.0; }

//implement mod the same way python does for unsigned integers.
inline uint64_t pyMod(uint64_t l, uint64_t r) {
    if (r == 0) {
        PyErr_Format(PyExc_ZeroDivisionError, "integer division or modulo by zero");
        throw PythonExceptionSet();
    }

    return l % r;
}

//implement mod the same way python does for signed integers
inline int64_t pyMod(int64_t l, int64_t r) {
    if (r == 1 || r == -1 || l == 0) {
        return 0;
    }

    if (r == 0) {
        PyErr_Format(PyExc_ZeroDivisionError, "integer division or modulo by zero");
        throw PythonExceptionSet();
    }

    if (r < 0) {
        if (l < 0) {
            return -((-l) % (-r));
        }
        return -((-r + ((-l) % -r)) % -r);
    }

    if (l < 0) {
        return (r + (l % r)) % r;
    }

    return l % r;
}

//implement mod the same way python does for floats and doubles
template<class T>
T pyModFloat(T l, T r) {
    if (std::isnan(r) || std::isnan(l)) {
        return NAN;
    }

    if (l == 0.0) {
        return 0.0;
    }
    if (r == 0.0) {
        PyErr_Format(PyExc_ZeroDivisionError, "float modulo");
        throw PythonExceptionSet();
    }

    double mod = fmod(l, r);
    if (mod) {
        if ((r < 0) != (mod < 0))
            mod += r;
    }
    return mod;
}

inline int32_t pyMod(int32_t l, int32_t r) { return (int32_t)pyMod((int64_t)l, (int64_t)r); }
inline int16_t pyMod(int16_t l, int16_t r) { return (int16_t)pyMod((int64_t)l, (int64_t)r); }
inline int8_t pyMod(int8_t l, int8_t r) { return (int8_t)pyMod((int64_t)l, (int64_t)r); }
inline uint32_t pyMod(uint32_t l, uint32_t r) { return (uint32_t)pyMod((uint64_t)l, (uint64_t)r); }
inline uint16_t pyMod(uint16_t l, uint16_t r) { return (uint16_t)pyMod((uint64_t)l, (uint64_t)r); }
inline uint8_t pyMod(uint8_t l, uint8_t r) { return (uint8_t)pyMod((uint64_t)l, (uint64_t)r); }
//inline bool pyMod(bool l, bool r) { return 0; }
inline double pyMod(double l, double r) { return pyModFloat(l, r); }
inline float pyMod(float l, float r) { return pyModFloat(l, r); }

inline int64_t pyAnd(int64_t l, int64_t r) { return l & r; }
inline int64_t pyAnd(int32_t l, int32_t r) { return l & r; }
inline int64_t pyAnd(int16_t l, int16_t r) { return l & r; }
inline int64_t pyAnd(int8_t l, int8_t r) { return l & r; }
inline int64_t pyAnd(uint64_t l, uint64_t r) { return l & r; }
inline int64_t pyAnd(uint32_t l, uint32_t r) { return l & r; }
inline int64_t pyAnd(uint16_t l, uint16_t r) { return l & r; }
inline int64_t pyAnd(uint8_t l, uint8_t r) { return l & r; }
inline int64_t pyAnd(double l, double r) {
    PyErr_Format(PyExc_TypeError, "'&' not supported for floating-point types");
    throw PythonExceptionSet();
}
inline int64_t pyAnd(float l, float r) {
    PyErr_Format(PyExc_TypeError, "'&' not supported for floating-point types");
    throw PythonExceptionSet();
}

inline int64_t pyOr(int64_t l, int64_t r) { return l | r; }
inline int64_t pyOr(int32_t l, int32_t r) { return l | r; }
inline int64_t pyOr(int16_t l, int16_t r) { return l | r; }
inline int64_t pyOr(int8_t l, int8_t r) { return l | r; }
inline int64_t pyOr(uint64_t l, uint64_t r) { return l | r; }
inline int64_t pyOr(uint32_t l, uint32_t r) { return l | r; }
inline int64_t pyOr(uint16_t l, uint16_t r) { return l | r; }
inline int64_t pyOr(uint8_t l, uint8_t r) { return l | r; }
inline int64_t pyOr(double l, double r) {
    PyErr_Format(PyExc_TypeError, "'|' not supported for floating-point types");
    throw PythonExceptionSet();
}
inline int64_t pyOr(float l, float r) {
    PyErr_Format(PyExc_TypeError, "'|' not supported for floating-point types");
    throw PythonExceptionSet();
}

inline int64_t pyXor(int8_t l, int8_t r) { return l ^ r; }
inline int64_t pyXor(uint8_t l, uint8_t r) { return l ^ r; }
inline int64_t pyXor(int16_t l, int16_t r) { return l ^ r; }
inline int64_t pyXor(uint16_t l, uint16_t r) { return l ^ r; }
inline int64_t pyXor(int32_t l, int32_t r) { return l ^ r; }
inline int64_t pyXor(uint32_t l, uint32_t r) { return l ^ r; }
inline int64_t pyXor(int64_t l, int64_t r) { return l ^ r; }
inline int64_t pyXor(uint64_t l, uint64_t r) { return l ^ r; }
inline int64_t pyXor(double l, double r) {
    PyErr_Format(PyExc_TypeError, "'^' not supported for floating-point types");
    throw PythonExceptionSet();
}
inline int64_t pyXor(float l, float r) {
    PyErr_Format(PyExc_TypeError, "'^' not supported for floating-point types");
    throw PythonExceptionSet();
}

inline int64_t pyLshift(int64_t l, int64_t r) {
    if (r < 0) {
        PyErr_Format(PyExc_ValueError, "negative shift count");
        throw PythonExceptionSet();
    }
    if ((l == 0 && r > SSIZE_MAX) || (l != 0 && r >= 1024)) { // 1024 is arbitrary
        PyErr_Format(PyExc_ValueError, "shift count too large");
        throw PythonExceptionSet();
    }
    return (l >= 0) ? l << r : -((-l) << r);
}
inline uint64_t pyLshift(uint64_t l, uint64_t r) {
    if ((l == 0 && r > SSIZE_MAX) || (l != 0 && r >= 1024)) { // 1024 is arbitrary
        PyErr_Format(PyExc_ValueError, "shift count too large");
        throw PythonExceptionSet();
    }
    return l << r;
}
inline int64_t pyLshift(int32_t l, int32_t r) { return pyLshift((int64_t)l, (int64_t)r); }
inline int64_t pyLshift(int16_t l, int16_t r) { return pyLshift((int64_t)l, (int64_t)r); }
inline int64_t pyLshift(int8_t l, int8_t r) { return pyLshift((int64_t)l, (int64_t)r); }
inline uint64_t pyLshift(uint32_t l, uint32_t r) { return pyLshift((uint64_t)l, (uint64_t)r); }
inline uint64_t pyLshift(uint16_t l, uint16_t r) { return pyLshift((uint64_t)l, (uint64_t)r); }
inline uint64_t pyLshift(uint8_t l, uint8_t r) { return pyLshift((uint64_t)l, (uint64_t)r); }
inline int64_t pyLshift(float l, float r) {
    PyErr_Format(PyExc_TypeError, "'<<' not supported for floating-point types");
    throw PythonExceptionSet();
}
inline int64_t pyLshift(double l, double r) {
    PyErr_Format(PyExc_TypeError, "'<<' not supported for floating-point types");
    throw PythonExceptionSet();
}

inline uint64_t pyRshift(uint64_t l, uint64_t r) {
    if (r > SSIZE_MAX) {
        PyErr_Format(PyExc_ValueError, "shift count too large");
        throw PythonExceptionSet();
    }
    if (r == 0)
        return l;
    if (r >= 64)
        return 0;
    return l >> r;
}
inline int64_t pyRshift(int64_t l, int64_t r) {
    if (r < 0) {
        PyErr_Format(PyExc_ValueError, "negative shift count");
        throw PythonExceptionSet();
    }
    if (r > SSIZE_MAX) {
        PyErr_Format(PyExc_ValueError, "shift count too large");
        throw PythonExceptionSet();
    }
    if (r == 0)
        return l;
    if (l >= 0)
        return l >> r;
    int64_t ret = (-l) >> r;
    if (ret == 0)
        return -1;
    if (l == -l)  // int64_min case
        return ret;
    return -ret;
}
inline int64_t pyRshift(int32_t l, int32_t r) { return pyRshift((int64_t)l, (int64_t)r); }
inline int64_t pyRshift(int16_t l, int16_t r) { return pyRshift((int64_t)l, (int64_t)r); }
inline int64_t pyRshift(int8_t l, int8_t r) { return pyRshift((int64_t)l, (int64_t)r); }
inline int64_t pyRshift(uint32_t l, uint32_t r) { return pyRshift((uint64_t)l, (uint64_t)r); }
inline int64_t pyRshift(uint16_t l, uint16_t r) { return pyRshift((uint64_t)l, (uint64_t)r); }
inline int64_t pyRshift(uint8_t l, uint8_t r) { return pyRshift((uint64_t)l, (uint64_t)r); }
inline int64_t pyRshift(double l, double r) {
    PyErr_Format(PyExc_TypeError, "'>>' not supported for floating-point types");
    throw PythonExceptionSet();
}
inline int64_t pyRshift(float l, float r) {
    PyErr_Format(PyExc_TypeError, "'>>' not supported for floating-point types");
    throw PythonExceptionSet();
}

//inline float pyFloatDiv(bool l, bool r)          { return ((float)l) / (float)r; }
inline double pyFloatDiv(uint64_t l, uint64_t r) { return ((double)l) / (double)r; }
inline float pyFloatDiv(uint32_t l, uint32_t r)  { return ((float)l) / (float)r; }
inline float pyFloatDiv(uint16_t l, uint16_t r)  { return ((float)l) / (float)r; }
inline float pyFloatDiv(uint8_t l, uint8_t r)    { return ((float)l) / (float)r; }
inline double pyFloatDiv(int64_t l, int64_t r)   { return ((double)l) / (double)r; }
inline float pyFloatDiv(int32_t l, int32_t r)    { return ((float)l) / (float)r; }
inline float pyFloatDiv(int16_t l, int16_t r)    { return ((float)l) / (float)r; }
inline float pyFloatDiv(int8_t l, int8_t r)      { return ((float)l) / (float)r; }
inline double pyFloatDiv(double l, double r)     { return l / r; }
inline float pyFloatDiv(float l, float r)        { return l / r; }

inline int64_t pyFloorDiv(int64_t l, int64_t r)   {
    if (r == 0) {
        PyErr_Format(PyExc_ZeroDivisionError, "integer division or modulo by zero");
        throw PythonExceptionSet();
    }
    if (l < 0 && l == -l && r == -1) {
        // overflow because int64_min / -1 > int64_max
        return l;
    }

    if ((l>0 && r>0) || (l<0 && r<0)) { //same signs
        return l / r;
    }
    // opposite signs
    return (l % r) ? l / r - 1 : l / r;
}
inline int32_t pyFloorDiv(int32_t l, int32_t r)    { return pyFloorDiv((int64_t)l, (int64_t)r); }
inline int16_t pyFloorDiv(int16_t l, int16_t r)    { return pyFloorDiv((int64_t)l, (int64_t)r); }
inline int8_t pyFloorDiv(int8_t l, int8_t r)       { return pyFloorDiv((int64_t)l, (int64_t)r); }
inline uint64_t pyFloorDiv(uint64_t l, uint64_t r) { return l / r; }
inline uint32_t pyFloorDiv(uint32_t l, uint32_t r) { return l / r; }
inline uint16_t pyFloorDiv(uint16_t l, uint16_t r) { return l / r; }
inline uint8_t pyFloorDiv(uint8_t l, uint8_t r)    { return l / r; }
inline bool pyFloorDiv(bool l, bool r)             { return l / r; }
inline double pyFloorDiv(double l, double r) {
    if (r == 0.0) {
        PyErr_Format(PyExc_ZeroDivisionError, "float divmod()");
        throw PythonExceptionSet();
    }
    double result = (l - pyMod(l, r))/r;
    double floorresult = std::floor(result);
    if (result - floorresult > 0.5)
        floorresult += 1.0;
    return floorresult;
}
inline float pyFloorDiv(float l, float r) { return pyFloorDiv((double)l, (double)r); }

// template for signed pow
template <class T>
inline double pyPow(T l, T r) {
    if (l == 0 && r < 0) {
        PyErr_Format(PyExc_TypeError, "0.0 cannot be raised to a negative power");
        throw PythonExceptionSet();
    }
    double result = std::pow(l, r);
    if (std::is_integral<T>::value && l < 0 && r > 0 && pyMod(r, (T)2) && result > 0)
        return -result;
    return result;
}
// specific for unsigned pow
//inline double pyPow(bool l, bool r) { return 1.0; }
inline double pyPow(uint8_t l, uint8_t r) { return std::pow(l,r); }
inline double pyPow(uint16_t l, uint16_t r) { return std::pow(l,r); }
inline double pyPow(uint32_t l, uint32_t r) { return std::pow(l,r); }
inline double pyPow(uint64_t l, uint64_t r) { return std::pow(l,r); }

inline double pyRound(double l, int64_t n)     { return nativepython_runtime_round_float64(l, n); }
inline float pyRound(float l, int64_t n)       { return nativepython_runtime_round_float64((double)l, n); }
inline uint64_t pyRound(uint64_t l, int64_t n) { if (n==0) return l; else return (uint64_t)pyRound((double)l, n); }
inline uint32_t pyRound(uint32_t l, int64_t n) { if (n==0) return l; else return (uint64_t)pyRound((double)l, n); }
inline uint16_t pyRound(uint16_t l, int64_t n) { if (n==0) return l; else return (uint64_t)pyRound((double)l, n); }
inline uint8_t pyRound(uint8_t l, int64_t n)   { if (n==0) return l; else return (uint64_t)pyRound((double)l, n); }
inline int64_t pyRound(int64_t l, int64_t n)   { if (n==0) return l; else return (uint64_t)pyRound((double)l, n); }
inline int32_t pyRound(int32_t l, int64_t n)   { if (n==0) return l; else return (uint64_t)pyRound((double)l, n); }
inline int16_t pyRound(int16_t l, int64_t n)   { if (n==0) return l; else return (uint64_t)pyRound((double)l, n); }
inline int8_t pyRound(int8_t l, int64_t n)     { if (n==0) return l; else return (uint64_t)pyRound((double)l, n); }
inline bool pyRound(bool l, int64_t n)     { return l && n >=0; }

template<class T>
static PyObject* pyOperatorConcreteForRegisterPromoted(T self, T other, const char* op, const char* opErr) {
    if (strcmp(op, "__add__") == 0) {
        return registerValueToPyValue(T(self+other));
    }

    if (strcmp(op, "__sub__") == 0) {
        return registerValueToPyValue(T(self-other));
    }

    if (strcmp(op, "__mul__") == 0) {
        return registerValueToPyValue(T(self*other));
    }

    if (strcmp(op, "__and__") == 0) {
        return registerValueToPyValue(T(pyAnd(self,other)));
    }

    if (strcmp(op, "__or__") == 0) {
        return registerValueToPyValue(T(pyOr(self,other)));
    }

    if (strcmp(op, "__xor__") == 0) {
        return registerValueToPyValue(T(pyXor(self,other)));
    }

    if (strcmp(op, "__lshift__") == 0) {
        return registerValueToPyValue(T(pyLshift(self,other)));
    }

    if (strcmp(op, "__rshift__") == 0) {
        return registerValueToPyValue(T(pyRshift(self,other)));
    }

    if (strcmp(op, "__pow__") == 0) {
        return registerValueToPyValue(pyPow(self,other));
    }

    if (strcmp(op, "__truediv__") == 0) {
        if (other == 0) {
            PyErr_Format(PyExc_ZeroDivisionError,
                std::is_floating_point<T>::value ? "float division by zero" : "division by zero");
            throw PythonExceptionSet();
        }
        return registerValueToPyValue(pyFloatDiv(self, other));
    }

    if (strcmp(op, "__floordiv__") == 0) {
        if (other == 0) {
            PyErr_Format(PyExc_ZeroDivisionError,
                std::is_floating_point<T>::value ? "float divmod()" : "integer division or modulo by zero");
            throw PythonExceptionSet();
        }
        return registerValueToPyValue(T(pyFloorDiv(self,other)));
    }

    if (strcmp(op, "__mod__") == 0) {
        if (other == 0) {
            PyErr_Format(PyExc_ZeroDivisionError,
                std::is_floating_point<T>::value ? "float modulo" : "integer division or modulo by zero");
            throw PythonExceptionSet();
        }

        return registerValueToPyValue(T(pyMod(self, other)));
    }

    return incref(Py_NotImplemented);
}

template<class T, class T2>
static PyObject* pyOperatorConcreteForRegister(T self, T2 other, const char* op, const char* opErr) {
    typedef typename PromotesTo<T, T2>::result_type target_type;
    if (strcmp(op, "__truediv__") == 0) {
        typedef typename PromotesTo<target_type, float>::result_type div_target_type;
        return pyOperatorConcreteForRegisterPromoted(div_target_type(self), div_target_type(other), op, opErr);
    }
    else if (strcmp(op, "__pow__") == 0) {
        typedef typename PromotesTo<target_type, uint64_t>::result_type pow_target_type;
        return pyOperatorConcreteForRegisterPromoted(pow_target_type(self), pow_target_type(other), op, opErr);
    }

    return pyOperatorConcreteForRegisterPromoted(target_type(self), target_type(other), op, opErr);
}

template<class T>
class PyRegisterTypeInstance : public PyInstance {
public:
    typedef RegisterType<T> modeled_type;

    inline T get() { return *(T*)dataPtr(); }

    static bool isNumpyFloatType(PyTypeObject* t) {
        return (
            t == &PyHalfArrType_Type
            || t == &PyFloatArrType_Type
            || t == &PyDoubleArrType_Type
            || t == &PyLongDoubleArrType_Type
        );
    }

    static bool isNumpyIntType(PyTypeObject* t) {
        return (
            t == &PyByteArrType_Type
            || t == &PyShortArrType_Type
            || t == &PyIntArrType_Type
            || t == &PyLongArrType_Type
            || t == &PyLongLongArrType_Type
            || t == &PyUByteArrType_Type
            || t == &PyUShortArrType_Type
            || t == &PyUIntArrType_Type
            || t == &PyULongArrType_Type
            || t == &PyULongLongArrType_Type
        );
    }

    static bool isNumpyScalarType(PyTypeObject* t) {
        return t == &PyBoolArrType_Type || isNumpyFloatType(t) || isNumpyIntType(t);
    }

    static Type::TypeCategory numpyScalarTypeToBestCategory(PyTypeObject* t) {
        if (t == &PyBoolArrType_Type) { return Type::TypeCategory::catBool; }
        if (t == &PyHalfArrType_Type) { return Type::TypeCategory::catFloat32; }
        if (t == &PyFloatArrType_Type) { return Type::TypeCategory::catFloat32; }
        if (t == &PyDoubleArrType_Type) { return Type::TypeCategory::catFloat64; }
        if (t == &PyLongDoubleArrType_Type) { return Type::TypeCategory::catFloat64; }
        if (t == &PyByteArrType_Type) { return Type::TypeCategory::catInt8; }
        if (t == &PyShortArrType_Type) { return Type::TypeCategory::catInt16; }
        if (t == &PyIntArrType_Type) { return Type::TypeCategory::catInt32; }
        if (t == &PyLongArrType_Type) {
            return sizeof(long) == 8 ? Type::TypeCategory::catInt64 : Type::TypeCategory::catInt32;
        }
        if (t == &PyLongLongArrType_Type) { return Type::TypeCategory::catInt64; }
        if (t == &PyUByteArrType_Type) { return Type::TypeCategory::catUInt8; }
        if (t == &PyUShortArrType_Type) { return Type::TypeCategory::catUInt16; }
        if (t == &PyUIntArrType_Type) { return Type::TypeCategory::catUInt32; }
        if (t == &PyULongArrType_Type) {
            return sizeof(long) == 8 ? Type::TypeCategory::catUInt64 : Type::TypeCategory::catUInt32;
        }
        if (t == &PyULongLongArrType_Type) { return Type::TypeCategory::catUInt64; }

        throw std::runtime_error("Type is not a numpy type.");
    }

    static void copyConstructFromPythonInstanceConcrete(RegisterType<T>* targetType, instance_ptr tgt, PyObject* pyRepresentation, ConversionLevel level) {
        Type::TypeCategory targetCat = targetType->getTypeCategory();

        std::pair<Type*, instance_ptr> typeAndPtr = extractTypeAndPtrFrom(pyRepresentation);
        Type* other = typeAndPtr.first;
        instance_ptr otherDataPtr = typeAndPtr.second;

        if (other) {
            Type::TypeCategory otherCat = other->getTypeCategory();

            if (RegisterTypeProperties::isValidConversion(otherCat, targetCat, level)) {
                if (otherCat == Type::TypeCategory::catUInt64) {
                    ((T*)tgt)[0] = *(uint64_t*)otherDataPtr;
                    return;
                }
                if (otherCat == Type::TypeCategory::catUInt32) {
                    ((T*)tgt)[0] = *(uint32_t*)otherDataPtr;
                    return;
                }
                if (otherCat == Type::TypeCategory::catUInt16) {
                    ((T*)tgt)[0] = *(uint16_t*)otherDataPtr;
                    return;
                }
                if (otherCat == Type::TypeCategory::catUInt8) {
                    ((T*)tgt)[0] = *(uint8_t*)otherDataPtr;
                    return;
                }
                if (otherCat == Type::TypeCategory::catInt64) {
                    ((T*)tgt)[0] = *(int64_t*)otherDataPtr;
                    return;
                }
                if (otherCat == Type::TypeCategory::catInt32) {
                    ((T*)tgt)[0] = *(int32_t*)otherDataPtr;
                    return;
                }
                if (otherCat == Type::TypeCategory::catInt16) {
                    ((T*)tgt)[0] = *(int16_t*)otherDataPtr;
                    return;
                }
                if (otherCat == Type::TypeCategory::catInt8) {
                    ((T*)tgt)[0] = *(int8_t*)otherDataPtr;
                    return;
                }
                if (otherCat == Type::TypeCategory::catBool) {
                    ((T*)tgt)[0] = *(bool*)otherDataPtr;
                    return;
                }
                if (otherCat == Type::TypeCategory::catFloat64) {
                    ((T*)tgt)[0] = *(double*)otherDataPtr;
                    return;
                }
                if (otherCat == Type::TypeCategory::catFloat32) {
                    ((T*)tgt)[0] = *(float*)otherDataPtr;
                    return;
                }
            }
        }

        if (PyBool_Check(pyRepresentation) || pyRepresentation->ob_type == &PyBoolArrType_Type) {
            if (RegisterTypeProperties::isValidConversion(Type::TypeCategory::catBool, targetCat, level)) {
                ((T*)tgt)[0] = PyObject_IsTrue(pyRepresentation);
                return;
            }
        }

        if (PyFloat_Check(pyRepresentation) || isNumpyFloatType(pyRepresentation->ob_type)) {
            Type::TypeCategory cat = Type::TypeCategory::catFloat64;

            if (isNumpyFloatType(pyRepresentation->ob_type)) {
                cat = numpyScalarTypeToBestCategory(pyRepresentation->ob_type);
            }

            if (RegisterTypeProperties::isValidConversion(cat, targetCat, level)) {
                ((T*)tgt)[0] = PyFloat_AsDouble(pyRepresentation);
                return;
            }
        }

        if ((PyLong_Check(pyRepresentation) && !PyBool_Check(pyRepresentation))
                || isNumpyIntType(pyRepresentation->ob_type)) {
            Type::TypeCategory cat = Type::TypeCategory::catInt64;

            if (isNumpyIntType(pyRepresentation->ob_type)) {
                cat = numpyScalarTypeToBestCategory(pyRepresentation->ob_type);
            }

            if (RegisterTypeProperties::isValidConversion(cat, targetCat, level)) {
                if (targetCat == Type::TypeCategory::catFloat64 || targetCat == Type::TypeCategory::catFloat32) {
                    ((T*)tgt)[0] = PyFloat_AsDouble(pyRepresentation);
                    return;
                } else {
                    int64_t l = PyLong_AsLongLong(pyRepresentation);
                    if (l == -1 && PyErr_Occurred()) {
                        PyErr_Clear();
                        // we always want to be able to cast, even if we throw information away.
                        uint64_t u = PyLong_AsUnsignedLongLongMask(pyRepresentation);

                        if (!(u == (uint64_t)(-1) && PyErr_Occurred())) {
                            ((T*)tgt)[0] = u;
                            return;
                        } else {
                            PyErr_Clear();
                        }
                    } else {
                        ((T*)tgt)[0] = l;
                        return;
                    }
                }
            }
        }

        if (level >= ConversionLevel::New) {
            //if this is an explicit cast, use python's internal type conversion
            //mechanisms, which will call methods like __bool__, __int__, or __float__ on
            //objects that have conversions defined.
            if (targetCat == Type::TypeCategory::catBool) {
                int result = PyObject_IsTrue(pyRepresentation);
                if (result == -1) {
                    // use the error message created by python
                    throw PythonExceptionSet();
                } else {
                    ((T*)tgt)[0] = (result == 1);
                    return;
                }
            }

            if (RegisterTypeProperties::isInteger(targetCat)) {
                PyObjectStealer asLong(PyNumber_Long(pyRepresentation));
                if (!asLong) {
                    // use the error message created by python
                    throw PythonExceptionSet();
                }

                int64_t l = PyLong_AsLongLong(asLong);
                if (l == -1 && PyErr_Occurred()) {
                    PyErr_Clear();
                    // we always want to be able to cast, even if we throw information away.
                    uint64_t u = PyLong_AsUnsignedLongLongMask(asLong);
                    if (!(u == (uint64_t)(-1) && PyErr_Occurred())) {
                        ((T*)tgt)[0] = u;
                        return;
                    } else {
                        // use the error message creaed by python
                        throw PythonExceptionSet();
                    }
                } else {
                    ((T*)tgt)[0] = l;
                    return;
                }
            }

            if (RegisterTypeProperties::isFloat(targetCat)) {
                PyObjectStealer asFloat(PyNumber_Float(pyRepresentation));
                if (!asFloat) {
                    // use the error message created by python
                    throw PythonExceptionSet();
                }

                double d = PyFloat_AsDouble(asFloat);

                if (d == -1.0 && PyErr_Occurred()) {
                    // use the error message created by python
                    throw PythonExceptionSet();
                } else {
                    ((T*)tgt)[0] = d;
                    return;
                }
            }
        }

        PyInstance::copyConstructFromPythonInstanceConcrete(targetType, tgt, pyRepresentation, level);
    }

    static bool pyValCouldBeOfTypeConcrete(modeled_type* t, PyObject* pyRepresentation, ConversionLevel level) {
        if (PyFloat_Check(pyRepresentation)) {
            return RegisterTypeProperties::isValidConversion(Type::TypeCategory::catFloat64, t->getTypeCategory(), level);
        }

        if ((PyLong_Check(pyRepresentation) && !PyBool_Check(pyRepresentation))) {
            return RegisterTypeProperties::isValidConversion(Type::TypeCategory::catInt64, t->getTypeCategory(), level);
        }

        if (PyBool_Check(pyRepresentation)) {
            return RegisterTypeProperties::isValidConversion(Type::TypeCategory::catBool, t->getTypeCategory(), level);
        }

        if (isNumpyScalarType(pyRepresentation->ob_type)) {
            return RegisterTypeProperties::isValidConversion(
                numpyScalarTypeToBestCategory(pyRepresentation->ob_type),
                t->getTypeCategory(),
                level
            );
        }

        if (Type* otherT = extractTypeFrom(pyRepresentation->ob_type)) {
            if (RegisterTypeProperties::isValidConversion(otherT->getTypeCategory(), t->getTypeCategory(), level)) {
                return true;
            }
        }

        if (level < ConversionLevel::New) {
            return false;
        }

        if (PyUnicode_Check(pyRepresentation) || PyBytes_Check(pyRepresentation)) {
            return true;
        }

        if (RegisterTypeProperties::isFloat(t->getTypeCategory()))  {
            if (!pyRepresentation->ob_type->tp_as_number) {
                return false;
            }

            return pyRepresentation->ob_type->tp_as_number->nb_float != nullptr;
        }

        if (RegisterTypeProperties::isInteger(t->getTypeCategory()))  {
            if (!pyRepresentation->ob_type->tp_as_number) {
                return false;
            }

            return pyRepresentation->ob_type->tp_as_number->nb_int != nullptr;
        }

        if (t->getTypeCategory() == Type::TypeCategory::catBool) {
            return true;
        }

        return false;
    }

    static PyObject* extractPythonObjectConcrete(RegisterType<T>* t, instance_ptr data) {
        if (t->getTypeCategory() == Type::TypeCategory::catInt64) {
            return PyLong_FromLongLong(*(int64_t*)data);
        }
        if (t->getTypeCategory() == Type::TypeCategory::catFloat64) {
            return PyFloat_FromDouble(*(double*)data);
        }
        if (t->getTypeCategory() == Type::TypeCategory::catBool) {
            return incref(*(bool*)data ? Py_True : Py_False);
        }
        return NULL;
    }

    int pyInquiryConcrete(const char* op, const char* opErrRep) {
        // op == '__bool__'
        return (get() != 0);
    }

    PyObject* pyUnaryOperatorConcrete(const char* op, const char* opErr) {
        if (strcmp(op, "__float__") == 0) {
            return PyFloat_FromDouble(get());
        }
        if (strcmp(op, "__int__") == 0) {
            Type::TypeCategory cat = type()->getTypeCategory();
            if (cat == Type::TypeCategory::catUInt64) {
                return PyLong_FromUnsignedLong(get());
            }
            return PyLong_FromLong(get());
        }
        if (strcmp(op, "__neg__") == 0) {
            T val = get();
            val = -val;
            return extractPythonObject((instance_ptr)&val, type());
        }
        if (strcmp(op, "__pos__") == 0) {
            T val = get();
            return extractPythonObject((instance_ptr)&val, type());
        }
        if (strcmp(op, "__invert__") == 0 && RegisterTypeProperties::isInteger(type()->getTypeCategory())) {
            T val = get();
            val = bitInvert(val);
            return extractPythonObject((instance_ptr)&val, type());
        }
        if (strcmp(op, "__index__") == 0 && RegisterTypeProperties::isInteger(type()->getTypeCategory())) {
            int64_t val = get();
            return PyLong_FromLong(val);
        }
        if (strcmp(op, "__abs__") == 0) {
            T val = get();
            if (val < 0)
                val = -val;
            return extractPythonObject((instance_ptr)&val, type());
        }

        return PyInstance::pyUnaryOperatorConcrete(op, opErr);
    }

    PyObject* pyTernaryOperatorConcrete(PyObject* rhs, PyObject* third, const char* op, const char* opErr) {
        // support binary operator __pow__ but not ternary
        if (strcmp(op, "__pow__") == 0 && third == Py_None)
            return pyOperatorConcrete(rhs, op, opErr);
        return PyInstance::pyOperatorConcrete(rhs, op, opErr);
    }

    PyObject* pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr) {
        if (PyLong_CheckExact(rhs)) {
            int64_t l = PyLong_AsLongLong(rhs);
            if (l == -1 && PyErr_Occurred()) {
                PyErr_Clear();
                uint64_t u = PyLong_AsUnsignedLongLongMask(rhs);
                if (u == (uint64_t)(-1) && PyErr_Occurred())
                    throw PythonExceptionSet();
                return pyOperatorConcreteForRegister<T, uint64_t>(get(), u, op, opErr);
            }
            return pyOperatorConcreteForRegister<T, int64_t>(get(), l, op, opErr);
        }
        if (PyBool_Check(rhs)) {
            return pyOperatorConcreteForRegister<T, bool>(get(), rhs == Py_True ? true : false, op, opErr);
        }
        if (PyFloat_CheckExact(rhs)) {
            return pyOperatorConcreteForRegister<T, double>(get(), PyFloat_AsDouble(rhs), op, opErr);
        }

        Type* rhsType = extractTypeFrom(rhs->ob_type);

        if (rhsType) {
            if (rhsType->getTypeCategory() == Type::TypeCategory::catBool) { return pyOperatorConcreteForRegister<T, bool>(get(), *(bool*)((PyInstance*)rhs)->dataPtr(), op, opErr); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catInt8) { return pyOperatorConcreteForRegister<T, int8_t>(get(), *(int8_t*)((PyInstance*)rhs)->dataPtr(), op, opErr); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catInt16) { return pyOperatorConcreteForRegister<T, int16_t>(get(), *(int16_t*)((PyInstance*)rhs)->dataPtr(), op, opErr); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catInt32) { return pyOperatorConcreteForRegister<T, int32_t>(get(), *(int32_t*)((PyInstance*)rhs)->dataPtr(), op, opErr); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catInt64) { return pyOperatorConcreteForRegister<T, int64_t>(get(), *(int64_t*)((PyInstance*)rhs)->dataPtr(), op, opErr); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catUInt8) { return pyOperatorConcreteForRegister<T, uint8_t>(get(), *(uint8_t*)((PyInstance*)rhs)->dataPtr(), op, opErr); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catUInt16) { return pyOperatorConcreteForRegister<T, uint16_t>(get(), *(uint16_t*)((PyInstance*)rhs)->dataPtr(), op, opErr); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catUInt32) { return pyOperatorConcreteForRegister<T, uint32_t>(get(), *(uint32_t*)((PyInstance*)rhs)->dataPtr(), op, opErr); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catUInt64) { return pyOperatorConcreteForRegister<T, uint64_t>(get(), *(uint64_t*)((PyInstance*)rhs)->dataPtr(), op, opErr); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catFloat32) { return pyOperatorConcreteForRegister<T, float>(get(), *(float*)((PyInstance*)rhs)->dataPtr(), op, opErr); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catFloat64) { return pyOperatorConcreteForRegister<T, double>(get(), *(double*)((PyInstance*)rhs)->dataPtr(), op, opErr); }
        }

        return PyInstance::pyOperatorConcrete(rhs, op, opErr);
    }

    PyObject* pyOperatorConcreteReverse(PyObject* rhs, const char* op, const char* opErr) {
        if (PyLong_CheckExact(rhs)) {
            int64_t l = PyLong_AsLongLong(rhs);
            if (l == -1 && PyErr_Occurred()) {
                PyErr_Clear();
                uint64_t u = PyLong_AsUnsignedLongLongMask(rhs);
                if (u == (uint64_t)(-1) && PyErr_Occurred())
                    throw PythonExceptionSet();
                return pyOperatorConcreteForRegister<uint64_t, T>(u, get(), op, opErr);
            }
            return pyOperatorConcreteForRegister<int64_t, T>(l, get(), op, opErr);
        }
        if (PyBool_Check(rhs)) {
            return pyOperatorConcreteForRegister<bool, T>(rhs == Py_True ? true : false, get(), op, opErr);
        }
        if (PyFloat_CheckExact(rhs)) {
            return pyOperatorConcreteForRegister<double, T>(PyFloat_AsDouble(rhs), get(), op, opErr);
        }

        Type* rhsType = extractTypeFrom(rhs->ob_type);

        if (rhsType) {
            if (rhsType->getTypeCategory() == Type::TypeCategory::catBool) { return pyOperatorConcreteForRegister<bool, T>(*(bool*)((PyInstance*)rhs)->dataPtr(), get(), op, opErr); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catInt8) { return pyOperatorConcreteForRegister<int8_t, T>(*(int8_t*)((PyInstance*)rhs)->dataPtr(), get(), op, opErr); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catInt16) { return pyOperatorConcreteForRegister<int16_t, T>(*(int16_t*)((PyInstance*)rhs)->dataPtr(), get(), op, opErr); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catInt32) { return pyOperatorConcreteForRegister<int32_t, T>(*(int32_t*)((PyInstance*)rhs)->dataPtr(), get(), op, opErr); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catInt64) { return pyOperatorConcreteForRegister<int64_t, T>(*(int64_t*)((PyInstance*)rhs)->dataPtr(), get(), op, opErr); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catUInt8) { return pyOperatorConcreteForRegister<uint8_t, T>(*(uint8_t*)((PyInstance*)rhs)->dataPtr(), get(), op, opErr); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catUInt16) { return pyOperatorConcreteForRegister<uint16_t, T>(*(uint16_t*)((PyInstance*)rhs)->dataPtr(), get(), op, opErr); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catUInt32) { return pyOperatorConcreteForRegister<uint32_t, T>(*(uint32_t*)((PyInstance*)rhs)->dataPtr(), get(), op, opErr); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catUInt64) { return pyOperatorConcreteForRegister<uint64_t, T>(*(uint64_t*)((PyInstance*)rhs)->dataPtr(), get(), op, opErr); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catFloat32) { return pyOperatorConcreteForRegister<float, T>(*(float*)((PyInstance*)rhs)->dataPtr(), get(), op, opErr); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catFloat64) { return pyOperatorConcreteForRegister<double, T>(*(double*)((PyInstance*)rhs)->dataPtr(), get(), op, opErr); }
        }

        return PyInstance::pyOperatorConcrete(rhs, op, opErr);
    }

    //compare two register types directly given the python
    //comparison operator 'pyComparisonOp'. We follow numpy's comparisons here,
    //where we use a signed compare anytime _either_ value is signed, which
    //is different than c++.
    template<class other_t>
    static bool pyCompare(T lhs, other_t rhs, int pyComparisonOp) {
        typedef typename PromotesTo<T, other_t>::result_type PT;

        if (pyComparisonOp == Py_EQ) { return ((PT)lhs) == ((PT)rhs); }
        if (pyComparisonOp == Py_NE) { return ((PT)lhs) != ((PT)rhs); }
        if (pyComparisonOp == Py_LT) { return ((PT)lhs) < ((PT)rhs); }
        if (pyComparisonOp == Py_GT) { return ((PT)lhs) > ((PT)rhs); }
        if (pyComparisonOp == Py_LE) { return ((PT)lhs) <= ((PT)rhs); }
        if (pyComparisonOp == Py_GE) { return ((PT)lhs) >= ((PT)rhs); }
        return false;
    }

    static bool compare_to_python_concrete(Type* t, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp) {
        //we need to ensure that we don't compare ourselves to any other
        if (exact) {
            if (pyComparisonOp == Py_NE) {
                return !compare_to_python_concrete(t, self, other, exact, Py_EQ);
            }

            if (pyComparisonOp != Py_EQ) {
                throw std::runtime_error("Exact must be used with Eq or Neq only");
            }

            if (t->getTypeCategory() == Type::TypeCategory::catBool) {
                if (!PyBool_Check(other)) {
                    return false;
                }
            } else
            if (t->getTypeCategory() == Type::TypeCategory::catInt64) {
                if (!PyLong_Check(other)) {
                    return false;
                }
            } else
            if (t->getTypeCategory() == Type::TypeCategory::catFloat64) {
                if (!PyFloat_Check(other)) {
                    return false;
                }
            } else {
                Type* rhsType = extractTypeFrom(other->ob_type);
                if (rhsType != t) {
                    return false;
                }
            }
        }

        if (PyBool_Check(other)) {
            return pyCompare(*(T*)self, other == Py_True ? true : false, pyComparisonOp);
        }

        if (PyLong_Check(other)) {
            int64_t l = PyLong_AsLongLong(other);
            if (l == -1 && PyErr_Occurred()) {
                PyErr_Clear();
                uint64_t u = PyLong_AsUnsignedLongLongMask(other);
                if (u == (uint64_t)(-1) && PyErr_Occurred()) {
                    throw PythonExceptionSet();
                }
                return pyCompare(*(T*)self, u, pyComparisonOp);
            }
            return pyCompare(*(T*)self, l, pyComparisonOp);
        }

        if (PyFloat_Check(other)) {
            return pyCompare(*(T*)self, PyFloat_AsDouble(other), pyComparisonOp);
        }

        Type* rhsType = extractTypeFrom(other->ob_type);

        if (rhsType) {
            if (rhsType->getTypeCategory() == Type::TypeCategory::catBool) { return pyCompare(*(T*)self, *(bool*)((PyInstance*)other)->dataPtr(), pyComparisonOp); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catInt8) { return pyCompare(*(T*)self, *(int8_t*)((PyInstance*)other)->dataPtr(), pyComparisonOp); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catInt16) { return pyCompare(*(T*)self, *(int16_t*)((PyInstance*)other)->dataPtr(), pyComparisonOp); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catInt32) { return pyCompare(*(T*)self, *(int32_t*)((PyInstance*)other)->dataPtr(), pyComparisonOp); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catInt64) { return pyCompare(*(T*)self, *(int64_t*)((PyInstance*)other)->dataPtr(), pyComparisonOp); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catUInt8) { return pyCompare(*(T*)self, *(uint8_t*)((PyInstance*)other)->dataPtr(), pyComparisonOp); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catUInt16) { return pyCompare(*(T*)self, *(uint16_t*)((PyInstance*)other)->dataPtr(), pyComparisonOp); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catUInt32) { return pyCompare(*(T*)self, *(uint32_t*)((PyInstance*)other)->dataPtr(), pyComparisonOp); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catUInt64) { return pyCompare(*(T*)self, *(uint64_t*)((PyInstance*)other)->dataPtr(), pyComparisonOp); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catFloat32) { return pyCompare(*(T*)self, *(float*)((PyInstance*)other)->dataPtr(), pyComparisonOp); }
            if (rhsType->getTypeCategory() == Type::TypeCategory::catFloat64) { return pyCompare(*(T*)self, *(double*)((PyInstance*)other)->dataPtr(), pyComparisonOp); }
        }

        return PyInstance::compare_to_python_concrete(t, self, other, exact, pyComparisonOp);
    }

    static void mirrorTypeInformationIntoPyTypeConcrete(RegisterType<T>* type, PyTypeObject* pyType) {
        //expose 'ElementType' as a member of the type object
        PyDict_SetItemString(pyType->tp_dict, "IsFloat",
            RegisterTypeProperties::isFloat(type->getTypeCategory()) ? Py_True : Py_False
            );
        PyDict_SetItemString(pyType->tp_dict, "IsInteger",
            RegisterTypeProperties::isInteger(type->getTypeCategory()) ? Py_True : Py_False
            );
        PyDict_SetItemString(pyType->tp_dict, "IsSignedInt",
            RegisterTypeProperties::isInteger(type->getTypeCategory())
            && !RegisterTypeProperties::isUnsigned(type->getTypeCategory()) ? Py_True : Py_False
            );
        PyDict_SetItemString(pyType->tp_dict, "IsUnsignedInt",
            RegisterTypeProperties::isUnsigned(type->getTypeCategory()) ? Py_True : Py_False
            );
        PyDict_SetItemString(pyType->tp_dict, "Bits",
            PyLong_FromLong(RegisterTypeProperties::bits(type))
            );
    }

private:
    static PyObject* _complex(PyObject* o, PyObject* args, PyObject* kwargs) {
        PyRegisterTypeInstance* self = (PyRegisterTypeInstance*)o;

        if (PyTuple_Size(args) > 0 || kwargs) {
            PyErr_Format(PyExc_TypeError, "__complex__ invalid number of parameters");
            return NULL;
        }

        T val = *(T*)(self->dataPtr());
        return PyComplex_FromDoubles((double)val,0);
    }

    static PyObject* _round(PyObject* o, PyObject* args, PyObject* kwargs) {
        PyRegisterTypeInstance* self = (PyRegisterTypeInstance*)o;

        if (PyTuple_Size(args) > 1 || kwargs) {
            PyErr_Format(PyExc_TypeError, "__round__ invalid number of parameters");
            return NULL;
        }

        T val = *(T*)(self->dataPtr());
        if (PyTuple_Size(args) == 0) {
            return registerValueToPyValue(T(pyRound(val, 0)));
        }
        PyObject *arg0 = PyTuple_GetItem(args, 0);
        if (!PyLong_Check(arg0)) {
            PyErr_Format(PyExc_TypeError, "__round__ 2nd parameter must be integer");
            return NULL;
        }

        return registerValueToPyValue(T(pyRound(val, PyLong_AsLong(arg0))));
    }

    static PyObject* _trunc(PyObject* o, PyObject* args, PyObject* kwargs) {
        PyRegisterTypeInstance* self = (PyRegisterTypeInstance*)o;

        if (PyTuple_Size(args) > 0 || kwargs) {
            PyErr_Format(PyExc_TypeError, "__trunc__ invalid number of parameters");
            return NULL;
        }

        // Float64 already has it
        if (std::is_same<T, float>::value) {
            T val = *(T*)(self->dataPtr());
            return registerValueToPyValue(T(trunc(val)));
        }
        // For other types, it's the identity function
        return incref(o);
    }

    static PyObject* _floor(PyObject* o, PyObject* args, PyObject* kwargs) {
        PyRegisterTypeInstance* self = (PyRegisterTypeInstance*)o;

        if (PyTuple_Size(args) > 0 || kwargs) {
            PyErr_Format(PyExc_TypeError, "__floor__ invalid number of parameters");
            return NULL;
        }

        // Float64 already has it
        if (std::is_same<T, float>::value) {
            T val = *(T*)(self->dataPtr());
            return registerValueToPyValue(T(floor(val)));
        }
        // For other types, it's the identity function
        return incref(o);
    }

    static PyObject* _ceil(PyObject* o, PyObject* args, PyObject* kwargs) {
        PyRegisterTypeInstance* self = (PyRegisterTypeInstance*)o;

        if (PyTuple_Size(args) > 0 || kwargs) {
            PyErr_Format(PyExc_TypeError, "__ceil__ invalid number of parameters");
            return NULL;
        }

        // Float64 already has it
        if (std::is_same<T, float>::value) {
            T val = *(T*)(self->dataPtr());
            return registerValueToPyValue(T(ceil(val)));
        }
        // For other types, it's the identity function
        return incref(o);
    }

public:
    static PyMethodDef* typeMethodsConcrete(Type* t) {
        return new PyMethodDef [6] {
            {"__complex__", (PyCFunction)_complex, METH_VARARGS | METH_KEYWORDS, _complex_doc},
            {"__round__", (PyCFunction)_round, METH_VARARGS | METH_KEYWORDS, _round_doc},
            {"__trunc__", (PyCFunction)_trunc, METH_VARARGS | METH_KEYWORDS, _trunc_doc},
            {"__floor__", (PyCFunction)_floor, METH_VARARGS | METH_KEYWORDS, _floor_doc},
            {"__ceil__", (PyCFunction)_ceil, METH_VARARGS | METH_KEYWORDS, _ceil_doc},
            {NULL, NULL}
            };
        }
};
