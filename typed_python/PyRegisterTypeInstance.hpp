#pragma once

#include "PyInstance.hpp"
#include "PromotesTo.hpp"
#include <cmath>
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
inline float bitInvert(float in) { return 0.0; }
inline double bitInvert(double in) { return 0.0; }


template<class T>
static PyObject* registerValueToPyValue(T val) {
    static Type* typeObj = GetRegisterType<T>()();

    return PyInstance::extractPythonObject((instance_ptr)&val, typeObj);
}

inline int64_t pyMod(int64_t l, int64_t r) { return l % r; }
inline int32_t pyMod(int32_t l, int32_t r) { return l % r; }
inline int16_t pyMod(int16_t l, int16_t r) { return l % r; }
inline int8_t pyMod(int8_t l, int8_t r) { return l % r; }
inline uint64_t pyMod(uint64_t l, uint64_t r) { return l % r; }
inline uint32_t pyMod(uint32_t l, uint32_t r) { return l % r; }
inline uint16_t pyMod(uint16_t l, uint16_t r) { return l % r; }
inline uint8_t pyMod(uint8_t l, uint8_t r) { return l % r; }
inline bool pyMod(bool l, bool r) { return 0; }
inline float pyMod(float l, float r) { return std::fmod(l,r); }
inline double pyMod(double l, double r) { return std::fmod(l,r); }

inline int64_t pyAnd(int8_t l, int8_t r) { return l & r; }
inline int64_t pyAnd(uint8_t l, uint8_t r) { return l & r; }
inline int64_t pyAnd(int16_t l, int16_t r) { return l & r; }
inline int64_t pyAnd(uint16_t l, uint16_t r) { return l & r; }
inline int64_t pyAnd(int32_t l, int32_t r) { return l & r; }
inline int64_t pyAnd(uint32_t l, uint32_t r) { return l & r; }
inline int64_t pyAnd(int64_t l, int64_t r) { return l & r; }
inline int64_t pyAnd(uint64_t l, uint64_t r) { return l & r; }
inline int64_t pyAnd(float l, float r) {
    PyErr_Format(PyExc_TypeError, "'&' not supported for floating-point types");
    throw PythonExceptionSet();
}
inline int64_t pyAnd(double l, double r) {
    PyErr_Format(PyExc_TypeError, "'&' not supported for floating-point types");
    throw PythonExceptionSet();
}

inline int64_t pyOr(int8_t l, int8_t r) { return l | r; }
inline int64_t pyOr(uint8_t l, uint8_t r) { return l | r; }
inline int64_t pyOr(int16_t l, int16_t r) { return l | r; }
inline int64_t pyOr(uint16_t l, uint16_t r) { return l | r; }
inline int64_t pyOr(int32_t l, int32_t r) { return l | r; }
inline int64_t pyOr(uint32_t l, uint32_t r) { return l | r; }
inline int64_t pyOr(int64_t l, int64_t r) { return l | r; }
inline int64_t pyOr(uint64_t l, uint64_t r) { return l | r; }
inline int64_t pyOr(float l, float r) {
    PyErr_Format(PyExc_TypeError, "'|' not supported for floating-point types");
    throw PythonExceptionSet();
}
inline int64_t pyOr(double l, double r) {
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
inline int64_t pyXor(float l, float r) {
    PyErr_Format(PyExc_TypeError, "'^' not supported for floating-point types");
    throw PythonExceptionSet();
}
inline int64_t pyXor(double l, double r) {
    PyErr_Format(PyExc_TypeError, "'^' not supported for floating-point types");
    throw PythonExceptionSet();
}

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

    if (strcmp(op, "__div__") == 0) {
        if (other == 0) {
            PyErr_Format(PyExc_ZeroDivisionError, "Divide by zero");
            throw PythonExceptionSet();
        }

        return registerValueToPyValue(T(self/other));
    }

    if (strcmp(op, "__floordiv__") == 0) {
        return registerValueToPyValue(T(std::floor(self/other)));
    }

    if (strcmp(op, "__mod__") == 0) {
        if (other == 0) {
            PyErr_Format(PyExc_ZeroDivisionError, "Divide by zero");
            throw PythonExceptionSet();
        }

        //for mod, bool has to go to int
        return registerValueToPyValue(T(pyMod(self, other)));
    }

    return incref(Py_NotImplemented);
}

template<class T, class T2>
static PyObject* pyOperatorConcreteForRegister(T self, T2 other, const char* op, const char* opErr) {
    typedef typename PromotesTo<T, T2>::result_type target_type;

    return pyOperatorConcreteForRegisterPromoted(target_type(self), target_type(other), op, opErr);
}

template<class T>
class PyRegisterTypeInstance : public PyInstance {
public:
    typedef RegisterType<T> modeled_type;

    T get() { return *(T*)dataPtr(); }

    static void copyConstructFromPythonInstanceConcrete(RegisterType<T>* eltType, instance_ptr tgt, PyObject* pyRepresentation, bool isExplicit) {
        Type::TypeCategory cat = eltType->getTypeCategory();

        if (isInteger(cat) || cat == Type::TypeCategory::catBool) {
            if (PyLong_Check(pyRepresentation)) {
                ((T*)tgt)[0] = PyLong_AsLong(pyRepresentation);
                return;
            }
            if (PyFloat_Check(pyRepresentation) && isExplicit) {
                ((T*)tgt)[0] = PyLong_AsLong(pyRepresentation);
                return;
            }
        }

        if (isFloat(cat)) {
            if (PyLong_Check(pyRepresentation)) {
                ((T*)tgt)[0] = PyLong_AsLong(pyRepresentation);
                return;
            }
            if (PyFloat_Check(pyRepresentation)) {
                ((T*)tgt)[0] = PyFloat_AsDouble(pyRepresentation);
                return;
            }
        }

        if (Type* other = extractTypeFrom(pyRepresentation->ob_type)) {
            Type::TypeCategory otherCat = other->getTypeCategory();

            if (otherCat == cat || isExplicit) {
                if (otherCat == Type::TypeCategory::catUInt64) {
                    ((T*)tgt)[0] = ((PyRegisterTypeInstance<uint64_t>*)pyRepresentation)->get();
                    return;
                }
                if (otherCat == Type::TypeCategory::catUInt32) {
                    ((T*)tgt)[0] = ((PyRegisterTypeInstance<uint32_t>*)pyRepresentation)->get();
                    return;
                }
                if (otherCat == Type::TypeCategory::catUInt16) {
                    ((T*)tgt)[0] = ((PyRegisterTypeInstance<uint16_t>*)pyRepresentation)->get();
                    return;
                }
                if (otherCat == Type::TypeCategory::catUInt8) {
                    ((T*)tgt)[0] = ((PyRegisterTypeInstance<uint8_t>*)pyRepresentation)->get();
                    return;
                }
                if (otherCat == Type::TypeCategory::catInt64) {
                    ((T*)tgt)[0] = ((PyRegisterTypeInstance<int64_t>*)pyRepresentation)->get();
                    return;
                }
                if (otherCat == Type::TypeCategory::catInt32) {
                    ((T*)tgt)[0] = ((PyRegisterTypeInstance<int32_t>*)pyRepresentation)->get();
                    return;
                }
                if (otherCat == Type::TypeCategory::catInt16) {
                    ((T*)tgt)[0] = ((PyRegisterTypeInstance<int16_t>*)pyRepresentation)->get();
                    return;
                }
                if (otherCat == Type::TypeCategory::catInt8) {
                    ((T*)tgt)[0] = ((PyRegisterTypeInstance<int8_t>*)pyRepresentation)->get();
                    return;
                }
                if (otherCat == Type::TypeCategory::catBool) {
                    ((T*)tgt)[0] = ((PyRegisterTypeInstance<bool>*)pyRepresentation)->get();
                    return;
                }
                if (otherCat == Type::TypeCategory::catFloat64) {
                    ((T*)tgt)[0] = ((PyRegisterTypeInstance<double>*)pyRepresentation)->get();
                    return;
                }
                if (otherCat == Type::TypeCategory::catFloat32) {
                    ((T*)tgt)[0] = ((PyRegisterTypeInstance<float>*)pyRepresentation)->get();
                    return;
                }
            }
        }

        PyInstance::copyConstructFromPythonInstanceConcrete(eltType, tgt, pyRepresentation, isExplicit);
    }

    static bool isUnsigned(Type::TypeCategory cat) {
        return (cat == Type::TypeCategory::catUInt64 ||
                cat == Type::TypeCategory::catUInt32 ||
                cat == Type::TypeCategory::catUInt16 ||
                cat == Type::TypeCategory::catUInt8 ||
                cat == Type::TypeCategory::catBool
                );
    }
    static bool isInteger(Type::TypeCategory cat) {
        return (cat == Type::TypeCategory::catInt64 ||
                cat == Type::TypeCategory::catInt32 ||
                cat == Type::TypeCategory::catInt16 ||
                cat == Type::TypeCategory::catInt8 ||
                cat == Type::TypeCategory::catUInt64 ||
                cat == Type::TypeCategory::catUInt32 ||
                cat == Type::TypeCategory::catUInt16 ||
                cat == Type::TypeCategory::catUInt8
                );
    }

    static bool isFloat(Type::TypeCategory cat) {
        return (cat == Type::TypeCategory::catFloat64 ||
                cat == Type::TypeCategory::catFloat32
                );
    }

    static bool pyValCouldBeOfTypeConcrete(modeled_type* t, PyObject* pyRepresentation) {
        if (isFloat(t->getTypeCategory()))  {
            return PyFloat_Check(pyRepresentation);
        }

        if (isInteger(t->getTypeCategory()))  {
            return PyLong_CheckExact(pyRepresentation);
        }

        if (t->getTypeCategory() == Type::TypeCategory::catBool) {
            return PyBool_Check(pyRepresentation);
        }

        if (Type* otherT = extractTypeFrom(pyRepresentation->ob_type)) {
            Type::TypeCategory otherCat = otherT->getTypeCategory();

            if (isInteger(otherCat) || isFloat(otherCat) || otherCat == Type::TypeCategory::catBool) {
                return true;
            }
        }

        return false;
    }

    static PyObject* extractPythonObjectConcrete(RegisterType<T>* t, instance_ptr data) {
        if (t->getTypeCategory() == Type::TypeCategory::catInt64) {
            return PyLong_FromLong(*(int64_t*)data);
        }
        if (t->getTypeCategory() == Type::TypeCategory::catFloat64) {
            return PyFloat_FromDouble(*(double*)data);
        }
        if (t->getTypeCategory() == Type::TypeCategory::catBool) {
            return incref(*(bool*)data ? Py_True : Py_False);
        }
        return NULL;
    }

    PyObject* pyUnaryOperatorConcrete(const char* op, const char* opErr) {
        Type::TypeCategory cat = type()->getTypeCategory();

        if (strcmp(op, "__float__") == 0) {
            return PyFloat_FromDouble(*(T*)dataPtr());
        }
        if (strcmp(op, "__int__") == 0) {
            if (cat == Type::TypeCategory::catUInt64) {
                return PyLong_FromUnsignedLong(*(T*)dataPtr());
            }
            return PyLong_FromLong(*(T*)dataPtr());
        }
        if (strcmp(op, "__neg__") == 0) {
            T val = *(T*)dataPtr();
            val = -val;
            return extractPythonObject((instance_ptr)&val, type());
        }
        if (strcmp(op, "__inv__") == 0 && isInteger(type()->getTypeCategory())) {
            T val = *(T*)dataPtr();
            val = bitInvert(val);
            return extractPythonObject((instance_ptr)&val, type());
        }
        if (strcmp(op, "__index__") == 0 && isInteger(type()->getTypeCategory())) {
            int64_t val = *(T*)dataPtr();
            return PyLong_FromLong(val);
        }

        return PyInstance::pyUnaryOperatorConcrete(op, opErr);
    }

    PyObject* pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr) {
        if (PyLong_CheckExact(rhs)) {
            return pyOperatorConcreteForRegister<T, int64_t>(*(T*)dataPtr(), PyLong_AsLong(rhs), op, opErr);
        }
        if (PyBool_Check(rhs)) {
            return pyOperatorConcreteForRegister<T, bool>(*(T*)dataPtr(), PyLong_AsLong(rhs), op, opErr);
        }
        if (PyFloat_CheckExact(rhs)) {
            return pyOperatorConcreteForRegister<T, double>(*(T*)dataPtr(), PyFloat_AsDouble(rhs), op, opErr);
        }

        Type* rhsType = extractTypeFrom(rhs->ob_type);

        if (rhsType->getTypeCategory() == Type::TypeCategory::catBool) { return pyOperatorConcreteForRegister<T, bool>(*(T*)dataPtr(), *(bool*)((PyInstance*)rhs)->dataPtr(), op, opErr); }
        if (rhsType->getTypeCategory() == Type::TypeCategory::catInt8) { return pyOperatorConcreteForRegister<T, int8_t>(*(T*)dataPtr(), *(int8_t*)((PyInstance*)rhs)->dataPtr(), op, opErr); }
        if (rhsType->getTypeCategory() == Type::TypeCategory::catInt16) { return pyOperatorConcreteForRegister<T, int16_t>(*(T*)dataPtr(), *(int16_t*)((PyInstance*)rhs)->dataPtr(), op, opErr); }
        if (rhsType->getTypeCategory() == Type::TypeCategory::catInt32) { return pyOperatorConcreteForRegister<T, int32_t>(*(T*)dataPtr(), *(int32_t*)((PyInstance*)rhs)->dataPtr(), op, opErr); }
        if (rhsType->getTypeCategory() == Type::TypeCategory::catInt64) { return pyOperatorConcreteForRegister<T, int64_t>(*(T*)dataPtr(), *(int64_t*)((PyInstance*)rhs)->dataPtr(), op, opErr); }
        if (rhsType->getTypeCategory() == Type::TypeCategory::catUInt8) { return pyOperatorConcreteForRegister<T, uint8_t>(*(T*)dataPtr(), *(uint8_t*)((PyInstance*)rhs)->dataPtr(), op, opErr); }
        if (rhsType->getTypeCategory() == Type::TypeCategory::catUInt16) { return pyOperatorConcreteForRegister<T, uint16_t>(*(T*)dataPtr(), *(uint16_t*)((PyInstance*)rhs)->dataPtr(), op, opErr); }
        if (rhsType->getTypeCategory() == Type::TypeCategory::catUInt32) { return pyOperatorConcreteForRegister<T, uint32_t>(*(T*)dataPtr(), *(uint32_t*)((PyInstance*)rhs)->dataPtr(), op, opErr); }
        if (rhsType->getTypeCategory() == Type::TypeCategory::catUInt64) { return pyOperatorConcreteForRegister<T, uint64_t>(*(T*)dataPtr(), *(uint64_t*)((PyInstance*)rhs)->dataPtr(), op, opErr); }
        if (rhsType->getTypeCategory() == Type::TypeCategory::catFloat32) { return pyOperatorConcreteForRegister<T, float>(*(T*)dataPtr(), *(float*)((PyInstance*)rhs)->dataPtr(), op, opErr); }
        if (rhsType->getTypeCategory() == Type::TypeCategory::catFloat64) { return pyOperatorConcreteForRegister<T, double>(*(T*)dataPtr(), *(double*)((PyInstance*)rhs)->dataPtr(), op, opErr); }

        return PyInstance::pyOperatorConcrete(rhs, op, opErr);
    }

    PyObject* pyOperatorConcreteReverse(PyObject* rhs, const char* op, const char* opErr) {
        if (PyLong_CheckExact(rhs)) {
            return pyOperatorConcreteForRegister<int64_t, T>(PyLong_AsLong(rhs), *(T*)dataPtr(), op, opErr);
        }
        if (PyBool_Check(rhs)) {
            return pyOperatorConcreteForRegister<bool, T>(PyLong_AsLong(rhs), *(T*)dataPtr(), op, opErr);
        }
        if (PyFloat_CheckExact(rhs)) {
            return pyOperatorConcreteForRegister<double, T>(PyFloat_AsDouble(rhs), *(T*)dataPtr(), op, opErr);
        }

        Type* rhsType = extractTypeFrom(rhs->ob_type);

        if (rhsType->getTypeCategory() == Type::TypeCategory::catBool) { return pyOperatorConcreteForRegister<bool, T>(*(bool*)((PyInstance*)rhs)->dataPtr(), *(T*)dataPtr(), op, opErr); }
        if (rhsType->getTypeCategory() == Type::TypeCategory::catInt8) { return pyOperatorConcreteForRegister<int8_t, T>(*(int8_t*)((PyInstance*)rhs)->dataPtr(), *(T*)dataPtr(), op, opErr); }
        if (rhsType->getTypeCategory() == Type::TypeCategory::catInt16) { return pyOperatorConcreteForRegister<int16_t, T>(*(int16_t*)((PyInstance*)rhs)->dataPtr(), *(T*)dataPtr(), op, opErr); }
        if (rhsType->getTypeCategory() == Type::TypeCategory::catInt32) { return pyOperatorConcreteForRegister<int32_t, T>(*(int32_t*)((PyInstance*)rhs)->dataPtr(), *(T*)dataPtr(), op, opErr); }
        if (rhsType->getTypeCategory() == Type::TypeCategory::catInt64) { return pyOperatorConcreteForRegister<int64_t, T>(*(int64_t*)((PyInstance*)rhs)->dataPtr(), *(T*)dataPtr(), op, opErr); }
        if (rhsType->getTypeCategory() == Type::TypeCategory::catUInt8) { return pyOperatorConcreteForRegister<uint8_t, T>(*(uint8_t*)((PyInstance*)rhs)->dataPtr(), *(T*)dataPtr(), op, opErr); }
        if (rhsType->getTypeCategory() == Type::TypeCategory::catUInt16) { return pyOperatorConcreteForRegister<uint16_t, T>(*(uint16_t*)((PyInstance*)rhs)->dataPtr(), *(T*)dataPtr(), op, opErr); }
        if (rhsType->getTypeCategory() == Type::TypeCategory::catUInt32) { return pyOperatorConcreteForRegister<uint32_t, T>(*(uint32_t*)((PyInstance*)rhs)->dataPtr(), *(T*)dataPtr(), op, opErr); }
        if (rhsType->getTypeCategory() == Type::TypeCategory::catUInt64) { return pyOperatorConcreteForRegister<uint64_t, T>(*(uint64_t*)((PyInstance*)rhs)->dataPtr(), *(T*)dataPtr(), op, opErr); }
        if (rhsType->getTypeCategory() == Type::TypeCategory::catFloat32) { return pyOperatorConcreteForRegister<float, T>(*(float*)((PyInstance*)rhs)->dataPtr(), *(T*)dataPtr(), op, opErr); }
        if (rhsType->getTypeCategory() == Type::TypeCategory::catFloat64) { return pyOperatorConcreteForRegister<double, T>(*(double*)((PyInstance*)rhs)->dataPtr(), *(T*)dataPtr(), op, opErr); }

        return PyInstance::pyOperatorConcrete(rhs, op, opErr);
    }


    static bool compare_to_python_concrete(Type* t, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp) {
        if (PyBool_Check(other) && (!exact || t->getTypeCategory() == Type::TypeCategory::catBool)) {
            bool value = (other == Py_True ? 1 : 0);

            return cmpResultToBoolForPyOrdering(
                pyComparisonOp,
                value < *(T*)self ? -1 :
                value > *(T*)self ? 1 :
                    0
                );
        }

        if (PyLong_CheckExact(other) && (!exact || t->getTypeCategory() == Type::TypeCategory::catInt64)) {
            int64_t value = PyLong_AsLong(other);

            return cmpResultToBoolForPyOrdering(
                pyComparisonOp,
                value < *(T*)self ? -1 :
                value > *(T*)self ? 1 :
                    0
                );
        }

        if (PyFloat_Check(other) && (!exact || t->getTypeCategory() == Type::TypeCategory::catFloat64)) {
            float value = PyFloat_AsDouble(other);

            return cmpResultToBoolForPyOrdering(
                pyComparisonOp,
                value < *(T*)self ? -1 :
                value > *(T*)self ? 1 :
                    0
                );
        }

        return cmpResultToBoolForPyOrdering(pyComparisonOp,-1);
    }

    static void mirrorTypeInformationIntoPyTypeConcrete(RegisterType<T>* type, PyTypeObject* pyType) {
        //expose 'ElementType' as a member of the type object
        PyDict_SetItemString(pyType->tp_dict, "IsFloat",
            isFloat(type->getTypeCategory()) ? Py_True : Py_False
            );
        PyDict_SetItemString(pyType->tp_dict, "IsSignedInt",
            isInteger(type->getTypeCategory()) && !isUnsigned(type->getTypeCategory()) ? Py_True : Py_False
            );
        PyDict_SetItemString(pyType->tp_dict, "IsUnsignedInt",
            isUnsigned(type->getTypeCategory()) ? Py_True : Py_False
            );
        PyDict_SetItemString(pyType->tp_dict, "Bits",
            PyLong_FromLong(
                type->getTypeCategory() == Type::TypeCategory::catBool ? 1 :
                type->getTypeCategory() == Type::TypeCategory::catInt8 ? 8 :
                type->getTypeCategory() == Type::TypeCategory::catInt16 ? 16 :
                type->getTypeCategory() == Type::TypeCategory::catInt32 ? 32 :
                type->getTypeCategory() == Type::TypeCategory::catInt64 ? 64 :
                type->getTypeCategory() == Type::TypeCategory::catUInt8 ? 8 :
                type->getTypeCategory() == Type::TypeCategory::catUInt16 ? 16 :
                type->getTypeCategory() == Type::TypeCategory::catUInt32 ? 32 :
                type->getTypeCategory() == Type::TypeCategory::catUInt64 ? 64 :
                type->getTypeCategory() == Type::TypeCategory::catFloat32 ? 32 :
                type->getTypeCategory() == Type::TypeCategory::catFloat64 ? 64 : -1
                )
            );
    }
};

