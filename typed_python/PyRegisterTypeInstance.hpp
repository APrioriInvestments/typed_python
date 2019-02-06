#pragma once

#include "PyInstance.hpp"

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

        return PyInstance::pyUnaryOperatorConcrete(op, opErr);
    }

    PyObject* pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr) {
        return PyInstance::pyOperatorConcrete(rhs, op, opErr);
    }


};

