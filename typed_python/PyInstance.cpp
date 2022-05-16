/******************************************************************************
   Copyright 2017-2019 typed_python Authors

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

#include <Python.h>
#include <pystate.h>

#include <numpy/arrayobject.h>
#include <type_traits>

#include "AllTypes.hpp"
#include "_runtime.h"
#include "PyInstance.hpp"
#include "PyDictInstance.hpp"
#include "PyConstDictInstance.hpp"
#include "PyTupleOrListOfInstance.hpp"
#include "PyPointerToInstance.hpp"
#include "PyRefToInstance.hpp"
#include "PyCompositeTypeInstance.hpp"
#include "PyClassInstance.hpp"
#include "PyHeldClassInstance.hpp"
#include "PyBoundMethodInstance.hpp"
#include "PyAlternativeMatcherInstance.hpp"
#include "PyAlternativeInstance.hpp"
#include "PyFunctionInstance.hpp"
#include "PyStringInstance.hpp"
#include "PyBytesInstance.hpp"
#include "PyNoneInstance.hpp"
#include "PyRegisterTypeInstance.hpp"
#include "PyValueInstance.hpp"
#include "PyPythonSubclassInstance.hpp"
#include "PyPythonObjectOfTypeInstance.hpp"
#include "PyOneOfInstance.hpp"
#include "PySubclassOfInstance.hpp"
#include "PyForwardInstance.hpp"
#include "PyEmbeddedMessageInstance.hpp"
#include "PyPyCellInstance.hpp"
#include "PyTypedCellInstance.hpp"
#include "PyTemporaryReferenceTracer.hpp"
#include "PySetInstance.hpp"
#include "_types.hpp"

Type* PyInstance::type() {
    return extractTypeFrom(((PyObject*)this)->ob_type);
}

std::pair<Type*, instance_ptr> PyInstance::derefAnyRefTo() {
    Type* t = type();
    instance_ptr p = dataPtr();

    if (t->isRefTo()) {
        t = ((RefTo*)t)->getEltType();
        p = *(instance_ptr*)p;
    }

    return std::make_pair(t, p);
}

void PyInstance::resolveTemporaryReference() {
    if (!mTemporaryRefTo) {
        return;
    }

    // duplicate the HeldClass we're holding onto
    mContainingInstance = Instance(mTemporaryRefTo, type());

    // and mark that we are no longer a temporary reference.
    mTemporaryRefTo = nullptr;
}

instance_ptr PyInstance::dataPtr() {
    if (mTemporaryRefTo) {
        return mTemporaryRefTo;
    }
    return mContainingInstance.data();
}

//static
PyObject* PyInstance::undefinedBehaviorException() {
    static PyObject* module = ::internalsModule();
    static PyObject* t = PyObject_GetAttrString(module, "UndefinedBehaviorException");
    return t;
}

//static
PyObject* PyInstance::nonTypesAcceptedAsTypes() {
    static PyObject* module = ::internalsModule();
    static PyObject* t = PyObject_GetAttrString(module, "_nonTypesAcceptedAsTypes");
    return t;
}

// static
PyMethodDef* PyInstance::typeMethods(Type* t) {
    return specializeStatic(t->getTypeCategory(), [&](auto* concrete_null_ptr) {
        typedef typename std::remove_reference<decltype(*concrete_null_ptr)>::type py_instance_type;

        return py_instance_type::typeMethodsConcrete(t);
    });
}

PyMethodDef* PyInstance::typeMethodsConcrete(Type* t) {
    return new PyMethodDef [2] {
        {NULL, NULL}
    };
}

// static
void PyInstance::tp_dealloc(PyObject* self) {
    PyInstance* wrapper = (PyInstance*)self;

    wrapper->mContainingInstance.~Instance();

    Py_TYPE(self)->tp_free((PyObject*)self);
}

// static
bool PyInstance::pyValCouldBeOfType(Type* t, PyObject* pyRepresentation, ConversionLevel level) {
    t->assertForwardsResolvedSufficientlyToInstantiate();

    Type* argType = extractTypeFrom(pyRepresentation->ob_type);

    if (argType && Type::typesEquivalent(argType, t)) {
        return true;
    }

    return specializeStatic(t->getTypeCategory(), [&](auto* concrete_null_ptr) {
        typedef typename std::remove_reference<decltype(*concrete_null_ptr)>::type py_instance_type;

        return py_instance_type::pyValCouldBeOfTypeConcrete(
            (typename py_instance_type::modeled_type*)t,
            pyRepresentation,
            level
        );
    });
}

// static
void PyInstance::copyConstructFromPythonInstance(Type* eltType, instance_ptr tgt, PyObject* pyRepresentation, ConversionLevel level) {
    eltType->assertForwardsResolvedSufficientlyToInstantiate();

    Type* argType = extractTypeFrom(pyRepresentation->ob_type);

    Type::TypeCategory cat = eltType->getTypeCategory();

    if (argType) {
        Type* argTypeToUse = argType;
        instance_ptr dataPtr = ((PyInstance*)pyRepresentation)->dataPtr();

        if (argTypeToUse && argTypeToUse->isRefTo()) {
            dataPtr = *(instance_ptr*)dataPtr;
            argTypeToUse = ((RefTo*)argTypeToUse)->getEltType();
        }

        if (argTypeToUse && (
                Type::typesEquivalent(argTypeToUse, eltType)
                || (argTypeToUse->isSubclassOf(eltType) && cat != Type::TypeCategory::catClass)
        )) {
            // it's already the right kind of instance. Note that we disallow classes in this
            // case because when child class C masquerades as class B, it needs to have a different
            // class dispatch offset encoded in it and we don't want to accidentally allow the wrong
            // dispatch table to come through here. PyClassInstance is supposed to handle this in the concrete
            // function below.
            eltType->copy_constructor(tgt, dataPtr);
            return;
        }
    }

    //dispatch to the appropriate Py[]Instance type
    specializeStatic(cat, [&](auto* concrete_null_ptr) {
        typedef typename std::remove_reference<decltype(*concrete_null_ptr)>::type py_instance_type;

        py_instance_type::copyConstructFromPythonInstanceConcrete(
            (typename py_instance_type::modeled_type*)eltType,
            tgt,
            pyRepresentation,
            level
        );
    });
}

void PyInstance::copyConstructFromPythonInstanceConcrete(Type* eltType, instance_ptr tgt, PyObject* pyRepresentation, ConversionLevel level) {
    std::string typeName = "an instance of " + eltType->name();
    std::string aNewName = "a new " + eltType->name();

    if (eltType->isValue()) {
        typeName = "the value " + eltType->name();
        aNewName = typeName;
    } else if (eltType->isNone()) {
        typeName = "the value None";
        aNewName = typeName;
    }

    std::string verb;
    if (level == ConversionLevel::Signature) {
        throw std::logic_error("Object of type " + std::string(pyRepresentation->ob_type->tp_name) + " is not " + typeName);
    }

    if (level < ConversionLevel::Implicit) {
        throw std::logic_error("Cannot upcast an object of type " + std::string(pyRepresentation->ob_type->tp_name) + " to " + typeName);
    }

    if (level == ConversionLevel::Implicit) {
        throw std::logic_error("Cannot implicitly convert an object of type " + std::string(pyRepresentation->ob_type->tp_name) + " to " + typeName);
    }

    throw std::logic_error("Cannot construct " + aNewName + " from an instance of " + std::string(pyRepresentation->ob_type->tp_name));
}


// static
void PyInstance::constructFromPythonArguments(uint8_t* data, Type* t, PyObject* args, PyObject* kwargs) {
    t->assertForwardsResolvedSufficientlyToInstantiate();

    //dispatch to the appropriate PyInstance subclass
    specializeStatic(t->getTypeCategory(), [&](auto* concrete_null_ptr) {
        typedef typename std::remove_reference<decltype(*concrete_null_ptr)>::type py_instance_type;

        py_instance_type::constructFromPythonArgumentsConcrete(
            (typename py_instance_type::modeled_type*)t,
            data,
            args,
            kwargs
            );
    });
}

void PyInstance::constructFromPythonArgumentsConcrete(Type* t, uint8_t* data, PyObject* args, PyObject* kwargs) {
    if ((kwargs == NULL || PyDict_Size(kwargs) == 0) && (args == NULL || PyTuple_Size(args) == 0)) {
        if (t->is_default_constructible() && !t->isHeldClass()) {
            t->constructor(data);
            return;
        }
    }

    if ((kwargs == NULL || PyDict_Size(kwargs) == 0) && (args && PyTuple_Size(args) == 1)) {
        PyObjectHolder argTuple(PyTuple_GetItem(args, 0));

        copyConstructFromPythonInstance(t, data, argTuple, ConversionLevel::New);

        return;
    }

    if (kwargs && PyDict_Size(kwargs)) {
        throw std::logic_error("Can't initialize " + t->name() + " with keyword arguments.");
    }

    throw std::logic_error("Can't initialize " + t->name() + " with " + format(args ? PyTuple_Size(args) : 0) + " arguments.");
}

/**
 * produce the pythonic representation of this object. for values that have a direct python representation,
 * such as integers, strings, bools, or None, we return an actual python object. Otherwise,
 * we return a pointer to a PyInstance representing the object.
 */

// static
PyObject* PyInstance::extractPythonObject(instance_ptr data, Type* eltType, bool createTemporaryRef) {
    return translateExceptionToPyObject([&]() {
        if (eltType->getTypeCategory() == Type::TypeCategory::catHeldClass && createTemporaryRef) {
            // we never return 'held class' instances directly. Instead, we
            // return a 'Temporary' reference to them and install a trace handler
            // that forces them to become non-refto objects on the execution of the
            // next instruction in the parent stack frame.

            // this allows us to mimic the behavior of the compiler when we're
            // using these objects: by default, we never get naked reftos anywhere
            // we can find them.
            PyObject* res = PyInstance::initializeTemporaryRef(eltType, data);

            PyThreadState *tstate = PyThreadState_GET();
            PyFrameObject *f = tstate->frame;

            if (f) {
                PyTemporaryReferenceTracer::traceObject(res, f);
            }

            return res;
        }

        //dispatch to the appropriate Py[]Instance type
        PyObject* result = specializeStatic(eltType->getTypeCategory(), [&](auto* concrete_null_ptr) {
            typedef typename std::remove_reference<decltype(*concrete_null_ptr)>::type py_instance_type;

            return py_instance_type::extractPythonObjectConcrete(
                (typename py_instance_type::modeled_type*)eltType,
                data
            );
        });

        if (result) {
            return result;
        }

        if (!result && PyErr_Occurred()) {
            throw PythonExceptionSet();
        }

        Type* concreteT = eltType->pickConcreteSubclass(data);

        return PyInstance::initialize(concreteT, [&](instance_ptr selfData) {
            concreteT->copy_constructor(selfData, data);
        });
    });
}

PyObject* PyInstance::extractPythonObject(const Instance& instance) {
    return extractPythonObject(instance.data(), instance.type());
}

PyObject* PyInstance::extractPythonObjectConcrete(Type* eltType, instance_ptr data) {
    return NULL;
}

// static
PyObject* PyInstance::tp_new_type(PyTypeObject *subtype, PyObject *args, PyObject *kwds) {
    Type::TypeCategory catToProduce = ((NativeTypeCategoryWrapper*)subtype)->mCategory;

    if (catToProduce != Type::TypeCategory::catNamedTuple &&
            catToProduce != Type::TypeCategory::catAlternative) {
        if (kwds && PyDict_Size(kwds)) {
            PyErr_Format(PyExc_TypeError, "Type %S does not accept keyword arguments", (PyObject*)subtype);
            return NULL;
        }
    }

    if (catToProduce == Type::TypeCategory::catListOf) { return MakeListOfType(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catTupleOf) { return MakeTupleOfType(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catPointerTo ) { return MakePointerToType(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catRefTo ) { return MakeRefToType(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catTuple ) { return MakeTupleType(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catConstDict ) { return MakeConstDictType(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catSet ) { return MakeSetType(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catDict ) { return MakeDictType(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catOneOf ) { return MakeOneOfType(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catNamedTuple ) { return MakeNamedTupleType(nullptr, args, kwds); }
    if (catToProduce == Type::TypeCategory::catBool ) { return MakeBoolType(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catInt8 ) { return MakeInt8Type(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catInt16 ) { return MakeInt16Type(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catInt32 ) { return MakeInt32Type(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catInt64 ) { return MakeInt64Type(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catFloat32 ) { return MakeFloat32Type(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catFloat64 ) { return MakeFloat64Type(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catUInt8 ) { return MakeUInt8Type(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catUInt16 ) { return MakeUInt16Type(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catUInt32 ) { return MakeUInt32Type(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catUInt64 ) { return MakeUInt64Type(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catString ) { return MakeStringType(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catBytes ) { return MakeBytesType(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catEmbeddedMessage ) { return MakeEmbeddedMessageType(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catPyCell ) { return MakePyCellType(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catTypedCell ) { return MakeTypedCellType(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catNone ) { return MakeNoneType(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catValue ) { return MakeValueType(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catBoundMethod ) { return MakeBoundMethodType(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catAlternativeMatcher ) { return MakeAlternativeMatcherType(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catFunction ) { return MakeFunctionType(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catClass ) { return MakeClassType(nullptr, args); }
    if (catToProduce == Type::TypeCategory::catAlternative ) { return MakeAlternativeType(nullptr, args, kwds); }
    if (catToProduce == Type::TypeCategory::catSubclassOf ) { return MakeSubclassOfType(nullptr, args); }

    PyErr_Format(PyExc_TypeError, "unknown TypeCategory %S", (PyObject*)subtype);
    return NULL;
}

// static
PyObject* PyInstance::tp_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds) {
    return translateExceptionToPyObject([&]() {
        Type* eltType = extractTypeFrom(subtype);

        if (!eltType) {
            throw std::runtime_error("Can't find a TypedPython type for " + std::string(subtype->tp_name));
        }

        eltType->assertForwardsResolvedSufficientlyToInstantiate();

        if (isSubclassOfNativeType(subtype)) {
            PyInstance* self = (PyInstance*)subtype->tp_alloc(subtype, 0);

            try {
                self->initializeEmpty();

                self->initialize([&](instance_ptr data) {
                    constructFromPythonArguments(data, eltType, args, kwds);
                }, eltType);

                return (PyObject*)self;
            } catch(...) {
                subtype->tp_dealloc((PyObject*)self);
                throw;
            }

            // not reachable
            assert(false);
        } else {
            if (eltType->isClass() && ((Class*)eltType)->getOwnClassMembers().find("__typed_python_template__")
                    != ((Class*)eltType)->getOwnClassMembers().end()) {
                return PyObject_Call(
                    ((Class*)eltType)->getOwnClassMembers().find("__typed_python_template__")->second,
                    args,
                    kwds
                );
            }

            Instance inst(eltType, [&](instance_ptr tgt) {
                constructFromPythonArguments(tgt, eltType, args, kwds);
            });

            return extractPythonObject(inst.data(), eltType, false);
        }
    });
}

PyObject* PyInstance::pyUnaryOperator(PyObject* lhs, const char* op, const char* opErrRep) {
    return specializeForType(lhs, [&](auto& subtype) {
        return subtype.pyUnaryOperatorConcrete(op, opErrRep);
    });
}

PyObject* PyInstance::pyOperator(PyObject* lhs, PyObject* rhs, const char* op, const char* opErrRep) {
    return translateExceptionToPyObject([&]() {
        if (extractTypeFrom(lhs->ob_type)) {
            PyObject* ret = nullptr;
            ret = specializeForType(lhs, [&](auto& subtype) {
                return subtype.pyOperatorConcrete(rhs, op, opErrRep);
            });

            if (ret != Py_NotImplemented) {
                return ret;
            }
        }

        if (extractTypeFrom(rhs->ob_type)) {
            return specializeForType(rhs, [&](auto& subtype) {
                return subtype.pyOperatorConcreteReverse(lhs, op, opErrRep);
            });
        }

        long opErrRepLen = strlen(opErrRep);

        if (op[0] == '_' && op[1] == '_' && op[2] == 'i' && opErrRepLen && opErrRep[opErrRepLen-1] == '=') {
            // we were called with __iadd__, but our implementation doesn't have an 'i' form.
            // just delegate to the regular one.
            std::string opnameStr(op);
            opnameStr = "__" + opnameStr.substr(3);

            std::string opErrRepStr(opErrRep);
            opErrRepStr = opErrRepStr.substr(0, opErrRepStr.size() - 1);

            return pyOperator(lhs, rhs, opnameStr.c_str(), opErrRepStr.c_str());
        }

        PyErr_Format(PyExc_TypeError, "Invalid type arguments of type '%S' and '%S' to binary operator %s",
            lhs->ob_type,
            rhs->ob_type,
            op
            );

        throw PythonExceptionSet();
    });
}

PyObject* PyInstance::pyTernaryOperator(PyObject* lhs, PyObject* rhs, PyObject* thirdArg, const char* op, const char* opErrRep) {
    // only supporting binary version of ternary __pow__
    return translateExceptionToPyObject([&]() {
        PyObject* ret = nullptr;
        if (extractTypeFrom(lhs->ob_type)) {
            ret = specializeForType(lhs, [&](auto& subtype) {
                return subtype.pyTernaryOperatorConcrete(rhs, thirdArg, op, opErrRep);
            });
        }

        if (ret && ret != Py_NotImplemented) {
            return ret;
        }

        if (thirdArg == Py_None && extractTypeFrom(rhs->ob_type)) {
            return specializeForType(rhs, [&](auto& subtype) {
                return subtype.pyOperatorConcreteReverse(lhs, op, opErrRep);
            });
        }

        PyErr_Format(PyExc_TypeError, "Invalid type arguments of type '%S' and '%S' to ternary operator %s",
            lhs->ob_type,
            rhs->ob_type,
            op
            );

        throw PythonExceptionSet();
    });
}

int PyInstance::pyInquiry(PyObject* lhs, const char* op, const char* opErrRep) {
    return specializeForTypeReturningInt(lhs, [&](auto& subtype) {
        return subtype.pyInquiryConcrete(op, opErrRep);
    });
}

PyObject* PyInstance::pyUnaryOperatorConcrete(const char* op, const char* opErrRep) {
    PyErr_Format(PyExc_TypeError, "%s.%s is not implemented", type()->name().c_str(), op);
    throw PythonExceptionSet();
}

PyObject* PyInstance::pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErrRep) {
    return incref(Py_NotImplemented);
}

PyObject* PyInstance::pyOperatorConcreteReverse(PyObject* lhs, const char* op, const char* opErrRep) {
    return incref(Py_NotImplemented);
}

PyObject* PyInstance::pyTernaryOperatorConcrete(PyObject* rhs, PyObject* third, const char* op, const char* opErrRep) {
    return incref(Py_NotImplemented);
}

int PyInstance::pyInquiryConcrete(const char* op, const char* opErrRep) {
    int typeCat = type()->getTypeCategory();
    PyErr_Format(PyExc_TypeError, "Operation %s not permitted on type '%S' %d", op, (PyObject*)((PyObject*)this)->ob_type, typeCat);
    return -1;
}

PyObject* PyInstance::nb_inplace_add(PyObject* lhs, PyObject* rhs) {
    return pyOperator(lhs, rhs, "__iadd__", "+=");
}

PyObject* PyInstance::nb_inplace_subtract(PyObject* lhs, PyObject* rhs) {
    return pyOperator(lhs, rhs, "__isub__", "-=");
}

PyObject* PyInstance::nb_inplace_multiply(PyObject* lhs, PyObject* rhs) {
    return pyOperator(lhs, rhs, "__imul__", "*=");
}

PyObject* PyInstance::nb_inplace_remainder(PyObject* lhs, PyObject* rhs) {
    return pyOperator(lhs, rhs, "__imod__", "%=");
}

PyObject* PyInstance::nb_inplace_power(PyObject* lhs, PyObject* rhs, PyObject* modOrNone) {
    return pyOperator(lhs, rhs, "__ipow__", "**=");
}

PyObject* PyInstance::nb_inplace_lshift(PyObject* lhs, PyObject* rhs) {
    return pyOperator(lhs, rhs, "__ilshift__", "<<=");
}

PyObject* PyInstance::nb_inplace_rshift(PyObject* lhs, PyObject* rhs) {
    return pyOperator(lhs, rhs, "__irshift__", ">>=");
}

PyObject* PyInstance::nb_inplace_and(PyObject* lhs, PyObject* rhs) {
    return pyOperator(lhs, rhs, "__iand__", "&=");
}

PyObject* PyInstance::nb_inplace_xor(PyObject* lhs, PyObject* rhs) {
    return pyOperator(lhs, rhs, "__ixor__", "^=");
}

PyObject* PyInstance::nb_inplace_or(PyObject* lhs, PyObject* rhs) {
    return pyOperator(lhs, rhs, "__ior__", "|=");
}

PyObject* PyInstance::nb_floor_divide(PyObject* lhs, PyObject* rhs) {
    return pyOperator(lhs, rhs, "__floordiv__", "//");
}

PyObject* PyInstance::nb_true_divide(PyObject* lhs, PyObject* rhs) {
    return pyOperator(lhs, rhs, "__truediv__", "/");  // __div__ replaced by __truediv__
}

PyObject* PyInstance::nb_inplace_floor_divide(PyObject* lhs, PyObject* rhs) {
    return pyOperator(lhs, rhs, "__ifloordiv__", "//=");
}

PyObject* PyInstance::nb_inplace_true_divide(PyObject* lhs, PyObject* rhs) {
    return pyOperator(lhs, rhs, "__itruediv__", "/=");
}

PyObject* PyInstance::nb_inplace_matrix_multiply(PyObject* lhs, PyObject* rhs) {
    return pyOperator(lhs, rhs, "__imatmul__", "@=");
}

// static
PyObject* PyInstance::nb_negative(PyObject* lhs) {
    return pyUnaryOperator(lhs, "__neg__", "-");
}

// static
PyObject* PyInstance::nb_positive(PyObject* lhs) {
    return pyUnaryOperator(lhs, "__pos__", "+");
}

// static
PyObject* PyInstance::nb_absolute(PyObject* lhs) {
    return pyUnaryOperator(lhs, "__abs__", "abs");
}

// static
PyObject* PyInstance::nb_invert(PyObject* lhs) {
    return pyUnaryOperator(lhs, "__invert__", "~");
}

// static
PyObject* PyInstance::nb_int(PyObject* lhs) {
    return pyUnaryOperator(lhs, "__int__", "int");
}

// static
PyObject* PyInstance::nb_float(PyObject* lhs) {
    return pyUnaryOperator(lhs, "__float__", "float");
}

// static
int PyInstance::nb_bool(PyObject* lhs) {
    return pyInquiry(lhs, "__bool__", "bool");
}

// static
PyObject* PyInstance::nb_index(PyObject* lhs) {
    return pyUnaryOperator(lhs, "__index__", "index");
}

// static
PyObject* PyInstance::nb_matmul(PyObject* lhs, PyObject* rhs) {
    return pyOperator(lhs, rhs, "__matmul__", "@");
}

// static
PyObject* PyInstance::nb_divmod(PyObject* lhs, PyObject* rhs) {
    return pyOperator(lhs, rhs, "divmod", "divmod");
}

// static
PyObject* PyInstance::nb_power(PyObject* lhs, PyObject* rhs, PyObject* modOrNone) {
    return pyTernaryOperator(lhs, rhs, modOrNone, "__pow__", "**");
}

// static
PyObject* PyInstance::nb_and(PyObject* lhs, PyObject* rhs) {
    return pyOperator(lhs, rhs, "__and__", "&");
}

// static
PyObject* PyInstance::nb_xor(PyObject* lhs, PyObject* rhs) {
    return pyOperator(lhs, rhs, "__xor__", "^");
}

// static
PyObject* PyInstance::nb_or(PyObject* lhs, PyObject* rhs) {
    return pyOperator(lhs, rhs, "__or__", "|");
}

// static
PyObject* PyInstance::nb_rshift(PyObject* lhs, PyObject* rhs) {
    return pyOperator(lhs, rhs, "__rshift__", ">>");
}

// static
PyObject* PyInstance::nb_lshift(PyObject* lhs, PyObject* rhs) {
    return pyOperator(lhs, rhs, "__lshift__", "<<");
}

// static
PyObject* PyInstance::nb_add(PyObject* lhs, PyObject* rhs) {
    return pyOperator(lhs, rhs, "__add__", "+");
}

// static
PyObject* PyInstance::nb_subtract(PyObject* lhs, PyObject* rhs) {
    return pyOperator(lhs, rhs, "__sub__", "-");
}

// static
PyObject* PyInstance::nb_multiply(PyObject* lhs, PyObject* rhs) {
    return pyOperator(lhs, rhs, "__mul__", "*");
}

// static
PyObject* PyInstance::nb_remainder(PyObject* lhs, PyObject* rhs) {
    return pyOperator(lhs, rhs, "__mod__", "%");
}

// static
PyObject* PyInstance::sq_item(PyObject* o, Py_ssize_t ix) {
    return specializeForType(o, [&](auto& subtype) {
        return subtype.sq_item_concrete(ix);
    });
}

PyObject* PyInstance::sq_item_concrete(Py_ssize_t ix) {
    PyErr_Format(PyExc_TypeError, "%S object is not subscriptable", (PyObject*)((PyObject*)this)->ob_type);
    return NULL;
}

// static
int PyInstance::sq_ass_item(PyObject* o, Py_ssize_t ix, PyObject* v) {
    return specializeForTypeReturningInt(o, [&](auto& subtype) {
        return subtype.sq_ass_item_concrete(ix, v);
    });
}

int PyInstance::sq_ass_item_concrete(Py_ssize_t ix, PyObject* v) {
    PyErr_Format(PyExc_TypeError, "%S object is not subscript assignable", (PyObject*)((PyObject*)this)->ob_type);
    return -1;
}

// static
PyTypeObject* PyInstance::typeObj(Type* inType) {
    if (!inType->getTypeRep()) {
        inType->setTypeRep(typeObjInternal(inType));
    }

    return inType->getTypeRep();
}

PyObject* PyInstance::typePtrToPyTypeRepresentation(Type* inType) {
    return (PyObject*)typeObj(inType);
}

// static
PySequenceMethods* PyInstance::sequenceMethodsFor(Type* t) {
    if (    t->getTypeCategory() == Type::TypeCategory::catAlternative ||
            t->getTypeCategory() == Type::TypeCategory::catConcreteAlternative ||
            t->getTypeCategory() == Type::TypeCategory::catClass ||
            t->getTypeCategory() == Type::TypeCategory::catHeldClass
            ) {
        PySequenceMethods* res =
            new PySequenceMethods {0,0,0,0,0,0,0,0};
            res->sq_contains = (objobjproc)PyInstance::sq_contains;
            res->sq_length = (lenfunc)PyInstance::mp_and_sq_length;
        return res;
    }

    if (    t->getTypeCategory() == Type::TypeCategory::catTupleOf ||
            t->getTypeCategory() == Type::TypeCategory::catListOf ||
            t->getTypeCategory() == Type::TypeCategory::catTuple ||
            t->getTypeCategory() == Type::TypeCategory::catNamedTuple ||
            t->getTypeCategory() == Type::TypeCategory::catString ||
            t->getTypeCategory() == Type::TypeCategory::catBytes ||
            t->getTypeCategory() == Type::TypeCategory::catSet ||
            t->getTypeCategory() == Type::TypeCategory::catDict ||
            t->getTypeCategory() == Type::TypeCategory::catConstDict) {
        PySequenceMethods* res =
            new PySequenceMethods {0,0,0,0,0,0,0,0};

        if (t->getTypeCategory() == Type::TypeCategory::catConstDict || t->getTypeCategory() == Type::TypeCategory::catDict || t->getTypeCategory() == Type::TypeCategory::catSet) {
            res->sq_contains = (objobjproc)PyInstance::sq_contains;
        } else {
            res->sq_length = (lenfunc)PyInstance::mp_and_sq_length;
            res->sq_item = (ssizeargfunc)PyInstance::sq_item;
        }

        return res;
    }

    return 0;
}

// static
PyNumberMethods* PyInstance::numberMethods(Type* t) {
    return new PyNumberMethods {
        // We should eventually only enable this for the types that have definitions.
        // otherwise, things like PyIndex_Check(x) will return True on all of our types,
        // but then fail when we go to implement them.
        nb_add, //binaryfunc nb_add
        nb_subtract, //binaryfunc nb_subtract
        nb_multiply, //binaryfunc nb_multiply
        nb_remainder, //binaryfunc nb_remainder
        nb_divmod, //binaryfunc nb_divmod
        nb_power, //ternaryfunc nb_power
        nb_negative, //unaryfunc nb_negative
        nb_positive, //unaryfunc nb_positive
        nb_absolute, //unaryfunc nb_absolute
        nb_bool, //inquiry nb_bool
        nb_invert, //unaryfunc nb_invert
        nb_lshift, //binaryfunc nb_lshift
        nb_rshift, //binaryfunc nb_rshift
        nb_and, //binaryfunc nb_and
        nb_xor, //binaryfunc nb_xor
        nb_or, //binaryfunc nb_or
        nb_int, //unaryfunc nb_int
        0, //void *nb_reserved
        nb_float, //unaryfunc nb_float
        nb_inplace_add, //binaryfunc nb_inplace_add
        nb_inplace_subtract, //binaryfunc nb_inplace_subtract
        nb_inplace_multiply, //binaryfunc nb_inplace_multiply
        nb_inplace_remainder, //binaryfunc nb_inplace_remainder
        nb_inplace_power, //ternaryfunc nb_inplace_power
        nb_inplace_lshift, //binaryfunc nb_inplace_lshift
        nb_inplace_rshift, //binaryfunc nb_inplace_rshift
        nb_inplace_and, //binaryfunc nb_inplace_and
        nb_inplace_xor, //binaryfunc nb_inplace_xor
        nb_inplace_or, //binaryfunc nb_inplace_or
        nb_floor_divide, //binaryfunc nb_floor_divide
        nb_true_divide, //binaryfunc nb_true_divide
        nb_inplace_floor_divide, //binaryfunc nb_inplace_floor_divide
        nb_inplace_true_divide, //binaryfunc nb_inplace_true_divide
        nb_index, //unaryfunc nb_index
        nb_matmul, //binaryfunc nb_matrix_multiply
        nb_inplace_matrix_multiply  //binaryfunc nb_inplace_matrix_multiply
    };
}

// static
Py_ssize_t PyInstance::mp_and_sq_length(PyObject* o) {
    return specializeForTypeReturningSizeT(o, [&](auto& subtype) {
        return subtype.mp_and_sq_length_concrete();
    });
}

Py_ssize_t PyInstance::mp_and_sq_length_concrete() {
    PyErr_Format(
        PyExc_TypeError,
        "object of type '%S' has no len()",
        (PyObject*)((PyObject*)this)->ob_type
        );
    return -1;
}


int PyInstance::sq_contains(PyObject* o, PyObject* item) {
    return specializeForTypeReturningInt(o, [&](auto& subtype) {
        return subtype.sq_contains_concrete(item);
    });
}

int PyInstance::sq_contains_concrete(PyObject* item) {
    PyErr_Format(PyExc_TypeError, "Argument of type '%S' is not iterable", (PyObject*)((PyObject*)this)->ob_type);
    return -1;
}

int PyInstance::mp_ass_subscript(PyObject* o, PyObject* item, PyObject* value) {
    return specializeForTypeReturningInt(o, [&](auto& subtype) {
        return subtype.mp_ass_subscript_concrete(item, value);
    });
}

int PyInstance::mp_ass_subscript_concrete(PyObject* item, PyObject* value) {
    PyErr_Format(PyExc_TypeError, "'%S' object does not support item assignment", (PyObject*)((PyObject*)this)->ob_type);
    return -1;
}

PyObject* PyInstance::mp_subscript(PyObject* o, PyObject* item) {
    return specializeForType(o, [&](auto& subtype) {
        return subtype.mp_subscript_concrete(item);
    });
}

PyObject* PyInstance::mp_subscript_concrete(PyObject* item) {
    PyErr_Format(PyExc_TypeError, "'%S' object is not subscriptable", (PyObject*)((PyObject*)this)->ob_type);
    return NULL;
}

// static
PyMappingMethods* PyInstance::mappingMethods(Type* t) {
    static PyMappingMethods* mapMethods =
        new PyMappingMethods {
            PyInstance::mp_and_sq_length, //mp_length
            PyInstance::mp_subscript, //mp_subscript
            PyInstance::mp_ass_subscript //mp_ass_subscript
            };

    if (t->getTypeCategory() == Type::TypeCategory::catConstDict ||
        t->getTypeCategory() == Type::TypeCategory::catDict ||
        t->getTypeCategory() == Type::TypeCategory::catSet ||
        t->getTypeCategory() == Type::TypeCategory::catTupleOf ||
        t->getTypeCategory() == Type::TypeCategory::catListOf ||
        t->getTypeCategory() == Type::TypeCategory::catAlternative ||
        t->getTypeCategory() == Type::TypeCategory::catConcreteAlternative ||
        t->getTypeCategory() == Type::TypeCategory::catClass ||
        t->getTypeCategory() == Type::TypeCategory::catHeldClass
        ) {
        return mapMethods;
    }


    static PyMappingMethods* mapMethodsLite =
        new PyMappingMethods {
            nullptr,
            PyInstance::mp_subscript, //mp_subscript
            PyInstance::mp_ass_subscript //mp_ass_subscript
        };

    if (t->getTypeCategory() == Type::TypeCategory::catPointerTo) {
        return mapMethodsLite;
    }

    return 0;
}

// static
PyBufferProcs* PyInstance::bufferProcs() {
    static PyBufferProcs* procs = new PyBufferProcs { 0, 0 };
    return procs;
}

/**
    Determine if a given PyTypeObject* is one of our types.

    We are using pointer-equality with the tp_as_buffer function pointer
    that we set on our types. This should be safe because:
    - No other type can be pointing to it, and
    - All of our types point to the unique instance of PyBufferProcs
*/
// static
bool PyInstance::isNativeType(PyTypeObject* typeObj) {
    return typeObj->tp_as_buffer == bufferProcs();
}

/**
 *  Return true if the given PyTypeObject* is a subclass of a NativeType.
 *  This will return false when called with a native type
*/
// static
bool PyInstance::isSubclassOfNativeType(PyTypeObject* typeObj) {
    if (isNativeType(typeObj)) {
        return false;
    }

    while (typeObj) {
        if (isNativeType(typeObj)) {
            return true;
        }
        typeObj = typeObj->tp_base;
    }
    return false;
}

Type* PyInstance::rootNativeType(PyTypeObject* typeObj) {
    PyTypeObject* baseType = typeObj;

    while (!isNativeType(baseType) && baseType) {
        baseType = baseType->tp_base;
    }

    if (!baseType) {
        return nullptr;
    }

    return ((NativeTypeWrapper*)baseType)->mType;
}

// static
Type* PyInstance::extractTypeFrom(PyTypeObject* typeObj) {
    if (isSubclassOfNativeType(typeObj)) {
        return PythonSubclass::Make(rootNativeType(typeObj), typeObj);
    }

    if (isNativeType(typeObj)) {
        return ((NativeTypeWrapper*)typeObj)->mType;
    } else {
        return nullptr;
    }
}

std::pair<Type*, instance_ptr> PyInstance::extractTypeAndPtrFrom(PyObject* obj) {
    Type* t = extractTypeFrom(obj->ob_type);

    if (!t) {
        return std::make_pair(t, (instance_ptr)nullptr);
    }

    return ((PyInstance*)obj)->derefAnyRefTo();
}


PyObject* PyInstance::getInternalModuleMember(const char* name) {
    static PyObject* internalsModule = ::internalsModule();

    if (!internalsModule) {
        PyErr_SetString(PyExc_TypeError, "Internal error: couldn't find typed_python.internals");
        return nullptr;
    }

    PyObject* result = PyObject_GetAttrString(internalsModule, name);

    if (!result) {
        PyErr_Format(PyExc_TypeError, "Internal error: couldn't find typed_python.internals.%s", name);
        return nullptr;
    }

    return result;
}

//construct the base class that all actual type instances of a given TypeCategory descend from
PyTypeObject* PyInstance::allTypesBaseType() {
    auto allocateBaseType = [&]() {
        PyTypeObject* result = new PyTypeObject {
            PyVarObject_HEAD_INIT(NULL, 0)              // TYPE (c.f., Type Objects)
            .tp_name = "typed_python._types.Type",          // const char*
            .tp_basicsize = sizeof(PyInstance),         // Py_ssize_t
            .tp_itemsize = 0,                           // Py_ssize_t
            .tp_dealloc = PyInstance::tp_dealloc,       // destructor

            #if PY_MINOR_VERSION < 8
            .tp_print = 0,                              // printfunc  (Changed to tp_vectorcall_offset in Python 3.8)
            #else
            .tp_vectorcall_offset = 0,                  // printfunc  (Changed to tp_vectorcall_offset in Python 3.8)
            #endif

            .tp_getattr = 0,                            // getattrfunc
            .tp_setattr = 0,                            // setattrfunc
            .tp_as_async = 0,                           // PyAsyncMethods*
            .tp_repr = tp_repr,                         // reprfunc
            .tp_as_number = 0,                          // PyNumberMethods*
            .tp_as_sequence = 0,                        // PySequenceMethods*
            .tp_as_mapping = 0,                         // PyMappingMethods*
            .tp_hash = tp_hash,                         // hashfunc
            .tp_call = tp_call,                         // ternaryfunc
            .tp_str = tp_str,                           // reprfunc
            .tp_getattro = 0,                           // getattrofunc
            .tp_setattro = 0,                           // setattrofunc
            .tp_as_buffer = 0,                          // PyBufferProcs*
            .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
                                                        // unsigned long
            .tp_doc = 0,                                // const char*
            .tp_traverse = 0,                           // traverseproc
            .tp_clear = 0,                              // inquiry
            .tp_richcompare = 0,                        // richcmpfunc
            .tp_weaklistoffset = 0,                     // Py_ssize_t
            .tp_iter = 0,                               // getiterfunc tp_iter;
            .tp_iternext = 0,                           // iternextfunc
            .tp_methods = 0,                            // struct PyMethodDef*
            .tp_members = 0,                            // struct PyMemberDef*
            .tp_getset = 0,                             // struct PyGetSetDef*
            .tp_base = 0,                               // struct _typeobject*
            .tp_dict = 0,                               // PyObject*
            .tp_descr_get = 0,                          // descrgetfunc
            .tp_descr_set = 0,                          // descrsetfunc
            .tp_dictoffset = 0,                         // Py_ssize_t
            .tp_init = 0,                               // initproc
            .tp_alloc = 0,                              // allocfunc
            .tp_new = 0,                                // newfunc
            .tp_free = 0,                               // freefunc /* Low-level free-memory routine */
            .tp_is_gc = 0,                              // inquiry  /* For PyObject_IS_GC */
            .tp_bases = 0,                              // PyObject*
            .tp_mro = 0,                                // PyObject* /* method resolution order */
            .tp_cache = 0,                              // PyObject*
            .tp_subclasses = 0,                         // PyObject*
            .tp_weaklist = 0,                           // PyObject*
            .tp_del = 0,                                // destructor
            .tp_version_tag = 0,                        // unsigned int
            .tp_finalize = 0,                           // destructor

            #if PY_MINOR_VERSION >= 8
            .tp_vectorcall = 0,
            #endif

            #if PY_MINOR_VERSION == 8
            .tp_print = 0,
            #endif

        };

        PyType_Ready(result);

        return result;
    };

    static PyTypeObject* baseType = allocateBaseType();

    return baseType;
}

//construct the base class that all actual type instances of a given TypeCategory descend from
PyTypeObject* PyInstance::typeCategoryBaseType(Type::TypeCategory category) {
    static std::map<Type::TypeCategory, NativeTypeCategoryWrapper*> types;

    if (types.find(category) == types.end()) {
        PyObject* classDict = PyDict_New();
        PyDict_SetItemString(classDict, "__module__", PyUnicode_FromString("typed_python._types"));

        types[category] = new NativeTypeCategoryWrapper { {
            PyVarObject_HEAD_INIT(NULL, 0)              // TYPE (c.f., Type Objects)
            .tp_name = (new std::string("typed_python._types." + Type::categoryToString(category)))->c_str(),          // const char*
            .tp_basicsize = sizeof(PyInstance),         // Py_ssize_t
            .tp_itemsize = 0,                           // Py_ssize_t
            .tp_dealloc = PyInstance::tp_dealloc,       // destructor

            #if PY_MINOR_VERSION < 8
            .tp_print = 0,                              // printfunc
            #else
            .tp_vectorcall_offset = 0,                  // printfunc  (Changed to tp_vectorcall_offset in Python 3.8)
            #endif

            .tp_getattr = 0,                            // getattrfunc
            .tp_setattr = 0,                            // setattrfunc
            .tp_as_async = 0,                           // PyAsyncMethods*
            .tp_repr = tp_repr,                         // reprfunc
            .tp_as_number = 0,                          // PyNumberMethods*
            .tp_as_sequence = 0,                        // PySequenceMethods*
            .tp_as_mapping = 0,                         // PyMappingMethods*
            .tp_hash = tp_hash,                         // hashfunc
            .tp_call = tp_call,                         // ternaryfunc
            .tp_str = tp_str,                           // reprfunc
            .tp_getattro = 0,                           // getattrofunc
            .tp_setattro = 0,                           // setattrofunc
            .tp_as_buffer = 0,                          // PyBufferProcs*
            .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
                                                        // unsigned long
            .tp_doc = 0,                                // const char*
            .tp_traverse = 0,                           // traverseproc
            .tp_clear = 0,                              // inquiry
            .tp_richcompare = 0,                        // richcmpfunc
            .tp_weaklistoffset = 0,                     // Py_ssize_t
            .tp_iter = 0,                               // getiterfunc tp_iter;
            .tp_iternext = 0,                           // iternextfunc
            .tp_methods = 0,                            // struct PyMethodDef*
            .tp_members = 0,                            // struct PyMemberDef*
            .tp_getset = 0,                             // struct PyGetSetDef*
            .tp_base = allTypesBaseType(),              // struct _typeobject*
            .tp_dict = classDict,                       // PyObject*
            .tp_descr_get = 0,                          // descrgetfunc
            .tp_descr_set = 0,                          // descrsetfunc
            .tp_dictoffset = 0,                         // Py_ssize_t
            .tp_init = 0,                               // initproc
            .tp_alloc = 0,                              // allocfunc
            .tp_new = PyInstance::tp_new_type,          // newfunc
            .tp_free = 0,                               // freefunc /* Low-level free-memory routine */
            .tp_is_gc = 0,                              // inquiry  /* For PyObject_IS_GC */
            .tp_bases = 0,                              // PyObject*
            .tp_mro = 0,                                // PyObject* /* method resolution order */
            .tp_cache = 0,                              // PyObject*
            .tp_subclasses = 0,                         // PyObject*
            .tp_weaklist = 0,                           // PyObject*
            .tp_del = 0,                                // destructor
            .tp_version_tag = 0,                        // unsigned int
            .tp_finalize = 0,                           // destructor


            #if PY_MINOR_VERSION >= 8
            .tp_vectorcall = 0,
            #endif

            #if PY_MINOR_VERSION == 8
            .tp_print = 0,
            #endif
            },
            category
        };

        if (category == Type::TypeCategory::catClass) {
            static PyTypeObject* classMetaclass = (PyTypeObject*)getInternalModuleMember("ClassMetaclass");
            ((PyObject*)&types[category]->typeObj)->ob_type = incref(classMetaclass);
        }

        PyType_Ready((PyTypeObject*)types[category]);
    }

    return (PyTypeObject*)types[category];
}

PyTypeObject* PyInstance::typeObjInternal(Type* inType) {
    if (inType->getTypeCategory() == ::Type::TypeCategory::catPythonSubclass) {
        return inType->getTypeRep();
    }
    if (inType->getTypeCategory() == ::Type::TypeCategory::catInt64) {
        return &PyLong_Type;
    }
    if (inType->getTypeCategory() == ::Type::TypeCategory::catFloat64) {
        return &PyFloat_Type;
    }
    if (inType->getTypeCategory() == ::Type::TypeCategory::catBool) {
        return &PyBool_Type;
    }
    if (inType->getTypeCategory() == ::Type::TypeCategory::catString) {
        return &PyUnicode_Type;
    }
    if (inType->getTypeCategory() == ::Type::TypeCategory::catBytes) {
        return &PyBytes_Type;
    }
    if (inType->getTypeCategory() == ::Type::TypeCategory::catNone) {
        return Py_None->ob_type;
    }
    if (inType->getTypeCategory() == ::Type::TypeCategory::catPythonObjectOfType) {
        return ((PythonObjectOfType*)inType)->pyType();
    }

    static std::map<Type*, NativeTypeWrapper*> types;

    auto it = types.find(inType);
    if (it != types.end()) {
        return (PyTypeObject*)it->second;
    }

    Type::TypeCategory cat = inType->getTypeCategory();

    types[inType] = new NativeTypeWrapper { {
            PyVarObject_HEAD_INIT(NULL, 0)              // TYPE (c.f., Type Objects)
            .tp_name = (new std::string(inType->nameWithModule()))->c_str(),          // const char*
            .tp_basicsize = sizeof(PyInstance),         // Py_ssize_t
            .tp_itemsize = 0,                           // Py_ssize_t
            .tp_dealloc = PyInstance::tp_dealloc,       // destructor

            #if PY_MINOR_VERSION < 8
            .tp_print = 0,                              // printfunc
            #else
            .tp_vectorcall_offset = 0,                  // printfunc  (Changed to tp_vectorcall_offset in Python 3.8)
            #endif

            .tp_getattr = 0,                            // getattrfunc
            .tp_setattr = 0,                            // setattrfunc
            .tp_as_async = 0,                           // PyAsyncMethods*
            .tp_repr = tp_repr,                         // reprfunc
            .tp_as_number = numberMethods(inType),      // PyNumberMethods*
            .tp_as_sequence = sequenceMethodsFor(inType),   // PySequenceMethods*
            .tp_as_mapping = mappingMethods(inType),    // PyMappingMethods*
            .tp_hash = tp_hash,                         // hashfunc
            .tp_call = tp_call,                         // ternaryfunc
            .tp_str = tp_str,                           // reprfunc
            .tp_getattro = PyInstance::tp_getattro,     // getattrofunc
            .tp_setattro = PyInstance::tp_setattro,     // setattrofunc
            .tp_as_buffer = bufferProcs(),              // PyBufferProcs*
            .tp_flags = typeCanBeSubclassed(inType) ?
                Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE
            :   Py_TPFLAGS_DEFAULT,                     // unsigned long
            .tp_doc = inType->doc(),                    // const char*
            .tp_traverse = 0,                           // traverseproc
            .tp_clear = 0,                              // inquiry
            .tp_richcompare = tp_richcompare,           // richcmpfunc
            .tp_weaklistoffset = 0,                     // Py_ssize_t
            .tp_iter = cat == Type::TypeCategory::catConstDict ||
                        cat == Type::TypeCategory::catDict ||
                        cat == Type::TypeCategory::catSet ||
                        cat == Type::TypeCategory::catConcreteAlternative ||
                        cat == Type::TypeCategory::catClass ||
                        cat == Type::TypeCategory::catPointerTo
                         ?
                PyInstance::tp_iter
            :   0,                                      // getiterfunc tp_iter;
            .tp_iternext = PyInstance::tp_iternext,     // iternextfunc
            .tp_methods = typeMethods(inType),          // struct PyMethodDef*
            .tp_members = 0,                            // struct PyMemberDef*
            .tp_getset = 0,                             // struct PyGetSetDef*
            .tp_base = 0,                               // struct _typeobject*
            .tp_dict = PyDict_New(),                    // PyObject*
            .tp_descr_get = inType->getTypeCategory() == Type::TypeCategory::catFunction ?
                PyInstance::tp_descr_get : 0,           // descrgetfunc
            .tp_descr_set = 0,                          // descrsetfunc
            .tp_dictoffset = 0,                         // Py_ssize_t
            .tp_init = 0,                               // initproc
            .tp_alloc = 0,                              // allocfunc
            .tp_new = PyInstance::tp_new,                // newfunc
            .tp_free = 0,                               // freefunc /* Low-level free-memory routine */
            .tp_is_gc = 0,                              // inquiry  /* For PyObject_IS_GC */
            .tp_bases = 0,                              // PyObject*
            .tp_mro = 0,                                // PyObject* /* method resolution order */
            .tp_cache = 0,                              // PyObject*
            .tp_subclasses = 0,                         // PyObject*
            .tp_weaklist = 0,                           // PyObject*
            .tp_del = 0,                                // destructor
            .tp_version_tag = 0,                        // unsigned int
            .tp_finalize = 0,                           // destructor


            #if PY_MINOR_VERSION >= 8
            .tp_vectorcall = 0,
            #endif

            #if PY_MINOR_VERSION == 8
            .tp_print = 0,
            #endif
            },
        inType
    };

    // at this point, the dictionary has an entry, so if we recurse back to this function
    // we will return the correct entry.
    if (inType->getBaseType()) {
        types[inType]->typeObj.tp_base = incref(typeObjInternal((Type*)inType->getBaseType()));
    } else  {
        types[inType]->typeObj.tp_base = incref(typeCategoryBaseType(inType->getTypeCategory()));
    }

    // if we are an instance of 'Class', we must explicitly set our Metaclass to internals.ClassMetaclass,
    // so that when other classes inherit from us they also inherit our metaclass.
    if (inType->getTypeCategory() == Type::TypeCategory::catClass) {
        static PyTypeObject* classMetaclass = (PyTypeObject*)getInternalModuleMember("ClassMetaclass");
        ((PyObject*)&types[inType]->typeObj)->ob_type = incref(classMetaclass);
    }

    PyType_Ready((PyTypeObject*)types[inType]);

    PyDict_SetItemString(
        types[inType]->typeObj.tp_dict,
        "__typed_python_category__",
        categoryToPyString(inType->getTypeCategory())
        );

    mirrorTypeInformationIntoPyType(inType, &types[inType]->typeObj);

    return (PyTypeObject*)types[inType];
}

// static
int PyInstance::tp_setattro(PyObject *o, PyObject* attrName, PyObject* attrVal) {
    if (!PyUnicode_Check(attrName)) {
        PyErr_Format(
            PyExc_AttributeError,
            "Cannot set attribute '%S' on instance of type '%S'. Attribute does not resolve to a string",
            attrName, o->ob_type
        );
        return -1;
    }

    return specializeForTypeReturningInt(o, [&](auto& subtype) {
        return subtype.tp_setattr_concrete(attrName, attrVal);
    });
}

int PyInstance::tp_setattr_concrete(PyObject* attrName, PyObject* attrVal) {
    PyErr_Format(
        PyExc_AttributeError,
        "'%s' object has no attribute '%S'",
        type()->name().c_str(),
        attrName
    );

    return -1;
}

// static
PyObject* PyInstance::tp_call(PyObject* o, PyObject* args, PyObject* kwargs) {
    return specializeForType(o, [&](auto& subtype) {
        return subtype.tp_call_concrete(args, kwargs);
    });
}

PyObject* PyInstance::tp_call_concrete(PyObject* args, PyObject* kwargs) {
    PyErr_Format(PyExc_TypeError, "'%s' object is not callable", type()->name().c_str());
    return nullptr;
}

PyObject* PyInstance::tp_getattr_concrete(PyObject* pyAttrName, const char* attrName) {
    return PyObject_GenericGetAttr((PyObject*)this, pyAttrName);
}

// static
PyObject* PyInstance::tp_getattro(PyObject *o, PyObject* attrName) {
    if (!PyUnicode_Check(attrName)) {
        PyErr_SetString(PyExc_AttributeError, "attribute is not a string");
        return nullptr;
    }

    const char *attr_name = PyUnicode_AsUTF8(attrName);

    return specializeForType(o, [&](auto& subtype) {
        return subtype.tp_getattr_concrete(attrName, attr_name);
    });
}

// static
Py_hash_t PyInstance::tp_hash(PyObject *o) {
    return translateExceptionToPyObjectReturningInt([&]() {
        Type* self_type = extractTypeFrom(o->ob_type);
        PyInstance* w = (PyInstance*)o;

        int64_t h = -1;
        if (self_type->getTypeCategory() == Type::TypeCategory::catConcreteAlternative) {
            h = ((PyConcreteAlternativeInstance*)o)->tryCallHashMethod();
        } else if (self_type->getTypeCategory() == Type::TypeCategory::catClass) {
            h = ((PyClassInstance*)o)->tryCallHashMemberFunction();
        }
        if (h == -1) {
            h = self_type->hash(w->dataPtr());
        }
        if (h == -1) {
            h = -2;
        }

        return h;
    });
}

// static
bool PyInstance::compare_to_python(Type* t, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp) {
    if (exact && pyComparisonOp != Py_EQ && pyComparisonOp != Py_NE) {
        throw std::runtime_error("Exact must be used with Py_EQ or Py_NE only");
    }

    if (t->getTypeCategory() == Type::TypeCategory::catValue) {
        Value* valType = (Value*)t;

        return compare_to_python(valType->value().type(), valType->value().data(), other, exact, pyComparisonOp);
    }

    if (t->getTypeCategory() != Type::TypeCategory::catConcreteAlternative) {
        Type* otherT = extractTypeFrom(other->ob_type);

        if (otherT && otherT == t) {
            return t->cmp(self, ((PyInstance*)other)->dataPtr(), pyComparisonOp, false);
        }
    }

    return specializeStatic(t->getTypeCategory(), [&](auto* concrete_null_ptr) {
        typedef typename std::remove_reference<decltype(*concrete_null_ptr)>::type py_instance_type;

        return py_instance_type::compare_to_python_concrete(
            (typename py_instance_type::modeled_type*)t,
            self,
            other,
            exact,
            pyComparisonOp
            );
    });
}

bool PyInstance::compare_to_python_concrete(Type* t, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp) {
    if (pyComparisonOp == Py_NE) {
        return true;
    }
    if (pyComparisonOp == Py_EQ) {
        return false;
    }

    throw std::runtime_error("Cannot compare instances of type " + t->name() + " and " + other->ob_type->tp_name);

    return false;
}

bool PyInstance::compare_as_iterator_to_python_concrete(PyObject* other, int pyComparisonOp) {
    if (pyComparisonOp == Py_NE) {
        return true;
    }
    if (pyComparisonOp == Py_EQ) {
        return false;
    }

    throw std::runtime_error("Cannot compare instances of type " + type()->name() + " and " + other->ob_type->tp_name);

    return false;
}

int PyInstance::reversePyOpOrdering(int op) {
    if (op == Py_LT) {
        return Py_GT;
    }
    if (op == Py_LE) {
        return Py_GE;
    }
    if (op == Py_GT) {
        return Py_LT;
    }
    if (op == Py_GE) {
        return Py_LE;
    }

    return op;
}

// static
PyObject* PyInstance::tp_richcompare(PyObject *a, PyObject *b, int op) {
    return translateExceptionToPyObject([&]() {
        Type* own = extractTypeFrom(a->ob_type);
        Type* other = extractTypeFrom(b->ob_type);

        if (!own && !other) {
            PyErr_Format(PyExc_TypeError, "Can't call tp_richcompare where neither object is a typed_python object!");
            return (PyObject*)NULL;
        }

        if (own && other && own == other && ((PyInstance*)a)->mIteratorOffset == -1 && ((PyInstance*)b)->mIteratorOffset == -1) {
            return incref(compare_to_python(own, ((PyInstance*)a)->dataPtr(), b, false, op) ? Py_True : Py_False);
        }

        if (!own) {
            op = reversePyOpOrdering(op);
            std::swap(a, b);
        }

        PyInstance* instance = (PyInstance*)a;

        bool cmp;

        if (instance->mIteratorOffset != -1) {
            cmp = instance->specializeStatic(instance->type()->getTypeCategory(), [&](auto* concrete_null_ptr) {
                typedef typename std::remove_reference<decltype(*concrete_null_ptr)>::type py_instance_type;

                return ((py_instance_type*)instance)->compare_as_iterator_to_python_concrete(b, op);
            });
        } else {
            cmp = compare_to_python(instance->type(), instance->dataPtr(), b, false, op);
        }

        return incref(cmp ? Py_True : Py_False);
    });
}

// static
PyObject* PyInstance::tp_iter(PyObject *self) {
    return specializeForType(self, [&](auto& subtype) {
        return subtype.tp_iter_concrete();
        }
    );
}

// static
PyObject* PyInstance::tp_iternext(PyObject *self) {
    return specializeForType(self, [&](auto& subtype) {
        return subtype.tp_iternext_concrete();
        }
    );
}

PyObject* PyInstance::tp_iter_concrete() {
    PyErr_Format(PyExc_TypeError, "Cannot iterate an instance of %s", type()->name().c_str());
    throw PythonExceptionSet();
}

PyObject* PyInstance::tp_iternext_concrete() {
    PyErr_Format(PyExc_TypeError, "Cannot iterate an instance of %s", type()->name().c_str());
    throw PythonExceptionSet();
}

// static
PyObject* PyInstance::tp_repr(PyObject *self) {
    return specializeForType(self, [&](auto& subtype) {
        return subtype.tp_repr_concrete();
        }
    );
}

// static
PyObject* PyInstance::tp_descr_get(PyObject* func, PyObject* obj, PyObject* type) {
    if (obj == Py_None || obj == NULL) {
        return incref(func);
    }
    return PyMethod_New(func, obj);
}

PyObject* PyInstance::tp_repr_concrete() {
    std::ostringstream str;
    ReprAccumulator accumulator(str);

    type()->repr(dataPtr(), accumulator, false);

    return PyUnicode_FromString(str.str().c_str());
}

// static
PyObject* PyInstance::tp_str(PyObject *self) {
    return specializeForType(self, [&](auto& subtype) {
        return subtype.tp_str_concrete();
        }
    );
}

PyObject* PyInstance::tp_str_concrete() {
    std::ostringstream str;
    ReprAccumulator accumulator(str);

    type()->repr(dataPtr(), accumulator, true);

    return PyUnicode_FromString(str.str().c_str());
}

// static
bool PyInstance::typeCanBeSubclassed(Type* t) {
    return (
        t->getTypeCategory() == Type::TypeCategory::catNamedTuple ||
        t->getTypeCategory() == Type::TypeCategory::catClass
    );
}

// static
void PyInstance::mirrorTypeInformationIntoPyType(Type* inType, PyTypeObject* pyType) {
    specializeStatic(inType->getTypeCategory(), [&](auto* concrete_null_ptr) {
        typedef typename std::remove_reference<decltype(*concrete_null_ptr)>::type py_instance_type;

        py_instance_type::mirrorTypeInformationIntoPyTypeConcrete(
            (typename py_instance_type::modeled_type*)inType,
            pyType
            );
    });
}
void PyInstance::mirrorTypeInformationIntoPyTypeConcrete(Type* inType, PyTypeObject* pyType) {
    //noop
}

/**
 *  We are repeating this here rather than using Type::categoryToName because we want to create a singleton PyUnicode
 *  object for each type category to make this function ultra fast. We need the 'static' declarations within each
 *  if-branch to be their own objects.
 */
// static
PyObject* PyInstance::categoryToPyString(Type::TypeCategory cat) {
    if (cat == Type::TypeCategory::catNone) { static PyObject* res = PyUnicode_FromString("None"); return res; }
    if (cat == Type::TypeCategory::catBool) { static PyObject* res = PyUnicode_FromString("Bool"); return res; }
    if (cat == Type::TypeCategory::catUInt8) { static PyObject* res = PyUnicode_FromString("UInt8"); return res; }
    if (cat == Type::TypeCategory::catUInt16) { static PyObject* res = PyUnicode_FromString("UInt16"); return res; }
    if (cat == Type::TypeCategory::catUInt32) { static PyObject* res = PyUnicode_FromString("UInt32"); return res; }
    if (cat == Type::TypeCategory::catUInt64) { static PyObject* res = PyUnicode_FromString("UInt64"); return res; }
    if (cat == Type::TypeCategory::catInt8) { static PyObject* res = PyUnicode_FromString("Int8"); return res; }
    if (cat == Type::TypeCategory::catInt16) { static PyObject* res = PyUnicode_FromString("Int16"); return res; }
    if (cat == Type::TypeCategory::catInt32) { static PyObject* res = PyUnicode_FromString("Int32"); return res; }
    if (cat == Type::TypeCategory::catInt64) { static PyObject* res = PyUnicode_FromString("Int64"); return res; }
    if (cat == Type::TypeCategory::catString) { static PyObject* res = PyUnicode_FromString("String"); return res; }
    if (cat == Type::TypeCategory::catBytes) { static PyObject* res = PyUnicode_FromString("Bytes"); return res; }
    if (cat == Type::TypeCategory::catFloat32) { static PyObject* res = PyUnicode_FromString("Float32"); return res; }
    if (cat == Type::TypeCategory::catFloat64) { static PyObject* res = PyUnicode_FromString("Float64"); return res; }
    if (cat == Type::TypeCategory::catValue) { static PyObject* res = PyUnicode_FromString("Value"); return res; }
    if (cat == Type::TypeCategory::catOneOf) { static PyObject* res = PyUnicode_FromString("OneOf"); return res; }
    if (cat == Type::TypeCategory::catTupleOf) { static PyObject* res = PyUnicode_FromString("TupleOf"); return res; }
    if (cat == Type::TypeCategory::catPointerTo) { static PyObject* res = PyUnicode_FromString("PointerTo"); return res; }
    if (cat == Type::TypeCategory::catRefTo) { static PyObject* res = PyUnicode_FromString("RefTo"); return res; }
    if (cat == Type::TypeCategory::catListOf) { static PyObject* res = PyUnicode_FromString("ListOf"); return res; }
    if (cat == Type::TypeCategory::catNamedTuple) { static PyObject* res = PyUnicode_FromString("NamedTuple"); return res; }
    if (cat == Type::TypeCategory::catTuple) { static PyObject* res = PyUnicode_FromString("Tuple"); return res; }
    if (cat == Type::TypeCategory::catSet) { static PyObject* res = PyUnicode_FromString("Set"); return res; }
    if (cat == Type::TypeCategory::catDict) { static PyObject* res = PyUnicode_FromString("Dict"); return res; }
    if (cat == Type::TypeCategory::catConstDict) { static PyObject* res = PyUnicode_FromString("ConstDict"); return res; }
    if (cat == Type::TypeCategory::catAlternative) { static PyObject* res = PyUnicode_FromString("Alternative"); return res; }
    if (cat == Type::TypeCategory::catConcreteAlternative) { static PyObject* res = PyUnicode_FromString("ConcreteAlternative"); return res; }
    if (cat == Type::TypeCategory::catPythonSubclass) { static PyObject* res = PyUnicode_FromString("PythonSubclass"); return res; }
    if (cat == Type::TypeCategory::catBoundMethod) { static PyObject* res = PyUnicode_FromString("BoundMethod"); return res; }
    if (cat == Type::TypeCategory::catAlternativeMatcher) { static PyObject* res = PyUnicode_FromString("AlternativeMatcher"); return res; }
    if (cat == Type::TypeCategory::catClass) { static PyObject* res = PyUnicode_FromString("Class"); return res; }
    if (cat == Type::TypeCategory::catHeldClass) { static PyObject* res = PyUnicode_FromString("HeldClass"); return res; }
    if (cat == Type::TypeCategory::catFunction) { static PyObject* res = PyUnicode_FromString("Function"); return res; }
    if (cat == Type::TypeCategory::catForward) { static PyObject* res = PyUnicode_FromString("Forward"); return res; }
    if (cat == Type::TypeCategory::catEmbeddedMessage) { static PyObject* res = PyUnicode_FromString("EmbeddedMessage"); return res; }
    if (cat == Type::TypeCategory::catPyCell) { static PyObject* res = PyUnicode_FromString("PyCell"); return res; }
    if (cat == Type::TypeCategory::catTypedCell) { static PyObject* res = PyUnicode_FromString("TypedCell"); return res; }
    if (cat == Type::TypeCategory::catPythonObjectOfType) { static PyObject* res = PyUnicode_FromString("PythonObjectOfType"); return res; }
    if (cat == Type::TypeCategory::catSubclassOf) { static PyObject* res = PyUnicode_FromString("SubclassOf"); return res; }

    static PyObject* res = PyUnicode_FromString("Unknown");
    return res;
}

// static
Instance PyInstance::unwrapPyObjectToInstance(PyObject* inst, bool allowArbitraryPyObjects) {
    if (inst == Py_None) {
        return Instance();
    }
    if (PyBool_Check(inst)) {
        return Instance::create(inst == Py_True);
    }
    if (PyLong_Check(inst)) {
        try {
            return Instance::create((int64_t)PyLong_AsLongLong(inst));
        }
        catch(...) {
            return Instance::create((uint64_t)PyLong_AsUnsignedLongLong(inst));
        }
    }
    if (PyFloat_Check(inst)) {
        return Instance::create(PyFloat_AsDouble(inst));
    }
    if (PyBytes_Check(inst)) {
        return Instance::createAndInitialize(
            BytesType::Make(),
            [&](instance_ptr i) {
                BytesType::Make()->constructor(i, PyBytes_GET_SIZE(inst), PyBytes_AsString(inst));
            }
        );
    }
    if (PyUnicode_Check(inst)) {
        auto kind = PyUnicode_KIND(inst);
        assert(
            kind == PyUnicode_1BYTE_KIND ||
            kind == PyUnicode_2BYTE_KIND ||
            kind == PyUnicode_4BYTE_KIND
            );
        int64_t bytesPerCodepoint =
            kind == PyUnicode_1BYTE_KIND ? 1 :
            kind == PyUnicode_2BYTE_KIND ? 2 :
                                           4 ;

        int64_t count = PyUnicode_GET_LENGTH(inst);

        const char* data =
            kind == PyUnicode_1BYTE_KIND ? (char*)PyUnicode_1BYTE_DATA(inst) :
            kind == PyUnicode_2BYTE_KIND ? (char*)PyUnicode_2BYTE_DATA(inst) :
                                           (char*)PyUnicode_4BYTE_DATA(inst);

        return Instance::createAndInitialize(
            StringType::Make(),
            [&](instance_ptr i) {
                StringType::Make()->constructor(i, bytesPerCodepoint, count, data);
            }
        );
    }
    if (PyType_Check(inst)) {
        return Instance::createAndInitialize(
            PythonObjectOfType::AnyPyType(),
            [&](instance_ptr i) {
                PythonObjectOfType::AnyPyType()->initializeFromPyObject(i, inst);
            }
        );
    }

    if (allowArbitraryPyObjects) {
        return Instance::createAndInitialize(
            PythonObjectOfType::AnyPyObject(),
            [&](instance_ptr i) {
                PythonObjectOfType::AnyPyObject()->initializeFromPyObject(i, inst);
            }
        );
    }

    assert(!PyErr_Occurred());

    PyErr_Format(
        PyExc_TypeError,
        "Cannot convert %S to an Instance in this context. Try wrapping it in Value to be more explicit.",
        inst
    );

    return Instance();  // when failed, return a None instance
}

// static
Type* PyInstance::tryUnwrapPyInstanceToValueType(PyObject* typearg, bool allowArbitraryPyObjects) {
    if (typearg == Py_None) {
        return NoneType::Make();
    }

    Type* nativeType = PyInstance::extractTypeFrom(typearg->ob_type);

    if (nativeType) {
        if (nativeType->getTypeCategory() == Type::TypeCategory::catClass) {
            return nullptr;
        }

        return Value::Make(
            Instance::create(
                nativeType,
                ((PyInstance*)typearg)->dataPtr()
            )
        );
    }

    Instance inst = unwrapPyObjectToInstance(typearg, allowArbitraryPyObjects);

    if (!PyErr_Occurred()) {
        return Value::Make(inst);
    }

    PyErr_Clear();
    return nullptr;
}

// static
Type* PyInstance::tryUnwrapPyInstanceToType(PyObject* arg) {
    if (PyType_Check(arg)) {
        Type* possibleType = PyInstance::unwrapTypeArgToTypePtr(arg);
        if (!possibleType) {
            return NULL;
        }
        return possibleType;
    }

    if (PyDict_Contains(nonTypesAcceptedAsTypes(), arg)) {
        PyObject* nonType = PyDict_GetItem(nonTypesAcceptedAsTypes(), arg);
        if (!nonType) {
            return nullptr;
        }

        return PythonObjectOfType::Make(
            (PyTypeObject*)nonType,
            arg
        );
    }

    if (arg == Py_None) {
        return NoneType::Make();
    }

    return  PyInstance::tryUnwrapPyInstanceToValueType(arg, false);
}

// static
Type* PyInstance::unwrapTypeArgToTypePtr(PyObject* typearg) {
    if (PyType_Check(typearg)) {
        PyTypeObject* pyType = (PyTypeObject*)typearg;

        if (pyType == &PyLong_Type) {
            return Int64::Make();
        }
        if (pyType == &PyFloat_Type) {
            return Float64::Make();
        }
        if (pyType == Py_None->ob_type) {
            return NoneType::Make();
        }
        if (pyType == &PyBool_Type) {
            return Bool::Make();
        }
        if (pyType == &PyBytes_Type) {
            return BytesType::Make();
        }
        if (pyType == &PyUnicode_Type) {
            return StringType::Make();
        }

        if (PyInstance::isSubclassOfNativeType(pyType)) {
            Type* nativeT = PyInstance::rootNativeType(pyType);

            if (!nativeT) {
                PyErr_SetString(PyExc_TypeError,
                    ("Type " + std::string(pyType->tp_name) + " looked like a native type subclass, but has no base").c_str()
                    );
                return NULL;
            }

            //this is now a permanent object
            incref(typearg);

            return PythonSubclass::Make(nativeT, pyType);
        } else {
            Type* res = PyInstance::extractTypeFrom(pyType);

            if (res) {
                // we have a native type -> return it
                return res;
            } else {
                // we have a python type -> wrap it
                return PythonObjectOfType::Make(pyType);
            }
        }
    }

    if (PyDict_Contains(nonTypesAcceptedAsTypes(), typearg)) {
        PyObject* nonType = PyDict_GetItem(nonTypesAcceptedAsTypes(), typearg);
        if (!nonType) {
            return nullptr;
        }

        return PythonObjectOfType::Make(
            (PyTypeObject*)nonType,
            typearg
        );
    }

    // else: typearg is not a type -> it is a value
    Type* valueType = PyInstance::tryUnwrapPyInstanceToValueType(typearg, false);

    if (valueType) {
        return valueType;
    }

    PyErr_Format(PyExc_TypeError, "Cannot convert %R to a typed_python Type", typearg);
    return NULL;
}
