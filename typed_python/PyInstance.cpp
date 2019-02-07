#include <Python.h>
#include <numpy/arrayobject.h>
#include <type_traits>

#include "AllTypes.hpp"
#include "_runtime.h"
#include "PyInstance.hpp"
#include "PyConstDictInstance.hpp"
#include "PyTupleOrListOfInstance.hpp"
#include "PyPointerToInstance.hpp"
#include "PyCompositeTypeInstance.hpp"
#include "PyClassInstance.hpp"
#include "PyHeldClassInstance.hpp"
#include "PyBoundMethodInstance.hpp"
#include "PyAlternativeInstance.hpp"
#include "PyFunctionInstance.hpp"
#include "PyStringInstance.hpp"
#include "PyBytesInstance.hpp"
#include "PyNoneInstance.hpp"
#include "PyRegisterTypeInstance.hpp"
#include "PyValueInstance.hpp"
#include "PyValueInstance.hpp"
#include "PyPythonSubclassInstance.hpp"
#include "PyPythonObjectOfTypeInstance.hpp"
#include "PyOneOfInstance.hpp"
#include "PyForwardInstance.hpp"

// static
bool PyInstance::guaranteeForwardsResolved(Type* t) {
    try {
        guaranteeForwardsResolvedOrThrow(t);
        return true;
    } catch(PythonExceptionSet& e) {
        return false;
    } catch(std::exception& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return false;
    }
}

// static
void PyInstance::guaranteeForwardsResolvedOrThrow(Type* t) {
    t->guaranteeForwardsResolved([&](PyObject* o) {
        PyObject* result = PyObject_CallFunctionObjArgs(o, NULL);
        if (!result) {
            PyErr_Clear();
            throw std::runtime_error("Python type callback threw an exception.");
        }

        if (!PyType_Check(result)) {
            Py_DECREF(result);
            throw std::runtime_error("Python type callback didn't return a type: got " +
                std::string(result->ob_type->tp_name));
        }

        Type* resType = unwrapTypeArgToTypePtr(result);
        Py_DECREF(result);

        if (!resType) {
            throw std::runtime_error("Python type callback didn't return a native type: got " +
                std::string(result->ob_type->tp_name));
        }

        return resType;
    });
}

Type* PyInstance::type() {
    return extractTypeFrom(((PyObject*)this)->ob_type);
}

instance_ptr PyInstance::dataPtr() {
    return mContainingInstance.data();
}

//static
PyObject* PyInstance::undefinedBehaviorException() {
    static PyObject* module = PyImport_ImportModule("typed_python.internals");
    static PyObject* t = PyObject_GetAttrString(module, "UndefinedBehaviorException");
    return t;
}

// static
PyMethodDef* PyInstance::typeMethods(Type* t) {
    return specializeStatic(t->getTypeCategory(), [&](auto* concrete_null_ptr) {
        typedef typename std::remove_reference<decltype(*concrete_null_ptr)>::type py_instance_type;

        return py_instance_type::typeMethodsConcrete();
    });
}

PyMethodDef* PyInstance::typeMethodsConcrete() {
    return new PyMethodDef [2] {
        {NULL, NULL}
    };
}

// static
void PyInstance::tp_dealloc(PyObject* self) {
    PyInstance* wrapper = (PyInstance*)self;

    if (wrapper->mIsInitialized) {
        wrapper->mContainingInstance.~Instance();
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}

// static
bool PyInstance::pyValCouldBeOfType(Type* t, PyObject* pyRepresentation) {
    guaranteeForwardsResolvedOrThrow(t);

    Type* argType = extractTypeFrom(pyRepresentation->ob_type);

    if (argType && argType->isBinaryCompatibleWith(argType)) {
        return true;
    }

    return specializeStatic(t->getTypeCategory(), [&](auto* concrete_null_ptr) {
        typedef typename std::remove_reference<decltype(*concrete_null_ptr)>::type py_instance_type;

        return py_instance_type::pyValCouldBeOfTypeConcrete(
            (typename py_instance_type::modeled_type*)t,
            pyRepresentation
            );
    });
}

// static
void PyInstance::copyConstructFromPythonInstance(Type* eltType, instance_ptr tgt, PyObject* pyRepresentation, bool isExplicit) {
    guaranteeForwardsResolvedOrThrow(eltType);

    Type* argType = extractTypeFrom(pyRepresentation->ob_type);

    if (argType && argType->isBinaryCompatibleWith(eltType)) {
        //it's already the right kind of instance
        eltType->copy_constructor(tgt, ((PyInstance*)pyRepresentation)->dataPtr());
        return;
    }

    Type::TypeCategory cat = eltType->getTypeCategory();

    //dispatch to the appropriate Py[]Instance type
    specializeStatic(cat, [&](auto* concrete_null_ptr) {
        typedef typename std::remove_reference<decltype(*concrete_null_ptr)>::type py_instance_type;

        py_instance_type::copyConstructFromPythonInstanceConcrete(
            (typename py_instance_type::modeled_type*)eltType,
            tgt,
            pyRepresentation,
            isExplicit
            );
    });
}

void PyInstance::copyConstructFromPythonInstanceConcrete(Type* eltType, instance_ptr tgt, PyObject* pyRepresentation, bool isExplicit) {
    throw std::logic_error("Couldn't initialize type " + eltType->name() + " from " + pyRepresentation->ob_type->tp_name);
}


// static
void PyInstance::constructFromPythonArguments(uint8_t* data, Type* t, PyObject* args, PyObject* kwargs) {
    guaranteeForwardsResolvedOrThrow(t);

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
    if (kwargs == NULL && (args == NULL || PyTuple_Size(args) == 0)) {
        if (t->is_default_constructible()) {
            t->constructor(data);
            return;
        }
    }

    if (kwargs == NULL && PyTuple_Size(args) == 1) {
        PyObject* argTuple = PyTuple_GetItem(args, 0);

        copyConstructFromPythonInstance(t, data, argTuple, true /* mark isExplicit */);

        return;
    }

    throw std::logic_error("Can't initialize " + t->name() + " with this signature.");
}

/**
 * produce the pythonic representation of this object. for values that have a direct python representation,
 * such as integers, strings, bools, or None, we return an actual python object. Otherwise,
 * we return a pointer to a PyInstance representing the object.
 */
// static
PyObject* PyInstance::extractPythonObject(instance_ptr data, Type* eltType) {
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
        return NULL;
    }

    try {
        Type* concreteT = eltType->pickConcreteSubclass(data);

        return PyInstance::initialize(concreteT, [&](instance_ptr selfData) {
            concreteT->copy_constructor(selfData, data);
        });
    } catch(PythonExceptionSet& e) {
        return NULL;
    } catch(std::exception& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    }
}

PyObject* PyInstance::extractPythonObjectConcrete(Type* eltType, instance_ptr data) {
    return NULL;
}

// static
PyObject* PyInstance::tp_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds) {
    Type* eltType = extractTypeFrom(subtype);

    if (!guaranteeForwardsResolved(eltType)) {
        return nullptr;
    }

    if (isSubclassOfNativeType(subtype)) {
        PyInstance* self = (PyInstance*)subtype->tp_alloc(subtype, 0);

        try {
            self->mIteratorOffset = -1;
            self->mIsMatcher = false;

            self->initialize([&](instance_ptr data) {
                constructFromPythonArguments(data, eltType, args, kwds);
            });

            return (PyObject*)self;
        } catch(PythonExceptionSet& e) {
            subtype->tp_dealloc((PyObject*)self);
            return NULL;
        } catch(std::exception& e) {
            subtype->tp_dealloc((PyObject*)self);

            PyErr_SetString(PyExc_TypeError, e.what());
            return NULL;
        }

        // not reachable
        assert(false);

    } else {
        instance_ptr tgt = (instance_ptr)malloc(eltType->bytecount());

        try {
            constructFromPythonArguments(tgt, eltType, args, kwds);
        } catch(std::exception& e) {
            free(tgt);
            PyErr_SetString(PyExc_TypeError, e.what());
            return NULL;
        } catch(PythonExceptionSet& e) {
            free(tgt);
            return NULL;
        }

        PyObject* result = extractPythonObject(tgt, eltType);

        eltType->destroy(tgt);
        free(tgt);

        return result;
    }
}

PyObject* PyInstance::pyUnaryOperator(PyObject* lhs, const char* op, const char* opErrRep) {
    return specializeForType(lhs, [&](auto& subtype) {
        return subtype.pyUnaryOperatorConcrete(op, opErrRep);
    });
}

PyObject* PyInstance::pyOperator(PyObject* lhs, PyObject* rhs, const char* op, const char* opErrRep) {
    if (extractTypeFrom(lhs->ob_type)) {
        return specializeForType(lhs, [&](auto& subtype) {
            return subtype.pyOperatorConcrete(rhs, op, opErrRep);
        });
    }

    if (extractTypeFrom(rhs->ob_type)) {
        return specializeForType(rhs, [&](auto& subtype) {
            return subtype.pyOperatorConcreteReverse(lhs, op, opErrRep);
        });
    }

    PyErr_Format(PyExc_TypeError, "Invalid type arguments of type '%S' and '%S' to binary operator %s",
        lhs->ob_type,
        rhs->ob_type,
        op
        );

    return NULL;
}

PyObject* PyInstance::pyTernaryOperator(PyObject* lhs, PyObject* rhs, PyObject* thirdArg, const char* op, const char* opErrRep) {
    if (extractTypeFrom(lhs->ob_type)) {
        return specializeForType(lhs, [&](auto& subtype) {
            return subtype.pyTernaryOperatorConcrete(rhs, thirdArg, op, opErrRep);
        });
    }

    PyErr_Format(PyExc_TypeError, "Invalid type arguments of type '%S' and '%S' to binary operator %s",
        lhs->ob_type,
        rhs->ob_type,
        op
        );

    return NULL;
}

PyObject* PyInstance::pyUnaryOperatorConcrete(const char* op, const char* opErrRep) {
    return incref(Py_NotImplemented);
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
    return pyOperator(lhs, rhs, "__div__", ".");
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
    return pyUnaryOperator(lhs, "__abs__", "+");
}

// static
PyObject* PyInstance::nb_invert(PyObject* lhs) {
    return pyUnaryOperator(lhs, "__invert__", "~");
}

// static
PyObject* PyInstance::nb_int(PyObject* lhs) {
    return pyUnaryOperator(lhs, "__int__", "+");
}

// static
PyObject* PyInstance::nb_float(PyObject* lhs) {
    return pyUnaryOperator(lhs, "__float__", "+");
}

// static
PyObject* PyInstance::nb_index(PyObject* lhs) {
    return pyUnaryOperator(lhs, "__index__", "+");
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
PyTypeObject* PyInstance::typeObj(Type* inType) {
    if (!inType->getTypeRep()) {
        inType->setTypeRep(typeObjInternal(inType));
    }

    return inType->getTypeRep();
}

// static
PySequenceMethods* PyInstance::sequenceMethodsFor(Type* t) {
    if (    t->getTypeCategory() == Type::TypeCategory::catTupleOf ||
            t->getTypeCategory() == Type::TypeCategory::catListOf ||
            t->getTypeCategory() == Type::TypeCategory::catTuple ||
            t->getTypeCategory() == Type::TypeCategory::catNamedTuple ||
            t->getTypeCategory() == Type::TypeCategory::catString ||
            t->getTypeCategory() == Type::TypeCategory::catBytes ||
            t->getTypeCategory() == Type::TypeCategory::catConstDict) {
        PySequenceMethods* res =
            new PySequenceMethods {0,0,0,0,0,0,0,0};

        if (t->getTypeCategory() == Type::TypeCategory::catConstDict) {
            res->sq_contains = (objobjproc)PyInstance::sq_contains;
        } else {
            res->sq_length = (lenfunc)PyInstance::mp_and_sq_length;
            res->sq_item = (ssizeargfunc)PyInstance::sq_item;
        }

        return res;
    }
    if (    t->getTypeCategory() == Type::TypeCategory::catPointerTo) {
        PySequenceMethods* res =
            new PySequenceMethods {0,0,0,0,0,0,0,0};

        res->sq_item = (ssizeargfunc)PyInstance::sq_item;

        return res;
    }

    return 0;
}

// static
PyNumberMethods* PyInstance::numberMethods(Type* t) {
    return new PyNumberMethods {
            //only enable this for the types that it operates on. Otherwise it disables the concatenation functions
            //we should probably just unify them
            nb_add, //binaryfunc nb_add
            nb_subtract, //binaryfunc nb_subtract
            nb_multiply, //binaryfunc nb_multiply
            nb_remainder, //binaryfunc nb_remainder
            nb_divmod, //binaryfunc nb_divmod
            nb_power, //ternaryfunc nb_power
            nb_negative, //unaryfunc nb_negative
            nb_positive, //unaryfunc nb_positive
            nb_absolute, //unaryfunc nb_absolute
            0, //inquiry nb_bool
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
    static PyMappingMethods* res =
        new PyMappingMethods {
            PyInstance::mp_and_sq_length, //mp_length
            PyInstance::mp_subscript, //mp_subscript
            PyInstance::mp_ass_subscript //mp_ass_subscript
            };

    if (t->getTypeCategory() == Type::TypeCategory::catConstDict ||
        t->getTypeCategory() == Type::TypeCategory::catTupleOf ||
        t->getTypeCategory() == Type::TypeCategory::catListOf ||
        t->getTypeCategory() == Type::TypeCategory::catClass) {
        return res;
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
inline bool PyInstance::isNativeType(PyTypeObject* typeObj) {
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

// static
Type* PyInstance::extractTypeFrom(PyTypeObject* typeObj, bool exact /*=false*/) {
    if (exact && isSubclassOfNativeType(typeObj)) {
        return PythonSubclass::Make(extractTypeFrom(typeObj), typeObj);
    }

    while (!exact && typeObj->tp_base && !isNativeType(typeObj)) {
        typeObj = typeObj->tp_base;
    }

    if (isNativeType(typeObj)) {
        return ((NativeTypeWrapper*)typeObj)->mType;
    } else {
        return nullptr;
    }

}

PyTypeObject* PyInstance::typeObjInternal(Type* inType) {
    static std::recursive_mutex mutex;
    static std::map<Type*, NativeTypeWrapper*> types;

    std::lock_guard<std::recursive_mutex> lock(mutex);

    auto it = types.find(inType);
    if (it != types.end()) {
        return (PyTypeObject*)it->second;
    }

    types[inType] = new NativeTypeWrapper { {
            PyVarObject_HEAD_INIT(NULL, 0)              // TYPE (c.f., Type Objects)
            .tp_name = (new std::string(inType->name()))->c_str(),          // const char*
            .tp_basicsize = sizeof(PyInstance),         // Py_ssize_t
            .tp_itemsize = 0,                           // Py_ssize_t
            .tp_dealloc = PyInstance::tp_dealloc,       // destructor
            .tp_print = 0,                              // printfunc
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
            .tp_doc = 0,                                // const char*
            .tp_traverse = 0,                           // traverseproc
            .tp_clear = 0,                              // inquiry
            .tp_richcompare = tp_richcompare,           // richcmpfunc
            .tp_weaklistoffset = 0,                     // Py_ssize_t
            .tp_iter = inType->getTypeCategory() == Type::TypeCategory::catConstDict ?
                PyInstance::tp_iter
            :   0,                                      // getiterfunc tp_iter;
            .tp_iternext = PyInstance::tp_iternext,// iternextfunc
            .tp_methods = typeMethods(inType),          // struct PyMethodDef*
            .tp_members = 0,                            // struct PyMemberDef*
            .tp_getset = 0,                             // struct PyGetSetDef*
            .tp_base = 0,                               // struct _typeobject*
            .tp_dict = PyDict_New(),                    // PyObject*
            .tp_descr_get = 0,                          // descrgetfunc
            .tp_descr_set = 0,                          // descrsetfunc
            .tp_dictoffset = 0,                         // Py_ssize_t
            .tp_init = 0,                               // initproc
            .tp_alloc = 0,                              // allocfunc
            .tp_new = PyInstance::tp_new,  // newfunc
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
            }, inType
            };

    // at this point, the dictionary has an entry, so if we recurse back to this function
    // we will return the correct entry.
    if (inType->getBaseType()) {
        types[inType]->typeObj.tp_base = typeObjInternal((Type*)inType->getBaseType());
        Py_INCREF(types[inType]->typeObj.tp_base);
    }

    PyType_Ready((PyTypeObject*)types[inType]);

    PyDict_SetItemString(
        types[inType]->typeObj.tp_dict,
        "__typed_python_category__",
        categoryToPyString(inType->getTypeCategory())
        );

    PyDict_SetItemString(
        types[inType]->typeObj.tp_dict,
        "__typed_python_basetype__",
        inType->getBaseType() ?
            (PyObject*)typeObjInternal(inType->getBaseType())
        :   Py_None
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

    Type* type = extractTypeFrom(o->ob_type);
    Type::TypeCategory cat = type->getTypeCategory();

    if (cat == Type::TypeCategory::catClass) {
        PyInstance* self_w = (PyInstance*)o;

        return PyClassInstance::classInstanceSetAttributeFromPyObject((Class*)type, self_w->dataPtr(), attrName, attrVal);
    } else if (cat == Type::TypeCategory::catNamedTuple ||
               cat == Type::TypeCategory::catConcreteAlternative) {
        PyErr_Format(
            PyExc_AttributeError,
            "Cannot set attributes on instance of type '%S' because it is immutable",
            o->ob_type
        );
        return -1;
    } else {
        PyErr_Format(
            PyExc_AttributeError,
            "Instances of type '%S' do not accept attributes",
            attrName, o->ob_type
        );
        return -1;
    }
}
// static
PyObject* PyInstance::tp_call(PyObject* o, PyObject* args, PyObject* kwargs) {
    return specializeForType(o, [&](auto& subtype) {
        return subtype.tp_call_concrete(args, kwargs);
    });
}

PyObject* PyInstance::tp_call_concrete(PyObject* args, PyObject* kwargs) {
    PyErr_Format(PyExc_TypeError, "'%s' object is not callable", type()->name().c_str());
    return 0;
}

// static
PyObject* PyInstance::tp_getattro(PyObject *o, PyObject* attrName) {
    if (!PyUnicode_Check(attrName)) {
        PyErr_SetString(PyExc_AttributeError, "attribute is not a string");
        return NULL;
    }

    char *attr_name = PyUnicode_AsUTF8(attrName);

    Type* t = extractTypeFrom(o->ob_type);

    PyInstance* w = (PyInstance*)o;

    Type::TypeCategory cat = t->getTypeCategory();

    if (w->mIsMatcher) {
        PyObject* res;

        if (cat == Type::TypeCategory::catAlternative) {
            Alternative* a = (Alternative*)t;
            if (a->subtypes()[a->which(w->dataPtr())].first == attr_name) {
                res = Py_True;
            } else {
                res = Py_False;
            }
        } else {
            ConcreteAlternative* a = (ConcreteAlternative*)t;
            if (a->getAlternative()->subtypes()[a->which()].first == attr_name) {
                res = Py_True;
            } else {
                res = Py_False;
            }
        }

        Py_INCREF(res);
        return res;
    }

    if (cat == Type::TypeCategory::catAlternative ||
            cat == Type::TypeCategory::catConcreteAlternative) {
        if (strcmp(attr_name,"matches") == 0) {
            PyInstance* self = (PyInstance*)o->ob_type->tp_alloc(o->ob_type, 0);

            self->mIteratorOffset = -1;
            self->mIsMatcher = true;

            self->initialize([&](instance_ptr data) {
                t->copy_constructor(data, w->dataPtr());
            });

            return (PyObject*)self;
        }

        //see if its a method
        Alternative* toCheck =
            (Alternative*)(cat == Type::TypeCategory::catConcreteAlternative ? t->getBaseType() : t)
            ;

        auto it = toCheck->getMethods().find(attr_name);
        if (it != toCheck->getMethods().end()) {
            return PyMethod_New((PyObject*)it->second->getOverloads()[0].getFunctionObj(), o);
        }
    }

    if (t->getTypeCategory() == Type::TypeCategory::catClass) {
        Class* nt = (Class*)t;

        for (long k = 0; k < nt->getMembers().size();k++) {
            if (nt->getMemberName(k) == attr_name) {
                Type* eltType = nt->getMemberType(k);

                if (!nt->checkInitializationFlag(w->dataPtr(),k)) {
                    PyErr_Format(
                        PyExc_AttributeError,
                        "Attribute '%S' is not initialized",
                        attrName
                    );
                    return NULL;
                }

                return extractPythonObject(
                    nt->eltPtr(w->dataPtr(), k),
                    eltType
                    );
            }
        }

        {
            auto it = nt->getPropertyFunctions().find(attr_name);
            if (it != nt->getPropertyFunctions().end()) {
                std::pair<bool, PyObject*> res = PyFunctionInstance::tryToCall(it->second, o);
                if (res.first) {
                    return res.second;
                }

                PyErr_Format(
                    PyExc_TypeError,
                    "Found a property for %s but failed to call it with 'self'",
                    attr_name
                    );
                return NULL;
            }
        }

        {
            auto it = nt->getMemberFunctions().find(attr_name);
            if (it != nt->getMemberFunctions().end()) {
                BoundMethod* bm = BoundMethod::Make(nt, it->second);

                return PyInstance::initializePythonRepresentation(bm, [&](instance_ptr data) {
                    bm->copy_constructor(data, w->dataPtr());
                });
            }
        }

        {
            auto it = nt->getClassMembers().find(attr_name);
            if (it != nt->getClassMembers().end()) {
                PyObject* res = it->second;
                Py_INCREF(res);
                return res;
            }
        }
    }

    PyObject* result = getattr(t, w->dataPtr(), attr_name);

    if (result) {
        return result;
    }

    return PyObject_GenericGetAttr(o, attrName);
}

// static
PyObject* PyInstance::getattr(Type* type, instance_ptr data, char* attr_name) {
    if (type->getTypeCategory() == Type::TypeCategory::catConcreteAlternative) {
        ConcreteAlternative* t = (ConcreteAlternative*)type;

        return getattr(
            t->getAlternative()->subtypes()[t->which()].second,
            t->getAlternative()->eltPtr(data),
            attr_name
            );
    }
    if (type->getTypeCategory() == Type::TypeCategory::catAlternative) {
        Alternative* t = (Alternative*)type;

        return getattr(
            t->subtypes()[t->which(data)].second,
            t->eltPtr(data),
            attr_name
            );
    }

    if (type->getTypeCategory() == Type::TypeCategory::catNamedTuple) {
        NamedTuple* nt = (NamedTuple*)type;
        for (long k = 0; k < nt->getNames().size();k++) {
            if (nt->getNames()[k] == attr_name) {
                return extractPythonObject(
                    nt->eltPtr(data, k),
                    nt->getTypes()[k]
                    );
            }
        }
    }

    return NULL;
}

// static
Py_hash_t PyInstance::tp_hash(PyObject *o) {
    Type* self_type = extractTypeFrom(o->ob_type);
    PyInstance* w = (PyInstance*)o;

    int32_t h = self_type->hash32(w->dataPtr());
    if (h == -1) {
        h = -2;
    }

    return h;
}

// static
bool PyInstance::compare_to_python(Type* t, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp) {
    if (t->getTypeCategory() == Type::TypeCategory::catValue) {
        Value* valType = (Value*)t;
        return compare_to_python(valType->value().type(), valType->value().data(), other, exact, pyComparisonOp);
    }

    auto convert = [&](char cmpValue) { return cmpResultToBoolForPyOrdering(pyComparisonOp, cmpValue); };

    Type* otherT = extractTypeFrom(other->ob_type);

    if (otherT) {
        if (otherT < t) {
            return convert(1);
        }
        if (otherT > t) {
            return convert(-1);
        }
        return t->cmp(self, ((PyInstance*)other)->dataPtr(), pyComparisonOp);
    }

    if (t->getTypeCategory() == Type::TypeCategory::catOneOf) {
        std::pair<Type*, instance_ptr> child = ((OneOf*)t)->unwrap(self);
        return compare_to_python(child.first, child.second, other, exact, pyComparisonOp);
    }

    if (other == Py_None) {
        return convert((t->getTypeCategory() == Type::TypeCategory::catNone ? 0 : 1));
    }

    if (PyBool_Check(other)) {
        int64_t other_l = other == Py_True ? 1 : 0;
        int64_t self_l;

        if (t->getTypeCategory() == Type::TypeCategory::catBool) {
            self_l = (*(bool*)self) ? 1 : 0;
        } else {
            return convert(-1);
        }

        if (other_l < self_l) { return convert(-1); }
        if (other_l > self_l) { return convert(1); }
        return convert(0);
    }

    if (PyLong_Check(other)) {
        int64_t other_l = PyLong_AsLong(other);
        int64_t self_l;

        if (t->getTypeCategory() == Type::TypeCategory::catInt64) {
            self_l = (*(int64_t*)self);
        } else if (t->getTypeCategory() == Type::TypeCategory::catInt32) {
            self_l = (*(int32_t*)self);
        } else if (t->getTypeCategory() == Type::TypeCategory::catInt16) {
            self_l = (*(int16_t*)self);
        } else if (t->getTypeCategory() == Type::TypeCategory::catInt8) {
            self_l = (*(int8_t*)self);
        } else if (t->getTypeCategory() == Type::TypeCategory::catBool) {
            self_l = (*(bool*)self) ? 1 : 0;
        } else if (t->getTypeCategory() == Type::TypeCategory::catUInt64) {
            self_l = (*(uint64_t*)self);
        } else if (t->getTypeCategory() == Type::TypeCategory::catUInt32) {
            self_l = (*(uint32_t*)self);
        } else if (t->getTypeCategory() == Type::TypeCategory::catUInt16) {
            self_l = (*(uint16_t*)self);
        } else if (t->getTypeCategory() == Type::TypeCategory::catUInt8) {
            self_l = (*(uint8_t*)self);
        } else if (t->getTypeCategory() == Type::TypeCategory::catFloat32) {
            if (exact) {
                return convert(-1);
            }
            if (other_l < *(float*)self) { return convert(-1); }
            if (other_l > *(float*)self) { return convert(1); }
            return convert(0);
        } else if (t->getTypeCategory() == Type::TypeCategory::catFloat64) {
            if (exact) {
                return convert(-1);
            }
            if (other_l < *(double*)self) { return convert(-1); }
            if (other_l > *(double*)self) { return convert(1); }
            return convert(0);
        } else {
            return convert(-1);
        }

        if (other_l < self_l) { return convert(-1); }
        if (other_l > self_l) { return convert(1); }
        return convert(0);
    }

    if (PyFloat_Check(other)) {
        double other_d = PyFloat_AsDouble(other);
        double self_d;

        if (t->getTypeCategory() == Type::TypeCategory::catFloat32) {
            self_d = (*(float*)self);
        } else if (t->getTypeCategory() == Type::TypeCategory::catFloat64) {
            self_d = (*(double*)self);
        } else {
            if (exact) {
                return convert(-1);
            }
            if (t->getTypeCategory() == Type::TypeCategory::catInt64) {
                self_d = (*(int64_t*)self);
            } else if (t->getTypeCategory() == Type::TypeCategory::catInt32) {
                self_d = (*(int32_t*)self);
            } else if (t->getTypeCategory() == Type::TypeCategory::catInt16) {
                self_d = (*(int16_t*)self);
            } else if (t->getTypeCategory() == Type::TypeCategory::catInt8) {
                self_d = (*(int8_t*)self);
            } else if (t->getTypeCategory() == Type::TypeCategory::catBool) {
                self_d = (*(bool*)self) ? 1 : 0;
            } else if (t->getTypeCategory() == Type::TypeCategory::catUInt64) {
                self_d = (*(uint64_t*)self);
            } else if (t->getTypeCategory() == Type::TypeCategory::catUInt32) {
                self_d = (*(uint32_t*)self);
            } else if (t->getTypeCategory() == Type::TypeCategory::catUInt16) {
                self_d = (*(uint16_t*)self);
            } else if (t->getTypeCategory() == Type::TypeCategory::catUInt8) {
                self_d = (*(uint8_t*)self);
            } else {
                return convert(-1);
            }
        }

        if (other_d < self_d) { return convert(-1); }
        if (other_d > self_d) { return convert(1); }
        return convert(0);
    }

    if (PyTuple_Check(other)) {
        if (t->getTypeCategory() == Type::TypeCategory::catTupleOf) {
            TupleOf* tupT = (TupleOf*)t;
            int lenO = PyTuple_Size(other);
            int lenS = tupT->count(self);

            for (long k = 0; k < lenO && k < lenS; k++) {
                if (!compare_to_python(tupT->getEltType(), tupT->eltPtr(self, k), PyTuple_GetItem(other,k), exact, Py_EQ)) {
                    if (compare_to_python(tupT->getEltType(), tupT->eltPtr(self, k), PyTuple_GetItem(other,k), exact, Py_LT)) {
                        return convert(-1);
                    }
                    return convert(1);
                }
            }

            if (lenS < lenO) { return convert(-1); }
            if (lenS > lenO) { return convert(1); }
            return convert(0);
        }
        if (t->getTypeCategory() == Type::TypeCategory::catTuple ||
                    t->getTypeCategory() == Type::TypeCategory::catNamedTuple) {
            CompositeType* tupT = (CompositeType*)t;
            int lenO = PyTuple_Size(other);
            int lenS = tupT->getTypes().size();

            for (long k = 0; k < lenO && k < lenS; k++) {
                if (!compare_to_python(tupT->getTypes()[k], tupT->eltPtr(self, k), PyTuple_GetItem(other,k), exact, Py_EQ)) {
                    if (compare_to_python(tupT->getTypes()[k], tupT->eltPtr(self, k), PyTuple_GetItem(other,k), exact, Py_LT)) {
                        return convert(-1);
                    }
                    return convert(1);
                }
            }

            if (lenS < lenO) { return convert(-1); }
            if (lenS > lenO) { return convert(1); }

            return convert(0);
        }
    }

    if (PyList_Check(other)) {
        if (t->getTypeCategory() == Type::TypeCategory::catListOf) {
            ListOf* listT = (ListOf*)t;
            int lenO = PyList_Size(other);
            int lenS = listT->count(self);
            for (long k = 0; k < lenO && k < lenS; k++) {
                if (!compare_to_python(listT->getEltType(), listT->eltPtr(self, k), PyList_GetItem(other,k), exact, Py_EQ)) {
                    if (compare_to_python(listT->getEltType(), listT->eltPtr(self, k), PyList_GetItem(other,k), exact, Py_LT)) {
                        return convert(-1);
                    }
                    return convert(1);
                }
            }

            if (lenS < lenO) { return convert(-1); }
            if (lenS > lenO) { return convert(1); }
            return convert(0);
        }
    }

    if (PyUnicode_Check(other) && t->getTypeCategory() == Type::TypeCategory::catString) {
        auto kind = PyUnicode_KIND(other);
        int bytesPer = kind == PyUnicode_1BYTE_KIND ? 1 :
            kind == PyUnicode_2BYTE_KIND ? 2 : 4;

        if (bytesPer != ((String*)t)->bytes_per_codepoint(self)) {
            return convert(-1);
        }

        if (PyUnicode_GET_LENGTH(other) != ((String*)t)->count(self)) {
            return convert(-1);
        }

        return convert(memcmp(
            kind == PyUnicode_1BYTE_KIND ? (const char*)PyUnicode_1BYTE_DATA(other) :
            kind == PyUnicode_2BYTE_KIND ? (const char*)PyUnicode_2BYTE_DATA(other) :
                                           (const char*)PyUnicode_4BYTE_DATA(other),
            ((String*)t)->eltPtr(self, 0),
            PyUnicode_GET_LENGTH(other) * bytesPer
            ));
    }

    if (PyBytes_Check(other) && t->getTypeCategory() == Type::TypeCategory::catBytes) {
        if (PyBytes_GET_SIZE(other) != ((Bytes*)t)->count(self)) {
            return convert(-1);
        }

        return convert(memcmp(
            PyBytes_AsString(other),
            ((Bytes*)t)->eltPtr(self, 0),
            PyBytes_GET_SIZE(other)
            ));
    }

    return convert(-1);
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
    try {
        Type* own = extractTypeFrom(a->ob_type);
        Type* other = extractTypeFrom(b->ob_type);

        if (!own && !other) {
            PyErr_Format(PyExc_TypeError, "Can't call tp_richcompare where neither object is a typed_python object!");
            return NULL;
        }

        if (!own || !other) {
            bool cmp;

            if (own) {
                cmp = compare_to_python(own, ((PyInstance*)a)->dataPtr(), b, false, op);
            } else {
                cmp = compare_to_python(other, ((PyInstance*)b)->dataPtr(), a, false, reversePyOpOrdering(op));
            }

            return incref(cmp ? Py_True : Py_False);
        } else {
            bool result;

            if (own < other) {
                result = cmpResultToBoolForPyOrdering(op, -1);
            } else if (own > other) {
                result = cmpResultToBoolForPyOrdering(op, 1);
            } else {
                result = own->cmp(((PyInstance*)a)->dataPtr(), ((PyInstance*)b)->dataPtr(), op);
            }

            return incref(result ? Py_True : Py_False);
        }
    } catch(PythonExceptionSet& e) {
        return NULL;
    } catch(std::exception& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    }
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
PyObject* PyInstance::tp_repr(PyObject *o) {
    Type* self_type = extractTypeFrom(o->ob_type);
    PyInstance* w = (PyInstance*)o;

    std::ostringstream str;
    ReprAccumulator accumulator(str);

    str << std::showpoint;

    self_type->repr(w->dataPtr(), accumulator);

    return PyUnicode_FromString(str.str().c_str());
}

// static
PyObject* PyInstance::tp_str(PyObject *o) {
    Type* self_type = extractTypeFrom(o->ob_type);
    PyInstance* self_w = (PyInstance*)o;

    if (self_type->getTypeCategory() == Type::TypeCategory::catConcreteAlternative) {
        self_type = self_type->getBaseType();
    }

    if (self_type->getTypeCategory() == Type::TypeCategory::catAlternative) {
        Alternative* a = (Alternative*)self_type;
        auto it = a->getMethods().find("__str__");
        if (it != a->getMethods().end()) {
            return PyObject_CallFunctionObjArgs(
                (PyObject*)it->second->getOverloads()[0].getFunctionObj(),
                o,
                NULL
                );
        }
    }

    std::ostringstream str;
    ReprAccumulator accumulator(str, true);

    str << std::showpoint;

    self_type->repr(self_w->dataPtr(), accumulator);

    return PyUnicode_FromString(str.str().c_str());
}

// static
bool PyInstance::typeCanBeSubclassed(Type* t) {
    return t->getTypeCategory() == Type::TypeCategory::catNamedTuple;
}

// static
void PyInstance::mirrorTypeInformationIntoPyType(Type* inType, PyTypeObject* pyType) {
    if (inType->getTypeCategory() == Type::TypeCategory::catAlternative) {
        Alternative* alt = (Alternative*)inType;

        PyObject* alternatives = PyTuple_New(alt->subtypes().size());

        for (long k = 0; k < alt->subtypes().size(); k++) {
            ConcreteAlternative* concrete = ConcreteAlternative::Make(alt, k);

            PyDict_SetItemString(
                pyType->tp_dict,
                alt->subtypes()[k].first.c_str(),
                (PyObject*)typeObjInternal(concrete)
                );

            PyTuple_SetItem(alternatives, k, incref((PyObject*)typeObjInternal(concrete)));
        }

        PyDict_SetItemString(
            pyType->tp_dict,
            "__typed_python_alternatives__",
            alternatives
            );

        Py_DECREF(alternatives);
    }

    if (inType->getTypeCategory() == Type::TypeCategory::catConcreteAlternative) {
        ConcreteAlternative* alt = (ConcreteAlternative*)inType;

        PyDict_SetItemString(
            pyType->tp_dict,
            "Alternative",
            (PyObject*)typeObjInternal(alt->getAlternative())
            );

        PyDict_SetItemString(
            pyType->tp_dict,
            "Index",
            PyLong_FromLong(alt->which())
            );

        PyDict_SetItemString(
            pyType->tp_dict,
            "Name",
            PyUnicode_FromString(alt->getAlternative()->subtypes()[alt->which()].first.c_str())
            );

        PyDict_SetItemString(
            pyType->tp_dict,
            "ElementType",
            (PyObject*)typeObjInternal(alt->elementType())
            );
    }


    if (inType->getTypeCategory() == Type::TypeCategory::catBoundMethod) {
        BoundMethod* methodT = (BoundMethod*)inType;

        PyDict_SetItemString(pyType->tp_dict, "FirstArgType", typePtrToPyTypeRepresentation(methodT->getFirstArgType()));
        PyDict_SetItemString(pyType->tp_dict, "Function", typePtrToPyTypeRepresentation(methodT->getFunction()));
    }

    if (inType->getTypeCategory() == Type::TypeCategory::catClass) {
        Class* classT = (Class*)inType;

        PyObject* types = PyTuple_New(classT->getMembers().size());
        for (long k = 0; k < classT->getMembers().size(); k++) {
            PyTuple_SetItem(types, k, incref(typePtrToPyTypeRepresentation(std::get<1>(classT->getMembers()[k]))));
        }

        PyObject* names = PyTuple_New(classT->getMembers().size());
        for (long k = 0; k < classT->getMembers().size(); k++) {
            PyObject* namePtr = PyUnicode_FromString(std::get<0>(classT->getMembers()[k]).c_str());
            PyTuple_SetItem(names, k, namePtr);
        }

        PyObject* defaults = PyDict_New();
        for (long k = 0; k < classT->getMembers().size(); k++) {

            if (classT->getHeldClass()->memberHasDefaultValue(k)) {
                const Instance& i = classT->getHeldClass()->getMemberDefaultValue(k);

                PyObject* defaultVal = PyInstance::extractPythonObject(i.data(), i.type());

                PyDict_SetItemString(
                    defaults,
                    classT->getHeldClass()->getMemberName(k).c_str(),
                    defaultVal
                    );

                Py_DECREF(defaultVal);
            }
        }

        PyObject* memberFunctions = PyDict_New();
        for (auto p: classT->getMemberFunctions()) {
            PyDict_SetItemString(memberFunctions, p.first.c_str(), typePtrToPyTypeRepresentation(p.second));
            PyDict_SetItemString(pyType->tp_dict, p.first.c_str(), typePtrToPyTypeRepresentation(p.second));
        }

        //expose 'ElementType' as a member of the type object
        PyDict_SetItemString(pyType->tp_dict, "HeldClass", typePtrToPyTypeRepresentation(classT->getHeldClass()));
        PyDict_SetItemString(pyType->tp_dict, "MemberTypes", types);
        PyDict_SetItemString(pyType->tp_dict, "MemberNames", names);
        PyDict_SetItemString(pyType->tp_dict, "MemberDefaultValues", defaults);

        PyDict_SetItemString(pyType->tp_dict, "MemberFunctions", memberFunctions);

        for (auto nameAndObj: ((Class*)inType)->getClassMembers()) {
            PyDict_SetItemString(
                pyType->tp_dict,
                nameAndObj.first.c_str(),
                nameAndObj.second
                );
        }

        for (auto nameAndObj: ((Class*)inType)->getStaticFunctions()) {
            PyDict_SetItemString(
                pyType->tp_dict,
                nameAndObj.first.c_str(),
                PyInstance::initializePythonRepresentation(nameAndObj.second, [&](instance_ptr data){
                    //nothing to do - functions like this are just types.
                })
                );
        }
    }

    if (inType->getTypeCategory() == Type::TypeCategory::catTupleOf ||
                    inType->getTypeCategory() == Type::TypeCategory::catListOf) {
        TupleOf* tupleOfType = (TupleOf*)inType;

        //expose 'ElementType' as a member of the type object
        PyDict_SetItemString(
                pyType->tp_dict,
                "ElementType",
                typePtrToPyTypeRepresentation(tupleOfType->getEltType())
                );
    }

    if (inType->getTypeCategory() == Type::TypeCategory::catPointerTo) {
        PointerTo* pointerT = (PointerTo*)inType;

        //expose 'ElementType' as a member of the type object
        PyDict_SetItemString(
                pyType->tp_dict,
                "ElementType",
                typePtrToPyTypeRepresentation(pointerT->getEltType())
                );
    }

    if (inType->getTypeCategory() == Type::TypeCategory::catConstDict) {
        ConstDict* constDictT = (ConstDict*)inType;

        //expose 'ElementType' as a member of the type object
        PyDict_SetItemString(pyType->tp_dict, "KeyType",
                typePtrToPyTypeRepresentation(constDictT->keyType())
                );
        PyDict_SetItemString(pyType->tp_dict, "ValueType",
                typePtrToPyTypeRepresentation(constDictT->valueType())
                );
    }

    if (inType->getTypeCategory() == Type::TypeCategory::catNamedTuple) {
        NamedTuple* tupleT = (NamedTuple*)inType;

        PyObject* types = PyTuple_New(tupleT->getTypes().size());
        for (long k = 0; k < tupleT->getTypes().size(); k++) {
            PyTuple_SetItem(types, k, incref(typePtrToPyTypeRepresentation(tupleT->getTypes()[k])));
        }

        PyObject* names = PyTuple_New(tupleT->getNames().size());
        for (long k = 0; k < tupleT->getNames().size(); k++) {
            PyObject* namePtr = PyUnicode_FromString(tupleT->getNames()[k].c_str());
            PyTuple_SetItem(names, k, namePtr);
        }

        //expose 'ElementType' as a member of the type object
        PyDict_SetItemString(pyType->tp_dict, "ElementTypes", types);
        PyDict_SetItemString(pyType->tp_dict, "ElementNames", names);

        Py_DECREF(names);
        Py_DECREF(types);
    }

    if (inType->getTypeCategory() == Type::TypeCategory::catOneOf) {
        OneOf* oneOfT = (OneOf*)inType;

        PyObject* types = PyTuple_New(oneOfT->getTypes().size());
        for (long k = 0; k < oneOfT->getTypes().size(); k++) {
            PyTuple_SetItem(types, k, incref(typePtrToPyTypeRepresentation(oneOfT->getTypes()[k])));
        }

        //expose 'ElementType' as a member of the type object
        PyDict_SetItemString(pyType->tp_dict, "Types", types);
        Py_DECREF(types);
    }

    if (inType->getTypeCategory() == Type::TypeCategory::catTuple) {
        Tuple* tupleT = (Tuple*)inType;

        PyObject* res = PyTuple_New(tupleT->getTypes().size());
        for (long k = 0; k < tupleT->getTypes().size(); k++) {
            PyTuple_SetItem(res, k, incref(typePtrToPyTypeRepresentation(tupleT->getTypes()[k])));
        }
        //expose 'ElementType' as a member of the type object
        PyDict_SetItemString(pyType->tp_dict, "ElementTypes", res);
    }

    if (inType->getTypeCategory() == Type::TypeCategory::catFunction) {
        //expose a list of overloads
        PyObject* overloads = PyFunctionInstance::createOverloadPyRepresentation((Function*)inType);

        PyDict_SetItemString(
                pyType->tp_dict,
                "overloads",
                overloads
                );

        Py_DECREF(overloads);
    }
}

// static
PyTypeObject* PyInstance::getObjectAsTypeObject() {
    static PyObject* module = PyImport_ImportModule("typed_python.internals");
    static PyObject* t = PyObject_GetAttrString(module, "object");
    return (PyTypeObject*)t;
}

// static
Type* PyInstance::pyFunctionToForward(PyObject* arg) {
    static PyObject* internalsModule = PyImport_ImportModule("typed_python.internals");

    if (!internalsModule) {
        throw std::runtime_error("Internal error: couldn't find typed_python.internals");
    }

    static PyObject* forwardToName = PyObject_GetAttrString(internalsModule, "forwardToName");

    if (!forwardToName) {
        throw std::runtime_error("Internal error: couldn't find typed_python.internals.makeFunction");
    }

    PyObject* fRes = PyObject_CallFunctionObjArgs(forwardToName, arg, NULL);

    std::string fwdName;

    if (!fRes) {
        fwdName = "<Internal Error>";
        PyErr_Clear();
    } else {
        if (!PyUnicode_Check(fRes)) {
            fwdName = "<Internal Error>";
            Py_DECREF(fRes);
        } else {
            fwdName = PyUnicode_AsUTF8(fRes);
            Py_DECREF(fRes);
        }
    }

    Py_INCREF(arg);
    return new Forward(arg, fwdName);
}

/**
 *  We are doing this here rather than in Type because we want to create a singleton PyUnicode
 *  object for each type category to make this function ultra fast.
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
    if (cat == Type::TypeCategory::catListOf) { static PyObject* res = PyUnicode_FromString("ListOf"); return res; }
    if (cat == Type::TypeCategory::catNamedTuple) { static PyObject* res = PyUnicode_FromString("NamedTuple"); return res; }
    if (cat == Type::TypeCategory::catTuple) { static PyObject* res = PyUnicode_FromString("Tuple"); return res; }
    if (cat == Type::TypeCategory::catConstDict) { static PyObject* res = PyUnicode_FromString("ConstDict"); return res; }
    if (cat == Type::TypeCategory::catAlternative) { static PyObject* res = PyUnicode_FromString("Alternative"); return res; }
    if (cat == Type::TypeCategory::catConcreteAlternative) { static PyObject* res = PyUnicode_FromString("ConcreteAlternative"); return res; }
    if (cat == Type::TypeCategory::catPythonSubclass) { static PyObject* res = PyUnicode_FromString("PythonSubclass"); return res; }
    if (cat == Type::TypeCategory::catBoundMethod) { static PyObject* res = PyUnicode_FromString("BoundMethod"); return res; }
    if (cat == Type::TypeCategory::catClass) { static PyObject* res = PyUnicode_FromString("Class"); return res; }
    if (cat == Type::TypeCategory::catHeldClass) { static PyObject* res = PyUnicode_FromString("HeldClass"); return res; }
    if (cat == Type::TypeCategory::catFunction) { static PyObject* res = PyUnicode_FromString("Function"); return res; }
    if (cat == Type::TypeCategory::catForward) { static PyObject* res = PyUnicode_FromString("Forward"); return res; }
    if (cat == Type::TypeCategory::catPythonObjectOfType) { static PyObject* res = PyUnicode_FromString("PythonObjectOfType"); return res; }

    static PyObject* res = PyUnicode_FromString("Unknown");
    return res;
}

// static
Instance PyInstance::unwrapPyObjectToInstance(PyObject* inst) {
    if (inst == Py_None) {
        return Instance();
    }
    if (PyBool_Check(inst)) {
        return Instance::create(inst == Py_True);
    }
    if (PyLong_Check(inst)) {
        return Instance::create(PyLong_AsLong(inst));
    }
    if (PyFloat_Check(inst)) {
        return Instance::create(PyFloat_AsDouble(inst));
    }
    if (PyBytes_Check(inst)) {
        return Instance::createAndInitialize(
            Bytes::Make(),
            [&](instance_ptr i) {
                Bytes::Make()->constructor(i, PyBytes_GET_SIZE(inst), PyBytes_AsString(inst));
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
            String::Make(),
            [&](instance_ptr i) {
                String::Make()->constructor(i, bytesPerCodepoint, count, data);
            }
        );

    }

    assert(!PyErr_Occurred());
    PyErr_Format(
        PyExc_TypeError,
        "Cannot convert %S to an Instance "
        "(only None, int, bool, bytes, and str are supported currently).",
        inst
    );

    return Instance();  // when failed, return a None instance
}

// static
Type* PyInstance::tryUnwrapPyInstanceToValueType(PyObject* typearg) {
    Instance inst = unwrapPyObjectToInstance(typearg);

    if (!PyErr_Occurred()) {
        return Value::Make(inst);
    }
    PyErr_Clear();

    Type* nativeType = PyInstance::extractTypeFrom(typearg->ob_type);
    if (nativeType) {
        return Value::Make(
            Instance::create(
                nativeType,
                ((PyInstance*)typearg)->dataPtr()
            )
        );
    }
    return nullptr;
}

//static
PyObject* PyInstance::typePtrToPyTypeRepresentation(Type* t) {
    return (PyObject*)typeObjInternal(t);
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

    if (arg == Py_None) {
        return None::Make();
    }

    if (PyFunction_Check(arg)) {
        return pyFunctionToForward(arg);
    }

    return  PyInstance::tryUnwrapPyInstanceToValueType(arg);
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
            return None::Make();
        }
        if (pyType == &PyBool_Type) {
            return Bool::Make();
        }
        if (pyType == &PyBytes_Type) {
            return Bytes::Make();
        }
        if (pyType == &PyUnicode_Type) {
            return String::Make();
        }

        if (PyInstance::isSubclassOfNativeType(pyType)) {
            Type* nativeT = PyInstance::extractTypeFrom(pyType);

            if (!nativeT) {
                PyErr_SetString(PyExc_TypeError,
                    ("Type " + std::string(pyType->tp_name) + " looked like a native type subclass, but has no base").c_str()
                    );
                return NULL;
            }

            //this is now a permanent object
            Py_INCREF(typearg);

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
    // else: typearg is not a type -> it is a value
    Type* valueType = PyInstance::tryUnwrapPyInstanceToValueType(typearg);

    if (valueType) {
        return valueType;
    }

    if (PyFunction_Check(typearg)) {
        return pyFunctionToForward(typearg);
    }


    PyErr_Format(PyExc_TypeError, "Cannot convert %S to a native type.", typearg);
    return NULL;
}
