#include <Python.h>
#include <map>
#include <memory>
#include <vector>
#include <string>
#include <iostream>

#include "Type.hpp"
#include "NullSerializationContext.hpp"

static_assert(PY_MAJOR_VERSION >= 3, "nativepython is a python3 project only");
static_assert(PY_MINOR_VERSION >= 6, "nativepython is a python3.6 project only");

//thread-local counter for how many times we've disabled native dispatch
thread_local int64_t native_dispatch_disabled = false;


inline PyObject* incref(PyObject* o) {
    Py_INCREF(o);
    return o;
}


//extension of PyTypeObject that adds a Type* at the end.
struct NativeTypeWrapper {
    PyTypeObject typeObj;
    Type* mType;
};

class InternalPyException {};

struct native_instance_wrapper {
    PyObject_HEAD

    bool mIsInitialized;
    bool mIsMatcher;
    char mIteratorFlag; //0 is keys, 1 is values, 2 is pairs
    int64_t mIteratorOffset; //-1 if we're not an iterator

    Instance mContainingInstance;

    static bool guaranteeForwardsResolved(Type* t) {
        try {
            guaranteeForwardsResolvedOrThrow(t);
            return true;
        } catch(std::exception& e) {
            PyErr_SetString(PyExc_TypeError, e.what());
            return false;
        }
    }

    static void guaranteeForwardsResolvedOrThrow(Type* t) {
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

    //return the standard python representation of an object of type 'eltType'
    template<class init_func>
    static PyObject* initializePythonRepresentation(Type* eltType, const init_func& f) {
        if (!guaranteeForwardsResolved(eltType)) {
            return nullptr;
        }

        Instance instance(eltType, f);

        return extractPythonObject(instance.data(), instance.type());
    }

    //initialize a native_instance_wrapper for 'eltType'. For ints, floats, etc, with
    //actual native representations, this will produce a wrapper object (maybe not what you want)
    //rather than the standard python representation.
    template<class init_func>
    static PyObject* initialize(Type* eltType, const init_func& f) {
        if (!guaranteeForwardsResolved(eltType)) {
            return nullptr;
        }

        native_instance_wrapper* self =
            (native_instance_wrapper*)typeObj(eltType)->tp_alloc(typeObj(eltType), 0);

        try {
            self->initialize(f);

            return (PyObject*)self;
        } catch(...) {
            typeObj(eltType)->tp_dealloc((PyObject*)self);
            throw;
        }
    }

    template<class init_func>
    void initialize(const init_func& i) {
        Type* type = extractTypeFrom(((PyObject*)this)->ob_type);
        guaranteeForwardsResolvedOrThrow(type);

        mIsInitialized = false;
        new (&mContainingInstance) Instance( type, i );
        mIsInitialized = true;
    }

    instance_ptr dataPtr() {
        return mContainingInstance.data();
    }

    static PyObject* constDictItems(PyObject *o) {
        Type* self_type = extractTypeFrom(o->ob_type);

        native_instance_wrapper* w = (native_instance_wrapper*)o;

        if (self_type && self_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
            native_instance_wrapper* self = (native_instance_wrapper*)o->ob_type->tp_alloc(o->ob_type, 0);

            self->mIteratorOffset = 0;
            self->mIteratorFlag = 2;
            self->mIsMatcher = false;

            self->initialize([&](instance_ptr data) {
                self_type->copy_constructor(data, w->dataPtr());
            });


            return (PyObject*)self;
        }

        PyErr_SetString(PyExc_TypeError, ("Cannot iterate an instance of " + self_type->name()).c_str());
        return NULL;
    }

    static PyObject* constDictKeys(PyObject *o) {
        Type* self_type = extractTypeFrom(o->ob_type);
        native_instance_wrapper* w = (native_instance_wrapper*)o;

        if (self_type && self_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
            native_instance_wrapper* self = (native_instance_wrapper*)o->ob_type->tp_alloc(o->ob_type, 0);

            self->mIteratorOffset = 0;
            self->mIteratorFlag = 0;
            self->mIsMatcher = false;

            self->initialize([&](instance_ptr data) {
                self_type->copy_constructor(data, w->dataPtr());
            });


            return (PyObject*)self;
        }

        PyErr_SetString(PyExc_TypeError, ("Cannot iterate an instance of " + self_type->name()).c_str());
        return NULL;
    }

    static PyObject* constDictValues(PyObject *o) {
        Type* self_type = extractTypeFrom(o->ob_type);
        native_instance_wrapper* w = (native_instance_wrapper*)o;

        if (self_type && self_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
            native_instance_wrapper* self = (native_instance_wrapper*)o->ob_type->tp_alloc(o->ob_type, 0);

            self->mIteratorOffset = 0;
            self->mIteratorFlag = 1;
            self->mIsMatcher = false;

            self->initialize([&](instance_ptr data) {
                self_type->copy_constructor(data, w->dataPtr());
            });


            return (PyObject*)self;
        }

        PyErr_SetString(PyExc_TypeError, ("Cannot iterate an instance of " + self_type->name()).c_str());
        return NULL;
    }

    static PyObject* constDictGet(PyObject* o, PyObject* args) {
        native_instance_wrapper* self_w = (native_instance_wrapper*)o;

        if (PyTuple_Size(args) < 1 || PyTuple_Size(args) > 2) {
            PyErr_SetString(PyExc_TypeError, "ConstDict.get takes one or two arguments");
            return NULL;
        }

        PyObject* item = PyTuple_GetItem(args,0);
        PyObject* ifNotFound = (PyTuple_Size(args) == 2 ? PyTuple_GetItem(args,1) : Py_None);

        Type* self_type = extractTypeFrom(o->ob_type);
        Type* item_type = extractTypeFrom(item->ob_type);

        if (self_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
            ConstDict* dict_t = (ConstDict*)self_type;

            if (item_type == dict_t->keyType()) {
                native_instance_wrapper* item_w = (native_instance_wrapper*)item;

                instance_ptr i = dict_t->lookupValueByKey(self_w->dataPtr(), item_w->dataPtr());

                if (!i) {
                    Py_INCREF(ifNotFound);
                    return ifNotFound;
                }

                return extractPythonObject(i, dict_t->valueType());
            } else {
                instance_ptr tempObj = (instance_ptr)malloc(dict_t->keyType()->bytecount());
                try {
                    copy_constructor(dict_t->keyType(), tempObj, item);
                } catch(std::exception& e) {
                    free(tempObj);
                    PyErr_SetString(PyExc_TypeError, e.what());
                    return NULL;
                }

                instance_ptr i = dict_t->lookupValueByKey(self_w->dataPtr(), tempObj);

                dict_t->keyType()->destroy(tempObj);
                free(tempObj);

                if (!i) {
                    Py_INCREF(ifNotFound);
                    return ifNotFound;
                }

                return extractPythonObject(i, dict_t->valueType());
            }

            PyErr_SetString(PyExc_TypeError, "Invalid ConstDict lookup type");
            return NULL;
        }

        PyErr_SetString(PyExc_TypeError, "Wrong type!");
        return NULL;
    }

    static PyMethodDef* typeMethods(Type* t) {
        if (t->getTypeCategory() == Type::TypeCategory::catConstDict) {
            return new PyMethodDef [5] {
                {"get", (PyCFunction)native_instance_wrapper::constDictGet, METH_VARARGS, NULL},
                {"items", (PyCFunction)native_instance_wrapper::constDictItems, METH_NOARGS, NULL},
                {"keys", (PyCFunction)native_instance_wrapper::constDictKeys, METH_NOARGS, NULL},
                {"values", (PyCFunction)native_instance_wrapper::constDictValues, METH_NOARGS, NULL},
                {NULL, NULL}
            };
        }

        return new PyMethodDef [2] {
            {NULL, NULL}
        };
    };

    static void tp_dealloc(PyObject* self) {
        native_instance_wrapper* wrapper = (native_instance_wrapper*)self;

        if (wrapper->mIsInitialized) {
            wrapper->mContainingInstance.~Instance();
        }

        Py_TYPE(self)->tp_free((PyObject*)self);
    }

    static bool pyValCouldBeOfType(Type* t, PyObject* pyRepresentation) {
        guaranteeForwardsResolvedOrThrow(t);

        if (t->getTypeCategory() == Type::TypeCategory::catPythonObjectOfType) {
            int isinst = PyObject_IsInstance(pyRepresentation, (PyObject*)((PythonObjectOfType*)t)->pyType());

            if (isinst == -1) {
                isinst = 0;
                PyErr_Clear();
            }

            return isinst > 0;
        }

        if (t->getTypeCategory() == Type::TypeCategory::catValue) {
            Value* valType = (Value*)t;
            if (compare_to_python(valType->value().type(), valType->value().data(), pyRepresentation, true) == 0) {
                return true;
            } else {
                return false;
            }
        }

        Type* argType = extractTypeFrom(pyRepresentation->ob_type);

        if (argType) {
            return argType->isBinaryCompatibleWith(argType);
        }

        if (t->getTypeCategory() == Type::TypeCategory::catNamedTuple ||
                t->getTypeCategory() == Type::TypeCategory::catTupleOf ||
                t->getTypeCategory() == Type::TypeCategory::catTuple
                ) {
            return PyTuple_Check(pyRepresentation) || PyList_Check(pyRepresentation) || PyDict_Check(pyRepresentation);
        }

        if (t->getTypeCategory() == Type::TypeCategory::catFloat64 ||
                t->getTypeCategory() == Type::TypeCategory::catFloat32)  {
            return PyFloat_Check(pyRepresentation);
        }

        if (t->getTypeCategory() == Type::TypeCategory::catInt64 ||
                t->getTypeCategory() == Type::TypeCategory::catInt32 ||
                t->getTypeCategory() == Type::TypeCategory::catInt16 ||
                t->getTypeCategory() == Type::TypeCategory::catInt8 ||
                t->getTypeCategory() == Type::TypeCategory::catUInt64 ||
                t->getTypeCategory() == Type::TypeCategory::catUInt32 ||
                t->getTypeCategory() == Type::TypeCategory::catUInt16 ||
                t->getTypeCategory() == Type::TypeCategory::catUInt8
                )  {
            return PyLong_CheckExact(pyRepresentation);
        }

        if (t->getTypeCategory() == Type::TypeCategory::catBool) {
            return PyBool_Check(pyRepresentation);
        }

        if (t->getTypeCategory() == Type::TypeCategory::catString) {
            return PyUnicode_Check(pyRepresentation);
        }

        if (t->getTypeCategory() == Type::TypeCategory::catBytes) {
            return PyBytes_Check(pyRepresentation);
        }

        return true;
    }

    static void copy_constructor(Type* eltType, instance_ptr tgt, PyObject* pyRepresentation) {
        guaranteeForwardsResolvedOrThrow(eltType);

        Type* argType = extractTypeFrom(pyRepresentation->ob_type);

        if (argType && argType->isBinaryCompatibleWith(eltType)) {
            //it's already the right kind of instance
            eltType->copy_constructor(tgt, ((native_instance_wrapper*)pyRepresentation)->dataPtr());
            return;
        }

        Type::TypeCategory cat = eltType->getTypeCategory();

        if (cat == Type::TypeCategory::catPythonSubclass) {
            copy_constructor((Type*)eltType->getBaseType(), tgt, pyRepresentation);
            return;
        }

        if (cat == Type::TypeCategory::catPythonObjectOfType) {
            int isinst = PyObject_IsInstance(pyRepresentation, (PyObject*)((PythonObjectOfType*)eltType)->pyType());
            if (isinst == -1) {
                isinst = 0;
                PyErr_Clear();
            }

            if (!isinst) {
                throw std::logic_error("Object of type " + std::string(pyRepresentation->ob_type->tp_name) +
                        " is not an instance of " + ((PythonObjectOfType*)eltType)->pyType()->tp_name);
            }

            Py_INCREF(pyRepresentation);
            ((PyObject**)tgt)[0] = pyRepresentation;
            return;
        }

        if (cat == Type::TypeCategory::catValue) {
            Value* v = (Value*)eltType;

            const Instance& elt = v->value();

            if (compare_to_python(elt.type(), elt.data(), pyRepresentation, false) != 0) {
                throw std::logic_error("Can't initialize a " + eltType->name() + " from an instance of " +
                    std::string(pyRepresentation->ob_type->tp_name));
            } else {
                //it's the value we want
                return;
            }
        }

        if (cat == Type::TypeCategory::catOneOf) {
            OneOf* oneOf = (OneOf*)eltType;

            for (long k = 0; k < oneOf->getTypes().size(); k++) {
                Type* subtype = oneOf->getTypes()[k];

                if (pyValCouldBeOfType(subtype, pyRepresentation)) {
                    try {
                        copy_constructor(subtype, tgt+1, pyRepresentation);
                        *(uint8_t*)tgt = k;
                        return;
                    } catch(...) {
                    }
                }
            }

            throw std::logic_error("Can't initialize a " + eltType->name() + " from an instance of " +
                std::string(pyRepresentation->ob_type->tp_name));
            return;
        }

        if (cat == Type::TypeCategory::catNone) {
            if (pyRepresentation == Py_None) {
                return;
            }
            throw std::logic_error("Can't initialize a None from an instance of " +
                std::string(pyRepresentation->ob_type->tp_name));
        }

        if (cat == Type::TypeCategory::catInt64) {
            if (PyLong_Check(pyRepresentation)) {
                ((int64_t*)tgt)[0] = PyLong_AsLong(pyRepresentation);
                return;
            }
            throw std::logic_error("Can't initialize an int64 from an instance of " +
                std::string(pyRepresentation->ob_type->tp_name));
        }

        if (cat == Type::TypeCategory::catInt32) {
            if (PyLong_Check(pyRepresentation)) {
                ((int32_t*)tgt)[0] = PyLong_AsLong(pyRepresentation);
                return;
            }
            throw std::logic_error("Can't initialize an int32 from an instance of " +
                std::string(pyRepresentation->ob_type->tp_name));
        }

        if (cat == Type::TypeCategory::catInt16) {
            if (PyLong_Check(pyRepresentation)) {
                ((int16_t*)tgt)[0] = PyLong_AsLong(pyRepresentation);
                return;
            }
            throw std::logic_error("Can't initialize an int16 from an instance of " +
                std::string(pyRepresentation->ob_type->tp_name));
        }

        if (cat == Type::TypeCategory::catInt8) {
            if (PyLong_Check(pyRepresentation)) {
                ((int8_t*)tgt)[0] = PyLong_AsLong(pyRepresentation);
                return;
            }
            throw std::logic_error("Can't initialize an int8 from an instance of " +
                std::string(pyRepresentation->ob_type->tp_name));
        }

        if (cat == Type::TypeCategory::catUInt64) {
            if (PyLong_Check(pyRepresentation)) {
                ((uint64_t*)tgt)[0] = PyLong_AsUnsignedLong(pyRepresentation);
                return;
            }
            throw std::logic_error("Can't initialize an uint64 from an instance of " +
                std::string(pyRepresentation->ob_type->tp_name));
        }

        if (cat == Type::TypeCategory::catUInt32) {
            if (PyLong_Check(pyRepresentation)) {
                ((uint32_t*)tgt)[0] = PyLong_AsUnsignedLong(pyRepresentation);
                return;
            }
            throw std::logic_error("Can't initialize an uint32 from an instance of " +
                std::string(pyRepresentation->ob_type->tp_name));
        }

        if (cat == Type::TypeCategory::catUInt16) {
            if (PyLong_Check(pyRepresentation)) {
                ((uint16_t*)tgt)[0] = PyLong_AsUnsignedLong(pyRepresentation);
                return;
            }
            throw std::logic_error("Can't initialize an uint16 from an instance of " +
                std::string(pyRepresentation->ob_type->tp_name));
        }

        if (cat == Type::TypeCategory::catUInt8) {
            if (PyLong_Check(pyRepresentation)) {
                ((uint8_t*)tgt)[0] = PyLong_AsUnsignedLong(pyRepresentation);
                return;
            }
            throw std::logic_error("Can't initialize an uint8 from an instance of " +
                std::string(pyRepresentation->ob_type->tp_name));
        }
        if (cat == Type::TypeCategory::catBool) {
            if (PyLong_Check(pyRepresentation)) {
                ((bool*)tgt)[0] = PyLong_AsLong(pyRepresentation) != 0;
                return;
            }
            throw std::logic_error("Can't initialize a Bool from an instance of " +
                std::string(pyRepresentation->ob_type->tp_name));
        }

        if (cat == Type::TypeCategory::catString) {
            if (PyUnicode_Check(pyRepresentation)) {
                auto kind = PyUnicode_KIND(pyRepresentation);
                assert(
                    kind == PyUnicode_1BYTE_KIND ||
                    kind == PyUnicode_2BYTE_KIND ||
                    kind == PyUnicode_4BYTE_KIND
                    );
                String().constructor(
                    tgt,
                    kind == PyUnicode_1BYTE_KIND ? 1 :
                    kind == PyUnicode_2BYTE_KIND ? 2 :
                                                    4,
                    PyUnicode_GET_LENGTH(pyRepresentation),
                    kind == PyUnicode_1BYTE_KIND ? (const char*)PyUnicode_1BYTE_DATA(pyRepresentation) :
                    kind == PyUnicode_2BYTE_KIND ? (const char*)PyUnicode_2BYTE_DATA(pyRepresentation) :
                                                   (const char*)PyUnicode_4BYTE_DATA(pyRepresentation)
                    );
                return;
            }
            throw std::logic_error("Can't initialize a String from an instance of " +
                std::string(pyRepresentation->ob_type->tp_name));
        }

        if (cat == Type::TypeCategory::catBytes) {
            if (PyBytes_Check(pyRepresentation)) {
                Bytes().constructor(
                    tgt,
                    PyBytes_GET_SIZE(pyRepresentation),
                    PyBytes_AsString(pyRepresentation)
                    );
                return;
            }
            throw std::logic_error("Can't initialize a Bytes object from an instance of " +
                std::string(pyRepresentation->ob_type->tp_name));
        }

        if (cat == Type::TypeCategory::catFloat64) {
            if (PyLong_Check(pyRepresentation)) {
                ((double*)tgt)[0] = PyLong_AsLong(pyRepresentation);
                return;
            }
            if (PyFloat_Check(pyRepresentation)) {
                ((double*)tgt)[0] = PyFloat_AsDouble(pyRepresentation);
                return;
            }
            throw std::logic_error("Can't initialize a float64 from an instance of " +
                std::string(pyRepresentation->ob_type->tp_name));
        }

        if (cat == Type::TypeCategory::catConstDict) {
            if (PyDict_Check(pyRepresentation)) {
                ConstDict* dictType = ((ConstDict*)eltType);
                dictType->constructor(tgt, PyDict_Size(pyRepresentation), false);

                try {
                    PyObject *key, *value;
                    Py_ssize_t pos = 0;

                    int i = 0;

                    while (PyDict_Next(pyRepresentation, &pos, &key, &value)) {
                        copy_constructor(dictType->keyType(), dictType->kvPairPtrKey(tgt, i), key);
                        try {
                            copy_constructor(dictType->valueType(), dictType->kvPairPtrValue(tgt, i), value);
                        } catch(...) {
                            dictType->keyType()->destroy(dictType->kvPairPtrKey(tgt,i));
                            throw;
                        }
                        dictType->incKvPairCount(tgt);
                        i++;
                    }

                    dictType->sortKvPairs(tgt);
                } catch(...) {
                    dictType->destroy(tgt);
                    throw;
                }
                return;
            }

            throw std::logic_error("Couldn't initialize internal elt of type " + eltType->name()
                    + " with a " + pyRepresentation->ob_type->tp_name);
        }

        if (cat == Type::TypeCategory::catTupleOf) {
            if (PyTuple_Check(pyRepresentation)) {
                ((TupleOf*)eltType)->constructor(tgt, PyTuple_Size(pyRepresentation),
                    [&](uint8_t* eltPtr, int64_t k) {
                        copy_constructor(((TupleOf*)eltType)->getEltType(), eltPtr, PyTuple_GetItem(pyRepresentation,k));
                        }
                    );
                return;
            }
            if (PyList_Check(pyRepresentation)) {
                ((TupleOf*)eltType)->constructor(tgt, PyList_Size(pyRepresentation),
                    [&](uint8_t* eltPtr, int64_t k) {
                        copy_constructor(((TupleOf*)eltType)->getEltType(), eltPtr, PyList_GetItem(pyRepresentation,k));
                        }
                    );
                return;
            }
            if (PySet_Check(pyRepresentation)) {
                if (PySet_Size(pyRepresentation) == 0) {
                    ((TupleOf*)eltType)->constructor(tgt);
                    return;
                }

                PyObject *iterator = PyObject_GetIter(pyRepresentation);

                ((TupleOf*)eltType)->constructor(tgt, PySet_Size(pyRepresentation),
                    [&](uint8_t* eltPtr, int64_t k) {
                        PyObject* item = PyIter_Next(iterator);
                        copy_constructor(((TupleOf*)eltType)->getEltType(), eltPtr, item);
                        Py_DECREF(item);
                        }
                    );

                Py_DECREF(iterator);

                return;
            }

            throw std::logic_error("Couldn't initialize internal elt of type " + eltType->name()
                    + " with a " + pyRepresentation->ob_type->tp_name);
        }

        if (eltType->isComposite()) {
            if (PyTuple_Check(pyRepresentation)) {
                if (((CompositeType*)eltType)->getTypes().size() != PyTuple_Size(pyRepresentation)) {
                    throw std::runtime_error("Wrong number of arguments");
                }

                ((CompositeType*)eltType)->constructor(tgt,
                    [&](uint8_t* eltPtr, int64_t k) {
                        copy_constructor(((CompositeType*)eltType)->getTypes()[k], eltPtr, PyTuple_GetItem(pyRepresentation,k));
                        }
                    );
                return;
            }
            if (PyList_Check(pyRepresentation)) {
                if (((CompositeType*)eltType)->getTypes().size() != PyList_Size(pyRepresentation)) {
                    throw std::runtime_error("Wrong number of arguments");
                }

                ((CompositeType*)eltType)->constructor(tgt,
                    [&](uint8_t* eltPtr, int64_t k) {
                        copy_constructor(((CompositeType*)eltType)->getTypes()[k], eltPtr, PyList_GetItem(pyRepresentation,k));
                        }
                    );
                return;
            }
        }

        if (eltType->getTypeCategory() == Type::TypeCategory::catNamedTuple) {
            if (PyDict_Check(pyRepresentation)) {
                if (((NamedTuple*)eltType)->getTypes().size() < PyDict_Size(pyRepresentation)) {
                    throw std::runtime_error("Couldn't initialize type of " + eltType->name() + " because supplied dictionary had too many items");
                }
                long actuallyUsed = 0;

                ((NamedTuple*)eltType)->constructor(tgt,
                    [&](uint8_t* eltPtr, int64_t k) {
                        const std::string& name = ((NamedTuple*)eltType)->getNames()[k];
                        Type* t = ((NamedTuple*)eltType)->getTypes()[k];

                        PyObject* o = PyDict_GetItemString(pyRepresentation, name.c_str());
                        if (o) {
                            copy_constructor(t, eltPtr, o);
                            actuallyUsed++;
                        }
                        else if (eltType->is_default_constructible()) {
                            t->constructor(eltPtr);
                        } else {
                            throw std::logic_error("Can't default initialize argument " + name);
                        }
                    });

                if (actuallyUsed != PyDict_Size(pyRepresentation)) {
                    throw std::runtime_error("Couldn't initialize type of " + eltType->name() + " because supplied dictionary had unused arguments");
                }

                return;
            }
        }

        throw std::logic_error("Couldn't initialize internal elt of type " + eltType->name() + " from " + pyRepresentation->ob_type->tp_name);
    }

    static void initialize(uint8_t* data, Type* t, PyObject* args, PyObject* kwargs) {
        guaranteeForwardsResolvedOrThrow(t);

        Type::TypeCategory cat = t->getTypeCategory();

        if (cat == Type::TypeCategory::catPythonSubclass) {
            initialize(data, (Type*)t->getBaseType(), args, kwargs);
            return;
        }

        if (cat == Type::TypeCategory::catConcreteAlternative) {
            ConcreteAlternative* alt = (ConcreteAlternative*)t;
            alt->constructor(data, [&](instance_ptr p) {
                initialize(p, alt->elementType(), args, kwargs);
            });
            return;
        }

        if (kwargs == NULL) {
            if (args == NULL || PyTuple_Size(args) == 0) {
                if (t->is_default_constructible()) {
                    t->constructor(data);
                    return;
                }
            }

            if (PyTuple_Size(args) == 1) {
                PyObject* argTuple = PyTuple_GetItem(args, 0);

                copy_constructor(t, data, argTuple);

                return;
            }

            throw std::logic_error("Can't initialize " + t->name() + " with these in-place arguments.");
        } else {
            if (cat == Type::TypeCategory::catNamedTuple) {
                long actuallyUsed = 0;

                CompositeType* compositeT = ((CompositeType*)t);

                compositeT->constructor(
                    data,
                    [&](uint8_t* eltPtr, int64_t k) {
                        Type* eltType = compositeT->getTypes()[k];
                        PyObject* o = PyDict_GetItemString(kwargs, compositeT->getNames()[k].c_str());
                        if (o) {
                            copy_constructor(eltType, eltPtr, o);
                            actuallyUsed++;
                        }
                        else if (eltType->is_default_constructible()) {
                            eltType->constructor(eltPtr);
                        } else {
                            throw std::logic_error("Can't default initialize argument " + compositeT->getNames()[k]);
                        }
                    });

                if (actuallyUsed != PyDict_Size(kwargs)) {
                    throw std::runtime_error("Couldn't initialize type of " + t->name() + " because supplied dictionary had unused arguments");
                }

                return;
            }

            throw std::logic_error("Can't initialize " + t->name() + " from python with kwargs.");
        }
    }


    //produce the pythonic representation of this object. for things like integers, string, etc,
    //convert them back to their python-native form. otherwise, a pointer back into a native python
    //structure
    static PyObject* extractPythonObject(instance_ptr data, Type* eltType) {
        if (eltType->getTypeCategory() == Type::TypeCategory::catPythonObjectOfType) {
            PyObject* res = *(PyObject**)data;
            Py_INCREF(res);
            return res;
        }
        if (eltType->getTypeCategory() == Type::TypeCategory::catValue) {
            Value* valueType = (Value*)eltType;
            return extractPythonObject(valueType->value().data(), valueType->value().type());
        }
        if (eltType->getTypeCategory() == Type::TypeCategory::catNone) {
            Py_INCREF(Py_None);
            return Py_None;
        }
        if (eltType->getTypeCategory() == Type::TypeCategory::catInt64) {
            return PyLong_FromLong(*(int64_t*)data);
        }
        if (eltType->getTypeCategory() == Type::TypeCategory::catBool) {
            PyObject* res = *(bool*)data ? Py_True : Py_False;
            Py_INCREF(res);
            return res;
        }
        if (eltType->getTypeCategory() == Type::TypeCategory::catFloat64) {
            return PyFloat_FromDouble(*(double*)data);
        }
        if (eltType->getTypeCategory() == Type::TypeCategory::catFloat32) {
            return PyFloat_FromDouble(*(float*)data);
        }
        if (eltType->getTypeCategory() == Type::TypeCategory::catBytes) {
            return PyBytes_FromStringAndSize(
                (const char*)Bytes().eltPtr(data, 0),
                Bytes().count(data)
                );
        }
        if (eltType->getTypeCategory() == Type::TypeCategory::catString) {
            int bytes_per_codepoint = String().bytes_per_codepoint(data);

            return PyUnicode_FromKindAndData(
                bytes_per_codepoint == 1 ? PyUnicode_1BYTE_KIND :
                bytes_per_codepoint == 2 ? PyUnicode_2BYTE_KIND :
                                           PyUnicode_4BYTE_KIND,
                String().eltPtr(data, 0),
                String().count(data)
                );
        }

        if (eltType->getTypeCategory() == Type::TypeCategory::catOneOf) {
            std::pair<Type*, instance_ptr> child = ((OneOf*)eltType)->unwrap(data);
            return extractPythonObject(child.second, child.first);
        }

        Type* concreteT = eltType->pickConcreteSubclass(data);

        try {
            return native_instance_wrapper::initialize(concreteT, [&](instance_ptr selfData) {
                concreteT->copy_constructor(selfData, data);
            });
        } catch(std::exception& e) {
            PyErr_SetString(PyExc_TypeError, e.what());
            return NULL;
        }
    }

    static PyObject *tp_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds) {
        Type* eltType = extractTypeFrom(subtype);

        if (!guaranteeForwardsResolved(eltType)) { return nullptr; }

        if (isSubclassOfNativeType(subtype)) {
            native_instance_wrapper* self = (native_instance_wrapper*)subtype->tp_alloc(subtype, 0);

            try {
                self->mIteratorOffset = -1;
                self->mIsMatcher = false;

                self->initialize([&](instance_ptr data) {
                    initialize(data, eltType, args, kwds);
                });

                return (PyObject*)self;
            } catch(std::exception& e) {
                subtype->tp_dealloc((PyObject*)self);

                PyErr_SetString(PyExc_TypeError, e.what());
                return NULL;
            }

            //not reachable
            assert(false);

        } else {
            instance_ptr tgt = (instance_ptr)malloc(eltType->bytecount());

            try{
                initialize(tgt, eltType, args, kwds);
            } catch(std::exception& e) {
                free(tgt);
                PyErr_SetString(PyExc_TypeError, e.what());
                return NULL;
            }


            PyObject* result = extractPythonObject(tgt, eltType);

            eltType->destroy(tgt);
            free(tgt);

            return result;
        }
    }

    static Py_ssize_t sq_length(PyObject* o) {
        native_instance_wrapper* w = (native_instance_wrapper*)o;

        Type* t = extractTypeFrom(o->ob_type);

        if (t->getTypeCategory() == Type::TypeCategory::catTupleOf) {
            return ((TupleOf*)t)->count(w->dataPtr());
        }
        if (t->isComposite()) {
            return ((CompositeType*)t)->getTypes().size();
        }
        if (t->getTypeCategory() == Type::TypeCategory::catString) {
            return String().count(w->dataPtr());
        }
        if (t->getTypeCategory() == Type::TypeCategory::catBytes) {
            return Bytes().count(w->dataPtr());
        }

        return 0;
    }

    static PyObject* nb_rshift(PyObject* lhs, PyObject* rhs) {
        std::pair<bool, PyObject*> res = checkForPyOperator(lhs, rhs, "__rshift__");
        if (res.first) {
            return res.second;
        }

        PyErr_Format(
            PyExc_TypeError,
            "Unsupported operand type(s) for >>: %S and %S",
            lhs->ob_type,
            rhs->ob_type,
            NULL
            );

        return NULL;
    }

    static std::pair<bool, PyObject*> checkForPyOperator(PyObject* lhs, PyObject* rhs, const char* op) {
        Type* lhs_type = extractTypeFrom(lhs->ob_type);

        Alternative* alt = nullptr;
        if (lhs_type && lhs_type->getTypeCategory() == Type::TypeCategory::catConcreteAlternative) {
            alt = ((ConcreteAlternative*)lhs_type)->getAlternative();
        }
        if (lhs_type && lhs_type->getTypeCategory() == Type::TypeCategory::catAlternative) {
            alt = ((Alternative*)lhs_type);
        }
        if (alt) {
            auto it = alt->getMethods().find(op);
            if (it != alt->getMethods().end()) {
                Function* f = it->second;

                PyObject* argTuple = PyTuple_Pack(2, lhs, rhs);

                for (const auto& overload: f->getOverloads()) {
                    std::pair<bool, PyObject*> res = tryToCallOverload(overload, nullptr, argTuple, nullptr);
                    if (res.first) {
                        Py_DECREF(argTuple);
                        return res;
                    }

                Py_DECREF(argTuple);
                }
            }
        }

        return std::pair<bool, PyObject*>(false, NULL);
    }

    static PyObject* nb_add(PyObject* lhs, PyObject* rhs) {
        std::pair<bool, PyObject*> res = checkForPyOperator(lhs, rhs, "__add__");
        if (res.first) {
            return res.second;
        }

        PyErr_Format(
            PyExc_TypeError,
            "Unsupported operand type(s) for +: %S and %S",
            lhs->ob_type,
            rhs->ob_type,
            NULL
            );

        return NULL;
    }

    static PyObject* nb_subtract(PyObject* lhs, PyObject* rhs) {
        Type* lhs_type = extractTypeFrom(lhs->ob_type);
        Type* rhs_type = extractTypeFrom(rhs->ob_type);

        if (lhs_type) {
            native_instance_wrapper* w_lhs = (native_instance_wrapper*)lhs;

            if (lhs_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
                ConstDict* dict_t = (ConstDict*)lhs_type;

                Type* tupleOfKeysType = dict_t->tupleOfKeysType();

                if (lhs_type == tupleOfKeysType) {
                    native_instance_wrapper* w_rhs = (native_instance_wrapper*)rhs;

                    native_instance_wrapper* self =
                        (native_instance_wrapper*)typeObj(lhs_type)->tp_alloc(typeObj(lhs_type), 0);

                    self->initialize([&](instance_ptr data) {
                        ((ConstDict*)lhs_type)->subtractTupleOfKeysFromDict(w_lhs->dataPtr(), w_rhs->dataPtr(), data);
                    });

                    return (PyObject*)self;
                } else {
                    //attempt to convert rhs to a relevant dict type.
                    instance_ptr tempObj = (instance_ptr)malloc(tupleOfKeysType->bytecount());

                    try {
                        copy_constructor(tupleOfKeysType, tempObj, rhs);
                    } catch(std::exception& e) {
                        free(tempObj);
                        PyErr_SetString(PyExc_TypeError, e.what());
                        return NULL;
                    }

                    native_instance_wrapper* self =
                        (native_instance_wrapper*)typeObj(lhs_type)->tp_alloc(typeObj(lhs_type), 0);

                    self->initialize([&](instance_ptr data) {
                        ((ConstDict*)lhs_type)->subtractTupleOfKeysFromDict(w_lhs->dataPtr(), tempObj, data);
                    });

                    tupleOfKeysType->destroy(tempObj);

                    free(tempObj);

                    return (PyObject*)self;
                }
            }
        }

        std::pair<bool, PyObject*> res = checkForPyOperator(lhs, rhs, "__sub__");
        if (res.first) {
            return res.second;
        }

        PyErr_Format(
            PyExc_TypeError,
            "Unsupported operand type(s) for -: %S and %S",
            lhs->ob_type,
            rhs->ob_type,
            NULL
            );

        return NULL;
    }

    static PyObject* sq_concat(PyObject* lhs, PyObject* rhs) {
        Type* lhs_type = extractTypeFrom(lhs->ob_type);
        Type* rhs_type = extractTypeFrom(rhs->ob_type);

        if (lhs_type) {
            native_instance_wrapper* w_lhs = (native_instance_wrapper*)lhs;

            if (lhs_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
                if (lhs_type == rhs_type) {
                    native_instance_wrapper* w_rhs = (native_instance_wrapper*)rhs;

                    native_instance_wrapper* self =
                        (native_instance_wrapper*)typeObj(lhs_type)->tp_alloc(typeObj(lhs_type), 0);

                    self->initialize([&](instance_ptr data) {
                        ((ConstDict*)lhs_type)->addDicts(w_lhs->dataPtr(), w_rhs->dataPtr(), data);
                    });

                    return (PyObject*)self;
                } else {
                    //attempt to convert rhs to a relevant dict type.
                    instance_ptr tempObj = (instance_ptr)malloc(lhs_type->bytecount());

                    try {
                        copy_constructor(lhs_type, tempObj, rhs);
                    } catch(std::exception& e) {
                        free(tempObj);
                        PyErr_SetString(PyExc_TypeError, e.what());
                        return NULL;
                    }

                    native_instance_wrapper* self =
                        (native_instance_wrapper*)typeObj(lhs_type)->tp_alloc(typeObj(lhs_type), 0);

                    self->initialize([&](instance_ptr data) {
                        ((ConstDict*)lhs_type)->addDicts(w_lhs->dataPtr(), tempObj, data);
                    });

                    lhs_type->destroy(tempObj);

                    free(tempObj);

                    return (PyObject*)self;
                }
            }
            if (lhs_type->getTypeCategory() == Type::TypeCategory::catTupleOf) {
                //TupleOf(X) + TupleOf(X) fastpath
                if (lhs_type == rhs_type) {
                    native_instance_wrapper* w_rhs = (native_instance_wrapper*)rhs;

                    TupleOf* tupT = (TupleOf*)lhs_type;
                    Type* eltType = tupT->getEltType();
                    native_instance_wrapper* self =
                        (native_instance_wrapper*)typeObj(tupT)
                            ->tp_alloc(typeObj(tupT), 0);

                    int count_lhs = tupT->count(w_lhs->dataPtr());
                    int count_rhs = tupT->count(w_rhs->dataPtr());

                    self->initialize([&](instance_ptr data) {
                        tupT->constructor(data, count_lhs + count_rhs,
                            [&](uint8_t* eltPtr, int64_t k) {
                                eltType->copy_constructor(
                                    eltPtr,
                                    k < count_lhs ? tupT->eltPtr(w_lhs->dataPtr(), k) :
                                        tupT->eltPtr(w_rhs->dataPtr(), k - count_lhs)
                                    );
                                }
                            );
                    });

                    return (PyObject*)self;
                }
                //generic path to add any kind of iterable.
                if (PyObject_Length(rhs) != -1) {
                    TupleOf* tupT = (TupleOf*)lhs_type;
                    Type* eltType = tupT->getEltType();

                    native_instance_wrapper* self =
                        (native_instance_wrapper*)typeObj(tupT)
                            ->tp_alloc(typeObj(tupT), 0);

                    int count_lhs = tupT->count(w_lhs->dataPtr());
                    int count_rhs = PyObject_Length(rhs);

                    try {
                        self->initialize([&](instance_ptr data) {
                            tupT->constructor(data, count_lhs + count_rhs,
                                [&](uint8_t* eltPtr, int64_t k) {
                                    if (k < count_lhs) {
                                        eltType->copy_constructor(
                                            eltPtr,
                                            tupT->eltPtr(w_lhs->dataPtr(), k)
                                            );
                                    } else {
                                        PyObject* kval = PyLong_FromLong(k - count_lhs);
                                        PyObject* o = PyObject_GetItem(rhs, kval);
                                        Py_DECREF(kval);

                                        if (!o) {
                                            throw InternalPyException();
                                        }

                                        try {
                                            copy_constructor(eltType, eltPtr, o);
                                        } catch(...) {
                                            Py_DECREF(o);
                                            throw;
                                        }

                                        Py_DECREF(o);
                                    }
                                });
                        });
                    } catch(std::exception& e) {
                        typeObj(tupT)->tp_dealloc((PyObject*)self);
                        PyErr_SetString(PyExc_TypeError, e.what());
                        return NULL;
                    }

                    return (PyObject*)self;
                }
            }
        }

        PyErr_SetString(
            PyExc_TypeError,
            (std::string("cannot concatenate ") + lhs->ob_type->tp_name + " and "
                    + rhs->ob_type->tp_name).c_str()
            );
        return NULL;
    }

    static PyObject* sq_item(PyObject* o, Py_ssize_t ix) {
        native_instance_wrapper* w = (native_instance_wrapper*)o;
        Type* t = extractTypeFrom(o->ob_type);

        if (t->getTypeCategory() == Type::TypeCategory::catTupleOf) {
            int64_t count = ((TupleOf*)t)->count(w->dataPtr());

            if (ix < 0) {
                ix += count;
            }

            if (ix >= count || ix < 0) {
                PyErr_SetString(PyExc_IndexError, "index out of range");
                return NULL;
            }

            Type* eltType = (Type*)((TupleOf*)t)->getEltType();
            return extractPythonObject(
                ((TupleOf*)t)->eltPtr(w->dataPtr(), ix),
                eltType
                );
        }

        if (t->isComposite()) {
            auto compType = (CompositeType*)t;

            if (ix < 0 || ix >= (int64_t)compType->getTypes().size()) {
                PyErr_SetString(PyExc_IndexError, "index out of range");
                return NULL;
            }

            Type* eltType = compType->getTypes()[ix];

            return extractPythonObject(
                compType->eltPtr(w->dataPtr(), ix),
                eltType
                );
        }

        if (t->getTypeCategory() == Type::TypeCategory::catBytes) {
            if (ix < 0 || ix >= (int64_t)Bytes().count(w->dataPtr())) {
                PyErr_SetString(PyExc_IndexError, "index out of range");
                return NULL;
            }

            return PyBytes_FromStringAndSize(
                (const char*)Bytes().eltPtr(w->dataPtr(), ix),
                1
                );
        }
        if (t->getTypeCategory() == Type::TypeCategory::catString) {
            if (ix < 0 || ix >= (int64_t)String().count(w->dataPtr())) {
                PyErr_SetString(PyExc_IndexError, "index out of range");
                return NULL;
            }

            int bytes_per_codepoint = String().bytes_per_codepoint(w->dataPtr());

            return PyUnicode_FromKindAndData(
                bytes_per_codepoint == 1 ? PyUnicode_1BYTE_KIND :
                bytes_per_codepoint == 2 ? PyUnicode_2BYTE_KIND :
                                           PyUnicode_4BYTE_KIND,
                String().eltPtr(w->dataPtr(), ix),
                1
                );
        }

        PyErr_SetString(PyExc_TypeError, "not a __getitem__'able thing.");
        return NULL;
    }

    static PyTypeObject* typeObj(Type* inType) {
        if (!inType->getTypeRep()) {
            inType->setTypeRep(typeObjInternal(inType));
        }

        return inType->getTypeRep();
    }

    static PySequenceMethods* sequenceMethodsFor(Type* t) {
        if (t->getTypeCategory() == Type::TypeCategory::catTupleOf ||
                t->getTypeCategory() == Type::TypeCategory::catTuple ||
                t->getTypeCategory() == Type::TypeCategory::catNamedTuple ||
                t->getTypeCategory() == Type::TypeCategory::catString ||
                t->getTypeCategory() == Type::TypeCategory::catBytes ||
                t->getTypeCategory() == Type::TypeCategory::catConstDict) {
            PySequenceMethods* res =
                new PySequenceMethods {0,0,0,0,0,0,0,0};

            if (t->getTypeCategory() == Type::TypeCategory::catConstDict) {
                res->sq_contains = (objobjproc)native_instance_wrapper::sq_contains;
            } else {
                res->sq_length = (lenfunc)native_instance_wrapper::sq_length;
                res->sq_item = (ssizeargfunc)native_instance_wrapper::sq_item;
            }

            res->sq_concat = native_instance_wrapper::sq_concat;

            return res;
        }

        return 0;
    }

    static PyNumberMethods* numberMethods(Type* t) {
        return new PyNumberMethods {
                //only enable this for the types that it operates on. Otherwise it disables the concatenation functions
                //we should probably just unify them
                t->getTypeCategory() == Type::TypeCategory::catConcreteAlternative ||
                    t->getTypeCategory() == Type::TypeCategory::catAlternative
                    ? nb_add : 0, //binaryfunc nb_add
                nb_subtract, //binaryfunc nb_subtract
                0, //binaryfunc nb_multiply
                0, //binaryfunc nb_remainder
                0, //binaryfunc nb_divmod
                0, //ternaryfunc nb_power
                0, //unaryfunc nb_negative
                0, //unaryfunc nb_positive
                0, //unaryfunc nb_absolute
                0, //inquiry nb_bool
                0, //unaryfunc nb_invert
                0, //binaryfunc nb_lshift
                nb_rshift, //binaryfunc nb_rshift
                0, //binaryfunc nb_and
                0, //binaryfunc nb_xor
                0, //binaryfunc nb_or
                0, //unaryfunc nb_int
                0, //void *nb_reserved
                0, //unaryfunc nb_float
                0, //binaryfunc nb_inplace_add
                0, //binaryfunc nb_inplace_subtract
                0, //binaryfunc nb_inplace_multiply
                0, //binaryfunc nb_inplace_remainder
                0, //ternaryfunc nb_inplace_power
                0, //binaryfunc nb_inplace_lshift
                0, //binaryfunc nb_inplace_rshift
                0, //binaryfunc nb_inplace_and
                0, //binaryfunc nb_inplace_xor
                0, //binaryfunc nb_inplace_or
                0, //binaryfunc nb_floor_divide
                0, //binaryfunc nb_true_divide
                0, //binaryfunc nb_inplace_floor_divide
                0, //binaryfunc nb_inplace_true_divide
                0, //unaryfunc nb_index
                0, //binaryfunc nb_matrix_multiply
                0  //binaryfunc nb_inplace_matrix_multiply
                };
    }

    static Py_ssize_t mp_length(PyObject* o) {
        native_instance_wrapper* w = (native_instance_wrapper*)o;

        Type* t = extractTypeFrom(o->ob_type);

        if (t->getTypeCategory() == Type::TypeCategory::catConstDict) {
            return ((ConstDict*)t)->size(w->dataPtr());
        }

        if (t->getTypeCategory() == Type::TypeCategory::catTupleOf) {
            return ((TupleOf*)t)->count(w->dataPtr());
        }

        return 0;
    }

    static int sq_contains(PyObject* o, PyObject* item) {
        native_instance_wrapper* self_w = (native_instance_wrapper*)o;

        Type* self_type = extractTypeFrom(o->ob_type);
        Type* item_type = extractTypeFrom(item->ob_type);

        if (self_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
            ConstDict* dict_t = (ConstDict*)self_type;

            if (item_type == dict_t->keyType()) {
                native_instance_wrapper* item_w = (native_instance_wrapper*)item;

                instance_ptr i = dict_t->lookupValueByKey(self_w->dataPtr(), item_w->dataPtr());

                if (!i) {
                    return 0;
                }

                return 1;
            } else {
                instance_ptr tempObj = (instance_ptr)malloc(dict_t->keyType()->bytecount());
                try {
                    copy_constructor(dict_t->keyType(), tempObj, item);
                } catch(std::exception& e) {
                    free(tempObj);
                    PyErr_SetString(PyExc_TypeError, e.what());
                    return -1;
                }

                instance_ptr i = dict_t->lookupValueByKey(self_w->dataPtr(), tempObj);

                dict_t->keyType()->destroy(tempObj);
                free(tempObj);

                if (!i) {
                    return 0;
                }

                return 1;
            }
        }

        return 0;
    }
    static PyObject* mp_subscript(PyObject* o, PyObject* item) {
        native_instance_wrapper* self_w = (native_instance_wrapper*)o;

        Type* self_type = extractTypeFrom(o->ob_type);
        Type* item_type = extractTypeFrom(item->ob_type);

        if (self_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
            ConstDict* dict_t = (ConstDict*)self_type;

            if (item_type == dict_t->keyType()) {
                native_instance_wrapper* item_w = (native_instance_wrapper*)item;

                instance_ptr i = dict_t->lookupValueByKey(self_w->dataPtr(), item_w->dataPtr());

                if (!i) {
                    PyErr_SetObject(PyExc_KeyError, item);
                    return NULL;
                }

                return extractPythonObject(i, dict_t->valueType());
            } else {
                instance_ptr tempObj = (instance_ptr)malloc(dict_t->keyType()->bytecount());
                try {
                    copy_constructor(dict_t->keyType(), tempObj, item);
                } catch(std::exception& e) {
                    free(tempObj);
                    PyErr_SetString(PyExc_TypeError, e.what());
                    return NULL;
                }

                instance_ptr i = dict_t->lookupValueByKey(self_w->dataPtr(), tempObj);

                dict_t->keyType()->destroy(tempObj);
                free(tempObj);

                if (!i) {
                    PyErr_SetObject(PyExc_KeyError, item);
                    return NULL;
                }

                return extractPythonObject(i, dict_t->valueType());
            }

            PyErr_SetString(PyExc_TypeError, "Invalid ConstDict lookup type");
            return NULL;
        }

        if (self_type->getTypeCategory() == Type::TypeCategory::catTupleOf) {
            if (PySlice_Check(item)) {
                TupleOf* tupType = (TupleOf*)self_type;

                Py_ssize_t start,stop,step,slicelength;
                if (PySlice_GetIndicesEx(item, tupType->count(self_w->dataPtr()), &start,
                            &stop, &step, &slicelength) == -1) {
                    return NULL;
                }

                Type* eltType = tupType->getEltType();

                native_instance_wrapper* result =
                    (native_instance_wrapper*)typeObj(tupType)->tp_alloc(typeObj(tupType), 0);

                result->initialize([&](instance_ptr data) {
                    tupType->constructor(data, slicelength,
                        [&](uint8_t* eltPtr, int64_t k) {
                            eltType->copy_constructor(
                                eltPtr,
                                tupType->eltPtr(self_w->dataPtr(), start + k * step)
                                );
                            }
                        );
                });

                return (PyObject*)result;
            }

            if (PyLong_Check(item)) {
                return sq_item((PyObject*)self_w, PyLong_AsLong(item));
            }
        }

        PyErr_SetObject(PyExc_KeyError, item);
        return NULL;
    }

    static PyMappingMethods* mappingMethods(Type* t) {
        static PyMappingMethods* res =
            new PyMappingMethods {
                native_instance_wrapper::mp_length, //mp_length
                native_instance_wrapper::mp_subscript, //mp_subscript
                0 //mp_ass_subscript
                };

        if (t->getTypeCategory() == Type::TypeCategory::catConstDict ||
            t->getTypeCategory() == Type::TypeCategory::catTupleOf) {
            return res;
        }

        return 0;
    }

    static bool isSubclassOfNativeType(PyTypeObject* typeObj) {
        if (typeObj->tp_as_buffer == bufferProcs()) {
            return false;
        }

        while (typeObj) {
            if (typeObj->tp_as_buffer == bufferProcs()) {
                return true;
            }
            typeObj = typeObj->tp_base;
        }

        return false;
    }

    static Type* extractTypeFrom(PyTypeObject* typeObj, bool exact=false) {
        if (exact && isSubclassOfNativeType(typeObj)) {
            return PythonSubclass::Make(extractTypeFrom(typeObj), typeObj);
        }

        while (!exact && typeObj->tp_base && typeObj->tp_as_buffer != bufferProcs()) {
            typeObj = typeObj->tp_base;
        }

        if (typeObj->tp_as_buffer == bufferProcs()) {
            return ((NativeTypeWrapper*)typeObj)->mType;
        }

        return nullptr;
    }

    static int tp_setattro(PyObject *o, PyObject* attrName, PyObject* attrVal) {
        if (!PyUnicode_Check(attrName)) {
            PyErr_Format(PyExc_AttributeError, "Instance of type %S has no attribute '%S'", o->ob_type, attrName);
            return -1;
        }

        Type* type = extractTypeFrom(o->ob_type);

        if (type->getTypeCategory() == Type::TypeCategory::catClass) {
            native_instance_wrapper* self_w = (native_instance_wrapper*)o;
            Class* nt = (Class*)type;

            int i = nt->memberNamed(PyUnicode_AsUTF8(attrName));

            if (i < 0) {
                PyErr_Format(PyExc_AttributeError, "Instance of type %S has no attribute '%S'", o->ob_type, attrName);
                return -1;
            }

            Type* eltType = nt->getMembers()[i].second;

            Type* attrType = extractTypeFrom(attrVal->ob_type);

            if (eltType == attrType) {
                native_instance_wrapper* item_w = (native_instance_wrapper*)attrVal;

                attrType->assign(nt->eltPtr(self_w->dataPtr(), i), item_w->dataPtr());

                return 0;
            } else {
                instance_ptr tempObj = (instance_ptr)malloc(eltType->bytecount());
                try {
                    copy_constructor(eltType, tempObj, attrVal);
                } catch(std::exception& e) {
                    free(tempObj);
                    PyErr_SetString(PyExc_TypeError, e.what());
                    return -1;
                }

                eltType->assign(nt->eltPtr(self_w->dataPtr(), i), tempObj);

                eltType->destroy(tempObj);
                free(tempObj);

                return 0;
            }
        }

        PyErr_Format(PyExc_AttributeError, "Instance of type %S has no attribute '%S'", o->ob_type, attrName);
        return -1;
    }

    static std::pair<bool, PyObject*> tryToCallOverload(const Function::Overload& f, PyObject* self, PyObject* args, PyObject* kwargs) {
        PyObject* targetArgTuple = PyTuple_New(PyTuple_Size(args)+(self?1:0));
        Function::Matcher matcher(f);

        int write_slot = 0;

        if (self) {
            Py_INCREF(self);
            PyTuple_SetItem(targetArgTuple, write_slot++, self);
            matcher.requiredTypeForArg(nullptr);
        }

        for (long k = 0; k < PyTuple_Size(args); k++) {
            PyObject* elt = PyTuple_GetItem(args, k);

            //what type would we need for this unnamed arg?
            Type* targetType = matcher.requiredTypeForArg(nullptr);

            if (!matcher.stillMatches()) {
                Py_DECREF(targetArgTuple);
                return std::make_pair(false, nullptr);
            }

            if (!targetType) {
                Py_INCREF(elt);
                PyTuple_SetItem(targetArgTuple, write_slot++, elt);
            }
            else {
                try {
                    PyObject* targetObj =
                        native_instance_wrapper::initializePythonRepresentation(targetType, [&](instance_ptr data) {
                            copy_constructor(targetType, data, elt);
                        });

                    PyTuple_SetItem(targetArgTuple, write_slot++, targetObj);
                } catch(...) {
                    //not a valid conversion, but keep going
                    Py_DECREF(targetArgTuple);
                    return std::make_pair(false, nullptr);
                }
            }
        }

        PyObject* newKwargs = nullptr;

        if (kwargs) {
            newKwargs = PyDict_New();

            PyObject *key, *value;
            Py_ssize_t pos = 0;

            while (PyDict_Next(kwargs, &pos, &key, &value)) {
                if (!PyUnicode_Check(key)) {
                    Py_DECREF(targetArgTuple);
                    Py_DECREF(newKwargs);
                    PyErr_SetString(PyExc_TypeError, "Keywords arguments must be strings.");
                    return std::make_pair(false, nullptr);
                }

                //what type would we need for this unnamed arg?
                Type* targetType = matcher.requiredTypeForArg(PyUnicode_AsUTF8(key));

                if (!matcher.stillMatches()) {
                    Py_DECREF(targetArgTuple);
                    Py_DECREF(newKwargs);
                    return std::make_pair(false, nullptr);
                }

                if (!targetType) {
                    PyDict_SetItem(newKwargs, key, value);
                }
                else {
                    try {
                        PyObject* convertedValue = native_instance_wrapper::initializePythonRepresentation(targetType, [&](instance_ptr data) {
                            copy_constructor(targetType, data, value);
                        });

                        PyDict_SetItem(newKwargs, key, convertedValue);
                        Py_DECREF(convertedValue);
                    } catch(...) {
                        //not a valid conversion
                        Py_DECREF(targetArgTuple);
                        Py_DECREF(newKwargs);
                        return std::make_pair(false, nullptr);
                    }
                }
            }
        }

        if (!matcher.definitelyMatches()) {
            Py_DECREF(targetArgTuple);
            return std::make_pair(false, nullptr);
        }

        PyObject* result;

        if (f.getEntrypoint() && !native_dispatch_disabled) {
            result = dispatchFunctionCallToNative(f, targetArgTuple, newKwargs);
        } else {
            result = PyObject_Call((PyObject*)f.getFunctionObj(), targetArgTuple, newKwargs);
        }

        Py_DECREF(targetArgTuple);
        if (newKwargs) {
            Py_DECREF(newKwargs);
        }

        //exceptions pass through directly
        if (!result) {
            return std::make_pair(true, result);
        }

        //force ourselves to convert to the native type
        if (f.getReturnType()) {
            try {
                PyObject* newRes = native_instance_wrapper::initializePythonRepresentation(f.getReturnType(), [&](instance_ptr data) {
                        copy_constructor(f.getReturnType(), data, result);
                    });
                Py_DECREF(result);
                return std::make_pair(true, newRes);
            } catch (std::exception& e) {
                Py_DECREF(result);
                PyErr_SetString(PyExc_TypeError, e.what());
                return std::make_pair(true, (PyObject*)nullptr);
            }
        }

        return std::make_pair(true, result);
    }

    static PyObject* dispatchFunctionCallToNative(const Function::Overload& overload, PyObject* argTuple, PyObject *kwargs) {
        Type* returnType = overload.getReturnType();

        if (!returnType) {
            returnType = PythonObjectOfType::Make(getObjectAsTypeObject());
        }

        if (PyTuple_Size(argTuple) != overload.getArgs().size()) {
            PyErr_Format(PyExc_TypeError, "Can't call %S natively with %d arguments", PyTuple_Size(argTuple));
            return NULL;
        }

        if (kwargs && PyDict_Size(kwargs)) {
            PyErr_Format(PyExc_TypeError, "Can't call %S natively with keyword arguments", PyTuple_Size(argTuple));
            return NULL;
        }

        std::vector<Instance> instances;

        for (long k = 0; k < overload.getArgs().size(); k++) {
            auto arg = overload.getArgs()[k];

            if (arg.getIsKwarg() || arg.getIsStarArg()) {
                PyErr_Format(PyExc_TypeError, "Can't call %S natively because we don't support *args and **kwargs yet", overload.getFunctionObj());
                return NULL;
            }

            Type* toUse = arg.getTypeFilter();
            if (!toUse) {
                toUse = PythonObjectOfType::Make(getObjectAsTypeObject());
            }

            instances.push_back(
                Instance::createAndInitialize(toUse, [&](instance_ptr p) {
                    copy_constructor(toUse, p, PyTuple_GetItem(argTuple, k));
                })
                );
        }

        try {
            Instance result = Instance::createAndInitialize(returnType, [&](instance_ptr returnData) {
                std::vector<instance_ptr> args;
                for (auto& i: instances) {
                    args.push_back(i.data());
                }

                overload.getEntrypoint()(returnData, &args[0]);
            });

            return extractPythonObject(result.data(), result.type());
        } catch(...) {
            PyErr_Format(PyExc_TypeError, "Generated code threw an exception!");
            return NULL;
        }
    }

    static PyObject* tp_call(PyObject* o, PyObject* args, PyObject* kwargs) {
        Type* self_type = extractTypeFrom(o->ob_type);
        native_instance_wrapper* w = (native_instance_wrapper*)o;

        if (self_type->getTypeCategory() == Type::TypeCategory::catFunction) {
            Function* methodType = (Function*)self_type;

            for (const auto& overload: methodType->getOverloads()) {
                std::pair<bool, PyObject*> res = tryToCallOverload(overload, nullptr, args, kwargs);
                if (res.first) {
                    return res.second;
                }
            }

            PyErr_Format(PyExc_TypeError, "'%s' cannot find a valid overload with these arguments", o->ob_type->tp_name);
            return 0;
        }


        if (self_type->getTypeCategory() == Type::TypeCategory::catBoundMethod) {
            BoundMethod* methodType = (BoundMethod*)self_type;

            Function* f = methodType->getFunction();
            Class* c = methodType->getClass();

            PyObject* objectInstance = native_instance_wrapper::initializePythonRepresentation(c, [&](instance_ptr d) {
                c->copy_constructor(d, w->dataPtr());
            });

            for (const auto& overload: f->getOverloads()) {
                std::pair<bool, PyObject*> res = tryToCallOverload(overload, objectInstance, args, kwargs);
                if (res.first) {
                    Py_DECREF(objectInstance);
                    return res.second;
                }
            }

            Py_DECREF(objectInstance);
            PyErr_Format(PyExc_TypeError, "'%s' cannot find a valid overload with these arguments", o->ob_type->tp_name);
            return 0;
        }

        PyErr_Format(PyExc_TypeError, "'%s' object is not callable", o->ob_type->tp_name);
        return 0;
    }

    static PyObject* tp_getattro(PyObject *o, PyObject* attrName) {
        if (!PyUnicode_Check(attrName)) {
            PyErr_SetString(PyExc_AttributeError, "attribute is not a string");
            return NULL;
        }

        char *attr_name = PyUnicode_AsUTF8(attrName);

        Type* t = extractTypeFrom(o->ob_type);

        native_instance_wrapper* w = (native_instance_wrapper*)o;

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
                native_instance_wrapper* self = (native_instance_wrapper*)o->ob_type->tp_alloc(o->ob_type, 0);

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
                if (nt->getMembers()[k].first == attr_name) {
                    Type* eltType = nt->getMembers()[k].second;

                    return extractPythonObject(
                        nt->eltPtr(w->dataPtr(), k),
                        nt->getMembers()[k].second
                        );
                }
            }

            for (long k = 0; k < nt->getMemberFunctions().size(); k++) {
                auto it = nt->getMemberFunctions().find(attr_name);
                if (it != nt->getMemberFunctions().end()) {
                    BoundMethod* bm = BoundMethod::Make(nt, it->second);

                    return native_instance_wrapper::initializePythonRepresentation(bm, [&](instance_ptr data) {
                        bm->copy_constructor(data, w->dataPtr());
                    });
                }
            }

            for (long k = 0; k < nt->getClassMembers().size(); k++) {
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

    static PyObject* getattr(Type* type, instance_ptr data, char* attr_name) {
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

    static Py_hash_t tp_hash(PyObject *o) {
        Type* self_type = extractTypeFrom(o->ob_type);
        native_instance_wrapper* w = (native_instance_wrapper*)o;

        int32_t h = self_type->hash32(w->dataPtr());
        if (h == -1) {
            h = -2;
        }

        return h;
    }

    static char compare_to_python(Type* t, instance_ptr self, PyObject* other, bool exact) {
        if (t->getTypeCategory() == Type::TypeCategory::catValue) {
            Value* valType = (Value*)t;
            return compare_to_python(valType->value().type(), valType->value().data(), other, exact);
        }

        Type* otherT = extractTypeFrom(other->ob_type);

        if (otherT) {
            if (otherT < t) {
                return 1;
            }
            if (otherT > t) {
                return -1;
            }
            return t->cmp(self, ((native_instance_wrapper*)other)->dataPtr());
        }

        if (t->getTypeCategory() == Type::TypeCategory::catOneOf) {
            std::pair<Type*, instance_ptr> child = ((OneOf*)t)->unwrap(self);
            return compare_to_python(child.first, child.second, other, exact);
        }

        if (other == Py_None) {
            return (t->getTypeCategory() == Type::TypeCategory::catNone ? 0 : 1);
        }

        if (PyBool_Check(other)) {
            int64_t other_l = other == Py_True ? 1 : 0;
            int64_t self_l;

            if (t->getTypeCategory() == Type::TypeCategory::catBool) {
                self_l = (*(bool*)self) ? 1 : 0;
            } else {
                return -1;
            }

            if (other_l < self_l) { return -1; }
            if (other_l > self_l) { return 1; }
            return 0;
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
                    return -1;
                }
                if (other_l < *(float*)self) { return -1; }
                if (other_l > *(float*)self) { return 1; }
                return 0;
            } else if (t->getTypeCategory() == Type::TypeCategory::catFloat64) {
                if (exact) {
                    return -1;
                }
                if (other_l < *(double*)self) { return -1; }
                if (other_l > *(double*)self) { return 1; }
                return 0;
            } else {
                return -1;
            }

            if (other_l < self_l) { return -1; }
            if (other_l > self_l) { return 1; }
            return 0;
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
                    return -1;
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
                    return -1;
                }
            }

            if (other_d < self_d) { return -1; }
            if (other_d > self_d) { return 1; }
            return 0;
        }

        if (PyTuple_Check(other)) {
            if (t->getTypeCategory() == Type::TypeCategory::catTupleOf) {
                TupleOf* tupT = (TupleOf*)t;
                int lenO = PyTuple_Size(other);
                int lenS = tupT->count(self);
                for (long k = 0; k < lenO && k < lenS; k++) {
                    char res = compare_to_python(tupT->getEltType(), tupT->eltPtr(self, k), PyTuple_GetItem(other,k), exact);
                    if (res) {
                        return res;
                    }
                }

                if (lenS < lenO) { return -1; }
                if (lenS > lenO) { return 1; }
                return 0;
            }
            if (t->getTypeCategory() == Type::TypeCategory::catTuple ||
                        t->getTypeCategory() == Type::TypeCategory::catNamedTuple) {
                CompositeType* tupT = (CompositeType*)t;
                int lenO = PyTuple_Size(other);
                int lenS = tupT->getTypes().size();

                for (long k = 0; k < lenO && k < lenS; k++) {
                    char res = compare_to_python(tupT->getTypes()[k], tupT->eltPtr(self, k), PyTuple_GetItem(other,k), exact);
                    if (res) {
                        return res;
                    }
                }

                if (lenS < lenO) { return -1; }
                if (lenS > lenO) { return 1; }

                return 0;
            }
        }

        if (PyUnicode_Check(other) && t->getTypeCategory() == Type::TypeCategory::catString) {
            auto kind = PyUnicode_KIND(other);
            int bytesPer = kind == PyUnicode_1BYTE_KIND ? 1 :
                kind == PyUnicode_2BYTE_KIND ? 2 : 4;

            if (bytesPer != ((String*)t)->bytes_per_codepoint(self)) {
                return -1;
            }

            if (PyUnicode_GET_LENGTH(other) != ((String*)t)->count(self)) {
                return -1;
            }

            return memcmp(
                kind == PyUnicode_1BYTE_KIND ? (const char*)PyUnicode_1BYTE_DATA(other) :
                kind == PyUnicode_2BYTE_KIND ? (const char*)PyUnicode_2BYTE_DATA(other) :
                                               (const char*)PyUnicode_4BYTE_DATA(other),
                ((String*)t)->eltPtr(self, 0),
                PyUnicode_GET_LENGTH(other) * bytesPer
                ) == 0 ? 0 : 1;
        }
        if (PyBytes_Check(other) && t->getTypeCategory() == Type::TypeCategory::catBytes) {
            if (PyBytes_GET_SIZE(other) != ((Bytes*)t)->count(self)) {
                return -1;
            }

            return memcmp(
                PyBytes_AsString(other),
                ((Bytes*)t)->eltPtr(self, 0),
                PyBytes_GET_SIZE(other)
                ) == 0 ? 0 : 1;
        }

        return -1;
    }

    static PyObject *tp_richcompare(PyObject *a, PyObject *b, int op) {
        Type* own = extractTypeFrom(a->ob_type);
        Type* other = extractTypeFrom(b->ob_type);


        if (!other) {
            char cmp = compare_to_python(own, ((native_instance_wrapper*)a)->dataPtr(), b, false);

            PyObject* res;
            if (op == Py_EQ) {
                res = cmp == 0 ? Py_True : Py_False;
            } else if (op == Py_NE) {
                res = cmp != 0 ? Py_True : Py_False;
            } else {
                PyErr_SetString(PyExc_TypeError, "invalid comparison");
                return NULL;
            }

            Py_INCREF(res);

            return res;
        } else {
            char cmp = 0;

            if (own == other) {
                cmp = own->cmp(((native_instance_wrapper*)a)->dataPtr(), ((native_instance_wrapper*)b)->dataPtr());
            } else if (own < other) {
                cmp = -1;
            } else {
                cmp = 1;
            }

            PyObject* res;

            if (op == Py_LT) {
                res = (cmp < 0 ? Py_True : Py_False);
            } else if (op == Py_LE) {
                res = (cmp <= 0 ? Py_True : Py_False);
            } else if (op == Py_EQ) {
                res = (cmp == 0 ? Py_True : Py_False);
            } else if (op == Py_NE) {
                res = (cmp != 0 ? Py_True : Py_False);
            } else if (op == Py_GT) {
                res = (cmp > 0 ? Py_True : Py_False);
            } else if (op == Py_GE) {
                res = (cmp >= 0 ? Py_True : Py_False);
            } else {
                res = Py_NotImplemented;
            }

            Py_INCREF(res);

            return res;
        }
    }

    static PyObject* tp_iter(PyObject *o) {
        Type* self_type = extractTypeFrom(o->ob_type);
        native_instance_wrapper* w = (native_instance_wrapper*)o;

        if (self_type && self_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
            native_instance_wrapper* self = (native_instance_wrapper*)o->ob_type->tp_alloc(o->ob_type, 0);

            self->mIteratorOffset = 0;
            self->mIteratorFlag = w->mIteratorFlag;
            self->mIsMatcher = false;

            self->initialize([&](instance_ptr data) {
                self_type->copy_constructor(data, w->dataPtr());
            });

            return (PyObject*)self;
        }

        PyErr_SetString(PyExc_TypeError, ("Cannot iterate an instance of " + self_type->name()).c_str());
        return NULL;
    }

    static PyObject* tp_iternext(PyObject *o) {
        Type* self_type = extractTypeFrom(o->ob_type);
        native_instance_wrapper* w = (native_instance_wrapper*)o;

        if (self_type->getTypeCategory() != Type::TypeCategory::catConstDict) {
            return NULL;
        }

        ConstDict* dict_t = (ConstDict*)self_type;

        if (w->mIteratorOffset >= dict_t->size(w->dataPtr())) {
            return NULL;
        }

        w->mIteratorOffset++;

        if (w->mIteratorFlag == 2) {
            auto t1 = extractPythonObject(
                    dict_t->kvPairPtrKey(w->dataPtr(), w->mIteratorOffset-1),
                    dict_t->keyType()
                    );
            auto t2 = extractPythonObject(
                    dict_t->kvPairPtrValue(w->dataPtr(), w->mIteratorOffset-1),
                    dict_t->valueType()
                    );

            auto res = PyTuple_Pack(2, t1, t2);

            Py_DECREF(t1);
            Py_DECREF(t2);

            return res;
        } else if (w->mIteratorFlag == 1) {
            return extractPythonObject(
                dict_t->kvPairPtrValue(w->dataPtr(), w->mIteratorOffset-1),
                dict_t->valueType()
                );
        } else {
            return extractPythonObject(
                dict_t->kvPairPtrKey(w->dataPtr(), w->mIteratorOffset-1),
                dict_t->keyType()
                );
        }
    }

    static PyObject* tp_repr(PyObject *o) {
        Type* self_type = extractTypeFrom(o->ob_type);
        native_instance_wrapper* w = (native_instance_wrapper*)o;

        std::ostringstream str;
        str << std::showpoint;

        self_type->repr(w->dataPtr(), str);

        return PyUnicode_FromString(str.str().c_str());
    }

    static PyObject* tp_str(PyObject *o) {
        Type* self_type = extractTypeFrom(o->ob_type);

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

        return tp_repr(o);
    }

    static bool typeCanBeSubclassed(Type* t) {
        return t->getTypeCategory() == Type::TypeCategory::catNamedTuple;
    }

    static PyBufferProcs* bufferProcs() {
        static PyBufferProcs* procs = new PyBufferProcs { 0, 0 };
        return procs;
    }

    static PyTypeObject* typeObjInternal(Type* inType) {
        static std::recursive_mutex mutex;
        static std::map<Type*, NativeTypeWrapper*> types;

        std::lock_guard<std::recursive_mutex> lock(mutex);

        auto it = types.find(inType);
        if (it != types.end()) {
            return (PyTypeObject*)it->second;
        }

        types[inType] = new NativeTypeWrapper { {
                PyVarObject_HEAD_INIT(NULL, 0)
                inType->name().c_str(),    /* tp_name */
                sizeof(native_instance_wrapper),       /* tp_basicsize */
                0,                         // tp_itemsize
                native_instance_wrapper::tp_dealloc,// tp_dealloc
                0,                         // tp_print
                0,                         // tp_getattr
                0,                         // tp_setattr
                0,                         // tp_reserved
                tp_repr,                   // tp_repr
                numberMethods(inType),     // tp_as_number
                sequenceMethodsFor(inType),   // tp_as_sequence
                mappingMethods(inType),    // tp_as_mapping
                tp_hash,                   // tp_hash
                tp_call,                   // tp_call
                tp_str,                    // tp_str
                native_instance_wrapper::tp_getattro, // tp_getattro
                native_instance_wrapper::tp_setattro, // tp_setattro
                bufferProcs(),             // tp_as_buffer
                typeCanBeSubclassed(inType) ?
                    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE
                :   Py_TPFLAGS_DEFAULT,    // tp_flags
                0,                         // tp_doc
                0,                         // traverseproc tp_traverse;
                0,                         // inquiry tp_clear;
                tp_richcompare,            // richcmpfunc tp_richcompare;
                0,                         // Py_ssize_t tp_weaklistoffset;
                inType->getTypeCategory() == Type::TypeCategory::catConstDict ?
                    native_instance_wrapper::tp_iter
                :   0,                     // getiterfunc tp_iter;
                native_instance_wrapper::tp_iternext,// iternextfunc tp_iternext;
                typeMethods(inType),       // struct PyMethodDef *tp_methods;
                0,                         // struct PyMemberDef *tp_members;
                0,                         // struct PyGetSetDef *tp_getset;
                0,                         // struct _typeobject *tp_base;
                PyDict_New(),              // PyObject *tp_dict;
                0,                         // descrgetfunc tp_descr_get;
                0,                         // descrsetfunc tp_descr_set;
                0,                         // Py_ssize_t tp_dictoffset;
                0,                         // initproc tp_init;
                0,                         // allocfunc tp_alloc;
                native_instance_wrapper::tp_new,// newfunc tp_new;
                0,                         // freefunc tp_free; /* Low-level free-memory routine */
                0,                         // inquiry tp_is_gc; /* For PyObject_IS_GC */
                0,                         // PyObject *tp_bases;
                0,                         // PyObject *tp_mro; /* method resolution order */
                0,                         // PyObject *tp_cache;
                0,                         // PyObject *tp_subclasses;
                0,                         // PyObject *tp_weaklist;
                0,                         // destructor tp_del;
                0,                         // unsigned int tp_version_tag;
                0,                         // destructor tp_finalize;
                }, inType
                };

        //at this point, the dictionary has an entry, so if we recurse back to this function
        //we will return the correct entry.
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

    static void mirrorTypeInformationIntoPyType(Type* inType, PyTypeObject* pyType) {
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
                "Index",
                PyLong_FromLong(alt->which())
                );

            PyDict_SetItemString(
                pyType->tp_dict,
                "ElementType",
                (PyObject*)typeObjInternal(alt->elementType())
                );
        }

        if (inType->getTypeCategory() == Type::TypeCategory::catClass) {
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
                    native_instance_wrapper::initializePythonRepresentation(nameAndObj.second, [&](instance_ptr data){
                        //nothing to do - functions like this are just types.
                    })
                    );
            }
        }

        if (inType->getTypeCategory() == Type::TypeCategory::catTupleOf) {
            TupleOf* tupleOfType = (TupleOf*)inType;

            //expose 'ElementType' as a member of the type object
            PyDict_SetItemString(
                    pyType->tp_dict,
                    "ElementType",
                    typePtrToPyTypeRepresentation(tupleOfType->getEltType())
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
            PyObject* overloads = createOverloadPyRepresentation((Function*)inType);

            PyDict_SetItemString(
                    pyType->tp_dict,
                    "overloads",
                    overloads
                    );

            Py_DECREF(overloads);
        }
    }

    static PyTypeObject* getObjectAsTypeObject() {
        static PyObject* module = PyImport_ImportModule("typed_python.internals");
        static PyObject* t = PyObject_GetAttrString(module, "object");
        return (PyTypeObject*)t;
    }

    static PyObject* createOverloadPyRepresentation(Function* f) {
        static PyObject* internalsModule = PyImport_ImportModule("typed_python.internals");

        if (!internalsModule) {
            throw std::runtime_error("Internal error: couldn't find typed_python.internals");
        }

        static PyObject* funcOverload = PyObject_GetAttrString(internalsModule, "FunctionOverload");

        if (!funcOverload) {
            throw std::runtime_error("Internal error: couldn't find typed_python.internals.FunctionOverload");
        }

        PyObject* overloadTuple = PyTuple_New(f->getOverloads().size());

        for (long k = 0; k < f->getOverloads().size(); k++) {
            auto& overload = f->getOverloads()[k];

            PyObject* pyIndex = PyLong_FromLong(k);

            PyObject* pyOverloadInst = PyObject_CallFunctionObjArgs(
                funcOverload,
                typePtrToPyTypeRepresentation(f),
                pyIndex,
                (PyObject*)overload.getFunctionObj(),
                overload.getReturnType() ? (PyObject*)typePtrToPyTypeRepresentation(overload.getReturnType()) : Py_None,
                NULL
                );

            Py_DECREF(pyIndex);

            if (pyOverloadInst) {
                for (auto arg: f->getOverloads()[k].getArgs()) {
                    PyObject* res = PyObject_CallMethod(pyOverloadInst, "addArg", "sOOOO",
                        arg.getName().c_str(),
                        arg.getDefaultValue() ? PyTuple_Pack(1, arg.getDefaultValue()) : Py_None,
                        arg.getTypeFilter() ? (PyObject*)typePtrToPyTypeRepresentation(arg.getTypeFilter()) : Py_None,
                        arg.getIsStarArg() ? Py_True : Py_False,
                        arg.getIsKwarg() ? Py_True : Py_False
                        );

                    if (!res) {
                        PyErr_PrintEx(0);
                    } else {
                        Py_DECREF(res);
                    }
                }

                PyTuple_SetItem(overloadTuple, k, pyOverloadInst);
            } else {
                PyErr_PrintEx(0);
                Py_INCREF(Py_None);
                PyTuple_SetItem(overloadTuple, k, Py_None);
            }
        }

        return overloadTuple;
    }

    static Type* pyFunctionToForward(PyObject* arg) {
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

    static Type* tryUnwrapPyInstanceToType(PyObject* arg) {
        if (PyType_Check(arg)) {
            Type* possibleType = native_instance_wrapper::unwrapTypeArgToTypePtr(arg);
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

        return  native_instance_wrapper::tryUnwrapPyInstanceToValueType(arg);
    }

    static PyObject* categoryToPyString(Type::TypeCategory cat) {
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

    static Type* tryUnwrapPyInstanceToValueType(PyObject* typearg) {
        if (PyBool_Check(typearg)) {
            return Value::MakeBool(typearg == Py_True);
        }
        if (PyLong_Check(typearg)) {
            int64_t val = PyLong_AsLong(typearg);
            return Value::MakeInt64(val);
        }
        if (PyFloat_Check(typearg)) {
            return Value::MakeFloat64(PyFloat_AsDouble(typearg));
        }
        if (PyBytes_Check(typearg)) {
            return Value::MakeBytes(PyBytes_AsString(typearg), PyBytes_GET_SIZE(typearg));
        }
        if (PyUnicode_Check(typearg)) {
            auto kind = PyUnicode_KIND(typearg);
            assert(
                kind == PyUnicode_1BYTE_KIND ||
                kind == PyUnicode_2BYTE_KIND ||
                kind == PyUnicode_4BYTE_KIND
                );
            return Value::MakeString(
                kind == PyUnicode_1BYTE_KIND ? 1 :
                kind == PyUnicode_2BYTE_KIND ? 2 :
                                                4,
                PyUnicode_GET_LENGTH(typearg),
                kind == PyUnicode_1BYTE_KIND ? (char*)PyUnicode_1BYTE_DATA(typearg) :
                kind == PyUnicode_2BYTE_KIND ? (char*)PyUnicode_2BYTE_DATA(typearg) :
                                               (char*)PyUnicode_4BYTE_DATA(typearg)
                );
        }

        Type* nativeType = native_instance_wrapper::extractTypeFrom(typearg->ob_type);
        if (nativeType) {
            return Value::Make(
                Instance::create(
                    nativeType,
                    ((native_instance_wrapper*)typearg)->dataPtr()
                    )
                );
        }

        return nullptr;
    }

    static PyObject* typePtrToPyTypeRepresentation(Type* t) {
        return (PyObject*)typeObjInternal(t);
    }

    static Type* unwrapTypeArgToTypePtr(PyObject* typearg) {
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

            if (native_instance_wrapper::isSubclassOfNativeType(pyType)) {
                Type* nativeT = native_instance_wrapper::extractTypeFrom(pyType);

                if (!nativeT) {
                    PyErr_SetString(PyExc_TypeError,
                        ("Type " + std::string(pyType->tp_name) + " looked like a native type subclass, but has no base").c_str()
                        );
                    return NULL;
                }

                //this is now a permanent object
                Py_INCREF(typearg);

                return PythonSubclass::Make(nativeT, pyType);
            }

            Type* res = native_instance_wrapper::extractTypeFrom(pyType);
            if (res) {
                return res;
            }

            return PythonObjectOfType::Make(pyType);
        }

        Type* valueType = native_instance_wrapper::tryUnwrapPyInstanceToValueType(typearg);

        if (valueType) {
            return valueType;
        }

        if (PyFunction_Check(typearg)) {
            return pyFunctionToForward(typearg);
        }


        PyErr_Format(PyExc_TypeError, "Cannot convert %S to a native type.", typearg);
        return NULL;
    }
};

PyObject *MakeTupleOfType(PyObject* nullValue, PyObject* args) {
    std::vector<Type*> types;
    for (long k = 0; k < PyTuple_Size(args); k++) {
        types.push_back(native_instance_wrapper::unwrapTypeArgToTypePtr(PyTuple_GetItem(args,k)));
        if (not types.back()) {
            return NULL;
        }
    }

    if (types.size() != 1) {
        PyErr_SetString(PyExc_TypeError, "TupleOf takes 1 positional argument.");
        return NULL;
    }

    PyObject* typeObj = (PyObject*)native_instance_wrapper::typeObj(TupleOf::Make(types[0]));

    Py_INCREF(typeObj);
    return typeObj;
}

PyObject *MakeTupleType(PyObject* nullValue, PyObject* args) {
    std::vector<Type*> types;
    for (long k = 0; k < PyTuple_Size(args); k++) {
        types.push_back(native_instance_wrapper::unwrapTypeArgToTypePtr(PyTuple_GetItem(args,k)));
        if (not types.back()) {
            return NULL;
        }
    }

    PyObject* typeObj = (PyObject*)native_instance_wrapper::typeObj(Tuple::Make(types));

    Py_INCREF(typeObj);
    return typeObj;
}

PyObject *MakeConstDictType(PyObject* nullValue, PyObject* args) {
    std::vector<Type*> types;
    for (long k = 0; k < PyTuple_Size(args); k++) {
        types.push_back(native_instance_wrapper::unwrapTypeArgToTypePtr(PyTuple_GetItem(args,k)));
        if (not types.back()) {
            return NULL;
        }
    }

    if (types.size() != 2) {
        PyErr_SetString(PyExc_TypeError, "ConstDict accepts two arguments");
        return NULL;
    }

    PyObject* typeObj = (PyObject*)native_instance_wrapper::typeObj(
        ConstDict::Make(types[0],types[1])
        );

    Py_INCREF(typeObj);
    return typeObj;
}

PyObject *MakeOneOfType(PyObject* nullValue, PyObject* args) {
    std::vector<Type*> types;
    for (long k = 0; k < PyTuple_Size(args); k++) {
        PyObject* arg = PyTuple_GetItem(args,k);

        Type* t = native_instance_wrapper::tryUnwrapPyInstanceToType(arg);

        if (t) {
            types.push_back(t);
        } else {
            PyErr_SetString(PyExc_TypeError, "Can't handle values like this in Types.");
            return NULL;
        }
    }

    PyObject* typeObj = (PyObject*)native_instance_wrapper::typeObj(OneOf::Make(types));

    Py_INCREF(typeObj);
    return typeObj;
}

PyObject *MakeNamedTupleType(PyObject* nullValue, PyObject* args, PyObject* kwargs) {
    if (args && PyTuple_Check(args) && PyTuple_Size(args)) {
        PyErr_SetString(PyExc_TypeError, "NamedTuple takes no positional arguments.");
        return NULL;
    }

    std::vector<std::pair<std::string, Type*> > namesAndTypes;

    if (kwargs) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;

        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if (!PyUnicode_Check(key)) {
                PyErr_SetString(PyExc_TypeError, "NamedTuple keywords are supposed to be strings.");
                return NULL;
            }

            namesAndTypes.push_back(
                std::make_pair(
                    PyUnicode_AsUTF8(key),
                    native_instance_wrapper::unwrapTypeArgToTypePtr(value)
                    )
                );

            if (not namesAndTypes.back().second) {
                return NULL;
            }
        }
    }

    if (PY_MINOR_VERSION <= 5) {
        //we cannot rely on the ordering of 'kwargs' here because of the python version, so
        //we sort it. this will be a problem for anyone running some processes using different
        //python versions that share python code.
        std::sort(namesAndTypes.begin(), namesAndTypes.end());
    }

    std::vector<std::string> names;
    std::vector<Type*> types;

    for (auto p: namesAndTypes) {
        names.push_back(p.first);
        types.push_back(p.second);
    }

    PyObject* typeObj = (PyObject*)native_instance_wrapper::typeObj(NamedTuple::Make(types, names));

    Py_INCREF(typeObj);
    return typeObj;
}


PyObject *MakeBoolType(PyObject* nullValue, PyObject* args) {
    PyObject* res = (PyObject*)native_instance_wrapper::typeObj(::Bool::Make());
    Py_INCREF(res);
    return res;
}
PyObject *MakeInt8Type(PyObject* nullValue, PyObject* args) {
    PyObject* res = (PyObject*)native_instance_wrapper::typeObj(::Int8::Make());
    Py_INCREF(res);
    return res;
}
PyObject *MakeInt16Type(PyObject* nullValue, PyObject* args) {
    PyObject* res = (PyObject*)native_instance_wrapper::typeObj(::Int16::Make());
    Py_INCREF(res);
    return res;
}
PyObject *MakeInt32Type(PyObject* nullValue, PyObject* args) {
    PyObject* res = (PyObject*)native_instance_wrapper::typeObj(::Int32::Make());
    Py_INCREF(res);
    return res;
}
PyObject *MakeInt64Type(PyObject* nullValue, PyObject* args) {
    PyObject* res = (PyObject*)native_instance_wrapper::typeObj(::Int64::Make());
    Py_INCREF(res);
    return res;
}
PyObject *MakeFloat32Type(PyObject* nullValue, PyObject* args) {
    PyObject* res = (PyObject*)native_instance_wrapper::typeObj(::Float32::Make());
    Py_INCREF(res);
    return res;
}
PyObject *MakeFloat64Type(PyObject* nullValue, PyObject* args) {
    PyObject* res = (PyObject*)native_instance_wrapper::typeObj(::Float64::Make());
    Py_INCREF(res);
    return res;
}
PyObject *MakeUInt8Type(PyObject* nullValue, PyObject* args) {
    PyObject* res = (PyObject*)native_instance_wrapper::typeObj(::UInt8::Make());
    Py_INCREF(res);
    return res;
}
PyObject *MakeUInt16Type(PyObject* nullValue, PyObject* args) {
    PyObject* res = (PyObject*)native_instance_wrapper::typeObj(::UInt16::Make());
    Py_INCREF(res);
    return res;
}
PyObject *MakeUInt32Type(PyObject* nullValue, PyObject* args) {
    PyObject* res = (PyObject*)native_instance_wrapper::typeObj(::UInt32::Make());
    Py_INCREF(res);
    return res;
}
PyObject *MakeUInt64Type(PyObject* nullValue, PyObject* args) {
    PyObject* res = (PyObject*)native_instance_wrapper::typeObj(::UInt64::Make());
    Py_INCREF(res);
    return res;
}
PyObject *MakeStringType(PyObject* nullValue, PyObject* args) {
    PyObject* res = (PyObject*)native_instance_wrapper::typeObj(::String::Make());
    Py_INCREF(res);
    return res;
}
PyObject *MakeBytesType(PyObject* nullValue, PyObject* args) {
    PyObject* res = (PyObject*)native_instance_wrapper::typeObj(::Bytes::Make());
    Py_INCREF(res);
    return res;
}

PyObject *MakeNoneTypeType(PyObject* nullValue, PyObject* args) {
    PyObject* res = (PyObject*)native_instance_wrapper::typeObj(::None::Make());
    Py_INCREF(res);
    return res;
}

PyObject *MakeTypeFor(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "TypeFor takes 1 positional argument");
        return NULL;
    }

    PyObject* arg = PyTuple_GetItem(args,0);

    if (arg == Py_None) {
        return incref((PyObject*)native_instance_wrapper::typeObj(::None::Make()));
    }

    if (!PyType_Check(arg)) {
        PyErr_Format(PyExc_TypeError, "TypeFor expects a python primitive or an existing native value, not %S", arg);
        return NULL;
    }

    Type* type = native_instance_wrapper::unwrapTypeArgToTypePtr(arg);

    if (type) {
        PyObject* typeObj = (PyObject*)native_instance_wrapper::typeObj(type);
        Py_INCREF(typeObj);
        return typeObj;
    }

    PyErr_Format(PyExc_TypeError, "Couldn't convert %S to a Type", arg);
    return NULL;
}

PyObject *MakeValueType(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "Value takes 1 positional argument");
        return NULL;
    }

    PyObject* arg = PyTuple_GetItem(args,0);

    if (PyType_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Value expects a python primitive or an existing native value");
        return NULL;
    }

    Type* type = native_instance_wrapper::tryUnwrapPyInstanceToValueType(arg);

    if (type) {
        PyObject* typeObj = (PyObject*)native_instance_wrapper::typeObj(type);
        Py_INCREF(typeObj);
        return typeObj;
    }

    PyErr_SetString(PyExc_TypeError, "Couldn't convert this to a value");
    return NULL;
}

bool unpackTupleToStringAndTypes(PyObject* tuple, std::vector<std::pair<std::string, Type*> >& out) {
    std::set<std::string> memberNames;

    for (int i = 0; i < PyTuple_Size(tuple); ++i) {
        PyObject* entry = PyTuple_GetItem(tuple, i);
        Type* targetType = NULL;

        if (!PyTuple_Check(entry) || PyTuple_Size(entry) != 2
                || !PyUnicode_Check(PyTuple_GetItem(entry, 0))
                || !(targetType =
                    native_instance_wrapper::tryUnwrapPyInstanceToType(
                        PyTuple_GetItem(entry, 1)
                        ))
                )
        {
            PyErr_SetString(PyExc_TypeError, "Badly formed class type argument.");
            return false;
        }

        std::string memberName = PyUnicode_AsUTF8(PyTuple_GetItem(entry,0));

        if (memberNames.find(memberName) != memberNames.end()) {
            PyErr_Format(PyExc_TypeError, "Cannot redefine Class member %s", memberName.c_str());
            return false;
        }

        memberNames.insert(memberName);

        out.push_back(
            std::make_pair(memberName, targetType)
            );
    }

    return true;
}

bool unpackTupleToStringAndObjects(PyObject* tuple, std::vector<std::pair<std::string, PyObject*> >& out) {
    std::set<std::string> memberNames;

    for (int i = 0; i < PyTuple_Size(tuple); ++i) {
        PyObject* entry = PyTuple_GetItem(tuple, i);
        Type* targetType = NULL;

        if (!PyTuple_Check(entry) || PyTuple_Size(entry) != 2
                || !PyUnicode_Check(PyTuple_GetItem(entry, 0))
                )
        {
            PyErr_SetString(PyExc_TypeError, "Badly formed class type argument.");
            return false;
        }

        std::string memberName = PyUnicode_AsUTF8(PyTuple_GetItem(entry,0));

        if (memberNames.find(memberName) != memberNames.end()) {
            PyErr_Format(PyExc_TypeError, "Cannot redefine Class member %s", memberName.c_str());
            return false;
        }

        memberNames.insert(memberName);

        PyObject* item = PyTuple_GetItem(entry, 1);
        Py_INCREF(item);

        out.push_back(
            std::make_pair(memberName, item)
            );
    }

    return true;
}

PyObject *MakeFunctionType(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 4 && PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "Function takes 2 or 4 arguments");
        return NULL;
    }

    Function* resType;

    if (PyTuple_Size(args) == 2) {
        PyObject* a0 = PyTuple_GetItem(args,0);
        PyObject* a1 = PyTuple_GetItem(args,1);

        Type* t0 = native_instance_wrapper::unwrapTypeArgToTypePtr(a0);
        Type* t1 = native_instance_wrapper::unwrapTypeArgToTypePtr(a1);

        if (!t0 || t0->getTypeCategory() != Type::TypeCategory::catFunction) {
            PyErr_SetString(PyExc_TypeError, "Expected first argument to be a function");
            return NULL;
        }
        if (!t1 || t1->getTypeCategory() != Type::TypeCategory::catFunction) {
            PyErr_SetString(PyExc_TypeError, "Expected second argument to be a function");
            return NULL;
        }

        resType = Function::merge((Function*)t0, (Function*)t1);
    } else {
        PyObject* nameObj = PyTuple_GetItem(args,0);
        if (!PyUnicode_Check(nameObj)) {
            PyErr_SetString(PyExc_TypeError, "First arg should be a string.");
            return NULL;
        }

        PyObject* retType = PyTuple_GetItem(args,1);
        PyObject* funcObj = PyTuple_GetItem(args,2);
        PyObject* argTuple = PyTuple_GetItem(args,3);

        if (!PyFunction_Check(funcObj)) {
            PyErr_SetString(PyExc_TypeError, "Third arg should be a function.");
            return NULL;
        }

        Type* rType = 0;

        if (retType != Py_None) {
            rType = native_instance_wrapper::unwrapTypeArgToTypePtr(retType);
            if (!rType) {
                PyErr_SetString(PyExc_TypeError, "Expected second argument to be None or a type");
                return NULL;
            }
        }

        if (!PyTuple_Check(argTuple)) {
            PyErr_SetString(PyExc_TypeError, "Expected fourth argument to be a tuple of args");
            return NULL;
        }

        std::vector<Function::FunctionArg> argList;

        for (long k = 0; k < PyTuple_Size(argTuple); k++) {
            PyObject* kTup = PyTuple_GetItem(argTuple, k);
            if (!PyTuple_Check(kTup) || PyTuple_Size(kTup) != 5) {
                PyErr_SetString(PyExc_TypeError, "Argtuple elements should be tuples of five things.");
                return NULL;
            }

            PyObject* k0 = PyTuple_GetItem(kTup, 0);
            PyObject* k1 = PyTuple_GetItem(kTup, 1);
            PyObject* k2 = PyTuple_GetItem(kTup, 2);
            PyObject* k3 = PyTuple_GetItem(kTup, 3);
            PyObject* k4 = PyTuple_GetItem(kTup, 4);

            if (!PyUnicode_Check(k0)) {
                PyErr_Format(PyExc_TypeError, "Argument %S has a name which is not a string.", k0);
                return NULL;
            }

            Type* argT = nullptr;
            if (k1 != Py_None) {
                argT = native_instance_wrapper::unwrapTypeArgToTypePtr(k1);
                if (!argT) {
                    PyErr_Format(PyExc_TypeError, "Argument %S has a type argument %S which "
                            "should be None or a Type.", k0, k1);
                    return NULL;
                }
            }

            if ((k3 != Py_True && k3 != Py_False) || (k4 != Py_True && k4 != Py_False)) {
                PyErr_Format(PyExc_TypeError, "Argument %S has a malformed type tuple", k0);
                return NULL;
            }

            PyObject* val = nullptr;
            if (k2 != Py_None) {
                if (!PyTuple_Check(k2) || PyTuple_Size(k2) != 1) {
                    PyErr_Format(PyExc_TypeError, "Argument %S has a malformed type tuple", k0);
                    return NULL;
                }

                val = PyTuple_GetItem(k2,0);
            }


            if (val) {
                Py_INCREF(val);
            }
            Py_INCREF(funcObj);

            argList.push_back(Function::FunctionArg(
                PyUnicode_AsUTF8(k0),
                argT,
                val,
                k3 == Py_True,
                k4 == Py_True
                ));
        }

        std::vector<Function::Overload> overloads;
        overloads.push_back(
            Function::Overload((PyFunctionObject*)funcObj, rType, argList)
            );

        resType = new Function(PyUnicode_AsUTF8(nameObj), overloads);
    }

    PyObject* typeObj = (PyObject*)native_instance_wrapper::typeObj(resType);
    Py_INCREF(typeObj);
    return typeObj;
}

Function* convertPythonObjectToFunction(PyObject* name, PyObject *funcObj) {
    static PyObject* internalsModule = PyImport_ImportModule("typed_python.internals");

    if (!internalsModule) {
        PyErr_SetString(PyExc_TypeError, "Internal error: couldn't find typed_python.internals");
        return nullptr;
    }

    static PyObject* makeFunction = PyObject_GetAttrString(internalsModule, "makeFunction");

    if (!makeFunction) {
        PyErr_SetString(PyExc_TypeError, "Internal error: couldn't find typed_python.internals.makeFunction");
        return nullptr;
    }

    PyObject* fRes = PyObject_CallFunctionObjArgs(makeFunction, name, funcObj, NULL);

    if (!fRes) {
        return nullptr;
    }

    if (!PyType_Check(fRes)) {
        PyErr_SetString(PyExc_TypeError, "Internal error: expected typed_python.internals.makeFunction to return a type");
        return nullptr;
    }

    Type* actualType = native_instance_wrapper::extractTypeFrom((PyTypeObject*)fRes);

    if (!actualType || actualType->getTypeCategory() != Type::TypeCategory::catFunction) {
        PyErr_Format(PyExc_TypeError, "Internal error: expected makeFunction to return a Function. Got %S", fRes);
        return nullptr;
    }

    return (Function*)actualType;
}

PyObject *MakeClassType(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 5) {
        PyErr_SetString(PyExc_TypeError, "Class takes 2 arguments (name and a list of class members)");
        return NULL;
    }

    PyObject* nameArg = PyTuple_GetItem(args,0);

    if (!PyUnicode_Check(nameArg)) {
        PyErr_SetString(PyExc_TypeError, "Class needs a string in the first argument");
        return NULL;
    }

    std::string name = PyUnicode_AsUTF8(nameArg);

    PyObject* memberTuple = PyTuple_GetItem(args,1);
    PyObject* memberFunctionTuple = PyTuple_GetItem(args,2);
    PyObject* staticFunctionTuple = PyTuple_GetItem(args,3);
    PyObject* classMemberTuple = PyTuple_GetItem(args,4);

    if (!PyTuple_Check(memberTuple)) {
        PyErr_SetString(PyExc_TypeError, "Class needs a tuple of (str, member_type) in the second argument");
        return NULL;
    }

    if (!PyTuple_Check(memberFunctionTuple)) {
        PyErr_SetString(PyExc_TypeError, "Class needs a tuple of (str, Function) in the third argument");
        return NULL;
    }

    if (!PyTuple_Check(memberFunctionTuple)) {
        PyErr_SetString(PyExc_TypeError, "Class needs a tuple of (str, Function) in the fourth argument");
        return NULL;
    }

    if (!PyTuple_Check(classMemberTuple)) {
        PyErr_SetString(PyExc_TypeError, "Class needs a tuple of (str, object) in the fifth argument");
        return NULL;
    }

    std::vector<std::pair<std::string, Type*> > members;
    std::vector<std::pair<std::string, Type*> > memberFunctions;
    std::vector<std::pair<std::string, Type*> > staticFunctions;
    std::vector<std::pair<std::string, PyObject*> > classMembers;

    if (!unpackTupleToStringAndTypes(memberTuple, members)) {
        return NULL;
    }
    if (!unpackTupleToStringAndTypes(memberFunctionTuple, memberFunctions)) {
        return NULL;
    }
    if (!unpackTupleToStringAndTypes(staticFunctionTuple, staticFunctions)) {
        return NULL;
    }
    if (!unpackTupleToStringAndObjects(classMemberTuple, classMembers)) {
        return NULL;
    }

    std::map<std::string, Function*> memberFuncs;
    std::map<std::string, Function*> staticFuncs;

    for (auto mf: memberFunctions) {
        if (mf.second->getTypeCategory() != Type::TypeCategory::catFunction) {
            PyErr_Format(PyExc_TypeError, "Class member %s is not a function.", mf.first.c_str());
            return NULL;
        }
        if (memberFuncs.find(mf.first) != memberFuncs.end()) {
            PyErr_Format(PyExc_TypeError, "Class member %s repeated. This should have"
                                    " been compressed as an overload.", mf.first.c_str());
            return NULL;
        }
        memberFuncs[mf.first] = (Function*)mf.second;
    }

    for (auto mf: staticFunctions) {
        if (mf.second->getTypeCategory() != Type::TypeCategory::catFunction) {
            PyErr_Format(PyExc_TypeError, "Class member %s is not a function.", mf.first.c_str());
            return NULL;
        }
        if (staticFuncs.find(mf.first) != staticFuncs.end()) {
            PyErr_Format(PyExc_TypeError, "Class member %s repeated. This should have"
                                    " been compressed as an overload.", mf.first.c_str());
            return NULL;
        }
        staticFuncs[mf.first] = (Function*)mf.second;
    }

    std::map<std::string, PyObject*> clsMembers;

    for (auto mf: classMembers) {
        clsMembers[mf.first] = mf.second;
    }

    PyObject* typeObj = (PyObject*)native_instance_wrapper::typeObj(
        Class::Make(name, members, memberFuncs, staticFuncs, clsMembers)
        );

    Py_INCREF(typeObj);
    return typeObj;
}

PyObject *refcount(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "refcount takes 1 positional argument");
        return NULL;
    }

    PyObject* a1 = PyTuple_GetItem(args, 0);

    Type* actualType = native_instance_wrapper::extractTypeFrom(a1->ob_type);

    if (!actualType || actualType->getTypeCategory() != Type::TypeCategory::catTupleOf) {
        PyErr_Format(
            PyExc_TypeError,
            "first argument to serialize must be one of ConstDict, TupleOf, or Class, not %S",
            a1
            );
        return NULL;
    }

    if (actualType->getTypeCategory() == Type::TypeCategory::catTupleOf) {
        return PyLong_FromLong(
            ((::TupleOf*)actualType)->refcount(((native_instance_wrapper*)a1)->dataPtr())
            );
    }


    PyErr_Format(
        PyExc_TypeError,
        "this code should be unreachable"
        );
    return NULL;
}

class PythonSerializationContext : public SerializationContext {
public:
    //enums in our protocol (serialized as uint8_t)
    enum {
        T_NATIVE,   //a native type, followed by instance data
        T_OBJECT,   //an object written either with a representation, or as a type and a dict, or as a name
        T_LIST,
        T_TUPLE,
        T_DICT,
        T_NONE,
        T_TRUE,
        T_FALSE,
        T_UNICODE,
        T_BYTES,
        T_FLOAT,
        T_LONG,      //currently serialized as an int64_t which is completely wrong.
        T_NATIVETYPE,
        T_OBJECT_NAMED,
        T_OBJECT_TYPEANDDICT,
        T_OBJECT_REPRESENTATION
    };

    PythonSerializationContext(PyObject* typeSetObj) :
            mContextObj(typeSetObj)
    {
    }

    virtual void serializePythonObject(PyObject* o, SerializationBuffer& b) const {
        Type* t = native_instance_wrapper::extractTypeFrom(o->ob_type, true);

        if (t) {
            b.write_uint8(T_NATIVE);
            //we have a natural serialization mechanism already
            serializeNativeType(t, b);
            t->serialize(((native_instance_wrapper*)o)->dataPtr(), b);
        } else {
            if (PyDict_Check(o)) {
                b.write_uint8(T_DICT);
                serializePyDict(o, b);
            } else
            if (PyList_Check(o)) {
                b.write_uint8(T_LIST);
                serializePyList(o, b);
            } else
            if (PyTuple_Check(o)) {
                b.write_uint8(T_TUPLE);
                serializePyTuple(o, b);
            } else
            if (PyType_Check(o)) {
                Type* nativeType = native_instance_wrapper::extractTypeFrom((PyTypeObject*)o, true);

                if (nativeType) {
                    b.write_uint8(T_NATIVETYPE);
                    serializeNativeType(nativeType, b);
                    return;
                } else {
                    b.write_uint8(T_OBJECT);
                    serializePythonObjectNamedOrAsObj(o, b);
                }
            } else
            if (PyBytes_Check(o)) {
                b.write_uint8(T_BYTES);
                b.write_uint32(PyBytes_GET_SIZE(o));
                b.write_bytes((uint8_t*)PyBytes_AsString(o), PyBytes_GET_SIZE(o));
            } else
            if (PyUnicode_Check(o)) {
                b.write_uint8(T_UNICODE);
                Py_ssize_t sz;
                const char* c = PyUnicode_AsUTF8AndSize(o, &sz);
                b.write_uint32(sz);
                b.write_bytes((uint8_t*)c, sz);
            } else
            if (o == Py_None) {
                b.write_uint8(T_NONE);
            } else
            if (o == Py_True) {
                b.write_uint8(T_TRUE);
            } else
            if (o == Py_False) {
                b.write_uint8(T_FALSE);
            } else
            if (PyLong_Check(o)) {
                b.write_uint8(T_LONG);
                b.write_int64(PyLong_AsLong(o));
            } else
            if (PyFloat_Check(o)) {
                b.write_uint8(T_FLOAT);
                b.write_double(PyFloat_AsDouble(o));
            } else {
                b.write_uint8(T_OBJECT);
                serializePythonObjectNamedOrAsObj(o, b);
            }
        }
    }

    void serializePyDict(PyObject* o, SerializationBuffer& b) const {
        std::pair<uint32_t, bool> idAndIsNew = b.cachePointer(o);
        b.write_uint32(idAndIsNew.first);

        if (idAndIsNew.second) {
            b.write_uint32(PyDict_Size(o));

            PyObject *key, *value;
            Py_ssize_t pos = 0;

            int i = 0;

            while (PyDict_Next(o, &pos, &key, &value)) {
                serializePythonObject(key, b);
                serializePythonObject(value, b);
            }
        }
    }

    PyObject* deserializePydict(DeserializationBuffer& b) const {
        uint32_t id = b.read_uint32();

        PyObject* res = (PyObject*)b.lookupCachedPointer(id);
        if (res) {
            return incref(res);
        }

        res = PyDict_New();
        b.addCachedPointer(id, incref(res), true);

        size_t sz = b.read_uint32();

        try {
            for (long k = 0; k < sz; k++) {
                PyObject* key = deserializePythonObject(b);
                PyObject* value = deserializePythonObject(b);

                if (PyDict_SetItem(res, key, value) != 0) {
                    PyErr_Clear();
                    throw std::runtime_error(std::string("Can't place value of type ") + key->ob_type->tp_name + " into a dict");
                }

                Py_DECREF(key);
                Py_DECREF(value);
            }
        } catch(...) {
            Py_DECREF(res);
            throw;
        }

        return res;
    }

    void serializePyList(PyObject* o, SerializationBuffer& b) const {
        std::pair<uint32_t, bool> idAndIsNew = b.cachePointer(o);
        b.write_uint32(idAndIsNew.first);

        if (idAndIsNew.second) {
            size_t sz = PyList_Size(o);

            b.write_uint32(sz);

            for (long k = 0; k < sz; k++) {
                serializePythonObject(PyList_GetItem(o,k), b);
            }
        }
    }

    PyObject* deserializePyList(DeserializationBuffer& b) const {
        uint32_t id = b.read_uint32();

        PyObject* res = (PyObject*)b.lookupCachedPointer(id);
        if (res) {
            return incref(res);
        }

        size_t sz = b.read_uint32();

        res = PyList_New(sz);
        b.addCachedPointer(id, incref(res), true);

        try {
            for (long k = 0; k < sz; k++) {
                PyList_SET_ITEM(res, k, deserializePythonObject(b));
            }
        } catch(...) {
            Py_DECREF(res);
            throw;
        }

        return res;
    }

    void serializePyTuple(PyObject* o, SerializationBuffer& b) const {
        std::pair<uint32_t, bool> idAndIsNew = b.cachePointer(o);
        b.write_uint32(idAndIsNew.first);

        if (idAndIsNew.second) {
            size_t sz = PyTuple_Size(o);

            b.write_uint32(sz);

            for (long k = 0; k < sz; k++) {
                serializePythonObject(PyTuple_GetItem(o,k), b);
            }
        }
    }
    PyObject* deserializePyTuple(DeserializationBuffer& b) const {
        uint32_t id = b.read_uint32();

        PyObject* res = (PyObject*)b.lookupCachedPointer(id);
        if (res) {
            return incref(res);
        }

        size_t sz = b.read_uint32();

        res = PyTuple_New(sz);
        b.addCachedPointer(id, incref(res), true);

        try {
            for (long k = 0; k < sz; k++) {
                PyTuple_SET_ITEM(res, k, deserializePythonObject(b));
            }
        } catch(...) {
            Py_DECREF(res);
            throw;
        }

        return res;
    }

    void serializePythonObjectNamedOrAsObj(PyObject* o, SerializationBuffer& b) const {
        std::pair<uint32_t, bool> idAndIsNew = b.cachePointer(o);
        b.write_uint32(idAndIsNew.first);

        if (!idAndIsNew.second) {
            return;
        }

        //see if the object has a name
        PyObject* typeName = PyObject_CallMethod(mContextObj, "nameForObject", "O", o);
        if (!typeName) {
            PyErr_PrintEx(1);
            throw std::runtime_error("Python threw an exception internally when we called 'nameForObject'.");
        }
        if (typeName != Py_None) {
            if (!PyUnicode_Check(typeName)) {
                Py_DECREF(typeName);
                throw std::runtime_error(std::string("nameForObject returned a non-string"));
            }

            b.write_uint8(T_OBJECT_NAMED);
            b.write_string(std::string(PyUnicode_AsUTF8(typeName)));
            Py_DECREF(typeName);
            return;
        }
        Py_DECREF(typeName);

        //give the plugin a chance to convert the instance to something else
        PyObject* representation = PyObject_CallMethod(mContextObj, "representationFor", "O", o);
        if (!representation) {
            PyErr_PrintEx(1);
            throw std::runtime_error("representationFor threw an exception.");
        }

        if (representation == Py_None) {
            //we did nothing interesting
            b.write_uint8(T_OBJECT_TYPEANDDICT);
            serializePythonObject((PyObject*)o->ob_type, b);

            PyObject* d = PyObject_GenericGetDict(o, nullptr);
            if (!d) {
                throw std::runtime_error(std::string("Object of type ") + o->ob_type->tp_name + " had no dict.");
            }
            serializePyDict(d, b);
            Py_DECREF(d);
        } else {
            //we're writing a pair
            b.write_uint8(T_OBJECT_REPRESENTATION);

            if (!PyTuple_Check(representation) || PyTuple_Size(representation) != 3) {
                Py_DECREF(representation);
                throw std::runtime_error("representationFor should return None or a tuple with 3 things");
            }
            if (!PyTuple_Check(PyTuple_GetItem(representation, 1))) {
                Py_DECREF(representation);
                throw std::runtime_error("representationFor second arguments should be a tuple");
            }

            serializePythonObject(PyTuple_GetItem(representation, 0), b);
            serializePythonObject(PyTuple_GetItem(representation, 1), b);
            serializePythonObject(PyTuple_GetItem(representation, 2), b);
        }

        Py_DECREF(representation);
    }

    PyObject* deserializePythonObjectNamedOrAsObj(DeserializationBuffer& b) const {
        uint32_t id = b.read_uint32();
        PyObject* p = (PyObject*)b.lookupCachedPointer(id);
        if (p) {
            return incref(p);
        }

        uint8_t code = b.read_uint8();

        if (code == T_OBJECT_TYPEANDDICT) {
            //no representation
            PyObject* t = deserializePythonObject(b);
            if (!PyType_Check(t)) {
                std::string tname = t->ob_type->tp_name;
                Py_DECREF(t);
                throw std::runtime_error("Expected a type object. Got " + tname + " instead");
            }

            PyObject* result = ((PyTypeObject*)t)->tp_new(((PyTypeObject*)t), PyTuple_Pack(0), NULL);

            if (!result) {
                throw std::runtime_error("tp_new threw an exception");
            }

            b.addCachedPointer(id, incref(result), true);

            PyObject* d = deserializePydict(b);
            PyObject_GenericSetDict(result, d, nullptr);
            Py_DECREF(d);
            return result;
        } else
        if (code == T_OBJECT_REPRESENTATION) {
            PyObject* t = deserializePythonObject(b);
            PyObject* tArgs = deserializePythonObject(b);

            if (!PyTuple_Check(tArgs)) {
                throw std::runtime_error("corrupt data: second reconstruction argument is not a tuple.");
            }

            PyObject* instance = PyObject_Call(t, tArgs, NULL);
            Py_DECREF(t);
            Py_DECREF(tArgs);

            if (!instance) {
                PyErr_PrintEx(1);
                throw std::runtime_error(std::string("Representation constructor of type ") +
                        t->ob_type->tp_name + " threw an exception.");
            }

            b.addCachedPointer(id, incref(instance), true);

            PyObject* rep = deserializePythonObject(b);
            PyObject* res = PyObject_CallMethod(mContextObj, "setInstanceStateFromRepresentation", "OO", instance, rep);
            Py_DECREF(rep);

            if (!res) {
                PyErr_PrintEx(1);
                throw std::runtime_error("setInstanceStateFromRepresentation threw an exception.");
            }
            if (res != Py_True) {
                Py_DECREF(res);
                throw std::runtime_error("setInstanceStateFromRepresentation didn't return True.");
            }
            Py_DECREF(res);

            return instance;
        } else
        if (code == T_OBJECT_NAMED) {
            std::string name = b.readString();

            PyObject* res = PyObject_CallMethod(mContextObj, "objectFromName", "s", name.c_str());

            if (!res) {
                PyErr_PrintEx(1);
                throw std::runtime_error("objectFromName threw an exception.");
            }

            return b.addCachedPointer(id, incref(res), true);
        } else {
            throw std::runtime_error("corrupt data: invalid code after T_OBJECT");
        }
    }

    void serializeNativeType(Type* nativeType, SerializationBuffer& b) const {
        PyObject* nameForObject = PyObject_CallMethod(mContextObj, "nameForObject", "O", native_instance_wrapper::typeObj(nativeType));

        if (!nameForObject) {
            PyErr_PrintEx(1);
            throw std::runtime_error("nameForObject threw an exception.");
        }

        if (nameForObject != Py_None) {
            if (!PyUnicode_Check(nameForObject)) {
                PyErr_PrintEx(1);
                Py_DECREF(nameForObject);
                throw std::runtime_error("nameForObject returned something other than None or a string.");
            }

            b.write_uint8(0);
            b.write_string(std::string(PyUnicode_AsUTF8(nameForObject)));
            Py_DECREF(nameForObject);
            return;
        }
        b.write_uint8(1);

        b.write_uint8(nativeType->getTypeCategory());

        if (nativeType->getTypeCategory() == Type::TypeCategory::catInt64 ||
                nativeType->getTypeCategory() == Type::TypeCategory::catFloat64 ||
                nativeType->getTypeCategory() == Type::TypeCategory::catNone ||
                nativeType->getTypeCategory() == Type::TypeCategory::catBytes ||
                nativeType->getTypeCategory() == Type::TypeCategory::catString ||
                nativeType->getTypeCategory() == Type::TypeCategory::catBool
                ) {
            return;
        }

        if (nativeType->getTypeCategory() == Type::TypeCategory::catConcreteAlternative) {
            serializeNativeType(nativeType->getBaseType(), b);
            b.write_uint32(((ConcreteAlternative*)nativeType)->which());
            return;
        }

        if (nativeType->getTypeCategory() == Type::TypeCategory::catConstDict) {
            serializeNativeType(((ConstDict*)nativeType)->keyType(), b);
            serializeNativeType(((ConstDict*)nativeType)->valueType(), b);
            return;
        }

        if (nativeType->getTypeCategory() == Type::TypeCategory::catTupleOf) {
            serializeNativeType(((TupleOf*)nativeType)->getEltType(), b);
            return;
        }

        if (nativeType->getTypeCategory() == Type::TypeCategory::catTuple) {
            b.write_uint32(((CompositeType*)nativeType)->getTypes().size());
            for (auto t: ((CompositeType*)nativeType)->getTypes()) {
                serializeNativeType(t, b);
            }
            return;
        }

        if (nativeType->getTypeCategory() == Type::TypeCategory::catNamedTuple) {
            b.write_uint32(((CompositeType*)nativeType)->getTypes().size());
            for (long k = 0; k < ((CompositeType*)nativeType)->getTypes().size(); k++) {
                b.write_string(((CompositeType*)nativeType)->getNames()[k]);
                serializeNativeType(((CompositeType*)nativeType)->getTypes()[k], b);
            }
            return;
        }

        if (nativeType->getTypeCategory() == Type::TypeCategory::catOneOf) {
            b.write_uint32(((OneOf*)nativeType)->getTypes().size());
            for (auto t: ((OneOf*)nativeType)->getTypes()) {
                serializeNativeType(t, b);
            }
            return;
        }

        if (nativeType->getTypeCategory() == Type::TypeCategory::catPythonObjectOfType) {
            serializePythonObject((PyObject*)((PythonObjectOfType*)nativeType)->pyType(), b);
            return;
        }

        if (nativeType->getTypeCategory() == Type::TypeCategory::catValue) {
            Instance i = ((Value*)nativeType)->value();
            serializeNativeType(i.type(), b);
            i.type()->serialize(i.data(), b);
            return;
        }

        throw std::runtime_error("Can't serialize native type " + nativeType->name() + " if its unnamed.");
    }

    Type* deserializeNativeType(DeserializationBuffer& b) const {
        if (b.read_uint8() == 0) {
            //it's a named type
            std::string name = b.readString();
            PyObject* res = PyObject_CallMethod(mContextObj, "objectFromName", "s", name.c_str());

            if (!res) {
                PyErr_PrintEx(1);
                throw std::runtime_error("objectFromName threw an exception.");
            }

            if (!PyType_Check(res)) {
                std::string msg = "objectFromName returned a non-type for name " + name + ". it has type " + res->ob_type->tp_name;
                Py_DECREF(res);
                throw std::runtime_error(msg);
            }

            Type* resultType = native_instance_wrapper::extractTypeFrom((PyTypeObject*)res, true);
            Py_DECREF(res);
            if (!resultType) {
                throw std::runtime_error("we expected objectFromName to return a native type in this context, but it didn't.");
            }
            return resultType;
        }

        uint8_t category = b.read_uint8();
        if (category == Type::TypeCategory::catInt64) {
            return ::Int64::Make();
        }
        if (category == Type::TypeCategory::catFloat64) {
            return ::Float64::Make();
        }
        if (category == Type::TypeCategory::catNone) {
            return ::None::Make();
        }
        if (category == Type::TypeCategory::catBytes) {
            return ::Bytes::Make();
        }
        if (category == Type::TypeCategory::catString) {
            return ::String::Make();
        }
        if (category == Type::TypeCategory::catBool) {
            return ::Bool::Make();
        }

        if (category == Type::TypeCategory::catConcreteAlternative) {
            Type* base = deserializeNativeType(b);
            if (base->getTypeCategory() != Type::TypeCategory::catAlternative) {
                throw std::runtime_error("corrupt data: expected an Alternative type here");
            }
            uint32_t which = b.read_uint32();

            Alternative* a = (Alternative*)base;
            if (which >= a->subtypes().size()) {
                throw std::runtime_error("corrupt data: invalid alternative specified");
            }

            return ::ConcreteAlternative::Make(a,which);
        }


        if (category == Type::TypeCategory::catConstDict) {
            Type* keyType = deserializeNativeType(b);
            Type* valueType = deserializeNativeType(b);

            return ::ConstDict::Make(keyType, valueType);
        }

        if (category == Type::TypeCategory::catTupleOf) {
            return ::TupleOf::Make(
                deserializeNativeType(b)
                );
        }

        if (category == Type::TypeCategory::catTuple) {
            std::vector<Type*> types;
            size_t count = b.read_uint32();
            for (long k = 0; k < count; k++) {
                types.push_back(deserializeNativeType(b));
            }
            return ::Tuple::Make(types);
        }

        if (category == Type::TypeCategory::catNamedTuple) {
            std::vector<Type*> types;
            std::vector<std::string> names;
            size_t count = b.read_uint32();
            for (long k = 0; k < count; k++) {
                names.push_back(b.readString());
                types.push_back(deserializeNativeType(b));
            }
            return ::NamedTuple::Make(types, names);
        }

        if (category == Type::TypeCategory::catOneOf) {
            std::vector<Type*> types;
            size_t count = b.read_uint32();
            for (long k = 0; k < count; k++) {
                types.push_back(deserializeNativeType(b));
            }
            return ::OneOf::Make(types);
        }

        if (category == Type::TypeCategory::catPythonObjectOfType) {
            PyObject* typeObj = deserializePythonObject(b);
            return ::PythonObjectOfType::Make((PyTypeObject*)typeObj);
        }

        if (category == Type::TypeCategory::catValue) {
            Type* t = deserializeNativeType(b);

            Instance i = Instance::createAndInitialize(t, [&](instance_ptr p) {
                t->deserialize(p, b);
            });

            return ::Value::Make(i);
        }

        throw std::runtime_error("corrupt data. invalid native type code.");
    }

    virtual PyObject* deserializePythonObject(DeserializationBuffer& b) const {
        uint8_t code = b.read_uint8();
        if (code == T_LIST) {
            return deserializePyList(b);
        } else
        if (code == T_TUPLE) {
            return deserializePyTuple(b);
        } else
        if (code == T_DICT) {
            return deserializePydict(b);
        } else
        if (code == T_NATIVE) {
            Type* t = deserializeNativeType(b);
            return native_instance_wrapper::initialize(t, [&](instance_ptr selfData) {
                t->deserialize(selfData, b);
            });
        } else
        if (code == T_NATIVETYPE) {
            return (PyObject*)native_instance_wrapper::typeObj(deserializeNativeType(b));
        } else
        if (code == T_OBJECT) {
            return deserializePythonObjectNamedOrAsObj(b);
        } else
        if (code == T_NONE) {
            return incref(Py_None);
        } else
        if (code == T_TRUE) {
            return incref(Py_True);
        } else
        if (code == T_FALSE) {
            return incref(Py_False);
        } else
        if (code == T_UNICODE) {
            PyObject* result;
            uint32_t sz = b.read_uint32();
            b.read_bytes_fun(sz, [&](uint8_t* ptr) {
                result = PyUnicode_DecodeUTF8((const char*)ptr, sz, nullptr);
                if (!result) {
                    PyErr_PrintEx(1);
                    throw std::runtime_error("corrupt data: invalid unicode");
                }
            });
            return result;
        } else
        if (code == T_BYTES) {
            PyObject* result;
            uint32_t sz = b.read_uint32();
            b.read_bytes_fun(sz, [&](uint8_t* ptr) {
                result = PyBytes_FromStringAndSize((const char*)ptr, sz);
            });
            return result;
        } else
        if (code == T_FLOAT) {
            return PyFloat_FromDouble(b.read_double());
        } else if (code == T_LONG) {
            return PyLong_FromLong(b.read_int64());
        }

        throw std::runtime_error("corrupt data: invalid deserialization code");
    }


private:
    PyObject* mContextObj;
};


PyObject *serialize(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 2 && PyTuple_Size(args) != 3) {
        PyErr_SetString(PyExc_TypeError, "serialize takes 2 or 3 positional arguments");
        return NULL;
    }

    PyObject* a1 = PyTuple_GetItem(args, 0);
    PyObject* a2 = PyTuple_GetItem(args, 1);
    PyObject* a3 = PyTuple_Size(args) == 3 ? PyTuple_GetItem(args, 2) : nullptr;

    Type* serializeType = native_instance_wrapper::unwrapTypeArgToTypePtr(a1);

    if (!serializeType) {
        PyErr_Format(
            PyExc_TypeError,
            "first argument to serialize must be a native type object, not %S",
            a1
            );
        return NULL;
    }

    Type* actualType = native_instance_wrapper::extractTypeFrom(a2->ob_type, true);

    std::shared_ptr<SerializationContext> context(new NullSerializationContext());
    if (a3 && a3 != Py_None) {
        context.reset(new PythonSerializationContext(a3));
    }

    SerializationBuffer b(*context);

    try{
        if (actualType == serializeType) {
            //the simple case
            actualType->serialize(((native_instance_wrapper*)a2)->dataPtr(), b);
        } else {
            //try to construct a 'serialize type' from the argument and then serialize that
            Instance i = Instance::createAndInitialize(serializeType, [&](instance_ptr p) {
                native_instance_wrapper::copy_constructor(serializeType, p, a2);
            });

            i.type()->serialize(i.data(), b);
        }
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    }

    return PyBytes_FromStringAndSize((const char*)b.buffer(), b.size());
}

PyObject *deserialize(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 2 && PyTuple_Size(args) != 3) {
        PyErr_SetString(PyExc_TypeError, "deserialize takes 2 or 3 positional arguments");
        return NULL;
    }
    PyObject* a1 = PyTuple_GetItem(args, 0);
    PyObject* a2 = PyTuple_GetItem(args, 1);
    PyObject* a3 = PyTuple_Size(args) == 3 ? PyTuple_GetItem(args, 2) : nullptr;

    Type* serializeType = native_instance_wrapper::unwrapTypeArgToTypePtr(a1);

    if (!serializeType) {
        PyErr_SetString(PyExc_TypeError, "first argument to serialize must be a native type object");
        return NULL;
    }
    if (!PyBytes_Check(a2)) {
        PyErr_SetString(PyExc_TypeError, "second argument to serialize must be a bytes object");
        return NULL;
    }

    std::shared_ptr<SerializationContext> context(new NullSerializationContext());
    if (a3 && a3 != Py_None) {
        context.reset(new PythonSerializationContext(a3));
    }
    SerializationBuffer b(*context);

    DeserializationBuffer buf((uint8_t*)PyBytes_AsString(a2), PyBytes_GET_SIZE(a2), *context);

    try {
        Instance i = Instance::createAndInitialize(serializeType, [&](instance_ptr p) {
            serializeType->deserialize(p, buf);
        });

        return native_instance_wrapper::extractPythonObject(i.data(), i.type());
    } catch(std::exception& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    }
}

PyObject *bytecount(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "bytecount takes 1 positional argument");
        return NULL;
    }
    PyObject* a1 = PyTuple_GetItem(args, 0);

    Type* t = native_instance_wrapper::unwrapTypeArgToTypePtr(a1);

    if (!t) {
        PyErr_SetString(PyExc_TypeError, "first argument to 'bytecount' must be a native type object");
        return NULL;
    }

    return PyLong_FromLong(t->bytecount());
}

PyObject *resolveForwards(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "resolveForwards takes 1 positional argument");
        return NULL;
    }

    PyObject* a1 = PyTuple_GetItem(args, 0);

    Type* t1 = native_instance_wrapper::unwrapTypeArgToTypePtr(a1);

    if (!t1) {
        PyErr_SetString(PyExc_TypeError, "first argument to 'resolveForwards' must be a native type object");
        return NULL;
    }

    if (native_instance_wrapper::guaranteeForwardsResolved(t1)) {
        return incref(Py_True);
    }

    return NULL;
}

PyObject *disableNativeDispatch(PyObject* nullValue, PyObject* args) {
    native_dispatch_disabled++;
    return incref(Py_None);
}

PyObject *enableNativeDispatch(PyObject* nullValue, PyObject* args) {
    native_dispatch_disabled++;
    return incref(Py_None);
}

PyObject *installNativeFunctionPointer(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 3) {
        PyErr_SetString(PyExc_TypeError, "installNativeFunctionPointer takes 3 positional arguments");
        return NULL;
    }
    PyObject* a1 = PyTuple_GetItem(args, 0);
    PyObject* a2 = PyTuple_GetItem(args, 1);
    PyObject* a3 = PyTuple_GetItem(args, 2);

    Type* t1 = native_instance_wrapper::unwrapTypeArgToTypePtr(a1);

    if (!t1 || t1->getTypeCategory() != Type::TypeCategory::catFunction) {
        PyErr_SetString(PyExc_TypeError, "first argument to 'installNativeFunctionPointer' must be a Function");
        return NULL;
    }

    if (!PyLong_Check(a2)) {
        PyErr_SetString(PyExc_TypeError, "second argument to 'installNativeFunctionPointer' must be an integer 'index'");
        return NULL;
    }

    if (!PyLong_Check(a3)) {
        PyErr_SetString(PyExc_TypeError, "third argument to 'installNativeFunctionPointer' must be an integer function pointer");
        return NULL;
    }

    Function* f = (Function*)t1;

    int index = PyLong_AsLong(a2);
    size_t ptr = PyLong_AsSize_t(a3);

    if (index < 0 || index >= f->getOverloads().size()) {
        PyErr_SetString(PyExc_TypeError, "index is out of bounds");
        return NULL;
    }

    f->setEntrypoint(index,(compiled_code_entrypoint)ptr);

    return incref(Py_None);
}

PyObject *isBinaryCompatible(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "isBinaryCompatible takes 2 positional arguments");
        return NULL;
    }
    PyObject* a1 = PyTuple_GetItem(args, 0);
    PyObject* a2 = PyTuple_GetItem(args, 1);

    Type* t1 = native_instance_wrapper::unwrapTypeArgToTypePtr(a1);
    Type* t2 = native_instance_wrapper::unwrapTypeArgToTypePtr(a2);

    if (!t1) {
        PyErr_SetString(PyExc_TypeError, "first argument to 'isBinaryCompatible' must be a native type object");
        return NULL;
    }
    if (!t2) {
        PyErr_SetString(PyExc_TypeError, "second argument to 'isBinaryCompatible' must be a native type object");
        return NULL;
    }

    PyObject* res = t1->isBinaryCompatibleWith(t2) ? Py_True : Py_False;
    Py_INCREF(res);
    return res;
}

PyObject *MakeAlternativeType(PyObject* nullValue, PyObject* args, PyObject* kwargs) {
    if (PyTuple_Size(args) != 1 || !PyUnicode_Check(PyTuple_GetItem(args,0))) {
        PyErr_SetString(PyExc_TypeError, "Alternative takes a single string positional argument.");
        return NULL;
    }

    std::string name = PyUnicode_AsUTF8(PyTuple_GetItem(args,0));

    std::vector<std::pair<std::string, NamedTuple*> > definitions;

    std::map<std::string, Function*> functions;

    PyObject *key, *value;
    Py_ssize_t pos = 0;

    int i = 0;

    while (kwargs && PyDict_Next(kwargs, &pos, &key, &value)) {
        assert(PyUnicode_Check(key));

        std::string fieldName(PyUnicode_AsUTF8(key));

        if (PyFunction_Check(value)) {
            functions[fieldName] = convertPythonObjectToFunction(key, value);
            if (functions[fieldName] == nullptr) {
                //error code is already set
                return nullptr;
            }
        }
        else {
            if (!PyDict_Check(value)) {
                PyErr_SetString(PyExc_TypeError, "Alternative members must be initialized with dicts.");
                return NULL;
            }

            PyObject* ntPyPtr = MakeNamedTupleType(nullptr, nullptr, value);
            if (!ntPyPtr) {
                return NULL;
            }

            NamedTuple* ntPtr = (NamedTuple*)native_instance_wrapper::extractTypeFrom((PyTypeObject*)ntPyPtr);

            assert(ntPtr);

            definitions.push_back(std::make_pair(fieldName, ntPtr));
        }
    };

    static_assert(PY_MAJOR_VERSION >= 3, "nativepython is a python3 project only");

    if (PY_MINOR_VERSION <= 5) {
        //we cannot rely on the ordering of 'kwargs' here because of the python version, so
        //we sort it. this will be a problem for anyone running some processes using different
        //python versions that share python code.
        std::sort(definitions.begin(), definitions.end());
    }

    PyObject* res = (PyObject*)native_instance_wrapper::typeObj(
        ::Alternative::Make(name, definitions, functions)
        );

    Py_INCREF(res);
    return res;
}

static PyMethodDef module_methods[] = {
    {"NoneType", (PyCFunction)MakeNoneTypeType, METH_VARARGS, NULL},
    {"Bool", (PyCFunction)MakeBoolType, METH_VARARGS, NULL},
    {"Int8", (PyCFunction)MakeInt8Type, METH_VARARGS, NULL},
    {"Int16", (PyCFunction)MakeInt16Type, METH_VARARGS, NULL},
    {"Int32", (PyCFunction)MakeInt32Type, METH_VARARGS, NULL},
    {"Int64", (PyCFunction)MakeInt64Type, METH_VARARGS, NULL},
    {"UInt8", (PyCFunction)MakeUInt8Type, METH_VARARGS, NULL},
    {"UInt16", (PyCFunction)MakeUInt16Type, METH_VARARGS, NULL},
    {"UInt32", (PyCFunction)MakeUInt32Type, METH_VARARGS, NULL},
    {"UInt64", (PyCFunction)MakeUInt64Type, METH_VARARGS, NULL},
    {"Float32", (PyCFunction)MakeFloat32Type, METH_VARARGS, NULL},
    {"Float64", (PyCFunction)MakeFloat64Type, METH_VARARGS, NULL},
    {"String", (PyCFunction)MakeStringType, METH_VARARGS, NULL},
    {"Bytes", (PyCFunction)MakeBytesType, METH_VARARGS, NULL},
    {"TupleOf", (PyCFunction)MakeTupleOfType, METH_VARARGS, NULL},
    {"Tuple", (PyCFunction)MakeTupleType, METH_VARARGS, NULL},
    {"NamedTuple", (PyCFunction)MakeNamedTupleType, METH_VARARGS | METH_KEYWORDS, NULL},
    {"OneOf", (PyCFunction)MakeOneOfType, METH_VARARGS, NULL},
    {"ConstDict", (PyCFunction)MakeConstDictType, METH_VARARGS, NULL},
    {"Alternative", (PyCFunction)MakeAlternativeType, METH_VARARGS | METH_KEYWORDS, NULL},
    {"Value", (PyCFunction)MakeValueType, METH_VARARGS, NULL},
    {"Class", (PyCFunction)MakeClassType, METH_VARARGS, NULL},
    {"Function", (PyCFunction)MakeFunctionType, METH_VARARGS, NULL},
    {"TypeFor", (PyCFunction)MakeTypeFor, METH_VARARGS, NULL},
    {"serialize", (PyCFunction)serialize, METH_VARARGS, NULL},
    {"deserialize", (PyCFunction)deserialize, METH_VARARGS, NULL},
    {"bytecount", (PyCFunction)bytecount, METH_VARARGS, NULL},
    {"isBinaryCompatible", (PyCFunction)isBinaryCompatible, METH_VARARGS, NULL},
    {"resolveForwards", (PyCFunction)resolveForwards, METH_VARARGS, NULL},
    {"installNativeFunctionPointer", (PyCFunction)installNativeFunctionPointer, METH_VARARGS, NULL},
    {"disableNativeDispatch", (PyCFunction)disableNativeDispatch, METH_VARARGS, NULL},
    {"enableNativeDispatch", (PyCFunction)enableNativeDispatch, METH_VARARGS, NULL},
    {"refcount", (PyCFunction)refcount, METH_VARARGS, NULL},
    {NULL, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_types",
    NULL,
    0,
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

void updateTypeRepForType(Type* type, PyTypeObject* pyType) {
    //deliberately leak the name.
    pyType->tp_name = (new std::string(type->name()))->c_str();

    native_instance_wrapper::mirrorTypeInformationIntoPyType(type, pyType);
}


PyMODINIT_FUNC
PyInit__types(void)
{
    PyObject *module = PyModule_Create(&moduledef);

    if (module == NULL)
        return NULL;

    return module;
}

