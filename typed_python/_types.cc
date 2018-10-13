#include "Python.h"
#include <map>
#include <memory>
#include <vector>
#include <string>
#include "Type.hpp"
#include <iostream>

static_assert(PY_MAJOR_VERSION >= 3, "nativepython is a python3 project only");


//extension of PyTypeObject that stashes a Type* on the end.
struct NativeTypeWrapper {
    PyTypeObject typeObj;
    const Type* mType;
};

class InternalPyException {};

struct native_instance_wrapper {
    PyObject_HEAD
  
    bool mIsInitialized;
    bool mIsMatcher; //-1 if we're not an iterator
    bool mIteratorIsPairs;
    int64_t mIteratorOffset; //-1 if we're not an iterator
    uint8_t data[0];

    static PyObject* bytecount(PyObject* o) {
        NativeTypeWrapper* w = (NativeTypeWrapper*)o;
        return PyLong_FromLong(w->mType->bytecount());
    }

    static PyObject* constDictItems(PyObject *o) {
        const Type* self_type = extractTypeFrom(o->ob_type);
        native_instance_wrapper* w = (native_instance_wrapper*)o;

        if (self_type && self_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
            native_instance_wrapper* self = (native_instance_wrapper*)o->ob_type->tp_alloc(o->ob_type, 0);

            self->mIteratorOffset = 0;
            self->mIteratorIsPairs = 1;
            self_type->copy_constructor(self->data, w->data);
            self->mIsInitialized = true;
            self->mIsMatcher = false;

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

        const Type* self_type = extractTypeFrom(o->ob_type);
        const Type* item_type = extractTypeFrom(item->ob_type);
        
        if (self_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
            ConstDict* dict_t = (ConstDict*)self_type;

            if (item_type == dict_t->keyType()) {
                native_instance_wrapper* item_w = (native_instance_wrapper*)item;

                instance_ptr i = dict_t->lookupValueByKey(self_w->data, item_w->data);
                
                if (!i) {
                    Py_INCREF(ifNotFound);
                    return ifNotFound;
                }

                return extractPythonObject(i, dict_t->valueType());
            } else {
                instance_ptr tempObj = (instance_ptr)malloc(dict_t->keyType()->bytecount());
                try {
                    copy_initialize(dict_t->keyType(), tempObj, item);
                } catch(std::exception& e) {
                    free(tempObj);
                    PyErr_SetString(PyExc_TypeError, e.what());
                    return NULL;
                }

                instance_ptr i = dict_t->lookupValueByKey(self_w->data, tempObj);

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

    static PyMethodDef* typeMethods(const Type* t) {
        if (t->getTypeCategory() == Type::TypeCategory::catConstDict) {
            return new PyMethodDef [4] {
                {"bytecount", (PyCFunction)native_instance_wrapper::bytecount, METH_CLASS | METH_NOARGS, NULL},
                {"get", (PyCFunction)native_instance_wrapper::constDictGet, METH_VARARGS, NULL},
                {"items", (PyCFunction)native_instance_wrapper::constDictItems, METH_NOARGS, NULL},
                {NULL, NULL}
            };
        }

        return new PyMethodDef [2] {
            {"bytecount", (PyCFunction)native_instance_wrapper::bytecount, METH_CLASS | METH_NOARGS, NULL},
            {NULL, NULL}
        };
    };

    static void tp_dealloc(PyObject* self) {
        native_instance_wrapper* wrapper = (native_instance_wrapper*)self;

        if (wrapper->mIsInitialized) {
            extractTypeFrom(self->ob_type)->destroy(wrapper->data);
        }

        Py_TYPE(self)->tp_free((PyObject*)self);
    }

    static bool pyValCouldBeOfType(const Type* t, PyObject* pyRepresentation) {
        if (t->getTypeCategory() == Type::TypeCategory::catValue) {
            Value* valType = (Value*)t;
            if (compare_to_python(valType->value().type(), valType->value().data(), pyRepresentation, true) == 0) {
                return true;
            } else {
                return false;
            }
        }

        const Type* argType = extractTypeFrom(pyRepresentation->ob_type);
        if (argType) {
            return argType == t || argType == t->getBaseType();
        }

        if (t->getTypeCategory() == Type::TypeCategory::catNamedTuple || 
                t->getTypeCategory() == Type::TypeCategory::catTupleOf || 
                t->getTypeCategory() == Type::TypeCategory::catTuple
                ) {
            return PyTuple_Check(pyRepresentation);
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

    static void copy_initialize(const Type* eltType, instance_ptr tgt, PyObject* pyRepresentation) {
        const Type* argType = extractTypeFrom(pyRepresentation->ob_type);

        if ((argType && argType->getBaseType() == eltType) || argType == eltType) {
            //it's already the right kind of instance
            eltType->copy_constructor(tgt, ((native_instance_wrapper*)pyRepresentation)->data);
            return;
        }

        Type::TypeCategory cat = eltType->getTypeCategory();

        if (cat == Type::TypeCategory::catPythonSubclass) {
            copy_initialize((const Type*)eltType->getBaseType(), tgt, pyRepresentation);
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
                const Type* subtype = oneOf->getTypes()[k];

                if (pyValCouldBeOfType(subtype, pyRepresentation)) {
                    try {
                        copy_initialize(subtype, tgt+1, pyRepresentation);
                        *(uint8_t*)tgt = k;
                        return;
                    } catch(...) {}
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
                        copy_initialize(dictType->keyType(), dictType->kvPairPtrKey(tgt, i), key);
                        try {
                            copy_initialize(dictType->valueType(), dictType->kvPairPtrValue(tgt, i), value);
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
                        copy_initialize(((TupleOf*)eltType)->getEltType(), eltPtr, PyTuple_GetItem(pyRepresentation,k));
                        }
                    );
                return;
            }
            if (PyList_Check(pyRepresentation)) {
                ((TupleOf*)eltType)->constructor(tgt, PyList_Size(pyRepresentation), 
                    [&](uint8_t* eltPtr, int64_t k) {
                        copy_initialize(((TupleOf*)eltType)->getEltType(), eltPtr, PyList_GetItem(pyRepresentation,k));
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
                        copy_initialize(((TupleOf*)eltType)->getEltType(), eltPtr, item);
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
                        copy_initialize(((CompositeType*)eltType)->getTypes()[k], eltPtr, PyTuple_GetItem(pyRepresentation,k));
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
                        copy_initialize(((CompositeType*)eltType)->getTypes()[k], eltPtr, PyList_GetItem(pyRepresentation,k));
                        }
                    );
                return;
            }
        }

        throw std::logic_error("Couldn't initialize internal elt of type " + eltType->name() + " from " + pyRepresentation->ob_type->tp_name);
    }

    static void initialize(uint8_t* data, const Type* t, PyObject* args, PyObject* kwargs) {
        Type::TypeCategory cat = t->getTypeCategory();

        if (cat == Type::TypeCategory::catPythonSubclass) {
            initialize(data, (const Type*)t->getBaseType(), args, kwargs);
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

                copy_initialize(t, data, argTuple);

                return;
            }

            throw std::logic_error("Can't initialize " + t->name() + " with these in-place arguments.");
        } else {
            if (cat == Type::TypeCategory::catNamedTuple) {
                CompositeType* compositeT = ((CompositeType*)t);

                compositeT->constructor(
                    data, 
                    [&](uint8_t* eltPtr, int64_t k) {
                        const Type* eltType = compositeT->getTypes()[k];
                        PyObject* o = PyDict_GetItemString(kwargs, compositeT->getNames()[k].c_str());
                        if (o) {
                            copy_initialize(eltType, eltPtr, o);
                        }
                        else if (eltType->is_default_constructible()) {
                            eltType->constructor(eltPtr);
                        } else {
                            throw std::logic_error("Can't default initialize argument " + compositeT->getNames()[k]);
                        }
                    });
                return;
            }

            throw std::logic_error("Can't initialize " + t->name() + " from python with kwargs.");
        }
    }


    //produce the pythonic representation of this object. for things like integers, string, etc,
    //convert them back to their python-native form. otherwise, a pointer back into a native python
    //structure
    static PyObject* extractPythonObject(instance_ptr data, const Type* eltType) {
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
            std::pair<const Type*, instance_ptr> child = ((OneOf*)eltType)->unwrap(data);
            return extractPythonObject(child.second, child.first);
        }

        const Type* concreteT = eltType->pickConcreteSubclass(data);

        native_instance_wrapper* self = (native_instance_wrapper*)typeObj(concreteT)->tp_alloc(typeObj(concreteT), 0);

        try {
            self->mIteratorOffset = -1;
            self->mIsInitialized = false;
            self->mIsMatcher = false;

            concreteT->copy_constructor(self->data, data);

            self->mIsInitialized = true;

            return (PyObject*)self;
        } catch(std::exception& e) {
            typeObj(concreteT)->tp_dealloc((PyObject*)self);

            PyErr_SetString(PyExc_TypeError, e.what());
            return NULL;
        }
    }

    static PyObject *tp_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds) {
        const Type* eltType = extractTypeFrom(subtype);

        if (isSubclassOfNativeType(subtype)) {
            native_instance_wrapper* self = (native_instance_wrapper*)subtype->tp_alloc(subtype, 0);

            try {
                self->mIteratorOffset = -1;
                self->mIsInitialized = false;
                self->mIsMatcher = false;

                initialize(self->data, eltType, args, kwds);

                self->mIsInitialized = true;

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

        const Type* t = extractTypeFrom(o->ob_type);

        if (t->getTypeCategory() == Type::TypeCategory::catTupleOf) {
            return ((TupleOf*)t)->count(w->data);
        }
        if (t->isComposite()) {
            return ((CompositeType*)t)->getTypes().size();
        }
        if (t->getTypeCategory() == Type::TypeCategory::catString) {
            return String().count(w->data);
        }
        if (t->getTypeCategory() == Type::TypeCategory::catBytes) {
            return Bytes().count(w->data);
        }

        return 0;
    }

    static PyObject* nb_subtract(PyObject* lhs, PyObject* rhs) {
        const Type* lhs_type = extractTypeFrom(lhs->ob_type);
        const Type* rhs_type = extractTypeFrom(rhs->ob_type);

        if (lhs_type) {
            native_instance_wrapper* w_lhs = (native_instance_wrapper*)lhs;

            if (lhs_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
                ConstDict* dict_t = (ConstDict*)lhs_type;

                const Type* tupleOfKeysType = dict_t->tupleOfKeysType();

                if (lhs_type == tupleOfKeysType) {
                    native_instance_wrapper* w_rhs = (native_instance_wrapper*)rhs;

                    native_instance_wrapper* self = 
                        (native_instance_wrapper*)typeObj(lhs_type)->tp_alloc(typeObj(lhs_type), 0);

                    ((ConstDict*)lhs_type)->subtractTupleOfKeysFromDict(w_lhs->data, w_rhs->data, self->data);

                    return (PyObject*)self;
                } else {
                    //attempt to convert rhs to a relevant dict type.
                    instance_ptr tempObj = (instance_ptr)malloc(tupleOfKeysType->bytecount());

                    try {
                        copy_initialize(tupleOfKeysType, tempObj, rhs);
                    } catch(std::exception& e) {
                        free(tempObj);
                        PyErr_SetString(PyExc_TypeError, e.what());
                        return NULL;
                    }

                    native_instance_wrapper* self = 
                        (native_instance_wrapper*)typeObj(lhs_type)->tp_alloc(typeObj(lhs_type), 0);

                    ((ConstDict*)lhs_type)->subtractTupleOfKeysFromDict(w_lhs->data, tempObj, self->data);

                    tupleOfKeysType->destroy(tempObj);

                    free(tempObj);

                    return (PyObject*)self;                    
                }
            }
        }

        PyErr_SetString(
            PyExc_TypeError, 
            (std::string("cannot subtract ") + rhs->ob_type->tp_name + " from "
                    + lhs->ob_type->tp_name).c_str()
            );
        return NULL;
    }

    static PyObject* sq_concat(PyObject* lhs, PyObject* rhs) {
        const Type* lhs_type = extractTypeFrom(lhs->ob_type);
        const Type* rhs_type = extractTypeFrom(rhs->ob_type);

        if (lhs_type) {
            native_instance_wrapper* w_lhs = (native_instance_wrapper*)lhs;

            if (lhs_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
                if (lhs_type == rhs_type) {
                    native_instance_wrapper* w_rhs = (native_instance_wrapper*)rhs;

                    native_instance_wrapper* self = 
                        (native_instance_wrapper*)typeObj(lhs_type)->tp_alloc(typeObj(lhs_type), 0);

                    ((ConstDict*)lhs_type)->addDicts(w_lhs->data, w_rhs->data, self->data);

                    return (PyObject*)self;
                } else {
                    //attempt to convert rhs to a relevant dict type.
                    instance_ptr tempObj = (instance_ptr)malloc(lhs_type->bytecount());

                    try {
                        copy_initialize(lhs_type, tempObj, rhs);
                    } catch(std::exception& e) {
                        free(tempObj);
                        PyErr_SetString(PyExc_TypeError, e.what());
                        return NULL;
                    }

                    native_instance_wrapper* self = 
                        (native_instance_wrapper*)typeObj(lhs_type)->tp_alloc(typeObj(lhs_type), 0);

                    ((ConstDict*)lhs_type)->addDicts(w_lhs->data, tempObj, self->data);

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
                    const Type* eltType = tupT->getEltType();
                    native_instance_wrapper* self = 
                        (native_instance_wrapper*)typeObj(tupT)
                            ->tp_alloc(typeObj(tupT), 0);

                    int count_lhs = tupT->count(w_lhs->data);
                    int count_rhs = tupT->count(w_rhs->data);

                    tupT->constructor(self->data, count_lhs + count_rhs, 
                        [&](uint8_t* eltPtr, int64_t k) {
                            eltType->copy_constructor(
                                eltPtr, 
                                k < count_lhs ? tupT->eltPtr(w_lhs->data, k) : 
                                    tupT->eltPtr(w_rhs->data, k - count_lhs)
                                );
                            }
                        );

                    return (PyObject*)self;
                }
                //generic path to add any kind of iterable.
                if (PyObject_Length(rhs) != -1) {
                    TupleOf* tupT = (TupleOf*)lhs_type;
                    const Type* eltType = tupT->getEltType();

                    native_instance_wrapper* self = 
                        (native_instance_wrapper*)typeObj(tupT)
                            ->tp_alloc(typeObj(tupT), 0);

                    int count_lhs = tupT->count(w_lhs->data);
                    int count_rhs = PyObject_Length(rhs);

                    try {
                        tupT->constructor(self->data, count_lhs + count_rhs, 
                            [&](uint8_t* eltPtr, int64_t k) {
                                if (k < count_lhs) {
                                    eltType->copy_constructor(
                                        eltPtr, 
                                        tupT->eltPtr(w_lhs->data, k)
                                        );
                                } else {
                                    PyObject* kval = PyLong_FromLong(k - count_lhs);
                                    PyObject* o = PyObject_GetItem(rhs, kval);
                                    Py_DECREF(kval);

                                    if (!o) {
                                        throw InternalPyException();
                                    }
                                    
                                    try {
                                        copy_initialize(eltType, eltPtr, o);
                                    } catch(...) {
                                        Py_DECREF(o);
                                        throw;
                                    }

                                    Py_DECREF(o);
                                }
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
        const Type* t = extractTypeFrom(o->ob_type);

        if (t->getTypeCategory() == Type::TypeCategory::catTupleOf) {
            int64_t count = ((TupleOf*)t)->count(w->data);

            if (ix < 0) {
                ix += count;
            }

            if (ix >= count || ix < 0) {
                PyErr_SetString(PyExc_IndexError, "index out of range");
                return NULL;
            }

            const Type* eltType = (const Type*)((TupleOf*)t)->getEltType();
            return extractPythonObject(
                ((TupleOf*)t)->eltPtr(w->data, ix), 
                eltType
                );
        }
        
        if (t->isComposite()) {
            auto compType = (CompositeType*)t;

            if (ix < 0 || ix >= (int64_t)compType->getTypes().size()) {
                PyErr_SetString(PyExc_IndexError, "index out of range");
                return NULL;
            }

            const Type* eltType = compType->getTypes()[ix];

            return extractPythonObject(
                compType->eltPtr(w->data, ix), 
                eltType
                );
        }

        if (t->getTypeCategory() == Type::TypeCategory::catBytes) {
            if (ix < 0 || ix >= (int64_t)Bytes().count(w->data)) {
                PyErr_SetString(PyExc_IndexError, "index out of range");
                return NULL;
            }

            return PyBytes_FromStringAndSize(
                (const char*)Bytes().eltPtr(w->data, ix),
                1
                );
        }
        if (t->getTypeCategory() == Type::TypeCategory::catString) {
            if (ix < 0 || ix >= (int64_t)String().count(w->data)) {
                PyErr_SetString(PyExc_IndexError, "index out of range");
                return NULL;
            }

            int bytes_per_codepoint = String().bytes_per_codepoint(w->data);

            return PyUnicode_FromKindAndData(
                bytes_per_codepoint == 1 ? PyUnicode_1BYTE_KIND :
                bytes_per_codepoint == 2 ? PyUnicode_2BYTE_KIND :
                                           PyUnicode_4BYTE_KIND,
                String().eltPtr(w->data, ix),
                1
                );
        }

        PyErr_SetString(PyExc_TypeError, "not a __getitem__'able thing.");
        return NULL;
    }

    static PyTypeObject* typeObj(const Type* inType) {
        if (!inType->getTypeRep()) {
            inType->setTypeRep(typeObjInternal(inType));
        }
            
        return inType->getTypeRep();
    }

    static PySequenceMethods* sequenceMethodsFor(const Type* t) {
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

    static PyNumberMethods* numberMethods(const Type* t) {
        static PyNumberMethods* res = 
            new PyNumberMethods {
                0, //binaryfunc nb_add
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
                0, //binaryfunc nb_rshift
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

        return res;
    }

    static Py_ssize_t mp_length(PyObject* o) {
        native_instance_wrapper* w = (native_instance_wrapper*)o;

        const Type* t = extractTypeFrom(o->ob_type);

        if (t->getTypeCategory() == Type::TypeCategory::catConstDict) {
            return ((ConstDict*)t)->size(w->data);
        }

        if (t->getTypeCategory() == Type::TypeCategory::catTupleOf) {
            return ((TupleOf*)t)->count(w->data);
        }

        return 0;
    }

    static int sq_contains(PyObject* o, PyObject* item) {
        native_instance_wrapper* self_w = (native_instance_wrapper*)o;

        const Type* self_type = extractTypeFrom(o->ob_type);
        const Type* item_type = extractTypeFrom(item->ob_type);

        if (self_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
            ConstDict* dict_t = (ConstDict*)self_type;

            if (item_type == dict_t->keyType()) {
                native_instance_wrapper* item_w = (native_instance_wrapper*)item;

                instance_ptr i = dict_t->lookupValueByKey(self_w->data, item_w->data);
                
                if (!i) {
                    return 0;
                }

                return 1;
            } else {
                instance_ptr tempObj = (instance_ptr)malloc(dict_t->keyType()->bytecount());
                try {
                    copy_initialize(dict_t->keyType(), tempObj, item);
                } catch(std::exception& e) {
                    free(tempObj);
                    PyErr_SetString(PyExc_TypeError, e.what());
                    return -1;
                }

                instance_ptr i = dict_t->lookupValueByKey(self_w->data, tempObj);

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

        const Type* self_type = extractTypeFrom(o->ob_type);
        const Type* item_type = extractTypeFrom(item->ob_type);

        if (self_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
            ConstDict* dict_t = (ConstDict*)self_type;

            if (item_type == dict_t->keyType()) {
                native_instance_wrapper* item_w = (native_instance_wrapper*)item;

                instance_ptr i = dict_t->lookupValueByKey(self_w->data, item_w->data);
                
                if (!i) {
                    PyErr_SetObject(PyExc_KeyError, item);
                    return NULL;
                }

                return extractPythonObject(i, dict_t->valueType());
            } else {
                instance_ptr tempObj = (instance_ptr)malloc(dict_t->keyType()->bytecount());
                try {
                    copy_initialize(dict_t->keyType(), tempObj, item);
                } catch(std::exception& e) {
                    free(tempObj);
                    PyErr_SetString(PyExc_TypeError, e.what());
                    return NULL;
                }

                instance_ptr i = dict_t->lookupValueByKey(self_w->data, tempObj);

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
                if (PySlice_GetIndicesEx(item, tupType->count(self_w->data), &start,
                            &stop, &step, &slicelength) == -1) {
                    return NULL;
                }

                const Type* eltType = tupType->getEltType();

                native_instance_wrapper* result = 
                    (native_instance_wrapper*)typeObj(tupType)->tp_alloc(typeObj(tupType), 0);

                tupType->constructor(result->data, slicelength, 
                    [&](uint8_t* eltPtr, int64_t k) {
                        eltType->copy_constructor(
                            eltPtr, 
                            tupType->eltPtr(self_w->data, start + k * step)
                            );
                        }
                    );

                return (PyObject*)result;
            }

            if (PyLong_Check(item)) {
                return sq_item((PyObject*)self_w, PyLong_AsLong(item));
            }
        }

        PyErr_SetObject(PyExc_KeyError, item);
        return NULL;
    }

    static PyMappingMethods* mappingMethods(const Type* t) {
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

    static const Type* extractTypeFrom(PyTypeObject* typeObj) {
        while (typeObj->tp_base && typeObj->tp_as_buffer != bufferProcs()) {
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

        const Type* type = extractTypeFrom(o->ob_type);

        if (type->getTypeCategory() == Type::TypeCategory::catClass) {
            native_instance_wrapper* self_w = (native_instance_wrapper*)o;
            Class* nt = (Class*)type;

            int i = nt->memberNamed(PyUnicode_AsUTF8(attrName));
            
            if (i < 0) {
                PyErr_Format(PyExc_AttributeError, "Instance of type %S has no attribute '%S'", o->ob_type, attrName);
                return -1;
            }

            const Type* eltType = nt->getMembers()[i].second;

            const Type* attrType = extractTypeFrom(attrVal->ob_type);

            if (eltType == attrType) {
                native_instance_wrapper* item_w = (native_instance_wrapper*)attrVal;

                attrType->assign(nt->eltPtr(self_w->data, i), item_w->data);

                return 0;
            } else {
                instance_ptr tempObj = (instance_ptr)malloc(eltType->bytecount());
                try {
                    copy_initialize(eltType, tempObj, attrVal);
                } catch(std::exception& e) {
                    free(tempObj);
                    PyErr_SetString(PyExc_TypeError, e.what());
                    return -1;
                }

                eltType->assign(nt->eltPtr(self_w->data, i), tempObj);

                eltType->destroy(tempObj);
                free(tempObj);

                return 0;
            }
        }

        PyErr_Format(PyExc_AttributeError, "Instance of type %S has no attribute '%S'", o->ob_type, attrName);
        return -1;
    }

    static PyObject* tp_getattro(PyObject *o, PyObject* attrName) {
        if (!PyUnicode_Check(attrName)) {
            PyErr_SetString(PyExc_AttributeError, "attribute is not a string");
            return NULL;
        }

        char *attr_name = PyUnicode_AsUTF8(attrName);

        const Type* t = extractTypeFrom(o->ob_type);
        
        native_instance_wrapper* w = (native_instance_wrapper*)o;

        Type::TypeCategory cat = t->getTypeCategory();

        if (w->mIsMatcher) {
            PyObject* res;
            
            if (cat == Type::TypeCategory::catAlternative) {
                Alternative* a = (Alternative*)t;
                if (a->subtypes()[a->which(w->data)].first == attr_name) {
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
                t->copy_constructor(self->data, w->data);
                self->mIsInitialized = true;
                self->mIsMatcher = true;

                return (PyObject*)self;
            }
        }

        PyObject* result = getattr(t, w->data, attr_name);

        if (result) {
            return result;
        }

        return PyObject_GenericGetAttr(o, attrName);
    }

    static PyObject* getattr(const Type* type, instance_ptr data, char* attr_name) {
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

        if (type->getTypeCategory() == Type::TypeCategory::catClass) {
            Class* nt = (Class*)type;
            for (long k = 0; k < nt->getMembers().size();k++) {
                if (nt->getMembers()[k].first == attr_name) {
                    return extractPythonObject(
                        nt->eltPtr(data, k), 
                        nt->getMembers()[k].second
                        );
                }
            }
        }

        return NULL;
    }
    
    static Py_hash_t tp_hash(PyObject *o) {
        const Type* self_type = extractTypeFrom(o->ob_type);
        native_instance_wrapper* w = (native_instance_wrapper*)o;

        int32_t h = self_type->hash32(w->data);
        if (h == -1) {
            h = -2;
        }

        return h;
    }

    static char compare_to_python(const Type* t, instance_ptr self, PyObject* other, bool exact) {
        if (t->getTypeCategory() == Type::TypeCategory::catValue) {
            Value* valType = (Value*)t;
            return compare_to_python(valType->value().type(), valType->value().data(), other, exact);
        }

        const Type* otherT = extractTypeFrom(other->ob_type);

        if (otherT) {
            if (otherT < t) {
                return 1;
            }
            if (otherT > t) {
                return -1;
            }
            return t->cmp(self, ((native_instance_wrapper*)other)->data);
        }

        if (t->getTypeCategory() == Type::TypeCategory::catOneOf) {
            std::pair<const Type*, instance_ptr> child = ((OneOf*)t)->unwrap(self);
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
        const Type* own = extractTypeFrom(a->ob_type);
        const Type* other = extractTypeFrom(b->ob_type);


        if (!other) {
            char cmp = compare_to_python(own, ((native_instance_wrapper*)a)->data, b, false);

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
                cmp = own->cmp(((native_instance_wrapper*)a)->data, ((native_instance_wrapper*)b)->data);
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
        const Type* self_type = extractTypeFrom(o->ob_type);
        native_instance_wrapper* w = (native_instance_wrapper*)o;

        if (self_type && self_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
            native_instance_wrapper* self = (native_instance_wrapper*)o->ob_type->tp_alloc(o->ob_type, 0);

            self->mIteratorOffset = 0;
            self->mIteratorIsPairs = w->mIteratorIsPairs;
            self_type->copy_constructor(self->data, w->data);
            self->mIsInitialized = true;
            self->mIsMatcher = false;

            return (PyObject*)self;
        }

        PyErr_SetString(PyExc_TypeError, ("Cannot iterate an instance of " + self_type->name()).c_str());
        return NULL;
    }

    static PyObject* tp_iternext(PyObject *o) {
        const Type* self_type = extractTypeFrom(o->ob_type);
        native_instance_wrapper* w = (native_instance_wrapper*)o;

        if (self_type->getTypeCategory() != Type::TypeCategory::catConstDict) {
            return NULL;
        }

        ConstDict* dict_t = (ConstDict*)self_type;

        if (w->mIteratorOffset >= dict_t->size(w->data)) {
            return NULL;
        }

        w->mIteratorOffset++;

        if (w->mIteratorIsPairs) {
            auto t1 = extractPythonObject(
                    dict_t->kvPairPtrKey(w->data, w->mIteratorOffset-1), 
                    dict_t->keyType()
                    );
            auto t2 = extractPythonObject(
                    dict_t->kvPairPtrValue(w->data, w->mIteratorOffset-1), 
                    dict_t->valueType()
                    );
            
            auto res = PyTuple_Pack(2, t1, t2);

            Py_DECREF(t1);
            Py_DECREF(t2);

            return res;
        } else {
            return extractPythonObject(
                dict_t->kvPairPtrKey(w->data, w->mIteratorOffset-1), 
                dict_t->keyType()
                );
        }
    }

    static PyObject* tp_repr(PyObject *o) {
        const Type* self_type = extractTypeFrom(o->ob_type);
        native_instance_wrapper* w = (native_instance_wrapper*)o;

        std::ostringstream str;
        str << std::showpoint;

        self_type->repr(w->data, str);

        return PyUnicode_FromString(str.str().c_str());
    }

    static bool typeCanBeSubclassed(const Type* t) {
        return t->getTypeCategory() == Type::TypeCategory::catNamedTuple;
    }

    static PyBufferProcs* bufferProcs() {
        static PyBufferProcs* procs = new PyBufferProcs { 0, 0 };
        return procs;
    }

    static PyTypeObject* typeObjInternal(const Type* inType) {
        static std::recursive_mutex mutex;
        static std::map<const Type*, NativeTypeWrapper*> types;

        std::lock_guard<std::recursive_mutex> lock(mutex);

        auto it = types.find(inType);
        if (it != types.end()) {
            return (PyTypeObject*)it->second;
        }

        types[inType] = new NativeTypeWrapper { {
                PyVarObject_HEAD_INIT(NULL, 0)
                inType->name().c_str(),    /* tp_name */
                sizeof(native_instance_wrapper) + inType->bytecount(),       /* tp_basicsize */
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
                0,                         // tp_call
                0,                         // tp_str
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
            types[inType]->typeObj.tp_base = typeObjInternal((const Type*)inType->getBaseType());
            Py_INCREF(types[inType]->typeObj.tp_base);
        }

        PyType_Ready((PyTypeObject*)types[inType]);

        if (inType->getTypeCategory() == Type::TypeCategory::catAlternative) {
            Alternative* alt = (Alternative*)inType;
            for (long k = 0; k < alt->subtypes().size(); k++) {
                PyDict_SetItemString(
                    types[inType]->typeObj.tp_dict, 
                    alt->subtypes()[k].first.c_str(), 
                    (PyObject*)typeObjInternal(ConcreteAlternative::Make(alt, k))
                    );
            }
        }

        if (inType->getTypeCategory() == Type::TypeCategory::catClass) {
            for (auto nameAndObj: ((Class*)inType)->getClassMembers()) {
                PyDict_SetItemString(
                    types[inType]->typeObj.tp_dict, 
                    nameAndObj.first.c_str(), 
                    nameAndObj.second
                    );
            }
        }

        return (PyTypeObject*)types[inType];
    }

    static const Type* tryUnwrapPyInstanceToType(PyObject* arg) {
        if (PyType_Check(arg)) {
            const Type* possibleType = native_instance_wrapper::unwrapTypeArgToTypePtr(arg);
            if (!possibleType) {
                return NULL;
            }
            return possibleType;
        }

        if (arg == Py_None) {
            return None::Make();
        }

        return  native_instance_wrapper::tryUnwrapPyInstanceToValueType(arg);
    }        

    static const Type* tryUnwrapPyInstanceToValueType(PyObject* typearg) {
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

        const Type* nativeType = native_instance_wrapper::extractTypeFrom(typearg->ob_type);
        if (nativeType) {
            return Value::Make(
                Instance::create(
                    nativeType, 
                    ((native_instance_wrapper*)typearg)->data
                    )
                );
        }

        return nullptr;
    }

    static const Type* unwrapTypeArgToTypePtr(PyObject* typearg) {
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
                const Type* nativeT = native_instance_wrapper::extractTypeFrom(pyType);

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

            const Type* res = native_instance_wrapper::extractTypeFrom(pyType);
            if (res) {
                return res;
            }

            PyErr_SetString(PyExc_TypeError, 
                ("Cannot convert " + std::string(pyType->tp_name) + " to a native type.").c_str()
                );
            return NULL;
        }

        const Type* valueType = native_instance_wrapper::tryUnwrapPyInstanceToValueType(typearg);

        if (valueType) {
            return valueType;
        }

        PyErr_SetString(PyExc_TypeError, "Cannot convert argument to a native type because is't not a type.");
        return NULL;
    }
};

PyObject *TupleOf(PyObject* nullValue, PyObject* args) {
    std::vector<const Type*> types;
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

PyObject *Tuple(PyObject* nullValue, PyObject* args) {
    std::vector<const Type*> types;
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

PyObject *ConstDict(PyObject* nullValue, PyObject* args) {
    std::vector<const Type*> types;
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

PyObject *OneOf(PyObject* nullValue, PyObject* args) {
    std::vector<const Type*> types;
    for (long k = 0; k < PyTuple_Size(args); k++) {
        PyObject* arg = PyTuple_GetItem(args,k);

        const Type* t = native_instance_wrapper::tryUnwrapPyInstanceToType(arg);

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

    std::vector<std::pair<std::string, const Type*> > namesAndTypes;

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
    std::vector<const Type*> types;

    for (auto p: namesAndTypes) {
        names.push_back(p.first);
        types.push_back(p.second);
    }

    PyObject* typeObj = (PyObject*)native_instance_wrapper::typeObj(NamedTuple::Make(types, names));

    Py_INCREF(typeObj);
    return typeObj;
}


PyObject *Int8(PyObject* nullValue, PyObject* args) {
    PyObject* res = (PyObject*)native_instance_wrapper::typeObj(::Int8::Make());
    Py_INCREF(res);
    return res;
}
PyObject *Int16(PyObject* nullValue, PyObject* args) {
    PyObject* res = (PyObject*)native_instance_wrapper::typeObj(::Int16::Make());
    Py_INCREF(res);
    return res;
}
PyObject *Int32(PyObject* nullValue, PyObject* args) {
    PyObject* res = (PyObject*)native_instance_wrapper::typeObj(::Int32::Make());
    Py_INCREF(res);
    return res;
}
PyObject *Int64(PyObject* nullValue, PyObject* args) {
    PyObject* res = (PyObject*)native_instance_wrapper::typeObj(::Int64::Make());
    Py_INCREF(res);
    return res;
}
PyObject *UInt8(PyObject* nullValue, PyObject* args) {
    PyObject* res = (PyObject*)native_instance_wrapper::typeObj(::UInt8::Make());
    Py_INCREF(res);
    return res;
}
PyObject *UInt16(PyObject* nullValue, PyObject* args) {
    PyObject* res = (PyObject*)native_instance_wrapper::typeObj(::UInt16::Make());
    Py_INCREF(res);
    return res;
}
PyObject *UInt32(PyObject* nullValue, PyObject* args) {
    PyObject* res = (PyObject*)native_instance_wrapper::typeObj(::UInt32::Make());
    Py_INCREF(res);
    return res;
}
PyObject *UInt64(PyObject* nullValue, PyObject* args) {
    PyObject* res = (PyObject*)native_instance_wrapper::typeObj(::UInt64::Make());
    Py_INCREF(res);
    return res;
}
PyObject *String(PyObject* nullValue, PyObject* args) {
    PyObject* res = (PyObject*)native_instance_wrapper::typeObj(::String::Make());
    Py_INCREF(res);
    return res;
}
PyObject *Bytes(PyObject* nullValue, PyObject* args) {
    PyObject* res = (PyObject*)native_instance_wrapper::typeObj(::Bytes::Make());
    Py_INCREF(res);
    return res;
}

PyObject *NoneType(PyObject* nullValue, PyObject* args) {
    PyObject* res = (PyObject*)native_instance_wrapper::typeObj(::None::Make());
    Py_INCREF(res);
    return res;
}

PyObject *Value(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "Value takes 1 positional argument");
        return NULL;
    }
    
    PyObject* arg = PyTuple_GetItem(args,0);

    if (PyType_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Value expects a python primitive or an existing native value");
        return NULL;
    }

    const Type* type = native_instance_wrapper::tryUnwrapPyInstanceToValueType(arg);
    
    if (type) {
        PyObject* typeObj = (PyObject*)native_instance_wrapper::typeObj(type);
        Py_INCREF(typeObj);
        return typeObj;
    }

    PyErr_SetString(PyExc_TypeError, "Couldn't convert this to a value");
    return NULL;    
}

bool unpackTupleToStringAndTypes(PyObject* tuple, std::vector<std::pair<std::string, const Type*> >& out) {
    std::set<std::string> memberNames;

    for (int i = 0; i < PyTuple_Size(tuple); ++i) {
        PyObject* entry = PyTuple_GetItem(tuple, i);
        const Type* targetType = NULL;

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
        const Type* targetType = NULL;

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

PyObject *MakeFunction(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 4 && PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "Function takes 2 or 4 arguments");
        return NULL;
    }

    const Function* resType;

    if (PyTuple_Size(args) == 2) {
        PyObject* a0 = PyTuple_GetItem(args,0);
        PyObject* a1 = PyTuple_GetItem(args,1);

        const Type* t0 = native_instance_wrapper::unwrapTypeArgToTypePtr(a0);
        const Type* t1 = native_instance_wrapper::unwrapTypeArgToTypePtr(a1);

        if (!t0 || t0->getTypeCategory() != Type::TypeCategory::catFunction) {
            PyErr_SetString(PyExc_TypeError, "Expected first argument to be a function");
            return NULL;
        }
        if (!t1 || t1->getTypeCategory() != Type::TypeCategory::catFunction) {
            PyErr_SetString(PyExc_TypeError, "Expected second argument to be a function");
            return NULL;
        }

        resType = Function::merge((const Function*)t0, (const Function*)t1);
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

        const Type* rType = 0;

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
            
            const Type* argT = nullptr;
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
                Py_INCREF(val);
            }

            argList.push_back(Function::FunctionArg(
                PyUnicode_AsUTF8(k0),
                argT,
                val,
                k3 == Py_True,
                k4 == Py_True
                ));
        }

        std::vector<Function::FunctionOverload> overloads;
        overloads.push_back(
            Function::FunctionOverload((PyFunctionObject*)funcObj, rType, argList) 
            );

        resType = new Function(PyUnicode_AsUTF8(nameObj), overloads);
    }



    PyObject* typeObj = (PyObject*)native_instance_wrapper::typeObj(resType);
    Py_INCREF(typeObj);
    return typeObj;
}

PyObject *Class(PyObject* nullValue, PyObject* args) {
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

    std::vector<std::pair<std::string, const Type*> > members;
    std::vector<std::pair<std::string, const Type*> > memberFunctions;
    std::vector<std::pair<std::string, const Type*> > staticFunctions;
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
    
    std::map<std::string, const Function*> memberFuncs;
    std::map<std::string, const Function*> staticFuncs;
    
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
        memberFuncs[mf.first] = (const Function*)mf.second;
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
        staticFuncs[mf.first] = (const Function*)mf.second;
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

PyObject *serialize(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "serialize takes 2 positional arguments");
        return NULL;
    }
    
    PyObject* a1 = PyTuple_GetItem(args, 0);
    PyObject* a2 = PyTuple_GetItem(args, 1);

    const Type* serializeType = native_instance_wrapper::unwrapTypeArgToTypePtr(a1);

    if (!serializeType) {
        PyErr_Format(
            PyExc_TypeError, 
            "first argument to serialize must be a native type object, not %S",
            a1
            );
        return NULL;
    }

    const Type* actualType = native_instance_wrapper::extractTypeFrom(a2->ob_type);

    SerializationBuffer b;
    
    if (actualType == serializeType) {
        //the simple case
        actualType->serialize(((native_instance_wrapper*)a2)->data, b);
    } else {
        //try to construct a 'serialize type' from the argument and then serialize that
        try{
            Instance i = Instance::createAndInitialize(serializeType, [&](instance_ptr p) {
                native_instance_wrapper::copy_initialize(serializeType, p, a2);
            });
            
            i.type()->serialize(i.data(), b);
        } catch (std::exception& e) {
            PyErr_SetString(PyExc_TypeError, e.what());
            return NULL;            
        }
    }

    return PyBytes_FromStringAndSize((const char*)b.buffer(), b.size());
}

PyObject *deserialize(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "serialize takes 1 positional argument");
        return NULL;
    }
    PyObject* a1 = PyTuple_GetItem(args, 0);
    PyObject* a2 = PyTuple_GetItem(args, 1);

    const Type* serializeType = native_instance_wrapper::unwrapTypeArgToTypePtr(a1);

    if (!serializeType) {
        PyErr_SetString(PyExc_TypeError, "first argument to serialize must be a native type object");
        return NULL;
    }
    if (!PyBytes_Check(a2)) {
        PyErr_SetString(PyExc_TypeError, "second argument to serialize must be a bytes object");
        return NULL;
    }

    DeserializationBuffer buf((uint8_t*)PyBytes_AsString(a2), PyBytes_GET_SIZE(a2));

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

PyObject *Alternative(PyObject* nullValue, PyObject* args, PyObject* kwargs) {
    if (PyTuple_Size(args) != 1 || !PyUnicode_Check(PyTuple_GetItem(args,0))) {
        PyErr_SetString(PyExc_TypeError, "Alternative takes a single string positional argument.");
        return NULL;
    }

    std::string name = PyUnicode_AsUTF8(PyTuple_GetItem(args,0));

    std::vector<std::pair<std::string, NamedTuple*> > definitions;

    PyObject *key, *value;
    Py_ssize_t pos = 0;

    int i = 0;

    while (kwargs && PyDict_Next(kwargs, &pos, &key, &value)) {
        assert(PyUnicode_Check(key));

        std::string fieldName(PyUnicode_AsUTF8(key));

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
    };

    static_assert(PY_MAJOR_VERSION >= 3, "nativepython is a python3 project only");

    if (PY_MINOR_VERSION <= 5) {
        //we cannot rely on the ordering of 'kwargs' here because of the python version, so
        //we sort it. this will be a problem for anyone running some processes using different
        //python versions that share python code.
        std::sort(definitions.begin(), definitions.end());
    }

    PyObject* res = (PyObject*)native_instance_wrapper::typeObj(
        ::Alternative::Make(name, definitions)
        );

    Py_INCREF(res);
    return res;
}

static PyMethodDef module_methods[] = {
    {"NoneType", (PyCFunction)NoneType, METH_VARARGS, NULL},
    {"Int8", (PyCFunction)Int8, METH_VARARGS, NULL},
    {"Int16", (PyCFunction)Int16, METH_VARARGS, NULL},
    {"Int32", (PyCFunction)Int32, METH_VARARGS, NULL},
    {"Int64", (PyCFunction)Int64, METH_VARARGS, NULL},
    {"UInt8", (PyCFunction)UInt8, METH_VARARGS, NULL},
    {"UInt16", (PyCFunction)UInt16, METH_VARARGS, NULL},
    {"UInt32", (PyCFunction)UInt32, METH_VARARGS, NULL},
    {"UInt64", (PyCFunction)UInt64, METH_VARARGS, NULL},
    {"String", (PyCFunction)String, METH_VARARGS, NULL},
    {"Bytes", (PyCFunction)Bytes, METH_VARARGS, NULL},
    {"TupleOf", (PyCFunction)TupleOf, METH_VARARGS, NULL},
    {"Tuple", (PyCFunction)Tuple, METH_VARARGS, NULL},
    {"NamedTuple", (PyCFunction)MakeNamedTupleType, METH_VARARGS | METH_KEYWORDS, NULL},
    {"OneOf", (PyCFunction)OneOf, METH_VARARGS, NULL},
    {"ConstDict", (PyCFunction)ConstDict, METH_VARARGS, NULL},
    {"Alternative", (PyCFunction)Alternative, METH_VARARGS | METH_KEYWORDS, NULL},
    {"Value", (PyCFunction)Value, METH_VARARGS, NULL},
    {"Class", (PyCFunction)Class, METH_VARARGS, NULL},
    {"Function", (PyCFunction)MakeFunction, METH_VARARGS, NULL},
    {"serialize", (PyCFunction)serialize, METH_VARARGS, NULL},
    {"deserialize", (PyCFunction)deserialize, METH_VARARGS, NULL},
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

PyMODINIT_FUNC
PyInit__types(void)
{
    PyObject *module = PyModule_Create(&moduledef);

    if (module == NULL)
        return NULL;

    return module;
}
