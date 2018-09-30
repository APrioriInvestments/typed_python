#include "Python.h"
#include <map>
#include <memory>
#include <vector>
#include <string>
#include "Type.hpp"
#include <iostream>

//extension of PyTypeObject that stashes a Type* on the end.
struct NativeTypeWrapper {
    PyTypeObject typeObj;
    Type* mType;
};

struct native_instance_wrapper {
    PyObject_HEAD
  
    Type* getType() const {
        return ((const NativeTypeWrapper*)((PyObject*)this)->ob_type)->mType;
    }

    int64_t mIsInitialized;
    int64_t mIteratorOffset; //-1 if we're not an iterator
    uint8_t data[0];

    static PyObject* bytecount(PyObject* o) {
        NativeTypeWrapper* w = (NativeTypeWrapper*)o;
        return PyLong_FromLong(w->mType->bytecount());
    }

    static PyMethodDef* typeMethods() {
        static PyMethodDef typed_python_TypeMethods[] = {
            {"bytecount", (PyCFunction)native_instance_wrapper::bytecount, METH_CLASS | METH_NOARGS, NULL},
            {NULL, NULL}
        };

        return typed_python_TypeMethods;
    };

    static void tp_dealloc(PyObject* self) {
        native_instance_wrapper* wrapper = (native_instance_wrapper*)self;

        if (wrapper->mIsInitialized) {
            wrapper->getType()->destroy(wrapper->data);
        }

        Py_TYPE(self)->tp_free((PyObject*)self);
    }

    static bool pythonObjectCouldBe(Type* eltType, PyObject* pyRepresentation) {
        return true;
    }

    static bool pythonObjectIsDefinitely(Type* eltType, instance_ptr data, PyObject* pyRepresentation) {
        if (eltType->getTypeCategory() == Type::TypeCategory::catNone &&
                pyRepresentation->ob_type == Py_None->ob_type)
            return true;

        return false;
    }

    static void copy_initialize(Type* eltType, instance_ptr tgt, PyObject* pyRepresentation) {
        if (pyRepresentation->ob_type == typeObj(eltType)) {
            //it's already the right kind of instance
            eltType->copy_constructor(tgt, ((native_instance_wrapper*)pyRepresentation)->data);
            return;
        }

        Type::TypeCategory cat = eltType->getTypeCategory();

        if (cat == Type::TypeCategory::catValue) {
            Value* v = (Value*)eltType;

            std::pair<Type*, instance_ptr> elt = v->unwrap();

            if (pythonObjectCouldBe(elt.first, pyRepresentation)) {
                if (pythonObjectIsDefinitely(elt.first, elt.second, pyRepresentation)) {
                    return;
                }

                //we have to see if it's possible. for the moment, punt because we
                //don't have comparators defined yet.
                throw std::logic_error("Can't initialize a " + eltType->name() + " from an instance of " + 
                    std::string(pyRepresentation->ob_type->tp_name));
                return;
            }
        }

        if (cat == Type::TypeCategory::catOneOf) {
            OneOf* oneOf = (OneOf*)eltType;

            for (long k = 0; k < oneOf->getTypes().size(); k++) {
                Type* subtype = oneOf->getTypes()[k];

                if (pythonObjectCouldBe(subtype, pyRepresentation)) {
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
                            dictType->keyType()->destroy(dictType->kvPairPtrValue(tgt,i));
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

        throw std::logic_error("Couldn't initialize internal elt of type " + eltType->name());
    }

    static void initialize(uint8_t* data, Type* t, PyObject* args, PyObject* kwargs) {
        Type::TypeCategory cat = t->getTypeCategory();

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

            throw std::logic_error("Can't initialize " + t->name() + " with these arguments.");
        } else {
            if (cat == Type::TypeCategory::catNamedTuple) {
                CompositeType* compositeT = ((CompositeType*)t);

                compositeT->constructor(
                    data, 
                    [&](uint8_t* eltPtr, int64_t k) {
                        Type* eltType = compositeT->getTypes()[k];
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

            throw std::logic_error("Can't initialize " + t->name() + " from python yet.");
        }
    }


    //produce the pythonic representation of this object. for things like integers, string, etc,
    //convert them back to their python-native form. otherwise, a pointer back into a native python
    //structure
    static PyObject* extractPythonObject(instance_ptr data, Type* eltType) {
        if (eltType->getTypeCategory() == Type::TypeCategory::catNone) {
            Py_INCREF(Py_None);
            return Py_None;
        }
        if (eltType->getTypeCategory() == Type::TypeCategory::catInt64) {
            return PyLong_FromLong(*(int64_t*)data);
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

        native_instance_wrapper* self = (native_instance_wrapper*)typeObj(eltType)->tp_alloc(typeObj(eltType), 0);

        try {
            self->mIteratorOffset = -1;
            self->mIsInitialized = false;

            eltType->copy_constructor(self->data, data);

            self->mIsInitialized = true;

            return (PyObject*)self;
        } catch(std::exception& e) {
            typeObj(self->getType())->tp_dealloc((PyObject*)self);

            PyErr_SetString(PyExc_TypeError, e.what());
            return NULL;
        }
    }

    static PyObject *tp_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds) {
        Type* eltType = extractTypeFrom(subtype);

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

    static Py_ssize_t sq_length(native_instance_wrapper* w) {
        if (w->getType()->getTypeCategory() == Type::TypeCategory::catTupleOf) {
            return ((TupleOf*)w->getType())->count(w->data);
        }
        if (w->getType()->isComposite()) {
            return ((CompositeType*)w->getType())->getTypes().size();
        }
        if (w->getType()->getTypeCategory() == Type::TypeCategory::catString) {
            return String().count(w->data);
        }
        if (w->getType()->getTypeCategory() == Type::TypeCategory::catBytes) {
            return Bytes().count(w->data);
        }

        return 0;
    }

    static PyObject* sq_item(native_instance_wrapper* w, Py_ssize_t ix) {
        if (w->getType()->getTypeCategory() == Type::TypeCategory::catTupleOf) {
            int64_t count = ((TupleOf*)w->getType())->count(w->data);

            if (ix >= count || ix < 0) {
                PyErr_SetString(PyExc_IndexError, "index out of range");
                return NULL;
            }

            Type* eltType = (Type*)((TupleOf*)w->getType())->getEltType();
            return extractPythonObject(
                ((TupleOf*)w->getType())->eltPtr(w->data, ix), 
                eltType
                );
        }
        
        if (w->getType()->isComposite()) {
            auto compType = (CompositeType*)w->getType();

            if (ix < 0 || ix >= (int64_t)compType->getTypes().size()) {
                PyErr_SetString(PyExc_IndexError, "index out of range");
                return NULL;
            }

            Type* eltType = compType->getTypes()[ix];

            return extractPythonObject(
                compType->eltPtr(w->data, ix), 
                eltType
                );
        }

        if (w->getType()->getTypeCategory() == Type::TypeCategory::catBytes) {
            if (ix < 0 || ix >= (int64_t)Bytes().count(w->data)) {
                PyErr_SetString(PyExc_IndexError, "index out of range");
                return NULL;
            }

            return PyBytes_FromStringAndSize(
                (const char*)Bytes().eltPtr(w->data, ix),
                1
                );
        }
        if (w->getType()->getTypeCategory() == Type::TypeCategory::catString) {
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
        if (w->getType()->getTypeCategory() == Type::TypeCategory::catConstDict) {
            if (ix < 0 || ix >= (int64_t)Bytes().count(w->data)) {
                PyErr_SetString(PyExc_IndexError, "index out of range");
                return NULL;
            }

            return PyBytes_FromStringAndSize(
                (const char*)Bytes().eltPtr(w->data, ix),
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

    static PySequenceMethods* sequenceMethods(Type* t) {
        static PySequenceMethods* res = 
            new PySequenceMethods {
                (lenfunc)native_instance_wrapper::sq_length,
                0,
                0,
                (ssizeargfunc)native_instance_wrapper::sq_item,
                0,
                0,
                0,
                0
                };

        if (t->getTypeCategory() == Type::TypeCategory::catTupleOf || 
                t->getTypeCategory() == Type::TypeCategory::catTuple || 
                t->getTypeCategory() == Type::TypeCategory::catNamedTuple || 
                t->getTypeCategory() == Type::TypeCategory::catString || 
                t->getTypeCategory() == Type::TypeCategory::catBytes) {
            return res;
        }

        return 0;
    }

    static PyNumberMethods* numberMethods(Type* t) {
        static PyNumberMethods* res = 
            new PyNumberMethods {
                0, //binaryfunc nb_add
                0, //binaryfunc nb_subtract
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

        Type* t = extractTypeFrom(o->ob_type);

        if (t->getTypeCategory() == Type::TypeCategory::catConstDict) {
            return ((ConstDict*)t)->size(w->data);
        }

        return 0;
    }

    static PyObject* mp_subscript(PyObject* o, PyObject* item) {
        native_instance_wrapper* self_w = (native_instance_wrapper*)o;

        Type* self_type = extractTypeFrom(o->ob_type);
        Type* item_type = extractTypeFrom(o->ob_type);

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

        if (t->getTypeCategory() == Type::TypeCategory::catConstDict) {
            return res;
        }

        return 0;
    }

    static Type* extractTypeFrom(PyTypeObject* typeObj) {
        if (typeObj->tp_dealloc == native_instance_wrapper::tp_dealloc) {
            return ((NativeTypeWrapper*)typeObj)->mType;
        }

        return nullptr;
    }

    static PyObject* tp_getattr(PyObject *o, char *attr_name) {
        native_instance_wrapper* w = (native_instance_wrapper*)o;

        if (w->getType()->getTypeCategory() == Type::TypeCategory::catNamedTuple) {
            NamedTuple* nt = (NamedTuple*)w->getType();
            for (long k = 0; k < nt->getNames().size();k++) {
                if (nt->getNames()[k] == attr_name) {
                    return extractPythonObject(
                        nt->eltPtr(w->data, k), 
                        nt->getTypes()[k]
                        );
                }
            }
        }

        PyErr_SetString(PyExc_AttributeError, attr_name);
        return NULL;
    }
    static PyObject *tp_richcompare(PyObject *a, PyObject *b, int op) {
        Type* own = extractTypeFrom(a->ob_type);
        Type* other = extractTypeFrom(b->ob_type);

        if (!other) {
            Py_INCREF(Py_NotImplemented);
            return Py_NotImplemented;
        }
        
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

    static PyObject* tp_iter(PyObject *o) {
        Type* self_type = extractTypeFrom(o->ob_type);
        native_instance_wrapper* w = (native_instance_wrapper*)o;

        if (self_type && self_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
            native_instance_wrapper* self = (native_instance_wrapper*)typeObj(self_type)->tp_alloc(o->ob_type, 0);

            self->mIteratorOffset = 0;
            self_type->copy_constructor(self->data, w->data);
            self->mIsInitialized = true;

            return (PyObject*)self;
        }

        PyErr_SetString(PyExc_TypeError, ("Cannot iterate an instance of " + self_type->name()).c_str());
        return NULL;
    }

    static PyObject* tp_iternext(PyObject *o) {
        Type* self_type = extractTypeFrom(o->ob_type);
        native_instance_wrapper* w = (native_instance_wrapper*)o;

        assert(self_type->getTypeCategory() == Type::TypeCategory::catConstDict);

        ConstDict* dict_t = (ConstDict*)self_type;

        if (w->mIteratorOffset >= dict_t->size(w->data)) {
            return NULL;
        }

        w->mIteratorOffset++;

        return extractPythonObject(dict_t->kvPairPtrKey(w->data, w->mIteratorOffset-1), dict_t->keyType());
    }

    static PyTypeObject* typeObjInternal(Type* inType) {
        static std::mutex mutex;
        static std::map<Type*, NativeTypeWrapper*> types;

        std::lock_guard<std::mutex> lock(mutex);

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
                native_instance_wrapper::tp_getattr,                         // tp_getattr
                0,                         // tp_setattr
                0,                         // tp_reserved
                0,                         // tp_repr
                numberMethods(inType),     // tp_as_number
                sequenceMethods(inType),   // tp_as_sequence
                mappingMethods(inType),    // tp_as_mapping
                0,                         // tp_hash
                0,                         // tp_call
                0,                         // tp_str
                0,                         // tp_getattro
                0,                         // tp_setattro
                0,                         // tp_as_buffer
                Py_TPFLAGS_DEFAULT,        // tp_flags
                0,                         // tp_doc
                0,                         // traverseproc tp_traverse;
                0,                         // inquiry tp_clear;
                tp_richcompare,            // richcmpfunc tp_richcompare;
                0,                         // Py_ssize_t tp_weaklistoffset;
                inType->getTypeCategory() == Type::TypeCategory::catConstDict ? 
                    native_instance_wrapper::tp_iter
                :   0,                     // getiterfunc tp_iter;
                native_instance_wrapper::tp_iternext,// iternextfunc tp_iternext;
                typeMethods(),             // struct PyMethodDef *tp_methods;
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

        PyType_Ready((PyTypeObject*)types[inType]);
        return (PyTypeObject*)types[inType];
    }
};

Type* unwrapTypeArgToTypePtr(PyObject* typearg) {
    if (PyType_Check(typearg)) {
        if ((PyTypeObject*)typearg == &PyLong_Type) {
            return Int64::Make();
        }
        if ((PyTypeObject*)typearg == &PyFloat_Type) {
            return Float64::Make();
        }
        if ((PyTypeObject*)typearg == Py_None->ob_type) {
            return None::Make();
        }
        if ((PyTypeObject*)typearg == &PyBool_Type) {
            return Bool::Make();
        }
        if ((PyTypeObject*)typearg == &PyBytes_Type) {
            return Bytes::Make();
        }
        if ((PyTypeObject*)typearg == &PyUnicode_Type) {
            return String::Make();
        }

        Type* res = native_instance_wrapper::extractTypeFrom((PyTypeObject*)typearg);
        if (res) {
            return res;
        }
    }

    PyErr_SetString(PyExc_TypeError, "Cannot convert argument to a native type.");
    return NULL;
}

PyObject *TupleOf(PyObject* nullValue, PyObject* args) {
    std::vector<Type*> types;
    for (long k = 0; k < PyTuple_Size(args); k++) {
        types.push_back(unwrapTypeArgToTypePtr(PyTuple_GetItem(args,k)));
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
    std::vector<Type*> types;
    for (long k = 0; k < PyTuple_Size(args); k++) {
        types.push_back(unwrapTypeArgToTypePtr(PyTuple_GetItem(args,k)));
        if (not types.back()) {
            return NULL;
        }
    }

    PyObject* typeObj = (PyObject*)native_instance_wrapper::typeObj(Tuple::Make(types));

    Py_INCREF(typeObj);
    return typeObj;
}

PyObject *ConstDict(PyObject* nullValue, PyObject* args) {
    std::vector<Type*> types;
    for (long k = 0; k < PyTuple_Size(args); k++) {
        types.push_back(unwrapTypeArgToTypePtr(PyTuple_GetItem(args,k)));
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
    std::vector<Type*> types;
    for (long k = 0; k < PyTuple_Size(args); k++) {
        PyObject* arg = PyTuple_GetItem(args,k);

        if (PyType_Check(arg)) {
            Type* possibleType = unwrapTypeArgToTypePtr(PyTuple_GetItem(args,k));
            if (!possibleType) {
                return NULL;
            }
            types.push_back(possibleType);
        } else {
            if (arg == Py_None) {
                types.push_back(None::Make());
            } else {
                PyErr_SetString(PyExc_TypeError, "Can't handle values like this in Types.");
                return NULL;
            }
        }
    }

    PyObject* typeObj = (PyObject*)native_instance_wrapper::typeObj(OneOf::Make(types));

    Py_INCREF(typeObj);
    return typeObj;
}

PyObject *NamedTuple(PyObject* nullValue, PyObject* args, PyObject* kwargs) {
    if (PyTuple_Size(args)) {
        PyErr_SetString(PyExc_TypeError, "NamedTuple takes no positional arguments.");
        return NULL;
    }

    std::vector<Type*> types;
    std::vector<std::string> names;

    PyObject *key, *value;
    Py_ssize_t pos = 0;

    while (PyDict_Next(kwargs, &pos, &key, &value)) {
        if (!PyUnicode_Check(key)) {
            PyErr_SetString(PyExc_TypeError, "NamedTuple keywords are supposed to be strings.");
            return NULL;
        }

        names.push_back(PyUnicode_AsUTF8(key));
        types.push_back(unwrapTypeArgToTypePtr(value));

        if (not types.back()) {
            return NULL;
        }
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
    {"NamedTuple", (PyCFunction)NamedTuple, METH_VARARGS | METH_KEYWORDS, NULL},
    {"OneOf", (PyCFunction)OneOf, METH_VARARGS, NULL},
    {"ConstDict", (PyCFunction)ConstDict, METH_VARARGS, NULL},
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
