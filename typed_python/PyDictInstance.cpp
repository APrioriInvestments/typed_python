/******************************************************************************
   Copyright 2017-2019 Nativepython Authors

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

#include "PyDictInstance.hpp"

DictType* PyDictInstance::type() {
    return (DictType*)extractTypeFrom(((PyObject*)this)->ob_type);
}

// static
PyObject* PyDictInstance::dictItems(PyObject *o) {
    PyInstance* result = ((PyInstance*)o)->duplicate();

    result->mIteratorOffset = 0;
    result->mIteratorFlag = 2;
    result->mIsMatcher = false;

    return (PyObject*)result;
}

// static
PyObject* PyDictInstance::dictKeys(PyObject *o) {
    PyInstance* result = ((PyInstance*)o)->duplicate();

    result->mIteratorOffset = 0;
    result->mIteratorFlag = 0;
    result->mIsMatcher = false;

    return (PyObject*)result;
}

// static
PyObject* PyDictInstance::dictValues(PyObject *o) {
    PyInstance* result = ((PyInstance*)o)->duplicate();

    result->mIteratorOffset = 0;
    result->mIteratorFlag = 1;
    result->mIsMatcher = false;

    return (PyObject*)result;
}

// static
PyObject* PyDictInstance::dictGet(PyObject* o, PyObject* args) {
    PyDictInstance* self_w = (PyDictInstance*)o;

    if (PyTuple_Size(args) < 1 || PyTuple_Size(args) > 2) {
        PyErr_SetString(PyExc_TypeError, "Dict.get takes one or two arguments");
        return NULL;
    }

    PyObjectHolder item(PyTuple_GetItem(args,0));
    PyObjectHolder ifNotFound((PyTuple_Size(args) == 2 ? PyTuple_GetItem(args,1) : Py_None));

    Type* self_type = extractTypeFrom(o->ob_type);
    Type* item_type = extractTypeFrom(item->ob_type);

    if (self_type->getTypeCategory() == Type::TypeCategory::catDict) {
        if (item_type == self_w->type()->keyType()) {
            PyInstance* item_w = (PyInstance*)(PyObject*)item;

            instance_ptr i = self_w->type()->lookupValueByKey(self_w->dataPtr(), item_w->dataPtr());

            if (!i) {
                return incref(ifNotFound);
            }

            return extractPythonObject(i, self_w->type()->valueType());
        } else {
            try {
                Instance key(self_w->type()->keyType(), [&](instance_ptr data) {
                    copyConstructFromPythonInstance(self_w->type()->keyType(), data, item);
                });

                instance_ptr i = self_w->type()->lookupValueByKey(self_w->dataPtr(), key.data());

                if (!i) {
                    return incref(ifNotFound);
                }

                return extractPythonObject(i, self_w->type()->valueType());
            } catch(PythonExceptionSet& e) {
                return NULL;
            } catch(std::exception& e) {
                PyErr_SetString(PyExc_TypeError, e.what());
                return NULL;
            }
        }

        PyErr_SetString(PyExc_TypeError, "Invalid Dict lookup type");
        return NULL;
    }

    PyErr_SetString(PyExc_TypeError, "Wrong type!");
    return NULL;
}

PyObject* PyDictInstance::tp_iter_concrete() {
    return createIteratorToSelf(mIteratorFlag);
}

PyObject* PyDictInstance::tp_iternext_concrete() {
    if (mIteratorOffset == 0) {
        //search forward to find the first slot
        while (mIteratorOffset < type()->slotCount(dataPtr()) && !type()->slotPopulated(dataPtr(), mIteratorOffset)) {
            mIteratorOffset++;
        }
    }

    if (mIteratorOffset >= type()->slotCount(dataPtr())) {
        return NULL;
    }

    int32_t curSlot = mIteratorOffset;

    mIteratorOffset++;
    while (mIteratorOffset < type()->slotCount(dataPtr()) && !type()->slotPopulated(dataPtr(), mIteratorOffset)) {
        mIteratorOffset++;
    }

    if (mIteratorFlag == 2) {
        PyObjectStealer t1(extractPythonObject(
                type()->keyAtSlot(dataPtr(), curSlot),
                type()->keyType()
                ));
        PyObjectStealer t2(extractPythonObject(
                type()->valueAtSlot(dataPtr(), curSlot),
                type()->valueType()
                ));

        return PyTuple_Pack(2, (PyObject*)t1, (PyObject*)t2);
    } else if (mIteratorFlag == 1) {
        return extractPythonObject(
            type()->valueAtSlot(dataPtr(), curSlot),
            type()->valueType()
            );
    } else {
        return extractPythonObject(
            type()->keyAtSlot(dataPtr(), curSlot),
            type()->keyType()
            );
    }
}

Py_ssize_t PyDictInstance::mp_and_sq_length_concrete() {
    return type()->size(dataPtr());
}

int PyDictInstance::sq_contains_concrete(PyObject* item) {
    Type* item_type = extractTypeFrom(item->ob_type);

    if (item_type == type()->keyType()) {
        PyInstance* item_w = (PyInstance*)item;

        instance_ptr i = type()->lookupValueByKey(dataPtr(), item_w->dataPtr());

        if (!i) {
            return 0;
        }

        return 1;
    } else {
        Instance key(type()->keyType(), [&](instance_ptr data) {
            copyConstructFromPythonInstance(type()->keyType(), data, item);
        });

        instance_ptr i = type()->lookupValueByKey(dataPtr(), key.data());

        return i ? 1 : 0;
    }
}

PyObject* PyDictInstance::pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr) {
    return ((PyInstance*)this)->pyOperatorConcrete(rhs, op, opErr);
}

PyObject* PyDictInstance::mp_subscript_concrete(PyObject* item) {
    Type* item_type = extractTypeFrom(item->ob_type);

    if (item_type == type()->keyType()) {
        PyInstance* item_w = (PyInstance*)item;

        instance_ptr i = type()->lookupValueByKey(dataPtr(), item_w->dataPtr());

        if (!i) {
            PyErr_SetObject(PyExc_KeyError, item);
            return NULL;
        }

        return extractPythonObject(i, type()->valueType());
    } else {
        Instance key(type()->keyType(), [&](instance_ptr data) {
            copyConstructFromPythonInstance(type()->keyType(), data, item);
        });

        instance_ptr i = type()->lookupValueByKey(dataPtr(), key.data());

        if (!i) {
            PyErr_SetObject(PyExc_KeyError, item);
            return NULL;
        }

        return extractPythonObject(i, type()->valueType());
    }

    PyErr_SetObject(PyExc_KeyError, item);
    return NULL;
}

int PyDictInstance::mp_ass_subscript_concrete_typed(instance_ptr key, instance_ptr value) {
    instance_ptr existingLoc = type()->lookupValueByKey(dataPtr(), key);
    if (existingLoc) {
        type()->valueType()->assign(existingLoc, value);
    } else {
        instance_ptr newLoc = type()->insertKey(dataPtr(), key);
        type()->valueType()->copy_constructor(newLoc, value);
    }

    return 0;
}

int PyDictInstance::mp_ass_subscript_concrete_keytyped(PyObject* pyKey, instance_ptr key, PyObject* value) {
    if (!value) {
        if (type()->deleteKey(dataPtr(), key)) {

            return 0;
        }

        PyErr_SetObject(PyExc_KeyError, pyKey);
        return -1;
    }

    Type* value_type = extractTypeFrom(value->ob_type);

    if (value_type == type()->valueType()) {
        PyInstance* value_w = (PyInstance*)value;
        return mp_ass_subscript_concrete_typed(key, value_w->dataPtr());
    } else {
        Instance val(type()->valueType(), [&](instance_ptr data) {
            copyConstructFromPythonInstance(type()->valueType(), data, value);
        });

        return mp_ass_subscript_concrete_typed(key, val.data());
    }
}

int PyDictInstance::mp_ass_subscript_concrete(PyObject* item, PyObject* value) {
    Type* item_type = extractTypeFrom(item->ob_type);

    if (item_type == type()->keyType()) {
        PyInstance* item_w = (PyInstance*)item;
        return mp_ass_subscript_concrete_keytyped(item, item_w->dataPtr(), value);
    } else {
        Instance key(type()->keyType(), [&](instance_ptr data) {
            copyConstructFromPythonInstance(type()->keyType(), data, item);
        });

        return mp_ass_subscript_concrete_keytyped(item, key.data(), value);
    }
}

PyMethodDef* PyDictInstance::typeMethodsConcrete(Type* t) {
    return new PyMethodDef [6] {
        {"get", (PyCFunction)PyDictInstance::dictGet, METH_VARARGS, NULL},
        {"items", (PyCFunction)PyDictInstance::dictItems, METH_NOARGS, NULL},
        {"keys", (PyCFunction)PyDictInstance::dictKeys, METH_NOARGS, NULL},
        {"values", (PyCFunction)PyDictInstance::dictValues, METH_NOARGS, NULL},
        {"setdefault", (PyCFunction)PyDictInstance::setDefault, METH_VARARGS, NULL},
        {NULL, NULL}
    };
}

void PyDictInstance::mirrorTypeInformationIntoPyTypeConcrete(DictType* dictT, PyTypeObject* pyType) {
    //expose 'ElementType' as a member of the type object
    PyDict_SetItemString(pyType->tp_dict, "KeyType",
            typePtrToPyTypeRepresentation(dictT->keyType())
            );
    PyDict_SetItemString(pyType->tp_dict, "ValueType",
            typePtrToPyTypeRepresentation(dictT->valueType())
            );
}


void PyDictInstance::copyConstructFromPythonInstanceConcrete(DictType* dictType, instance_ptr dictTgt, PyObject* pyRepresentation, bool isExplicit) {
    if (PyDict_Check(pyRepresentation)) {
        dictType->constructor(dictTgt);

        try {
            PyObject *key, *value;
            Py_ssize_t pos = 0;

            while (PyDict_Next(pyRepresentation, &pos, &key, &value)) {
                Instance keyInst(dictType->keyType(), [&](instance_ptr data) {
                    copyConstructFromPythonInstance(dictType->keyType(), data, key);
                });

                Instance valueInst(dictType->valueType(), [&](instance_ptr data) {
                    copyConstructFromPythonInstance(dictType->valueType(), data, value);
                });

                instance_ptr valueTgt = dictType->lookupValueByKey(dictTgt, keyInst.data());

                if (valueTgt) {
                    dictType->valueType()->assign(dictTgt, valueInst.data());
                } else {
                    valueTgt = dictType->insertKey(dictTgt, keyInst.data());
                    dictType->valueType()->copy_constructor(valueTgt, valueInst.data());
                }
            }

            return;
        } catch(...) {
            dictType->destroy(dictTgt);
            throw;
        }
    }

    PyInstance::copyConstructFromPythonInstanceConcrete(dictType, dictTgt, pyRepresentation, isExplicit);
}

bool PyDictInstance::compare_to_python_concrete(DictType* dictType, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp) {
    if (pyComparisonOp != Py_EQ && pyComparisonOp != Py_NE) {
        return PyInstance::compare_to_python_concrete(dictType, self, other, exact, pyComparisonOp);
    }

    if (pyComparisonOp == Py_NE) {
        return !compare_to_python_concrete(dictType, self, other, exact, Py_EQ);
    }

    //only Py_EQ at this point in the function
    if (pyComparisonOp != Py_EQ) {
        throw std::runtime_error("Somehow pyComparisonOp is not Py_EQ");
    }

    if (!PyDict_Check(other)) {
        return false;
    }

    int lenO = PyDict_Size(other);
    int lenS = dictType->size(self);

    if (lenO != lenS) {
        return false;
    }

    // iterate the python dictionary and check if each item in it matches
    // our values. if 'exact', then don't try to coerce the key types.

    PyObject *key, *value;
    Py_ssize_t pos = 0;

    while (PyDict_Next(other, &pos, &key, &value)) {
        Instance keyInst;

        try {
            keyInst = Instance(dictType->keyType(), [&](instance_ptr data) {
                copyConstructFromPythonInstance(dictType->keyType(), data, key, !exact /* explicit */);
            });
        } catch(PythonExceptionSet&) {
            PyErr_Clear();
        } catch(...) {
            //if we can't convert to KeyType, we're not equal
            return false;
        }

        instance_ptr valueTgt = dictType->lookupValueByKey(self, keyInst.data());

        if (!valueTgt) {
            //if we don't have the value, we're not equal
            return false;
        }

        if (!compare_to_python(dictType->valueType(), valueTgt, value, exact, Py_EQ)) {
            return false;
        }
    }

    return true;
}

//static 
PyObject* PyDictInstance::setDefault(PyObject* o, PyObject* args) {
    return translateExceptionToPyObject([&]() {
        const int argsNumber = PyTuple_Size(args);

        if (argsNumber < 1 || argsNumber > 2) {
            throw std::runtime_error("Dict.setdefault takes one or two arguments.");
        }

        PyDictInstance *self = (PyDictInstance *) o;

        /*
         * The function is called with setdefault(tem, ifNotFound=None)
         */
        PyObjectHolder item(PyTuple_GetItem(args, 0));
        PyObjectHolder ifNotFound(argsNumber == 2 ? PyTuple_GetItem(args, 1) : Py_None);

        Type *selfType = extractTypeFrom(o->ob_type);
        Type *itemType = extractTypeFrom(item->ob_type);

        Type *keyType = self->type()->keyType();
        Type *valueType = self->type()->valueType();

        if (selfType->getTypeCategory() != Type::TypeCategory::catDict) {
            throw std::runtime_error("Wrong type!");
        }

        instance_ptr i;
        instance_ptr lookupKey;
        Instance key;

        if (itemType == keyType) {
            PyInstance *item_w = (PyInstance *) (PyObject *) item;
            lookupKey = item_w->dataPtr();
            i = self->type()->lookupValueByKey(self->dataPtr(), item_w->dataPtr());
        } else {
            key = Instance(keyType, [&](instance_ptr data) {
                copyConstructFromPythonInstance(keyType, data, item);
            });
            lookupKey = key.data();
            i = self->type()->lookupValueByKey(self->dataPtr(), key.data());
        }

        // there is no value, we need to insert this to the dictionary
        if (!i) {
            if (extractTypeFrom(ifNotFound->ob_type) == self->type()->valueType()) {
                instance_ptr ifNotFoundValue = ((PyInstance *) (PyObject *) ifNotFound)->dataPtr();
                instance_ptr valueTgt = self->type()->insertKey(self->dataPtr(), lookupKey);
                self->type()->valueType()->copy_constructor(valueTgt, ifNotFoundValue);
                return extractPythonObject(valueTgt, self->type()->valueType());
            } else {
                Instance i(valueType, [&](instance_ptr data) {
                    copyConstructFromPythonInstance(valueType, data, ifNotFound);
                });
                instance_ptr ifNotFoundValue = i.data();
                instance_ptr valueTgt = self->type()->insertKey(self->dataPtr(), lookupKey);
                self->type()->valueType()->copy_constructor(valueTgt, ifNotFoundValue);
                return extractPythonObject(valueTgt, self->type()->valueType());

            }
        }
        // there is this value, we should return the one which is stored
        return extractPythonObject(i, self->type()->valueType());

    });
}

int PyDictInstance::pyInquiryConcrete(const char* op, const char* opErrRep) {
    // op == '__bool__'
    return type()->size(dataPtr()) != 0;
}
