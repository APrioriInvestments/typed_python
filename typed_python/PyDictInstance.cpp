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

#include "PyDictInstance.hpp"

DictType* PyDictInstance::type() {
    return (DictType*)extractTypeFrom(((PyObject*)this)->ob_type);
}

// static
PyObject* PyDictInstance::dictItems(PyObject *o) {
    if (((PyInstance*)o)->mIteratorOffset != -1) {
        PyErr_SetString(PyExc_TypeError, "dict iterators don't support 'items'");
        return NULL;
    }

    PyInstance* result = ((PyInstance*)o)->duplicate();

    result->mIteratorOffset = 0;
    result->mIteratorFlag = 2;
    result->mIsMatcher = false;

    return (PyObject*)result;
}

// static
PyObject* PyDictInstance::dictKeys(PyObject *o) {
    if (((PyInstance*)o)->mIteratorOffset != -1) {
        PyErr_SetString(PyExc_TypeError, "dict iterators don't support 'keys'");
        return NULL;
    }

    PyInstance* result = ((PyInstance*)o)->duplicate();

    result->mIteratorOffset = 0;
    result->mIteratorFlag = 0;
    result->mIsMatcher = false;

    return (PyObject*)result;
}

// static
PyObject* PyDictInstance::dictValues(PyObject *o) {
    if (((PyInstance*)o)->mIteratorOffset != -1) {
        PyErr_SetString(PyExc_TypeError, "dict iterators don't support 'values'");
        return NULL;
    }

    PyInstance* result = ((PyInstance*)o)->duplicate();

    result->mIteratorOffset = 0;
    result->mIteratorFlag = 1;
    result->mIsMatcher = false;

    return (PyObject*)result;
}

// static
PyObject* PyDictInstance::dictUpdate(PyObject* o, PyObject* args) {
    PyDictInstance* self_w = (PyDictInstance*)o;

    if (self_w->mIteratorOffset != -1) {
        PyErr_SetString(PyExc_TypeError, "dict iterators don't support 'update'");
        return NULL;
    }

    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "Dict.update takes one argument");
        return NULL;
    }

    PyObjectHolder item(PyTuple_GetItem(args,0));

    Type* item_type = extractTypeFrom(item->ob_type);
    DictType* self_type = (DictType*)extractTypeFrom(o->ob_type);

    if (self_type == item_type) {
        instance_ptr selfPtr = self_w->dataPtr();

        ((DictType*)item_type)->visitKeyValuePairsAsSeparateArgs(
            ((PyDictInstance*)(PyObject*)item)->dataPtr(),
            [&](instance_ptr key, instance_ptr value) {
                instance_ptr existingLoc = self_type->lookupValueByKey(selfPtr, key);
                if (existingLoc) {
                    self_type->valueType()->assign(existingLoc, value);
                } else {
                    instance_ptr newLoc = self_type->insertKey(selfPtr, key);
                    self_type->valueType()->copy_constructor(newLoc, value);
                }

                return true;
            }
        );

        return incref(Py_None);
    } else {
        return translateExceptionToPyObject([&]() {
            iterate(item, [&](PyObject* key) {
                PyObjectHolder value(PyObject_GetItem(item, key));
                if (!value) {
                    throw PythonExceptionSet();
                }

                if (self_w->mp_ass_subscript_concrete(key, value) == -1) {
                    throw PythonExceptionSet();
                }
            });

            return incref(Py_None);
        });
    }
}

// static
PyObject* PyDictInstance::dictGet(PyObject* o, PyObject* args) {
    PyDictInstance* self_w = (PyDictInstance*)o;

    if (self_w->mIteratorOffset != -1) {
        PyErr_SetString(PyExc_TypeError, "dict iterators are not subscriptable");
        return NULL;
    }

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

// static
PyObject* PyDictInstance::dictClear(PyObject* o) {
    PyDictInstance* self_w = (PyDictInstance*)o;

    if (self_w->mIteratorOffset != -1) {
        PyErr_SetString(PyExc_TypeError, "dict iterators don't allow 'clear'");
        return NULL;
    }

    Type* self_type = extractTypeFrom(o->ob_type);

    ((DictType*)self_type)->clear(self_w->dataPtr());

    return incref(Py_None);
}

void copy_elements(PyObject* dst, PyObject* src) {
    PyDictInstance* dst_w = (PyDictInstance*)dst;
    iterate(src, [&](PyObject* key) {
        PyObjectHolder value(PyObject_GetItem(src, key));
        if (!value) {
            throw PythonExceptionSet();
        }

        if (dst_w->mp_ass_subscript_concrete(key, value) == -1) {
            throw PythonExceptionSet();
        }
    });
}

PyObject* PyDictInstance::dictCopy(PyObject* o, PyObject* args) {
    if (args && PyTuple_Size(args) != 0) {
        PyErr_SetString(PyExc_TypeError, "Dict.copy takes no arguments");
        return NULL;
    }

    PySetInstance* new_inst = (PySetInstance*)PyInstance::tp_new(Py_TYPE(o), NULL, NULL);
    if (!new_inst) {
        return NULL;
    }
    new_inst->mIteratorOffset = -1;
    new_inst->mIteratorFlag = 0;
    new_inst->mIsMatcher = false;

    try {
        copy_elements((PyObject*)new_inst, (PyObject*)(PySetInstance*)o);
    } catch (PythonExceptionSet& e) {
        decref((PyObject*)new_inst);
        return NULL;
    } catch (std::exception& e) {
        decref((PyObject*)new_inst);
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    }
    return (PyObject*)new_inst;
}

PyObject* PyDictInstance::tp_iter_concrete() {
    return createIteratorToSelf(mIteratorFlag);
}

bool PyDictInstance::compare_as_iterator_to_python_concrete(PyObject* other, int pyComparisonOp) {
    if (extractTypeFrom(other->ob_type) != type()) {
        return ((PyInstance*)this)->compare_as_iterator_to_python_concrete(other, pyComparisonOp);
    }

    if (((PyInstance*)other)->mIteratorFlag != mIteratorFlag) {
        return ((PyInstance*)this)->compare_as_iterator_to_python_concrete(other, pyComparisonOp);
    }

    if (mIteratorFlag == 1) {
        //.values() are never comparable.
        return ((PyInstance*)this)->compare_as_iterator_to_python_concrete(other, pyComparisonOp);
    }

    if (mIteratorFlag == 2) {
        // just compare the dicts
        return type()->cmp(dataPtr(), ((PyInstance*)other)->dataPtr(), pyComparisonOp, false);
    }

    // just compare the dict keys
    return type()->cmp(dataPtr(), ((PyInstance*)other)->dataPtr(), pyComparisonOp, false, false /* don't compare values */);
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

    if (mIteratorFlag == 1) {
        // we're a values iterator
        Instance value(type()->valueType(), [&](instance_ptr data) {
            copyConstructFromPythonInstance(type()->valueType(), data, item);
        });

        bool found = false;
        type()->visitValues(dataPtr(), [&](instance_ptr valueInst) {
            if (type()->valueType()->cmp(value.data(), valueInst, Py_EQ)) {
                found = true;
                return false;
            }

            return true;
        });

        return found;
    }

    if (mIteratorFlag == 2) {
        // we're an items iterator
        bool found = false;

        Type* tupType = Tuple::Make({type()->keyType(), type()->valueType()});

        Instance value(tupType, [&](instance_ptr data) {
            copyConstructFromPythonInstance(tupType, data, item);
        });

        type()->visitKeyValuePairs(dataPtr(), [&](instance_ptr keyValuePairInst) {
            if (tupType->cmp(value.data(), keyValuePairInst, Py_EQ)) {
                found = true;
                return false;
            }

            return true;
        });

        return found;
    }

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
    if (mIteratorOffset != -1) {
        return incref(Py_NotImplemented);
    }

    return PyInstance::pyOperatorConcrete(rhs, op, opErr);
}

PyObject* PyDictInstance::mp_subscript_concrete(PyObject* item) {
    if (mIteratorOffset != -1) {
        PyErr_SetString(PyExc_TypeError, "dict iterators are not subscriptable");
        return NULL;
    }

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
    if (mIteratorOffset != -1) {
        PyErr_SetString(PyExc_TypeError, "dict iterators are not subscriptable");
        return -1;
    }

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
            copyConstructFromPythonInstance(type()->valueType(), data, value, true);
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
    return new PyMethodDef [10] {
        {"get", (PyCFunction)PyDictInstance::dictGet, METH_VARARGS, NULL},
        {"clear", (PyCFunction)PyDictInstance::dictClear, METH_NOARGS, NULL},
        {"copy", (PyCFunction)PyDictInstance::dictCopy, METH_VARARGS, NULL},
        {"update", (PyCFunction)PyDictInstance::dictUpdate, METH_VARARGS, NULL},
        {"items", (PyCFunction)PyDictInstance::dictItems, METH_NOARGS, NULL},
        {"keys", (PyCFunction)PyDictInstance::dictKeys, METH_NOARGS, NULL},
        {"values", (PyCFunction)PyDictInstance::dictValues, METH_NOARGS, NULL},
        {"setdefault", (PyCFunction)PyDictInstance::setDefault, METH_VARARGS, NULL},
        {"pop", (PyCFunction)PyDictInstance::pop, METH_VARARGS, NULL},
        {NULL, NULL}
    };
}

void PyDictInstance::mirrorTypeInformationIntoPyTypeConcrete(DictType* dictT, PyTypeObject* pyType) {
    PyDict_SetItemString(pyType->tp_dict, "KeyType",
        typePtrToPyTypeRepresentation(dictT->keyType())
    );
    PyDict_SetItemString(pyType->tp_dict, "ElementType",  // ElementType is what you get if you iterate
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
            return false;
        } catch(...) {
            //if we can't convert to keyType, we're not equal
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

        if (self->mIteratorOffset != -1) {
            PyErr_SetString(PyExc_TypeError, "dict iterators don't support setdefault");
            throw PythonExceptionSet();
        }

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
            throw std::runtime_error("Somehow 'self' was not a Dictionary.");
        }

        instance_ptr i;
        instance_ptr lookupKey;
        Instance key;

        if (itemType == keyType) {
            PyInstance *item_w = (PyInstance*)(PyObject*)item;
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
            if (argsNumber == 1) {
                if (!valueType->is_default_constructible()) {
                    throw std::runtime_error("Can't default construct " + valueType->name());
                }

                // we don't try to convert 'None' directly to a value because in most cases
                // that wouldn't work, and it's not that useful. Instead, we'll construct
                // the default item for the set.
                Instance i(valueType, [&](instance_ptr data) {
                    valueType->constructor(data);
                });

                instance_ptr valueTgt = self->type()->insertKey(self->dataPtr(), lookupKey);
                self->type()->valueType()->copy_constructor(valueTgt, i.data());

                return extractPythonObject(valueTgt, self->type()->valueType());
            }
            else if (extractTypeFrom(ifNotFound->ob_type) == self->type()->valueType()) {
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

//static
PyObject* PyDictInstance::pop(PyObject* o, PyObject* args) {
    return translateExceptionToPyObject([&]() {
        const int argsNumber = PyTuple_Size(args);

        if (argsNumber < 1 || argsNumber > 2) {
            throw std::runtime_error("Dict.pop takes one or two arguments.");
        }

        PyDictInstance *self = (PyDictInstance *) o;

        if (self->mIteratorOffset != -1) {
            PyErr_SetString(PyExc_TypeError, "dict iterators don't support 'pop'");
            throw PythonExceptionSet();
        }

        /*
         * The function is called with pop(key, ifNotFound=None)
         */
        PyObjectHolder item(PyTuple_GetItem(args, 0));
        PyObjectHolder ifNotFound(argsNumber == 2 ? PyTuple_GetItem(args, 1) : Py_None);

        Type *selfType = extractTypeFrom(o->ob_type);
        Type *itemType = extractTypeFrom(item->ob_type);

        Type *keyType = self->type()->keyType();
        Type *valueType = self->type()->valueType();

        if (selfType->getTypeCategory() != Type::TypeCategory::catDict) {
            throw std::runtime_error("Somehow 'self' was not a Dictionary.");
        }

        Instance key;
        instance_ptr keyPtr;

        if (itemType == keyType) {
            PyInstance *item_w = (PyInstance*)(PyObject*)item;
            keyPtr = item_w->dataPtr();
        } else {
            key = Instance(keyType, [&](instance_ptr data) {
                copyConstructFromPythonInstance(keyType, data, item);
            });
            keyPtr = key.data();
        }

        instance_ptr valuePtr = self->type()->lookupValueByKey(self->dataPtr(), keyPtr);

        if (valuePtr) {
            PyObject* result = extractPythonObject(valuePtr, valueType);

            self->type()->deleteKey(self->dataPtr(), keyPtr);

            return result;
        }

        if (argsNumber == 1) {
            PyErr_SetObject(PyExc_KeyError, item);
            throw PythonExceptionSet();
        }

        return incref((PyObject*)ifNotFound);
    });
}

int PyDictInstance::pyInquiryConcrete(const char* op, const char* opErrRep) {
    // op == '__bool__'
    return type()->size(dataPtr()) != 0;
}

PyObject* PyDictInstance::tp_str_concrete() {
    return tp_repr_concrete();
}

PyObject* PyDictInstance::tp_repr_concrete() {
    if (mIteratorOffset == -1) {
        return ((PyInstance*)this)->tp_repr_concrete();
    }

    std::ostringstream str;
    ReprAccumulator accumulator(str);

    if (mIteratorFlag == 0) {
        type()->repr_keys(dataPtr(), accumulator);
    }
    if (mIteratorFlag == 1) {
        type()->repr_values(dataPtr(), accumulator);
    }
    if (mIteratorFlag == 2) {
        type()->repr_items(dataPtr(), accumulator);
    }

    return PyUnicode_FromString(str.str().c_str());
}
