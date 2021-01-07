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

#include "PyConstDictInstance.hpp"

ConstDictType* PyConstDictInstance::type() {
    return (ConstDictType*)extractTypeFrom(((PyObject*)this)->ob_type);
}

// static
PyDoc_STRVAR(constDictItems_doc,
    "D.items() -> an iterable containing D's items."
);

PyObject* PyConstDictInstance::constDictItems(PyObject *o) {
    if (((PyInstance*)o)->mIteratorOffset != -1) {
        PyErr_SetString(PyExc_TypeError, "ConstDict iterators don't support 'items'");
        return NULL;
    }

    Type* self_type = extractTypeFrom(o->ob_type);

    PyInstance* w = (PyInstance*)o;

    if (self_type && self_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
        PyInstance* self = (PyInstance*)o->ob_type->tp_alloc(o->ob_type, 0);

        self->mIteratorOffset = 0;
        self->mIteratorFlag = 2;

        self->initialize([&](instance_ptr data) {
            self_type->copy_constructor(data, w->dataPtr());
        });

        return (PyObject*)self;
    }

    PyErr_SetString(PyExc_TypeError, ("Cannot iterate an instance of " + self_type->name()).c_str());
    return NULL;
}

// static
PyDoc_STRVAR(constDictKeys_doc,
    "D.keys() -> an iterable containing D's keys."
);
PyObject* PyConstDictInstance::constDictKeys(PyObject *o) {
    if (((PyInstance*)o)->mIteratorOffset != -1) {
        PyErr_SetString(PyExc_TypeError, "ConstDict iterators don't support 'keys'");
        return NULL;
    }

    Type* self_type = extractTypeFrom(o->ob_type);
    PyInstance* w = (PyInstance*)o;

    if (self_type && self_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
        PyInstance* self = (PyInstance*)o->ob_type->tp_alloc(o->ob_type, 0);

        self->mIteratorOffset = 0;
        self->mIteratorFlag = 0;

        self->initialize([&](instance_ptr data) {
            self_type->copy_constructor(data, w->dataPtr());
        });

        return (PyObject*)self;
    }

    PyErr_SetString(PyExc_TypeError, ("Cannot iterate an instance of " + self_type->name()).c_str());
    return NULL;
}

// static
PyDoc_STRVAR(constDictValues_doc,
    "D.values() -> an iterable containing D's values."
    );
PyObject* PyConstDictInstance::constDictValues(PyObject *o) {
    if (((PyInstance*)o)->mIteratorOffset != -1) {
        PyErr_SetString(PyExc_TypeError, "ConstDict iterators don't support 'values'");
        return NULL;
    }

    Type* self_type = extractTypeFrom(o->ob_type);
    PyInstance* w = (PyInstance*)o;

    if (self_type && self_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
        PyInstance* self = (PyInstance*)o->ob_type->tp_alloc(o->ob_type, 0);

        self->mIteratorOffset = 0;
        self->mIteratorFlag = 1;

        self->initialize([&](instance_ptr data) {
            self_type->copy_constructor(data, w->dataPtr());
        });


        return (PyObject*)self;
    }

    PyErr_SetString(PyExc_TypeError, ("Cannot iterate an instance of " + self_type->name()).c_str());
    return NULL;
}

// static
PyDoc_STRVAR(constDictGet_doc,
    "D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None."
    );
PyObject* PyConstDictInstance::constDictGet(PyObject* o, PyObject* args) {
    if (((PyInstance*)o)->mIteratorOffset != -1) {
        PyErr_SetString(PyExc_TypeError, "ConstDict iterators don't support 'get'");
        return NULL;
    }

    PyConstDictInstance* self_w = (PyConstDictInstance*)o;

    if (PyTuple_Size(args) < 1 || PyTuple_Size(args) > 2) {
        PyErr_SetString(PyExc_TypeError, "ConstDict.get takes one or two arguments");
        return NULL;
    }

    PyObjectHolder item(PyTuple_GetItem(args,0));
    PyObjectHolder ifNotFound(PyTuple_Size(args) == 2 ? PyTuple_GetItem(args,1) : Py_None);

    Type* self_type = extractTypeFrom(o->ob_type);
    Type* item_type = extractTypeFrom(item->ob_type);

    if (self_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
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
                    copyConstructFromPythonInstance(self_w->type()->keyType(), data, item, ConversionLevel::UpcastContainers);
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

        PyErr_SetString(PyExc_TypeError, "Invalid ConstDict lookup type");
        return NULL;
    }

    PyErr_SetString(PyExc_TypeError, "Wrong type!");
    return NULL;
}

PyObject* PyConstDictInstance::tp_iter_concrete() {
    PyInstance* output = (PyInstance*)PyInstance::initialize(type(), [&](instance_ptr data) {
        type()->copy_constructor(data, dataPtr());
    });

    output->mIteratorOffset = 0;
    output->mIteratorFlag = mIteratorFlag;

    return (PyObject*)output;
}

PyObject* PyConstDictInstance::tp_iternext_concrete() {
    if (mIteratorOffset >= type()->size(dataPtr())) {
        return NULL;
    }

    mIteratorOffset++;

    if (mIteratorFlag == 2) {
        auto t1 = extractPythonObject(
                type()->kvPairPtrKey(dataPtr(), mIteratorOffset-1),
                type()->keyType()
                );
        auto t2 = extractPythonObject(
                type()->kvPairPtrValue(dataPtr(), mIteratorOffset-1),
                type()->valueType()
                );

        auto res = PyTuple_Pack(2, t1, t2);

        decref(t1);
        decref(t2);

        return res;
    } else if (mIteratorFlag == 1) {
        return extractPythonObject(
            type()->kvPairPtrValue(dataPtr(), mIteratorOffset-1),
            type()->valueType()
            );
    } else {
        return extractPythonObject(
            type()->kvPairPtrKey(dataPtr(), mIteratorOffset-1),
            type()->keyType()
            );
    }
}

Py_ssize_t PyConstDictInstance::mp_and_sq_length_concrete() {
    return type()->size(dataPtr());
}

int PyConstDictInstance::sq_contains_concrete(PyObject* item) {
    Type* item_type = extractTypeFrom(item->ob_type);

    if (mIteratorFlag == 1) {
        // we're a values iterator
        Instance value(type()->valueType(), [&](instance_ptr data) {
            copyConstructFromPythonInstance(type()->valueType(), data, item, ConversionLevel::UpcastContainers);
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
            copyConstructFromPythonInstance(tupType, data, item, ConversionLevel::Implicit);
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
            copyConstructFromPythonInstance(type()->keyType(), data, item, ConversionLevel::UpcastContainers);
        });

        instance_ptr i = type()->lookupValueByKey(dataPtr(), key.data());

        return i ? 1 : 0;
    }
}

PyObject* PyConstDictInstance::pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr) {
    if (mIteratorOffset != -1) {
        return incref(Py_NotImplemented);
    }

    if (strcmp(op, "__add__") == 0) {
        Type* rhs_type = extractTypeFrom(rhs->ob_type);

        if (type() == rhs_type) {
            PyInstance* w_rhs = (PyInstance*)rhs;

            return PyInstance::initialize(type(), [&](instance_ptr data) {
                type()->addDicts(dataPtr(), w_rhs->dataPtr(), data);
            });
        } else {
            // to convert the RHS to the appropriate const dict type, RHS must
            // be a finite, iterable mapping.

            // check that it's mappable
            if (!PyMapping_Check(rhs)) {
                return incref(Py_NotImplemented);
            }

            // check that its iterable
            PyObjectStealer iterator(PyObject_GetIter(rhs));

            if (!iterator) {
                PyErr_Clear();
                return incref(Py_NotImplemented);
            }

            // check that its finite
            int64_t len = PyObject_Length(rhs);

            if (len == -1) {
                PyErr_Clear();
                return incref(Py_NotImplemented);
            }

            Instance other(type(), [&](instance_ptr data) {
                copyConstructFromPythonInstance(type(), data, rhs, ConversionLevel::Implicit);
            });

            return PyInstance::initialize(type(), [&](instance_ptr data) {
                type()->addDicts(dataPtr(), other.data(), data);
            });
        }
    }
    if (strcmp(op, "__sub__") == 0) {
        if (!PySequence_Check(rhs))
            return incref(Py_NotImplemented);
        Type* tupleOfKeysType = type()->tupleOfKeysType();

        //convert rhs to a relevant tuple of keys type.
        Instance keys(tupleOfKeysType, [&](instance_ptr data) {
            copyConstructFromPythonInstance(tupleOfKeysType, data, rhs, ConversionLevel::Implicit);
        });

        return PyInstance::initialize(type(), [&](instance_ptr data) {
            type()->subtractTupleOfKeysFromDict(dataPtr(), keys.data(), data);
        });
    }

    return PyInstance::pyOperatorConcrete(rhs, op, opErr);
}

bool PyConstDictInstance::compare_as_iterator_to_python_concrete(PyObject* other, int pyComparisonOp) {
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

PyObject* PyConstDictInstance::mp_subscript_concrete(PyObject* item) {
    if (mIteratorOffset != -1) {
        PyErr_SetString(PyExc_TypeError, "ConstDict iterators are not subscriptable");
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
            copyConstructFromPythonInstance(type()->keyType(), data, item, ConversionLevel::UpcastContainers);
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

PyMethodDef* PyConstDictInstance::typeMethodsConcrete(Type* t) {
    return new PyMethodDef [5] {
        {"get", (PyCFunction)PyConstDictInstance::constDictGet, METH_VARARGS, constDictGet_doc},
        {"items", (PyCFunction)PyConstDictInstance::constDictItems, METH_NOARGS, constDictItems_doc},
        {"keys", (PyCFunction)PyConstDictInstance::constDictKeys, METH_NOARGS, constDictKeys_doc},
        {"values", (PyCFunction)PyConstDictInstance::constDictValues, METH_NOARGS, constDictValues_doc},
        {NULL, NULL}
    };
}

void PyConstDictInstance::mirrorTypeInformationIntoPyTypeConcrete(ConstDictType* constDictT, PyTypeObject* pyType) {
    PyDict_SetItemString(pyType->tp_dict, "KeyType",
        typePtrToPyTypeRepresentation(constDictT->keyType())
    );
    PyDict_SetItemString(pyType->tp_dict, "ElementType",  // ElementType is what you get if you iterate
        typePtrToPyTypeRepresentation(constDictT->keyType())
    );
    PyDict_SetItemString(pyType->tp_dict, "ValueType",
        typePtrToPyTypeRepresentation(constDictT->valueType())
    );
}

bool PyConstDictInstance::compare_to_python_concrete(ConstDictType* dictType, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp) {
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
                copyConstructFromPythonInstance(
                    dictType->keyType(),
                    data,
                    key,
                    exact ? ConversionLevel::Signature : ConversionLevel::UpcastContainers
                );
            });
        } catch(PythonExceptionSet&) {
            PyErr_Clear();
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

bool PyConstDictInstance::pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, ConversionLevel level) {
    if (level != ConversionLevel::Signature) {
        if (PyDict_Check(pyRepresentation)) {
            return true;
        }

        return false;
    }

    return false;
}

void PyConstDictInstance::copyConstructFromPythonInstanceConcrete(ConstDictType* dictType, instance_ptr tgt, PyObject* pyRepresentation, ConversionLevel level) {
    if (level == ConversionLevel::Signature) {
        return PyInstance::copyConstructFromPythonInstanceConcrete(dictType, tgt, pyRepresentation, level);
    }

    if (PyDict_Check(pyRepresentation)) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;

        dictType->constructor(tgt, PyDict_Size(pyRepresentation), [&](instance_ptr outKey, instance_ptr outVal) {
            PyDict_Next(pyRepresentation, &pos, &key, &value);

            copyConstructFromPythonInstance(dictType->keyType(), outKey, key, ConversionLevel::UpcastContainers);

            try {
                copyConstructFromPythonInstance(dictType->valueType(), outVal, value, ConversionLevel::Implicit);
            } catch(...) {
                dictType->keyType()->destroy(outKey);
                throw;
            }
        });
        return;
    }

    PyObjectStealer iterator(PyObject_GetIter(pyRepresentation));

    if (!iterator) {
        throw PythonExceptionSet();
    }

    int64_t len = PyObject_Length(pyRepresentation);

    if (len == -1) {
        if (PyErr_Occurred()) {
            throw PythonExceptionSet();
        }
        len = 0;
    }

    dictType->constructor(tgt, len, [&](instance_ptr outKey, instance_ptr outVal) {
        PyObjectStealer key(PyIter_Next(iterator));

        if (!key) {
            if (PyErr_Occurred()) {
                throw PythonExceptionSet();
            }
            throw std::runtime_error("Length didn't match number of iterator items");
        }

        PyObjectStealer value(PyObject_GetItem(pyRepresentation, key));

        if (!value) {
            throw PythonExceptionSet();
        }

        copyConstructFromPythonInstance(dictType->keyType(), outKey, key, ConversionLevel::UpcastContainers);

        try {
            copyConstructFromPythonInstance(dictType->valueType(), outVal, value, ConversionLevel::Implicit);
        } catch(...) {
            dictType->keyType()->destroy(outKey);
            throw;
        }
    });
}

int PyConstDictInstance::pyInquiryConcrete(const char* op, const char* opErrRep) {
    // op == '__bool__'
    return type()->size(dataPtr()) != 0;
}

PyObject* PyConstDictInstance::tp_str_concrete() {
    return tp_repr_concrete();
}

PyObject* PyConstDictInstance::tp_repr_concrete() {
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
