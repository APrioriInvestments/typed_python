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

#include "PySetInstance.hpp"

PyMethodDef* PySetInstance::typeMethodsConcrete(Type* t) {
    return new PyMethodDef[6]{{"add", (PyCFunction)PySetInstance::setAdd, METH_VARARGS, NULL},
                              {"discard", (PyCFunction)PySetInstance::setDiscard, METH_VARARGS,
                               NULL},
                              {"remove", (PyCFunction)PySetInstance::setRemove, METH_VARARGS, NULL},
                              {"clear", (PyCFunction)PySetInstance::setClear, METH_VARARGS, NULL},
                              {NULL, NULL}};
}

PyObject* PySetInstance::try_remove(PyObject* o, PyObject* args, bool assertKeyError) {
    PySetInstance* self_w = (PySetInstance*)o;
    PyObjectHolder item(PyTuple_GetItem(args, 0));
    Type* item_type = extractTypeFrom(item->ob_type);
    Type* self_type = extractTypeFrom(o->ob_type);

    if (self_type->getTypeCategory() == Type::TypeCategory::catSet) {
        if (item_type == self_w->type()->keyType()) {
            PyInstance* item_w = (PyInstance*)(PyObject*)item;
            instance_ptr key = item_w->dataPtr();
            bool found_and_discarded = self_w->type()->discard(self_w->dataPtr(), key);
            if (assertKeyError && !found_and_discarded) {
                PyErr_SetObject(PyExc_KeyError, item);
                return NULL;
            }
        } else {
            try {
                Instance key(self_w->type()->keyType(), [&](instance_ptr data) {
                    copyConstructFromPythonInstance(self_w->type()->keyType(), data, item);
                });
                bool found_and_discarded = self_w->type()->discard(self_w->dataPtr(), key.data());
                if (assertKeyError && !found_and_discarded) {
                    PyErr_SetObject(PyExc_KeyError, item);
                    return NULL;
                }
            } catch (PythonExceptionSet& e) {
                return NULL;
            } catch (std::exception& e) {
                PyErr_SetString(PyExc_TypeError, e.what());
                return NULL;
            }
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Wrong type!");
        return NULL;
    }

    Py_RETURN_NONE;
}

PyObject* PySetInstance::setRemove(PyObject* o, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "Set.remove takes one argument");
        return NULL;
    }

    return try_remove(o, args, true);
}

PyObject* PySetInstance::setDiscard(PyObject* o, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "Set.discard takes one argument");
        return NULL;
    }

    return try_remove(o, args, false);
}

PyObject* PySetInstance::setClear(PyObject* o, PyObject* args) {
    if (PyTuple_Size(args) != 0) {
        PyErr_SetString(PyExc_TypeError, "Set.clear takes no arguments");
        return NULL;
    }
    PySetInstance* self_w = (PySetInstance*)o;
    self_w->type()->clear(self_w->dataPtr());
    Py_RETURN_NONE;
}

int PySetInstance::sq_contains_concrete(PyObject* item) {
    Type* item_type = extractTypeFrom(Py_TYPE(item));
    if (item_type == type()->keyType()) {
        PyInstance* item_w = (PyInstance*)item;
        instance_ptr i = type()->lookupKey(dataPtr(), item_w->dataPtr());
        return i ? 1 : 0;
    } else {
        Instance key(type()->keyType(), [&](instance_ptr data) {
            copyConstructFromPythonInstance(type()->keyType(), data, item);
        });
        instance_ptr i = type()->lookupKey(dataPtr(), key.data());
        return i ? 1 : 0;
    }
}

PyObject* PySetInstance::setAdd(PyObject* o, PyObject* args) {
    PySetInstance* self_w = (PySetInstance*)o;
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "Set.add takes one argument");
        return NULL;
    }

    PyObjectHolder item(PyTuple_GetItem(args, 0));
    Type* item_type = extractTypeFrom(item->ob_type);
    Type* self_type = extractTypeFrom(o->ob_type);

    if (self_type->getTypeCategory() == Type::TypeCategory::catSet) {
        if (item_type == self_w->type()->keyType()) {
            PyInstance* item_w = (PyInstance*)(PyObject*)item;
            instance_ptr key = item_w->dataPtr();
            if (try_insert_key(self_w, item, key) != 0)
                return NULL;
        } else {
            try {
                Instance key(self_w->type()->keyType(), [&](instance_ptr data) {
                    copyConstructFromPythonInstance(self_w->type()->keyType(), data, item);
                });
                if (try_insert_key(self_w, item, key.data()) != 0)
                    return NULL;
            } catch (PythonExceptionSet& e) {
                return NULL;
            } catch (std::exception& e) {
                PyErr_SetString(PyExc_TypeError, e.what());
                return NULL;
            }
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Wrong type!");
        return NULL;
    }

    Py_RETURN_NONE;
}

Py_ssize_t PySetInstance::mp_and_sq_length_concrete() {
    return type()->size(dataPtr());
}

PyObject* PySetInstance::tp_iter_concrete() {
    return createIteratorToSelf(mIteratorFlag);
}

PyObject* PySetInstance::tp_iternext_concrete() {
    if (mIteratorOffset == 0) {
        // search forward to find the first slot
        while (mIteratorOffset < type()->slotCount(dataPtr())
               && !type()->slotPopulated(dataPtr(), mIteratorOffset)) {
            mIteratorOffset++;
        }
    }

    if (mIteratorOffset >= type()->slotCount(dataPtr())) {
        return NULL;
    }

    int32_t curSlot = mIteratorOffset;

    mIteratorOffset++;
    while (mIteratorOffset < type()->slotCount(dataPtr())
           && !type()->slotPopulated(dataPtr(), mIteratorOffset)) {
        mIteratorOffset++;
    }

    return extractPythonObject(type()->keyAtSlot(dataPtr(), curSlot), type()->keyType());
}

int PySetInstance::try_insert_key(PySetInstance* self, PyObject* pyKey, instance_ptr key) {
    instance_ptr existingLoc = self->type()->lookupKey(self->dataPtr(), key);
    if (!existingLoc)
        self->type()->insertKey(self->dataPtr(), key);
    return 0;
}

void PySetInstance::mirrorTypeInformationIntoPyTypeConcrete(SetType* setType,
                                                            PyTypeObject* pyType) {
    PyDict_SetItemString(pyType->tp_dict, "KeyType",
                         typePtrToPyTypeRepresentation(setType->keyType()));
}

SetType* PySetInstance::type() {
    return (SetType*)extractTypeFrom(((PyObject*)this)->ob_type);
}

void PySetInstance::copyConstructFromPythonInstanceConcrete(SetType* setType, instance_ptr tgt,
                                                            PyObject* pyRepresentation,
                                                            bool isExplicit) {
    setType->constructor(tgt);
    try {
        if ((PyList_Check(pyRepresentation) || PySet_Check(pyRepresentation)
             || PyBytes_Check(pyRepresentation) || PyTuple_Check(pyRepresentation))) {
            PyObjectStealer iterator(PyObject_GetIter(pyRepresentation));
            if (!iterator) {
                throw PythonExceptionSet();
            }

            PyObject* item;
            while ((item = PyObjectStealer(PyIter_Next(iterator)))) {
                if (!item) {
                    if (PyErr_Occurred()) {
                        throw PythonExceptionSet();
                    }
                }

                Instance key(setType->keyType(), [&](instance_ptr data) {
                    copyConstructFromPythonInstance(setType->keyType(), data, item);
                });

                instance_ptr found = setType->lookupKey(tgt, key.data());
                if (!found) {
                    setType->insertKey(tgt, key.data());
                }
            }
            return;
        } else if (PyNumber_Check(pyRepresentation)) {
            Instance key(setType->keyType(), [&](instance_ptr data) {
                copyConstructFromPythonInstance(setType->keyType(), data, pyRepresentation);
            });

            instance_ptr found = setType->lookupKey(tgt, key.data());
            if (!found) {
                setType->insertKey(tgt, key.data());
            }
            return;
        }

    } catch (...) {
        setType->destroy(tgt);
        throw;
    }
    PyInstance::copyConstructFromPythonInstanceConcrete(setType, tgt, pyRepresentation, isExplicit);
}
