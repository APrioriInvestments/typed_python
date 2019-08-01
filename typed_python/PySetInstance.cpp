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
    return new PyMethodDef[9]{{"add", (PyCFunction)PySetInstance::setAdd, METH_VARARGS, NULL},
                              {"discard", (PyCFunction)PySetInstance::setDiscard, METH_VARARGS,
                               NULL},
                              {"remove", (PyCFunction)PySetInstance::setRemove, METH_VARARGS, NULL},
                              {"clear", (PyCFunction)PySetInstance::setClear, METH_VARARGS, NULL},
                              {"copy", (PyCFunction)PySetInstance::setCopy, METH_VARARGS, NULL},
                              {"union", (PyCFunction)PySetInstance::setUnion, METH_VARARGS, NULL},
                              {"intersection", (PyCFunction)PySetInstance::setIntersection,
                               METH_VARARGS, NULL},
                              {"difference", (PyCFunction)PySetInstance::setDifference,
                               METH_VARARGS, NULL},
                              {NULL, NULL}};
}

PyObject* PySetInstance::try_remove(PyObject* o, PyObject* item, bool assertKeyError) {
    PySetInstance* self_w = (PySetInstance*)o;
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

    PyObjectHolder item(PyTuple_GetItem(args, 0));
    return try_remove(o, item, true);
}

PyObject* PySetInstance::setDiscard(PyObject* o, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "Set.discard takes one argument");
        return NULL;
    }

    PyObjectHolder item(PyTuple_GetItem(args, 0));
    return try_remove(o, item, false);
}

PyObject* PySetInstance::setClear(PyObject* o, PyObject* args) {
    if (args && PyTuple_Size(args) != 0) {
        PyErr_SetString(PyExc_TypeError, "Set.clear takes no arguments");
        return NULL;
    }
    PySetInstance* self_w = (PySetInstance*)o;
    self_w->type()->clear(self_w->dataPtr());
    Py_RETURN_NONE;
}

void PySetInstance::copy_elements(PyObject* dst, PyObject* src) {
    PySetInstance* dst_w = (PySetInstance*)dst;

    PyObjectStealer iterator(PyObject_GetIter(src));
    if (!iterator) {
        throw PythonExceptionSet();
    }

    PyObject* item;
    while ((item = PyObjectStealer(PyIter_Next(iterator)))) {

        if (!item) {
            if (PyErr_Occurred()) {
                throw PythonExceptionSet();
            } else {
                // no more values to iterate over
                break;
            }
        }

        Instance key(dst_w->type()->keyType(), [&](instance_ptr data) {
            PyInstance::copyConstructFromPythonInstance(dst_w->type()->keyType(), data, item, true);
        });

        if (try_insert_key(dst_w, item, key.data()) != 0) {
            throw PythonExceptionSet();
        }
    }
}

PyObject* PySetInstance::setCopy(PyObject* o, PyObject* args) {
    if (args && PyTuple_Size(args) != 0) {
        PyErr_SetString(PyExc_TypeError, "Set.copy takes no arguments");
        return NULL;
    }

    PySetInstance* new_inst = (PySetInstance*)PyInstance::tp_new(Py_TYPE(o), NULL, NULL);
    if (!new_inst) {
        return NULL;
    }
    new_inst->mIteratorOffset = 0;
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

PyObject* PySetInstance::set_intersection(PyObject* o, PyObject* other) {
    PySetInstance* new_inst = (PySetInstance*)PyInstance::tp_new(Py_TYPE(o), NULL, NULL);
    if (!new_inst) {
        return NULL;
    }
    new_inst->mIteratorOffset = 0;
    new_inst->mIteratorFlag = 0;
    new_inst->mIsMatcher = false;

    PyObjectStealer iterator(PyObject_GetIter(other));
    if (!iterator) {
        decref((PyObject*)new_inst);
        return NULL;
    }

    PySetInstance* self_w = (PySetInstance*)o;
    PyObject* item;
    while ((item = PyObjectStealer(PyIter_Next(iterator))) != NULL) {

        Instance key(self_w->type()->keyType(), [&](instance_ptr data) {
            PyInstance::copyConstructFromPythonInstance(self_w->type()->keyType(), data, item,
                                                        true);
        });

        instance_ptr existingLoc = self_w->type()->lookupKey(self_w->dataPtr(), key.data());
        if (!existingLoc) {
            continue;
        }
        if (existingLoc && try_insert_key(new_inst, item, key.data()) != 0) {
            decref((PyObject*)new_inst);
            return NULL;
        }
    }
    if (PyErr_Occurred()) {
        decref((PyObject*)new_inst);
        return NULL;
    }
    return (PyObject*)new_inst;
}

PyObject* PySetInstance::set_difference(PyObject* o, PyObject* other) {
    if (!PySet_Check(other)) {
        PyObject* new_inst = setCopy(o, NULL);
        if (!new_inst) {
            return NULL;
        }
        if (set_difference_update(new_inst, other) < 0) {
            decref(new_inst);
            return NULL;
        }
        return new_inst;
    }

    PySetInstance* new_inst = (PySetInstance*)PyInstance::tp_new(Py_TYPE(o), NULL, NULL);
    if (!new_inst) {
        return NULL;
    }
    new_inst->mIteratorOffset = 0;
    new_inst->mIteratorFlag = 0;
    new_inst->mIsMatcher = false;

    PyObjectStealer iterator(PyObject_GetIter(o));
    if (!iterator) {
        decref((PyObject*)new_inst);
        return NULL;
    }

    PySetInstance* other_w = (PySetInstance*)other;
    PyObject* item;
    while ((item = PyObjectStealer(PyIter_Next(iterator))) != NULL) {

        Instance key(other_w->type()->keyType(), [&](instance_ptr data) {
            PyInstance::copyConstructFromPythonInstance(other_w->type()->keyType(), data, item,
                                                        true);
        });

        instance_ptr existingLoc = other_w->type()->lookupKey(other_w->dataPtr(), key.data());
        if (!existingLoc && try_insert_key(new_inst, item, key.data()) != 0) {
            decref((PyObject*)new_inst);
            return NULL;
        }
    }
    if (PyErr_Occurred()) {
        decref((PyObject*)new_inst);
        return NULL;
    }
    return (PyObject*)new_inst;
}

int PySetInstance::set_difference_update(PyObject* o, PyObject* other) {
    if (o == other) {
        setClear(o, NULL);
        return 0;
    }

    PyObjectStealer iterator(PyObject_GetIter(other));
    if (!iterator) {
        return -1;
    }

    PySetInstance* self_w = (PySetInstance*)o;
    PyObject* item;
    while ((item = PyObjectStealer(PyIter_Next(iterator))) != NULL) {
        if (!try_remove(o, item)) {
            return -1;
        }
    }
    if (PyErr_Occurred()) {
        return -1;
    }
    return 0;
}

PyObject* PySetInstance::setDifference(PyObject* o, PyObject* args) {
    if (PyTuple_Size(args) == 0) {
        return setCopy(o, NULL);
    }

    PyObject* result;
    PyObjectHolder other(PyTuple_GetItem(args, 0));
    result = set_difference(o, other);
    if (!result) {
        return NULL;
    }

    for (Py_ssize_t i = 1; i < PyTuple_Size(args); ++i) {
        PyObjectHolder other(PyTuple_GetItem(args, i));
        if (set_difference_update(result, other) < 0) {
            decref(result);
            return NULL;
        }
    }

    return result;
}

PyObject* PySetInstance::setIntersection(PyObject* o, PyObject* args) {
    if (PyTuple_Size(args) == 0) {
        return setCopy(o, NULL);
    }

    for (size_t k = 0; k < PyTuple_Size(args); ++k) {
        PyObjectHolder item(PyTuple_GetItem(args, k));
        if (!PyIter_Check(item) && !PyList_Check(item) && !PySet_Check(item)
            && !PyTuple_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "Set.intersection one of args has wrong type");
            return NULL;
        }
    }

    PyObject* self_w = o;
    incref(o);
    try {
        for (size_t k = 0; k < PyTuple_Size(args); ++k) {
            PyObjectHolder item(PyTuple_GetItem(args, k));
            PyObject* set_result = set_intersection(self_w, item);
            if (!set_result) {
                decref(self_w);
                return NULL;
            }
            decref(self_w);
            self_w = set_result;
        }
    } catch (PythonExceptionSet& e) {
        decref(self_w);
        return NULL;
    } catch (std::exception& e) {
        decref(self_w);
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    }
    return self_w;
}

PyObject* PySetInstance::setUnion(PyObject* o, PyObject* args) {
    if (PyTuple_Size(args) < 1) {
        PyErr_SetString(PyExc_TypeError, "Set.union takes at least one argument");
        return NULL;
    }

    for (size_t k = 0; k < PyTuple_Size(args); ++k) {
        PyObjectHolder item(PyTuple_GetItem(args, k));
        if (!PyIter_Check(item) && !PyList_Check(item) && !PyAnySet_Check(item)
            && !PyTuple_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "Set.union one of args has wrong type");
            return NULL;
        }
    }

    PySetInstance* self_w = (PySetInstance*)o;
    PySetInstance* new_inst = (PySetInstance*)PyInstance::tp_new(Py_TYPE(o), NULL, NULL);
    if (!new_inst) {
        return NULL;
    }
    new_inst->mIteratorOffset = 0;
    new_inst->mIteratorFlag = 0;
    new_inst->mIsMatcher = false;

    try {
        // copy lhs to new set
        copy_elements((PyObject*)new_inst, (PyObject*)self_w);

        // copy rhs arg items to new set
        for (size_t k = 0; k < PyTuple_Size(args); ++k) {
            PyObjectHolder arg_item(PyTuple_GetItem(args, k));
            Type* item_type = extractTypeFrom(Py_TYPE((PyObject*)arg_item));
            if (item_type == self_w->type()->keyType()) {
                continue;
            }

            copy_elements((PyObject*)new_inst, (PyObject*)arg_item);
        }

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
                    } else {
                        break;
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
        } else if (PyUnicode_Check(pyRepresentation)) {
            if (PyUnicode_READY(pyRepresentation) < 0) {
                if (PyErr_Occurred()) {
                    throw PythonExceptionSet();
                }
            }

            Py_ssize_t str_size = PyUnicode_GetLength(pyRepresentation);
            for (size_t i = 0; i < str_size; ++i) {
                Py_UCS4 c = PyUnicode_ReadChar(pyRepresentation, i);
                Instance key(setType->keyType(), [&](instance_ptr data) {
                    copyConstructFromPythonInstance(setType->keyType(), data,
                                                    PyUnicode_FromFormat("%c", c));
                });

                instance_ptr found = setType->lookupKey(tgt, key.data());
                if (!found) {
                    setType->insertKey(tgt, key.data());
                }
            }
            return;
        }

    } catch (...) {
        setType->destroy(tgt);
        throw;
    }

    PyInstance::copyConstructFromPythonInstanceConcrete(setType, tgt, pyRepresentation, isExplicit);
}
