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
#include "PySetInstance.hpp"

void PySetInstance::getDataFromNative(PySetInstance* src, std::function<void(instance_ptr)> func) {
    for (size_t i = 0; i < src->type()->slotCount(src->dataPtr())
                       && src->type()->slotPopulated(src->dataPtr(), i);
         ++i) {
        instance_ptr key = src->type()->keyAtSlot(src->dataPtr(), i);
        func(key);
    }
}

void PySetInstance::getDataFromNative(PyTupleOrListOfInstance* src,
                                      std::function<void(instance_ptr)> func) {
    for (size_t i = 0; i < src->type()->count(src->dataPtr()); ++i) {
        instance_ptr key = src->type()->eltPtr(src->dataPtr(), i);
        func(key);
    }
}


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
    Type* src_type = extractTypeFrom(Py_TYPE(src));
    // fastpath: first check whether 'src' is a typed_python Set, Tuple, or ListOf with the same
    // element type, and if so, bypass the python interpreter and insert the elements directly.
    if (src_type && (src_type->getTypeCategory() == Type::TypeCategory::catSet)
        && ((SetType*)src_type)->keyType() == dst_w->type()->keyType()) {
        getDataFromNative((PySetInstance*)src, [&](instance_ptr key) {
            if (try_insert_key(dst_w, src, key) != 0) {
                throw PythonExceptionSet();
            }
        });
    } else if (src_type
               && (src_type->getTypeCategory() == Type::TypeCategory::catListOf
                   || src_type->getTypeCategory() == Type::TypeCategory::catTupleOf)
               && ((TupleOrListOfType*)src_type)->getEltType() == dst_w->type()->keyType()) {
        getDataFromNative((PyTupleOrListOfInstance*)src, [&](instance_ptr key) {
            if (try_insert_key(dst_w, (PyObject*)src, key) != 0) {
                throw PythonExceptionSet();
            }
        });
    } else {
        iterate(src, [&](PyObject* item) {
            Instance key(dst_w->type()->keyType(), [&](instance_ptr data) {
                PyInstance::copyConstructFromPythonInstance(dst_w->type()->keyType(), data, item,
                                                            true);
            });
            if (try_insert_key(dst_w, item, key.data()) != 0) {
                throw PythonExceptionSet();
            }
        });
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

    PySetInstance* self_w = (PySetInstance*)o;
    Type* src_type = extractTypeFrom(Py_TYPE(other));
    try {
        if (src_type && (src_type->getTypeCategory() == Type::TypeCategory::catSet)
            && ((SetType*)src_type)->keyType() == self_w->type()->keyType()) {
            getDataFromNative((PySetInstance*)other, [&](instance_ptr key) {
                instance_ptr existingLoc = self_w->type()->lookupKey(self_w->dataPtr(), key);
                if (existingLoc && try_insert_key(new_inst, other, key) != 0) {
                    throw PythonExceptionSet();
                }
            });
        } else if (src_type
                   && (src_type->getTypeCategory() == Type::TypeCategory::catListOf
                       || src_type->getTypeCategory() == Type::TypeCategory::catTupleOf)
                   && ((TupleOrListOfType*)src_type)->getEltType() == self_w->type()->keyType()) {
            getDataFromNative((PyTupleOrListOfInstance*)other, [&](instance_ptr key) {
                instance_ptr existingLoc = self_w->type()->lookupKey(self_w->dataPtr(), key);
                if (existingLoc && try_insert_key(new_inst, other, key) != 0) {
                    throw PythonExceptionSet();
                }
            });
        } else {
            iterate(other, [&](PyObject* item) {
                Instance key(self_w->type()->keyType(), [&](instance_ptr data) {
                    PyInstance::copyConstructFromPythonInstance(self_w->type()->keyType(), data,
                                                                item, true);
                });
                instance_ptr existingLoc = self_w->type()->lookupKey(self_w->dataPtr(), key.data());
                if (existingLoc && try_insert_key(new_inst, item, key.data()) != 0) {
                    throw PythonExceptionSet();
                }
            });
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

PyObject* PySetInstance::set_difference(PyObject* o, PyObject* other) {
    PySetInstance* new_inst = NULL;
    PySetInstance* self_w = (PySetInstance*)o;
    Type* src_type = extractTypeFrom(Py_TYPE(other));

    if (src_type && (src_type->getTypeCategory() == Type::TypeCategory::catSet)
        && ((SetType*)src_type)->keyType() == self_w->type()->keyType()) {
        new_inst = (PySetInstance*)PyInstance::tp_new(Py_TYPE(o), NULL, NULL);
        if (!new_inst) {
            return NULL;
        }
        new_inst->mIteratorOffset = 0;
        new_inst->mIteratorFlag = 0;
        new_inst->mIsMatcher = false;

        try {
            PySetInstance* other_w = (PySetInstance*)other;
            getDataFromNative((PySetInstance*)o, [&](instance_ptr key) {
                instance_ptr existingLoc = other_w->type()->lookupKey(other_w->dataPtr(), key);
                if (!existingLoc && try_insert_key(new_inst, other, key) != 0) {
                    throw PythonExceptionSet();
                }
            });
        } catch (PythonExceptionSet& e) {
            decref((PyObject*)new_inst);
            return NULL;
        } catch (std::exception& e) {
            decref((PyObject*)new_inst);
            PyErr_SetString(PyExc_TypeError, e.what());
            return NULL;
        }
    } else {
        new_inst = (PySetInstance*)setCopy(o, NULL);
        if (!new_inst) {
            return NULL;
        }
        if (set_difference_update((PyObject*)new_inst, other) < 0) {
            decref((PyObject*)new_inst);
            return NULL;
        }
    }

    return (PyObject*)new_inst;
}

int PySetInstance::set_difference_update(PyObject* o, PyObject* other) {
    if (o == other) {
        setClear(o, NULL);
        return 0;
    }

    try {
        iterate(other, [&](PyObject* item) {
            if (!try_remove(o, item)) {
                throw PythonExceptionSet();
            }
        });
    } catch (PythonExceptionSet& e) {
        return -1;
    } catch (std::exception& e) {
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
             || PyUnicode_Check(pyRepresentation) || PyBytes_Check(pyRepresentation)
             || PyTuple_Check(pyRepresentation))) {
            iterate(pyRepresentation, [&](PyObject* item) {
                Instance key(setType->keyType(), [&](instance_ptr data) {
                    copyConstructFromPythonInstance(setType->keyType(), data, item);
                });

                instance_ptr found = setType->lookupKey(tgt, key.data());
                if (!found) {
                    setType->insertKey(tgt, key.data());
                }
            });
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

int PySetInstance::pyInquiryConcrete(const char* op, const char* opErrRep) {
    // op == '__bool__'
    return type()->size(dataPtr()) != 0;
}
