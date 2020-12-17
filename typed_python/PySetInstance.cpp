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
                    copyConstructFromPythonInstance(self_w->type()->keyType(), data, item, ConversionLevel::UpcastContainers);
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

PyObject* PySetInstance::try_add_if_not_found(PyObject* o, PySetInstance* to_be_added, PyObject* item) {
    PySetInstance* self_w = (PySetInstance*)o;
    Type* item_type = extractTypeFrom(item->ob_type);
    Type* self_type = extractTypeFrom(o->ob_type);

    if (self_type->getTypeCategory() == Type::TypeCategory::catSet) {
        if (item_type == self_w->type()->keyType()) {
            PyInstance* item_w = (PyInstance*)(PyObject*)item;
            instance_ptr key = item_w->dataPtr();
            if (!self_w->type()->lookupKey(self_w->dataPtr(), key)) {
                to_be_added->type()->insertKey(to_be_added->dataPtr(), key);
            }
            else
                return NULL;
        } else {
            try {
                Instance key(self_w->type()->keyType(), [&](instance_ptr data) {
                    copyConstructFromPythonInstance(self_w->type()->keyType(), data, item, ConversionLevel::UpcastContainers);
                });
                if (!self_w->type()->lookupKey(self_w->dataPtr(), key.data())) {
                    to_be_added->type()->insertKey(to_be_added->dataPtr(), key.data());
                }
                else
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

PyDoc_STRVAR(setRemove_doc,
    "s.remove(item) -> None, and item is removed from s\n"
    "\n"
    "Raises KeyError if item is not in set.\n"
    );
PyObject* PySetInstance::setRemove(PyObject* o, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "Set.remove takes one argument");
        return NULL;
    }

    PyObjectHolder item(PyTuple_GetItem(args, 0));
    return try_remove(o, item, true);
}

PyDoc_STRVAR(setDiscard_doc,
    "s.discard(item) -> None, and item is removed from s\n"
    "\n"
    "No action if item not found.  Compare s.remove().\n"
    );
PyObject* PySetInstance::setDiscard(PyObject* o, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "Set.discard takes one argument");
        return NULL;
    }

    PyObjectHolder item(PyTuple_GetItem(args, 0));
    return try_remove(o, item, false);
}

PyDoc_STRVAR(setClear_doc,
    "s.clear() -> None, and s becomes the empty set"
    );
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
            insertKey(dst_w, src, key);
        });
    } else if (src_type
               && (src_type->getTypeCategory() == Type::TypeCategory::catListOf
                   || src_type->getTypeCategory() == Type::TypeCategory::catTupleOf)
               && ((TupleOrListOfType*)src_type)->getEltType() == dst_w->type()->keyType()) {
        getDataFromNative((PyTupleOrListOfInstance*)src, [&](instance_ptr key) {
            insertKey(dst_w, (PyObject*)src, key);
        });
    } else {
        iterate(src, [&](PyObject* item) {
            Instance key(dst_w->type()->keyType(), [&](instance_ptr data) {
                PyInstance::copyConstructFromPythonInstance(dst_w->type()->keyType(), data, item,
                                                            ConversionLevel::UpcastContainers);
            });
            insertKey(dst_w, item, key.data());
        });
    }
}

PyDoc_STRVAR(setCopy_doc,
    "s.copy() -> shallow copy of s"
    );
PyObject* PySetInstance::setCopy(PyObject* o, PyObject* args) {
    if (args && PyTuple_Size(args) != 0) {
        PyErr_SetString(PyExc_TypeError, "Set.copy takes no arguments");
        return NULL;
    }

    PySetInstance* new_inst = (PySetInstance*)PyInstance::tp_new(Py_TYPE(o), NULL, NULL);
    if (!new_inst) {
        return NULL;
    }
    new_inst->mIteratorOffset = -1;
    new_inst->mIteratorFlag = 0;

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
    new_inst->mIteratorOffset = -1;
    new_inst->mIteratorFlag = 0;

    PySetInstance* self_w = (PySetInstance*)o;
    Type* src_type = extractTypeFrom(Py_TYPE(other));
    try {
        if (src_type && (src_type->getTypeCategory() == Type::TypeCategory::catSet)
            && ((SetType*)src_type)->keyType() == self_w->type()->keyType()) {
            getDataFromNative((PySetInstance*)other, [&](instance_ptr key) {
                instance_ptr existingLoc = self_w->type()->lookupKey(self_w->dataPtr(), key);
                if (existingLoc) {
                    insertKey(new_inst, other, key);
                }
            });
        } else if (src_type
                   && (src_type->getTypeCategory() == Type::TypeCategory::catListOf
                       || src_type->getTypeCategory() == Type::TypeCategory::catTupleOf)
                   && ((TupleOrListOfType*)src_type)->getEltType() == self_w->type()->keyType()) {
            getDataFromNative((PyTupleOrListOfInstance*)other, [&](instance_ptr key) {
                instance_ptr existingLoc = self_w->type()->lookupKey(self_w->dataPtr(), key);
                if (existingLoc) {
                    insertKey(new_inst, other, key);
                }
            });
        } else {
            iterate(other, [&](PyObject* item) {
                Instance key(self_w->type()->keyType(), [&](instance_ptr data) {
                    PyInstance::copyConstructFromPythonInstance(self_w->type()->keyType(), data,
                                                                item, ConversionLevel::UpcastContainers);
                });
                instance_ptr existingLoc = self_w->type()->lookupKey(self_w->dataPtr(), key.data());
                if (existingLoc) {
                    insertKey(new_inst, item, key.data());
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

// returns 0 on success, -1 on error
int PySetInstance::set_intersection_update(PyObject* o, PyObject* other) {
    if (o == other) {
        return 0;
    }
    Type* src_type = extractTypeFrom(Py_TYPE(other));
    bool fastpath = src_type && (src_type->getTypeCategory() == Type::TypeCategory::catSet)
        && ((SetType*)src_type)->keyType() == ((PySetInstance*)o)->type()->keyType();

    try {
        if (fastpath) {
            SetType* o_type = (SetType*)extractTypeFrom(o->ob_type);
            PySetInstance *o_w = (PySetInstance*)o;
            PySetInstance *other_w = (PySetInstance*)other;
            o_type->visitSetElements(o_w->dataPtr(), [&](instance_ptr item) {
                if (!o_type->lookupKey(other_w->dataPtr(), item)) {
                    o_type->discard(o_w->dataPtr(), item);
                }
                return true;
            });
        }
        // TODO: implement efficient algorithms for sequence objects
        // else if (PySequence_Check(other)) {
        // }
        else {
            PyObject *copy = PySetInstance::setCopy(o, NULL);
            PySetInstance* copy_w = (PySetInstance*)copy;
            PySetInstance* self_w = (PySetInstance*)o;
            self_w->type()->clear(self_w->dataPtr());

            iterate(other, [&](PyObject* item) {
                    Instance key(self_w->type()->keyType(), [&](instance_ptr data) {
                        copyConstructFromPythonInstance(self_w->type()->keyType(), data, item, ConversionLevel::UpcastContainers);
                    });
                    if (copy_w->type()->lookupKey(copy_w->dataPtr(), key.data()))
                        insertKey(self_w, item, key.data());
            });
            decref(copy);
        }
    } catch (PythonExceptionSet& e) {
        return -1;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return -1;
    }

    return 0;
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
        new_inst->mIteratorOffset = -1;
        new_inst->mIteratorFlag = 0;

        try {
            PySetInstance* other_w = (PySetInstance*)other;
            getDataFromNative((PySetInstance*)o, [&](instance_ptr key) {
                instance_ptr existingLoc = other_w->type()->lookupKey(other_w->dataPtr(), key);
                if (!existingLoc) {
                    insertKey(new_inst, other, key);
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

// returns 0 on success, -1 on error
int PySetInstance::set_difference_update(PyObject* o, PyObject* other) {
    if (o == other) {
        setClear(o, NULL);
        return 0;
    }

    int ret = 0;
    try {
        iterateWithEarlyExit(other, [&](PyObject* item) {
            if (!try_remove(o, item)) {
                ret = -1;
                return false;
            }
            return true;
        });
    } catch (PythonExceptionSet& e) {
        return -1;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return -1;
    }

    return ret;
}

// returns 1 on true, 0 on false, -1 on error
int PySetInstance::set_is_subset(PyObject *o, PyObject* other) {
    if (o == other) {
        return (int)true;
    }
    PySetInstance* self_w = (PySetInstance*)o;
    PySetInstance* shadow = (PySetInstance*)PyInstance::tp_new(Py_TYPE(o), NULL, NULL);
    if (!shadow) {
        return -1;
    }

    if (self_w->mp_and_sq_length_concrete() == 0) {
        return (int)true;
    }

    bool ret = false;
    try {
        iterateWithEarlyExit(other, [&](PyObject* item) {
            Instance key(self_w->type()->keyType(), [&](instance_ptr data) {
                copyConstructFromPythonInstance(self_w->type()->keyType(), data, item, ConversionLevel::UpcastContainers);
            });
            if (self_w->type()->lookupKey(self_w->dataPtr(), key.data())) {
                insertKey(shadow, item, key.data());
                if (shadow->mp_and_sq_length_concrete() == self_w->mp_and_sq_length_concrete()) {
                    ret = true;
                    return false;
                }
            }
            return true;
        });
    } catch (PythonExceptionSet& e) {
        return -1;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return -1;
    }
    return (int)ret;
}

// returns 1 on true, 0 on false, -1 on error
int PySetInstance::set_is_superset(PyObject *o, PyObject* other) {
    if (o == other) {
        return (int)true;
    }
    PySetInstance* self_w = (PySetInstance*)o;
    bool contains = true;
    try {
        iterateWithEarlyExit(other, [&](PyObject* item) {
            contains = self_w->sq_contains_concrete(item);
            return contains;
        });
    } catch (PythonExceptionSet& e) {
        return -1;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return -1;
    }

    return (int)contains;
}

// returns 1 on true, 0 on false, -1 on error
int PySetInstance::set_is_disjoint(PyObject *o, PyObject* other) {
    if (o == other) {
        return (int)false;
    }
    PySetInstance* self_w = (PySetInstance*)o;
    bool contains = false;
    try {
        iterateWithEarlyExit(other, [&](PyObject* item) {
            contains = self_w->sq_contains_concrete(item);
            return !contains;
        });
    } catch (PythonExceptionSet& e) {
        return -1;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return -1;
    }

    return (int)(!contains);
}

PyObject* PySetInstance::set_symmetric_difference(PyObject* o, PyObject* other) {
    PySetInstance* new_inst = NULL;
    PySetInstance* self_w = (PySetInstance*)o;
    Type* src_type = extractTypeFrom(Py_TYPE(other));

    if (src_type && (src_type->getTypeCategory() == Type::TypeCategory::catSet)
            && ((SetType*)src_type)->keyType() == self_w->type()->keyType()) {
        new_inst = (PySetInstance*)PyInstance::tp_new(Py_TYPE(o), NULL, NULL);
        if (!new_inst) {
            return NULL;
        }
        new_inst->mIteratorOffset = -1;
        new_inst->mIteratorFlag = 0;

        try {
            PySetInstance* other_w = (PySetInstance*)other;
            getDataFromNative((PySetInstance*)o, [&](instance_ptr key) {
                instance_ptr existingLoc = other_w->type()->lookupKey(other_w->dataPtr(), key);
                if (!existingLoc) {
                    insertKey(new_inst, other, key);
                }
            });
            getDataFromNative((PySetInstance*)other, [&](instance_ptr key) {
                instance_ptr existingLoc = self_w->type()->lookupKey(self_w->dataPtr(), key);
                if (!existingLoc) {
                    insertKey(new_inst, o, key);
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
            PyErr_SetString(PyExc_TypeError, "symmetric_difference allocation error");
            return NULL;
        }
        if (set_symmetric_difference_update((PyObject*)new_inst, other) < 0) {
            decref((PyObject*)new_inst);
            PyErr_SetString(PyExc_TypeError, "symmetric_difference type mismatch");
            return NULL;
        }
    }

    return (PyObject*)new_inst;
}

int PySetInstance::set_symmetric_difference_update(PyObject* o, PyObject* other) {
    if (o == other) {
        setClear(o, NULL);
        return 0;
    }

    PySetInstance* self_w = (PySetInstance*)o;

    PySetInstance* to_be_added = (PySetInstance*)PyInstance::tp_new(Py_TYPE(o), NULL, NULL);
    if (!to_be_added) {
        return -1;
    }
    PySetInstance* to_be_removed = (PySetInstance*)PyInstance::tp_new(Py_TYPE(o), NULL, NULL);
    if (!to_be_removed) {
        decref((PyObject*)to_be_added);
        return -1;
    }

    try {
        iterate(other, [&](PyObject* item) {
            Instance key(self_w->type()->keyType(), [&](instance_ptr data) {
                copyConstructFromPythonInstance(self_w->type()->keyType(), data, item, ConversionLevel::UpcastContainers);
            });
            instance_ptr i = self_w->type()->lookupKey(self_w->dataPtr(), key.data());
            if (i)
                insertKey(to_be_removed, item, key.data());
            else
                insertKey(to_be_added, item, key.data());
        });

        set_difference_update(o, (PyObject*)to_be_removed);
        copy_elements((PyObject*)o, (PyObject*)to_be_added);
        decref((PyObject*)to_be_added);
        decref((PyObject*)to_be_removed);
    } catch (PythonExceptionSet& e) {
        decref((PyObject*)to_be_added);
        decref((PyObject*)to_be_removed);
        return -1;
    } catch (std::exception& e) {
        decref((PyObject*)to_be_added);
        decref((PyObject*)to_be_removed);
        return -1;
    }

    return 0;
}

PyObject* PySetInstance::set_union(PyObject* o, PyObject* other) {
    PySetInstance* new_inst = NULL;
    PySetInstance* self_w = (PySetInstance*)o;
    Type* src_type = extractTypeFrom(Py_TYPE(other));

    if (src_type && (src_type->getTypeCategory() == Type::TypeCategory::catSet)
            && ((SetType*)src_type)->keyType() == self_w->type()->keyType()) {
        new_inst = (PySetInstance*)PyInstance::tp_new(Py_TYPE(o), NULL, NULL);
        if (!new_inst) {
            return NULL;
        }
    }
    else {
        return NULL;
    }

    new_inst->mIteratorOffset = -1;
    new_inst->mIteratorFlag = 0;

    try {
        // copy lhs to new set
        copy_elements((PyObject*)new_inst, (PyObject*)self_w);
        copy_elements((PyObject*)new_inst, other);
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

PyDoc_STRVAR(setDifference_doc,
    "s.difference(s1) -> set that contain elements that are in s but not s1\n"
    "s.difference(s1, s2, ...) -> set of elements that are in s but not s1, s2, ...\n"
    "\n"
    "s1, s2, ... can be sets or any iterable yielding the same element type as s.\n"
    );
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

PyDoc_STRVAR(setSymmetricDifference_doc,
    "s.symmetric_difference(s1) -> set of elements in one of s or s1, but not both\n"
    "\n"
    "s1 can be a set or any iterable yielding the same element type as s.\n"
    );
PyObject* PySetInstance::setSymmetricDifference(PyObject* o, PyObject* args) {
    if (Py_ssize_t n = PyTuple_Size(args) != 1) {
        PyErr_Format(PyExc_TypeError, "symmetric_difference() takes exactly one argument (%d given)", n);
        return NULL;
    }

    PyObject* result;
    PyObjectHolder other(PyTuple_GetItem(args, 0));
    result = set_symmetric_difference(o, other);
    return result;
}

PyDoc_STRVAR(setIsSubset_doc,
    "s.issubset(s1) -> Is s a subset of s1?"
    );
PyObject* PySetInstance::setIsSubset(PyObject* o, PyObject* args) {
    if (Py_ssize_t n = PyTuple_Size(args) != 1) {
        PyErr_Format(PyExc_TypeError, "issubset() takes exactly one argument (%d given)", n);
        return NULL;
    }
    PyObjectHolder arg(PyTuple_GetItem(args, 0));

    if (!PyIter_Check(arg) && !PyList_Check(arg) && !PySet_Check(arg)
            && !PyTuple_Check(arg) && !PySequence_Check(arg)) {
        PyErr_Format(PyExc_TypeError, "'%s' object is not iterable", Py_TYPE(arg)->tp_name);
        return NULL;
    }

    int ret = set_is_subset(o, arg);
    if (ret < 0)
        return NULL;
    return incref(ret ? Py_True : Py_False);
}

PyDoc_STRVAR(setIsSuperset_doc,
    "s.issuperset(s1) -> Is s a superset of s1?"
    );
PyObject* PySetInstance::setIsSuperset(PyObject* o, PyObject* args) {
    if (Py_ssize_t n = PyTuple_Size(args) != 1) {
        PyErr_Format(PyExc_TypeError, "issuperset() takes exactly one argument (%d given)", n);
        return NULL;
    }
    PyObjectHolder arg(PyTuple_GetItem(args, 0));
    if (!PyIter_Check(arg) && !PyList_Check(arg) && !PySet_Check(arg)
            && !PyTuple_Check(arg) && !PySequence_Check(arg)) {
        PyErr_Format(PyExc_TypeError, "'%s' object is not iterable", Py_TYPE(arg)->tp_name);
        return NULL;
    }

    int ret = set_is_superset(o, arg);
    if (ret < 0)
        return NULL;
    return incref(ret ? Py_True : Py_False);
}

PyDoc_STRVAR(setIsDisjoint_doc,
    "s.isdisjoint(s1) -> Are s and s1 disjoint?"
    );
PyObject* PySetInstance::setIsDisjoint(PyObject* o, PyObject* args) {
    if (Py_ssize_t n = PyTuple_Size(args) != 1) {
        PyErr_Format(PyExc_TypeError, "isdisjoint() takes exactly one argument (%d given)", n);
        return NULL;
    }
    PyObjectHolder arg(PyTuple_GetItem(args, 0));
    if (!PyIter_Check(arg) && !PyList_Check(arg) && !PySet_Check(arg)
            && !PyTuple_Check(arg) && !PySequence_Check(arg)) {
        PyErr_Format(PyExc_TypeError, "'%s' object is not iterable", Py_TYPE(arg)->tp_name);
        return NULL;
    }

    int ret = set_is_disjoint(o, arg);
    if (ret < 0)
        return NULL;
    return incref(ret ? Py_True : Py_False);
}

PyDoc_STRVAR(setIntersection_doc,
    "s.intersection(s1) -> set of elements that are in both s and s1\n"
    "s.intersection(s1, s2, ...) -> set of elements that are in s, s1, s2, ...\n"
    "\n"
    "s1, s2, ... can be sets or any iterable yielding the same element type as s.\n"
    );
PyObject* PySetInstance::setIntersection(PyObject* o, PyObject* args) {
    if (PyTuple_Size(args) == 0) {
        return setCopy(o, NULL);
    }

    for (size_t k = 0; k < PyTuple_Size(args); ++k) {
        PyObjectHolder item(PyTuple_GetItem(args, k));
        if (!PyIter_Check(item) && !PyList_Check(item) && !PySet_Check(item)
                && !PyTuple_Check(item) && !PySequence_Check(item)) {
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

PyDoc_STRVAR(setUnion_doc,
    "s.union(s1) -> set of elements that are in s or s1\n"
    "s.union(s1, s2, ...) -> set of elements that are in one of s, s1, s2, ...\n"
    "\n"
    "s1, s2, ... can be sets or any iterable yielding the same element type as s.\n"
    );
PyObject* PySetInstance::setUnion(PyObject* o, PyObject* args) {
    if (PyTuple_Size(args) == 0) {
        return setCopy(o, NULL);
    }

    for (size_t k = 0; k < PyTuple_Size(args); ++k) {
        PyObjectHolder item(PyTuple_GetItem(args, k));
        if (!PyIter_Check(item) && !PyList_Check(item) && !PyAnySet_Check(item)
                && !PyTuple_Check(item) && !PySequence_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "Set.union one of args has wrong type");
            return NULL;
        }
    }

    PySetInstance* self_w = (PySetInstance*)o;
    PySetInstance* new_inst = (PySetInstance*)PyInstance::tp_new(Py_TYPE(o), NULL, NULL);
    if (!new_inst) {
        return NULL;
    }
    new_inst->mIteratorOffset = -1;
    new_inst->mIteratorFlag = 0;

    try {
        // copy lhs to new set
        copy_elements((PyObject*)new_inst, (PyObject*)self_w);

        // copy rhs arg items to new set
        for (size_t k = 0; k < PyTuple_Size(args); ++k) {
            PyObjectHolder arg_item(PyTuple_GetItem(args, k));
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

void PySetInstance::constructFromPythonArgumentsConcrete(SetType* t, uint8_t* data, PyObject* args, PyObject* kwargs) {
    if ((!kwargs || PyDict_Size(kwargs) == 0) && (args && PyTuple_Size(args) == 1)) {
        PyObject* arg = PyTuple_GetItem(args, 0);
        Type* argType = extractTypeFrom(arg->ob_type);

        if (Type::typesEquivalent(argType, t)) {
            //following python semantics, this needs to produce a new object
            //that's a copy of the original dict. We can't just incref it and return
            //the original object because it has state and we don't want the objects
            //to alias each other
            t->constructor(data);
            t->visitSetElements(
                ((PyInstance*)arg)->dataPtr(),
                [&](instance_ptr key) {
                    t->insertKey(data, key);
                    return true;
                }
            );
            return;
        }
    }

    PyInstance::constructFromPythonArgumentsConcrete(t, data, args, kwargs);
}

PyDoc_STRVAR(setIntersectionUpdate_doc,
    "s.intersection_update(s1) -> None, and s=s.intersection(s1)\n"
    "s.intersection_update(s1, s2, ...) -> None, and s=s.intersection(s1, s2, ...)\n"
    "\n"
    "s1, s2, ... can be sets or any iterable yielding the same element type as s.\n"
    );
PyObject* PySetInstance::setIntersectionUpdate(PyObject* o, PyObject* args) {
    if (PyTuple_Size(args) == 0) {
        Py_RETURN_NONE;
    }

    for (size_t k = 0; k < PyTuple_Size(args); ++k) {
        PyObjectHolder item(PyTuple_GetItem(args, k));
        if (!PyIter_Check(item) && !PyList_Check(item) && !PySet_Check(item)
                && !PyTuple_Check(item) && !PySequence_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "Set.intersection_update one of args has wrong type");
            return NULL;
        }
    }

    try {
        for (size_t k = 0; k < PyTuple_Size(args); ++k) {
            PyObjectHolder item(PyTuple_GetItem(args, k));

            if (set_intersection_update(o, item) < 0) {
                return NULL;
            }
        }
    } catch (PythonExceptionSet& e) {
        return NULL;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(setDifferenceUpdate_doc,
    "s.difference_update(s1) -> None, and s=s.difference(s1)\n"
    "s.difference_update(s1, s2, ...) -> None, and s=s.difference(s1, s2, ...)\n"
    "\n"
    "s1, s2, ... can be sets or any iterable yielding the same element type as s.\n"
    );
PyObject* PySetInstance::setDifferenceUpdate(PyObject* o, PyObject* args) {
    if (PyTuple_Size(args) == 0) {
        Py_RETURN_NONE;
    }

    for (size_t k = 0; k < PyTuple_Size(args); ++k) {
        PyObjectHolder item(PyTuple_GetItem(args, k));
        if (!PyIter_Check(item) && !PyList_Check(item) && !PySet_Check(item)
                && !PyTuple_Check(item) && !PySequence_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "difference_update one of args has wrong type");
            return NULL;
        }
    }

    try {
        for (size_t k = 0; k < PyTuple_Size(args); ++k) {
            PyObjectHolder item(PyTuple_GetItem(args, k));
            if (set_difference_update(o, item) < 0) {
                PyErr_SetString(PyExc_TypeError, "difference_update element type conversion error");
                return NULL;
            }
        }
    } catch (PythonExceptionSet& e) {
        return NULL;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(setSymmetricDifferenceUpdate_doc,
    "s.symmetric_difference_update(s1) -> None, and s=s.symmetric_difference(s1)\n"
    "\n"
    "s1 can be a set or any iterable yielding the same element type as s.\n"
    );
PyObject* PySetInstance::setSymmetricDifferenceUpdate(PyObject* o, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "symmetric_difference_update takes exactly one argument");
        return NULL;
    }

    PyObjectHolder item(PyTuple_GetItem(args, 0));
    if (!PyIter_Check(item) && !PyList_Check(item) && !PySet_Check(item)
            && !PyTuple_Check(item) && !PySequence_Check(item)) {
        PyErr_SetString(PyExc_TypeError, "symmetric_difference_update arg has wrong type");
        return NULL;
    }

    try {
        PyObjectHolder item(PyTuple_GetItem(args, 0));
        if (set_symmetric_difference_update(o, item) < 0) {
            PyErr_SetString(PyExc_TypeError, "symmetric_difference_update element type conversion error");
            return NULL;
        }
    } catch (PythonExceptionSet& e) {
        return NULL;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(setUpdate_doc,
    "s.update(s1) -> None, and s=s.union(s1)\n"
    "s.update(s1, s2, ...) -> None, and s=s.union(s1, s2, ...)\n"
    "\n"
    "s1, s2, ... can be sets or any iterable yielding the same element type as s.\n"
    );
PyObject* PySetInstance::setUpdate(PyObject* o, PyObject* args) {
    if (PyTuple_Size(args) == 0) {
        Py_RETURN_NONE;
    }

    for (size_t k = 0; k < PyTuple_Size(args); ++k) {
        PyObjectHolder item(PyTuple_GetItem(args, k));
        if (!PyIter_Check(item) && !PyList_Check(item) && !PyAnySet_Check(item)
                && !PyTuple_Check(item) && !PySequence_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "Set.update one of args has wrong type");
            return NULL;
        }
    }

    PySetInstance* self_w = (PySetInstance*)o;
    self_w->mIteratorOffset = -1;
    self_w->mIteratorFlag = 0;

    try {
        // copy rhs arg items to new set
        for (size_t k = 0; k < PyTuple_Size(args); ++k) {
            PyObjectHolder arg_item(PyTuple_GetItem(args, k));
            copy_elements((PyObject*)self_w, (PyObject*)arg_item);
        }
    } catch (PythonExceptionSet& e) {
        return NULL;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    }

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
            copyConstructFromPythonInstance(type()->keyType(), data, item, ConversionLevel::UpcastContainers);
        });
        instance_ptr i = type()->lookupKey(dataPtr(), key.data());
        return i ? 1 : 0;
    }
}

PyDoc_STRVAR(setPop_doc,
    "s.pop() -> an arbitrary element of the set, which is removed"
    );
PyObject* PySetInstance::setPop(PyObject* o, PyObject* args) {
    PySetInstance* self_w = (PySetInstance*)o;
    if (PyTuple_Size(args) != 0) {
        PyErr_SetString(PyExc_TypeError, "Set.pop takes no arguments");
        return NULL;
    }

    SetType* self_type = (SetType*)extractTypeFrom(o->ob_type);
    instance_ptr self_ptr = self_w->dataPtr();

    int iteratorOffset = 0;

    while (iteratorOffset < self_type->slotCount(self_ptr) &&
                !self_type->slotPopulated(self_ptr, iteratorOffset)) {
        iteratorOffset++;
    }

    if (iteratorOffset == self_type->slotCount(self_ptr)) {
        PyErr_SetString(PyExc_KeyError, "pop from an empty set");
        return NULL;
    }

    instance_ptr keyPtr = self_type->keyAtSlot(self_ptr, iteratorOffset);

    PyObject* result = extractPythonObject(keyPtr, self_type->keyType());

    self_type->discard(self_ptr, keyPtr);

    return result;
}

PyDoc_STRVAR(setAdd_doc,
    "s.add(item) -> None, and item is added to the set s"
    );
PyObject* PySetInstance::setAdd(PyObject* o, PyObject* args) {
    PySetInstance* self_w = (PySetInstance*)o;
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "Set.add takes one argument");
        return NULL;
    }

    PyObjectHolder item(PyTuple_GetItem(args, 0));
    Type* item_type = extractTypeFrom(item->ob_type);
    Type* self_type = extractTypeFrom(o->ob_type);

    if (self_type->getTypeCategory() != Type::TypeCategory::catSet) {
        PyErr_SetString(PyExc_TypeError, "Wrong type!");
        return NULL;
    }

    if (item_type == self_w->type()->keyType()) {
        PyInstance* item_w = (PyInstance*)(PyObject*)item;
        instance_ptr key = item_w->dataPtr();
        insertKey(self_w, item, key);
    } else {
        try {
            Instance key(self_w->type()->keyType(), [&](instance_ptr data) {
                copyConstructFromPythonInstance(self_w->type()->keyType(), data, item, ConversionLevel::UpcastContainers);
            });
            insertKey(self_w, item, key.data());

        } catch (PythonExceptionSet& e) {
            return NULL;
        } catch (std::exception& e) {
            PyErr_SetString(PyExc_TypeError, e.what());
            return NULL;
        }
    }

    Py_RETURN_NONE;
}

Py_ssize_t PySetInstance::mp_and_sq_length_concrete() {
    return type()->size(dataPtr());
}

PyObject* PySetInstance::tp_iter_concrete() {
    return createIteratorToSelf(mIteratorFlag, type()->size(dataPtr()));
}

PyObject* PySetInstance::tp_iternext_concrete() {
    if (mIteratorOffset == 0) {
        // search forward to find the first slot
        while (mIteratorOffset < type()->slotCount(dataPtr())
               && !type()->slotPopulated(dataPtr(), mIteratorOffset)) {
            mIteratorOffset++;
        }
    }

    if (type()->size(dataPtr()) != mContainerSize) {
        PyErr_Format(PyExc_RuntimeError, "set size changed during iteration");
        return NULL;
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

void PySetInstance::insertKey(PySetInstance* self, PyObject* pyKey, instance_ptr key) {
    instance_ptr existingLoc = self->type()->lookupKey(self->dataPtr(), key);
    if (!existingLoc) {
        self->type()->insertKey(self->dataPtr(), key);
    }
}

void PySetInstance::mirrorTypeInformationIntoPyTypeConcrete(SetType* setType,
                                                            PyTypeObject* pyType) {
    PyDict_SetItemString(pyType->tp_dict, "ElementType", // was KeyType
                         typePtrToPyTypeRepresentation(setType->keyType()));
}

SetType* PySetInstance::type() {
    return (SetType*)extractTypeFrom(((PyObject*)this)->ob_type);
}

void PySetInstance::copyConstructFromPythonInstanceConcrete(
    SetType* setType, instance_ptr tgt,
    PyObject* pyRepresentation,
    ConversionLevel level
) {
    if (level < ConversionLevel::ImplicitContainers) {
        return PyInstance::copyConstructFromPythonInstanceConcrete(setType, tgt, pyRepresentation, level);
    }

    ConversionLevel childLevel;

    if (level == ConversionLevel::ImplicitContainers) {
        childLevel = ConversionLevel::UpcastContainers;
    } else {
        childLevel = ConversionLevel::Implicit;
    }

    bool setIsInitialized = false;

    std::pair<Type*, instance_ptr> typeAndPtr = extractTypeAndPtrFrom(pyRepresentation);
    Type* argType = typeAndPtr.first;
    instance_ptr argDataPtr = typeAndPtr.second;

    try {
        if (argType && argType->getTypeCategory() == Type::TypeCategory::catTupleOf) {
            TupleOfType* tupType = (TupleOfType*)argType;

            if (tupType->getEltType() == setType->keyType()) {
                setType->constructor(tgt);
                setIsInitialized = true;

                for (long k = 0; k < tupType->count(argDataPtr); k++) {
                    instance_ptr data = tupType->eltPtr(argDataPtr, k);
                    instance_ptr found = setType->lookupKey(tgt, data);
                    if (!found) {
                        setType->insertKey(tgt, data);
                    }
                }

                return;
            }
        }

        if (argType && argType->getTypeCategory() == Type::TypeCategory::catListOf) {
            ListOfType* listType = (ListOfType*)argType;

            if (listType->getEltType() == setType->keyType()) {
                setType->constructor(tgt);
                setIsInitialized = true;

                for (long k = 0; k < listType->count(argDataPtr); k++) {
                    instance_ptr data = listType->eltPtr(argDataPtr, k);
                    instance_ptr found = setType->lookupKey(tgt, data);
                    if (!found) {
                         setType->insertKey(tgt, data);
                    }
                }

                return;
            }
        }

        setType->constructor(tgt);
        setIsInitialized = true;

        iterate(pyRepresentation, [&](PyObject* item) {
            Instance key(setType->keyType(), [&](instance_ptr data) {
                copyConstructFromPythonInstance(setType->keyType(), data, item, childLevel);
                return true;
            });

            instance_ptr found = setType->lookupKey(tgt, key.data());
            if (!found) {
                setType->insertKey(tgt, key.data());
            }
        });

        return;
    } catch (...) {
        if (setIsInitialized) {
            setType->destroy(tgt);
        }

        throw;
    }
}

int PySetInstance::pyInquiryConcrete(const char* op, const char* opErrRep) {
    // op == '__bool__'
    return type()->size(dataPtr()) != 0;
}

PyObject* PySetInstance::pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr) {
    if (strcmp(op, "__sub__") == 0) {
        return pyOperatorDifference(rhs, op, opErr, false);
    }
    if (strcmp(op, "__xor__") == 0) {
        return pyOperatorSymmetricDifference(rhs, op, opErr, false);
    }
    if (strcmp(op, "__or__") == 0) {
        return pyOperatorUnion(rhs, op, opErr, false);
    }
    if (strcmp(op, "__and__") == 0) {
        return pyOperatorIntersection(rhs, op, opErr, false);
    }

    return PyInstance::pyOperatorConcrete(rhs, op, opErr);
}

PyObject* PySetInstance::pyOperatorDifference(PyObject* rhs, const char* op, const char* opErr, bool reversed) {
    Type* rhs_type = extractTypeFrom(rhs->ob_type);

    PySetInstance* w_lhs = (PySetInstance*)this;
    PySetInstance* w_rhs = (PySetInstance*)rhs;

    //Set(X) - Set(X) fastpath
    if (type() == rhs_type) {
        PyObject* result = set_difference((PyObject*)w_lhs, (PyObject*)w_rhs);
        if (!result) {
            return NULL;
        }
        return result;
    }

    return PyInstance::pyOperatorConcrete(rhs, op, opErr);
}

PyObject* PySetInstance::pyOperatorUnion(PyObject* rhs, const char* op, const char* opErr, bool reversed) {
    Type* rhs_type = extractTypeFrom(rhs->ob_type);

    PySetInstance* w_lhs = (PySetInstance*)this;
    PySetInstance* w_rhs = (PySetInstance*)rhs;

    //Set(X) | Set(X) fastpath
    if (type() == rhs_type) {
        PyObject* result = set_union((PyObject*)w_lhs, (PyObject*)w_rhs);
        if (!result) {
            return NULL;
        }
        return result;
    }

    return PyInstance::pyOperatorConcrete(rhs, op, opErr);
}

PyObject* PySetInstance::pyOperatorIntersection(PyObject* rhs, const char* op, const char* opErr, bool reversed) {
    Type* rhs_type = extractTypeFrom(rhs->ob_type);

    PySetInstance* w_lhs = (PySetInstance*)this;
    PySetInstance* w_rhs = (PySetInstance*)rhs;

    //Set(X) & Set(X) fastpath
    if (type() == rhs_type) {
        PyObject* result = set_intersection((PyObject*)w_lhs, (PyObject*)w_rhs);
        if (!result) {
            return NULL;
        }
        return result;
    }

    return PyInstance::pyOperatorConcrete(rhs, op, opErr);
}

PyObject* PySetInstance::pyOperatorSymmetricDifference(PyObject* rhs, const char* op, const char* opErr, bool reversed) {
    Type* rhs_type = extractTypeFrom(rhs->ob_type);

    PySetInstance* w_lhs = (PySetInstance*)this;
    PySetInstance* w_rhs = (PySetInstance*)rhs;

    //Set(X) ^ Set(X) fastpath
    if (type() == rhs_type) {
        PyObject* result = set_symmetric_difference((PyObject*)w_lhs, (PyObject*)w_rhs);
        if (!result) {
            return NULL;
        }
        return result;
    }

    return PyInstance::pyOperatorConcrete(rhs, op, opErr);
}

bool PySetInstance::subset(SetType *setT, instance_ptr left, PyObject* right) {
    int found = 1;

    setT->visitSetElements(left, [&](instance_ptr key) {
        PyObject* keyObject = extractPythonObject(key, setT->keyType());
        found = PySequence_Contains(right, keyObject);
        return (found <= 0);
    });
    return found == 1;
}

bool PySetInstance::superset(SetType *setT, instance_ptr left, PyObject* right) {
    int found = 1;
    long found_count = 0;
    long right_count = PyObject_Length(right);

    if (right_count == 0) {
        return true;
    }
    setT->visitSetElements(left, [&](instance_ptr key) {
        PyObject* keyObject = extractPythonObject(key, setT->keyType());
        found = PySequence_Contains(right, keyObject);
        if (found == 1) {
            found_count++;
            if (found_count >= right_count) {
                return false;
            }
        }
        return true;
    });
    return found_count >= right_count;
}

bool PySetInstance::compare_to_python_concrete(SetType* setT, instance_ptr left, PyObject* right, bool exact, int pyComparisonOp) {
    Type* rightType = extractTypeFrom(right->ob_type);

    if (!PySet_Check(right) && (!rightType || rightType->getTypeCategory() != Type::TypeCategory::catSet)) {
        if (pyComparisonOp == Py_EQ || pyComparisonOp == Py_NE) {
            return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
        }
        PyErr_Format(
            PyExc_TypeError,
            "Comparison not supported between instances of '%s' and '%s'.",
            setT->name().c_str(),
            right->ob_type->tp_name
        );
        throw PythonExceptionSet();
    }

    long left_count = setT->size(left);
    long right_count = PyObject_Length(right);

    if (pyComparisonOp == Py_EQ || pyComparisonOp == Py_NE) {
        if (left_count != right_count) {
            return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
        }
        return cmpResultToBoolForPyOrdering(pyComparisonOp, subset(setT, left, right) ? 0 : 1);
    }
    else if (pyComparisonOp == Py_LE) {
        if (left_count > right_count) {
            return false;
        }
        return subset(setT, left, right);
    }
    else if (pyComparisonOp == Py_LT) {
        if (left_count >= right_count) {
            return false;
        }
        return subset(setT, left, right);
    }
    else if (pyComparisonOp == Py_GE) {
        if (left_count < right_count) {
            return false;
        }
        return superset(setT, left, right);
    }
    else if (pyComparisonOp == Py_GT) {
        if (left_count <= right_count) {
            return false;
        }
        return superset(setT, left, right);
    }

    assert(false);
    return false;
}

bool PySetInstance::pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, ConversionLevel level) {
    if (level >= ConversionLevel::ImplicitContainers) {
        return true;
    }

    return false;
}

PyMethodDef* PySetInstance::typeMethodsConcrete(Type* t) {
    return new PyMethodDef[18]{{"add", (PyCFunction)PySetInstance::setAdd, METH_VARARGS, setAdd_doc},
                              {"pop", (PyCFunction)PySetInstance::setPop, METH_VARARGS, setPop_doc},
                              {"discard", (PyCFunction)PySetInstance::setDiscard, METH_VARARGS, setDiscard_doc},
                              {"remove", (PyCFunction)PySetInstance::setRemove, METH_VARARGS, setRemove_doc},
                              {"clear", (PyCFunction)PySetInstance::setClear, METH_VARARGS, setClear_doc},
                              {"copy", (PyCFunction)PySetInstance::setCopy, METH_VARARGS, setCopy_doc},
                              {"union", (PyCFunction)PySetInstance::setUnion, METH_VARARGS, setUnion_doc},
                              {"update", (PyCFunction)PySetInstance::setUpdate, METH_VARARGS, setUpdate_doc},
                              {"intersection", (PyCFunction)PySetInstance::setIntersection, METH_VARARGS, setIntersection_doc},
                              {"intersection_update", (PyCFunction)PySetInstance::setIntersectionUpdate, METH_VARARGS, setIntersectionUpdate_doc},
                              {"difference", (PyCFunction)PySetInstance::setDifference, METH_VARARGS, setDifference_doc},
                              {"difference_update", (PyCFunction)PySetInstance::setDifferenceUpdate, METH_VARARGS, setDifferenceUpdate_doc},
                              {"symmetric_difference", (PyCFunction)PySetInstance::setSymmetricDifference, METH_VARARGS, setSymmetricDifference_doc},
                              {"symmetric_difference_update", (PyCFunction)PySetInstance::setSymmetricDifferenceUpdate, METH_VARARGS, setSymmetricDifferenceUpdate_doc},
                              {"issubset", (PyCFunction)PySetInstance::setIsSubset, METH_VARARGS, setIsSubset_doc},
                              {"issuperset", (PyCFunction)PySetInstance::setIsSuperset, METH_VARARGS, setIsSuperset_doc},
                              {"isdisjoint", (PyCFunction)PySetInstance::setIsDisjoint, METH_VARARGS, setIsDisjoint_doc},
                              {NULL, NULL}};
}
