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

#include "PyTupleOrListOfInstance.hpp"

TupleOrListOfType* PyTupleOrListOfInstance::type() {
    return (TupleOrListOfType*)extractTypeFrom(((PyObject*)this)->ob_type);
}

TupleOfType* PyTupleOfInstance::type() {
    return (TupleOfType*)extractTypeFrom(((PyObject*)this)->ob_type);
}

ListOfType* PyListOfInstance::type() {
    return (ListOfType*)extractTypeFrom(((PyObject*)this)->ob_type);
}

bool PyTupleOrListOfInstance::pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, ConversionLevel level) {
    if (type->isListOf()) {
        return level >= ConversionLevel::ImplicitContainers;
    }

    std::pair<Type*, instance_ptr> typeAndPtrOfArg = extractTypeAndPtrFrom(pyRepresentation);

    if (level == ConversionLevel::Signature) {
        return Type::typesEquivalent(typeAndPtrOfArg.first, type);
    }

    if (level == ConversionLevel::Upcast) {
        return false;
    }

    if (level >= ConversionLevel::ImplicitContainers) {
        return true;
    }

    // only allow implicit conversion from tuple/list of. We don't want dicts and sets to implicitly
    // convert to tuples
    if (PyTuple_Check(pyRepresentation) || PyList_Check(pyRepresentation)) {
        return true;
    }

    if (typeAndPtrOfArg.first && (
            typeAndPtrOfArg.first->isListOf()
            || typeAndPtrOfArg.first->isTupleOf()
            || typeAndPtrOfArg.first->isComposite()
            )
    ) {
        return true;
    }

    return false;
}

PyObject* PyTupleOrListOfInstance::pyOperatorConcreteReverse(PyObject* lhs, const char* op, const char* opErrRep) {
    if (strcmp(op, "__add__") == 0) {
        return pyOperatorAdd(lhs, op, opErrRep, true);
    }

    return PyInstance::pyOperatorConcreteReverse(lhs, op, opErrRep);
}

PyObject* PyTupleOrListOfInstance::pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr) {
    if (strcmp(op, "__add__") == 0) {
        return pyOperatorAdd(rhs, op, opErr, false);
    }

    return PyInstance::pyOperatorConcrete(rhs, op, opErr);
}

PyObject* PyTupleOrListOfInstance::pyOperatorAdd(PyObject* rhs, const char* op, const char* opErr, bool reversed) {
    Type* rhs_type = extractTypeFrom(rhs->ob_type);

    //TupleOrListOf(X) + TupleOrListOf(X) fastpath
    if (type() == rhs_type) {
        PyTupleOrListOfInstance* w_lhs;
        PyTupleOrListOfInstance* w_rhs;

        if (reversed) {
            w_lhs = (PyTupleOrListOfInstance*)rhs;
            w_rhs = (PyTupleOrListOfInstance*)this;
        } else {
            w_lhs = (PyTupleOrListOfInstance*)this;
            w_rhs = (PyTupleOrListOfInstance*)rhs;
        }

        Type* eltType = type()->getEltType();

        return PyInstance::initialize(type(), [&](instance_ptr data) {
            int count_lhs = type()->count(w_lhs->dataPtr());
            int count_rhs = type()->count(w_rhs->dataPtr());

            type()->constructor(data, count_lhs + count_rhs,
                [&](uint8_t* eltPtr, int64_t k) {
                    eltType->copy_constructor(
                        eltPtr,
                        k < count_lhs ?
                            type()->eltPtr(w_lhs->dataPtr(), k) :
                            type()->eltPtr(w_rhs->dataPtr(), k - count_lhs)
                        );
                    }
                );
            });
    }

    //generic path to add any kind of iterable.
    if (PyObject_Length(rhs) == -1) {
        // if not iterable, we want to fall through to Py_NotImplemented, which will try reverse operator,
        // instead of generating an actual error
        PyErr_Clear();
    } else {
        Type* eltType = type()->getEltType();

        return PyInstance::initialize(type(), [&](instance_ptr data) {
            int count_lhs = type()->count(dataPtr());
            int count_rhs = PyObject_Length(rhs);

            type()->constructor(data, count_lhs + count_rhs,
                [&](uint8_t* eltPtr, int64_t k) {
                    if ((!reversed && k < count_lhs) || (reversed && k >= count_rhs)) {
                        eltType->copy_constructor(
                            eltPtr,
                            type()->eltPtr(dataPtr(), reversed ? k - count_rhs : k)
                            );
                    } else {
                        PyObjectStealer kval(PyLong_FromLong(reversed ? k : k - count_lhs));
                        PyObjectStealer o(PyObject_GetItem(rhs, kval));

                        if (!o) {
                            throw PythonExceptionSet();
                        }

                        PyInstance::copyConstructFromPythonInstance(eltType, eltPtr, o, ConversionLevel::ImplicitContainers);
                    }
                });
        });
    }

    return PyInstance::pyOperatorConcrete(rhs, op, opErr);
}


//static
PyDoc_STRVAR(listSetSizeUnsafe_doc,
    "lst.setSizeUnsafe(n) -> None, and set length of lst to n\n"
    "\n"
    "No adjustments to elements or reallocations are done.\n"
    "The size field is simply set to n."
    "This is intended for compiled code.\n"
    );
PyObject* PyListOfInstance::listSetSizeUnsafe(PyObject* o, PyObject* args) {
    PyListOfInstance* self_w = (PyListOfInstance*)o;

    if (PyTuple_Size(args) != 1 || !PyLong_Check(PyTuple_GetItem(args, 0))) {
        PyErr_SetString(PyExc_TypeError, "ListOf.getUnsafe takes one integer argument");
        return NULL;
    }

    int64_t ix = PyLong_AsLongLong(PyTuple_GetItem(args,0));

    if (ix < 0) {
        PyErr_SetString(undefinedBehaviorException(), "setSizeUnsafe passed negative index");
        return NULL;
    }

    self_w->type()->setSizeUnsafe(self_w->dataPtr(), ix);

    return incref(Py_None);
}


template<class dest_t, class source_t>
void constructTupleOrListInst(TupleOrListOfType* tupT, instance_ptr tgt, size_t count, long* strides, uint8_t* source_data) {
    tupT->constructor(tgt, count,
        [&](uint8_t* eltPtr, int64_t k) {
            ((dest_t*)eltPtr)[0] = ((source_t*)(source_data + k * strides[0]))[0];
            }
        );
}

template<class dest_t>
bool constructTupleOrListInstFromNumpy(TupleOrListOfType* tupT, instance_ptr tgt, size_t size, long* strides, uint8_t* data, int numpyType) {
    if (numpyType == NPY_FLOAT64) {
        constructTupleOrListInst<dest_t, double>(tupT, tgt, size, strides, data);
    } else if (numpyType == NPY_FLOAT32) {
        constructTupleOrListInst<dest_t, float>(tupT, tgt, size, strides, data);
    } else if (numpyType == NPY_INT64) {
        constructTupleOrListInst<dest_t, int64_t>(tupT, tgt, size, strides, data);
    } else if (numpyType == NPY_INT32) {
        constructTupleOrListInst<dest_t, int32_t>(tupT, tgt, size, strides, data);
    } else if (numpyType == NPY_INT16) {
        constructTupleOrListInst<dest_t, int16_t>(tupT, tgt, size, strides, data);
    } else if (numpyType == NPY_INT8) {
        constructTupleOrListInst<dest_t, int8_t>(tupT, tgt, size, strides, data);
    } else if (numpyType == NPY_UINT64) {
        constructTupleOrListInst<dest_t, uint64_t>(tupT, tgt, size, strides, data);
    } else if (numpyType == NPY_UINT32) {
        constructTupleOrListInst<dest_t, uint32_t>(tupT, tgt, size, strides, data);
    } else if (numpyType == NPY_UINT16) {
        constructTupleOrListInst<dest_t, uint16_t>(tupT, tgt, size, strides, data);
    } else if (numpyType == NPY_UINT8) {
        constructTupleOrListInst<dest_t, uint8_t>(tupT, tgt, size, strides, data);
    } else if (numpyType == NPY_BOOL) {
        constructTupleOrListInst<dest_t, dest_t>(tupT, tgt, size, strides, data);
    } else {
        return false;
    }

    return true;
}
void PyTupleOrListOfInstance::copyConstructFromPythonInstanceConcrete(TupleOrListOfType* tupT, instance_ptr tgt, PyObject* pyRepresentation, ConversionLevel level) {
    if (PyArray_Check(pyRepresentation) && level >= ConversionLevel::ImplicitContainers) {
        if (!PyArray_ISBEHAVED_RO(pyRepresentation)) {
            throw std::logic_error("Can't convert a numpy array that's not contiguous and in machine-native byte order.");
        }

        if (PyArray_NDIM(pyRepresentation) != 1) {
            throw std::logic_error("Can't convert a numpy array with more than 1 dimension. please flatten it.");
        }

        uint8_t* data = (uint8_t*)PyArray_BYTES(pyRepresentation);
        size_t size = PyArray_SIZE(pyRepresentation);
	long* strides = PyArray_STRIDES(pyRepresentation);

        if (tupT->getEltType()->getTypeCategory() == Type::TypeCategory::catBool) {
            if (constructTupleOrListInstFromNumpy<bool>(tupT, tgt, size, strides, data, PyArray_TYPE(pyRepresentation))) {
                return;
            }
        }
        if (tupT->getEltType()->getTypeCategory() == Type::TypeCategory::catInt64) {
            if (constructTupleOrListInstFromNumpy<int64_t>(tupT, tgt, size, strides, data, PyArray_TYPE(pyRepresentation))) {
                return;
            }
        }
        if (tupT->getEltType()->getTypeCategory() == Type::TypeCategory::catUInt64) {
            if (constructTupleOrListInstFromNumpy<uint64_t>(tupT, tgt, size, strides, data, PyArray_TYPE(pyRepresentation))) {
                return;
            }
        }
        if (tupT->getEltType()->getTypeCategory() == Type::TypeCategory::catInt32) {
            if (constructTupleOrListInstFromNumpy<int32_t>(tupT, tgt, size, strides, data, PyArray_TYPE(pyRepresentation))) {
                return;
            }
        }
        if (tupT->getEltType()->getTypeCategory() == Type::TypeCategory::catUInt32) {
            if (constructTupleOrListInstFromNumpy<uint32_t>(tupT, tgt, size, strides, data, PyArray_TYPE(pyRepresentation))) {
                return;
            }
        }
        if (tupT->getEltType()->getTypeCategory() == Type::TypeCategory::catInt16) {
            if (constructTupleOrListInstFromNumpy<int16_t>(tupT, tgt, size, strides, data, PyArray_TYPE(pyRepresentation))) {
                return;
            }
        }
        if (tupT->getEltType()->getTypeCategory() == Type::TypeCategory::catUInt16) {
            if (constructTupleOrListInstFromNumpy<uint16_t>(tupT, tgt, size, strides, data, PyArray_TYPE(pyRepresentation))) {
                return;
            }
        }
        if (tupT->getEltType()->getTypeCategory() == Type::TypeCategory::catInt8) {
            if (constructTupleOrListInstFromNumpy<int8_t>(tupT, tgt, size, strides, data, PyArray_TYPE(pyRepresentation))) {
                return;
            }
        }
        if (tupT->getEltType()->getTypeCategory() == Type::TypeCategory::catUInt8) {
            if (constructTupleOrListInstFromNumpy<uint8_t>(tupT, tgt, size, strides, data, PyArray_TYPE(pyRepresentation))) {
                return;
            }
        }
        if (tupT->getEltType()->getTypeCategory() == Type::TypeCategory::catFloat64) {
            if (constructTupleOrListInstFromNumpy<double>(tupT, tgt, size, strides, data, PyArray_TYPE(pyRepresentation))) {
                return;
            }
        }
        if (tupT->getEltType()->getTypeCategory() == Type::TypeCategory::catFloat32) {
            if (constructTupleOrListInstFromNumpy<float>(tupT, tgt, size, strides, data, PyArray_TYPE(pyRepresentation))) {
                return;
            }
        }
    }

    ConversionLevel childLevel = ConversionLevel::ImplicitContainers;

    // determine the level of conversion we'll use when converting the children if we're not a New call
    if (level < ConversionLevel::ImplicitContainers) {
        // ListOf only operates if we're above ImplicitContainers
        if (tupT->isListOf()) {
            throw std::logic_error("Can't implicitly convert from " +
                std::string(pyRepresentation->ob_type->tp_name) + " to " + tupT->name()
            );
        }

        if (level == ConversionLevel::Upcast) {
            throw std::logic_error("Can't upcast from " +
                std::string(pyRepresentation->ob_type->tp_name) + " to " + tupT->name()
            );
        } else if (level == ConversionLevel::UpcastContainers) {
            childLevel = level;
        } else if (level == ConversionLevel::Signature) {
            PyInstance::copyConstructFromPythonInstanceConcrete(tupT, tgt, pyRepresentation, level);
            return;
        }
    }

    if (level < ConversionLevel::ImplicitContainers) {
        // only allow implicit conversion from tuple/list of. We don't want dicts and sets to implicitly
        // convert to tuples
        bool isValid = false;
        if (PyTuple_Check(pyRepresentation) || PyList_Check(pyRepresentation)) {
            isValid = true;
        }

        std::pair<Type*, instance_ptr> typeAndPtrOfArg = extractTypeAndPtrFrom(pyRepresentation);
        if (typeAndPtrOfArg.first && (
                typeAndPtrOfArg.first->isListOf()
                || typeAndPtrOfArg.first->isTupleOf()
                || typeAndPtrOfArg.first->isComposite())
        ) {
            isValid = true;
        }

        if (!isValid) {
            PyInstance::copyConstructFromPythonInstanceConcrete(tupT, tgt, pyRepresentation, level);
            return;
        }
    }

    if (PyBytes_Check(pyRepresentation) && tupT->getEltType()->getTypeCategory() == Type::TypeCategory::catUInt8) {
        tupT->constructor(tgt);

        size_t sz = PyBytes_GET_SIZE(pyRepresentation);

        tupT->reserve(tgt, sz);

        memcpy(tupT->eltPtr(tgt, 0), PyBytes_AsString(pyRepresentation), sz);
        tupT->setSizeUnsafe(tgt, sz);

        return;
    }

    PyObjectStealer iterator(PyObject_GetIter(pyRepresentation));

    if (iterator) {
        tupT->constructorUnbounded(tgt,
            [&](uint8_t* eltPtr, int64_t k) {
                PyObjectStealer item(PyIter_Next(iterator));

                if (!item) {
                    if (PyErr_Occurred()) {
                        throw PythonExceptionSet();
                    }

                    return false;
                }

                PyInstance::copyConstructFromPythonInstance(tupT->getEltType(), eltPtr, item, childLevel);

                return true;
            });

        return;
    } else {
        throw PythonExceptionSet();
    }
}

PyObject* PyTupleOrListOfInstance::toBytes(PyObject* o, PyObject* args) {
    PyListOfInstance* self_w = (PyListOfInstance*)o;

    if (!self_w->type()->getEltType()->isPOD()) {
        PyErr_Format(
            PyExc_TypeError,
            "Can't convert %s to bytes because internals are not POD",
            self_w->type()->name().c_str()
        );

        return NULL;
    }

    return PyBytes_FromStringAndSize(
        (const char*)self_w->type()->eltPtr(self_w->dataPtr(), 0),
        self_w->type()->count(self_w->dataPtr()) *
        self_w->type()->getEltType()->bytecount()
    );
}

PyObject* PyTupleOrListOfInstance::fromBytes(PyObject* o, PyObject* args, PyObject* kwds) {
    static const char *kwlist[] = {"bytesObj", NULL};

    PyObject* bytesObj;

    if (!PyArg_ParseTupleAndKeywords(args, NULL, "O", (char**)kwlist, &bytesObj)) {
        return nullptr;
    }

    if (!PyBytes_Check(bytesObj)) {
        PyErr_Format(
            PyExc_TypeError,
            "Expected 'bytesObj' to be a bytes object. Got %s",
            bytesObj->ob_type->tp_name
        );
    }

    Type* selfType = PyInstance::unwrapTypeArgToTypePtr(o);

    if (!selfType || !selfType->isTupleOrListOf()) {
        PyErr_Format(PyExc_TypeError, "Expected cls to be a Type");
        return nullptr;
    }

    TupleOrListOfType* tupT = (TupleOrListOfType*)selfType;

    if (!tupT->getEltType()->isPOD()) {
        PyErr_Format(
            PyExc_TypeError,
            "Can't convert %s from bytes because internals are not POD",
            tupT->name().c_str()
        );
        return nullptr;
    }

    int64_t bytecount = tupT->getEltType()->bytecount();
    size_t sz = PyBytes_GET_SIZE(bytesObj);

    if ((bytecount == 0 && sz) || sz % bytecount != 0) {
        PyErr_Format(PyExc_ValueError, "Byte array must be an integer multiple of underlying type.");
        return nullptr;
    }

    size_t eltCount = bytecount > 0 ? sz / bytecount : 0;

    Instance outConverted(selfType, [&](instance_ptr data) {
        tupT->constructor(data);
        tupT->reserve(data, eltCount);
        memcpy(tupT->eltPtr(data, 0), PyBytes_AsString(bytesObj), sz);
        tupT->setSizeUnsafe(data, eltCount);
    });

    return PyInstance::fromInstance(outConverted);
}

PyDoc_STRVAR(TupleOf_toArray_doc,
    "t.toarray() -> numpy array\n"
    "\n"
    "Converts a TupleOf() instance to a numpy array\n"
    "Raises TypeError on failure.\n"
    );
PyDoc_STRVAR(ListOf_toArray_doc,
    "lst.toarray() -> numpy array\n"
    "\n"
    "Converts a ListOf() instance to a numpy array\n"
    "Raises TypeError on failure.\n"
    );
PyObject* PyTupleOrListOfInstance::toArray(PyObject* o, PyObject* args) {
    PyListOfInstance* self_w = (PyListOfInstance*)o;
    npy_intp dims[1] = { self_w->type()->count(self_w->dataPtr()) };

    int typenum = -1;
    int bytecount = 0;

    if (self_w->type()->getEltType()->getTypeCategory() == Type::TypeCategory::catBool) {
        typenum = NPY_BOOL;
        bytecount = sizeof(bool);
    }

    if (self_w->type()->getEltType()->getTypeCategory() == Type::TypeCategory::catInt64) {
        typenum = NPY_INT64;
        bytecount = sizeof(int64_t);
    }

    if (self_w->type()->getEltType()->getTypeCategory() == Type::TypeCategory::catInt32) {
        typenum = NPY_INT32;
        bytecount = sizeof(int32_t);
    }

    if (self_w->type()->getEltType()->getTypeCategory() == Type::TypeCategory::catInt16) {
        typenum = NPY_INT16;
        bytecount = sizeof(int16_t);
    }

    if (self_w->type()->getEltType()->getTypeCategory() == Type::TypeCategory::catInt8) {
        typenum = NPY_INT8;
        bytecount = sizeof(int8_t);
    }

    if (self_w->type()->getEltType()->getTypeCategory() == Type::TypeCategory::catUInt64) {
        typenum = NPY_UINT64;
        bytecount = sizeof(uint64_t);
    }

    if (self_w->type()->getEltType()->getTypeCategory() == Type::TypeCategory::catUInt32) {
        typenum = NPY_UINT32;
        bytecount = sizeof(uint32_t);
    }

    if (self_w->type()->getEltType()->getTypeCategory() == Type::TypeCategory::catUInt16) {
        typenum = NPY_UINT16;
        bytecount = sizeof(uint16_t);
    }

    if (self_w->type()->getEltType()->getTypeCategory() == Type::TypeCategory::catUInt8) {
        typenum = NPY_UINT8;
        bytecount = sizeof(uint8_t);
    }

    if (self_w->type()->getEltType()->getTypeCategory() == Type::TypeCategory::catFloat64) {
        typenum = NPY_FLOAT64;
        bytecount = sizeof(double);
    }

    if (self_w->type()->getEltType()->getTypeCategory() == Type::TypeCategory::catFloat32) {
        typenum = NPY_FLOAT32;
        bytecount = sizeof(float);
    }

    if (bytecount) {
        PyObject* resultArray = PyArray_SimpleNew(
            1,
            dims,
            typenum
            );
        memcpy(
            PyArray_BYTES(resultArray),
            self_w->type()->eltPtr(self_w->dataPtr(), 0),
            dims[0] * bytecount
            );

        return resultArray;
    }

    PyErr_Format(PyExc_TypeError, "Can't convert %s to a numpy array.", self_w->type()->name().c_str());
    return NULL;
}

PyObject* PyTupleOrListOfInstance::sq_item_concrete(Py_ssize_t ix) {
    int64_t count = type()->count(dataPtr());

    if (ix < 0) {
        ix += count;
    }

    if (ix >= count || ix < 0) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }

    Type* eltType = type()->getEltType();

    return extractPythonObject(
        type()->eltPtr(dataPtr(), ix),
        eltType
    );
}

Py_ssize_t PyTupleOrListOfInstance::mp_and_sq_length_concrete() {
    return type()->count(dataPtr());
}

PyObject* PyTupleOrListOfInstance::mp_subscript_concrete(PyObject* item) {
    if (PySlice_Check(item)) {
        Py_ssize_t start,stop,step,slicelength;

        if (PySlice_GetIndicesEx(item, type()->count(dataPtr()), &start,
                    &stop, &step, &slicelength) == -1) {
            return NULL;
        }

        Type* eltType = type()->getEltType();

        return PyInstance::initialize(type(), [&](instance_ptr data) {
            type()->constructor(data, slicelength,
                [&](uint8_t* eltPtr, int64_t k) {
                    eltType->copy_constructor(
                        eltPtr,
                        type()->eltPtr(dataPtr(), start + k * step)
                        );
                    }
                );
        });
    }

    if (PyLong_Check(item)) {
        return sq_item((PyObject*)this, PyLong_AsLongLong(item));
    }

    if (PyIndex_Check(item)) {
        PyObjectStealer res(PyNumber_Index(item));

        if (!res) {
            return NULL;
        }

        if (!PyLong_Check(res)) {
            PyErr_Format(PyExc_TypeError, "__index__ returned a non int for %S", item);
            return NULL;
        }

        return sq_item((PyObject*)this, PyLong_AsLongLong(res));
    }

    PyErr_Format(PyExc_TypeError, "ListOf indices must be integers or slices, not %S", item);
    return NULL;
}

const char* TUPLE_CONVERT_DOCSTRING =
    "Convert an arbitrary value to a `TupleOf(T)`\n\n"
    "Unlike calling `TupleOf(T)`, `TupleOf(T).convert` will recursively attempt\n"
    "to convert all of the interior values in the argument as well.\n"
    "Normally, `TupleOf(T)([x])` will only succeed if `x` is implicitly convertible to `T`\n"
    "TupleOf(T).convert([x]) will call 'convert' on each of the members of the iterable.\n"
    "Note that this means that you may get deepcopies of objects: `TupleOf(TupleOf(int))([[x]])`\n"
    "will duplicate the inner list as a `TupleOf(int)`."
;


const char* TUPLE_TO_BYTES_DOCSTRING =
    "Convert a TupleOf(T) to the raw bytes underneath it.\n\n"
    "This is only valid if T is \"POD\" (Plain Old Data), meaning it must be\n"
    "an integer, float, or bool type, or some combination of those in a Tuple or \n"
    "NamedTuple."
;

const char* TUPLE_FROM_BYTES_DOCSTRING =
    "Construct a TupleOf(T) from the raw bytes that would underly it.\n\n"
    "This is only valid if T is \"POD\" (Plain Old Data), meaning it must be\n"
    "an integer, float, or bool type, or some combination of those in a Tuple or \n"
    "NamedTuple."
;

PyDoc_STRVAR(tuplePointerUnsafe_doc,
    "tup.pointerUnsafe(i) -> pointer to element i of tup\n"
    "\n"
    "If tup if of type TupleOf(T), the return value is of type PointerTo(T).\n"
    "This is intended for compiled code, where pointerUnsafe is expected to be\n"
    "faster than the usual bounds-checked indexed access to an instance of\n"
    "TupleOf(T).\n"
    "In C++ terms, tup.pointerUnsafe(i) is &tup[i].\n"

);
PyObject* PyTupleOrListOfInstance::pointerUnsafe(PyObject* o, PyObject* args) {
    PyTupleOrListOfInstance* self_w = (PyTupleOrListOfInstance*)o;

    if (PyTuple_Size(args) != 1 || !PyLong_Check(PyTuple_GetItem(args, 0))) {
        PyErr_SetString(PyExc_TypeError, "ListOf.pointerUnsafe takes one integer argument");
        return NULL;
    }

    int64_t ix = PyLong_AsLongLong(PyTuple_GetItem(args,0));

    void* ptr = (void*)self_w->type()->eltPtr(self_w->dataPtr(), ix);

    return extractPythonObject((instance_ptr)&ptr, PointerTo::Make(self_w->type()->getEltType()));
}



PyMethodDef* PyTupleOfInstance::typeMethodsConcrete(Type* t) {
    return new PyMethodDef [6] {
        {"toArray", (PyCFunction)PyTupleOrListOfInstance::toArray, METH_VARARGS, TupleOf_toArray_doc},
        {"toBytes", (PyCFunction)PyTupleOrListOfInstance::toBytes, METH_VARARGS, TUPLE_TO_BYTES_DOCSTRING},
        {"fromBytes", (PyCFunction)PyTupleOrListOfInstance::fromBytes, METH_VARARGS | METH_KEYWORDS | METH_CLASS, TUPLE_FROM_BYTES_DOCSTRING},
        {"pointerUnsafe", (PyCFunction)PyTupleOrListOfInstance::pointerUnsafe, METH_VARARGS, tuplePointerUnsafe_doc},
        {NULL, NULL}
    };
}

//static
PyDoc_STRVAR(listPointerUnsafe_doc,
    "lst.pointerUnsafe(i) -> pointer to element i of lst\n"
    "\n"
    "If lst if of type ListOf(T), the return value is of type PointerTo(T).\n"
    "This is intended for compiled code, where pointerUnsafe is expected to be\n"
    "faster than the usual bounds-checked indexed access to an instance of\n"
    "ListOf(T).\n"
    "In C++ terms, lst.pointerUnsafe(i) is &lst[i].\n"
);

// static
PyDoc_STRVAR(listAppend_doc,
    "lst.append(e) -> None, and appends element e to lst"
    );
PyObject* PyListOfInstance::listAppend(PyObject* o, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "ListOf.append takes one argument");
        return NULL;
    }

    return listAppendDirect(o, PyTuple_GetItem(args, 0));
}

PyObject* PyListOfInstance::listAppendDirect(PyObject* o, PyObject* value) {
    try {
        PyListOfInstance* self_w = (PyListOfInstance*)o;

        Type* value_type = extractTypeFrom(value->ob_type);

        Type* eltType = self_w->type()->getEltType();

        if (value_type == eltType) {
            PyInstance* value_w = (PyInstance*)(PyObject*)value;

            self_w->type()->append(self_w->dataPtr(), value_w->dataPtr());
        } else {
            Instance temp(eltType, [&](instance_ptr data) {
                PyInstance::copyConstructFromPythonInstance(
                    eltType,
                    data,
                    value,
                    ConversionLevel::ImplicitContainers
                );
            });

            self_w->type()->append(self_w->dataPtr(), temp.data());
        }

        return incref(Py_None);
    } catch(std::exception& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    } catch(PythonExceptionSet& e) {
        return NULL;
    }
}


// static
PyDoc_STRVAR(listExtend_doc,
    "lst.extend(c) -> None, and appends elements of the container c to lst"
    );
PyObject* PyListOfInstance::listExtend(PyObject* o, PyObject* args) {
    return translateExceptionToPyObject([&]() {
        if (PyTuple_Size(args) != 1) {
            throw std::runtime_error("ListOf.extend takes one argument");
        }

        PyObjectHolder value(PyTuple_GetItem(args, 0));

        PyListOfInstance* self_w = (PyListOfInstance*)o;

        ListOfType* self_type = (ListOfType*)self_w->type();

        Type* value_type = extractTypeFrom(value->ob_type);

        auto extendFromBinaryCompatiblePtr = [&](instance_ptr otherObj) {
            // directly extend ourselves, because we are binary compatible with the other type.
            self_type->getEltType()->check([&](auto& concrete_subtype) {
                self_type->extend(self_w->dataPtr(), self_type->count(otherObj),
                    [&](instance_ptr tgt, size_t i) {
                        concrete_subtype.copy_constructor(tgt, self_type->eltPtr(otherObj, i));
                    }
                );
            });
        };

        if (value_type && (value_type->getTypeCategory() == Type::TypeCategory::catListOf ||
                value_type->getTypeCategory() == Type::TypeCategory::catTupleOf) &&
                ((TupleOrListOfType*)value_type)->getEltType() == self_type->getEltType()) {

            extendFromBinaryCompatiblePtr(((PyInstance*)(PyObject*)value)->dataPtr());

            return incref(Py_None);
        }

        // iterate
        iterate(value, [&](PyObject* arg) {
            if (!listAppendDirect(o, arg)) {
                throw PythonExceptionSet();
            }
        });

        return incref(Py_None);
    });
}

// static
PyDoc_STRVAR(listReserved_doc,
    "lst.reserved() -> number of elements lst could hold, as currently allocated\n"
    "\n"
    "Contrast this with the actual length len(lst) of the lst.\n"
    "lst.reserved() >= len(lst)\n"
    );
PyObject* PyListOfInstance::listReserved(PyObject* o, PyObject* args) {
    if (PyTuple_Size(args) != 0) {
        PyErr_SetString(PyExc_TypeError, "ListOf.reserved takes no arguments");
        return NULL;
    }

    PyListOfInstance* self_w = (PyListOfInstance*)o;

    return PyLong_FromLong(self_w->type()->reserved(self_w->dataPtr()));
}

PyDoc_STRVAR(listReserve_doc,
    "lst.reserve(n) -> None, and allocates space to potentially hold n elements\n"
    "\n"
    "Expands or shrinks the memory allocated for lst.\n"
    "Won't shrink lst smaller than the elements it currently contains.\n"
    "For example, n=0 will reallocate lst to the minimum possible size, \n"
    "containing len(lst) elements and no unused allocation.\n"
    );
PyObject* PyListOfInstance::listReserve(PyObject* o, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "ListOf.reserve takes one argument");
        return NULL;
    }

    PyObject* pyReserveSize = PyTuple_GetItem(args, 0);

    if (!PyLong_Check(pyReserveSize)) {
        PyErr_SetString(PyExc_TypeError, "ListOf.reserve takes an integer");
        return NULL;
    }

    int size = PyLong_AsLongLong(pyReserveSize);

    PyListOfInstance* self_w = (PyListOfInstance*)o;

    self_w->type()->reserve(self_w->dataPtr(), size);

    return incref(Py_None);
}

PyDoc_STRVAR(listClear_doc,
    "lst.clear() -> None, and resize lst to 0."
    );
PyObject* PyListOfInstance::listClear(PyObject* o, PyObject* args) {
    if (PyTuple_Size(args) != 0) {
        PyErr_SetString(PyExc_TypeError, "ListOf.clear takes no arguments");
        return NULL;
    }

    PyListOfInstance* self_w = (PyListOfInstance*)o;

    self_w->type()->resize(self_w->dataPtr(), 0);

    return incref(Py_None);
}

PyDoc_STRVAR(listResize_doc,
    "lst.resize(n[, e]) -> None, and lst now contains n elements\n"
    "\n"
    "if n > len(lst), default-constructed elements (or e if provided) are\n"
    "placed the end of lst.\n"
    "if n < len(lst), elements are deleted from the end of lst.\n"
    );
PyObject* PyListOfInstance::listResize(PyObject* o, PyObject* args) {
    try {
        if (PyTuple_Size(args) != 1 && PyTuple_Size(args) != 2) {
            PyErr_SetString(PyExc_TypeError, "ListOf.resize takes one argument");
            return NULL;
        }

        PyObject* pySize = PyTuple_GetItem(args, 0);

        if (!PyLong_Check(pySize)) {
            PyErr_SetString(PyExc_TypeError, "ListOf.resize takes an integer");
            return NULL;
        }

        int64_t size = PyLong_AsLongLong(pySize);

        PyListOfInstance* self_w = (PyListOfInstance*)o;
        Type* eltType = self_w->type()->getEltType();

        if (self_w->type()->count(self_w->dataPtr()) > size) {
            self_w->type()->resize(self_w->dataPtr(), size);
        } else {
            if (PyTuple_Size(args) == 2) {
                Instance temp(eltType, [&](instance_ptr data) {
                    PyInstance::copyConstructFromPythonInstance(
                        eltType,
                        data,
                        PyTuple_GetItem(args, 1),
                        ConversionLevel::ImplicitContainers
                    );
                });

                self_w->type()->resize(self_w->dataPtr(), size, temp.data());
            } else {
                if (!self_w->type()->getEltType()->is_default_constructible()) {
                    PyErr_SetString(PyExc_TypeError, "Cannot increase the size of this list without an object to copy in because the"
                        " element type is not default-constructible");
                    return NULL;
                }

                self_w->type()->resize(self_w->dataPtr(), size);
            }
        }

        return incref(Py_None);
    } catch(std::exception& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    } catch(PythonExceptionSet& e) {
        return NULL;
    }
}

PyDoc_STRVAR(listPop_doc,
    "lst.pop() -> last element of lst, and remove last element from lst\n"
    "lst.pop(n) -> nth element of lst, and remove nth element from lst\n"
    "\n"
    "Raises IndexError when called on an empty list.\n"
    "Raises IndexError when n is out of range.\n"
    "Negative n is interpreted as in slice notation.\n"
    );
PyObject* PyListOfInstance::listPop(PyObject* o, PyObject* args) {
    if (PyTuple_Size(args) != 0 && PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "ListOf.pop takes zero or one argument");
        return NULL;
    }

    int64_t which = -1;

    if (PyTuple_Size(args)) {
        PyObject* pySize = PyTuple_GetItem(args, 0);

        if (!PyLong_Check(pySize)) {
            PyErr_SetString(PyExc_TypeError, "ListOf.pop takes an integer");
            return NULL;
        }

        which = PyLong_AsLongLong(pySize);
    }

    PyListOfInstance* self_w = (PyListOfInstance*)o;

    int64_t listSize = self_w->type()->count(self_w->dataPtr());

    if (listSize == 0) {
        PyErr_SetString(PyExc_IndexError, "pop from empty list");
        return NULL;
    }

    if (which < 0) {
        which += listSize;
    }

    if (which < 0 || which >= listSize) {
        PyErr_SetString(PyExc_IndexError, "pop index out of range");
        return NULL;
    }

    PyObject* result = extractPythonObject(
            self_w->type()->eltPtr(self_w->dataPtr(), which),
            self_w->type()->getEltType()
            );

    self_w->type()->remove(self_w->dataPtr(), which);

    return result;
}

PyDoc_STRVAR(listTranspose_doc,
    "lst.transpose() -> transposition of ListOf(Tuple()) or ListOf(NamedTuple())\n"
    "\n"
    "This is a matrix transposition that swaps the types of the rows and columns.\n"
    "If lst is a ListOf(Tuple(X1, X2, X3)), then lst.transpose() is a\n"
    "Tuple(ListOf(X1), ListOf(X2), ListOf(X3)).\n"
    "If lst is a ListOf(NamedTuple(a=X1, b=X2, c=X3)), then lst.transpose() is a\n"
    "NamedTuple(a=ListOf(X1), b=ListOf(X2), c=ListOf(X3)).\n"
    );
PyObject* PyListOfInstance::listTranspose(PyObject* o, PyObject* args) {
    if (PyTuple_Size(args) != 0) {
        PyErr_SetString(PyExc_TypeError, "ListOf.transpose takes zero arguments");
        return NULL;
    }

    PyListOfInstance* self_w = (PyListOfInstance*)o;

    ListOfType* ourListType = (ListOfType*)self_w->type();

    int64_t listSize = self_w->type()->count(self_w->dataPtr());
    instance_ptr ourListPtr = self_w->dataPtr();

    if (ourListType->getEltType()->getTypeCategory() != Type::TypeCategory::catNamedTuple &&
            ourListType->getEltType()->getTypeCategory() != Type::TypeCategory::catTuple) {
        PyErr_Format(
            PyExc_TypeError,
            "Can't transpose %s because it's not a List of NamedTuple or Tuple objects.",
            self_w->type()->name().c_str()
        );

        return NULL;
    }

    CompositeType* ourListEltType = (CompositeType*)ourListType->getEltType();

    std::vector<Type*> childListTypes;
    for (auto t: ourListEltType->getTypes()) {
        childListTypes.push_back(ListOfType::Make(t));
    }

    CompositeType* newListType;

    if (ourListEltType->getTypeCategory() == Type::TypeCategory::catNamedTuple) {
        newListType = NamedTuple::Make(childListTypes, ourListEltType->getNames());
    } else {
        newListType = Tuple::Make(childListTypes);
    }

    //create each list
    std::vector<Instance> listInstances;

    for (long k = 0; k < ourListEltType->getOffsets().size(); k++) {
        size_t offset = ourListEltType->getOffsets()[k];
        Type* eltType = ourListEltType->getTypes()[k];
        ListOfType* tgtList = (ListOfType*)childListTypes[k];

        listInstances.push_back(Instance(tgtList, [&](instance_ptr listBody) {
            eltType->check([&](auto& eltTypeConcrete) {
                tgtList->constructor(listBody, listSize, [&](instance_ptr toCopy, size_t index) {
                    eltTypeConcrete.copy_constructor(
                        toCopy,
                        ourListType->eltPtr(ourListPtr, index) + offset
                    );
                });
            });
        }));
    }

    Instance outTuple(newListType, [&](instance_ptr outComposite) {
        newListType->constructor(outComposite, [&](instance_ptr outSublist, size_t outSublistIx) {
            ((ListOfType*)childListTypes[outSublistIx])->copy_constructor(outSublist, listInstances[outSublistIx].data());
        });
    });

    return PyInstance::fromInstance(outTuple);
}

int PyListOfInstance::mp_ass_subscript_concrete(PyObject* item, PyObject* value) {
    if (!value) {
        PyErr_SetString(PyExc_TypeError, "Item deletion is not implemented yet");
        throw PythonExceptionSet();
    }

    Type* value_type = extractTypeFrom(value->ob_type);

    Type* eltType = type()->getEltType();

    PyObjectHolder index(item);

    if (!PyLong_Check(item)) {
        index.set(PyNumber_Index(item));

        if (!index) {
            throw PythonExceptionSet();
        }
    }

    if (PyLong_Check(index)) {
        int64_t ix = PyLong_AsLongLong(index);
        int64_t count = type()->count(dataPtr());

        if (ix < 0) {
            ix += count;
        }

        if (ix >= count || ix < 0) {
            PyErr_SetString(PyExc_IndexError, "index out of range");
            return -1;
        }

        if (value_type == eltType) {
            PyInstance* value_w = (PyInstance*)value;

            eltType->assign(
                type()->eltPtr(dataPtr(), ix),
                value_w->dataPtr()
                );

            return 0;
        } else {
            Instance toAssign(eltType, [&](instance_ptr data) {
                PyInstance::copyConstructFromPythonInstance(
                    eltType,
                    data,
                    value,
                    ConversionLevel::ImplicitContainers
                );
            });

            eltType->assign(
                type()->eltPtr(dataPtr(), ix),
                toAssign.data()
                );

            return 0;
        }
    }

    return PyInstance::mp_ass_subscript_concrete(item, value);
}

PyDoc_STRVAR(
    LIST_TO_BYTES_DOCSTRING,
    "ListOf(T).toBytes(self) -> bytes\n\nConvert a ListOf(T) to the raw bytes underneath it.\n\n"
    "This is only valid if T is \"POD\" (Plain Old Data), meaning it must be\n"
    "an integer, float, or bool type, or some combination of those in a Tuple or \n"
    "NamedTuple."
);

PyDoc_STRVAR(
    LIST_FROM_BYTES_DOCSTRING,
    "ListOf(T).fromBytes(b: bytes) -> ListOf(T)\n\nConstruct a ListOf(T) from the raw bytes that would underly it.\n\n"
    "This is only valid if T is \"POD\" (Plain Old Data), meaning it must be\n"
    "an integer, float, or bool type, or some combination of those in a Tuple or \n"
    "NamedTuple."
);

PyMethodDef* PyListOfInstance::typeMethodsConcrete(Type* t) {
    return new PyMethodDef [14] {
        {"toArray", (PyCFunction)PyTupleOrListOfInstance::toArray, METH_VARARGS, ListOf_toArray_doc},
        {"toBytes", (PyCFunction)PyTupleOrListOfInstance::toBytes, METH_VARARGS, LIST_TO_BYTES_DOCSTRING},
        {"fromBytes", (PyCFunction)PyTupleOrListOfInstance::fromBytes, METH_VARARGS | METH_KEYWORDS | METH_CLASS, LIST_FROM_BYTES_DOCSTRING},
        {"append", (PyCFunction)PyListOfInstance::listAppend, METH_VARARGS, listAppend_doc},
        {"extend", (PyCFunction)PyListOfInstance::listExtend, METH_VARARGS, listExtend_doc},
        {"clear", (PyCFunction)PyListOfInstance::listClear, METH_VARARGS, listClear_doc},
        {"reserved", (PyCFunction)PyListOfInstance::listReserved, METH_VARARGS, listReserved_doc},
        {"reserve", (PyCFunction)PyListOfInstance::listReserve, METH_VARARGS, listReserve_doc},
        {"resize", (PyCFunction)PyListOfInstance::listResize, METH_VARARGS, listResize_doc},
        {"pop", (PyCFunction)PyListOfInstance::listPop, METH_VARARGS, listPop_doc},
        {"setSizeUnsafe", (PyCFunction)PyListOfInstance::listSetSizeUnsafe, METH_VARARGS, listSetSizeUnsafe_doc},
        {"pointerUnsafe", (PyCFunction)PyTupleOrListOfInstance::pointerUnsafe, METH_VARARGS, listPointerUnsafe_doc},
        {"transpose", (PyCFunction)PyListOfInstance::listTranspose, METH_VARARGS, listTranspose_doc},
        {NULL, NULL}
    };
}

void PyListOfInstance::constructFromPythonArgumentsConcrete(ListOfType* t, uint8_t* data, PyObject* args, PyObject* kwargs) {
    if ((!kwargs || PyDict_Size(kwargs) == 0) && (args && PyTuple_Size(args) == 1)) {
        PyObject* arg = PyTuple_GetItem(args, 0);
        Type* argType = extractTypeFrom(arg->ob_type);

        if (Type::typesEquivalent(argType, t)) {
            //following python semantics, this needs to produce a new object
            //that's a copy of the original list. We can't just incref it and return
            //the original object because it has state.
            ListOfType* listT = (ListOfType*)t;
            listT->copyListObject(data, ((PyInstance*)arg)->dataPtr());
            return;
        }
    }

    PyInstance::constructFromPythonArgumentsConcrete(t, data, args, kwargs);
}

void PyTupleOrListOfInstance::mirrorTypeInformationIntoPyTypeConcrete(TupleOrListOfType* inType, PyTypeObject* pyType) {
    //expose 'ElementType' as a member of the type object
    PyDict_SetItemString(
        pyType->tp_dict,
        "ElementType",
        typePtrToPyTypeRepresentation(inType->getEltType())
        );
}

int PyTupleOrListOfInstance::pyInquiryConcrete(const char* op, const char* opErrRep) {
    // op == '__bool__'
    return type()->count(dataPtr()) != 0;
}

bool PyListOfInstance::compare_to_python_concrete(ListOfType* listT, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp) {
    auto convert = [&](char cmpValue) { return cmpResultToBoolForPyOrdering(pyComparisonOp, cmpValue); };

    Type* otherType = extractTypeFrom(other->ob_type);

    if (PyList_Check(other) || (otherType && otherType->getTypeCategory() == Type::TypeCategory::catListOf)) {
        int lenS = listT->count(self);
        int indexInOwn = 0;

        int result = 0;

        iterateWithEarlyExit(other, [&](PyObject* listItem) {
            if (indexInOwn >= lenS) {
                // we ran out of items in our list
                result = -1;
                return false;
            }

            if (!compare_to_python(listT->getEltType(), listT->eltPtr(self, indexInOwn), listItem, exact, Py_EQ)) {
                if (pyComparisonOp == Py_EQ || pyComparisonOp == Py_NE) {
                    result = 1;
                    return false;
                }
                if (compare_to_python(listT->getEltType(), listT->eltPtr(self, indexInOwn), listItem, exact, Py_LT)) {
                    result = -1;
                    return false;
                }

                result = 1;
                return false;
            }

            indexInOwn += 1;

            return true;
        });

        if (result) {
            return convert(result);
        }

        if (indexInOwn == lenS) {
            return convert(0);
        }

        return convert(1);
    }

    if (pyComparisonOp == Py_EQ || pyComparisonOp == Py_NE) {
        return convert(1);
    }

    PyErr_Format(
        PyExc_TypeError,
        "Comparison not supported between instances of '%s' and '%s'.",
        listT->name().c_str(),
        other->ob_type->tp_name
    );

    throw PythonExceptionSet();
}

bool PyTupleOfInstance::compare_to_python_concrete(TupleOfType* tupT, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp) {
    auto convert = [&](char cmpValue) { return cmpResultToBoolForPyOrdering(pyComparisonOp, cmpValue); };

    Type* otherType = extractTypeFrom(other->ob_type);

    if (PyTuple_Check(other) || (otherType && (otherType->getTypeCategory() == Type::TypeCategory::catTupleOf || otherType->isComposite()))) {
        int lenS = tupT->count(self);
        int indexInOwn = 0;

        int result = 0;

        iterateWithEarlyExit(other, [&](PyObject* tupleItem) {
            if (indexInOwn >= lenS) {
                // we ran out of items in our list
                result = -1;
                return false;
            }

            if (!compare_to_python(tupT->getEltType(), tupT->eltPtr(self, indexInOwn), tupleItem, exact, Py_EQ)) {
                if (pyComparisonOp == Py_EQ || pyComparisonOp == Py_NE) {
                    result = 1;
                    return false;
                }
                if (compare_to_python(tupT->getEltType(), tupT->eltPtr(self, indexInOwn), tupleItem, exact, Py_LT)) {
                    result = -1;
                    return false;
                }

                result = 1;
                return false;
            }

            indexInOwn += 1;

            return true;
        });

        if (result) {
            return convert(result);
        }

        if (indexInOwn == lenS) {
            return convert(0);
        }

        return convert(1);
    }

    if (pyComparisonOp == Py_EQ || pyComparisonOp == Py_NE) {
        return convert(1);
    }

    PyErr_Format(
        PyExc_TypeError,
        "Comparison not supported between instances of '%s' and '%s'.",
        tupT->name().c_str(),
        other->ob_type->tp_name
    );

    throw PythonExceptionSet();
}
