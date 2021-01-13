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

#include "PyPointerToInstance.hpp"

PointerTo* PyPointerToInstance::type() {
    return (PointerTo*)extractTypeFrom(((PyObject*)this)->ob_type);
}

//static
PyDoc_STRVAR(pointerInitialize_doc,
    "p.initialize() -> None, and sets p to point to default-initialized value\n"
    "p.initialize(v) -> None, and sets p to point to a copy of v\n"
    "\n"
    "In C++ terms, if p is type PointerTo(T):\n"
    "(no arguments) p = new T;\n"
    "(one argument) p = new T(v);\n"
    );
PyObject* PyPointerToInstance::pointerInitialize(PyObject* o, PyObject* args) {
    PyInstance* self_w = (PyInstance*)o;
    PointerTo* pointerT = (PointerTo*)extractTypeFrom(o->ob_type);

    if (PyTuple_Size(args) != 0 && PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "PointerTo.initialize takes zero or one argument");
        return NULL;
    }

    instance_ptr target = (instance_ptr)*(void**)self_w->dataPtr();

    if (PyTuple_Size(args) == 0) {
        if (!pointerT->getEltType()->is_default_constructible()) {
            PyErr_Format(PyExc_TypeError, "%s is not default initializable", pointerT->getEltType()->name().c_str());
            return NULL;
        }

        pointerT->getEltType()->constructor(target);
        return incref(Py_None);
    } else {
        try {
            PyObjectHolder arg0(PyTuple_GetItem(args,0));

            copyConstructFromPythonInstance(pointerT->getEltType(), target, arg0, ConversionLevel::Implicit);
            return incref(Py_None);
        } catch(std::exception& e) {
            PyErr_SetString(PyExc_TypeError, e.what());
            return NULL;
        } catch(PythonExceptionSet& e) {
            return NULL;
        }
    }
}

//static
PyDoc_STRVAR(pointerSet_doc,
    "p.set(v) -> None, and sets where p points to to v.\n"
    "\n"
    "In C++ terms: *p = v;\n"
    );
PyObject* PyPointerToInstance::pointerSet(PyObject* o, PyObject* args) {
    PyInstance* self_w = (PyInstance*)o;
    PointerTo* pointerT = (PointerTo*)extractTypeFrom(o->ob_type);

    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "PointerTo.set takes one argument");
        return NULL;
    }

    instance_ptr target = (instance_ptr)*(void**)self_w->dataPtr();

    instance_ptr tempObj = (instance_ptr)tp_malloc(pointerT->getEltType()->bytecount());

    try {
        PyObjectHolder arg0(PyTuple_GetItem(args,0));

        copyConstructFromPythonInstance(pointerT->getEltType(), tempObj, arg0, ConversionLevel::Implicit);
    } catch(std::exception& e) {
        tp_free(tempObj);
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    } catch(PythonExceptionSet& e) {
        return NULL;
    }

    pointerT->getEltType()->assign(target, tempObj);
    pointerT->getEltType()->destroy(tempObj);
    tp_free(tempObj);

    return incref(Py_None);
}

PyDoc_STRVAR(
    pointerDestroy_doc,
    "p.destroy() -> None, and destroys the value pointed to by 'p'.\n"
    "\n"
    "In C++ terms, if p is type PointerTo(T), then this would be `p->~T()`.\n"
);

//static
PyObject* PyPointerToInstance::pointerDestroy(PyObject* o, PyObject* args) {
    PyInstance* self_w = (PyInstance*)o;
    PointerTo* pointerT = (PointerTo*)extractTypeFrom(o->ob_type);

    if (PyTuple_Size(args) != 0) {
        PyErr_SetString(PyExc_TypeError, "PointerTo.destroy takes no arguments");
        return NULL;
    }

    instance_ptr target = (instance_ptr)*(void**)self_w->dataPtr();

    pointerT->getEltType()->destroy(target);

    return incref(Py_None);
}

PyObject* PyPointerToInstance::mp_subscript_concrete(PyObject* key) {
    return translateExceptionToPyObject([&]() {
        int64_t offset;

        copyConstructFromPythonInstance(Int64::Make(), (instance_ptr)&offset, key, ConversionLevel::Implicit);

        instance_ptr output;

        type()->offsetBy((instance_ptr)&output, dataPtr(), offset);

        return extractPythonObject(output, type()->getEltType());
    });
}

int PyPointerToInstance::mp_ass_subscript_concrete(PyObject* item, PyObject* value) {
    return translateExceptionToPyObjectReturningInt([&]() {
        int64_t offset;

        copyConstructFromPythonInstance(Int64::Make(), (instance_ptr)&offset, item, ConversionLevel::Implicit);

        Instance val = Instance::createAndInitialize(
            type()->getEltType(),
            [&](instance_ptr i) {
                copyConstructFromPythonInstance(type()->getEltType(), i, value, ConversionLevel::Implicit);
            }
        );

        void* output;

        type()->offsetBy((instance_ptr)&output, dataPtr(), offset);

        type()->getEltType()->assign((instance_ptr)output, val.data());

        return 0;
    });
}

//static
PyDoc_STRVAR(pointerGet_doc,
    "p.get() -> element pointed to by p\n"
    "\n"
    "If p is type PointerTo(T), p.get() is type T.\n"
    "In C++ terms: *p\n"
    );
PyObject* PyPointerToInstance::pointerGet(PyObject* o, PyObject* args) {
    PyInstance* self_w = (PyInstance*)o;
    PointerTo* pointerT = (PointerTo*)extractTypeFrom(o->ob_type);

    if (PyTuple_Size(args) != 0) {
        PyErr_SetString(PyExc_TypeError, "PointerTo.get takes zero arguments");
        return NULL;
    }

    instance_ptr target = (instance_ptr)*(void**)self_w->dataPtr();

    return extractPythonObject(target, pointerT->getEltType());
}

//static
PyDoc_STRVAR(pointerCast_doc,
    "p.cast(T) -> pointer p cast to PointerTo(T)\n"
    "\n"
    "In C++ terms: (T*)p\n"
    );
PyObject* PyPointerToInstance::pointerCast(PyObject* o, PyObject* args) {
    PyInstance* self_w = (PyInstance*)o;

    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "PointerTo.cast takes one argument");
        return NULL;
    }

    PyObjectHolder arg0(PyTuple_GetItem(args, 0));

    Type* targetType = PyPointerToInstance::unwrapTypeArgToTypePtr(arg0);

    if (!targetType) {
        PyErr_SetString(PyExc_TypeError, "PointerTo.cast requires a type argument");
        return NULL;
    }

    Type* newType = PointerTo::Make(targetType);

    return extractPythonObject(self_w->dataPtr(), newType);
}

PyObject* PyPointerToInstance::pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr) {
    if (strcmp(op, "__add__") == 0 || strcmp(op, "__iadd__") == 0) {
        if (!PyIndex_Check(rhs)) {
            return PyInstance::pyOperatorConcrete(rhs, op, opErr);
        }

        PyObjectStealer index(PyNumber_Index(rhs));
        if (!index) {
            throw PythonExceptionSet();
        }

        int64_t ix = PyLong_AsLongLong(rhs);
        void* output;

        type()->offsetBy((instance_ptr)&output, dataPtr(), ix);

        return extractPythonObject((instance_ptr)&output, type());
    }

    if (strcmp(op, "__sub__") == 0 || strcmp(op, "__isub__") == 0) {
        PointerTo* otherPointer = (PointerTo*)extractTypeFrom(rhs->ob_type);

        if (otherPointer != type()) {
            //call 'super'
            return PyInstance::pyOperatorConcrete(rhs, op, opErr);
        }

        PyInstance* other_w = (PyPointerToInstance*)rhs;

        uint8_t* ptr = *(uint8_t**)dataPtr();
        uint8_t* other_ptr = *(uint8_t**)other_w->dataPtr();

        return PyLong_FromLong((ptr-other_ptr) / type()->getEltType()->bytecount());
    }

    return PyInstance::pyOperatorConcrete(rhs, op, opErr);
}

PyMethodDef* PyPointerToInstance::typeMethodsConcrete(Type* t) {
    return new PyMethodDef [6] {
        {"initialize", (PyCFunction)PyPointerToInstance::pointerInitialize, METH_VARARGS, pointerInitialize_doc},
        {"set", (PyCFunction)PyPointerToInstance::pointerSet, METH_VARARGS, pointerSet_doc},
        {"get", (PyCFunction)PyPointerToInstance::pointerGet, METH_VARARGS, pointerGet_doc},
        {"cast", (PyCFunction)PyPointerToInstance::pointerCast, METH_VARARGS, pointerCast_doc},
        {"destroy", (PyCFunction)PyPointerToInstance::pointerDestroy, METH_VARARGS, pointerDestroy_doc},
        {NULL, NULL}
    };
}

void PyPointerToInstance::mirrorTypeInformationIntoPyTypeConcrete(PointerTo* pointerT, PyTypeObject* pyType) {
    //expose 'ElementType' as a member of the type object
    PyDict_SetItemString(
            pyType->tp_dict,
            "ElementType",
            typePtrToPyTypeRepresentation(pointerT->getEltType())
            );
}

int PyPointerToInstance::pyInquiryConcrete(const char* op, const char* opErrRep) {
    // op == '__bool__'
    return *(void**)dataPtr() != nullptr;
}

PyObject* PyPointerToInstance::tp_getattr_concrete(PyObject* pyAttrName, const char* attrName) {
    if (type()->getEltType()->getTypeCategory() == Type::TypeCategory::catHeldClass) {
        HeldClass* clsType = (HeldClass*)type()->getEltType();

        int index = clsType->getMemberIndex(attrName);

        // we're accessing a member 'm' of a pointer to a held class.
        // we figure out the pointer to the instance and get that pointer.
        if (index >= 0) {
            Type* memberType = clsType->getMemberType(index);
            void* ptr = clsType->eltPtr(*(instance_ptr*)dataPtr(), index);

            return PyInstance::extractPythonObject((instance_ptr)&ptr, PointerTo::Make(memberType));
        }
    }

    if (type()->getEltType()->isNamedTuple()) {
        NamedTuple* nt = (NamedTuple*)type()->getEltType();

        auto it = nt->getNameToIndex().find(attrName);

        if (it != nt->getNameToIndex().end()) {
            int index = it->second;

            Type* memberType = nt->getTypes()[index];
            size_t offset = nt->getOffsets()[index];

            instance_ptr ptr = *(instance_ptr*)dataPtr() + offset;

            return PyInstance::extractPythonObject((instance_ptr)&ptr, PointerTo::Make(memberType));
        }
    }

    return PyInstance::tp_getattr_concrete(pyAttrName, attrName);
}

PyObject* PyPointerToInstance::pyUnaryOperatorConcrete(const char* op, const char* opErr) {
    if (strcmp(op, "__int__") == 0) {
        return PyLong_FromUnsignedLong((uint64_t)*(void**)dataPtr());
    }

    return PyInstance::pyUnaryOperatorConcrete(op, opErr);
}
