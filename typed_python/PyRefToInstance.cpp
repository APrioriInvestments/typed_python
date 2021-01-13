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

#include "PyRefToInstance.hpp"

RefTo* PyRefToInstance::type() {
    return (RefTo*)extractTypeFrom(((PyObject*)this)->ob_type);
}

PyMethodDef* PyRefToInstance::typeMethodsConcrete(Type* t) {
    return new PyMethodDef [5] {
        {NULL, NULL}
    };
}

void PyRefToInstance::mirrorTypeInformationIntoPyTypeConcrete(RefTo* pointerT, PyTypeObject* pyType) {
    //expose 'ElementType' as a member of the type object
    PyDict_SetItemString(
            pyType->tp_dict,
            "ElementType",
            typePtrToPyTypeRepresentation(pointerT->getEltType())
            );
}

int PyRefToInstance::pyInquiryConcrete(const char* op, const char* opErrRep) {
    // op == '__bool__'
    return *(void**)dataPtr() != nullptr;
}

PyObject* PyRefToInstance::tp_getattr_concrete(PyObject* pyAttrName, const char* attrName) {
    if (type()->getEltType()->getTypeCategory() != Type::TypeCategory::catHeldClass) {
        throw std::runtime_error("RefTo only works with HeldClass");
    }

    HeldClass* clsType = (HeldClass*)type()->getEltType();

    int index = clsType->getMemberIndex(attrName);

    // we're accessing a member of a reference to a held class.
    // we figure out the pointer to the instance and get that pointer.
    if (index >= 0) {
        Type* eltType = clsType->getMemberType(index);

        instance_ptr heldClassBody = *(instance_ptr*)dataPtr();

        if (!clsType->checkInitializationFlag(heldClassBody, index)) {
            PyErr_Format(
                PyExc_AttributeError,
                "Attribute '%S' is not initialized",
                pyAttrName
            );
            return NULL;
        }

        return extractPythonObject(clsType->eltPtr(heldClassBody, index), eltType);
    }

    BoundMethod* method = clsType->getMemberFunctionMethodType(attrName, true /* forHeld */);
    if (method) {
        if (method->getFirstArgType() != type()) {
            throw std::runtime_error("somehow our bound method has the wrong type!");
        }

        return PyInstance::initializePythonRepresentation(method, [&](instance_ptr data) {
            method->copy_constructor(data, dataPtr());
        });
    }

    {
        auto it = clsType->getPropertyFunctions().find(attrName);
        if (it != clsType->getPropertyFunctions().end()) {
            auto res = PyFunctionInstance::tryToCall(it->second, nullptr, (PyObject*)this);
            if (res.first) {
                return res.second;
            }

            PyErr_Format(
                PyExc_TypeError,
                "Found a property for %s but failed to call it with 'self'",
                attrName
                );
            return NULL;
        }
    }

    {
        auto it = clsType->getClassMembers().find(attrName);
        if (it != clsType->getClassMembers().end()) {
            return incref(it->second);
        }
    }

    PyObject* ret = PyInstance::tp_getattr_concrete(pyAttrName, attrName);
    if (!ret && PyErr_Occurred() && PyErr_ExceptionMatches(PyExc_AttributeError)) {
        PyErr_Clear();
        auto p = callMemberFunction("__getattr__", pyAttrName);
        if (p.first) {
            return p.second;
        }
        else {
            PyErr_Format(PyExc_AttributeError, "no attribute %s for instance of type %s", attrName, type()->name().c_str());
        }
    }

    return ret;
}

int PyRefToInstance::tp_setattr_concrete(PyObject* attrName, PyObject* attrVal) {
    if (type()->getEltType()->getTypeCategory() != Type::TypeCategory::catHeldClass) {
        throw std::runtime_error("RefTo only works with HeldClass");
    }

    HeldClass* clsType = (HeldClass*)type()->getEltType();

    int i = clsType->memberNamed(PyUnicode_AsUTF8(attrName));

    if (i < 0) {
        auto it = clsType->getClassMembers().find(PyUnicode_AsUTF8(attrName));
        if (it == clsType->getClassMembers().end()) {
            PyErr_Format(
                PyExc_AttributeError,
                "'%s' object has no attribute '%S' and cannot add attributes to instances of this type",
                clsType->name().c_str(), attrName
            );
        } else {
            PyErr_Format(
                PyExc_AttributeError,
                "Cannot modify read-only class member '%S' of instance of type '%s'",
                attrName, clsType->name().c_str()
            );
        }
        return -1;
    }

    Type* eltType = clsType->getMemberType(i);

    Type* attrType = extractTypeFrom(attrVal->ob_type);

    instance_ptr heldClassBody = *(instance_ptr*)dataPtr();

    if (Type::typesEquivalent(eltType, attrType)) {
        PyInstance* item_w = (PyInstance*)attrVal;

        clsType->setAttribute(heldClassBody, i, item_w->dataPtr());

        return 0;
    } else if (attrType && attrType->getTypeCategory() == Type::TypeCategory::catRefTo &&
            ((RefTo*)attrType)->getEltType() == eltType) {
        PyInstance* item_w = (PyInstance*)attrVal;

        clsType->setAttribute(heldClassBody, i, *(instance_ptr*)item_w->dataPtr());

        return 0;
    } else {
        instance_ptr tempObj = (instance_ptr)tp_malloc(eltType->bytecount());
        try {
            copyConstructFromPythonInstance(eltType, tempObj, attrVal, ConversionLevel::Implicit);
        } catch(PythonExceptionSet& e) {
            tp_free(tempObj);
            return -1;
        } catch(std::exception& e) {
            tp_free(tempObj);
            PyErr_SetString(PyExc_TypeError, e.what());
            return -1;
        }

        clsType->setAttribute(heldClassBody, i, tempObj);

        eltType->destroy(tempObj);
        tp_free(tempObj);

        return 0;
    }
}


std::pair<bool, PyObject*> PyRefToInstance::callMemberFunction(const char* name, PyObject* arg0, PyObject* arg1, PyObject* arg2) {
    if (type()->getEltType()->getTypeCategory() == Type::TypeCategory::catHeldClass) {
        return std::make_pair(false, (PyObject*)nullptr);
    }
    HeldClass* cls = (HeldClass*)type()->getEltType();

    auto it = cls->getMemberFunctions().find(name);

    if (it == cls->getMemberFunctions().end()) {
        return std::make_pair(false, (PyObject*)nullptr);
    }

    Function* method = it->second;

    int argCount = 1;
    if (arg0) {
        argCount += 1;
    }
    if (arg1) {
        argCount += 1;
    }
    if (arg2) {
        argCount += 1;
    }

    PyObjectStealer targetArgTuple(PyTuple_New(argCount));

    PyTuple_SetItem(targetArgTuple, 0, incref((PyObject*)this)); //steals a reference

    if (arg0) {
        PyTuple_SetItem(targetArgTuple, 1, incref(arg0)); //steals a reference
    }
    if (arg1) {
        PyTuple_SetItem(targetArgTuple, 2, incref(arg1)); //steals a reference
    }
    if (arg2) {
        PyTuple_SetItem(targetArgTuple, 3, incref(arg2)); //steals a reference
    }

    auto res = PyFunctionInstance::tryToCallAnyOverload(method, nullptr, nullptr, targetArgTuple, nullptr);

    if (res.first) {
        return res;
    }

    PyErr_Format(
        PyExc_TypeError,
        "'%s.%s' cannot find a valid overload with these arguments",
        type()->name().c_str(),
        name
    );
    return res;
}
