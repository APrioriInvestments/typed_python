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

#pragma once

#include "PyInstance.hpp"

class PyHeldClassInstance : public PyInstance {
public:
    typedef HeldClass modeled_type;

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, ConversionLevel level) {
        return true;
    }

    PyObject* tp_getattr_concrete(PyObject* pyAttrName, const char* attrName) {
        if (type()->getTypeCategory() != Type::TypeCategory::catHeldClass) {
            throw std::runtime_error("RefTo only works with HeldClass");
        }

        HeldClass* clsType = (HeldClass*)type();

        int index = clsType->getMemberIndex(attrName);

        // we're accessing a member of a reference to a held class.
        // we figure out the pointer to the instance and get that pointer.
        if (index >= 0) {
            Type* eltType = clsType->getMemberType(index);

            instance_ptr heldClassBody = dataPtr();

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
            if (method->getFirstArgType() != clsType->getRefToType()) {
                throw std::runtime_error("somehow our bound method has the wrong type!");
            }

            instance_ptr heldData = dataPtr();

            return PyInstance::initializePythonRepresentation(method, [&](instance_ptr data) {
                method->copy_constructor(data, (instance_ptr)&heldData);
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

        PyErr_Format(
            PyExc_AttributeError,
            "no attribute %s for instance of type %s", attrName, type()->name().c_str()
        );

        return NULL;
    }

    int tp_setattr_concrete(PyObject* attrName, PyObject* attrVal) {
        HeldClass* clsType = (HeldClass*)type();

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

        instance_ptr heldClassBody = dataPtr();

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

    static void mirrorTypeInformationIntoPyTypeConcrete(HeldClass* classT, PyTypeObject* pyType) {
        PyClassInstance::mirrorTypeInformationIntoPyTypeConcrete(
            classT->getClassType(),
            pyType,
            true /* asHeldClass */
        );
    }
};
