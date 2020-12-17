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

class PyAlternativeMatcherInstance : public PyInstance {
public:
    typedef AlternativeMatcher modeled_type;

    AlternativeMatcher* type() {
        return (AlternativeMatcher*)extractTypeFrom(((PyObject*)this)->ob_type);
    }

    Alternative* baseAlternative() {
        if (type()->getAlternative()->isAlternative()) {
            return (Alternative*)(type()->getAlternative());
        }

        if (type()->getAlternative()->isConcreteAlternative()) {
            return ((ConcreteAlternative*)(type()->getAlternative()))->getAlternative();
        }

        throw std::runtime_error("Invalid base alternative in PyAlternativeMatcherInstance");
    }

    static void mirrorTypeInformationIntoPyTypeConcrete(AlternativeMatcher* altMatcherT, PyTypeObject* pyType) {
        PyDict_SetItemString(pyType->tp_dict, "Alternative", typePtrToPyTypeRepresentation(altMatcherT->getBaseAlternative()));
    }

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, ConversionLevel level) {
        return true;
    }

    PyObject* tp_getattr_concrete(PyObject* pyAttrName, const char* attrName) {
        Alternative* alt = baseAlternative();

        bool matches = alt->subtypes()[
            alt->which(dataPtr())
        ].first == attrName;

        return incref(matches ? Py_True : Py_False);
    }
};
