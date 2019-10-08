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

#include "PyBoundMethodInstance.hpp"
#include "PyFunctionInstance.hpp"

BoundMethod* PyBoundMethodInstance::type() {
    return (BoundMethod*)extractTypeFrom(((PyObject*)this)->ob_type);
}



PyObject* PyBoundMethodInstance::tp_call_concrete(PyObject* args, PyObject* kwargs) {
    Function* f = type()->getFunction();
    Type* c = type()->getFirstArgType();

    PyObjectStealer objectInstance(
        PyInstance::initializePythonRepresentation(c, [&](instance_ptr d) {
            c->copy_constructor(d, dataPtr());
        })
    );

    for (long convertExplicitly = 0; convertExplicitly <= 1; convertExplicitly++) {
        for (const auto& overload: f->getOverloads()) {
            std::pair<bool, PyObject*> res = PyFunctionInstance::tryToCallOverload(overload, objectInstance, args, kwargs, convertExplicitly);
            if (res.first) {
                return res.second;
            }
        }
    }

    std::string argTupleTypeDesc = PyFunctionInstance::argTupleTypeDescription(args, kwargs);

    PyErr_Format(
        PyExc_TypeError, "'%s' cannot find a valid overload with arguments of type %s",
        type()->name().c_str(),
        argTupleTypeDesc.c_str()
        );

    return NULL;
}

void PyBoundMethodInstance::mirrorTypeInformationIntoPyTypeConcrete(BoundMethod* methodT, PyTypeObject* pyType) {
    PyDict_SetItemString(pyType->tp_dict, "FirstArgType", typePtrToPyTypeRepresentation(methodT->getFirstArgType()));
    PyDict_SetItemString(pyType->tp_dict, "FuncName", PyUnicode_FromString(methodT->getFuncName().c_str()));
}


int PyBoundMethodInstance::pyInquiryConcrete(const char* op, const char* opErrRep) {
    // op == '__bool__'
    return 1;
}
