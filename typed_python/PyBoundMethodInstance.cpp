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
    Type* firstArgType = type()->getFirstArgType();

    if (!f) {
        PyErr_Format(
            PyExc_TypeError,
            "'%s' can be held in a bound method, but actually calling "
            "it isn't supported yet.",
            type()->name().c_str()
            );

        return NULL;
    }

    //if we are an entrypoint, map any untyped function arguments to typed functions
    PyObjectHolder mappedArgs;
    PyObjectHolder mappedKwargs;

    if (f->isEntrypoint()) {
        mappedArgs.steal(PyTuple_New(PyTuple_Size(args)));

        for (long k = 0; k < PyTuple_Size(args); k++) {
            PyTuple_SetItem(mappedArgs, k, PyFunctionInstance::prepareArgumentToBePassedToCompiler(PyTuple_GetItem(args, k)));
        }

        mappedKwargs.steal(PyDict_New());

        if (kwargs) {
            PyObject *key, *value;
            Py_ssize_t pos = 0;

            while (PyDict_Next(kwargs, &pos, &key, &value)) {
                PyObjectStealer mapped(PyFunctionInstance::prepareArgumentToBePassedToCompiler(value));
                PyDict_SetItem(mappedKwargs, key, mapped);
            }
        }
    } else {
        mappedArgs.set(args);
        mappedKwargs.set(kwargs);
    }

    PyObjectHolder objectInstance;

    if (firstArgType->isRefTo()) {
        objectInstance.steal(
            PyInstance::initializeTemporaryRef(
                ((RefTo*)firstArgType)->getEltType(),
                *(instance_ptr*)dataPtr()
            )
        );
    } else {
        objectInstance.steal(
            PyInstance::initializePythonRepresentation(firstArgType, [&](instance_ptr d) {
                firstArgType->copy_constructor(d, dataPtr());
            })
        );
    }

    for (ConversionLevel conversionLevel: {
        ConversionLevel::Signature,
        ConversionLevel::Upcast,
        ConversionLevel::UpcastContainers,
        ConversionLevel::Implicit,
        ConversionLevel::ImplicitContainers
    }) {
        for (long overloadIx = 0; overloadIx < f->getOverloads().size(); overloadIx++) {
            std::pair<bool, PyObject*> res = PyFunctionInstance::tryToCallOverload(
                f,
                nullptr,
                overloadIx,
                objectInstance,
                mappedArgs,
                mappedKwargs,
                conversionLevel
            );

            if (res.first) {
                return res.second;
            }
        }
    }

    std::string argTupleTypeDesc = PyFunctionInstance::argTupleTypeDescription(objectInstance, args, kwargs);

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
