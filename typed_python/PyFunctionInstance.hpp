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

class PyFunctionInstance : public PyInstance {
public:
    typedef Function modeled_type;

    Function* type();

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, bool isExplicit) {
        return true;
    }

    static std::pair<bool, PyObject*> tryToCall(const Function* f, PyObject* arg0=nullptr, PyObject* arg1=nullptr, PyObject* arg2=nullptr);

    static std::pair<bool, PyObject*> tryToCallAnyOverload(const Function* f, PyObject* self, PyObject* args, PyObject* kwargs);

    static std::pair<bool, PyObject*> tryToCallOverload(const Function::Overload& f, PyObject* self, PyObject* args, PyObject* kwargs, bool convertExplicitly, bool dontActuallyCall=false);

    //perform a linear scan of all specializations contained in overload and attempt to dispatch to each one.
    //returns <true, result or none> if we dispatched.
    static std::pair<bool, PyObject*> dispatchFunctionCallToNative(const Function::Overload& overload, PyObject* argTuple, PyObject *kwargs);

    //attempt to dispatch to this one exact specialization by converting each arg to the relevant type. if
    //we can't convert, then return <false, nullptr>. If we do dispatch, return <true, result or none> and set
    //the python exception if native code returns an exception.
    static std::pair<bool, PyObject*> dispatchFunctionCallToCompiledSpecialization(
                                                const Function::Overload& overload,
                                                const Function::CompiledSpecialization& specialization,
                                                PyObject* argTuple,
                                                PyObject *kwargs
                                                );

    static PyObject* createOverloadPyRepresentation(Function* f);

    PyObject* tp_call_concrete(PyObject* args, PyObject* kwargs);

    static std::string argTupleTypeDescription(PyObject* args, PyObject* kwargs);

    static void mirrorTypeInformationIntoPyTypeConcrete(Function* inType, PyTypeObject* pyType);

    int pyInquiryConcrete(const char* op, const char* opErrRep);

    static PyMethodDef* typeMethodsConcrete(Type* t);

    static PyObject* indexOfOverloadMatching(PyObject* self, PyObject* args, PyObject* kwargs);
};
