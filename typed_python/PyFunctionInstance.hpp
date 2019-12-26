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

class FunctionCallArgMapping;

class PyFunctionInstance : public PyInstance {
public:
    typedef Function modeled_type;

    Function* type();

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, bool isExplicit);

    static void copyConstructFromPythonInstanceConcrete(modeled_type* type, instance_ptr tgt, PyObject* pyRepresentation, bool isExplicit);

    static std::pair<bool, PyObject*> tryToCall(const Function* f, PyObject* arg0=nullptr, PyObject* arg1=nullptr, PyObject* arg2=nullptr);

    static std::pair<bool, PyObject*> tryToCallAnyOverload(const Function* f, PyObject* self, PyObject* args, PyObject* kwargs);

    static std::pair<bool, PyObject*> tryToCallOverload(const Function* f, long overloadIx, PyObject* self, PyObject* args, PyObject* kwargs, bool convertExplicitly, bool dontActuallyCall);

    //perform a linear scan of all specializations contained in overload and attempt to dispatch to each one.
    //returns <true, result or none> if we dispatched..
    //if 'isEntrypoint', then if we don't match a compiled specialization, ask the runtime to produce
    //one for us.
    static std::pair<bool, PyObject*> dispatchFunctionCallToNative(const Function* f, long overloadIx, const FunctionCallArgMapping& mapping);

    //attempt to dispatch to this one exact specialization by converting each arg to the relevant type. if
    //we can't convert, then return <false, nullptr>. If we do dispatch, return <true, result or none> and set
    //the python exception if native code returns an exception.
    static std::pair<bool, PyObject*> dispatchFunctionCallToCompiledSpecialization(
                                                const Function::Overload& overload,
                                                const Function::CompiledSpecialization& specialization,
                                                const FunctionCallArgMapping& mapping
                                                );

    static PyObject* createOverloadPyRepresentation(Function* f);

    PyObject* tp_call_concrete(PyObject* args, PyObject* kwargs);

    static std::string argTupleTypeDescription(PyObject* self, PyObject* args, PyObject* kwargs);

    static void mirrorTypeInformationIntoPyTypeConcrete(Function* inType, PyTypeObject* pyType);

    int pyInquiryConcrete(const char* op, const char* opErrRep);

    static PyMethodDef* typeMethodsConcrete(Type* t);

    static PyObject* overload(PyObject* cls, PyObject* args, PyObject* kwargs);

    static PyObject* withEntrypoint(PyObject* funcObj, PyObject* args, PyObject* kwargs);

    static PyObject* resultTypeFor(PyObject* funcObj, PyObject* args, PyObject* kwargs);

    static Function* convertPythonObjectToFunction(PyObject* name, PyObject *funcObj);

};
