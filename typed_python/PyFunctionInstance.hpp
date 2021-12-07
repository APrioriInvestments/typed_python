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

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, ConversionLevel level);

    static void copyConstructFromPythonInstanceConcrete(modeled_type* type, instance_ptr tgt, PyObject* pyRepresentation, ConversionLevel level);

    static std::pair<bool, PyObject*> tryToCall(const Function* f, instance_ptr functionClosure, PyObject* arg0=nullptr, PyObject* arg1=nullptr, PyObject* arg2=nullptr);

    static std::pair<bool, PyObject*> tryToCallAnyOverload(const Function* f, instance_ptr functionClosure, PyObject* self, PyObject* args, PyObject* kwargs);

    // determine the exact return type of a specific overload. Returns <result, isException>
    static std::pair<Type*, bool> getOverloadReturnType(
        const Function* f,
        long overloadIx,
        FunctionCallArgMapping& matchedArgs
    );

    // Given that we have matched a collection of arguments, what return type are we?
    // this calls signature functions, and checks for return type consistency.
    // Returns <result, isException>
    static std::pair<Type*, bool> determineReturnTypeForMatchedCall(
        const Function* f,
        long overloadIx,
        FunctionCallArgMapping& matchedArgs,
        PyObject* self,
        PyObject* args,
        PyObject* kwargs
    );

    // determine the 'compiler type' of an argument 'o'. If 'o' is already the right type, just
    // use that. But for untyped function objects, and for Function objects with interpreter
    // closures, we attempt to walk the closure graph and build a better type signature.
    // returns a new reference to an object.
    static PyObject* prepareArgumentToBePassedToCompiler(PyObject* o);

    static std::pair<bool, PyObject*> tryToCallOverload(
        const Function* f,
        instance_ptr funcClosure,
        long overloadIx,
        PyObject* self,
        PyObject* args,
        PyObject* kwargs,
        ConversionLevel conversionLevel
    );

    //perform a linear scan of all specializations contained in overload and attempt to dispatch to each one.
    //returns <true, result or none> if we dispatched..
    //if 'isEntrypoint', then if we don't match a compiled specialization, ask the runtime to produce
    //one for us.
    static std::pair<bool, PyObject*> dispatchFunctionCallToNative(
        const Function* f,
        instance_ptr functionClosure,
        long overloadIx,
        const FunctionCallArgMapping& mapping
    );

    //attempt to dispatch to this one exact specialization by converting each arg to the relevant type. if
    //we can't convert, then return <false, nullptr>. If we do dispatch, return <true, result or none> and set
    //the python exception if native code returns an exception.
    static std::pair<bool, PyObject*> dispatchFunctionCallToCompiledSpecialization(
        const Function::Overload& overload,
        Type* closureType,
        instance_ptr closureData,
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

    static PyObject* getClosure(PyObject* funcObj, PyObject* args, PyObject* kwargs);

    static PyObject* extractPyFun(PyObject* funcObj, PyObject* args, PyObject* kwargs);

    static PyObject* extractOverloadGlobals(PyObject* funcObj, PyObject* args, PyObject* kwargs);

    static PyObject* typeWithEntrypoint(PyObject* cls, PyObject* args, PyObject* kwargs);

    static PyObject* typeWithNocompile(PyObject* cls, PyObject* args, PyObject* kwargs);

    static PyObject* withEntrypoint(PyObject* funcObj, PyObject* args, PyObject* kwargs);

    static PyObject* withNocompile(PyObject* funcObj, PyObject* args, PyObject* kwargs);

    static PyObject* resultTypeFor(PyObject* funcObj, PyObject* args, PyObject* kwargs);

    static PyObject* withClosureType(PyObject* cls, PyObject* args, PyObject* kwargs);

    static PyObject* withOverloadVariableBindings(PyObject* cls, PyObject* args, PyObject* kwargs);

    static Function* convertPythonObjectToFunctionType(
        PyObject* name,
        PyObject *funcObj,
        bool assumeClosuresGlobal, // if true, then place closures in the function type itself
                                   // this is appropriate if this function is a method of a class
        bool ignoreAnnotations // if true, then the resulting function has no explicit annotations
    );

};


class Path {
public:
    Path() {
        mPathVals.reset(new std::vector<int>());
    }

    Path(int i) {
        mPathVals.reset(new std::vector<int>());
        mPathVals->push_back(i);
    }

    Path(const Path& parent, int i) {
        mPathVals.reset(new std::vector<int>(*parent.mPathVals));
        mPathVals->push_back(i);
    }

    Path(const Path& other) : mPathVals(other.mPathVals)
    {
    }

    Path& operator=(const Path& other) {
        mPathVals = other.mPathVals;
        return *this;
    }

    bool operator<(const Path& other) const {
        return *mPathVals < *other.mPathVals;
    }

    bool operator==(const Path& other) const {
        return *mPathVals == *other.mPathVals;
    }

    Path operator+(int val) {
        return Path(*this, val);
    }

    std::string toString() const {
        std::ostringstream out;

        out << "Path(";
        for (long k = 0; k < mPathVals->size(); k++) {
            if (k) {
                out << ", ";
            }
            out << (*mPathVals)[k];
        }
        out << ")";

        return out.str();
    }

private:
    std::shared_ptr<std::vector<int> > mPathVals;
};
