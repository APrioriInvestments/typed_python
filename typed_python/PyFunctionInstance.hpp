#pragma once

#include "PyInstance.hpp"

class PyFunctionInstance : public PyInstance {
public:
    typedef Function modeled_type;

    Function* type();

    static std::pair<bool, PyObject*> tryToCallOverload(const Function::Overload& f, PyObject* self, PyObject* args, PyObject* kwargs);

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
};
