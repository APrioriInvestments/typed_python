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

#include "PyFunctionInstance.hpp"
#include "FunctionCallArgMapping.hpp"
#include "TypedClosureBuilder.hpp"

Function* PyFunctionInstance::type() {
    return (Function*)extractTypeFrom(((PyObject*)this)->ob_type);
}

// static
PyObject* PyFunctionInstance::prepareArgumentToBePassedToCompiler(PyObject* o) {
    TypedClosureBuilder builder;

    if (builder.isFunctionObject(o)) {
        return builder.convert(o);
    }

    return incref(o);
}

// static
std::pair<bool, PyObject*>
PyFunctionInstance::tryToCallAnyOverload(const Function* f, instance_ptr funcClosure, PyObject* self,
                                         PyObject* args, PyObject* kwargs) {
    //if we are an entrypoint, map any untyped function arguments to typed functions
    PyObjectHolder mappedArgs;
    PyObjectHolder mappedKwargs;

    if (f->isEntrypoint()) {
        mappedArgs.steal(PyTuple_New(PyTuple_Size(args)));

        for (long k = 0; k < PyTuple_Size(args); k++) {
            PyTuple_SetItem(mappedArgs, k, prepareArgumentToBePassedToCompiler(PyTuple_GetItem(args, k)));
        }

        mappedKwargs.steal(PyDict_New());

        if (kwargs) {
            PyObject *key, *value;
            Py_ssize_t pos = 0;

            while (PyDict_Next(kwargs, &pos, &key, &value)) {
                PyObjectStealer mapped(prepareArgumentToBePassedToCompiler(value));
                PyDict_SetItem(mappedKwargs, key, mapped);
            }
        }
    } else {
        mappedArgs.set(args);
        mappedKwargs.set(kwargs);
    }

    for (ConversionLevel conversionLevel: {
        ConversionLevel::Signature,
        ConversionLevel::Upcast,
        ConversionLevel::UpcastContainers,
        ConversionLevel::Implicit,
        ConversionLevel::ImplicitContainers
    }) {
        for (long overloadIx = 0; overloadIx < f->getOverloads().size(); overloadIx++) {
            std::pair<bool, PyObject*> res =
                PyFunctionInstance::tryToCallOverload(
                    f, funcClosure, overloadIx, self,
                    mappedArgs, mappedKwargs, conversionLevel
                );

            if (res.first) {
                return res;
            }
        }
    }

    std::string argTupleTypeDesc = PyFunctionInstance::argTupleTypeDescription(self, args, kwargs);

    PyErr_Format(
        PyExc_TypeError, "Cannot find a valid overload of '%s' with arguments of type %s",
        f->name().c_str(),
        argTupleTypeDesc.c_str()
        );

    return std::pair<bool, PyObject*>(false, nullptr);
}

// static
std::pair<bool, PyObject*> PyFunctionInstance::tryToCallOverload(
        const Function* f,
        instance_ptr functionClosure,
        long overloadIx,
        PyObject* self,
        PyObject* args,
        PyObject* kwargs,
        ConversionLevel conversionLevel
) {
    const Function::Overload& overload(f->getOverloads()[overloadIx]);

    FunctionCallArgMapping mapping(overload);

    mapping.pushArguments(self, args, kwargs);

    if (!mapping.isValid()) {
        return std::make_pair(false, nullptr);
    }

    //first, see if we can short-circuit without producing temporaries, which
    //can be slow.
    if (mapping.definitelyDoesntMatch(conversionLevel)) {
        return std::pair<bool, PyObject*>(false, (PyObject*)nullptr);
    }

    //perform argument coercion
    mapping.applyTypeCoercion(conversionLevel);

    if (!mapping.isValid()) {
        return std::make_pair(false, nullptr);
    }

    PyObjectHolder result;

    bool hadNativeDispatch = false;

    if (!native_dispatch_disabled) {
        auto tried_and_result = dispatchFunctionCallToNative(f, functionClosure, overloadIx, mapping);
        hadNativeDispatch = tried_and_result.first;
        result.steal(tried_and_result.second);
    }

    if (!hadNativeDispatch) {
        PyObjectStealer argTup(mapping.buildPositionalArgTuple());
        PyObjectStealer kwargD(mapping.buildKeywordArgTuple());
        PyObjectStealer funcObj(
            overload.buildFunctionObj(
                f->getClosureType(),
                functionClosure
            )
        );

        result.steal(PyObject_Call((PyObject*)funcObj, (PyObject*)argTup, (PyObject*)kwargD));
    }

    //exceptions pass through directly
    if (!result) {
        return std::make_pair(true, result);
    }

    // determine if we have a valid type-signature. If this is a method of a class,
    // we'll need to look up the chain and see whether there are any terms that match
    // this signature, and verify that all the signature types above us are valid.
    std::pair<Type*, bool> returnTypeAndIsException = determineReturnTypeForMatchedCall(
        f,
        overloadIx,
        mapping,
        self,
        args,
        kwargs
    );

    if (returnTypeAndIsException.second) {
        return std::make_pair(true, nullptr);
    }

    Type* returnType = returnTypeAndIsException.first;

    //force ourselves to convert to the returnType
    if (returnType) {
        try {
            PyObject* newRes = PyInstance::initializePythonRepresentation(returnType, [&](instance_ptr data) {
                copyConstructFromPythonInstance(returnType, data, result, ConversionLevel::ImplicitContainers);
            });

            return std::make_pair(true, newRes);
        } catch (std::exception& e) {
            PyErr_SetString(PyExc_TypeError, e.what());
            return std::make_pair(true, (PyObject*)nullptr);
        }
    }

    return std::make_pair(true, incref(result));
}

std::pair<Type*, bool> PyFunctionInstance::getOverloadReturnType(
    const Function* f,
    long overloadIx,
    FunctionCallArgMapping& matchedArgs
) {
    const Function::Overload& overload(f->getOverloads()[overloadIx]);
    Type* returnType = overload.getReturnType();

    if (overload.getSignatureFunction()) {
        // this function has a signature function. we need to invoke it to determine
        // what type we're going to be returning
        PyObjectHolder sigFuncReturnType;

        PyObjectStealer argTypeTup(matchedArgs.buildPositionalArgTuple(true /*types*/));
        PyObjectStealer kwargTypeDict(matchedArgs.buildKeywordArgTuple(true /*types*/));

        sigFuncReturnType.steal(
            PyObject_Call(overload.getSignatureFunction(), (PyObject*)argTypeTup, (PyObject*)kwargTypeDict)
        );

        if (!sigFuncReturnType) {
            // user code threw an exception
            return std::make_pair(nullptr, true);
        }

        returnType = PyInstance::unwrapTypeArgToTypePtr(sigFuncReturnType);

        if (!returnType) {
            // user code didn't return a proper type!
            return std::make_pair(nullptr, true);
        }
    }

    return std::make_pair(returnType, false);
}

std::pair<Type*, bool> PyFunctionInstance::determineReturnTypeForMatchedCall(
    const Function* f,
    long overloadIx,
    FunctionCallArgMapping& matchedArgs,
    PyObject* self,
    PyObject* args,
    PyObject* kwargs
) {
    std::pair<Type*, bool> returnTypeAndIsException = getOverloadReturnType(f, overloadIx, matchedArgs);

    if (returnTypeAndIsException.second) {
        return returnTypeAndIsException;
    }

    if (!f->getOverloads()[overloadIx].getMethodOf()) {
        return returnTypeAndIsException;
    }

    Type* returnType = returnTypeAndIsException.first;

    std::vector<std::pair<Type*, Type*> > returnTypeAndCls;

    // this is the method of a class. We need to check that all types in classes
    // above this are valid also. We will do that by grouping the overloads
    // into blocks based on the class they're a method of, and for each base class,
    // determining what return type it would have for this set of arguments.
    size_t overloadCount = f->getOverloads().size();

    auto methodFor = [&](int64_t ix) {
        return f->getOverloads()[ix].getMethodOf();
    };

    auto blockEndIx = [&](int64_t ix) {
        int64_t startIx = ix;
        while (ix < overloadCount && methodFor(ix) == methodFor(startIx)) {
            ix++;
        }
        return ix;
    };

    returnTypeAndCls.push_back(std::make_pair(returnType, methodFor(overloadIx)));

    int64_t nextOverloadIx = blockEndIx(overloadIx);

    while (nextOverloadIx < overloadCount) {
        int64_t topIx = blockEndIx(nextOverloadIx);

        bool anyMatched = false;

        // within this block of overloads, find the first overload
        // we would have matched
        for (ConversionLevel conversionLevel: {
            ConversionLevel::Signature,
            ConversionLevel::Upcast,
            ConversionLevel::UpcastContainers,
            ConversionLevel::Implicit,
            ConversionLevel::ImplicitContainers
        }) {
            if (!anyMatched) {
                for (long matchingIx = nextOverloadIx; matchingIx < topIx && !anyMatched; matchingIx++) {
                    FunctionCallArgMapping subMapping(f->getOverloads()[matchingIx]);

                    subMapping.pushArguments(self, args, kwargs);

                    if (!subMapping.definitelyDoesntMatch(conversionLevel)) {
                        subMapping.applyTypeCoercion(conversionLevel);

                        if (subMapping.isValid()) {
                            anyMatched = true;

                            std::pair<Type*, bool> baseClassReturnType = getOverloadReturnType(f, matchingIx, subMapping);

                            // if the signature function fails, we have to stop processing
                            if (baseClassReturnType.second) {
                                return baseClassReturnType;
                            }

                            returnTypeAndCls.push_back(std::make_pair(baseClassReturnType.first, methodFor(matchingIx)));
                        }
                    }
                }
            }
        }

        // roll to the next block
        nextOverloadIx = topIx;
    }

    // now we have a list of matched types. We need to ensure that they're progressively
    // tighter in reverse order.
    std::pair<Type*, Type*> actualReturnTypeAndCls;

    for (int64_t ix = (int64_t)returnTypeAndCls.size() - 1; ix >= 0; ix--) {
        if (!actualReturnTypeAndCls.first) {
            // if we don't have a stated return type yet, anything is OK
            actualReturnTypeAndCls = returnTypeAndCls[ix];
        } else
        if (returnTypeAndCls[ix].first) {
            if (!returnTypeAndCls[ix].first->canConvertToTrivially(actualReturnTypeAndCls.first)) {
                Type* baseType = actualReturnTypeAndCls.second;
                Type* childType = returnTypeAndCls[ix].second;

                if (baseType->isHeldClass()) {
                    baseType = ((HeldClass*)baseType)->getClassType();
                }

                if (childType->isHeldClass()) {
                    childType = ((HeldClass*)childType)->getClassType();
                }

                PyErr_Format(
                    PyExc_TypeError,
                    "Method %s.%s promised a return type of '%s', but subclass %s proposed to return '%s'. ",
                    baseType->name().c_str(),
                    f->name().c_str(),
                    actualReturnTypeAndCls.first->name().c_str(),
                    childType->name().c_str(),
                    returnTypeAndCls[ix].first->name().c_str()
                );
                return std::make_pair(nullptr, true);
            } else {
                actualReturnTypeAndCls = returnTypeAndCls[ix];
            }
        }
    }

    return std::make_pair(actualReturnTypeAndCls.first, false);
}

std::pair<bool, PyObject*> PyFunctionInstance::tryToCall(const Function* f, instance_ptr closure, PyObject* arg0, PyObject* arg1, PyObject* arg2) {
    PyObjectStealer argTuple(
        (arg0 && arg1 && arg2) ?
            PyTuple_Pack(3, arg0, arg1, arg2) :
        (arg0 && arg1) ?
            PyTuple_Pack(2, arg0, arg1) :
        arg0 ?
            PyTuple_Pack(1, arg0) :
            PyTuple_Pack(0)
        );
    return PyFunctionInstance::tryToCallAnyOverload(f, closure, nullptr, argTuple, nullptr);
}

// static
std::pair<bool, PyObject*> PyFunctionInstance::dispatchFunctionCallToNative(
        const Function* f,
        instance_ptr functionClosure,
        long overloadIx,
        const FunctionCallArgMapping& mapper
    ) {
    const Function::Overload& overload(f->getOverloads()[overloadIx]);

    for (const auto& spec: overload.getCompiledSpecializations()) {
        auto res = dispatchFunctionCallToCompiledSpecialization(
            overload,
            f->getClosureType(),
            functionClosure,
            spec,
            mapper
        );

        if (res.first) {
            return res;
        }
    }

    if (f->isEntrypoint()) {
        // package 'f' back up as an object and pass it to the closure-type-generator.
        // otherwise when we call this in the compiler, we'll have PyCell objects
        // instead of proper closures
        PyObjectStealer fAsObj(PyInstance::extractPythonObject(functionClosure, (Type*)f));
        PyObjectStealer fConvertedAsObj(prepareArgumentToBePassedToCompiler(fAsObj));

        // convert the object back to a function
        Type* convertedFType = PyInstance::extractTypeFrom(fConvertedAsObj->ob_type);

        if (!convertedFType || convertedFType->getTypeCategory() != Type::TypeCategory::catFunction) {
            throw std::runtime_error("prepareArgumentToBePassedToCompiler returned a non-function!");
        }

        Function* convertedF = (Function*)convertedFType;

        instance_ptr convertedFData = ((PyInstance*)(PyObject*)fConvertedAsObj)->dataPtr();

        // attempt a second dispatch. Some functions change every time they're called
        // because of what's held in their closures.

        // the overloadIx shouldn't change because 'prepareArgumentToBePassedToCompiler'
        // doesn't fundamentally change the nature of the function - it just picks
        // appropriate types for the closures.
        if (convertedF->getOverloads().size() != f->getOverloads().size()) {
            throw std::runtime_error("Somehow, the number of overloads in the function changed.");
        }

        const Function::Overload& overload2(convertedF->getOverloads()[overloadIx]);

        for (const auto& spec: overload2.getCompiledSpecializations()) {
            auto res = dispatchFunctionCallToCompiledSpecialization(
                overload2,
                convertedF->getClosureType(),
                convertedFData,
                spec,
                mapper
            );

            if (res.first) {
                return res;
            }
        }

        static PyObject* runtimeModule = ::runtimeModule();

        if (!runtimeModule) {
            throw std::runtime_error("Internal error: couldn't find typed_python.compiler.runtime");
        }

        PyObject* runtimeClass = PyObject_GetAttrString(runtimeModule, "Runtime");

        if (!runtimeClass) {
            throw std::runtime_error("Internal error: couldn't find typed_python.compiler.runtime.Runtime");
        }

        PyObject* singleton = PyObject_CallMethod(runtimeClass, "singleton", "");

        if (!singleton) {
            if (PyErr_Occurred()) {
                PyErr_Clear();
            }

            throw std::runtime_error("Internal error: couldn't call typed_python.compiler.runtime.Runtime.singleton");
        }

        PyObjectStealer arguments(mapper.extractFunctionArgumentValues());

        PyObject* res = PyObject_CallMethod(
            singleton,
            "compileFunctionOverload",
            "OlO",
            PyInstance::typePtrToPyTypeRepresentation((Type*)convertedF),
            overloadIx,
            (PyObject*)arguments
        );

        if (!res) {
            throw PythonExceptionSet();
        }

        decref(res);

        const Function::Overload& convertedOverload(convertedF->getOverloads()[overloadIx]);

        for (const auto& spec: convertedOverload.getCompiledSpecializations()) {
            auto res = dispatchFunctionCallToCompiledSpecialization(
                convertedOverload,
                convertedF->getClosureType(),
                convertedFData,
                spec,
                mapper
            );

            if (res.first) {
                return res;
            }
        }

        throw std::runtime_error(
            "Compiled but then failed to dispatch to one of "
             + format(convertedOverload.getCompiledSpecializations().size())
             + " specializations"
        );
    }

    return std::pair<bool, PyObject*>(false, (PyObject*)nullptr);
}

std::pair<bool, PyObject*> PyFunctionInstance::dispatchFunctionCallToCompiledSpecialization(
                                                        const Function::Overload& overload,
                                                        Type* closureType,
                                                        instance_ptr closureData,
                                                        const Function::CompiledSpecialization& specialization,
                                                        const FunctionCallArgMapping& mapper
                                                        ) {
    Type* returnType = specialization.getReturnType();

    if (!returnType) {
        throw std::runtime_error("Malformed function specialization: missing a return type.");
    }

    std::vector<Instance> instances;

    // first, see if we can short-circuit
    for (long k = 0; k < overload.getArgs().size(); k++) {
        auto arg = overload.getArgs()[k];
        if (arg.getIsNormalArg()) {
            Type* argType = specialization.getArgTypes()[k];

            if (!PyInstance::pyValCouldBeOfType(argType, mapper.getSingleValueArgs()[k], ConversionLevel::Signature)) {
                return std::pair<bool, PyObject*>(false, (PyObject*)nullptr);
            }
        }
    }

    for (long k = 0; k < overload.getArgs().size(); k++) {
        auto arg = overload.getArgs()[k];
        Type* argType = specialization.getArgTypes()[k];

        std::pair<Instance, bool> res = mapper.extractArgWithType(k, argType);

        if (res.second) {
            instances.push_back(res.first);
        } else {
            return std::pair<bool, PyObject*>(false, (PyObject*)nullptr);
        }
    }

    Instance result = Instance::createAndInitialize(returnType, [&](instance_ptr returnData) {
        std::vector<Instance> closureCells;

        std::vector<instance_ptr> args;

        // we pass each closure variable first (sorted lexically), then the actual function arguments.
        for (auto nameAndPath: overload.getClosureVariableBindings()) {
            closureCells.push_back(nameAndPath.second.extractValueOrContainingClosure(closureType, closureData));
            args.push_back(closureCells.back().data());
        }

        for (auto& i: instances) {
            args.push_back(i.data());
        }

        auto functionPtr = specialization.getFuncPtr();

        PyEnsureGilReleased releaseTheGIL;

        try {
            functionPtr(returnData, &args[0]);
        }
        catch(...) {
            // exceptions coming out of compiled code always use the python interpreter
            throw PythonExceptionSet();
        }
    });

    return std::pair<bool, PyObject*>(true, (PyObject*)extractPythonObject(result.data(), result.type()));
}


// static
PyObject* PyFunctionInstance::createOverloadPyRepresentation(Function* f) {
    static PyObject* internalsModule = ::internalsModule();

    if (!internalsModule) {
        throw std::runtime_error("Internal error: couldn't find typed_python.internals");
    }

    static PyObject* funcOverload = PyObject_GetAttrString(internalsModule, "FunctionOverload");

    if (!funcOverload) {
        throw std::runtime_error("Internal error: couldn't find typed_python.internals.FunctionOverload");
    }

    static PyObject* closureVariableCellLookupSingleton = PyObject_GetAttrString(internalsModule, "CellAccess");

    if (!closureVariableCellLookupSingleton) {
        throw std::runtime_error("Internal error: couldn't find typed_python.internals.CellAccess");
    }

    static PyObject* funcOverloadArg = PyObject_GetAttrString(internalsModule, "FunctionOverloadArg");

    if (!funcOverloadArg) {
        throw std::runtime_error("Internal error: couldn't find typed_python.internals.FunctionOverloadArg");
    }

    PyObjectStealer overloadTuple(PyTuple_New(f->getOverloads().size()));

    for (long k = 0; k < f->getOverloads().size(); k++) {
        auto& overload = f->getOverloads()[k];

        PyObjectStealer pyIndex(PyLong_FromLong(k));

        PyObjectStealer pyGlobalCellDict(PyDict_New());

        for (auto nameAndCell: overload.getFunctionGlobalsInCells()) {
            PyDict_SetItemString(pyGlobalCellDict, nameAndCell.first.c_str(), nameAndCell.second);
        }

        PyObjectStealer pyClosureVarsDict(PyDict_New());

        for (auto nameAndClosureVar: overload.getClosureVariableBindings()) {
            PyObjectStealer bindingObj(PyTuple_New(nameAndClosureVar.second.size()));

            for (long k = 0; k < nameAndClosureVar.second.size(); k++) {
                ClosureVariableBindingStep step = nameAndClosureVar.second[k];

                if (step.isFunction()) {
                    // recall that 'PyTuple_SetItem' steals a reference, so we need to incref it here
                    PyTuple_SetItem(bindingObj, k, incref((PyObject*)typePtrToPyTypeRepresentation(step.getFunction())));
                } else
                if (step.isNamedField()) {
                    PyTuple_SetItem(bindingObj, k, PyUnicode_FromString(step.getNamedField().c_str()));
                } else
                if (step.isIndexedField()) {
                    PyTuple_SetItem(bindingObj, k, PyLong_FromLong(step.getIndexedField()));
                } else
                if (step.isCellAccess()) {
                    PyTuple_SetItem(bindingObj, k, incref(closureVariableCellLookupSingleton));
                } else {
                    throw std::runtime_error("Corrupt ClosureVariableBindingStep encountered");
                }
            }

            PyDict_SetItemString(pyClosureVarsDict, nameAndClosureVar.first.c_str(), bindingObj);
        }

        PyObjectStealer emptyTup(PyTuple_New(0));
        PyObjectStealer emptyDict(PyDict_New());

        // note that we can't actually call into the Python interpreter during this call,
        // because that can release the GIL and allow other threads to access our type
        // object before it's done.
        PyObjectStealer argsTup(PyTuple_New(f->getOverloads()[k].getArgs().size()));

        for (long argIx = 0; argIx < f->getOverloads()[k].getArgs().size(); argIx++) {
            auto arg = f->getOverloads()[k].getArgs()[argIx];

            PyObjectStealer pyArgInst(
                ((PyTypeObject*)funcOverloadArg)->tp_new((PyTypeObject*)funcOverloadArg, emptyTup, emptyDict)
            );

            PyObjectStealer pyArgInstDict(PyObject_GenericGetDict(pyArgInst, nullptr));

            PyObjectStealer pyName(PyUnicode_FromString(arg.getName().c_str()));
            PyDict_SetItemString(pyArgInstDict, "name", pyName);
            PyDict_SetItemString(pyArgInstDict, "defaultValue", arg.getDefaultValue() ? PyTuple_Pack(1, arg.getDefaultValue()) : Py_None);
            PyDict_SetItemString(pyArgInstDict, "_typeFilter", arg.getTypeFilter() ? (PyObject*)typePtrToPyTypeRepresentation(arg.getTypeFilter()) : Py_None);
            PyDict_SetItemString(pyArgInstDict, "isStarArg", arg.getIsStarArg() ? Py_True : Py_False);
            PyDict_SetItemString(pyArgInstDict, "isKwarg", arg.getIsKwarg() ? Py_True : Py_False);

            PyTuple_SetItem(argsTup, argIx, incref(pyArgInst));
        }

        PyObjectStealer pyOverloadInst(
            ((PyTypeObject*)funcOverload)->tp_new((PyTypeObject*)funcOverload, emptyTup, emptyDict)
        );

        PyObjectStealer pyOverloadInstDict(PyObject_GenericGetDict(pyOverloadInst, nullptr));

        PyDict_SetItemString(pyOverloadInstDict, "functionTypeObject", typePtrToPyTypeRepresentation(f));
        PyDict_SetItemString(pyOverloadInstDict, "index", (PyObject*)pyIndex);
        PyDict_SetItemString(pyOverloadInstDict, "closureVarLookups", (PyObject*)pyClosureVarsDict);
        PyDict_SetItemString(pyOverloadInstDict, "functionCode", (PyObject*)overload.getFunctionCode());
        PyDict_SetItemString(pyOverloadInstDict, "funcGlobalsInCells", (PyObject*)pyGlobalCellDict);
        PyDict_SetItemString(pyOverloadInstDict, "returnType", overload.getReturnType() ? (PyObject*)typePtrToPyTypeRepresentation(overload.getReturnType()) : Py_None);
        PyDict_SetItemString(pyOverloadInstDict, "signatureFunction", overload.getSignatureFunction() ? (PyObject*)overload.getSignatureFunction() : Py_None);
        PyDict_SetItemString(pyOverloadInstDict, "methodOf", overload.getMethodOf() ? (PyObject*)typePtrToPyTypeRepresentation(overload.getMethodOf()) : Py_None);
        PyDict_SetItemString(pyOverloadInstDict, "_realizedGlobals", Py_None);
        PyDict_SetItemString(pyOverloadInstDict, "args", argsTup);

        PyTuple_SetItem(overloadTuple, k, incref(pyOverloadInst));
    }

    return incref(overloadTuple);
}

PyObject* PyFunctionInstance::tp_call_concrete(PyObject* args, PyObject* kwargs) {
    return PyFunctionInstance::tryToCallAnyOverload(type(), dataPtr(), nullptr, args, kwargs).second;
}

std::string PyFunctionInstance::argTupleTypeDescription(PyObject* self, PyObject* args, PyObject* kwargs) {
    std::ostringstream outTypes;
    outTypes << "(";
    bool first = true;

    if (self) {
        outTypes << self->ob_type->tp_name;
        first = false;
    }

    for (long k = 0; k < PyTuple_Size(args); k++) {
        if (!first) {
            outTypes << ",";
        } else {
            first = false;
        }
        outTypes << PyTuple_GetItem(args,k)->ob_type->tp_name;
    }
    if (kwargs) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;

        while (kwargs && PyDict_Next(kwargs, &pos, &key, &value)) {
            if (!first) {
                outTypes << ",";
            } else {
                first = false;
            }
            outTypes << PyUnicode_AsUTF8(key) << "=" << value->ob_type->tp_name;
        }
    }

    outTypes << ")";

    return outTypes.str();
}

void PyFunctionInstance::mirrorTypeInformationIntoPyTypeConcrete(Function* inType, PyTypeObject* pyType) {
    PyObjectHolder overloads(createOverloadPyRepresentation(inType));

    if (!overloads) {
        throw PythonExceptionSet();
    }

    PyDict_SetItemString(
        pyType->tp_dict,
        "__name__",
        PyUnicode_FromString(inType->name().c_str())
    );

    PyDict_SetItemString(
        pyType->tp_dict,
        "__qualname__",
        PyUnicode_FromString(inType->qualname().c_str())
    );

    PyDict_SetItemString(
        pyType->tp_dict,
        "__module__",
        PyUnicode_FromString(inType->moduleName().c_str())
    );

    PyDict_SetItemString(
        pyType->tp_dict,
        "overloads",
        overloads
    );

    PyObject* closureTypeObj = PyInstance::typePtrToPyTypeRepresentation(inType->getClosureType());

    if (closureTypeObj) {
        PyDict_SetItemString(
            pyType->tp_dict,
            "ClosureType",
            closureTypeObj
        );
    } else {
        throw std::runtime_error("Couldn't get a type object for the closure of " + inType->name());
    }

    PyDict_SetItemString(
        pyType->tp_dict,
        "isEntrypoint",
        inType->isEntrypoint() ? Py_True : Py_False
    );

    PyDict_SetItemString(
        pyType->tp_dict,
        "isNocompile",
        inType->isNocompile() ? Py_True : Py_False
    );
}

int PyFunctionInstance::pyInquiryConcrete(const char* op, const char* opErrRep) {
    // op == '__bool__'
    return 1;
}

/* static */
PyObject* PyFunctionInstance::extractPyFun(PyObject* funcObj, PyObject* args, PyObject* kwargs) {
    static const char *kwlist[] = {"overloadIx", NULL};

    long overloadIx;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "l", (char**)kwlist, &overloadIx)) {
        return nullptr;
    }

    Function* fType = (Function*)((PyInstance*)(funcObj))->type();

    if (overloadIx < 0 || overloadIx >= fType->getOverloads().size()) {
        PyErr_SetString(PyExc_IndexError, "Overload index out of bounds");
        return nullptr;
    }

    return translateExceptionToPyObject([&]() {
        return fType->getOverloads()[overloadIx].buildFunctionObj(
            fType->getClosureType(),
            ((PyInstance*)funcObj)->dataPtr()
        );
    });
}

/* static */
PyObject* PyFunctionInstance::extractOverloadGlobals(PyObject* cls, PyObject* args, PyObject* kwargs) {
    Type* selfType = PyInstance::unwrapTypeArgToTypePtr(cls);

    if (!selfType || selfType->getTypeCategory() != Type::TypeCategory::catFunction) {
        PyErr_Format(PyExc_TypeError, "Expected class to be a Function");
        return nullptr;
    }

    Function* fType = (Function*)selfType;

    static const char *kwlist[] = {"overloadIx", NULL};

    long overloadIx;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "l", (char**)kwlist, &overloadIx)) {
        return nullptr;
    }

    if (overloadIx < 0 || overloadIx >= fType->getOverloads().size()) {
        PyErr_SetString(PyExc_IndexError, "Overload index out of bounds");
        return nullptr;
    }

    return translateExceptionToPyObject([&]() {
        return incref(fType->getOverloads()[overloadIx].getFunctionGlobals());
    });
}

/* static */
PyObject* PyFunctionInstance::getClosure(PyObject* funcObj, PyObject* args, PyObject* kwargs) {
    static const char *kwlist[] = {NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
        return nullptr;
    }

    Function* fType = (Function*)((PyInstance*)(funcObj))->type();

    return PyInstance::extractPythonObject(((PyInstance*)funcObj)->dataPtr(), fType->getClosureType());
}

/* static */
PyObject* PyFunctionInstance::typeWithEntrypoint(PyObject* cls, PyObject* args, PyObject* kwargs) {
    Type* selfType = PyInstance::unwrapTypeArgToTypePtr(cls);

    if (!selfType || selfType->getTypeCategory() != Type::TypeCategory::catFunction) {
        PyErr_Format(PyExc_TypeError, "Expected class to be a Function");
        return nullptr;
    }

    static const char *kwlist[] = {"isEntrypoint", NULL};
    int isWithEntrypoint;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "p", (char**)kwlist, &isWithEntrypoint)) {
        return nullptr;
    }

    return PyInstance::typePtrToPyTypeRepresentation(
        ((Function*)selfType)->withEntrypoint(isWithEntrypoint)
    );
}

/* static */
PyObject* PyFunctionInstance::typeWithNocompile(PyObject* cls, PyObject* args, PyObject* kwargs) {
    Type* selfType = PyInstance::unwrapTypeArgToTypePtr(cls);

    if (!selfType || selfType->getTypeCategory() != Type::TypeCategory::catFunction) {
        PyErr_Format(PyExc_TypeError, "Expected class to be a Function");
        return nullptr;
    }

    static const char *kwlist[] = {"isNocompile", NULL};
    int isWithNocompile;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "p", (char**)kwlist, &isWithNocompile)) {
        return nullptr;
    }

    return PyInstance::typePtrToPyTypeRepresentation(
        ((Function*)selfType)->withNocompile(isWithNocompile)
    );
}

/* static */
PyObject* PyFunctionInstance::withEntrypoint(PyObject* funcObj, PyObject* args, PyObject* kwargs) {
    static const char *kwlist[] = {"isEntrypoint", NULL};
    int isWithEntrypoint;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "p", (char**)kwlist, &isWithEntrypoint)) {
        return nullptr;
    }

    Function* resType = (Function*)((PyInstance*)(funcObj))->type();

    resType = resType->withEntrypoint(isWithEntrypoint);

    return PyInstance::extractPythonObject(((PyInstance*)funcObj)->dataPtr(), resType);
}

/* static */
PyObject* PyFunctionInstance::withNocompile(PyObject* funcObj, PyObject* args, PyObject* kwargs) {
    static const char *kwlist[] = {"isNocompile", NULL};
    int isNocompile;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "p", (char**)kwlist, &isNocompile)) {
        return nullptr;
    }

    Function* resType = (Function*)((PyInstance*)(funcObj))->type();

    resType = resType->withNocompile(isNocompile);

    return PyInstance::extractPythonObject(((PyInstance*)funcObj)->dataPtr(), resType);
}

/* static */
PyObject* PyFunctionInstance::overload(PyObject* funcObj, PyObject* args, PyObject* kwargs) {
    return translateExceptionToPyObject([&]() {
        if (kwargs && PyDict_Size(kwargs)) {
            throw std::runtime_error("Can't call 'overload' with kwargs");
        }

        if (PyTuple_Size(args) != 1) {
            throw std::runtime_error("'overload' expects one argument");
        }

        Function* ownType = (Function*)((PyInstance*)(funcObj))->type();
        instance_ptr ownClosure = ((PyInstance*)(funcObj))->dataPtr();

        if (!ownType || ownType->getTypeCategory() != Type::TypeCategory::catFunction) {
            throw std::runtime_error("Expected 'cls' to be a Function.");
        }

        PyObject* arg = PyTuple_GetItem(args, 0);

        Type* argT = PyInstance::extractTypeFrom(arg->ob_type);
        Instance otherFuncAsInstance;

        Function* otherType;
        instance_ptr otherClosure;

        if (!argT) {
            argT = PyInstance::unwrapTypeArgToTypePtr(arg);

            if (!argT && PyFunction_Check(arg)) {
                // unwrapTypeArgToTypePtr sets an exception if it can't convert.
                // we want to clear it so we can try the unwrapping directly.
                PyErr_Clear();

                PyObjectStealer name(PyObject_GetAttrString(arg, "__name__"));
                if (!name) {
                    throw PythonExceptionSet();
                }

                argT = convertPythonObjectToFunctionType(name, arg, false, false);
            }

            if (!argT) {
                throw PythonExceptionSet();
            }

            otherFuncAsInstance = Instance::createAndInitialize(
                argT,
                [&](instance_ptr ptr) {
                    PyInstance::copyConstructFromPythonInstance(argT, ptr, arg, ConversionLevel::New);
                }
            );

            otherType = (Function*)argT;
            otherClosure = otherFuncAsInstance.data();
        } else {
            if (argT->getTypeCategory() != Type::TypeCategory::catFunction) {
                throw std::runtime_error("'overload' requires arguments to be Function types");
            }
            otherType = (Function*)argT;
            otherClosure = ((PyInstance*)arg)->dataPtr();
        }

        Function* mergedType = Function::merge(ownType, otherType);

        // closures are packed in
        return PyInstance::initialize(mergedType, [&](instance_ptr p) {
            ownType->getClosureType()->copy_constructor(p, ownClosure);
            otherType->getClosureType()->copy_constructor(
                p + ownType->getClosureType()->bytecount(),
                otherClosure
            );
        });
    });
}

/* static */
PyObject* PyFunctionInstance::resultTypeFor(PyObject* funcObj, PyObject* args, PyObject* kwargs) {
    static PyObject* runtimeModule = ::runtimeModule();

    if (!runtimeModule) {
        throw std::runtime_error("Internal error: couldn't find typed_python.compiler.runtime");
    }

    static PyObject* runtimeClass = PyObject_GetAttrString(runtimeModule, "Runtime");

    if (!runtimeClass) {
        throw std::runtime_error("Internal error: couldn't find typed_python.compiler.runtime.Runtime");
    }

    static PyObject* singleton = PyObject_CallMethod(runtimeClass, "singleton", "");

    if (!singleton) {
        if (PyErr_Occurred()) {
            PyErr_Clear();
        }

        throw std::runtime_error("Internal error: couldn't call typed_python.compiler.runtime.Runtime.singleton");
    }

    if (!kwargs) {
        static PyObject* emptyDict = PyDict_New();
        kwargs = emptyDict;
    }

    return PyObject_CallMethod(
        singleton,
        "resultTypeForCall",
        "OOO",
        funcObj,
        args,
        kwargs
    );
}

/* static */
PyMethodDef* PyFunctionInstance::typeMethodsConcrete(Type* t) {
    return new PyMethodDef[12] {
        {"overload", (PyCFunction)PyFunctionInstance::overload, METH_VARARGS | METH_KEYWORDS, NULL},
        {"withEntrypoint", (PyCFunction)PyFunctionInstance::withEntrypoint, METH_VARARGS | METH_KEYWORDS, NULL},
        {"typeWithEntrypoint", (PyCFunction)PyFunctionInstance::typeWithEntrypoint, METH_VARARGS | METH_KEYWORDS | METH_CLASS, NULL},
        {"withNocompile", (PyCFunction)PyFunctionInstance::withNocompile, METH_VARARGS | METH_KEYWORDS, NULL},
        {"typeWithNocompile", (PyCFunction)PyFunctionInstance::typeWithNocompile, METH_VARARGS | METH_KEYWORDS | METH_CLASS, NULL},
        {"resultTypeFor", (PyCFunction)PyFunctionInstance::resultTypeFor, METH_VARARGS | METH_KEYWORDS, NULL},
        {"extractPyFun", (PyCFunction)PyFunctionInstance::extractPyFun, METH_VARARGS | METH_KEYWORDS, NULL},
        {"extractOverloadGlobals", (PyCFunction)PyFunctionInstance::extractOverloadGlobals, METH_VARARGS | METH_KEYWORDS | METH_CLASS, NULL},
        {"getClosure", (PyCFunction)PyFunctionInstance::getClosure, METH_VARARGS | METH_KEYWORDS, NULL},
        {"withClosureType", (PyCFunction)PyFunctionInstance::withClosureType, METH_VARARGS | METH_KEYWORDS | METH_CLASS, NULL},
        {"withOverloadVariableBindings", (PyCFunction)PyFunctionInstance::withOverloadVariableBindings, METH_VARARGS | METH_KEYWORDS | METH_CLASS, NULL},
        {NULL, NULL}
    };
}

PyObject* PyFunctionInstance::withClosureType(PyObject* cls, PyObject* args, PyObject* kwargs) {
    static const char *kwlist[] = {"newType", NULL};

    PyObject* newType;

    if (!PyArg_ParseTupleAndKeywords(args, NULL, "O", (char**)kwlist, &newType)) {
        return nullptr;
    }

    Type* newTypeAsType = PyInstance::unwrapTypeArgToTypePtr(newType);

    if (!newTypeAsType) {
        PyErr_Format(PyExc_TypeError, "Expected a typed-python Type");
        return nullptr;
    }

    Type* selfType = PyInstance::unwrapTypeArgToTypePtr(cls);

    if (!selfType || selfType->getTypeCategory() != Type::TypeCategory::catFunction) {
        PyErr_Format(PyExc_TypeError, "Expected class to be a Function");
        return nullptr;
    }

    Function* fType = (Function*)selfType;

    return PyInstance::typePtrToPyTypeRepresentation(fType->replaceClosure(newTypeAsType));
}

PyObject* PyFunctionInstance::withOverloadVariableBindings(PyObject* cls, PyObject* args, PyObject* kwargs) {
    static const char *kwlist[] = {"overloadIx", "closureVarBindings", NULL};

    PyObject* pyBindingDict;
    long overloadIx;

    if (!PyArg_ParseTupleAndKeywords(args, NULL, "lO", (char**)kwlist, &overloadIx, &pyBindingDict)) {
        return nullptr;
    }

    Type* selfType = PyInstance::unwrapTypeArgToTypePtr(cls);

    if (!selfType || selfType->getTypeCategory() != Type::TypeCategory::catFunction) {
        PyErr_Format(PyExc_TypeError, "Expected class to be a Function");
        return nullptr;
    }

    Function* fType = (Function*)selfType;

    if (!PyDict_Check(pyBindingDict)) {
        PyErr_Format(PyExc_TypeError, "Expected 'closureVarBindings' to be a dict");
        return nullptr;
    }

    PyObject *key, *value;
    Py_ssize_t pos = 0;

    std::map<std::string, ClosureVariableBinding> bindingDict;

    return translateExceptionToPyObject([&]() {
        while (PyDict_Next(pyBindingDict, &pos, &key, &value)) {
            if (!PyUnicode_Check(key)) {
                PyErr_SetString(PyExc_TypeError, "closureVarBindings keys are supposed to be strings.");
                return (PyObject*)nullptr;
            }

            ClosureVariableBinding binding;

            iterate(value, [&](PyObject* step) {
                if (PyLong_Check(step)) {
                    binding = binding + ClosureVariableBindingStep(PyLong_AsLong(step));
                } else
                if (PyUnicode_Check(step)) {
                    binding = binding + ClosureVariableBindingStep(std::string(PyUnicode_AsUTF8(step)));
                } else
                if (PyType_Check(step) && ::strcmp(((PyTypeObject*)step)->tp_name, "CellAccess") == 0) {
                    binding = binding + ClosureVariableBindingStep::AccessCell();
                } else {
                    Type* t = PyInstance::unwrapTypeArgToTypePtr(step);
                    if (t) {
                        binding = binding + ClosureVariableBindingStep(t);
                    } else {
                        throw std::runtime_error("Invalid argument to closureVarBindings.");
                    }
                }
            });

            bindingDict[PyUnicode_AsUTF8(key)] = binding;
        }

        return PyInstance::typePtrToPyTypeRepresentation(
            fType->replaceOverloadVariableBindings(overloadIx, bindingDict)
        );
    });
}

Function* PyFunctionInstance::convertPythonObjectToFunctionType(
    PyObject* name,
    PyObject *funcObj,
    bool assumeClosuresGlobal,
    bool ignoreAnnotations
) {
    typedef std::tuple<PyObject*, bool, bool> key_type;

    key_type memoKey(funcObj, assumeClosuresGlobal, ignoreAnnotations);

    static std::map<key_type, Function*> memo;

    auto memo_it = memo.find(memoKey);

    if (memo_it != memo.end()) {
        return memo_it->second;
    }

    static PyObject* internalsModule = ::internalsModule();

    if (!internalsModule) {
        PyErr_SetString(PyExc_TypeError, "Internal error: couldn't find typed_python.internals");
        return nullptr;
    }

    static PyObject* makeFunctionType = PyObject_GetAttrString(internalsModule, "makeFunctionType");

    if (!makeFunctionType) {
        PyErr_SetString(PyExc_TypeError, "Internal error: couldn't find typed_python.internals.makeFunctionType");
        return nullptr;
    }

    PyObjectStealer args(PyTuple_Pack(2, name, funcObj));
    PyObjectStealer kwargs(PyDict_New());

    if (assumeClosuresGlobal) {
        PyDict_SetItemString(kwargs, "assumeClosuresGlobal", Py_True);
    }

    if (ignoreAnnotations) {
        PyDict_SetItemString(kwargs, "ignoreAnnotations", Py_True);
    }

    PyObject* fRes = PyObject_Call(makeFunctionType, args, kwargs);

    if (!fRes) {
        return nullptr;
    }

    if (!PyType_Check(fRes)) {
        PyErr_SetString(PyExc_TypeError, "Internal error: expected typed_python.internals.makeFunctionType to return a type");
        return nullptr;
    }

    Type* actualType = PyInstance::extractTypeFrom((PyTypeObject*)fRes);

    if (!actualType || actualType->getTypeCategory() != Type::TypeCategory::catFunction) {
        PyErr_Format(PyExc_TypeError, "Internal error: expected makeFunctionType to return a Function. Got %S", fRes);
        return nullptr;
    }

    // if the closures can be assumed global, then its OK to put this in a memo,
    // because this is a 'typelike' function (its in an Alternative or Class)
    // if not, then the memo will just bloat because we'll have different function
    // objects for each possible type.
    if (assumeClosuresGlobal) {
        //make sure this memo stays valid.
        incref(funcObj);

        memo[memoKey] = (Function*)actualType;
    }

    return (Function*)actualType;
}

/* static */
bool PyFunctionInstance::pyValCouldBeOfTypeConcrete(Function* type, PyObject* pyRepresentation, ConversionLevel level) {
    if (!PyFunction_Check(pyRepresentation)) {
        return false;
    }

    if (type->getOverloads().size() != 1) {
        return false;
    }

    return type->getOverloads()[0].getFunctionCode() == PyFunction_GetCode(pyRepresentation);
}

/* static */
void PyFunctionInstance::copyConstructFromPythonInstanceConcrete(Function* type, instance_ptr tgt, PyObject* pyRepresentation, ConversionLevel level) {
    if (level < ConversionLevel::New) {
        throw std::runtime_error("Can't convert to " + type->name() + " with " + format(conversionLevelToInt(level)));
    }

    // see if we have just been handed the closure.
    if (pyRepresentation->ob_type == PyInstance::typeObj(type->getClosureType())) {
        type->getClosureType()->copy_constructor(tgt, ((PyInstance*)pyRepresentation)->dataPtr());
        return;
    }

    if (!pyValCouldBeOfTypeConcrete(type, pyRepresentation, level)) {
        throw std::runtime_error("Can't convert to " + type->name() + " pyrep");
    }

    if (!type->getClosureType()->isTuple()) {
        throw std::runtime_error("expected untyped closures to be Tuples");
    }

    Tuple* containingClosureType = (Tuple*)type->getClosureType();

    if (containingClosureType->bytecount() == 0) {
        // there's nothing to do.
        return;
    }

    if (containingClosureType->getTypes().size() != 1 || !containingClosureType->getTypes()[0]->isNamedTuple()) {
        throw std::runtime_error("expected a single overload in the untyped closure");
    }

    NamedTuple* closureType = (NamedTuple*)containingClosureType->getTypes()[0];

    PyObject* pyClosure = PyFunction_GetClosure(pyRepresentation);

    if (!pyClosure || !PyTuple_Check(pyClosure) || PyTuple_Size(pyClosure) != closureType->getTypes().size()) {
        throw std::runtime_error("Expected the pyClosure to have " + format(closureType->getTypes().size()) + " cells.");
    }

    closureType->constructor(tgt, [&](instance_ptr tgtCell, int index) {
        Type* closureTypeInst = closureType->getTypes()[index];

        PyObject* cell = PyTuple_GetItem(pyClosure, index);
        if (!cell) {
            throw PythonExceptionSet();
        }

        if (!PyCell_Check(cell)) {
            throw std::runtime_error("Expected function closure to be made up of cells.");
        }

        if (closureTypeInst->getTypeCategory() == Type::TypeCategory::catPyCell) {
            // our representation in the closure is itself a PyCell, so we just reference
            // the actual cell object.
            static PyCellType* pct = PyCellType::Make();
            pct->initializeFromPyObject(tgtCell, cell);
        } else {
            if (!PyCell_GET(cell)) {
                throw std::runtime_error("Cell for " + closureType->getNames()[index] + " was empty.");
            }

            PyInstance::copyConstructFromPythonInstance(
                closureType->getTypes()[index],
                tgtCell,
                PyCell_GET(cell),
                ConversionLevel::Implicit
            );
        }
    });
}

/* static */
std::map<
    //unresolved functions and closure variables
    std::pair<std::map<Path, Function*>, std::map<Path, Type*> >,
    //the closure type and the resolved function types
    std::tuple<Type*, std::map<Path, size_t>, std::map<Path, Type*> >
> TypedClosureBuilder::sResolvedTypes;
