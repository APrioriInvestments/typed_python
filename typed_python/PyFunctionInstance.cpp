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

Function* PyFunctionInstance::type() {
    return (Function*)extractTypeFrom(((PyObject*)this)->ob_type);
}

// static
std::pair<bool, PyObject*>
PyFunctionInstance::tryToCallAnyOverload(const Function* f, PyObject* self,
                                         PyObject* args, PyObject* kwargs) {
    //first try to match arguments with no explicit conversion.
    //if that fails, try explicit conversion
    for (long tryToConvertExplicitly = 0; tryToConvertExplicitly <= 1; tryToConvertExplicitly++) {
        for (long overloadIx = 0; overloadIx < f->getOverloads().size(); overloadIx++) {
            std::pair<bool, PyObject*> res =
                PyFunctionInstance::tryToCallOverload(f, overloadIx, self, args, kwargs, tryToConvertExplicitly, false);
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
        long overloadIx,
        PyObject* self,
        PyObject* args,
        PyObject* kwargs,
        bool convertExplicitly,
        bool dontActuallyCall
) {
    const Function::Overload& overload(f->getOverloads()[overloadIx]);

    FunctionCallArgMapping mapping(overload);

    if (self) {
        mapping.pushPositionalArg(self);
    }

    for (long k = 0; k < PyTuple_Size(args); k++) {
        mapping.pushPositionalArg(PyTuple_GetItem(args, k));
    }

    if (kwargs) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;

        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if (!PyUnicode_Check(key)) {
                PyErr_SetString(PyExc_TypeError, "Keywords arguments must be strings.");
                return std::make_pair(true, nullptr);
            }

            mapping.pushKeywordArg(PyUnicode_AsUTF8(key), value);
        }
    }

    mapping.finishedPushing();

    if (!mapping.isValid()) {
        return std::make_pair(false, nullptr);
    }

    //first, see if we can short-circuit without producing temporaries, which
    //can be slow.
    for (long k = 0; k < overload.getArgs().size(); k++) {
        auto arg = overload.getArgs()[k];

        if (arg.getIsNormalArg() && arg.getTypeFilter()) {
            if (!PyInstance::pyValCouldBeOfType(arg.getTypeFilter(), mapping.getSingleValueArgs()[k], convertExplicitly)) {
                return std::pair<bool, PyObject*>(false, (PyObject*)nullptr);
            }
        }
    }

    //perform argument coercion
    mapping.applyTypeCoercion(convertExplicitly);

    if (!mapping.isValid()) {
        return std::make_pair(false, nullptr);
    }

    // pathway to let us test which form we'd call without actually dispatching.
    if (dontActuallyCall) {
        return std::make_pair(true, nullptr);
    }

    PyObjectHolder result;

    bool hadNativeDispatch = false;

    if (!native_dispatch_disabled) {
        auto tried_and_result = dispatchFunctionCallToNative(f, overloadIx, mapping);
        hadNativeDispatch = tried_and_result.first;
        result.steal(tried_and_result.second);
    }

    if (!hadNativeDispatch) {
        PyObjectStealer argTup(mapping.buildPositionalArgTuple());
        PyObjectStealer kwargD(mapping.buildKeywordArgTuple());

        result.steal(PyObject_Call((PyObject*)overload.getFunctionObj(), (PyObject*)argTup, (PyObject*)kwargD));
    }

    //exceptions pass through directly
    if (!result) {
        return std::make_pair(true, result);
    }

    //force ourselves to convert to the native type
    if (overload.getReturnType()) {
        try {
            PyObject* newRes = PyInstance::initializePythonRepresentation(overload.getReturnType(), [&](instance_ptr data) {
                copyConstructFromPythonInstance(overload.getReturnType(), data, result, true);
            });

            return std::make_pair(true, newRes);
        } catch (std::exception& e) {
            PyErr_SetString(PyExc_TypeError, e.what());
            return std::make_pair(true, (PyObject*)nullptr);
        }
    }

    return std::make_pair(true, incref(result));
}

std::pair<bool, PyObject*> PyFunctionInstance::tryToCall(const Function* f, PyObject* arg0, PyObject* arg1, PyObject* arg2) {
    PyObjectStealer argTuple(
        (arg0 && arg1 && arg2) ?
            PyTuple_Pack(3, arg0, arg1, arg2) :
        (arg0 && arg1) ?
            PyTuple_Pack(2, arg0, arg1) :
        arg0 ?
            PyTuple_Pack(1, arg0) :
            PyTuple_Pack(0)
        );
    return PyFunctionInstance::tryToCallAnyOverload(f, nullptr, argTuple, nullptr);
}

// static
std::pair<bool, PyObject*> PyFunctionInstance::dispatchFunctionCallToNative(const Function* f, long overloadIx, const FunctionCallArgMapping& mapper) {
    const Function::Overload& overload(f->getOverloads()[overloadIx]);

    for (const auto& spec: overload.getCompiledSpecializations()) {
        auto res = dispatchFunctionCallToCompiledSpecialization(overload, spec, mapper);
        if (res.first) {
            return res;
        }
    }

    if (f->isEntrypoint()) {
        static PyObject* runtimeModule = PyImport_ImportModule("typed_python.compiler.runtime");

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
        PyObjectStealer funcAsObj(PyInstance::initialize((Type*)f, [&](instance_ptr p) {}));

        PyObject* res = PyObject_CallMethod(
            singleton,
            "compileFunctionOverload",
            "OlO",
            (PyObject*)funcAsObj,
            overloadIx,
            (PyObject*)arguments
        );

        if (!res) {
            throw PythonExceptionSet();
        }

        decref(res);

        for (const auto& spec: overload.getCompiledSpecializations()) {
            auto res = dispatchFunctionCallToCompiledSpecialization(overload, spec, mapper);
            if (res.first) {
                return res;
            }
        }

        throw std::runtime_error("Compiled but then failed to dispatch!");
    }

    return std::pair<bool, PyObject*>(false, (PyObject*)nullptr);
}

std::pair<bool, PyObject*> PyFunctionInstance::dispatchFunctionCallToCompiledSpecialization(
                                                        const Function::Overload& overload,
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

            if (!PyInstance::pyValCouldBeOfType(argType, mapper.getSingleValueArgs()[k], false)) {
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

    try {
        Instance result = Instance::createAndInitialize(returnType, [&](instance_ptr returnData) {
            std::vector<instance_ptr> args;
            for (auto& i: instances) {
                args.push_back(i.data());
            }

            auto functionPtr = specialization.getFuncPtr();

            PyEnsureGilReleased releaseTheGIL;

            functionPtr(returnData, &args[0]);
        });

        return std::pair<bool, PyObject*>(true, (PyObject*)extractPythonObject(result.data(), result.type()));
    }
    catch(...) {
        // exceptions coming out of compiled code always use the python interpreter
        throw PythonExceptionSet();
    }
}


// static
PyObject* PyFunctionInstance::createOverloadPyRepresentation(Function* f) {
    static PyObject* internalsModule = PyImport_ImportModule("typed_python.internals");

    if (!internalsModule) {
        throw std::runtime_error("Internal error: couldn't find typed_python.internals");
    }

    static PyObject* funcOverload = PyObject_GetAttrString(internalsModule, "FunctionOverload");

    if (!funcOverload) {
        throw std::runtime_error("Internal error: couldn't find typed_python.internals.FunctionOverload");
    }

    PyObjectStealer overloadTuple(PyTuple_New(f->getOverloads().size()));

    for (long k = 0; k < f->getOverloads().size(); k++) {
        auto& overload = f->getOverloads()[k];

        PyObjectStealer pyIndex(PyLong_FromLong(k));

        PyObjectStealer pyOverloadInst(
            PyObject_CallFunctionObjArgs(
                funcOverload,
                typePtrToPyTypeRepresentation(f),
                (PyObject*)pyIndex,
                (PyObject*)overload.getFunctionObj(),
                overload.getReturnType() ? (PyObject*)typePtrToPyTypeRepresentation(overload.getReturnType()) : Py_None,
                NULL
                )
            );

        if (pyOverloadInst) {
            for (auto arg: f->getOverloads()[k].getArgs()) {
                PyObjectStealer res(
                    PyObject_CallMethod(
                        (PyObject*)pyOverloadInst,
                        "addArg",
                        "sOOOO",
                        arg.getName().c_str(),
                        arg.getDefaultValue() ? PyTuple_Pack(1, arg.getDefaultValue()) : Py_None,
                        arg.getTypeFilter() ? (PyObject*)typePtrToPyTypeRepresentation(arg.getTypeFilter()) : Py_None,
                        arg.getIsStarArg() ? Py_True : Py_False,
                        arg.getIsKwarg() ? Py_True : Py_False
                        )
                    );

                if (!res) {
                    PyErr_PrintEx(0);
                }
            }

            PyTuple_SetItem(overloadTuple, k, incref(pyOverloadInst));
        } else {
            PyErr_PrintEx(0);
            PyTuple_SetItem(overloadTuple, k, incref(Py_None));
        }
    }

    return incref(overloadTuple);
}

PyObject* PyFunctionInstance::tp_call_concrete(PyObject* args, PyObject* kwargs) {
    for (long convertExplicitly = 0; convertExplicitly <= 1; convertExplicitly++) {
        for (long overloadIx = 0; overloadIx < type()->getOverloads().size(); overloadIx++) {
            std::pair<bool, PyObject*> res = PyFunctionInstance::tryToCallOverload(
                type(),
                overloadIx,
                nullptr,
                args,
                kwargs,
                convertExplicitly,
                false
            );

            if (res.first) {
                return res.second;
            }
        }
    }

    std::string argTupleTypeDesc = argTupleTypeDescription(nullptr, args, kwargs);

    PyErr_Format(
        PyExc_TypeError, "'%s' cannot find a valid overload with arguments of type %s",
        type()->name().c_str(),
        argTupleTypeDesc.c_str()
        );

    return NULL;
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
    //expose a list of overloads
    PyObjectStealer overloads(createOverloadPyRepresentation(inType));

    PyDict_SetItemString(
        pyType->tp_dict,
        "__name__",
        PyUnicode_FromString(inType->name().c_str())
    );

    PyDict_SetItemString(
        pyType->tp_dict,
        "__qualname__",
        PyUnicode_FromString(inType->name().c_str())
    );

    PyDict_SetItemString(
        pyType->tp_dict,
        "overloads",
        overloads
    );

}

int PyFunctionInstance::pyInquiryConcrete(const char* op, const char* opErrRep) {
    // op == '__bool__'
    return 1;
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

    return PyInstance::initialize(resType, [&](instance_ptr p) {});
}

/* static */
PyObject* PyFunctionInstance::overload(PyObject* funcObj, PyObject* args, PyObject* kwargs) {
    return translateExceptionToPyObject([&]() {
        if (kwargs && PyDict_Size(kwargs)) {
            throw std::runtime_error("Can't call 'overload' with kwargs");
        }

        Function* resType = (Function*)((PyInstance*)(funcObj))->type();

        if (!resType || resType->getTypeCategory() != Type::TypeCategory::catFunction) {
            throw std::runtime_error("Expected 'cls' to be a Function.");
        }

        iterate(args, [&](PyObject* arg) {
            Type* argT = PyInstance::extractTypeFrom(arg->ob_type);

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

                    argT = convertPythonObjectToFunction(name, arg);
                }

                if (!argT) {
                    throw PythonExceptionSet();
                }
            }

            if (argT->getTypeCategory() != Type::TypeCategory::catFunction) {
                throw std::runtime_error("'overload' requires arguments to be Function types");
            }

            resType = Function::merge(resType, (Function*)argT);
        });

        return PyInstance::initialize(resType, [&](instance_ptr p) {});
    });
}

/* static */
PyObject* PyFunctionInstance::resultTypeFor(PyObject* funcObj, PyObject* args, PyObject* kwargs) {
    static PyObject* runtimeModule = PyImport_ImportModule("typed_python.compiler.runtime");

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
    return new PyMethodDef [5] {
        {"overload", (PyCFunction)PyFunctionInstance::overload, METH_VARARGS | METH_KEYWORDS, NULL},
        {"withEntrypoint", (PyCFunction)PyFunctionInstance::withEntrypoint, METH_VARARGS | METH_KEYWORDS, NULL},
        {"resultTypeFor", (PyCFunction)PyFunctionInstance::resultTypeFor, METH_VARARGS | METH_KEYWORDS, NULL},
        {NULL, NULL}
    };
}

Function* PyFunctionInstance::convertPythonObjectToFunction(PyObject* name, PyObject *funcObj) {
    static PyObject* internalsModule = PyImport_ImportModule("typed_python.internals");

    if (!internalsModule) {
        PyErr_SetString(PyExc_TypeError, "Internal error: couldn't find typed_python.internals");
        return nullptr;
    }

    static PyObject* makeFunction = PyObject_GetAttrString(internalsModule, "makeFunction");

    if (!makeFunction) {
        PyErr_SetString(PyExc_TypeError, "Internal error: couldn't find typed_python.internals.makeFunction");
        return nullptr;
    }

    PyObject* fRes = PyObject_CallFunctionObjArgs(makeFunction, name, funcObj, NULL);

    if (!fRes) {
        return nullptr;
    }

    if (!PyType_Check(fRes)) {
        PyErr_SetString(PyExc_TypeError, "Internal error: expected typed_python.internals.makeFunction to return a type");
        return nullptr;
    }

    Type* actualType = PyInstance::extractTypeFrom((PyTypeObject*)fRes);

    if (!actualType || actualType->getTypeCategory() != Type::TypeCategory::catFunction) {
        PyErr_Format(PyExc_TypeError, "Internal error: expected makeFunction to return a Function. Got %S", fRes);
        return nullptr;
    }

    return (Function*)actualType;
}

/* static */
bool PyFunctionInstance::pyValCouldBeOfTypeConcrete(Function* type, PyObject* pyRepresentation, bool isExplicit) {
    if (!PyFunction_Check(pyRepresentation)) {
        return false;
    }

    if (type->getOverloads().size() != 1) {
        return false;
    }

    return (PyObject*)type->getOverloads()[0].getFunctionObj() == pyRepresentation;
}

/* static */
void PyFunctionInstance::copyConstructFromPythonInstanceConcrete(Function* type, instance_ptr tgt, PyObject* pyRepresentation, bool isExplicit) {
    if (pyValCouldBeOfTypeConcrete(type, pyRepresentation, isExplicit)) {
        return;
    }

    throw std::runtime_error("Can't convert to " + type->name());
}
