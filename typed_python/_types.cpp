
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

#include <Python.h>
#include <frameobject.h>
#include <numpy/arrayobject.h>
#include <map>
#include <memory>
#include <vector>
#include <string>
#include <iostream>

#include "AllTypes.hpp"
#include "NullSerializationContext.hpp"
#include "util.hpp"
#include "PyInstance.hpp"
#include "PyFunctionInstance.hpp"
#include "SerializationBuffer.hpp"
#include "DeserializationBuffer.hpp"
#include "PythonSerializationContext.hpp"
#include "UnicodeProps.hpp"
#include "_types.hpp"

PyObject *MakeTupleOrListOfType(PyObject* nullValue, PyObject* args, bool isTuple) {
    std::vector<Type*> types;

    if (!unpackTupleToTypes(args, types)) {
        return nullptr;
    }

    if (types.size() != 1) {
        if (isTuple) {
            PyErr_SetString(PyExc_TypeError, "TupleOfType takes 1 positional argument.");
        } else {
            PyErr_SetString(PyExc_TypeError, "ListOfType takes 1 positional argument.");
        }
        return NULL;
    }

    return incref(
        (PyObject*)PyInstance::typeObj(
            isTuple ? (TupleOrListOfType*)TupleOfType::Make(types[0]) : (TupleOrListOfType*)ListOfType::Make(types[0])
            )
        );
}

PyObject *MakePointerToType(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "PointerTo takes 1 positional argument.");
        return NULL;
    }

    PyObjectHolder tupleItem(PyTuple_GetItem(args, 0));

    Type* t = PyInstance::unwrapTypeArgToTypePtr(tupleItem);

    if (!t) {
        PyErr_SetString(PyExc_TypeError, "PointerTo needs a type.");
        return NULL;
    }

    return incref((PyObject*)PyInstance::typeObj(PointerTo::Make(t)));
}

PyObject *MakeRefToType(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "RefTo takes 1 positional argument.");
        return NULL;
    }

    PyObjectHolder tupleItem(PyTuple_GetItem(args, 0));

    Type* t = PyInstance::unwrapTypeArgToTypePtr(tupleItem);

    if (!t) {
        PyErr_SetString(PyExc_TypeError, "RefTo needs a type.");
        return NULL;
    }

    return translateExceptionToPyObject([&]{
        return incref((PyObject*)PyInstance::typeObj(RefTo::Make(t)));
    });
}

PyObject *MakeTypedCellType(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "TypedCell takes 1 positional argument.");
        return NULL;
    }

    PyObjectHolder tupleItem(PyTuple_GetItem(args, 0));

    Type* t = PyInstance::unwrapTypeArgToTypePtr(tupleItem);

    if (!t) {
        PyErr_SetString(PyExc_TypeError, "TypedCell needs a type.");
        return NULL;
    }

    return incref((PyObject*)PyInstance::typeObj(TypedCellType::Make(t)));
}

PyObject *MakeTupleOfType(PyObject* nullValue, PyObject* args) {
    return MakeTupleOrListOfType(nullValue, args, true);
}

PyObject *MakeListOfType(PyObject* nullValue, PyObject* args) {
    return MakeTupleOrListOfType(nullValue, args, false);
}

PyObject *MakeTupleType(PyObject* nullValue, PyObject* args) {
    std::vector<Type*> types;
    if (!unpackTupleToTypes(args, types)) {
        return NULL;
    }

    return incref((PyObject*)PyInstance::typeObj(Tuple::Make(types)));
}

PyObject *MakeConstDictType(PyObject* nullValue, PyObject* args) {
    std::vector<Type*> types;
    for (long k = 0; k < PyTuple_Size(args); k++) {
        PyObjectHolder item(PyTuple_GetItem(args,k));
        types.push_back(PyInstance::unwrapTypeArgToTypePtr(item));
        if (not types.back()) {
            return NULL;
        }
    }

    if (types.size() != 2) {
        PyErr_SetString(PyExc_TypeError, "ConstDict accepts two arguments");
        return NULL;
    }

    PyObject* typeObj = (PyObject*)PyInstance::typeObj(
        ConstDictType::Make(types[0],types[1])
        );

    return incref(typeObj);
}

PyObject* MakeSetType(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args)!=1) {
        PyErr_SetString(PyExc_TypeError, "Set takes 1 positional arguments");
        return NULL;
    }
    PyObjectHolder tupleItem(PyTuple_GetItem(args, 0));
    Type* t = PyInstance::unwrapTypeArgToTypePtr(tupleItem);
    if (!t) {
        PyErr_SetString(PyExc_TypeError, "Set needs a type.");
        return NULL;
    }
    SetType* setT = SetType::Make(t);
    return incref((PyObject*)PyInstance::typeObj(setT));
}

PyObject *MakeDictType(PyObject* nullValue, PyObject* args) {
    std::vector<Type*> types;
    for (long k = 0; k < PyTuple_Size(args); k++) {
        PyObjectHolder item(PyTuple_GetItem(args,k));
        types.push_back(PyInstance::unwrapTypeArgToTypePtr(item));
        if (not types.back()) {
            return NULL;
        }
    }

    if (types.size() != 2) {
        PyErr_SetString(PyExc_TypeError, "Dict accepts two arguments");
        return NULL;
    }

    return incref(
        (PyObject*)PyInstance::typeObj(
            DictType::Make(types[0],types[1])
            )
        );
}

PyObject *MakeOneOfType(PyObject* nullValue, PyObject* args) {
    std::vector<Type*> types;
    for (long k = 0; k < PyTuple_Size(args); k++) {
        PyObjectHolder item(PyTuple_GetItem(args,k));

        Type* t = PyInstance::tryUnwrapPyInstanceToType(item);

        if (t) {
            types.push_back(t);
        } else {
            PyErr_Format(PyExc_TypeError,
                "Type arguments must be types or simple values (like ints, strings, etc.), not %S. "
                "If you need a more complex value (such as a type object itself), wrap it in 'Value'.",
                (PyObject*)item
            );

            return NULL;
        }
    }

    PyObject* typeObj = (PyObject*)PyInstance::typeObj(OneOfType::Make(types));

    return incref(typeObj);
}

PyObject *MakeNamedTupleType(PyObject* nullValue, PyObject* args, PyObject* kwargs) {
    if (args && PyTuple_Check(args) && PyTuple_Size(args)) {
        PyErr_SetString(PyExc_TypeError, "NamedTuple takes no positional arguments.");
        return NULL;
    }

    std::vector<std::pair<std::string, Type*> > namesAndTypes;

    if (kwargs) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;

        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if (!PyUnicode_Check(key)) {
                PyErr_SetString(PyExc_TypeError, "NamedTuple keywords are supposed to be strings.");
                return NULL;
            }

            namesAndTypes.push_back(
                std::make_pair(
                    PyUnicode_AsUTF8(key),
                    PyInstance::unwrapTypeArgToTypePtr(value)
                    )
                );

            if (not namesAndTypes.back().second) {
                return NULL;
            }
        }
    }

    if (PY_MINOR_VERSION <= 5) {
        //we cannot rely on the ordering of 'kwargs' here because of the python version, so
        //we sort it. this will be a problem for anyone running some processes using different
        //python versions that share python code.
        std::sort(namesAndTypes.begin(), namesAndTypes.end());
    }

    std::vector<std::string> names;
    std::vector<Type*> types;

    for (auto p: namesAndTypes) {
        names.push_back(p.first);
        types.push_back(p.second);
    }

    return incref((PyObject*)PyInstance::typeObj(NamedTuple::Make(types, names)));
}


PyObject *MakePyCellType(PyObject* nullValue, PyObject* args) {
    return incref((PyObject*)PyInstance::typeObj(::PyCellType::Make()));
}
PyObject *MakeBoolType(PyObject* nullValue, PyObject* args) {
    return incref((PyObject*)PyInstance::typeObj(::Bool::Make()));
}
PyObject *MakeInt8Type(PyObject* nullValue, PyObject* args) {
    return incref((PyObject*)PyInstance::typeObj(::Int8::Make()));
}
PyObject *MakeInt16Type(PyObject* nullValue, PyObject* args) {
    return incref((PyObject*)PyInstance::typeObj(::Int16::Make()));
}
PyObject *MakeInt32Type(PyObject* nullValue, PyObject* args) {
    return incref((PyObject*)PyInstance::typeObj(::Int32::Make()));
}
PyObject *MakeInt64Type(PyObject* nullValue, PyObject* args) {
    return incref((PyObject*)PyInstance::typeObj(::Int64::Make()));
}
PyObject *MakeFloat32Type(PyObject* nullValue, PyObject* args) {
    return incref((PyObject*)PyInstance::typeObj(::Float32::Make()));
}
PyObject *MakeFloat64Type(PyObject* nullValue, PyObject* args) {
    return incref((PyObject*)PyInstance::typeObj(::Float64::Make()));
}
PyObject *MakeUInt8Type(PyObject* nullValue, PyObject* args) {
    return incref((PyObject*)PyInstance::typeObj(::UInt8::Make()));
}
PyObject *MakeUInt16Type(PyObject* nullValue, PyObject* args) {
    return incref((PyObject*)PyInstance::typeObj(::UInt16::Make()));
}
PyObject *MakeUInt32Type(PyObject* nullValue, PyObject* args) {
    return incref((PyObject*)PyInstance::typeObj(::UInt32::Make()));
}
PyObject *MakeUInt64Type(PyObject* nullValue, PyObject* args) {
    return incref((PyObject*)PyInstance::typeObj(::UInt64::Make()));
}
PyObject *MakeStringType(PyObject* nullValue, PyObject* args) {
    return incref((PyObject*)PyInstance::typeObj(::StringType::Make()));
}
PyObject *MakeBytesType(PyObject* nullValue, PyObject* args) {
    return incref((PyObject*)PyInstance::typeObj(::BytesType::Make()));
}
PyObject *MakeEmbeddedMessageType(PyObject* nullValue, PyObject* args) {
    return incref((PyObject*)PyInstance::typeObj(::EmbeddedMessageType::Make()));
}
PyObject *MakeNoneType(PyObject* nullValue, PyObject* args) {
    return incref((PyObject*)PyInstance::typeObj(::NoneType::Make()));
}

PyObject *getVTablePointer(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1 || !PyInstance::unwrapTypeArgToTypePtr(PyTuple_GetItem(args,0))) {
        PyErr_SetString(PyExc_TypeError, "getVTablePointer takes 1 positional argument (a type)");
        return NULL;
    }

    Type* type = PyInstance::unwrapTypeArgToTypePtr(PyTuple_GetItem(args,0));

    if (type->getTypeCategory() != Type::TypeCategory::catClass) {
        PyErr_Format(PyExc_TypeError, "Expected a Class, not %s", type->name().c_str());
        return NULL;
    }

    return PyLong_FromLong((size_t)((Class*)type)->getHeldClass()->getVTable());
}

PyObject *allocateClassMethodDispatch(PyObject* nullValue, PyObject* args, PyObject* kwargs)
{
    static const char *kwlist[] = {"classType", "methodName", "retType", "argTupleType", "kwargTupleType", NULL};

    PyObject* pyClassType;
    const char* methodName;
    PyObject* pyFuncRetType;
    PyObject* pyFuncArgType;
    PyObject* pyFuncKwargType;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OsOOO", (char**)kwlist, &pyClassType, &methodName, &pyFuncRetType, &pyFuncArgType, &pyFuncKwargType)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        Type* classType = PyInstance::unwrapTypeArgToTypePtr(pyClassType);
        if (!classType || classType->getTypeCategory() != Type::TypeCategory::catClass) {
            throw std::runtime_error("Expected 'classType' to be a Class");
        }

        Type* funcRetType = PyInstance::unwrapTypeArgToTypePtr(pyFuncRetType);
        if (!funcRetType) {
            throw std::runtime_error("Expected 'retType' to be a Type");
        }

        Type* funcArgType = PyInstance::unwrapTypeArgToTypePtr(pyFuncArgType);
        if (!funcArgType || funcArgType->getTypeCategory() != Type::TypeCategory::catTuple) {
            throw std::runtime_error("Expected 'argTuple' to be a Tuple");
        }

        Type* funcKwargType = PyInstance::unwrapTypeArgToTypePtr(pyFuncKwargType);
        if (!funcKwargType || funcKwargType->getTypeCategory() != Type::TypeCategory::catNamedTuple) {
            throw std::runtime_error("Expected 'kwargTupleType' to be a Tuple");
        }

        size_t dispatchSlot = ((Class*)classType)->getHeldClass()->allocateMethodDispatch(
            methodName,
            function_call_signature_type(
                funcRetType,
                (Tuple*)funcArgType,
                (NamedTuple*)funcKwargType
            )
        );

        return PyLong_FromLong(dispatchSlot);
    });
}

PyObject *getNextUnlinkedClassMethodDispatch(PyObject* nullValue, PyObject* args, PyObject* kwargs)
{
    static const char *kwlist[] = {NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        auto& needing = ClassDispatchTable::globalPointersNeedingCompile();

        if (needing.size() == 0) {
            return incref(Py_None);
        }

        std::pair<ClassDispatchTable*, size_t> dispatchAndSlot = *needing.begin();
        needing.erase(*needing.begin());

        return PyTuple_Pack(
            3,
            PyInstance::typeObj(dispatchAndSlot.first->getInterfaceClass()->getClassType()),
            PyInstance::typeObj(dispatchAndSlot.first->getImplementingClass()->getClassType()),
            PyLong_FromLong(dispatchAndSlot.second)
        );
    });
}

PyObject *getClassMethodDispatchSignature(PyObject* nullValue, PyObject* args, PyObject* kwargs)
{
    static const char *kwlist[] = {"interfaceClass", "implementingClass", "slot", NULL};

    PyObject* pyInterfaceClass;
    PyObject* pyImplementingClass;
    size_t slot;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOl", (char**)kwlist, &pyInterfaceClass, &pyImplementingClass, &slot)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        Type* interfaceClass = PyInstance::unwrapTypeArgToTypePtr(pyInterfaceClass);
        Type* implementingClass = PyInstance::unwrapTypeArgToTypePtr(pyImplementingClass);

        if (!interfaceClass || interfaceClass->getTypeCategory() != Type::TypeCategory::catClass) {
            throw std::runtime_error("Expected 'interfaceClass' to be a Class");
        }

        if (!implementingClass || implementingClass->getTypeCategory() != Type::TypeCategory::catClass) {
            throw std::runtime_error("Expected 'implementingClass' to be a Class");
        }

        HeldClass* heldInterface = ((Class*)interfaceClass)->getHeldClass();
        HeldClass* heldImplementing = ((Class*)implementingClass)->getHeldClass();

        ClassDispatchTable* cdt = heldImplementing->dispatchTableAs(heldInterface);

        method_call_signature_type sig = cdt->dispatchDefinitionForSlot(slot);

        return PyTuple_Pack(
            4,
            PyUnicode_FromString(sig.first.c_str()),
            PyInstance::typeObj(std::get<0>(sig.second)),
            PyInstance::typeObj(std::get<1>(sig.second)),
            PyInstance::typeObj(std::get<2>(sig.second))
        );
    });
}

PyObject *installClassMethodDispatch(PyObject* nullValue, PyObject* args, PyObject* kwargs)
{
    static const char *kwlist[] = {"interfaceClass", "implementingClass", "slot", "funcPtr", NULL};

    PyObject* pyInterfaceClass;
    PyObject* pyImplementingClass;
    size_t slot;
    size_t funcPtr;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOll", (char**)kwlist, &pyInterfaceClass, &pyImplementingClass, &slot, &funcPtr)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        Type* interfaceClass = PyInstance::unwrapTypeArgToTypePtr(pyInterfaceClass);
        Type* implementingClass = PyInstance::unwrapTypeArgToTypePtr(pyImplementingClass);

        if (!interfaceClass || interfaceClass->getTypeCategory() != Type::TypeCategory::catClass) {
            throw std::runtime_error("Expected 'interfaceClass' to be a Class");
        }

        if (!implementingClass || implementingClass->getTypeCategory() != Type::TypeCategory::catClass) {
            throw std::runtime_error("Expected 'implementingClass' to be a Class");
        }

        HeldClass* heldInterface = ((Class*)interfaceClass)->getHeldClass();
        HeldClass* heldImplementing = ((Class*)implementingClass)->getHeldClass();

        ClassDispatchTable* cdt = heldImplementing->dispatchTableAs(heldInterface);

        cdt->define(slot, (untyped_function_ptr)funcPtr);

        return incref(Py_None);
    });
}

PyObject *prepareArgumentToBePassedToCompiler(PyObject* nullValue, PyObject* args, PyObject* kwargs) {
    static const char *kwlist[] = {"obj", NULL};

    PyObject* obj;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist, &obj)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        return PyFunctionInstance::prepareArgumentToBePassedToCompiler(obj);
    });
}

PyObject *getDispatchIndexForType(PyObject* nullValue, PyObject* args, PyObject* kwargs)
{
    static const char *kwlist[] = {"interfaceClass", "implementingClass", NULL};

    PyObject* pyInterfaceClass;
    PyObject* pyImplementingClass;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", (char**)kwlist, &pyInterfaceClass, &pyImplementingClass)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        Type* interfaceClass = PyInstance::unwrapTypeArgToTypePtr(pyInterfaceClass);
        Type* implementingClass = PyInstance::unwrapTypeArgToTypePtr(pyImplementingClass);

        if (!interfaceClass || interfaceClass->getTypeCategory() != Type::TypeCategory::catClass) {
            throw std::runtime_error("Expected 'interfaceClass' to be a Class");
        }

        if (!implementingClass || implementingClass->getTypeCategory() != Type::TypeCategory::catClass) {
            throw std::runtime_error("Expected 'implementingClass' to be a Class");
        }

        HeldClass* heldInterface = ((Class*)interfaceClass)->getHeldClass();
        HeldClass* heldImplementing = ((Class*)implementingClass)->getHeldClass();

        return PyLong_FromLong(heldImplementing->getMroIndex(heldInterface));
    });
}

PyObject* classGetDispatchIndex(PyObject* nullValue, PyObject* args, PyObject* kwargs)
{
    static const char *kwlist[] = {"instance", NULL};

    PyObject* instance;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist, &instance)) {
        return NULL;
    }


    return translateExceptionToPyObject([&]() {
        Type* actualType = PyInstance::extractTypeFrom((PyTypeObject*)instance->ob_type);

        if (!actualType || actualType->getTypeCategory() != Type::TypeCategory::catClass) {
            throw std::runtime_error("Expected 'instance' to be an instance of a class");
        }

        return PyLong_FromLong(Class::instanceToDispatchTableIndex(((PyInstance*)instance)->dataPtr()));
    });
}

PyObject *installClassDestructor(PyObject* nullValue, PyObject* args, PyObject* kwargs)
{
    static const char *kwlist[] = {"implementingClass", "funcPtr", NULL};

    PyObject* pyImplementingClass;
    size_t funcPtr;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Ol", (char**)kwlist, &pyImplementingClass, &funcPtr)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        Type* implementingClass = PyInstance::unwrapTypeArgToTypePtr(pyImplementingClass);

        if (!implementingClass || implementingClass->getTypeCategory() != Type::TypeCategory::catClass) {
            throw std::runtime_error("Expected 'implementingClass' to be a Class");
        }

        HeldClass* heldImplementing = ((Class*)implementingClass)->getHeldClass();

        heldImplementing->getVTable()->installDestructor((destructor_fun_type)funcPtr);

        return incref(Py_None);
    });
}

PyObject *MakeTypeFor(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "TypeFor takes 1 positional argument");
        return NULL;
    }

    PyObjectHolder arg(PyTuple_GetItem(args,0));

    if (arg == Py_None) {
        return incref((PyObject*)Py_None->ob_type);
    }

    if (!PyType_Check(arg)) {
        PyErr_Format(PyExc_TypeError, "TypeFor expects a python primitive or an existing native value, not %S", (PyObject*)arg);
        return NULL;
    }

    if (arg == &PyLong_Type ||
        arg == &PyFloat_Type ||
        arg == Py_None->ob_type ||
        arg == &PyBool_Type ||
        arg == &PyBytes_Type ||
        arg == &PyUnicode_Type
    ) {
        return incref(arg);
    }

    Type* type = PyInstance::unwrapTypeArgToTypePtr(arg);

    if (type) {
        return incref((PyObject*)PyInstance::typeObj(type));
    }

    PyErr_Format(PyExc_TypeError, "Couldn't convert %S to a Type", (PyObject*)arg);
    return NULL;
}

PyObject *MakeValueType(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "Value takes 1 positional argument");
        return NULL;
    }

    PyObjectHolder arg(PyTuple_GetItem(args,0));

    Type* type = PyInstance::tryUnwrapPyInstanceToValueType(arg, true);

    if (type) {
        return incref((PyObject*)PyInstance::typeObj(type));
    }

    PyErr_Format(PyExc_TypeError, "Couldn't convert %S to a value", (PyObject*)arg);
    return NULL;
}

PyObject *MakeBoundMethodType(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "BoundMethod takes 2 arguments");
        return NULL;
    }

    PyObjectHolder a0(PyTuple_GetItem(args,0));
    PyObjectHolder a1(PyTuple_GetItem(args,1));

    Type* t0 = PyInstance::unwrapTypeArgToTypePtr(a0);

    if (!PyUnicode_Check(a1)) {
        PyErr_SetString(PyExc_TypeError, "Expected second argument to be a string");
        return NULL;
    }

    Type* resType = BoundMethod::Make((Class*)t0, PyUnicode_AsUTF8(a1));

    return incref((PyObject*)PyInstance::typeObj(resType));
}

PyObject *MakeFunctionType(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 6 && PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "Function takes 2 or 6 arguments");
        return NULL;
    }

    Function* resType;

    if (PyTuple_Size(args) == 2) {
        PyObjectHolder a0(PyTuple_GetItem(args, 0));
        PyObjectHolder a1(PyTuple_GetItem(args, 1));

        Type* t0 = PyInstance::unwrapTypeArgToTypePtr(a0);
        Type* t1 = PyInstance::unwrapTypeArgToTypePtr(a1);

        if (!t0 || t0->getTypeCategory() != Type::TypeCategory::catFunction) {
            PyErr_SetString(PyExc_TypeError, "Expected first argument to be a function");
            return NULL;
        }
        if (!t1 || t1->getTypeCategory() != Type::TypeCategory::catFunction) {
            PyErr_SetString(PyExc_TypeError, "Expected second argument to be a function");
            return NULL;
        }

        resType = Function::merge((Function*)t0, (Function*)t1);
    } else {
        PyObjectHolder nameObj(PyTuple_GetItem(args, 0));
        if (!PyUnicode_Check(nameObj)) {
            PyErr_SetString(PyExc_TypeError, "First arg should be a string.");
            return NULL;
        }
        PyObjectHolder qualnameObj(PyTuple_GetItem(args, 1));
        if (!PyUnicode_Check(qualnameObj)) {
            PyErr_SetString(PyExc_TypeError, "First arg should be a string.");
            return NULL;
        }
        PyObjectHolder retType(PyTuple_GetItem(args, 2));
        PyObjectHolder funcObj(PyTuple_GetItem(args, 3));
        PyObjectHolder argTuple(PyTuple_GetItem(args, 4));

        int assumeClosureGlobal = PyObject_IsTrue(PyTuple_GetItem(args, 5));

        if (assumeClosureGlobal == -1) {
            return NULL;
        }

        if (!PyFunction_Check(funcObj)) {
            PyErr_SetString(PyExc_TypeError, "Third arg should be a function object");
            return NULL;
        }

        Type* rType = 0;

        if (retType != Py_None) {
            rType = PyInstance::unwrapTypeArgToTypePtr(retType);
            if (!rType) {
                PyErr_SetString(PyExc_TypeError, "Expected second argument to be None or a type");
                return NULL;
            }
        }

        if (!PyTuple_Check(argTuple)) {
            PyErr_SetString(PyExc_TypeError, "Expected fourth argument to be a tuple of args");
            return NULL;
        }

        std::vector<Function::FunctionArg> argList;

        for (long k = 0; k < PyTuple_Size(argTuple); k++) {
            PyObjectHolder kTup(PyTuple_GetItem(argTuple, k));
            if (!PyTuple_Check(kTup) || PyTuple_Size(kTup) != 5) {
                PyErr_SetString(PyExc_TypeError, "Argtuple elements should be tuples of five things.");
                return NULL;
            }

            PyObjectHolder k0(PyTuple_GetItem(kTup, 0));
            PyObjectHolder k1(PyTuple_GetItem(kTup, 1));
            PyObjectHolder k2(PyTuple_GetItem(kTup, 2));
            PyObjectHolder k3(PyTuple_GetItem(kTup, 3));
            PyObjectHolder k4(PyTuple_GetItem(kTup, 4));

            if (!PyUnicode_Check(k0)) {
                PyErr_Format(PyExc_TypeError, "Argument %S has a name which is not a string.", (PyObject*)k0);
                return NULL;
            }

            Type* argT = nullptr;
            if (k1 != Py_None) {
                argT = PyInstance::unwrapTypeArgToTypePtr(k1);
                if (!argT) {
                    PyErr_Format(PyExc_TypeError, "Argument %S has a type argument %S which should be None or a Type.", k0->ob_type, k1->ob_type);
                    return NULL;
                }
            }

            if ((k3 != Py_True && k3 != Py_False) || (k4 != Py_True && k4 != Py_False)) {
                PyErr_Format(PyExc_TypeError, "Argument %S has a malformed type tuple", (PyObject*)k0);
                return NULL;
            }

            PyObject* val = nullptr;
            if (k2 != Py_None) {
                if (!PyTuple_Check(k2) || PyTuple_Size(k2) != 1) {
                    PyErr_Format(PyExc_TypeError, "Argument %S has a malformed type tuple", (PyObject*)k0);
                    return NULL;
                }

                val = PyTuple_GetItem(k2,0);
            }

            if (val) {
                incref(val);
            }

            argList.push_back(Function::FunctionArg(
                PyUnicode_AsUTF8(k0),
                argT,
                val,
                k3 == Py_True,
                k4 == Py_True
                ));
        }

        std::string moduleName = "<unknown>";

        if (PyObject_HasAttrString(funcObj, "__module__")) {
            PyObject* pyModulename = PyObject_GetAttrString(funcObj, "__module__");

            if (!pyModulename) {
                return NULL;
            }

            if (PyUnicode_Check(pyModulename)) {
                moduleName = PyUnicode_AsUTF8(pyModulename);
            }

            decref(pyModulename);
        }

        std::vector<Function::Overload> overloads;

        std::vector<std::string> closureVarnames;
        std::vector<Type*> closureVarTypes;
        std::map<std::string, PyObject*> globalsInCells;
        std::map<std::string, ClosureVariableBinding> closureBindings;

        PyObject* closure = PyFunction_GetClosure(funcObj);

        if (closure) {
            PyObjectStealer coFreevars(PyObject_GetAttrString(PyFunction_GetCode(funcObj), "co_freevars"));

            if (!coFreevars) {
                return NULL;
            }

            if (!PyTuple_Check(coFreevars)) {
                PyErr_Format(PyExc_TypeError, "f.__code__.co_freevars was not a tuple");
                return NULL;
            }

            if (PyTuple_Size(coFreevars) != PyTuple_Size(closure)) {
                PyErr_Format(PyExc_TypeError, "f.__code__.co_freevars had a different number of elements than the closure");
                return NULL;
            }

            for (long ix = 0; ix < PyTuple_Size(coFreevars); ix++) {
                PyObject* varname = PyTuple_GetItem(coFreevars, ix);
                if (!PyUnicode_Check(varname)) {
                    PyErr_Format(PyExc_TypeError, "f.__code__.co_freevars was not all strings");
                    return NULL;
                }
                closureVarnames.push_back(std::string(PyUnicode_AsUTF8(varname)));
            }

            if (assumeClosureGlobal) {
                for (long ix = 0; ix < PyTuple_Size(closure); ix++) {
                    PyObject* cell = PyTuple_GetItem(closure, ix);
                    if (!PyCell_Check(cell)) {
                        PyErr_Format(PyExc_TypeError, "Function closure needs to all be cells.");
                        return NULL;
                    }

                    globalsInCells[closureVarnames[ix]] = incref(cell);
                }
            }
            else {
                for (long ix = 0; ix < PyTuple_Size(closure); ix++) {
                    closureVarTypes.push_back(PyCellType::Make());
                    closureBindings[closureVarnames[ix]] = (
                        ClosureVariableBinding() + 0 + ix + ClosureVariableBindingStep::AccessCell()
                    );
                }
            }
        }

        overloads.push_back(
            Function::Overload(
                PyFunction_GetCode(funcObj),
                PyFunction_GetGlobals(funcObj),
                PyFunction_GetDefaults(funcObj),
                PyFunction_GetAnnotations(funcObj),
                globalsInCells,
                closureVarnames,
                closureBindings,
                rType,
                argList
            )
        );

        resType = Function::Make(
            PyUnicode_AsUTF8(nameObj),
            PyUnicode_AsUTF8(qualnameObj),
            moduleName,
            overloads,
            Tuple::Make({
                assumeClosureGlobal ?
                    NamedTuple::Make({}, {}) :
                    NamedTuple::Make(closureVarTypes, closureVarnames)
                }),
            false,
            false
        );
    }

    return incref((PyObject*)PyInstance::typeObj(resType));
}

PyObject *MakeClassType(PyObject* nullValue, PyObject* args) {
    int expected_args = 8;
    if (PyTuple_Size(args) != expected_args) {
        PyErr_Format(PyExc_TypeError, "Class takes %S arguments", expected_args);
        return NULL;
    }

    PyObjectHolder nameArg(PyTuple_GetItem(args,0));

    if (!PyUnicode_Check(nameArg)) {
        PyErr_SetString(PyExc_TypeError, "Class needs a string in the first argument");
        return NULL;
    }

    std::string name = PyUnicode_AsUTF8(nameArg);

    PyObjectHolder basesTuple(PyTuple_GetItem(args, 1));
    PyObjectHolder final(PyTuple_GetItem(args, 2));
    PyObjectHolder memberTuple(PyTuple_GetItem(args, 3));
    PyObjectHolder memberFunctionTuple(PyTuple_GetItem(args, 4));
    PyObjectHolder staticFunctionTuple(PyTuple_GetItem(args, 5));
    PyObjectHolder propertyFunctionTuple(PyTuple_GetItem(args, 6));
    PyObjectHolder classMemberTuple(PyTuple_GetItem(args, 7));

    if (!PyTuple_Check(basesTuple)) {
        PyErr_SetString(PyExc_TypeError, "Class needs a tuple of Class type objects in the second argument");
        return NULL;
    }

    if (!PyBool_Check(final)) {
        PyErr_SetString(PyExc_TypeError, "Class needs a bool in the third argument");
        return NULL;
    }

    if (!PyTuple_Check(memberTuple)) {
        PyErr_SetString(PyExc_TypeError, "Class needs a tuple of (str, member_type) in the fourth argument");
        return NULL;
    }

    if (!PyTuple_Check(memberFunctionTuple)) {
        PyErr_SetString(PyExc_TypeError, "Class needs a tuple of (str, Function) in the fifth argument");
        return NULL;
    }

    if (!PyTuple_Check(memberFunctionTuple)) {
        PyErr_SetString(PyExc_TypeError, "Class needs a tuple of (str, Function) in the sixth argument");
        return NULL;
    }

    if (!PyTuple_Check(propertyFunctionTuple)) {
        PyErr_SetString(PyExc_TypeError, "Class needs a tuple of (str, object) in the seventh argument");
        return NULL;
    }

    if (!PyTuple_Check(classMemberTuple)) {
        PyErr_SetString(PyExc_TypeError, "Class needs a tuple of (str, object) in the eighth argument");
        return NULL;
    }

    std::vector<Type*> bases;
    std::vector<std::tuple<std::string, Type*, Instance> > members;
    std::vector<std::pair<std::string, Type*> > memberFunctions;
    std::vector<std::pair<std::string, Type*> > staticFunctions;
    std::vector<std::pair<std::string, Type*> > propertyFunctions;
    std::vector<std::pair<std::string, PyObject*> > classMembers;

    if (!unpackTupleToTypes(basesTuple, bases)) {
        return NULL;
    }
    std::vector<Class*> baseClasses;

    for (auto t: bases) {
        if (t->getTypeCategory() != Type::TypeCategory::catClass) {
            PyErr_SetString(PyExc_TypeError, "Classes must descend from other Class types");
            return NULL;
        }
        baseClasses.push_back((Class*)t);
    }

    if (!unpackTupleToStringTypesAndValues(memberTuple, members)) {
        return NULL;
    }
    if (!unpackTupleToStringAndTypes(memberFunctionTuple, memberFunctions)) {
        return NULL;
    }
    if (!unpackTupleToStringAndTypes(staticFunctionTuple, staticFunctions)) {
        return NULL;
    }
    if (!unpackTupleToStringAndTypes(propertyFunctionTuple, propertyFunctions)) {
        return NULL;
    }
    if (!unpackTupleToStringAndObjects(classMemberTuple, classMembers)) {
        return NULL;
    }

    std::map<std::string, Function*> memberFuncs;
    std::map<std::string, Function*> staticFuncs;
    std::map<std::string, Function*> propertyFuncs;

    for (auto mf: memberFunctions) {
        if (mf.second->getTypeCategory() != Type::TypeCategory::catFunction) {
            PyErr_Format(PyExc_TypeError, "Class member %s is not a function.", mf.first.c_str());
            return NULL;
        }
        if (memberFuncs.find(mf.first) != memberFuncs.end()) {
            PyErr_Format(PyExc_TypeError, "Class member %s repeated. This should have"
                                    " been compressed as an overload.", mf.first.c_str());
            return NULL;
        }
        memberFuncs[mf.first] = (Function*)mf.second;
    }
    for (auto pf: propertyFunctions) {
        if (pf.second->getTypeCategory() != Type::TypeCategory::catFunction) {
            PyErr_Format(PyExc_TypeError, "Class member %s is not a function.", pf.first.c_str());
            return NULL;
        }
        if (propertyFuncs.find(pf.first) != propertyFuncs.end()) {
            PyErr_Format(PyExc_TypeError, "Class member %s repeated. This should have"
                                    " been compressed as an overload.", pf.first.c_str());
            return NULL;
        }
        propertyFuncs[pf.first] = (Function*)pf.second;
    }

    for (auto mf: staticFunctions) {
        if (mf.second->getTypeCategory() != Type::TypeCategory::catFunction) {
            PyErr_Format(PyExc_TypeError, "Class member %s is not a function.", mf.first.c_str());
            return NULL;
        }
        if (staticFuncs.find(mf.first) != staticFuncs.end()) {
            PyErr_Format(PyExc_TypeError, "Class member %s repeated. This should have"
                                    " been compressed as an overload.", mf.first.c_str());
            return NULL;
        }
        staticFuncs[mf.first] = (Function*)mf.second;
    }

    std::map<std::string, PyObject*> clsMembers;

    for (auto mf: classMembers) {
        clsMembers[mf.first] = mf.second;
    }

    return translateExceptionToPyObject([&]() {
        return incref(
            (PyObject*)PyInstance::typeObj(
                Class::Make(
                    name,
                    baseClasses,
                    final == Py_True,
                    members,
                    memberFuncs,
                    staticFuncs,
                    propertyFuncs,
                    clsMembers
                )
            )
        );
    });
}

PyObject *pyInstanceHeldObjectAddress(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "pyInstanceHeldObjectAddress takes a PyInstance");
        return NULL;
    }

    PyObject* a1(PyTuple_GetItem(args, 0));

    Type* actualType = PyInstance::extractTypeFrom(a1->ob_type);

    if (!actualType) {
        PyErr_Format(
            PyExc_TypeError,
            "pyInstanceHeldObjectAddress takes a PyInstance",
            (PyObject*)a1
            );
        return NULL;
    }

    return PyLong_FromLong((size_t)((PyInstance*)a1)->dataPtr());
}

PyObject *pointerTo(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "pointerTo takes a single class instance as an argument");
        return NULL;
    }

    PyObjectHolder a1(PyTuple_GetItem(args, 0));

    Type* actualType = PyInstance::extractTypeFrom(a1->ob_type);

    if (!actualType || actualType->getTypeCategory() != Type::TypeCategory::catClass) {
        PyErr_Format(
            PyExc_TypeError,
            "first argument to pointerTo '%S' must be a Class",
            (PyObject*)a1
            );
        return NULL;
    }

    Class* clsType = (Class*)actualType;
    Class::layout* layout = clsType->instanceToLayout(((PyInstance*)(PyObject*)a1)->dataPtr());

    PointerTo* ptrType = PointerTo::Make(clsType->getHeldClass());
    void* ptrValue = layout->data;

    return PyInstance::extractPythonObject((instance_ptr)&ptrValue, ptrType);
}

PyObject *refTo(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "refTo takes a single class instance as an argument");
        return NULL;
    }

    PyObjectHolder a1(PyTuple_GetItem(args, 0));

    Type* actualType = PyInstance::extractTypeFrom(a1->ob_type);

    if (!actualType || actualType->getTypeCategory() != Type::TypeCategory::catClass) {
        PyErr_Format(
            PyExc_TypeError,
            "first argument to refTo '%S' must be a Class",
            (PyObject*)a1
            );
        return NULL;
    }

    Class* clsType = (Class*)actualType;
    Class::layout* layout = clsType->instanceToLayout(((PyInstance*)(PyObject*)a1)->dataPtr());

    RefTo* refType = RefTo::Make(clsType->getHeldClass());
    void* ptrValue = layout->data;

    return PyInstance::extractPythonObject((instance_ptr)&ptrValue, refType);
}

PyObject *copyRefTo(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "copy takes a single RefTo instance as an argument");
        return NULL;
    }

    PyObjectHolder a1(PyTuple_GetItem(args, 0));

    Type* actualType = PyInstance::extractTypeFrom(a1->ob_type);

    if (!actualType || actualType->getTypeCategory() != Type::TypeCategory::catRefTo) {
        PyErr_Format(
            PyExc_TypeError,
            "first argument to copy '%S' must be a RefTo",
            (PyObject*)a1
            );
        return NULL;
    }

    instance_ptr reffedHeldClassData = *(instance_ptr*)((PyInstance*)(PyObject*)a1)->dataPtr();

    RefTo* refTo = (RefTo*)actualType;
    HeldClass* heldCls = (HeldClass*)refTo->getEltType();
    Class* cls = heldCls->getClassType();

    return PyInstance::initialize(cls, [&](instance_ptr data) {
        cls->constructorInitializingHeld(data, [&](instance_ptr heldClsData) {
            heldCls->copy_constructor(heldClsData, reffedHeldClassData);
        });
    });
}

PyObject *refcount(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "refcount takes 1 positional argument");
        return NULL;
    }

    PyObjectHolder a1(PyTuple_GetItem(args, 0));

    Type* actualType = PyInstance::extractTypeFrom(a1->ob_type);

    if (!actualType || (
            actualType->getTypeCategory() != Type::TypeCategory::catTupleOf &&
            actualType->getTypeCategory() != Type::TypeCategory::catListOf &&
            actualType->getTypeCategory() != Type::TypeCategory::catClass &&
            actualType->getTypeCategory() != Type::TypeCategory::catConstDict &&
            actualType->getTypeCategory() != Type::TypeCategory::catDict &&
            actualType->getTypeCategory() != Type::TypeCategory::catSet &&
            actualType->getTypeCategory() != Type::TypeCategory::catAlternative &&
            actualType->getTypeCategory() != Type::TypeCategory::catConcreteAlternative
            )) {
        PyErr_Format(
            PyExc_TypeError,
            "first argument to refcount '%S' not a permitted Type",
            (PyObject*)a1
            );
        return NULL;
    }

    if (actualType->getTypeCategory() == Type::TypeCategory::catTupleOf) {
        return PyLong_FromLong(
            ((::TupleOfType*)actualType)->refcount(((PyInstance*)(PyObject*)a1)->dataPtr())
            );
    }

    if (actualType->getTypeCategory() == Type::TypeCategory::catListOf) {
        return PyLong_FromLong(
            ((::ListOfType*)actualType)->refcount(((PyInstance*)(PyObject*)a1)->dataPtr())
            );
    }

    if (actualType->getTypeCategory() == Type::TypeCategory::catClass) {
        return PyLong_FromLong(
            ((::Class*)actualType)->refcount(((PyInstance*)(PyObject*)a1)->dataPtr())
            );
    }

    if (actualType->getTypeCategory() == Type::TypeCategory::catConstDict) {
        return PyLong_FromLong(
            ((::ConstDictType*)actualType)->refcount(((PyInstance*)(PyObject*)a1)->dataPtr())
            );
    }
    if (actualType->getTypeCategory() == Type::TypeCategory::catSet) {
        return PyLong_FromLong(
            ((::SetType*)actualType)->refcount(((PyInstance*)(PyObject*)a1)->dataPtr())
            );
    }
    if (actualType->getTypeCategory() == Type::TypeCategory::catDict) {
        return PyLong_FromLong(
            ((::DictType*)actualType)->refcount(((PyInstance*)(PyObject*)a1)->dataPtr())
            );
    }
    if (actualType->getTypeCategory() == Type::TypeCategory::catAlternative) {
        return PyLong_FromLong(
            ((::Alternative*)actualType)->refcount(((PyInstance*)(PyObject*)a1)->dataPtr())
            );
    }
    if (actualType->getTypeCategory() == Type::TypeCategory::catConcreteAlternative) {
        return PyLong_FromLong(
            ((::ConcreteAlternative*)actualType)->refcount(((PyInstance*)(PyObject*)a1)->dataPtr())
            );
    }


    PyErr_Format(
        PyExc_TypeError,
        "this code should be unreachable"
        );
    return NULL;
}

/**
    Serializes an object instance and returns a pointer to a PyBytes object

    Note: we list the params to be extracted from `args`
    @param a1: Serialization Type
    @param a2: Instance
    @param a3: Serialization Context (optional)
*/
PyObject *serialize(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 2 && PyTuple_Size(args) != 3) {
        PyErr_SetString(PyExc_TypeError, "serialize takes 2 or 3 positional arguments");
        return NULL;
    }

    PyObjectHolder a1(PyTuple_GetItem(args, 0));
    PyObjectHolder a2(PyTuple_GetItem(args, 1));
    PyObjectHolder a3(PyTuple_Size(args) == 3 ? PyTuple_GetItem(args, 2) : nullptr);

    Type* serializeType = PyInstance::unwrapTypeArgToTypePtr(a1);

    if (!serializeType) {
        PyErr_Format(
            PyExc_TypeError,
            "first argument to serialize must be a native type object, not %S",
            (PyObject*)a1
            );
        return NULL;
    }

    serializeType->assertForwardsResolved();

    Type* actualType = PyInstance::extractTypeFrom(a2->ob_type);

    std::shared_ptr<SerializationContext> context(new NullSerializationContext());

    try {
        if (a3 && a3 != Py_None) {
            context.reset(new PythonSerializationContext(a3));
        }
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    } catch(PythonExceptionSet& e) {
        return NULL;
    }

    SerializationBuffer b(*context);

    try{
        if (actualType == serializeType) {
            //the simple case
            PyEnsureGilReleased releaseTheGil;

            actualType->serialize(((PyInstance*)(PyObject*)a2)->dataPtr(), b, 0);
        } else {
            //try to construct a 'serialize type' from the argument and then serialize that
            Instance i = Instance::createAndInitialize(serializeType, [&](instance_ptr p) {
                PyInstance::copyConstructFromPythonInstance(serializeType, p, a2, true);
            });

            PyEnsureGilReleased releaseTheGil;

            i.type()->serialize(i.data(), b, 0);
        }

        b.finalize();
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    } catch(PythonExceptionSet& e) {
        return NULL;
    }

    return PyBytes_FromStringAndSize((const char*)b.buffer(), b.size());
}


PyObject *createPyCell(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "createPyCell takes 1 positional argument");
        return NULL;
    }

    return PyCell_New(PyTuple_GetItem(args, 0));
}

PyObject *identityHash(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "identityHash takes 1 positional argument");
        return NULL;
    }
    PyObjectHolder a1(PyTuple_GetItem(args, 0));

    ShaHash hash;
    Type* typeArg = nullptr;

    return translateExceptionToPyObject([&]() {
        if (PyType_Check(a1) && (typeArg = PyInstance::extractTypeFrom(a1))) {
            hash = typeArg->identityHash();
        } else {
            hash = MutuallyRecursiveTypeGroup::pyObjectShaHash(a1, nullptr);
        }

        return PyBytes_FromStringAndSize((const char*)&hash, sizeof(ShaHash));
    });
}

/**
    Serializes a container instance as a stream of concatenated messages, and return a pointer to a PyBytes object

    Note: we list the params to be extracted from `args`
    @param a1: Serialization Type
    @param a2: Instance
    @param a3: Serialization Context (optional)
*/
PyObject *serializeStream(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 2 && PyTuple_Size(args) != 3) {
        PyErr_SetString(PyExc_TypeError, "serialize takes 2 or 3 positional arguments");
        return NULL;
    }

    PyObjectHolder a1(PyTuple_GetItem(args, 0));
    PyObjectHolder a2(PyTuple_GetItem(args, 1));
    PyObjectHolder a3(PyTuple_Size(args) == 3 ? PyTuple_GetItem(args, 2) : nullptr);

    Type* serializeType = PyInstance::unwrapTypeArgToTypePtr(a1);

    if (!serializeType) {
        PyErr_Format(
            PyExc_TypeError,
            "first argument to serializeStream must be a native type object, not %S",
            (PyObject*)a1
            );
        return NULL;
    }

    serializeType->assertForwardsResolved();

    Type* actualType = PyInstance::extractTypeFrom(a2->ob_type);

    std::shared_ptr<SerializationContext> context(new NullSerializationContext());

    try{
        if (a3 && a3 != Py_None) {
            context.reset(new PythonSerializationContext(a3));
        }
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    } catch(PythonExceptionSet& e) {
        return NULL;
    }

    SerializationBuffer b(*context);

    try{
        if (actualType && (actualType->getTypeCategory() == Type::TypeCategory::catListOf ||
                actualType->getTypeCategory() == Type::TypeCategory::catTupleOf)) {
            if (((TupleOrListOfType*)actualType)->getEltType() == serializeType) {
                PyEnsureGilReleased releaseTheGil;

                ((TupleOrListOfType*)actualType)->serializeStream(((PyInstance*)(PyObject*)a2)->dataPtr(), b);

                b.finalize();

                return PyBytes_FromStringAndSize((const char*)b.buffer(), b.size());
            }
        }

        //try to construct a 'serialize type' from the argument and then serialize that
        TupleOfType* serializeTupleType = TupleOfType::Make(serializeType);

        Instance i = Instance::createAndInitialize(serializeTupleType, [&](instance_ptr p) {
            PyInstance::copyConstructFromPythonInstance(serializeTupleType, p, a2, true);
        });

        {
            PyEnsureGilReleased releaseTheGil;
            serializeTupleType->serializeStream(i.data(), b);
        }

        b.finalize();

        return PyBytes_FromStringAndSize((const char*)b.buffer(), b.size());
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    } catch(PythonExceptionSet& e) {
        return NULL;
    }
}

PyObject *deserialize(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 2 && PyTuple_Size(args) != 3) {
        PyErr_SetString(PyExc_TypeError, "deserialize takes 2 or 3 positional arguments");
        return NULL;
    }
    PyObjectHolder a1(PyTuple_GetItem(args, 0));
    PyObjectHolder a2(PyTuple_GetItem(args, 1));
    PyObjectHolder a3(PyTuple_Size(args) == 3 ? PyTuple_GetItem(args, 2) : nullptr);

    Type* serializeType = PyInstance::unwrapTypeArgToTypePtr(a1);

    if (!serializeType) {
        PyErr_SetString(PyExc_TypeError, "first argument to deserialize must be a native type object");
        return NULL;
    }
    if (!PyBytes_Check(a2)) {
        PyErr_SetString(PyExc_TypeError, "second argument to deserialize must be a bytes object");
        return NULL;
    }

    std::shared_ptr<SerializationContext> context(new NullSerializationContext());
    if (a3 && a3 != Py_None) {
        context.reset(new PythonSerializationContext(a3));
    }

    return translateExceptionToPyObject([&]() {
        DeserializationBuffer buf((uint8_t*)PyBytes_AsString(a2), PyBytes_GET_SIZE((PyObject*)a2), *context);

        serializeType->assertForwardsResolved();

        Instance i = Instance::createAndInitialize(serializeType, [&](instance_ptr p) {
            PyEnsureGilReleased releaseTheGil;
            auto fieldAndWireType = buf.readFieldNumberAndWireType();
            serializeType->deserialize(p, buf, fieldAndWireType.second);
        });

        return PyInstance::extractPythonObject(i.data(), i.type());
    });
}

PyObject *decodeSerializedObject(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "validateSerializedObject takes 1 bytes argument");
        return NULL;
    }
    PyObjectHolder a1(PyTuple_GetItem(args, 0));

    if (!PyBytes_Check(a1)) {
        PyErr_SetString(PyExc_TypeError, "first argument to validateSerializedObject must be a bytes object");
        return NULL;
    }

    std::shared_ptr<SerializationContext> context(new NullSerializationContext());

    DeserializationBuffer buf((uint8_t*)PyBytes_AsString(a1), PyBytes_GET_SIZE((PyObject*)a1), *context);

    std::function<PyObject* (size_t)> decode = [&](size_t wireType) {
        if (wireType == WireType::VARINT) {
            return PyLong_FromLong(buf.readSignedVarint());
        }
        if (wireType == WireType::EMPTY) {
            return PyList_New(0);
        }
        if (wireType == WireType::BITS_32) {
            float f = buf.read<float>();
            Instance i((instance_ptr)&f, ::Float32::Make());

            return PyInstance::fromInstance(i);
        }
        if (wireType == WireType::BITS_64) {
            return PyFloat_FromDouble(buf.read<double>());
        }
        if (wireType == WireType::BYTES) {
            size_t count = buf.readUnsignedVarint();
            return buf.read_bytes_fun(count, [&](uint8_t* bytes) {
                return PyBytes_FromStringAndSize((const char*)bytes, count);
            });
        }
        if (wireType == WireType::SINGLE) {
            auto fieldAndWire = buf.readFieldNumberAndWireType();

            PyObjectStealer res(decode(fieldAndWire.second));

            PyObjectStealer result(PyList_New(0));
            PyObjectStealer key(PyLong_FromLong(fieldAndWire.first));
            PyObjectStealer tup(PyTuple_Pack(2, (PyObject*)key, (PyObject*)res));
            PyList_Append(result, tup);
            return incref(result);
        }
        if (wireType == WireType::BEGIN_COMPOUND) {
            PyObjectStealer result(PyList_New(0));

            while (true) {
                auto fieldAndWire = buf.readFieldNumberAndWireType();

                if (fieldAndWire.second == WireType::END_COMPOUND) {
                    return incref(result);
                }

                PyObjectStealer res(decode(fieldAndWire.second));

                PyObjectStealer key(PyLong_FromLong(fieldAndWire.first));
                //(PyObject*) are because this is a variadic function and doesn't cast the pointers
                //correctly
                PyObjectStealer tup(PyTuple_Pack(2, (PyObject*)key, (PyObject*)res));
                PyList_Append(result, tup);
            }
        }
        throw std::runtime_error("Invalid wire type encountered.");
    };

    return translateExceptionToPyObject([&](){
        auto msg = buf.readFieldNumberAndWireType();

        PyObjectStealer res(decode(msg.second));

        if (!buf.isDone()) {
            PyObjectStealer resDict(PyDict_New());
            PyDict_SetItemString(resDict, "result", res);
            PyDict_SetItemString(
                resDict,
                "remainingBytes",
                PyBytes_FromStringAndSize(
                    (const char*)PyBytes_AsString(a1) + buf.pos(),
                    PyBytes_GET_SIZE((PyObject*)a1) - buf.pos()
                )
            );

            return incref(resDict);
        }

        return incref(res);
    });
}

PyObject *validateSerializedObject(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "validateSerializedObject takes 1 bytes argument");
        return NULL;
    }
    PyObjectHolder a1(PyTuple_GetItem(args, 0));

    if (!PyBytes_Check(a1)) {
        PyErr_SetString(PyExc_TypeError, "first argument to validateSerializedObject must be a bytes object");
        return NULL;
    }

    std::shared_ptr<SerializationContext> context(new NullSerializationContext());

    DeserializationBuffer buf((uint8_t*)PyBytes_AsString(a1), PyBytes_GET_SIZE((PyObject*)a1), *context);

    return translateExceptionToPyObject([&](){
        try {
            buf.readMessageAndDiscard();

            if (buf.isDone()) {
                return incref(Py_None);
            }

            std::ostringstream remaining;
            remaining << PyBytes_GET_SIZE((PyObject*)a1) - buf.pos();
            throw std::runtime_error(
                "Stream had " +
                remaining.str() +
                " bytes remaining to process."
            );
        } catch(std::exception& e) {
            return PyUnicode_FromString(e.what());
        }
    });
}

PyObject *validateSerializedObjectStream(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "validateSerializedObjectStream takes 1 bytes argument");
        return NULL;
    }
    PyObjectHolder a1(PyTuple_GetItem(args, 0));

    if (!PyBytes_Check(a1)) {
        PyErr_SetString(PyExc_TypeError, "first argument to validateSerializedObjectStream must be a bytes object");
        return NULL;
    }

    std::shared_ptr<SerializationContext> context(new NullSerializationContext());

    DeserializationBuffer buf((uint8_t*)PyBytes_AsString(a1), PyBytes_GET_SIZE((PyObject*)a1), *context);

    return translateExceptionToPyObject([&](){
        try {
            while (!buf.isDone()) {
                buf.readMessageAndDiscard();
            }

            return incref(Py_None);
        } catch(std::exception& e) {
            return PyUnicode_FromString(e.what());
        }
    });
}

PyObject *deserializeStream(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 2 && PyTuple_Size(args) != 3) {
        PyErr_SetString(PyExc_TypeError, "deserialize takes 2 or 3 positional arguments");
        return NULL;
    }
    PyObjectHolder a1(PyTuple_GetItem(args, 0));
    PyObjectHolder a2(PyTuple_GetItem(args, 1));
    PyObjectHolder a3(PyTuple_Size(args) == 3 ? PyTuple_GetItem(args, 2) : nullptr);

    Type* serializeType = PyInstance::unwrapTypeArgToTypePtr(a1);

    if (!serializeType) {
        PyErr_SetString(PyExc_TypeError, "first argument to deserialize must be a native type object");
        return NULL;
    }
    if (!PyBytes_Check(a2)) {
        PyErr_SetString(PyExc_TypeError, "second argument to deserialize must be a bytes object");
        return NULL;
    }

    std::shared_ptr<SerializationContext> context(new NullSerializationContext());
    if (a3 && a3 != Py_None) {
        context.reset(new PythonSerializationContext(a3));
    }

    DeserializationBuffer buf((uint8_t*)PyBytes_AsString(a2), PyBytes_GET_SIZE((PyObject*)a2), *context);

    try {
        serializeType->assertForwardsResolved();

        TupleOfType* tupType = TupleOfType::Make(serializeType);

        tupType->assertForwardsResolved();

        Instance i = Instance::createAndInitialize(tupType, [&](instance_ptr p) {
            PyEnsureGilReleased releaseTheGil;

            tupType->constructorUnbounded(p, [&](instance_ptr tupElt, int index) {
                if (buf.isDone()) {
                    return false;
                }

                auto fieldAndWireType = buf.readFieldNumberAndWireType();
                serializeType->deserialize(tupElt, buf, fieldAndWireType.second);

                return true;
            });
        });

        return PyInstance::extractPythonObject(i.data(), i.type());
    } catch(std::exception& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    } catch(PythonExceptionSet& e) {
        return NULL;
    }
}

PyObject *isSimple(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "isSimple takes 1 positional argument");
        return NULL;
    }
    PyObjectHolder a1(PyTuple_GetItem(args, 0));

    Type* t = PyInstance::unwrapTypeArgToTypePtr(a1);

    if (!t) {
        PyErr_SetString(PyExc_TypeError, "first argument to 'isSimple' must be a native type object");
        return NULL;
    }

    return incref(t->isSimple() ? Py_True : Py_False);
}

PyObject *is_default_constructible(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "is_default_constructible takes 1 positional argument");
        return NULL;
    }
    PyObjectHolder a1(PyTuple_GetItem(args, 0));

    Type* t = PyInstance::unwrapTypeArgToTypePtr(a1);

    if (!t) {
        PyErr_SetString(PyExc_TypeError, "first argument to 'is_default_constructible' must be a native type object");
        return NULL;
    }

    return incref(t->is_default_constructible() ? Py_True : Py_False);
}

PyObject *all_alternatives_empty(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "all_alternatives_empty takes 1 positional argument");
        return NULL;
    }
    PyObjectHolder a1(PyTuple_GetItem(args, 0));

    Type* t = PyInstance::unwrapTypeArgToTypePtr(a1);

    if (!t) {
        PyErr_SetString(PyExc_TypeError, "first argument to 'all_alternatives_empty' must be a native type object");
        return NULL;
    }

    if (t->getTypeCategory() == Type::TypeCategory::catAlternative) {
        return incref(((Alternative*)t)->all_alternatives_empty() ? Py_True : Py_False);
    }
    else if (t->getTypeCategory() == Type::TypeCategory::catConcreteAlternative) {
        return incref(((ConcreteAlternative*)t)->getAlternative()->all_alternatives_empty() ? Py_True : Py_False);
    }
    else {
        PyErr_SetString(PyExc_TypeError, "first argument to 'all_alternatives_empty' must be an Alternative or ConcreteAlternative");
        return NULL;
    }

    return incref(((Alternative*)t)->all_alternatives_empty() ? Py_True : Py_False);
}

PyObject *wantsToDefaultConstruct(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "wantsToDefaultConstruct takes 1 positional argument");
        return NULL;
    }
    PyObjectHolder a1(PyTuple_GetItem(args, 0));

    Type* t = PyInstance::unwrapTypeArgToTypePtr(a1);

    if (!t) {
        PyErr_SetString(PyExc_TypeError, "first argument to 'wantsToDefaultConstruct' must be a native type object");
        return NULL;
    }

    return incref(HeldClass::wantsToDefaultConstruct(t) ? Py_True : Py_False);
}

PyObject *bytecount(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "bytecount takes 1 positional argument");
        return NULL;
    }
    PyObjectHolder a1(PyTuple_GetItem(args, 0));

    Type* t = PyInstance::unwrapTypeArgToTypePtr(a1);

    if (!t) {
        PyErr_SetString(PyExc_TypeError, "first argument to 'bytecount' must be a native type object");
        return NULL;
    }

    return PyLong_FromLong(t->bytecount());
}

PyObject *canConstructFrom(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "canConstructFrom takes 3 positional arguments");
        return NULL;
    }
    PyObjectHolder a1(PyTuple_GetItem(args, 0));
    PyObjectHolder a2(PyTuple_GetItem(args, 1));

    Type* t1 = PyInstance::unwrapTypeArgToTypePtr(a1);

    if (!t1) {
        PyErr_SetString(PyExc_TypeError, "first argument to 'canConstructFrom' must be a native type object");
        return NULL;
    }

    Type* t2 = PyInstance::unwrapTypeArgToTypePtr(a2);

    if (!t2) {
        PyErr_SetString(PyExc_TypeError, "second argument to 'canConstructFrom' must be a native type object");
        return NULL;
    }

    Maybe can = t1->canConstructFrom(t2, PyTuple_GetItem(args, 2) == Py_True);

    if (can == Maybe::Maybe) {
        return PyUnicode_FromString("Maybe");
    }

    if (can == Maybe::True) {
        return incref(Py_True);
    }

    return incref(Py_False);
}

PyObject *disableNativeDispatch(PyObject* nullValue, PyObject* args) {
    native_dispatch_disabled++;
    return incref(Py_None);
}

PyObject *enableNativeDispatch(PyObject* nullValue, PyObject* args) {
    native_dispatch_disabled--;
    return incref(Py_None);
}

PyObject *isDispatchEnabled(PyObject* nullValue, PyObject* args) {
    return incref(native_dispatch_disabled ? Py_False : Py_True);
}

PyObject *installNativeFunctionPointer(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 5) {
        PyErr_SetString(PyExc_TypeError, "installNativeFunctionPointer takes 5 positional arguments");
        return NULL;
    }
    PyObjectHolder a1(PyTuple_GetItem(args, 0));
    PyObjectHolder a2(PyTuple_GetItem(args, 1));
    PyObjectHolder a3(PyTuple_GetItem(args, 2));
    PyObjectHolder a4(PyTuple_GetItem(args, 3));
    PyObjectHolder a5(PyTuple_GetItem(args, 4));

    Type* t1 = PyInstance::unwrapTypeArgToTypePtr(a1);

    if (!t1 || t1->getTypeCategory() != Type::TypeCategory::catFunction) {
        PyErr_SetString(PyExc_TypeError, "first argument to 'installNativeFunctionPointer' must be a Function");
        return NULL;
    }

    if (!PyLong_Check(a2)) {
        PyErr_SetString(PyExc_TypeError, "second argument to 'installNativeFunctionPointer' must be an integer 'index'");
        return NULL;
    }

    if (!PyLong_Check(a3)) {
        PyErr_SetString(PyExc_TypeError, "third argument to 'installNativeFunctionPointer' must be an integer function pointer");
        return NULL;
    }


    Type* returnType = PyInstance::unwrapTypeArgToTypePtr(a4);

    if (!returnType) {
        PyErr_SetString(PyExc_TypeError, "fourth argument to 'installNativeFunctionPointer' must be a type object (return type)");
        return NULL;
    }

    std::vector<Type*> argTypes;

    if (!unpackTupleToTypes(a5, argTypes)) {
        return NULL;
    }

    Function* f = (Function*)t1;

    int index = PyLong_AsLong(a2);
    size_t ptr = PyLong_AsSize_t(a3);

    if (index < 0 || index >= f->getOverloads().size()) {
        PyErr_SetString(PyExc_TypeError, "index is out of bounds");
        return NULL;
    }

    f->addCompiledSpecialization(index,(compiled_code_entrypoint)ptr, returnType, argTypes);

    return incref(Py_None);
}

PyObject *touchCompiledSpecializations(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "touchCompiledSpecializations takes 2 positional arguments");
        return NULL;
    }
    PyObjectHolder a1(PyTuple_GetItem(args, 0));
    PyObjectHolder a2(PyTuple_GetItem(args, 1));

    Type* t1 = PyInstance::unwrapTypeArgToTypePtr(a1);

    if (!t1 || t1->getTypeCategory() != Type::TypeCategory::catFunction) {
        PyErr_SetString(PyExc_TypeError, "first argument to 'touchCompiledSpecializations' must be a Function");
        return NULL;
    }

    if (!PyLong_Check(a2)) {
        PyErr_SetString(PyExc_TypeError, "second argument to 'touchCompiledSpecializations' must be an integer 'index'");
        return NULL;
    }

    Function* f = (Function*)t1;

    int index = PyLong_AsLong(a2);

    if (index < 0 || index >= f->getOverloads().size()) {
        PyErr_SetString(PyExc_TypeError, "index is out of bounds");
        return NULL;
    }

    f->touchCompiledSpecializations(index);

    return incref(Py_None);
}

PyObject *isRecursive(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "isRecursive takes 1 positional argument");
        return NULL;
    }
    PyObjectHolder a1(PyTuple_GetItem(args, 0));

    MutuallyRecursiveTypeGroup* group = nullptr;

    return translateExceptionToPyObject([&]() {
        if (PyType_Check(a1)) {
            Type* typeArg = PyInstance::extractTypeFrom(a1);
            if (typeArg) {
                group = typeArg->getRecursiveTypeGroup();
            }
        }

        if (!group) {
            group = MutuallyRecursiveTypeGroup::pyObjectGroupHeadAndIndex(a1).first;
        }

        if (!group) {
            return incref(Py_False);
        }

        return incref(group->getIndexToObject().size() > 1 ? Py_True : Py_False);
    });
}

PyObject *typesAndObjectsVisibleToCompilerFrom(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "typesAndObjectsVisibleToCompilerFrom takes 1 positional argument");
        return NULL;
    }
    PyObjectHolder a1(PyTuple_GetItem(args, 0));

    return translateExceptionToPyObject([&]() {
        std::vector<TypeOrPyobj> visible;
        Type* typeArg = nullptr;

        if (PyType_Check(a1) && (typeArg=PyInstance::extractTypeFrom(a1))) {
            MutuallyRecursiveTypeGroup::visibleFrom(typeArg, visible);
        } else {
            MutuallyRecursiveTypeGroup::visibleFrom((PyObject*)a1, visible);
        }

        PyObjectStealer res(PyList_New(0));

        for (auto& t: visible) {
            PyList_Append(
                res,
                t.type() ?
                    (PyObject*)PyInstance::typeObj(t.type())
                :   t.pyobj()
            );
        }

        return incref(res);
    });
}

PyObject *recursiveTypeGroup(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "recursiveTypeGroup takes 1 positional argument");
        return NULL;
    }
    PyObjectHolder a1(PyTuple_GetItem(args, 0));

    MutuallyRecursiveTypeGroup* group = nullptr;

    return translateExceptionToPyObject([&]() {
        if (PyType_Check(a1)) {
            Type* typeArg = PyInstance::extractTypeFrom(a1);
            if (typeArg) {
                group = typeArg->getRecursiveTypeGroup();
            }
        }

        if (!group) {
            group = MutuallyRecursiveTypeGroup::pyObjectGroupHeadAndIndex(a1).first;
        }

        if (!group) {
            return incref(Py_None);
        }

        PyObjectStealer res(PyList_New(0));

        for (auto ixAndType: group->getIndexToObject()) {
            PyList_Append(
                res,
                ixAndType.second.type() ?
                    (PyObject*)PyInstance::typeObj(ixAndType.second.type())
                :   ixAndType.second.pyobj()
            );
        }

        return incref(res);
    });
}

PyObject *referencedTypes(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "referencedTypes takes 1 positional argument");
        return NULL;
    }
    PyObjectHolder a1(PyTuple_GetItem(args, 0));

    Type* t1 = PyInstance::unwrapTypeArgToTypePtr(a1);

    if (!t1) {
        PyErr_SetString(PyExc_TypeError, "first argument to 'referencedTypes' must be a native type object");
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        PyObjectStealer res(PyList_New(0));

        t1->visitReferencedTypes([&](Type* refType) {
            PyList_Append(
                res,
                (PyObject*)PyInstance::typeObj(refType)
            );
        });

        return incref(res);
    });
}

PyObject *isBinaryCompatible(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "isBinaryCompatible takes 2 positional arguments");
        return NULL;
    }
    PyObjectHolder a1(PyTuple_GetItem(args, 0));
    PyObjectHolder a2(PyTuple_GetItem(args, 1));

    Type* t1 = PyInstance::unwrapTypeArgToTypePtr(a1);
    Type* t2 = PyInstance::unwrapTypeArgToTypePtr(a2);

    if (!t1) {
        PyErr_SetString(PyExc_TypeError, "first argument to 'isBinaryCompatible' must be a native type object");
        return NULL;
    }
    if (!t2) {
        PyErr_SetString(PyExc_TypeError, "second argument to 'isBinaryCompatible' must be a native type object");
        return NULL;
    }

    return incref(t1->isBinaryCompatibleWith(t2) ? Py_True : Py_False);
}

PyObject *MakeForward(PyObject* nullValue, PyObject* args) {
    int num_args = PyTuple_Size(args);

    PyThreadState * ts = PyThreadState_Get();
    std::string moduleName;
    PyObject* pyModuleName;

    if (ts->frame && ts->frame->f_globals &&
            (pyModuleName = PyDict_GetItemString(ts->frame->f_globals, "__name__"))) {
        if (PyUnicode_Check(pyModuleName)) {
            moduleName = PyUnicode_AsUTF8(pyModuleName);
        }
    }

    if (num_args > 1 || !PyUnicode_Check(PyTuple_GetItem(args,0))) {
        PyErr_SetString(PyExc_TypeError, "Forward takes a zero or one string positional arguments.");
        return NULL;
    }

    if (num_args == 1) {
        std::string name = PyUnicode_AsUTF8(PyTuple_GetItem(args,0));

        if (moduleName.size()) {
            name = moduleName + "." + name;
        }

        return incref((PyObject*)PyInstance::typeObj(
            ::Forward::Make(name)
            ));
    } else {
        return incref((PyObject*)PyInstance::typeObj(
            ::Forward::Make()
            ));
    }
}

PyObject* buildPyFunctionObject(PyObject* nullValue, PyObject* args, PyObject* kwargs) {
    static const char *kwlist[] = {"code", "globals", "closure", NULL};

    PyObject* pyCode;
    PyObject* pyGlobals;
    PyObject* pyClosure;

    return translateExceptionToPyObject([&]() {
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO", (char**)kwlist, &pyCode, &pyGlobals, &pyClosure)) {
            throw PythonExceptionSet();
        }

        if (!PyCode_Check(pyCode)) {
            throw std::runtime_error("'code' argument to buildPyFunctionObject must be a Code object.");
        }

        if (!PyDict_Check(pyGlobals)) {
            throw std::runtime_error("'globals' argument to buildPyFunctionObject must be a dict.");
        }

        if (!PyTuple_Check(pyClosure)) {
            throw std::runtime_error("'closure' argument to buildPyFunctionObject must be a tuple.");
        }

        iterate(pyClosure, [&](PyObject* shouldBeCell) {
            if (!PyCell_Check(shouldBeCell)) {
                throw std::runtime_error("closure elements must be cells.");
            }
        });

        PyObject* result = PyFunction_New(pyCode, pyGlobals);
        if (!result) {
            throw PythonExceptionSet();
        }

        if (PyFunction_SetClosure(result, pyClosure) == -1) {
            throw PythonExceptionSet();
        }

        return result;
    });
}

PyObject *MakeAlternativeType(PyObject* nullValue, PyObject* args, PyObject* kwargs) {
    if (PyTuple_Size(args) != 1 || !PyUnicode_Check(PyTuple_GetItem(args,0))) {
        PyErr_SetString(PyExc_TypeError, "Alternative takes a single string positional argument.");
        return NULL;
    }

    std::string name = PyUnicode_AsUTF8(PyTuple_GetItem(args,0));

    std::vector<std::pair<std::string, NamedTuple*> > definitions;

    std::map<std::string, Function*> functions;

    PyObject *key, *value;
    Py_ssize_t pos = 0;

    while (kwargs && PyDict_Next(kwargs, &pos, &key, &value)) {
        assert(PyUnicode_Check(key));

        std::string fieldName(PyUnicode_AsUTF8(key));

        if (PyFunction_Check(value)) {
            functions[fieldName] = PyFunctionInstance::convertPythonObjectToFunctionType(key, value, true, false);

            if (functions[fieldName] == nullptr) {
                //error code is already set
                return nullptr;
            }
        }
        else {
            if (!PyDict_Check(value)) {
                PyErr_SetString(PyExc_TypeError, "Alternative members must be initialized with dicts.");
                return NULL;
            }

            PyObject* ntPyPtr = MakeNamedTupleType(nullptr, nullptr, value);
            if (!ntPyPtr) {
                return NULL;
            }

            NamedTuple* ntPtr = (NamedTuple*)PyInstance::extractTypeFrom((PyTypeObject*)ntPyPtr);

            assert(ntPtr);

            definitions.push_back(std::make_pair(fieldName, ntPtr));
        }
    };

    static_assert(PY_MAJOR_VERSION >= 3, "typed_python is a python3 project only");

    PyThreadState * ts = PyThreadState_Get();
    std::string moduleName;
    PyObject* pyModuleName;

    if (ts->frame && ts->frame->f_globals &&
            (pyModuleName = PyDict_GetItemString(ts->frame->f_globals, "__name__"))) {
        if (PyUnicode_Check(pyModuleName)) {
            moduleName = PyUnicode_AsUTF8(pyModuleName);
        }
    }

    if (PY_MINOR_VERSION <= 5) {
        //we cannot rely on the ordering of 'kwargs' here because of the python version, so
        //we sort it. this will be a problem for anyone running some processes using different
        //python versions that share python code.
        std::sort(definitions.begin(), definitions.end());
    }

    return incref((PyObject*)PyInstance::typeObj(
        ::Alternative::Make(name, moduleName, definitions, functions)
    ));
}

PyObject *getTypePointer(PyObject* nullValue, PyObject* args) {
    if (PyTuple_Size(args) != 1 || !PyInstance::unwrapTypeArgToTypePtr(PyTuple_GetItem(args,0))) {
        PyErr_SetString(PyExc_TypeError, "getTypePointer takes 1 positional argument (a type)");
        return NULL;
    }

    Type* type = PyInstance::unwrapTypeArgToTypePtr(PyTuple_GetItem(args,0));

    return PyLong_FromLong((uint64_t)type);
}

static PyMethodDef module_methods[] = {
    {"TypeFor", (PyCFunction)MakeTypeFor, METH_VARARGS, NULL},
    {"serialize", (PyCFunction)serialize, METH_VARARGS, NULL},
    {"deserialize", (PyCFunction)deserialize, METH_VARARGS, NULL},
    {"decodeSerializedObject", (PyCFunction)decodeSerializedObject, METH_VARARGS, NULL},
    {"validateSerializedObject", (PyCFunction)validateSerializedObject, METH_VARARGS, NULL},
    {"validateSerializedObjectStream", (PyCFunction)validateSerializedObjectStream, METH_VARARGS, NULL},
    {"identityHash", (PyCFunction)identityHash, METH_VARARGS, NULL},
    {"serializeStream", (PyCFunction)serializeStream, METH_VARARGS, NULL},
    {"deserializeStream", (PyCFunction)deserializeStream, METH_VARARGS, NULL},
    {"is_default_constructible", (PyCFunction)is_default_constructible, METH_VARARGS, NULL},
    {"buildPyFunctionObject", (PyCFunction)buildPyFunctionObject, METH_VARARGS | METH_KEYWORDS, NULL},
    {"isSimple", (PyCFunction)isSimple, METH_VARARGS, NULL},
    {"canConstructFrom", (PyCFunction)canConstructFrom, METH_VARARGS, NULL},
    {"bytecount", (PyCFunction)bytecount, METH_VARARGS, NULL},
    {"isBinaryCompatible", (PyCFunction)isBinaryCompatible, METH_VARARGS, NULL},
    {"Forward", (PyCFunction)MakeForward, METH_VARARGS, NULL},
    {"recursiveTypeGroup", (PyCFunction)recursiveTypeGroup, METH_VARARGS, NULL},
    {"typesAndObjectsVisibleToCompilerFrom", (PyCFunction)typesAndObjectsVisibleToCompilerFrom, METH_VARARGS, NULL},
    {"isRecursive", (PyCFunction)isRecursive, METH_VARARGS, NULL},
    {"referencedTypes", (PyCFunction)referencedTypes, METH_VARARGS, NULL},
    {"wantsToDefaultConstruct", (PyCFunction)wantsToDefaultConstruct, METH_VARARGS, NULL},
    {"all_alternatives_empty", (PyCFunction)all_alternatives_empty, METH_VARARGS, NULL},
    {"installNativeFunctionPointer", (PyCFunction)installNativeFunctionPointer, METH_VARARGS, NULL},
    {"touchCompiledSpecializations", (PyCFunction)touchCompiledSpecializations, METH_VARARGS, NULL},
    {"disableNativeDispatch", (PyCFunction)disableNativeDispatch, METH_VARARGS, NULL},
    {"enableNativeDispatch", (PyCFunction)enableNativeDispatch, METH_VARARGS, NULL},
    {"isDispatchEnabled", (PyCFunction)isDispatchEnabled, METH_VARARGS, NULL},
    {"refcount", (PyCFunction)refcount, METH_VARARGS, NULL},
    {"getOrSetTypeResolver", (PyCFunction)getOrSetTypeResolver, METH_VARARGS, NULL},
    {"pointerTo", (PyCFunction)pointerTo, METH_VARARGS, NULL},
    {"pyInstanceHeldObjectAddress", (PyCFunction)pyInstanceHeldObjectAddress, METH_VARARGS, NULL},
    {"createPyCell", (PyCFunction)createPyCell, METH_VARARGS, NULL},
    {"copy", (PyCFunction)copyRefTo, METH_VARARGS, NULL},
    {"refTo", (PyCFunction)refTo, METH_VARARGS, NULL},
    {"getTypePointer", (PyCFunction)getTypePointer, METH_VARARGS, NULL},
    {"_vtablePointer", (PyCFunction)getVTablePointer, METH_VARARGS, NULL},
    {"allocateClassMethodDispatch", (PyCFunction)allocateClassMethodDispatch, METH_VARARGS | METH_KEYWORDS, NULL},
    {"getNextUnlinkedClassMethodDispatch", (PyCFunction)getNextUnlinkedClassMethodDispatch, METH_VARARGS | METH_KEYWORDS, NULL},
    {"getClassMethodDispatchSignature", (PyCFunction)getClassMethodDispatchSignature, METH_VARARGS | METH_KEYWORDS, NULL},
    {"installClassMethodDispatch", (PyCFunction)installClassMethodDispatch, METH_VARARGS | METH_KEYWORDS, NULL},
    {"installClassDestructor", (PyCFunction)installClassDestructor, METH_VARARGS | METH_KEYWORDS, NULL},
    {"classGetDispatchIndex", (PyCFunction)classGetDispatchIndex, METH_VARARGS | METH_KEYWORDS, NULL},
    {"getDispatchIndexForType", (PyCFunction)getDispatchIndexForType, METH_VARARGS | METH_KEYWORDS, NULL},
    {"prepareArgumentToBePassedToCompiler", (PyCFunction)prepareArgumentToBePassedToCompiler, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_types",
    .m_doc = NULL,
    .m_size = 0,
    .m_methods = module_methods,
    .m_slots = NULL,
    .m_traverse = NULL,
    .m_clear = NULL,
    .m_free = NULL
};

void updateTypeRepForType(Type* type, PyTypeObject* pyType) {
    //deliberately leak the name.
    pyType->tp_name = (new std::string(type->name()))->c_str();

    PyInstance::mirrorTypeInformationIntoPyType(type, pyType);
}

PyMODINIT_FUNC
PyInit__types(void)
{
    // initialize unicode property table, for StringType
    initialize_uprops();

    //initialize numpy. This is only OK because all the .cpp files get
    //glommed together in a single file. If we were to change that behavior,
    //then additional steps must be taken as per the API documentation.
    import_array();

    PyObject *module = PyModule_Create(&moduledef);

    PyModule_AddObject(module, "Type", (PyObject*)incref(PyInstance::allTypesBaseType()));
    PyModule_AddObject(module, "ListOf", (PyObject*)incref(PyInstance::typeCategoryBaseType(Type::TypeCategory::catListOf)));
    PyModule_AddObject(module, "Int8", (PyObject*)incref(PyInstance::typeCategoryBaseType(Type::TypeCategory::catInt8)));
    PyModule_AddObject(module, "Int16", (PyObject*)incref(PyInstance::typeCategoryBaseType(Type::TypeCategory::catInt16)));
    PyModule_AddObject(module, "Int32", (PyObject*)incref(PyInstance::typeCategoryBaseType(Type::TypeCategory::catInt32)));
    PyModule_AddObject(module, "UInt8", (PyObject*)incref(PyInstance::typeCategoryBaseType(Type::TypeCategory::catUInt8)));
    PyModule_AddObject(module, "UInt16", (PyObject*)incref(PyInstance::typeCategoryBaseType(Type::TypeCategory::catUInt16)));
    PyModule_AddObject(module, "UInt32", (PyObject*)incref(PyInstance::typeCategoryBaseType(Type::TypeCategory::catUInt32)));
    PyModule_AddObject(module, "UInt64", (PyObject*)incref(PyInstance::typeCategoryBaseType(Type::TypeCategory::catUInt64)));
    PyModule_AddObject(module, "Float32", (PyObject*)incref(PyInstance::typeCategoryBaseType(Type::TypeCategory::catFloat32)));
    PyModule_AddObject(module, "TupleOf", (PyObject*)incref(PyInstance::typeCategoryBaseType(Type::TypeCategory::catTupleOf)));
    PyModule_AddObject(module, "PointerTo", (PyObject*)incref(PyInstance::typeCategoryBaseType(Type::TypeCategory::catPointerTo)));
    PyModule_AddObject(module, "RefTo", (PyObject*)incref(PyInstance::typeCategoryBaseType(Type::TypeCategory::catRefTo)));
    PyModule_AddObject(module, "Tuple", (PyObject*)incref(PyInstance::typeCategoryBaseType(Type::TypeCategory::catTuple)));
    PyModule_AddObject(module, "NamedTuple", (PyObject*)incref(PyInstance::typeCategoryBaseType(Type::TypeCategory::catNamedTuple)));
    PyModule_AddObject(module, "OneOf", (PyObject*)incref(PyInstance::typeCategoryBaseType(Type::TypeCategory::catOneOf)));
    PyModule_AddObject(module, "Set", (PyObject*)incref(PyInstance::typeCategoryBaseType(Type::TypeCategory::catSet)));
    PyModule_AddObject(module, "Dict", (PyObject*)incref(PyInstance::typeCategoryBaseType(Type::TypeCategory::catDict)));
    PyModule_AddObject(module, "ConstDict", (PyObject*)incref(PyInstance::typeCategoryBaseType(Type::TypeCategory::catConstDict)));
    PyModule_AddObject(module, "Alternative", (PyObject*)incref(PyInstance::typeCategoryBaseType(Type::TypeCategory::catAlternative)));
    PyModule_AddObject(module, "Value", (PyObject*)incref(PyInstance::typeCategoryBaseType(Type::TypeCategory::catValue)));
    PyModule_AddObject(module, "Class", (PyObject*)incref(PyInstance::typeCategoryBaseType(Type::TypeCategory::catClass)));
    PyModule_AddObject(module, "Function", (PyObject*)incref(PyInstance::typeCategoryBaseType(Type::TypeCategory::catFunction)));
    PyModule_AddObject(module, "BoundMethod", (PyObject*)incref(PyInstance::typeCategoryBaseType(Type::TypeCategory::catBoundMethod)));
    PyModule_AddObject(module, "EmbeddedMessage", (PyObject*)incref(PyInstance::typeCategoryBaseType(Type::TypeCategory::catEmbeddedMessage)));
    PyModule_AddObject(module, "TypedCell", (PyObject*)incref(PyInstance::typeCategoryBaseType(Type::TypeCategory::catTypedCell)));
    PyModule_AddObject(module, "PyCell", (PyObject*)incref(PyInstance::typeCategoryBaseType(Type::TypeCategory::catPyCell)));
    PyModule_AddObject(module, "PythonObjectOfType", (PyObject*)incref(PyInstance::typeCategoryBaseType(Type::TypeCategory::catPythonObjectOfType)));
    PyModule_AddObject(module, "PythonSubclass", (PyObject*)incref(PyInstance::typeCategoryBaseType(Type::TypeCategory::catPythonSubclass)));


    if (module == NULL)
        return NULL;

    return module;
}
