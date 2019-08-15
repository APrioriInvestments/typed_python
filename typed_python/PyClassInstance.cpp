/******************************************************************************
   Copyright 2017-2019 Nativepython Authors

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

#include "PyClassInstance.hpp"

Class* PyClassInstance::type() {
    return (Class*)extractTypeFrom(((PyObject*)this)->ob_type);
}

bool PyClassInstance::pyValCouldBeOfTypeConcrete(Class* type, PyObject* pyRepresentation, bool isExplicit) {
    Type* argType = extractTypeFrom(pyRepresentation->ob_type);

    return argType && ((argType == type) || argType->isSubclassOf(type));
}

PyObject* PyClassInstance::extractPythonObjectConcrete(Type* eltType, instance_ptr data) {
    // we need to make sure we always produce python objects with a 0 classDispatchOffset.
    // the standard 'extractPythonObjectConcrete' assumes you can simply pick the concrete subclass
    // and copy construct as if the binary layouts were identical. But because we have this
    // odd multi-vtable thing, we need to ensure we produce the proper classDispatch. Our
    // standard is that the python layer only ever sees actual proper classes
    Type* concreteT = eltType->pickConcreteSubclass(data);

    return PyInstance::initialize(concreteT, [&](instance_ptr selfData) {
        Class::initializeInstance(selfData, Class::instanceToLayout(data), 0)->refcount++;
    });
}

void PyClassInstance::copyConstructFromPythonInstanceConcrete(Class* eltType, instance_ptr tgt, PyObject* pyRepresentation, bool isExplicit) {
    Type* argType = extractTypeFrom(pyRepresentation->ob_type);

    if (argType && argType->isSubclassOf(eltType)) {
        instance_ptr argDataPtr = ((PyInstance*)pyRepresentation)->dataPtr();

        // arg data ptrs here should _always_ have the 0 classDispatchOffset, because
        // we never allow the python representation of a class to be a subclass.
        if (Class::instanceToDispatchTableIndex(argDataPtr) != 0) {
            throw std::runtime_error("Corrupt class instance. Expected a zero classDispatchOffset.");
        }

        Class* childType = (Class*)argType;

        Class::initializeInstance(
            tgt,
            Class::instanceToLayout(argDataPtr),
            childType->getHeldClass()->getMroIndex(eltType->getHeldClass())
        )->refcount++;

        return;
    }

    PyInstance::copyConstructFromPythonInstanceConcrete(eltType, tgt, pyRepresentation, isExplicit);
}

void PyClassInstance::initializeClassWithDefaultArguments(Class* cls, uint8_t* data, PyObject* args, PyObject* kwargs) {
    if (PyTuple_Size(args)) {
        PyErr_Format(PyExc_TypeError,
            "default __init__ for instances of '%s' doesn't accept positional arguments.",
            cls->name().c_str()
            );
        throw PythonExceptionSet();
    }

    if (!kwargs) {
        return;
    }

    PyObject *key, *value;
    Py_ssize_t pos = 0;

    while (PyDict_Next(kwargs, &pos, &key, &value)) {
        int res = classInstanceSetAttributeFromPyObject(cls, data, key, value);

        if (res != 0) {
            throw PythonExceptionSet();
        }
    }
}

/**
 *  Return 0 if successful and -1 if it failed
 */
// static
int PyClassInstance::classInstanceSetAttributeFromPyObject(Class* cls, instance_ptr data, PyObject* attrName, PyObject* attrVal) {
    int i = cls->memberNamed(PyUnicode_AsUTF8(attrName));

    if (i < 0) {
        auto it = cls->getClassMembers().find(PyUnicode_AsUTF8(attrName));
        if (it == cls->getClassMembers().end()) {
            PyErr_Format(
                PyExc_AttributeError,
                "'%s' object has no attribute '%S' and cannot add attributes to instances of this type",
                cls->name().c_str(), attrName
            );
        } else {
            PyErr_Format(
                PyExc_AttributeError,
                "Cannot modify read-only class member '%S' of instance of type '%s'",
                attrName, cls->name().c_str()
            );
        }
        return -1;
    }

    Type* eltType = cls->getMemberType(i);

    Type* attrType = extractTypeFrom(attrVal->ob_type);

    if (eltType == attrType) {
        PyInstance* item_w = (PyInstance*)attrVal;

        cls->setAttribute(data, i, item_w->dataPtr());

        return 0;
    } else {
        instance_ptr tempObj = (instance_ptr)malloc(eltType->bytecount());
        try {
            copyConstructFromPythonInstance(eltType, tempObj, attrVal, true /* set isExplicit to True */ );
        } catch(PythonExceptionSet& e) {
            free(tempObj);
            return -1;
        } catch(std::exception& e) {
            free(tempObj);
            PyErr_SetString(PyExc_TypeError, e.what());
            return -1;
        }


        cls->setAttribute(data, i, tempObj);

        eltType->destroy(tempObj);
        free(tempObj);

        return 0;
    }
}

PyObject* PyClassInstance::pyUnaryOperatorConcrete(const char* op, const char* opErr) {
    std::pair<bool, PyObject*> res = callMemberFunction(op);

    if (!res.first) {
        return PyInstance::pyUnaryOperatorConcrete(op, opErr);
    }

    return res.second;
}

PyObject* PyClassInstance::pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr) {
    std::pair<bool, PyObject*> res = callMemberFunction(op, rhs);

    if (!res.first) {
        return PyInstance::pyOperatorConcrete(rhs, op, opErr);
    }

    return res.second;
}

PyObject* PyClassInstance::pyOperatorConcreteReverse(PyObject* lhs, const char* op, const char* opErr) {
    char buf[50];
    strncpy(buf+1, op, 45);
    buf[0] = '_';
    buf[2] = 'r';

    std::pair<bool, PyObject*> res = callMemberFunction(buf, lhs);

    if (!res.first) {
        return PyInstance::pyOperatorConcreteReverse(lhs, buf, opErr);
    }

    return res.second;
}

PyObject* PyClassInstance::pyTernaryUnaryOperatorConcrete(PyObject* rhs, PyObject* ternaryArg, const char* op, const char* opErr) {
    if (ternaryArg == Py_None) {
        //if you pass 'None' as the third argument, python calls your class
        //__pow__ function with two arguments. This is the behavior for
        //'instance ** b' as well as if you write 'pow(instance,b,None)'
        return pyOperatorConcrete(rhs, op, opErr);
    }

    std::pair<bool, PyObject*> res = callMemberFunction(op, rhs, ternaryArg);

    if (!res.first) {
        return PyInstance::pyTernaryOperatorConcrete(rhs, ternaryArg, op, opErr);
    }

    return res.second;
}

int PyClassInstance::pyInquiryConcrete(const char* op, const char* opErrRep) {
    // op == '__bool__'
    std::pair<bool, PyObject*> p = callMemberFunction("__bool__", nullptr, nullptr);
    if (!p.first) {
        p = callMemberFunction("__len__", nullptr, nullptr);
        // if neither __bool__ nor __len__ is available, return True
        if (!p.first)
            return 1;
    }
    return PyObject_IsTrue(p.second);
}

std::pair<bool, PyObject*> PyClassInstance::callMemberFunction(const char* name, PyObject* arg0, PyObject* arg1) {
    auto it = type()->getMemberFunctions().find(name);

    if (it == type()->getMemberFunctions().end()) {
        return std::make_pair(false, (PyObject*)nullptr);
    }

    Function* method = it->second;

    int argCount = 1;
    if (arg0) {
        argCount += 1;
    }
    if (arg1) {
        argCount += 1;
    }

    PyObjectStealer targetArgTuple(PyTuple_New(argCount));

    PyTuple_SetItem(targetArgTuple, 0, incref((PyObject*)this)); //steals a reference

    if (arg0) {
        PyTuple_SetItem(targetArgTuple, 1, incref(arg0)); //steals a reference
    }

    if (arg1) {
        PyTuple_SetItem(targetArgTuple, 2, incref(arg1)); //steals a reference
    }

    auto res = PyFunctionInstance::tryToCallAnyOverload(method, nullptr, targetArgTuple, nullptr);
    if (res.first) {
        return res;
    }
    PyErr_Format(
        PyExc_TypeError,
        "'%s.%s' cannot find a valid overload with these arguments",
        type()->name().c_str(),
        name
    );
    return res;
}

PyObject* PyClassInstance::mp_subscript_concrete(PyObject* item) {
    std::pair<bool, PyObject*> res = callMemberFunction("__getitem__", item);

    if (res.first) {
        return res.second;
    }

    return PyInstance::mp_subscript_concrete(item);
}

Py_ssize_t PyClassInstance::mp_and_sq_length_concrete() {
    std::pair<bool, PyObject*> res = callMemberFunction("__len__");

    if (!res.first) {
        return PyInstance::mp_and_sq_length_concrete();
    }

    if (!res.second) {
        return -1;
    }

    if (!PyLong_Check(res.second)) {
        PyErr_Format(
            PyExc_TypeError,
            "'%s.__len__' returned an object of type %s",
            type()->name().c_str(),
            res.second->ob_type->tp_name
            );
        return -1;
    }

    long result = PyLong_AsLong(res.second);

    if (result < 0) {
        PyErr_Format(PyExc_ValueError, "'__len__()' should return >= 0");
        return -1;
    }

    return result;
}

void PyClassInstance::constructFromPythonArgumentsConcrete(Class* classT, uint8_t* data, PyObject* args, PyObject* kwargs) {
    classT->constructor(data);

    auto it = classT->getMemberFunctions().find("__init__");
    if (it == classT->getMemberFunctions().end()) {
        //run the default constructor
        PyClassInstance::initializeClassWithDefaultArguments(classT, data, args, kwargs);
        return;
    }

    Function* initMethod = it->second;

    PyObject* selfAsObject = PyInstance::initialize(classT, [&](instance_ptr selfData) {
        classT->copy_constructor(selfData, data);
    });

    PyObjectStealer targetArgTuple(PyTuple_New(PyTuple_Size(args)+1));

    PyTuple_SetItem(targetArgTuple, 0, selfAsObject); //steals the reference to the new 'selfAsObject'

    for (long k = 0; k < PyTuple_Size(args); k++) {
        PyTuple_SetItem(targetArgTuple, k+1, incref(PyTuple_GetItem(args, k))); //have to incref because of stealing
    }

    auto res = PyFunctionInstance::tryToCallAnyOverload(initMethod, nullptr, targetArgTuple, kwargs);
    if (!res.first) {
        throw std::runtime_error("Cannot find a valid overload of __init__ with these arguments.");
    }

    if (res.second) {
        decref(res.second);
    } else {
        throw PythonExceptionSet();
    }
}

PyObject* PyClassInstance::tp_getattr_concrete(PyObject* pyAttrName, const char* attrName) {
    int index = type()->getMemberIndex(attrName);

    if (index >= 0) {
        Type* eltType = type()->getMemberType(index);

        if (!type()->checkInitializationFlag(dataPtr(), index)) {
            PyErr_Format(
                PyExc_AttributeError,
                "Attribute '%S' is not initialized",
                pyAttrName
            );
            return NULL;
        }

        return extractPythonObject(type()->eltPtr(dataPtr(), index), eltType);
    }

    BoundMethod* method = type()->getMemberFunctionMethodType(attrName);
    if (method) {
        return PyInstance::initializePythonRepresentation(method, [&](instance_ptr data) {
            method->copy_constructor(data, dataPtr());
        });
    }

    {
        auto it = type()->getPropertyFunctions().find(attrName);
        if (it != type()->getPropertyFunctions().end()) {
            std::pair<bool, PyObject*> res = PyFunctionInstance::tryToCall(it->second, (PyObject*)this);
            if (res.first) {
                return res.second;
            }

            PyErr_Format(
                PyExc_TypeError,
                "Found a property for %s but failed to call it with 'self'",
                attrName
                );
            return NULL;
        }
    }

    {
        auto it = type()->getClassMembers().find(attrName);
        if (it != type()->getClassMembers().end()) {
            return incref(it->second);
        }
    }

    return PyInstance::tp_getattr_concrete(pyAttrName, attrName);
}

void PyClassInstance::mirrorTypeInformationIntoPyTypeConcrete(Class* classT, PyTypeObject* pyType) {
    PyObjectStealer bases(PyTuple_New(classT->getHeldClass()->getBases().size()));

    for (long k = 0; k < classT->getHeldClass()->getBases().size(); k++) {
        PyTuple_SetItem(
            bases,
            k,
            incref(typePtrToPyTypeRepresentation(classT->getHeldClass()->getBases()[k]->getClassType()))
        );
    }

    PyObjectStealer mro(PyTuple_New(classT->getHeldClass()->getMro().size()));

    for (long k = 0; k < classT->getHeldClass()->getMro().size(); k++) {
        PyTuple_SetItem(
            mro,
            k,
            incref(typePtrToPyTypeRepresentation(classT->getHeldClass()->getMro()[k]->getClassType()))
        );
    }

    PyObjectStealer types(PyTuple_New(classT->getMembers().size()));

    for (long k = 0; k < classT->getMembers().size(); k++) {
        PyTuple_SetItem(types, k, incref(typePtrToPyTypeRepresentation(std::get<1>(classT->getMembers()[k]))));
    }

    PyObjectStealer names(PyTuple_New(classT->getMembers().size()));
    for (long k = 0; k < classT->getMembers().size(); k++) {
        PyObject* namePtr = PyUnicode_FromString(std::get<0>(classT->getMembers()[k]).c_str());
        PyTuple_SetItem(names, k, namePtr);
    }

    PyObjectStealer defaults(PyDict_New());

    for (long k = 0; k < classT->getMembers().size(); k++) {

        if (classT->getHeldClass()->memberHasDefaultValue(k)) {
            const Instance& i = classT->getHeldClass()->getMemberDefaultValue(k);

            PyObjectStealer defaultVal(
                PyInstance::extractPythonObject(i.data(), i.type())
                );

            PyDict_SetItemString(
                defaults,
                classT->getHeldClass()->getMemberName(k).c_str(),
                defaultVal
                );
        }
    }

    PyObjectStealer memberFunctions(PyDict_New());

    for (auto p: classT->getMemberFunctions()) {
        PyDict_SetItemString(memberFunctions, p.first.c_str(), typePtrToPyTypeRepresentation(p.second));
        PyDict_SetItemString(pyType->tp_dict, p.first.c_str(), typePtrToPyTypeRepresentation(p.second));
    }

    //expose 'ElementType' as a member of the type object
    PyDict_SetItemString(pyType->tp_dict, "HeldClass", typePtrToPyTypeRepresentation(classT->getHeldClass()));
    PyDict_SetItemString(pyType->tp_dict, "MemberTypes", types);
    PyDict_SetItemString(pyType->tp_dict, "BaseClasses", bases);
    PyDict_SetItemString(pyType->tp_dict, "IsFinal", classT->isFinal() ? Py_True : Py_False);
    PyDict_SetItemString(pyType->tp_dict, "MRO", mro);
    PyDict_SetItemString(pyType->tp_dict, "MemberNames", names);
    PyDict_SetItemString(pyType->tp_dict, "MemberDefaultValues", defaults);

    PyDict_SetItemString(pyType->tp_dict, "MemberFunctions", memberFunctions);

    for (auto nameAndObj: classT->getClassMembers()) {
        PyDict_SetItemString(
            pyType->tp_dict,
            nameAndObj.first.c_str(),
            nameAndObj.second
            );
    }

    for (auto nameAndObj: classT->getStaticFunctions()) {
        PyDict_SetItemString(
            pyType->tp_dict,
            nameAndObj.first.c_str(),
            PyInstance::initializePythonRepresentation(nameAndObj.second, [&](instance_ptr data){
                //nothing to do - functions like this are just types.
            })
            );
    }
}

int PyClassInstance::tp_setattr_concrete(PyObject* attrName, PyObject* attrVal) {
    return PyClassInstance::classInstanceSetAttributeFromPyObject(type(), dataPtr(), attrName, attrVal);
}


PyObject* PyClassInstance::tp_call_concrete(PyObject* args, PyObject* kwargs) {
    auto it = type()->getMemberFunctions().find("__call__");

    if (it == type()->getMemberFunctions().end()) {
        PyErr_Format(
            PyExc_TypeError,
            "'%s' object is not callable because '__call__' was not defined",
            type()->name().c_str()
            );
        throw PythonExceptionSet();
    }
    // else
    Function* method = it->second;

    auto res = PyFunctionInstance::tryToCallAnyOverload(method, (PyObject*)this, args, kwargs);

    if (res.first) {
        return res.second;
    }
    // else
    throw PythonExceptionSet();
}
