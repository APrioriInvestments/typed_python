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

#include "PyClassInstance.hpp"

Class* PyClassInstance::type() {
    return (Class*)extractTypeFrom(((PyObject*)this)->ob_type);
}

bool PyClassInstance::pyValCouldBeOfTypeConcrete(Class* type, PyObject* pyRepresentation, ConversionLevel level) {
    Type* argType = extractTypeFrom(pyRepresentation->ob_type);

    return argType && (
        (Type::typesEquivalent(argType, type) || argType->isSubclassOf(type))
    );
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

void PyClassInstance::copyConstructFromPythonInstanceConcrete(Class* eltType, instance_ptr tgt, PyObject* pyRepresentation, ConversionLevel level) {
    std::pair<Type*, instance_ptr> typeAndPtr = extractTypeAndPtrFrom(pyRepresentation);
    Type* argType = typeAndPtr.first;
    instance_ptr argDataPtr = typeAndPtr.second;

    if (argType && argType->isSubclassOf(eltType)) {

        // arg data ptrs here should _always_ have the 0 classDispatchOffset, because
        // we never allow the python representation of a class to be a subclass.
        if (Class::instanceToDispatchTableIndex(argDataPtr) != 0) {
            throw std::runtime_error("Corrupt class instance. Expected a zero classDispatchOffset.");
        }

        Class* childType = (Class*)argType;

        int mroIndex = childType->getHeldClass()->getMroIndex(
            eltType->getHeldClass()
        );

        if (mroIndex < 0) {
            throw std::runtime_error("Failed to make an " + eltType->name() + " out of a " + argType->name() +
                " because even though the latter is a subclass of the former, we got an invalid MRO index"
            );
        }

        Class::initializeInstance(
            tgt,
            Class::instanceToLayout(argDataPtr),
            mroIndex
        )->refcount++;

        return;
    }

    PyInstance::copyConstructFromPythonInstanceConcrete(eltType, tgt, pyRepresentation, level);
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
        int res = tpSetattrGeneric(nullptr, cls, data, key, value);

        if (res != 0) {
            throw PythonExceptionSet();
        }
    }
}

/**
 *  Return 0 if successful and -1 if it failed
 */
// static
int PyClassInstance::tpSetattrGeneric(
    PyObject* self,
    Type* t,
    instance_ptr data,
    PyObject* attrName,
    PyObject* attrVal
) {
    HeldClass* heldClass = getHeldClassType(t);
    instance_ptr heldData = getHeldClassData(t, data);

    if (!attrVal && heldClass->hasDelAttrMagicMethod() && self) {
        auto p = callMemberFunctionGeneric(self, t, data, "__delattr__", attrName);
        if (p.first) {
            if (p.second) {
                decref(p.second);
                return 0;
            } else {
                return -1;
            }
        }
    }
    else if (heldClass->hasSetAttrMagicMethod() && self) {
        auto p = callMemberFunctionGeneric(self, t, data, "__setattr__", attrName, attrVal);
        if (p.first) {
            if (p.second) {
                decref(p.second);
                return 0;
            } else {
                return -1;
            }
        }
    }

    int memberIndex = heldClass->memberNamed(PyUnicode_AsUTF8(attrName));

    if (memberIndex < 0) {
        auto it = heldClass->getClassMembers().find(
            PyUnicode_AsUTF8(attrName)
        );

        if (it == heldClass->getClassMembers().end()) {
            PyErr_Format(
                PyExc_AttributeError,
                "'%s' object has no attribute '%S' and cannot add attributes to instances of this type",
                t->name().c_str(), attrName
            );
        } else {
            PyErr_Format(
                PyExc_AttributeError,
                "Cannot modify read-only class member '%S' of instance of type '%s'",
                attrName, t->name().c_str()
            );
        }

        return -1;
    }

    if (!attrVal) {
        if (heldClass->getMemberIsNonempty(memberIndex)) {
            PyErr_Format(
                PyExc_AttributeError,
                "Attribute '%S' cannot be deleted",
                attrName
            );
            return -1;
        }

        if (!heldClass->checkInitializationFlag(heldData, memberIndex)) {
            PyErr_Format(
                PyExc_AttributeError,
                "Attribute '%S' is not initialized",
                attrName
            );
            return -1;
        }

        heldClass->delAttribute(heldData, memberIndex);
        return 0;
    }

    Type* eltType = heldClass->getMemberType(memberIndex);

    Type* attrType = extractTypeFrom(attrVal->ob_type);

    if (Type::typesEquivalent(eltType, attrType)) {
        PyInstance* item_w = (PyInstance*)attrVal;

        heldClass->setAttribute(heldData, memberIndex, item_w->dataPtr());

        return 0;
    }
    else if (attrType && attrType->isRefTo() &&
            ((RefTo*)attrType)->getEltType() == eltType) {
        PyInstance* item_w = (PyInstance*)attrVal;

        heldClass->setAttribute(heldData, memberIndex, *(instance_ptr*)item_w->dataPtr());

        return 0;
    } else {
        Instance temp = Instance::createAndInitialize(eltType, [&](instance_ptr tempObj) {
            copyConstructFromPythonInstance(eltType, tempObj, attrVal, ConversionLevel::ImplicitContainers);
        });

        heldClass->setAttribute(heldData, memberIndex, temp.data());

        return 0;
    }
}

PyObject* PyClassInstance::pyUnaryOperatorConcrete(const char* op, const char* opErr) {
    return pyUnaryOperatorConcreteGeneric((PyObject*)this, type(), dataPtr(), op, opErr);
}

PyObject* PyClassInstance::pyUnaryOperatorConcreteGeneric(PyObject* self, Type* t, instance_ptr data, const char* op, const char* opErr) {
    auto res = callMemberFunctionGeneric(self, t, data, op);

    if (!res.first) {
        // dispatch to the base class for a generic error message
        return ((PyInstance*)self)->pyUnaryOperatorConcrete(op, opErr);
    }

    return res.second;
}

PyObject* PyClassInstance::pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr) {
    return pyOperatorConcreteGeneric((PyObject*)this, type(), dataPtr(), rhs, op, opErr);
}

PyObject* PyClassInstance::pyOperatorConcreteGeneric(PyObject* self, Type* t, instance_ptr data, PyObject* rhs, const char* op, const char* opErr) {
    auto res = callMemberFunctionGeneric(self, t, data, op, rhs);

    if (res.first) {
        return res.second;
    }

    if (strlen(op) > 2 && strlen(op) < 16 && op[2] == 'i') {
        // an inplace operator should fall back to the regular operator
        char reg_op[16] = "__";
        strcpy(&reg_op[2], op + 3);

        auto res = callMemberFunctionGeneric(self, t, data, reg_op, rhs);
        if (res.first) {
            return res.second;
        }
    }

    return ((PyInstance*)self)->pyOperatorConcrete(rhs, op, opErr);
}

PyObject* PyClassInstance::pyOperatorConcreteReverse(PyObject* lhs, const char* op, const char* opErr) {
    return pyOperatorConcreteReverseGeneric((PyObject*)this, type(), dataPtr(), lhs, op, opErr);
}

PyObject* PyClassInstance::pyOperatorConcreteReverseGeneric(PyObject* self, Type* t, instance_ptr data, PyObject* lhs, const char* op, const char* opErr) {
    char buf[50];
    strncpy(buf+1, op, 45);
    buf[0] = '_';
    buf[2] = 'r';

    auto res = callMemberFunctionGeneric(self, t, data, buf, lhs);

    if (!res.first) {
        return ((PyInstance*)self)->pyOperatorConcreteReverse(lhs, buf, opErr);
    }

    return res.second;
}

PyObject* PyClassInstance::pyTernaryOperatorConcrete(PyObject* rhs, PyObject* ternaryArg, const char* op, const char* opErr) {
    return pyTernaryOperatorConcreteGeneric((PyObject*)this, type(), dataPtr(), rhs, ternaryArg, op, opErr);
}

PyObject* PyClassInstance::pyTernaryOperatorConcreteGeneric(PyObject* self, Type* t, instance_ptr data, PyObject* rhs, PyObject* ternaryArg, const char* op, const char* opErr) {
    if (ternaryArg == Py_None) {
        //if you pass 'None' as the third argument, python calls your class
        //__pow__ function with two arguments. This is the behavior for
        //'instance ** b' as well as if you write 'pow(instance,b,None)'
        return pyOperatorConcreteGeneric(self, t, data, rhs, op, opErr);
    }

    auto res = callMemberFunctionGeneric(self, t, data, op, rhs, ternaryArg);

    if (!res.first) {
        return ((PyInstance*)self)->pyTernaryOperatorConcrete(rhs, ternaryArg, op, opErr);
    }

    return res.second;
}

int PyClassInstance::pyInquiryConcrete(const char* op, const char* opErrRep) {
    return pyInquiryGeneric((PyObject*)this, type(), dataPtr(), op, opErrRep);
}

int PyClassInstance::pyInquiryGeneric(
    PyObject* self, Type* t, instance_ptr data, const char* op, const char* opErrRep
) {
    // op == '__bool__'
    auto p = callMemberFunctionGeneric(self, t, data, "__bool__");
    if (!p.first) {
        p = callMemberFunctionGeneric(self, t, data, "__len__");
        // if neither __bool__ nor __len__ is available, return True
        if (!p.first) {
            return 1;
        }
    }

    if (!p.second) {
        return -1;
    }

    int result = PyObject_IsTrue(p.second);
    decref(p.second);
    return result;
}

// try to call user-defined hash method
// returns -1 if not defined or if it returns an invalid value
int64_t PyClassInstance::tryCallHashMemberFunction() {
    auto result = callMemberFunction("__hash__");

    if (!result.first)
        return -1;

    if (!PyLong_Check(result.second)) {
        decref(result.second);
        return -1;
    }

    int64_t res = PyLong_AsLong(result.second);
    decref(result.second);
    return res;
}

std::pair<bool, PyObject*> PyClassInstance::callMemberFunction(const char* name, PyObject* arg0, PyObject* arg1, PyObject* arg2) {
    return callMemberFunctionGeneric((PyObject*)this, type(), dataPtr(), name, arg0, arg1, arg2);
}

std::pair<bool, PyObject*> PyClassInstance::callMemberFunctionGeneric(
    PyObject* self,
    Type* t,
    instance_ptr data,
    const char* name,
    PyObject* arg0,
    PyObject* arg1,
    PyObject* arg2
) {
    HeldClass* heldClass = getHeldClassType(t);

    auto it = heldClass->getMemberFunctions().find(name);

    if (it == heldClass->getMemberFunctions().end()) {
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
    if (arg2) {
        argCount += 1;
    }

    PyObjectStealer targetArgTuple(PyTuple_New(argCount));

    PyTuple_SetItem(targetArgTuple, 0, incref(self)); //steals a reference

    if (arg0) {
        PyTuple_SetItem(targetArgTuple, 1, incref(arg0)); //steals a reference
    }
    if (arg1) {
        PyTuple_SetItem(targetArgTuple, 2, incref(arg1)); //steals a reference
    }
    if (arg2) {
        PyTuple_SetItem(targetArgTuple, 3, incref(arg2)); //steals a reference
    }

    auto res = PyFunctionInstance::tryToCallAnyOverload(method, nullptr, nullptr, targetArgTuple, nullptr);

    if (res.first) {
        return res;
    }

    PyErr_Format(
        PyExc_TypeError,
        "'%s.%s' cannot find a valid overload with these arguments",
        t->name().c_str(),
        name
    );
    return std::make_pair(true, (PyObject*)nullptr);
}

Py_ssize_t PyClassInstance::mp_and_sq_length_concrete() {
    return mpAndSqLengthGeneric((PyObject*)this, type(), dataPtr());
}

Py_ssize_t PyClassInstance::mpAndSqLengthGeneric(PyObject* self, Type* t, instance_ptr data) {
    auto res = callMemberFunctionGeneric(self, t, data, "__len__");

    if (!res.first) {
        // call the generic form of the function which produces the error message
        return ((PyInstance*)self)->mp_and_sq_length_concrete();
    }

    if (!res.second) {
        decref(res.second);
        return -1;
    }

    if (!PyLong_Check(res.second)) {
        PyErr_Format(
            PyExc_TypeError,
            "'%s.__len__' returned an object of type %s",
            t->name().c_str(),
            res.second->ob_type->tp_name
            );
        decref(res.second);
        return -1;
    }

    long result = PyLong_AsLong(res.second);
    decref(res.second);

    if (result < 0) {
        PyErr_Format(PyExc_ValueError, "'__len__()' should return >= 0");
        return -1;
    }

    return result;
}

void PyClassInstance::constructFromPythonArgumentsConcrete(Class* classT, uint8_t* data, PyObject* args, PyObject* kwargs) {
    classT->constructor(data, true /* allowEmpty */);

    auto it = classT->getMemberFunctions().find("__init__");
    if (it == classT->getMemberFunctions().end()) {
        //run the default constructor
        PyClassInstance::initializeClassWithDefaultArguments(classT, data, args, kwargs);
        return;
    }

    Function* initMethod = it->second;

    PyObjectStealer selfAsObject(
        PyInstance::initialize(classT, [&](instance_ptr selfData) {
            classT->copy_constructor(selfData, data);
        })
    );

    auto res = PyFunctionInstance::tryToCallAnyOverload(initMethod, nullptr, selfAsObject, args, kwargs);

    if (!res.first) {
        throw std::runtime_error("Cannot find a valid overload of __init__ with these arguments.");
    }

    if (res.second) {
        decref(res.second);
    } else {
        throw PythonExceptionSet();
    }
}

instance_ptr PyClassInstance::getHeldClassData(Type* t, instance_ptr data) {
    if (t->isHeldClass()) {
        return data;
    }

    if (t->isRefTo()) {
        return *(instance_ptr*)data;
    }

    if (t->isClass()) {
        return (instance_ptr)Class::instanceToLayout(data)->data;
    }

    throw std::runtime_error("Can't extract a HeldClass* from " + t->name());
}

HeldClass* PyClassInstance::getHeldClassType(Type* t) {
    if (t->isRefTo()) {
        return (HeldClass*)((RefTo*)t)->getEltType();
    }

    if (t->isHeldClass()) {
        return (HeldClass*)t;
    }

    if (t->isClass()) {
        return ((Class*)t)->getHeldClass();
    }

    throw std::runtime_error("Can't extract a HeldClass* from " + t->name());
}

PyObject* PyClassInstance::tp_getattr_concrete(PyObject* pyAttrName, const char* attrName) {
    return tpGetattrGeneric((PyObject*)this, type(), dataPtr(), pyAttrName, attrName);
}

PyObject* PyClassInstance::tpGetattrGeneric(
    PyObject* self,
    Type* t,
    instance_ptr data,
    PyObject* pyAttrName,
    const char* attrName
) {
    HeldClass* heldClass = getHeldClassType(t);
    instance_ptr heldData = getHeldClassData(t, data);

    if (getHeldClassType(t)->hasGetAttributeMagicMethod()) {
        auto p = callMemberFunctionGeneric(self, t, data, "__getattribute__", pyAttrName);
        if (PyErr_Occurred() && PyErr_ExceptionMatches(PyExc_AttributeError)) {
            PyErr_Clear();
        }
        else {
            if (p.first) {
                return p.second;
            }
        }
    }

    int index = heldClass->getMemberIndex(attrName);

    if (index >= 0) {
        Type* eltType = heldClass->getMemberType(index);

        if (!heldClass->checkInitializationFlag(heldData, index)) {
            PyErr_Format(
                PyExc_AttributeError,
                "Attribute '%S' is not initialized",
                pyAttrName,
                eltType->name().c_str()
            );
            return NULL;
        }

        return extractPythonObject(heldClass->eltPtr(heldData, index), eltType);
    }

    BoundMethod* method = heldClass->getMemberFunctionMethodType(
        attrName,
        t->isClass() ? false : true /* for held */
    );

    if (method) {
        if (t->isClass()) {
            return PyInstance::initializePythonRepresentation(method, [&](instance_ptr methodData) {
                method->copy_constructor(methodData, data);
            });
        } else {
            if (!method->getFirstArgType()->isRefTo()) {
                throw std::runtime_error("BoundMethod of HeldClass should take a RefTo");
            }

            return PyInstance::initializePythonRepresentation(method, [&](instance_ptr methodData) {
                method->copy_constructor(methodData, (instance_ptr)&heldData);
            });
        }
    }

    {
        auto it = heldClass->getPropertyFunctions().find(attrName);
        if (it != heldClass->getPropertyFunctions().end()) {
            auto res = PyFunctionInstance::tryToCall(it->second, nullptr, self);

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
        auto it = heldClass->getClassMembers().find(attrName);
        if (it != heldClass->getClassMembers().end()) {
            // if this follows the descriptor protocol, use that
            if (PyObject_HasAttrString(it->second, "__get__")) {
                return PyObject_CallMethod(it->second, "__get__", "OO", self, (PyObject*)self->ob_type);
            }

            return incref(it->second);
        }
    }

    // call the generic tp_getattr, which can bind descriptors defined in classes correctly.
    PyObject* ret = ((PyInstance*)self)->tp_getattr_concrete(pyAttrName, attrName);

    if (PyErr_Occurred() && PyErr_ExceptionMatches(PyExc_AttributeError)) {
        PyErr_Clear();
        auto p = callMemberFunctionGeneric(self, t, data, "__getattr__", pyAttrName);
        if (p.first) {
            return p.second;
        }
        else {
            PyErr_Format(
                PyExc_AttributeError,
                "no attribute %s for instance of type %s",
                attrName, t->name().c_str()
            );
            return NULL;
        }
    }

    return ret;
}

/* initialize the type object's dict.

if 'asHeldClass', then this is the held class's dict we're initializing.
*/
void PyClassInstance::mirrorTypeInformationIntoPyTypeConcrete(Class* classT, PyTypeObject* pyType, bool asHeldClass) {
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
        PyTuple_SetItem(types, k, incref(typePtrToPyTypeRepresentation(classT->getMembers()[k].getType())));
    }

    PyObjectStealer names(PyTuple_New(classT->getMembers().size()));
    for (long k = 0; k < classT->getMembers().size(); k++) {
        PyObject* namePtr = PyUnicode_FromString(classT->getMembers()[k].getName().c_str());
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

        // TODO: find a predefined function that does this method search
        PyMethodDef* defined = pyType->tp_methods;
        while (defined && defined->ml_name && !!strcmp(defined->ml_name, p.first.c_str()))
            defined++;

        if (!defined || !defined->ml_name) {
            if (p.second->getClosureType()->bytecount()) {
                std::cout << "WARNING: invalid class member " << classT->name() << p.first << " had a nonempty closure.\n";
            } else {
                if (p.second->bytecount()) {
                    throw std::runtime_error("Somehow, a Class got a function with a closure type.");
                }
                PyDict_SetItemString(pyType->tp_dict, p.first.c_str(), PyInstance::initialize(p.second, [&](instance_ptr) {}));
            }
        }
    }

    PyObjectStealer propertyFunctions(PyDict_New());

    for (auto p: classT->getPropertyFunctions()) {
        PyDict_SetItemString(propertyFunctions, p.first.c_str(), typePtrToPyTypeRepresentation(p.second));
    }

    //expose 'ElementType' as a member of the type object
    if (asHeldClass) {
        PyDict_SetItemString(pyType->tp_dict, "Class", typePtrToPyTypeRepresentation(classT));
    } else {
        PyDict_SetItemString(pyType->tp_dict, "HeldClass", typePtrToPyTypeRepresentation(classT->getHeldClass()));
    }

    PyDict_SetItemString(pyType->tp_dict, "MemberTypes", types);
    PyDict_SetItemString(pyType->tp_dict, "BaseClasses", bases);
    PyDict_SetItemString(pyType->tp_dict, "IsFinal", classT->isFinal() ? Py_True : Py_False);
    PyDict_SetItemString(pyType->tp_dict, "MRO", mro);
    PyDict_SetItemString(pyType->tp_dict, "MemberNames", names);
    PyDict_SetItemString(pyType->tp_dict, "MemberDefaultValues", defaults);

    PyDict_SetItemString(pyType->tp_dict, "PropertyFunctions", propertyFunctions);
    PyDict_SetItemString(pyType->tp_dict, "MemberFunctions", memberFunctions);

    PyObjectStealer classMembers(PyDict_New());
    for (auto nameAndObj: classT->getClassMembers()) {
        PyDict_SetItemString(
            pyType->tp_dict,
            nameAndObj.first.c_str(),
            nameAndObj.second
        );
        PyDict_SetItemString(
            classMembers,
            nameAndObj.first.c_str(),
            nameAndObj.second
        );
    }

    PyDict_SetItemString(pyType->tp_dict, "ClassMembers", classMembers);

    PyObjectStealer staticMemberFunctions(PyDict_New());
    PyDict_SetItemString(pyType->tp_dict, "StaticMemberFunctions", staticMemberFunctions);
    for (auto nameAndObj: classT->getStaticFunctions()) {
        PyDict_SetItemString(staticMemberFunctions, nameAndObj.first.c_str(), typePtrToPyTypeRepresentation(nameAndObj.second));

        if (nameAndObj.second->getClosureType()->bytecount()) {
            throw std::runtime_error(
                "Somehow, " + classT->name() + "."
                + nameAndObj.first + " has a populated closure."
            );
        }

        PyDict_SetItemString(
            pyType->tp_dict,
            nameAndObj.first.c_str(),
            PyStaticMethod_New(
                PyInstance::initialize(nameAndObj.second, [&](instance_ptr data){
                    //nothing to do - functions like this are just types.
                })
            )
        );
    }

    PyObjectStealer classMemberFunctions(PyDict_New());
    PyDict_SetItemString(pyType->tp_dict, "ClassMemberFunctions", classMemberFunctions);
    for (auto nameAndObj: classT->getClassMethods()) {
        PyDict_SetItemString(classMemberFunctions, nameAndObj.first.c_str(), typePtrToPyTypeRepresentation(nameAndObj.second));

        if (nameAndObj.second->getClosureType()->bytecount()) {
            throw std::runtime_error(
                "Somehow, " + classT->name() + "."
                + nameAndObj.first + " has a populated closure."
            );
        }

        PyDict_SetItemString(
            pyType->tp_dict,
            nameAndObj.first.c_str(),
            PyClassMethod_New(
                PyInstance::initialize(nameAndObj.second, [&](instance_ptr data){
                    //nothing to do - functions like this are just types.
                })
            )
        );
    }

    if (!asHeldClass) {
        // mirror the MRO + the hierarchy above into the actual __mro__ variable, which
        // over the long run should replace MRO and Bases.
        // we have to include the base classes above us (Class, Type, object)
        // so that standard python object model behaviors work.
        PyObjectStealer realMro(PyTuple_New(classT->getHeldClass()->getMro().size() + 3));

        long k = 0;
        for (;k < classT->getHeldClass()->getMro().size(); k++) {
            PyTuple_SetItem(
                realMro,
                k,
                incref(typePtrToPyTypeRepresentation(classT->getHeldClass()->getMro()[k]->getClassType()))
            );
        }

        PyTuple_SetItem(
            realMro,
            k++,
            incref((PyObject*)pyType->tp_base)
        );

        PyTuple_SetItem(
            realMro,
            k++,
            incref((PyObject*)pyType->tp_base->tp_base)
        );

        PyTuple_SetItem(
            realMro,
            k++,
            incref((PyObject*)pyType->tp_base->tp_base->tp_base)
        );

        pyType->tp_mro = incref(realMro);
    }
}

int PyClassInstance::tp_setattr_concrete(PyObject* attrName, PyObject* attrVal) {
    return PyClassInstance::tpSetattrGeneric(
        (PyObject*)this,
        type(),
        dataPtr(),
        attrName,
        attrVal
    );
}

/* static */
bool PyClassInstance::compare_to_python_concrete(Class* t, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp) {
    if (t->getHeldClass()->hasAnyComparisonOperators()) {
        auto it = t->getHeldClass()->getMemberFunctions().find(Class::pyComparisonOpToMethodName(pyComparisonOp));

        if (it != t->getHeldClass()->getMemberFunctions().end()) {
            //we found a user-defined method for this comparison function.
            PyObjectStealer selfAsPyObj(PyInstance::extractPythonObject(self, t));

            std::pair<bool, PyObject*> res = PyFunctionInstance::tryToCall(
                it->second,
                nullptr,
                selfAsPyObj,
                other
            );

            if (res.first && !res.second) {
                throw PythonExceptionSet();
            }

            int result = PyObject_IsTrue(res.second);
            decref(res.second);

            if (result == -1) {
                throw PythonExceptionSet();
            }

            return result != 0;
        }
    }

    return PyInstance::compare_to_python_concrete(t, self, other, exact, pyComparisonOp);
}


PyObject* PyClassInstance::tp_call_concrete(PyObject* args, PyObject* kwargs) {
    return tpCallConcreteGeneric((PyObject*)this, type(), dataPtr(), args, kwargs);
}

PyObject* PyClassInstance::tpCallConcreteGeneric(
    PyObject* self, Type* t, instance_ptr data, PyObject* args, PyObject* kwargs
) {
    HeldClass* heldCls = getHeldClassType(t);

    auto it = heldCls->getMemberFunctions().find("__call__");

    if (it == heldCls->getMemberFunctions().end()) {
        PyErr_Format(
            PyExc_TypeError,
            "'%s' object is not callable because '__call__' was not defined",
            t->name().c_str()
        );

        throw PythonExceptionSet();
    }
    // else
    Function* method = it->second;

    auto res = PyFunctionInstance::tryToCallAnyOverload(
        method,
        nullptr,
        (PyObject*)self,
        args,
        kwargs
    );

    if (res.first) {
        return res.second;
    }

    throw PythonExceptionSet();
}


int PyClassInstance::sqContainsGeneric(PyObject* self, Type* t, instance_ptr data, PyObject* item) {
    auto p = callMemberFunctionGeneric(self, t, data, "__contains__", item);
    if (!p.first) {
        return 0;
    }
    if (p.second) {
        int res = PyObject_IsTrue(p.second);
        decref(p.second);
        return res;
    } else {
        return -1;
    }
}

int PyClassInstance::sq_contains_concrete(PyObject* item) {
    return sqContainsGeneric((PyObject*)this, type(), dataPtr(), item);
}

PyObject* PyClassInstance::mp_subscript_concrete(PyObject* item) {
    return mpSubscriptGeneric((PyObject*)this, type(), dataPtr(), item);
}

PyObject* PyClassInstance::mpSubscriptGeneric(PyObject* self, Type* t, instance_ptr data, PyObject* item) {
    auto p = callMemberFunctionGeneric(self, t, data, "__getitem__", item);
    if (!p.first) {
        PyErr_Format(PyExc_TypeError, "__getitem__ not defined for type %s", t->name().c_str());
        return NULL;
    }
    return p.second;
}

int PyClassInstance::mp_ass_subscript_concrete(PyObject* item, PyObject* v) {
    return mpAssignSubscriptGeneric((PyObject*)this, type(), dataPtr(), item, v);
}

int PyClassInstance::mpAssignSubscriptGeneric(PyObject* self, Type* t, instance_ptr data, PyObject* item, PyObject* v) {
    std::pair<bool, PyObject*> p;

    if (!v) {
        p = callMemberFunctionGeneric(self, t, data, "__delitem__", item);
    } else {
        p = callMemberFunctionGeneric(self, t, data, "__setitem__", item, v);
    }

    if (!p.first) {
        if (!v) {
            PyErr_Format(PyExc_TypeError, "__delitem__ not defined for type %s", t->name().c_str());
        } else {
            PyErr_Format(PyExc_TypeError, "__setitem__ not defined for type %s", t->name().c_str());
        }
        return -1;
    }

    if (p.second) {
        decref(p.second);
        return 0;
    } else {
        return -1;
    }
}

PyObject* PyClassInstance::tp_iter_concrete() {
    return tpIterGeneric((PyObject*)this, type(), dataPtr());
}

PyObject* PyClassInstance::tpIterGeneric(PyObject* self, Type* t, instance_ptr data) {
    auto p = callMemberFunctionGeneric(self, t, data, "__iter__");
    if (!p.first) {
        PyErr_Format(PyExc_TypeError, "__iter__ not defined for type %s", t->name().c_str());
        return NULL;
    }
    return p.second;
}

PyObject* PyClassInstance::tp_iternext_concrete() {
    return tpIternextGeneric((PyObject*)this, type(), dataPtr());
}

PyObject* PyClassInstance::tpIternextGeneric(PyObject* self, Type* t, instance_ptr data) {
    auto p = callMemberFunctionGeneric(self, t, data, "__next__");
    if (!p.first) {
        PyErr_Format(PyExc_TypeError, "__next__ not defined for type %s", t->name().c_str());
        return NULL;
    }
    return p.second;
}

// static
PyObject* PyClassInstance::clsFormatGeneric(PyObject* o, Type* t, instance_ptr data, PyObject* args, PyObject* kwargs) {
    return translateExceptionToPyObject([&]() -> PyObject* {
        PyClassInstance* self = (PyClassInstance*)o;

        if (PyTuple_Size(args) != 1 || kwargs) {
            PyErr_Format(PyExc_TypeError, "__format__ invalid number of parameters");
            return NULL;
        }
        PyObjectStealer arg0(PyTuple_GetItem(args, 0));

        auto result = self->callMemberFunctionGeneric(o, t, data, "__format__", arg0);
        if (!result.first) {
            PyErr_Format(PyExc_TypeError, "__format__ not defined for type %s", self->type()->name().c_str());
            return NULL;
        }
        if (!PyUnicode_Check(result.second)) {
            PyErr_Format(PyExc_TypeError, "__format__ returned non-string for type %s", self->type()->name().c_str());
            return NULL;
        }

        return result.second;
    });
}

// static
PyObject* PyClassInstance::clsBytesGeneric(PyObject* o, Type* t, instance_ptr data, PyObject* args, PyObject* kwargs) {
    return translateExceptionToPyObject([&]() -> PyObject* {
        PyClassInstance* self = (PyClassInstance*)o;

        if (PyTuple_Size(args) > 0 || kwargs) {
            PyErr_Format(PyExc_TypeError, "__bytes__ invalid number of parameters");
            return NULL;
        }

        auto result = self->callMemberFunctionGeneric(o, t, data, "__bytes__");
        if (!result.first) {
            PyErr_Format(PyExc_TypeError, "__bytes__ not defined for type %s", self->type()->name().c_str());
            return NULL;
        }
        if (!PyBytes_Check(result.second)) {
            PyErr_Format(PyExc_TypeError, "__bytes__ returned non-bytes %s for type %s", result.second->ob_type->tp_name, self->type()->name().c_str());
            return NULL;
        }

        return result.second;
    });
}

// static
PyObject* PyClassInstance::clsDirGeneric(PyObject* o, Type* t, instance_ptr data, PyObject* args, PyObject* kwargs) {
    return translateExceptionToPyObject([&]() -> PyObject* {
        PyClassInstance* self = (PyClassInstance*)o;

        if (PyTuple_Size(args) > 0 || kwargs) {
            PyErr_Format(PyExc_TypeError, "__dir__ invalid number of parameters");
            return NULL;
        }

        auto result = self->callMemberFunctionGeneric(o, t, data, "__dir__");
        if (!result.first) {
            PyErr_Format(PyExc_TypeError, "__dir__ missing");
            return NULL;
        }
        if (!PySequence_Check(result.second)) {
            PyErr_Format(PyExc_TypeError, "__dir__ returned non-sequence %s for type %s", result.second->ob_type->tp_name, self->type()->name().c_str());
            return NULL;
        }

        return result.second;
    });
}

// static
PyObject* PyClassInstance::clsReversedGeneric(PyObject* o, Type* t, instance_ptr data, PyObject* args, PyObject* kwargs) {
    return translateExceptionToPyObject([&]() -> PyObject* {
        PyClassInstance* self = (PyClassInstance*)o;

        if (PyTuple_Size(args) > 0 || kwargs) {
            PyErr_Format(PyExc_TypeError, "__reversed__ invalid number of parameters");
            return NULL;
        }

        auto result = self->callMemberFunctionGeneric(o, t, data, "__reversed__");
        if (!result.first) {
            PyErr_Format(PyExc_TypeError, "__reversed__ missing");
            return NULL;
        }

        return result.second;
    });
}

// static
PyObject* PyClassInstance::clsComplexGeneric(PyObject* o, Type* t, instance_ptr data, PyObject* args, PyObject* kwargs) {
    return translateExceptionToPyObject([&]() -> PyObject* {
        PyClassInstance* self = (PyClassInstance*)o;

        if (PyTuple_Size(args) > 0 || kwargs) {
            PyErr_Format(PyExc_TypeError, "__complex__ invalid number of parameters");
            return NULL;
        }

        auto result = self->callMemberFunctionGeneric(o, t, data, "__complex__");
        if (!result.first) {
            PyErr_Format(PyExc_TypeError, "__complex__ missing");
            return NULL;
        }
        if (!PyComplex_Check(result.second)) {
            PyErr_Format(PyExc_TypeError, "__complex__ returned non-complex %s for type %s", result.second->ob_type->tp_name, self->type()->name().c_str());
            return NULL;
        }

        return result.second;
    });
}

// static
PyObject* PyClassInstance::clsRoundGeneric(PyObject* o, Type* t, instance_ptr data, PyObject* args, PyObject* kwargs) {
    return translateExceptionToPyObject([&]() -> PyObject* {
        PyClassInstance* self = (PyClassInstance*)o;

        if (PyTuple_Size(args) > 1 || kwargs) {
            PyErr_Format(PyExc_TypeError, "__round__ invalid number of parameters %d %d", PyTuple_Size(args), kwargs);
            return NULL;
        }

        if (PyTuple_Size(args) == 1) {
            PyObjectStealer arg0(PyTuple_GetItem(args, 0));
            auto result = self->callMemberFunctionGeneric(o, t, data, "__round__", arg0);
            if (!result.first) {
                PyErr_Format(PyExc_TypeError, "__round__ missing");
                return NULL;
            }
            return result.second;
        }

        auto result = self->callMemberFunctionGeneric(o, t, data, "__round__");
        if (!result.first) {
            PyErr_Format(PyExc_TypeError, "__round__ missing");
            return NULL;
        }
        if (!PyLong_Check(result.second)) {
            PyErr_Format(PyExc_TypeError, "__round__ returned non-integer %s for type %s", result.second->ob_type->tp_name, self->type()->name().c_str());
            return NULL;
        }

        return result.second;
    });
}

// static
PyObject* PyClassInstance::clsTruncGeneric(PyObject* o, Type* t, instance_ptr data, PyObject* args, PyObject* kwargs) {
    return translateExceptionToPyObject([&]() -> PyObject* {
        PyClassInstance* self = (PyClassInstance*)o;

        if (PyTuple_Size(args) > 0 || kwargs) {
            PyErr_Format(PyExc_TypeError, "__trunc__ invalid number of parameters");
            return NULL;
        }

        auto result = self->callMemberFunctionGeneric(o, t, data, "__trunc__");
        if (!result.first) {
            PyErr_Format(PyExc_TypeError, "__trunc__ missing");
            return NULL;
        }
        if (!PyLong_Check(result.second)) {
            PyErr_Format(PyExc_TypeError, "__trunc__ returned non-integer %s for type %s", result.second->ob_type->tp_name, self->type()->name().c_str());
            return NULL;
        }

        return result.second;
    });
}

// static
PyObject* PyClassInstance::clsFloorGeneric(PyObject* o, Type* t, instance_ptr data, PyObject* args, PyObject* kwargs) {
    return translateExceptionToPyObject([&]() -> PyObject* {
        PyClassInstance* self = (PyClassInstance*)o;

        if (PyTuple_Size(args) > 0 || kwargs) {
            PyErr_Format(PyExc_TypeError, "__floor__ invalid number of parameters");
            return NULL;
        }

        auto result = self->callMemberFunctionGeneric(o, t, data, "__floor__");
        if (!result.first) {
            PyErr_Format(PyExc_TypeError, "__floor__ missing");
            return NULL;
        }
        if (!PyLong_Check(result.second)) {
            PyErr_Format(PyExc_TypeError, "__floor__ returned non-integer %s for type %s", result.second->ob_type->tp_name, self->type()->name().c_str());
            return NULL;
        }

        return result.second;
    });
}

// static
PyObject* PyClassInstance::clsCeilGeneric(PyObject* o, Type* t, instance_ptr data, PyObject* args, PyObject* kwargs) {
    return translateExceptionToPyObject([&]() -> PyObject* {
        PyClassInstance* self = (PyClassInstance*)o;

        if (PyTuple_Size(args) > 0 || kwargs) {
            PyErr_Format(PyExc_TypeError, "__ceil__ invalid number of parameters");
            return NULL;
        }

        auto result = self->callMemberFunctionGeneric(o, t, data, "__ceil__");
        if (!result.first) {
            PyErr_Format(PyExc_TypeError, "__ceil__ missing");
            return NULL;
        }
        if (!PyLong_Check(result.second)) {
            PyErr_Format(PyExc_TypeError, "__ceil__ returned non-integer %s for type %s", result.second->ob_type->tp_name, self->type()->name().c_str());
            return NULL;
        }

        return result.second;
    });
}

// static
PyObject* PyClassInstance::clsEnterGeneric(PyObject* o, Type* t, instance_ptr data, PyObject* args, PyObject* kwargs) {
    return translateExceptionToPyObject([&]() -> PyObject* {
        PyClassInstance* self = (PyClassInstance*)o;

        if (PyTuple_Size(args) > 0 || kwargs) {
            PyErr_Format(PyExc_TypeError, "__enter__ invalid number of parameters");
            return NULL;
        }

        auto result = self->callMemberFunctionGeneric(o, t, data, "__enter__");

        if (!result.first) {
            PyErr_Format(PyExc_TypeError, "__enter__ missing");
            return NULL;
        }

        return result.second;
    });
}

// static
PyObject* PyClassInstance::clsExitGeneric(PyObject* o, Type* t, instance_ptr data, PyObject* args, PyObject* kwargs) {
    return translateExceptionToPyObject([&]() -> PyObject* {
        PyClassInstance* self = (PyClassInstance*)o;

        if (PyTuple_Size(args) != 3 || kwargs) {
            PyErr_Format(PyExc_TypeError, "__exit__ invalid number of parameters");
            return NULL;
        }
        PyObject* arg0(PyTuple_GetItem(args, 0));
        PyObject* arg1(PyTuple_GetItem(args, 1));
        PyObject* arg2(PyTuple_GetItem(args, 2));

        auto result = self->callMemberFunctionGeneric(o, t, data, "__exit__", arg0, arg1, arg2);
        if (!result.first) {
            PyErr_Format(PyExc_TypeError, "__exit__ missing");
            return NULL;
        }

        return result.second;
    });
}

// static
PyMethodDef* PyClassInstance::typeMethodsConcrete(Type* t) {

    // List of magic methods that are not attached to direct function pointers in PyTypeObject.
    //   These need to be defined by adding entries to PyTypeObject.tp_methods
    //   and we need to avoid adding them to PyTypeObject.tp_dict ourselves.
    //   Also, we only want to add the entry to tp_methods if they are explicitly defined.
    const std::map<const char*, PyCFunction> special_magic_methods = {
            {"__format__", (PyCFunction)clsFormat},
            {"__bytes__", (PyCFunction)clsBytes},
            {"__dir__", (PyCFunction)clsDir},
            {"__reversed__", (PyCFunction)clsReversed},
            {"__complex__", (PyCFunction)clsComplex},
            {"__round__", (PyCFunction)clsRound},
            {"__trunc__", (PyCFunction)clsTrunc},
            {"__floor__", (PyCFunction)clsFloor},
            {"__ceil__", (PyCFunction)clsCeil},
            {"__enter__", (PyCFunction)clsEnter},
            {"__exit__", (PyCFunction)clsExit}
        };

    int cur = 0;
    auto clsMethods = ((Class*)t)->getMemberFunctions();
    PyMethodDef* ret = new PyMethodDef[special_magic_methods.size() + 1];
    for (auto m: special_magic_methods) {
        if (clsMethods.find(m.first) != clsMethods.end()) {
            ret[cur++] =  {m.first, m.second, METH_VARARGS | METH_KEYWORDS, NULL};
        }
    }
    ret[cur] = {NULL, NULL};
    return ret;
}
