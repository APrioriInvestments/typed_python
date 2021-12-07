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

#include "PyAlternativeInstance.hpp"
#include "PyFunctionInstance.hpp"

Alternative* PyAlternativeInstance::type() {
    return (Alternative*)extractTypeFrom(((PyObject*)this)->ob_type);
}

ConcreteAlternative* PyConcreteAlternativeInstance::type() {
    return (ConcreteAlternative*)extractTypeFrom(((PyObject*)this)->ob_type);
}

PyObject* PyConcreteAlternativeInstance::pyTernaryOperatorConcrete(PyObject* rhs, PyObject* thirdArg, const char* op, const char* opErr) {
    if (thirdArg == Py_None) {
        return pyOperatorConcrete(rhs, op, opErr);
    }

    auto res = callMethod(op, rhs, thirdArg);
    if (!res.first) {
        return PyInstance::pyTernaryOperatorConcrete(rhs, thirdArg, op, opErr);
    }
    return res.second;
}

PyObject* PyConcreteAlternativeInstance::pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr) {
    auto res = callMethod(op, rhs);

    if (res.first) {
        return res.second;
    }

    if (strlen(op) > 2 && strlen(op) < 16 && op[2] == 'i') { // an inplace operator should fall back to the regular operator
        char reg_op[16] = "__";
        strcpy(&reg_op[2], op + 3);

        auto res = callMethod(reg_op, rhs);
        if (res.first)
            return res.second;
    }

    return PyInstance::pyOperatorConcrete(rhs, op, opErr);
}

PyObject* PyConcreteAlternativeInstance::pyOperatorConcreteReverse(PyObject* rhs, const char* op, const char* opErr) {
    if (strlen(op) < 2 || strlen(op) > 14) {
        return PyInstance::pyOperatorConcrete(rhs, op, opErr);
    }
    char rev_op[16] = "__r";
    strcpy(&rev_op[3], op + 2);
    auto res = callMethod(rev_op, rhs);

    if (!res.first) {
        return PyInstance::pyOperatorConcrete(rhs, op, opErr);
    }

    return res.second;
}

PyObject* PyConcreteAlternativeInstance::pyUnaryOperatorConcrete(const char* op, const char* opErr) {
    auto res = callMethod(op);

    if (!res.first) {
        return PyInstance::pyUnaryOperatorConcrete(op, opErr);
    }

    return res.second;
}

void PyConcreteAlternativeInstance::constructFromPythonArgumentsConcrete(ConcreteAlternative* alt, uint8_t* data, PyObject* args, PyObject* kwargs) {
    alt->constructor(data, [&](instance_ptr p) {
        if ((kwargs == nullptr || PyDict_Size(kwargs) == 0) && PyTuple_Size(args) == 1) {
            //construct an alternative from a single argument.
            //if it's a binary compatible subtype of the alternative we're constructing, then
            //invoke the copy constructor.
            PyObjectHolder arg(PyTuple_GetItem(args, 0));
            Type* argType = extractTypeFrom(arg->ob_type);

            if (argType && alt == argType) {
                //it's already the right kind of instance, so we can copy-through the underlying element
                alt->elementType()->copy_constructor(p, alt->eltPtr(((PyInstance*)(PyObject*)arg)->dataPtr()));
                return;
            }

            //otherwise, if we have exactly one subelement, attempt to construct from that
            if (alt->elementType()->getTypeCategory() != Type::TypeCategory::catNamedTuple) {
                throw std::runtime_error("ConcreteAlternatives are supposed to only contain NamedTuples");
            }

            NamedTuple* alternativeEltType = (NamedTuple*)alt->elementType();

            if (alternativeEltType->getTypes().size() != 1) {
                throw std::logic_error("Can't initialize " + alt->name() + " with positional arguments because it doesn't have only one field.");
            }

            PyInstance::copyConstructFromPythonInstance(alternativeEltType->getTypes()[0], p, arg, ConversionLevel::ImplicitContainers);
        } else if (PyTuple_Size(args) == 0) {
            //construct an alternative from Kwargs
            constructFromPythonArguments(p, alt->elementType(), args, kwargs);
        } else {
            throw std::logic_error("Can only initialize " + alt->name() + " from python with kwargs or a single in-place argument");
        }
    });
}

void PyAlternativeInstance::copyConstructFromPythonInstanceConcrete(Alternative* altType, instance_ptr tgt, PyObject* pyRepresentation, ConversionLevel level) {
    std::pair<Type*, instance_ptr> typeAndPtr = extractTypeAndPtrFrom(pyRepresentation);

    if (typeAndPtr.first && typeAndPtr.first->getTypeCategory() == Type::TypeCategory::catConcreteAlternative &&
            ((ConcreteAlternative*)typeAndPtr.first)->getAlternative() == altType) {
        altType->copy_constructor(tgt, typeAndPtr.second);
        return;
    }

    PyInstance::copyConstructFromPythonInstanceConcrete(altType, tgt, pyRepresentation, level);
}


PyObject* PyAlternativeInstance::tp_getattr_concrete(PyObject* pyAttrName, const char* attrName) {
    if (strcmp(attrName,"matches") == 0) {
        // concrete alternative should handle this
        throw std::runtime_error("PyAlternativeInstance should not be receiving 'matches' inquiries.");
    }

    //see if its a method
    Alternative* toCheck = type();

    auto it = toCheck->getMethods().find(attrName);

    if (it != toCheck->getMethods().end()) {
        return translateExceptionToPyObject([&]() {
            PyObjectStealer funcObj(
                it->second->getOverloads()[0].buildFunctionObj(
                    it->second->getClosureType(),
                    nullptr
                )
            );

            return PyMethod_New(
                (PyObject*)funcObj,
                (PyObject*)this
            );
        });
    }

    //see if its a member of our held type
    NamedTuple* heldT = (NamedTuple*)type()->subtypes()[type()->which(dataPtr())].second;
    instance_ptr heldData = type()->eltPtr(dataPtr());

    int ix = heldT->indexOfName(attrName);
    if (ix >= 0) {
        return extractPythonObject(
            heldT->eltPtr(heldData, ix),
            heldT->getTypes()[ix]
            );
    }

    return PyInstance::tp_getattr_concrete(pyAttrName, attrName);
}

PyObject* PyConcreteAlternativeInstance::tp_getattr_concrete(PyObject* pyAttrName, const char* attrName) {
    if (type()->getAlternative()->hasGetAttributeMagicMethod()) {
        auto p = callMethod("__getattribute__", pyAttrName);
        if (PyErr_Occurred() && PyErr_ExceptionMatches(PyExc_AttributeError)) {
            PyErr_Clear();
        }
        else {
            if (p.first) {
                return p.second;
            }
        }
    }

    if (strcmp(attrName, "matches") == 0) {
        Type* matcherType = AlternativeMatcher::Make(type());

        return PyInstance::initializePythonRepresentation(matcherType, [&](instance_ptr data) {
            matcherType->copy_constructor(data, dataPtr());
        });
    }

    //see if its a method
    Alternative* toCheck = type()->getAlternative();

    auto it = toCheck->getMethods().find(attrName);
    if (it != toCheck->getMethods().end()) {
        return translateExceptionToPyObject([&]() {
            PyObjectStealer funcObj(
                it->second->getOverloads()[0].buildFunctionObj(
                    it->second->getClosureType(),
                    nullptr
                )
            );

            return PyMethod_New(
                (PyObject*)funcObj,
                (PyObject*)this
            );
        });
    }

    //see if its a member of our held type
    NamedTuple* heldT = (NamedTuple*)type()->getAlternative()->subtypes()[type()->which()].second;
    instance_ptr heldData = type()->getAlternative()->eltPtr(dataPtr());

    int ix = heldT->indexOfName(attrName);
    if (ix >= 0) {
        return extractPythonObject(
            heldT->eltPtr(heldData, ix),
            heldT->getTypes()[ix]
            );
    }

    PyObject* ret = PyInstance::tp_getattr_concrete(pyAttrName, attrName);
    if (PyErr_Occurred() && PyErr_ExceptionMatches(PyExc_AttributeError)) {
        PyErr_Clear();
        auto p = callMethod("__getattr__", pyAttrName);
        if (p.first) {
            return p.second;
        }
        else {
            PyErr_Format(PyExc_AttributeError, "no attribute %s for instance of type %s", attrName, type()->name().c_str());
        }
    }
    return ret;
}

void PyAlternativeInstance::mirrorTypeInformationIntoPyTypeConcrete(Alternative* alt, PyTypeObject* pyType) {
    PyObjectStealer alternatives(PyTuple_New(alt->subtypes().size()));

    for (long k = 0; k < alt->subtypes().size(); k++) {
        ConcreteAlternative* concrete = ConcreteAlternative::Make(alt, k);

        PyDict_SetItemString(
            pyType->tp_dict,
            alt->subtypes()[k].first.c_str(),
            (PyObject*)typeObjInternal(concrete)
            );

        PyTuple_SetItem(alternatives, k, incref((PyObject*)typeObjInternal(concrete)));
    }

    PyDict_SetItemString(
        pyType->tp_dict,
        "__typed_python_alternatives__",
        alternatives
        );

    PyObjectStealer methodsDict(PyDict_New());

    for (auto method_pair: alt->getMethods()) {
        PyDict_SetItemString(
            methodsDict,
            method_pair.first.c_str(),
            (PyObject*)typeObj(method_pair.second)
        );

        PyDict_SetItemString(
            pyType->tp_dict,
            method_pair.first.c_str(),
            (PyObject*)typeObj(method_pair.second)
        );
    }

    PyDict_SetItemString(
        pyType->tp_dict,
        "__typed_python_methods__",
        methodsDict
    );
}

void PyConcreteAlternativeInstance::mirrorTypeInformationIntoPyTypeConcrete(ConcreteAlternative* alt, PyTypeObject* pyType) {
    PyDict_SetItemString(
        pyType->tp_dict,
        "Alternative",
        (PyObject*)typeObjInternal(alt->getAlternative())
        );

    PyDict_SetItemString(
        pyType->tp_dict,
        "Index",
        PyLong_FromLong(alt->which())
        );

    PyDict_SetItemString(
        pyType->tp_dict,
        "Name",
        PyUnicode_FromString(alt->getAlternative()->subtypes()[alt->which()].first.c_str())
        );

    PyDict_SetItemString(
        pyType->tp_dict,
        "ElementType",
        (PyObject*)typeObjInternal(alt->elementType())
        );

    for (auto method_pair: alt->getAlternative()->getMethods()) {
        // TODO: find a predefined function that does this method search
        PyMethodDef* defined = pyType->tp_methods;
        while (defined && defined->ml_name && !!strcmp(defined->ml_name, method_pair.first.c_str()))
            defined++;

        if (!defined || !defined->ml_name) {
            PyDict_SetItemString(
                pyType->tp_dict,
                method_pair.first.c_str(),
                (PyObject*)typeObj(method_pair.second)
            );
        }
    }
}

int PyAlternativeInstance::tp_setattr_concrete(PyObject* attrName, PyObject* attrVal) {
    PyErr_Format(
        PyExc_AttributeError,
        "Cannot set attributes on instance of type '%s' because it is immutable",
        type()->name().c_str()
    );
    return -1;
}

int PyConcreteAlternativeInstance::tp_setattr_concrete(PyObject* attrName, PyObject* attrVal) {
    if (!attrVal) {
        auto p = callMethod("__delattr__", attrName);
        if (p.first)
            return 0;
    }
    else {
        auto p = callMethod("__setattr__", attrName, attrVal);
        if (p.first)
            return 0;
    }

    PyErr_Format(
        PyExc_AttributeError,
        "Cannot set attributes on instance of type '%s' because it is immutable",
        type()->name().c_str()
    );
    return -1;
}

PyObject* PyConcreteAlternativeInstance::tp_call_concrete(PyObject* args, PyObject* kwargs) {
    auto it = type()->getAlternative()->getMethods().find("__call__");

    if (it == type()->getAlternative()->getMethods().end()) {
        PyErr_Format(
            PyExc_TypeError,
            "'%s' object is not callable because '__call__' was not defined",
            type()->name().c_str()
            );
        throw PythonExceptionSet();
    }
    // else
    Function* f = it->second;
    auto res = PyFunctionInstance::tryToCallAnyOverload(f, nullptr, (PyObject*)this, args, kwargs);
    if (res.first) {
        return res.second;
    }
    // else
    throw PythonExceptionSet();
}

// try to call user-defined hash method
// returns -1 if not defined or if it returns an invalid value
int64_t PyConcreteAlternativeInstance::tryCallHashMethod() {
    auto result = callMethod("__hash__");
    if (!result.first)
        return -1;
    if (!PyLong_Check(result.second))
        return -1;
    return PyLong_AsLong(result.second);
}

std::pair<bool, PyObject*> PyConcreteAlternativeInstance::callMethod(const char* name, PyObject* arg0, PyObject* arg1, PyObject* arg2) {
    auto it = type()->getAlternative()->getMethods().find(name);

    if (it == type()->getAlternative()->getMethods().end()) {
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

    PyTuple_SetItem(targetArgTuple, 0, incref((PyObject*)this)); //steals a reference

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
        type()->name().c_str(),
        name
    );
    return res;
}

int PyConcreteAlternativeInstance::pyInquiryConcrete(const char* op, const char* opErrRep) {
    // op == '__bool__'
    auto p = callMethod("__bool__");
    if (!p.first) {
        p = callMethod("__len__");
        // if neither __bool__ nor __len__ is available, return True
        if (!p.first)
            return 1;
    }
    return PyObject_IsTrue(p.second);
}

int PyConcreteAlternativeInstance::sq_contains_concrete(PyObject* item) {
    auto p = callMethod("__contains__", item);
    if (!p.first) {
        return 0;
    }
    return PyObject_IsTrue(p.second);
}

Py_ssize_t PyConcreteAlternativeInstance::mp_and_sq_length_concrete() {
    auto p = callMethod("__len__");
    if (!p.first) {
        return -1;
    }
    if (!PyLong_Check(p.second)) {
        return -1;
    }
    return PyLong_AsLong(p.second);
}

PyObject* PyConcreteAlternativeInstance::mp_subscript_concrete(PyObject* item) {
    auto p = callMethod("__getitem__", item);
    if (!p.first) {
        PyErr_Format(PyExc_TypeError, "__getitem__ not defined for type %s", type()->name().c_str());
        return NULL;
    }
    return p.second;
}

int PyConcreteAlternativeInstance::mp_ass_subscript_concrete(PyObject* item, PyObject* v) {
    auto p = callMethod("__setitem__", item, v);
    if (!p.first) {
        PyErr_Format(PyExc_TypeError, "__setitem__ not defined for type %s", type()->name().c_str());
        return -1;
    }
    return 0;
}

PyObject* PyConcreteAlternativeInstance::tp_iter_concrete() {
    auto p = callMethod("__iter__");
    if (!p.first) {
        PyErr_Format(PyExc_TypeError, "__iter__ not defined for type %s", type()->name().c_str());
        return NULL;
    }
    return p.second;
}

PyObject* PyConcreteAlternativeInstance::tp_iternext_concrete() {
    auto p = callMethod("__next__");
    if (!p.first) {
        PyErr_Format(PyExc_TypeError, "__next__ not defined for type %s", type()->name().c_str());
        return NULL;
    }
    return p.second;
}

// static
bool PyConcreteAlternativeInstance::compare_to_python_concrete(ConcreteAlternative* altT, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp) {
    PyObjectStealer self_object(extractPythonObject(self, altT));
    if (!self_object) {
        PyErr_PrintEx(0);
        throw std::runtime_error("failed to extract python object");
    }
    PyConcreteAlternativeInstance *self_inst = (PyConcreteAlternativeInstance*)(PyObject*)self_object;

    auto p = self_inst->callMethod(pyCompareFlagToMethod(pyComparisonOp), other);
    if (p.first)
        return PyObject_IsTrue(p.second);

    Type* otherT = extractTypeFrom(other->ob_type);
    if (otherT && otherT->getTypeCategory() == Type::TypeCategory::catConcreteAlternative) {
        if (((ConcreteAlternative *)otherT)->getAlternative() == altT->getAlternative())
        {
            PyConcreteAlternativeInstance* altInstance = (PyConcreteAlternativeInstance*)other;
            return altT->cmp(self, altInstance->dataPtr(), pyComparisonOp, false);
        }
    }

    return PyInstance::compare_to_python_concrete(altT, self, other, exact, pyComparisonOp);
}

// static
PyObject* PyConcreteAlternativeInstance::altFormat(PyObject* o, PyObject* args, PyObject* kwargs) {
    PyConcreteAlternativeInstance* self = (PyConcreteAlternativeInstance*)o;

    if (PyTuple_Size(args) != 1 || kwargs) {
        PyErr_Format(PyExc_TypeError, "__format__ invalid number of parameters");
        return NULL;
    }
    PyObjectStealer arg0(PyTuple_GetItem(args, 0));

    auto result = self->callMethod("__format__", arg0);
    if (!result.first) {
        PyErr_Format(PyExc_TypeError, "__format__ not defined for type %s", self->type()->name().c_str());
        return NULL;
    }
    if (!PyUnicode_Check(result.second)) {
        PyErr_Format(PyExc_TypeError, "__format__ returned non-string for type %s", self->type()->name().c_str());
        return NULL;
    }

    return result.second;
}

// static
PyObject* PyConcreteAlternativeInstance::altBytes(PyObject* o, PyObject* args, PyObject* kwargs) {
    PyConcreteAlternativeInstance* self = (PyConcreteAlternativeInstance*)o;

    if (PyTuple_Size(args) > 0 || kwargs) {
        PyErr_Format(PyExc_TypeError, "__bytes__ invalid number of parameters");
        return NULL;
    }

    auto result = self->callMethod("__bytes__");
    if (!result.first) {
        PyErr_Format(PyExc_TypeError, "__bytes__ not defined for type %s", self->type()->name().c_str());
        return NULL;
    }
    if (!PyBytes_Check(result.second)) {
        PyErr_Format(PyExc_TypeError, "__bytes__ returned non-bytes %s for type %s", result.second->ob_type->tp_name, self->type()->name().c_str());
        return NULL;
    }

    return result.second;
}

// static
PyObject* PyConcreteAlternativeInstance::altDir(PyObject* o, PyObject* args, PyObject* kwargs) {
    PyConcreteAlternativeInstance* self = (PyConcreteAlternativeInstance*)o;

    if (PyTuple_Size(args) > 0 || kwargs) {
        PyErr_Format(PyExc_TypeError, "__dir__ invalid number of parameters");
        return NULL;
    }

    auto result = self->callMethod("__dir__");
    if (!result.first) {
        PyErr_Format(PyExc_TypeError, "__dir__ missing");
        return NULL;
    }
    if (!PySequence_Check(result.second)) {
        PyErr_Format(PyExc_TypeError, "__dir__ returned non-sequence %s for type %s", result.second->ob_type->tp_name, self->type()->name().c_str());
        return NULL;
    }

    return result.second;
}

// static
PyObject* PyConcreteAlternativeInstance::altReversed(PyObject* o, PyObject* args, PyObject* kwargs) {
    PyConcreteAlternativeInstance* self = (PyConcreteAlternativeInstance*)o;

    if (PyTuple_Size(args) > 0 || kwargs) {
        PyErr_Format(PyExc_TypeError, "__reversed__ invalid number of parameters");
        return NULL;
    }

    auto result = self->callMethod("__reversed__");
    if (!result.first) {
        PyErr_Format(PyExc_TypeError, "__reversed__ missing");
        return NULL;
    }

    return result.second;
}

// static
PyObject* PyConcreteAlternativeInstance::altComplex(PyObject* o, PyObject* args, PyObject* kwargs) {
    PyConcreteAlternativeInstance* self = (PyConcreteAlternativeInstance*)o;

    if (PyTuple_Size(args) > 0 || kwargs) {
        PyErr_Format(PyExc_TypeError, "__complex__ invalid number of parameters");
        return NULL;
    }

    auto result = self->callMethod("__complex__");
    if (!result.first) {
        PyErr_Format(PyExc_TypeError, "__complex__ missing");
        return NULL;
    }
    if (!PyComplex_Check(result.second)) {
        PyErr_Format(PyExc_TypeError, "__complex__ returned non-complex %s for type %s", result.second->ob_type->tp_name, self->type()->name().c_str());
        return NULL;
    }

    return result.second;
}

// static
PyObject* PyConcreteAlternativeInstance::altRound(PyObject* o, PyObject* args, PyObject* kwargs) {
    PyConcreteAlternativeInstance* self = (PyConcreteAlternativeInstance*)o;

    if (PyTuple_Size(args) > 1 || kwargs) {
        PyErr_Format(PyExc_TypeError, "__round__ invalid number of parameters %d %d", PyTuple_Size(args), kwargs);
        return NULL;
    }

    if (PyTuple_Size(args) == 1) {
        PyObjectStealer arg0(PyTuple_GetItem(args, 0));
        auto result = self->callMethod("__round__", arg0);
        if (!result.first) {
            PyErr_Format(PyExc_TypeError, "__round__ missing");
            return NULL;
        }
        return result.second;
    }

    auto result = self->callMethod("__round__");
    if (!result.first) {
        PyErr_Format(PyExc_TypeError, "__round__ missing");
        return NULL;
    }
    if (!PyLong_Check(result.second)) {
        PyErr_Format(PyExc_TypeError, "__round__ returned non-integer %s for type %s", result.second->ob_type->tp_name, self->type()->name().c_str());
        return NULL;
    }

    return result.second;
}

// static
PyObject* PyConcreteAlternativeInstance::altTrunc(PyObject* o, PyObject* args, PyObject* kwargs) {
    PyConcreteAlternativeInstance* self = (PyConcreteAlternativeInstance*)o;

    if (PyTuple_Size(args) > 0 || kwargs) {
        PyErr_Format(PyExc_TypeError, "__trunc__ invalid number of parameters");
        return NULL;
    }

    auto result = self->callMethod("__trunc__");
    if (!result.first) {
        PyErr_Format(PyExc_TypeError, "__trunc__ missing");
        return NULL;
    }
    if (!PyLong_Check(result.second)) {
        PyErr_Format(PyExc_TypeError, "__trunc__ returned non-integer %s for type %s", result.second->ob_type->tp_name, self->type()->name().c_str());
        return NULL;
    }

    return result.second;
}

// static
PyObject* PyConcreteAlternativeInstance::altFloor(PyObject* o, PyObject* args, PyObject* kwargs) {
    PyConcreteAlternativeInstance* self = (PyConcreteAlternativeInstance*)o;

    if (PyTuple_Size(args) > 0 || kwargs) {
        PyErr_Format(PyExc_TypeError, "__floor__ invalid number of parameters");
        return NULL;
    }

    auto result = self->callMethod("__floor__");
    if (!result.first) {
        PyErr_Format(PyExc_TypeError, "__floor__ missing");
        return NULL;
    }
    if (!PyLong_Check(result.second)) {
        PyErr_Format(PyExc_TypeError, "__floor__ returned non-integer %s for type %s", result.second->ob_type->tp_name, self->type()->name().c_str());
        return NULL;
    }

    return result.second;
}

// static
PyObject* PyConcreteAlternativeInstance::altCeil(PyObject* o, PyObject* args, PyObject* kwargs) {
    PyConcreteAlternativeInstance* self = (PyConcreteAlternativeInstance*)o;

    if (PyTuple_Size(args) > 0 || kwargs) {
        PyErr_Format(PyExc_TypeError, "__ceil__ invalid number of parameters");
        return NULL;
    }

    auto result = self->callMethod("__ceil__");
    if (!result.first) {
        PyErr_Format(PyExc_TypeError, "__ceil__ missing");
        return NULL;
    }
    if (!PyLong_Check(result.second)) {
        PyErr_Format(PyExc_TypeError, "__ceil__ returned non-integer %s for type %s", result.second->ob_type->tp_name, self->type()->name().c_str());
        return NULL;
    }

    return result.second;
}

// static
PyObject* PyConcreteAlternativeInstance::altEnter(PyObject* o, PyObject* args, PyObject* kwargs) {
    PyConcreteAlternativeInstance* self = (PyConcreteAlternativeInstance*)o;

    if (PyTuple_Size(args) > 0 || kwargs) {
        PyErr_Format(PyExc_TypeError, "__enter__ invalid number of parameters");
        return NULL;
    }

    auto result = self->callMethod("__enter__");
    if (!result.first) {
        PyErr_Format(PyExc_TypeError, "__enter__ missing");
        return NULL;
    }

    return result.second;
}

// static
PyObject* PyConcreteAlternativeInstance::altExit(PyObject* o, PyObject* args, PyObject* kwargs) {
    PyConcreteAlternativeInstance* self = (PyConcreteAlternativeInstance*)o;

    if (PyTuple_Size(args) != 3 || kwargs) {
        PyErr_Format(PyExc_TypeError, "__exit__ invalid number of parameters");
        return NULL;
    }
    PyObjectStealer arg0(PyTuple_GetItem(args, 0));
    PyObjectStealer arg1(PyTuple_GetItem(args, 1));
    PyObjectStealer arg2(PyTuple_GetItem(args, 2));

    auto result = self->callMethod("__exit__", arg0, arg1, arg2);
    if (!result.first) {
        PyErr_Format(PyExc_TypeError, "__exit__ missing");
        return NULL;
    }

    return result.second;
}

// static
PyMethodDef* PyConcreteAlternativeInstance::typeMethodsConcrete(Type* t) {

    // List of magic methods that are not attached to direct function pointers in PyTypeObject.
    //   These need to be defined by adding entries to PyTypeObject.tp_methods
    //   and we need to avoid adding them to PyTypeObject.tp_dict ourselves.
    //   Also, we only want to add the entry to tp_methods if they are explicitly defined.
    const std::map<const char*, PyCFunction> special_magic_methods = {
            {"__format__", (PyCFunction)altFormat},
            {"__bytes__", (PyCFunction)altBytes},
            {"__dir__", (PyCFunction)altDir},
            {"__reversed__", (PyCFunction)altReversed},
            {"__complex__", (PyCFunction)altComplex},
            {"__round__", (PyCFunction)altRound},
            {"__trunc__", (PyCFunction)altTrunc},
            {"__floor__", (PyCFunction)altFloor},
            {"__ceil__", (PyCFunction)altCeil},
            {"__enter__", (PyCFunction)altEnter},
            {"__exit__", (PyCFunction)altExit}
        };

    int cur = 0;
    auto altMethods = ((ConcreteAlternative*)t)->getAlternative()->getMethods();
    PyMethodDef* ret = new PyMethodDef[special_magic_methods.size() + 1];
    for (auto m: special_magic_methods) {
        if (altMethods.find(m.first) != altMethods.end()) {
            ret[cur++] =  {m.first, m.second, METH_VARARGS | METH_KEYWORDS, NULL};
        }
    }
    ret[cur] = {NULL, NULL};
    return ret;
}
