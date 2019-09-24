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

#include "PyAlternativeInstance.hpp"
#include "PyFunctionInstance.hpp"

Alternative* PyAlternativeInstance::type() {
    return (Alternative*)extractTypeFrom(((PyObject*)this)->ob_type);
}

ConcreteAlternative* PyConcreteAlternativeInstance::type() {
    return (ConcreteAlternative*)extractTypeFrom(((PyObject*)this)->ob_type);
}

PyObject* PyAlternativeInstance::pyTernaryOperatorConcrete(PyObject* rhs, PyObject* thirdArg, const char* op, const char* opErr) {
    auto it = type()->getMethods().find(op);

    if (it != type()->getMethods().end()) {
        Function* f = it->second;

        // for now, restrict usage to only 2 arguments
        PyObjectStealer argTuple(
            PyTuple_Pack(2, (PyObject*)this, (PyObject*)rhs)
            );

        std::pair<bool, PyObject*> res =
            PyFunctionInstance::tryToCallAnyOverload(f, nullptr, argTuple, nullptr);
        if (res.first) {
            return res.second;
        }
    }

    return ((PyInstance*)this)->pyTernaryOperatorConcrete(rhs, thirdArg, op, opErr);
}
PyObject* PyConcreteAlternativeInstance::pyTernaryOperatorConcrete(PyObject* rhs, PyObject* thirdArg, const char* op, const char* opErr) {
    auto it = type()->getAlternative()->getMethods().find(op);

    if (it != type()->getAlternative()->getMethods().end()) {
        Function* f = it->second;

        // for now, restrict usage to only 2 arguments
        PyObjectStealer argTuple(
            PyTuple_Pack(2, (PyObject*)this, (PyObject*)rhs)
            );
        std::pair<bool, PyObject*> res =
            PyFunctionInstance::tryToCallAnyOverload(f, nullptr, argTuple, nullptr);
        if (res.first) {
            return res.second;
        }
    }

    return ((PyInstance*)this)->pyTernaryOperatorConcrete(rhs, thirdArg, op, opErr);
}

PyObject* PyAlternativeInstance::pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr) {
    auto it = type()->getMethods().find(op);

    if (it != type()->getMethods().end()) {
        Function* f = it->second;

        PyObjectStealer argTuple(
            PyTuple_Pack(2, (PyObject*)this, (PyObject*)rhs)
            );

        std::pair<bool, PyObject*> res =
            PyFunctionInstance::tryToCallAnyOverload(f, nullptr, argTuple, nullptr);
        if (res.first) {
            return res.second;
        }
    }

    return ((PyInstance*)this)->pyOperatorConcrete(rhs, op, opErr);
}
PyObject* PyConcreteAlternativeInstance::pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr) {
    auto it = type()->getAlternative()->getMethods().find(op);

    if (it != type()->getAlternative()->getMethods().end()) {
        Function* f = it->second;

        PyObjectStealer argTuple(
            PyTuple_Pack(2, (PyObject*)this, (PyObject*)rhs)
            );

        std::pair<bool, PyObject*> res =
            PyFunctionInstance::tryToCallAnyOverload(f, nullptr, argTuple, nullptr);
        if (res.first) {
            return res.second;
        }
    }

    return ((PyInstance*)this)->pyOperatorConcrete(rhs, op, opErr);
}

PyObject* PyAlternativeInstance::pyUnaryOperatorConcrete(const char* op, const char* opErr) {
    auto it = type()->getMethods().find(op);

    if (it != type()->getMethods().end()) {
        Function* f = it->second;

        PyObjectStealer argTuple(
            PyTuple_Pack(1, (PyObject*)this)
            );

        std::pair<bool, PyObject*> res =
            PyFunctionInstance::tryToCallAnyOverload(f, nullptr, argTuple, nullptr);
        if (res.first) {
            return res.second;
        }
    }

    return ((PyInstance*)this)->pyUnaryOperatorConcrete(op, opErr);
}
PyObject* PyConcreteAlternativeInstance::pyUnaryOperatorConcrete(const char* op, const char* opErr) {
    auto it = type()->getAlternative()->getMethods().find(op);

    if (it != type()->getAlternative()->getMethods().end()) {
        Function* f = it->second;

        PyObjectStealer argTuple(
            PyTuple_Pack(1, (PyObject*)this)
            );

        std::pair<bool, PyObject*> res =
            PyFunctionInstance::tryToCallAnyOverload(f, nullptr, argTuple, nullptr);
        if (res.first) {
            return res.second;
        }
    }

    return ((PyInstance*)this)->pyUnaryOperatorConcrete(op, opErr);
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

            PyInstance::copyConstructFromPythonInstance(alternativeEltType->getTypes()[0], p, arg);
        } else if (PyTuple_Size(args) == 0) {
            //construct an alternative from Kwargs
            constructFromPythonArguments(p, alt->elementType(), args, kwargs);
        } else {
            throw std::logic_error("Can only initialize " + alt->name() + " from python with kwargs or a single in-place argument");
        }
    });
}

void PyAlternativeInstance::copyConstructFromPythonInstanceConcrete(Alternative* altType, instance_ptr tgt, PyObject* pyRepresentation, bool isExplicit) {
    Type* argType = extractTypeFrom(pyRepresentation->ob_type);

    if (argType && argType->getTypeCategory() == Type::TypeCategory::catConcreteAlternative &&
            ((ConcreteAlternative*)argType)->getAlternative() == altType) {
        altType->copy_constructor(tgt, ((PyInstance*)pyRepresentation)->dataPtr());
        return;
    }

    PyInstance::copyConstructFromPythonInstanceConcrete(altType, tgt, pyRepresentation, isExplicit);
}


PyObject* PyAlternativeInstance::tp_getattr_concrete(PyObject* pyAttrName, const char* attrName) {
    if (mIsMatcher) {
        if (type()->subtypes()[type()->which(dataPtr())].first == attrName) {
            return incref(Py_True);
        }
        return incref(Py_False);
    }

    if (strcmp(attrName,"matches") == 0) {
        PyInstance* self = duplicate();

        self->mIteratorOffset = -1;
        self->mIsMatcher = true;

        return (PyObject*)self;
    }

    //see if its a method
    Alternative* toCheck = type();

    auto it = toCheck->getMethods().find(attrName);
    if (it != toCheck->getMethods().end()) {
        return PyMethod_New(
            (PyObject*)it->second->getOverloads()[0].getFunctionObj(),
            (PyObject*)this
            );
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
    std::pair<bool, PyObject*> p = callMethod("__getattribute__", pyAttrName, nullptr);
    if (PyErr_Occurred() && PyErr_ExceptionMatches(PyExc_AttributeError)) {
        PyErr_Clear();
    }
    else {
        if (p.first) {
            return p.second;
        }
    }

    if (mIsMatcher) {
        if (type()->getAlternative()->subtypes()[type()->getAlternative()->which(dataPtr())].first == attrName) {
            return incref(Py_True);
        }
        return incref(Py_False);
    }

    if (strcmp(attrName,"matches") == 0) {
        PyInstance* self = duplicate();

        self->mIteratorOffset = -1;
        self->mIsMatcher = true;

        return (PyObject*)self;
    }

    //see if its a method
    Alternative* toCheck = type()->getAlternative();

    auto it = toCheck->getMethods().find(attrName);
    if (it != toCheck->getMethods().end()) {
        return PyMethod_New(
            (PyObject*)it->second->getOverloads()[0].getFunctionObj(),
            (PyObject*)this
            );
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
        std::pair<bool, PyObject*> p = callMethod("__getattr__", pyAttrName, nullptr);
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

    for (auto method_pair: alt->getMethods()) {
        PyDict_SetItemString(
            pyType->tp_dict,
            method_pair.first.c_str(),
            (PyObject*)typeObj(method_pair.second)
        );
    }
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
        if (method_pair.first == "__format__"
                || method_pair.first == "__bytes__"
                || method_pair.first == "__dir__"
                )
            continue;
        PyDict_SetItemString(
            pyType->tp_dict,
            method_pair.first.c_str(),
            (PyObject*)typeObj(method_pair.second)
        );
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
        std::pair<bool, PyObject*> p = callMethod("__delattr__", attrName, attrVal);
        if (p.first)
            return 0;
    }
    else {
        std::pair<bool, PyObject*> p = callMethod("__setattr__", attrName, attrVal);
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
    auto res = PyFunctionInstance::tryToCallAnyOverload(f, (PyObject*)this, args, kwargs);
    if (res.first) {
        return res.second;
    }
    // else
    throw PythonExceptionSet();
}

// try to call user-defined hash method
// returns -1 if not defined or if it returns an invalid value
int64_t PyConcreteAlternativeInstance::tryCallHashMethod() {
    auto result = callMethod("__hash__", nullptr, nullptr);
    if (!result.first)
        return -1;
    if (!PyLong_Check(result.second))
        return -1;
    return PyLong_AsLong(result.second);
}

std::pair<bool, PyObject*> PyConcreteAlternativeInstance::callMethod(const char* name, PyObject* arg0, PyObject* arg1) {
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

int PyConcreteAlternativeInstance::pyInquiryConcrete(const char* op, const char* opErrRep) {
    // op == '__bool__'
    std::pair<bool, PyObject*> p = callMethod("__bool__", nullptr, nullptr);
    if (!p.first) {
        p = callMethod("__len__", nullptr, nullptr);
        // if neither __bool__ nor __len__ is available, return True
        if (!p.first)
            return 1;
    }
    return PyObject_IsTrue(p.second);
}

int PyConcreteAlternativeInstance::sq_contains_concrete(PyObject* item) {
    std::pair<bool, PyObject*> p = callMethod("__contains__", item, nullptr);
    if (!p.first) {
        return 0;
    }
    return PyObject_IsTrue(p.second);
}

Py_ssize_t PyConcreteAlternativeInstance::mp_and_sq_length_concrete() {
    std::pair<bool, PyObject*> p = callMethod("__len__", nullptr, nullptr);
    if (!p.first) {
        return 0;
    }
    if (!PyLong_Check(p.second)) {
        return 0;
    }
    return PyLong_AsLong(p.second);
}

PyObject* PyConcreteAlternativeInstance::sq_item_concrete(Py_ssize_t ix) {
    PyObjectStealer arg0(PyLong_FromUnsignedLong(ix));
    std::pair<bool, PyObject*> p = callMethod("__getitem__", arg0, nullptr);
    if (!p.first) {
        PyErr_Format(PyExc_TypeError, "__getitem__ not defined for type %s", type()->name().c_str());
        return NULL;
    }
    return p.second;
}

int PyConcreteAlternativeInstance::sq_ass_item_concrete(Py_ssize_t ix, PyObject* v) {
    PyObjectStealer arg0(PyLong_FromUnsignedLong(ix));
    std::pair<bool, PyObject*> p = callMethod("__setitem__", arg0, v);
    if (!p.first) {
        PyErr_Format(PyExc_TypeError, "__setitem__ not defined for type %s", type()->name().c_str());
        return -1;
    }
    if (!PyLong_Check(p.second)) {
        PyErr_Format(PyExc_TypeError, "__setitem__ returned non-integer");
        return -1;
    }
    return PyLong_AsLong(p.second);
}

PyObject* PyConcreteAlternativeInstance::tp_iter_concrete() {
    std::pair<bool, PyObject*> p = callMethod("__iter__", nullptr, nullptr);
    if (!p.first) {
        PyErr_Format(PyExc_TypeError, "__iter__ not defined for type %s", type()->name().c_str());
        return NULL;
    }
    return p.second;
}

PyObject* PyConcreteAlternativeInstance::tp_iternext_concrete() {
    std::pair<bool, PyObject*> p = callMethod("__next__", nullptr, nullptr);
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

    std::pair<bool, PyObject*> p = self_inst->callMethod(pyCompareFlagToMethod(pyComparisonOp), other, nullptr);
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

    auto result = self->callMethod("__format__", nullptr, nullptr);
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

    auto result = self->callMethod("__bytes__", nullptr, nullptr);
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

    auto result = self->callMethod("__dir__", nullptr, nullptr);
    if (!result.first) {
        PyErr_Format(PyExc_TypeError, "shouldn't happen");
        return NULL;
    }
    if (!PySequence_Check(result.second)) {
        PyErr_Format(PyExc_TypeError, "__dir__ returned non-sequence %s for type %s", result.second->ob_type->tp_name, self->type()->name().c_str());
        return NULL;
    }

    return result.second;
}

// static
PyMethodDef* PyConcreteAlternativeInstance::typeMethodsConcrete(Type* t) {
    const int max_entries = 3;
    int cur= 0;
    auto altMethods = ((ConcreteAlternative*)t)->getAlternative()->getMethods();
    PyMethodDef* ret = new PyMethodDef[max_entries + 1];
    if (altMethods.find("__format__") != altMethods.end()) {
        ret[cur++] =  {"__format__", (PyCFunction)PyConcreteAlternativeInstance::altFormat, METH_VARARGS | METH_KEYWORDS, NULL};
    }
    if (altMethods.find("__bytes__") != altMethods.end()) {
        ret[cur++] =  {"__bytes__", (PyCFunction)PyConcreteAlternativeInstance::altBytes, METH_VARARGS | METH_KEYWORDS, NULL};
    }
    if (altMethods.find("__dir__") != altMethods.end()) {
        ret[cur++] =  {"__dir__", (PyCFunction)PyConcreteAlternativeInstance::altDir, METH_VARARGS | METH_KEYWORDS, NULL};
    }
    ret[cur++] = {NULL, NULL};
    return ret;
}
