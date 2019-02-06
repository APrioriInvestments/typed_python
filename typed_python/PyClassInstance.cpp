#include "PyClassInstance.hpp"

Class* PyClassInstance::type() {
    return (Class*)extractTypeFrom(((PyObject*)this)->ob_type);
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

    PyObject* targetArgTuple = PyTuple_New(argCount);

    PyTuple_SetItem(targetArgTuple, 0, incref((PyObject*)this)); //steals a reference

    if (arg0) {
        PyTuple_SetItem(targetArgTuple, 1, incref(arg0)); //steals a reference
    }

    if (arg1) {
        PyTuple_SetItem(targetArgTuple, 2, incref(arg1)); //steals a reference
    }

    bool threw = false;
    bool ran = false;

    for (const auto& overload: method->getOverloads()) {
        std::pair<bool, PyObject*> res = PyFunctionInstance::tryToCallOverload(overload, nullptr, targetArgTuple, nullptr);
        if (res.first) {
            //res.first is true if we matched and tried to call this function
            if (res.second) {
                //don't need the result.
                Py_DECREF(targetArgTuple);
                return std::make_pair(true, res.second);
            } else {
                //it threw an exception
                Py_DECREF(targetArgTuple);
                return std::make_pair(true, (PyObject*)nullptr);
            }
        }
    }

    PyErr_Format(PyExc_TypeError, "'%s.%s' cannot find a valid overload with these arguments", type()->name().c_str(), name);
    return std::make_pair(true, (PyObject*)nullptr);
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

    PyObject* targetArgTuple = PyTuple_New(PyTuple_Size(args)+1);

    PyTuple_SetItem(targetArgTuple, 0, selfAsObject); //steals the reference to the new 'selfAsObject'

    for (long k = 0; k < PyTuple_Size(args); k++) {
        PyTuple_SetItem(targetArgTuple, k+1, incref(PyTuple_GetItem(args, k))); //have to incref because of stealing
    }

    bool threw = false;
    bool ran = false;

    for (const auto& overload: initMethod->getOverloads()) {
        std::pair<bool, PyObject*> res = PyFunctionInstance::tryToCallOverload(overload, nullptr, targetArgTuple, kwargs);
        if (res.first) {
            //res.first is true if we matched and tried to call this function
            if (res.second) {
                //don't need the result.
                Py_DECREF(res.second);
                ran = true;
            } else {
                //it threw an exception
                ran = true;
                threw = true;
            }

            break;
        }
    }

    Py_DECREF(targetArgTuple);

    if (!ran) {
        throw std::runtime_error("Cannot find a valid overload of __init__ with these arguments.");
    }

    if (threw) {
        throw PythonExceptionSet();
    }
}