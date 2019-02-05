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
            copyConstructFromPythonInstance(eltType, tempObj, attrVal);
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

std::pair<bool, PyObject*> PyClassInstance::callMemberFunction(const char* name, PyObject* arg0) {
    auto it = type()->getMemberFunctions().find(name);

    if (it == type()->getMemberFunctions().end()) {
        return std::make_pair(false, (PyObject*)nullptr);
    }

    Function* method = it->second;

    int argCount = 1;
    if (arg0) {
        argCount += 1;
    }

    PyObject* targetArgTuple = PyTuple_New(argCount);

    PyTuple_SetItem(targetArgTuple, 0, incref((PyObject*)this)); //steals a reference

    if (arg0) {
        PyTuple_SetItem(targetArgTuple, 1, incref(arg0)); //steals a reference
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
