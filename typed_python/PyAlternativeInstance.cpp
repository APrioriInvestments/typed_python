#include "PyAlternativeInstance.hpp"
#include "PyFunctionInstance.hpp"

Alternative* PyAlternativeInstance::type() {
    return (Alternative*)extractTypeFrom(((PyObject*)this)->ob_type);
}

ConcreteAlternative* PyConcreteAlternativeInstance::type() {
    return (ConcreteAlternative*)extractTypeFrom(((PyObject*)this)->ob_type);
}

PyObject* PyAlternativeInstance::pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr) {
    auto it = type()->getMethods().find(op);

    if (it != type()->getMethods().end()) {
        Function* f = it->second;

        PyObject* argTuple = PyTuple_Pack(2, (PyObject*)this, rhs);

        for (const auto& overload: f->getOverloads()) {
            std::pair<bool, PyObject*> res = PyFunctionInstance::tryToCallOverload(overload, nullptr, argTuple, nullptr);
            if (res.first) {
                Py_DECREF(argTuple);
                return res.second;
            }

        Py_DECREF(argTuple);
        }
    }

    return ((PyInstance*)this)->pyOperatorConcrete(rhs, op, opErr);
}
PyObject* PyConcreteAlternativeInstance::pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr) {
    auto it = type()->getAlternative()->getMethods().find(op);

    if (it != type()->getAlternative()->getMethods().end()) {
        Function* f = it->second;

        PyObject* argTuple = PyTuple_Pack(2, (PyObject*)this, rhs);

        for (const auto& overload: f->getOverloads()) {
            std::pair<bool, PyObject*> res = PyFunctionInstance::tryToCallOverload(overload, nullptr, argTuple, nullptr);
            if (res.first) {
                Py_DECREF(argTuple);
                return res.second;
            }

        Py_DECREF(argTuple);
        }
    }

    return ((PyInstance*)this)->pyOperatorConcrete(rhs, op, opErr);
}