#include "PyConstDictInstance.hpp"

ConstDict* PyConstDictInstance::type() {
    return (ConstDict*)extractTypeFrom(((PyObject*)this)->ob_type);
}

PyObject* PyConstDictInstance::sq_concat_concrete(PyObject* rhs) {
    Type* rhs_type = extractTypeFrom(rhs->ob_type);

    if (type() == rhs_type) {
        PyInstance* w_rhs = (PyInstance*)rhs;

        return PyInstance::initialize(type(), [&](instance_ptr data) {
            type()->addDicts(dataPtr(), w_rhs->dataPtr(), data);
        });
    } else {
        Instance other(type(), [&](instance_ptr data) {
            copyConstructFromPythonInstance(type(), data, rhs);
        });

        return PyInstance::initialize(type(), [&](instance_ptr data) {
            type()->addDicts(dataPtr(), other.data(), data);
        });
    }
}

