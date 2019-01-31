#include "PyCompositeTypeInstance.hpp"

CompositeType* PyCompositeTypeInstance::type() {
    return (CompositeType*)extractTypeFrom(((PyObject*)this)->ob_type);
}

Tuple* PyTupleInstance::type() {
    return (Tuple*)extractTypeFrom(((PyObject*)this)->ob_type);
}

NamedTuple* PyNamedTupleInstance::type() {
    return (NamedTuple*)extractTypeFrom(((PyObject*)this)->ob_type);
}

PyObject* PyCompositeTypeInstance::sq_item_concrete(Py_ssize_t ix) {
    if (ix < 0 || ix >= (int64_t)type()->getTypes().size()) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }

    Type* eltType = type()->getTypes()[ix];

    return extractPythonObject(type()->eltPtr(dataPtr(), ix), eltType);
}


Py_ssize_t PyCompositeTypeInstance::mp_and_sq_length() {
    return type()->getTypes().size();
}
