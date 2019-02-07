#include "PyBoundMethodInstance.hpp"
#include "PyFunctionInstance.hpp"

BoundMethod* PyBoundMethodInstance::type() {
    return (BoundMethod*)extractTypeFrom(((PyObject*)this)->ob_type);
}



PyObject* PyBoundMethodInstance::tp_call_concrete(PyObject* args, PyObject* kwargs) {
    Function* f = type()->getFunction();
    Type* c = type()->getFirstArgType();

    PyObjectStealer objectInstance(
        PyInstance::initializePythonRepresentation(c, [&](instance_ptr d) {
            c->copy_constructor(d, dataPtr());
        })
    );

    for (const auto& overload: f->getOverloads()) {
        std::pair<bool, PyObject*> res = PyFunctionInstance::tryToCallOverload(overload, objectInstance, args, kwargs);
        if (res.first) {
            return res.second;
        }
    }

    std::string argTupleTypeDesc = PyFunctionInstance::argTupleTypeDescription(args, kwargs);

    PyErr_Format(
        PyExc_TypeError, "'%s' cannot find a valid overload with arguments of type %s",
        type()->name().c_str(),
        argTupleTypeDesc.c_str()
        );

    return NULL;
}

void PyBoundMethodInstance::mirrorTypeInformationIntoPyTypeConcrete(BoundMethod* methodT, PyTypeObject* pyType) {
    PyDict_SetItemString(pyType->tp_dict, "FirstArgType", typePtrToPyTypeRepresentation(methodT->getFirstArgType()));
    PyDict_SetItemString(pyType->tp_dict, "Function", typePtrToPyTypeRepresentation(methodT->getFunction()));
}


