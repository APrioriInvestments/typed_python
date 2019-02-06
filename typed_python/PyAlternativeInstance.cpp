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

        PyObjectStealer argTuple(
            PyTuple_Pack(3, (PyObject*)this, (PyObject*)rhs, (PyObject*)thirdArg)
            );

        for (const auto& overload: f->getOverloads()) {
            std::pair<bool, PyObject*> res = PyFunctionInstance::tryToCallOverload(overload, nullptr, argTuple, nullptr);
            if (res.first) {
                return res.second;
            }
        }
    }

    return ((PyInstance*)this)->pyTernaryOperatorConcrete(rhs, thirdArg, op, opErr);
}
PyObject* PyConcreteAlternativeInstance::pyTernaryOperatorConcrete(PyObject* rhs, PyObject* thirdArg, const char* op, const char* opErr) {
    auto it = type()->getAlternative()->getMethods().find(op);

    if (it != type()->getAlternative()->getMethods().end()) {
        Function* f = it->second;

        PyObjectStealer argTuple(
            PyTuple_Pack(3, (PyObject*)this, (PyObject*)rhs, (PyObject*)thirdArg)
            );

        for (const auto& overload: f->getOverloads()) {
            std::pair<bool, PyObject*> res = PyFunctionInstance::tryToCallOverload(overload, nullptr, argTuple, nullptr);
            if (res.first) {
                return res.second;
            }
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

        for (const auto& overload: f->getOverloads()) {
            std::pair<bool, PyObject*> res = PyFunctionInstance::tryToCallOverload(overload, nullptr, argTuple, nullptr);
            if (res.first) {
                return res.second;
            }
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

        for (const auto& overload: f->getOverloads()) {
            std::pair<bool, PyObject*> res = PyFunctionInstance::tryToCallOverload(overload, nullptr, argTuple, nullptr);
            if (res.first) {
                return res.second;
            }
        }
    }

    return ((PyInstance*)this)->pyOperatorConcrete(rhs, op, opErr);
}

PyObject* PyAlternativeInstance::pyUnaryOperatorConcrete(const char* op, const char* opErr) {
    auto it = type()->getMethods().find(op);

    if (it != type()->getMethods().end()) {
        Function* f = it->second;

        PyObjectStealer argTuple(
            PyTuple_Pack(2, (PyObject*)this)
            );

        for (const auto& overload: f->getOverloads()) {
            std::pair<bool, PyObject*> res = PyFunctionInstance::tryToCallOverload(overload, nullptr, argTuple, nullptr);
            if (res.first) {
                return res.second;
            }
        }
    }

    return ((PyInstance*)this)->pyUnaryOperatorConcrete(op, opErr);
}
PyObject* PyConcreteAlternativeInstance::pyUnaryOperatorConcrete(const char* op, const char* opErr) {
    auto it = type()->getAlternative()->getMethods().find(op);

    if (it != type()->getAlternative()->getMethods().end()) {
        Function* f = it->second;

        PyObjectStealer argTuple(
            PyTuple_Pack(2, (PyObject*)this)
            );

        for (const auto& overload: f->getOverloads()) {
            std::pair<bool, PyObject*> res = PyFunctionInstance::tryToCallOverload(overload, nullptr, argTuple, nullptr);
            if (res.first) {
                return res.second;
            }
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
            PyObject* arg = PyTuple_GetItem(args, 0);
            Type* argType = extractTypeFrom(arg->ob_type);

            if (argType && argType->isBinaryCompatibleWith(alt)) {
                //it's already the right kind of instance, so we can copy-through the underlying element
                alt->elementType()->copy_constructor(p, alt->eltPtr(((PyInstance*)arg)->dataPtr()));
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