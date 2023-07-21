#include "PyInstance.hpp"
#include "FunctionType.hpp"

Instance ClosureVariableBinding::extractValueOrContainingClosure(Type* closureType, instance_ptr data) {
    for (long stepIx = 0; stepIx < size(); stepIx++) {
        ClosureVariableBindingStep step = (*this)[stepIx];

        if (step.isFunction()) {
            closureType = (Type*)step.getFunction();
        } else
        if (step.isNamedField()) {
            if (closureType->isPythonObjectOfType() && closureType == PythonObjectOfType::AnyPyDict()) {
                PyObject* dict = PythonObjectOfType::getPyObj(data);
                if (!dict || !PyDict_Check(dict)) {
                    throw std::runtime_error("Invalid closure: expected a populated PyDict");
                }
                PyObject* entry = PyDict_GetItemString(dict, step.getNamedField().c_str());
                if (!entry) {
                    // TODO: this is wrong. Really we need a pattern that lets us communicate
                    // the fact that this entry is empty and that we should throw a name error.
                    throw std::runtime_error("Invalid closure: expected " + step.getNamedField() + " of a Dict to be populated.");
                }

                return Instance::create(entry);
            } else
            if (closureType->isNamedTuple()) {
                NamedTuple* tupType = (NamedTuple*)closureType;
                auto it = tupType->getNameToIndex().find(step.getNamedField());
                if (it == tupType->getNameToIndex().end()) {
                    throw std::runtime_error(
                        "Invalid closure: expected NamedTuple to have field " +
                        step.getNamedField() + " but it doesn't."
                    );
                }

                closureType = tupType->getTypes()[it->second];
                data = data + tupType->getOffsets()[it->second];
            } else
            if (closureType->getTypeCategory() == Type::TypeCategory::catClass) {
                Class* clsType = (Class*)closureType;
                int index = clsType->getMemberIndex(step.getNamedField().c_str());
                if (index == -1) {
                    throw std::runtime_error("Can't find a field " + step.getNamedField() + " in class " + clsType->name());
                }

                if (!clsType->checkInitializationFlag(data, index)) {
                    throw std::runtime_error("Closure field " + step.getNamedField() + " is not populated.");
                }

                closureType = clsType->getMemberType(index);
                data = clsType->eltPtr(data, index);
            } else {
                throw std::runtime_error(
                    "Invalid closure: expected to find a Class or a NamedTuple."
                );
            }
        } else
        if (step.isIndexedField()) {
            if (!closureType->isComposite()) {
                throw std::runtime_error("Invalid closure: expected a NamedTuple or Tuple but got " + closureType->name());
            }

            CompositeType* tupType = (CompositeType*)closureType;

            if (step.getIndexedField() < 0 || step.getIndexedField() >= tupType->getTypes().size()) {
                throw std::runtime_error(
                    "Invalid closure: index " + format(step.getIndexedField()) + " is out of bounds in closure " + tupType->name()
                );
            }

            closureType = tupType->getTypes()[step.getIndexedField()];
            data = data + tupType->getOffsets()[step.getIndexedField()];
        } else
        if (step.isCellAccess()) {
            if (!(closureType->getTypeCategory() == Type::TypeCategory::catPyCell ||
                    closureType->getTypeCategory() == Type::TypeCategory::catTypedCell)) {
                throw std::runtime_error(
                    "Invalid closure: expected a cell, but got "
                    + Type::categoryToString(closureType->getTypeCategory())
                );
            }

            if (stepIx + 1 == size()) {
                // do nothing, because this function grabs the containing
                // closure
            } else {
                if (closureType->getTypeCategory() == Type::TypeCategory::catPyCell) {
                    throw std::runtime_error("Corrupt closure encountered: a PyCell should always be the last step");
                }

                // it's a typed closure
                data = ((TypedCellType*)closureType)->get(data);
                closureType = ((TypedCellType*)closureType)->getHeldType();
            }
        } else {
            throw std::runtime_error("Corrupt closure variable binding enountered.");
        }
    }

    return Instance(data, closureType);
}
