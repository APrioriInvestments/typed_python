#include "PyInstance.hpp"
#include "FunctionType.hpp"

Instance ClosureVariableBinding::extractValueOrContainingClosure(Type* closureType, instance_ptr data) {
    for (long stepIx = 0; stepIx < size(); stepIx++) {
        ClosureVariableBindingStep step = (*this)[stepIx];

        if (step.isFunction()) {
            closureType = (Type*)step.getFunction();
        } else
        if (step.isNamedField()) {
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


/* static */
PyObject* Function::Overload::buildFunctionObj(Type* closureType, instance_ptr closureData) const {
    if (mCachedFunctionObj) {
        return incref(mCachedFunctionObj);
    }

    PyObject* res = PyFunction_New(mFunctionCode, mFunctionGlobals);

    if (!res) {
        throw PythonExceptionSet();
    }

    if (mFunctionDefaults) {
        if (PyFunction_SetDefaults(res, mFunctionDefaults) == -1) {
            throw PythonExceptionSet();
        }
    }

    if (mFunctionAnnotations) {
        if (PyFunction_SetAnnotations(res, mFunctionAnnotations) == -1) {
            throw PythonExceptionSet();
        }
    }

    int closureVarCount = PyCode_GetNumFree((PyCodeObject*)mFunctionCode);

    if (mFunctionClosureVarnames.size() != closureVarCount) {
        throw std::runtime_error(
            "Invalid closure: wrong number of cells: wanted " +
            format(closureVarCount) +
            " but got " +
            format(mFunctionClosureVarnames.size())
        );
    }

    if (closureVarCount) {
        PyObjectStealer closureTup(PyTuple_New(closureVarCount));

        for (long k = 0; k < closureVarCount; k++) {
            std::string varname = mFunctionClosureVarnames[k];

            auto globalIt = mFunctionGlobalsInCells.find(varname);
            if (globalIt != mFunctionGlobalsInCells.end()) {
                if (varname == "__class__" && getMethodOf()) {
                    PyTuple_SetItem(
                        (PyObject*)closureTup,
                        k,
                        PyCell_New((PyObject*)PyInstance::typeObj(
                            ((HeldClass*)getMethodOf())->getClassType()
                        ))
                    );
                } else {
                    PyTuple_SetItem(
                        (PyObject*)closureTup,
                        k,
                        incref(globalIt->second)
                    );
                }
            } else {
                auto bindingIt = mClosureBindings.find(varname);
                if (bindingIt == mClosureBindings.end()) {
                    throw std::runtime_error("Closure variable " + varname + " not in globals or closure bindings.");
                }

                ClosureVariableBinding binding = bindingIt->second;

                Instance bindingValue = binding.extractValueOrContainingClosure(closureType, closureData);

                if (bindingValue.type()->getTypeCategory() == Type::TypeCategory::catPyCell) {
                    PyTuple_SetItem(
                        (PyObject*)closureTup,
                        k,
                        incref(
                            ((PyCellType*)bindingValue.type())->getPyObj(bindingValue.data())
                        )
                    );
                } else {
                    // it's not a cell. We have to encode it as a PyCell for the interpreter
                    // to be able to handle it.
                    PyObjectHolder newCellContents;

                    if (bindingValue.type()->getTypeCategory() == Type::TypeCategory::catTypedCell) {
                        newCellContents.steal(
                            PyInstance::extractPythonObject(
                                ((TypedCellType*)bindingValue.type())->get(bindingValue.data()),
                                ((TypedCellType*)bindingValue.type())->getHeldType()
                            )
                        );
                    } else {
                        newCellContents.steal(
                            PyInstance::fromInstance(bindingValue)
                        );
                    }

                    PyTuple_SetItem(
                        (PyObject*)closureTup,
                        k,
                        PyCell_New(newCellContents)
                    );
                }
            }
        }

        if (PyFunction_SetClosure(res, (PyObject*)closureTup) == -1) {
            throw PythonExceptionSet();
        }
    }

    if (closureType->bytecount() == 0) {
        mCachedFunctionObj = incref(res);
    }

    return res;
}
