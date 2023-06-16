#include "FunctionOverload.hpp"


/* static */
PyObject* FunctionOverload::buildFunctionObj(Type* closureType, instance_ptr closureData) const {
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
