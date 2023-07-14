#include "FunctionOverload.hpp"


/* static */
PyObject* FunctionOverload::buildFunctionObj(Type* closureType, instance_ptr closureData) {
    if (mCachedFunctionObj) {
        return incref(mCachedFunctionObj);
    }

    std::set<std::string> globalsInCells(
        mFunctionClosureVarnames.begin(),
        mFunctionClosureVarnames.end()
    );

    PyObjectStealer funcGlobals(PyDict_New());

    for (auto& nameAndGlobal: mGlobals) {
        if (globalsInCells.find(nameAndGlobal.first) == globalsInCells.end()) {
            PyObject* val = nameAndGlobal.second.getValueAsPyobj();

            if (val) {
                PyDict_SetItemString(
                    (PyObject*)funcGlobals,
                    nameAndGlobal.first.c_str(),
                    val
                );
            }
        }
    }

    PyObject* res = PyFunction_New(mFunctionCode, (PyObject*)funcGlobals);

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

            if (varname == "__class__" && getMethodOf()) {
                PyTuple_SetItem(
                    (PyObject*)closureTup,
                    k,
                    PyCell_New((PyObject*)PyInstance::typeObj(
                        ((HeldClass*)getMethodOf())->getClassType()
                    ))
                );
            } else {
                auto globalIt = mGlobals.find(varname);
                if (globalIt != mGlobals.end()) {
                    PyObject* globalVal = globalIt->second.getValueAsPyobj();

                    PyTuple_SetItem(
                        (PyObject*)closureTup,
                        k,
                        PyCell_New(globalVal)
                    );
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
                            PyCell_New(
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
