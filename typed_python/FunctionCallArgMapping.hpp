/******************************************************************************
   Copyright 2017-2019 typed_python Authors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

#pragma once

#include "FunctionType.hpp"
#include "PyInstance.hpp"

/*******
Utility class for mapping arguments in a function call to args, keyword args, *args, and **kwargs
in a python function signature.
*******/

class FunctionCallArgMapping {
    typedef PyObject* py_obj_ptr;

    FunctionCallArgMapping(const FunctionCallArgMapping&) = delete;
    FunctionCallArgMapping& operator=(const FunctionCallArgMapping&) = delete;

public:
    FunctionCallArgMapping(const Function::Overload& overload) :
            mArgs(overload.getArgs()),
            mCurrentArgumentIx(0),
            mIsValid(true)
    {
        mSingleValueArgs.resize(mArgs.size());
    }

    ~FunctionCallArgMapping() {
        for (auto td: mToTeardown) {
            decref(td);
        }
    }

    void coerceToType(py_obj_ptr& ptr, Type* target, ConversionLevel level) {
        try {
            PyObject* coerced = PyInstance::initializePythonRepresentation(
                target,
                [&](instance_ptr data) {
                    PyInstance::copyConstructFromPythonInstance(
                        target,
                        data,
                        ptr,
                        level
                    );
                }
            );

            mToTeardown.push_back(coerced);

            ptr = coerced;
        }
        catch(PythonExceptionSet& e) {
            PyErr_Clear();
            mIsValid = false;
        }
        catch(...) {
            mIsValid = false;
        }
    }

    void applyTypeCoercion(ConversionLevel level) {
        for (long k = 0; k < mArgs.size(); k++) {
            if (mArgs[k].getTypeFilter()) {
                if (mArgs[k].getIsNormalArg()) {
                    coerceToType(mSingleValueArgs[k], mArgs[k].getTypeFilter(), level);
                } else if (mArgs[k].getIsStarArg()) {
                    for (long j = 0; j < mStarArgValues.size(); j++) {
                        coerceToType(mStarArgValues[j], mArgs[k].getTypeFilter(), level);
                    }
                } else if (mArgs[k].getIsKwarg()) {
                    for (long j = 0; j < mKwargValues.size(); j++) {
                        coerceToType(mKwargValues[j].second, mArgs[k].getTypeFilter(), level);
                    }
                }
            }
        }
    }

    void pushPositionalArg(PyObject* arg) {
        if (!arg) {
            PyErr_Format(
                PyExc_RuntimeError,
                "Can't push an empty positional argument!"
            );
            throw PythonExceptionSet();
        }

        if (!mIsValid) {
            return;
        }

        if (mCurrentArgumentIx >= mArgs.size()) {
            mIsValid = false;
            return;
        }

        if (mArgs[mCurrentArgumentIx].getIsNormalArg()) {
            mSingleValueArgs[mCurrentArgumentIx++] = arg;
        } else if (mArgs[mCurrentArgumentIx].getIsStarArg()) {
            mStarArgValues.push_back(arg);
        } else if (mArgs[mCurrentArgumentIx].getIsKwarg()) {
            mIsValid = false;
        }
    }

    void pushKeywordArg(std::string argName, PyObject* arg) {
        if (!mIsValid) {
            return;
        }

        for (long k = 0; k < mArgs.size(); k++) {
            if (mArgs[k].getIsNormalArg() && mArgs[k].getName() == argName) {
                if (mSingleValueArgs[k]) {
                    mIsValid = false;
                    return;
                }

                mSingleValueArgs[k] = arg;
                return;
            } else if (mArgs[k].getIsStarArg()) {
                // do nothing - just skip.
            } else if (mArgs[k].getIsKwarg()) {
                mKwargValues.push_back(std::make_pair(argName, arg));
                return;
            }
        }

        // we failed to place the argument
        mIsValid = false;
        return;
    }

    void finishedPushing() {
        if (!mIsValid) {
            return;
        }

        for (long k = 0; k < mArgs.size(); k++) {
            if (mArgs[k].getIsNormalArg() && !mSingleValueArgs[k]) {
                if (mArgs[k].getDefaultValue()) {
                    mSingleValueArgs[k] = mArgs[k].getDefaultValue();
                } else {
                    mIsValid = false;
                    return;
                }
            }
        }
    }

    void pushArguments(PyObject* self, PyObject* args, PyObject* kwargs) {
        if (self) {
            pushPositionalArg(self);
        }

        for (long k = 0; k < PyTuple_Size(args); k++) {
            pushPositionalArg(PyTuple_GetItem(args, k));
        }

        if (kwargs) {
            PyObject *key, *value;
            Py_ssize_t pos = 0;

            while (PyDict_Next(kwargs, &pos, &key, &value)) {
                if (!PyUnicode_Check(key)) {
                    PyErr_SetString(PyExc_TypeError, "Keywords arguments must be strings.");
                    throw PythonExceptionSet();
                }

                pushKeywordArg(PyUnicode_AsUTF8(key), value);
            }
        }

        finishedPushing();
    }

    // return true if we can show we don't match without having to do
    // type coersion
    bool definitelyDoesntMatch(ConversionLevel conversionLevel) {
        if (!mIsValid) {
            return true;
        }

        for (long k = 0; k < mArgs.size(); k++) {
            auto& arg = mArgs[k];

            if (arg.getIsNormalArg() && arg.getTypeFilter()) {
                if (k >= mSingleValueArgs.size()) {
                    return false;
                } else {
                    if (!mSingleValueArgs[k]) {
                        PyErr_Format(
                            PyExc_RuntimeError,
                            "Found an empty arg at slot %d / %d",
                            k,
                            mSingleValueArgs.size()
                        );
                        throw PythonExceptionSet();
                    }

                    if (!PyInstance::pyValCouldBeOfType(
                        arg.getTypeFilter(),
                        mSingleValueArgs[k],
                        conversionLevel
                        )
                    ) {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    bool isValid() const {
        return mIsValid;
    }

    const std::vector<PyObject*>& getSingleValueArgs() const {
        return mSingleValueArgs;
    }

    const std::vector<PyObject*>& getStarArgValues() const {
        return mStarArgValues;
    }

    const std::vector<std::pair<std::string, PyObject*> >& getKwargValues() const {
        return mKwargValues;
    }

    // flatten all the args for passing to regular python
    PyObject* buildPositionalArgTuple(bool returnTypes=false) const {
        size_t count = 0;
        for (long k = 0; k < mArgs.size(); k++) {
            if (mArgs[k].getIsNormalArg()) {
                count++;
            } else if (mArgs[k].getIsStarArg()) {
                count += mStarArgValues.size();
            }
        }

        PyObject* tup = PyTuple_New(count);

        count = 0;

        for (long k = 0; k < mArgs.size(); k++) {
            if (mArgs[k].getIsNormalArg()) {
                if (!mSingleValueArgs[k]) {
                    throw std::runtime_error("Expected a populated value here.");
                }

                PyTuple_SetItem(tup, count++, incref(returnTypes ? (PyObject*)mSingleValueArgs[k]->ob_type : mSingleValueArgs[k]));
            } else if (mArgs[k].getIsStarArg()) {
                for (auto subElt: mStarArgValues) {
                    PyTuple_SetItem(tup, count++, incref(returnTypes ? (PyObject*)subElt->ob_type : subElt));
                }
            }
        }

        return tup;
    }

    PyObject* buildStarArgTuple() const {
        for (long k = 0; k < mArgs.size(); k++) {
            if (mArgs[k].getIsStarArg()) {
                PyObject* res = PyTuple_New(mStarArgValues.size());
                long count = 0;
                for (auto subElt: mStarArgValues) {
                    PyTuple_SetItem(res, count++, incref(subElt));
                }
                return res;
            }
        }

        return nullptr;
    }

    PyObject* buildKeywordArgTuple(bool returnTypes=false) const {
        for (long k = 0; k < mArgs.size(); k++) {
            if (mArgs[k].getIsKwarg()) {
                PyObject* res = PyDict_New();
                for (auto nameAndObj: mKwargValues) {
                    PyDict_SetItemString(
                        res,
                        nameAndObj.first.c_str(),
                        returnTypes ? (PyObject*)nameAndObj.second->ob_type : nameAndObj.second
                    );
                }
                return res;
            }
        }

        return nullptr;
    }

    PyObject* extractFunctionArgumentValues() const {
        PyObject* res = PyTuple_New(mArgs.size());

        for (long k = 0; k < mArgs.size(); k++) {
            if (mArgs[k].getIsNormalArg()) {
                PyTuple_SetItem(res, k, incref(mSingleValueArgs[k]));
            } else if (mArgs[k].getIsStarArg()) {
                PyTuple_SetItem(res, k, buildStarArgTuple());
            } else if (mArgs[k].getIsKwarg()) {
                PyTuple_SetItem(res, k, buildKeywordArgTuple());
            } else {
                throw std::runtime_error("unreachable");
            }
        }

        return res;
    }

    std::pair<Instance, bool> extractArgWithType(int argIx, Type* argType) const {
        if (mArgs[argIx].getIsNormalArg()) {
            try {
                return std::make_pair(
                    Instance::createAndInitialize(argType, [&](instance_ptr p) {
                        PyInstance::copyConstructFromPythonInstance(
                            argType, p, mSingleValueArgs[argIx], ConversionLevel::Signature
                        );
                    }),
                    true
                );
            } catch(PythonExceptionSet& s) {
                // failed to convert, but keep going
                PyErr_Clear();
                return std::pair<Instance, bool>(Instance(), false);
            }
            catch(...) {
                // not a valid conversion
                return std::pair<Instance, bool>(Instance(), false);
            }
        } else if (mArgs[argIx].getIsStarArg()) {
            if (argType->getTypeCategory() != Type::TypeCategory::catTuple) {
                throw std::runtime_error("Invalid type signature for *args: " + argType->name());
            }

            Tuple* tup = (Tuple*)argType;

            if (mStarArgValues.size() != tup->getTypes().size()) {
                return std::pair<Instance, bool>(Instance(), false);
            }

            try {
                return std::make_pair(
                    Instance::createAndInitialize(tup, [&](instance_ptr p) {
                        tup->constructor(p, [&](instance_ptr subElt, int tupArg) {
                            PyInstance::copyConstructFromPythonInstance(
                                tup->getTypes()[tupArg], subElt, mStarArgValues[tupArg],
                                ConversionLevel::Signature
                            );
                        });
                    }),
                    true
                );
            } catch(PythonExceptionSet& s) {
                // failed to convert, but keep going
                PyErr_Clear();
                return std::pair<Instance, bool>(Instance(), false);
            }
            catch(...) {
                // not a valid conversion
                return std::pair<Instance, bool>(Instance(), false);
            }
        } else if (mArgs[argIx].getIsKwarg()) {
            if (argType->getTypeCategory() != Type::TypeCategory::catNamedTuple) {
                throw std::runtime_error("Invalid type signature for **kwargs");
            }

            NamedTuple* tup = (NamedTuple*)argType;

            if (mKwargValues.size() != tup->getTypes().size()) {
                return std::pair<Instance, bool>(Instance(), false);
            }

            for (long k = 0; k < mKwargValues.size(); k++) {
                if (mKwargValues[k].first != tup->getNames()[k]) {
                    return std::pair<Instance, bool>(Instance(), false);
                }
            }

            try {
                return std::make_pair(
                    Instance::createAndInitialize(tup, [&](instance_ptr p) {
                        tup->constructor(p, [&](instance_ptr subElt, int tupArg) {
                            PyInstance::copyConstructFromPythonInstance(
                                tup->getTypes()[tupArg],
                                subElt,
                                mKwargValues[tupArg].second,
                                ConversionLevel::Signature
                            );
                        });
                    }),
                    true
                );
            } catch(PythonExceptionSet& s) {
                // failed to convert, but keep going
                PyErr_Clear();
                return std::pair<Instance, bool>(Instance(), false);
            }
            catch(...) {
                // not a valid conversion
                return std::pair<Instance, bool>(Instance(), false);
            }
        }

        throw std::runtime_error("unreachable");
    }


private:
    const std::vector<Function::FunctionArg>& mArgs;

    std::vector<PyObject*> mSingleValueArgs;
    std::vector<PyObject*> mStarArgValues;
    std::vector<std::pair<std::string, PyObject*> > mKwargValues;

    std::vector<PyObject*> mToTeardown;

    bool mIsValid;

    size_t mCurrentArgumentIx;
};
