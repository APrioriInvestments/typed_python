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

#include "PyInstance.hpp"
#include "PyFunctionInstance.hpp"

/******************
Walk a graph of python function and cell objects, building a new graph
where each 'cell' becomes strongly typed, and where we walk the closures
of cells that contain functions.

Each cell in the closure can be one of two things:

1. a normal value, that we will map into a typed value if possible
   (int, strings, etc). We won't try to map normal python objects
   such as tuples if they're not already typed.
2. a function object, which will bind the closure with a different
   type to represent the new code.

*******************/

class TypedClosureBuilder {
public:
    TypedClosureBuilder() :
        mVerbose(true)
    {
    }

    // primary entrypoint to the builder.
    PyObject* convert(PyObject* o) {
        // 'true' indicates that we went to filter out empty closures.
        // we only want to do that here, because there is no reason to process
        // 'Function' type objects that don't have complex closures.
        // however, we do want to process any such closures inside because
        // we don't want to have to bind the full closure to them.
        if (!isFunctionObject(o, true)) {
            return incref(o);
        }

        // find all the functions and PyCells reachable in this closure
        walkFunction(o, Path());

        // build the recursive type closure.
        buildTypes();

        return buildFinalResult();
    }

    bool isFunctionObject(PyObject* o, bool excludeEmptyClosures=false) {
        if (PyFunction_Check(o)) {
            return true;
        }

        Type* argT = PyInstance::extractTypeFrom(o->ob_type);

        if (!(argT && argT->getTypeCategory() == Type::TypeCategory::catFunction)) {
            return false;
        }

        Function* funcT = (Function*)argT;

        if ((funcT->isEmptyClosure() && excludeEmptyClosures) || !funcT->isFullyUntypedClosure()) {
            return false;
        }

        return true;
    }

    PyObjectHolder unwrapToFuncObj(PyObject* o) {
        PyObjectHolder result(o);

        if (PyFunction_Check(o)) {
            PyObjectStealer name(PyObject_GetAttrString(o, "__name__"));

            Function* funcType = PyFunctionInstance::convertPythonObjectToFunctionType(name, o, false, true);

            if (!funcType) {
                throw PythonExceptionSet();
            }

            Instance funcAsInstance = Instance::createAndInitialize(
                funcType,
                [&](instance_ptr ptr) {
                    PyInstance::copyConstructFromPythonInstance(funcType, ptr, o, ConversionLevel::New);
                }
            );

            result.steal(PyInstance::extractPythonObject(funcAsInstance));
        }

        return result;
    }

    void walkCell(PyObject* cell, Path p) {
        // don't walk anything more than one time.
        if (mObjectPaths.find(cell) != mObjectPaths.end()) {
            return;
        }

        mObjectPaths[cell] = p;

        Path contentsPath = p + 0;

        PyObject* cellContents = PyCell_GET(cell);

        if (!cellContents) {
            // nothing to do, as the cell is empty
            mCellTypes[contentsPath] = nullptr;
            return;
        }

        // if it's a function, walk down into it. the function walker will decide
        // if we've already seen the function before.
        if (isFunctionObject(cellContents)) {
            walkFunction(cellContents, contentsPath);
            return;
        }

        // if not, don't walk into it again.
        if (mObjectPaths.find(cellContents) != mObjectPaths.end()) {
            // we've already seen this cell contents.
            return;
        }

        mObjectPaths[cellContents] = contentsPath;
        mPathToCellContents[contentsPath] = cellContents;

        Type* argT = PyInstance::extractTypeFrom(cellContents->ob_type);

        if (argT) {
            mCellTypes[contentsPath] = argT;
            return;
        }

        if (PyType_Check(cellContents)) {
            argT = PyInstance::tryUnwrapPyInstanceToValueType(cellContents, true);

            if (argT) {
                mCellTypes[contentsPath] = argT;
                return;
            }
        }

        // this is how generic objects can communicate to TypedPython that they should be
        // considered 'compile time' objects.
        if (PyObject_HasAttrString(cellContents, "__typed_python_is_compile_time_constant__")) {
            Type* type = PyInstance::tryUnwrapPyInstanceToValueType(cellContents, true);

            if (type) {
                mCellTypes[contentsPath] = type;
                return;
            }
        }


        if (PyLong_Check(cellContents)) {
            mCellTypes[contentsPath] = Int64::Make();
        } else if (PyUnicode_Check(cellContents)) {
            mCellTypes[contentsPath] = StringType::Make();
        } else if (PyBytes_Check(cellContents)) {
            mCellTypes[contentsPath] = BytesType::Make();
        } else if (PyFloat_Check(cellContents)) {
            mCellTypes[contentsPath] = Float64::Make();
        } else if (cellContents == Py_None) {
            mCellTypes[contentsPath] = NoneType::Make();
        } else if (cellContents == Py_True || cellContents == Py_False) {
            mCellTypes[contentsPath] = Bool::Make();
        } else {
            mCellTypes[contentsPath] = PythonObjectOfType::AnyPyObject();
        }
    }

    void walkFunction(PyObject* o, Path p) {
        // don't walk anything more than one time.
        if (mObjectPaths.find(o) != mObjectPaths.end()) {
            return;
        }

        mObjectPaths[o] = p;

        //convert untyped functions to typed functions.
        PyObjectHolder funcObj = unwrapToFuncObj(o);

        Type* argT = PyInstance::extractTypeFrom(funcObj->ob_type);

        if (!(argT && argT->getTypeCategory() == Type::TypeCategory::catFunction)) {
            throw std::runtime_error("Expected this to be a function object.");
        }

        Function* funcT = (Function*)argT;
        instance_ptr closureData = ((PyInstance*)(PyObject*)funcObj)->dataPtr();

        mUnresolvedFunctionTypes[p] = funcT;
        mUnresolvedFunctionObjects[p] = o;

        // even though we have a two-level structure between overloads and
        // their cells, for this purpose it's OK to just index everything
        // by the index in the overall closure in which we find each cell.
        long overallCellIndex = 0;

        if (funcT->getClosureType()->getTypeCategory() != Type::TypeCategory::catTuple) {
            throw std::runtime_error("Expected untyped closure to be a tuple");
        }

        Tuple* closureType = (Tuple*)funcT->getClosureType();

        // functions that came from the interpreter have a particular structure:
        // they should be a Tuple() with one element per overload, and each
        // element of that Tuple should be a NamedTuple containing the closure
        // elements, each of which is a PyCell
        for (long overloadIx = 0; overloadIx < closureType->getTypes().size(); overloadIx++) {
            if (!closureType->getTypes()[overloadIx]->isNamedTuple()) {
                throw std::runtime_error("Expected untyped closure elements to be named tuples");
            }

            NamedTuple* overloadClosure = (NamedTuple*)closureType->getTypes()[overloadIx];
            long overloadOffset = closureType->getOffsets()[overloadIx];

            for (long cellIx = 0; cellIx < overloadClosure->getTypes().size(); cellIx++) {
                long cellOffset = overloadClosure->getOffsets()[cellIx];

                if (overloadClosure->getTypes()[cellIx]->getTypeCategory() != Type::TypeCategory::catPyCell) {
                    throw std::runtime_error("Untyped closure elements should be PyCell objects");
                }

                PyObject* cell = PyObjectHandleTypeBase::getPyObj(closureData + overloadOffset + cellOffset);

                walkCell(cell, p + overallCellIndex);

                overallCellIndex++;
            }
        }
    }

    void buildTypes() {
        auto it = sResolvedTypes.find(std::make_pair(mUnresolvedFunctionTypes, mCellTypes));

        if (it != sResolvedTypes.end()) {
            mClosureType = std::get<0>(it->second);
            mClosureIndices = std::get<1>(it->second);
            mResolvedFunctionTypes = std::get<2>(it->second);
            mClosureIsCell = mClosureType->getTypeCategory() == Type::TypeCategory::catTypedCell;
            return;
        }

        buildClosureType();
        buildFunctionTypes();

        sResolvedTypes[std::make_pair(mUnresolvedFunctionTypes, mCellTypes)] =
            std::make_tuple(mClosureType, mClosureIndices, mResolvedFunctionTypes);
    }

    void buildClosureType() {
        std::vector<Type*> types;

        size_t index = 0;

        for (auto pathAndType: mCellTypes) {
            if (pathAndType.second) {
                types.push_back(pathAndType.second);
                mClosureIndices[pathAndType.first] = index++;
            } else {
                // this variable is not assigned.
                types.push_back(NoneType::Make());
            }
        }

        Tuple* closureTuple = Tuple::Make(types);

        if (closureTuple->isPOD()) {
            mClosureType = closureTuple;
            mClosureIsCell = false;
        } else {
            mClosureType = TypedCellType::Make(closureTuple);
            mClosureIsCell = true;
        }
    }

    void buildFunctionTypes() {
        for (auto pathAndType: mUnresolvedFunctionTypes) {
            //this function needs to be converted.
            mResolvedFunctionTypes[pathAndType.first] = Forward::Make(pathAndType.second->name());
        }

        // now, walk each resolved closure and rebuild its type.
        for (auto& pathAndForwardType: mResolvedFunctionTypes) {
            Forward* f = (Forward*)pathAndForwardType.second;

            pathAndForwardType.second = f->define(
                computeFunctionTypeForPath(pathAndForwardType.first)
            );
        }
    }

    Function* computeFunctionTypeForPath(Path p) {
        // get the existing object
        PyObject* existingFunction = mUnresolvedFunctionObjects[p];
        if (!existingFunction) {
            throw std::runtime_error("Somehow, path didn't lead to a function");
        }

        //convert untyped functions to typed functions.
        PyObjectHolder funcObj = unwrapToFuncObj(existingFunction);

        Type* argT = PyInstance::extractTypeFrom(funcObj->ob_type);

        if (!(argT && argT->getTypeCategory() == Type::TypeCategory::catFunction)) {
            throw std::runtime_error("Expected this to be a function object.");
        }

        Function* funcT = (Function*)argT;

        if (funcT->getClosureType()->getTypeCategory() != Type::TypeCategory::catTuple) {
            throw std::runtime_error("Expected untyped closure to be a Tuple");
        }

        instance_ptr closureData = ((PyInstance*)(PyObject*)funcObj)->dataPtr();
        Tuple* closureType = (Tuple*)funcT->getClosureType();

        std::vector<Function::Overload> newOverloads;

        if (closureType->getTypes().size() != funcT->getOverloads().size()) {
            throw std::runtime_error("Untyped closures should have one element per overload.");
        }

        for (long overloadIx = 0; overloadIx < closureType->getTypes().size(); overloadIx++) {
            if (!closureType->getTypes()[overloadIx]->isNamedTuple()) {
                throw std::runtime_error("Untyped closures should be Tuples of NamedTuples");
            }
            newOverloads.push_back(
                mapOverload(
                    funcT->getOverloads()[overloadIx],
                    (NamedTuple*)closureType->getTypes()[overloadIx],
                    closureData + closureType->getOffsets()[overloadIx]
                )
            );
        }

        return funcT->replaceClosure(mClosureType)->replaceOverloads(newOverloads);
    }

    Function::Overload mapOverload(const Function::Overload& overload, NamedTuple* untypedClosure, instance_ptr untypedClosureData) {
        std::map<std::string, ClosureVariableBinding> bindings;

        for (long cellIx = 0; cellIx < untypedClosure->getTypes().size(); cellIx++) {
            std::string closureVarName = untypedClosure->getNames()[cellIx];

            if (untypedClosure->getTypes()[cellIx]->getTypeCategory() != Type::TypeCategory::catPyCell) {
                throw std::runtime_error("Untyped closures are supposed to hold PyCells");
            }

            PyObject* cell = PyObjectHandleTypeBase::getPyObj(
                untypedClosureData + untypedClosure->getOffsets()[cellIx]
            );

            if (!cell || !PyCell_Check(cell)) {
                throw std::runtime_error("Somehow this is not a cell?");
            }

            PyObject* cellContents = PyCell_GET(cell);

            if (!cellContents) {
                // this cell is empty, and so we just don't bind it at all
            } else {
                if (mObjectPaths.find(cellContents) == mObjectPaths.end()) {
                    throw std::runtime_error("Failed to find a record of this function cell even though we walked it already");
                }

                Path contentsPath = mObjectPaths[cellContents];

                if (mResolvedFunctionTypes.find(contentsPath) != mResolvedFunctionTypes.end()) {
                    bindings[closureVarName] = ClosureVariableBinding() + ClosureVariableBindingStep(
                        (Function*)mResolvedFunctionTypes.find(contentsPath)->second
                    );
                } else {
                    // this is a function
                    ClosureVariableBinding binding;
                    if (mClosureIsCell) {
                        binding = binding + ClosureVariableBindingStep::AccessCell();
                    }

                    auto it = mClosureIndices.find(contentsPath);
                    if (it == mClosureIndices.end()) {
                        throw std::runtime_error("Path " + contentsPath.toString() + " isn't in the closure indices");
                    }

                    bindings[closureVarName] = binding + ClosureVariableBindingStep(it->second);
                }
            }
        }

        return overload.withClosureBindings(bindings);
    }

    PyObject* buildFinalResult() {
        Function* outType = (Function*)mResolvedFunctionTypes[Path()];

        if (!outType) {
            throw std::runtime_error("Somehow, we don't have a function at the root of the closure converter");
        }

        if (outType->getTypeCategory() != Type::TypeCategory::catFunction) {
            throw std::runtime_error("Somehow, our function type didn't resolve.");
        }

        return PyInstance::initialize(outType, [&](instance_ptr outClosure) {
            if (mClosureIsCell) {
                ((TypedCellType*)mClosureType)->initializeHandleAt(outClosure);

                ((TypedCellType*)mClosureType)->set(
                    outClosure,
                    [&](instance_ptr data) {
                        buildClosureValues(
                            (Tuple*)((TypedCellType*)mClosureType)->getHeldType(),
                            data
                        );
                    }
                );
           } else {
                buildClosureValues(
                    (Tuple*)mClosureType,
                    outClosure
                );
            }
        });
    }

    void buildClosureValues(Tuple* closureTuple, instance_ptr data) {
        for (auto& pathAndContents: mPathToCellContents) {
            auto it = mClosureIndices.find(pathAndContents.first);
            if (it == mClosureIndices.end()) {
                throw std::runtime_error("Couldn't find an index for path " + pathAndContents.first.toString());
            }

            PyInstance::copyConstructFromPythonInstance(
                closureTuple->getTypes()[it->second],
                data + closureTuple->getOffsets()[it->second],
                pathAndContents.second,
                ConversionLevel::New
            );
        }
    }


private:
    // for each _cell_ object, cell contents, or function, the first path
    // in our search that reaches it.
    std::map<PyObject*, Path> mObjectPaths;

    // for each cell object (by path), the type of the _contents_ of the cell
    // as we'll be storing it, if it is not a function.
    std::map<Path, Type*> mCellTypes;

    std::map<Path, PyObject*> mPathToCellContents;

    // for each cell object (by path), that is a function, the type of the function
    // excluding the closure.
    std::map<Path, Function*> mUnresolvedFunctionTypes;

    std::map<Path, PyObject*> mUnresolvedFunctionObjects;

    // for each object in our closure, where is it in the closure tuple
    std::map<Path, size_t> mClosureIndices;

    // the type of the actual closure
    Type* mClosureType;

    // is this closure a cell? if so, then we always have to dereference it,
    // otherwise, it's just a Tuple
    bool mClosureIsCell;

    std::map<Path, Type*> mResolvedFunctionTypes;

    static std::map<
        //unresolved functions and closure variables
        std::pair<std::map<Path, Function*>, std::map<Path, Type*> >,
        //the closure type, mClosureIndices, and the resolved function types
        std::tuple<Type*, std::map<Path, size_t>, std::map<Path, Type*> >
    > sResolvedTypes;

    bool mVerbose;
};
