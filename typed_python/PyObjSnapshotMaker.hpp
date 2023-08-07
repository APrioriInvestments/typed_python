/******************************************************************************
   Copyright 2017-2023 typed_python Authors

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

#include <vector>
#include "Type.hpp"
#include "Instance.hpp"
#include "PythonTypeInternals.hpp"

class PyObjGraphSnapshot;
class PyObjSnapshot;
class FunctionGlobal;
class FunctionOverload;


class PyObjSnapshotMaker {
public:
    PyObjSnapshotMaker(
        std::unordered_map<PyObject*, PyObjSnapshot*>& inObjMapCache,
        std::unordered_map<Type*, PyObjSnapshot*>& inTypeMapCache,
        std::unordered_map<InstanceRef, PyObjSnapshot*>& inInstanceCache,
        PyObjGraphSnapshot* inGraph,
        bool inLinkBackToOriginalObject,
        bool linkToInternal
    ) :
        mObjMapCache(inObjMapCache),
        mTypeMapCache(inTypeMapCache),
        mInstanceCache(inInstanceCache),
        mGraph(inGraph),
        mLinkBackToOriginalObject(inLinkBackToOriginalObject),
        mLinkToInternal(linkToInternal)
    {
    }

    PyObjSnapshot* internalize(const std::string& def);
    PyObjSnapshot* internalize(const MemberDefinition& def);
    PyObjSnapshot* internalize(const FunctionGlobal& def);
    PyObjSnapshot* internalize(const FunctionOverload& def);
    PyObjSnapshot* internalize(const FunctionArg& def);
    PyObjSnapshot* internalize(const ClosureVariableBinding& def);
    PyObjSnapshot* internalize(const ClosureVariableBindingStep& def);
    PyObjSnapshot* internalize(const std::vector<FunctionOverload>& def);
    PyObjSnapshot* internalize(const std::vector<FunctionArg>& inArgs);
    PyObjSnapshot* internalize(const std::vector<HeldClass*>& inArgs);
    PyObjSnapshot* internalize(const std::vector<Type*>& inArgs);
    PyObjSnapshot* internalize(const std::vector<std::string>& inArgs);
    PyObjSnapshot* internalize(const std::map<std::string, ClosureVariableBinding>& inBindings);
    PyObjSnapshot* internalize(const std::map<std::string, FunctionGlobal>& inGlobals);
    PyObjSnapshot* internalize(const std::map<std::string, Function*>& inMethods);
    PyObjSnapshot* internalize(const std::map<std::string, PyObject*>& inMethods);
    PyObjSnapshot* internalize(const std::vector<MemberDefinition>& inMethods);
    PyObjSnapshot* internalize(PyObject* val);
    PyObjSnapshot* internalize(Type* val);
    PyObjSnapshot* internalize(const Instance& val) {
        return internalize(val.ref());
    }
    PyObjSnapshot* internalize(InstanceRef val);

    bool linkBackToOriginalObject() const {
        return mLinkBackToOriginalObject;
    }

    PyObjGraphSnapshot* graph() const {
        return mGraph;
    }

private:
    std::unordered_map<PyObject*, PyObjSnapshot*>& mObjMapCache;
    std::unordered_map<Type*, PyObjSnapshot*>& mTypeMapCache;
    std::unordered_map<InstanceRef, PyObjSnapshot*>& mInstanceCache;
    PyObjGraphSnapshot* mGraph;
    bool mLinkBackToOriginalObject;
    bool mLinkToInternal;
};

