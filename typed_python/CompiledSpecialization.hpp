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


#include "Type.hpp"
#include <vector>

class CompiledSpecialization {
public:
    CompiledSpecialization(
                compiled_code_entrypoint funcPtr,
                Type* returnType,
                const std::vector<Type*>& argTypes
                ) :
        mFuncPtr(funcPtr),
        mReturnType(returnType),
        mArgTypes(argTypes)
    {}

    compiled_code_entrypoint getFuncPtr() const {
        return mFuncPtr;
    }

    Type* getReturnType() const {
        return mReturnType;
    }

    const std::vector<Type*>& getArgTypes() const {
        return mArgTypes;
    }

    bool operator==(const CompiledSpecialization& other) const {
        return mFuncPtr == other.mFuncPtr
            && mReturnType == other.mReturnType
            && mArgTypes == other.mArgTypes
            ;
    }

private:
    compiled_code_entrypoint mFuncPtr;
    Type* mReturnType;
    std::vector<Type*> mArgTypes;
};
