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

#include <map>

class Type;


// model a call stack of types above us in a search. Clients can ask if a type
// is above us in the stack and if so, how many levels up, using 'indexOf'
// and push new types to the stack using PushTypeStack. The stack can't have
// duplicates.

class TypeStack {
public:
    long indexOf(Type* t) {
        auto it = mTypeIndices.find(t);

        if (it == mTypeIndices.end()) {
            return -1;
        }

        return mTypeIndices.size() - it->second - 1;
    }

    void push(Type* t) {
        if (mTypeIndices.find(t) != mTypeIndices.end()) {
            throw std::runtime_error("Type is already in the type stack");
        }
        size_t s = mTypeIndices.size();

        mTypeIndices[t] = s;
    }

    void pop(Type* t) {
        mTypeIndices.erase(t);
    }

private:
    std::map<Type*, long> mTypeIndices;
};


class PushTypeStack {
public:
    PushTypeStack(TypeStack& stack, Type* t) : mStack(stack), mT(t) {
        stack.push(mT);
    }

    ~PushTypeStack() {
        mStack.pop(mT);
    }

private:
    TypeStack& mStack;
    Type* mT;
};
