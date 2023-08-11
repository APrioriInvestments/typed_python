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
template<class T>
class PtrStack {
public:
    long indexOf(T* t) {
        auto it = mTypeIndices.find(t);

        if (it == mTypeIndices.end()) {
            return -1;
        }

        return mTypeIndices.size() - it->second - 1;
    }

    void push(T* t) {
        if (mTypeIndices.find(t) != mTypeIndices.end()) {
            throw std::runtime_error("Type is already in the type stack");
        }
        size_t s = mTypeIndices.size();

        mTypeIndices[t] = s;
    }

    void pop(T* t) {
        mTypeIndices.erase(t);
    }

private:
    std::map<T*, long> mTypeIndices;
};

typedef PtrStack<Type> TypeStack;


template<class T>
class PushPtrStack {
public:
    PushPtrStack(PtrStack<T>& stack, T* t) : mStack(stack), mT(t) {
        stack.push(mT);
    }

    ~PushPtrStack() {
        mStack.pop(mT);
    }

private:
    PtrStack<T>& mStack;
    T* mT;
};


typedef PushPtrStack<Type> PushTypeStack;
