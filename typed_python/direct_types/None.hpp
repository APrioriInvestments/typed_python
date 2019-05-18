/******************************************************************************
   Copyright 2017-2019 Nativepython Authors

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

#include "../Type.hpp"
#include "../PyInstance.hpp"

class None {
public:
    static NoneType* getType() {
        static NoneType* t = NoneType::Make();
        return t;
    }
    None() {}
    ~None() {}
    None(None& other) {}
    None& operator=(const None& other) { return *this; }
};

template<>
class TypeDetails<None> {
public:
    static Type* getType() {
        static Type* t = None::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("None: somehow we have the wrong bytecount!");
        }
        return t;
    }

    static const uint64_t bytecount = 0;
};
