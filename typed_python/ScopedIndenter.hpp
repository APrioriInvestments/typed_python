/******************************************************************************
   Copyright 2017-2022 typed_python Authors

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

#include <string>

// use to print indented log/debug messages. The static thread-local lets you keep track
// of how many stack frames there are above you

class ScopedIndenter {
public:
    ScopedIndenter() {
        get()++;
    }
    ~ScopedIndenter() {
        get()--;
    }

    int& get() {
        static thread_local int i = 0;
        return i;
    }

    std::string prefix() {
        return std::string(get() * 4, ' ');
    }
};
