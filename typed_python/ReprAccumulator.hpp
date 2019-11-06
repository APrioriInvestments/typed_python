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

#include <sstream>
#include <set>

class ReprAccumulator {
public:
    ReprAccumulator(std::ostringstream& stream) :
        m_stream(stream)
    {
    }

    //record that we have seen this object, and then return whether it was redundant.
    bool pushObject(void* ptr) {
        bool hasSeen = m_seen_objects[ptr] > 0;
        m_seen_objects[ptr]++;
        return !hasSeen;
    }

    void popObject(void* ptr) {
        m_seen_objects[ptr]--;
    }

    template<class T>
    ReprAccumulator& operator<<(const T& in) {
        m_stream << in;
        return *this;
    }

    //if true, then this was invoked by a 'str' call, not a 'repr'.
    bool isStrCall() const {
        return m_isStr;
    }

private:
    std::ostringstream& m_stream;

    std::map<void*, int64_t> m_seen_objects;

    bool m_isStr;
};

class PushReprState {
public:
    template<class T>
    PushReprState(ReprAccumulator& accumulator, T* ptr) :
        m_accumulator(accumulator),
        m_ptr((void*)ptr),
        m_was_new(false)
    {
        m_was_new = m_accumulator.pushObject(m_ptr);
    }

    ~PushReprState() {
        m_accumulator.popObject(m_ptr);
    }

    operator bool() const {
        return m_was_new;
    }

private:
    ReprAccumulator& m_accumulator;
    void* m_ptr;
    bool m_was_new;
};
