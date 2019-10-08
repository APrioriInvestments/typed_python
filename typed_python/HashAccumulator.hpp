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


typedef int32_t typed_python_hash_type;


class HashAccumulator {
public:
    HashAccumulator(typed_python_hash_type init = 0) :
        m_state(init)
    {
    }

    void add(int32_t i) {
        m_state = int32_t(m_state * 1000003) ^ i;
    }

    void addBytes(uint8_t* bytes, int64_t count) {
        while (count >= 4) {
            add(*(int32_t*)bytes);
            bytes += 4;
            count -= 4;
        }
        while (count) {
            add((uint8_t)*bytes);
            bytes++;
            count--;
        }
    }

    typed_python_hash_type get() const {
        return m_state;
    }

    void addRegister(bool i) { add(i ? 1:0); }
    void addRegister(uint8_t i) { add(i); }
    void addRegister(uint16_t i) { add(i); }
    void addRegister(uint32_t i) { add(i); }
    void addRegister(uint64_t i) {
        add(i >> 32);
        add(i & 0xFFFFFFFF);
    }

    void addRegister(int8_t i) { add(i); }
    void addRegister(int16_t i) { add(i); }
    void addRegister(int32_t i) { add(i); }
    void addRegister(int64_t i) {
        add(i >> 32);
        add(i & 0xFFFFFFFF);
    }

    void addRegister(float i) {
      addRegister((double)i);
    }

    void addRegister(double i) {
      if (i == int32_t(i)) {
        add((int32_t)i);
      } else {
        addBytes((uint8_t*)&i, sizeof(i));
      }
    }

private:
    typed_python_hash_type m_state;
};
