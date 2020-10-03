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

class Bytes {
public:
    static BytesType* getType() {
        static BytesType* t = BytesType::Make();
        return t;
    }

    static Bytes fromPython(PyObject* p) {
        BytesType::layout* l = nullptr;
        PyInstance::copyConstructFromPythonInstance(getType(), (instance_ptr)&l, p, ConversionLevel::Implicit);
        return Bytes(l);
    }

    PyObject* toPython() {
        return PyInstance::extractPythonObject((instance_ptr)&mLayout, getType());
    }

    Bytes():mLayout(0) {}

    explicit Bytes(const char *pc, size_t bytecount):mLayout(0) {
        getType()->constructor((instance_ptr)&mLayout, bytecount, pc);
    }

    explicit Bytes(const char *pc):mLayout(0) {
        getType()->constructor((instance_ptr)&mLayout, strlen(pc), pc);
    }

    explicit Bytes(const std::string& s):mLayout(0) {
        getType()->constructor((instance_ptr)&mLayout, s.length(), s.data());
    }

    ~Bytes() {
        getType()->destroy((instance_ptr)&mLayout);
    }

    Bytes(const Bytes& other) {
        getType()->copy_constructor(
           (instance_ptr)&mLayout,
           (instance_ptr)&other.mLayout
           );
    }

    Bytes& operator=(const Bytes& other) {
        getType()->assign(
           (instance_ptr)&mLayout,
           (instance_ptr)&other.mLayout
           );
        return *this;
    }

    template<class buf_t>
    void serialize(buf_t& buffer) {
        getType()->serialize<buf_t>((instance_ptr)&mLayout, buffer);
    }

    template<class buf_t>
    void deserialize(buf_t& buffer) {
        getType()->deserialize<buf_t>((instance_ptr)&mLayout, buffer);
    }

    size_t size() const {
        return getType()->count((instance_ptr)&mLayout);
    }

    typed_python_hash_type hashValue() const {
        return getType()->hash((instance_ptr)&mLayout);
    }

    static inline char cmp(const Bytes& left, const Bytes& right) {
        return BytesType::cmpStatic(left.mLayout, right.mLayout);
    }
    bool operator==(const Bytes& right) const { return cmp(*this, right) == 0; }
    bool operator!=(const Bytes& right) const { return cmp(*this, right) != 0; }
    bool operator<(const Bytes& right) const { return cmp(*this, right) < 0; }
    bool operator>(const Bytes& right) const { return cmp(*this, right) > 0; }
    bool operator<=(const Bytes& right) const { return cmp(*this, right) <= 0; }
    bool operator>=(const Bytes& right) const { return cmp(*this, right) >= 0; }

    const uint8_t& operator[](size_t s) const {
        return *(uint8_t*)getType()->eltPtr((instance_ptr)&mLayout, s);
    }

    static Bytes concatenate(const Bytes& left, const Bytes& right) {
        return Bytes(BytesType::concatenate(left.getLayout(), right.getLayout()));
    }

    Bytes operator+(const Bytes& right) const {
        return Bytes(BytesType::concatenate(mLayout, right.getLayout()));
    }

    Bytes& operator+=(const Bytes& right) {
        return *this = Bytes(BytesType::concatenate(mLayout, right.getLayout()));
    }

    BytesType::layout* getLayout() const { return mLayout; }

    std::string toStdString() const {
        return std::string(&(*this)[0], &(*this)[0] + size());
    }

private:
    explicit Bytes(BytesType::layout* l): mLayout(l) {
        // deliberately stealing a reference
    }

    BytesType::layout* mLayout;
};

template<>
class TypeDetails<Bytes> {
public:
    static Type* getType() {
        static Type* t = BytesType::Make();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("Bytes: somehow we have the wrong bytecount!");
        }
        return t;
    }

    static const uint64_t bytecount = sizeof(void*);
};
