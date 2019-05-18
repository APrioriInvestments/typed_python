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

#include "ListOf.hpp"
#include "../Type.hpp"
#include "../PyInstance.hpp"

class String {
public:
    static StringType* getType() {
        static StringType* t = StringType::Make();
        return t;
    }

    String():mLayout(0) {}

    explicit String(const char *pc):mLayout(0) {
        getType()->constructor((instance_ptr)&mLayout, 1, strlen(pc), pc);
    }

    explicit String(const std::string& s):mLayout(0) {
        getType()->constructor((instance_ptr)&mLayout, 1, s.length(), s.data());
    }

    explicit String(StringType::layout* l):mLayout(0) {
        getType()->constructor((instance_ptr)&mLayout, l->bytes_per_codepoint, l->pointcount, (const char*)l->data);
    }

    ~String() {
        StringType::destroyStatic((instance_ptr)&mLayout);
    }

    String(const String& other)
    {
        getType()->copy_constructor(
           (instance_ptr)&mLayout,
           (instance_ptr)&other.mLayout
           );
    }

    String& operator=(const String& other)
    {
        getType()->assign(
           (instance_ptr)&mLayout,
           (instance_ptr)&other.mLayout
           );
        return *this;
    }

    String& operator=(const char* other) {
        return *this = String(other);
    }

    String& operator=(const std::string& other) {
        return *this = String(other);
    }

    template<class buf_t>
    void serialize(buf_t& buffer) {
        getType()->serialize<buf_t>((instance_ptr)&mLayout, buffer);
    }

    template<class buf_t>
    void deserialize(buf_t& buffer) {
        getType()->deserialize<buf_t>((instance_ptr)&mLayout, buffer);
    }

    static inline char cmp(const String& left, const String& right) {
        return StringType::cmpStatic(left.mLayout, right.mLayout);
    }
    bool operator==(const String& right) const { return cmp(*this, right) == 0; }
    bool operator!=(const String& right) const { return cmp(*this, right) != 0; }
    bool operator<(const String& right) const { return cmp(*this, right) < 0; }
    bool operator>(const String& right) const { return cmp(*this, right) > 0; }
    bool operator<=(const String& right) const { return cmp(*this, right) <= 0; }
    bool operator>=(const String& right) const { return cmp(*this, right) >= 0; }

    size_t size() const {
        return getType()->count((instance_ptr)&mLayout);
    }

    static String concatenate(const String& left, const String& right) {
        return String(StringType::concatenate(left.getLayout(), right.getLayout()));
    }
    String operator+(const String& right) {
        return String(StringType::concatenate(mLayout, right.getLayout()));
    }
    String& operator+=(const String& right) {
        return *this = String(StringType::concatenate(mLayout, right.getLayout()));
    }
    String upper() const { return String(StringType::upper(mLayout)); }
    String lower() const { return String(StringType::lower(mLayout)); }
    int64_t find(const String& sub, int64_t start, int64_t end) const { return StringType::find(mLayout, sub.getLayout(), start, end); }
    int64_t find(const String& sub, int64_t start) const { return StringType::find(mLayout, sub.getLayout(), start, mLayout ? mLayout->pointcount : 0); }
    int64_t find(const String& sub) const { return StringType::find(mLayout, sub.getLayout(), 0, mLayout ? mLayout->pointcount : 0); }
    ListOf<String> split(const String& sep, int64_t max = -1) const {
       ListOf<String> ret;
       StringType::split(ret.getLayout(), mLayout, sep.getLayout(), max);
       return ret;
    }
    ListOf<String> split(int64_t max = -1) {
       ListOf<String> ret;
       StringType::split_3(ret.getLayout(), mLayout, max);
       return ret;
    }
    bool isalpha() const { return StringType::isalpha(mLayout); }
    bool isalnum() const { return StringType::isalnum(mLayout); }
    bool isdecimal() const { return StringType::isdecimal(mLayout); }
    bool isdigit() const { return StringType::isdigit(mLayout); }
    bool islower() const { return StringType::islower(mLayout); }
    bool isnumeric() const { return StringType::isnumeric(mLayout); }
    bool isprintable() const { return StringType::isprintable(mLayout); }
    bool isspace() const { return StringType::isspace(mLayout); }
    bool istitle() const { return StringType::istitle(mLayout); }
    bool isupper() const { return StringType::isupper(mLayout); }

    String substr(int64_t start, int64_t stop) const { return String(StringType::getsubstr(mLayout, start, stop)); }

    StringType::layout* getLayout() const {
        return mLayout;
    }

private:
    StringType::layout* mLayout;
};

template<>
class TypeDetails<String> {
public:
    static Type* getType() {
        static Type* t = StringType::Make();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("String: somehow we have the wrong bytecount!");
        }
        return t;
    }

    static const uint64_t bytecount = sizeof(void*);
};
