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

#include <string>
#include <sstream>

/**********
Format: a simple utility for converting ostreamable types to strings.

Usage:

    std::string message = "This is an int as a string " + format(10);

Anything that will stream into std::ostringstream will work.
**********/

template<class T>
class Format {
public:
    Format(const T& in) : m(in) {

    }

    std::string toString() const {
        std::ostringstream s;
        s << m;
        return s.str();
    }

    friend std::string operator+(const char* s, const Format& f) {
        return std::string(s) + f.toString();
    }

    friend std::string operator+(const std::string& s, const Format& f) {
        return s + f.toString();
    }

    friend std::string operator+(const Format& f, const std::string& s) {
        return f.toString() + s;
    }

    template<class stream>
    friend stream& operator<<(stream& s, const Format& f) {
        s << f.toString();
        return s;
    }

private:
    T m;
};

template<class T>
class Format<std::vector<T> > {
public:
    Format(const std::vector<T>& in) : m(in) {

    }

    std::string toString() const {
        std::ostringstream s;

        s << "[";
        for (long i = 0; i < m.size(); i++) {
            if (i) {
                s << ", ";
            }

            s << m[i];
        }
        s << "]";

        return s.str();
    }

    friend std::string operator+(const char* s, const Format& f) {
        return std::string(s) + f.toString();
    }

    friend std::string operator+(const std::string& s, const Format& f) {
        return s + f.toString();
    }

    friend std::string operator+(const Format& f, const std::string& s) {
        return f.toString() + s;
    }

    template<class stream>
    friend stream& operator<<(stream& s, const Format& f) {
        s << f.toString();
        return s;
    }

private:
    std::vector<T> m;
};

template<class T>
Format<T> format(const T& in) {
    return Format<T>(in);
}
