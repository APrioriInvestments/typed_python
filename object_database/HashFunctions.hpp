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

#include <string>

/*******
Overrides of std::hash so we can use unordered containers.
*******/

class DatabaseConnectionState;

namespace std {
    template<>
    struct hash<std::pair<std::string, DatabaseConnectionState*> > {
        typedef std::pair<std::string, DatabaseConnectionState*> argument_type;
        typedef std::size_t result_type;

        result_type operator()(argument_type const& s) const noexcept {
            return std::hash<std::string>()(s.first) ^ (size_t)s.second;
        }
    };

    template<>
    struct hash<IndexKey> {
        typedef IndexKey argument_type;
        typedef std::size_t result_type;

        result_type operator()(argument_type const& s) const noexcept {
            return s.hash();
        }
    };

    template<>
    struct hash<std::pair<int64_t, int64_t> > {
        typedef std::pair<int64_t, int64_t> argument_type;
        typedef std::size_t result_type;

        result_type operator()(argument_type const& s) const noexcept {
            return s.first ^ s.second;
        }
    };

    template<>
    struct hash<std::pair<std::string, std::string> > {
        typedef std::pair<std::string, std::string> argument_type;
        typedef std::size_t result_type;

        result_type operator()(argument_type const& s) const noexcept {
            return std::hash<std::string>()(s.first) ^ std::hash<std::string>()(s.second);
        }
    };
}
