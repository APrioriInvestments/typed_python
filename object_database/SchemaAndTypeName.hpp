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

/**********
SchemaAndTypeName

An interned pointer to a schema/typename pair. We use this to represent a database
type independently of a specific codebase (and therefore Type*).
**********/

#include <string>

class SchemaAndTypeName {
public:
  SchemaAndTypeName() : m_interned_name_pair(0)
  {
  }

  SchemaAndTypeName(const std::string& schemaName, const std::string& typeName) :
      m_interned_name_pair(intern(schemaName, typeName))
  {
  }

  bool operator==(const SchemaAndTypeName& other) const {
    return m_interned_name_pair == other.m_interned_name_pair;
  }

  bool operator<(const SchemaAndTypeName& other) const {
    return m_interned_name_pair < other.m_interned_name_pair;
  }

  bool operator!=(const SchemaAndTypeName& other) const {
    return m_interned_name_pair != other.m_interned_name_pair;
  }

  const std::string& schemaName() const {
    return m_interned_name_pair->first;
  }

  const std::string& typeName() const {
    return m_interned_name_pair->second;
  }

  std::size_t hash() const {
    return std::hash<size_t>()((size_t)m_interned_name_pair);
  }

  operator std::string() const {
    if (!m_interned_name_pair) {
      return "<null>";
    }

    return m_interned_name_pair->first + "." + m_interned_name_pair->second;
  }

private:
  const std::pair<std::string, std::string>* intern(std::string schema, std::string type) {
    static std::set<std::pair<std::string, std::string> > internedValues;

    static std::mutex guard;
    std::lock_guard<std::mutex> lock(guard);

    auto it = internedValues.find(std::make_pair(schema, type));
    if (it == internedValues.end()) {
      internedValues.insert(std::make_pair(schema, type));
      it = internedValues.find(std::make_pair(schema, type));
    }

    return &*it;
  }

  const std::pair<std::string, std::string>* m_interned_name_pair;
};

template<class stream>
inline stream& operator<<(stream& s, const SchemaAndTypeName& type) {
  s << std::string(type);
  return s;
}

inline std::string operator+(std::string s, const SchemaAndTypeName& type) {
  return s + std::string(type);
}

namespace std {
  template<>
  struct hash<SchemaAndTypeName> {
    typedef SchemaAndTypeName argument_type;
    typedef std::size_t result_type;

    result_type operator()(argument_type const& s) const noexcept {
      return s.hash();
    }
  };
}
