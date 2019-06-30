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

#include <stdint.h>
#include "../typed_python/direct_types/all.hpp"
#include "../typed_python/Format.hpp"
#include "SchemaAndTypeName.hpp"

enum { NO_TRANSACTION = -1 };
enum { NO_OBJECT = -1 };
enum { NO_FIELD = -1 };

typedef int64_t field_id;
typedef Bytes index_value;
typedef int64_t object_id;
typedef int64_t transaction_id;

//lookup key for a single field/index pair.
//optimized to use the bytes hash value.
class IndexKey {
public:
  IndexKey(field_id f, index_value i) :
        m_fieldId(f),
        m_index(i),
        m_hash_val(i.hashValue())
  {
  }

  bool operator<(const IndexKey& other) const {
     if (m_fieldId < other.m_fieldId) {
        return true;
     }
     if (m_fieldId > other.m_fieldId) {
        return false;
     }
     if (m_hash_val < other.m_hash_val) {
        return true;
     }
     if (m_hash_val > other.m_hash_val) {
        return false;
     }
     return m_index < other.m_index;
  }

  bool operator==(const IndexKey& other) const {
    return !(*this < other) && !(other < *this);
  }

  const field_id& fieldId() const {
     return m_fieldId;
  }

  const index_value& indexValue() const {
     return m_index;
  }

  size_t hash() const {
    return m_fieldId ^ m_hash_val;
  }

private:
  field_id m_fieldId;
  index_value m_index;
  typed_python_hash_type m_hash_val;
};
