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

#include "Type.hpp"
#include "SerializationContext.hpp"

class NullSerializationContext : public SerializationContext {
public:
    virtual void serializePythonObject(PyObject* o, SerializationBuffer& b, size_t fieldNumber) const {
        throw std::runtime_error("No serialization plugin provided, so we can't serialize arbitrary python objects.");
    }
    virtual PyObject* deserializePythonObject(DeserializationBuffer& b, size_t wireType) const {
        throw std::runtime_error("No serialization plugin provided, so we can't deserialize arbitrary python objects.");
    }

    virtual void serializeNativeType(Type* o, SerializationBuffer& b, size_t fieldNumber) const {
        throw std::runtime_error("No serialization plugin provided, so we can't serialize arbitrary python objects.");
    }

    virtual Type* deserializeNativeType(DeserializationBuffer& b, size_t wireType) const {
        throw std::runtime_error("No serialization plugin provided, so we can't serialize arbitrary python objects.");
    }


    virtual bool isCompressionEnabled() const {
        return false;
    }
};
