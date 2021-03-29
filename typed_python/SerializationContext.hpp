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

#include <memory>
#include "WireType.hpp"

class SerializationBuffer;
class DeserializationBuffer;
class Type;


class SerializationContext {
public:
    virtual ~SerializationContext() {};

    virtual void serializePythonObject(PyObject* o, SerializationBuffer& b, size_t fieldNumber) const = 0;
    virtual PyObject* deserializePythonObject(DeserializationBuffer& b, size_t wireType) const = 0;

    virtual void serializeNativeType(Type* o, SerializationBuffer& b, size_t fieldNumber) const = 0;
    virtual Type* deserializeNativeType(DeserializationBuffer& b, size_t wireType) const = 0;

    virtual bool isCompressionEnabled() const = 0;
};
