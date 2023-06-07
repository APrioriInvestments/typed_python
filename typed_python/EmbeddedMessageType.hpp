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

#include "BytesType.hpp"
#include "NullSerializationContext.hpp"
#include "SerializationBuffer.hpp"

PyDoc_STRVAR(EmbeddedMessage_doc,
    "EmbeddedMessage: represents an embedded message in a serialization graph.\n\n"
    "If you have a type T in a type graph and you replace it with EmbeddedMessage and \n"
    "deserialize an instance of that type, instead of the actual instance you'll get a\n"
    "block of bytes representing the embedded type. This can be used in some circumstances\n"
    "to build tooling that can handle messages without knowing their full type. It cannot\n"
    "be used in contexts where there may be embedded memos within the subgraph, so use with\n"
    "care. This will probably get deprecated at some point.\n"
);


class EmbeddedMessageType : public BytesType {
public:
    EmbeddedMessageType()
    {
        m_name = "EmbeddedMessage";
        m_is_default_constructible = true;
        m_size = sizeof(layout*);
        m_typeCategory = TypeCategory::catEmbeddedMessage;
    }

    const char* docConcrete() {
        return EmbeddedMessage_doc;
    }

    static EmbeddedMessageType* Make() {
        static EmbeddedMessageType* res = new EmbeddedMessageType();
        return res;
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber) {
        //by default we're the empty message
        if (count(self) == 0) {
            buffer.writeEmpty(fieldNumber);
            return;
        }

        //we represent a message with fieldNumber 0. Our first byte is whatever
        //our own wire type is, so we can just add the field number to it
        buffer.writeUnsignedVarint((fieldNumber << 3) + *(uint8_t*)eltPtr(self, 0));

        //then we write the rest of the message
        buffer.write_bytes(eltPtr(self, 1), count(self) - 1);
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t wireType) {
        NullSerializationContext context;
        SerializationBuffer outBuffer(context);

        outBuffer.write<uint8_t>(wireType);
        buffer.copyMessageToOtherBuffer(wireType, outBuffer);

        constructor((instance_ptr)self, outBuffer.size(), (const char*)outBuffer.buffer());
    }

    void postInitializeConcrete() {}
};
