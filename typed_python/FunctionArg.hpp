/******************************************************************************
   Copyright 2017-2022 typed_python Authors

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

class FunctionArg {
public:
    FunctionArg(std::string name, Type* typeFilterOrNull, PyObject* defaultValue, bool isStarArg, bool isKwarg) :
        m_name(name),
        m_typeFilter(typeFilterOrNull),
        m_defaultValue(defaultValue),
        m_isStarArg(isStarArg),
        m_isKwarg(isKwarg)
    {
        assert(!(isStarArg && isKwarg));
    }

    std::string getName() const {
        return m_name;
    }

    PyObject* getDefaultValue() const {
        return m_defaultValue;
    }

    Type* getTypeFilter() const {
        return m_typeFilter;
    }

    bool getIsStarArg() const {
        return m_isStarArg;
    }

    bool getIsKwarg() const {
        return m_isKwarg;
    }

    bool getIsNormalArg() const {
        return !m_isKwarg && !m_isStarArg;
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        if (m_typeFilter) {
            visitor(m_typeFilter);
        }
    }

    bool operator<(const FunctionArg& other) const {
        if (m_name < other.m_name) {
            return true;
        }
        if (m_name > other.m_name) {
            return false;
        }
        if (m_typeFilter < other.m_typeFilter) {
            return true;
        }
        if (m_typeFilter > other.m_typeFilter) {
            return false;
        }
        if (m_defaultValue < other.m_defaultValue) {
            return true;
        }
        if (m_defaultValue > other.m_defaultValue) {
            return false;
        }
        if (m_isStarArg < other.m_isStarArg) {
            return true;
        }
        if (m_isStarArg > other.m_isStarArg) {
            return false;
        }
        if (m_isKwarg < other.m_isKwarg) {
            return true;
        }
        if (m_isKwarg > other.m_isKwarg) {
            return false;
        }

        return false;
    }

    template<class serialization_context_t, class buf_t>
    void serialize(serialization_context_t& context, buf_t& buffer, int fieldNumber) const {
        buffer.writeBeginCompound(fieldNumber);

        buffer.writeStringObject(0, m_name);
        if (m_typeFilter) {
            context.serializeNativeType(m_typeFilter, buffer, 1);
        }
        if (m_defaultValue) {
            context.serializePythonObject(m_defaultValue, buffer, 2);
        }
        buffer.writeUnsignedVarintObject(3, m_isStarArg ? 1 : 0);
        buffer.writeUnsignedVarintObject(4, m_isKwarg ? 1 : 0);

        buffer.writeEndCompound();
    }

    template<class serialization_context_t, class buf_t>
    static FunctionArg deserialize(serialization_context_t& context, buf_t& buffer, int wireType) {
        std::string name;
        Type* typeFilterOrNull = nullptr;
        PyObjectHolder defaultValue;
        bool isStarArg = false;
        bool isKwarg = false;

        buffer.consumeCompoundMessage(wireType, [&](size_t fieldNumber, size_t wireType) {
            if (fieldNumber == 0) {
                assertWireTypesEqual(wireType, WireType::BYTES);
                name = buffer.readStringObject();
            }
            else if (fieldNumber == 1) {
                typeFilterOrNull = context.deserializeNativeType(buffer, wireType);
            }
            else if (fieldNumber == 2) {
                defaultValue.steal(context.deserializePythonObject(buffer, wireType));
            }
            else if (fieldNumber == 3) {
                assertWireTypesEqual(wireType, WireType::VARINT);
                isStarArg = buffer.readUnsignedVarint();
            }
            else if (fieldNumber == 4) {
                assertWireTypesEqual(wireType, WireType::VARINT);
                isKwarg = buffer.readUnsignedVarint();
            }
        });

        return FunctionArg(name, typeFilterOrNull, defaultValue, isStarArg, isKwarg);
    }

    template<class visitor_type>
    void _visitCompilerVisibleInternals(const visitor_type& v) {
        v.visitName(m_name);
        if (m_defaultValue) {
            v.visitTopo((PyObject*)m_defaultValue);
        } else {
            v.visitHash(ShaHash());
        }

        if (m_typeFilter) {
            v.visitTopo(m_typeFilter);
        } else {
            v.visitHash(ShaHash());
        }

        v.visitHash(
            ShaHash((m_isStarArg ? 2 : 1) + (m_isKwarg ? 10: 11))
        );
    }

private:
    std::string m_name;
    Type* m_typeFilter;
    PyObjectHolder m_defaultValue;
    bool m_isStarArg;
    bool m_isKwarg;
};
