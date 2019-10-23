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
#include "ReprAccumulator.hpp"
#include <unordered_map>

class CompositeType : public Type {
public:
    CompositeType(
                TypeCategory in_typeCategory,
                const std::vector<Type*>& types,
                const std::vector<std::string>& names
                ) :
            Type(in_typeCategory),
            m_types(types),
            m_names(names)
    {
        endOfConstructorInitialization(); // finish initializing the type object.
    }

    bool isBinaryCompatibleWithConcrete(Type* other);

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
        for (auto& typePtr: m_types) {
            visitor(typePtr);
        }
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        _visitContainedTypes(visitor);
    }

    bool _updateAfterForwardTypesChanged();

    instance_ptr eltPtr(instance_ptr self, int64_t ix) const {
        return self + m_byte_offsets[ix];
    }

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions);

    /*******

        deserialize a named tuple using field codes.

        we decode an id for each field type. fields we don't recognize
        are discarded. fields that are not mentioned are default-initialized.

    ********/
    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t wireType) {
        //some versions of GCC appear to crash if you use variable-length arrays for this.
        uint8_t* initialized = (uint8_t*)alloca(m_types.size());

        for (long k = 0; k < m_types.size();k++) {
            initialized[k] = false;
        }

        buffer.consumeCompoundMessage(wireType, [&](size_t fieldNumber, size_t subWireType) {
            auto it = m_serialize_typecodes_to_position.find(fieldNumber);
            if (it != m_serialize_typecodes_to_position.end()) {
                initialized[it->second] = true;
                getTypes()[it->second]->deserialize(eltPtr(self, it->second), buffer, subWireType);
            } else {
                buffer.finishReadingMessageAndDiscard(subWireType);
            }
        });

        for (long k = 0; k < m_types.size();k++) {
            if (!initialized[k]) {
                getTypes()[k]->constructor(eltPtr(self, k));
            }
        }
    }

    /*******

        serialize a named tuple using field codes.

        we write a field count, and then an id before each field, followed
        by the field data.

    ********/
    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber) {
        if (getTypes().size() == 0) {
            buffer.writeEmpty(fieldNumber);
            return;
        }
        if (getTypes().size() == 1) {
            buffer.writeBeginSingle(fieldNumber);
        } else {
            buffer.writeBeginCompound(fieldNumber);
        }

        for (long k = 0; k < getTypes().size();k++) {
            getTypes()[k]->serialize(eltPtr(self,k), buffer, m_serialize_typecodes[k]);
        }

        if (getTypes().size() > 1) {
            buffer.writeEndCompound();
        }
    }

    void repr(instance_ptr self, ReprAccumulator& stream);

    typed_python_hash_type hash(instance_ptr left);

    template<class sub_constructor>
    void constructor(instance_ptr self, const sub_constructor& initializer) {
        for (int64_t k = 0; k < getTypes().size(); k++) {
            try {
                initializer(eltPtr(self, k), k);
            } catch(...) {
                for (long k2 = k-1; k2 >= 0; k2--) {
                    m_types[k2]->destroy(eltPtr(self,k2));
                }
                throw;
            }
        }
    }

    void constructor(instance_ptr self);

    void destroy(instance_ptr self);

    void copy_constructor(instance_ptr self, instance_ptr other);

    void assign(instance_ptr self, instance_ptr other);

    int64_t indexOfName(const char* name) {
        for (int64_t i = 0; i < m_names.size(); i++) {
            if (m_names[i] == name) {
                return i;
            }
        }
        return -1;
    }

    const std::vector<Type*>& getTypes() const {
        return m_types;
    }
    const std::vector<size_t>& getOffsets() const {
        return m_byte_offsets;
    }
    const std::vector<std::string>& getNames() const {
        return m_names;
    }

protected:
    template<class subtype>
    static subtype* MakeSubtype(const std::vector<Type*>& types, const std::vector<std::string>& names, subtype* knownType = nullptr) {
        static std::mutex guard;

        std::lock_guard<std::mutex> lock(guard);

        typedef std::pair<const std::vector<Type*>, const std::vector<std::string> > keytype;

        static std::map<keytype, subtype*> m;

        auto it = m.find(keytype(types, names));
        if (it == m.end()) {
            it = m.insert(
                std::make_pair(
                    keytype(types, names),
                    knownType ? knownType : new subtype(types, names)
                )
            ).first;
        }

        return it->second;
    }

    std::vector<Type*> m_types;
    std::vector<size_t> m_byte_offsets;
    std::vector<std::string> m_names;
    std::vector<size_t> m_serialize_typecodes; //codes to use when serializing/deserializing
    std::unordered_map<size_t, size_t> m_serialize_typecodes_to_position; //codes to use when serializing/deserializing
};

class NamedTuple : public CompositeType {
public:
    NamedTuple(const std::vector<Type*>& types, const std::vector<std::string>& names) :
            CompositeType(TypeCategory::catNamedTuple, types, names)
    {
        assert(types.size() == names.size());

        endOfConstructorInitialization(); // finish initializing the type object.
    }

    bool _updateAfterForwardTypesChanged();

    void _updateTypeMemosAfterForwardResolution() {
        NamedTuple::Make(m_types, m_names, this);
    }

    static NamedTuple* Make(const std::vector<Type*>& types, const std::vector<std::string>& names, NamedTuple* knownType = nullptr) {
        return MakeSubtype<NamedTuple>(types, names, knownType);
    }
};

class Tuple : public CompositeType {
public:
    Tuple(const std::vector<Type*>& types, const std::vector<std::string>& names) :
            CompositeType(TypeCategory::catTuple, types, names)
    {
        endOfConstructorInitialization(); // finish initializing the type object.
    }

    bool _updateAfterForwardTypesChanged();

    void _updateTypeMemosAfterForwardResolution() {
        Tuple::Make(m_types, this);
    }

    static Tuple* Make(const std::vector<Type*>& types, Tuple* knownType=nullptr) {
        return MakeSubtype<Tuple>(types, std::vector<std::string>(), knownType);
    }
};
