/******************************************************************************
   Copyright 2017-2020 typed_python Authors

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
        for (long k = 0; k < names.size(); k++) {
            m_nameToIndex[names[k]] = k;
        }

        endOfConstructorInitialization(); // finish initializing the type object.
    }

    ShaHash _computeIdentityHash(MutuallyRecursiveTypeGroup* groupHead = nullptr) {
        ShaHash res = ShaHash(1, m_typeCategory);

        for (long k = 0; k < m_types.size(); k++) {
            res = res + ShaHash(m_types[k]->identityHash(groupHead));
        }
        for (long k = 0; k < m_names.size(); k++) {
            res = res + ShaHash(m_names[k]);
        }

        return res;
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

    bool isPODConcrete() {
        for (auto t: m_types) {
            if (!t->isPOD()) {
                return false;
            }
        }

        return true;
    }

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions);

    void deepcopyConcrete(
        instance_ptr dest,
        instance_ptr src,
        DeepcopyContext& context
    ) {
        for (long k = (long)m_types.size() - 1; k >= 0; k--) {
            m_types[k]->deepcopy(
                dest + m_byte_offsets[k],
                src + m_byte_offsets[k],
                context
            );
        }
    }

    size_t deepBytecountConcrete(instance_ptr instance, std::unordered_set<void*>& alreadyVisited, std::set<Slab*>* outSlabs) {
        size_t res = 0;

        for (long k = 0; k < getTypes().size(); k++) {
            res += getTypes()[k]->deepBytecount(eltPtr(instance, k), alreadyVisited, outSlabs);
        }

        return res;
    }

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

    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr);

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

    int64_t indexOfName(const std::string& name) {
        for (int64_t i = 0; i < m_names.size(); i++) {
            if (m_names[i] == name) {
                return i;
            }
        }
        return -1;
    }

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
    const std::map<std::string, int>& getNameToIndex() const {
        return m_nameToIndex;
    }

protected:
    template<class subtype>
    static subtype* MakeSubtype(const std::vector<Type*>& types, const std::vector<std::string>& names, subtype* knownType = nullptr) {
        PyEnsureGilAcquired getTheGil;

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
    std::map<std::string, int> m_nameToIndex;
    std::vector<size_t> m_serialize_typecodes; //codes to use when serializing/deserializing
    std::unordered_map<size_t, size_t> m_serialize_typecodes_to_position; //codes to use when serializing/deserializing
};

PyDoc_STRVAR(NamedTuple_doc,
    "NamedTuple(kw)() -> new default-initialized typed named tuple with names and types from kwargs kw\n"
    "NamedTuple(kw)(t) -> new typed named tuple with names and types from kw, initialized from tuple t\n"
    "\n"
    "Raises TypeError if types don't match.\n"
    );

class NamedTuple : public CompositeType {
public:
    NamedTuple(const std::vector<Type*>& types, const std::vector<std::string>& names) :
            CompositeType(TypeCategory::catNamedTuple, types, names)
    {
        assert(types.size() == names.size());

        m_doc = NamedTuple_doc;

        endOfConstructorInitialization(); // finish initializing the type object.
    }

    bool _updateAfterForwardTypesChanged();

    void _updateTypeMemosAfterForwardResolution() {
        NamedTuple::Make(m_types, m_names, this);
    }

    static NamedTuple* Make(const std::vector<Type*>& types, const std::vector<std::string>& names, NamedTuple* knownType = nullptr) {
        if (names.size() != types.size()) {
            throw std::runtime_error("Names mismatched with types!");
        }
        return MakeSubtype<NamedTuple>(types, names, knownType);
    }
};

PyDoc_STRVAR(Tuple_doc,
    "Tuple(T1, T2, ...)() -> new default-initialized typed tuple with element types T1, T2, ...\n"
    "Tuple(T1, T2, ...)(t) -> new typed tuple with types T1, T2, ..., initialized from tuple t\n"
    "\n"
    "Raises TypeError if types don't match.\n"
    );

class Tuple : public CompositeType {
public:
    Tuple(const std::vector<Type*>& types, const std::vector<std::string>& names) :
            CompositeType(TypeCategory::catTuple, types, names)
    {
        m_doc = Tuple_doc;
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
