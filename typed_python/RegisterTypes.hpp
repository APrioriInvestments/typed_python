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
#include <iomanip>

#include "Type.hpp"
#include "ConversionLevel.hpp"

// class to contain methods that describe properties of register types.
class RegisterTypeProperties {
public:
    // assign a value 'v' to the memory at 'tgt' assuming it points to an element of type 'category'
    template<class other_type>
    static void assign(instance_ptr tgt, Type::TypeCategory category, const other_type& v) {
        if (category == Type::TypeCategory::catUInt64) {
            ((uint64_t*)tgt)[0] = v;
        } else
        if (category == Type::TypeCategory::catUInt32) {
            ((uint32_t*)tgt)[0] = v;
        } else
        if (category == Type::TypeCategory::catUInt16) {
            ((uint16_t*)tgt)[0] = v;
        } else
        if (category == Type::TypeCategory::catUInt8) {
            ((uint8_t*)tgt)[0] = v;
        } else
        if (category == Type::TypeCategory::catInt64) {
            ((int64_t*)tgt)[0] = v;
        } else
        if (category == Type::TypeCategory::catInt32) {
            ((int32_t*)tgt)[0] = v;
        } else
        if (category == Type::TypeCategory::catInt16) {
            ((int16_t*)tgt)[0] = v;
        } else
        if (category == Type::TypeCategory::catInt8) {
            ((int8_t*)tgt)[0] = v;
        } else
        if (category == Type::TypeCategory::catBool) {
            ((bool*)tgt)[0] = v;
        } else
        if (category == Type::TypeCategory::catFloat64) {
            ((double*)tgt)[0] = v;
        } else
        if (category == Type::TypeCategory::catFloat32) {
            ((float*)tgt)[0] = v;
        } else {
            throw std::runtime_error("invalid register category");
        }
    }

    static bool isValidConversion(Type* fromT, Type* toT, ConversionLevel level) {
        return isValidConversion(fromT->getTypeCategory(), toT->getTypeCategory(), level);
    }

    static bool isValidConversion(Type::TypeCategory fromCat, Type::TypeCategory toCat, ConversionLevel level) {
        if (level >= ConversionLevel::Implicit) {
            return true;
        }

        if (fromCat == toCat) {
            return true;
        }

        if (level >= ConversionLevel::Upcast && isValidUpcast(fromCat, toCat)) {
            return true;
        }

        return false;
    }


    // is this a 'lossless' cast, meaning that the number of bits isn't
    // decreasing, nor are we losing signage, nor are we going from float->int?
    static bool isValidUpcast(Type* fromType, Type* toType) {
        return isValidUpcast(fromType->getTypeCategory(), toType->getTypeCategory());
    }

    static bool isValidUpcast(Type::TypeCategory c1, Type::TypeCategory c2) {
        return isValidUpcast_(c1, c2);
    }

    static bool isValidUpcast_(Type::TypeCategory fromType, Type::TypeCategory toType) {
        if (fromType == Type::TypeCategory::catBool) {
            return true;
        }

        if (isFloat(fromType) && !isFloat(toType)) {
            return false;
        }

        if (bits(fromType) > bits(toType)) {
            return false;
        }

        if (isFloat(toType)) {
            return true;
        }

        // casting to strictly more bits and also signedness will always succeed
        if (!isUnsigned(toType) && bits(toType) > bits(fromType)) {
            return true;
        }

        if (isUnsigned(toType) != isUnsigned(fromType)) {
            return false;
        }

        return true;
    }

    static bool isUnsigned(Type* t) {
        return isUnsigned(t->getTypeCategory());
    }

    static bool isUnsigned(Type::TypeCategory cat) {
        return (cat == Type::TypeCategory::catUInt64 ||
                cat == Type::TypeCategory::catUInt32 ||
                cat == Type::TypeCategory::catUInt16 ||
                cat == Type::TypeCategory::catUInt8 ||
                cat == Type::TypeCategory::catBool
                );
    }

    static int bits(Type* t) {
        return bits(t->getTypeCategory());
    }

    static int bits(Type::TypeCategory cat) {
        return
            cat == Type::TypeCategory::catBool ? 1 :
            cat == Type::TypeCategory::catInt8 ? 8 :
            cat == Type::TypeCategory::catInt16 ? 16 :
            cat == Type::TypeCategory::catInt32 ? 32 :
            cat == Type::TypeCategory::catInt64 ? 64 :
            cat == Type::TypeCategory::catUInt8 ? 8 :
            cat == Type::TypeCategory::catUInt16 ? 16 :
            cat == Type::TypeCategory::catUInt32 ? 32 :
            cat == Type::TypeCategory::catUInt64 ? 64 :
            cat == Type::TypeCategory::catFloat32 ? 32 :
            cat == Type::TypeCategory::catFloat64 ? 64 : -1
            ;
    }

    static bool isInteger(Type* t) {
        return isInteger(t->getTypeCategory());
    }

    static bool isInteger(Type::TypeCategory cat) {
        return (cat == Type::TypeCategory::catInt64 ||
                cat == Type::TypeCategory::catInt32 ||
                cat == Type::TypeCategory::catInt16 ||
                cat == Type::TypeCategory::catInt8 ||
                cat == Type::TypeCategory::catUInt64 ||
                cat == Type::TypeCategory::catUInt32 ||
                cat == Type::TypeCategory::catUInt16 ||
                cat == Type::TypeCategory::catUInt8
                );
    }

    static bool isFloat(Type* t) {
        return isFloat(t->getTypeCategory());
    }

    static bool isFloat(Type::TypeCategory cat) {
        return (cat == Type::TypeCategory::catFloat64 ||
                cat == Type::TypeCategory::catFloat32
                );
    }
};


template<class T>
class RegisterType : public Type {
public:
    RegisterType(TypeCategory kind) : Type(kind)
    {
        m_size = sizeof(T);
        m_is_default_constructible = true;

        endOfConstructorInitialization(); // finish initializing the type object.
    }

    bool isBinaryCompatibleWithConcrete(Type* other) {
        if (other->getTypeCategory() != m_typeCategory) {
            return false;
        }

        return true;
    }

    bool isPODConcrete() {
        return true;
    }

    bool _updateAfterForwardTypesChanged() { return false; }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& v) {}

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& v) {}

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions) {
        if ( (*(T*)left) < (*(T*)right) ) {
            return cmpResultToBoolForPyOrdering(pyComparisonOp, -1);
        }
        if ( (*(T*)left) > (*(T*)right) ) {
            return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
        }

        return cmpResultToBoolForPyOrdering(pyComparisonOp, 0);
    }

    typed_python_hash_type hash(instance_ptr left) {
        HashAccumulator acc;

        acc.addRegister(*(T*)left);

        return acc.get();
    }

    void constructor(instance_ptr self) {
        new ((T*)self) T();
    }

    void destroy(instance_ptr self) {}

    void copy_constructor(instance_ptr self, instance_ptr other) {
        *((T*)self) = *((T*)other);
    }

    void assign(instance_ptr self, instance_ptr other) {
        *((T*)self) = *((T*)other);
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t wireType) {
        buffer.readRegisterType((T*)self, wireType);
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber) {
        buffer.writeRegisterType(fieldNumber, *(T*)self);
    }

    size_t deepBytecountConcrete(instance_ptr instance, std::unordered_set<void*>& alreadyVisited, std::set<Slab*>* outSlabs) {
        return 0;
    }

    void deepcopyConcrete(
        instance_ptr dest,
        instance_ptr src,
        DeepcopyContext& context
    ) {
        copy_constructor(dest, src);
    }
};


PyDoc_STRVAR(Bool_doc,
    "Bool(x=False) -> register type for bool"
    );
class Bool : public RegisterType<bool> {
public:
    Bool() : RegisterType(TypeCategory::catBool)
    {
        m_name = "bool";
        m_doc = Bool_doc;
    }

    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr) {
        stream << (*(bool*)self ? "True":"False");
    }

    static Bool* Make() {
        static Bool* res = new Bool();
        return res;
    }
};

PyDoc_STRVAR(UInt8_doc,
    "UInt8(x=0) -> register type for uint8_t"
    );
class UInt8 : public RegisterType<uint8_t> {
public:
    UInt8() : RegisterType(TypeCategory::catUInt8)
    {
        m_name = "UInt8";
        m_doc = UInt8_doc;
    }

    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr) {
        stream << (uint64_t)*(uint8_t*)self << "u8";
    }

    static UInt8* Make() {
        static UInt8* res = new UInt8();
        return res;
    }
};

PyDoc_STRVAR(UInt16_doc,
    "UInt16(x=0) -> register type for uint16_t"
    );
class UInt16 : public RegisterType<uint16_t> {
public:
    UInt16() : RegisterType(TypeCategory::catUInt16)
    {
        m_name = "UInt16";
        m_doc = UInt16_doc;
    }

    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr) {
        stream << (uint64_t)*(uint16_t*)self << "u16";
    }

    static UInt16* Make() { static UInt16* res = new UInt16(); return res; }
};

PyDoc_STRVAR(UInt32_doc,
    "UInt32(x=0) -> register type for uint32_t"
    );
class UInt32 : public RegisterType<uint32_t> {
public:
    UInt32() : RegisterType(TypeCategory::catUInt32)
    {
        m_name = "UInt32";
        m_doc = UInt32_doc;
    }

    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr) {
        stream << (uint64_t)*(uint32_t*)self << "u32";
    }

    static UInt32* Make() { static UInt32* res = new UInt32(); return res; }
};

PyDoc_STRVAR(UInt64_doc,
    "UInt64(x=0) -> register type for uint64_t"
    );
class UInt64 : public RegisterType<uint64_t> {
public:
    UInt64() : RegisterType(TypeCategory::catUInt64)
    {
        m_name = "UInt64";
        m_doc = UInt64_doc;
    }

    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr) {
        stream << *(uint64_t*)self << "u64";
    }

    static UInt64* Make() { static UInt64* res = new UInt64(); return res; }
};

PyDoc_STRVAR(Int8_doc,
    "Int8(x=0) -> register type for int8_t"
    );
class Int8 : public RegisterType<int8_t> {
public:
    Int8() : RegisterType(TypeCategory::catInt8)
    {
        m_name = "Int8";
        m_doc = Int8_doc;
        m_size = 1; // Why only specify this here and not in other specializations?
    }

    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr) {
        stream << (int64_t)*(int8_t*)self << "i8";
    }

    static Int8* Make() { static Int8* res = new Int8(); return res; }
};

PyDoc_STRVAR(Int16_doc,
    "Int16(x=0) -> register type for int16_t"
    );
class Int16 : public RegisterType<int16_t> {
public:
    Int16() : RegisterType(TypeCategory::catInt16)
    {
        m_name = "Int16";
        m_doc = Int16_doc;
    }

    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr) {
        stream << (int64_t)*(int16_t*)self << "i16";
    }

    static Int16* Make() { static Int16* res = new Int16(); return res; }
};

PyDoc_STRVAR(Int32_doc,
    "Int32(x=0) -> register type for int32_t"
    );
class Int32 : public RegisterType<int32_t> {
public:
    Int32() : RegisterType(TypeCategory::catInt32)
    {
        m_name = "Int32";
        m_doc = Int32_doc;
    }

    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr) {
        stream << (int64_t)*(int32_t*)self << "i32";
    }

    static Int32* Make() { static Int32* res = new Int32(); return res; }
};

PyDoc_STRVAR(Int64_doc,
    "Int64(x=0) -> register type for int64_t"
    );
class Int64 : public RegisterType<int64_t> {
public:
    Int64() : RegisterType(TypeCategory::catInt64)
    {
        m_name = "int";
        m_doc = Int64_doc;
    }

    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr) {
        stream << *(int64_t*)self;
    }

    static Int64* Make() { static Int64* res = new Int64(); return res; }
};

PyDoc_STRVAR(Float32_doc,
    "Float32(x=0) -> register type for (C++) float"
    );
class Float32 : public RegisterType<float> {
public:
    Float32() : RegisterType(TypeCategory::catFloat32)
    {
        m_name = "Float32";
        m_doc = Float32_doc;
    }

    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr) {
        stream << *(float*)self << "f32";
    }

    static Float32* Make() { static Float32* res = new Float32(); return res; }
};

PyDoc_STRVAR(Float64_doc,
    "Float64(x=0) -> register type for (C++) double"
    );
class Float64 : public RegisterType<double> {
public:
    Float64() : RegisterType(TypeCategory::catFloat64)
    {
        m_name = "float";
        m_doc = Float64_doc;
    }

    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr) {
        // this is never actually called
        stream << *(double*)self;
    }

    static Float64* Make() { static Float64* res = new Float64(); return res; }
};

template<class T>
class GetRegisterType {};

template<> class GetRegisterType<bool> { public: Type* operator()() const { return Bool::Make(); } };
template<> class GetRegisterType<int8_t> { public: Type* operator()() const { return Int8::Make(); } };
template<> class GetRegisterType<int16_t> { public: Type* operator()() const { return Int16::Make(); } };
template<> class GetRegisterType<int32_t> { public: Type* operator()() const { return Int32::Make(); } };
template<> class GetRegisterType<int64_t> { public: Type* operator()() const { return Int64::Make(); } };
template<> class GetRegisterType<uint8_t> { public: Type* operator()() const { return UInt8::Make(); } };
template<> class GetRegisterType<uint16_t> { public: Type* operator()() const { return UInt16::Make(); } };
template<> class GetRegisterType<uint32_t> { public: Type* operator()() const { return UInt32::Make(); } };
template<> class GetRegisterType<uint64_t> { public: Type* operator()() const { return UInt64::Make(); } };
template<> class GetRegisterType<float> { public: Type* operator()() const { return Float32::Make(); } };
template<> class GetRegisterType<double> { public: Type* operator()() const { return Float64::Make(); } };

template<>
class TypeDetails<int64_t> {
public:
    static Type* getType() { return Int64::Make(); }

    static const uint64_t bytecount = sizeof(int64_t);
};

template<>
class TypeDetails<uint64_t> {
public:
    static Type* getType() { return UInt64::Make(); }

    static const uint64_t bytecount = sizeof(uint64_t);
};

template<>
class TypeDetails<int32_t> {
public:
    static Type* getType() { return Int32::Make(); }

    static const uint64_t bytecount = sizeof(int32_t);
};

template<>
class TypeDetails<uint32_t> {
public:
    static Type* getType() { return UInt32::Make(); }

    static const uint64_t bytecount = sizeof(uint32_t);
};

template<>
class TypeDetails<int16_t> {
public:
    static Type* getType() { return Int16::Make(); }

    static const uint64_t bytecount = sizeof(int16_t);
};

template<>
class TypeDetails<uint16_t> {
public:
    static Type* getType() { return UInt16::Make(); }

    static const uint64_t bytecount = sizeof(uint16_t);
};

template<>
class TypeDetails<int8_t> {
public:
    static Type* getType() { return Int8::Make(); }

    static const uint64_t bytecount = sizeof(int8_t);
};

template<>
class TypeDetails<uint8_t> {
public:
    static Type* getType() { return UInt8::Make(); }

    static const uint64_t bytecount = sizeof(uint8_t);
};

template<>
class TypeDetails<bool> {
public:
    static Type* getType() { return Bool::Make(); }

    static const uint64_t bytecount = sizeof(bool);
};

template<>
class TypeDetails<float> {
public:
    static Type* getType() { return Float32::Make(); }

    static const uint64_t bytecount = sizeof(float);
};

template<>
class TypeDetails<double> {
public:
    static Type* getType() { return Float64::Make(); }

    static const uint64_t bytecount = sizeof(double);
};
