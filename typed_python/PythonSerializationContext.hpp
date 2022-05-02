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
#include <Python.h>
#include "util.hpp"
#include "Type.hpp"
#include "SerializationContext.hpp"

// PySet_CheckExact is missing from the CPython API for some reason
#ifndef PySet_CheckExact
  #define PySet_CheckExact(obj)        (Py_TYPE(obj) == &PySet_Type)
#endif


// Wrapping this macro with a function so we can use it in templated code
inline PyObject* PyList_Get_Item_No_Checks(PyObject* obj, long idx) {
    return PyList_GET_ITEM(obj, idx);
}

// Wrapping this macro with a function so we can use it in templated code
inline void PyList_Set_Item_No_Checks(PyObject* obj, long idx, PyObject* item) {
    PyList_SET_ITEM(obj, idx, item);
}

// Wrapping this macro with a function so we can use it in templated code
inline PyObject* PyTuple_Get_Item_No_Checks(PyObject* obj, long idx) {
    return PyTuple_GET_ITEM(obj, idx);
}

// Wrapping this macro with a function so we can use it in templated code
inline void PyTuple_Set_Item_No_Checks(PyObject* o, long k, PyObject* item) {
    PyTuple_SET_ITEM(o, k, item);
}

class PythonSerializationContext : public SerializationContext {
public:
    //enums in our protocol (serialized as uint8_t)
    class FieldNumbers {
    public:
        enum {
            MEMO = 0, //a varint encoding the ID of the object in the memo stream.
                      //if this memo has been defined already in the stream, no other
                      //fields should be present in the stream.
            NATIVE_INSTANCE = 2, //field 0 is the type, field 1 is the data.
            OBJECT_NAME = 3, //a string encoding the name of the object in the current codebase
            OBJECT_TYPEANDDICT = 4, //an object where the object's python type is encoded as
                                    //field 0, and the dictionary is encoded as field 1
            OBJECT_REPRESENTATION = 5, //a python object representing an objects' representation
            FLOAT = 6, //the object is a 64-bit float
            LONG = 7, //the object is a varint encoding a python long
            BOOL = 8, //the object is a varint encoding a python bool (1 for true, 0 for False)
            LIST = 9, //the object is a list with items encoded by index in a child compound
            TUPLE = 10, //the object is a tuple
            SET = 11, //the object is a set
            DICT = 12, //the object is a dict with keys and values encoded in alternating order
            NONE = 13, //the object is an empty compound encoding None.
            UNICODE = 14, //the object is a BYTES encoding a utf8-encoded string
            BYTES = 15, //the object is a BYTES encoding actual the actual bytes
            FROZENSET = 16, //the object is a frozenset with items encoded by index
            CELL = 17, //the object is a cell object
            NATIVE_TYPE = 18, //a native type, which will consist of a number in the group
                              //and a recursive type group
            RECURSIVE_OBJECT = 19, //a global object that's got a sha-hash. for example,
                                   //a class defined in a TypeFunction. this will consist of
                                   //an index and a recursive type group.
        };
    };

    PythonSerializationContext(PyObject* typeSetObj) :
            mContextObj(typeSetObj),
            mCompressionEnabled(false),
            mSerializeHashSequence(false)
    {
        setFlags();
    }

    void setFlags();

    bool isCompressionEnabled() const {
        return mCompressionEnabled;
    }

    // should we serialize an integer in the order of the
    // hash sequence rather than the hash itself?
    bool shouldSerializeHashSequence() const {
        return mSerializeHashSequence;
    }

    virtual void serializePythonObject(PyObject* o, SerializationBuffer& b, size_t fieldNumber) const;

    void serializePythonObjectNamedOrAsObj(PyObject* o, SerializationBuffer& b) const;

    void serializePythonObjectRepresentation(PyObject* o, SerializationBuffer& b, size_t fieldNumber) const;

    void serializeNativeType(Type* nativeType, SerializationBuffer& b, size_t fieldNumber) const;

    void serializeNativeTypeInner(Type* nativeType, SerializationBuffer& b, size_t fieldNumber) const;

    void serializeClassMembers(const std::vector<MemberDefinition>& members, SerializationBuffer& b, int fieldNumber) const;

    void serializeAlternativeMembers(const std::vector<std::pair<std::string, NamedTuple*> >& members, SerializationBuffer& b, int fieldNumber) const;

    void serializeClassFunDict(const std::map<std::string, Function*>& dict, SerializationBuffer& b, int fieldNumber) const;

    void serializeClassClassMemberDict(const std::map<std::string, PyObject*>& dict, SerializationBuffer& b, int fieldNumber) const;

    void deserializeAlternativeMembers(std::vector<std::pair<std::string, NamedTuple*> >& members, DeserializationBuffer& b, int wireType) const;

    void deserializeClassMembers(std::vector<MemberDefinition>& members, DeserializationBuffer& b, int wireType) const;

    void deserializeClassFunDict(std::string className, std::map<std::string, Function*>& dict, DeserializationBuffer& b, int wireType) const;

    void deserializeClassClassMemberDict(std::map<std::string, PyObjectHolder>& dict, DeserializationBuffer& b, int wireType) const;

    Type* deserializeNativeTypeInner(DeserializationBuffer& b, size_t wireType, bool actuallyProduceResult) const;

    Instance deserializeNativeInstance(DeserializationBuffer& b, size_t wireType) const;

    virtual PyObject* deserializePythonObject(DeserializationBuffer& b, size_t wireType) const;

    Type* deserializeNativeType(DeserializationBuffer& b, size_t wireType) const;

    PyObject* deserializeRecursiveObject(DeserializationBuffer& b, size_t wireType) const;

    void serializeMutuallyRecursiveTypeGroup(MutuallyRecursiveTypeGroup* group, SerializationBuffer& b, size_t fieldNumber) const;

    MutuallyRecursiveTypeGroup* deserializeMutuallyRecursiveTypeGroup(DeserializationBuffer& b, size_t inWireType) const;

    PyObject* deserializePythonObjectFromName(DeserializationBuffer& b, size_t wireType, int64_t memo) const;

    PyObject* deserializePythonObjectFromTypeAndDict(DeserializationBuffer& b, size_t wireType, int64_t memo) const;

    PyObject* deserializePythonObjectFromRepresentation(DeserializationBuffer& b, size_t wireType, int64_t memo) const;

    std::string getNameForPyObj(PyObject* o) const;

private:
    template<class Factory_Fn, class SetItem_Fn>
    inline PyObject* deserializeIndexable(DeserializationBuffer& b, size_t wireType, Factory_Fn factory_fn, SetItem_Fn set_item_and_steal_ref_fn, int64_t memo) const;

    void serializeIterable(PyObject* o, SerializationBuffer& b, size_t fieldNumber) const;

    template<class Factory_Fn, class AddItem_Fn, class Clear_Fn>
    inline PyObject* deserializeIterable(DeserializationBuffer &b, size_t wireType, Factory_Fn factory_fn, AddItem_Fn add_item_fn, Clear_Fn clear_fn, int64_t memo) const;

    void serializePyList(PyObject* o, SerializationBuffer& b) const;

    PyObject* deserializePyList(DeserializationBuffer& b, size_t wireType, int64_t memo) const;

    void serializePyTuple(PyObject* o, SerializationBuffer& b) const;

    PyObject* deserializePyTuple(DeserializationBuffer& b, size_t wireType, int64_t memo) const;

    void serializePySet(PyObject* o, SerializationBuffer& b) const;

    PyObject* deserializePySet(DeserializationBuffer &b, size_t wireType, int64_t memo) const;

    void serializePyDict(PyObject* o, SerializationBuffer& b) const;

    PyObject* deserializePyDict(DeserializationBuffer& b, size_t wireType, int64_t memo) const;

    void serializePyCell(PyObject* o, SerializationBuffer& b) const;

    PyObject* deserializePyCell(DeserializationBuffer& b, size_t wireType, int64_t memo) const;

    void serializePyFrozenSet(PyObject* o, SerializationBuffer& b) const;

    PyObject* deserializePyFrozenSet(DeserializationBuffer &b, size_t wireType, int64_t memo) const;

    PyObject* mContextObj;

    bool mCompressionEnabled;

    bool mInternalizeTypeGroups;

    bool mSerializeHashSequence;
};
