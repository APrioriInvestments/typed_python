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
#include <Python.h>
#include "util.hpp"
#include "Type.hpp"
#include "SerializationContext.hpp"

// PySet_CheckExact is missing from the CPython API for some reason
#ifndef PySet_CheckExact
  #define PySet_CheckExact(obj)        (Py_TYPE(obj) == &PySet_Type)
#endif


static inline void throwDerivedClassError(std::string type) {
    throw std::runtime_error(
        std::string("Classes derived from `" + type + "` cannot be serialized")
    );
}

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

//models bytes held in a python 'bytes' object.
class PyBytesByteBuffer : public ByteBuffer {
public:
    explicit PyBytesByteBuffer(PyObject* obj) : m_obj(incref(obj)) {
        if (!PyObject_CheckBuffer(obj)) {
            PyErr_Format(PyExc_TypeError, "Not a buffer object.");
            throw PythonExceptionSet();
        }

        if (PyObject_GetBuffer(obj, &m_buffer, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS) == -1) {
            throw PythonExceptionSet();
        }
    }

    virtual ~PyBytesByteBuffer() {
        PyBuffer_Release(&m_buffer);
        decref(m_obj);
    }

    virtual std::pair<uint8_t*, uint8_t*> range() {
        assertHoldingTheGil();
        return std::make_pair((uint8_t*)m_buffer.buf, (uint8_t*)m_buffer.buf + m_buffer.len);
    }

private:
    PyObject* m_obj;
    Py_buffer m_buffer;
};

class PythonSerializationContext : public SerializationContext {
public:
    //enums in our protocol (serialized as uint8_t)
    enum {
        T_NATIVE,   //a native type, followed by instance data
        T_OBJECT,   //an object written either with a representation, or as a type and a dict, or as a name
        T_LIST,
        T_TUPLE,
        T_SET,
        T_FROZENSET,
        T_DICT,
        T_NONE,
        T_TRUE,
        T_FALSE,
        T_UNICODE,
        T_BYTES,
        T_FLOAT,
        T_LONG,      //currently serialized as an int64_t which is completely wrong.
        T_NATIVETYPE,
        T_OBJECT_NAMED,
        T_OBJECT_TYPEANDDICT,
        T_OBJECT_REPRESENTATION,
        T_NATIVETYPE_BY_CATEGORY
    };

    PythonSerializationContext(PyObject* typeSetObj) :
            mContextObj(typeSetObj)
    {
    }

    std::shared_ptr<ByteBuffer> compress(uint8_t* begin, uint8_t* end) const;

    std::shared_ptr<ByteBuffer> decompress(uint8_t* begin, uint8_t* end) const;

    std::shared_ptr<ByteBuffer> compressOrDecompress(uint8_t* begin, uint8_t* end, bool compress) const;

    virtual void serializePythonObject(PyObject* o, SerializationBuffer& b) const;

    void serializePyDict(PyObject* o, SerializationBuffer& b) const;

    PyObject* deserializePydict(DeserializationBuffer& b) const;

    PyObject* deserializePyFrozenSet(DeserializationBuffer &b) const;

    void serializePythonObjectNamedOrAsObj(PyObject* o, SerializationBuffer& b) const;

    PyObject* deserializePythonObjectNamedOrAsObj(DeserializationBuffer& b) const;

    void serializeNativeType(Type* nativeType, SerializationBuffer& b, bool allowCaching=false) const;

    Type* deserializeNativeType(DeserializationBuffer& b, bool allowCaching=false) const;

    Type* deserializeNativeTypeUncached(DeserializationBuffer& b) const;

    virtual PyObject* deserializePythonObject(DeserializationBuffer& b) const;

    void serializePyRepresentation(PyObject* representation, SerializationBuffer& b) const;

    PyObject* deserializePyRepresentation(DeserializationBuffer& b, int32_t objectId) const;

private:
    template<class Size_Fn, class GetItem_Fn>
    inline void serializeIndexable(PyObject* o, SerializationBuffer& b, Size_Fn size_fn, GetItem_Fn get_item_fn) const;

    template<class Factory_Fn, class SetItem_Fn>
    inline PyObject* deserializeIndexable(DeserializationBuffer& b, Factory_Fn factory_fn, SetItem_Fn set_item_fn) const;

    template<class Size_Fn>
    inline void serializeIterable(PyObject* o, SerializationBuffer& b, Size_Fn size_fn) const;

    template<class Factory_Fn, class AddItem_Fn, class Clear_Fn>
    inline PyObject* deserializeIterable(DeserializationBuffer &b, Factory_Fn factory_fn, AddItem_Fn add_item_fn, Clear_Fn clear_fn) const;

    void serializePyList(PyObject* o, SerializationBuffer& b) const;

    PyObject* deserializePyList(DeserializationBuffer& b) const;

    void serializePyTuple(PyObject* o, SerializationBuffer& b) const;

    PyObject* deserializePyTuple(DeserializationBuffer& b) const;

    void serializePySet(PyObject* o, SerializationBuffer& b) const;

    PyObject* deserializePySet(DeserializationBuffer &b) const;

    void serializePyFrozenSet(PyObject* o, SerializationBuffer& b) const;

    PyObject* mContextObj;
};

