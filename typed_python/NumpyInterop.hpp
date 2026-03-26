/******************************************************************************
   Copyright 2017-2024 typed_python Authors

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
#include <string>
#include <cstring>
#include "Type.hpp"

// Numpy interop without compile-time numpy dependency.
//
// Instead of #include <numpy/arrayobject.h>, we use:
//   - The Python buffer protocol (PEP 3118) to read/write array data
//   - Runtime-cached numpy type objects for scalar type detection
//   - Runtime calls to numpy.empty() for array creation
//
// If numpy is not installed, all detection functions return false and
// array creation returns nullptr with a Python error set.

namespace NumpyInterop {

// Cached numpy type objects, populated at module init time.
struct NumpyTypeCache {
    bool initialized = false;
    bool available = false;

    PyObject* numpy_module = nullptr;

    // Scalar type objects
    PyTypeObject* bool_type = nullptr;
    PyTypeObject* float16_type = nullptr;
    PyTypeObject* float32_type = nullptr;
    PyTypeObject* float64_type = nullptr;
    PyTypeObject* longdouble_type = nullptr;

    PyTypeObject* int8_type = nullptr;
    PyTypeObject* int16_type = nullptr;
    PyTypeObject* int32_type = nullptr;
    PyTypeObject* int64_type = nullptr;
    // numpy 'long' maps to C long, which is platform-dependent
    PyTypeObject* long_type = nullptr;
    PyTypeObject* longlong_type = nullptr;

    PyTypeObject* uint8_type = nullptr;
    PyTypeObject* uint16_type = nullptr;
    PyTypeObject* uint32_type = nullptr;
    PyTypeObject* uint64_type = nullptr;
    PyTypeObject* ulong_type = nullptr;
    PyTypeObject* ulonglong_type = nullptr;

    // The ndarray type itself
    PyTypeObject* ndarray_type = nullptr;
};

inline NumpyTypeCache& getCache() {
    static NumpyTypeCache cache;
    return cache;
}

// Attempt to cache a type from numpy module. Returns nullptr if not found.
inline PyTypeObject* cacheType(PyObject* mod, const char* name) {
    PyObject* obj = PyObject_GetAttrString(mod, name);
    if (!obj) {
        PyErr_Clear();
        return nullptr;
    }
    if (PyType_Check(obj)) {
        return (PyTypeObject*)obj;  // keeps a reference
    }
    Py_DECREF(obj);
    return nullptr;
}

// Initialize the numpy type cache. Call once at module init.
// Safe to call if numpy is not installed.
inline void init() {
    NumpyTypeCache& c = getCache();
    if (c.initialized) return;
    c.initialized = true;

    c.numpy_module = PyImport_ImportModule("numpy");
    if (!c.numpy_module) {
        PyErr_Clear();
        c.available = false;
        return;
    }
    c.available = true;

    c.ndarray_type = cacheType(c.numpy_module, "ndarray");

    c.bool_type = cacheType(c.numpy_module, "bool_");
    c.float16_type = cacheType(c.numpy_module, "float16");
    c.float32_type = cacheType(c.numpy_module, "float32");
    c.float64_type = cacheType(c.numpy_module, "float64");
    c.longdouble_type = cacheType(c.numpy_module, "longdouble");

    c.int8_type = cacheType(c.numpy_module, "int8");
    c.int16_type = cacheType(c.numpy_module, "int16");
    c.int32_type = cacheType(c.numpy_module, "int32");
    c.int64_type = cacheType(c.numpy_module, "int64");
    c.long_type = cacheType(c.numpy_module, "long");
    c.longlong_type = cacheType(c.numpy_module, "longlong");

    c.uint8_type = cacheType(c.numpy_module, "uint8");
    c.uint16_type = cacheType(c.numpy_module, "uint16");
    c.uint32_type = cacheType(c.numpy_module, "uint32");
    c.uint64_type = cacheType(c.numpy_module, "uint64");
    c.ulong_type = cacheType(c.numpy_module, "ulong");
    c.ulonglong_type = cacheType(c.numpy_module, "ulonglong");
}

// Check if an object is a numpy ndarray.
inline bool isNumpyArray(PyObject* obj) {
    NumpyTypeCache& c = getCache();
    if (!c.available || !c.ndarray_type) return false;
    return PyObject_IsInstance(obj, (PyObject*)c.ndarray_type) == 1;
}

// Check if a type is a numpy float scalar type.
inline bool isNumpyFloatType(PyTypeObject* t) {
    NumpyTypeCache& c = getCache();
    if (!c.available) return false;
    return (
        t == c.float16_type
        || t == c.float32_type
        || t == c.float64_type
        || t == c.longdouble_type
    );
}

// Check if a type is a numpy integer scalar type.
inline bool isNumpyIntType(PyTypeObject* t) {
    NumpyTypeCache& c = getCache();
    if (!c.available) return false;
    return (
        t == c.int8_type
        || t == c.int16_type
        || t == c.int32_type
        || t == c.int64_type
        || t == c.long_type
        || t == c.longlong_type
        || t == c.uint8_type
        || t == c.uint16_type
        || t == c.uint32_type
        || t == c.uint64_type
        || t == c.ulong_type
        || t == c.ulonglong_type
    );
}

// Check if a type is any numpy scalar type (bool, float, or int).
inline bool isNumpyScalarType(PyTypeObject* t) {
    NumpyTypeCache& c = getCache();
    if (!c.available) return false;
    return t == c.bool_type || isNumpyFloatType(t) || isNumpyIntType(t);
}

// Map a numpy scalar type to the best typed_python type category.
inline Type::TypeCategory numpyScalarTypeToBestCategory(PyTypeObject* t) {
    NumpyTypeCache& c = getCache();
    if (t == c.bool_type) { return Type::TypeCategory::catBool; }
    if (t == c.float16_type) { return Type::TypeCategory::catFloat32; }
    if (t == c.float32_type) { return Type::TypeCategory::catFloat32; }
    if (t == c.float64_type) { return Type::TypeCategory::catFloat64; }
    if (t == c.longdouble_type) { return Type::TypeCategory::catFloat64; }
    if (t == c.int8_type) { return Type::TypeCategory::catInt8; }
    if (t == c.int16_type) { return Type::TypeCategory::catInt16; }
    if (t == c.int32_type) { return Type::TypeCategory::catInt32; }
    if (t == c.long_type) {
        return sizeof(long) == 8 ? Type::TypeCategory::catInt64 : Type::TypeCategory::catInt32;
    }
    if (t == c.int64_type) { return Type::TypeCategory::catInt64; }
    if (t == c.longlong_type) { return Type::TypeCategory::catInt64; }
    if (t == c.uint8_type) { return Type::TypeCategory::catUInt8; }
    if (t == c.uint16_type) { return Type::TypeCategory::catUInt16; }
    if (t == c.uint32_type) { return Type::TypeCategory::catUInt32; }
    if (t == c.ulong_type) {
        return sizeof(long) == 8 ? Type::TypeCategory::catUInt64 : Type::TypeCategory::catUInt32;
    }
    if (t == c.uint64_type) { return Type::TypeCategory::catUInt64; }
    if (t == c.ulonglong_type) { return Type::TypeCategory::catUInt64; }

    throw std::runtime_error("Type is not a numpy type.");
}

// Buffer protocol element type enum (replaces NPY_* constants)
enum class BufferDtype {
    Unknown,
    Bool,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float32, Float64
};

// Map a buffer protocol format character to our dtype enum.
// See PEP 3118 / struct module format strings.
inline BufferDtype formatCharToDtype(char fmt) {
    switch (fmt) {
        case '?': return BufferDtype::Bool;
        case 'b': return BufferDtype::Int8;
        case 'h': return BufferDtype::Int16;
        case 'i': return BufferDtype::Int32;
        case 'q': return BufferDtype::Int64;
        case 'B': return BufferDtype::UInt8;
        case 'H': return BufferDtype::UInt16;
        case 'I': return BufferDtype::UInt32;
        case 'Q': return BufferDtype::UInt64;
        case 'f': return BufferDtype::Float32;
        case 'd': return BufferDtype::Float64;
        case 'l': // C long - platform dependent
            return sizeof(long) == 8 ? BufferDtype::Int64 : BufferDtype::Int32;
        case 'L': // C unsigned long
            return sizeof(unsigned long) == 8 ? BufferDtype::UInt64 : BufferDtype::UInt32;
        case 'n': // ssize_t
            return sizeof(Py_ssize_t) == 8 ? BufferDtype::Int64 : BufferDtype::Int32;
        case 'N': // size_t
            return sizeof(size_t) == 8 ? BufferDtype::UInt64 : BufferDtype::UInt32;
        default: return BufferDtype::Unknown;
    }
}

// Parse a buffer format string to a dtype. Handles optional byte-order prefix
// (e.g. "<d", "=i", "@f") and numpy's format strings.
inline BufferDtype parseBufferFormat(const char* format) {
    if (!format || !format[0]) return BufferDtype::Unknown;

    const char* p = format;
    // Skip byte-order/alignment prefix characters
    if (*p == '@' || *p == '=' || *p == '<' || *p == '>' || *p == '!') {
        p++;
    }
    if (!*p) return BufferDtype::Unknown;
    // Should be a single format char
    if (p[1] != '\0') return BufferDtype::Unknown;
    return formatCharToDtype(*p);
}

// Create a numpy array of given size and dtype string.
// dtype_str should be a numpy dtype name like "float64", "int32", etc.
// Returns a new reference, or nullptr with Python error set.
inline PyObject* createNumpyArray(Py_ssize_t size, const char* dtype_str) {
    NumpyTypeCache& c = getCache();
    if (!c.available || !c.numpy_module) {
        PyErr_SetString(PyExc_ImportError, "numpy is not available");
        return nullptr;
    }

    PyObject* empty_func = PyObject_GetAttrString(c.numpy_module, "empty");
    if (!empty_func) return nullptr;

    PyObject* shape = PyLong_FromSsize_t(size);
    PyObject* dtype = PyObject_GetAttrString(c.numpy_module, dtype_str);
    if (!dtype) {
        Py_DECREF(empty_func);
        Py_DECREF(shape);
        return nullptr;
    }

    PyObject* args = PyTuple_Pack(1, shape);
    PyObject* kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "dtype", dtype);

    PyObject* result = PyObject_Call(empty_func, args, kwargs);

    Py_DECREF(empty_func);
    Py_DECREF(shape);
    Py_DECREF(dtype);
    Py_DECREF(args);
    Py_DECREF(kwargs);

    return result;
}

// RAII wrapper for Py_buffer
struct ScopedBuffer {
    Py_buffer view;
    bool valid;

    ScopedBuffer(PyObject* obj, int flags = PyBUF_FORMAT | PyBUF_STRIDES) {
        valid = (PyObject_GetBuffer(obj, &view, flags) == 0);
    }

    ~ScopedBuffer() {
        if (valid) PyBuffer_Release(&view);
    }

    ScopedBuffer(const ScopedBuffer&) = delete;
    ScopedBuffer& operator=(const ScopedBuffer&) = delete;
};

} // namespace NumpyInterop
