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

#include "PythonSerializationContext.hpp"
#include "AllTypes.hpp"
#include "PyInstance.hpp"

void PythonSerializationContext::setCompressionEnabled() {
    PyEnsureGilAcquired acquireTheGil;

    PyObjectStealer isEnabled(PyObject_GetAttrString(mContextObj, "compressionEnabled"));

    if (!isEnabled) {
        throw PythonExceptionSet();
    }

    mCompressionEnabled = ((PyObject*)isEnabled) == Py_True;
}

// virtual
void PythonSerializationContext::serializePythonObject(PyObject* o, SerializationBuffer& b, size_t fieldNumber) const {
    PyEnsureGilAcquired acquireTheGil;

    b.writeBeginCompound(fieldNumber);

    Type* t = PyInstance::extractTypeFrom(o->ob_type);

    if (t) {
        b.writeBeginCompound(FieldNumbers::NATIVE_INSTANCE);

        serializeNativeType(t, b, 0);

        PyEnsureGilReleased releaseTheGil;
        t->serialize(((PyInstance*)o)->dataPtr(), b, 1);

        b.writeEndCompound();
    } else {
        if (o == Py_None) {
            b.writeEmpty(FieldNumbers::NONE);
        } else
        if (o == Py_True) {
            b.writeUnsignedVarintObject(FieldNumbers::BOOL, 1);
        } else
        // The checks for 'True' and 'False' must happen before the test for PyLong
        // because bool is a subtype of int in Python
        if (o == Py_False) {
            b.writeUnsignedVarintObject(FieldNumbers::BOOL, 0);
        } else
        if (PyLong_CheckExact(o)) {
            //this will fail for very large integers. We should fix this
            b.writeSignedVarintObject(FieldNumbers::LONG, PyLong_AsLongLong(o));
        } else
        if (PyFloat_CheckExact(o)) {
            b.writeRegisterType(FieldNumbers::FLOAT, PyFloat_AsDouble(o));
        } else
        if (PyComplex_CheckExact(o)) {
            throw std::runtime_error(std::string("`complex` objects cannot be serialized yet"));
        } else
        if (PyBytes_CheckExact(o)) {
            b.writeBeginBytes(FieldNumbers::BYTES, PyBytes_GET_SIZE(o));
            b.write_bytes((uint8_t*)PyBytes_AsString(o), PyBytes_GET_SIZE(o));
        } else
        if (PyUnicode_CheckExact(o)) {
            Py_ssize_t sz;
            const char* c = PyUnicode_AsUTF8AndSize(o, &sz);
            b.writeBeginBytes(FieldNumbers::UNICODE, sz);
            b.write_bytes((uint8_t*)c, sz);
        } else
        if (PyList_CheckExact(o)) {
            serializePyList(o, b);
        } else
        if (PyTuple_CheckExact(o)) {
            serializePyTuple(o, b);
        } else
        if (PySet_CheckExact(o)) {
            serializePySet(o, b);
        } else
        if (PyFrozenSet_CheckExact(o)) {
            serializePyFrozenSet(o, b);
        } else
        if (PyDict_CheckExact(o)) {
            static PyObject* builtinsModule = PyImport_ImportModule("builtins");
            static PyObject* builtinsModuleDict = PyObject_GetAttrString(builtinsModule, "__dict__");

            if (builtinsModuleDict == o) {
                serializePythonObjectNamedOrAsObj(o, b);
            } else {
                serializePyDict(o, b);
            }
        } else
        if (PyType_Check(o)) {
            Type* nativeType = PyInstance::extractTypeFrom((PyTypeObject*)o);

            if ((PyTypeObject*)o == &PyLong_Type) {
                nativeType = Int64::Make();
            }
            else if ((PyTypeObject*)o == &PyFloat_Type) {
                nativeType = Float64::Make();
            }
            else if ((PyTypeObject*)o == Py_None->ob_type) {
                nativeType = NoneType::Make();
            }
            else if ((PyTypeObject*)o == &PyBool_Type) {
                nativeType = Bool::Make();
            }
            else if ((PyTypeObject*)o == &PyBytes_Type) {
                nativeType = BytesType::Make();
            }
            else if ((PyTypeObject*)o == &PyUnicode_Type) {
                nativeType = StringType::Make();
            }

            if (nativeType) {
                serializeNativeType(nativeType, b, FieldNumbers::NATIVE_TYPE);
            } else {
                serializePythonObjectNamedOrAsObj(o, b);
            }
        } else
        if (PyCell_Check(o)) {
            serializePyCell(o, b);
        } else {
            serializePythonObjectNamedOrAsObj(o, b);
        }
    }

    b.writeEndCompound();
}

// virtual
PyObject* PythonSerializationContext::deserializePythonObject(DeserializationBuffer& b, size_t inWireType) const {
    PyEnsureGilAcquired acquireTheGil;

    PyObject* result = nullptr;
    int64_t memo = -1;

    b.consumeCompoundMessage(inWireType, [&](size_t fieldNumber, size_t wireType) {
        if (fieldNumber == FieldNumbers::MEMO) {
            assertWireTypesEqual(wireType, WireType::VARINT);

            if (memo != -1) {
                throw std::runtime_error("Corrupt stream: multiple memos found.");
            }

            memo = b.readUnsignedVarint();

            if (result) {
                throw std::runtime_error("Corrupt stream: memo found after object definition.");
            }

            PyObject* memoObj = (PyObject*)b.lookupCachedPointer(memo);
            if (memoObj) {
                result = incref(memoObj);
            }
        } else {
            if (result) {
                throw std::runtime_error("result already populated");
            }

            if (fieldNumber == FieldNumbers::NONE) {
                assertWireTypesEqual(wireType, WireType::EMPTY);
                result = incref(Py_None);
            } else if (fieldNumber == FieldNumbers::BOOL) {
                assertWireTypesEqual(wireType, WireType::VARINT);
                size_t val = b.readUnsignedVarint();
                result = incref(val ? Py_True : Py_False);
            } else if (fieldNumber == FieldNumbers::FLOAT) {
                assertWireTypesEqual(wireType, WireType::BITS_64);
                double val = b.read_double();
                result = PyFloat_FromDouble(val);
            } else if (fieldNumber == FieldNumbers::LONG) {
                assertWireTypesEqual(wireType, WireType::VARINT);
                int64_t val = b.readSignedVarint();
                result = PyLong_FromLong(val);
            }else if (fieldNumber == FieldNumbers::BYTES) {
                assertWireTypesEqual(wireType, WireType::BYTES);
                int64_t sz = b.readUnsignedVarint();

                b.read_bytes_fun(sz, [&](uint8_t* ptr) {
                    result = PyBytes_FromStringAndSize((const char*)ptr, sz);
                });
            } else if (fieldNumber == FieldNumbers::UNICODE) {
                assertWireTypesEqual(wireType, WireType::BYTES);
                int64_t sz = b.readUnsignedVarint();

                b.read_bytes_fun(sz, [&](uint8_t* ptr) {
                    result = PyUnicode_DecodeUTF8((const char*)ptr, sz, nullptr);
                    if (!result) {
                        throw PythonExceptionSet();
                    }
                });
            } else if (fieldNumber == FieldNumbers::LIST) {
                result = deserializePyList(b, wireType, memo);
            } else if (fieldNumber == FieldNumbers::DICT) {
                result = deserializePyDict(b, wireType, memo);
            } else if (fieldNumber == FieldNumbers::SET) {
                result = deserializePySet(b, wireType, memo);
            } else if (fieldNumber == FieldNumbers::TUPLE) {
                result = deserializePyTuple(b, wireType, memo);
            } else if (fieldNumber == FieldNumbers::FROZENSET) {
                result = deserializePyFrozenSet(b, wireType, memo);
            } else if (fieldNumber == FieldNumbers::NATIVE_INSTANCE) {
                result = PyInstance::fromInstance(deserializeNativeInstance(b, wireType));
            } else if (fieldNumber == FieldNumbers::NATIVE_TYPE) {
                result = incref((PyObject*)PyInstance::typeObj(deserializeNativeType(b, wireType)));
            } else if (fieldNumber == FieldNumbers::OBJECT_NAME) {
                result = deserializePythonObjectFromName(b, wireType, memo);
            } else if (fieldNumber == FieldNumbers::OBJECT_REPRESENTATION) {
                result = deserializePythonObjectFromRepresentation(b, wireType, memo);
            } else if (fieldNumber == FieldNumbers::CELL) {
                result = deserializePyCell(b, wireType, memo);
            } else if (fieldNumber == FieldNumbers::OBJECT_TYPEANDDICT) {
                result = deserializePythonObjectFromTypeAndDict(b, wireType, memo);
            } else {
                throw std::runtime_error("Unknown field number " + format(fieldNumber) + " deserializing python object.");
            }
        }
    });

    if (!result) {
        throw std::runtime_error("Corrupt state: neither a memo nor a result.");
    }

    return result;
}

void PythonSerializationContext::serializePyCell(PyObject* o, SerializationBuffer& b) const {
    PyEnsureGilAcquired acquireTheGil;

    uint32_t id;
    bool isNew;
    std::tie(id, isNew) = b.cachePointer(o);

    b.writeUnsignedVarintObject(FieldNumbers::MEMO, id);

    if (!isNew) {
        return;
    }

    b.writeBeginCompound(FieldNumbers::CELL);

    PyObject* cellContents = PyCell_GET(o);
    if (cellContents) {
        serializePythonObject(cellContents, b, 0);
    }

    b.writeEndCompound();
}

PyObject* PythonSerializationContext::deserializePyCell(DeserializationBuffer& b, size_t inWireType, int64_t memo) const {
    PyEnsureGilAcquired acquireTheGil;

    PyObjectStealer cell(PyCell_New(nullptr));

    if (memo != -1) {
        b.addCachedPyObj(memo, incref(cell));
    }

    b.consumeCompoundMessageWithImpliedFieldNumbers(inWireType, [&](size_t fieldNumber, size_t wiretype) {
        PyObjectHolder cellContents;
        cellContents.steal(deserializePythonObject(b, wiretype));

        PyCell_Set((PyObject*)cell, (PyObject*)cellContents);
    });

    return cell.extract();
}

void PythonSerializationContext::serializePyDict(PyObject* o, SerializationBuffer& b) const {
    PyEnsureGilAcquired acquireTheGil;

    uint32_t id;
    bool isNew;
    std::tie(id, isNew) = b.cachePointer(o);

    b.writeUnsignedVarintObject(FieldNumbers::MEMO, id);

    if (!isNew) {
        return;
    }

    b.writeBeginCompound(FieldNumbers::DICT);

    PyObject *key, *value;
    Py_ssize_t pos = 0;

    while (PyDict_Next(o, &pos, &key, &value)) {
        serializePythonObject(key, b, 0);
        serializePythonObject(value, b, 0);
    }

    b.writeEndCompound();
}

PyObject* PythonSerializationContext::deserializePyDict(DeserializationBuffer& b, size_t inWireType, int64_t memo) const {
    PyEnsureGilAcquired acquireTheGil;

    PyObjectStealer res(PyDict_New());

    if (memo != -1) {
        b.addCachedPyObj(memo, incref(res));
    }

    PyObjectHolder key;
    PyObjectHolder value;

    b.consumeCompoundMessageWithImpliedFieldNumbers(inWireType, [&](size_t fieldNumber, size_t wiretype) {
        if (fieldNumber % 2 == 0) {
            if (key) {
                throw std::runtime_error("Corrupt stream: key already populated.");
            }
            key.steal(deserializePythonObject(b, wiretype));
            if (!key) {
                throw PythonExceptionSet();
            }
        } else {
            if (value) {
                throw std::runtime_error("Corrupt stream: value already populated.");
            }
            if (!key) {
                throw std::runtime_error("Corrupt stream: key not populated.");
            }

            value.steal(deserializePythonObject(b, wiretype));

            if (!value) {
                throw PythonExceptionSet();
            }

            if (PyDict_SetItem(res, key, value) != 0) {
                throw PythonExceptionSet();
            }

            key.release();
            value.release();
        }
    });

    return res.extract();
}

void PythonSerializationContext::serializePyFrozenSet(PyObject* o, SerializationBuffer& b) const {
    serializeIterable(o, b, FieldNumbers::FROZENSET);
}

PyObject* PythonSerializationContext::deserializePyFrozenSet(DeserializationBuffer &b, size_t inWireType, int64_t memo) const {
    PyEnsureGilAcquired acquireTheGil;

    PyObject* res = PyFrozenSet_New(NULL);
    if (!res) {
        PyErr_PrintEx(1);
        throw std::runtime_error(
            std::string("Failed to allocate memory for frozen set deserialization"));
    }

    if (memo != -1) {
        b.addCachedPyObj(memo, incref(res));
    }

    try {
        b.consumeCompoundMessageWithImpliedFieldNumbers(inWireType, [&](size_t fieldNumber, size_t wireType) {
            if (fieldNumber == 0) {
                b.finishReadingMessageAndDiscard(wireType);
                return;
            }

            PyObject* item = deserializePythonObject(b, wireType);

            if (!item) {
                throw std::runtime_error(std::string("object in frozenset couldn't be deserialized"));
            }
            // In the process of deserializing a member, we may have increfed the frozenset
            // currently being deserialized. In that case PySet_Add will fail, so we temporarily
            // decref it.
            auto refcount = Py_REFCNT(res);
            for (int i = 1; i < refcount; i++) decref(res);

            int success = PySet_Add(res, item);

            for (int i = 1; i < refcount; i++) incref(res);

            decref(item);
            if (success < 0) {
                throw PythonExceptionSet();
            }
        });
    } catch(...) {
        PySet_Clear(res);
        decref(res);
        throw;
    }

    return res;
}

void PythonSerializationContext::serializePythonObjectNamedOrAsObj(PyObject* o, SerializationBuffer& b) const {
    PyEnsureGilAcquired acquireTheGil;

    uint32_t id;
    bool isNew;
    std::tie(id, isNew) = b.cachePointer(o);

    b.writeUnsignedVarintObject(FieldNumbers::MEMO, id);

    if (!isNew) {
        return;
    }

    //see if the object has a name
    PyObjectStealer typeName(PyObject_CallMethod(mContextObj, "nameForObject", "O", o));
    if (!typeName) {
        throw PythonExceptionSet();
    }

    if (typeName != Py_None) {
        if (!PyUnicode_Check(typeName)) {
            throw std::runtime_error(std::string("nameForObject returned a non-string"));
        }

        b.writeStringObject(FieldNumbers::OBJECT_NAME, std::string(PyUnicode_AsUTF8(typeName)));
        return;
    }

    //give the plugin a chance to convert the instance to something else
    PyObjectStealer representation(PyObject_CallMethod(mContextObj, "representationFor", "O", o));
    if (!representation) {
        throw PythonExceptionSet();
    }

    if (representation != Py_None) {
        serializePythonObjectRepresentation(representation, b);
        return;
    }

    //check whether this is a type derived from a serializable native type, which we don't support
    if (PyLong_Check(o)) {
        throwDerivedClassError("long");
    }
    if (PyFloat_Check(o)) {
        throwDerivedClassError("float");
    }
    if (PyBytes_Check(o)) {
        throwDerivedClassError("bytes");
    }
    if (PyComplex_Check(o)) {
        throwDerivedClassError("complex");
    }
    if (PyUnicode_Check(o)) {
        throwDerivedClassError("str");
    }
    if (PyList_Check(o)) {
        throwDerivedClassError("list");
    }
    if (PyTuple_Check(o)) {
        throwDerivedClassError("tuple");
    }
    if (PySet_Check(o)) {
        throwDerivedClassError("set");
    }
    if (PyFrozenSet_Check(o)) {
        throwDerivedClassError("frozenset");
    }
    if (PyDict_Check(o)) {
        throwDerivedClassError("dict");
    }
    if (PyModule_Check(o)) {
        throw std::runtime_error(
            std::string("Cannot serialize module '") + PyModule_GetName(o) + ("' because it's not explicitly named. "
                "Please ensure that it's imported as a module variable somewhere in your codebase.")
            );
    }

    b.writeBeginCompound(FieldNumbers::OBJECT_TYPEANDDICT);
    serializePythonObject((PyObject*)o->ob_type, b, 0);
    PyObjectStealer objDict(PyObject_GenericGetDict(o, nullptr));
    if (!objDict) {
        PyErr_Format(PyExc_TypeError,
            "Object %S (of type %S) had no dict", o, o->ob_type
        );

        throw PythonExceptionSet();
    }
    serializePythonObject(objDict, b, 1);
    b.writeEndCompound();
}

PyObject* PythonSerializationContext::deserializePythonObjectFromName(DeserializationBuffer& b, size_t wireType, int64_t memo) const {
    PyEnsureGilAcquired acquireTheGil;

    assertWireTypesEqual(wireType, WireType::BYTES);

    std::string name = b.readStringObject();

    PyObject* result = PyObject_CallMethod(mContextObj, "objectFromName", "s", name.c_str());

    if (!result) {
        throw PythonExceptionSet();
    }

    if (memo != -1) {
        b.addCachedPyObj(memo, incref(result));
    }

    return result;
}

Type* PythonSerializationContext::deserializeNativeTypeFromRepresentation(DeserializationBuffer& b, size_t inWireType, int64_t memo) const {
    PyEnsureGilAcquired acquireTheGil;

    PyObjectHolder factory, factoryArgs;
    PyObjectHolder value;
    PyObjectHolder state;

    b.consumeCompoundMessage(inWireType, [&](size_t fieldNumber, size_t wireType) {
        if (fieldNumber == 0) {
            factory.steal(deserializePythonObject(b, wireType));
        }
        else if (fieldNumber == 1) {
            if (!factory) {
                throw std::runtime_error("Corrupt stream: no factory defined for python object representation");
            }

            factoryArgs.steal(deserializePythonObject(b, wireType));

            if (!factoryArgs) {
                throw PythonExceptionSet();
            }

            value.steal(PyObject_Call((PyObject*)factory, (PyObject*)factoryArgs, NULL));

            if (!value) {
                throw PythonExceptionSet();
            }
        }
        else if (fieldNumber == 2) {
            if (!value) {
                throw std::runtime_error("Invalid representation.");
            }

            state.steal(deserializePythonObject(b, wireType));

            if (!state) {
                throw PythonExceptionSet();
            }

            PyObjectStealer res(
                PyObject_CallMethod(mContextObj, "setInstanceStateFromRepresentation", "OO", (PyObject*)value, (PyObject*)state)
            );

            if (!res) {
                throw PythonExceptionSet();
            }
            if (res != Py_True) {
                throw std::runtime_error("setInstanceStateFromRepresentation didn't return True.");
            }
        } else {
            throw std::runtime_error("corrupt python object representation");
        }
    });

    if (!value) {
        throw std::runtime_error("Invalid representation.");
    }

    if (!PyType_Check(value)) {
        throw std::runtime_error("Expected value from representation to be a type.");
    }

    Type* resultType = PyInstance::extractTypeFrom((PyTypeObject*)(PyObject*)value);

    if (!resultType) {
        throw std::runtime_error("Expected value from representation to be a type.");
    }

    if (memo != -1) {
        b.addCachedPointer(memo, resultType);
    }

    return resultType;
}

PyObject* PythonSerializationContext::deserializePythonObjectFromRepresentation(DeserializationBuffer& b, size_t inWireType, int64_t memo) const {
    PyEnsureGilAcquired acquireTheGil;

    PyObjectHolder factory, factoryArgs;
    PyObjectHolder value;
    PyObjectHolder state;

    b.consumeCompoundMessage(inWireType, [&](size_t fieldNumber, size_t wireType) {
        if (fieldNumber == 0) {
            factory.steal(deserializePythonObject(b, wireType));
        }
        else if (fieldNumber == 1) {
            if (!factory) {
                throw std::runtime_error("Corrupt stream: no factory defined for python object representation");
            }

            factoryArgs.steal(deserializePythonObject(b, wireType));

            if (!factoryArgs) {
                throw PythonExceptionSet();
            }

            value.steal(PyObject_Call((PyObject*)factory, (PyObject*)factoryArgs, NULL));

            if (!value) {
                throw PythonExceptionSet();
            }

            if (memo != -1) {
                b.addCachedPyObj(memo, incref(value));
            }
        }
        else if (fieldNumber == 2) {
            if (!value) {
                throw std::runtime_error("Invalid representation.");
            }

            state.steal(deserializePythonObject(b, wireType));

            if (!state) {
                throw PythonExceptionSet();
            }

            PyObjectStealer res(
                PyObject_CallMethod(mContextObj, "setInstanceStateFromRepresentation", "OO", (PyObject*)value, (PyObject*)state)
            );

            if (!res) {
                throw PythonExceptionSet();
            }
            if (res != Py_True) {
                throw std::runtime_error("setInstanceStateFromRepresentation didn't return True.");
            }
        } else {
            throw std::runtime_error("corrupt python object representation");
        }
    });

    if (!value) {
        throw std::runtime_error("Invalid representation.");
    }

    return incref(value);
}

PyObject* PythonSerializationContext::deserializePythonObjectFromTypeAndDict(DeserializationBuffer& b, size_t inWireType, int64_t memo) const {
    PyEnsureGilAcquired acquireTheGil;

    PyObjectHolder type;
    PyObjectHolder value;
    PyObjectHolder dict;

    b.consumeCompoundMessage(inWireType, [&](size_t fieldNumber, size_t wireType) {
        if (fieldNumber == 0) {
            type.steal(deserializePythonObject(b, wireType));

            if (!type) {
                throw std::runtime_error("Invalid representation.");
            }

            if (!PyType_Check(type)) {
                std::string tname = type->ob_type->tp_name;
                throw std::runtime_error("Expected a type object. Got " + tname + " instead");
            }

            static PyObject* emptyTuple = PyTuple_Pack(0);
            value.steal(((PyTypeObject*)type)->tp_new(((PyTypeObject*)type), emptyTuple, NULL));
            if (!value) {
                throw std::runtime_error("tp_new for " + std::string(((PyTypeObject*)type)->tp_name) + " threw an exception");
            }

            if (memo != -1) {
                b.addCachedPyObj(memo, incref(value));
            }
        }
        if (fieldNumber == 1) {
            if (!value) {
                throw std::runtime_error("Invalid representation.");
            }
            dict.steal(deserializePythonObject(b, wireType));

            if (PyObject_GenericSetDict(value, dict, nullptr) == -1) {
                throw PythonExceptionSet();
            }
        }
    });

    if (!value) {
        throw std::runtime_error("Invalid representation.");
    }

    return incref(value);
}

void PythonSerializationContext::serializePythonObjectRepresentation(PyObject* representation, SerializationBuffer& b) const {
    PyEnsureGilAcquired acquireTheGil;

    if (!PyTuple_Check(representation) || PyTuple_Size(representation) != 3) {
        throw std::runtime_error("representationFor should return None or a tuple with 3 things");
    }
    if (!PyTuple_Check(PyTuple_GetItem(representation, 1))) {
        throw std::runtime_error("representationFor second arguments should be a tuple");
    }

    PyObjectHolder rep0(PyTuple_GetItem(representation, 0));
    PyObjectHolder rep1(PyTuple_GetItem(representation, 1));
    PyObjectHolder rep2(PyTuple_GetItem(representation, 2));

    b.writeBeginCompound(FieldNumbers::OBJECT_REPRESENTATION);
    serializePythonObject(rep0, b, 0);
    serializePythonObject(rep1, b, 1);
    serializePythonObject(rep2, b, 2);
    b.writeEndCompound();
}

void PythonSerializationContext::serializeNativeType(Type* nativeType, SerializationBuffer& b, size_t fieldNumber) const {
    b.writeBeginCompound(fieldNumber);
    serializeNativeTypeInner(nativeType, b);
    b.writeEndCompound();
}

void PythonSerializationContext::serializeNativeTypeInner(
            Type* nativeType,
            SerializationBuffer& b
            ) const {
    Type::TypeCategory cat = nativeType->getTypeCategory();

    if (cat == Type::TypeCategory::catInt8 ||
        cat == Type::TypeCategory::catInt16 ||
        cat == Type::TypeCategory::catInt32 ||
        cat == Type::TypeCategory::catInt64 ||
        cat == Type::TypeCategory::catUInt8 ||
        cat == Type::TypeCategory::catUInt16 ||
        cat == Type::TypeCategory::catUInt32 ||
        cat == Type::TypeCategory::catUInt64 ||
        cat == Type::TypeCategory::catFloat32 ||
        cat == Type::TypeCategory::catFloat64 ||
        cat == Type::TypeCategory::catNone ||
        cat == Type::TypeCategory::catBytes ||
        cat == Type::TypeCategory::catString ||
        cat == Type::TypeCategory::catBool ||
        cat == Type::TypeCategory::catPyCell)
    {
        b.writeBeginCompound(FieldNumbers::NATIVE_TYPE);
        b.writeUnsignedVarintObject(0, nativeType->getTypeCategory());
        b.writeEndCompound();
        return;
    }

    PyEnsureGilAcquired acquireTheGil;

    uint32_t id;
    bool isNew;
    std::tie(id, isNew) = b.cachePointer(nativeType, nullptr);

    b.writeUnsignedVarintObject(FieldNumbers::MEMO, id);

    if (!isNew) {
        return;
    }

    PyObjectStealer nameForObject(PyObject_CallMethod(mContextObj, "nameForObject", "O", PyInstance::typeObj(nativeType)));

    if (!nameForObject) {
        throw PythonExceptionSet();
    }

    if (nameForObject != Py_None) {
        if (!PyUnicode_Check(nameForObject)) {
            decref(nameForObject);
            throw std::runtime_error("nameForObject returned something other than None or a string.");
        }

        b.writeStringObject(FieldNumbers::OBJECT_NAME, std::string(PyUnicode_AsUTF8(nameForObject)));

        return;
    }

    PyObjectStealer representation(
        PyObject_CallMethod(mContextObj, "representationFor", "O", PyInstance::typeObj(nativeType))
    );

    if (!representation) {
        throw PythonExceptionSet();
    }

    if (representation != Py_None) {
        serializePythonObjectRepresentation(representation, b);
        return;
    }

    MarkTypeBeingSerialized marker(nativeType, b);

    if (nativeType->isRecursiveForward()) {
        b.writeBeginCompound(FieldNumbers::RECURSIVE_NATIVE_TYPE);
        b.writeStringObject(0, nativeType->name());
        b.writeBeginCompound(1);
    } else {
        b.writeBeginCompound(FieldNumbers::NATIVE_TYPE);
    }

    b.writeUnsignedVarintObject(0, nativeType->getTypeCategory());

    if (nativeType->getTypeCategory() == Type::TypeCategory::catConcreteAlternative) {
        serializeNativeType(nativeType->getBaseType(), b, 1);
        b.writeUnsignedVarintObject(2, ((ConcreteAlternative*)nativeType)->which());
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catSet) {
        serializeNativeType(((SetType*)nativeType)->keyType(), b, 1);
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catConstDict) {
        serializeNativeType(((ConstDictType*)nativeType)->keyType(), b, 1);
        serializeNativeType(((ConstDictType*)nativeType)->valueType(), b, 2);
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catDict) {
        serializeNativeType(((DictType*)nativeType)->keyType(), b, 1);
        serializeNativeType(((DictType*)nativeType)->valueType(), b, 2);
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catTupleOf) {
        serializeNativeType(((TupleOfType*)nativeType)->getEltType(), b, 1);
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catListOf) {
        serializeNativeType(((ListOfType*)nativeType)->getEltType(), b, 2);
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catTuple) {
        size_t index = 1;
        for (auto t: ((CompositeType*)nativeType)->getTypes()) {
            serializeNativeType(t, b, index++);
        }
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catNamedTuple) {
        size_t index = 1;
        for (long k = 0; k < ((CompositeType*)nativeType)->getTypes().size(); k++) {
            b.writeStringObject(index++, ((CompositeType*)nativeType)->getNames()[k]);
            serializeNativeType(((CompositeType*)nativeType)->getTypes()[k], b, index++);
        }
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catOneOf) {
        size_t index = 1;
        for (auto t: ((OneOfType*)nativeType)->getTypes()) {
            serializeNativeType(t, b, index++);
        }
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catPythonObjectOfType) {
        serializePythonObject((PyObject*)((PythonObjectOfType*)nativeType)->pyType(), b, 1);
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catValue) {
        b.writeBeginCompound(1);

        Instance i = ((Value*)nativeType)->value();
        serializeNativeType(i.type(), b, 0);

        PyEnsureGilReleased releaseTheGil;
        i.type()->serialize(i.data(), b, 1);
        b.writeEndCompound();
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catFunction) {
        Function* ftype = (Function*)nativeType;

        serializeNativeType(ftype->getClosureType(), b, 1);
        b.writeStringObject(2, ftype->name());
        b.writeUnsignedVarintObject(3, ftype->isEntrypoint() ? 1 : 0);
        b.writeUnsignedVarintObject(4, ftype->isNocompile() ? 1 : 0);

        int whichIndex = 5;
        for (auto& overload: ftype->getOverloads()) {
            overload.serialize(*this, b, whichIndex++);
        }
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catHeldClass) {
        serializeNativeType(((HeldClass*)nativeType)->getClassType(), b, 1);
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catForward) {
        b.writeStringObject(1, nativeType->name());
        if (((Forward*)nativeType)->getTarget()) {
            serializeNativeType(((Forward*)nativeType)->getTarget(), b, 2);
        }
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catClass) {
        Class* cls = (Class*)nativeType;

        b.writeStringObject(1, cls->name());

        //members
        serializeClassMembers(cls->getOwnMembers(), b, 2);

        //methods
        serializeClassFunDict(cls->getOwnMemberFunctions(), b, 3);
        serializeClassFunDict(cls->getOwnStaticFunctions(), b, 4);
        serializeClassFunDict(cls->getOwnPropertyFunctions(), b, 5);
        serializeClassClassMemberDict(cls->getOwnClassMembers(), b, 6);

        b.writeBeginCompound(7);

        {
            int which = 0;
            for (auto t: cls->getBases()) {
                serializeNativeType(t->getClassType(), b, which++);
            }
        }

        b.writeEndCompound();

        b.writeUnsignedVarintObject(8, cls->isFinal() ? 1 : 0);
    } else {
        throw std::runtime_error(
            "Can't serialize native type " + nativeType->name()
            + " of category " + Type::categoryToString(nativeType->getTypeCategory())
            + " if its unnamed."
        );
    }

    if (nativeType->isRecursiveForward()) {
        b.writeEndCompound();
        b.writeEndCompound();
    } else {
        b.writeEndCompound();
    }
}

void PythonSerializationContext::serializeClassMembers(
    const std::vector<std::tuple<std::string, Type*, Instance> >& members,
    SerializationBuffer& b,
    int fieldNumber
) const {
    b.writeBeginCompound(fieldNumber);
    for (long k = 0; k < members.size(); k++) {
        b.writeBeginCompound(k);
            b.writeStringObject(0, std::get<0>(members[k]));

            serializeNativeType(std::get<1>(members[k]), b, 1);

            Instance defaultValue = std::get<2>(members[k]);

            if (defaultValue.type()->getTypeCategory() != Type::TypeCategory::catNone) {
                b.writeBeginCompound(2);

                serializeNativeType(defaultValue.type(), b, 0);

                PyEnsureGilReleased releaseTheGil;
                defaultValue.type()->serialize(defaultValue.data(), b, 1);

                b.writeEndCompound();
            }
        b.writeEndCompound();
    }
    b.writeEndCompound();
}

void PythonSerializationContext::serializeClassFunDict(
    const std::map<std::string, Function*>& dict,
    SerializationBuffer& b,
    int fieldNumber
) const {
    b.writeBeginCompound(fieldNumber);

    int which = 0;

    for (auto& nameAndFunType: dict) {
        b.writeBeginCompound(which++);

        b.writeStringObject(0, nameAndFunType.first);
        serializeNativeType(nameAndFunType.second, b, 1);

        b.writeEndCompound();
    }

    b.writeEndCompound();
}

void PythonSerializationContext::serializeClassClassMemberDict(
    const std::map<std::string, PyObject*>& dict,
    SerializationBuffer& b,
    int fieldNumber
) const {
    b.writeBeginCompound(fieldNumber);

    int which = 0;

    for (auto& nameAndObj: dict) {
        b.writeBeginCompound(which++);

        b.writeStringObject(0, nameAndObj.first);
        serializePythonObject(nameAndObj.second, b, 1);

        b.writeEndCompound();
    }

    b.writeEndCompound();
}

void PythonSerializationContext::deserializeClassMembers(
    std::vector<std::tuple<std::string, Type*, Instance> >& members,
    DeserializationBuffer& b,
    int inWireType
) const {
    b.consumeCompoundMessage(inWireType, [&](size_t fieldNumber, size_t wireType) {
        std::string name;
        Type* t;
        Instance i;

        b.consumeCompoundMessage(wireType, [&](size_t fieldNumber2, size_t wireType2) {
            if (fieldNumber2 == 0) {
                name = b.readStringObject();
            } else if (fieldNumber2 == 1) {
                t = deserializeNativeType(b, wireType2);
            } else if (fieldNumber2 == 2) {
                i = deserializeNativeInstance(b, wireType2);
            } else {
                throw std::runtime_error("Corrupt ClassMember definition when deserializing class.");
            }
        });

        if (!t || name.size() == 0) {
            throw std::runtime_error("Corrupt ClassMember definition when deserializing class.");
        }

        members.push_back(std::tuple<std::string, Type*, Instance>(name, t, i));
    });
}

void PythonSerializationContext::deserializeClassFunDict(
    std::map<std::string, Function*>& dict,
    DeserializationBuffer& b,
    int inWireType
) const {
    b.consumeCompoundMessage(inWireType, [&](size_t fieldNumber, size_t wireType) {
        std::string name;
        Type* fun = nullptr;

        b.consumeCompoundMessage(wireType, [&](size_t fieldNumber2, size_t wireType2) {
            if (fieldNumber2 == 0) {
                name = b.readStringObject();
            } else if (fieldNumber2 == 1) {
                fun = deserializeNativeType(b, wireType2);
            } else {
                throw std::runtime_error("Corrupt function definition when deserializing class.");
            }
        });

        if (!fun || name.size() == 0 || fun->getTypeCategory() != Type::TypeCategory::catFunction) {
            throw std::runtime_error("Corrupt function definition when deserializing class.");
        }

        dict[name] = (Function*)fun;
    });
}

void PythonSerializationContext::deserializeClassClassMemberDict(
    std::map<std::string, PyObject*>& dict,
    DeserializationBuffer& b,
    int inWireType
) const {
    b.consumeCompoundMessage(inWireType, [&](size_t fieldNumber, size_t wireType) {
        std::string name;
        PyObject* o = nullptr;

        b.consumeCompoundMessage(wireType, [&](size_t fieldNumber2, size_t wireType2) {
            if (fieldNumber2 == 0) {
                name = b.readStringObject();
            } else if (fieldNumber2 == 1) {
                o = deserializePythonObject(b, wireType2);
            } else {
                throw std::runtime_error("Corrupt ClassMember definition when deserializing class.");
            }
        });

        if (!o || name.size() == 0) {
            throw std::runtime_error("Corrupt ClassMember definition when deserializing class.");
        }

        dict[name] = o;
    });
}

Type* PythonSerializationContext::deserializeNativeType(DeserializationBuffer& b, size_t inWireType) const {
    PyEnsureGilAcquired acquireTheGil;

    Type* resultType = nullptr;
    int32_t memo = -1;

    b.consumeCompoundMessage(inWireType, [&](size_t fieldNumber, size_t wireType) {
        if (fieldNumber == FieldNumbers::MEMO) {
            assertWireTypesEqual(wireType, WireType::VARINT);

            if (memo != -1) {
                throw std::runtime_error("Corrupt stream: multiple memos found.");
            }

            memo = b.readUnsignedVarint();

            if (resultType) {
                throw std::runtime_error("Corrupt stream: memo found after type definition.");
            }

            resultType = (Type*)b.lookupCachedPointer(memo);
        }
        else if (fieldNumber == FieldNumbers::OBJECT_NAME) {
            assertWireTypesEqual(wireType, WireType::BYTES);

            std::string name = b.readStringObject();

            PyObjectStealer result(PyObject_CallMethod(mContextObj, "objectFromName", "s", name.c_str()));

            if (!result) {
                throw PythonExceptionSet();
            }

            if (!PyType_Check(result)) {
                throw std::runtime_error("Expected value named " + name + " to be a type.");
            }

            if (result == &PyLong_Type) {
                resultType = Int64::Make();
            }
            else if (result == &PyFloat_Type) {
                resultType = Float64::Make();
            }
            else if (result == &PyBool_Type) {
                resultType = Bool::Make();
            }
            else if (result == &PyUnicode_Type) {
                resultType = StringType::Make();
            }
            else if (result == &PyBytes_Type) {
                resultType = BytesType::Make();
            }
            else if (result == Py_None->ob_type) {
                resultType = NoneType::Make();
            } else {
                resultType = PyInstance::extractTypeFrom((PyTypeObject*)result);
            }

            if (!resultType) {
                throw std::runtime_error("Expected value named " + name + " to be a type.");
            }

            if (memo != -1) {
                b.addCachedPointer(memo, resultType);
            }
        } else if (fieldNumber == FieldNumbers::OBJECT_REPRESENTATION) {
            resultType = deserializeNativeTypeFromRepresentation(b, wireType, memo);
        } else if (fieldNumber == FieldNumbers::NATIVE_TYPE) {
            resultType = deserializeNativeTypeInner(b, wireType, memo);

            if (!resultType) {
                throw std::runtime_error("Corrupt native type: deserializeNativeTypeInner returned Null");
            }
        } else if (fieldNumber == FieldNumbers::RECURSIVE_NATIVE_TYPE) {
            Forward* fwd = nullptr;

            if (memo == -1) {
                throw std::runtime_error("Corrupt Recursive Forward");
            }

            b.consumeCompoundMessage(wireType, [&](size_t subFieldNumber, size_t subWireType) {
                if (subFieldNumber == 0) {
                    assertWireTypesEqual(subWireType, WireType::BYTES);

                    std::string name = b.readStringObject();

                    fwd = Forward::Make(name);

                    b.addCachedPointer(memo, fwd);
                } else if (subFieldNumber == 1) {
                    if (!fwd) {
                        throw std::runtime_error("Corrupt Recursive Forward");
                    }
                    resultType = deserializeNativeTypeInner(b, subWireType, -1);

                    fwd->define(resultType);

                    b.updateCachedPointer(memo, resultType);
                } else {
                    throw std::runtime_error("Corrupt Recursive Forward");
                }
            });
        }
    });

    if (!resultType) {
        throw std::runtime_error("Corrupt native type");
    }

    return resultType;
}

Instance PythonSerializationContext::deserializeNativeInstance(DeserializationBuffer& b, size_t inWireType) const {
    PyEnsureGilAcquired acquireTheGil;

    Instance result;
    Type* type = nullptr;

    b.consumeCompoundMessage(inWireType, [&](size_t fieldNumber, size_t wireType) {
        if (fieldNumber == 0) {
            type = deserializeNativeType(b, wireType);
        }
        if (fieldNumber == 1) {
            if (!type) {
                throw std::runtime_error("Invalid native Instance: no type object.");
            }
            result = Instance(type, [&](instance_ptr data) {
                PyEnsureGilReleased releaseTheGil;
                type->deserialize(data, b, wireType);
            });
        }
    });

    return result;
}

// deserialize a type. if memo is != -1, then we are intended to stash a forward in the global memo
// before continuing deserialization, and then to resolve the forward to the appropriate type
// once it's known. this is necessary for things like Class objects, which may not be recursive
// forwards as defined in a normal codebase, but which need recursion to deserialize properly because
// we have detached their function bodies from the standard global scope, and therefore they are now
// able to find themselves without a global 'dict'  (which is memoized) standing in between.
Type* PythonSerializationContext::deserializeNativeTypeInner(DeserializationBuffer& b, size_t inWireType, int32_t memo) const {
    PyEnsureGilAcquired acquireTheGil;

    Forward* addedToMemoEarly = nullptr;

    int category = -1;

    std::vector<Type*> types;
    std::vector<std::string> names;
    int64_t whichIndex = -1;
    PyObjectHolder obj;
    Instance instance;

    std::vector<Function::Overload> overloads;
    Type* closureType = 0;
    int isEntrypoint = 0;
    int isNocompile = 0;

    std::vector<Class*> classBases;
    bool classIsFinal = false;
    std::map<std::string, Function*> classMethods, classStatics, classPropertyFunctions;
    std::map<std::string, PyObject*> classClassMembers;
    std::vector<std::tuple<std::string, Type*, Instance> > classMembers;

    b.consumeCompoundMessage(inWireType, [&](size_t fieldNumber, size_t wireType) {
        if (fieldNumber == 0) {
            assertWireTypesEqual(wireType, WireType::VARINT);
            category = b.readUnsignedVarint();
        } else {
            if (category == -1) {
                throw std::runtime_error("Corrupt native type found.");
            }
            if (category == Type::TypeCategory::catForward) {
                if (fieldNumber == 1) {
                    names.push_back(b.readStringObject());
                } else if (fieldNumber == 2) {
                    types.push_back(deserializeNativeType(b, wireType));
                }
            } else
            if (category == Type::TypeCategory::catHeldClass) {
                types.push_back(deserializeNativeType(b, wireType));
            } else
            if (category == Type::TypeCategory::catClass) {
                if (fieldNumber == 1) {
                    names.push_back(b.readStringObject());

                    if (memo != -1) {
                        // Class objects can refer to themselves implicitly,
                        // and therefore may require us to create a temporary
                        // forward to hold them.
                        addedToMemoEarly = Forward::Make(names[0]);
                        b.addCachedPointer(memo, addedToMemoEarly);
                    }
                } else if (fieldNumber == 2) {
                    deserializeClassMembers(classMembers, b, wireType);
                } else if (fieldNumber == 3) {
                    deserializeClassFunDict(classMethods, b, wireType);
                } else if (fieldNumber == 4) {
                    deserializeClassFunDict(classStatics, b, wireType);
                } else if (fieldNumber == 5) {
                    deserializeClassFunDict(classPropertyFunctions, b, wireType);
                } else if (fieldNumber == 6) {
                    deserializeClassClassMemberDict(classClassMembers, b, wireType);
                } else if (fieldNumber == 7) {
                    b.consumeCompoundMessage(wireType, [&](int which, size_t wireType2) {
                        Type* t = deserializeNativeType(b, wireType2);
                        if (!t || t->getTypeCategory() != Type::TypeCategory::catClass) {
                            throw std::runtime_error("Corrupt class base found.");
                        }
                        classBases.push_back((Class*)t);
                    });
                } else if (fieldNumber == 8) {
                    assertWireTypesEqual(wireType, WireType::VARINT);
                    classIsFinal = b.readUnsignedVarint();
                }
            } else
            if (category == Type::TypeCategory::catFunction) {
                if (fieldNumber == 1) {
                    closureType = deserializeNativeType(b, wireType);
                }
                else if (fieldNumber == 2) {
                    assertWireTypesEqual(wireType, WireType::BYTES);
                    names.push_back(b.readStringObject());
                }
                else if (fieldNumber == 3) {
                    assertWireTypesEqual(wireType, WireType::VARINT);
                    isEntrypoint = b.readUnsignedVarint();
                }
                else if (fieldNumber == 4) {
                    assertWireTypesEqual(wireType, WireType::VARINT);
                    isNocompile = b.readUnsignedVarint();
                }
                else {
                    overloads.push_back(
                        Function::Overload::deserialize(*this, b, wireType)
                    );
                }
            } else
            if (category == Type::TypeCategory::catOneOf ||
                category == Type::TypeCategory::catTupleOf ||
                category == Type::TypeCategory::catListOf ||
                category == Type::TypeCategory::catSet ||
                category == Type::TypeCategory::catDict ||
                category == Type::TypeCategory::catConstDict ||
                category == Type::TypeCategory::catPointerTo ||
                category == Type::TypeCategory::catTuple ||
                //named tuples alternate between strings for names and type values
                (category == Type::TypeCategory::catNamedTuple && (fieldNumber % 2 == 0)) ||
                //alternatives encode one type exactly
                (category == Type::TypeCategory::catConcreteAlternative && fieldNumber == 1)
            ) {
                types.push_back(deserializeNativeType(b, wireType));
            }
            else if (category == Type::TypeCategory::catNamedTuple) {
                assertWireTypesEqual(wireType, WireType::BYTES);
                names.push_back(b.readStringObject());
            } else if (category == Type::TypeCategory::catPythonObjectOfType && fieldNumber == 1) {
                obj.steal(deserializePythonObject(b, wireType));
            } else if (category == Type::TypeCategory::catValue && fieldNumber == 1) {
                instance = deserializeNativeInstance(b, wireType);
            } else if (category == Type::TypeCategory::catConcreteAlternative && fieldNumber == 2) {
                assertWireTypesEqual(wireType, WireType::VARINT);
                whichIndex = b.readUnsignedVarint();
            } else {
                throw std::runtime_error("invalid category and fieldNumber");
            }
        }
    });

    Type* resultType = nullptr;

    if (category == Type::TypeCategory::catHeldClass) {
        if (types.size() != 1 || types[0]->getTypeCategory() != Type::TypeCategory::catClass) {
            throw std::runtime_error("Corrupt 'HeldClass' encountered.");
        }
        resultType = ((Class*)types[0])->getHeldClass();
    }
    else if (category == Type::TypeCategory::catForward) {
        if (names.size() != 1 || types.size() > 1) {
            throw std::runtime_error("Corrupt 'Forward' encountered.");
        }
        resultType = Forward::Make(names[0]);
        if (types.size()) {
            ((Forward*)resultType)->define(types[0]);
        }
    }
    else if (category == Type::TypeCategory::catClass) {
        if (names.size() != 1) {
            throw std::runtime_error("Corrupt 'Class' encountered.");
        }

        resultType = Class::Make(
            names[0],
            classBases,
            classIsFinal,
            classMembers,
            classMethods,
            classStatics,
            classPropertyFunctions,
            classClassMembers
        );
    }
    else if (category == Type::TypeCategory::catFunction) {
        if (names.size() != 1) {
            throw std::runtime_error("Badly structured 'Function' encountered.");
        }

        resultType = Function::Make(
            names[0],
            overloads,
            closureType,
            isEntrypoint,
            isNocompile
        );
    }
    else if (category == Type::TypeCategory::catUInt8) {
        resultType = ::UInt8::Make();
    }
    else if (category == Type::TypeCategory::catUInt16) {
        resultType = ::UInt16::Make();
    }
    else if (category == Type::TypeCategory::catUInt32) {
        resultType = ::UInt32::Make();
    }
    else if (category == Type::TypeCategory::catUInt64) {
        resultType = ::UInt64::Make();
    }
    else if (category == Type::TypeCategory::catInt8) {
        resultType = ::Int8::Make();
    }
    else if (category == Type::TypeCategory::catInt16) {
        resultType = ::Int16::Make();
    }
    else if (category == Type::TypeCategory::catInt32) {
        resultType = ::Int32::Make();
    }
    else if (category == Type::TypeCategory::catInt64) {
        resultType = ::Int64::Make();
    }
    else if (category == Type::TypeCategory::catFloat32) {
        resultType = ::Float32::Make();
    }
    else if (category == Type::TypeCategory::catFloat64) {
        resultType = ::Float64::Make();
    }
    else if (category == Type::TypeCategory::catNone) {
        resultType = ::NoneType::Make();
    }
    else if (category == Type::TypeCategory::catBytes) {
        resultType = ::BytesType::Make();
    }
    else if (category == Type::TypeCategory::catString) {
        resultType = ::StringType::Make();
    }
    else if (category == Type::TypeCategory::catPyCell) {
        resultType = ::PyCellType::Make();
    }
    else if (category == Type::TypeCategory::catBool) {
        resultType = ::Bool::Make();
    }
    else if (category == Type::TypeCategory::catOneOf) {
        resultType = ::OneOfType::Make(types);
    }
    else if (category == Type::TypeCategory::catValue) {
        resultType = ::Value::Make(instance);
    }
    else if (category == Type::TypeCategory::catTupleOf) {
        if (types.size() != 1) {
            throw std::runtime_error("Invalid native type: TupleOf needs exactly 1 type.");
        }
        resultType = ::TupleOfType::Make(types[0]);
    }
    else if (category == Type::TypeCategory::catListOf) {
        if (types.size() != 1) {
            throw std::runtime_error("Invalid native type: ListOf needs exactly 1 type.");
        }
        resultType = ::ListOfType::Make(types[0]);
    }
    else if (category == Type::TypeCategory::catPointerTo) {
        if (types.size() != 1) {
            throw std::runtime_error("Invalid native type: PointerTo needs exactly 1 type.");
        }
        resultType = ::PointerTo::Make(types[0]);
    }
    else if (category == Type::TypeCategory::catNamedTuple) {
        resultType = ::NamedTuple::Make(types, names);
    }
    else if (category == Type::TypeCategory::catTuple) {
        resultType = ::Tuple::Make(types);
    }
    else if (category == Type::TypeCategory::catSet) {
        if (types.size() != 1) {
            throw std::runtime_error("Invalid native type: Set needs exactly 1 type.");
        }
        resultType = ::SetType::Make(types[0]);
    }
    else if (category == Type::TypeCategory::catDict) {
        if (types.size() != 2) {
            throw std::runtime_error("Invalid native type: Dict needs exactly 2 types.");
        }
        resultType = ::DictType::Make(types[0], types[1]);
    }
    else if (category == Type::TypeCategory::catConstDict) {
        if (types.size() != 2) {
            throw std::runtime_error("Invalid native type: ConstDict needs exactly 2 types.");
        }
        resultType = ::ConstDictType::Make(types[0], types[1]);
    }
    else if (category == Type::TypeCategory::catPythonObjectOfType) {
        if (!obj || !PyType_Check(obj)) {
            throw std::runtime_error("Invalid native type: PythonObjectOfType needs a python type.");
        }

        resultType = ::PythonObjectOfType::Make((PyTypeObject*)(PyObject*)obj);
    }
    else if (category == Type::TypeCategory::catConcreteAlternative) {
        if (types.size() != 1) {
            throw std::runtime_error("Invalid native type: ConcreteAlternative needs exactly 1 type.");
        }
        if (whichIndex < 0) {
            throw std::runtime_error("Invalid native type: ConcreteAlternative needs an integer value");
        }

        Type* base = types[0];
        if (base->getTypeCategory() != Type::TypeCategory::catAlternative) {
            throw std::runtime_error("corrupt data: expected an Alternative type here");
        }

        Alternative* a = (Alternative*)base;
        if (whichIndex >= a->subtypes().size()) {
            throw std::runtime_error("corrupt data: invalid alternative specified");
        }

        resultType = ::ConcreteAlternative::Make(a, whichIndex);
    } else {
        throw std::runtime_error("Invalid native type category");
    }

    if (!resultType) {
        throw std::runtime_error("Corrupt nativeType.");
    }

    if (memo != -1) {
        // if we had to add this to the memo early, it's because it was a
        // class type that might be able to find itself through its function
        // definitions. In this case, we create a Forward to represent it, and then
        // define the forward.
        if (addedToMemoEarly) {
            addedToMemoEarly->define(resultType);
            b.updateCachedPointer(memo, resultType);
        } else {
            b.addCachedPointer(memo, resultType);
        }
    }

    return resultType;
}

template<class Factory_Fn, class SetItem_Fn>
inline PyObject* PythonSerializationContext::deserializeIndexable(
                DeserializationBuffer& b,
                size_t inWireType,
                Factory_Fn factory_fn,
                SetItem_Fn set_item_and_steal_ref_fn,
                int64_t memo
            ) const {
    PyEnsureGilAcquired acquireTheGil;

    PyObjectHolder res;
    int64_t count = -1;

    b.consumeCompoundMessageWithImpliedFieldNumbers(inWireType, [&](size_t fieldNumber, size_t wireType) {
        if (fieldNumber == 0) {
            assertWireTypesEqual(wireType, WireType::VARINT);
            count = b.readUnsignedVarint();

            res.steal(factory_fn(count));

            if (memo != -1) {
                b.addCachedPyObj(memo, incref(res));
            }
        } else {
            if (!res) {
                throw std::runtime_error("Corrupt stream. no result defined.");
            }
            PyObjectHolder elt;
            elt.steal(deserializePythonObject(b, wireType));
            if (fieldNumber < 1 || fieldNumber > count) {
                throw std::runtime_error("Corrupt record number.");
            }

            set_item_and_steal_ref_fn((PyObject*)res, fieldNumber - 1, elt.extract());
        }
    });

    return incref(res);
}

void PythonSerializationContext::serializeIterable(PyObject* o, SerializationBuffer& b, size_t fieldNumber) const {
    PyEnsureGilAcquired acquireTheGil;

    uint32_t id;
    bool isNew;
    std::tie(id, isNew) = b.cachePointer(o);

    b.writeUnsignedVarintObject(FieldNumbers::MEMO, id);

    if (!isNew) {
        return;
    }

    b.writeBeginCompound(fieldNumber);

    b.writeUnsignedVarintObject(0, PyObject_Length(o));

    PyObject* iter = PyObject_GetIter(o);
    if (!iter) {
        return;
    }
    PyObject* item = PyIter_Next(iter);
    while (item) {
        serializePythonObject(item, b, 0);
        decref(item);
        item = PyIter_Next(iter);
    }
    decref(iter);

    b.writeEndCompound();
}

template<class Factory_Fn, class AddItem_Fn, class Clear_Fn>
inline PyObject* PythonSerializationContext::deserializeIterable(
                DeserializationBuffer &b,
                size_t inWireType,
                Factory_Fn factory_fn,
                AddItem_Fn add_item_fn,
                Clear_Fn clear_fn,
                int64_t memo
            ) const {
    PyEnsureGilAcquired acquireTheGil;

    PyObjectHolder res;

    b.consumeCompoundMessageWithImpliedFieldNumbers(inWireType, [&](size_t fieldNumber, size_t wireType) {
        if (fieldNumber == 0) {
            assertWireTypesEqual(wireType, WireType::VARINT);
            //drop the size value on the floor since we don't need it.
            b.readUnsignedVarint();

            res.steal(factory_fn(NULL));

            if (memo != -1) {
                b.addCachedPyObj(memo, incref(res));
            }
        } else {
            PyObjectHolder elt;

            elt.steal(deserializePythonObject(b, wireType));

            if (add_item_fn((PyObject*)res, (PyObject*)elt) < 0) {
                throw PythonExceptionSet();
            }
        }
    });

    return incref(res);
}

void PythonSerializationContext::serializePyList(PyObject* o, SerializationBuffer& b) const {
    serializeIterable(o, b, FieldNumbers::LIST);
}

PyObject* PythonSerializationContext::deserializePyList(DeserializationBuffer& b, size_t wireType, int64_t memo) const {
    return deserializeIndexable(b, wireType, PyList_New, PyList_Set_Item_No_Checks, memo);
}

void PythonSerializationContext::serializePyTuple(PyObject* o, SerializationBuffer& b) const {
    serializeIterable(o, b, FieldNumbers::TUPLE);
}

PyObject* PythonSerializationContext::deserializePyTuple(DeserializationBuffer& b, size_t wireType, int64_t memo) const {
    return deserializeIndexable(b, wireType, PyTuple_New, PyTuple_Set_Item_No_Checks, memo);
}

void PythonSerializationContext::serializePySet(PyObject* o, SerializationBuffer& b) const {
    serializeIterable(o, b, FieldNumbers::SET);
}

PyObject* PythonSerializationContext::deserializePySet(DeserializationBuffer &b, size_t wireType, int64_t memo) const {
    return deserializeIterable(b, wireType, PySet_New, PySet_Add, PySet_Clear, memo);
}

std::shared_ptr<ByteBuffer> PythonSerializationContext::compressOrDecompress(uint8_t* begin, uint8_t* end, bool compress) const {
    assertHoldingTheGil();

    if (!mContextObj) {
        return std::shared_ptr<ByteBuffer>(new RangeByteBuffer(begin,end));
    }

    PyObjectStealer pyBytes(
        PyBytes_FromStringAndSize((const char*)begin, end-begin)
        );

    PyObjectStealer outBytes(
        PyObject_CallMethod(
            mContextObj,
            compress ? "compress" : "decompress",
            "O",
            (PyObject*)pyBytes //without this cast, the actual "Stealer" object gets passed
                               //because this is a C varargs function and it doesn't know
                               //that the intended type is PyObject*.
            )
        );

    if (!outBytes) {
        throw PythonExceptionSet();
    }

    if (!PyBytes_Check(outBytes)) {
        PyErr_Format(PyExc_TypeError,
            compress ?
                    "'compress' method didn't return bytes object."
                :   "'decompress' method didn't return bytes object."
            );
        throw PythonExceptionSet();
    }

    return std::shared_ptr<ByteBuffer>(new PyBytesByteBuffer(outBytes));
}

std::shared_ptr<ByteBuffer> PythonSerializationContext::compress(uint8_t* begin, uint8_t* end) const {
    return compressOrDecompress(begin, end, true);
}

std::shared_ptr<ByteBuffer> PythonSerializationContext::decompress(uint8_t* begin, uint8_t* end) const {
    return compressOrDecompress(begin, end, false);
}
