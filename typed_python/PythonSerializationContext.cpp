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

        serializeNativeTypeInCompound(t, b, 0);

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
            serializePyDict(o, b);
        } else
        if (PyType_Check(o)) {
            Type* nativeType = PyInstance::extractTypeFrom((PyTypeObject*)o);

            if (nativeType) {
                serializeNativeType(nativeType, b);
            } else {
                serializePythonObjectNamedOrAsObj(o, b);
            }
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
                result = incref((PyObject*)PyInstance::typeObj(deserializeNativeType(b, wireType, memo)));
            } else if (fieldNumber == FieldNumbers::OBJECT_NAME) {
                result = deserializePythonObjectFromName(b, wireType, memo);
            } else if (fieldNumber == FieldNumbers::OBJECT_REPRESENTATION) {
                result = deserializePythonObjectFromRepresentation(b, wireType, memo);
            } else if (fieldNumber == FieldNumbers::OBJECT_TYPEANDDICT) {
                result = deserializePythonObjectFromTypeAndDict(b, wireType, memo);
            } else {
                throw std::runtime_error("Unknown field number deserializing python object.");
            }
        }
    });

    if (!result) {
        throw std::runtime_error("Corrupt state: neither a memo nor a result.");
    }

    return result;
}

void PythonSerializationContext::serializePyDict(PyObject* o, SerializationBuffer& b) const {
    PyEnsureGilAcquired acquireTheGil;

    static Type* anyPyObjType = PythonObjectOfType::AnyPyObject();

    uint32_t id;
    bool isNew;
    std::tie(id, isNew) = b.cachePointer(o, anyPyObjType);

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
        static Type* anyPyObjType = PythonObjectOfType::AnyPyObject();

        b.addCachedPointer(memo, incref(res), anyPyObjType);
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
        static Type* anyPyObjType = PythonObjectOfType::AnyPyObject();

        b.addCachedPointer(memo, incref(res), anyPyObjType);
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

    static Type* anyPyObjType = PythonObjectOfType::AnyPyObject();

    uint32_t id;
    bool isNew;
    std::tie(id, isNew) = b.cachePointer(o, anyPyObjType);

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
        throw std::runtime_error(std::string("Object of type ") + o->ob_type->tp_name + " had no dict.");
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
        static Type* anyPyObjType = PythonObjectOfType::AnyPyObject();

        b.addCachedPointer(memo, incref(result), anyPyObjType);
    }

    return result;
}

PyObject* PythonSerializationContext::deserializePythonObjectFromRepresentation(DeserializationBuffer& b, size_t inWireType, int64_t memo) const {
    PyEnsureGilAcquired acquireTheGil;

    PyObjectHolder factory, factoryArgs;
    PyObjectHolder value;
    PyObjectHolder state;

    static Type* anyPyObjType = PythonObjectOfType::AnyPyObject();

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
                b.addCachedPointer(memo, incref(value), anyPyObjType);
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

    static Type* anyPyObjType = PythonObjectOfType::AnyPyObject();

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
                b.addCachedPointer(memo, incref(value), anyPyObjType);
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

void PythonSerializationContext::serializeNativeTypeInCompound(Type* nativeType, SerializationBuffer& b, size_t fieldNumber) const {
    b.writeBeginCompound(fieldNumber);
    serializeNativeType(nativeType, b);
    b.writeEndCompound();
}

void PythonSerializationContext::serializeNativeType(
            Type* nativeType,
            SerializationBuffer& b
            ) const {
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

    b.writeBeginCompound(FieldNumbers::NATIVE_TYPE);
    b.writeUnsignedVarintObject(0, nativeType->getTypeCategory());

    if (nativeType->getTypeCategory() == Type::TypeCategory::catInt8 ||
            nativeType->getTypeCategory() == Type::TypeCategory::catInt16 ||
            nativeType->getTypeCategory() == Type::TypeCategory::catInt32 ||
            nativeType->getTypeCategory() == Type::TypeCategory::catInt64 ||
            nativeType->getTypeCategory() == Type::TypeCategory::catUInt8 ||
            nativeType->getTypeCategory() == Type::TypeCategory::catUInt16 ||
            nativeType->getTypeCategory() == Type::TypeCategory::catUInt32 ||
            nativeType->getTypeCategory() == Type::TypeCategory::catUInt64 ||
            nativeType->getTypeCategory() == Type::TypeCategory::catFloat32 ||
            nativeType->getTypeCategory() == Type::TypeCategory::catFloat64 ||
            nativeType->getTypeCategory() == Type::TypeCategory::catNone ||
            nativeType->getTypeCategory() == Type::TypeCategory::catBytes ||
            nativeType->getTypeCategory() == Type::TypeCategory::catString ||
            nativeType->getTypeCategory() == Type::TypeCategory::catBool
            ) {
        //do nothing
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catConcreteAlternative) {
        serializeNativeTypeInCompound(nativeType->getBaseType(), b, 1);
        b.writeUnsignedVarintObject(2, ((ConcreteAlternative*)nativeType)->which());
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catSet) {
        serializeNativeTypeInCompound(((SetType*)nativeType)->keyType(), b, 1);
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catConstDict) {
        serializeNativeTypeInCompound(((ConstDictType*)nativeType)->keyType(), b, 1);
        serializeNativeTypeInCompound(((ConstDictType*)nativeType)->valueType(), b, 2);
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catDict) {
        serializeNativeTypeInCompound(((DictType*)nativeType)->keyType(), b, 1);
        serializeNativeTypeInCompound(((DictType*)nativeType)->valueType(), b, 2);
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catTupleOf) {
        serializeNativeTypeInCompound(((TupleOfType*)nativeType)->getEltType(), b, 1);
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catListOf) {
        serializeNativeTypeInCompound(((ListOfType*)nativeType)->getEltType(), b, 2);
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catTuple) {
        size_t index = 1;
        for (auto t: ((CompositeType*)nativeType)->getTypes()) {
            serializeNativeTypeInCompound(t, b, index++);
        }
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catNamedTuple) {
        size_t index = 1;
        for (long k = 0; k < ((CompositeType*)nativeType)->getTypes().size(); k++) {
            b.writeStringObject(index++, ((CompositeType*)nativeType)->getNames()[k]);
            serializeNativeTypeInCompound(((CompositeType*)nativeType)->getTypes()[k], b, index++);
        }
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catOneOf) {
        size_t index = 1;
        for (auto t: ((OneOfType*)nativeType)->getTypes()) {
            serializeNativeTypeInCompound(t, b, index++);
        }
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catPythonObjectOfType) {
        serializePythonObject((PyObject*)((PythonObjectOfType*)nativeType)->pyType(), b, 1);
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catValue) {
        b.writeBeginCompound(1);

        Instance i = ((Value*)nativeType)->value();
        serializeNativeTypeInCompound(i.type(), b, 0);

        PyEnsureGilReleased releaseTheGil;
        i.type()->serialize(i.data(), b, 1);
        b.writeEndCompound();
    } else {
        throw std::runtime_error("Can't serialize native type " + nativeType->name() + " if its unnamed.");
    }

    b.writeEndCompound();
}

Type* PythonSerializationContext::deserializePythonObjectExpectingNativeType(DeserializationBuffer& b, size_t wireType) const {
    PyEnsureGilAcquired acquireTheGil;

    PyObject* res = deserializePythonObject(b, wireType);

    Type* nativeType = PyInstance::extractTypeFrom((PyTypeObject*)res);

    if (!nativeType) {
        throw std::runtime_error("Expected a native type but didn't get one.");
    }

    if (!nativeType->resolved()) {
        throw std::runtime_error("Can't deserialize into an unresolved type " + nativeType->name());
    }
    return nativeType;
}

Instance PythonSerializationContext::deserializeNativeInstance(DeserializationBuffer& b, size_t inWireType) const {
    PyEnsureGilAcquired acquireTheGil;

    Instance result;
    Type* type = nullptr;

    b.consumeCompoundMessage(inWireType, [&](size_t fieldNumber, size_t wireType) {
        if (fieldNumber == 0) {
            type = deserializePythonObjectExpectingNativeType(b, wireType);
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

Type* PythonSerializationContext::deserializeNativeType(DeserializationBuffer& b, size_t inWireType, int64_t memo) const {
    PyEnsureGilAcquired acquireTheGil;

    int category = -1;

    static Type* anyPyObjType = PythonObjectOfType::AnyPyObject();

    std::vector<Type*> types;
    std::vector<std::string> names;
    int64_t whichIndex = -1;
    PyObjectHolder obj;
    Instance instance;

    b.consumeCompoundMessage(inWireType, [&](size_t fieldNumber, size_t wireType) {
        if (fieldNumber == 0) {
            assertWireTypesEqual(wireType, WireType::VARINT);
            category = b.readUnsignedVarint();
        } else {
            if (category == -1) {
                throw std::runtime_error("Corrupt native type found.");
            }
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
                types.push_back(deserializePythonObjectExpectingNativeType(b, wireType));
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

    if (category == Type::TypeCategory::catUInt8) {
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
        b.addCachedPointer(memo, incref((PyObject*)PyInstance::typeObj(resultType)), anyPyObjType);
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

    static Type* anyPyObjType = PythonObjectOfType::AnyPyObject();

    PyObjectHolder res;
    int64_t count = -1;

    b.consumeCompoundMessageWithImpliedFieldNumbers(inWireType, [&](size_t fieldNumber, size_t wireType) {
        if (fieldNumber == 0) {
            assertWireTypesEqual(wireType, WireType::VARINT);
            count = b.readUnsignedVarint();

            res.steal(factory_fn(count));

            if (memo != -1) {
                b.addCachedPointer(memo, incref(res), anyPyObjType);
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

    static Type* anyPyObjType = PythonObjectOfType::AnyPyObject();

    uint32_t id;
    bool isNew;
    std::tie(id, isNew) = b.cachePointer(o, anyPyObjType);

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

    static Type* anyPyObjType = PythonObjectOfType::AnyPyObject();

    PyObjectHolder res;

    b.consumeCompoundMessageWithImpliedFieldNumbers(inWireType, [&](size_t fieldNumber, size_t wireType) {
        if (fieldNumber == 0) {
            assertWireTypesEqual(wireType, WireType::VARINT);
            //drop the size value on the floor since we don't need it.
            b.readUnsignedVarint();

            res.steal(factory_fn(NULL));

            if (memo != -1) {
                b.addCachedPointer(memo, incref(res), anyPyObjType);
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

