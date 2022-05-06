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
#include "MutuallyRecursiveTypeGroup.hpp"

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
        } else if (fieldNumber == FieldNumbers::RECURSIVE_OBJECT) {
            result = deserializeRecursiveObject(b, wireType);
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
                Type* nativeType = deserializeNativeType(b, wireType);
                result = incref((PyObject*)PyInstance::typeObj(nativeType));
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

    if (memo != -1) {
        if (!b.lookupCachedPointer(memo)) {
            if (result) {
                throw std::runtime_error("FAILED to populate memo=" + format(memo));
            } else {
                throw std::runtime_error("FAILED: no result, and we had memo=" + format(memo));
            }
        }
    }

    if (!result) {
        throw std::runtime_error("FAILED: no result populated. memo=" + format(memo));
    }

    return result;
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

PyObject* PythonSerializationContext::deserializePyFrozenSet(DeserializationBuffer &b, size_t inWireType, int64_t memo) const {
    PyEnsureGilAcquired acquireTheGil;

    PyObject* res = PyFrozenSet_New(NULL);
    if (!res) {
        PyErr_PrintEx(1);
        throw std::runtime_error(
            std::string("Failed to allocate memory for frozen set deserialization")
        );
    }

    if (memo != -1) {
        b.addCachedPyObj(memo, incref(res));
    }

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

    return res;
}

PyObject* PythonSerializationContext::deserializePythonObjectFromName(DeserializationBuffer& b, size_t wireType, int64_t memo) const {
    PyEnsureGilAcquired acquireTheGil;

    assertWireTypesEqual(wireType, WireType::BYTES);

    std::string name = b.readStringObject();

    PyObject* result = PyObject_CallMethod(mContextObj, "objectFromName", "s", name.c_str());

    if (!result) {
        throw PythonExceptionSet();
    }
    if (result == Py_None){
        throw std::runtime_error("Failed to deserialize Type '" + name + "'");
    }

    if (result == Py_None) {
        throw std::runtime_error(
            "Object named " + name + " doesn't exist in this codebase. Perhaps you "
            "are deserializing an object from an earlier codebase?"
        );
    }

    if (memo != -1) {
        b.addCachedPyObj(memo, incref(result));
    }

    return result;
}

PyObject* PythonSerializationContext::deserializePythonObjectFromRepresentation(DeserializationBuffer& b, size_t inWireType, int64_t memo) const {
    PyEnsureGilAcquired acquireTheGil;

    PyObjectHolder value;

    std::vector<PyObjectHolder> holders;

    b.consumeCompoundMessage(inWireType, [&](size_t fieldNumber, size_t wireType) {
        if (fieldNumber != holders.size()) {
            throw std::runtime_error("Corrupt stream: pythonObjectRepresentation");
        } else {
            PyObjectHolder holder;
            holder.steal(deserializePythonObject(b, wireType));

            if (!holder) {
                throw PythonExceptionSet();
            }

            holders.push_back(holder);
        }

        if (holders.size() == 2) {
            value.steal(PyObject_Call((PyObject*)holders[0], (PyObject*)holders[1], NULL));

            if (!value) {
                throw PythonExceptionSet();
            }

            if (memo != -1) {
                b.addCachedPyObj(memo, incref(value));
            }
        }
    });

    if (holders.size() < 2 || holders.size() > 6) {
        throw std::runtime_error("Corrupt stream: pythonObjectRepresentation should be 2-6 items");
    }

    if (!value) {
        throw std::runtime_error("Invalid representation.");
    }

    PyObjectHolder res;

    static PyObject* methodName = PyUnicode_FromString(
        "setInstanceStateFromRepresentation"
    );

    res.steal(
        PyObject_CallMethodObjArgs(
            mContextObj,
            methodName,
            (PyObject*)value,
            (holders.size() > 2 ? (PyObject*)holders[2] : (PyObject*)nullptr),
            (holders.size() > 3 ? (PyObject*)holders[3] : (PyObject*)nullptr),
            (holders.size() > 4 ? (PyObject*)holders[4] : (PyObject*)nullptr),
            (holders.size() > 5 ? (PyObject*)holders[5] : (PyObject*)nullptr),
            (PyObject*)nullptr
        )
    );

    if (!res) {
        throw PythonExceptionSet();
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

void PythonSerializationContext::deserializeAlternativeMembers(
    std::vector<std::pair<std::string, NamedTuple*> >& members,
    DeserializationBuffer& b,
    int inWireType
) const {
    b.consumeCompoundMessage(inWireType, [&](size_t fieldNumber, size_t wireType) {
        std::string name;
        Type* t = nullptr;

        b.consumeCompoundMessage(wireType, [&](size_t fieldNumber2, size_t wireType2) {
            if (fieldNumber2 == 0) {
                name = b.readStringObject();
            } else if (fieldNumber2 == 1) {
                t = deserializeNativeType(b, wireType2);
            } else {
                throw std::runtime_error("Corrupt Alternative member definition when deserializing: " + format(fieldNumber2));
            }
        });

        if (!t || name.size() == 0) {
            throw std::runtime_error("Corrupt Alternative member when deserializing.");
        }

        if (t->getTypeCategory() != Type::TypeCategory::catNamedTuple) {
            throw std::runtime_error(
                "Corrupt Alternative member definition when deserializing: category is " +
                Type::categoryToString(t->getTypeCategory())
            );
        }

        members.push_back(std::pair<std::string, NamedTuple*>(name, (NamedTuple*)t));
    });
}

void PythonSerializationContext::deserializeClassMembers(
    std::vector<MemberDefinition>& members,
    DeserializationBuffer& b,
    int inWireType
) const {
    b.consumeCompoundMessage(inWireType, [&](size_t fieldNumber, size_t wireType) {
        std::string name;
        Type* t = nullptr;
        Instance i;
        bool isNonempty = false;

        b.consumeCompoundMessage(wireType, [&](size_t fieldNumber2, size_t wireType2) {
            if (fieldNumber2 == 0) {
                name = b.readStringObject();
            } else if (fieldNumber2 == 1) {
                t = deserializeNativeType(b, wireType2);
            } else if (fieldNumber2 == 2) {
                i = deserializeNativeInstance(b, wireType2);
            } else if (fieldNumber2 == 3) {
                isNonempty = b.readUnsignedVarint();
            } else {
                throw std::runtime_error("Corrupt ClassMember definition when deserializing class.");
            }
        });

        if (!t || name.size() == 0) {
            throw std::runtime_error("Corrupt ClassMember definition when deserializing class.");
        }

        members.push_back(MemberDefinition(name, t, i, isNonempty));
    });
}

void PythonSerializationContext::deserializeClassFunDict(
    std::string className,
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
                throw std::runtime_error("Corrupt function definition when deserializing class "
                    + className
                    + ": bad field " + format(fieldNumber2));
            }
        });

        if (!fun) {
            throw std::runtime_error("Corrupt function definition when deserializing class "
                + className
                + ": no function");
        }

        if (name.size() == 0) {
            throw std::runtime_error("Corrupt function definition when deserializing class "
                + className
                + ": no name");
        }

        if (fun->getTypeCategory() == Type::TypeCategory::catForward && !((Forward*)fun)->getTarget()) {
            throw std::runtime_error(
                "Corrupt function definition when deserializing class "
                + className + "." + name
                + ": function is an untargeted forward."
            );
        }

        if (fun->getTypeCategory() != Type::TypeCategory::catFunction) {
            throw std::runtime_error(
                "Corrupt function definition when deserializing class fun "
                + name + ": not a function: "
                + fun->name() + ". category=" + Type::categoryToString(fun->getTypeCategory())
            );
        }

        dict[name] = (Function*)fun;
    });
}

void PythonSerializationContext::deserializeClassClassMemberDict(
    std::map<std::string, PyObjectHolder>& dict,
    DeserializationBuffer& b,
    int inWireType
) const {
    b.consumeCompoundMessage(inWireType, [&](size_t fieldNumber, size_t wireType) {
        std::string name;
        PyObjectHolder o;

        b.consumeCompoundMessage(wireType, [&](size_t fieldNumber2, size_t wireType2) {
            if (fieldNumber2 == 0) {
                name = b.readStringObject();
            } else if (fieldNumber2 == 1) {
                o.steal(deserializePythonObject(b, wireType2));
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

PyObject* prepareBlankInstanceOfType(MutuallyRecursiveTypeGroup* outGroup, PyTypeObject* type, int indexInGroup) {
    if (type == &PyCell_Type) {
        return PyCell_New(nullptr);
    }

    Type* nt;
    PyObjectHolder obj;

    if ((nt = PyInstance::extractTypeFrom(type))) {
        // this is going to be an instance
        obj.steal(
            PyInstance::initialize(
                nt,
                [&](instance_ptr) { /* do nothing. we'll initialize this later.*/ })
        );
    } else {
        // this is a regular python type
        static PyObject* emptyTuple = PyTuple_Pack(0);

        obj.steal(type->tp_new(type, emptyTuple, NULL));

        if (!obj) {
            throw std::runtime_error(
                "tp_new for " + std::string(type->tp_name) + " threw an exception"
            );
        }
    }

    outGroup->setIndexToObject(indexInGroup, (PyObject*)obj);
    return obj;
}

MutuallyRecursiveTypeGroup* PythonSerializationContext::deserializeMutuallyRecursiveTypeGroup(
        DeserializationBuffer& b, size_t inWireType
) const {
    PyEnsureGilAcquired acquireTheGil;

    MutuallyRecursiveTypeGroup* outGroup = nullptr;

    bool hasGroupHash = false;
    ShaHash groupHash;
    int32_t memo = -1;

    bool actuallyBuildGroup = false;

    // map from the index in the group to the index of the type object in the group
    std::map<int32_t, int32_t> indexOfObjToIndexOfType;

    std::map<int32_t, Type*> indicesOfNativeTypes;
    std::map<int32_t, Type::TypeCategory> indicesOfNativeTypeCategories;
    std::map<int32_t, PyObject*> indicesWrittenAsObjectAndRep;

    std::map<int32_t, PyObject*> indicesWithSerializedBodies;
    std::map<int32_t, PyTypeObject*> indicesWithSerializedBodiesTypes;

    bool cameFromMemo = false;

    b.consumeCompoundMessage(inWireType, [&](size_t fieldNumber, size_t wireType) {
        if (fieldNumber == FieldNumbers::MEMO) {
            memo = b.readUnsignedVarint();

            outGroup = (MutuallyRecursiveTypeGroup*)b.lookupCachedPointer(memo);

            if (outGroup) {
                cameFromMemo = true;
            }
        } else
        if (fieldNumber == 1) {
            assertWireTypesEqual(wireType, WireType::BYTES);
            groupHash = ShaHash::fromDigest(b.readStringObject());
            hasGroupHash = true;

            if (mInternalizeTypeGroups) {
                groupHash = MutuallyRecursiveTypeGroup::sourceToDestHashLookup(groupHash);
                // check if a group exists with this hash. If so, we'll consume our objects
                // but we'll just drop them on the floor because we don't really need them.
                outGroup = MutuallyRecursiveTypeGroup::getGroupFromHash(groupHash);
            }

            if (memo != -1 && outGroup) {
                b.addCachedPointer(memo, (void*)outGroup, nullptr);
            }
        } else
        if (fieldNumber == 2) {
            if (!hasGroupHash) {
                throw std::runtime_error("Corrupt MutuallyRecursiveTypeGroup: no hash");
            }

            if (!outGroup) {
                outGroup = new MutuallyRecursiveTypeGroup();

                if (memo != -1) {
                    b.addCachedPointer(memo, (void*)outGroup, nullptr);
                }

                actuallyBuildGroup = true;
            }

            // consume each sub object
            b.consumeCompoundMessage(wireType, [&](size_t indexInGroup, size_t subWireType) {
                int32_t kind = -1;
                PyObjectHolder rep0;
                PyObjectHolder rep1;
                bool setSomething = false;

                b.consumeCompoundMessage(subWireType, [&](size_t fieldInIndex, size_t subSubWireType) {
                    if (fieldInIndex == 0 && kind == -1) {
                        assertWireTypesEqual(subSubWireType, WireType::VARINT);
                        kind = b.readUnsignedVarint();
                    } else
                    if (fieldInIndex == 1 && kind == 0) {
                        // this is a native type and this value is the name
                        assertWireTypesEqual(subSubWireType, WireType::BYTES);
                        std::string fwdName = b.readStringObject();

                        if (actuallyBuildGroup) {
                            Forward* f = Forward::Make(fwdName);

                            outGroup->setIndexToObject(indexInGroup, f);
                            indicesOfNativeTypes[indexInGroup] = f;
                        } else {
                            indicesOfNativeTypes[indexInGroup] = nullptr;
                        }
                        setSomething = true;
                    } else
                    if (fieldInIndex == 2 && kind == 0) {
                        assertWireTypesEqual(subSubWireType, WireType::VARINT);
                        indicesOfNativeTypeCategories[indexInGroup] = Type::TypeCategory(
                            b.readUnsignedVarint()
                        );
                    } else
                    if (fieldInIndex == 1 && kind == 1) {
                        // this is the now-deprecated field code for sending a named object.
                        // now if we send a named object, we just send one and use it to find
                        // the MRTG.
                        throw std::runtime_error(
                            "Deprecated object-by-name in MRTG. This stream was written by an earlier "
                            "and not compatible version of typed_python."
                        );
                    } else
                    if (fieldInIndex == 1 && kind == 2) {
                        // this is a representation object
                        rep0.steal(deserializePythonObject(b, subSubWireType));

                        if (!rep0) {
                            throw PythonExceptionSet();
                        }
                    } else
                    if (fieldInIndex == 2 && kind == 2) {
                        // this is a representation object
                        rep1.steal(deserializePythonObject(b, subSubWireType));

                        if (!rep1) {
                            throw PythonExceptionSet();
                        }

                        if (actuallyBuildGroup) {
                            PyObjectStealer obj(PyObject_Call((PyObject*)rep0, (PyObject*)rep1, NULL));

                            if (!obj) {
                                throw PythonExceptionSet();
                            }

                            outGroup->setIndexToObject(indexInGroup, (PyObject*)obj);
                            indicesWrittenAsObjectAndRep[indexInGroup] = (PyObject*)obj;
                        } else {
                            indicesWrittenAsObjectAndRep[indexInGroup] = nullptr;
                        }
                        setSomething = true;
                    } else
                    if (fieldInIndex == 1 && kind == 3) {
                        // this is the type
                        rep0.steal(deserializePythonObject(b, subSubWireType));

                        if (!rep0) {
                            throw PythonExceptionSet();
                        }

                        if (!PyType_Check((PyObject*)rep0)) {
                            throw std::runtime_error("Not a type object");
                        }

                        indicesWithSerializedBodiesTypes[indexInGroup] = (PyTypeObject*)rep0;

                        if (actuallyBuildGroup) {
                            PyTypeObject* type = (PyTypeObject*)(PyObject*)rep0;

                            indicesWithSerializedBodies[indexInGroup] = (
                                prepareBlankInstanceOfType(outGroup, type, indexInGroup)
                            );
                            outGroup->setIndexToObject(indexInGroup, indicesWithSerializedBodies[indexInGroup]);
                        } else {
                            indicesWithSerializedBodies[indexInGroup] = nullptr;
                        }

                        setSomething = true;
                    } else
                    if (fieldInIndex == 1 && kind == 4) {
                        // this is the index of the type in this recursive type group
                        assertWireTypesEqual(subSubWireType, WireType::VARINT);
                        setSomething = true;
                        indexOfObjToIndexOfType[indexInGroup] = b.readUnsignedVarint();
                    } else
                    if (fieldInIndex == 1 && kind == 5) {
                        assertWireTypesEqual(subSubWireType, WireType::VARINT);
                        indicesWithSerializedBodiesTypes[indexInGroup] = &PyTuple_Type;
                        int tupSize = b.readUnsignedVarint();
                        if (actuallyBuildGroup) {
                            indicesWithSerializedBodies[indexInGroup] = PyTuple_New(tupSize);
                            outGroup->setIndexToObject(indexInGroup, indicesWithSerializedBodies[indexInGroup]);
                        } else {
                            indicesWithSerializedBodies[indexInGroup] = nullptr;
                        }
                        setSomething = true;
                    } else {
                        throw std::runtime_error("Invalid object kind/field combination");
                    }
                });

                if (!setSomething) {
                    throw std::runtime_error("corrupt MutuallyRecursiveTypeGroup item");
                }
            });
        } else
        if (fieldNumber == 3) {
            b.consumeCompoundMessage(wireType, [&](size_t indexInGroup, size_t subWireType) {

                if (indexOfObjToIndexOfType.find(indexInGroup) != indexOfObjToIndexOfType.end()) {
                    // this instance doesn't have a body yet because it was an instance of an object defined
                    // within this type group. So we have to provide one.
                    size_t indexOfType = indexOfObjToIndexOfType[indexInGroup];

                    auto it = outGroup->getIndexToObject().find(indexOfType);
                    if (it == outGroup->getIndexToObject().end()) {
                        throw std::runtime_error("Invalid mutually recursive type group: internal type not defined yet");
                    }

                    TypeOrPyobj objT = it->second;

                    PyTypeObject* typeObj;

                    if (objT.pyobj()) {
                        if (!PyType_Check(objT.pyobj())) {
                            throw std::runtime_error("Invalid mutually recursive type group: internal type is not a type");
                        }
                        typeObj = (PyTypeObject*)objT.pyobj();
                    } else {
                        typeObj = PyInstance::typeObj(objT.type());
                    }

                    indicesWithSerializedBodiesTypes[indexInGroup] = typeObj;

                    if (actuallyBuildGroup) {
                        indicesWithSerializedBodies[indexInGroup] = (
                            prepareBlankInstanceOfType(outGroup, typeObj, indexInGroup)
                        );
                        outGroup->setIndexToObject(indexInGroup, indicesWithSerializedBodies[indexInGroup]);
                    }
                }

                if (indicesWrittenAsObjectAndRep.find(indexInGroup) != indicesWrittenAsObjectAndRep.end()) {
                    // the remaining parts of the __reduce__ call are encoded here
                    PyObjectStealer trailingRepresentationParts(deserializePythonObject(b, subWireType));
                    if (!trailingRepresentationParts) {
                        throw PythonExceptionSet();
                    }

                    if (actuallyBuildGroup) {
                        if (!indicesWrittenAsObjectAndRep[indexInGroup]) {
                            throw std::runtime_error("Somehow indicesWrittenAsObjectAndRep[indexInGroup] is empty");
                        }

                        PyObjectHolder res;

                        if (!PyTuple_Check(trailingRepresentationParts) ||
                                PyTuple_Size(trailingRepresentationParts) > 4) {
                            throw std::runtime_error(
                                "trailingRepresentationParts is not a tuple with <= 4 elements"
                            );
                        }

                        // this just seems so inelegant...
                        int argCount = PyTuple_Size(trailingRepresentationParts);

                        static PyObject* methodName = PyUnicode_FromString(
                            "setInstanceStateFromRepresentation"
                        );

                        res.steal(
                            PyObject_CallMethodObjArgs(
                                mContextObj,
                                methodName,
                                (PyObject*)indicesWrittenAsObjectAndRep[indexInGroup],
                                argCount > 0 ? PyTuple_GetItem((PyObject*)trailingRepresentationParts, 0) : (PyObject*)nullptr,
                                argCount > 1 ? PyTuple_GetItem((PyObject*)trailingRepresentationParts, 1) : (PyObject*)nullptr,
                                argCount > 2 ? PyTuple_GetItem((PyObject*)trailingRepresentationParts, 2) : (PyObject*)nullptr,
                                argCount > 3 ? PyTuple_GetItem((PyObject*)trailingRepresentationParts, 3) : (PyObject*)nullptr,
                                (PyObject*)nullptr
                            )
                        );

                        if (!res) {
                            throw PythonExceptionSet();
                        }
                    }
                } else
                if (indicesWithSerializedBodiesTypes.find(indexInGroup) != indicesWithSerializedBodiesTypes.end()) {
                    PyTypeObject* pyType = indicesWithSerializedBodiesTypes[indexInGroup];

                    Type* nt = PyInstance::extractTypeFrom(pyType);

                    if (nt) {
                        if (actuallyBuildGroup) {
                            nt->deserialize(
                                ((PyInstance*)indicesWithSerializedBodies[indexInGroup])->dataPtr(),
                                b,
                                subWireType
                            );
                        } else {
                            Instance i(nt, [&](instance_ptr toDrop) { nt->deserialize(toDrop, b, subWireType); });
                        }
                    } else {
                        PyObjectStealer state(deserializePythonObject(b, subWireType));
                        if (!state) {
                            throw PythonExceptionSet();
                        }
                        if (actuallyBuildGroup) {
                            PyObject* inst = indicesWithSerializedBodies[indexInGroup];

                            if (!inst) {
                                throw std::runtime_error("Somehow indicesWithSerializedBodies[indexInGroup] is empty");
                            }

                            if (PyCell_Check(inst)) {
                                PyCell_Set(inst, state);
                            } else
                            if (PyTuple_Check(inst)) {
                                if (!PyTuple_Check((PyObject*)state)) {
                                    throw std::runtime_error("Somehow we deserialized something other than a tuple");
                                }
                                if (PyTuple_Size((PyObject*)state) != PyTuple_Size(inst)) {
                                    throw std::runtime_error("Somehow our deserialized tuple had the wrong size");
                                }

                                for (long k = 0; k < PyTuple_Size((PyObject*)state); k++) {
                                    //recall that setitem steals a reference
                                    PyObject* inItem = PyTuple_GetItem((PyObject*)state, k);
                                    if (!inItem) {
                                        throw std::runtime_error(
                                            "Somehow our deserialized tuple  for " +
                                            format(indexInGroup) + " has a null value"
                                        );
                                    }

                                    PyTuple_SET_ITEM(inst, k, incref(inItem));
                                }
                            } else {
                                if (!PyDict_Check(state)) {
                                    throw std::runtime_error(
                                        "Expected object state to be a dict, not " +
                                        std::string(state->ob_type->tp_name) + " when inst is "
                                         + inst->ob_type->tp_name
                                    );
                                }

                                if (PyObject_GenericSetDict(inst, state, nullptr) == -1) {
                                    throw PythonExceptionSet();
                                }
                            }
                        }
                    }
                } else
                if (indicesOfNativeTypes.find(indexInGroup) != indicesOfNativeTypes.end()) {
                    if (indicesOfNativeTypeCategories[indexInGroup] == Type::TypeCategory::catForward) {
                        Type* t = deserializeNativeType(b, subWireType);
                        if (actuallyBuildGroup) {
                            ((Forward*)indicesOfNativeTypes[indexInGroup])->define(t);
                        }
                    } else {
                        Type* t = deserializeNativeTypeInner(b, subWireType, actuallyBuildGroup);

                        if (actuallyBuildGroup) {
                            if (!t) {
                                throw std::runtime_error("Somehow, deserializeNativeTypeInner didn't return a type.");
                            }

                            if (!indicesOfNativeTypes[indexInGroup]) {
                                throw std::runtime_error("indicesOfNativeTypes[indexInGroup] is somehow none");
                            }
                            if (!indicesOfNativeTypes[indexInGroup]->isForward()) {
                                throw std::runtime_error("indicesOfNativeTypes[indexInGroup] was already resolved");
                            }
                            ((Forward*)indicesOfNativeTypes[indexInGroup])->define(t);
                            indicesOfNativeTypes[indexInGroup] = t;

                            // update the mutually recursive group or we'll end up with
                            // downstream consumers actually pulling out the forwards
                            outGroup->setIndexToObject(indexInGroup, t);
                        }
                    }
                } else {
                    throw std::runtime_error("Corrupt MutuallyRecursiveTypeGroup group");
                }
            });
        } else
        if (fieldNumber == 4) {
            // fill out the globals of function objects. we have to deserialize these
            // in a separate final pass because they can have references to existing
            // native types, and we need those objects to be filled out fully.
            b.consumeCompoundMessage(wireType, [&](size_t indexInGroup, size_t subWireType) {
                b.consumeCompoundMessage(subWireType, [&](size_t overloadIx, size_t subSubWireType) {
                    PyObjectStealer overloadGlobals(deserializePythonObject(b, subSubWireType));

                    if (actuallyBuildGroup) {
                        if (!indicesOfNativeTypes[indexInGroup] ||
                                !indicesOfNativeTypes[indexInGroup]->isFunction()) {
                            throw std::runtime_error("Invalid function overload globals: not a function");
                        }

                        Function* f = (Function*)indicesOfNativeTypes[indexInGroup];
                        if (overloadIx >= f->getOverloads().size()) {
                            throw std::runtime_error(
                                "Invalid function overload globals: overload index out of bounds"
                            );
                        }

                        ((Function::Overload*)&f->getOverloads()[overloadIx])->setGlobals(overloadGlobals);
                    }
                });
            });

            // now that we've set function globals, we need to go over all the classes and
            // held classes and make sure the versions of the function types they're actually
            // using have the new definitions. Unfortunately, Overload objects are not pointers
            // (maybe they should be?) and so when we merge Overloads from base/child classes
            // they don't get their globals replaced with the appropriate values
            for (auto& indexAndNativeType: indicesOfNativeTypes) {
                Type* nt = indexAndNativeType.second;
                if (nt && nt->isHeldClass()) {
                    ((HeldClass*)nt)->mergeOwnFunctionsIntoInheritanceTree();
                }
            }
        } else
        if (fieldNumber == 5) {
            // field 5 indicates that this is a mutually recursive type group
            // identified by the name of an object inside of the group.
            std::string name = b.readStringObject();

            PyObjectStealer namedObj(PyObject_CallMethod(mContextObj, "objectFromName", "s", name.c_str()));

            if (!namedObj) {
                throw PythonExceptionSet();
            }

            if (namedObj == Py_None) {
                throw std::runtime_error(
                    "Named object " + name + " doesn't exist. This serialized data must "
                    "have come from a different version of the codebase."
                );
            }

            MutuallyRecursiveTypeGroup* group = nullptr;

            if (PyType_Check(namedObj)) {
                Type* typeArg = PyInstance::extractTypeFrom(namedObj);
                if (typeArg) {
                    group = typeArg->getRecursiveTypeGroup();
                }
            }

            if (!group) {
                group = (
                    MutuallyRecursiveTypeGroup::pyObjectGroupHeadAndIndex(namedObj).first
                );
            }

            if (!outGroup) {
                outGroup = group;
                if (memo != -1) {
                    b.addCachedPointer(memo, (void*)outGroup, nullptr);
                }
            }
        } else {
            throw std::runtime_error("Invalid subfield");
        }
    });

    if (!outGroup) {
        throw std::runtime_error("Somehow we didn't create a MutuallyRecursiveTypeGroup");
    }

    if (outGroup->getIndexToObject().size() == 0) {
        if (cameFromMemo) {
            throw std::runtime_error(
                "Somehow we accessed an empty memoized MutuallyRecursiveTypeGroup: \n"
                + outGroup->repr()
            );
        } else {
            throw std::runtime_error(
                "Somehow we deserialized an empty MutuallyRecursiveTypeGroup: \n"
                + outGroup->repr()
            );
        }
    }

    if (actuallyBuildGroup) {
        outGroup->computeHashAndInstall();

        if (outGroup->hash() != groupHash) {
            // make sure that next time we deserialize this group
            // we get the same one. Just because we have different internal
            // hashes doesn't mean we should duplicate the types.
            MutuallyRecursiveTypeGroup::installSourceToDestHashLookup(
                groupHash,
                outGroup->hash()
            );
        }
    }

    return outGroup;
}

Type* catToSimpleType(Type::TypeCategory category) {
    if (category == Type::TypeCategory::catUInt8) {
        return ::UInt8::Make();
    }
    else if (category == Type::TypeCategory::catUInt16) {
        return ::UInt16::Make();
    }
    else if (category == Type::TypeCategory::catUInt32) {
        return ::UInt32::Make();
    }
    else if (category == Type::TypeCategory::catUInt64) {
        return ::UInt64::Make();
    }
    else if (category == Type::TypeCategory::catInt8) {
        return ::Int8::Make();
    }
    else if (category == Type::TypeCategory::catInt16) {
        return ::Int16::Make();
    }
    else if (category == Type::TypeCategory::catInt32) {
        return ::Int32::Make();
    }
    else if (category == Type::TypeCategory::catInt64) {
        return ::Int64::Make();
    }
    else if (category == Type::TypeCategory::catEmbeddedMessage) {
        return ::EmbeddedMessageType::Make();
    }
    else if (category == Type::TypeCategory::catFloat32) {
        return ::Float32::Make();
    }
    else if (category == Type::TypeCategory::catFloat64) {
        return ::Float64::Make();
    }
    else if (category == Type::TypeCategory::catNone) {
        return ::NoneType::Make();
    }
    else if (category == Type::TypeCategory::catBytes) {
        return ::BytesType::Make();
    }
    else if (category == Type::TypeCategory::catString) {
        return ::StringType::Make();
    }
    else if (category == Type::TypeCategory::catPyCell) {
        return ::PyCellType::Make();
    }
    else if (category == Type::TypeCategory::catBool) {
        return ::Bool::Make();
    }

    throw std::runtime_error("Category " + Type::categoryToString(category) + " is not simple");
}

Type* PythonSerializationContext::deserializeNativeType(DeserializationBuffer& b, size_t inWireType) const {
    PyEnsureGilAcquired acquireTheGil;

    MutuallyRecursiveTypeGroup* group = nullptr;
    std::string name;
    PyObjectHolder typeFromRep;
    int32_t indexInGroup = -1;
    int32_t kind = -1;
    int32_t which = -1;
    int32_t category = -1;

    b.consumeCompoundMessage(inWireType, [&](size_t fieldNumber, size_t wireType) {
        if (fieldNumber == 0) {
            assertWireTypesEqual(wireType, WireType::VARINT);
            kind = b.readUnsignedVarint();
        } else
        if (kind == 0 && fieldNumber == 1) {
            assertWireTypesEqual(wireType, WireType::VARINT);
            category = b.readUnsignedVarint();
        } else
        if (kind == 1 && fieldNumber == 1) {
            assertWireTypesEqual(wireType, WireType::BYTES);
            name = b.readStringObject();
        } else
        if (kind == 1 && fieldNumber == 2) {
            assertWireTypesEqual(wireType, WireType::VARINT);
            which = b.readUnsignedVarint();
        } else
        if (kind == 2 && fieldNumber == 1) {
            assertWireTypesEqual(wireType, WireType::BYTES);
            name = b.readStringObject();
        } else
        if (kind == 3 && fieldNumber == 1) {
            typeFromRep.steal(deserializePythonObjectFromRepresentation(b, wireType, -1));
        } else
        if (kind == 4 && fieldNumber == 1) {
            group = deserializeMutuallyRecursiveTypeGroup(b, wireType);
        } else
        if (kind == 4 && fieldNumber == 2) {
            assertWireTypesEqual(wireType, WireType::VARINT);
            indexInGroup = b.readUnsignedVarint();
        } else {
            throw std::runtime_error("Invalid nativeType: kind/fieldNumber error.");
        }
    });

    if (kind == -1) {
        throw std::runtime_error("Invalid native type: no 'kind' field");
    }

    if (kind == 0) {
        if (category == -1) {
            throw std::runtime_error("Invalid inline type.");
        }
        return catToSimpleType(Type::TypeCategory(category));
    }

    if (kind == 1) {
        if (name.size() == 0) {
            throw std::runtime_error("Invalid inline named concrete alternative");
        }
        if (which == -1) {
            throw std::runtime_error("Invalid inline named concrete alternative: no index.");
        }

        PyObjectStealer namedObj(PyObject_CallMethod(mContextObj, "objectFromName", "s", name.c_str()));

        if (!namedObj) {
            throw PythonExceptionSet();
        }

        if (!PyType_Check(namedObj)) {
            throw std::runtime_error("Expected " + name + " to name an Alternative");
        }

        Type* altType = PyInstance::extractTypeFrom((PyTypeObject*)(PyObject*)namedObj);
        if (!altType || altType->getTypeCategory() != Type::TypeCategory::catAlternative) {
            throw std::runtime_error("Expected " + name + " to name an Alternative");
        }

        Alternative* alt = (Alternative*)altType;
        if (!alt->concreteSubtype(which)) {
            throw std::runtime_error("Invalid inline named concrete alternative: invalid index.");
        }

        Type* resultType = alt->concreteSubtype(which);

        if (resultType->getTypeCategory() != Type::TypeCategory::catConcreteAlternative) {
            throw std::runtime_error("concrete subtype should return a ConcreteAlternative");
        }

        return resultType;
    }

    if (kind == 2) {
        if (name.size() == 0) {
            throw std::runtime_error("Invalid inline named type");
        }

        PyObjectStealer namedObj(PyObject_CallMethod(mContextObj, "objectFromName", "s", name.c_str()));

        if (!namedObj) {
            throw PythonExceptionSet();
        }

        if (!PyType_Check(namedObj)) {
            throw std::runtime_error("Expected " + name + " to name a native type");
        }

        Type* nativeType = PyInstance::extractTypeFrom((PyTypeObject*)(PyObject*)namedObj);
        if (!nativeType) {
            nativeType = PythonObjectOfType::Make(namedObj);
        }

        if (nativeType->getTypeCategory() == Type::TypeCategory::catForward) {
            Type* target = ((Forward*)nativeType)->getTarget();
            if (target) {
                nativeType = target;
            }
        }

        return nativeType;
    }

    if (kind == 3) {
        if (!typeFromRep) {
            throw std::runtime_error("We expected a named type from a type representation");
        }

        if (!PyType_Check(typeFromRep)) {
            throw std::runtime_error(
                "We expect the type representation to produce a type object but got " +
                std::string(typeFromRep->ob_type->tp_name)
            );
        }

        Type* nativeType = PyInstance::extractTypeFrom(typeFromRep);

        if (!nativeType) {
            // the type representation could be a regular python class
            return PythonObjectOfType::Make(typeFromRep);
        }

        return nativeType;
    }

    if (kind == 4) {
        if (!group || indexInGroup == -1) {
            throw std::runtime_error("Corrupt native type: missing group or group index");
        }

        auto it = group->getIndexToObject().find(indexInGroup);
        if (it == group->getIndexToObject().end()) {
            throw std::runtime_error("Corrupt native type: index " + format(indexInGroup) + " doesn't exist in group");
        }

        if (!it->second.typeOrPyobjAsType())  {
            throw std::runtime_error("Corrupt native type: indexed group item is not a Type");
        }

        Type* resultType = it->second.typeOrPyobjAsType();

        if (resultType->getTypeCategory() == Type::TypeCategory::catForward) {
            Type* target = ((Forward*)resultType)->getTarget();
            if (target) {
                resultType = target;
            }
        }

        return resultType;
    }

    throw std::runtime_error("Unreachable");
}

PyObject* PythonSerializationContext::deserializeRecursiveObject(DeserializationBuffer& b, size_t inWireType) const {
    PyEnsureGilAcquired acquireTheGil;

    MutuallyRecursiveTypeGroup* group = nullptr;
    int32_t indexInGroup = -1;

    b.consumeCompoundMessage(inWireType, [&](size_t fieldNumber, size_t wireType) {
        if (fieldNumber == 0) {
            group = deserializeMutuallyRecursiveTypeGroup(b, wireType);
        }
        if (fieldNumber == 1) {
            assertWireTypesEqual(wireType, WireType::VARINT);
            indexInGroup = b.readUnsignedVarint();
        }
    });

    if (!group || indexInGroup == -1) {
        throw std::runtime_error("Corrupt native pyobj: missing group or group index");
    }

    auto it = group->getIndexToObject().find(indexInGroup);
    if (it == group->getIndexToObject().end()) {
        throw std::runtime_error(
            "Corrupt native pyobj: index "
            + format(indexInGroup)
            + " doesn't exist in group:\n\n"
            + group->repr()
        );
    }

    if (!it->second.pyobj())  {
        throw std::runtime_error("Corrupt native pyobj: indexed group item is not a Type");
    }

    return incref(it->second.pyobj());
}


Instance PythonSerializationContext::deserializeNativeInstance(DeserializationBuffer& b, size_t inWireType) const {
    PyEnsureGilAcquired acquireTheGil;

    Instance result;
    Type* type = nullptr;

    b.consumeCompoundMessage(inWireType, [&](size_t fieldNumber, size_t wireType) {
        if (fieldNumber == 0) {
            type = deserializeNativeType(b, wireType);

            if (type->getTypeCategory() == Type::TypeCategory::catForward) {
                throw std::runtime_error("shomehow we deserialized a forward type here.");
            }
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

// if actuallyProduceResult == false, then we still need to walk over the objects to make sure
// that our memos are correctly populated, but we don't need to construct objects, which will prevent us
// from creating new objects and leaking them when deserializing classes we already hold.
Type* PythonSerializationContext::deserializeNativeTypeInner(DeserializationBuffer& b, size_t inWireType, bool actuallyProduceResult) const {
    PyEnsureGilAcquired acquireTheGil;

    int category = -1;

    Type* namedType = nullptr;

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
    std::vector<std::pair<std::string, NamedTuple*> > alternativeMembers;
    std::map<std::string, Function*> classMethods, classStatics, classClassMethods, classPropertyFunctions;
    std::map<std::string, PyObjectHolder> classClassMembers;
    std::vector<MemberDefinition> classMembers;

    b.consumeCompoundMessage(inWireType, [&](size_t fieldNumber, size_t wireType) {
        if (fieldNumber == 0) {
            if (wireType == WireType::VARINT) {
                category = b.readUnsignedVarint();
            } else
            if (wireType == WireType::BYTES) {
                std::string objectName = b.readStringObject();

                PyObjectStealer namedObj(PyObject_CallMethod(mContextObj, "objectFromName", "s", objectName.c_str()));

                if (!namedObj) {
                    throw PythonExceptionSet();
                }

                if (!PyType_Check(namedObj)) {
                    throw std::runtime_error("Corrupt native type found: object named " + objectName + " was not a type.");
                }
                namedType = PyInstance::extractTypeFrom((PyTypeObject*)(PyObject*)namedObj);
                if (!namedType) {
                    throw std::runtime_error("Corrupt native type found: object named " + objectName + " was not a native type.");
                }
            } else {
                throw std::runtime_error("Corrupt native type found: 0 field should be a category or a name");
            }
        } else {
            if (category == -1) {
                throw std::runtime_error("Corrupt native type found.");
            }
            if (category == Type::TypeCategory::catForward
                    || category == Type::TypeCategory::catBoundMethod
            ) {
                if (fieldNumber == 1) {
                    names.push_back(b.readStringObject());
                } else if (fieldNumber == 2) {
                    types.push_back(deserializeNativeType(b, wireType));
                }
            } else
            if (category == Type::TypeCategory::catAlternativeMatcher
            ) {
                if (fieldNumber == 1) {
                    types.push_back(deserializeNativeType(b, wireType));
                }
            } else
            if (category == Type::TypeCategory::catClass) {
                types.push_back(deserializeNativeType(b, wireType));
            } else
            if (category == Type::TypeCategory::catAlternative) {
                if (fieldNumber == 1 || fieldNumber == 2) {
                    names.push_back(b.readStringObject());
                } else if (fieldNumber == 3) {
                    deserializeAlternativeMembers(alternativeMembers, b, wireType);
                } else if (fieldNumber == 4) {
                    deserializeClassFunDict(names.back(), classMethods, b, wireType);
                }
            } else
            if (category == Type::TypeCategory::catHeldClass) {
                if (fieldNumber == 1) {
                    names.push_back(b.readStringObject());
                } else if (fieldNumber == 2) {
                    deserializeClassMembers(classMembers, b, wireType);
                } else if (fieldNumber == 3) {
                    deserializeClassFunDict(names.back(), classMethods, b, wireType);
                } else if (fieldNumber == 4) {
                    deserializeClassFunDict(names.back(), classStatics, b, wireType);
                } else if (fieldNumber == 9) {
                    deserializeClassFunDict(names.back(), classClassMethods, b, wireType);
                } else if (fieldNumber == 5) {
                    deserializeClassFunDict(names.back(), classPropertyFunctions, b, wireType);
                } else if (fieldNumber == 6) {
                    deserializeClassClassMemberDict(classClassMembers, b, wireType);
                } else if (fieldNumber == 7) {
                    b.consumeCompoundMessage(wireType, [&](int which, size_t wireType2) {
                        Type* t = deserializeNativeType(b, wireType2);
                        if (!t) {
                            throw std::runtime_error(
                                "Empty class base found"
                            );
                        }
                        if (!t->isHeldClass()) {
                            throw std::runtime_error(
                                "Corrupt class base found: got " +
                                t->name() + " of category " + t->getTypeCategoryString()
                            );
                        }
                        classBases.push_back(((HeldClass*)t)->getClassType());
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
                else if (fieldNumber >= 2 && fieldNumber <= 4) {
                    assertWireTypesEqual(wireType, WireType::BYTES);
                    names.push_back(b.readStringObject());
                }
                else if (fieldNumber == 5) {
                    assertWireTypesEqual(wireType, WireType::VARINT);
                    isEntrypoint = b.readUnsignedVarint();
                }
                else if (fieldNumber == 6) {
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
                category == Type::TypeCategory::catTypedCell ||
                category == Type::TypeCategory::catSet ||
                category == Type::TypeCategory::catDict ||
                category == Type::TypeCategory::catConstDict ||
                category == Type::TypeCategory::catPointerTo ||
                category == Type::TypeCategory::catRefTo ||
                category == Type::TypeCategory::catSubclassOf ||
                category == Type::TypeCategory::catTuple ||
                (category == Type::TypeCategory::catPythonSubclass && fieldNumber == 1) ||
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
            } else if (category == Type::TypeCategory::catPythonSubclass && fieldNumber == 2) {
                obj.steal(deserializePythonObject(b, wireType));
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

    // if the type was just named in the codebase, return it
    if (namedType) {
        return namedType;
    }

    if (!actuallyProduceResult) {
        return nullptr;
    }

    // otherwise, construct it from the pieces we were given.
    Type* resultType = nullptr;

    if (category == Type::TypeCategory::catClass) {
        if (types.size() != 1) {
            throw std::runtime_error("Corrupt 'Class' encountered: can't have multiple corresponding HeldClass objects.");
        }
        if (types[0]->getTypeCategory() != Type::TypeCategory::catHeldClass) {
            throw std::runtime_error(
                "Corrupt 'Class' encountered: HeldClass is a " +
                Type::categoryToString(types[0]->getTypeCategory()) + " not a HeldClass"
            );
        }
        resultType = ((HeldClass*)types[0])->getClassType();
    }
    else if (category == Type::TypeCategory::catAlternative) {
        if (names.size() != 2) {
            throw std::runtime_error("Corrupt 'Alternative' encountered: invalid number of names");
        }

        resultType = Alternative::Make(
            names[0],
            names[1],
            alternativeMembers,
            classMethods
        );
    }
    else if (category == Type::TypeCategory::catForward) {
        throw std::runtime_error("We shouldn't be deserializing forwards directly.");
    }
    else if (category == Type::TypeCategory::catHeldClass) {
        if (names.size() != 1) {
            throw std::runtime_error("Corrupt 'HeldClass' encountered.");
        }

        std::map<std::string, PyObject*> classClassMembersRaw;
        for (auto& nameAndObj: classClassMembers) {
            classClassMembersRaw[nameAndObj.first] = incref((PyObject*)nameAndObj.second);
        }

        resultType = Class::Make(
            names[0],
            classBases,
            classIsFinal,
            classMembers,
            classMethods,
            classStatics,
            classPropertyFunctions,
            classClassMembersRaw,
            classClassMethods,
            // the methodOf should already be set in the class
            false
        )->getHeldClass();
    }
    else if (category == Type::TypeCategory::catFunction) {
        if (names.size() != 3) {
            throw std::runtime_error("Badly structured 'Function' encountered.");
        }

        resultType = Function::Make(
            names[0],
            names[1],
            names[2],
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
    else if (category == Type::TypeCategory::catEmbeddedMessage) {
        resultType = ::EmbeddedMessageType::Make();
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
    else if (category == Type::TypeCategory::catTypedCell) {
        if (types.size() != 1) {
            throw std::runtime_error("Invalid native type: TypedCell needs exactly 1 type.");
        }
        resultType = ::TypedCellType::Make(types[0]);
    }
    else if (category == Type::TypeCategory::catPointerTo) {
        if (types.size() != 1) {
            throw std::runtime_error("Invalid native type: PointerTo needs exactly 1 type.");
        }
        resultType = ::PointerTo::Make(types[0]);
    }
    else if (category == Type::TypeCategory::catBoundMethod) {
        if (types.size() != 1) {
            throw std::runtime_error("Invalid native type: BoundMethod needs exactly 1 type.");
        }
        if (names.size() != 1) {
            throw std::runtime_error("Invalid native type: BoundMethod needs exactly 1 name.");
        }
        resultType = ::BoundMethod::Make(types[0], names[0]);
    }
    else if (category == Type::TypeCategory::catAlternativeMatcher) {
        if (types.size() != 1) {
            throw std::runtime_error("Invalid native type: AternativeMatcher needs exactly 1 type.");
        }
        resultType = ::AlternativeMatcher::Make(types[0]);
    }
    else if (category == Type::TypeCategory::catSubclassOf) {
        if (types.size() != 1) {
            throw std::runtime_error("Invalid native type: SubclassOf needs exactly 1 type.");
        }
        resultType = ::SubclassOfType::Make(types[0]);
    }
    else if (category == Type::TypeCategory::catRefTo) {
        if (types.size() != 1) {
            throw std::runtime_error("Invalid native type: RefTo needs exactly 1 type.");
        }
        resultType = ::RefTo::Make(types[0]);
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
    else if (category == Type::TypeCategory::catPythonSubclass) {
        if (!obj || !PyType_Check(obj)) {
            throw std::runtime_error("Invalid native type: PythonSubclass needs a python type.");
        }

        if (types.size() != 1) {
            throw std::runtime_error("Invalid native type: PythonSubclass needs a native type.");
        }

        resultType = ::PythonSubclass::Make(types[0], (PyTypeObject*)(PyObject*)obj);
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

        resultType = a->concreteSubtype(whichIndex);
    } else {
        throw std::runtime_error("Invalid native type category");
    }

    if (!resultType) {
        throw std::runtime_error("Corrupt nativeType.");
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

PyObject* PythonSerializationContext::deserializePyList(DeserializationBuffer& b, size_t wireType, int64_t memo) const {
    return deserializeIndexable(b, wireType, PyList_New, PyList_Set_Item_No_Checks, memo);
}

PyObject* PythonSerializationContext::deserializePyTuple(DeserializationBuffer& b, size_t wireType, int64_t memo) const {
    return deserializeIndexable(b, wireType, PyTuple_New, PyTuple_Set_Item_No_Checks, memo);
}

PyObject* PythonSerializationContext::deserializePySet(DeserializationBuffer &b, size_t wireType, int64_t memo) const {
    return deserializeIterable(b, wireType, PySet_New, PySet_Add, PySet_Clear, memo);
}
