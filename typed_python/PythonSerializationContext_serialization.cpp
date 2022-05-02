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
void PythonSerializationContext::serializePythonObject(PyObject* o, SerializationBuffer& b, size_t fieldNumber) const {
    PyEnsureGilAcquired acquireTheGil;

    b.writeBeginCompound(fieldNumber);

    // if this is already part of the recursive type group that we're currently writing, and its unnamed,
    // then write a reference to this mutually recursive type group.
    // however, if its named, we want to fall back to the normal mechanism for writing it.
    // we can't actually _start_ serializing a mutually recursive type group here, however,
    // because otherwise we'd create a dependency during serialization on _whether_ the
    // group was already created, which would introduce a nondeterminism into serilization
    // based on the compiler state
    if (PyCell_Check(o) || PyFunction_Check(o)) {
        auto groupAndIndex = MutuallyRecursiveTypeGroup::pyObjectGroupHeadAndIndex(o, false);

        // if we have a group, check whether we've already memoized this group in the stream
        // this prevents us from initiating the serialization of a new group here.
        if (groupAndIndex.first && b.isAlreadyCached(groupAndIndex.first)) {
            //see if the object has a name. If so, we always want to use that.
            PyObjectStealer typeName(PyObject_CallMethod(mContextObj, "nameForObject", "(O)", o));
            if (!typeName) {
                throw PythonExceptionSet();
            }

            if (typeName == Py_None) {
                b.writeBeginCompound(FieldNumbers::RECURSIVE_OBJECT);
                serializeMutuallyRecursiveTypeGroup(groupAndIndex.first, b, 0);
                b.writeUnsignedVarintObject(1, groupAndIndex.second);
                b.writeEndCompound();
                b.writeEndCompound();
                return;
            }
        }
    }

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
            Py_ssize_t sz = 0;
            const char* c = PyUnicode_AsUTF8AndSize(o, &sz);

            if (!c) {
                throw PythonExceptionSet();
            }

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

void PythonSerializationContext::serializePyDict(PyObject* o, SerializationBuffer& b) const {
    PyEnsureGilAcquired acquireTheGil;

    b.writeBeginCompound(FieldNumbers::DICT);

    PyObject *key, *value;
    Py_ssize_t pos = 0;

    while (PyDict_Next(o, &pos, &key, &value)) {
        serializePythonObject(key, b, 0);
        serializePythonObject(value, b, 0);
    }

    b.writeEndCompound();
}

void PythonSerializationContext::serializePyFrozenSet(PyObject* o, SerializationBuffer& b) const {
    serializeIterable(o, b, FieldNumbers::FROZENSET);
}

void PythonSerializationContext::serializePythonObjectNamedOrAsObj(PyObject* o, SerializationBuffer& b) const {
    PyEnsureGilAcquired acquireTheGil;

    if (b.isAlreadyCached(o)) {
        b.writeUnsignedVarintObject(FieldNumbers::MEMO, b.memoFor(o));
        return;
    }

    //see if the object has a name
    PyObjectStealer typeName(PyObject_CallMethod(mContextObj, "nameForObject", "(O)", o));
    if (!typeName) {
        throw PythonExceptionSet();
    }

    if (typeName != Py_None) {
        if (!PyUnicode_Check(typeName)) {
            throw std::runtime_error(std::string("nameForObject returned a non-string"));
        }

        b.writeUnsignedVarintObject(FieldNumbers::MEMO, b.cachePointer(o).first);
        b.writeStringObject(FieldNumbers::OBJECT_NAME, std::string(PyUnicode_AsUTF8(typeName)));
        return;
    }

    if (PyDict_CheckExact(o)) {
        b.writeUnsignedVarintObject(FieldNumbers::MEMO, b.cachePointer(o).first);
        serializePyDict(o, b);
        return;
    }

    //give the plugin a chance to convert the instance to something else
    PyObjectStealer representation(PyObject_CallMethod(mContextObj, "representationFor", "(O)", o));
    if (!representation) {
        throw PythonExceptionSet();
    }

    bool representationIsPythonTypeObject = (
        PyType_Check(o)
        && PyTuple_Check(representation)
        && PyTuple_Size(representation) > 1
        && PyTuple_GetItem(representation, 0) == (PyObject*)&PyType_Type
    );

    if (representation != Py_None && !representationIsPythonTypeObject) {
        b.writeUnsignedVarintObject(FieldNumbers::MEMO, b.cachePointer(o).first);
        serializePythonObjectRepresentation(representation, b, FieldNumbers::OBJECT_REPRESENTATION);
        return;
    }

    // we don't write a memo here
    if (PyType_Check(o)) {
        auto groupAndIndex = MutuallyRecursiveTypeGroup::pyObjectGroupHeadAndIndex(o);

        b.writeBeginCompound(FieldNumbers::RECURSIVE_OBJECT);
        serializeMutuallyRecursiveTypeGroup(groupAndIndex.first, b, 0);
        b.writeUnsignedVarintObject(1, groupAndIndex.second);
        b.writeEndCompound();
        return;
    }

    b.writeUnsignedVarintObject(FieldNumbers::MEMO, b.cachePointer(o).first);

    b.writeBeginCompound(FieldNumbers::OBJECT_TYPEANDDICT);
    serializePythonObject((PyObject*)o->ob_type, b, 0);
    PyObjectStealer objDict(PyObject_GenericGetDict(o, nullptr));
    if (!objDict) {
        if (PyCFunction_Check(o)) {
            PyErr_Format(PyExc_TypeError,
                "Method %s (of instance %S) had no dict.",
                ((PyCFunctionObject*)o)->m_ml ? ((PyCFunctionObject*)o)->m_ml->ml_name : "<NULL>",
                PyCFunction_GetSelf(o) ? (PyObject*)PyCFunction_GetSelf(o) : Py_None
            );
        } else {
            PyErr_Format(PyExc_TypeError,
                "Object %S (of type %S) had no dict.",
                o,
                o->ob_type
            );
        }
        throw PythonExceptionSet();
    }
    serializePythonObject(objDict, b, 1);
    b.writeEndCompound();
}

void PythonSerializationContext::serializePythonObjectRepresentation(PyObject* representation, SerializationBuffer& b, size_t fieldNumber) const {
    PyEnsureGilAcquired acquireTheGil;

    if (!PyTuple_Check(representation) || PyTuple_Size(representation) < 2 
            || PyTuple_Size(representation) > 6) {
        throw std::runtime_error("representationFor should return None or a tuple with 2 to 6 things");
    }
    if (!PyTuple_Check(PyTuple_GetItem(representation, 1))) {
        throw std::runtime_error("representationFor second arguments should be a tuple");
    }

    b.writeBeginCompound(fieldNumber);
    for (long ix = 0; ix < PyTuple_Size(representation); ix++) {
        PyObjectHolder rep(PyTuple_GetItem(representation, ix));
        serializePythonObject(rep, b, ix);
    }
    b.writeEndCompound();
}

bool isSimpleType(Type* t) {
    Type::TypeCategory cat = t->getTypeCategory();

    return
        cat == Type::TypeCategory::catInt8 ||
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
        cat == Type::TypeCategory::catEmbeddedMessage ||
        cat == Type::TypeCategory::catBool ||
        cat == Type::TypeCategory::catPyCell;
}

void PythonSerializationContext::serializeNativeType(Type* nativeType, SerializationBuffer& b, size_t fieldNumber) const {
    PyEnsureGilAcquired getTheGil;

    b.writeBeginCompound(fieldNumber);

    std::string name;

    if (isSimpleType(nativeType)) {
        // this is an inline type
        b.writeUnsignedVarintObject(0, 0);
        b.writeUnsignedVarintObject(1, nativeType->getTypeCategory());
    } else
    if (nativeType->getTypeCategory() == Type::TypeCategory::catConcreteAlternative &&
            (name = getNameForPyObj((PyObject*)PyInstance::typeObj(nativeType->getBaseType()))).size()) {
        // this is an inline named concrete alternative
        b.writeUnsignedVarintObject(0, 1);
        b.writeStringObject(1, name);
        b.writeUnsignedVarintObject(2, ((ConcreteAlternative*)nativeType)->which());
    } else
    if ((name = getNameForPyObj((PyObject*)PyInstance::typeObj(nativeType))).size()) {
        // this is an inline named type
        b.writeUnsignedVarintObject(0, 2);
        b.writeStringObject(1, name);
    } else {
        //give the plugin a chance to convert the instance to something else
        PyObjectStealer representation(
            PyObject_CallMethod(mContextObj, "representationFor", "(O)", (PyObject*)PyInstance::typeObj(nativeType))
        );

        if (!representation) {
            throw PythonExceptionSet();
        }

        if (representation != Py_None) {
            b.writeUnsignedVarintObject(0, 3);
            serializePythonObjectRepresentation(representation, b, 1);
        } else {
            // this is a member of a recursive type group
            b.writeUnsignedVarintObject(0, 4);
            serializeMutuallyRecursiveTypeGroup(nativeType->getRecursiveTypeGroup(), b, 1);
            b.writeUnsignedVarintObject(2, nativeType->getRecursiveTypeGroupIndex());
        }
    }

    b.writeEndCompound();
}

void PythonSerializationContext::serializeMutuallyRecursiveTypeGroup(MutuallyRecursiveTypeGroup* group, SerializationBuffer& b, size_t fieldNumber) const {
    PyEnsureGilAcquired acquireTheGil;

    if (group->hash().isPoison()) {
        for (auto ixAndMember: group->getIndexToObject()) {
            std::cout << "INVALID: " << ixAndMember.first << " -> " << ixAndMember.second.name() << "\n";
        }
        throw std::runtime_error("Can't serialize a mutually recursive type group with an invalid hash.");
    }

    b.writeBeginCompound(fieldNumber);
        // group memo
        uint32_t id;
        bool isNew;

        std::tie(id, isNew) = b.cachePointer(group, nullptr);

        // write the memo first
        b.writeUnsignedVarintObject(FieldNumbers::MEMO, id);

        // if this is not the first time we write this, we can exit now
        if (!isNew) {
            b.writeEndCompound();
            return;
        }

        // group hash
        if (shouldSerializeHashSequence()) {
            b.writeUnsignedVarintObject(1, b.getGroupCounter(group));
        } else {
            b.writeStringObject(1, group->hash().digestAsString());
        }

        // check if any element of this group is named. If so, we can just write the name and
        // then grab the group on the other side
        for (auto& indexAndObj: group->getIndexToObject()) {
            std::string name;

            if (indexAndObj.second.typeOrPyobjAsType()) {
                name = getNameForPyObj((PyObject*)PyInstance::typeObj(indexAndObj.second.typeOrPyobjAsType()));
            } else {
                name = getNameForPyObj(indexAndObj.second.pyobj());
            }

            if (name.size()) {
                // field '5' indicates that this is a mutually recursive type group
                // identified by the name of an object within it. We can skip writing
                // the rest of the group since the other side will have the same
                // representation.
                b.writeStringObject(5, name);
                b.writeEndCompound();
                return;
            }
        }

        std::map<int32_t, PyObjectHolder> indicesWrittenAsObjectAndRep;
        std::set<int32_t> indicesWrittenAsExternalObjectAndDict;
        std::set<int32_t> indicesWrittenAsInternalObjectAndDict;

        // for each item in the group, enough information to allocate it
        // as a forward:
        //      0 if its a native type,
        //      1 if its a named object
        //      2 if its an object-representation
        //      3 if its an object-and-dict whose type is not included
        //           in this group, and we serialize the type as a python object
        //      4 if its an object-and-dict whose type is included
        //           in this group, and we serialize it as an index.
        //      5 if its a tuple
        b.writeBeginCompound(2);
            std::set<int> writtenSoFar;

            // write the 'header' for object in the group indexed by 'index'.
            // we do this recursively, so that we can ensure we write out the
            // headers for any base classes of classes in this group that are also
            // in this group, first. We have to do it this way because
            // we can't instantiate a class without the base class object being
            // instantiated as a type first. The 'object representation' of a
            // type contains its bases as part of the constructor, which can
            // lead to cycles.
            std::function<void (int)> writeObjectHeader = [&](int index) {
                // if we wrote this already, we can exit immediately
                if (writtenSoFar.find(index) != writtenSoFar.end()) {
                    return;
                }

                // mark that we wrote it
                writtenSoFar.insert(index);

                TypeOrPyobj obj = group->getIndexToObject().find(index)->second;

                // check whether this is a type object and if so, check all the bases
                if (obj.pyobj() && PyType_Check(obj.pyobj())) {
                    PyTypeObject* typeObj = (PyTypeObject*)obj.pyobj();

                    if (typeObj->tp_bases) {
                        iterate(typeObj->tp_bases, [&](PyObject* baseCls) {
                            int index = group->indexOfObjectInThisGroup(baseCls);
                            if (index != -1) {
                                writeObjectHeader(index);
                            }
                        });
                    }
                }

                b.writeBeginCompound(index);

                if (obj.typeOrPyobjAsType()) {
                    // it's a type object. write it as a native type and serialize its inner pieces.
                    b.writeUnsignedVarintObject(0, 0);
                    b.writeStringObject(1, obj.typeOrPyobjAsType()->name());
                    b.writeUnsignedVarintObject(2, obj.typeOrPyobjAsType()->getTypeCategory());
                } else {
                    //give the plugin a chance to convert the instance to something else
                    PyObjectStealer representation(
                        PyObject_CallMethod(mContextObj, "representationFor", "(O)", obj.pyobj())
                    );

                    if (!representation) {
                        throw PythonExceptionSet();
                    }

                    if (representation != Py_None) {
                        if (!PyTuple_Check(representation) || PyTuple_Size(representation) < 2 
                                || PyTuple_Size(representation) > 6) {
                            throw std::runtime_error("representationFor should return None or a tuple with 2 to 6 things");
                        }
                        if (!PyTuple_Check(PyTuple_GetItem(representation, 1))) {
                            throw std::runtime_error("representationFor second arguments should be a tuple");
                        }

                        // indicate that this is a 'representation' object
                        b.writeUnsignedVarintObject(0, 2);

                        // and serialize the first two parts of the representation, which are
                        // the object factory and its arguments. we'll include the state later.
                        serializePythonObject(PyTuple_GetItem(representation, 0), b, 1);
                        serializePythonObject(PyTuple_GetItem(representation, 1), b, 2);

                        indicesWrittenAsObjectAndRep[index].steal(
                            PyTuple_GetSlice(representation, 2, PyTuple_Size(representation))
                        );
                    } else
                    if (PyTuple_Check(obj.pyobj())) {
                        b.writeUnsignedVarintObject(0, 5);
                        b.writeUnsignedVarintObject(1, PyTuple_Size(obj.pyobj()));
                        indicesWrittenAsExternalObjectAndDict.insert(index);
                    } else {
                        // we're going to serialize this as a regular python object with
                        // a dict. we need to serialize the type object first.

                        // it's either a native type, or not.
                        // and it's either in our group or not.

                        int32_t indexInThisGroup = group->indexOfObjectInThisGroup(
                            (PyObject*)obj.pyobj()->ob_type
                        );

                        if (indexInThisGroup == -1) {
                            indicesWrittenAsExternalObjectAndDict.insert(index);
                            b.writeUnsignedVarintObject(0, 3);
                            serializePythonObject((PyObject*)obj.pyobj()->ob_type, b, 1);
                        } else {
                            indicesWrittenAsInternalObjectAndDict.insert(index);
                            b.writeUnsignedVarintObject(0, 4);
                            b.writeUnsignedVarintObject(1, indexInThisGroup);
                        }
                    }
                }

                b.writeEndCompound();
            };

            for (auto& indexAndObj: group->getIndexToObject()) {
                writeObjectHeader(indexAndObj.first);
            }

        b.writeEndCompound();

        // then the object bodies. we have to write native types first, so that
        // they can be instantiated fully, then objects and dictionaries. otherwise,
        // we end up with uninstantiated forwards leaking into our object graph
        b.writeBeginCompound(3);
            auto writeObjectBody = [&](int index, TypeOrPyobj obj) {
                if (indicesWrittenAsObjectAndRep.find(index) != indicesWrittenAsObjectAndRep.end()) {
                    // write the context
                    serializePythonObject(
                        indicesWrittenAsObjectAndRep[index],
                        b,
                        index
                    );
                } else
                if (
                    indicesWrittenAsExternalObjectAndDict.find(index) != indicesWrittenAsExternalObjectAndDict.end() ||
                    indicesWrittenAsInternalObjectAndDict.find(index) != indicesWrittenAsInternalObjectAndDict.end()
                ) {
                    Type* nt = PyInstance::extractTypeFrom(obj.pyobj()->ob_type);

                    if (nt) {
                        // this is an instance of a Type
                        nt->serialize(((PyInstance*)obj.pyobj())->dataPtr(), b, index);
                    } else
                    if (PyCell_Check(obj.pyobj())) {
                        serializePythonObject(PyCell_Get(obj.pyobj()), b, index);
                    } else
                    if (PyTuple_Check(obj.pyobj())) {
                        serializePythonObject(obj.pyobj(), b, index);
                    } else {
                        // write the context
                        PyObjectStealer objDict(PyObject_GenericGetDict(obj.pyobj(), nullptr));

                        if (!objDict) {
                            if (PyCFunction_Check(obj.pyobj())) {
                                auto o = obj.pyobj();

                                PyErr_Format(PyExc_TypeError,
                                    "Method %s (of instance %S) had no dict.",
                                    ((PyCFunctionObject*)o)->m_ml ? ((PyCFunctionObject*)o)->m_ml->ml_name : "<NULL>",
                                    PyCFunction_GetSelf(o) ? (PyObject*)PyCFunction_GetSelf(o) : Py_None
                                );
                            } else {
                                PyErr_Format(PyExc_TypeError,
                                    "Object %S (of type %S) had no dict.",
                                    obj.pyobj(),
                                    obj.pyobj()->ob_type
                                );
                            }

                            throw PythonExceptionSet();
                        }
                        serializePythonObject(objDict, b, index);
                    }
                } else {
                    Type* toSerialize = obj.typeOrPyobjAsType();
                    if (toSerialize->isForward()) {
                        Forward* f = (Forward*)toSerialize;
                        if (f->getTarget()) {
                            // we already indicated that this is a forward,
                            // so we just want to serialize the type of the inner
                            // which should be another MRTG reference
                            serializeNativeType(f->getTarget(), b, index);
                        }
                    } else {
                        serializeNativeTypeInner(toSerialize, b, index);
                    }
                }
            };

            // write functions first
            for (auto& indexAndObj: group->getIndexToObject()) {
                if (
                    indexAndObj.second.typeOrPyobjAsType() &&
                    indexAndObj.second.typeOrPyobjAsType()->isFunction()
                ) {
                    writeObjectBody(indexAndObj.first, indexAndObj.second);
                }
            }

            // then HeldClass objects, ordered with all bases before
            // their children (so that when we deserialize, any base classes
            // are no longer forwards)
            std::map<HeldClass*, int> heldClasses;
            for (auto& indexAndObj: group->getIndexToObject()) {
                if (
                    indexAndObj.second.typeOrPyobjAsType() &&
                    indexAndObj.second.typeOrPyobjAsType()->isHeldClass()
                ) {
                    heldClasses[(HeldClass*)indexAndObj.second.typeOrPyobjAsType()] = indexAndObj.first;
                }
            }

            std::set<HeldClass*> alreadyWrittenHeldClass;
            std::function<void (HeldClass*)> visit = [&](HeldClass* cls) {
                if (alreadyWrittenHeldClass.find(cls) != alreadyWrittenHeldClass.end()) {
                    return;
                }

                alreadyWrittenHeldClass.insert(cls);

                // walk the base class list and ensure all of them are serialized
                // first
                for (HeldClass* ancestor: cls->getMro()) {
                    if (heldClasses.find(ancestor) != heldClasses.end()) {
                        visit(ancestor);
                    }
                }

                // now we can write this object.
                writeObjectBody(heldClasses[cls], cls);
            };

            for (auto classAndIndex: heldClasses) {
                visit(classAndIndex.first);
            }


            // everything else
            for (auto& indexAndObj: group->getIndexToObject()) {
                if (indexAndObj.second.typeOrPyobjAsType()
                        && !indexAndObj.second.typeOrPyobjAsType()->isFunction()
                        && !indexAndObj.second.typeOrPyobjAsType()->isHeldClass()
                    ) {
                    writeObjectBody(indexAndObj.first, indexAndObj.second);
                }
            }

            // write out python objects that are types
            for (auto& indexAndObj: group->getIndexToObject()) {
                if (!indexAndObj.second.typeOrPyobjAsType() &&
                        PyType_Check(indexAndObj.second.pyobj())) {
                    writeObjectBody(indexAndObj.first, indexAndObj.second);
                }
            }

            // write out the remainder (python objects that are not types)
            for (auto& indexAndObj: group->getIndexToObject()) {
                if (!indexAndObj.second.typeOrPyobjAsType() &&
                        !PyType_Check(indexAndObj.second.pyobj())) {
                    writeObjectBody(indexAndObj.first, indexAndObj.second);
                }
            }

        b.writeEndCompound();

        b.writeBeginCompound(4);

            for (auto& indexAndObj: group->getIndexToObject()) {
                if (indexAndObj.second.typeOrPyobjAsType() && indexAndObj.second.typeOrPyobjAsType()->isFunction()) {
                    // we need to serialize the 'globals' dict of each function object
                    Function* f = (Function*)indexAndObj.second.typeOrPyobjAsType();

                    b.writeBeginCompound(indexAndObj.first);

                        for (long k = 0; k < f->getOverloads().size(); k++) {
                            PyObjectStealer globals(f->getOverloads()[k].getUsedGlobals());
                            serializePythonObject(globals, b, k);
                        }

                    b.writeEndCompound();
                }
            }

        b.writeEndCompound();

    b.writeEndCompound();
}

void PythonSerializationContext::serializeNativeTypeInner(
            Type* nativeType,
            SerializationBuffer& b,
            size_t fieldNumber
            ) const {
    b.writeBeginCompound(fieldNumber);

    if (isSimpleType(nativeType)) {
        b.writeUnsignedVarintObject(0, nativeType->getTypeCategory());
        b.writeEndCompound();
        return;
    }

    PyEnsureGilAcquired acquireTheGil;

    PyObjectStealer nameForObject(PyObject_CallMethod(mContextObj, "nameForObject", "(O)", PyInstance::typeObj(nativeType)));

    if (!nameForObject) {
        throw PythonExceptionSet();
    }

    if (nameForObject != Py_None) {
        if (!PyUnicode_Check(nameForObject)) {
            throw std::runtime_error("nameForObject returned something other than None or a string.");
        }

        b.writeStringObject(0, std::string(PyUnicode_AsUTF8(nameForObject)));
        b.writeEndCompound();
        return;
    }

    b.writeUnsignedVarintObject(0, nativeType->getTypeCategory());

    if (nativeType->getTypeCategory() == Type::TypeCategory::catConcreteAlternative) {
        serializeNativeType(nativeType->getBaseType(), b, 1);
        b.writeUnsignedVarintObject(2, ((ConcreteAlternative*)nativeType)->which());
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catAlternative) {
        Alternative* altType = (Alternative*)nativeType;

        b.writeStringObject(1, altType->name());
        b.writeStringObject(2, altType->moduleName());
        serializeAlternativeMembers(altType->subtypes(), b, 3);
        serializeClassFunDict(altType->getMethods(), b, 4);
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catSet) {
        serializeNativeType(((SetType*)nativeType)->keyType(), b, 1);
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catConstDict) {
        serializeNativeType(((ConstDictType*)nativeType)->keyType(), b, 1);
        serializeNativeType(((ConstDictType*)nativeType)->valueType(), b, 2);
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catDict) {
        serializeNativeType(((DictType*)nativeType)->keyType(), b, 1);
        serializeNativeType(((DictType*)nativeType)->valueType(), b, 2);
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catPythonSubclass) {
        serializeNativeType(((PythonSubclass*)nativeType)->baseType(), b, 1);
        serializePythonObject((PyObject*)((PythonSubclass*)nativeType)->pyType(), b, 2);
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catTupleOf) {
        serializeNativeType(((TupleOfType*)nativeType)->getEltType(), b, 1);
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catPointerTo) {
        serializeNativeType(((PointerTo*)nativeType)->getEltType(), b, 1);
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catRefTo) {
        serializeNativeType(((RefTo*)nativeType)->getEltType(), b, 1);
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catSubclassOf) {
        serializeNativeType(((SubclassOfType*)nativeType)->getSubclassOf(), b, 1);
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catBoundMethod) {
        b.writeStringObject(1, ((BoundMethod*)nativeType)->getFuncName());
        serializeNativeType(((BoundMethod*)nativeType)->getFirstArgType(), b, 2);
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catAlternativeMatcher) {
        serializeNativeType(((AlternativeMatcher*)nativeType)->getAlternative(), b, 1);
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catListOf) {
        serializeNativeType(((ListOfType*)nativeType)->getEltType(), b, 2);
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catTypedCell) {
        serializeNativeType(((TypedCellType*)nativeType)->getHeldType(), b, 2);
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
        b.writeStringObject(3, ftype->qualname());
        b.writeStringObject(4, ftype->moduleName());
        b.writeUnsignedVarintObject(5, ftype->isEntrypoint() ? 1 : 0);
        b.writeUnsignedVarintObject(6, ftype->isNocompile() ? 1 : 0);

        int whichIndex = 7;
        for (auto& overload: ftype->getOverloads()) {
            overload.serialize(*this, b, whichIndex++);
        }
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catForward) {
        throw std::runtime_error("We shouldn't be serializing forwards directly");
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catClass) {
        serializeNativeType(((Class*)nativeType)->getHeldClass(), b, 1);
    } else if (nativeType->getTypeCategory() == Type::TypeCategory::catHeldClass) {
        HeldClass* cls = (HeldClass*)nativeType;

        // note that we serialize the name of the class (not the held class)
        // which has 'Held' in the name. Practically speaking, we're serializing
        // the data necessary to call Class::Make, even though the object we're
        // serializing is the HeldClass.
        b.writeStringObject(1, cls->getClassType()->name());

        //members
        serializeClassMembers(cls->getOwnMembers(), b, 2);

        //methods
        serializeClassFunDict(cls->getOwnMemberFunctions(), b, 3);
        serializeClassFunDict(cls->getOwnStaticFunctions(), b, 4);
        serializeClassFunDict(cls->getOwnClassMethods(), b, 9);
        serializeClassFunDict(cls->getOwnPropertyFunctions(), b, 5);
        serializeClassClassMemberDict(cls->getOwnClassMembers(), b, 6);

        b.writeBeginCompound(7);

        {
            int which = 0;
            for (auto t: cls->getBases()) {
                serializeNativeType(t, b, which++);
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

    b.writeEndCompound();
}

void PythonSerializationContext::serializeClassMembers(
    const std::vector<MemberDefinition>& members,
    SerializationBuffer& b,
    int fieldNumber
) const {
    b.writeBeginCompound(fieldNumber);
    for (long k = 0; k < members.size(); k++) {
        b.writeBeginCompound(k);
            b.writeStringObject(0, members[k].getName());

            serializeNativeType(members[k].getType(), b, 1);

            Instance defaultValue = members[k].getDefaultValue();

            if (defaultValue.type()->getTypeCategory() != Type::TypeCategory::catNone) {
                b.writeBeginCompound(2);

                serializeNativeType(defaultValue.type(), b, 0);

                PyEnsureGilReleased releaseTheGil;
                defaultValue.type()->serialize(defaultValue.data(), b, 1);

                b.writeEndCompound();
            }

            b.writeUnsignedVarintObject(3, members[k].getIsNonempty() ? 1 : 0);
        b.writeEndCompound();
    }
    b.writeEndCompound();
}

void PythonSerializationContext::serializeAlternativeMembers(
    const std::vector<std::pair<std::string, NamedTuple*> >& members,
    SerializationBuffer& b,
    int fieldNumber
) const {
    b.writeBeginCompound(fieldNumber);
    for (long k = 0; k < members.size(); k++) {
        b.writeBeginCompound(k);
            b.writeStringObject(0, members[k].first);

            serializeNativeType(members[k].second, b, 1);
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

    PyObjectStealer iter(PyObject_GetIter(o));
    if (!iter) {
        return;
    }
    PyObjectStealer item(PyIter_Next(iter));
    while (item) {
        serializePythonObject(item, b, 0);
        item.steal(PyIter_Next(iter));
    }

    b.writeEndCompound();
}

void PythonSerializationContext::serializePyList(PyObject* o, SerializationBuffer& b) const {
    serializeIterable(o, b, FieldNumbers::LIST);
}

void PythonSerializationContext::serializePyTuple(PyObject* o, SerializationBuffer& b) const {
    serializeIterable(o, b, FieldNumbers::TUPLE);
}

void PythonSerializationContext::serializePySet(PyObject* o, SerializationBuffer& b) const {
    serializeIterable(o, b, FieldNumbers::SET);
}
