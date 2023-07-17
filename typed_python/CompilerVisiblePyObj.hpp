/******************************************************************************
   Copyright 2017-2023 typed_python Authors

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

#include <vector>
#include "Type.hpp"
#include "Instance.hpp"


/*********************************
CompilerVisiblePyObject

a representation of a python object that's owned by TypedPython.  We hold these by
pointer and leak them indiscriminately - like Type objects they're considered to be permanent
and singletonish

**********************************/

class CompilerVisiblePyObj {
    enum class Kind {
        // this should never be visible in a running program
        Uninitialized = 0,
        // we're pointing back into a fully resolved Type
        Type = 1,
        // we're pointing into a TP instannce that doesn't reach 
        // a more complex object. It can have Type leaves in it 
        Instance = 2,
        // this is a python tuple object
        PyTuple = 3,
        // the bailout pathway for cases we don't handle well. We should
        // assume that the compiler will treat this object as a plain
        // PyObject without looking inside of it, and the details of this
        // object are insufficient to differentiate two different Function
        // types that both refer to different ArbitraryPyObject instances.
        ArbitraryPyObject = 4
    };

    CompilerVisiblePyObj() :
        mKind(Kind::Uninitialized),
        mType(nullptr),
        mPyObject(nullptr)
    {
    }

public:
    bool isUninitialized() const {
        return mKind == Kind::Uninitialized;
    }

    bool isType() const {
        return mKind == Kind::Type;
    }

    bool isInstance() const {
        return mKind == Kind::Instance;
    }

    bool isPyTuple() const {
        return mKind == Kind::PyTuple;
    }

    bool isArbitraryPyObject() const {
        return mKind == Kind::ArbitraryPyObject;
    }

    static CompilerVisiblePyObj* Type(Type* t) {
        CompilerVisiblePyObj* res = new CompilerVisiblePyObj();

        res->mKind = Kind::Type;
        res->mType = t;

        return res;
    }

    static CompilerVisiblePyObj* Instance(Instance i) {
        CompilerVisiblePyObj* res = new CompilerVisiblePyObj();

        res->mKind = Kind::Instance;
        res->mInstance = i;

        return res;
    }

    static CompilerVisiblePyObj* PyTuple() {
        CompilerVisiblePyObj* res = new CompilerVisiblePyObj();

        res->mKind = Kind::PyTuple;

        return res;
    }

    static CompilerVisiblePyObj* ArbitraryPyObject(PyObject* val) {
        CompilerVisiblePyObj* res = new CompilerVisiblePyObj();

        res->mKind = Kind::ArbitraryPyObject;
        res->mPyObject = incref(val);

        return res;
    }

    static CompilerVisiblePyObj* PyTuple(const std::vector<CompilerVisiblePyObj*>& elts) {
        CompilerVisiblePyObj* res = new CompilerVisiblePyObj();

        res->mKind = Kind::PyTuple;
        res->mElements = elts;

        return res;
    }

    // return a CVPO for 'val', stashing it in 'constantMapCache' 
    // in case we hit a recursion.
    static CompilerVisiblePyObj* internalizePyObj(
        PyObject* val,
        std::unordered_map<PyObject*, CompilerVisiblePyObj*>& constantMapCache,
        const std::map<::Type*, ::Type*>& groupMap
    ) {
        auto it = constantMapCache.find(val);
        
        if (it != constantMapCache.end()) {
            return it->second;
        }

        constantMapCache[val] = new CompilerVisiblePyObj();

        constantMapCache[val]->becomeInternalizedOf(val, constantMapCache, groupMap);

        return constantMapCache[val];
    }

    void becomeInternalizedOf(
        PyObject* val,
        std::unordered_map<PyObject*, CompilerVisiblePyObj*>& constantMapCache,
        const std::map<::Type*, ::Type*>& groupMap
    ) {
        ::Type* t = PyInstance::extractTypeFrom(val);

        if (t) {
            if (groupMap.find(t) != groupMap.end()) {
                t = groupMap.find(t)->second;
            } else {
                if (t->isForwardDefined()) {
                    if (t->isResolved()) {
                        t = t->forwardResolvesTo();
                    }
                }
            }

            mKind = Kind::Type;
            mType = t;
            return;
        }

        if (PyTuple_Check(val)) {
            mKind = Kind::PyTuple;
            for (long i = 0; i < PyTuple_Size(val); i++) {
                mElements.push_back(
                    CompilerVisiblePyObj::internalizePyObj(
                        PyTuple_GetItem(val, i),
                        constantMapCache,
                        groupMap
                    )
                );
            }
            return;
        }

        ::Type* instanceType = PyInstance::extractTypeFrom(val->ob_type);
        if (instanceType) {
            mKind = Kind::Instance;
            mInstance = ::Instance::create(
                instanceType, 
                ((PyInstance*)val)->dataPtr()
            );
            return;
        }

        if (val == Py_None) {
            mKind = Kind::Instance;
            return;
        }

        if (PyBool_Check(val)) {
            mKind = Kind::Instance;
            mInstance = Instance::create(val == Py_True);
            return;
        }

        if (PyLong_Check(val)) {
            mKind = Kind::Instance;
            
            try {
                mInstance = Instance::create((int64_t)PyLong_AsLongLong(val));
            }
            catch(...) {
                mInstance = Instance::create((uint64_t)PyLong_AsUnsignedLongLong(val));
            }

            return;
        }

        if (PyFloat_Check(val)) {
            mKind = Kind::Instance;
            mInstance = Instance::create(PyFloat_AsDouble(val));
            return;
        }

        if (PyBytes_Check(val)) {
            mKind = Kind::Instance;
            mInstance = Instance::createAndInitialize(
                BytesType::Make(),
                [&](instance_ptr i) {
                    BytesType::Make()->constructor(
                        i, 
                        PyBytes_GET_SIZE(val), 
                        PyBytes_AsString(val)
                    );
                }
            );
            return;
        }

        if (PyUnicode_Check(val)) {
            mKind = Kind::Instance;

            auto kind = PyUnicode_KIND(val);
            assert(
                kind == PyUnicode_1BYTE_KIND ||
                kind == PyUnicode_2BYTE_KIND ||
                kind == PyUnicode_4BYTE_KIND
                );
            int64_t bytesPerCodepoint =
                kind == PyUnicode_1BYTE_KIND ? 1 :
                kind == PyUnicode_2BYTE_KIND ? 2 :
                                               4 ;

            int64_t count = PyUnicode_GET_LENGTH(val);

            const char* data =
                kind == PyUnicode_1BYTE_KIND ? (char*)PyUnicode_1BYTE_DATA(val) :
                kind == PyUnicode_2BYTE_KIND ? (char*)PyUnicode_2BYTE_DATA(val) :
                                               (char*)PyUnicode_4BYTE_DATA(val);

            mInstance = Instance::createAndInitialize(
                StringType::Make(),
                [&](instance_ptr i) {
                    StringType::Make()->constructor(i, bytesPerCodepoint, count, data);
                }
            );

            return;
        }

        mKind = Kind::ArbitraryPyObject;
        mPyObject = incref(val);
    }

    void append(CompilerVisiblePyObj* elt) {
        if (mKind != Kind::PyTuple) {
            throw std::runtime_error("Expected a PyTuple");
        }

        mElements.push_back(elt);
    }

    const std::vector<CompilerVisiblePyObj*>& elements() const {
        return mElements;
    }

    ::Type* getType() const {
        return mType;
    }

    const ::Instance& getInstance() const {
        return mInstance;
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& v) {
        if (mKind == Kind::Type) {
            v(mType);
            return;
        }

        if (mKind == Kind::Instance) {
            // TODO: what to do here?
        }

        if (mKind == Kind::PyTuple) {
            // TODO: what to do here?
        }
    }

    template<class visitor_type>
    void _visitCompilerVisibleInternals(const visitor_type& visitor) {
        if (mKind == Kind::Type) {
            visitor.visitTopo(mType);
        }

        if (mKind == Kind::Instance) {
            // TODO: what to do here?
            visitor.visitInstance(mInstance.type(), mInstance.data());
        }

        if (mKind == Kind::PyTuple) {
            // TODO: what to do here?
            throw std::runtime_error("TODO: CompilerVisiblePyObj::_visitCompilerVisibleInternals PyTuple");
        }

        if (mKind == Kind::ArbitraryPyObject) {
            visitor.visitTopo(mPyObject);
        }
    }

    // get the python object representation of this object, which isn't guaranteed
    // to exist and may need to be constructed on demand.
    PyObject* getPyObj() {
        if (mKind == Kind::Type) {
            return (PyObject*)PyInstance::typeObj(mType);
        }

        if (mKind == Kind::ArbitraryPyObject) {
            return mPyObject;
        }

        if (mKind == Kind::Instance) {
            if (!mPyObject) {
                mPyObject = PyInstance::extractPythonObject(mInstance);
            }

            return mPyObject;
        }

        throw std::runtime_error("Can't make a python object representation for this pyobj");
    }

    std::string toString() {
        if (mKind == Kind::Type) {
            return "CompilerVisiblePyObj.Type(" + mType->name() + ")";
        }

        if (mKind == Kind::Instance) {
            return "CompilerVisiblePyObj.Instance(" + mInstance.type()->name() + ")";
        }

        if (mKind == Kind::PyTuple) {
            return "CompilerVisiblePyObj.PyTuple()";
        }

        if (mKind == Kind::ArbitraryPyObject) {
            return std::string("CompilerVisiblePyObj.ArbitraryPyObject(type=")
                + mPyObject->ob_type->tp_name + ")";
        }

        throw std::runtime_error("Unknown CompilerVisiblePyObj Kind.");
    }

    template<class serialization_context_t, class buf_t>
    void serialize(serialization_context_t& context, buf_t& buffer, int fieldNumber) const {
        uint32_t id;
        bool isNew;
        std::tie(id, isNew) = buffer.cachePointer((void*)this, nullptr);

        if (!isNew) {
            buffer.writeBeginCompound(fieldNumber);
            buffer.writeUnsignedVarintObject(0, id);
            buffer.writeEndCompound();
            return;
        } else {
            buffer.writeBeginCompound(fieldNumber);
            buffer.writeUnsignedVarintObject(0, id);
            buffer.writeUnsignedVarintObject(1, (int)mKind);

            if (mKind == Kind::Type) {
                context.serializeNativeType(mType, buffer, 2);
            } else
            if (mKind == Kind::Instance) {
                buffer.writeBeginCompound(3);
                context.serializeNativeType(mInstance.type(), buffer, 0);
                mInstance.type()->serialize(mInstance.data(), buffer, 1);
                buffer.writeEndCompound();
            } else
            if (mKind == Kind::PyTuple) {
                buffer.writeBeginCompound(4);
                for (long i = 0; i < mElements.size(); i++) {
                    mElements[i]->serialize(context, buffer, i);
                }
                buffer.writeEndCompound();
            } else
            if (mKind == Kind::ArbitraryPyObject) {
                context.serializePythonObject(mPyObject, buffer, 5);
            }

            buffer.writeEndCompound();
        }
    }

    template<class serialization_context_t, class buf_t>
    static CompilerVisiblePyObj* deserialize(serialization_context_t& context, buf_t& buffer, int wireType) {
        int64_t kind = 0;

        ::Type* type = nullptr;

        std::vector<CompilerVisiblePyObj*> vec;
        ::Instance i;
        uint32_t id = -1;
        PyObjectHolder pyobj;

        buffer.consumeCompoundMessage(wireType, [&](size_t fieldNumber, size_t wireType) {
            if (fieldNumber == 0) {
                id = buffer.readUnsignedVarint();
            } else
            if (fieldNumber == 1) {
                kind = buffer.readUnsignedVarint();
            } else
            if (fieldNumber == 2) {
                type = context.deserializeNativeType(buffer, wireType);
            } else
            if (fieldNumber == 3) {
                i = context.deserializeNativeInstance(buffer, wireType);
            } else
            if (fieldNumber == 4) {
                buffer.consumeCompoundMessage(wireType, [&](size_t fieldNumber, size_t wireType) {
                    vec.push_back(CompilerVisiblePyObj::deserialize(context, buffer, wireType));
                });
            } else
            if (fieldNumber == 5) {
                pyobj.steal(context.deserializePythonObject(buffer, wireType));
            }
        });

        void* ptr = buffer.lookupCachedPointer(id);
        if (ptr) {
            return (CompilerVisiblePyObj*)ptr;
        }

        if (kind == (int)Kind::Type) {
            if (!type) {
                throw std::runtime_error("Corrupt CompilerVisiblePyObj::Type");
            }
            return CompilerVisiblePyObj::Type(type);
        }

        if (kind == (int)Kind::Instance) {
            return CompilerVisiblePyObj::Instance(i);
        }

        if (kind == (int)Kind::PyTuple) {
            return CompilerVisiblePyObj::PyTuple(vec);
        }

        if (kind == (int)Kind::Uninitialized) {
            return new CompilerVisiblePyObj();
        }

        if (kind == (int)Kind::ArbitraryPyObject) {
            return CompilerVisiblePyObj::ArbitraryPyObject(pyobj);
        }

        throw std::runtime_error("Corrupt CompilerVisiblePyObj - invalid kind");
    }

private:
    Kind mKind;

    ::Type* mType;
    ::Instance mInstance;

    // if we are an ArbitraryPythonObject this is always populated
    // otherwise, it will be a cache for a constructed instance
    PyObject* mPyObject;

    std::vector<CompilerVisiblePyObj*> mElements;
};
