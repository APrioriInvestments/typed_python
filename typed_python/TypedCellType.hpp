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

#include "util.hpp"
#include "Type.hpp"

//wraps a python Cell
class TypedCellType : public Type {
public:
    class layout {
    public:
        std::atomic<int64_t> refcount;
        int64_t initialized;
        unsigned char data[];
    };

    typedef layout* layout_ptr;

    TypedCellType(Type* heldType) :
            Type(TypeCategory::catTypedCell),
            mHeldType(heldType)
    {
        m_name = std::string("TypedCell(") + heldType->name() + ")";

        m_is_simple = false;

        m_size = sizeof(layout*);

        m_is_default_constructible = true;

        endOfConstructorInitialization(); // finish initializing the type object.
    }

    bool isBinaryCompatibleWithConcrete(Type* other) {
        return other == this;
    }

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        visitor(mHeldType);
    }

    bool _updateAfterForwardTypesChanged() {
        std::string newName = std::string("TypedCell(") + mHeldType->name() + ")";

        bool nameChanged = newName != m_name;

        m_name = newName;

        return nameChanged;
    }

    void _updateTypeMemosAfterForwardResolution() {
        TypedCellType::Make(mHeldType, this);
    }

    layout_ptr& getLayoutPtr(instance_ptr self) {
        return *(layout**)self;
    }

    typed_python_hash_type hash(instance_ptr left) {
        return mHeldType->hash(getLayoutPtr(left)->data);
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber) {
        throw std::runtime_error("Cells are not serializable.");
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t wireType) {
        throw std::runtime_error("Cells are not serializable.");
    }

    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr) {
        stream << "TypedCell()";
    }

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions) {
        if (pyComparisonOp == Py_EQ) {
            return getLayoutPtr(left) == getLayoutPtr(right);
        }
        if (pyComparisonOp != Py_NE) {
            return getLayoutPtr(left) == getLayoutPtr(right);
        }

        if (suppressExceptions) {
            return false;
        }

        throw std::runtime_error("Can't order TypedCell instances");
    }

    void initializeHandleAt(instance_ptr data) {
        getLayoutPtr(data) = (layout*)malloc(sizeof(layout) + mHeldType->bytecount());
        getLayoutPtr(data)->refcount = 1;
        getLayoutPtr(data)->initialized = false;
    }

    void constructor(instance_ptr self) {
        initializeHandleAt(self);
    }

    void destroy(instance_ptr self) {
        getLayoutPtr(self)->refcount--;

        if (getLayoutPtr(self)->refcount == 0) {
            if (getLayoutPtr(self)->initialized) {
                mHeldType->destroy(getLayoutPtr(self)->data);
            }
            free(getLayoutPtr(self));
        }
    }

    void copy_constructor(instance_ptr self, instance_ptr other) {
        getLayoutPtr(self) = getLayoutPtr(other);
        getLayoutPtr(self)->refcount++;
    }

    void assign(instance_ptr self, instance_ptr other) {
        if (getLayoutPtr(self) == getLayoutPtr(other)) {
            return;
        }
        getLayoutPtr(other)->refcount++;
        destroy(self);
        getLayoutPtr(self) = getLayoutPtr(other);
    }

    template<class initializer_fun>
    void set(instance_ptr self, initializer_fun initializer) {
        clear(self);

        initializer((instance_ptr)getLayoutPtr(self)->data);

        getLayoutPtr(self)->initialized = true;
    }

    void clear(instance_ptr self) {
        if (getLayoutPtr(self)->initialized) {
            mHeldType->destroy(getLayoutPtr(self)->data);
            getLayoutPtr(self)->initialized = false;
        }
    }

    bool isSet(instance_ptr self) {
        return getLayoutPtr(self)->initialized;
    }

    instance_ptr get(instance_ptr self) {
        return getLayoutPtr(self)->data;
    }

    static TypedCellType* Make(Type* t, TypedCellType* knownType = nullptr) {
        static std::mutex guard;

        std::lock_guard<std::mutex> lock(guard);

        typedef Type* keytype;

        static std::map<keytype, TypedCellType*> m;

        auto it = m.find(t);
        if (it == m.end()) {
            it = m.insert(
                std::make_pair(
                    t,
                    knownType ? knownType : new TypedCellType(t)
                )
            ).first;
        }

        return it->second;
    }

    Type* getHeldType() const {
        return mHeldType;
    }

private:
    Type* mHeldType;
};
