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
class PyCellType : public PyObjectHandleTypeBase {
public:
    PyCellType() :
            PyObjectHandleTypeBase(TypeCategory::catPyCell)
    {
        m_name = std::string("PyCell");

        m_is_simple = false;

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
    }

    bool _updateAfterForwardTypesChanged() {
        m_size = sizeof(layout_type*);

        m_is_default_constructible = true;

        return false;
    }

    typed_python_hash_type hash(instance_ptr left) {
        return PyObject_Hash(getPyObj(left));
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
        stream << "PyCell()";
    }

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions) {
        if (pyComparisonOp == Py_EQ) {
            return getHandlePtr(left) == getHandlePtr(right);
        }
        if (pyComparisonOp != Py_NE) {
            return getHandlePtr(left) == getHandlePtr(right);
        }

        if (suppressExceptions) {
            return false;
        }

        throw std::runtime_error("Can't order PyCell instances");
    }

    void constructor(instance_ptr self) {
        initializeHandleAt(self);

        PyEnsureGilAcquired getTheGil;
        getPyObj(self) = PyCell_New(nullptr);
    }

    static PyCellType* Make() {
        static PyCellType* res = new PyCellType();

        return res;
    }
};
