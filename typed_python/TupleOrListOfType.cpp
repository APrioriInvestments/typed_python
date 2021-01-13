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

#include "AllTypes.hpp"

bool TupleOrListOfType::isBinaryCompatibleWithConcrete(Type* other) {
    if (other->getTypeCategory() != m_typeCategory) {
        return false;
    }

    TupleOfType* otherO = (TupleOfType*)other;

    return m_element_type->isBinaryCompatibleWith(otherO->m_element_type);
}

bool TupleOrListOfType::_updateAfterForwardTypesChanged() {
    std::string name = (m_is_tuple ? "TupleOf(" : "ListOf(") + m_element_type->name(true) + ")";

    if (m_is_recursive_forward) {
        name = m_recursive_name;
    }

    bool anyChanged = name != m_name;

    m_name = name;
    m_stripped_name = "";

    return anyChanged;
}

void TupleOrListOfType::repr(instance_ptr self, ReprAccumulator& stream, bool isStr) {
    PushReprState isNew(stream, self);

    if (!isNew) {
        if (m_is_tuple) {
            stream << m_name << "(" << (void*)self << ")";
        } else {
            stream << m_name << "[" << (void*)self << "]";
        }

        return;
    }

    stream << (m_is_tuple ? "(" : "[");

    int32_t ct = count(self);

    for (long k = 0; k < ct;k++) {
        if (k > 0) {
            stream << ", ";
        }

        m_element_type->repr(eltPtr(self,k), stream, false);
    }

    stream << (m_is_tuple ? ")" : "]");
}

typed_python_hash_type TupleOrListOfType::hash(instance_ptr left) {
    if (!(*(layout**)left)) {
        return 0x123;
    }

    if ((*(layout**)left)->hash_cache == -1) {
        HashAccumulator acc(0);

        int32_t ct = count(left);

        for (long k = 0; k < ct;k++) {
            acc.add(m_element_type->hash(eltPtr(left, k)));
        }

        (*(layout**)left)->hash_cache = acc.get();
        if ((*(layout**)left)->hash_cache == -1) {
            (*(layout**)left)->hash_cache = -2;
        }
    }

    return (*(layout**)left)->hash_cache;
}

bool TupleOrListOfType::cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions) {
    layout& left_layout = **(layout**)left;
    layout& right_layout = **(layout**)right;

    bool leftEmpty = !(*(layout**)left) || left_layout.count == 0;
    bool rightEmpty = !(*(layout**)right) || right_layout.count == 0;

    if (leftEmpty && rightEmpty) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, 0);
    }
    if (leftEmpty && !rightEmpty) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, -1);
    }
    if (rightEmpty && !leftEmpty) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
    }

    if (&left_layout == &right_layout) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, 0);
    }

    if (pyComparisonOp == Py_NE) {
        return !cmp(left, right, Py_EQ, suppressExceptions);
    }

    size_t bytesPer = m_element_type->bytecount();

    if (pyComparisonOp == Py_EQ) {
        if (left_layout.count != right_layout.count) {
            return false;
        }

        for (long k = 0; k < left_layout.count && k < right_layout.count; k++) {
            if (m_element_type->cmp(left_layout.data + bytesPer * k,
                                           right_layout.data + bytesPer * k, Py_NE, suppressExceptions)) {
                return false;
            }
        }

        return true;
    }

    for (long k = 0; k < left_layout.count && k < right_layout.count; k++) {
        if (m_element_type->cmp(left_layout.data + bytesPer * k,
                                       right_layout.data + bytesPer * k, Py_NE, suppressExceptions)) {
            return cmpResultToBoolForPyOrdering(pyComparisonOp,
                m_element_type->cmp(left_layout.data + bytesPer * k,
                                       right_layout.data + bytesPer * k, Py_LT, suppressExceptions)
                    ? -1 : 1
                );
        }
    }

    if (left_layout.count < right_layout.count) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp,-1);
    }

    if (left_layout.count > right_layout.count) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp,1);
    }

    return cmpResultToBoolForPyOrdering(pyComparisonOp,0);
}

// static
TupleOfType* TupleOfType::Make(Type* elt, TupleOfType* knownType) {
    PyEnsureGilAcquired getTheGil;

    static std::map<Type*, TupleOfType*> m;

    auto it = m.find(elt);

    if (it == m.end()) {
        if (knownType == nullptr) {
            knownType = new TupleOfType(elt);
        }

        it = m.insert(std::make_pair(elt, knownType)).first;
    }

    return it->second;
}

// static
ListOfType* ListOfType::Make(Type* elt, ListOfType* knownType) {
    PyEnsureGilAcquired getTheGil;

    static std::map<Type*, ListOfType*> m;

    auto it = m.find(elt);

    if (it == m.end()) {
        if (knownType == nullptr) {
            knownType = new ListOfType(elt);
        }

        it = m.insert(std::make_pair(elt, knownType)).first;
    }

    return it->second;
}

int64_t TupleOrListOfType::count(instance_ptr self) const {
    if (!(*(layout**)self)) {
        return 0;
    }

    return (*(layout**)self)->count;
}

int64_t TupleOrListOfType::refcount(instance_ptr self) const {
    if (!(*(layout**)self)) {
        return 0;
    }

    return (*(layout**)self)->refcount;
}

void TupleOrListOfType::constructor(instance_ptr self) {
    assertForwardsResolvedSufficientlyToInstantiate();
    constructor(self, 0, [](instance_ptr i, int64_t k) {});
}

void TupleOrListOfType::destroy(instance_ptr selfPtr) {
    layout_ptr& self = *(layout_ptr*)selfPtr;

    if (!self) {
        return;
    }

    if (self->refcount.fetch_sub(1) == 1) {
        m_element_type->destroy(self->count, [&](int64_t k) {return eltPtr(self,k);});
        tp_free(self->data);
        tp_free(self);
    }
}

void TupleOrListOfType::copy_constructor(instance_ptr self, instance_ptr other) {
    (*(layout**)self) = (*(layout**)other);
    if (*(layout**)self) {
        (*(layout**)self)->refcount++;
    }
}

void TupleOrListOfType::assign(instance_ptr self, instance_ptr other) {
    if (self == other)
        return;

    layout* old = (*(layout**)self);

    (*(layout**)self) = (*(layout**)other);

    if (*(layout**)self) {
        (*(layout**)self)->refcount++;
    }

    destroy((instance_ptr)&old);
}

void TupleOrListOfType::setSizeUnsafe(instance_ptr self, size_t target) {
    layout_ptr& self_layout = *(layout_ptr*)self;

    self_layout->count = target;
}

void TupleOrListOfType::reserve(instance_ptr self, size_t target) {
    layout_ptr& self_layout = *(layout_ptr*)self;

    if (target < self_layout->count) {
        target = self_layout->count;
    }

    self_layout->data = (uint8_t*)tp_realloc(
        self_layout->data,
        getEltType()->bytecount() * self_layout->reserved,
        getEltType()->bytecount() * target
    );
    self_layout->reserved = target;
}

void TupleOrListOfType::reverse(instance_ptr self) {
    layout_ptr& self_layout = *(layout_ptr*)self;
    if (!self_layout || self_layout->count <= 1) return;

    long elt_size = getEltType()->bytecount();

    // swap memory without caring what the elements are
    uint8_t* swap = (uint8_t*)tp_malloc(elt_size);
    for (long k = 0; k < self_layout->count / 2; k++) {
        uint8_t* ptr1 = (uint8_t*)eltPtr(self_layout, k);
        uint8_t* ptr2 = (uint8_t*)eltPtr(self_layout, self_layout->count - 1 - k);
        memcpy(swap, ptr1, elt_size);
        memcpy(ptr1, ptr2, elt_size);
        memcpy(ptr2, swap, elt_size);
    }
    tp_free(swap);
}

//static
void ListOfType::copyListObject(instance_ptr target, instance_ptr src) {
    constructor(target, count(src), [&](instance_ptr tgt, long k) {
        return m_element_type->copy_constructor(tgt, eltPtr(src,k));
    });
}

void ListOfType::setSizeUnsafe(instance_ptr self, size_t count) {
    layout_ptr& self_layout = *(layout_ptr*)self;

    self_layout->count = count;
}

void ListOfType::append(instance_ptr self, instance_ptr other) {
    layout_ptr& self_layout = *(layout_ptr*)self;

    if (!self_layout) {
        self_layout = (layout_ptr)tp_malloc(sizeof(layout));
        self_layout->data = (uint8_t*)tp_malloc(getEltType()->bytecount());

        self_layout->count = 1;
        self_layout->refcount = 1;
        self_layout->reserved = 1;
        self_layout->hash_cache = -1;

        getEltType()->copy_constructor(eltPtr(self, 0), other);
    } else {
        if (self_layout->count == self_layout->reserved) {
            int64_t new_reserved = self_layout->reserved * 1.25 + 1;
            self_layout->data = (uint8_t*)tp_realloc(
                self_layout->data,
                getEltType()->bytecount() * self_layout->reserved,
                getEltType()->bytecount() * new_reserved
            );
            self_layout->reserved = new_reserved;
        }

        getEltType()->copy_constructor(eltPtr(self, self_layout->count), other);
        self_layout->count++;
    }
}

void ListOfType::ensureSpaceFor(instance_ptr data, size_t N) {
    if (reserved(data) < count(data) + N) {
        reserve(data, count(data) + N);
    }
}

size_t ListOfType::reserved(instance_ptr self) {
    layout_ptr& self_layout = *(layout_ptr*)self;

    return self_layout->reserved;
}
void ListOfType::remove(instance_ptr self, size_t index) {
    layout_ptr& self_layout = *(layout_ptr*)self;

    getEltType()->destroy(eltPtr(self, index));
    memmove(eltPtr(self, index), eltPtr(self, index+1), (self_layout->count - index - 1) * getEltType()->bytecount());
    self_layout->count--;
}

void ListOfType::resize(instance_ptr self, size_t count) {
    layout_ptr& self_layout = *(layout_ptr*)self;

    if (count > self_layout->reserved) {
        reserve(self, count);
    }

    if (count < self_layout->count) {
        getEltType()->destroy(self_layout->count - count, [&](int64_t k) {return eltPtr(self,k + count);});
        self_layout->count = count;
    }
    else if (count > self_layout->count) {
        getEltType()->constructor(count - self_layout->count, [&](int64_t k) {return eltPtr(self,k + self_layout->count);});
        self_layout->count = count;
    }
}

void ListOfType::resize(instance_ptr self, size_t count, instance_ptr value) {
    layout_ptr& self_layout = *(layout_ptr*)self;

    if (count > self_layout->reserved) {
        reserve(self, count);
    }

    if (count < self_layout->count) {
        getEltType()->destroy(self_layout->count - count, [&](int64_t k) {return eltPtr(self,k + count);});
        self_layout->count = count;
    }
    else if (count > self_layout->count) {
        getEltType()->copy_constructor(
            count - self_layout->count,
            [&](int64_t k) {return eltPtr(self,k + self_layout->count);},
            [&](int64_t k) {return value;}
            );
        self_layout->count = count;
    }
}
