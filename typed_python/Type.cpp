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

#include "AllTypes.hpp"

void Type::repr(instance_ptr self, ReprAccumulator& out) {
    assertForwardsResolved();

    this->check([&](auto& subtype) {
        subtype.repr(self, out);
    });
}

bool Type::cmp(instance_ptr left, instance_ptr right, int pyComparisonOp) {
    assertForwardsResolved();

    return this->check([&](auto& subtype) {
        return subtype.cmp(left, right, pyComparisonOp);
    });
}

int32_t Type::hash32(instance_ptr left) {
    assertForwardsResolved();

    return this->check([&](auto& subtype) {
        return subtype.hash32(left);
    });
}

void Type::move(instance_ptr dest, instance_ptr src) {
    //right now, this is legal because we have no self references.
    swap(dest, src);
}

void Type::swap(instance_ptr left, instance_ptr right) {
    assertForwardsResolved();

    if (left == right) {
        return;
    }

    size_t remaining = m_size;
    while (remaining >= 8) {
        int64_t temp = *(int64_t*)left;
        *(int64_t*)left = *(int64_t*)right;
        *(int64_t*)right = temp;

        remaining -= 8;
        left += 8;
        right += 8;
    }

    while (remaining > 0) {
        int8_t temp = *(int8_t*)left;
        *(int8_t*)left = *(int8_t*)right;
        *(int8_t*)right = temp;

        remaining -= 1;
        left += 1;
        right += 1;
    }
}

// static
char Type::byteCompare(uint8_t* l, uint8_t* r, size_t count) {
    while (count >= 8 && *(uint64_t*)l == *(uint64_t*)r) {
        l += 8;
        r += 8;
        count -= 8;
    }

    for (long k = 0; k < count; k++) {
        if (l[k] < r[k]) {
            return -1;
        }
        if (l[k] > r[k]) {
            return 1;
        }
    }
    return 0;
}

void Type::constructor(instance_ptr self) {
    assertForwardsResolved();

    this->check([&](auto& subtype) { subtype.constructor(self); } );
}

void Type::destroy(instance_ptr self) {
    assertForwardsResolved();

    this->check([&](auto& subtype) { subtype.destroy(self); } );
}


// common calculations
void Type::forwardTypesMayHaveChanged() {
    this->check([&](auto& subtype) { subtype._forwardTypesMayHaveChanged(); });

    if (m_resolved)
        return;
    if (getTypeCategory() == catForward)
        return;
    m_resolved = true;
    visitReferencedTypes([&](Type* t) { if (!t->m_resolved) m_resolved = false; });

    // TODO: this doesn't need to be recalculated each time
    visitReferencedTypes([&](Type* t) { if (t != this) {t->m_used_by.insert(this); } });

    if (m_resolved) {
        this->check([&](auto& subtype) { subtype._forwardTypesMayHaveChanged(); });
        visitReferencedTypes(
            [&](Type*& t) {
                if (t->getTypeCategory() == catForward) {
                    Type *newt = ((Forward *)t)->getTarget();
                    if (newt) {
                        t = newt;
                    }
                }
            });

        TypeCategory cat = getTypeCategory();
        m_is_simple = (cat != catAlternative && cat != catClass && cat != catHeldClass);
        if (m_is_simple)
            visitReferencedTypes([&](Type* t) { if (!t->m_is_simple) m_is_simple = false; });

        if (mTypeRep) {
            updateTypeRepForType(this, mTypeRep);
        }
    }
}

void Type::copy_constructor(instance_ptr self, instance_ptr other) {
    assertForwardsResolved();

    this->check([&](auto& subtype) { subtype.copy_constructor(self, other); } );
}

void Type::assign(instance_ptr self, instance_ptr other) {
    assertForwardsResolved();

    this->check([&](auto& subtype) { subtype.assign(self, other); } );
}

bool Type::isBinaryCompatibleWith(Type* other) {
    if (other == this) {
        return true;
    }

    while (other->getTypeCategory() == TypeCategory::catPythonSubclass) {
        other = other->getBaseType();
    }

    auto it = mIsBinaryCompatible.find(other);
    if (it != mIsBinaryCompatible.end()) {
        return it->second != BinaryCompatibilityCategory::Incompatible;
    }

    //mark that we are recursing through this datastructure. we don't want to
    //loop indefinitely.
    mIsBinaryCompatible[other] = BinaryCompatibilityCategory::Checking;

    bool isCompatible = this->check([&](auto& subtype) {
        return subtype.isBinaryCompatibleWithConcrete(other);
    });

    mIsBinaryCompatible[other] = isCompatible ?
        BinaryCompatibilityCategory::Compatible :
        BinaryCompatibilityCategory::Incompatible
        ;

    return isCompatible;
}

PyObject* getOrSetTypeResolver(PyObject* resolver) {
    static PyObject* curResolver = nullptr;
    assertHoldingTheGil();
    if (!resolver) {
        return curResolver;
    }

    decref(curResolver);
    incref(resolver);
    curResolver = resolver;
    return curResolver;
}
