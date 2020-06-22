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

#include "AllTypes.hpp"

void Type::repr(instance_ptr self, ReprAccumulator& out, bool isStr) {
    assertForwardsResolvedSufficientlyToInstantiate();

    this->check([&](auto& subtype) {
        subtype.repr(self, out, isStr);
    });
}

ShaHash Type::pyObjectShaHash(PyObject* h) {
    // handle basic constants
    if (h == Py_None) {
        return ShaHash(0);
    }

    if (h == Py_True) {
        return ShaHash(1);
    }

    if (h == Py_False) {
        return ShaHash(2);
    }

    if (PyLong_Check(h)) {
        return ShaHash(3) + ShaHash(PyLong_AsLong(h));
    }

    if (PyBytes_Check(h)) {
        return ShaHash(4) + ShaHash::SHA1(
            PyBytes_AsString(h),
            PyBytes_GET_SIZE(h)
        );
    }

    if (PyUnicode_Check(h)) {
        Py_ssize_t s;
        const char* c = PyUnicode_AsUTF8AndSize(h, &s);

        return ShaHash(5) + ShaHash::SHA1(c, s);
    }

    if (PyTuple_Check(h)) {
        ShaHash res(6);
        res += ShaHash(PyTuple_Size(h));
        for (long k = 0; k < PyTuple_Size(h); k++) {
            res += pyObjectShaHash(PyTuple_GetItem(h, k));
        }
        return res;
    }

    //TODO: actually handle this correctly.
    PyObject* repr = PyObject_Repr(h);
    if (!repr) {
        std::cout << "WARNING: tried to hash an unprintable object of type "
            << repr->ob_type->tp_name << "\n";
    } else {
        std::cout << "WARNING: tried to hash " << PyUnicode_AsUTF8(repr) << "\n";
        decref(repr);
    }
    return ShaHash::poison();
}

ShaHash Type::pyObjectShaHash(Instance h) {
    if (h.type()->getTypeCategory() == Type::TypeCategory::catNone) {
        return ShaHash(0);
    }

    if (h.type()->getTypeCategory() == Type::TypeCategory::catBool) {
        return ShaHash(h.cast<bool>() ? 1 : 2);
    }

    if (h.type()->getTypeCategory() == Type::TypeCategory::catInt64) {
        return ShaHash(3) + ShaHash(h.cast<int64_t>());
    }

    std::cout << "WARNING: tried to hash Instance " << h.repr() << "\n";

    return ShaHash::poison();
}


bool Type::cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions) {
    assertForwardsResolvedSufficientlyToInstantiate();

    return this->check([&](auto& subtype) {
        return subtype.cmp(left, right, pyComparisonOp, suppressExceptions);
    });
}

typed_python_hash_type Type::hash(instance_ptr left) {
    assertForwardsResolvedSufficientlyToInstantiate();

    return this->check([&](auto& subtype) {
        return subtype.hash(left);
    });
}

void Type::move(instance_ptr dest, instance_ptr src) {
    //right now, this is legal because we have no self references.
    swap(dest, src);
}

void Type::swap(instance_ptr left, instance_ptr right) {
    assertForwardsResolvedSufficientlyToInstantiate();

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
    assertForwardsResolvedSufficientlyToInstantiate();

    this->check([&](auto& subtype) { subtype.constructor(self); } );
}

void Type::destroy(instance_ptr self) {
    this->check([&](auto& subtype) { subtype.destroy(self); } );
}

void Type::copy_constructor(instance_ptr self, instance_ptr other) {
    assertForwardsResolvedSufficientlyToInstantiate();

    this->check([&](auto& subtype) { subtype.copy_constructor(self, other); } );
}

void Type::assign(instance_ptr self, instance_ptr other) {
    this->check([&](auto& subtype) { subtype.assign(self, other); } );
}

bool Type::isBinaryCompatibleWith(Type* other) {
    if (other == this) {
        return true;
    }

    if (isSubclassOf(other)) {
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

Maybe Type::canConstructFrom(Type* otherType, bool isExplicit) {
    if (otherType == this) {
        return Maybe::True;
    }

    if (mCanConvertOnStack.find(otherType) != mCanConvertOnStack.end()) {
        return Maybe::Maybe;
    }

    auto it = mCanConvert.find(otherType);
    if (it != mCanConvert.end()) {
        return it->second;
    }

    mCanConvertOnStack.insert(otherType);

    try {
        mCanConvert[otherType] = this->check([&](auto& subtype) {
            return subtype.canConstructFromConcrete(otherType, isExplicit);
        });
        mCanConvertOnStack.erase(otherType);
    } catch (...) {
        mCanConvertOnStack.erase(otherType);
        throw;
    }

    return mCanConvert[otherType];
}

void Type::buildMutuallyRecursiveTypeCycle() {
    if (mMutuallyRecursiveTypeGroupHead) {
        return;
    }

    std::map<Type*, int> index;
    std::vector<Type*> stack;
    std::set<Type*> stackContents;
    std::map<Type*, int> link;

    long curIndex = 0;

    std::function<void (Type*)> visit = [&](Type* cur) {
        index[cur] = curIndex;
        link[cur] = curIndex;
        curIndex += 1;

        stack.push_back(cur);
        stackContents.insert(cur);

        cur->visitReferencedTypes([&](Type* child) {
            if (cur->mMutuallyRecursiveTypeGroupHead) {
                // don't consider this node at all
                return;
            }

            if (index.find(child) == index.end()) {
                visit(child);
                link[cur] = std::min(link[cur], link[child]);
            }
            else if (stackContents.find(child) != stackContents.end()) {
                link[cur] = std::min(link[cur], index[child]);
            }
        });

        if (link[cur] == index[cur]) {
            // we're the top of a connected component
            std::vector<Type*> curComponent;

            while (stackContents.find(cur) != stackContents.end()) {
                curComponent.push_back(stack.back());
                stackContents.erase(stack.back());
                stack.pop_back();
            }

            Type* lowest = nullptr;

            for (auto t: curComponent) {
                if (t->m_is_recursive_forward && (!lowest || m_recursive_forward_index < lowest->m_recursive_forward_index)) {
                    lowest = t;
                }
            }

            if (!lowest) {
                // pick the one with the lowest name
                for (auto t: curComponent) {
                    if (!lowest || t->name() < lowest->name()) {
                        lowest = t;
                    }
                }
            }

            for (auto t: curComponent) {
                t->mMutuallyRecursiveTypeGroupHead = lowest;
                t->mMutuallyRecursiveTypeGroupIndex = -1;
            }

            std::set<Type*> curComponentSet;
            curComponentSet.insert(curComponent.begin(), curComponent.end());

            // order the nodes in the group in the order in which we traverse them
            // in the graph, filling out 'mMutuallyRecursiveTypeGroupIndex'
            long curIndex = 0;
            std::function<void (Type*)> visitForIndex = [&](Type* cur) {
                // if the node isn't in this connected component, do nothing
                if (curComponentSet.find(cur) == curComponentSet.end()) {
                    return;
                }

                // if the node already has an index, do nothing.
                if (cur->mMutuallyRecursiveTypeGroupIndex >= 0) {
                    return;
                }

                cur->mMutuallyRecursiveTypeGroupIndex = curIndex;
                lowest->mMutuallyRecursiveTypeGroup[curIndex] = cur;
                curIndex++;

                cur->visitReferencedTypes(visitForIndex);
            };

            visitForIndex(lowest);
        }
    };

    visit(this);
}

void Type::endOfConstructorInitialization() {
    visitReferencedTypes([&](Type* &t) {
        while (t->getTypeCategory() == TypeCategory::catForward && ((Forward*)t)->getTarget()) {
            t = ((Forward*)t)->getTarget();
        }

        if (t == this) {
            return;
        }

        if (t->getTypeCategory() == TypeCategory::catForward) {
            m_referenced_forwards.insert((Forward*)t);
            ((Forward*)t)->markIndirectForwardUse(this);
        } else {
            for (auto referencedT: t->getReferencedForwards()) {
                m_referenced_forwards.insert(referencedT);
                referencedT->markIndirectForwardUse(this);
            }
        }
    });

    visitContainedTypes([&](Type* t) {
        if (t->getTypeCategory() == TypeCategory::catForward) {
            m_contained_forwards.insert((Forward*)t);
        } else {
            for (auto containedT: t->getContainedForwards()) {
                m_contained_forwards.insert(containedT);
            }
        }
    });

    if (!m_referenced_forwards.size()) {
        forwardTypesAreResolved();
        mMutuallyRecursiveTypeGroup[0] = this;
        mMutuallyRecursiveTypeGroupHead = this;
    } else {
        updateAfterForwardTypesChanged();
    }
}

void Type::forwardTypesAreResolved() {
    m_resolved = true;

    updateAfterForwardTypesChanged();

    if (m_is_simple) {
        bool isSimple = true;

        visitReferencedTypes([&](Type* t) { if (!t->m_is_simple) isSimple = false; });

        if (!isSimple) {
            std::function<void (Type*)> markNotSimple([&](Type* t) {
                if (t->m_is_simple) {
                    t->m_is_simple = false;
                    markNotSimple(t);
                }
            });

            markNotSimple(this);
        }
    }

    if (mTypeRep) {
        updateTypeRepForType(this, mTypeRep);
    }
}

void Type::forwardResolvedTo(Forward* forward, Type* resolvedTo) {
    // make sure we reference this forward properly
    if (m_referenced_forwards.find(forward) == m_referenced_forwards.end()) {
        throw std::runtime_error(
            "Internal error: we are supposed to reference forward " +
            forward->name() + " but we don't have it marked."
        );
    }

    // swap out the type representation. we shouldn't reference this forward any more
    visitReferencedTypes([&](Type* &subtype) {
        if (subtype == forward) {
            subtype = resolvedTo;
        }
    });

    // update the 'containment' forward graph
    if (m_contained_forwards.find(forward) != m_contained_forwards.end()) {
        m_contained_forwards.erase(forward);
        for (auto contained: resolvedTo->getContainedForwards()) {
            if (contained != forward) {
                m_contained_forwards.insert(contained);
            }
        }
    }

    m_referenced_forwards.erase(forward);

    if (resolvedTo->getTypeCategory() == TypeCategory::catForward) {
        Forward* tgtForward = (Forward*)resolvedTo;

        if (tgtForward->getTarget()) {
            throw std::runtime_error("Forwards must not resolve to other forwards that are already resolved!");
        }

        m_referenced_forwards.insert(tgtForward);
    } else {
        for (auto referenced: resolvedTo->getReferencedForwards()) {
            if (referenced != forward) {
                referenced->markIndirectForwardUse(this);
                m_referenced_forwards.insert(referenced);
            }
        }

        if (m_referenced_forwards.size() == 0) {
            forwardTypesAreResolved();
        }
    }

    this->check([&](auto& subtype) {
        subtype._updateTypeMemosAfterForwardResolution();
    });
}


PyObject* getOrSetTypeResolver(PyObject* module, PyObject* args) {
    int num_args = 0;
    if (args)
        num_args = PyTuple_Size(args);
    if (num_args > 1) {
        PyErr_SetString(PyExc_TypeError, "getOrSetTypeResolver takes 0 or 1 positional argument");
        return NULL;
    }
    static PyObject* curResolver = nullptr;
    assertHoldingTheGil();
    if (num_args == 0) {
        return curResolver;
    }

    PyObjectHolder resolver(PyTuple_GetItem(args, 0));
    //std::cerr<<" " << Py_TYPE(module)->tp_name << " " << Py_TYPE(resolver)->tp_name <<std::endl;

    decref(curResolver);
    incref(resolver);
    curResolver = resolver;
    return curResolver;
}
