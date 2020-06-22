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


// these types can all see each other through their references, either
// through the compiler, or just through normal type references. We need to
// pick a 'first' type, which we can do by picking the first type to be defined
// in the program, and then walk through the group placing them in order.
void Type::buildCompilerRecursiveGroup(const std::set<Type*>& types) {
    if (types.size() == 0) {
        throw std::runtime_error("Empty compiler recursive group makes no sense.");
    }

    // order the types by their type index
    std::map<int64_t, Type*> typeIndex;
    for (auto t: types) {
        typeIndex[t->m_global_type_index] = t;
    }

    // then collapse that to the index with which we visit them here
    int visitIx = 0;
    std::map<int32_t, Type*> visitOrder;
    for (auto orderAndType: typeIndex) {
        visitOrder[visitIx] = orderAndType.second;
        orderAndType.second->mCompilerRecursiveTypeGroupIndex = visitIx;

        visitIx++;
    }

    Type* typeHead = visitOrder[0];

    for (auto t: types) {
        t->mCompilerRecursiveTypeGroupHead = typeHead;
    }

    typeHead->mCompilerRecursiveTypeGroup = visitOrder;
}

Type* Type::computeCompilerReferenceGroupHead() {
    std::vector<Type*> callStack;
    std::vector<std::shared_ptr<std::set<Type*> > > aboveUs;
    std::set<Type*> aboveUsSet;

    std::function<void (Type*)> visit = [&](Type* t) {
        // exclude any type that already has a recursive type group head.
        if (t->mCompilerRecursiveTypeGroupHead) {
            return;
        }

        if (aboveUsSet.find(t) == aboveUsSet.end()) {
            // we've never seen this type before
            callStack.push_back(t);
            aboveUsSet.insert(t);

            aboveUs.push_back(
                std::shared_ptr<std::set<Type*> >(
                    new std::set<Type*>()
                )
            );
            aboveUs.back()->insert(t);

            // now recurse into the subtypes
            t->visitReferencedTypes(visit);
            t->visitCompilerVisibleReferencedTypes(visit);

            callStack.pop_back();

            if (aboveUs.size() > callStack.size()) {
                // this group is now complete. Every element in it needs
                // to get a group head, and then get removed.
                Type::buildCompilerRecursiveGroup(*aboveUs.back());

                for (auto subT: *aboveUs.back()) {
                    aboveUsSet.erase(subT);
                }

                aboveUs.pop_back();
            }
        } else {
            // this is not a new type. We need to collapse this group into its parent,
            // since all of these types are mutually recursive
            while (aboveUs.back()->find(t) == aboveUs.back()->end()) {
                if (aboveUs.size() == 1) {
                    throw std::runtime_error(
                        "Somehow, we can't find this type even though we know"
                        " its above us."
                    );
                }

                aboveUs[aboveUs.size() - 2]->insert(
                    aboveUs.back()->begin(), aboveUs.back()->end()
                );
                aboveUs.pop_back();
            }
        }
    };

    visit(this);

    if (!mCompilerRecursiveTypeGroupHead) {
        throw std::runtime_error("Somehow we don't have a recursive type group head");
    }

    return mCompilerRecursiveTypeGroupHead;
}

ShaHash Type::pyObjectShaHash(PyObject* h, Type* groupHead) {
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
            res += pyObjectShaHash(PyTuple_GetItem(h, k), groupHead);
        }
        return res;
    }

    static PyObject* builtinsModule = PyImport_ImportModule("builtins");
    static PyObject* builtinsModuleDict = PyObject_GetAttrString(builtinsModule, "__dict__");

    if (h == builtinsModule) {
        return ShaHash(7);
    }
    if (h == builtinsModuleDict) {
        return ShaHash(8);
    }

    Type* argType = PyInstance::extractTypeFrom(h->ob_type);
    if (argType) {
        // this is wrong because we're not including the closure
        return argType->identityHash(groupHead);
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

ShaHash Type::pyObjectShaHash(Instance h, Type* groupHead) {
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
