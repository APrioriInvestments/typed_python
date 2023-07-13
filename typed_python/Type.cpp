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

bool Type::isValidUpcastType(Type* t1, Type* t2) {
    if (typesEquivalent(t1, t2)) {
        return true;
    }

    if (t1->isRegister() && t2->isRegister()) {
        return RegisterTypeProperties::isValidUpcast(t1, t2);
    }

    if (t1->isValue()) {
        return isValidUpcastType(((Value*)t1)->value().type(), t2);
    }

    if (t2->isValue()) {
        return false;
    }

    if (t1->isOneOf()) {
        for (auto t: ((OneOfType*)t1)->getTypes()) {
            if (!RegisterTypeProperties::isValidUpcast(t, t2)) {
                return false;
            }
        }
    }

    if (t2->isOneOf()) {
        for (auto t: ((OneOfType*)t2)->getTypes()) {
            if (RegisterTypeProperties::isValidUpcast(t1, t)) {
                return true;
            }
        }
    }

    if (t1->isTupleOf() && t2->isTupleOf()) {
        return RegisterTypeProperties::isValidUpcast(
            ((TupleOfType*)t1)->getEltType(),
            ((TupleOfType*)t2)->getEltType()
        );
    }

    return false;
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


void Type::deepcopy(
    instance_ptr dest,
    instance_ptr src,
    DeepcopyContext& context
) {
    if (context.tpTypeMap.size()) {
        bool foundOne = false;

        visitMRO([&](Type* baseType) {
            if (foundOne) {
                return;
            }

            auto it = context.tpTypeMap.find(baseType);

            if (it != context.tpTypeMap.end()) {
                PyEnsureGilAcquired getTheGil;

                PyObjectStealer asPyObj(PyInstance::extractPythonObject(src, this));

                PyObjectStealer result(
                    PyObject_CallFunction(
                        it->second,
                        "O",
                        (PyObject*)asPyObj
                    )
                );

                if (!result) {
                    throw PythonExceptionSet();
                }

                PyInstance::copyConstructFromPythonInstance(
                    this,
                    dest,
                    result,
                    ConversionLevel::Signature
                );

                foundOne = true;
            }
        });

        if (foundOne) {
            return;
        }
    }

    this->check([&](auto& subtype) {
        subtype.deepcopyConcrete(dest, src, context);
    });
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

bool Type::canConvertToTrivially(Type* otherType) {
    if (typesEquivalent(this, otherType)) {
        return true;
    }

    if (this->isSubclassOf(otherType)) {
        return true;
    }

    if (isValue()) {
        return ((Value*)this)->value().type()->canConvertToTrivially(otherType);
    }

    if (isSubclassOf() && otherType->isSubclassOf()) {
        return ((SubclassOfType*)this)->getSubclassOf()->isSubclassOf(
            ((SubclassOfType*)otherType)->getSubclassOf()
        );
    }

    if (isSubclassOf()) {
        return ((SubclassOfType*)this)->getSubclassOf()->isSubclassOf(otherType);
    }

    if (isOneOf() && otherType->isOneOf()) {
        for (Type* ownOneofElt: ((OneOfType*)this)->getTypes()) {
            if (!ownOneofElt->canConvertToTrivially(otherType)) {
                return false;
            }

            return true;
        }
    } else
    if (otherType->isOneOf()) {
        for (Type* otherOneofT: ((OneOfType*)otherType)->getTypes()) {
            if (canConvertToTrivially(otherOneofT)) {
                return true;
            }
        }
    }

    return false;
}

void reachableUnresolvedTypes(Type* root, std::set<Type*>& outTypes) {
    std::set<Type*> toVisit;
    toVisit.insert(root);

    // walk the graph and determine all forwards that are not resolved
    while (toVisit.size()) {
        Type* toCheck = *toVisit.begin();
        toVisit.erase(toCheck);

        if (toCheck->isForwardDefined() && !toCheck->isResolved()) {
            if (outTypes.find(toCheck) == outTypes.end()) {
                outTypes.insert(toCheck);

                toCheck->visitReferencedTypes([&](Type* subtype) {
                    if (subtype->isForwardDefined() && !subtype->isResolved()) {
                        toVisit.insert(subtype);
                    }
                });
            }
        }
    }
}

bool Type::looksResolvable(bool unambiguously) {
    if (!m_is_forward_defined) {
        return true;
    }

    if (m_forward_resolves_to) {
        return true;
    }

    // do the entire type resolution process while holding the GIL
    PyEnsureGilAcquired getTheGil;

    std::set<Type*> typesNeedingResolution;
    reachableUnresolvedTypes(this, typesNeedingResolution);

    for (auto t: typesNeedingResolution) {
        if (t->isForward() && !((Forward*)t)->getTarget() && !((Forward*)t)->lambdaDefinition()) {
            return false;
        }
    }

    if (unambiguously) {
        for (auto t: typesNeedingResolution) {
            if (t->isFunction() && ((Function*)t)->hasUnresolvedSymbols(false)) {
                return false;
            }
        }
    }

    return true;
}

void Type::assertForwardsResolvedSufficientlyToInstantiateConcrete() {
    if (!m_is_forward_defined) {
        return;
    }

    if (m_forward_resolves_to) {
        throw std::logic_error(
            nameWithModule() + " is forward-declared and cannot be instantiated. However, "
                + "has already been resolved, so it needs only to be replaced with a concrete "
                + "type by calling resolveForwardDeclaredType and it can be used."
        );
    }

    // do the entire type resolution process while holding the GIL
    PyEnsureGilAcquired getTheGil;

    std::set<Type*> typesNeedingResolution;
    reachableUnresolvedTypes(this, typesNeedingResolution);

    for (auto t: typesNeedingResolution) {
        if (t->isForward() && !((Forward*)t)->getTarget() && !((Forward*)t)->lambdaDefinition()) {
            throw std::logic_error(
                nameWithModule()  + " is forward-declared and cannot be autoresolved because "
                + "it refers to forward " + t->nameWithModule() + " which has no definition."
            );
        }
    }

    for (auto t: typesNeedingResolution) {
        if (t->isFunction() && ((Function*)t)->hasUnresolvedSymbols(false)) {
            throw std::logic_error(
                nameWithModule()  + " is forward-declared and cannot be autoresolved because "
                + "it refers to forward " + t->nameWithModule() + " which has a reference to "
                + "symbol '" + ((Function*)t)->firstUnresolvedSymbol(false) + "' which is "
                + "unresolved."
            );
        }
    }

    throw std::logic_error(
        nameWithModule() + " is forward-declared and cannot be instantiated. However, "
            + "it is resolvable, so it needs only to be replaced with a concrete "
            + "type by calling resolveForwardDeclaredType and it can be used."
    );
}

bool Type::assertResolvable(bool unambiguously) {
    if (!m_is_forward_defined) {
        return true;
    }

    if (m_forward_resolves_to) {
        return true;
    }

    // do the entire type resolution process while holding the GIL
    PyEnsureGilAcquired getTheGil;

    std::set<Type*> typesNeedingResolution;
    reachableUnresolvedTypes(this, typesNeedingResolution);

    for (auto t: typesNeedingResolution) {
        if (t->isForward() && !((Forward*)t)->getTarget() && !((Forward*)t)->lambdaDefinition()) {
            throw std::runtime_error("Forward " + t->name() + " has no definition.");
        }
    }

    if (unambiguously) {
        for (auto t: typesNeedingResolution) {
            if (t->isFunction() && ((Function*)t)->hasUnresolvedSymbols(false)) {
                throw std::runtime_error(
                    "Function " + t->nameWithModule() + " has unresolved symbol '" +
                        ((Function*)t)->firstUnresolvedSymbol(false) + "'"
                );
            }
        }
    }

    return true;
}

// try to resolve this forward type. If we can't, we'll throw an exception. On exit,
// we will have thrown, or m_forward_resolves_to will be populated.
void Type::attemptToResolve() {
    if (m_forward_resolves_to) {
        return;
    }

    if (!m_is_forward_defined) {
        return;
    }

    // do the entire type resolution process while holding the GIL
    PyEnsureGilAcquired getTheGil;

    std::set<Type*> typesNeedingResolution;
    reachableUnresolvedTypes(this, typesNeedingResolution);

    std::set<Type*> existingReferencedTypes;

    for (auto t: typesNeedingResolution) {
        if (t->isForward() && !((Forward*)t)->getTarget() && !((Forward*)t)->lambdaDefinition()) {
            throw std::runtime_error(
                "Forward defined as " + t->nameWithModule() + " has not been defined."
            );
        }
    }

    for (auto t: typesNeedingResolution) {
        if (t->isForward() && !((Forward*)t)->getTarget()) {
            ((Forward*)t)->define(
                ((Forward*)t)->lambdaDefinition()
            );
        }
    }

    // for each type that we have defined in our graph, what type are we mapping it to?
    // for Forwards of primitive types like ListOf or TupleOf that can see themselves, this
    // this will be a 'named copy' of the type depending on the Forward name.
    std::map<Type*, Type*> resolutionMapping;

    // for each type we end up defining, which type did it come from
    std::map<Type*, Type*> resolutionSource;

    // allocate new target type bodies for all regular types in the graph
    // that are forward declared.
    for (auto t: typesNeedingResolution) {
        if (!t->isForward()) {
            // we're resolving to this type directly
            resolutionMapping[t] = t->cloneForForwardResolution();
            resolutionSource[resolutionMapping[t]] = t;
        }
    }

    // now ensure that the target for any forward is the underlying type
    // note that we don't need a source for any of these since nobody will end up
    // actually having a forward in their graph
    for (auto t: typesNeedingResolution) {
        if (t->isForward()) {
            Forward* f = (Forward*)t;

            Type* tgt = f->getTargetTransitive();

            if (!tgt) {
                throw std::runtime_error("somehow this forward doesn't have a target");
            }

            if (typesNeedingResolution.find(tgt) == typesNeedingResolution.end()) {
                resolutionMapping[t] = tgt;
                existingReferencedTypes.insert(tgt);
            } else {
                // resolve this forward to whatever its target resolves to
                resolutionMapping[t] = resolutionMapping[tgt];
            }
        }
    }

    // copy all the types. At this point, they all know that they are not forwards
    // but none of them is ready to be identity-hashed yet. They all will be in a state of
    // where m_needs_post_init is true.
    for (auto typeAndSource: resolutionSource) {
        typeAndSource.first->initializeFrom(typeAndSource.second);
        typeAndSource.first->updateInternalTypePointers(resolutionMapping);
    }

    // cause each type to recompute its name. We have to do this in a single pass
    // without any types being aware of their final name before we run the post initialize
    // step. Otherwise, it's possible for the names to depend on the order of initialization
    // and we want them to be stable.
    for (auto typeAndSource: resolutionSource) {
        typeAndSource.first->recomputeName();
    }

    // cause each type to post-initialize itself, which lets it update internal bytecounts and
    // default initialization flags
    bool anyUpdated = true;
    size_t passCt = 0;
    while (anyUpdated) {
        anyUpdated = false;
        for (auto typeAndSource: resolutionSource) {
            if (typeAndSource.first->postInitialize()) {
                anyUpdated = true;
            }
        }
        passCt += 1;

        // we can run this algorithm until all type sizes have stabilized. Conceivably we
        // could introduce an error that would cause this to not converge - this should
        // detect that.
        if (passCt > resolutionSource.size() * 2 + 10) {
            throw std::runtime_error("Type size graph is not stabilizing.");
        }
    }

    // let each type update any internal caches it might need before it gets instantiated
    for (auto typeAndSource: resolutionSource) {
        typeAndSource.first->finalizeType();
    }

    // std::cout << "resolving group:\n";
    // for (auto typeAndSource: resolutionSource) {
    //     std::cout << "    " << TypeOrPyobj(typeAndSource.first).name() << " from " << TypeOrPyobj(typeAndSource.second).name() << "\n";
    // }

    // now internalize the types by their hash. For each type, we compute a hash
    // and then look to see if we've seen it before. We build a lookup table from
    // each existing type to the internalized type, and then do the same process we did
    // above to map any roots across.

    std::map<Type*, Type*> resolvedToInternal; // each 'resolved' type what should it be replaced with

    std::set<Type*> newTypes;
    std::set<Type*> redundantTypes;

    for (auto typeAndSource: resolutionSource) {
        ShaHash h = typeAndSource.first->identityHash();

        auto it = mInternalizedIdentityHashToType.find(h);
        if (it == mInternalizedIdentityHashToType.end()) {
            // this is a new type. We're going to put it in the memo and
            // update it so that it ponts
            newTypes.insert(typeAndSource.first);
        } else {
            resolvedToInternal[typeAndSource.first] = it->second;
            redundantTypes.insert(typeAndSource.first);
        }
    }

    // update all the new types to no longer look at the redundant types
    for (auto t: newTypes) {
        t->updateInternalTypePointers(resolvedToInternal);
        mInternalizedIdentityHashToType[t->identityHash()] = t;
        resolvedToInternal[t] = t;
    }

    for (auto t: redundantTypes) {
        t->markRedundant();
    }

    // tell each source type which type it actually resolves to. We're holding the GIL so
    // nobody should see anything until we finish this process.
    for (auto typeAndTarget: resolutionMapping) {
        auto it = resolvedToInternal.find(typeAndTarget.second);
        if (it == resolvedToInternal.end()) {
            // this must be an existing type
            if (existingReferencedTypes.find(typeAndTarget.second) == existingReferencedTypes.end()) {
                throw std::runtime_error(
                    "Failed to get an internalization of Type "
                    + typeAndTarget.second->name() + " and it wasn't a pre-existing type either."
                );
            }

            // we resolve to it directly
            typeAndTarget.first->m_forward_resolves_to = typeAndTarget.second;
        } else {
            typeAndTarget.first->m_forward_resolves_to = it->second;
        }
    }

    if (!m_forward_resolves_to) {
        throw std::runtime_error("Somehow, we didn't resolve???");
    }
}

void Type::internalize() {
    if (mIdentityHash == ShaHash()) {
        throw std::runtime_error("This type should already have been hashed!");
    }

    if (mInternalizedIdentityHashToType.find(mIdentityHash)
            != mInternalizedIdentityHashToType.end()) {
        throw std::runtime_error("This type is already internalized!");
    }

    mInternalizedIdentityHashToType[mIdentityHash] = this;
}

void Type::tryToAutoresolve() {
    // some types don't need resolution
    if (!m_is_forward_defined) {
        return;
    }

    // some types are already resolved
    if (m_forward_resolves_to) {
        return;
    }

    if (!looksResolvable(true)) {
        return;
    }

    std::set<Type*> typesNeedingResolution;
    reachableUnresolvedTypes(this, typesNeedingResolution);

    if (::getenv("TP_AUTORESOLVE_VERBOSE") && ::getenv("TP_AUTORESOLVE_VERBOSE")[0]) {
        std::cout << "try to autoresolve " << TypeOrPyobj(this).name() << ":\n";
        for (auto t: typesNeedingResolution) {
            std::cout << "    " << TypeOrPyobj(t).name() << "\n";
        }
    }

    attemptToResolve();

    for (auto t: typesNeedingResolution) {
        t->attemptAutoresolveWrite();
    }

    for (auto t: typesNeedingResolution) {
        if (t->isForward()) {
            ((Forward*)t)->installDefinitionIfLambda();
        }

        if (t->isFunction()) {
            ((Function*)t)->autoresolveGlobals(typesNeedingResolution);
        }
    }
}

void Type::attemptAutoresolveWrite() {
    if (!m_is_forward_defined || !m_forward_resolves_to) {
        return;
    }

    if (!mAutoresolveFrameOwners.size()) {
        return;
    }

    long updateCount = 0;

    for (auto f: mAutoresolveFrameOwners) {
        // force the locals to be populated with a dict so we can muck with them
        PyFrame_FastToLocals((PyFrameObject*)f);

        PyObject* locals = ((PyFrameObject*)f)->f_locals;

        if (locals && PyDict_Check(locals)) {
            PyObject *key, *value;
            Py_ssize_t pos = 0;

            PyObject* ownTypeObj = (PyObject*)PyInstance::typeObj(this);
            PyObject* resolvedTypeObj = (PyObject*)PyInstance::typeObj(m_forward_resolves_to);

            bool updated = false;
            while (PyDict_Next(locals, &pos, &key, &value)) {
                if (value == ownTypeObj) {
                    PyDict_SetItem(locals, key, resolvedTypeObj);
                    updated = true;
                }
            }

            if (updated) {
                updateCount += 1;
                PyFrame_LocalsToFast((PyFrameObject*)f, 0);
            }
        }

        decref(f);
    }

    mAutoresolveFrameOwners.clear();

    if (::getenv("TP_AUTORESOLVE_VERBOSE") && ::getenv("TP_AUTORESOLVE_VERBOSE")[0]) {
        std::cout << "Autoresolve wrote " << name() << " into " << updateCount << " stackframe slots\n";
    }
}

void Type::typeFinishedBeingDeserializedPhase1() {
    if (m_is_being_deserialized) {
        // during deserialization we have to defer anything that actually
        // looks hard at the type and copies data into the PyTypeObject
        // because its not ready yet. This allows us to refer to the object
        // without it being finished.
        PyTypeObject* typeObj = PyInstance::typeObj(this);

        PyInstance::finalizePyTypeObjectPhase1(this, typeObj);
    }
}

void Type::typeFinishedBeingDeserializedPhase2() {
    if (m_is_being_deserialized) {
        // during deserialization we have to defer anything that actually
        // looks hard at the type and copies data into the PyTypeObject
        // because its not ready yet. This allows us to refer to the object
        // without it being finished.
        PyTypeObject* typeObj = PyInstance::typeObj(this);

        PyInstance::finalizePyTypeObjectPhase2(this, typeObj);

        m_is_being_deserialized = false;
    }
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

//static
std::map<ShaHash, Type*> Type::mInternalizedIdentityHashToType;
