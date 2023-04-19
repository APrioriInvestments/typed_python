/******************************************************************************
   Copyright 2017-2022 typed_python Authors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY visibility, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

#include "CompilerVisibleObjectVisitor.hpp"
#include "MutuallyRecursiveTypeGroup.hpp"


MutuallyRecursiveTypeGroup::MutuallyRecursiveTypeGroup(VisibilityType visibility) :
    mAnyPyObjectsIncorrectlyOrdered(false),
    mIsDeserializerGroup(false),
    mVisibilityType(visibility)
{
}

MutuallyRecursiveTypeGroup::MutuallyRecursiveTypeGroup(
    ShaHash hash,
    VisibilityType visibility
) :
    mAnyPyObjectsIncorrectlyOrdered(false),
    mIntendedHash(hash),
    mVisibilityType(visibility),
    mIsDeserializerGroup(true)
{
}


MutuallyRecursiveTypeGroup* MutuallyRecursiveTypeGroup::DeserializerGroup(
    ShaHash hash,
    VisibilityType k
) {
    return new MutuallyRecursiveTypeGroup(hash, k);
}


void MutuallyRecursiveTypeGroup::finalizeDeserializerGroup() {
    PyEnsureGilAcquired getTheGil;

    if (!mIsDeserializerGroup) {
        throw std::runtime_error("This MRTG is not a builder group so you can't finalize it");
    }

    if (mHash != ShaHash()) {
        throw std::runtime_error("Type group is already installed\n" + repr());
    }

    // we are a builder group. this means some deserializer started constructing this
    // from data. Unfortunately, there is no guarantee that its belief about the hash of this
    // group matches our own.  We want to make sure that we memoize the hashes as the writer
    // saw them, but don't want to actually assume that the hash is correct!

    // A discrepancy can arise from one of two situations: either (a)
    // the serializer has a different version of TP itself, and has hashed something
    // differently, or (b) there is a globally visible object reachable from this group and the
    // serializer's version of the code is different. For instance, the source code of a
    // globally visible function may have changed.

    // We know that this particular mutually recursive type group can't have a globally
    // visible object in it directly (since otherwise it would be serialized as a name only).
    // So, its sufficient to attempt to install us, and if we're already installed, replace our
    // deserialized topos with the ones in the installed type group.

    // in some ways, its a little strange that we're memoizing by the intended hash here,
    // in the global singleton, instead of in the deserialization buffer - after all, we could
    // be reading from multiple different corrupted streams with different definitions that
    // conflict with each other.  However, this protocol is definitely not robust to
    // adversarial attacks, and we want to prevent cases where we are repeatedly deserializing
    // and building the same MRTG over and over, since we don't clean them up after the fact!

    // it's also the case that some of our internal objects may not be in the cache memo
    // since the deserializer can build compound objects. As a result, we can simply
    // build a group around _any_ of our objects (which should end up in the same order)
    // and then copy them in

    MutuallyRecursiveTypeGroup::ensureRecursiveTypeGroup(
        mIndexToObject.begin()->second,
        mVisibilityType
    );

    MutuallyRecursiveTypeGroup* canonical = MutuallyRecursiveTypeGroup::groupAndIndexFor(
        mIndexToObject.begin()->second,
        mVisibilityType
    ).first;

    if (mIndexToObject.size() != canonical->mIndexToObject.size()) {
        throw std::runtime_error(
            "Somehow, when we deserialized this group:\n"
            + repr()
            + "\n\nand hashed it, we ended up with this group\n\n"
            + canonical->repr()
            + "\n\nwhich has a different number of elements"
        );
    }

    // note that this check is somewhat expensive - it SHOULD always pass, but in theory
    // we should only need it on for debug builds.
    computeHash();

    if (mHash != canonical->mHash) {
        throw std::runtime_error(
            "Somehow, when we deserialized this group:\n"
            + repr()
            + "\n\nand hashed it, we ended up with this group\n\n"
            + canonical->repr()
            + "\n\nwhich has a different hash!"
        );
    }

    // simply use the objects from the canonical group.
    mIndexToObject = canonical->mIndexToObject;
    mObjectToIndex = canonical->mObjectToIndex;

    // install ourselves by 'intended hash' so that future deserializers can just use this
    // instead of building a new group
    {
        std::lock_guard<std::recursive_mutex> lock(mMutex);

        // if this intended hash doesn't match anything, just copy it in
        if (mIntendedHashToGroup[mVisibilityType].find(mIntendedHash) == mIntendedHashToGroup[mVisibilityType].end()) {
            mIntendedHashToGroup[mVisibilityType][mIntendedHash] = this;
        }
    }
}

void MutuallyRecursiveTypeGroup::_computeHashAndInstall() {
    if (mHash != ShaHash()) {
        throw std::runtime_error("Type group is already installed\n" + repr());
    }

    computeHash();

    {
        std::lock_guard<std::recursive_mutex> lock(mMutex);

        // see if this official hash has been seen yet
        if (mHashToGroup[mVisibilityType].find(mHash) == mHashToGroup[mVisibilityType].end()) {
            // this is the official MRTG for this hash, so we install ourselves
            // as the official group and register ourselves in the various type memos.
            // this group will come up in the reverse lookup now.
            mHashToGroup[mVisibilityType][mHash] = this;
        }

        for (auto& typeAndOrder: mObjectToIndex) {
            auto it = mTypeGroups[mVisibilityType].find(typeAndOrder.first);

            if (it == mTypeGroups[mVisibilityType].end()) {
                // this type has never been installed - go ahead and add it. It's possible
                // there are multiple copies of this type in the system
                mTypeGroups[mVisibilityType][typeAndOrder.first] = std::make_pair(this, typeAndOrder.second);

                if (typeAndOrder.first.type() && mVisibilityType == VisibilityType::Identity) {
                    Type* t = typeAndOrder.first.type();
                    t->setRecursiveTypeGroup(this, typeAndOrder.second);
                }
            }
        }

    }
}

//static
void MutuallyRecursiveTypeGroup::visibleFrom(
    TypeOrPyobj root,
    std::vector<TypeOrPyobj>& outReachable,
    VisibilityType visibility
) {
    CompilerVisibleObjectVisitor::singleton().visit(
        root,
        visibility,
        [&](ShaHash h) {},
        [&](const std::string& s) {},
        [&](TypeOrPyobj t) { outReachable.push_back(t); },
        [&](const std::string& s, TypeOrPyobj t) { outReachable.push_back(t); },
        [&](const std::string& err) {}
    );
}

void MutuallyRecursiveTypeGroup::setIndexToObject(int32_t index, TypeOrPyobj obj) {
    if (!mIsDeserializerGroup) {
        throw std::runtime_error("This MRTG is not a builder group.");
    }

    auto it = mIndexToObject.find(index);
    if (it != mIndexToObject.end()) {
        mObjectToIndex.erase(it->second);
        mObjectToIndex.insert({obj, index});
        it->second = obj;
    } else {
        mIndexToObject.insert({index, obj});
        mObjectToIndex.insert({obj, index});
    }
}

bool MutuallyRecursiveTypeGroup::objectIsUnassigned(TypeOrPyobj obj, VisibilityType visibility) {
    // we can check if we're installed without hitting the lock
    if (visibility == VisibilityType::Identity && obj.type() && obj.type()->hasTypeGroup()) {
        return false;
    }

    std::lock_guard<std::recursive_mutex> lock(mMutex);

    // if we've already installed this group into 'mTypeGroups'
    if (mTypeGroups[visibility].find(obj) != mTypeGroups[visibility].end()) {
        return false;
    }

    // if its a constant not worth delving into
    if (obj.pyobj() && CompilerVisibleObjectVisitor::isSimpleConstant(obj.pyobj())) {
        return false;
    }

    return true;
}

int32_t MutuallyRecursiveTypeGroup::indexOfObjectInThisGroup(TypeOrPyobj o) {
    auto it = mObjectToIndex.find(o);
    if (it == mObjectToIndex.end()) {
        return -1;
    }
    return it->second;
}

std::string MutuallyRecursiveTypeGroup::repr(bool deep) {
    std::ostringstream s;

    std::set<MutuallyRecursiveTypeGroup*> seen;

    std::function<void (MutuallyRecursiveTypeGroup*, int)> dump = [&](MutuallyRecursiveTypeGroup* group, int level) {
        seen.insert(group);

        std::string levelPrefix(level, ' ');

        s << levelPrefix << "group with hash " << group->hash().digestAsHexString() << ":\n";

        // sort lexically and then by level, so that
        // we can tell what's going on when we have a discrepancy
        std::map<std::string, std::vector<int> > ordered;
        for (auto& ixAndObj: group->mIndexToObject) {
            ordered[ixAndObj.second.name()].push_back(ixAndObj.first);
        }

        for (auto& nameAndIndices: ordered) {
            for (auto ix: nameAndIndices.second) {
                TypeOrPyobj obj = group->mIndexToObject.find(ix)->second;

                s << levelPrefix << " " << ix << " -> " << obj.name() << "\n";

                if (!deep) {
                    s << CompilerVisibleObjectVisitor::recordWalkAsString(
                        obj, mVisibilityType
                    ) << "\n";
                }

                if (deep) {
                    std::vector<TypeOrPyobj> visible;
                    visibleFrom(obj, visible, mVisibilityType);

                    for (auto v: visible) {
                        MutuallyRecursiveTypeGroup* subgroup;
                        int ixInSubgroup;

                        MutuallyRecursiveTypeGroup::ensureRecursiveTypeGroup(v, mVisibilityType);
                        std::tie(subgroup, ixInSubgroup) =
                            MutuallyRecursiveTypeGroup::groupAndIndexFor(v, mVisibilityType);

                        if (subgroup) {
                            if (seen.find(subgroup) == seen.end()) {
                                dump(subgroup, level + 2);
                            } else {
                                s   << levelPrefix << "  "
                                    << "group with hash "
                                    << group->hash().digestAsHexString()
                                    << " item " << ixInSubgroup << "\n";
                            }
                        }
                    }
                }
            }
        }
    };

    dump(this, 0);

    return s.str();
}

// these types can all see each other through their references, either
// through the compiler, or just through normal type references. We need to
// pick a 'first' type, which we can do by picking the first type to be defined
// in the program, and then walk through the group placing them in order..
// returns the root, and a bool that's false normally, true if we
// were unable to uniquely pick it.

// static
std::pair<TypeOrPyobj, bool> MutuallyRecursiveTypeGroup::computeRoot(
    const std::set<TypeOrPyobj>& topos,
    VisibilityType visibility
) {
    // we look at python objects in this function, so we need to be holding the GIL
    PyEnsureGilAcquired getTheGil;

    // we need to pick a single type or object to be the 'root' of the group
    // so we can order the hashes properly

    // ideally we'd use a named class, function, or barring that an alternative.
    // If we use a named type, it ensures that the type layout for an MRTG where some of the topos
    // are named is stable even if the hashes change because we modify the code.
    std::map<std::string, std::vector<TypeOrPyobj> > namedThings;

    // first see if there are any TP class or function objects
    for (auto &t: topos) {
        if (t.type() && (t.type()->isClass() || t.type()->isFunction() || t.type()->isAlternative())) {
            namedThings[t.type()->name()].push_back(t);
        }
    }

    // take the first named thing where only one thing has that name
    for (auto it = namedThings.rbegin(); it != namedThings.rend(); ++it) {
        if (it->second.size() == 1) {
            return std::make_pair(it->second[0], false);
        }
    }

    namedThings.clear();

    // now python class, function, or type objects
    for (auto &t: topos) {
        if (t.pyobj()) {
            if (PyModule_Check(t.pyobj())) {
                PyObjectStealer name(PyModule_GetNameObject(t.pyobj()));
                if (!name) {
                    PyErr_Clear();
                } else {
                    if (PyUnicode_Check(name)) {
                        namedThings[PyUnicode_AsUTF8(name)].push_back(t);
                    }
                }
            } else
            if (PyType_Check(t.pyobj())) {
                namedThings[((PyTypeObject*)t.pyobj())->tp_name].push_back(t);
            } else
            if (PyFunction_Check(t.pyobj())) {
                PyObjectStealer moduleName(PyObject_GetAttrString(t.pyobj(), "__module__"));
                if (!moduleName) {
                    PyErr_Clear();
                } else {
                    PyObjectStealer name(PyObject_GetAttrString(t.pyobj(), "__qualname__"));
                    if (!name) {
                        PyErr_Clear();
                    } else {
                        if (PyUnicode_Check(moduleName) && PyUnicode_Check(name)) {
                            namedThings[
                                std::string(PyUnicode_AsUTF8(moduleName))
                                + "."
                                + PyUnicode_AsUTF8(name)
                            ].push_back(t);
                        }
                    }
                }

            }
        }
    }

    // take the first named thing where only one thing has that name
    for (auto it = namedThings.rbegin(); it != namedThings.rend(); ++it) {
        if (it->second.size() == 1) {
            return std::make_pair(it->second[0], false);
        }
    }

    // we have nothing named. Fall back to a process that depends on the
    // hashes.
    std::map<TypeOrPyobj, ShaHash> curHashes;
    std::map<TypeOrPyobj, ShaHash> newHashes;

    for (auto& t: topos) {
        curHashes[t] = ShaHash();
    }

    // hash each type's visible inputs, using 'curHashes' as the hash
    // for any type that's in the group. After N passes, the SHA hash
    // contains information about paths through the graph of length "N"
    // and so if there is a unique element, it should be visible after
    // two passes.
    for (long k = 0; k < 2; k++) {
        for (auto& t: topos) {
            ShaHash newHash;

            if (t.type()) {
                newHash = ShaHash(t.type()->name());
            } else {
                newHash = ShaHash(std::string(t.pyobj()->ob_type->tp_name));
            }

            CompilerVisibleObjectVisitor::singleton().visit(
                t,
                visibility,
                [&](ShaHash h) { newHash += h; },
                [&](const std::string& s) { newHash += ShaHash(s); },
                [&](TypeOrPyobj t) {
                    if (curHashes.find(t) != curHashes.end()) {
                        newHash += curHashes[t];
                    } else {
                        newHash += shaHash(t, visibility);
                    }
                },
                [&](const std::string& s, TypeOrPyobj t) {
                    newHash += ShaHash(s);

                    if (curHashes.find(t) != curHashes.end()) {
                        newHash += curHashes[t];
                    } else {
                        newHash += shaHash(t, visibility);
                    }
                },
                [&](const std::string& err) {}
            );

            newHashes[t] = newHash;
        }

        curHashes = newHashes;
    }

    std::map<ShaHash, std::vector<TypeOrPyobj> > names;
    for (auto& t: topos) {
        names[curHashes[t]].push_back(t);
    }


    for (auto& nameAndTypes: names) {
        if (nameAndTypes.second.size() == 1) {
            return std::make_pair(nameAndTypes.second[0], false);
        }
    }

    // just pick the first one. It's not stable.
    for (auto& nameAndTypes: names) {
        return std::make_pair(nameAndTypes.second[0], true);
    }

    throw std::runtime_error("this should never happen");
}

// static
void MutuallyRecursiveTypeGroup::buildCompilerRecursiveGroup(
    const std::set<TypeOrPyobj>& topos,
    VisibilityType visibility
) {
    // check to see if any of our topos has been installed in a type group already. If so,
    // then they ALL should have been installed in a type group. This can happen if two threads
    // are computing type groups at the same time: imagine a cycle of many objects - both
    // threads start walking the cycle. If one finishes, it marks the entire group
    // and then the second thread, when it sees these objects, assume they're already in
    // a group and ignores them. It will then attempt to complete the second, smaller group,
    // but that group should be completely contained in whichever version finished first.
    bool anyAreAssigned = false;
    bool anyAreNotAssigned = false;

    for (auto t: topos) {
        if (MutuallyRecursiveTypeGroup::objectIsUnassigned(t, visibility)) {
            anyAreNotAssigned = true;
        } else {
            anyAreAssigned = true;
        }
    }

    if (anyAreAssigned) {
        if (anyAreNotAssigned) {
            CompilerVisibleObjectVisitor::singleton().checkForInstability(visibility);

            throw std::runtime_error(
                "Somehow we have an MRTG where some of its members "
                "were found by another group, but not all?"
            );
        }

        // there's nothing to do
        return;
    }


    MutuallyRecursiveTypeGroup* group = new MutuallyRecursiveTypeGroup(visibility);

    if (topos.size() == 0) {
        throw std::runtime_error("Empty compiler recursive group makes no sense.");
    }

    TypeOrPyobj root;
    std::tie(root, group->mAnyPyObjectsIncorrectlyOrdered) = computeRoot(topos, visibility);

    std::map<TypeOrPyobj, int32_t> ordering;

    // now walk our object graph from the root and observe the objects
    // in the order in which we find them. This should be stable across
    // program invocations and also serialization.
    std::function<void (TypeOrPyobj)> visit = [&](TypeOrPyobj parent) {
        if (topos.find(parent) == topos.end() || ordering.find(parent) != ordering.end()) {
            return;
        }

        int index = ordering.size();
        ordering[parent] = index;

        CompilerVisibleObjectVisitor::singleton().visit(
            parent,
            visibility,
            [&](ShaHash h) {},
            [&](const std::string& s) {},
            [&](TypeOrPyobj t) { visit(t); },
            [&](const std::string& s, TypeOrPyobj t) { visit(t); },
            [&](const std::string& err) {}
        );
    };

    visit(root);

    if (ordering.size() != topos.size()) {
        throw std::runtime_error("Couldn't find all the topos: " + format(ordering.size()) + " vs " + format(topos.size()));
    }

    for (auto& typeAndOrder: ordering) {
        group->mIndexToObject.insert({typeAndOrder.second, typeAndOrder.first});
        group->mObjectToIndex.insert({typeAndOrder.first, typeAndOrder.second});
    }

    group->_computeHashAndInstall();
}


MutuallyRecursiveTypeGroup* MutuallyRecursiveTypeGroup::getGroupFromHash(ShaHash hash, VisibilityType visibility) {
    std::lock_guard<std::recursive_mutex> lock(mMutex);

    auto it = mHashToGroup[visibility].find(hash);
    if (it == mHashToGroup[visibility].end()) {
        return nullptr;
    }

    return it->second;
}

MutuallyRecursiveTypeGroup* MutuallyRecursiveTypeGroup::getGroupFromIntendedHash(ShaHash hash, VisibilityType visibility) {
    std::lock_guard<std::recursive_mutex> lock(mMutex);

    auto it = mIntendedHashToGroup[visibility].find(hash);
    if (it != mIntendedHashToGroup[visibility].end()) {
        return it->second;
    }

    auto it2 = mHashToGroup[visibility].find(hash);
    if (it2 != mHashToGroup[visibility].end()) {
        return it2->second;
    }

    return nullptr;
}

void MutuallyRecursiveTypeGroup::computeHash() {
    if (mAnyPyObjectsIncorrectlyOrdered) {
        mHash = ShaHash::poison();
        return;
    }

    // the group we're already hashing on this thread. This should always be the nullptr
    // since the current hashing model insists that all objects below this MRTG are either
    // in the MRTG or are in MRTGs that have already been hashed and installed.
    thread_local static MutuallyRecursiveTypeGroup* currentlyHashing = nullptr;

    if (currentlyHashing) {
        CompilerVisibleObjectVisitor::singleton().checkForInstability(mVisibilityType);

        std::ostringstream errMsg;

        errMsg << "Somehow we are already computing the hash of another MRTG. \n\n";

        errMsg << currentlyHashing->repr(false) << "\n\n";
        errMsg << this->repr(false) << "\n";

        throw std::runtime_error(errMsg.str());
    }

    try {
        // mark that we're currently hashing.
        currentlyHashing = this;

        // we are a recursive group head. We want to compute the hash
        // of all of our constituents where, when each of them looks at
        // other topos within _our_ group, we simply hash in a placeholder
        // for the position.
        ShaHash wholeGroupHash;

        for (auto idAndObj: mIndexToObject) {
            wholeGroupHash += computeTopoHash(idAndObj.second);
        }

        mHash = wholeGroupHash;
        currentlyHashing = nullptr;
    } catch(...) {
        currentlyHashing = nullptr;
        throw;
    }
}

ShaHash MutuallyRecursiveTypeGroup::computeTopoHash(TypeOrPyobj toHash) {
    ShaHash res;

    if (toHash.type()) {
        res += ShaHash(1, toHash.type()->getTypeCategory());
        if (mObjectToIndex.size() != 1) {
            res += ShaHash(toHash.type()->name());
        }
    } else {
        res += ShaHash(2);
    }

    auto visitTopo = [&](TypeOrPyobj o) {
        if (o.pyobj() && CompilerVisibleObjectVisitor::isSimpleConstant(o.pyobj())) {
            res += ShaHash(1) + shaHashOfSimplePyObjectConstant(o.pyobj());
            return;
        }

        // this object must either be in our group, or must already be
        // installed as a type hash
        auto it = mObjectToIndex.find(o);
        if (it != mObjectToIndex.end()) {
            res += ShaHash(3, it->second);
            return;
        }

        auto groupAndIx = groupAndIndexFor(o, mVisibilityType);

        if (!groupAndIx.first || groupAndIx.second == -1) {
            CompilerVisibleObjectVisitor::singleton().checkForInstability(mVisibilityType);

            throw std::runtime_error(
                "All reachable objects should be in this group or hashed already. "
                "Instead, this object is not hashed and not local: " + o.name() +
                (groupAndIx.first ? "\n\nhas a group but not an index\n\n":"") +
                + "\n\nwithin\n\n" + toHash.name() + "\n\nwalk is\n"
                + CompilerVisibleObjectVisitor::recordWalkAsString(toHash, mVisibilityType)
                + "\n\nwalk of not hashed is\n"
                + CompilerVisibleObjectVisitor::recordWalkAsString(o, mVisibilityType)
                + "\nVT = " + visibilityTypeToStr(mVisibilityType)
            );
        }

        res += ShaHash(4) + groupAndIx.first->hash() + ShaHash(groupAndIx.second);
    };

    // walk one layer deep into our objects
    CompilerVisibleObjectVisitor::singleton().visit(
        toHash,
        mVisibilityType,
        [&](ShaHash h) { res += ShaHash(1) + h; },
        [&](std::string o) { res += ShaHash(2) + ShaHash(o); },
        [&](TypeOrPyobj o) {
            visitTopo(o);
        },
        [&](std::string name, TypeOrPyobj o) {
            res += ShaHash(5) + ShaHash(name);
            visitTopo(o);
        },
        [&](const std::string& err) {}
    );

    return res;
}

// find strongly-connected groups of python objects and Type objects (as far as the
// compiler is concerned)
class MutuallyRecursiveTypeGroupSearch {
public:
    MutuallyRecursiveTypeGroupSearch(VisibilityType visibility) :
        mVisibilityType(visibility)
    {}

    // this object is new and needs its own group
    void pushGroup(TypeOrPyobj o) {
        if (mInAGroup.find(o) != mInAGroup.end()) {
            throw std::runtime_error("Can't create a new group for an object we're already considering");
        }

        mInAGroup.insert(o);

        mGroups.push_back(
            std::shared_ptr<std::set<TypeOrPyobj> >(
                new std::set<TypeOrPyobj>()
            )
        );

        mGroupOutboundEdges.push_back(
            std::shared_ptr<std::vector<TypeOrPyobj> >(
                new std::vector<TypeOrPyobj>()
            )
        );

        mGroups.back()->insert(o);

        // now recurse into the subtypes
        CompilerVisibleObjectVisitor::singleton().visit(
            o,
            mVisibilityType,
            [&](ShaHash h) {},
            [&](std::string o) {},
            [&](TypeOrPyobj o) {
                if (MutuallyRecursiveTypeGroup::objectIsUnassigned(o, mVisibilityType)) {
                    mGroupOutboundEdges.back()->push_back(o);
                }
            },
            [&](std::string name, TypeOrPyobj o) {
                if (MutuallyRecursiveTypeGroup::objectIsUnassigned(o, mVisibilityType)) {
                    mGroupOutboundEdges.back()->push_back(o);
                }
            },
            [&](const std::string& err) {}
        );
    }

    void findAllGroups() {
        while (mGroups.size()) {
            doOneStep();
        }
    }

    void doOneStep() {
        if (mGroupOutboundEdges.back()->size() == 0) {
            // this group is finished....
            MutuallyRecursiveTypeGroup::buildCompilerRecursiveGroup(*mGroups.back(), mVisibilityType);

            for (auto gElt: *mGroups.back()) {
                mInAGroup.erase(gElt);
            }

            mGroups.pop_back();
            mGroupOutboundEdges.pop_back();
        } else {
            // pop an outbound edge and see where does it go?
            TypeOrPyobj o = mGroupOutboundEdges.back()->back();

            mGroupOutboundEdges.back()->pop_back();

            if (!MutuallyRecursiveTypeGroup::objectIsUnassigned(o, mVisibilityType)) {
                return;
            }

            if (mInAGroup.find(o) == mInAGroup.end()) {
                // this is new
                pushGroup(o);
            } else {
                // this is above us somewhere
                while (mGroups.size() && mGroups.back()->find(o) == mGroups.back()->end()) {
                    // merge these two mGroups together
                    mGroups[mGroups.size() - 2]->insert(
                        mGroups.back()->begin(), mGroups.back()->end()
                    );
                    mGroups.pop_back();

                    mGroupOutboundEdges[mGroupOutboundEdges.size() - 2]->insert(
                        mGroupOutboundEdges[mGroupOutboundEdges.size() - 2]->end(),
                        mGroupOutboundEdges.back()->begin(),
                        mGroupOutboundEdges.back()->end()
                    );
                    mGroupOutboundEdges.pop_back();
                }

                if (!mGroups.size()) {
                    // this shouldn't ever happen, but we want to know if it does
                    throw std::runtime_error("Never found parent of object!");
                }
            }
        }
    }

private:
    // everything in a group somewhere
    std::set<TypeOrPyobj> mInAGroup;

    // each group
    std::vector<std::shared_ptr<std::set<TypeOrPyobj> > > mGroups;

    // for each group, the set of things it reaches out to
    std::vector<std::shared_ptr<std::vector<TypeOrPyobj> > > mGroupOutboundEdges;

    VisibilityType mVisibilityType;
};

// static
void MutuallyRecursiveTypeGroup::ensureRecursiveTypeGroup(TypeOrPyobj root, VisibilityType visibility) {
    // we do things with pyobj refcounts, so we need to hold the gil.
    PyEnsureGilAcquired getTheGil;

    MutuallyRecursiveTypeGroupSearch groupFinder(visibility);

    static thread_local int count = 0;
    count++;
    if (count > 1) {
        throw std::runtime_error(
            "There should be only one group algo running at once. Somehow, "
            "our reference to " + root.name() + " wasn't captured correctly."
        );
    }
    try {
        groupFinder.pushGroup(root);

        groupFinder.findAllGroups();
    } catch(...) {
        count --;
        throw;
    }
    count--;
}

ShaHash MutuallyRecursiveTypeGroup::shaHash(TypeOrPyobj o, VisibilityType visibility) {
    if (o.pyobj() && CompilerVisibleObjectVisitor::isSimpleConstant(o.pyobj())) {
        return shaHashOfSimplePyObjectConstant(o.pyobj());
    }

    auto groupAndIx = MutuallyRecursiveTypeGroup::groupAndIndexFor(o, visibility);

    if (!groupAndIx.first) {
        return ShaHash();
    }

    return ShaHash(2) + groupAndIx.first->hash() + ShaHash(groupAndIx.second);
}

// static
std::pair<MutuallyRecursiveTypeGroup*, int> MutuallyRecursiveTypeGroup::groupAndIndexFor(
    PyObject* o, VisibilityType visibility
) {
    return groupAndIndexFor(TypeOrPyobj::withoutIntern(o), visibility);
}

// static
std::pair<MutuallyRecursiveTypeGroup*, int> MutuallyRecursiveTypeGroup::groupAndIndexFor(
    TypeOrPyobj o,
    VisibilityType visibility
) {
    std::lock_guard<std::recursive_mutex> lock(mMutex);

    auto it = mTypeGroups[visibility].find(o);
    if (it != mTypeGroups[visibility].end()) {
        return std::pair<MutuallyRecursiveTypeGroup*, int>(
            it->second.first, it->second.first->indexOfObjectInThisGroup(o)
        );
    }

    return {nullptr, 0};
}

ShaHash MutuallyRecursiveTypeGroup::shaHashOfSimplePyObjectConstant(PyObject* h) {
    PyEnsureGilAcquired getTheGil;

    // handle basic constants
    if (h == Py_None) {
        return ShaHash(100, 1);
    }

    if (h == Py_True) {
        return ShaHash(100, 2);
    }

    if (h == Py_False) {
        return ShaHash(100, 3);
    }

    if (PyLong_Check(h)) {
        ShaHash hash(100, 4);

        size_t l = PyLong_AsLong(h);
        if (!(l == -1 && PyErr_Occurred())) {
            return hash + ShaHash(100, l);
        }

        PyErr_Clear();

        // this is an overflow
        PyObjectHolder lng(h);

        while (true) {
            int overflow = 0;

            long l = PyLong_AsLongAndOverflow(lng, &overflow);

            if (l == -1 && PyErr_Occurred()) {
                throw PythonExceptionSet();
            }

            hash += ShaHash(l);
            if (overflow == 0) {
                return hash;
            }

            static PyObject* thirtyTwo = PyLong_FromLong(32);

            lng.set(PyNumber_Rshift(lng, thirtyTwo));
        }

        return hash;
    }

    if (PyBytes_Check(h)) {
        return ShaHash(100, 5) + ShaHash::SHA1(
            PyBytes_AsString(h),
            PyBytes_GET_SIZE(h)
        );
    }

    if (PyUnicode_Check(h)) {
        Py_ssize_t s;
        const char* c = PyUnicode_AsUTF8AndSize(h, &s);

        return ShaHash(100, 6) + ShaHash::SHA1(c, s);
    }

    PyObject* builtinsModule = staticPythonInstance("builtins", "");
    PyObject* builtinsModuleDict = staticPythonInstance("builtins", "__dict__");

    if (h == builtinsModule) {
        return ShaHash(100, 8);
    }

    if (h == builtinsModuleDict) {
        return ShaHash(100, 9);
    }

    // 'object'
    if (h == (PyObject*)PyDict_Type.tp_base) {
        return ShaHash(100, 10);
    }
    // 'type'
    if (h == (PyObject*)&PyType_Type) {
        return ShaHash(100, 11);
    }

    if (h == (PyObject*)&PyDict_Type) {
        return ShaHash(100, 12);
    }

    if (h == (PyObject*)&PyList_Type) {
        return ShaHash(100, 13);
    }

    if (h == (PyObject*)&PySet_Type) {
        return ShaHash(100, 14);
    }

    if (h == (PyObject*)&PyLong_Type) {
        return ShaHash(100, 15);
    }

    if (h == (PyObject*)&PyUnicode_Type) {
        return ShaHash(100, 16);
    }

    if (h == (PyObject*)&PyFloat_Type) {
        return ShaHash(100, 17);
    }

    if (h == (PyObject*)&PyBytes_Type) {
        return ShaHash(100, 18);
    }

    if (h == (PyObject*)&PyBool_Type) {
        return ShaHash(100, 19);
    }

    if (h == (PyObject*)Py_None->ob_type) {
        return ShaHash(100, 20);
    }

    if (h == (PyObject*)&PyProperty_Type) {
        return ShaHash(100, 21);
    }

    if (h == (PyObject*)&PyClassMethodDescr_Type) {
        return ShaHash(100, 22);
    }

    if (h == (PyObject*)&PyGetSetDescr_Type) {
        return ShaHash(100, 23);
    }

    if (h == (PyObject*)&PyMemberDescr_Type) {
        return ShaHash(100, 24);
    }

    if (h == (PyObject*)&PyMethodDescr_Type) {
        return ShaHash(100, 25);
    }

    if (h == (PyObject*)&PyWrapperDescr_Type) {
        return ShaHash(100, 26);
    }

    if (h == (PyObject*)&PyDictProxy_Type) {
        return ShaHash(100, 27);
    }

    if (h == (PyObject*)&_PyMethodWrapper_Type) {
        return ShaHash(100, 28);
    }

    if (h == (PyObject*)&PyCFunction_Type) {
        return ShaHash(100, 29);
    }

    if (h == (PyObject*)&PyFunction_Type) {
        return ShaHash(100, 30);
    }

    if (PyFloat_Check(h)) {
        double d = PyFloat_AsDouble(h);

        return ShaHash(100, 31) + ShaHash::SHA1((void*)&d, sizeof(d));
    }

    if (h->ob_type == &PyProperty_Type
        || h->ob_type == &PyGetSetDescr_Type
        || h->ob_type == &PyMemberDescr_Type
        || h->ob_type == &PyWrapperDescr_Type
        || h->ob_type == &PyDictProxy_Type
        || h->ob_type == &_PyMethodWrapper_Type
    ) {
        // the compiler doesn't look inside of these, so we don't need to
        // understand what goes on inside of them.
        return ShaHash(100, 32);
    }

    return ShaHash();
}

//static
std::map<
    VisibilityType,
    std::unordered_map<TypeOrPyobj, std::pair<MutuallyRecursiveTypeGroup*, int> >
> MutuallyRecursiveTypeGroup::mTypeGroups;

//static
std::map<
    VisibilityType,
    std::map<ShaHash, MutuallyRecursiveTypeGroup*>
> MutuallyRecursiveTypeGroup::mHashToGroup;

//static
std::map<
    VisibilityType,
    std::map<ShaHash, MutuallyRecursiveTypeGroup*>
> MutuallyRecursiveTypeGroup::mIntendedHashToGroup;

//static
std::recursive_mutex MutuallyRecursiveTypeGroup::mMutex;
