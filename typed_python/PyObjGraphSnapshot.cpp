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

#include "PyObjGraphSnapshot.hpp"
#include "PyObjSnapshotGroupSearch.hpp"

PyObjGraphSnapshot::PyObjGraphSnapshot(Type* root, bool linkBak) :
    mGroupsConsumed(0),
    mGroupSearch(this)
{
    std::unordered_map<PyObject*, PyObjSnapshot*> objMapCache;
    std::unordered_map<Type*, PyObjSnapshot*> typeMapCache;
    std::unordered_map<InstanceRef, PyObjSnapshot*> instanceCache;

    // this finds every reachable python objects from 'root', with the search terminating at
    // named module objects and named module dicts. This is enough information to completely
    // characterize a type's "identity hash", since there can be at most one version of a
    // given named module per program.
    PyObjSnapshotMaker snapMaker(
        objMapCache,
        typeMapCache,
        instanceCache,
        this,
        linkBak
    );

    snapMaker.internalize(root);
}

/* walk over the graph and point any forwards to the type they actually point to.

This will modify nodes in the following way:
    Any Kind::ForwardType will have a target set if its resolvable
    Any snapshot pointing at a Forward will now point at what the forward points to
    Any snapshot with a valid cache that can reach a type that was changed will have its
        cache cleared

*/
void PyObjGraphSnapshot::resolveForwards() {
    // any snapshot that was modified
    std::set<PyObjSnapshot*> modified;

    // point forwards where they go
    for (auto o: mObjects) {
        if (o->getKind() == PyObjSnapshot::Kind::ForwardType) {
            o->pointForwardToFinalType();
            modified.insert(o);
        }

        if (o->willBeATpType()) {
            if (o->markTypeNotFwdDefined()) {
                modified.insert(0);
            }
        }
    }

    for (auto o: mObjects) {
        if (o->replaceOutboundForwardsWithTargets()) {
            modified.insert(o);
        }
    }

    // build a reverse graph map
    std::map<PyObjSnapshot*, std::set<PyObjSnapshot*> > inbound;

    for (auto o: mObjects) {
        o->visitOutbound([&](PyObjSnapshot* outbound) {
            if (outbound->getGraph() == this) {
                inbound[outbound].insert(o);
            }
        });
    }

    std::set<PyObjSnapshot*> modifiedTransitive;
    std::function<void (PyObjSnapshot*)> check = [&](PyObjSnapshot* s) {
        if (modifiedTransitive.find(s) == modifiedTransitive.end()) {
            modifiedTransitive.insert(s);
            for (auto upstream: inbound[s]) {
                check(upstream);
            }
        }
    };

    for (auto m: modified) {
        check(m);
    }

    for (auto m: modifiedTransitive) {
        m->clearCache();
    }
}


void PyObjGraphSnapshot::internalize() {
    std::map<ShaHash, std::pair<PyObjSnapshot*, PyObjSnapshot*> > skeletons;

    for (auto s: mObjects) {
        ShaHash h = hashFor(s);

        if (!internal().snapshotForHash(h)) {
             skeletons[h] = std::make_pair(s, internal().createSkeleton(h));
        }
    }

    for (auto hashAndSkeleton: skeletons) {
        hashAndSkeleton.second.second->cloneFromSnapByHash(hashAndSkeleton.second.first);
    }

    // rehydrate this portion of the graph, so that we have concrete objects for everything
    // we just interned
    for (auto hashAndSkeleton: skeletons) {
        hashAndSkeleton.second.second->rehydrate();
    }

    for (auto hashAndSkeleton: skeletons) {
        hashAndSkeleton.second.second->markInternalizedOnType();
    }
}

PyObjSnapshot* PyObjGraphSnapshot::createSkeleton(ShaHash h) {
    if (mHashToSnap.find(h) != mHashToSnap.end()) {
        throw std::runtime_error("Can't create a skeleton for an object that already exists");
    }

    PyObjSnapshot* snap = new PyObjSnapshot(this);

    mHashToSnap[h] = snap;
    mObjects.insert(snap);
    return snap;
}


template<class compute_type>
ShaHash computeHashFor(PyObjSnapshot* snap, const compute_type& compute) {
    if (snap->getKind() == PyObjSnapshot::Kind::Instance) {
        if (snap->getInstance().type()->isPOD()) {
            return ShaHash(int(snap->getKind()))
                + ShaHash::SHA1(snap->getInstance().data(), snap->getInstance().type()->bytecount());
        }
        throw std::runtime_error(
            "Can't hash a PyObjSnapshot Instance of type " + snap->getInstance().type()->name()
        );
    }
    if (snap->getKind() == PyObjSnapshot::Kind::PrimitiveType) {
        return ShaHash(int(snap->getKind())) + ShaHash(int(snap->getType()->getTypeCategory()));
    }
    if (snap->getKind() == PyObjSnapshot::Kind::ArbitraryPyObject) {
        // probably we should try to serialize it?
        throw std::runtime_error(
            "Can't hash a PyObjSnapshot.ArbitraryPyObject of type "
            + std::string(snap->getPyObj()->ob_type->tp_name)
        );
    }

    ShaHash res(int(snap->getKind()));

    if (snap->getStringValue().size()) {
        res = res + ShaHash(1) + ShaHash(snap->getStringValue());
    }
    if (snap->getName().size()) {
        res = res + ShaHash(2) + ShaHash(snap->getName());
    }
    if (snap->getModuleName().size()) {
        res = res + ShaHash(3) + ShaHash(snap->getModuleName());
    }
    if (snap->getQualname().size()) {
        res = res + ShaHash(4) + ShaHash(snap->getQualname());
    }
    if (snap->names().size()) {
        res = res + ShaHash(6) + ShaHash(snap->names().size());

        for (auto n: snap->names()) {
            res = res + ShaHash(n);
        }
    }
    if (snap->elements().size()) {
        res = res + ShaHash(7) + ShaHash(snap->elements().size());

        for (auto n: snap->elements()) {
            res = res + compute(n);
        }
    }
    if (snap->keys().size()) {
        res = res + ShaHash(8) + ShaHash(snap->keys().size());

        for (auto n: snap->keys()) {
            res = res + compute(n);
        }
    }
    if (snap->namedElements().size()) {
        res = res + ShaHash(10) + ShaHash(snap->namedElements().size());

        for (auto& nameAndElt: snap->namedElements()) {
            res = res + ShaHash(nameAndElt.first) + compute(nameAndElt.second);
        }
    }
    if (snap->namedInts().size()) {
        res = res + ShaHash(11) + ShaHash(snap->namedInts().size());

        for (auto& nameAndElt: snap->namedInts()) {
            res = res + ShaHash(nameAndElt.first) + ShaHash(nameAndElt.second);
        }
    }

    return res;
}


PyObjSnapshot* PyObjGraphSnapshot::snapshotForHash(ShaHash hash) {
    auto it = mHashToSnap.find(hash);
    if (it != mHashToSnap.end()) {
        return it->second;
    }
    return nullptr;
}


ShaHash PyObjGraphSnapshot::hashFor(PyObjSnapshot* snap) {
    if (snap->getGraph() != this) {
        throw std::runtime_error("Can't graph a hash of a PyObjSnapshot we don't have");
    }

    auto it = mSnapToHash.find(snap);
    if (it != mSnapToHash.end()) {
        return it->second;
    }

    mGroupSearch.add(snap);

    while (mGroupsConsumed < mGroupSearch.getGroups().size()) {
        computeHashesFor(*mGroupSearch.getGroups()[mGroupsConsumed++]);
    }

    it = mSnapToHash.find(snap);

    if (it != mSnapToHash.end()) {
        return it->second;
    }

    throw std::runtime_error("PyObjGraphSnapshot failed to produce a hash for one of its objects.");
}

void PyObjGraphSnapshot::computeHashesFor(const std::unordered_set<PyObjSnapshot*>& group) {
    PyObjSnapshot* root = pickRootFor(group);

    // order our snapshots lexically
    std::unordered_map<PyObjSnapshot*, int> snapsSeen;
    std::vector<PyObjSnapshot*> snapsInOrder;

    std::function<void (PyObjSnapshot*)> visit = [&](PyObjSnapshot* s) {
        // if its not in the group, ignore it
        if (group.find(s) == group.end()) {
            return;
        }

        // if we saw it already, skip it
        if (snapsSeen.find(s) != snapsSeen.end()) {
            return;
        }

        // record it
        int snapIx = snapsSeen.size();
        snapsSeen[s] = snapIx;
        snapsInOrder.push_back(s);

        // visit our outbound edges
        s->visitOutbound(visit);
    };

    visit(root);

    if (snapsSeen.size() != group.size()) {
        throw std::runtime_error("PyObjGraphSnapshot::computeHashFor failed to find all snaps");
    }

    ShaHash groupHash = ShaHash(snapsInOrder.size());

    for (long i = 0; i < snapsInOrder.size(); i++) {
        ShaHash thisSnapHash = computeHashFor(
            snapsInOrder[i],
            [&](PyObjSnapshot* snap) {
                auto it = snapsSeen.find(snap);
                if (it != snapsSeen.end()) {
                    return ShaHash(1) + ShaHash(it->second);
                }
                auto it2 = mSnapToHash.find(snap);
                if (it2 != mSnapToHash.end()) {
                    return ShaHash(2) + it2->second;
                }

                throw std::runtime_error("PyObjGraphSnapshot::computeHashFor failed to find a snap.");
            }
        );

        groupHash = groupHash + ShaHash(i) + thisSnapHash;
    }

    for (long i = 0; i < snapsInOrder.size(); i++) {
        installSnapHash(snapsInOrder[i], groupHash + ShaHash(i));
    }
}

void PyObjGraphSnapshot::installSnapHash(PyObjSnapshot* snap, ShaHash h) {
    mSnapToHash[snap] = h;

    if (!mHashToSnap[h]) {
        mHashToSnap[h] = snap;
    } else {
        // this can happen any time in a single graph we have multiple copies of objects that
        // look the same (as far as what can be seen from their references). As an example,
        // the empty tuple or empty list will always be the same hash in any graph
        mRedundantSnaps.insert(snap);
    }
}

PyObjSnapshot* PyObjGraphSnapshot::pickRootFor(const std::unordered_set<PyObjSnapshot*>& group) {
    // pick a 'root' node by computing a shallow hash and taking the first value for which
    // there is only one hash
    std::unordered_map<PyObjSnapshot*, ShaHash> curHashes;
    std::unordered_map<PyObjSnapshot*, ShaHash> newHashes;
    long passes = 0;

    while (passes < group.size() + 1) {
        for (auto s: group) {
            ShaHash h = computeHashFor(s, [&](PyObjSnapshot* edge) {
                if (group.find(edge) == group.end()) {
                    auto it = mSnapToHash.find(edge);
                    if (it != mSnapToHash.end()) {
                        return it->second;
                    }

                    throw std::runtime_error("Found a PyObjSnapshot that's not in our graph");
                }

                return curHashes[s];
            });

            newHashes[s] = h;
        }

        // find the first hash that's unique
        std::map<ShaHash, std::set<PyObjSnapshot*> > toHash;
        for (auto snapAndHash: newHashes) {
            toHash[snapAndHash.second].insert(snapAndHash.first);
        }

        for (auto& hashAndObj: toHash) {
            if (hashAndObj.second.size() == 1) {
                return *hashAndObj.second.begin();
            }
        }
    }

    // we didn't find one even after N passes, so our objects look identical.
    // just pick one.
    return *group.begin();
}

void PyObjGraphSnapshot::getTypes(std::unordered_set<PyObjSnapshot*>& outTypes) {
    for (auto snap: mObjects) {
        if (snap->willBeATpType()) {
            outTypes.insert(snap);
        }
    }
}
