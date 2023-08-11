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

#include "PyObjSnapshot.hpp"

// find strongly-connected groups of python objects and Type objects (as far as the
// compiler is concerned)
class PyObjSnapshotGroupSearch {
public:
    PyObjSnapshotGroupSearch(PyObjGraphSnapshot* graph) : mGraph(graph)
    {
    }

    void add(PyObjSnapshot* o, bool insistNew=false) {
        if (mSnapToOutGroupIx.find(o) != mSnapToOutGroupIx.end()) {
            if (insistNew) {
                throw std::runtime_error("We've already seen this element");
            } else {
                return;
            }
        }

        pushGroup(o);
        findAllGroups();
    }

    const std::vector<std::shared_ptr<std::unordered_set<PyObjSnapshot*> > >& getGroups() const {
        return mOutGroups;
    }

private:
    void pushGroup(PyObjSnapshot* o) {
        if (mInAGroup.find(o) != mInAGroup.end()) {
            return;
        }

        if (mSnapToOutGroupIx.find(o) != mSnapToOutGroupIx.end()) {
            return;
        }

        mInAGroup.insert(o);

        mGroups.push_back(
            std::shared_ptr<std::unordered_set<PyObjSnapshot*> >(
                new std::unordered_set<PyObjSnapshot*>()
            )
        );

        mGroupOutboundEdges.push_back(
            std::shared_ptr<std::vector<PyObjSnapshot*> >(
                new std::vector<PyObjSnapshot*>()
            )
        );

        mGroups.back()->insert(o);

        o->visitOutbound([&](PyObjSnapshot* snap) {
            if (snap->getGraph() == mGraph) {
                mGroupOutboundEdges.back()->push_back(snap);
            }
        });
    }

    void findAllGroups() {
        while (mGroups.size()) {
            doOneStep();
        }
    }

    void doOneStep() {
        if (mGroupOutboundEdges.back()->size() == 0) {
            // this group is finished....
            mOutGroups.push_back(mGroups.back());

            for (auto gElt: *mGroups.back()) {
                mSnapToOutGroupIx[gElt] = mOutGroups.size() - 1;
                mInAGroup.erase(gElt);
            }

            mGroups.pop_back();
            mGroupOutboundEdges.pop_back();
        } else {
            // pop an outbound edge and see where does it go?
            PyObjSnapshot* o = mGroupOutboundEdges.back()->back();

            mGroupOutboundEdges.back()->pop_back();

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
    PyObjGraphSnapshot* mGraph;

    // everything in a group somewhere
    std::unordered_set<PyObjSnapshot*> mInAGroup;

    // each group
    std::vector<std::shared_ptr<std::unordered_set<PyObjSnapshot*> > > mGroups;

    // for each group, the set of things it reaches out to
    std::vector<std::shared_ptr<std::vector<PyObjSnapshot*> > > mGroupOutboundEdges;

    // the final groups we ended up with
    std::vector<std::shared_ptr<std::unordered_set<PyObjSnapshot*> > > mOutGroups;

    std::unordered_map<PyObjSnapshot*, size_t> mSnapToOutGroupIx;
};
