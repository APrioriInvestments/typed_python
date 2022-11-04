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

#include "TypeOrPyobj.hpp"
#include "ShaHash.hpp"
#include <map>
#include <unordered_map>


class MutuallyRecursiveTypeGroupSearch;


// represents a group of Type* and PyObject* instances that can 'see' each other according
// to the compiler. These instances need to be serialized and hashed as a group.  The hash
// of the instances depends on the hash of the group, and the hash of the group depends on
// the set of instances contained inside of it.
class MutuallyRecursiveTypeGroup {
public:
    //return an empty 'builder' group. Builder groups can be built up and then
    //installed into the graph. You're not guaranteed that your copy of the builder group
    //is the one we take, since (a) there could be two threads trying to build this particular
    //MRTG, or (b) you could be deserializing data from a version of your codebase where
    //the code has changed but the types are stable and you still want to deserialize objects
    //even though the hashes in the core data will have changed.

    //Regardless, you're guaranteed that after you finalize this group, there will be only
    //one copy of the types or singleton objects it contains. The types you put in may be
    //discarded, so don't keep a reference to them.
    static MutuallyRecursiveTypeGroup* DeserializerGroup(ShaHash intendedHash);

    bool isDeserializerGroup() const {
        return mIsDeserializerGroup;
    }

    void finalizeDeserializerGroup();

    // assuming we're a builder group, set the value of one of our objects. These need
    // not be the final versions of the objects.
    void setIndexToObject(int32_t index, TypeOrPyobj obj);

    // is this object in this group? If this is a native type, we'll unpack it and check that.
    // returns -1 if its not.
    int32_t indexOfObjectInThisGroup(TypeOrPyobj o);

    ShaHash hash() const {
        return mHash;
    }

    const std::map<int32_t, TypeOrPyobj>& getIndexToObject() const {
        return mIndexToObject;
    }

    std::string repr(bool deep=false);

    // if we have an official MRTG for this hash, return it.
    static MutuallyRecursiveTypeGroup* getGroupFromHash(ShaHash h);

    // if we have an MRTG with this 'intended' hash, return that.
    static MutuallyRecursiveTypeGroup* getGroupFromIntendedHash(ShaHash h);

    // construct an MRTG on this. After this call, this object will be in the type memo
    // and calls to 'groupAndIndexFor' and shaHash will succeed
    static void ensureRecursiveTypeGroup(TypeOrPyobj root);

    // return the current group head, assuming one has been created. If you haven't called
    // ensureRecursiveTypeGroup, you may get a nullptr for the group. Note that creating
    // a TypeOrPyobj will leak a reference to 'o', so don't do it on any old object.
    static std::pair<MutuallyRecursiveTypeGroup*, int> groupAndIndexFor(TypeOrPyobj o);

    // lookup an object without increffing it
    static std::pair<MutuallyRecursiveTypeGroup*, int> groupAndIndexFor(PyObject* o);

    // return the current sha-hash for this object, or ShaHash() if its not in a type
    // group yet.
    static ShaHash shaHash(TypeOrPyobj t);

    // ensure this object is in a type group and then return the set of reachable instances
    static void visibleFrom(TypeOrPyobj root, std::vector<TypeOrPyobj>& outReachable);

private:
    MutuallyRecursiveTypeGroup();

    MutuallyRecursiveTypeGroup(ShaHash hash);

    static ShaHash shaHashOfSimplePyObjectConstant(PyObject* h);

    static bool objectIsUnassigned(TypeOrPyobj obj);

    ShaHash computeTopoHash(TypeOrPyobj t);

    void _computeHashAndInstall();

    static void buildCompilerRecursiveGroup(const std::set<TypeOrPyobj>& types);

    static void installTypeHash(Type* t);

    static ShaHash pyObjectShaHashByVisiting(PyObject* obj, MutuallyRecursiveTypeGroup* groupHead);

    // given a collection of TypeOrPyobj, figure out which one is the 'root' for the group
    // this must be stable regardless of the ordering.
    static std::pair<TypeOrPyobj, bool> computeRoot(const std::set<TypeOrPyobj>& types);

    // static mutex that guards all the relevant data structures. Changes to all static
    // structures below must be made while holding the mutex. You may not hold the mutex
    // and call back into any python code of any kind, since that can produce deadlocks.
    static std::recursive_mutex mMutex;

    // for each python object that's in a type group, the group and index within the group
    static std::unordered_map<TypeOrPyobj, std::pair<MutuallyRecursiveTypeGroup*, int> > mTypeGroups;

    static std::map<ShaHash, MutuallyRecursiveTypeGroup*> mIntendedHashToGroup;

    static std::map<ShaHash, MutuallyRecursiveTypeGroup*> mHashToGroup;

    void computeHash();

    // is this a builder group? If so, we can modify it and types may be forwards
    // if not, we can't modify it and it may or may not be officially installed.
    bool mIsDeserializerGroup;

    // the sha hash of this group within this codebase
    ShaHash mHash;

    // the sha-hash of this group within the codebase from which it originated.
    // if this is not equal to mHash, then this group was a builder group that didn't
    // match, and will not be the official group for any type. It may, however, be
    // returned by the memo for this same 'intended' hash so that we don't need to
    // actually do the work of fully deserializing the relevant types.
    ShaHash mIntendedHash;

    std::map<int32_t, TypeOrPyobj> mIndexToObject;

    std::unordered_map<TypeOrPyobj, int32_t> mObjectToIndex;

    bool mAnyPyObjectsIncorrectlyOrdered;

    friend class MutuallyRecursiveTypeGroupSearch;
};
