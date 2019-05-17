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

#pragma once

#include <unordered_map>
#include <unordered_set>

#include "DictInstance.hpp"
#include "Common.hpp"
#include "HashFunctions.hpp"

/*************

VersionedObjectsOfMultiType stores a collection of TypedPython objects that
represent the same database field, but that can have different representations
in different codebases.

*************/

class VersionedObjectsOfMultiType {
    enum { NO_TRANSACTION = -1 };

    class ObjectAndVersion {
    public:
        ObjectAndVersion(object_id objId, transaction_id verId) :
                objectId(objId),
                version(verId)
        {
        }

        object_id objectId;
        transaction_id version;
    };

    class NextAndPrior {
    public:
        object_id nextId;
        object_id priorId;
    };

    class ObjectData {
    public:
        unsigned char object_data[8]; //can be any number, but need something since otherwise this is empty
    };

public:
    VersionedObjectsOfMultiType(field_id in_field_id) :
            m_field_id(in_field_id),
            m_guaranteed_lowest_id(NO_TRANSACTION)
    {
    }

    bool empty() const {
        return m_serialized_values.size() == 0 && m_deleted_values.size() == 0;
    }

    transaction_id getGuaranteedLowestId() const {
        return m_guaranteed_lowest_id;
    }

    transaction_id getBottomTid(object_id objectId) {
        auto min_max_it = m_min_max_versions.find(objectId);
        if (min_max_it == m_min_max_versions.end()) {
            return NO_TRANSACTION;
        }
        return min_max_it->second.first;
    }

    transaction_id getTopTid(object_id objectId) {
        auto min_max_it = m_min_max_versions.find(objectId);
        if (min_max_it == m_min_max_versions.end()) {
            return NO_TRANSACTION;
        }
        return min_max_it->second.second;
    }

    transaction_id nextTid(object_id objectId, transaction_id tid) {
        auto it = m_prior_and_next.find(std::make_pair(objectId, tid));

        if (it == m_prior_and_next.end()) {
            return NO_TRANSACTION;
        }

        return it->second.second;
    }

    transaction_id priorTid(object_id objectId, transaction_id tid) {
        auto it = m_prior_and_next.find(std::make_pair(objectId, tid));

        if (it == m_prior_and_next.end()) {
            return NO_TRANSACTION;
        }

        return it->second.first;
    }

    void moveGuaranteedLowestIdForward(transaction_id t) {
        if (t <= m_guaranteed_lowest_id) {
            return;
        }

        m_guaranteed_lowest_id = t;

        while (m_version_numbers_to_check.size() && m_version_numbers_to_check.begin()->first < t) {
            // this is the ID we're consuming, which is the lowest id mentioned in the
            // entire object.
            transaction_id lowestId = m_version_numbers_to_check.begin()->first;

            std::set<object_id> toCheck;
            std::swap(toCheck, m_version_numbers_to_check.begin()->second);
            m_version_numbers_to_check.erase(lowestId);

            for (auto objectId: toCheck) {
                removeLowestIfPossible(objectId);
            }
        }
    }

    transaction_id bestTransactionId(object_id objectId, transaction_id version) {
        if (version < m_guaranteed_lowest_id) {
            return NO_TRANSACTION;
        }

        auto range_it = m_min_max_versions.find(objectId);
        if (range_it == m_min_max_versions.end()) {
            return NO_TRANSACTION;
        }

        if (version < range_it->second.first) {
            return NO_TRANSACTION;
        }

        if (version >= range_it->second.second) {
            return range_it->second.second;
        }

        transaction_id id = range_it->second.second;
        while (id > version) {
            id = priorTid(objectId, id);
        }

        return id;
    }

    std::pair<instance_ptr, transaction_id> best(Type* valueType, SerializationContext& ctx, object_id objectId, transaction_id version) {
        transaction_id bestTid = bestTransactionId(objectId, version);

        if (bestTid == NO_TRANSACTION) {
            return std::pair<instance_ptr, transaction_id>(nullptr, NO_TRANSACTION);
        }

        if (isDeleted(objectId, bestTid)) {
            return std::pair<instance_ptr, transaction_id>(nullptr, NO_TRANSACTION);
        }

        auto& dataForType = dataCacheForType(valueType);

        ObjectData* od = dataForType.lookupKey(ObjectAndVersion(objectId, bestTid));
        if (od) {
            return std::pair<instance_ptr, transaction_id>(od->object_data, bestTid);
        }

        //the data is not already in the cache, so we have to produce it
        auto serialized_it = m_serialized_values.find(std::make_pair(objectId, bestTid));

        if (serialized_it == m_serialized_values.end()) {
            throw std::runtime_error("Somehow, we don't have serialized data for this object.");
        }

        Bytes& serializedVal(serialized_it->second);

        DeserializationBuffer buffer((uint8_t*)&serializedVal[0], serializedVal.size(), ctx);

        od = dataForType.insertKey(ObjectAndVersion(objectId, bestTid));

        try {
            auto fieldAndWireType = buffer.readFieldNumberAndWireType();
            valueType->deserialize(od->object_data, buffer, fieldAndWireType.second);
        } catch(std::exception& e) {
            dataForType.deleteKeyWithUninitializedValue(ObjectAndVersion(objectId, bestTid));

            throw std::runtime_error("Failed deserializing a " + valueType->name() + ": " + e.what());
        }

        return std::pair<instance_ptr, transaction_id>(od->object_data, bestTid);
    }

    bool isDeleted(object_id objectId, transaction_id tid) {
        return m_deleted_values.find(std::make_pair(objectId, tid)) != m_deleted_values.end();
    }

    //consume any values in the object that are _lower_ than 'version'
    void removeLowestIfPossible(object_id objectId) {
        transaction_id topTid = getTopTid(objectId);

        if (topTid == NO_TRANSACTION) {
            return;
        }

        //check whether our top transaction has rolled off
        if (topTid <= m_guaranteed_lowest_id) {
            //if the object was deleted, we can remove it
            if (isDeleted(objectId, topTid)) {
                removeObject(objectId);
                return;
            }
        }

        transaction_id bottomTid = getBottomTid(objectId);

        while (bottomTid != topTid && bottomTid < m_guaranteed_lowest_id) {
            transaction_id nextBottomTid = nextTid(objectId, bottomTid);

            if (nextBottomTid == NO_TRANSACTION) {
                throw std::runtime_error("Expected next TID to be populated");
            }

            if (nextBottomTid > m_guaranteed_lowest_id) {
                return;
            }

            dropObjectVersion(objectId, bottomTid);
            bottomTid = nextBottomTid;
        }
    }

    //remove all traces of an object from the transaction stream
    void removeObject(object_id objectId) {
        transaction_id bottomTid = getBottomTid(objectId);
        if (bottomTid == NO_TRANSACTION) {
            return;
        }

        while (bottomTid != NO_TRANSACTION) {
            transaction_id next = nextTid(objectId, bottomTid);
            dropObjectVersion(objectId, bottomTid);
            bottomTid = next;
        }

        m_min_max_versions.erase(objectId);
    }

    // mark an object 'deleted' as of a particular version number. once deleted,
    // it can't come back. transactions coming it at or above it will fail.
    bool markDeleted(object_id objectId, transaction_id version) {
        if (version < m_guaranteed_lowest_id) {
            return false;
        }

        transaction_id topTid = getTopTid(objectId);

        if (topTid == NO_TRANSACTION) {
            //can't delete something that doesn't exist
            return false;
        }

        if (topTid > version) {
            // makes no sense to delete before the current version
            return false;
        }

        if (isDeleted(objectId, topTid)) {
            //no reason to re-delete
            return false;
        }

        if (topTid == version) {
            //can't delete a value that's known to be non-deleted
            return false;
        }

        m_deleted_values.insert(std::make_pair(objectId, version));

        m_min_max_versions[objectId].second = version;
        m_prior_and_next[std::make_pair(objectId, topTid)].second = version;
        m_prior_and_next[std::make_pair(objectId, version)].first = topTid;
        m_prior_and_next[std::make_pair(objectId, version)].second = NO_TRANSACTION;

        registerObjectAndVersion(objectId, topTid);

        return true;
    }

    /****
    adds an object by id. If the object already exists or an error occurs,
    returns 'false' and does nothing. Otherwise, add it and return 'true'.

    The version number and object ids must be nonnegative.
    *****/
    bool add(object_id objectId, transaction_id version, Bytes data) {
        if (version < m_guaranteed_lowest_id) {
            return false;
        }

        transaction_id tid = getTopTid(objectId);

        m_serialized_values[std::make_pair(objectId, version)] = data;

        if (tid == NO_TRANSACTION) {
            //this is new
            m_min_max_versions[objectId] = std::make_pair(version, version);
            m_prior_and_next[std::make_pair(objectId, version)] = std::make_pair(NO_TRANSACTION, NO_TRANSACTION);
            return true;
        }

        if (version > tid) {
            //we're inserting on the front
            m_min_max_versions[objectId].second = version;
            m_prior_and_next[std::make_pair(objectId, version)].second = NO_TRANSACTION;
            m_prior_and_next[std::make_pair(objectId, version)].first = tid;
            m_prior_and_next[std::make_pair(objectId, tid)].second = version;

            //make sure we check this version later
            registerObjectAndVersion(objectId, tid);

            return true;
        }

        transaction_id prior = priorTid(objectId, tid);

        while (prior != NO_TRANSACTION && prior > version) {
            tid = prior;
            prior = priorTid(objectId, tid);
        }

        if (prior == version || tid == version) {
            return false;
        }

        //inserting on the back
        if (prior == NO_TRANSACTION) {
            m_min_max_versions[objectId].first = version;
            m_prior_and_next[std::make_pair(objectId, tid)].first = version;
            m_prior_and_next[std::make_pair(objectId, version)].second = tid;
            m_prior_and_next[std::make_pair(objectId, version)].first = NO_TRANSACTION;

            registerObjectAndVersion(objectId, version);

            return true;
        }

        if (!(prior < version && version < tid)) {
            throw std::runtime_error("Expected to be in between two transactions: " +
                    format(version) + " not in between " + format(prior) + " and " + format(tid));
        }

        //inserting in the middle
        m_prior_and_next[std::make_pair(objectId, tid)].first = version;
        m_prior_and_next[std::make_pair(objectId, prior)].second = version;
        m_prior_and_next[std::make_pair(objectId, version)].first = prior;
        m_prior_and_next[std::make_pair(objectId, version)].second = tid;

        registerObjectAndVersion(objectId, version);

        return true;
    }

    void dropObjectVersion(object_id oid, transaction_id tid) {
        auto range_it = m_min_max_versions.find(oid);

        if (range_it == m_min_max_versions.end()) {
            return;
        }

        m_deleted_values.erase(std::make_pair(oid, tid));
        m_serialized_values.erase(std::make_pair(oid, tid));

        for (auto& typeAndData: m_data) {
            typeAndData.second.deleteKey(ObjectAndVersion(oid, tid));
        }
        std::pair<transaction_id, transaction_id> priorAndNext = m_prior_and_next[std::make_pair(oid, tid)];
        m_prior_and_next.erase(std::make_pair(oid, tid));

        if (priorAndNext.first == NO_TRANSACTION && priorAndNext.second == NO_TRANSACTION) {
            m_min_max_versions.erase(oid);
            return;
        }

        //update the bounds if we're removing the bottom value
        if (tid == range_it->second.first) {
            range_it->second.first = priorAndNext.second;
        }

        //update the bounds if we're removing the top value
        if (tid == range_it->second.second) {
            range_it->second.second = priorAndNext.first;
        }

        if (priorAndNext.first != NO_TRANSACTION) {
            m_prior_and_next[std::make_pair(oid, priorAndNext.first)].second = priorAndNext.second;
        }
        if (priorAndNext.second != NO_TRANSACTION) {
            m_prior_and_next[std::make_pair(oid, priorAndNext.second)].first = priorAndNext.first;
        }
    }

    void registerObjectAndVersion(object_id oid, transaction_id tid) {
        m_version_numbers_to_check[tid].insert(oid);
    }

    size_t objectCount() const {
        return m_min_max_versions.size();
    }

    DictInstance<ObjectAndVersion, ObjectData>& dataCacheForType(Type* t) {
        auto it = m_data.find(t);
        if (it != m_data.end()) {
            return it->second;
        }

        m_data[t] = DictInstance<ObjectAndVersion, ObjectData>(
            Tuple::Make(std::vector<Type*>({Int64::Make(),Int64::Make()})),
            t
        );

        return m_data[t];
    }

    void check(object_id oid) {
        auto bottom = getBottomTid(oid);
        auto top = getTopTid(oid);

        if (bottom == NO_TRANSACTION) {
            return;
        }

        while (bottom != NO_TRANSACTION) {
            if (m_serialized_values.find(std::make_pair(oid, bottom)) == m_serialized_values.end()) {
                throw std::runtime_error("missing");
            }

            auto next = nextTid(oid, bottom);

            if (next == NO_TRANSACTION) {
                if (bottom != top) {
                    throw std::runtime_error("next is empty when it shouldn't be");
                }
            } else if (priorTid(oid, next) != bottom) {
                throw std::runtime_error("prev(next) != next");
            }

            bottom = next;
        }
    }

private:
    //the field we represent
    field_id m_field_id;

    //the lowest transaction anyone will ever ask us about
    transaction_id m_guaranteed_lowest_id;

    //the first and last transaction id for each object id
    std::unordered_map<object_id, std::pair<transaction_id, transaction_id> > m_min_max_versions;

    //the next and prior transaction ids for a given object/transaction id pair
    std::unordered_map<std::pair<object_id, transaction_id>, std::pair<transaction_id, transaction_id> > m_prior_and_next;

    //the serialized representation of each value
    std::unordered_map<std::pair<object_id, transaction_id>, Bytes> m_serialized_values;

    //any deleted values
    std::unordered_set<std::pair<object_id, transaction_id> > m_deleted_values;

    //a cache of the deserialized versions of each value
    std::unordered_map<Type*, DictInstance<ObjectAndVersion, ObjectData> > m_data;

    //for each transaction, a list of objects we want to check when that version number
    //gets consumed by the m_guaranteed_lowest_id, in case we want to delete things.
    std::map<transaction_id, std::set<object_id> > m_version_numbers_to_check;
};
