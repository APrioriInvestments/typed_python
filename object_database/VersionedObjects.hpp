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

#include <memory>
#include <map>

#include "../typed_python/Type.hpp"
#include "Common.hpp"
#include "HashFunctions.hpp"
#include "VersionedObjectsOfType.hpp"
#include "VersionedObjectsOfMultiType.hpp"

/*************

VersionedObjects stores a collection of TypedPython objects that are each
indexed by a pair of int64s (objectid, and fieldid) and a version.

We provide functionality to
* perform a fast lookup of values by (object,field,version) tuples
* discarding values below a given global version
* merge in data with new version numbers
* tag that data exists with a given version number but that's not loaded

*************/

class VersionedObjects {
public:
    std::pair<instance_ptr, transaction_id> bestObjectVersion(Type* t, SerializationContext& ctx, field_id fieldId, object_id objectId, transaction_id version) {
        return versionedObjectsForFieldId(fieldId)->best(t, ctx, objectId, version);
    }

    bool addObjectVersion(field_id fieldId, object_id oid, transaction_id tid, Bytes data) {
        //mark this field on this transaction so we can garbage collect it
        m_fields_needing_check.insert(std::make_pair(tid,fieldId));

        return versionedObjectsForFieldId(fieldId)->add(oid, tid, data);
    }

    bool markObjectVersionDeleted(field_id fieldId, object_id objectId, transaction_id version) {
        //mark this field on this transaction so we can garbage collect it
        m_fields_needing_check.insert(std::make_pair(version, fieldId));

        return versionedObjectsForFieldId(fieldId)->markDeleted(objectId, version);
    }

    object_id indexLookupOne(field_id fid, index_value i, transaction_id t) {
        IndexKey key(fid,i);

        auto it = m_index_to_versioned_id_sets.find(key);
        if (it == m_index_to_versioned_id_sets.end()) {
            return NO_OBJECT;
        }

        return it->second.lookupOne(t);
    }

    object_id indexLookupFirst(field_id fid, index_value i, transaction_id t) {
        return indexLookupNext(fid, i, t, NO_OBJECT);
    }

    object_id indexLookupNext(field_id fid, index_value i, transaction_id t, object_id o) {
        IndexKey key(fid,i);

        auto it = m_index_to_versioned_id_sets.find(key);
        if (it == m_index_to_versioned_id_sets.end()) {
            return NO_OBJECT;
        }

        return it->second.lookupNext(t, o);
    }

    bool indexContains(field_id fid, index_value i, transaction_id t, object_id o) {
        IndexKey key(fid,i);

        auto it = m_index_to_versioned_id_sets.find(key);
        if (it == m_index_to_versioned_id_sets.end()) {
            return false;
        }

        return it->second.isActive(t, o);
    }

    void indexAdd(field_id fid, index_value i, transaction_id t, object_id o) {
        m_indices_needing_check.insert(std::pair<transaction_id, IndexKey>(t, IndexKey(fid, i)));

        return m_index_to_versioned_id_sets[IndexKey(fid, i)].add(t,o);
    }

    void indexRemove(field_id fid, index_value i, transaction_id t, object_id o) {
        m_indices_needing_check.insert(std::pair<transaction_id, IndexKey>(t, IndexKey(fid, i)));

        return m_index_to_versioned_id_sets[IndexKey(fid, i)].remove(t,o);
    }

    void moveGuaranteedLowestIdForward(transaction_id t) {
        while (m_fields_needing_check.size() && m_fields_needing_check.begin()->first < t) {
            //grab the field id and consume it from the queue
            field_id fieldId = m_fields_needing_check.begin()->second;
            m_fields_needing_check.erase(m_fields_needing_check.begin());

            m_field_to_versioned_objects[fieldId]->moveGuaranteedLowestIdForward(t);
        }

        while (m_indices_needing_check.size() && m_indices_needing_check.begin()->first < t) {
            //grab the field id and consume it from the queue
            IndexKey indexId = m_indices_needing_check.begin()->second;
            m_indices_needing_check.erase(m_indices_needing_check.begin());

            auto& versionedIdSet = m_index_to_versioned_id_sets[indexId];

            transaction_id next = versionedIdSet.moveGuaranteedLowestIdForward(t);

            if (next != NO_TRANSACTION) {
                m_indices_needing_check.insert(std::make_pair(next, indexId));
            } else {
                if (versionedIdSet.empty()) {
                    m_index_to_versioned_id_sets.erase(indexId);
                }
            }
        }
    }

    VersionedObjectsOfMultiType* versionedObjectsForFieldId(field_id field) {
        auto it = m_field_to_versioned_objects.find(field);

        if (it == m_field_to_versioned_objects.end()) {
            m_field_to_versioned_objects[field].reset(new VersionedObjectsOfMultiType(field));
            return m_field_to_versioned_objects[field].get();
        }

        return it->second.get();
    }

    size_t objectCount() const {
        size_t res = 0;

        for (auto& fieldAndObjects: m_field_to_versioned_objects) {
            res += fieldAndObjects.second->objectCount();
        }

        return res;
    }

private:
    std::unordered_map<field_id, std::shared_ptr<VersionedObjectsOfMultiType> > m_field_to_versioned_objects;

    std::unordered_map<IndexKey, VersionedIdSet> m_index_to_versioned_id_sets;

    std::set<std::pair<transaction_id, field_id> > m_fields_needing_check;

    std::set<std::pair<transaction_id, IndexKey> > m_indices_needing_check;
};
