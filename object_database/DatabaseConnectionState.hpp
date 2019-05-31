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

#include <map>
#include <memory>

#include "../typed_python/SerializationContext.hpp"
#include "../typed_python/DeserializationBuffer.hpp"
#include "VersionedObjects.hpp"
#include "ObjectFieldId.hpp"
#include "IndexId.hpp"
#include "../typed_python/direct_types/all.hpp"

/*************
DatabaseConnectionState stores a set of versioned objects for a single database
connection. It provides methods for tracking which object versions have refcounts,
cleaning up the connection state, serializing/deserializing values from transactions,
etc.
*************/

class DatabaseConnectionState {
public:
   DatabaseConnectionState() :
         m_next_identity(-1),
         m_cur_transaction_id(-1),
         m_min_transaction_id(-1)
   {
      m_objects.reset(new VersionedObjects());
   }

   void setTriggerLazyLoad(PyObject* o) {
      m_trigger_lazy_load = PyObjectHolder(o);
   }

   void setIdentityRoot(transaction_id id) {
      m_next_identity = id;
   }

   transaction_id allocateIdentity() {
      return m_next_identity++;
   }

   void incomingTransaction(
         transaction_id tid,
         ConstDict<ObjectFieldId, OneOf<None, Bytes> > writes,
         ConstDict<IndexId, TupleOf<object_id> > setAdds,
         ConstDict<IndexId, TupleOf<object_id> > setRemoves
         ) {
      for (auto keyValuePair: writes) {
         None n;

         if (keyValuePair.second.getValue(n)) {
            m_objects->markObjectVersionDeleted(keyValuePair.first.fieldId(), keyValuePair.first.objId(), tid);
         } else {
            Bytes serializedVal;
            if (!keyValuePair.second.getValue(serializedVal)) {
               throw std::runtime_error("Corrupt 'OneOf'");
            }

            m_objects->addObjectVersion(keyValuePair.first.fieldId(), keyValuePair.first.objId(), tid, serializedVal);
         }
      }

      for (auto indexAndOids: setAdds) {
         for (auto o: indexAndOids.second) {
            m_objects->indexAdd(indexAndOids.first.fieldId(), indexAndOids.first.indexValue(), tid, o);
         }
      }

      for (auto indexAndOids: setRemoves) {
         for (auto o: indexAndOids.second) {
            m_objects->indexRemove(indexAndOids.first.fieldId(), indexAndOids.first.indexValue(), tid, o);
         }
      }

      cleanup(tid);
   }

   void setContext(std::shared_ptr<SerializationContext> inContext) {
      m_serialization_context = inContext;
   }

   std::shared_ptr<SerializationContext> getContext() {
      return m_serialization_context;
   }

   std::shared_ptr<VersionedObjects> getVersionedObjects() {
      return m_objects;
   }

   void increfVersion(transaction_id id) {
      m_version_refcounts[id] += 1;
   }

   void decrefVersion(transaction_id id) {
      m_version_refcounts[id] -= 1;
      if (m_version_refcounts[id] == 0) {
         m_version_refcounts.erase(id);
      }

      checkMinId();
   }

   int64_t outstandingViewCount() const {
      int64_t res = 0;

      for (auto versionAndCount: m_version_refcounts) {
         res += versionAndCount.second;
      }

      return res;
   }

   void cleanup(transaction_id observed_tid) {
      if (observed_tid <= m_cur_transaction_id) {
         return;
      }

      m_cur_transaction_id = observed_tid;

      checkMinId();
   }

   transaction_id getMinId() {
      return m_min_transaction_id;
   }

   void checkMinId() {
      transaction_id minId = m_cur_transaction_id;

      if (m_version_refcounts.size()) {
         minId = std::min(m_version_refcounts.begin()->first, minId);
      }

      if (minId > m_min_transaction_id) {
         m_min_transaction_id = minId;

         m_objects->moveGuaranteedLowestIdForward(m_min_transaction_id);
      }
   }

   field_id getFieldId(SchemaAndTypeName type, std::string fieldName) {
      auto type_it = m_field_ids.find(type);
      if (type_it == m_field_ids.end()) {
         throw std::runtime_error("Type '" + type + "' has not had any fields defined yet.");
      }

      auto field_it = type_it->second.find(fieldName);

      if (field_it == type_it->second.end()) {
         throw std::runtime_error("Type '" + type + "' has no field " + fieldName + " defined.");
      }

      return field_it->second;
   }

   void setFieldId(SchemaAndTypeName type, std::string fieldName, field_id fieldId) {
      m_field_ids[type][fieldName] = fieldId;
   }

   transaction_id typeSubscriptionLowestTransaction(SchemaAndTypeName t) {
      auto it = m_subscribed_types.find(t);

      if (it == m_subscribed_types.end()) {
         return NO_TRANSACTION;
      }

      return it->second;
   }

   transaction_id objectSubscriptionLowestTransaction(object_id i) {
      auto it = m_subscribed_objects.find(i);

      if (it == m_subscribed_objects.end()) {
         return NO_TRANSACTION;
      }

      return it->second;
   }

   // is an object visible to transaction 'tid'?
   bool objectIsVisible(SchemaAndTypeName t, object_id i, transaction_id tid) {
      transaction_id tidForType = typeSubscriptionLowestTransaction(t);
      if (tidForType != NO_TRANSACTION && tid >= tidForType) {
         return true;
      }

      transaction_id tidForOid = objectSubscriptionLowestTransaction(i);

      if (tidForOid == NO_TRANSACTION) {
         return false;
      }

      return tidForOid <= tid;
   }

   void markTypeSubscribed(SchemaAndTypeName t, transaction_id tid) {
      auto it = m_subscribed_types.find(t);

      if (it == m_subscribed_types.end()) {
         m_subscribed_types[t] = tid;
         return;
      }

      it->second = std::max(it->second, tid);
   }

   void markObjectSubscribed(object_id i, transaction_id tid) {
      auto it = m_subscribed_objects.find(i);

      if (it == m_subscribed_objects.end()) {
         m_subscribed_objects[i] = tid;
         return;
      }

      it->second = std::max(it->second, tid);
   }

   void markObjectSubscribed(SchemaAndTypeName t, object_id oid, transaction_id tid) {
      if (objectIsVisible(t, oid, tid)) {
         return;
      }

      markObjectSubscribed(oid, tid);
   }

   void markObjectLazy(SchemaAndTypeName schemaAndType, object_id oid) {
      m_lazy_objects[oid] = schemaAndType;
   }

   void markObjectNotLazy(object_id oid) {
      m_lazy_objects.erase(oid);
   }

   void loadLazyObjectIfNeeded(object_id oid) {
      auto it = m_lazy_objects.find(oid);

      if (it == m_lazy_objects.end()) {
         return;
      }

      if (!m_trigger_lazy_load) {
         throw std::runtime_error("No lazy object loader was defined.");
      }

      PyObject* res = PyObject_CallFunction(
         m_trigger_lazy_load,
         "lss",
         oid,
         it->second.schemaName().c_str(),
         it->second.typeName().c_str()
      );

      if (!res) {
         throw PythonExceptionSet();
      }

      decref(res);
   }

   PyObject* getLazyLoadTrigger() {
      return m_trigger_lazy_load;
   }

private:
   //the next guid we will create.
   transaction_id m_next_identity;

   //all the different version numbers for our objects
   std::shared_ptr<VersionedObjects> m_objects;

   //highest transaction number we've seen. The next view
   //will be at this transaction or higher
   transaction_id m_cur_transaction_id;

   //the minimum transaction we're keeping
   transaction_id m_min_transaction_id;

   //for each version number, how many views are outstanding on it?
   //we have to be careful not to delete behind these.
   std::map<transaction_id, int> m_version_refcounts;

   std::shared_ptr<SerializationContext> m_serialization_context;

   std::map<SchemaAndTypeName, std::map<std::string, field_id> > m_field_ids;

   //for each type where we're subscribed to the entire type, the transaction
   //id where that became effective
   std::unordered_map<SchemaAndTypeName, transaction_id> m_subscribed_types;

   //for each object where we're explicitly subscribed (because of an index or
   //object-level subscription), the transaction id where that became effective.
   std::unordered_map<object_id, transaction_id> m_subscribed_objects;

   PyObjectHolder m_trigger_lazy_load;

   std::unordered_map<object_id, SchemaAndTypeName> m_lazy_objects;
};
