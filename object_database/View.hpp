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

#include "DatabaseConnectionState.hpp"
#include "HashFunctions.hpp"
#include <unordered_set>
#include <unordered_map>

/**********
Views provide running (native)python threads with a snapshotted view of the current
object-database's subscribed data. A view holds a reference to a collection of objects
that have different representation at different transaction_ids, along with a single
transaction_id. We show a coherent view of all the objects whose versions are <= the
given transaction_id.

We also track all objects and indices we read or write during execution.
***********/

class View {
public:
   View(std::shared_ptr<DatabaseConnectionState> connection, transaction_id tid, bool allowWrites) :
      m_tid(tid),
      m_allow_writes(allowWrites),
      m_enclosing_view(nullptr),
      m_is_entered(false),
      m_ever_entered(false),
      m_versioned_objects(*connection->getVersionedObjects()),
      m_connection_state(connection)
   {
      m_connection_state->increfVersion(m_tid);
      m_serialization_context = m_connection_state->getContext();
   }

   ~View() {
      if (!m_ever_entered) {
         m_connection_state->decrefVersion(m_tid);
      }
   }

   void setSerializationContext(std::shared_ptr<SerializationContext> context) {
      m_serialization_context = context;
   }

   std::shared_ptr<SerializationContext> getSerializationContext() {
      return m_serialization_context;
   }

   void enter() {
      if (m_is_entered || m_ever_entered) {
         throw std::runtime_error("Can't enter a view twice.");
      }
      m_ever_entered = true;
      m_is_entered = true;
      m_enclosing_view = s_current_view;
      s_current_view = this;
   }

   static View* currentView() {
      return s_current_view;
   }

   void exit() {
      if (!m_is_entered) {
         throw std::runtime_error("Can't exit an un-entered view.");
      }

      m_is_entered = false;
      s_current_view = m_enclosing_view;
      m_enclosing_view = nullptr;
      m_connection_state->decrefVersion(m_tid);
   }

   bool objectIsVisible(SchemaAndTypeName objType, object_id oid) {
      return m_connection_state->objectIsVisible(objType, oid, m_tid);
   }

   void loadLazyObjectIfNeeded(object_id oid) {
      m_connection_state->loadLazyObjectIfNeeded(oid);
   }

   void newObject(SchemaAndTypeName obType, object_id oid) {
      m_connection_state->markObjectSubscribed(obType, oid, m_tid);
   }

   // lookup the current value of an object. if we have written to it, use that value.
   // otherwise use the value in the view. If the value does not exist, returns a null pointer.
   // we also record what values were read
   instance_ptr getField(field_id field, object_id oid, Type* t) {
      auto delete_it = m_delete_cache.find(std::make_pair(field, oid));
      if (delete_it != m_delete_cache.end()) {
         return nullptr;
      }

      auto write_it = m_write_cache.find(std::make_pair(field, oid));
      if (write_it != m_write_cache.end()) {
         return write_it->second.data();
      }

      instance_ptr i = m_versioned_objects.bestObjectVersion(t, *m_serialization_context, field, oid, m_tid).first;

      m_read_values.insert(std::make_pair(field, oid));

      return i;
   }

   void setField(field_id field, object_id oid, Type* t, instance_ptr data) {
      if (!m_allow_writes) {
         throw std::runtime_error("Please use a transaction if you wish to write to object_database fields.");
      }

      auto delete_it = m_delete_cache.find(std::make_pair(field, oid));

      if (delete_it != m_delete_cache.end()) {
         throw std::runtime_error("Value is deleted.");
      }

      if (data) {
         //if we're writing a new value, record whether this is a new object
         //that we're populating into 'm_new_writes'
         bool existsAlready = getField(field, oid, t) != nullptr;
         if (!existsAlready) {
            m_new_writes.insert(std::make_pair(field, oid));
         }

         m_write_cache[std::make_pair(field, oid)] = Instance(data, t);
      } else {
         bool wasNewWrite = m_new_writes.find(std::make_pair(field, oid)) != m_new_writes.end();

         m_write_cache.erase(std::make_pair(field, oid));

         if (!wasNewWrite) {
            //if the object already existed, we need to mark that we're deleting it
            m_delete_cache.insert(std::make_pair(field, oid));
         }
      }
   }

   void indexAdd(field_id fid, index_value i, object_id o) {
      IndexKey key(fid, i);

      m_set_reads.insert(key);

      //check if we are adding something back to an index it was removed from already
      auto remove_it = m_set_removes.find(key);
      if (remove_it != m_set_removes.end()) {
         auto it = remove_it->second.find(o);
         if (it != remove_it->second.end()) {
            remove_it->second.erase(it);
            return;
         }
      }

      if (m_versioned_objects.indexContains(fid, i, m_tid, o)) {
         throw std::runtime_error("Index already contains this value.");
      }

      m_set_adds[key].insert(o);
   }

   void indexRemove(field_id fid, index_value i, object_id o) {
      IndexKey key(fid, i);

      m_set_reads.insert(key);

      //check if we are adding something back to an index it was removed from already
      auto add_it = m_set_adds.find(key);
      if (add_it != m_set_adds.end()) {
         auto it = add_it->second.find(o);
         if (it != add_it->second.end()) {
            add_it->second.erase(it);
            return;
         }
      }

      if (!m_versioned_objects.indexContains(fid, i, m_tid, o)) {
         throw std::runtime_error("Index doesn't contain this value.");
      }

      m_set_removes[key].insert(o);
   }

   bool shouldSuppressIndexValue(field_id fid, index_value i, object_id o) {
      auto remove_it = m_set_removes.find(IndexKey(fid, i));
      if (remove_it == m_set_removes.end()) {
         return false;
      }

      return remove_it->second.find(o) != remove_it->second.end();
   }

   object_id firstAddedIndexValue(field_id fid, index_value i) {
      auto add_it = m_set_adds.find(IndexKey(fid, i));
      if (add_it == m_set_adds.end()) {
         return NO_OBJECT;
      }

      if (add_it->second.size() == 0) {
         return NO_OBJECT;
      }

      return *add_it->second.begin();
   }

   object_id nextAddedIndexValue(field_id fid, index_value i, object_id o) {
      auto add_it = m_set_adds.find(IndexKey(fid, i));
      if (add_it == m_set_adds.end()) {
         return NO_OBJECT;
      }

      auto next_ob_it = add_it->second.upper_bound(o);

      if (next_ob_it == add_it->second.end()) {
         return NO_OBJECT;
      }

      return *next_ob_it;
   }

   object_id indexLookupFirst(field_id fid, index_value i) {
      m_set_reads.insert(IndexKey(fid, i));

      //we need to suppress anything in 'm_set_removes' and add anything in 'm_set_adds'
      object_id first = m_versioned_objects.indexLookupFirst(fid, i, m_tid);
      object_id firstAdded = firstAddedIndexValue(fid, i);

      if (first == NO_OBJECT) {
         return firstAdded;
      }

      //loop until we find a value in the main set that's lower than 'firstAdded'
      while (true) {
         if (firstAdded != NO_OBJECT && firstAdded < first) {
            return firstAdded;
         }

         if (!shouldSuppressIndexValue(fid, i, first)) {
            return first;
         }

         first = m_versioned_objects.indexLookupNext(fid, i, m_tid, first);

         if (first == NO_OBJECT) {
            return firstAdded;
         }
      }
   }

   object_id indexLookupNext(field_id fid, index_value i, object_id o) {
      m_set_reads.insert(IndexKey(fid, i));

      object_id next = m_versioned_objects.indexLookupNext(fid, i, m_tid, o);

      object_id nextAdded = nextAddedIndexValue(fid, i, o);

      if (next == NO_OBJECT) {
         return nextAdded;
      }

      //loop until we find a value in the main set that's lower than 'firstAdded'
      while (true) {
         if (nextAdded != NO_OBJECT && nextAdded < next) {
            return nextAdded;
         }

         if (!shouldSuppressIndexValue(fid, i, next)) {
            return next;
         }

         next = m_versioned_objects.indexLookupNext(fid, i, m_tid, next);

         if (next == NO_OBJECT) {
            return nextAdded;
         }
      }
   }

   DatabaseConnectionState& getConnectionState() {
      return *m_connection_state;
   }

   bool isWriteable() const {
      return m_allow_writes;
   }

   const std::unordered_set<std::pair<field_id, object_id> >& getReadValues() const {
      return m_read_values;
   }

   const std::unordered_map<std::pair<field_id, object_id>, Instance>& getWriteCache() const {
      return m_write_cache;
   }

   const std::unordered_set<std::pair<field_id, object_id> >& getDeleteCache() const {
      return m_delete_cache;
   }

   const std::unordered_map<IndexKey, std::set<object_id> >& getSetAdds() const {
      return m_set_adds;
   }

   const std::unordered_map<IndexKey, std::set<object_id> >& getSetRemoves() const {
      return m_set_removes;
   }

   const std::unordered_set<IndexKey >& getSetReads() const {
      return m_set_reads;
   }

   transaction_id getTransactionId() const {
      return m_tid;
   }


private:
   static thread_local View* s_current_view;

   //the transaction id that snapshots this view
   transaction_id m_tid;

   //is this a view or a transaction?
   bool m_allow_writes;

   View* m_enclosing_view;

   bool m_is_entered;

   bool m_ever_entered;

   std::unordered_set<std::pair<field_id, object_id> > m_read_values;

   std::unordered_map<std::pair<field_id, object_id>, Instance> m_write_cache;

   std::unordered_set<std::pair<field_id, object_id> > m_new_writes;

   std::unordered_set<std::pair<field_id, object_id> > m_delete_cache;

   std::unordered_map<IndexKey, std::set<object_id> > m_set_adds;

   std::unordered_map<IndexKey, std::set<object_id> > m_set_removes;

   std::unordered_set<IndexKey > m_set_reads;

   VersionedObjects& m_versioned_objects;

   std::shared_ptr<DatabaseConnectionState> m_connection_state;

   std::shared_ptr<SerializationContext> m_serialization_context;
};
