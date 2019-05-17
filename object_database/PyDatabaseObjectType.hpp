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

#include <Python.h>
#include "Common.hpp"
#include "View.hpp"
#include "DatabaseConnectionState.hpp"
#include "HashFunctions.hpp"

//these are always subclasses of NamedTuple with '_identity', so that
//serialization can happen
struct PyDatabaseObjectType {
  PyTypeObject typeObj;

  bool m_is_finalized;

  bool m_lazy_by_default;

  PyObject* m_init_method;

  PyObject* m_del_method;

  SchemaAndTypeName m_schema_and_typename;

  //the fields defined in this type
  std::unordered_map<std::string, Type*> m_fields;

  //the per-connection mapping from fieldname to field id. the fieldname->field id mapping
  //covers both indices and regular fields. (the index for a single field has the same id as the field
  //itself)
  std::unordered_map<std::pair<std::string, DatabaseConnectionState*>, field_id> m_field_ids;

  //the indices defined on this type
  std::unordered_map<std::string, std::vector<std::string> > m_indices;

  //the type of each index value
  std::unordered_map<std::string, Type*> m_indexTypes;

  //for each of our fields, which indices is it in?
  std::unordered_map<std::string, std::set<std::string> > m_field_to_indices;

  std::unordered_map<std::string, PyObject*> m_methods;

  std::unordered_map<std::string, PyObject*> m_static_methods;

  std::unordered_map<std::string, std::pair<PyObject*, PyObject*> > m_properties;

  static std::unordered_set<PyDatabaseObjectType*> s_database_object_types;


  /*******
    ensure that all fields for value 'oid' have valid populated values in the given transaction.
  *******/
  void ensureAllFieldsInitialized(View* view, object_id oid);

  /*******
    ensure that all fields for value 'oid' have been removed
  *******/
  void removeAllFields(View* view, object_id oid);

  /*****
    extract the object id from an instance of a PyDatabaseObjectType.
    no checks are performed
  *****/
  static object_id& getObjectId(PyObject* o);

  //check if an object is a PyDatabaseObject. Returns the converted value
  //if so, nullptr if not.
  static PyDatabaseObjectType* check(PyObject* o);

  static PyTypeObject* createDatabaseObjectType(PyObject* schema, std::string name);

  /*****
  lookup the field_id for a given field name in the cache. Grab it from
  the actual DatabaseConnection if we don't have it.
  *****/
  field_id fieldIdForNameAndState(std::string name, DatabaseConnectionState* state);


  /******
  calculate the value of an index tuple.

  returns None if any of the values are not set.
  ******/
  OneOf<None, index_value> calcCurIndexValue(View* view, std::string indexName, field_id indexFieldId, object_id oid);

  //lookup the ObjectDoesntExistException python exception object.
  static PyObject* getObjectDoesntExistException();

  //python entrypoint for calling the 'fromIdentity' method
  static PyObject* fromIdentity(PyObject* self, PyObject* args);

  void finalize();

  void assertNameDoesntExist(std::string name);

  void addField(std::string name, Type* fieldType);

  void addIndex(std::string name, const std::vector<std::string>& names);

  void addMethod(std::string name, PyObject* method);

  void addStaticMethod(std::string name, PyObject* method);

  void addProperty(std::string name, PyObject* getter, PyObject* setter);

  PyObject* fromIntegerIdentity(object_id oid);

  static int tp_init(PyObject* self, PyObject* args, PyObject* kwargs);

  static PyObject* tp_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds);

  static int tp_setattro(PyObject* o, PyObject* attrName, PyObject* attrVal);

  static PyObject* tp_repr(PyObject* o);

  static PyObject* tp_str(PyObject* o);

  static PyObject* tp_getattro(PyObject* o, PyObject* attrName);

  static PyObject* pyExists(PyObject *self, PyObject* args, PyObject* kwargs);

  static PyObject* pyDelete(PyObject *self, PyObject* args, PyObject* kwargs);

  static PyObject* lookupFieldValue(PyDatabaseObjectType* obType, object_id oid, std::string attributeName, Type* fieldType);

  static void setFieldValue(PyDatabaseObjectType* obType, object_id oid, std::string attr, Type* fieldType, instance_ptr data);

  static PyObject* pyAddField(PyObject *none, PyObject* args, PyObject* kwargs);

  static PyObject* pyAddMethod(PyObject *none, PyObject* args, PyObject* kwargs);

  static PyObject* pyAddStaticMethod(PyObject *none, PyObject* args, PyObject* kwargs);

  static PyObject* pyAddProperty(PyObject *none, PyObject* args, PyObject* kwargs);

  static PyObject* pyFinalize(PyObject *none, PyObject* args, PyObject* kwargs);

  static PyObject* pyMarkLazyByDefault(PyObject *none, PyObject* args, PyObject* kwargs);

  static PyObject* pyAddIndex(PyObject *none, PyObject* args, PyObject* kwargs);

  static PyObject* pyLookupOne(PyObject *none, PyObject* args, PyObject* kwargs);

  static PyObject* pyLookupAny(PyObject *none, PyObject* args, PyObject* kwargs);

  static PyObject* pyLookupAll(PyObject *none, PyObject* args, PyObject* kwargs);

  std::pair<field_id, index_value> parseIndexLookupKwarg(View* view, PyObject* kwargs);

  //check if an object is visible (via subscriptions) in the current view.
  //sets a python exception and throws PythonExceptionSet if not.
  static void checkVisible(View* view, PyObject* o);
};
