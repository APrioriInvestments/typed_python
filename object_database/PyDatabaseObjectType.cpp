#include "PyDatabaseObjectType.hpp"
#include "../typed_python/PyInstance.hpp"

std::unordered_set<PyDatabaseObjectType*> PyDatabaseObjectType::s_database_object_types;

PyDatabaseObjectType* PyDatabaseObjectType::check(PyObject* o) {
    auto it = s_database_object_types.find((PyDatabaseObjectType*)o);

    if (it != s_database_object_types.end()) {
        return (PyDatabaseObjectType*)o;
    }

    return nullptr;
}

PyObject* PyDatabaseObjectType::tp_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds) {
    PyObjectStealer noArgs(PyTuple_Pack(0));

    return PyInstance::tp_new(subtype, noArgs, NULL);
}

PyObject* PyDatabaseObjectType::fromIdentity(PyObject* databaseType, PyObject* args) {
    return translateExceptionToPyObject([&]{
        PyDatabaseObjectType* dbType = check(databaseType);
        if (!dbType) {
            throw std::runtime_error("First argument to fromIdentity should be a dbtype");
        }

        object_id oid;
        if (!PyArg_ParseTuple(args, "l", &oid)) {
            throw PythonExceptionSet();
        }

        return dbType->fromIntegerIdentity(oid);
    });
}

PyObject* PyDatabaseObjectType::fromIntegerIdentity(object_id oid) {
    PyObjectStealer noArgs(PyTuple_Pack(0));

    PyObject* res = ((PyTypeObject*)this)->tp_new(((PyTypeObject*)this), noArgs, NULL);

    getObjectId(res) = oid;

    return res;
}

PyTypeObject* PyDatabaseObjectType::createDatabaseObjectType(PyObject* schema, std::string name) {
    static Type* baseType = NamedTuple::Make({::Int64::Make()}, {std::string("_identity")});
    static PyTypeObject* base = incref(PyInstance::typeObj(baseType));

    if (!base) {
        throw std::runtime_error("Expected a valid base type.");
    }

    PyObject* classDict = PyDict_New();

    PyDict_SetItemString(
        classDict,
        "__schema__",
        schema
        );

    PyDict_SetItemString(
        classDict,
        "__is_database_object_type__",
        Py_True
        );

    PyMethodDef* methods = new PyMethodDef[13] {
        {"fromIdentity", (PyCFunction)PyDatabaseObjectType::fromIdentity, METH_VARARGS | METH_CLASS, NULL},
        {"lookupAny", (PyCFunction)PyDatabaseObjectType::pyLookupAny, METH_VARARGS | METH_KEYWORDS | METH_CLASS, NULL},
        {"lookupAll", (PyCFunction)PyDatabaseObjectType::pyLookupAll, METH_VARARGS | METH_KEYWORDS | METH_CLASS, NULL},
        {"lookupOne", (PyCFunction)PyDatabaseObjectType::pyLookupOne, METH_VARARGS | METH_KEYWORDS | METH_CLASS, NULL},
        {"markLazyByDefault", (PyCFunction)PyDatabaseObjectType::pyMarkLazyByDefault, METH_VARARGS | METH_KEYWORDS | METH_CLASS, NULL},
        {"isLazyByDefault", (PyCFunction)PyDatabaseObjectType::pyIsLazyByDefault, METH_VARARGS | METH_KEYWORDS | METH_CLASS, NULL},
        {"finalize", (PyCFunction)PyDatabaseObjectType::pyFinalize, METH_VARARGS | METH_KEYWORDS | METH_CLASS, NULL},
        {"addField", (PyCFunction)PyDatabaseObjectType::pyAddField, METH_VARARGS | METH_KEYWORDS | METH_CLASS, NULL},
        {"addMethod", (PyCFunction)PyDatabaseObjectType::pyAddMethod, METH_VARARGS | METH_KEYWORDS | METH_CLASS, NULL},
        {"addStaticMethod", (PyCFunction)PyDatabaseObjectType::pyAddStaticMethod, METH_VARARGS | METH_KEYWORDS | METH_CLASS, NULL},
        {"addProperty", (PyCFunction)PyDatabaseObjectType::pyAddProperty, METH_VARARGS | METH_KEYWORDS | METH_CLASS, NULL},
        {"addIndex", (PyCFunction)PyDatabaseObjectType::pyAddIndex, METH_VARARGS | METH_KEYWORDS | METH_CLASS, NULL},
        {NULL, NULL}
    };

    PyDatabaseObjectType* result = new PyDatabaseObjectType { {
        PyVarObject_HEAD_INIT(NULL, 0)              // TYPE (c.f., Type Objects)
        .tp_name = (new std::string(name))->c_str(),// const char*
        .tp_basicsize = sizeof(PyInstance),         // Py_ssize_t
        .tp_itemsize = 0,                           // Py_ssize_t
        .tp_dealloc = 0,                            // destructor
        .tp_print = 0,                              // printfunc
        .tp_getattr = 0,                            // getattrfunc
        .tp_setattr = 0,                            // setattrfunc
        .tp_as_async = 0,                           // PyAsyncMethods*
        .tp_repr = PyDatabaseObjectType::tp_repr,    // reprfunc
        .tp_as_number = 0,                          // PyNumberMethods*
        .tp_as_sequence = 0,                        // PySequenceMethods*
        .tp_as_mapping = 0,                         // PyMappingMethods*
        .tp_hash = 0,                               // hashfunc
        .tp_call = 0,                               // ternaryfunc
        .tp_str = PyDatabaseObjectType::tp_str,     // reprfunc
        .tp_getattro = PyDatabaseObjectType::tp_getattro,// getattrofunc
        .tp_setattro = PyDatabaseObjectType::tp_setattro,// setattrofunc

        //this is necessary to ensure that our tp_as_buffer is _NOT_ the same
        //as the tp_as_buffer of the base class, because 'isNativeType' depends on this.
        //python's docs say that this field is not inherited, but apparently it is if its NULL
        //(see the source for PyType_Ready)
        .tp_as_buffer = new PyBufferProcs { 0, 0 }, // PyBufferProcs*
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, // unsigned long
        .tp_doc = 0,                                // const char*
        .tp_traverse = 0,                           // traverseproc
        .tp_clear = 0,                              // inquiry
        .tp_richcompare = 0,                        // richcmpfunc
        .tp_weaklistoffset = 0,                     // Py_ssize_t
        .tp_iter = 0,                               // getiterfunc tp_iter;
        .tp_iternext = 0,                           // iternextfunc
        .tp_methods = methods,                      // struct PyMethodDef*
        .tp_members = 0,                            // struct PyMemberDef*
        .tp_getset = 0,                             // struct PyGetSetDef*
        .tp_base = base,                            // struct _typeobject*
        .tp_dict = classDict,                       // PyObject*
        .tp_descr_get = 0,                          // descrgetfunc
        .tp_descr_set = 0,                          // descrsetfunc
        .tp_dictoffset = 0,                         // Py_ssize_t
        .tp_init = (initproc)PyDatabaseObjectType::tp_init,// initproc
        .tp_alloc = 0,                              // allocfunc
        .tp_new = (newfunc)PyDatabaseObjectType::tp_new,// newfunc
        .tp_free = 0,                               // freefunc /* Low-level free-memory routine */
        .tp_is_gc = 0,                              // inquiry  /* For PyObject_IS_GC */
        .tp_bases = 0,                              // PyObject*
        .tp_mro = 0,                                // PyObject* /* method resolution order */
        .tp_cache = 0,                              // PyObject*
        .tp_subclasses = 0,                         // PyObject*
        .tp_weaklist = 0,                           // PyObject*
        .tp_del = 0,                                // destructor
        .tp_version_tag = 0,                        // unsigned int
        .tp_finalize = 0,                           // destructor
        },
    };

    PyObject* pySchemaName = PyObject_GetAttrString(schema, "_name");
    if (!pySchemaName || !PyUnicode_Check(pySchemaName)) {
        throw std::runtime_error(pySchemaName ? "Invalid schema name" : "Invalid schema");
    }

    result->m_schema_and_typename = SchemaAndTypeName(PyUnicode_AsUTF8(pySchemaName), name);
    result->m_is_finalized = false;
    result->m_init_method = nullptr;
    result->m_del_method = nullptr;
    result->m_lazy_by_default = false;

    if (PyType_Ready((PyTypeObject*)result) < 0) {
        throw PythonExceptionSet();
    }

    if (PyInstance::isNativeType((PyTypeObject*)result)) {
        throw std::runtime_error("Somehow our type '" + name + "' thinks it's actually a native type.");
    }

    if (!PyInstance::isSubclassOfNativeType((PyTypeObject*)result)) {
        throw std::runtime_error("Somehow our type '" + name + "' is not a subclass of a native type");
    }

    if (!PyInstance::extractTypeFrom((PyTypeObject*)result)) {
        throw std::runtime_error("Somehow our type '" + name + "' does not have a TypedPython type");
    }

    if (PyInstance::extractTypeFrom((PyTypeObject*)result)->getTypeCategory() != Type::TypeCategory::catPythonSubclass)  {
        throw std::runtime_error("Somehow our type '" + name + "' does not have a TypedPython type");
    }

    s_database_object_types.insert(result);

    return (PyTypeObject*)result;
}

void PyDatabaseObjectType::finalize() {
    m_is_finalized = true;
}

void PyDatabaseObjectType::assertNameDoesntExist(std::string name) {
    if (m_fields.find(name) != m_fields.end() ||
            m_static_methods.find(name) != m_static_methods.end() ||
            m_properties.find(name) != m_properties.end()) {
        throw std::runtime_error("Member named '" + name + "' already exists.");
    }
}

void PyDatabaseObjectType::addMethod(std::string name, PyObject* method) {
    assertNameDoesntExist(name);

    if (name == "__init__") {
        m_init_method = method;
    }

    if (name == "__del__") {
        m_del_method = method;
    }

    m_methods[name] = incref(method);
}

void PyDatabaseObjectType::addStaticMethod(std::string name, PyObject* method) {
    assertNameDoesntExist(name);

    PyDict_SetItemString(((PyTypeObject*)this)->tp_dict, name.c_str(), method);

    m_static_methods[name] = incref(method);
}

void PyDatabaseObjectType::addField(std::string name, Type* fieldType) {
    assertNameDoesntExist(name);

    m_fields[name] = fieldType;
}

void PyDatabaseObjectType::addIndex(std::string index_name, const std::vector<std::string>& field_names) {
    if (m_indices.find(index_name) != m_indices.end()) {
        throw std::runtime_error("Index '" + index_name + "' already exists.");
    }

    for (auto field: field_names) {
        if (m_fields.find(field) == m_fields.end()) {
            throw std::runtime_error("Field '" + field + "' doesn't exist.");
        }
    }

    m_indices[index_name] = field_names;

    std::vector<Type*> types;

    for (auto field: field_names) {
        m_field_to_indices[field].insert(index_name);
        types.push_back(m_fields[field]);
    }

    if (types.size() != 1) {
        m_indexTypes[index_name] = ::Tuple::Make(types);
    } else {
        m_indexTypes[index_name] = types[0];
    }
}

void PyDatabaseObjectType::addProperty(std::string name, PyObject* getter, PyObject* setter) {
    assertNameDoesntExist(name);

    m_properties[name] = std::make_pair(incref(getter), incref(setter));
}

void PyDatabaseObjectType::ensureAllFieldsInitialized(View* view, object_id oid) {
    for (auto fieldnameAndType: m_fields) {
        if (!lookupFieldValue(this, oid, fieldnameAndType.first, fieldnameAndType.second)) {
            Instance defaultValue(fieldnameAndType.second, [&](instance_ptr tgt) {
                fieldnameAndType.second->constructor(tgt);
            });

            setFieldValue(this, oid, fieldnameAndType.first, fieldnameAndType.second, defaultValue.data());
        }
    }
}

void PyDatabaseObjectType::removeAllFields(View* view, object_id oid) {
    for (auto fieldnameAndType: m_fields) {
        if (fieldnameAndType.first != " exists") {
            if (lookupFieldValue(this, oid, fieldnameAndType.first, fieldnameAndType.second)) {
                setFieldValue(this, oid, fieldnameAndType.first, fieldnameAndType.second, nullptr);
            }
        }
    }

    if (lookupFieldValue(this, oid, " exists", ::Bool::Make())) {
        setFieldValue(this, oid, " exists", ::Bool::Make(), nullptr);
    }
}

int PyDatabaseObjectType::tp_init(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyDatabaseObjectType* self_t = (PyDatabaseObjectType*)self->ob_type;

    return translateExceptionToPyObjectReturningInt([&] {
        View* v = View::currentView();
        if (!v) {
            throw std::runtime_error(
                "Can't create instances of " + self_t->m_schema_and_typename + " outside of a transaction."
            );
        }

        if (!v->isWriteable()) {
            throw std::runtime_error(
                "Can't create instances of " + self_t->m_schema_and_typename + " in a view. Open a transaction."
            );
        }

        if (!args || !PyTuple_Check(args)) {
            throw std::runtime_error("Invalid arg tuple");
        }

        if (kwargs && !PyDict_Check(kwargs)) {
            throw std::runtime_error("Invalid kwarg dict");
        }

        if (self_t->m_init_method) {
            object_id oid = v->getConnectionState().allocateIdentity();

            getObjectId(self) = oid;

            //ensure that we can see this object. if we're not subscribed to
            //the entire type, we need to create an implicit object-level subscription
            //to any objects we create directly.
            v->newObject(self_t->m_schema_and_typename, oid);

            bool exists = true;
            setFieldValue(self_t, oid, " exists", ::Bool::Make(), (instance_ptr)&exists);

            PyObjectStealer initArgs(PyTuple_New(1 + PyTuple_Size(args)));

            PyTuple_SetItem(initArgs, 0, incref(self));
            for (long k = 0; k < PyTuple_Size(args); k++) {
                PyTuple_SetItem(initArgs, k+1, incref(PyTuple_GetItem(args, k)));
            }

            try {
                PyObject* res = PyObject_Call(self_t->m_init_method, initArgs, kwargs);
                if (!res) {
                    throw PythonExceptionSet();
                } else {
                    decref(res);
                }

                self_t->ensureAllFieldsInitialized(v, oid);
            } catch(...) {
                self_t->removeAllFields(v, oid);
                throw;
            }

            return 0;
        }

        if (PyTuple_Size(args)) {
            throw std::runtime_error(
                "Can't construct instances of " + self_t->m_schema_and_typename + " with positional arguments."
            );
        }

        std::map<std::string, Instance> typedArgs;

        bool isTrue = true;
        typedArgs[" exists"] = Instance((instance_ptr)&isTrue, ::Bool::Make());

        //check that no arguments are invalid
        PyObject *key, *value;
        Py_ssize_t pos = 0;

        while (kwargs && PyDict_Next(kwargs, &pos, &key, &value)) {
            if (!PyUnicode_Check(key)) {
                throw std::runtime_error("Invalid keyword argument: not a string");
            }

            const char* argName = PyUnicode_AsUTF8(key);

            auto argIt = self_t->m_fields.find(argName);

            if (argIt == self_t->m_fields.end()) {
                throw std::runtime_error(
                    "Can't construct instances of " + self_t->m_schema_and_typename +
                        " with an argument named " + argName
                );
            }

            Type* argType = argIt->second;

            try {
                typedArgs[argName] = Instance(argType, [&](instance_ptr tgt) {
                    PyInstance::copyConstructFromPythonInstance(argType, tgt, value, true);
                });
            } catch(std::exception& e) {
                throw std::runtime_error(
                    "Failed to initialize field " + self_t->m_schema_and_typename +
                        "." + argName + ": " + e.what()
                );
            }
        }

        for (auto argIt: self_t->m_fields) {
            if (typedArgs.find(argIt.first) == typedArgs.end()) {
                if (!argIt.second->is_default_constructible()) {
                    throw std::runtime_error(
                        "Can't construct instances of " + self_t->m_schema_and_typename + "." +
                            " without providing a value for " + argIt.first
                    );
                }

                try {
                    typedArgs[argIt.first] = Instance(argIt.second, [&](instance_ptr tgt) {
                        argIt.second->constructor(tgt);
                    });
                } catch(std::exception& e) {
                    throw std::runtime_error(
                        "Failed to default-initialize field " + self_t->m_schema_and_typename + "." +
                            argIt.first + ": " + e.what()
                    );
                }
            }
        }

        object_id oid = v->getConnectionState().allocateIdentity();

        getObjectId(self) = oid;

        v->newObject(self_t->m_schema_and_typename, oid);

        for (auto nameAndVal: typedArgs) {
            try {
                setFieldValue(self_t, oid, nameAndVal.first, nameAndVal.second.type(), nameAndVal.second.data());
            } catch(std::exception& e) {
                throw std::runtime_error(
                    "Failed to assign field " + self_t->m_schema_and_typename + "." +
                        nameAndVal.first + ": " + e.what()
                );
            }
        }

        return 0;
    });
}

/* static */
object_id& PyDatabaseObjectType::getObjectId(PyObject* o) {
    return *(object_id*)(((PyInstance*)o)->mContainingInstance.data());
}

int PyDatabaseObjectType::tp_setattro(PyObject *o, PyObject* attrName, PyObject* attrVal) {
    return translateExceptionToPyObjectReturningInt([&] {
        if (!PyUnicode_Check(attrName)) {
            throw std::runtime_error("Expected a string for attribute name");
        }

        View* view = View::currentView();
        if (!view) {
            throw std::runtime_error("Database attributes cannot be set without an active transaction.");
        }

        std::string attr(PyUnicode_AsUTF8(attrName));

        object_id oid = getObjectId(o);

        PyDatabaseObjectType* obType = (PyDatabaseObjectType*)o->ob_type;

        checkVisible(view, o);

        auto fieldIt = obType->m_fields.find(attr);
        if (fieldIt != obType->m_fields.end()) {
            Type* fieldType = fieldIt->second;

            Instance i(fieldType, [&](instance_ptr tgt) {
                PyInstance::copyConstructFromPythonInstance(fieldType, tgt, attrVal, true);
            });

            setFieldValue(obType, oid, attr, fieldType, i.data());
            return 0;
        }

        auto propIt = obType->m_properties.find(attr);

        if (propIt != obType->m_properties.end()) {
            if (propIt->second.second == Py_None) {
                throw std::runtime_error("Attribute " + attr + " is not settable.");
            }

            PyObject* result = PyObject_CallFunctionObjArgs(propIt->second.second, o, attrVal, NULL);

            if (!result) {
                throw PythonExceptionSet();
            }

            decref(result);

            return 0;
        }

        throw std::runtime_error("Attribute " + attr + " is not settable.");
    });
}

field_id PyDatabaseObjectType::fieldIdForNameAndState(std::string name, DatabaseConnectionState* state) {
    auto fieldIdIt = m_field_ids.find(std::make_pair(name, state));

    if (fieldIdIt == m_field_ids.end()) {
        field_id fieldId = state->getFieldId(m_schema_and_typename, name);

        m_field_ids[std::make_pair(name, state)] = fieldId;

        return fieldId;
    } else {
        return fieldIdIt->second;
    }
}

void PyDatabaseObjectType::setFieldValue(PyDatabaseObjectType* obType, object_id oid, std::string attr, Type* fieldType, instance_ptr data) {
    //get the current view
    View* view = View::currentView();
    if (!view) {
        throw std::runtime_error("Database attributes cannot be set without an active transaction.");
    }

    field_id fieldId = obType->fieldIdForNameAndState(attr, &view->getConnectionState());

    //pull each value out of the index if its already populated
    auto indexList = obType->m_field_to_indices.find(attr);
    if (indexList != obType->m_field_to_indices.end()) {
        for (auto index: indexList->second) {
            field_id fieldIdForIndex = obType->fieldIdForNameAndState(index, &view->getConnectionState());

            OneOf<None, index_value> curIndexValue = obType->calcCurIndexValue(view, index, fieldIdForIndex, oid);

            index_value val;

            if (curIndexValue.getValue(val)) {
                view->indexRemove(fieldIdForIndex, val, oid);
            }
        }
    }

    view->setField(fieldId, oid, fieldType, data);

    //now add each value back to the index if its not already populated
    if (indexList != obType->m_field_to_indices.end()) {
        for (auto index: indexList->second) {
            field_id fieldIdForIndex = obType->fieldIdForNameAndState(index, &view->getConnectionState());

            OneOf<None, index_value> curIndexValue = obType->calcCurIndexValue(view, index, fieldIdForIndex, oid);

            index_value val;

            if (curIndexValue.getValue(val)) {
                view->indexAdd(fieldIdForIndex, val, oid);
            }
        }
    }
}

OneOf<None, index_value> PyDatabaseObjectType::calcCurIndexValue(View* view, std::string indexName, field_id indexFieldId, object_id oid) {
    SerializationBuffer buffer(*view->getSerializationContext());

    if (m_indices[indexName].size() != 1) {
        //the index lookup will be a tuple. We need to replicate its serialization format.
        buffer.writeBeginCompound(0);
    }

    size_t fieldIndex = 0;

    for (auto fieldname: m_indices[indexName]) {
        field_id fieldId = fieldIdForNameAndState(fieldname, &view->getConnectionState());

        instance_ptr data = view->getField(fieldId, oid, m_fields[fieldname]);

        if (!data) {
            //doesn't exist
            return OneOf<None, index_value>(None());
        }

        m_fields[fieldname]->serialize(data, buffer, fieldIndex++);
    }


    if (m_indices[indexName].size() != 1) {
        //the index lookup will be a tuple.
        buffer.writeEndCompound();
    }

    buffer.finalize();

    //right now, index_value is just the serialization of the value.
    return OneOf<None, index_value>(Bytes((const char*)buffer.buffer(), buffer.size()));
}

/* static */
void PyDatabaseObjectType::checkVisible(View* view, PyObject* o) {
    if (!view) {
        view = View::currentView();
        if (!view) {
            throw std::runtime_error("Database attributes cannot be set without an active transaction.");
        }
    }

    object_id oid = getObjectId(o);

    PyDatabaseObjectType* obType = (PyDatabaseObjectType*)o->ob_type;

    if (!view->objectIsVisible(obType->m_schema_and_typename, oid)) {
        PyErr_SetObject(getObjectDoesntExistException(), o);
        throw PythonExceptionSet();
    }

    view->loadLazyObjectIfNeeded(oid);
}

PyObject* PyDatabaseObjectType::tp_getattro(PyObject *o, PyObject* attrName) {
    return translateExceptionToPyObject([&] {
        if (!PyUnicode_Check(attrName)) {
            throw std::runtime_error("Expected a string for attribute name");
        }

        PyDatabaseObjectType* obType = (PyDatabaseObjectType*)o->ob_type;

        std::string attr(PyUnicode_AsUTF8(attrName));

        object_id oid = getObjectId(o);

        if (attr == "_identity") {
            return PyLong_FromLong(oid);
        }

        if (attr == "__class__") {
            return incref((PyObject*)obType);
        }

        if (attr == "__schema__") {
            return PyObject_GetAttr((PyObject*)obType, attrName);
        }

        if (attr == "exists") {
            static PyMethodDef exists = {"exists", (PyCFunction)PyDatabaseObjectType::pyExists, METH_VARARGS | METH_KEYWORDS, NULL};
            return PyCFunction_New(&exists, o);
        }

        if (attr == "delete") {
            static PyMethodDef deleteFun = {"delete", (PyCFunction)PyDatabaseObjectType::pyDelete, METH_VARARGS | METH_KEYWORDS, NULL};
            return PyCFunction_New(&deleteFun, o);
        }

        auto fieldIt = obType->m_fields.find(attr);
        if (fieldIt != obType->m_fields.end()) {
            checkVisible(nullptr, o);

            return lookupFieldValue(obType, oid, attr, fieldIt->second);
        }

        auto staticMethodIt = obType->m_static_methods.find(attr);

        if (staticMethodIt != obType->m_static_methods.end()) {
            return incref(staticMethodIt->second);
        }

        auto methodIt = obType->m_methods.find(attr);

        if (methodIt != obType->m_methods.end()) {
            return PyMethod_New(methodIt->second, o);
        }

        auto propIt = obType->m_properties.find(attr);

        if (propIt != obType->m_properties.end()) {
            return PyObject_CallFunctionObjArgs(propIt->second.first, o, NULL);
        }

        PyErr_Format(
            PyExc_AttributeError,
            "%S",
            attrName
        );

        throw PythonExceptionSet();
    });
}

PyObject* PyDatabaseObjectType::lookupFieldValue(PyDatabaseObjectType* obType, object_id oid, std::string attr, Type* fieldType)
{
    View* view = View::currentView();

    if (!view) {
        throw std::runtime_error(
            "Database attributes cannot be read without an active view or transaction."
            );
    }

    field_id fieldId = obType->fieldIdForNameAndState(attr, &view->getConnectionState());

    instance_ptr data = view->getField(fieldId, oid, fieldType);

    if (!data) {
        field_id existsFID = obType->fieldIdForNameAndState(" exists", &view->getConnectionState());
        if (view->getField(existsFID, oid, ::Bool::Make())) {
            // return the default-constructed instance.
            Instance i(fieldType, [&](instance_ptr tgt) {
                fieldType->constructor(tgt);
            });
            return PyInstance::extractPythonObject(i.data(), fieldType);
        } else {
            PyErr_SetObject(getObjectDoesntExistException(), obType->fromIntegerIdentity(oid));
            throw PythonExceptionSet();
        }
    }

    return PyInstance::extractPythonObject(data, fieldType);
}

PyObject* PyDatabaseObjectType::pyExists(PyObject *self, PyObject* args, PyObject* kwargs)
{
    PyDatabaseObjectType* obType = (PyDatabaseObjectType*)self->ob_type;

    object_id oid = getObjectId(self);

    return translateExceptionToPyObject([&]() {
        View* view = View::currentView();

        if (!view) {
            throw std::runtime_error(
                "Database attributes cannot be read without an active view or transaction."
                );
        }

        field_id fieldId = obType->fieldIdForNameAndState(" exists", &view->getConnectionState());

        view->loadLazyObjectIfNeeded(oid);

        if (!view->objectIsVisible(obType->m_schema_and_typename, oid)) {
            return incref(Py_False);
        }

        return incref(view->getField(fieldId, oid, ::Bool::Make()) ? Py_True : Py_False);
    });
}

/* static */
PyObject* PyDatabaseObjectType::getObjectDoesntExistException() {
    static PyObject* viewModule = PyImport_ImportModule("object_database.view");

    if (!viewModule) {
        throw std::runtime_error("Can't find object_database.view");
    }

    static PyObject* objectDoesntExistException = PyObject_GetAttrString(viewModule, "ObjectDoesntExistException");

    if (!viewModule) {
        throw std::runtime_error("Can't find object_database.view.ObjectDoesntExistException");
    }

    return objectDoesntExistException;
}

PyObject* PyDatabaseObjectType::pyDelete(PyObject *self, PyObject* args, PyObject* kwargs)
{
    PyDatabaseObjectType* obType = (PyDatabaseObjectType*)self->ob_type;

    return translateExceptionToPyObject([&]() {
        View* view = View::currentView();

        if (!view) {
            throw std::runtime_error(
                "Database attributes cannot be read without an active view or transaction."
                );
        }

        object_id oid = getObjectId(self);

        checkVisible(view, self);

        if (obType->m_del_method) {
            PyObject* o = PyObject_CallFunctionObjArgs(obType->m_del_method, self, NULL);

            if (!o) {
                throw PythonExceptionSet();
            }

            decref(o);
        }

        instance_ptr data = view->getField(
            obType->fieldIdForNameAndState(" exists", &view->getConnectionState()),
            oid,
            ::Bool::Make()
        );

        if (!data) {
            PyErr_SetObject(getObjectDoesntExistException(), self);
            throw PythonExceptionSet();
        }

        for (auto fieldnameAndType: obType->m_fields) {
            setFieldValue(obType, oid, fieldnameAndType.first, fieldnameAndType.second, nullptr);
        }

        return incref(Py_None);
    });
}

PyObject* PyDatabaseObjectType::pyAddField(PyObject *databaseType, PyObject* args, PyObject* kwargs)
{
    static const char *kwlist[] = {"field", "type", NULL};
    const char* field;
    PyObject* type;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sO", (char**)kwlist, &field, &type)) {
        return nullptr;
    }

    Type* t = PyInstance::unwrapTypeArgToTypePtr(type);

    if (!t) {
        PyErr_Format(PyExc_TypeError, "Expected third argument to be convertible to a nativepython type.");
        return NULL;
    }

    PyDatabaseObjectType* obType = PyDatabaseObjectType::check(databaseType);
    if (!obType) {
        PyErr_Format(PyExc_TypeError, "Expected first argument to be a database type.");
        return NULL;
    }

    return translateExceptionToPyObject([&] {
        obType->addField(field, t);
        return incref(Py_None);
    });
}

PyObject* PyDatabaseObjectType::pyAddMethod(PyObject *databaseType, PyObject* args, PyObject* kwargs)
{
    static const char *kwlist[] = {"field", "method", NULL};
    const char* field;
    PyObject* method;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sO", (char**)kwlist, &field, &method)) {
        return nullptr;
    }

    PyDatabaseObjectType* obType = PyDatabaseObjectType::check(databaseType);
    if (!obType) {
        PyErr_Format(PyExc_TypeError, "Expected first argument to be a database type.");
        return NULL;
    }

    return translateExceptionToPyObject([&] {
        obType->addMethod(field, method);
        return incref(Py_None);
    });
}

PyObject* PyDatabaseObjectType::pyAddStaticMethod(PyObject *databaseType, PyObject* args, PyObject* kwargs)
{
    static const char *kwlist[] = {"field", "method", NULL};
    const char* field;
    PyObject* method;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sO", (char**)kwlist, &field, &method)) {
        return nullptr;
    }

    PyDatabaseObjectType* obType = PyDatabaseObjectType::check(databaseType);
    if (!obType) {
        PyErr_Format(PyExc_TypeError, "Expected first argument to be a database type.");
        return NULL;
    }

    return translateExceptionToPyObject([&] {
        obType->addStaticMethod(field, method);
        return incref(Py_None);
    });
}

PyObject* PyDatabaseObjectType::pyAddProperty(PyObject *databaseType, PyObject* args, PyObject* kwargs)
{
    static const char *kwlist[] = {"field", "getter", "setter", NULL};
    const char* field;
    PyObject* getter;
    PyObject* setter;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sOO", (char**)kwlist, &field, &getter, &setter)) {
        return nullptr;
    }

    PyDatabaseObjectType* obType = PyDatabaseObjectType::check(databaseType);
    if (!obType) {
        PyErr_Format(PyExc_TypeError, "Expected first argument to be a database type.");
        return NULL;
    }

    return translateExceptionToPyObject([&] {
        obType->addProperty(field, getter, setter);
        return incref(Py_None);
    });
}

PyObject* PyDatabaseObjectType::pyFinalize(PyObject *databaseType, PyObject* args, PyObject* kwargs)
{
    static const char *kwlist[] = {NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
        return nullptr;
    }

    PyDatabaseObjectType* obType = PyDatabaseObjectType::check(databaseType);
    if (!obType) {
        PyErr_Format(PyExc_TypeError, "Expected first argument to be a database type.");
        return NULL;
    }

    return translateExceptionToPyObject([&] {
        obType->finalize();
        return incref(Py_None);
    });
}

PyObject* PyDatabaseObjectType::pyMarkLazyByDefault(PyObject *databaseType, PyObject* args, PyObject* kwargs)
{
    static const char *kwlist[] = {NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
        return nullptr;
    }

    PyDatabaseObjectType* obType = PyDatabaseObjectType::check(databaseType);
    if (!obType) {
        PyErr_Format(PyExc_TypeError, "Expected first argument to be a database type.");
        return NULL;
    }

    obType->m_lazy_by_default = true;

    return incref(Py_None);
}

PyObject* PyDatabaseObjectType::pyIsLazyByDefault(PyObject *databaseType, PyObject* args, PyObject* kwargs)
{
    static const char *kwlist[] = {NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
        return nullptr;
    }

    PyDatabaseObjectType* obType = PyDatabaseObjectType::check(databaseType);
    if (!obType) {
        PyErr_Format(PyExc_TypeError, "Expected first argument to be a database type.");
        return NULL;
    }

    return incref(obType->m_lazy_by_default ? Py_True : Py_False);
}

PyObject* PyDatabaseObjectType::pyAddIndex(PyObject *databaseType, PyObject* args, PyObject* kwargs)
{
    static const char *kwlist[] = {"field", "fieldList", NULL};
    const char* field;
    PyObject* fieldList;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sO", (char**)kwlist, &field, &fieldList)) {
        return nullptr;
    }

    if (!PyTuple_Check(fieldList)) {
        PyErr_Format(PyExc_TypeError, "Expected third argument to be a list of fieldnames.");
        return NULL;
    }

    std::vector<std::string> fieldnames;
    for (long k = 0; k < PyTuple_Size(fieldList); k++) {
        PyObject* arg = PyTuple_GetItem(fieldList, k);
        if (!PyUnicode_Check(arg)) {
            PyErr_Format(PyExc_TypeError, "Expected fieldnames to be strings.");
            return NULL;
        }

        fieldnames.push_back(PyUnicode_AsUTF8(arg));
    }

    PyDatabaseObjectType* obType = PyDatabaseObjectType::check(databaseType);
    if (!obType) {
        PyErr_Format(PyExc_TypeError, "Expected first argument to be a database type.");
        return NULL;
    }

    return translateExceptionToPyObject([&] {
        obType->addIndex(field, fieldnames);
        return incref(Py_None);
    });
}


PyObject* PyDatabaseObjectType::pyLookupOne(PyObject *databaseType, PyObject* args, PyObject* kwargs) {
    return translateExceptionToPyObject([&] {
        if (PyTuple_Size(args)) {
            throw std::runtime_error("lookupOne does not accept positional arguments");
        }

        PyDatabaseObjectType* obType = PyDatabaseObjectType::check(databaseType);
        if (!obType) {
            throw std::runtime_error("Expected first argument to be a database type.");
        }

        View* view = View::currentView();
        if (!view) {
            throw std::runtime_error(
                "Can't lookup instances of " + obType->m_schema_and_typename + " outside of a view."
            );
        }

        std::pair<field_id, index_value> lookup = obType->parseIndexLookupKwarg(view, kwargs);

        object_id oid = view->indexLookupFirst(lookup.first, lookup.second);

        if (oid == NO_OBJECT) {
            throw std::runtime_error(
                "Can't find any instances of " + obType->m_schema_and_typename
            );
        }

        return obType->fromIntegerIdentity(oid);
    });
}

PyObject* PyDatabaseObjectType::pyLookupAny(PyObject *databaseType, PyObject* args, PyObject* kwargs) {
    return translateExceptionToPyObject([&] {
        if (PyTuple_Size(args)) {
            throw std::runtime_error("lookupOne does not accept positional arguments");
        }

        PyDatabaseObjectType* obType = PyDatabaseObjectType::check(databaseType);
        if (!obType) {
            throw std::runtime_error("Expected first argument to be a database type.");
        }

        View* view = View::currentView();
        if (!view) {
            throw std::runtime_error(
                "Can't lookup instances of " + obType->m_schema_and_typename + " outside of a view."
            );
        }

        std::pair<field_id, index_value> lookup = obType->parseIndexLookupKwarg(view, kwargs);

        object_id oid = view->indexLookupFirst(lookup.first, lookup.second);

        if (oid == NO_OBJECT) {
            return incref(Py_None);
        }

        return obType->fromIntegerIdentity(oid);
    });
}

PyObject* PyDatabaseObjectType::pyLookupAll(PyObject *databaseType, PyObject* args, PyObject* kwargs) {
    return translateExceptionToPyObject([&] {
        if (PyTuple_Size(args)) {
            throw std::runtime_error("lookupOne does not accept positional arguments");
        }

        PyDatabaseObjectType* obType = PyDatabaseObjectType::check(databaseType);
        if (!obType) {
            throw std::runtime_error("Expected first argument to be a database type.");
        }

        View* view = View::currentView();
        if (!view) {
            throw std::runtime_error(
                "Can't lookup instances of " + obType->m_schema_and_typename + " outside of a view."
            );
        }

        std::pair<field_id, index_value> lookup = obType->parseIndexLookupKwarg(view, kwargs);

        object_id oid = view->indexLookupFirst(lookup.first, lookup.second);

        TupleOf<object_id> oids = TupleOf<object_id>::createUnbounded([&](object_id* tgt, int index) {
            if (oid != NO_OBJECT) {
                *tgt = oid;
                oid = view->indexLookupNext(lookup.first, lookup.second, oid);
                return true;
            }

            return false;
        });

        return oids.toPython(
            // element type override
            PyInstance::unwrapTypeArgToTypePtr(databaseType)
        );
    });
}

std::pair<field_id, index_value> PyDatabaseObjectType::parseIndexLookupKwarg(View* view, PyObject* kwargs) {
    if (kwargs && !PyDict_Check(kwargs)) {
        throw std::runtime_error("Kwargs was not a Dict");
    }

    if (!kwargs || PyDict_Size(kwargs) == 0) {
        //this is how we encode a single bool
        return std::make_pair(
            fieldIdForNameAndState(" exists", &view->getConnectionState()),
            SerializationBuffer::serializeSingleBoolToBytes(true)
        );
    }

    if (PyDict_Size(kwargs) != 1) {
        throw std::runtime_error("You may lookup from at most one index at a time.");
    }

    PyObject *key, *value;
    Py_ssize_t pos = 0;

    if (!PyDict_Next(kwargs, &pos, &key, &value)) {
        throw std::runtime_error("Failed to iterate a dict even though it has items.");
    }

    if (!PyUnicode_Check(key)) {
        throw std::runtime_error("Invalid keyword argument: not a string");
    }

    const char* argName = PyUnicode_AsUTF8(key);

    if (m_indexTypes.find(argName) == m_indexTypes.end()) {
        throw std::runtime_error("No index named " + std::string(argName) + " defined on "
            + m_schema_and_typename);
    }

    Type* indexValType = m_indexTypes[argName];

    Instance indexVal(indexValType, [&](instance_ptr tgt) {
        PyInstance::copyConstructFromPythonInstance(indexValType, tgt, value, true);
    });

    SerializationBuffer buffer(*view->getSerializationContext());

    indexValType->serialize(indexVal.data(), buffer, 0);

    buffer.finalize();

    //right now, index_value is just the serialization of the value.
    return std::make_pair(
        fieldIdForNameAndState(argName, &view->getConnectionState()),
        Bytes((const char*)buffer.buffer(), buffer.size())
    );
}

/* static */
PyObject* PyDatabaseObjectType::tp_repr(PyObject* o) {
    return translateExceptionToPyObject([&] {
        PyDatabaseObjectType* obType = (PyDatabaseObjectType*)o->ob_type;

        auto method_it = obType->m_methods.find("__repr__");
        if (method_it != obType->m_methods.end()) {
            return PyObject_CallFunctionObjArgs(method_it->second, o, NULL);
        }

        std::ostringstream s;
        s << obType->m_schema_and_typename << "(id=" << getObjectId(o) << ")";

        return PyUnicode_FromString(s.str().c_str());
    });
}

/* static */
PyObject* PyDatabaseObjectType::tp_str(PyObject* o) {
    return translateExceptionToPyObject([&] {
        PyDatabaseObjectType* obType = (PyDatabaseObjectType*)o->ob_type;

        auto method_it = obType->m_methods.find("__str__");
        if (method_it != obType->m_methods.end()) {
            return PyObject_CallFunctionObjArgs(method_it->second, o, NULL);
        }

        std::ostringstream s;
        s << obType->m_schema_and_typename << "(id=" << getObjectId(o) << ")";

        return PyUnicode_FromString(s.str().c_str());
    });
}

