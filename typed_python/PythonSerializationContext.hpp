#pragma once

#include "util.hpp"
#include "SerializationContext.hpp"
#include "native_instance_wrapper.h"

class PythonSerializationContext : public SerializationContext {
public:
    //enums in our protocol (serialized as uint8_t)
    enum {
        T_NATIVE,   //a native type, followed by instance data
        T_OBJECT,   //an object written either with a representation, or as a type and a dict, or as a name
        T_LIST,
        T_TUPLE,
        T_DICT,
        T_NONE,
        T_TRUE,
        T_FALSE,
        T_UNICODE,
        T_BYTES,
        T_FLOAT,
        T_LONG,      //currently serialized as an int64_t which is completely wrong.
        T_NATIVETYPE,
        T_OBJECT_NAMED,
        T_OBJECT_TYPEANDDICT,
        T_OBJECT_REPRESENTATION
    };

    PythonSerializationContext(PyObject* typeSetObj) :
            mContextObj(typeSetObj)
    {
    }

    virtual void serializePythonObject(PyObject* o, SerializationBuffer& b) const {
        Type* t = native_instance_wrapper::extractTypeFrom(o->ob_type, true);

        if (t) {
            b.write_uint8(T_NATIVE);
            //we have a natural serialization mechanism already
            serializeNativeType(t, b);
            t->serialize(((native_instance_wrapper*)o)->dataPtr(), b);
        } else {
            if (PyDict_Check(o)) {
                b.write_uint8(T_DICT);
                serializePyDict(o, b);
            } else
            if (PyList_Check(o)) {
                b.write_uint8(T_LIST);
                serializePyList(o, b);
            } else
            if (PyTuple_Check(o)) {
                b.write_uint8(T_TUPLE);
                serializePyTuple(o, b);
            } else
            if (PyType_Check(o)) {
                Type* nativeType = native_instance_wrapper::extractTypeFrom((PyTypeObject*)o, true);

                if (nativeType) {
                    b.write_uint8(T_NATIVETYPE);
                    serializeNativeType(nativeType, b);
                    return;
                } else {
                    b.write_uint8(T_OBJECT);
                    serializePythonObjectNamedOrAsObj(o, b);
                }
            } else
            if (PyBytes_Check(o)) {
                b.write_uint8(T_BYTES);
                b.write_uint32(PyBytes_GET_SIZE(o));
                b.write_bytes((uint8_t*)PyBytes_AsString(o), PyBytes_GET_SIZE(o));
            } else
            if (PyUnicode_Check(o)) {
                b.write_uint8(T_UNICODE);
                Py_ssize_t sz;
                const char* c = PyUnicode_AsUTF8AndSize(o, &sz);
                b.write_uint32(sz);
                b.write_bytes((uint8_t*)c, sz);
            } else
            if (o == Py_None) {
                b.write_uint8(T_NONE);
            } else
            if (o == Py_True) {
                b.write_uint8(T_TRUE);
            } else
            if (o == Py_False) {
                b.write_uint8(T_FALSE);
            } else
            if (PyLong_Check(o)) {
                b.write_uint8(T_LONG);
                b.write_int64(PyLong_AsLong(o));
            } else
            if (PyFloat_Check(o)) {
                b.write_uint8(T_FLOAT);
                b.write_double(PyFloat_AsDouble(o));
            } else {
                b.write_uint8(T_OBJECT);
                serializePythonObjectNamedOrAsObj(o, b);
            }
        }
    }

    void serializePyDict(PyObject* o, SerializationBuffer& b) const {
        std::pair<uint32_t, bool> idAndIsNew = b.cachePointer(o);
        b.write_uint32(idAndIsNew.first);

        if (idAndIsNew.second) {
            b.write_uint32(PyDict_Size(o));

            PyObject *key, *value;
            Py_ssize_t pos = 0;

            int i = 0;

            while (PyDict_Next(o, &pos, &key, &value)) {
                serializePythonObject(key, b);
                serializePythonObject(value, b);
            }
        }
    }

    PyObject* deserializePydict(DeserializationBuffer& b) const {
        uint32_t id = b.read_uint32();

        PyObject* res = (PyObject*)b.lookupCachedPointer(id);
        if (res) {
            return incref(res);
        }

        res = PyDict_New();
        b.addCachedPointer(id, incref(res), true);

        size_t sz = b.read_uint32();

        try {
            for (long k = 0; k < sz; k++) {
                PyObject* key = deserializePythonObject(b);
                PyObject* value = deserializePythonObject(b);

                if (PyDict_SetItem(res, key, value) != 0) {
                    PyErr_Clear();
                    throw std::runtime_error(std::string("Can't place value of type ") + key->ob_type->tp_name + " into a dict");
                }

                Py_DECREF(key);
                Py_DECREF(value);
            }
        } catch(...) {
            Py_DECREF(res);
            throw;
        }

        return res;
    }

    void serializePyList(PyObject* o, SerializationBuffer& b) const {
        std::pair<uint32_t, bool> idAndIsNew = b.cachePointer(o);
        b.write_uint32(idAndIsNew.first);

        if (idAndIsNew.second) {
            size_t sz = PyList_Size(o);

            b.write_uint32(sz);

            for (long k = 0; k < sz; k++) {
                serializePythonObject(PyList_GetItem(o,k), b);
            }
        }
    }

    PyObject* deserializePyList(DeserializationBuffer& b) const {
        uint32_t id = b.read_uint32();

        PyObject* res = (PyObject*)b.lookupCachedPointer(id);
        if (res) {
            return incref(res);
        }

        size_t sz = b.read_uint32();

        res = PyList_New(sz);
        b.addCachedPointer(id, incref(res), true);

        try {
            for (long k = 0; k < sz; k++) {
                PyList_SET_ITEM(res, k, deserializePythonObject(b));
            }
        } catch(...) {
            Py_DECREF(res);
            throw;
        }

        return res;
    }

    void serializePyTuple(PyObject* o, SerializationBuffer& b) const {
        std::pair<uint32_t, bool> idAndIsNew = b.cachePointer(o);
        b.write_uint32(idAndIsNew.first);

        if (idAndIsNew.second) {
            size_t sz = PyTuple_Size(o);

            b.write_uint32(sz);

            for (long k = 0; k < sz; k++) {
                serializePythonObject(PyTuple_GetItem(o,k), b);
            }
        }
    }
    PyObject* deserializePyTuple(DeserializationBuffer& b) const {
        uint32_t id = b.read_uint32();

        PyObject* res = (PyObject*)b.lookupCachedPointer(id);
        if (res) {
            return incref(res);
        }

        size_t sz = b.read_uint32();

        res = PyTuple_New(sz);
        b.addCachedPointer(id, incref(res), true);

        try {
            for (long k = 0; k < sz; k++) {
                PyTuple_SET_ITEM(res, k, deserializePythonObject(b));
            }
        } catch(...) {
            Py_DECREF(res);
            throw;
        }

        return res;
    }

    void serializePythonObjectNamedOrAsObj(PyObject* o, SerializationBuffer& b) const {
        std::pair<uint32_t, bool> idAndIsNew = b.cachePointer(o);
        b.write_uint32(idAndIsNew.first);

        if (!idAndIsNew.second) {
            return;
        }

        //see if the object has a name
        PyObject* typeName = PyObject_CallMethod(mContextObj, "nameForObject", "O", o);
        if (!typeName) {
            PyErr_PrintEx(1);
            throw std::runtime_error("Python threw an exception internally when we called 'nameForObject'.");
        }
        if (typeName != Py_None) {
            if (!PyUnicode_Check(typeName)) {
                Py_DECREF(typeName);
                throw std::runtime_error(std::string("nameForObject returned a non-string"));
            }

            b.write_uint8(T_OBJECT_NAMED);
            b.write_string(std::string(PyUnicode_AsUTF8(typeName)));
            Py_DECREF(typeName);
            return;
        }
        Py_DECREF(typeName);

        //give the plugin a chance to convert the instance to something else
        PyObject* representation = PyObject_CallMethod(mContextObj, "representationFor", "O", o);
        if (!representation) {
            PyErr_PrintEx(1);
            throw std::runtime_error("representationFor threw an exception.");
        }

        if (representation == Py_None) {
            //we did nothing interesting
            b.write_uint8(T_OBJECT_TYPEANDDICT);
            serializePythonObject((PyObject*)o->ob_type, b);

            PyObject* d = PyObject_GenericGetDict(o, nullptr);
            if (!d) {
                throw std::runtime_error(std::string("Object of type ") + o->ob_type->tp_name + " had no dict.");
            }
            serializePyDict(d, b);
            Py_DECREF(d);
        } else {
            //we're writing a pair
            b.write_uint8(T_OBJECT_REPRESENTATION);

            if (!PyTuple_Check(representation) || PyTuple_Size(representation) != 3) {
                Py_DECREF(representation);
                throw std::runtime_error("representationFor should return None or a tuple with 3 things");
            }
            if (!PyTuple_Check(PyTuple_GetItem(representation, 1))) {
                Py_DECREF(representation);
                throw std::runtime_error("representationFor second arguments should be a tuple");
            }

            serializePythonObject(PyTuple_GetItem(representation, 0), b);
            serializePythonObject(PyTuple_GetItem(representation, 1), b);
            serializePythonObject(PyTuple_GetItem(representation, 2), b);
        }

        Py_DECREF(representation);
    }

    PyObject* deserializePythonObjectNamedOrAsObj(DeserializationBuffer& b) const {
        uint32_t id = b.read_uint32();
        PyObject* p = (PyObject*)b.lookupCachedPointer(id);
        if (p) {
            return incref(p);
        }

        uint8_t code = b.read_uint8();

        if (code == T_OBJECT_TYPEANDDICT) {
            //no representation
            PyObject* t = deserializePythonObject(b);
            if (!PyType_Check(t)) {
                std::string tname = t->ob_type->tp_name;
                Py_DECREF(t);
                throw std::runtime_error("Expected a type object. Got " + tname + " instead");
            }

            PyObject* result = ((PyTypeObject*)t)->tp_new(((PyTypeObject*)t), PyTuple_Pack(0), NULL);

            if (!result) {
                throw std::runtime_error("tp_new threw an exception");
            }

            b.addCachedPointer(id, incref(result), true);

            PyObject* d = deserializePydict(b);
            PyObject_GenericSetDict(result, d, nullptr);
            Py_DECREF(d);
            return result;
        } else
        if (code == T_OBJECT_REPRESENTATION) {
            PyObject* t = deserializePythonObject(b);
            PyObject* tArgs = deserializePythonObject(b);

            if (!PyTuple_Check(tArgs)) {
                throw std::runtime_error("corrupt data: second reconstruction argument is not a tuple.");
            }

            PyObject* instance = PyObject_Call(t, tArgs, NULL);
            Py_DECREF(t);
            Py_DECREF(tArgs);

            if (!instance) {
                PyErr_PrintEx(1);
                throw std::runtime_error(std::string("Representation constructor of type ") +
                        t->ob_type->tp_name + " threw an exception.");
            }

            b.addCachedPointer(id, incref(instance), true);

            PyObject* rep = deserializePythonObject(b);
            PyObject* res = PyObject_CallMethod(mContextObj, "setInstanceStateFromRepresentation", "OO", instance, rep);
            Py_DECREF(rep);

            if (!res) {
                PyErr_PrintEx(1);
                throw std::runtime_error("setInstanceStateFromRepresentation threw an exception.");
            }
            if (res != Py_True) {
                Py_DECREF(res);
                throw std::runtime_error("setInstanceStateFromRepresentation didn't return True.");
            }
            Py_DECREF(res);

            return instance;
        } else
        if (code == T_OBJECT_NAMED) {
            std::string name = b.readString();

            PyObject* res = PyObject_CallMethod(mContextObj, "objectFromName", "s", name.c_str());

            if (!res) {
                PyErr_PrintEx(1);
                throw std::runtime_error("objectFromName threw an exception.");
            }

            return b.addCachedPointer(id, incref(res), true);
        } else {
            throw std::runtime_error("corrupt data: invalid code after T_OBJECT");
        }
    }

    void serializeNativeType(Type* nativeType, SerializationBuffer& b) const {
        PyObject* nameForObject = PyObject_CallMethod(mContextObj, "nameForObject", "O", native_instance_wrapper::typeObj(nativeType));

        if (!nameForObject) {
            PyErr_PrintEx(1);
            throw std::runtime_error("nameForObject threw an exception.");
        }

        if (nameForObject != Py_None) {
            if (!PyUnicode_Check(nameForObject)) {
                PyErr_PrintEx(1);
                Py_DECREF(nameForObject);
                throw std::runtime_error("nameForObject returned something other than None or a string.");
            }

            b.write_uint8(0);
            b.write_string(std::string(PyUnicode_AsUTF8(nameForObject)));
            Py_DECREF(nameForObject);
            return;
        }
        b.write_uint8(1);

        b.write_uint8(nativeType->getTypeCategory());

        if (nativeType->getTypeCategory() == Type::TypeCategory::catInt64 ||
                nativeType->getTypeCategory() == Type::TypeCategory::catFloat64 ||
                nativeType->getTypeCategory() == Type::TypeCategory::catNone ||
                nativeType->getTypeCategory() == Type::TypeCategory::catBytes ||
                nativeType->getTypeCategory() == Type::TypeCategory::catString ||
                nativeType->getTypeCategory() == Type::TypeCategory::catBool
                ) {
            return;
        }

        if (nativeType->getTypeCategory() == Type::TypeCategory::catConcreteAlternative) {
            serializeNativeType(nativeType->getBaseType(), b);
            b.write_uint32(((ConcreteAlternative*)nativeType)->which());
            return;
        }

        if (nativeType->getTypeCategory() == Type::TypeCategory::catConstDict) {
            serializeNativeType(((ConstDict*)nativeType)->keyType(), b);
            serializeNativeType(((ConstDict*)nativeType)->valueType(), b);
            return;
        }

        if (nativeType->getTypeCategory() == Type::TypeCategory::catTupleOf) {
            serializeNativeType(((TupleOf*)nativeType)->getEltType(), b);
            return;
        }

        if (nativeType->getTypeCategory() == Type::TypeCategory::catTuple) {
            b.write_uint32(((CompositeType*)nativeType)->getTypes().size());
            for (auto t: ((CompositeType*)nativeType)->getTypes()) {
                serializeNativeType(t, b);
            }
            return;
        }

        if (nativeType->getTypeCategory() == Type::TypeCategory::catNamedTuple) {
            b.write_uint32(((CompositeType*)nativeType)->getTypes().size());
            for (long k = 0; k < ((CompositeType*)nativeType)->getTypes().size(); k++) {
                b.write_string(((CompositeType*)nativeType)->getNames()[k]);
                serializeNativeType(((CompositeType*)nativeType)->getTypes()[k], b);
            }
            return;
        }

        if (nativeType->getTypeCategory() == Type::TypeCategory::catOneOf) {
            b.write_uint32(((OneOf*)nativeType)->getTypes().size());
            for (auto t: ((OneOf*)nativeType)->getTypes()) {
                serializeNativeType(t, b);
            }
            return;
        }

        if (nativeType->getTypeCategory() == Type::TypeCategory::catPythonObjectOfType) {
            serializePythonObject((PyObject*)((PythonObjectOfType*)nativeType)->pyType(), b);
            return;
        }

        if (nativeType->getTypeCategory() == Type::TypeCategory::catValue) {
            Instance i = ((Value*)nativeType)->value();
            serializeNativeType(i.type(), b);
            i.type()->serialize(i.data(), b);
            return;
        }

        throw std::runtime_error("Can't serialize native type " + nativeType->name() + " if its unnamed.");
    }

    Type* deserializeNativeType(DeserializationBuffer& b) const {
        if (b.read_uint8() == 0) {
            //it's a named type
            std::string name = b.readString();
            PyObject* res = PyObject_CallMethod(mContextObj, "objectFromName", "s", name.c_str());

            if (!res) {
                PyErr_PrintEx(1);
                throw std::runtime_error("objectFromName threw an exception.");
            }

            if (!PyType_Check(res)) {
                std::string msg = "objectFromName returned a non-type for name " + name + ". it has type " + res->ob_type->tp_name;
                Py_DECREF(res);
                throw std::runtime_error(msg);
            }

            Type* resultType = native_instance_wrapper::extractTypeFrom((PyTypeObject*)res, true);
            Py_DECREF(res);
            if (!resultType) {
                throw std::runtime_error("we expected objectFromName to return a native type in this context, but it didn't.");
            }
            return resultType;
        }

        uint8_t category = b.read_uint8();
        if (category == Type::TypeCategory::catInt64) {
            return ::Int64::Make();
        }
        if (category == Type::TypeCategory::catFloat64) {
            return ::Float64::Make();
        }
        if (category == Type::TypeCategory::catNone) {
            return ::None::Make();
        }
        if (category == Type::TypeCategory::catBytes) {
            return ::Bytes::Make();
        }
        if (category == Type::TypeCategory::catString) {
            return ::String::Make();
        }
        if (category == Type::TypeCategory::catBool) {
            return ::Bool::Make();
        }

        if (category == Type::TypeCategory::catConcreteAlternative) {
            Type* base = deserializeNativeType(b);
            if (base->getTypeCategory() != Type::TypeCategory::catAlternative) {
                throw std::runtime_error("corrupt data: expected an Alternative type here");
            }
            uint32_t which = b.read_uint32();

            Alternative* a = (Alternative*)base;
            if (which >= a->subtypes().size()) {
                throw std::runtime_error("corrupt data: invalid alternative specified");
            }

            return ::ConcreteAlternative::Make(a,which);
        }


        if (category == Type::TypeCategory::catConstDict) {
            Type* keyType = deserializeNativeType(b);
            Type* valueType = deserializeNativeType(b);

            return ::ConstDict::Make(keyType, valueType);
        }

        if (category == Type::TypeCategory::catTupleOf) {
            return ::TupleOf::Make(
                deserializeNativeType(b)
                );
        }

        if (category == Type::TypeCategory::catTuple) {
            std::vector<Type*> types;
            size_t count = b.read_uint32();
            for (long k = 0; k < count; k++) {
                types.push_back(deserializeNativeType(b));
            }
            return ::Tuple::Make(types);
        }

        if (category == Type::TypeCategory::catNamedTuple) {
            std::vector<Type*> types;
            std::vector<std::string> names;
            size_t count = b.read_uint32();
            for (long k = 0; k < count; k++) {
                names.push_back(b.readString());
                types.push_back(deserializeNativeType(b));
            }
            return ::NamedTuple::Make(types, names);
        }

        if (category == Type::TypeCategory::catOneOf) {
            std::vector<Type*> types;
            size_t count = b.read_uint32();
            for (long k = 0; k < count; k++) {
                types.push_back(deserializeNativeType(b));
            }
            return ::OneOf::Make(types);
        }

        if (category == Type::TypeCategory::catPythonObjectOfType) {
            PyObject* typeObj = deserializePythonObject(b);
            return ::PythonObjectOfType::Make((PyTypeObject*)typeObj);
        }

        if (category == Type::TypeCategory::catValue) {
            Type* t = deserializeNativeType(b);

            Instance i = Instance::createAndInitialize(t, [&](instance_ptr p) {
                t->deserialize(p, b);
            });

            return ::Value::Make(i);
        }

        throw std::runtime_error("corrupt data. invalid native type code.");
    }

    virtual PyObject* deserializePythonObject(DeserializationBuffer& b) const {
        uint8_t code = b.read_uint8();
        if (code == T_LIST) {
            return deserializePyList(b);
        } else
        if (code == T_TUPLE) {
            return deserializePyTuple(b);
        } else
        if (code == T_DICT) {
            return deserializePydict(b);
        } else
        if (code == T_NATIVE) {
            Type* t = deserializeNativeType(b);
            return native_instance_wrapper::initialize(t, [&](instance_ptr selfData) {
                t->deserialize(selfData, b);
            });
        } else
        if (code == T_NATIVETYPE) {
            return (PyObject*)native_instance_wrapper::typeObj(deserializeNativeType(b));
        } else
        if (code == T_OBJECT) {
            return deserializePythonObjectNamedOrAsObj(b);
        } else
        if (code == T_NONE) {
            return incref(Py_None);
        } else
        if (code == T_TRUE) {
            return incref(Py_True);
        } else
        if (code == T_FALSE) {
            return incref(Py_False);
        } else
        if (code == T_UNICODE) {
            PyObject* result;
            uint32_t sz = b.read_uint32();
            b.read_bytes_fun(sz, [&](uint8_t* ptr) {
                result = PyUnicode_DecodeUTF8((const char*)ptr, sz, nullptr);
                if (!result) {
                    PyErr_PrintEx(1);
                    throw std::runtime_error("corrupt data: invalid unicode");
                }
            });
            return result;
        } else
        if (code == T_BYTES) {
            PyObject* result;
            uint32_t sz = b.read_uint32();
            b.read_bytes_fun(sz, [&](uint8_t* ptr) {
                result = PyBytes_FromStringAndSize((const char*)ptr, sz);
            });
            return result;
        } else
        if (code == T_FLOAT) {
            return PyFloat_FromDouble(b.read_double());
        } else if (code == T_LONG) {
            return PyLong_FromLong(b.read_int64());
        }

        throw std::runtime_error("corrupt data: invalid deserialization code");
    }


private:
    PyObject* mContextObj;
};

