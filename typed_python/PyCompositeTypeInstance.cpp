/******************************************************************************
   Copyright 2017-2019 typed_python Authors

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

#include "PyCompositeTypeInstance.hpp"

CompositeType* PyCompositeTypeInstance::type() {
    Type* t = extractTypeFrom(((PyObject*)this)->ob_type);

    if (t->getBaseType()) {
        t = t->getBaseType();
    }

    if (t->getTypeCategory() != Type::TypeCategory::catNamedTuple &&
            t->getTypeCategory() != Type::TypeCategory::catTuple) {
        throw std::runtime_error("Invalid type object found in PyCompositeTypeInstance");
    }

    return (CompositeType*)t;
}

Tuple* PyTupleInstance::type() {
    Type* t = extractTypeFrom(((PyObject*)this)->ob_type);

    if (t->getBaseType()) {
        t = t->getBaseType();
    }

    if (t->getTypeCategory() != Type::TypeCategory::catTuple) {
        throw std::runtime_error("Invalid type object found in PyTupleInstance");
    }

    return (Tuple*)t;
}

NamedTuple* PyNamedTupleInstance::type() {
    Type* t = extractTypeFrom(((PyObject*)this)->ob_type);

    if (t->getBaseType()) {
        t = t->getBaseType();
    }

    if (t->getTypeCategory() != Type::TypeCategory::catNamedTuple) {
        throw std::runtime_error("Invalid type object found in PyTupleInstance");
    }

    return (NamedTuple*)t;
}

PyObject* PyCompositeTypeInstance::sq_item_concrete(Py_ssize_t ix) {
    if (ix < 0 || ix >= (int64_t)type()->getTypes().size()) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }

    Type* eltType = type()->getTypes()[ix];

    return extractPythonObject(type()->eltPtr(dataPtr(), ix), eltType);
}


Py_ssize_t PyCompositeTypeInstance::mp_and_sq_length_concrete() {
    return type()->getTypes().size();
}

int PyCompositeTypeInstance::pyInquiryConcrete(const char* op, const char* opErrRep) {
    // op == '__bool__'
    return type()->getTypes().size() != 0;
}

void PyNamedTupleInstance::copyConstructFromPythonInstanceConcrete(NamedTuple* namedTupleT, instance_ptr tgt, PyObject* pyRepresentation, bool isExplicit) {
    if (PyDict_Check(pyRepresentation)) {
        static PyObject* emptyTuple = PyTuple_New(0);

        constructFromPythonArgumentsConcrete(namedTupleT, tgt, emptyTuple, pyRepresentation, isExplicit);

        return;
    }

    PyCompositeTypeInstance::copyConstructFromPythonInstanceConcrete(namedTupleT, tgt, pyRepresentation, isExplicit);
}

void PyNamedTupleInstance::constructFromPythonArgumentsConcrete(NamedTuple* namedTupleT, uint8_t* data, PyObject* args, PyObject* kwargs, bool isExplicit) {
    if (kwargs) {
        iterate(kwargs, [&](PyObject* name) {
            if (!PyUnicode_Check(name)) {
                throw std::runtime_error(
                    "Can't construct an instance of "
                    + namedTupleT->name() + " with dictionary keys that aren't strings."
                );
            }

            const char* nameAsCstr = PyUnicode_AsUTF8(name);

            if (namedTupleT->getNameToIndex().find(nameAsCstr) == namedTupleT->getNameToIndex().end()) {
                throw std::runtime_error(
                    namedTupleT->name() + " doesn't have a member named '" + nameAsCstr + "'"
                );
            }
        });

        namedTupleT->constructor(data,
            [&](uint8_t* eltPtr, int64_t k) {
                const std::string& name = namedTupleT->getNames()[k];
                Type* t = namedTupleT->getTypes()[k];

                PyObject* o = PyDict_GetItemString(kwargs, name.c_str());
                if (o) {
                    try {
                        copyConstructFromPythonInstance(t, eltPtr, o, isExplicit);
                    } catch(PythonExceptionSet&) {
                        PyErr_Clear();
                        throw std::runtime_error(
                            "Can't initialize member '" + name + "' in " + namedTupleT->name() +
                            " with an instance of type " + o->ob_type->tp_name
                        );
                    } catch(...) {
                        throw std::runtime_error(
                            "Can't initialize member '" + name + "' in " + namedTupleT->name() +
                            " with an instance of type " + o->ob_type->tp_name
                        );
                    }
                }
                else if (namedTupleT->is_default_constructible()) {
                    t->constructor(eltPtr);
                } else {
                    throw std::logic_error("Can't default initialize member " + name + " of " + namedTupleT->name());
                }
            });

        return;
    }

    return PyInstance::constructFromPythonArgumentsConcrete(namedTupleT, data, args, kwargs);
}

PyObject* PyNamedTupleInstance::tp_getattr_concrete(PyObject* pyAttrName, const char* attrName) {
    //see if its a member of our held type
    int ix = type()->indexOfName(attrName);

    if (ix >= 0) {
        return extractPythonObject(
            type()->eltPtr(dataPtr(), ix),
            type()->getTypes()[ix]
            );
    }

    return PyInstance::tp_getattr_concrete(pyAttrName, attrName);
}

void PyTupleInstance::mirrorTypeInformationIntoPyTypeConcrete(Tuple* tupleT, PyTypeObject* pyType) {
    PyObject* res = PyTuple_New(tupleT->getTypes().size());
    for (long k = 0; k < tupleT->getTypes().size(); k++) {
        PyTuple_SetItem(res, k, incref(typePtrToPyTypeRepresentation(tupleT->getTypes()[k])));
    }
    //expose 'ElementType' as a member of the type object
    PyDict_SetItemString(pyType->tp_dict, "ElementTypes", res);
}

void PyNamedTupleInstance::mirrorTypeInformationIntoPyTypeConcrete(NamedTuple* tupleT, PyTypeObject* pyType) {
    PyObjectStealer types(PyTuple_New(tupleT->getTypes().size()));

    for (long k = 0; k < tupleT->getTypes().size(); k++) {
        PyTuple_SetItem(types, k, incref(typePtrToPyTypeRepresentation(tupleT->getTypes()[k])));
    }

    PyObjectStealer names(PyTuple_New(tupleT->getNames().size()));

    for (long k = 0; k < tupleT->getNames().size(); k++) {
        PyObject* namePtr = PyUnicode_FromString(tupleT->getNames()[k].c_str());
        PyTuple_SetItem(names, k, namePtr);
    }

    //expose 'ElementType' as a member of the type object
    PyDict_SetItemString(pyType->tp_dict, "ElementTypes", types);
    PyDict_SetItemString(pyType->tp_dict, "ElementNames", names);
}

int PyNamedTupleInstance::tp_setattr_concrete(PyObject* attrName, PyObject* attrVal) {
    PyErr_Format(
        PyExc_AttributeError,
        "Cannot set attributes on instance of type '%s' because it is immutable",
        type()->name().c_str()
    );
    return -1;
}

/**
 * Searches for the index of the element in the container.
 * Returns the position, if the element doesn't exist, returns -1.
 */
int PyNamedTupleInstance::findElementIndex(const std::vector<std::string>& container, const std::string &element) {
    auto it = std::find(container.begin(), container.end(), element);
    int index = it == container.end() ? -1 : std::distance(container.begin(), it);
    return index;
}

// static
PyObject* PyNamedTupleInstance::replacing(PyObject* o, PyObject* args, PyObject* kwargs) {
    PyNamedTupleInstance* self = (PyNamedTupleInstance*)o;

    NamedTuple* tupType = self->type();

    // we should not allow passing any args, only kwargs are allower
    if (!PyArg_ParseTuple(args, "")) {
        PyErr_Format(PyExc_ValueError, "Only keyword arguments are allowed.");
        return NULL;
    }

    // don't allow for calling the function without any arguments
    if (!kwargs) {
        PyErr_Format(PyExc_ValueError, "No arguments provided.");
        return NULL;
    }

    // fields from the tuple definition
    const std::vector<std::string>& names = self->type()->getNames();

    PyObject *key, *value;
    Py_ssize_t pos = 0;

    // check if the names are fine
    while (PyDict_Next(kwargs, &pos, &key, &value)) {
        std::string stringKey = std::string(PyUnicode_AsUTF8(key));
        int index = PyNamedTupleInstance::findElementIndex(names, stringKey);
        if (index == -1) {
            PyErr_Format(PyExc_ValueError, "Argument '%s' is not in the tuple definition.", stringKey.c_str());
            return nullptr;
        }
    }

    // return a copy with updated valuess
    return PyInstance::initialize(tupType, [&](instance_ptr newInstanceData) {
        //newInstanceData will point to the uninitialized memory we've allocated for the new tuple

        tupType->constructor(newInstanceData, [&](instance_ptr item_data, int index) {
            //item_data is a pointer to the uninitialized value in the 'index'th slot in the new tuple
            Type *itemType = tupType->getTypes()[index];
            std::string itemName = tupType->getNames()[index];

            // value passed in kwargs
            PyObject* value = PyDict_GetItemString(kwargs, itemName.c_str());

            if (value != NULL) {
                PyInstance::copyConstructFromPythonInstance(
                        itemType,
                        item_data,
                        value
                );
            } else {
                //on failure, PyDict_GetItemString doesn't actually throw an exception,
                //so we don't have to do anything.

                //we don't have a replacement, so copy the existing value over.
                itemType->copy_constructor(item_data,  tupType->eltPtr(self->dataPtr(), index));
            };
        });
    });

}

PyMethodDef* PyNamedTupleInstance::typeMethodsConcrete(Type* t) {

    return new PyMethodDef[2] {
        {"replacing", (PyCFunction)PyNamedTupleInstance::replacing, METH_VARARGS | METH_KEYWORDS, NULL},
        {NULL, NULL}
    };

}
