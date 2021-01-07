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

#include "PyCompositeTypeInstance.hpp"

Type* PyCompositeTypeInstance::actualType() {
    return extractTypeFrom(((PyObject*)this)->ob_type);
}

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

void PyCompositeTypeInstance::copyConstructFromPythonInstanceConcrete(CompositeType* eltType, instance_ptr tgt, PyObject* pyRepresentation, ConversionLevel level) {
    if (level < ConversionLevel::Upcast) {
        PyInstance::copyConstructFromPythonInstanceConcrete(eltType, tgt, pyRepresentation, level);
        return;
    }

    ConversionLevel childConversionLevel = ConversionLevel::Implicit;

    if (level == ConversionLevel::New) {
        childConversionLevel = ConversionLevel::ImplicitContainers;
    }
    if (level == ConversionLevel::UpcastContainers) {
        childConversionLevel = ConversionLevel::UpcastContainers;
    }
    if (level == ConversionLevel::Upcast) {
        childConversionLevel = ConversionLevel::Upcast;
    }


    if (level < ConversionLevel::New) {
        // only allow implicit conversion from tuple/list of. We don't want dicts and sets to implicitly
        // convert to tuples
        bool isValid = false;
        if (PyTuple_Check(pyRepresentation) || PyList_Check(pyRepresentation)) {
            isValid = true;
        }

        std::pair<Type*, instance_ptr> typeAndPtrOfArg = extractTypeAndPtrFrom(pyRepresentation);
        if (typeAndPtrOfArg.first && (
                typeAndPtrOfArg.first->isListOf()
                || typeAndPtrOfArg.first->isTupleOf()
                || typeAndPtrOfArg.first->isComposite())
        ) {
            isValid = true;
        }

        if (!isValid) {
            PyInstance::copyConstructFromPythonInstanceConcrete(eltType, tgt, pyRepresentation, level);
            return;
        }
    }

    std::pair<Type*, instance_ptr> typeAndPtrOfArg = extractTypeAndPtrFrom(pyRepresentation);

    if (eltType->isNamedTuple() && typeAndPtrOfArg.first && typeAndPtrOfArg.first->isNamedTuple()) {
        NamedTuple* targetType = (NamedTuple*)eltType;
        NamedTuple* argType = (NamedTuple*)typeAndPtrOfArg.first;

        for (long k = 0; k < argType->getNames().size(); k++) {
            if (targetType->indexOfName(argType->getNames()[k]) == -1) {
                throw std::runtime_error(
                    "Can't convert a "
                    + argType->name()
                    + " to "
                    + targetType->name()
                    + " because the target doesn't have a member named "
                    + argType->getNames()[k]
                );
            }
        }

        for (long k = 0; k < targetType->getNames().size(); k++) {
            if (argType->indexOfName(targetType->getNames()[k]) == -1) {
                if (!targetType->getTypes()[k]->is_default_constructible()) {
                    throw std::runtime_error(
                        "Can't convert a "
                        + argType->name()
                        + " to "
                        + targetType->name()
                        + " because the type for field '"
                        + targetType->getNames()[k]
                        + "' is not default-constructible."
                    );
                }
            }
        }

        eltType->constructor(tgt,
            [&](uint8_t* eltPtr, int64_t k) {
                int otherIx = argType->indexOfName(eltType->getNames()[k]);

                if (otherIx == -1) {
                    //default constructor
                    eltType->getTypes()[k]->constructor(eltPtr);
                } else {
                    PyObjectStealer item(PyObject_GetAttrString(pyRepresentation, eltType->getNames()[k].c_str()));

                    PyInstance::copyConstructFromPythonInstance(
                        eltType->getTypes()[k],
                        eltPtr,
                        item,
                        childConversionLevel
                    );
                }

                return true;
            }
        );

        return;
    }

    int containerSize = PyObject_Length(pyRepresentation);
    if (containerSize == -1) {
        PyErr_Clear();
        PyInstance::copyConstructFromPythonInstanceConcrete(eltType, tgt, pyRepresentation, level);
        return;
    }

    if (containerSize != eltType->getTypes().size()) {
        throw std::runtime_error(
            "Can't convert a "
            + std::string(pyRepresentation->ob_type->tp_name)
            + " with "
            + format(containerSize)
            + " arguments to a "
            + eltType->name()
        );
    }

    PyObjectStealer iterator(PyObject_GetIter(pyRepresentation));

    if (iterator) {
        eltType->constructor(tgt,
            [&](uint8_t* eltPtr, int64_t k) {
                PyObjectStealer item(PyIter_Next(iterator));

                if (!item) {
                    if (PyErr_Occurred()) {
                        throw PythonExceptionSet();
                    }

                    return false;
                }

                PyInstance::copyConstructFromPythonInstance(eltType->getTypes()[k], eltPtr, item, childConversionLevel);

                return true;
            });

        return;
    } else {
        PyErr_Clear();
        PyInstance::copyConstructFromPythonInstanceConcrete(eltType, tgt, pyRepresentation, level);
        return;
    }

    PyInstance::copyConstructFromPythonInstanceConcrete(eltType, tgt, pyRepresentation, level);
}

bool PyCompositeTypeInstance::compare_to_python_concrete(CompositeType* tupT, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp) {
    auto convert = [&](char cmpValue) { return cmpResultToBoolForPyOrdering(pyComparisonOp, cmpValue); };

    Type* otherType = extractTypeFrom(other->ob_type);

    if (PyTuple_Check(other) || (otherType && (otherType->getTypeCategory() == Type::TypeCategory::catTupleOf || otherType->isComposite()))) {
        int lenS = tupT->getTypes().size();
        int indexInOwn = 0;

        int result = 0;

        iterateWithEarlyExit(other, [&](PyObject* tupleItem) {
            if (indexInOwn >= lenS) {
                // we ran out of items in our list
                result = -1;
                return false;
            }

            if (!compare_to_python(tupT->getTypes()[indexInOwn], tupT->eltPtr(self, indexInOwn), tupleItem, exact, Py_EQ)) {
                if (pyComparisonOp == Py_EQ || pyComparisonOp == Py_NE) {
                    result = 1;
                    return false;
                }
                if (compare_to_python(tupT->getTypes()[indexInOwn], tupT->eltPtr(self, indexInOwn), tupleItem, exact, Py_LT)) {
                    result = -1;
                    return false;
                }

                result = 1;
                return false;
            }

            indexInOwn += 1;

            return true;
        });

        if (result) {
            return convert(result);
        }

        if (indexInOwn == lenS) {
            return convert(0);
        }

        return convert(1);
    }

    if (pyComparisonOp == Py_EQ || pyComparisonOp == Py_NE) {
        return convert(1);
    }

    PyErr_Format(
        PyExc_TypeError,
        "Comparison not supported between instances of '%s' and '%s'.",
        tupT->name().c_str(),
        other->ob_type->tp_name
    );

    throw PythonExceptionSet();
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

void PyNamedTupleInstance::copyConstructFromPythonInstanceConcrete(
    NamedTuple* namedTupleT, instance_ptr tgt, PyObject* pyRepresentation, ConversionLevel level)
 {
    // allow implicit conversion of dict to named tuple.
    if (level >= ConversionLevel::UpcastContainers && PyDict_Check(pyRepresentation)) {
        static PyObject* emptyTuple = PyTuple_New(0);

        constructFromPythonArgumentsConcreteWithLevel(
            namedTupleT, tgt, emptyTuple, pyRepresentation,
            // convert at our level, but no higher than implicit containers
            level >= ConversionLevel::ImplicitContainers ?
                ConversionLevel::ImplicitContainers
            :   level
        );

        return;
    }

    PyCompositeTypeInstance::copyConstructFromPythonInstanceConcrete(namedTupleT, tgt, pyRepresentation, level);
}

void PyNamedTupleInstance::constructFromPythonArgumentsConcrete(
    NamedTuple* namedTupleT, uint8_t* data, PyObject* args, PyObject* kwargs
) {
    constructFromPythonArgumentsConcreteWithLevel(namedTupleT, data, args, kwargs, ConversionLevel::ImplicitContainers);
}

void PyNamedTupleInstance::constructFromPythonArgumentsConcreteWithLevel(
    NamedTuple* namedTupleT, uint8_t* data, PyObject* args, PyObject* kwargs, ConversionLevel level
) {
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

                // note that 'PyDict_GetItemString' returns a borrowed reference
                PyObject* o = PyDict_GetItemString(kwargs, name.c_str());
                if (o) {
                    try {
                        copyConstructFromPythonInstance(
                            t,
                            eltPtr,
                            o,
                            level
                        );
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
                else if (t->is_default_constructible()) {
                    t->constructor(eltPtr);
                } else {
                    throw std::logic_error(
                        "Can't default initialize member '" + name + "' of " + namedTupleT->name()
                        + " because it's not default-constructible"
                    );
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
    if (!PyUnicode_Check(attrName)) {
        PyErr_SetString(PyExc_AttributeError, "attribute is not a string");
        return -1;
    }

    int ix = type()->indexOfName(PyUnicode_AsUTF8(attrName));

    if (ix >= 0) {
        PyErr_Format(
            PyExc_AttributeError,
            "Cannot set attributes on instance of type '%s' because it is immutable",
            ((PyObject*)this)->ob_type->tp_name
        );
        return -1;
    }

    return PyInstance::tp_setattr_concrete(attrName, attrVal);
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
PyDoc_STRVAR(replacing_doc,
    "t.replacing(**kwargs) -> copy of t with updated fields and values\n"
    "\n"
    "Each keyword argument specifies a field and value to update.\n"
    );
PyObject* PyNamedTupleInstance::replacing(PyObject* o, PyObject* args, PyObject* kwargs) {
    PyNamedTupleInstance* self = (PyNamedTupleInstance*)o;

    NamedTuple* tupType = self->type();

    Type* tupOrSubclassType = self->actualType();

    // we should not allow passing any args, only kwargs are allowed
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

    // return a copy with updated values. We instantiate the subclass type. if it's a subclass,
    // then it will have the correct memory layout.
    return PyInstance::initialize(tupOrSubclassType, [&](instance_ptr newInstanceData) {
        //newInstanceData will point to the uninitialized memory we've allocated for the new tuple

        tupType->constructor(newInstanceData, [&](instance_ptr item_data, int index) {
            //item_data is a pointer to the uninitialized value in the 'index'th slot in the new tuple
            Type *itemType = tupType->getTypes()[index];
            std::string itemName = tupType->getNames()[index];

            // value passed in kwargs
            // note that 'value' is a borrowed reference
            PyObject* value = PyDict_GetItemString(kwargs, itemName.c_str());

            if (value != NULL) {
                PyInstance::copyConstructFromPythonInstance(
                        itemType,
                        item_data,
                        value,
                        ConversionLevel::ImplicitContainers
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
        {"replacing", (PyCFunction)PyNamedTupleInstance::replacing, METH_VARARGS | METH_KEYWORDS, replacing_doc},
        {NULL, NULL}
    };

}
