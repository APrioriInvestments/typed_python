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

#pragma once

#include <Python.h>

PyObject *MakeTupleOrListOfType(PyObject* nullValue, PyObject* args, bool isTuple);
PyObject *MakePointerToType(PyObject* nullValue, PyObject* args);
PyObject *MakeTupleOfType(PyObject* nullValue, PyObject* args);
PyObject *MakeListOfType(PyObject* nullValue, PyObject* args);
PyObject *MakeTupleType(PyObject* nullValue, PyObject* args);
PyObject *MakeConstDictType(PyObject* nullValue, PyObject* args);
PyObject* MakeSetType(PyObject* nullValue, PyObject* args);
PyObject *MakeDictType(PyObject* nullValue, PyObject* args);
PyObject *MakeOneOfType(PyObject* nullValue, PyObject* args);
PyObject *MakeNamedTupleType(PyObject* nullValue, PyObject* args, PyObject* kwargs);
PyObject *MakeBoolType(PyObject* nullValue, PyObject* args);
PyObject *MakeInt8Type(PyObject* nullValue, PyObject* args);
PyObject *MakeInt16Type(PyObject* nullValue, PyObject* args);
PyObject *MakeInt32Type(PyObject* nullValue, PyObject* args);
PyObject *MakeInt64Type(PyObject* nullValue, PyObject* args);
PyObject *MakeFloat32Type(PyObject* nullValue, PyObject* args);
PyObject *MakeFloat64Type(PyObject* nullValue, PyObject* args);
PyObject *MakeUInt8Type(PyObject* nullValue, PyObject* args);
PyObject *MakeUInt16Type(PyObject* nullValue, PyObject* args);
PyObject *MakeUInt32Type(PyObject* nullValue, PyObject* args);
PyObject *MakeUInt64Type(PyObject* nullValue, PyObject* args);
PyObject *MakeStringType(PyObject* nullValue, PyObject* args);
PyObject *MakeBytesType(PyObject* nullValue, PyObject* args);
PyObject *MakeEmbeddedMessageType(PyObject* nullValue, PyObject* args);
PyObject *MakePyCellType(PyObject* nullValue, PyObject* args);
PyObject *MakeTypedCellType(PyObject* nullValue, PyObject* args);
PyObject *MakeNoneType(PyObject* nullValue, PyObject* args);
PyObject *MakeValueType(PyObject* nullValue, PyObject* args);
PyObject *MakeBoundMethodType(PyObject* nullValue, PyObject* args);
PyObject *MakeAlternativeMatcherType(PyObject* nullValue, PyObject* args);
PyObject *MakeFunctionType(PyObject* nullValue, PyObject* args);
PyObject *MakeClassType(PyObject* nullValue, PyObject* args);
PyObject *MakeSubclassOfType(PyObject* nullValue, PyObject* args);
PyObject *MakeAlternativeType(PyObject* nullValue, PyObject* args, PyObject* kwargs);


typedef struct {
    PyObject_HEAD
    PyObject *prop_get;
    PyObject *prop_set;
    PyObject *prop_del;
    PyObject *prop_doc;

    #if PY_MINOR_VERSION >= 10
    PyObject *prop_name;
    #endif

    int getter_doc;
} JustLikeAPropertyObject;

typedef struct {
    PyObject_HEAD
    PyObject *cm_callable;
    PyObject *cm_dict;
} JustLikeAClassOrStaticmethod;
