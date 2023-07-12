/******************************************************************************
   Copyright 2017-2023 typed_python Authors

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

#include "PyFunctionOverload.hpp"


PyDoc_STRVAR(PyFunctionOverload_doc,
    "A single overload of a Function type object.\n\n"
);

PyMethodDef PyFunctionOverload_methods[] = {
    {NULL}  /* Sentinel */
};

/* static */
void PyFunctionOverload::dealloc(PyFunctionOverload *self)
{
    Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject* PyFunctionOverload::newPyFunctionOverload(Function* f, int64_t overloadIndex) {
    PyFunctionOverload* self = (PyFunctionOverload*)PyType_FunctionOverload.tp_alloc(&PyType_FunctionOverload, 0);
    
    self->mFunction = f;
    self->mOverloadIx = overloadIndex;
    self->initFields();

    return (PyObject*)self;
}

/* static */
PyObject* PyFunctionOverload::new_(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyFunctionOverload* self;

    self = (PyFunctionOverload*)type->tp_alloc(type, 0);

    if (self != NULL) {
        self->mFunction = nullptr;
        self->mOverloadIx = 0;
    }

    return (PyObject*)self;
}

void PyFunctionOverload::initFields() {
    FunctionOverload& overload = mFunction->getOverloads()[mOverloadIx];

    PyObjectStealer pyClosureVarsDict(PyDict_New());

    // note that we can't actually call into the Python interpreter during this call,
    // because that can release the GIL and allow other threads to access our type
    // object before it's done.
    PyObjectStealer argsTup(PyTuple_New(overload.getArgs().size()));

    PyObject* closureVariableCellLookupSingleton = staticPythonInstance("typed_python.internals", "CellAccess");
    PyObject* funcOverloadArg = staticPythonInstance("typed_python.internals", "FunctionOverloadArg");
    
    PyObjectStealer pyFunctionGlobals(PyDict_New());

    for (auto nameAndGlobal: overload.getGlobals()) {
        PyObject* val = nameAndGlobal.second.getValueAsPyobj();
        if (val) {
            PyDict_SetItemString(
                pyFunctionGlobals, 
                nameAndGlobal.first.c_str(),
                PyFunctionGlobal::newPyFunctionGlobal(
                    mFunction,
                    mOverloadIx,
                    nameAndGlobal.first
                )
            );
        }
    }

    for (auto nameAndClosureVar: overload.getClosureVariableBindings()) {
        PyObjectStealer bindingObj(PyTuple_New(nameAndClosureVar.second.size()));

        for (long k = 0; k < nameAndClosureVar.second.size(); k++) {
            ClosureVariableBindingStep step = nameAndClosureVar.second[k];

            if (step.isFunction()) {
                // recall that 'PyTuple_SetItem' steals a reference, so we need to incref it here
                PyTuple_SetItem(bindingObj, k, incref((PyObject*)PyInstance::typePtrToPyTypeRepresentation(step.getFunction())));
            } else
            if (step.isNamedField()) {
                PyTuple_SetItem(bindingObj, k, PyUnicode_FromString(step.getNamedField().c_str()));
            } else
            if (step.isIndexedField()) {
                PyTuple_SetItem(bindingObj, k, PyLong_FromLong(step.getIndexedField()));
            } else
            if (step.isCellAccess()) {
                PyTuple_SetItem(bindingObj, k, incref(closureVariableCellLookupSingleton));
            } else {
                throw std::runtime_error("Corrupt ClosureVariableBindingStep encountered");
            }
        }

        PyDict_SetItemString(pyClosureVarsDict, nameAndClosureVar.first.c_str(), bindingObj);
    }

    PyObjectStealer emptyTup(PyTuple_New(0));
    PyObjectStealer emptyDict(PyDict_New());

    for (long argIx = 0; argIx < overload.getArgs().size(); argIx++) {
        auto arg = overload.getArgs()[argIx];

        PyObjectStealer pyArgInst(
            ((PyTypeObject*)funcOverloadArg)->tp_new((PyTypeObject*)funcOverloadArg, emptyTup, emptyDict)
        );

        PyObjectStealer pyArgInstDict(PyObject_GenericGetDict(pyArgInst, nullptr));

        PyObjectStealer pyName(PyUnicode_FromString(arg.getName().c_str()));
        PyDict_SetItemString(pyArgInstDict, "name", pyName);
        PyDict_SetItemString(pyArgInstDict, "defaultValue", arg.getDefaultValue() ? PyTuple_Pack(1, arg.getDefaultValue()) : Py_None);
        PyDict_SetItemString(pyArgInstDict, "_typeFilter", arg.getTypeFilter() ? (PyObject*)PyInstance::typePtrToPyTypeRepresentation(arg.getTypeFilter()) : Py_None);
        PyDict_SetItemString(pyArgInstDict, "isStarArg", arg.getIsStarArg() ? Py_True : Py_False);
        PyDict_SetItemString(pyArgInstDict, "isKwarg", arg.getIsKwarg() ? Py_True : Py_False);

        PyTuple_SetItem(argsTup, argIx, incref(pyArgInst));
    }

    PyObjectStealer pyOverloadInstDict(PyObject_GenericGetDict((PyObject*)this, nullptr));

    PyDict_SetItemString(pyOverloadInstDict, "functionTypeObject", PyInstance::typePtrToPyTypeRepresentation(mFunction));
    PyDict_SetItemString(pyOverloadInstDict, "index", (PyObject*)PyLong_FromLong(mOverloadIx));

    PyDict_SetItemString(pyOverloadInstDict, "closureVarLookups", (PyObject*)pyClosureVarsDict);
    PyDict_SetItemString(pyOverloadInstDict, "functionCode", (PyObject*)overload.getFunctionCode());
    PyDict_SetItemString(pyOverloadInstDict, "functionGlobals", (PyObject*)pyFunctionGlobals);
    PyDict_SetItemString(pyOverloadInstDict, "returnType", overload.getReturnType() ? (PyObject*)PyInstance::typePtrToPyTypeRepresentation(overload.getReturnType()) : Py_None);
    PyDict_SetItemString(pyOverloadInstDict, "signatureFunction", overload.getSignatureFunction() ? (PyObject*)overload.getSignatureFunction() : Py_None);
    PyDict_SetItemString(pyOverloadInstDict, "methodOf", overload.getMethodOf() ? (PyObject*)PyInstance::typePtrToPyTypeRepresentation(overload.getMethodOf()) : Py_None);
    PyDict_SetItemString(pyOverloadInstDict, "args", argsTup);
    PyDict_SetItemString(pyOverloadInstDict, "name", PyUnicode_FromString(mFunction->name().c_str()));
}

/* static */
int PyFunctionOverload::init(PyFunctionOverload *self, PyObject *args, PyObject *kwargs)
{
    PyErr_Format(PyExc_RuntimeError, "FunctionOverload cannot be initialized directly");
    return -1;
}

// static
PyObject* PyFunctionOverload::tp_repr(PyObject *selfObj) {
    PyFunctionOverload* self = (PyFunctionOverload*)selfObj;

    return PyUnicode_FromString(
        self->mFunction->getOverloads()[self->mOverloadIx].signatureStr().c_str()
    );
}

PyTypeObject PyType_FunctionOverload = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "FunctionOverload",
    .tp_basicsize = sizeof(PyFunctionOverload),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor) PyFunctionOverload::dealloc,
    #if PY_MINOR_VERSION < 8
    .tp_print = 0,
    #else
    .tp_vectorcall_offset = 0,                  // printfunc  (Changed to tp_vectorcall_offset in Python 3.8)
    #endif
    .tp_getattr = 0,
    .tp_setattr = 0,
    .tp_as_async = 0,
    .tp_repr = PyFunctionOverload::tp_repr,
    .tp_as_number = 0,
    .tp_as_sequence = 0,
    .tp_as_mapping = 0,
    .tp_hash = 0,
    .tp_call = 0,
    .tp_str = 0,
    .tp_getattro = 0,
    .tp_setattro = 0,
    .tp_as_buffer = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = PyFunctionOverload_doc,
    .tp_traverse = 0,
    .tp_clear = 0,
    .tp_richcompare = 0,
    .tp_weaklistoffset = 0,
    .tp_iter = 0,
    .tp_iternext = 0,
    .tp_methods = PyFunctionOverload_methods,
    .tp_members = 0,
    .tp_getset = 0,
    .tp_base = 0,
    .tp_dict = 0,
    .tp_descr_get = 0,
    .tp_descr_set = 0,
    .tp_dictoffset = 0,
    .tp_init = (initproc) PyFunctionOverload::init,
    .tp_alloc = 0,
    .tp_new = PyFunctionOverload::new_,
    .tp_free = 0,
    .tp_is_gc = 0,
    .tp_bases = 0,
    .tp_mro = 0,
    .tp_cache = 0,
    .tp_subclasses = 0,
    .tp_weaklist = 0,
    .tp_del = 0,
    .tp_version_tag = 0,
    .tp_finalize = 0,
};
