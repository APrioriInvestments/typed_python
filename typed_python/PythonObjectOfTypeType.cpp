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

#include "AllTypes.hpp"

void PythonObjectOfType::repr(instance_ptr self, ReprAccumulator& stream) {
    PyEnsureGilAcquired acquireTheGil;

    PyObject* p = *(PyObject**)self;

    PyObjectStealer o(PyObject_Repr(p));

    if (!o) {
        stream << "<EXCEPTION>";
        PyErr_Clear();
        return;
    }

    if (!PyUnicode_Check(o)) {
        stream << "<EXCEPTION>";
        return;
    }

    stream << PyUnicode_AsUTF8(o);
}

bool PythonObjectOfType::cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions) {
    PyEnsureGilAcquired acquireTheGil;

    PyObject* l = *(PyObject**)left;
    PyObject* r = *(PyObject**)right;

    int res = PyObject_RichCompareBool(l, r, pyComparisonOp);

    if (res == -1) {
        if (suppressExceptions) {
            PyErr_Clear();
            if (l->ob_type != r->ob_type) {
                return cmpResultToBoolForPyOrdering(pyComparisonOp, ::strcmp(l->ob_type->tp_name, r->ob_type->tp_name));
            } else {
                return cmpResultToBoolForPyOrdering(pyComparisonOp, l < r ? 1 : l > r ? -1 : 0);
            }
        }

        throw PythonExceptionSet();
    }

    return res;
}

// static
PythonObjectOfType* PythonObjectOfType::Make(PyTypeObject* pyType) {
    static std::mutex guard;

    std::lock_guard<std::mutex> lock(guard);

    typedef PyTypeObject* keytype;

    static std::map<keytype, PythonObjectOfType*> m;

    auto it = m.find(pyType);

    if (it == m.end()) {
        it = m.insert(
            std::make_pair(pyType, new PythonObjectOfType(pyType))
            ).first;
    }

    return it->second;
}

PythonObjectOfType* PythonObjectOfType::AnyPyObject() {
    static PyObject* module = PyImport_ImportModule("typed_python.internals");
    static PyObject* t = PyObject_GetAttrString(module, "object");

    return Make((PyTypeObject*)t);
}

PythonObjectOfType* PythonObjectOfType::AnyPyType() {
    static PyObject* module = PyImport_ImportModule("typed_python.internals");
    static PyObject* t = PyObject_GetAttrString(module, "type");

    return Make((PyTypeObject*)t);
}
