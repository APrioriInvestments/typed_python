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

void PythonObjectOfType::repr(instance_ptr self, ReprAccumulator& stream, bool isStr) {
    PyEnsureGilAcquired acquireTheGil;

    PyObject* p = getPyObj(self);

    PyObjectStealer o(isStr ? PyObject_Str(p) : PyObject_Repr(p));

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

size_t PythonObjectOfType::deepBytecountForPyObj(PyObject* o, std::unordered_set<void*>& alreadyVisited) {
    PyEnsureGilAcquired getTheGil;

    if (alreadyVisited.find((void*)o) != alreadyVisited.end()) {
        return 0;
    }

    alreadyVisited.insert((void*)o);

    if (PyType_Check(o)) {
        return sizeof(PyTypeObject);
    }

    if (Type* t = PyInstance::extractTypeFrom(o->ob_type)) {
        PyEnsureGilReleased releaseTheGil;

        return t->bytecount() + t->deepBytecount(((PyInstance*)o)->dataPtr(), alreadyVisited) + sizeof(PyInstance);
    }

    if (PyDict_Check(o)) {
        size_t res = 0;

        PyObject *key, *value;
        Py_ssize_t pos = 0;

        while (PyDict_Next(o, &pos, &key, &value)) {
            res += deepBytecountForPyObj(key, alreadyVisited);
            res += deepBytecountForPyObj(value, alreadyVisited);
        }

        return res;
    }

    if (PyList_Check(o)) {
        size_t res = 0;
        for (long k = 0; k < PyList_Size(o); k++) {
            res += deepBytecountForPyObj(PyList_GetItem(o, k), alreadyVisited);
        }
        return res;
    }

    if (PyTuple_Check(o)) {
        size_t res = 0;
        for (long k = 0; k < PyTuple_Size(o); k++) {
            res += deepBytecountForPyObj(PyTuple_GetItem(o, k), alreadyVisited);
        }
        return res;
    }

    if (PySet_Check(o)) {
        size_t res = 0;

        iterate(o, [&](PyObject* o2) { res += deepBytecountForPyObj(o2, alreadyVisited); });

        return res;
    }

    if (PyObject_HasAttrString(o, "__dict__")) {
        PyObjectStealer dict(PyObject_GetAttrString(o, "__dict__"));

        if (dict) {
            return sizeof(PyObject) + deepBytecountForPyObj(dict, alreadyVisited);
        }
    }

    return sizeof(PyObject);
}


bool PythonObjectOfType::cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions) {
    PyEnsureGilAcquired acquireTheGil;

    PyObject* l = getPyObj(left);
    PyObject* r = getPyObj(right);

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
PythonObjectOfType* PythonObjectOfType::Make(PyTypeObject* pyType, PyObject* givenType) {
    PyEnsureGilAcquired getTheGil;

    typedef std::pair<PyTypeObject*, PyObject*> keytype;

    static std::map<keytype, PythonObjectOfType*> m;

    keytype key(pyType, givenType);

    auto it = m.find(key);

    if (it == m.end()) {
        it = m.insert(
            std::make_pair(key, new PythonObjectOfType(pyType, givenType))
        ).first;
    }

    return it->second;
}

PythonObjectOfType* PythonObjectOfType::AnyPyObject() {
    static PyObject* module = internalsModule();
    static PyObject* t = PyObject_GetAttrString(module, "object");

    return Make((PyTypeObject*)t);
}

PythonObjectOfType* PythonObjectOfType::AnyPyType() {
    static PyObject* module = internalsModule();
    static PyObject* t = PyObject_GetAttrString(module, "type");

    return Make((PyTypeObject*)t);
}
