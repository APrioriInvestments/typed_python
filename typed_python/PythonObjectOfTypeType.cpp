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

bool PythonObjectOfType::cmp(instance_ptr left, instance_ptr right, int pyComparisonOp) {
    PyEnsureGilAcquired acquireTheGil;

    PyObject* l = *(PyObject**)left;
    PyObject* r = *(PyObject**)right;

    int res = PyObject_RichCompareBool(l, r, pyComparisonOp);

    if (res == -1) {
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


