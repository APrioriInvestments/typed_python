#include "AllTypes.hpp"

void PythonObjectOfType::repr(instance_ptr self, ReprAccumulator& stream) {
    PyObject* p = *(PyObject**)self;

    PyObject* o = PyObject_Repr(p);

    if (!o) {
        stream << "<EXCEPTION>";
        PyErr_Clear();
        return;
    }

    if (!PyUnicode_Check(o)) {
        stream << "<EXCEPTION>";
        Py_DECREF(o);
        return;
    }

    stream << PyUnicode_AsUTF8(o);

    Py_DECREF(o);
}

char PythonObjectOfType::cmp(instance_ptr left, instance_ptr right) {
    PyObject* l = *(PyObject**)left;
    PyObject* r = *(PyObject**)right;

    int res = PyObject_RichCompareBool(l, r, Py_EQ);
    if (res == -1) {
        PyErr_Clear();
        if (l < r) {
            return -1;
        }
        if (l > r) {
            return 1;
        }
        return 0;
    }

    if (res) {
        return 0;
    }

    res = PyObject_RichCompareBool(l, r, Py_LT);

    if (res == -1) {
        PyErr_Clear();
        if (l < r) {
            return -1;
        }
        if (l > r) {
            return 1;
        }
        return 0;
    }

    if (res) {
        return -1;
    }
    return 1;
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

