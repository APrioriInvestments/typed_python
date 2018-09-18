#include "Python.h"

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

struct module_state {
};

static PyObject *
return_self(PyObject *m) {
    Py_INCREF(m);
    return m;
}

static PyMethodDef module_methods[] = {
    {"return_self", (PyCFunction)return_self, METH_NOARGS, NULL},
    {NULL, NULL}
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_types",
        NULL,
        sizeof(struct module_state),
        module_methods,
        NULL,
        NULL,
        NULL,
        NULL
};

PyMODINIT_FUNC
PyInit__types(void)
{
    PyObject *module = PyModule_Create(&moduledef);

    if (module == NULL)
        return NULL;

    struct module_state *st = GETSTATE(module);

    return module;
}
