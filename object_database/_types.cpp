#include <Python.h>
#include <numpy/arrayobject.h>
#include "../typed_python/AllTypes.hpp"
#include "../typed_python/PyInstance.hpp"
#include "PyVersionedIdSet.hpp"
#include "PyDatabaseObjectType.hpp"
#include "PyDatabaseConnectionState.hpp"
#include "PyView.hpp"

PyObject* createDatabaseObjectType(PyObject *none, PyObject* args, PyObject* kwargs)
{
    static const char *kwlist[] = {"schema", "name", NULL};
    PyObject* schema;
    const char* name;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Os", (char**)kwlist, &schema, &name)) {
        return nullptr;
    }

    return translateExceptionToPyObject([&] {
        return (PyObject*)PyDatabaseObjectType::createDatabaseObjectType(schema, name);
    });
}

static PyMethodDef module_methods[] = {
    {"createDatabaseObjectType", (PyCFunction)createDatabaseObjectType, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_types",
    .m_doc = NULL,
    .m_size = 0,
    .m_methods = module_methods,
    .m_slots = NULL,
    .m_traverse = NULL,
    .m_clear = NULL,
    .m_free = NULL
};

PyMODINIT_FUNC
PyInit__types(void)
{
    //initialize numpy. This is only OK because all the .cpp files get
    //glommed together in a single file. If we were to change that behavior,
    //then additional steps must be taken as per the API documentation.
    import_array();

    if (PyType_Ready(&PyType_VersionedIdSet) < 0)
        return NULL;

    if (PyType_Ready(&PyType_DatabaseConnectionState) < 0)
        return NULL;

    if (PyType_Ready(&PyType_View) < 0)
        return NULL;

    PyObject *module = PyModule_Create(&moduledef);

    if (module == NULL)
        return NULL;

    PyModule_AddObject(module, "VersionedIdSet", (PyObject *)&PyType_VersionedIdSet);
    PyModule_AddObject(module, "DatabaseConnectionState", (PyObject *)&PyType_DatabaseConnectionState);
    PyModule_AddObject(module, "View", (PyObject *)&PyType_View);

    return module;
}
