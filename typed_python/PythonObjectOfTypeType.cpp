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

// should return a reference to the object with an incref
PyObject* PythonObjectOfType::deepcopyPyObject(
    PyObject* o,
    DeepcopyContext& context
) {
    PyEnsureGilAcquired getTheGil;

    if (!o) {
        throw std::runtime_error("Can't deepcopy the null pyobj.");
    }

    if (context.alreadyAllocated.find((instance_ptr)o) != context.alreadyAllocated.end()) {
        return incref((PyObject*)context.alreadyAllocated[(instance_ptr)o]);
    }

    if (PyType_Check(o) || PyModule_Check(o)) {
        return incref(o);
    }

    if (Type* t = PyInstance::extractTypeFrom(o->ob_type)) {
        PyObject* res = PyInstance::initialize(t, [&](instance_ptr p) {
            PyEnsureGilReleased releaseTheGil;
            t->deepcopy(p, ((PyInstance*)o)->dataPtr(), context);
        });

        context.memoize(o, res);
        return res;
    }

    if (context.pyTypeMap.size()) {
        PyTypeObject* t = o->ob_type;

        while (t) {
            auto it = context.pyTypeMap.find((PyObject*)t);

            if (it != context.pyTypeMap.end()) {
                PyObject* result(
                    PyObject_CallFunction(
                        it->second,
                        "(O)",
                        o
                    )
                );

                if (!result) {
                    throw PythonExceptionSet();
                }

                context.memoize(o, result);
                return result;
            }

            t = t->tp_base;
        }
    }

    if (PyDict_Check(o)) {
        PyObject* res = PyDict_New();
        context.memoize(o, res);

        // this increfs 'res'
        PyObject *key, *value;
        Py_ssize_t pos = 0;

        while (PyDict_Next(o, &pos, &key, &value)) {
            PyObjectStealer k(deepcopyPyObject(key, context));
            PyObjectStealer v(deepcopyPyObject(value, context));

            // this takes a reference to k and v so we have to
            // decref them because the entomber returns it with an incref
            PyDict_SetItem(res, k, v);
        }

        return res;
    }

    if (PyList_Check(o)) {
        PyObject* res = PyList_New(PyList_Size(o));
        context.memoize(o, res);

        for (long k = 0; k < PyList_Size(o); k++) {
            PyObject* val = deepcopyPyObject(PyList_GetItem(o, k), context);
            PyList_SetItem(res, k, val);
        }

        return res;
    }

    if (PyTuple_Check(o) && PyTuple_Size(o)) {
        // don't try to deepcopy the empty tuple because it's a singleton
        PyObject* res = PyTuple_New(PyTuple_Size(o));

        if (!res) {
            throw PythonExceptionSet();
        }

        context.memoize(o, res);
        for (long k = 0; k < PyTuple_Size(o); k++) {
            PyObject* entombed = deepcopyPyObject(PyTuple_GetItem(o, k), context);
            //PyTuple_SET_ITEM steals a reference
            PyTuple_SET_ITEM(res, k, entombed);
        }

        return res;
    }

    if (PyTuple_Check(o)) {
        context.memoize(o, o);
        return incref(o);
    }

    if (PySet_Check(o)) {
        PyObject* res = PySet_New(nullptr);
        context.memoize(o, res);

        iterate(o, [&](PyObject* o2) {
            PyObjectStealer setItem(deepcopyPyObject(o2, context));
            PySet_Add(res, setItem);
        });

        return res;
    }

    if (PyLong_CheckExact(o) ||
        PyFloat_CheckExact(o) || 
        o == Py_None || 
        o == Py_True || 
        o == Py_False || 
        PyUnicode_CheckExact(o) || 
        PyBytes_CheckExact(o) || 
        PyComplex_CheckExact(o)        
    ) {
        // don't duplicate primitives
        return incref(o);
    }

    if ((
        PyObject_HasAttrString((PyObject*)o->ob_type, "__reduce__")
        || PyObject_HasAttrString((PyObject*)o->ob_type, "__reduce_ex__")
    ) && !PyFunction_Check(o)) {
        PyObjectHolder reduceRes;

        if (PyObject_HasAttrString((PyObject*)o->ob_type, "__reduce_ex__")) {
            static PyObject* four = PyLong_FromLong(4);

            reduceRes.steal(
                PyObject_CallMethod(o, "__reduce_ex__", "(O)", four)
            );
        } else {
            reduceRes.steal(
                PyObject_CallMethod(o, "__reduce__", NULL)
            );
        }

        if (!reduceRes) {
            // this can't be deepcopied
            PyErr_Clear();
            return incref(o);
        }

        if (PyUnicode_Check(reduceRes) || reduceRes == Py_None) {
            return incref(o);
        }

        if (!PyTuple_Check(reduceRes) 
                || PyTuple_Size(reduceRes) < 2 
                || PyTuple_Size(reduceRes) > 6) {
            PyObjectStealer repr(PyObject_Repr(o));
            if (!repr) {
                throw PythonExceptionSet();
            }

            throw std::runtime_error(
                "deepcopy encountered an invalid __reduce__ result for "
                + std::string(PyUnicode_AsUTF8(repr))
            );
        }

        // now take the first two arguments, and use them to produce the
        // interior object
        PyObjectStealer callable(
            deepcopyPyObject(PyTuple_GetItem(reduceRes, 0), context)
        );
        PyObjectStealer args(
            deepcopyPyObject(PyTuple_GetItem(reduceRes, 1), context)
        );

        if (!PyTuple_Check(args)) {
            PyObjectStealer repr(PyObject_Repr(o));
            if (!repr) {
                throw PythonExceptionSet();
            }

            throw std::runtime_error(
                "deepcopy encountered an invalid __reduce__ result for "
                + std::string(PyUnicode_AsUTF8(repr)) + ": callable arg was not a tuple"
            );
        }

        PyObjectStealer value(PyObject_Call(
            (PyObject*)callable,
            (PyObject*)args,
            NULL
        ));

        if (!value) {
            throw PythonExceptionSet();
        }

        // memoize the result since it may be referred to within the
        // state of our objects
        context.memoize(o, value);

        std::vector<PyObjectHolder> remainingArgs;

        for (long ix = 2; ix < PyTuple_Size(reduceRes); ix++) {
            remainingArgs.push_back(
                PyObjectStealer(
                    deepcopyPyObject(PyTuple_GetItem(reduceRes, ix), context)
                )
            );
        }

        // now implement the standard pickle protocol on the 
        // remaining arguments. Recall that the 3rd tuple element is the 'state'
        // the 4th and 5th are iterators of elements, and the 6th is a state setter
        if (remainingArgs.size() == 4 && remainingArgs[3] != Py_None) {
            PyObjectStealer setStateRes(
                PyObject_CallFunction(
                    (PyObject*)remainingArgs[3],
                    "OO",
                    (PyObject*)value,
                    (PyObject*)remainingArgs[0]
                )
            );
            if (!setStateRes) {
                throw PythonExceptionSet();
            }
        } else if (remainingArgs.size() > 0 && remainingArgs[0] != Py_None) {
            if (!PyObject_HasAttrString((PyObject*)value->ob_type, "__setstate__")) {
                PyObjectStealer repr(PyObject_Repr(o));
                if (!repr) {
                    throw PythonExceptionSet();
                }

                PyObjectStealer dict(
                    PyObject_GenericGetDict((PyObject*)value, nullptr)
                );

                if (!dict) {
                    throw PythonExceptionSet();
                }

                if (!PyDict_Check(dict)) {
                    throw std::runtime_error("GenericGetDict didn't return a dict!");
                }

                if (!PyDict_Check(remainingArgs[0])) {
                    throw std::runtime_error("remainingArgs[0] is not a dict!");
                }
                
                PyObjectStealer updateResult(
                    PyObject_CallMethod(
                        (PyObject*)dict, 
                        "update", 
                        "(O)", 
                        (PyObject*)remainingArgs[0]
                    )
                );

                if (!updateResult) {
                    throw PythonExceptionSet();
                }
            } else {
                PyObjectStealer setStateRes(
                    PyObject_CallMethod(
                        (PyObject*)value, 
                        "__setstate__", 
                        "(O)", 
                        (PyObject*)remainingArgs[0]
                    )
                );
                if (!setStateRes) {
                    throw PythonExceptionSet();
                }
            }
        }

        if (remainingArgs.size() > 1 && remainingArgs[1] != Py_None) {
            PyObjectStealer extendRes(
                PyObject_CallMethod(
                    (PyObject*)value, 
                    "extend", 
                    "(O)",
                    (PyObject*)remainingArgs[1]
                )
            );
            if (!extendRes) {
                throw PythonExceptionSet();
            }
        }

        if (remainingArgs.size() > 2 && remainingArgs[2] != Py_None) {
            iterate(remainingArgs[2], [&](PyObject* kvPair) {
                // TODO: should we be iterating here?
                static PyObject* zero = PyLong_FromLong(0);
                static PyObject* one = PyLong_FromLong(1);

                PyObjectStealer getItem0(PyObject_GetItem(kvPair, zero));
                
                if (!getItem0) {
                    throw PythonExceptionSet();
                }
                
                PyObjectStealer getItem1(PyObject_GetItem(kvPair, one));

                if (!getItem1) {
                    throw PythonExceptionSet();
                }
                
                if (PyObject_SetItem((PyObject*)value, getItem0, getItem1) == -1) {
                    throw PythonExceptionSet();
                }
            });
        }

        return incref((PyObject*)value);
    }

    return incref(o);
}

void PythonObjectOfType::deepcopyConcrete(
    instance_ptr dest,
    instance_ptr src,
    DeepcopyContext& context
) {
    layout_ptr& destPtr = *(layout_ptr*)dest;
    layout_ptr& srcPtr = *(layout_ptr*)src;

    // check if we already did this one
    if (context.alreadyAllocated.find((instance_ptr)srcPtr) != context.alreadyAllocated.end()) {
        destPtr = (layout_ptr)context.alreadyAllocated[(instance_ptr)srcPtr];
        destPtr->refcount++;
        return;
    }

    destPtr = (layout_ptr)context.slab->allocate(sizeof(layout_type), this);
    context.alreadyAllocated[(instance_ptr)srcPtr] = (instance_ptr)destPtr;

    destPtr->refcount = 1;

    destPtr->pyObj = deepcopyPyObject(srcPtr->pyObj, context);
}

size_t PythonObjectOfType::deepBytecountConcrete(
    instance_ptr instance,
    std::unordered_set<void*>& alreadyVisited,
    std::set<Slab*>* outSlabs
) {
    layout_type* layoutPtr = *(layout_type**)instance;

    if (alreadyVisited.find((void*)layoutPtr) != alreadyVisited.end()) {
        return 0;
    }

    alreadyVisited.insert((void*)layoutPtr);

    if (outSlabs && Slab::slabForAlloc(layoutPtr)) {
        outSlabs->insert(Slab::slabForAlloc(layoutPtr));
        return 0;
    }

    return bytesRequiredForAllocation(sizeof(layout_type)) + deepBytecountForPyObj(layoutPtr->pyObj, alreadyVisited, outSlabs);
}

size_t PythonObjectOfType::deepBytecountForPyObj(PyObject* o, std::unordered_set<void*>& alreadyVisited, std::set<Slab*>* outSlabs) {
    PyEnsureGilAcquired getTheGil;

    if (!o) {
        throw std::runtime_error("Can't deepcopy the null pyobj.");
    }

    if (alreadyVisited.find((void*)o) != alreadyVisited.end()) {
        return 0;
    }

    alreadyVisited.insert((void*)o);

    if (PyType_Check(o) || PyModule_Check(o)) {
        return 0;
    }

    if (Type* t = PyInstance::extractTypeFrom(o->ob_type)) {
        PyEnsureGilReleased releaseTheGil;

        return t->deepBytecount(((PyInstance*)o)->dataPtr(), alreadyVisited, outSlabs);
    }

    if (PyDict_Check(o)) {
        size_t res = 0;

        PyObject *key, *value;
        Py_ssize_t pos = 0;

        while (PyDict_Next(o, &pos, &key, &value)) {
            res += deepBytecountForPyObj(key, alreadyVisited, outSlabs);
            res += deepBytecountForPyObj(value, alreadyVisited, outSlabs);
        }

        return res;
    }

    if (PyList_Check(o)) {
        size_t res = 0;
        for (long k = 0; k < PyList_Size(o); k++) {
            res += deepBytecountForPyObj(PyList_GetItem(o, k), alreadyVisited, outSlabs);
        }
        return res;
    }

    if (PyTuple_Check(o)) {
        size_t res = 0;
        for (long k = 0; k < PyTuple_Size(o); k++) {
            res += deepBytecountForPyObj(PyTuple_GetItem(o, k), alreadyVisited, outSlabs);
        }
        return res;
    }

    if (PySet_Check(o)) {
        size_t res = 0;

        iterate(o, [&](PyObject* o2) { res += deepBytecountForPyObj(o2, alreadyVisited, outSlabs); });

        return res;
    }

    if (PyObject_HasAttrString(o, "__dict__") && !PyFunction_Check(o)) {
        PyObjectStealer dict(PyObject_GetAttrString(o, "__dict__"));

        if (dict) {
            return deepBytecountForPyObj(dict, alreadyVisited, outSlabs);
        }
    }

    return 0;
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
