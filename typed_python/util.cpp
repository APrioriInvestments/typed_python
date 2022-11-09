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

#include "util.hpp"
#include "Instance.hpp"
#include "PyInstance.hpp"

thread_local int64_t native_dispatch_disabled = 0;

bool unpackTupleToTypes(PyObject* tuple, std::vector<Type*>& out) {
    if (!PyTuple_Check(tuple)) {
        PyErr_SetString(PyExc_TypeError, "Argument to type tuple was not a tuple");
        return false;
    }
    for (int i = 0; i < PyTuple_Size(tuple); ++i) {
        PyObjectHolder entry(PyTuple_GetItem(tuple, i));
        Type* targetType = NULL;

        targetType = PyInstance::tryUnwrapPyInstanceToType(entry);

        if (!targetType) {
            PyErr_Format(PyExc_TypeError, "Expected a type in position %d of type tuple. Got %S instead.", i, (PyObject*)entry);
            return false;
        }

        out.push_back(targetType);
    }

    return true;
}

bool unpackTupleToStringAndTypes(PyObject* tuple, std::vector<std::pair<std::string, Type*> >& out) {
    std::set<std::string> memberNames;

    for (int i = 0; i < PyTuple_Size(tuple); ++i) {
        PyObjectHolder entry(PyTuple_GetItem(tuple, i));
        Type* targetType = NULL;

        if (!PyTuple_Check(entry) || PyTuple_Size(entry) != 2
                || !PyUnicode_Check(PyTuple_GetItem(entry, 0))
                || !(targetType =
                    PyInstance::tryUnwrapPyInstanceToType(
                        PyTuple_GetItem(entry, 1)
                        ))
                )
        {
            PyErr_SetString(PyExc_TypeError, "Badly formed class type argument.");
            return false;
        }

        std::string memberName = PyUnicode_AsUTF8(PyTuple_GetItem(entry, 0));

        if (memberNames.find(memberName) != memberNames.end()) {
            PyErr_Format(PyExc_TypeError, "Cannot redefine Class member %s", memberName.c_str());
            return false;
        }

        memberNames.insert(memberName);

        out.push_back(
            std::make_pair(memberName, targetType)
            );
    }

    return true;
}

bool unpackTupleToMemberDefinition(
    PyObject* tuple,
    std::vector<MemberDefinition>& out
) {
    std::set<std::string> memberNames;

    for (int i = 0; i < PyTuple_Size(tuple); ++i) {
        PyObjectHolder entry(PyTuple_GetItem(tuple, i));
        Type* targetType = NULL;

        if (!PyTuple_Check(entry) || PyTuple_Size(entry) != 4
                || !PyUnicode_Check(PyTuple_GetItem(entry, 0))
                || !PyBool_Check(PyTuple_GetItem(entry, 3))
                || !(targetType =
                    PyInstance::tryUnwrapPyInstanceToType(
                        PyTuple_GetItem(entry, 1)
                        ))
                )
        {
            PyErr_SetString(PyExc_TypeError, "Badly formed class MemberDefinition.");
            return false;
        }

        std::string memberName = PyUnicode_AsUTF8(PyTuple_GetItem(entry, 0));

        if (memberNames.find(memberName) != memberNames.end()) {
            PyErr_Format(PyExc_TypeError, "Cannot redefine Class member %s", memberName.c_str());
            return false;
        }

        memberNames.insert(memberName);

        PyObjectHolder entry_2(PyTuple_GetItem(entry, 2));

        Instance inst = PyInstance::unwrapPyObjectToInstance(entry_2, false);

        if (PyErr_Occurred()) {
            return false;
        }

        out.push_back(
            MemberDefinition(
                memberName,
                targetType,
                inst,
                PyTuple_GetItem(entry, 3) == Py_True
            )
        );
    }

    return true;
}

bool unpackTupleToStringAndObjects(PyObject* tuple, std::vector<std::pair<std::string, PyObject*> >& out) {
    std::set<std::string> memberNames;

    for (int i = 0; i < PyTuple_Size(tuple); ++i) {
        PyObjectHolder entry(PyTuple_GetItem(tuple, i));

        if (!PyTuple_Check(entry) || PyTuple_Size(entry) != 2
                || !PyUnicode_Check(PyTuple_GetItem(entry, 0))
                )
        {
            PyErr_SetString(PyExc_TypeError, "Badly formed class type argument.");
            return false;
        }

        std::string memberName = PyUnicode_AsUTF8(PyTuple_GetItem(entry,0));

        if (memberNames.find(memberName) != memberNames.end()) {
            PyErr_Format(PyExc_TypeError, "Cannot redefine Class member %s", memberName.c_str());
            return false;
        }

        memberNames.insert(memberName);

        out.push_back(
            std::make_pair(memberName, incref(PyTuple_GetItem(entry, 1)))
            );
    }

    return true;
}

PyObject* staticPythonInstance(std::string module, std::string member) {
    typedef std::pair<std::string, std::string> key_type;

    static std::map<key_type, PyObject*> cache;
    static std::mutex mutex;

    key_type key(module, member);

    // see if this is in our memo
    {
        std::lock_guard<std::mutex> lock(mutex);

        auto it = cache.find(key);

        if (it != cache.end()) {
            return it->second;
        }
    }

    // nope - we have to call into python and get it
    PyEnsureGilAcquired getTheGil;

    PyObject* val = PyImport_ImportModule(module.c_str());

    if (!val) {
        throw PythonExceptionSet();
    }

    size_t curOffset = 0;

    while (curOffset < member.size()) {
        size_t dot = member.find('.', curOffset);
        if (dot == std::string::npos) {
            dot = member.size();
        }

        std::string memberName = member.substr(curOffset, dot - curOffset);

        bool call = false;
        if (memberName.size() > 2 && memberName.substr(memberName.size() - 2) == "()") {
            call = true;
            memberName = memberName.substr(0, memberName.size() - 2);
        }

        val = PyObject_GetAttrString(
            val,
            memberName.c_str()
        );

        if (!val) {
            throw PythonExceptionSet();
        }

        if (call) {
            val = PyObject_CallFunction(val, "");

            if (!val) {
                throw PythonExceptionSet();
            }
        }

        curOffset = dot + 1;
    }

    std::lock_guard<std::mutex> lock(mutex);

    cache[key] = val;

    return val;
}
