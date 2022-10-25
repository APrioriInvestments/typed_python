/******************************************************************************
   Copyright 2017-2022 typed_python Authors

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

#include "ShaHash.hpp"
#include "util.hpp"
#include <unordered_map>

/******************************

This module provides services for walking the python object and Type object graph
with the same level of detail that the compiler does.  We use this to build a
unique hash for types and functions.

******************************/

bool isCanonicalName(std::string name) {
    // this is the list of standard library modules in python 3.8
    static std::set<std::string> canonicalPythonModuleNames({
        "abc", "aifc", "antigravity", "argparse", "ast", "asynchat", "asyncio", "asyncore",
        "base64", "bdb", "binhex", "bisect", "_bootlocale", "bz2", "calendar", "cgi", "cgitb",
        "chunk", "cmd", "codecs", "codeop", "code", "collections", "_collections_abc",
        "colorsys", "_compat_pickle", "compileall", "_compression", "concurrent",
        "configparser", "contextlib", "contextvars", "copy", "copyreg", "cProfile", "crypt",
        "csv", "ctypes", "curses", "dataclasses", "datetime", "dbm", "decimal", "difflib",
        "dis", "distutils", "doctest", "dummy_threading", "_dummy_thread", "email",
        "encodings", "ensurepip", "enum", "filecmp", "fileinput", "fnmatch", "formatter",
        "fractions", "ftplib", "functools", "__future__", "genericpath", "getopt", "getpass",
        "gettext", "glob", "gzip", "hashlib", "heapq", "hmac", "html", "http", "idlelib",
        "imaplib", "imghdr", "importlib", "imp", "inspect", "io", "ipaddress", "json",
        "keyword", "lib2to3", "linecache", "locale", "logging", "lzma", "mailbox", "mailcap",
        "marshal",
        "_markupbase", "mimetypes", "modulefinder", "msilib", "multiprocessing", "netrc",
        "nntplib", "ntpath", "nturl2path", "numbers", "opcode", "operator", "optparse", "os",
        "_osx_support", "pathlib", "pdb", "pickle", "pickletools", "pipes", "pkgutil",
        "platform", "plistlib", "poplib", "posixpath", "pprint", "profile", "pstats", "pty",
        "_py_abc", "pyclbr", "py_compile", "_pydecimal", "pydoc_data", "pydoc", "_pyio",
        "queue", "quopri", "random", "reprlib", "re", "rlcompleter", "runpy", "sched",
        "secrets", "selectors", "shelve", "shlex", "shutil", "signal", "_sitebuiltins",
        "site-packages", "site", "smtpd", "smtplib", "sndhdr", "socket", "socketserver",
        "sqlite3", "sre_compile", "sre_constants", "sre_parse", "ssl", "statistics", "stat",
        "stringprep", "string", "_strptime", "struct", "subprocess", "sunau", "symbol",
        "symtable", "sysconfig", "tabnanny", "tarfile", "telnetlib", "tempfile", "test",
        "textwrap", "this", "_threading_local", "threading", "timeit", "tkinter", "tokenize",
        "token", "traceback", "tracemalloc", "trace", "tty", "turtledemo", "turtle", "types",
        "typing", "unittest", "urllib", "uuid", "uu", "venv", "warnings", "wave", "weakref",
        "_weakrefset", "webbrowser", "wsgiref", "xdrlib", "xml", "xmlrpc", "zipapp",
        "zipfile", "zipimport", "pytz", "psutil",

        // and some standard ones we might commonly install
        "numpy", "pandas", "scipy", "pytest", "_pytest", "typed_python", "object_database", "llvmlite",
        "requests", "redis", "websockets", "boto3", "py", "xdist", "pytest_jsonreport",
        "pytest_metadata", "flask", "flaky", "coverage", "pyasn1", "cryptography", "paramiko",
        "six", "torch"
    });

    std::string moduleNameRoot;

    int posOfDot = name.find(".");
    if (posOfDot != std::string::npos) {
        moduleNameRoot = name.substr(0, posOfDot);
    } else {
        moduleNameRoot = name;
    }

    return canonicalPythonModuleNames.find(moduleNameRoot) != canonicalPythonModuleNames.end();
}

// is this a special name in a dict, module, or class that we shouldn't hash?
// we do want to hash methods like __init__
bool isSpecialIgnorableName(const std::string& name) {
    static std::set<std::string> canonicalMagicMethods({
        "__abs__", "__add__", "__and__", "__bool__",
        "__bytes__", "__call__", "__contains__", "__del__",
        "__delattr__", "__eq__", "__float__", "__floordiv__",
        "__format__", "__ge__", "__getitem__", "__gt__",
        "__hash__", "__iadd__", "__iand__", "__ieq__",
        "__ifloordiv__", "__ige__", "__igt__", "__ile__",
        "__ilshift__", "__ilt__", "__imatmul__", "__imod__",
        "__imul__", "__index__", "__ine__", "__init__",
        "__int__", "__invert__", "__ior__", "__ipow__",
        "__irshift__", "__isub__", "__itruediv__", "__ixor__",
        "__le__", "__len__", "__lshift__", "__lt__",
        "__matmul__", "__mod__", "__mul__", "__ne__",
        "__neg__", "__not__", "__or__", "__pos__",
        "__pow__", "__radd__", "__rand__", "__repr__",
        "__rfloordiv__", "__rlshift__", "__rmatmul__", "__rmod__",
        "__rmul__", "__ror__", "__round__", "__round__",
        "__rpow__", "__rrshift__", "__rshift__", "__rsub__",
        "__rtruediv__", "__rxor__", "__setattr__", "__setitem__",
        "__str__", "__sub__", "__truediv__", "__xor__",
    });

    return (
        name.substr(0, 2) == "__"
        && name.substr(name.size() - 2) == "__"
        && canonicalMagicMethods.find(name) == canonicalMagicMethods.end()
    );
}


class VisitRecord {
public:
    enum class kind { Hash=0, String=1, Instance=2, NameValuePair=3, Error=4 };

    VisitRecord() : mKind(kind::Error) {}

    VisitRecord(ShaHash hash) :
        mHash(hash),
        mKind(kind::Hash)
    {}

    VisitRecord(std::string name) :
        mName(name), mKind(kind::String)
    {}

    VisitRecord(std::string name, TypeOrPyobj instance)
        : mName(name), mInstance(instance), mKind(kind::NameValuePair)
    {}

    VisitRecord(TypeOrPyobj instance)
        : mInstance(instance), mKind(kind::Instance)
    {}

    static VisitRecord Err(std::string err) {
        VisitRecord res;
        res.mErr = err;
        return res;
    }

    bool operator==(const VisitRecord& other) const {
        if (mKind != other.mKind) {
            return false;
        }

        if (mKind == kind::Hash) {
            return mHash == other.mHash;
        }

        if (mKind == kind::Instance) {
            return mInstance == other.mInstance;
        }

        if (mKind == kind::String) {
            return mName == other.mName;
        }

        if (mKind == kind::NameValuePair) {
            return mName == other.mName && mInstance == other.mInstance;
        }

        if (mKind == kind::Error) {
            return mErr == other.mErr;
        }

        return true;
    }

    std::string err() const {
        return mErr;
    }

    std::string name() const {
        return mName;
    }

    TypeOrPyobj instance() const {
        return mInstance;
    }

    ShaHash hash() const {
        return mHash;
    }

    kind getKind() const {
        return mKind;
    }

    std::string toString() const {
        if (mKind == kind::Error) {
            return "Err(" + mErr + ")";
        }

        if (mKind == kind::String) {
            return "String(" + mName + ")";
        }
        if (mKind == kind::Hash) {
            return "Hash(" + mHash.digestAsHexString() + ")";
        }
        if (mKind == kind::Instance) {
            return "Instance(" + mInstance.name() + ")";
        }
        if (mKind == kind::NameValuePair) {
            return "NameValuePair(" + mName + "=" + mInstance.name() + ")";
        }

        return "<Unknown>";
    }

private:
    kind mKind;
    std::string mName;
    std::string mErr;
    ShaHash mHash;
    TypeOrPyobj mInstance;
};


class CompilerVisibleObjectVisitor {
public:
    static CompilerVisibleObjectVisitor& singleton() {
        static CompilerVisibleObjectVisitor* visitor = new CompilerVisibleObjectVisitor();

        return *visitor;
    }

    /*******
        This function defines  generic visitor pattern for looking inside of a Type or a PyObject to see
          which pieces of are visible to the compiler. We try to hold this all in one place so that we can
          have a single well-defined semantic for how we're visiting and hashing our objects.

        Our general rule is that objects visible at module level scope will never have their identities
        reassigned, nor will regular class members be reassigned. However, mutable containers may change.

        This function accepts a set of template parameters that get called with the internal pieces of the
        object:
            hashVisit(ShaHash): used to visit a single hash-hash
            nameVisit(string): used to visit a string (say, the name of a function)
            visit(TypeOrPyobj): looks at the actual instances
            namedVisitor(string, TypeOrPyobj): looks at (name, TypeOrPyobj) pairs (for walking dicts)
            onErr(): gets called if something odd happens (missing or badly typed member)
    ********/
    template<class visitor_1, class visitor_2, class visitor_3, class visitor_4, class visitor_5>
    void visit(
        TypeOrPyobj obj,
        const visitor_1& hashVisit,
        const visitor_2& nameVisit,
        const visitor_3& instanceVisit,
        const visitor_4& namedVisitor,
        const visitor_5& onErr
    ) {
        std::vector<VisitRecord> records = recordWalk(obj);

        auto it = mPastVisits.find(obj);
        if (it == mPastVisits.end()) {
            mPastVisits[obj] = records;
        } else {
            if (it->second != records) {
                checkForInstability();

                throw std::runtime_error(
                    "Found unstable object, but somehow our instability check"
                    " didn't throw an exception?" + obj.name()
                );
            }
        }

        walk(obj, hashVisit, nameVisit, instanceVisit, namedVisitor, onErr);
    }

    static std::vector<VisitRecord> recordWalk(TypeOrPyobj obj) {
        std::vector<VisitRecord> records;

        walk(
            obj,
            [&](ShaHash h) { records.push_back(VisitRecord(h)); },
            [&](std::string h) { records.push_back(VisitRecord(h)); },
            [&](TypeOrPyobj o) { records.push_back(VisitRecord(o)); },
            [&](std::string n, TypeOrPyobj o) { records.push_back(VisitRecord(n, o)); },
            [&](std::string err) { records.push_back(VisitRecord::Err(err)); }
        );

        return records;
    }

    void checkForInstability() {
        std::vector<TypeOrPyobj> unstable;

        for (auto it = mPastVisits.begin(); it != mPastVisits.end(); ++it) {
            if (it->second != recordWalk(it->first)) {
                unstable.push_back(it->first);
            }
        }

        if (!unstable.size()) {
            return;
        }

        std::ostringstream s;

        s << "Found " << unstable.size() << " unstable objects\n";

        for (long k = 0; k < unstable.size() && k < 1000; k++) {
            s << k << " -> " << unstable[k].name() << "\n";

            std::vector<std::string> linesLeft = stringifyVisitRecord(recordWalk(unstable[k]));
            std::vector<std::string> linesRight = stringifyVisitRecord(mPastVisits[unstable[k]]);

            auto pad = [&](std::string s, int ct) {
                if (s.size() > ct) {
                    return s.substr(0, ct);
                }

                return s + std::string(ct - s.size(), ' ');
            };

            for (long j = 0; j < linesLeft.size() || j < linesRight.size(); j++) {
                s << "    ";

                if (j < linesLeft.size()) {
                    s << pad(linesLeft[j], 80);
                } else {
                    s << pad("", 80);
                }

                s << "   |   ";

                if (j < linesRight.size()) {
                    s << pad(linesRight[j], 80);
                } else {
                    s << pad("", 80);
                }

                s << "\n";
            }
        }

        throw std::runtime_error(s.str());
    }

    std::vector<std::string> stringifyVisitRecord(const std::vector<VisitRecord>& records) {
        std::vector<std::string> lines;

        for (auto& record: records) {
            lines.push_back(record.toString());
        }

        return lines;
    }

private:
    template<class visitor_1, class visitor_2, class visitor_3, class visitor_4, class visitor_5>
    static void walk(
        TypeOrPyobj obj,
        const visitor_1& hashVisit,
        const visitor_2& nameVisit,
        const visitor_3& instanceVisit,
        const visitor_4& namedVisitor,
        const visitor_5& onErr
    ) {
        auto visitDict = [&](PyObject* d, bool ignoreSpecialNames=false) {
            if (!d) {
                hashVisit(0);
                return;
            }

            if (!PyDict_Check(d)) {
                onErr(std::string("not a dict: ") + d->ob_type->tp_name);
                return;
            }

            // get a list of the names in order. We have to walk them
            // in lexical order to make sure that our hash is stable.
            std::map<std::string, PyObject*> names;
            iterate(d, [&](PyObject* o) {
                if (PyUnicode_Check(o)) {
                    std::string name = PyUnicode_AsUTF8(o);

                    // we don't want module members to hash their file paths
                    // or their module loader info, because then they can't be
                    // moved around without violating the cache (and in fact their
                    // hashes are not stable at all)
                    if (!(ignoreSpecialNames && isSpecialIgnorableName(name))) {
                        names[name] = o;
                    }
                }
            });

            hashVisit(ShaHash(names.size()));

            for (auto nameAndO: names) {
                PyObject* val = PyDict_GetItem(d, nameAndO.second);
                if (!val) {
                    PyErr_Clear();
                    onErr("dict getitem empty");
                } else {
                    namedVisitor(nameAndO.first, val);
                }
            }
        };

        auto visitTuple = [&](PyObject* t) {
            if (!t) {
                hashVisit(ShaHash(0));
                return;
            }

            hashVisit(ShaHash(PyTuple_Size(t)));
            for (long k = 0; k < PyTuple_Size(t); k++) {
                instanceVisit(PyTuple_GetItem(t, k));
            }
        };

        if (obj.type()) {
            Type* objType = obj.type();

            hashVisit(ShaHash(1));

            objType->visitReferencedTypes(instanceVisit);

            // ensure that held and non-held versions of Class are
            // always visible to each other.
            if (objType->isHeldClass()) {
                Type* t = ((HeldClass*)objType)->getClassType();
                instanceVisit(t);
            }

            if (objType->isClass()) {
                Type* t = ((Class*)objType)->getHeldClass();
                instanceVisit(t);
            }

            objType->visitCompilerVisiblePythonObjects(instanceVisit);
            objType->visitCompilerVisibleInstances([&](Instance i) {
                instanceVisit(i.type());

                if (i.type()->getTypeCategory() == Type::TypeCategory::catPythonObjectOfType) {
                    return instanceVisit(i.cast<PythonObjectOfType::layout_type*>()->pyObj);
                }
            });

            return;
        }

        auto visitDictOrTuple = [&](PyObject* t) {
            if (!t) {
                hashVisit(ShaHash(0));
                return;
            }

            if (PyDict_Check(t)) {
                visitDict(t);
                return;
            }

            if (PyTuple_Check(t)) {
                visitTuple(t);
                return;
            }

            onErr("not a dict or tuple");
        };

        static PyObject* osModule = ::osModule();
        static PyObject* environType = PyObject_GetAttrString(osModule, "_Environ");

        if (obj.pyobj()->ob_type == (PyTypeObject*)environType) {
            // don't ever hash the environment.
            hashVisit(ShaHash(13));
            return;
        }

        // don't visit into constants
        if (MutuallyRecursiveTypeGroup::computePyObjectShaHashConstant(obj.pyobj()) != ShaHash()) {
            return;
        }

        Type* argType = PyInstance::extractTypeFrom(obj.pyobj()->ob_type);
        if (argType) {
            hashVisit(ShaHash(2));
            instanceVisit(argType);
            return;
        }

        // don't walk into canonical modules
        if (PyModule_Check(obj.pyobj())) {
            static PyObject* sysModule = ::sysModule();
            static PyObject* sysModuleModules = PyObject_GetAttrString(sysModule, "modules");

            PyObjectStealer name(PyObject_GetAttrString(obj.pyobj(), "__name__"));
            if (name) {
                if (PyUnicode_Check(name)) {
                    PyObjectStealer moduleObject(PyObject_GetItem(sysModuleModules, name));
                    if (moduleObject) {
                        if (moduleObject == obj.pyobj()) {
                            // this module is a canonical module. Lets not walk it as it's a standard
                            // system module
                            std::string moduleName = PyUnicode_AsUTF8(name);

                            //exclude modules that shouldn't change underneath us.
                            if (isCanonicalName(moduleName)) {
                                hashVisit(ShaHash(12));
                                nameVisit(moduleName);
                                return;
                            }
                        }
                    } else {
                        PyErr_Clear();
                    }
                }
            } else {
                PyErr_Clear();
            }
        }

        // this might be a named object. Let's see if its name actually resolves it correctly,
        // in which case we can hash its name (and its contents if the compiler could see
        // through it)
        if (MutuallyRecursiveTypeGroup::pyObjectGloballyIdentifiable(obj.pyobj())) {
            std::string moduleName = std::string(PyUnicode_AsUTF8(PyObject_GetAttrString(obj.pyobj(), "__module__")));
            std::string clsName = std::string(PyUnicode_AsUTF8(PyObject_GetAttrString(obj.pyobj(), "__name__")));

            if (isCanonicalName(moduleName) || obj.pyobj()->ob_type == &PyCFunction_Type) {
                hashVisit(ShaHash(2));
                nameVisit(moduleName + "|" + clsName);
                return;
            }
        }

        if (PyType_Check(obj.pyobj())) {
            argType = PyInstance::extractTypeFrom((PyTypeObject*)obj.pyobj());
            if (argType) {
                hashVisit(ShaHash(3));
                instanceVisit(argType);
                return;
            }
        }

        if (PyCode_Check(obj.pyobj())) {
            PyCodeObject* co = (PyCodeObject*)obj.pyobj();

            hashVisit(ShaHash(4));
            hashVisit(ShaHash(co->co_argcount));
            hashVisit(co->co_kwonlyargcount);
            hashVisit(co->co_nlocals);
            hashVisit(co->co_stacksize);
            // don't serialize the 'co_flags' field because it's not actually stable
            // and it doesn't contain any semantic information not available elsewhere.
            // hashVisit(co->co_flags);
            hashVisit(co->co_firstlineno);
            hashVisit(ShaHash::SHA1(PyBytes_AsString(co->co_code), PyBytes_GET_SIZE(co->co_code)));
            instanceVisit(co->co_consts);
            instanceVisit(co->co_names);
            instanceVisit(co->co_varnames);
            instanceVisit(co->co_freevars);
            instanceVisit(co->co_cellvars);
            // we ignore this, because otherwise, we'd have the hash change
            // whenever we instantiate code in a new location
            // visit(co->co_filename)
            instanceVisit(co->co_name);

    #       if PY_MINOR_VERSION >= 10
                instanceVisit(co->co_linetable);
    #       else
                instanceVisit(co->co_lnotab);
    #       endif
            return;
        }

        if (PyFunction_Check(obj.pyobj())) {
            hashVisit(ShaHash(5));

            PyFunctionObject* f = (PyFunctionObject*)obj.pyobj();

            if (f->func_closure) {
                hashVisit(ShaHash(PyTuple_Size(f->func_closure)));

                for (long k = 0; k < PyTuple_Size(f->func_closure); k++) {
                    PyObject* o = PyTuple_GetItem(f->func_closure, k);
                    if (o && PyCell_Check(o)) {
                        instanceVisit(o);
                    }
                }
            } else {
                hashVisit(ShaHash(0));
            }

            instanceVisit(f->func_name);
            instanceVisit(f->func_code);
            visitDictOrTuple(f->func_annotations);
            visitTuple(f->func_defaults);
            visitDictOrTuple(f->func_kwdefaults);

            hashVisit(ShaHash(1));

            if (f->func_globals && PyDict_Check(f->func_globals)) {

                std::vector<std::vector<PyObject*> > dotAccesses;

                Function::Overload::visitCompilerVisibleGlobals(
                    [&](std::string name, PyObject* val) {
                        if (!isSpecialIgnorableName(name)) {
                            namedVisitor(name, val);
                        }
                    },
                    (PyCodeObject*)f->func_code,
                    f->func_globals
                );
            }

            hashVisit(ShaHash(0));
            return;
        }

        if (PyType_Check(obj.pyobj())) {
            hashVisit(ShaHash(6));

            PyTypeObject* tp = (PyTypeObject*)obj.pyobj();

            hashVisit(ShaHash(0));
            if (tp->tp_dict) {
                visitDict(tp->tp_dict, true);
            }
            hashVisit(ShaHash(0));

            if (tp->tp_bases) {
                iterate(tp->tp_bases, instanceVisit);
            }

            hashVisit(ShaHash(0));

            return;
        }

        if (obj.pyobj()->ob_type == &PyStaticMethod_Type || obj.pyobj()->ob_type == &PyClassMethod_Type) {
            if (obj.pyobj()->ob_type == &PyStaticMethod_Type) {
                hashVisit(ShaHash(7));
            } else {
                hashVisit(ShaHash(8));
            }

            PyObjectStealer funcObj(PyObject_GetAttrString(obj.pyobj(), "__func__"));

            if (!funcObj) {
                onErr("not a func obj");
            } else {
                instanceVisit((PyObject*)funcObj);
            }

            return;
        }

        if (PyTuple_Check(obj.pyobj())) {
            hashVisit(ShaHash(9));
            hashVisit(ShaHash(PyTuple_Size(obj.pyobj())));

            for (long k = 0; k < PyTuple_Size(obj.pyobj()); k++) {
                instanceVisit(PyTuple_GetItem(obj.pyobj(), k));
            }

            return;
        }

        static PyObject* weakrefModule = ::weakrefModule();
        static PyObject* weakSetType = PyObject_GetAttrString(weakrefModule, "WeakSet");
        static PyObject* weakKeyDictType = PyObject_GetAttrString(weakrefModule, "WeakKeyDictionary");
        static PyObject* weakValueDictType = PyObject_GetAttrString(weakrefModule, "WeakValueDictionary");


        if (
            // dict, set and list are all mutable - we can't rely on their contents,
            // and the compiler shouldn't look inside of them anyways.
            PyDict_Check(obj.pyobj())
            || PySet_Check(obj.pyobj())
            || PyList_Check(obj.pyobj())
            // similarly, we shouldn't depend on the internals of a weakset/dict
            || obj.pyobj()->ob_type == (PyTypeObject*)weakSetType
            || obj.pyobj()->ob_type == (PyTypeObject*)weakKeyDictType
            || obj.pyobj()->ob_type == (PyTypeObject*)weakValueDictType
        ) {
            hashVisit(ShaHash(10));
            instanceVisit((PyObject*)obj.pyobj()->ob_type);
            return;
        }

        if (PyCell_Check(obj.pyobj())) {
            hashVisit(ShaHash(11));

            if (PyCell_Get(obj.pyobj())) {
                hashVisit(ShaHash(1));
                instanceVisit(PyCell_Get(obj.pyobj()));
            } else {
                hashVisit(ShaHash(0));
            }
            return;
        }

        // we do want to visit the internals of arbitrary objects, because
        // the compiler will attempt to do so as well.
        if (PyObject_HasAttrString(obj.pyobj(), "__dict__")) {
            PyObjectStealer dict(PyObject_GetAttrString(obj.pyobj(), "__dict__"));

            if (dict) {
                hashVisit(ShaHash(12));

                instanceVisit((PyObject*)obj.pyobj()->ob_type);
                visitDict(dict, true);
                return;
            }
        }

        instanceVisit((PyObject*)obj.pyobj()->ob_type);
    }

    std::unordered_map<TypeOrPyobj, std::vector<VisitRecord> > mPastVisits;
};
