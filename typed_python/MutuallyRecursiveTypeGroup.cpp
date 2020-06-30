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

#include "MutuallyRecursiveTypeGroup.hpp"

std::string TypeOrPyobj::name() {
    if (mType) {
        return "<Type " + mType->name() + ">";
    }

    if (mPyObj) {
        std::string lexical = MutuallyRecursiveTypeGroup::pyObjectSortName(mPyObj);
        if (lexical != "<UNNAMED>") {
            return "<PyObj named " + lexical + ">";
        }

        PyObjectStealer repr(PyObject_Repr(mPyObj));

        if (repr) {
            return "<PyObj of type " + std::string(mPyObj->ob_type->tp_name) +
                " with repr " + std::string(PyUnicode_AsUTF8(repr)).substr(0, 50) + ">";
            decref(repr);
        }

        return "<PyObj of type " + std::string(mPyObj->ob_type->tp_name) + ">";
    }

    throw std::runtime_error("Invalid TypeOrPyobj encountered.");
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
void visitCompilerVisibleTypesAndPyobjects(
    TypeOrPyobj obj,
    const visitor_1& hashVisit,
    const visitor_2& nameVisit,
    const visitor_3& visit,
    const visitor_4& namedVisitor,
    const visitor_5& onErr
) {
    auto visitDict = [&](PyObject* d) {
        if (!d) {
            hashVisit(0);
            return;
        }

        if (!PyDict_Check(d)) {
            onErr();
            return;
        }

        // get a list of the names in order. We have to walk them
        // in lexical order to make sure that our hash is stable.
        std::map<std::string, PyObject*> names;
        iterate(d, [&](PyObject* o) {
            if (PyUnicode_Check(o)) {
                names[PyUnicode_AsUTF8(o)] = o;
            }
        });

        hashVisit(ShaHash(names.size()));

        for (auto nameAndO: names) {
            PyObject* val = PyDict_GetItem(d, nameAndO.second);
            if (!val) {
                PyErr_Clear();
                onErr();
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
            visit(PyTuple_GetItem(t, k));
        }
    };

    if (obj.type()) {
        hashVisit(ShaHash(1));
        obj.type()->visitReferencedTypes(visit);
        obj.type()->visitCompilerVisiblePythonObjects(visit);

        obj.type()->visitCompilerVisibleInstances([&](Instance i) {
            if (i.type()->getTypeCategory() == Type::TypeCategory::catPythonObjectOfType) {
                return visit(i.cast<PythonObjectOfType::layout_type*>()->pyObj);
            }
        });

        return;
    }

    // don't visit into constants
    if (MutuallyRecursiveTypeGroup::computePyObjectShaHashConstant(obj.pyobj()) != ShaHash()) {
        return;
    }

    Type* argType = PyInstance::extractTypeFrom(obj.pyobj()->ob_type);
    if (argType) {
        hashVisit(ShaHash(2));
        visit(argType);
        return;
    }

    // don't walk into canonical modules
    if (PyModule_Check(obj.pyobj())) {
        static PyObject* sysModule = PyImport_ImportModule("sys");
        static PyObject* sysModuleModules = PyObject_GetAttrString(sysModule, "modules");

        PyObjectStealer name(PyObject_GetAttrString(obj.pyobj(), "__name__"));
        if (name) {
            if (PyUnicode_Check(name)) {
                PyObjectStealer moduleObject(PyObject_GetItem(sysModuleModules, name));
                if (moduleObject) {
                    if (moduleObject == obj.pyobj()) {
                        // this module is a canonical module. Lets not walk it its a standard
                        // system module
                        std::string moduleName = PyUnicode_AsUTF8(name);

                        // this is the list of standard library modules in python 3.8
                        static std::set<string> canonicalPythonModuleNames({
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
                            "zipfile", "zipimport",

                            // and some standard ones we might commonly install
                            "numpy", "pandas", "scipy", "pytest", "_pytest", "typed_python", "llvmlite",
                            "requests", "redis", "websockets", "boto3", "py", "xdist", "pytest_jsonreport",
                            "pytest_metadata", "flask", "flaky", "coverage"
                        });

                        std::string moduleNameRoot;

                        int posOfDot = moduleName.find(".");
                        if (posOfDot != std::string::npos) {
                            moduleNameRoot = moduleName.substr(0, posOfDot);
                        } else {
                            moduleNameRoot = moduleName;
                        }

                        //exclude modules that shouldn't change underneath us.
                        if (canonicalPythonModuleNames.find(moduleNameRoot) != canonicalPythonModuleNames.end()) {
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

        if (moduleName.substr(0, 13) == "typed_python." || obj.pyobj()->ob_type == &PyCFunction_Type) {
            hashVisit(ShaHash(2));
            nameVisit(moduleName + "|" + clsName);
            return;
        }
    }

    if (PyType_Check(obj.pyobj())) {
        argType = PyInstance::extractTypeFrom((PyTypeObject*)obj.pyobj());
        if (argType) {
            hashVisit(ShaHash(3));
            visit(argType);
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
        hashVisit(co->co_flags);
        hashVisit(co->co_firstlineno);
        hashVisit(ShaHash::SHA1(PyBytes_AsString(co->co_code), PyBytes_GET_SIZE(co->co_code)));
        visit(co->co_consts);
        visit(co->co_names);
        visit(co->co_varnames);
        visit(co->co_freevars);
        visit(co->co_cellvars);
        // we ignore this, because otherwise, we'd have the hash change
        // whenever we instantiate code in a new location
        // + visit(co->co_filename)
        visit(co->co_name);
        visit(co->co_lnotab);
        return;
    }

    if (PyFunction_Check(obj.pyobj())) {
        hashVisit(ShaHash(5));

        PyFunctionObject* f = (PyFunctionObject*)obj.pyobj();

        if (f->func_closure) {
            hashVisit(ShaHash(PyTuple_Size(f->func_closure)));

            for (long k = 0; k < PyTuple_Size(f->func_closure); k++) {
                PyObject* o = PyTuple_GetItem(f->func_closure, k);
                if (o && PyCell_Check(o) && PyCell_Get(o)) {
                    hashVisit(ShaHash(1));
                    visit(PyCell_Get(o));
                } else {
                    hashVisit(ShaHash(0));
                }
            }
        } else {
            hashVisit(ShaHash(0));
        }

        visit(f->func_name);
        visit(f->func_code);

        visitDict(f->func_annotations);
        visitTuple(f->func_defaults);
        visitDict(f->func_kwdefaults);

        hashVisit(ShaHash(1));

        if (f->func_globals && PyDict_Check(f->func_globals)) {
            std::set<std::string> names;

            // recursively walk through the code object and see which globals it reads from.
            // we'll assume it reads from all of its locals.
            MutuallyRecursiveTypeGroup::extractNamesFromCode((PyCodeObject*)f->func_code, names);

            for (auto name: names) {
                PyObject* val(PyDict_GetItemString(f->func_globals, name.c_str()));

                if (!val) {
                    PyErr_Clear();
                } else {
                    namedVisitor(name, val);
                }
            }
        }
        hashVisit(ShaHash(0));
        return;
    }

    if (PyType_Check(obj.pyobj())) {
        hashVisit(ShaHash(6));

        PyTypeObject* tp = (PyTypeObject*)obj.pyobj();

        hashVisit(ShaHash(0));
        if (tp->tp_dict) {
            visitDict(tp->tp_dict);
        }
        hashVisit(ShaHash(0));

        if (tp->tp_bases) {
            iterate(tp->tp_bases, visit);
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
            onErr();
        } else {
            visit((PyObject*)funcObj);
        }

        return;
    }

    if (PyTuple_Check(obj.pyobj())) {
        hashVisit(ShaHash(9));
        hashVisit(ShaHash(PyTuple_Size(obj.pyobj())));

        for (long k = 0; k < PyTuple_Size(obj.pyobj()); k++) {
            visit(PyTuple_GetItem(obj.pyobj(), k));
        }

        return;
    }

    static PyObject* weakrefModule = PyImport_ImportModule("weakref");
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
        visit((PyObject*)obj.pyobj()->ob_type);
        return;
    }

    if (PyObject_HasAttrString(obj.pyobj(), "__dict__")) {
        PyObjectStealer dict(PyObject_GetAttrString(obj.pyobj(), "__dict__"));

        if (dict) {
            hashVisit(ShaHash(11));

            visit((PyObject*)obj.pyobj()->ob_type);

            visitDict(dict);
            return;
        }
    }

    visit((PyObject*)obj.pyobj()->ob_type);
}


//static
void MutuallyRecursiveTypeGroup::visibleFrom(TypeOrPyobj root, std::vector<TypeOrPyobj>& outReachable) {
    visitCompilerVisibleTypesAndPyobjects(
        root,
        [&](ShaHash h) {},
        [&](const std::string& s) {},
        [&](TypeOrPyobj t) { outReachable.push_back(t); },
        [&](const std::string& s, TypeOrPyobj t) { outReachable.push_back(t); },
        [&]() {}
    );
}

std::string MutuallyRecursiveTypeGroup::pyObjectSortName(PyObject* o) {
    if (PyObject_HasAttrString(o, "__module__") && PyObject_HasAttrString(o, "__name__")) {
        std::string modulename, clsname;

        PyObjectStealer moduleName(PyObject_GetAttrString(o, "__module__"));
        if (!moduleName) {
            PyErr_Clear();
        } else {
            if (PyUnicode_Check(moduleName)) {
                modulename = std::string(PyUnicode_AsUTF8(moduleName));
            }
        }

        PyObjectStealer clsName(PyObject_GetAttrString(o, "__name__"));
        if (!clsName) {
            PyErr_Clear();
        } else {
            if (PyUnicode_Check(clsName)) {
                modulename = std::string(PyUnicode_AsUTF8(clsName));
            }
        }

        if (clsname.size()) {
            if (modulename.size()) {
                return modulename + "|" + clsname;
            }

            return clsname;
        }
    }

    return "<UNNAMED>";
}


// these types can all see each other through their references, either
// through the compiler, or just through normal type references. We need to
// pick a 'first' type, which we can do by picking the first type to be defined
// in the program, and then walk through the group placing them in order..

// static
void MutuallyRecursiveTypeGroup::buildCompilerRecursiveGroup(const std::set<TypeOrPyobj>& types) {
    MutuallyRecursiveTypeGroup* group = new MutuallyRecursiveTypeGroup();

    if (types.size() == 0) {
        throw std::runtime_error("Empty compiler recursive group makes no sense.");
    }

    // we need to find the lexically lowest unique type or named pyobject in the group.
    std::map<std::string, std::vector<TypeOrPyobj> > names;
    for (auto t: types) {
        names[t.name()].push_back(t);
    }

    TypeOrPyobj* root = nullptr;

    for (auto& nameAndTypes: names) {
        if (nameAndTypes.second.size() == 1) {
            root = &nameAndTypes.second[0];
            break;
        }
    }

    if (!root) {
        group->mAnyPyObjectsIncorrectlyOrdered = true;
        // just pick the first one. It's not stable.
        for (auto& nameAndTypes: names) {
            root = &nameAndTypes.second[0];
            break;
        }

        if (!root) {
            throw std::runtime_error("this should never happen");
        }
    }

    std::map<TypeOrPyobj, int> ordering;

    // now walk our object graph from the root and observe the objects
    // in the order in which we find them. This should be stable across
    // program invocations and also serialization.
    std::function<void (TypeOrPyobj)> visit = [&](TypeOrPyobj parent) {
        if (types.find(parent) == types.end() || ordering.find(parent) != ordering.end()) {
            return;
        }

        // uncomment to get a better handle on which objects are in which groups.
        // std::cout << "    Group item " << ordering.size() << " is " << parent.name() << "\n";

        // visitCompilerVisibleTypesAndPyobjects(
        //     parent,
        //     [&](ShaHash h) {},
        //     [&](const std::string& s) {},
        //     [&](TypeOrPyobj t) { std::cout << "      -> " << t.name() << "\n"; },
        //     [&](const std::string& s, TypeOrPyobj t) { std::cout << "      -> " << t.name() << "\n"; },
        //     [&]() {}
        // );

        int index = ordering.size();
        ordering[parent] = index;

        visitCompilerVisibleTypesAndPyobjects(
            parent,
            [&](ShaHash h) {},
            [&](const std::string& s) {},
            [&](TypeOrPyobj t) { visit(t); },
            [&](const std::string& s, TypeOrPyobj t) { visit(t); },
            [&]() {}
        );
    };

    // std::cout << "Finish group with " << types.size() << " items\n";

    visit(*root);

    if (ordering.size() != types.size()) {
        throw std::runtime_error("Couldn't find all the types: " + format(ordering.size()) + " vs " + format(types.size()));
    }

    for (auto& typeAndOrder: ordering) {
        if (typeAndOrder.first.pyobj()) {
            PyObject* o = typeAndOrder.first.pyobj();

            mPythonObjectTypeGroups[o] = std::make_pair(group, typeAndOrder.second);
        } else {
            Type* t = typeAndOrder.first.type();
            t->setRecursiveTypeGroup(group, typeAndOrder.second);
        }

        group->mIndexToObject.insert({typeAndOrder.second, typeAndOrder.first});
    }

    group->hash();
}

void MutuallyRecursiveTypeGroup::computeHash() {
    if (mAnyPyObjectsIncorrectlyOrdered) {
        mHash = ShaHash::poison();
        return;
    }

    // we are a recursive group head. We want to compute the hash
    // of all of our constituents where, when each of them looks at
    // other types within _our_ group, we simply hash in a placeholder
    // for the position.
    ShaHash wholeGroupHash;

    for (auto idAndType: mIndexToObject) {
        Type* t = idAndType.second.type();

        if (t) {
            // this actually walks the type and computes a sha-hash based on
            // what's inside the type one layer down.
            wholeGroupHash += t->computeIdentityHash(this);

            if (t->isRecursive()) {
                wholeGroupHash += ShaHash(t->name());
            }
        } else {
            wholeGroupHash += computePyObjectShaHash(idAndType.second.pyobj(), this);
        }
    }

    mHash = wholeGroupHash;
}

// find strongly-connected groups of python objects and Type objects (as far as the
// compiler is concerned)
class MutuallyRecursiveTypeGroupSearch {
public:
    MutuallyRecursiveTypeGroupSearch() {}

    // this object is new and needs its own group
    void pushGroup(TypeOrPyobj o) {
        if (mInAGroup.find(o) != mInAGroup.end()) {
            throw std::runtime_error("Can't create a new group for an object we're already considering");
        }

        mInAGroup.insert(o);

        mGroups.push_back(
            std::shared_ptr<std::set<TypeOrPyobj> >(
                new std::set<TypeOrPyobj>()
            )
        );

        mGroupOutboundEdges.push_back(
            std::shared_ptr<std::set<TypeOrPyobj> >(
                new std::set<TypeOrPyobj>()
            )
        );

        mGroups.back()->insert(o);

        // now recurse into the subtypes
        visitCompilerVisibleTypesAndPyobjects(
            o,
            [&](ShaHash h) {},
            [&](std::string o) {},
            [&](TypeOrPyobj o) {
                if (MutuallyRecursiveTypeGroup::objectIsUnassigned(o)) {
                    mGroupOutboundEdges.back()->insert(o);
                }
            },
            [&](std::string name, PyObject* o) {
                if (MutuallyRecursiveTypeGroup::objectIsUnassigned(o)) {
                    mGroupOutboundEdges.back()->insert(o);
                }
            },
            [&]() {}
        );
    }

    void findAllGroups() {
        while (mGroups.size()) {
            doOneStep();
        }
    }

    void doOneStep() {
        if (mGroupOutboundEdges.back()->size() == 0) {
            // this group is finished....
            MutuallyRecursiveTypeGroup::buildCompilerRecursiveGroup(*mGroups.back());

            for (auto gElt: *mGroups.back()) {
                mInAGroup.erase(gElt);
            }

            mGroups.pop_back();
            mGroupOutboundEdges.pop_back();
        } else {
            // pop an outbound edge and see where does it go?
            TypeOrPyobj o = *mGroupOutboundEdges.back()->begin();
            mGroupOutboundEdges.back()->erase(o);

            if (!MutuallyRecursiveTypeGroup::objectIsUnassigned(o)) {
                return;
            }

            if (mInAGroup.find(o) == mInAGroup.end()) {
                // this is new
                pushGroup(o);
            } else {
                // this is above us somewhere
                while (mGroups.size() && mGroups.back()->find(o) == mGroups.back()->end()) {
                    // merge these two mGroups together
                    mGroups[mGroups.size() - 2]->insert(
                        mGroups.back()->begin(), mGroups.back()->end()
                    );
                    mGroups.pop_back();

                    mGroupOutboundEdges[mGroupOutboundEdges.size() - 2]->insert(
                        mGroupOutboundEdges.back()->begin(), mGroupOutboundEdges.back()->end()
                    );
                    mGroupOutboundEdges.pop_back();
                }

                if (!mGroups.size()) {
                    // this shouldn't ever happen, but we want to know if it does
                    throw std::runtime_error("Never found parent of object!");
                }
            }
        }
    }

private:
    // everything in a group somewhere
    std::set<TypeOrPyobj> mInAGroup;

    // each group
    std::vector<std::shared_ptr<std::set<TypeOrPyobj> > > mGroups;

    // for each group, the set of things it reaches out to
    std::vector<std::shared_ptr<std::set<TypeOrPyobj> > > mGroupOutboundEdges;
};

bool MutuallyRecursiveTypeGroup::objectIsUnassigned(TypeOrPyobj obj) {
    // exclude any type that already has a recursive type group head.
    if (obj.type() && obj.type()->hasTypeGroup()) {
        return false;
    }

    if (obj.pyobj() && mPythonObjectTypeGroups.find(obj.pyobj()) != mPythonObjectTypeGroups.end()) {
        // if this is a python object we've already given a hash to
        return false;
    }

    // if its a constant not worth delving into
    if (obj.pyobj() && isSimpleConstant(obj.pyobj())) {
        return false;
    }

    return true;
}

// static
void MutuallyRecursiveTypeGroup::constructRecursiveTypeGroup(TypeOrPyobj root) {
    // we do things with pyobj refcounts, so we need to hold the gil.
    PyEnsureGilAcquired getTheGil;

    MutuallyRecursiveTypeGroupSearch groupFinder;

    static thread_local int count = 0;
    count++;
    if (count > 1) {
        throw std::runtime_error(
            "There should be only one group algo running at once. Somehow, "
            "our reference to " + root.name() + " wasn't captured correctly."
        );
    }
    try {
        groupFinder.pushGroup(root);

        groupFinder.findAllGroups();
    } catch(...) {
        count --;
        throw;
    }
    count--;
}

// is this object is globally identifiable by module name and object name?

// static
bool MutuallyRecursiveTypeGroup::pyObjectGloballyIdentifiable(PyObject* h) {
    static PyObject* sysModule = PyImport_ImportModule("sys");
    static PyObject* sysModuleModules = PyObject_GetAttrString(sysModule, "modules");

    if (PyObject_HasAttrString(h, "__module__") && PyObject_HasAttrString(h, "__name__")) {
        PyObjectStealer moduleName(PyObject_GetAttrString(h, "__module__"));
        if (!moduleName) {
            PyErr_Clear();
            return false;
        }

        PyObjectStealer clsName(PyObject_GetAttrString(h, "__name__"));
        if (!clsName) {
            PyErr_Clear();
            return false;
        }

        if (!PyUnicode_Check(moduleName) || !PyUnicode_Check(clsName)) {
            return false;
        }

        PyObjectStealer moduleObject(PyObject_GetItem(sysModuleModules, moduleName));

        if (!moduleObject) {
            PyErr_Clear();
            return false;
        }

        PyObjectStealer obj(PyObject_GetAttr(moduleObject, clsName));

        if (!obj) {
            PyErr_Clear();
            return false;
        }
        if ((PyObject*)obj == h) {
            return true;
        }
    }

    return false;
}


// static
void MutuallyRecursiveTypeGroup::extractNamesFromCode(PyCodeObject* code, std::set<std::string>& outNames) {
    if (!PyCode_Check(code)) {
        return;
    }

    auto adder = [&](PyObject* o) {
        if (PyUnicode_Check(o)) {
            outNames.insert(PyUnicode_AsUTF8(o));
        }
    };

    iterate(code->co_names, adder);
    iterate(code->co_consts, [&](PyObject* o) {
        if (PyCode_Check(o)) {
            extractNamesFromCode((PyCodeObject*)o, outNames);
        }
    });
}


// static
ShaHash MutuallyRecursiveTypeGroup::pyObjectShaHashByVisiting(PyObject* obj, MutuallyRecursiveTypeGroup* groupHead) {
    ShaHash res;

    visitCompilerVisibleTypesAndPyobjects(
        obj,
        [&](ShaHash h) {
            res += h;
        },
        [&](std::string o) {
            res += ShaHash(o);
        },
        [&](TypeOrPyobj o) {
            if (o.type()) {
                res += o.type()->identityHash(groupHead);
            } else {
                res += pyObjectShaHash(o.pyobj(), groupHead);
            }
        },
        [&](std::string name, PyObject* o) {
            res += ShaHash(name) + pyObjectShaHash(o, groupHead);
        },
        [&]() {
            res += ShaHash::poison();
        }
    );

    return res;
}

// static
ShaHash MutuallyRecursiveTypeGroup::pyObjectShaHash(PyObject* h, MutuallyRecursiveTypeGroup* groupHead) {
    assertHoldingTheGil();

    if (!h) {
        return ShaHash(0);
    }

    // check if its a simple constant
    ShaHash constRes = computePyObjectShaHashConstant(h);

    if (constRes != ShaHash()) {
        return constRes;
    }

    // it's not a simple constant. get the group it's in
    auto groupHeadAndHash = pyObjectGroupHeadAndIndex(h);

    // if we are hashing ourselves within our own group head
    if (groupHead && groupHeadAndHash.first == groupHead) {
        return ShaHash(3, groupHeadAndHash.second);
    }

    auto hashIt = mPythonObjectShaHashes.find(h);
    if (hashIt != mPythonObjectShaHashes.end()) {
        return hashIt->second;
    }

    // if the group head we're asking about is not our group head
    ShaHash res = ShaHash(2) +
        groupHeadAndHash.first->hash() +
        ShaHash(groupHeadAndHash.second)
        ;

    // stash the hash in a lookup
    mPythonObjectShaHashes[incref(h)] = res;
    mHashToObject[res] = incref(h);

    return res;
}

// static
std::pair<MutuallyRecursiveTypeGroup*, int> MutuallyRecursiveTypeGroup::pyObjectGroupHeadAndIndex(PyObject* o) {
    auto it = mPythonObjectTypeGroups.find(o);
    if (it != mPythonObjectTypeGroups.end()) {
        return it->second;
    }

    constructRecursiveTypeGroup(o);

    it = mPythonObjectTypeGroups.find(o);
    if (it != mPythonObjectTypeGroups.end()) {
        return it->second;
    }

    throw std::runtime_error(
        "Somehow, even after computing a recursive group, this pyobj doesn't have a sha hash."
    );
}

ShaHash MutuallyRecursiveTypeGroup::computePyObjectShaHashConstant(PyObject* h) {
    // handle basic constants
    if (h == Py_None) {
        return ShaHash(100, 1);
    }

    if (h == Py_True) {
        return ShaHash(100, 2);
    }

    if (h == Py_False) {
        return ShaHash(100, 3);
    }

    if (PyLong_Check(h)) {
        ShaHash hash(100, 4);

        size_t l = PyLong_AsLong(h);
        if (!(l == -1 && PyErr_Occurred())) {
            return hash + ShaHash(100, l);
        }

        PyErr_Clear();

        // this is an overflow
        PyObjectHolder lng(h);

        while (true) {
            int overflow = 0;

            long l = PyLong_AsLongAndOverflow(lng, &overflow);

            if (l == -1 && PyErr_Occurred()) {
                throw PythonExceptionSet();
            }

            hash += ShaHash(l);
            if (overflow == 0) {
                return hash;
            }

            static PyObject* thirtyTwo = PyLong_FromLong(32);

            lng.set(PyNumber_Rshift(lng, thirtyTwo));
        }

        return hash;
    }

    if (PyBytes_Check(h)) {
        return ShaHash(100, 5) + ShaHash::SHA1(
            PyBytes_AsString(h),
            PyBytes_GET_SIZE(h)
        );
    }

    if (PyUnicode_Check(h)) {
        Py_ssize_t s;
        const char* c = PyUnicode_AsUTF8AndSize(h, &s);

        return ShaHash(100, 6) + ShaHash::SHA1(c, s);
    }

    static PyObject* builtinsModule = PyImport_ImportModule("builtins");
    static PyObject* builtinsModuleDict = PyObject_GetAttrString(builtinsModule, "__dict__");

    if (h == builtinsModule) {
        return ShaHash(100, 8);
    }

    if (h == builtinsModuleDict) {
        return ShaHash(100, 9);
    }

    // 'object'
    if (h == (PyObject*)PyDict_Type.tp_base) {
        return ShaHash(100, 10);
    }
    // 'type'
    if (h == (PyObject*)&PyType_Type) {
        return ShaHash(100, 11);
    }

    if (h == (PyObject*)&PyDict_Type) {
        return ShaHash(100, 12);
    }

    if (h == (PyObject*)&PyList_Type) {
        return ShaHash(100, 13);
    }

    if (h == (PyObject*)&PySet_Type) {
        return ShaHash(100, 14);
    }

    if (h == (PyObject*)&PyLong_Type) {
        return ShaHash(100, 15);
    }

    if (h == (PyObject*)&PyUnicode_Type) {
        return ShaHash(100, 16);
    }

    if (h == (PyObject*)&PyFloat_Type) {
        return ShaHash(100, 17);
    }

    if (h == (PyObject*)&PyBytes_Type) {
        return ShaHash(100, 18);
    }

    if (h == (PyObject*)&PyBool_Type) {
        return ShaHash(100, 19);
    }

    if (h == (PyObject*)Py_None->ob_type) {
        return ShaHash(100, 20);
    }

    if (h == (PyObject*)&PyProperty_Type) {
        return ShaHash(100, 21);
    }

    if (h == (PyObject*)&PyClassMethodDescr_Type) {
        return ShaHash(100, 22);
    }

    if (h == (PyObject*)&PyGetSetDescr_Type) {
        return ShaHash(100, 23);
    }

    if (h == (PyObject*)&PyMemberDescr_Type) {
        return ShaHash(100, 24);
    }

    if (h == (PyObject*)&PyMethodDescr_Type) {
        return ShaHash(100, 25);
    }

    if (h == (PyObject*)&PyWrapperDescr_Type) {
        return ShaHash(100, 26);
    }

    if (h == (PyObject*)&PyDictProxy_Type) {
        return ShaHash(100, 27);
    }

    if (h == (PyObject*)&_PyMethodWrapper_Type) {
        return ShaHash(100, 28);
    }

    if (h == (PyObject*)&PyCFunction_Type) {
        return ShaHash(100, 29);
    }

    if (h == (PyObject*)&PyFunction_Type) {
        return ShaHash(100, 30);
    }

    if (PyFloat_Check(h)) {
        double d = PyFloat_AsDouble(h);

        return ShaHash(100, 31) + ShaHash::SHA1((void*)&d, sizeof(d));
    }

    if (h->ob_type == &PyProperty_Type
        || h->ob_type == &PyProperty_Type
        || h->ob_type == &PyClassMethodDescr_Type
        || h->ob_type == &PyGetSetDescr_Type
        || h->ob_type == &PyMemberDescr_Type
        || h->ob_type == &PyMethodDescr_Type
        || h->ob_type == &PyWrapperDescr_Type
        || h->ob_type == &PyDictProxy_Type
        || h->ob_type == &_PyMethodWrapper_Type
    ) {
        // the compiler doesn't look inside of these, so we don't need to
        // understand what goes on inside of them.
        return ShaHash(100, 32);
    }

    return ShaHash();
}

bool MutuallyRecursiveTypeGroup::isSimpleConstant(PyObject* h) {
    static PyObject* builtinsModule = PyImport_ImportModule("builtins");
    static PyObject* builtinsModuleDict = PyObject_GetAttrString(builtinsModule, "__dict__");

    // handle basic constants
    return (
           h == Py_None
        || h == Py_True
        || h == Py_False
        || PyLong_Check(h)
        || PyBytes_Check(h)
        || PyUnicode_Check(h)
        || h == builtinsModule
        || h == builtinsModuleDict
        || h == (PyObject*)PyDict_Type.tp_base
        || h == (PyObject*)&PyType_Type
        || h == (PyObject*)&PyDict_Type
        || h == (PyObject*)&PyList_Type
        || h == (PyObject*)&PySet_Type
        || h == (PyObject*)&PyLong_Type
        || h == (PyObject*)&PyUnicode_Type
        || h == (PyObject*)&PyFloat_Type
        || h == (PyObject*)&PyBytes_Type
        || h == (PyObject*)&PyBool_Type
        || h == (PyObject*)Py_None->ob_type
        || h == (PyObject*)&PyProperty_Type
        || h == (PyObject*)&PyClassMethodDescr_Type
        || h == (PyObject*)&PyGetSetDescr_Type
        || h == (PyObject*)&PyMemberDescr_Type
        || h == (PyObject*)&PyMethodDescr_Type
        || h == (PyObject*)&PyWrapperDescr_Type
        || h == (PyObject*)&PyDictProxy_Type
        || h == (PyObject*)&_PyMethodWrapper_Type
        || h == (PyObject*)&PyCFunction_Type
        || h == (PyObject*)&PyFunction_Type
        || PyFloat_Check(h)
        || h->ob_type == &PyProperty_Type
        || h->ob_type == &PyProperty_Type
        || h->ob_type == &PyClassMethodDescr_Type
        || h->ob_type == &PyGetSetDescr_Type
        || h->ob_type == &PyMemberDescr_Type
        || h->ob_type == &PyMethodDescr_Type
        || h->ob_type == &PyWrapperDescr_Type
        || h->ob_type == &PyDictProxy_Type
        || h->ob_type == &_PyMethodWrapper_Type
    );
}

// static
ShaHash MutuallyRecursiveTypeGroup::computePyObjectShaHash(PyObject* h, MutuallyRecursiveTypeGroup* groupHead) {
    ShaHash constRes = computePyObjectShaHashConstant(h);

    if (constRes != ShaHash()) {
        return constRes;
    }

    return pyObjectShaHashByVisiting(h, groupHead);
}

// static
ShaHash MutuallyRecursiveTypeGroup::tpInstanceShaHash(Instance h, MutuallyRecursiveTypeGroup* groupHead) {
    if (h.type()->getTypeCategory() == Type::TypeCategory::catNone) {
        return ShaHash(0);
    }

    if (h.type()->getTypeCategory() == Type::TypeCategory::catBool) {
        return ShaHash(h.cast<bool>() ? 1 : 2);
    }

    if (h.type()->getTypeCategory() == Type::TypeCategory::catInt64) {
        return ShaHash(3) + ShaHash(h.cast<int64_t>());
    }

    if (h.type()->getTypeCategory() == Type::TypeCategory::catPythonObjectOfType) {
        return pyObjectShaHash(h.cast<PythonObjectOfType::layout_type*>()->pyObj, groupHead);
    }

    std::cout << "WARNING: tried to hash Instance " << h.repr() << " of type " << h.type()->name() << "\n";

    return ShaHash::poison();
}

void MutuallyRecursiveTypeGroup::installTypeHash(Type* t) {
    // now fill out 'mHashToType' so that the deserializer can look types up by hash.
    // this will call back into the 'hash' function for this group, but we filled it out
    // already, so that should be OK.
    std::lock_guard<std::recursive_mutex> lock(mHashToTypeMutex);

    if (!t->identityHash().isPoison()) {
        auto it = mHashToType.find(t->identityHash());

        // don't replace a type that's already in there, because when we deserialize
        // we may rebuild the same type again, but we want to use the primary version
        if (it == mHashToType.end()) {
            // std::cout << "INSTALL " << t->name() << " = " << t->identityHash().digestAsHexString() << "\n";
            mHashToType[t->identityHash()] = t;
        }
    }
}

//static
PyObject* MutuallyRecursiveTypeGroup::lookupObject(const ShaHash& hash) {
    assertHoldingTheGil();

    auto it = mHashToObject.find(hash);

    if (it != mHashToObject.end()) {
        return it->second;
    }

    return nullptr;
}

//static
Type* MutuallyRecursiveTypeGroup::lookupType(const ShaHash& hash) {
    std::lock_guard<std::recursive_mutex> lock(mHashToTypeMutex);

    auto it = mHashToType.find(hash);

    if (it != mHashToType.end()) {
        return it->second;
    }

    return nullptr;
}

//static
std::unordered_map<PyObject*, ShaHash> MutuallyRecursiveTypeGroup::mPythonObjectShaHashes;

//static
std::unordered_map<PyObject*, std::pair<MutuallyRecursiveTypeGroup*, int> > MutuallyRecursiveTypeGroup::mPythonObjectTypeGroups;

//static
std::recursive_mutex MutuallyRecursiveTypeGroup::mHashToTypeMutex;

//static
std::map<ShaHash, Type*> MutuallyRecursiveTypeGroup::mHashToType;

//static
std::map<ShaHash, PyObject*> MutuallyRecursiveTypeGroup::mHashToObject;
