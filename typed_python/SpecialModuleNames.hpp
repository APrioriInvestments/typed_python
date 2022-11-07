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

inline bool isCanonicalName(std::string name) {
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
inline bool isSpecialIgnorableName(const std::string& name) {
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
