#!/usr/bin/env python3

#   Copyright 2017-2019 Nativepython Authors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


def gen_tuple_type(name, *args):
    """
    Generate direct c++ wrapper code for a particular Tuple type.

    Args:
        name: string name of this Tuple type
        *args: sequence of python Types
    Returns:
        A list of strings, containing c++ code implementing this wrapper.
    """
    keys = ["a" + str(i) for i in range(len(args))]
    items = list(zip(keys, list(args)))

    revkeys = list(keys)[::-1]
    ret = list()
    ret.append(f'// Generated Tuple {name}')
    for key, value in items:
        ret.append(f'//    {key}={value}')

    ret.append(f'class {name} {{')
    ret.append('public:')
    for key, value in items:
        ret.append(f'    typedef {value} {key}_type;')
    for i, (key, value) in enumerate(items):
        offset = '' if i == 0 else ' + ' + ' + '.join([f'size' + str(j) for j in range(1, i + 1)])
        ret.append(f'    {key}_type& {key}() const {{ return *({key}_type*)(data{offset}); }}')
    ret.append('    static Tuple* getType() {')
    ret.append('        static Tuple* t = Tuple::Make({')
    ret.append(',\n'.join(
        [f'                TypeDetails<{name}::{key}_type>::getType()' for key in keys]))
    ret.append('            });')
    ret.append('        return t;')
    ret.append('        }')
    ret.append('')

    ret.append(f'    static {name} fromPython(PyObject* p) {{')
    ret.append(f'        {name} l;')
    ret.append('        PyInstance::copyConstructFromPythonInstance(getType(), (instance_ptr)&l, p, true);')
    ret.append('        return l;')
    ret.append('    }')
    ret.append('')
    ret.append('    PyObject* toPython() {')
    ret.append('        return PyInstance::extractPythonObject((instance_ptr)this, getType());')
    ret.append('    }')
    ret.append('')

    ret.append(f'    {name}& operator = (const {name}& other) {{')
    for key in keys:
        ret.append(f'        {key}() = other.{key}();')
    ret.append('        return *this;')
    ret.append('    }')
    ret.append('')

    ret.append(f'    {name}(const {name}& other) {{')
    for key in keys:
        ret.append(f'        new (&{key}()) {key}_type(other.{key}());')
    ret.append('    }')
    ret.append('')

    ret.append(f'    ~{name}() {{')
    for key in revkeys:
        ret.append(f'        {key}().~{key}_type();')
    ret.append('    }')
    ret.append('')

    ret.append(f'    {name}() {{')
    for key in keys:
        ret.append(f'        bool init{key} = false;')
    ret.append('        try {')
    for key in keys:
        ret.append(f'            new (&{key}()) {key}_type();')
        ret.append(f'            init{key} = true;')
    ret.append('        } catch(...) {')
    ret.append('            try {')
    for key in revkeys:
        ret.append(f'                if (init{key}) {key}().~{key}_type();')
    ret.append('            } catch(...) {')
    ret.append('            }')
    ret.append('            throw;')
    ret.append('        }')
    ret.append('    }')
    ret.append('')

    if len(keys) > 0:
        ret.append(f'    {name}(' + ', '.join([f'const {key}_type& {key}_val' for key in keys]) + ') {')
        for key in keys:
            ret.append(f'        bool init{key} = false;')
        ret.append('        try {')
        for key in keys:
            ret.append(f'            new (&{key}()) {key}_type({key}_val);')
            ret.append(f'            init{key} = true;')
        ret.append('        } catch(...) {')
        ret.append('            try {')
        for key in revkeys:
            ret.append(f'                if (init{key}) {key}().~{key}_type();')
        ret.append('            } catch(...) {')
        ret.append('            }')
        ret.append('            throw;')
        ret.append('        }')
        ret.append('    }')
    ret.append('private:')
    for i, key in enumerate(keys):
        ret.append(f'    static const int size{i + 1} = sizeof({key}_type);')
    ret.append('    uint8_t data[{}];'.format(' + '.join(['size' + str(i) for i in range(1, len(keys) + 1)])))
    ret.append('};')
    ret.append('')

    ret.append('template <>')
    ret.append(f'class TypeDetails<{name}> {{')
    ret.append('public:')
    ret.append('    static Type* getType() {')
    ret.append(f'        static Type* t = {name}::getType();')
    ret.append('        if (t->bytecount() != bytecount) {')
    ret.append(f'            throw std::runtime_error("{name} somehow we have the wrong bytecount!");')
    ret.append('        }')
    ret.append('        return t;')
    ret.append('    }')
    ret.append('    static const uint64_t bytecount = ')
    ret.append(' +\n'.join([f'        sizeof({name}::{key}_type)' for key in keys]) + ';')
    ret.append('};')
    ret.append('')
    ret.append(f'// END Generated Tuple {name}')
    ret.append('')

    return [e + '\n' for e in ret]
