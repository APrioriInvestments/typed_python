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


def return_type(set_of_types):
    """Given a set of types, return a suitable type name covering the possibilities.
    """
    list_of_types = list(set_of_types)
    if len(list_of_types) == 0:
        return 'None'  # shouldn't happen
    if len(list_of_types) == 1:
        return list_of_types[0]
    return 'OneOf<' + ','.join(list_of_types) + '>'


def gen_alternative_type(full_name, d):
    """Generate direct c++ wrapper code for a particular Alternative type.

    Args:
        full_name: fully specified dotted name from codebase, module.class.subclass. ... .typename
        d: dict, where keys are subtypes of this Alternative type,
            and values are corresponding named tuples, represented as a list of (param, type) pairs
    Returns:
        A list of strings, containing c++ code implementing this wrapper.
    """
    name = full_name.rsplit('.', 1)[-1]

    nts = d.keys()
    members = dict()  # set of possible types for each member
    for nt in nts:
        for a, t in d[nt]:
            if a in members:
                members[a].add(t)
            else:
                members[a] = {t}
    ret = list()
    ret.append(f'// Generated Alternative {name}=')
    for nt in nts:
        ret.append('//     {}=({})'.format(nt, ", ".join([f'{a}={t}' for a, t in d[nt]])))
    ret.append('')
    for nt in nts:
        ret.append(f'class {name}_{nt};')
    ret.append('')
    ret.append(f'class {name} {{')
    ret.append('public:')
    ret.append('    enum class kind {{ {} }};'.format(
        ", ".join([f'{nt}={i}' for i, nt in enumerate(nts)])))
    ret.append('')
    for nt in nts:
        ret.append(f'    static NamedTuple* {nt}_Type;')
    ret.append('')
    ret.append('    static Alternative* getType() {')
    ret.append('        PyObject* resolver = getOrSetTypeResolver();')
    ret.append('        if (!resolver)')
    ret.append(f'            throw std::runtime_error("{name}: no resolver");')
    ret.append(f'        PyObject* res = PyObject_CallMethod(resolver, "resolveTypeByName", "s", "{full_name}");')
    ret.append('        if (!res)')
    ret.append(f'            throw std::runtime_error("{full_name}: did not resolve");')
    ret.append('        return (Alternative*)PyInstance::unwrapTypeArgToTypePtr(res);')
    ret.append('    }')

    ret.append(f'    static {name} fromPython(PyObject* p) {{')
    ret.append('        Alternative::layout* l = nullptr;')
    ret.append('        PyInstance::copyConstructFromPythonInstance(getType(), (instance_ptr)&l, p, true);')
    ret.append(f'        return {name}(l);')
    ret.append('    }')
    ret.append('')
    ret.append('    PyObject* toPython() {')
    ret.append('        return PyInstance::extractPythonObject((instance_ptr)&mLayout, getType());')
    ret.append('    }')
    ret.append('')

    ret.append(f'    ~{name}() {{ getType()->destroy((instance_ptr)&mLayout); }}')
    ret.append(f'    {name}():mLayout(0) {{ getType()->constructor((instance_ptr)&mLayout); }}')
    ret.append(f'    {name}(kind k):mLayout(0) {{ '
               'ConcreteAlternative::Make(getType(), (int64_t)k)->constructor((instance_ptr)&mLayout); }')
    ret.append(f'    {name}(const {name}& in) '
               '{ getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&in.mLayout); }')
    ret.append(f'    {name}& operator=(const {name}& other) '
               '{ getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout); return *this; }')
    ret.append('')
    for nt in nts:
        ret.append(f'    static {name} {nt}('
                   + ", ".join([f'const {t}& {a}' for a, t in d[nt]])
                   + ');')
    ret.append('')
    ret.append('    kind which() const { return (kind)mLayout->which; }')
    ret.append('')
    ret.append('    template <class F>')
    ret.append('    auto check(const F& f) {')
    for nt in nts:
        ret.append(f'        if (is{nt}()) {{ return f(*({name}_{nt}*)this); }}')
    ret.append('    }')
    ret.append('')
    for nt in nts:
        ret.append(f'    bool is{nt}() const {{ return which() == kind::{nt}; }}')
    ret.append('')
    ret.append('    // Accessors for members')
    for m in members:
        m_type = return_type(members[m])
        ret.append(f'    {m_type} {m}() const;')
    ret.append('')
    ret.append('    Alternative::layout* getLayout() const { return mLayout; }')
    ret.append('protected:')
    ret.append(f'    explicit {name}(Alternative::layout* l): mLayout(l) {{}}')
    ret.append('    Alternative::layout *mLayout;')
    ret.append('};')
    ret.append('')
    for nt in nts:
        ret.append(f'NamedTuple* {name}::{nt}_Type = NamedTuple::Make(')
        ret.append('    {' + ", ".join([f'TypeDetails<{t}>::getType()' for _, t in d[nt]]) + '},')
        ret.append('    {' + ", ".join([f'"{a}"' for a, _ in d[nt]]) + '}')
        ret.append(');')
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
    ret.append('    static const uint64_t bytecount = sizeof(void*);')
    ret.append('};')
    ret.append('')
    for nt in nts:
        ret.append(f'class {name}_{nt} : public {name} {{')
        ret.append('public:')
        ret.append('    static ConcreteAlternative* getType() {')

        ret.append(f'        static ConcreteAlternative* t = ConcreteAlternative::Make({name}::getType(), static_cast<int>(kind::{nt}));')
        ret.append('        return t;')
        ret.append('    }')
        ret.append(f'    static Alternative* getAlternative() {{ return {name}::getType(); }}')
        # ret.append(f'    static NamedTuple* elementType() {{ return {nt}_Type; }}')
        ret.append('')
        ret.append(f'    {name}_{nt}():{name}(kind::{nt}) {{}}')
        if len(d[nt]) > 0:
            ret.append(f'    {name}_{nt}('
                       + ", ".join([f' const {t}& {a}1' for a, t in d[nt]])
                       + f'):{name}(kind::{nt}) {{')
            for a, _ in d[nt]:
                ret.append(f'        {a}() = {a}1;')
            ret.append('    }')
        ret.append(f'    {name}_{nt}(const {name}_{nt}& other):{name}(kind::{nt}) {{')
        ret.append(f'        getType()->copy_constructor((instance_ptr)&mLayout, '
                   '(instance_ptr)&other.mLayout);')
        ret.append('    }')
        ret.append(f'    {name}_{nt}& operator=(const {name}_{nt}& other) {{')
        ret.append('         getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);')
        ret.append('         return *this;')
        ret.append('    }')
        ret.append(f'    ~{name}_{nt}() {{}}')
        ret.append('')
        for i, (a, t) in enumerate(d[nt]):
            offset = '' if i == 0 else ' + ' + ' + '.join([f'size' + str(j) for j in range(1, i + 1)])
            ret.append(f'    {t}& {a}() const {{ return *({t}*)(mLayout->data{offset}); }}')
        ret.append('private:')
        for i, (_, t) in list(enumerate(d[nt]))[:-1]:
            ret.append(f'    static const int size{i + 1} = sizeof({t});')
        ret.append('};')
        ret.append('')
        ret.append(f'{name} {name}::{nt}('
                   + ", ".join([f'const {t}& {a}' for a, t in d[nt]])
                   + ') {')
        ret.append(f'    return {name}_{nt}('
                   + ', '.join([a for a, _ in d[nt]])
                   + ');')
        ret.append('}')
        ret.append('')
    for m in members:
        m_type = return_type(members[m])
        multiple_types = (len(members[m]) > 1)
        ret.append(f'{m_type} {name}::{m}() const {{')
        for nt in nts:
            if m in [e[0] for e in d[nt]]:
                ret.append(f'    if (is{nt}())')
                if multiple_types:
                    ret.append(f'        return {m_type}((({name}_{nt}*)this)->{m}());')
                else:
                    ret.append(f'        return (({name}_{nt}*)this)->{m}();')
        ret.append(f'    throw std::runtime_error("\\"{name}\\" subtype does not contain \\"{m}\\"");')
        ret.append('}')
        ret.append('')
    ret.append(f'// END Generated Alternative {name}')
    ret.append('')
    return [e + '\n' for e in ret]
