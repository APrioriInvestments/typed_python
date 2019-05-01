import sys
import argparse
from typed_python._types import NamedTuple
from typed_python._types import ListOf, TupleOf, OneOf, Alternative


def gen_named_tuple_type(name, **kwargs):
    items = kwargs.items()
    keys = kwargs.keys()
    revkeys = list(keys)[::-1]
    ret = list()
    ret.append(f'// Generated NamedTuple {name}')
    for key, value in items:
        ret.append(f'//    {key}={value}')

    ret.append(f'class {name} {{')
    ret.append('public:')
    for key, value in items:
        ret.append(f'    typedef {value} {key}_type;')
    for i, (key, value) in enumerate(items):
        offset = '' if i == 0 else ' + ' + ' + '.join([f'size' + str(j) for j in range(1, i + 1)])
        ret.append(f'    {key}_type& {key}() const {{ return *({key}_type*)(data{offset}); }}')
    ret.append('private:')
    for i, key in enumerate(keys):
        ret.append(f'    static const int size{i + 1} = sizeof({key}_type);')
    ret.append('    uint8_t data[{}];'.format(' + '.join(['size' + str(i) for i in range(1, len(keys) + 1)])))
    ret.append('public:')
    ret.append('    static NamedTuple* getType() {')
    ret.append('        static NamedTuple* t = NamedTuple::Make({')
    ret.append(',\n'.join(
        [f'                TypeDetails<{name}::{key}_type>::getType()' for key in keys]))
    ret.append('            },{')
    ret.append(',\n'.join(
        [f'                "{key}"' for key in keys]))
    ret.append('            });')
    ret.append('        return t;')
    ret.append('        }')

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
    ret.append(f'// END Generated NamedTuple {name}')
    ret.append('')

    return [e + '\n' for e in ret]

def return_type(set_of_types):
    list_of_types = list(set_of_types)
    if len(list_of_types) == 0:
        return 'None'  # shouldn't happen
    if len(list_of_types) == 1:
        return list_of_types[0]
    return 'OneOf<' + ','.join(list_of_types) + '>'


# d is dictionary subtype->named_tuple,
# where named_tuple is represented as [(param, type),...]
def gen_alternative_type(name, d):
    nts = d.keys()
    members = dict()  # set of possible types for each member
    for nt in nts:
        for a, t in d[nt]:
            rt = resolved(t)
            if a in members:
                members[a].add(rt)
            else:
                members[a] = {rt}
    ret = list()
    ret.append(f'// Generated Alternative {name}=')
    for nt in nts:
        ret.append('//     {}=({})'.format(nt, ", ".join([f'{a}={resolved(t)}' for a, t in d[nt]])))
    ret.append('')
    for nt in nts:
        ret.append(f'class {name}_{nt};')
    ret.append('')
    ret.append(f'class {name} {{')
    ret.append('public:')
    ret.append('    struct e {')
    ret.append('        enum kind {{ {} }};'.format(
        ", ".join([f'{nt}={i}' for i, nt in enumerate(nts)])))
    ret.append('    };')
    ret.append('')
    for nt in nts:
        ret.append(f'    static NamedTuple* {nt}_Type;')
    ret.append('')
    ret.append('    static Alternative* getType();')
    ret.append(f'    ~{name}() {{ getType()->destroy((instance_ptr)&mLayout); }}')
    ret.append(f'    {name}():mLayout(0) {{ getType()->constructor((instance_ptr)&mLayout); }}')
    ret.append(f'    {name}(e::kind k):mLayout(0) {{ '
               'ConcreteAlternative::Make(getType(), (int64_t)k)->constructor((instance_ptr)&mLayout); }')
    ret.append(f'    {name}(const {name}& in) '
               '{ getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&in.mLayout); }')
    ret.append(f'    {name}& operator=(const {name}& other) '
               '{ getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout); return *this; }')
    ret.append('')
    for nt in nts:
        ret.append(f'    static {name} {nt}('
                   + ", ".join([f'const {resolved(t)}& {a}' for a, t in d[nt]])
                   + ');')
    ret.append('')
    ret.append('    e::kind which() const { return (e::kind)mLayout->which; }')
    ret.append('')
    ret.append('    template <class F>')
    ret.append('    auto check(const F& f) {')
    for nt in nts:
        ret.append(f'        if (is{nt}()) {{ return f(*({name}_{nt}*)this); }}')
    ret.append('    }')
    ret.append('')
    for nt in nts:
        ret.append(f'    bool is{nt}() const {{ return which() == e::{nt}; }}')
    ret.append('')
    ret.append('    // Accessors for members')
    for m in members:
        m_type = return_type(members[m])
        ret.append(f'    {m_type} {m}() const;')
    ret.append('')
    ret.append('    Alternative::layout* getLayout() const { return mLayout; }')
    ret.append('protected:')
    ret.append('    Alternative::layout *mLayout;')
    ret.append('};')
    ret.append('')
    ret.append('template <>')
    ret.append(f'class TypeDetails<{name}*> {{')
    ret.append('public:')
    ret.append('    static Type* getType() {')
    ret.append(f'        static Type* t = new Forward(0, "{name}");')
    ret.append('        return t;')
    ret.append('    }')
    ret.append('    static const uint64_t bytecount = sizeof(void*);')
    ret.append('};')
    ret.append('')
    for nt in nts:
        ret.append(f'NamedTuple* {name}::{nt}_Type = NamedTuple::Make(')
        ret.append('    {' + ", ".join([f'TypeDetails<{t}>::getType()' for _, t in d[nt]]) + '},')
        ret.append('    {' + ", ".join([f'"{a}"' for a, _ in d[nt]]) + '}')
        ret.append(');')
        ret.append('')
    ret.append('// static')
    ret.append(f'Alternative* {name}::getType() {{')
    ret.append(f'    static Alternative* t = Alternative::Make("{name}", {{')
    ret.append(f',\n'.join([f'        {{"{nt}", {nt}_Type}}' for nt in nts]))
    ret.append('    }, {});')
    for nt in nts:
        for _, t in d[nt]:
            if t.endswith('*'):
                ret.append(f'    {nt}_Type->directResolveForward(TypeDetails<{name}*>::getType(), t);')
                break
    ret.append('    return t;')
    ret.append('}')
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

        ret.append(f'        static ConcreteAlternative* t = ConcreteAlternative::Make({name}::getType(), e::{nt});')
        ret.append('        return t;')
        ret.append('    }')
        ret.append(f'    static Alternative* getAlternative() {{ return {name}::getType(); }}')
        # ret.append(f'    static NamedTuple* elementType() {{ return {nt}_Type; }}')
        ret.append('')
        ret.append(f'    {name}_{nt}():{name}(e::{nt}) {{}}')
        ret.append(f'    {name}_{nt}('
                   + ", ".join([f' const {resolved(t)}& {a}1' for a, t in d[nt]])
                   + f'):{name}(e::{nt}) {{')
        for a, _ in d[nt]:
            ret.append(f'        {a}() = {a}1;')
        ret.append('    }')
        ret.append(f'    {name}_{nt}(const {name}_{nt}& other):{name}(e::{nt}) {{')
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
            ret.append(f'    {resolved(t)}& {a}() const {{ return *({resolved(t)}*)(mLayout->data{offset}); }}')
        ret.append('private:')
        for i, (_, t) in list(enumerate(d[nt]))[:-1]:
            ret.append(f'    static const int size{i + 1} = sizeof({resolved(t)});')
        ret.append('};')
        ret.append('')
        ret.append(f'{name} {name}::{nt}('
                   + ", ".join([f'const {resolved(t)}& {a}' for a, t in d[nt]])
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


cpp_type_mapping = dict()


# py type -> c++ direct type
# Int64 -> int64_t
# Bool -> bool
# ListOf(Int64) -> ListOf<int64_t>
# TupleOf(Bool) -> TupleOf<bool>
# for generated types with arbitrary names, just use the name
# Arb=NamedTuple(X=Int64,Y=Bool) -> Arb
# Either assume Arb is defined in a previous stage, or keep track of it
def cpp_type(py_type):
    simple_cats = {
        'Int64': 'int64_t',
        'UInt64': 'uint64_t',
        'Int32': 'uint32_t',
        'UInt32': 'uint32_t',
        'Int16': 'uint16_t',
        'UInt16': 'uint16_t',
        'Int8': 'uint8_t',
        'UInt8': 'uint8_t',
        'Bool': 'bool',
        'Float64': 'double',
        'Float32': 'float',
        'String': 'String'
    }
    cat = py_type.__typed_python_category__
    if cat in simple_cats.keys():
        return simple_cats[cat]
    # if cat == 'NamedTuple': # not supported
    #     return 'NamedTuple({})'.format(
    #         ', '.join(['{}={}'.format(n,cpp_type(t))
    #             for n, t in zip(py_type.ElementNames, py_type.ElementTypes)])
    #         )
    if cat == 'NoneType':
        return 'None'
    if cat == 'ListOf' or cat == 'TupleOf':
        return '{}<{}>'.format(cat, cpp_type(py_type.ElementType))
    if cat == 'OneOf':
        return 'OneOf<{}>'.format(
            ', '.join([cpp_type(t) for t in py_type.Types])
        )
    if cat == 'Forward':
        return str(py_type)[8:-2] + '*'  # just for now!
    if cat == 'NamedTuple' or cat == 'Alternative':
        return cpp_type_mapping[py_type]
    return 'undefined_type'


def resolved(t):
    return t[:-1] if t.endswith('*') else t


def typed_python_codegen(**kwargs):
    ret = []
    for k, v in kwargs.items():
        if v.__typed_python_category__ == 'NamedTuple':
            ret += gen_named_tuple_type(k, **{n: cpp_type(t) for n, t in zip(v.ElementNames, v.ElementTypes)})
            cpp_type_mapping[v] = k
        elif v.__typed_python_category__ == 'Alternative':
            d = {nt.Name:
                 [(a, cpp_type(t)) for a, t in zip(nt.ElementType.ElementNames, nt.ElementType.ElementTypes)]
                 for nt in v.__typed_python_alternatives__}
            ret += gen_alternative_type(k, d)
            cpp_type_mapping[v] = k
    return ret


def generate_some_types(dest):
    Bexpress = lambda: Bexpress
    Bexpress = Alternative(
        "BooleanExpr",
        BinOp={
            "left": Bexpress,
            "op": str,
            "right": Bexpress,
        },
        UnaryOp={
            "op": str,
            "right": Bexpress
        },
        Leaf={
            "value": bool
        }
    )
    with open(dest, 'w') as f:
        f.writelines(typed_python_codegen(
            Overlap=Alternative('Overlap', Sub1={'b': bool, 'c': int}, Sub2={'b': str, 'c': TupleOf(str)},
                                Sub3={'b': int}),
            A=Alternative('A', Sub1={'b': int, 'c': int}, Sub2={'d': str, 'e': str}),
            Bexpress=Bexpress,
            NamedTupleTwoStrings=NamedTuple(X=str, Y=str),
            NamedTupleBoolIntStr=NamedTuple(b=bool, i=int, s=str),
            Choice=NamedTuple(A=NamedTuple(X=str, Y=str), B=Bexpress),
            NamedTupleIntFloatDesc=NamedTuple(a=OneOf(int, float, bool), b=float, desc=str),
            NamedTupleBoolListOfInt=NamedTuple(X=bool, Y=ListOf(int)),
            NamedTupleAttrAndValues=NamedTuple(attributes=TupleOf(str), values=TupleOf(int))
        ))


def main(argv):
    parser = argparse.ArgumentParser(description='Generate types')
    parser.add_argument('dest', nargs='?', default='DefaultGeneratedTestTypes.hpp')
    parser.add_argument('-t', '--testTypes', action='store_true')
    args = parser.parse_args()

    if args.testTypes:
        try:
            generate_some_types(args.dest)
        except Exception:
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
