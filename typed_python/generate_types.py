import sys
import argparse
from typed_python._types import NamedTuple
from typed_python._types import ListOf, TupleOf, OneOf, Alternative


def gen_named_tuple_type(name, **kwargs):
    items = kwargs.items()
    keys = kwargs.keys()
    revkeys = list(keys)[::-1]
    ret = []
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
    ret.append('};')
    ret.append('')

    ret.append('template <>')
    ret.append(f'class TypeDetails<{name}> {{')
    ret.append('public:')
    ret.append('    static Type* getType() {')
    ret.append(f'        static Type* t = {name}::getType();')
    ret.append('        if (t->bytecount() != bytecount) {')
    ret.append('            throw std::runtime_error("somehow we have the wrong bytecount!");')
    ret.append('        }')
    ret.append('        return t;')
    ret.append('    }')
    ret.append('    static const uint64_t bytecount = ')
    ret.append(' +\n'. join([f'        sizeof({name}::{key}_type)' for key in keys]) + ';')
    ret.append('};')
    ret.append('')

    return [e + '\n' for e in ret]


# d is dictionary subtype->named_tuple,
# where named_tuple is represented as [(param, type),...]
def gen_alternative_type(name, d):
    nts = d.keys()
    ret = []
    ret.append(f'// Generated Alternative {name}')
    ret.append(f'//    {name}=')
    for nt in nts:
        ret.append('//   {}=({})'.format(nt, ", ".join([f'{a}={t}' for a, t in d[nt]])))
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
        for a, t in d[nt]:
            ret.append(f'    typedef {t} {nt}_{a}_type;')
        for i, (a, _) in enumerate(d[nt]):
            ret.append(f'    static const int {nt}_size{i+1} = sizeof({nt}_{a}_type);')
    ret.append('')
    ret.append('    static Alternative* getType() {')
    ret.append(f'        static Alternative* t = Alternative::Make("{name}", {{')
    ret.append(f',\n'.join([f'            {{"{nt}", {nt}_Type}}' for nt in nts]))
    ret.append('        }, {});')
    ret.append('        return t;')
    ret.append('    }')
    ret.append(f'    ~{name}();')
    ret.append(f'    {name}(); // only if the whole alternative is default initializable')
    ret.append(f'    {name}(const {name}& in);')
    ret.append(f'    {name}& operator=(const {name}& other);')
    ret.append('')
    for nt in nts:
        ret.append(f'    static {name} {nt}('
                   + ", ".join([f'{nt}_{a}_type {a}' for a, _ in d[nt]])
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
    ret.append('    // Accessors for elements of all subtypes here')
    ret.append('    // But account for element name overlap and type differences...')
    for nt in nts:
        for a, _ in d[nt]:
            ret.append(f'    // const {nt}_{a}_type& {a}() const {{}}')
    ret.append('    Alternative::layout* getLayout() const { return mLayout; }')
    ret.append('private:')
    ret.append('    Alternative::layout *mLayout;')
    ret.append('};')
    ret.append('')
    for nt in nts:
        ret.append(f'NamedTuple* {name}::{nt}_Type = NamedTuple::Make(')
        ret.append('    {' + ", ".join([f'TypeDetails<{nt}_{a}_type>::getType()' for a, _ in d[nt]]) + '},')
        ret.append('    {' + ", ".join([f'"{a}"' for a, _ in d[nt]]) + '}')
        ret.append(');')
    ret.append('')
    for nt in nts:
        ret.append(f'class {name}_{nt} : public {name} {{')
        ret.append('public:')
        ret.append('    static ConcreteAlternative* getType() {')
        ret.append(f'        static ConcreteAlternative* t = ConcreteAlternative::Make(A::getType(), e::{nt});')
        ret.append('        return t;')
        ret.append('    }')
        ret.append(f'    static Alternative* getAlternative() {{ return {name}::getType(); }}')
        # ret.append(f'    static NamedTuple* elementType() {{ return {nt}_Type; }}')
        ret.append('')
        ret.append(f'    {name}_{nt}() {{ ')
        ret.append('        getType()->constructor(')
        ret.append('            (instance_ptr)getLayout(),')
        ret.append(f'            [](instance_ptr p) {{{nt}_Type->constructor(p);}});')
        ret.append('    }')
        ret.append(f'    {name}_{nt}('
                   + ", ".join([f'{nt}_{a}_type {a}1' for a, _ in d[nt]])
                   + f') {{')
        ret.append(f'        {name}_{nt}(); ')
        for a, _ in d[nt]:
            ret.append(f'        {a}() = {a}1;')
        ret.append('     }')
        ret.append(f'    {name}_{nt}(const {name}_{nt}& other) {{')
        ret.append(f'        getType()->copy_constructor((instance_ptr)getLayout(), '
                   '(instance_ptr)other.getLayout());')
        ret.append('    }')
        ret.append(f'    {name}_{nt}& operator=(const {name}_{nt}& other) {{')
        ret.append('         getType()->assign((instance_ptr)getLayout(), (instance_ptr)other.getLayout());')
        ret.append('    }')
        ret.append(f'    ~{name}_{nt}() {{')
        ret.append(f'        getType()->destroy((instance_ptr)getLayout());')
        ret.append('    }')
        ret.append('')
        for i, (a, _) in enumerate(d[nt]):
            offset = '' if i == 0 else ' + ' + ' + '.join([f'{nt}_size' + str(j) for j in range(1, i+1)])
            ret.append(f'    {nt}_{a}_type& {a}() const {{ return *({nt}_{a}_type*)(getLayout()->data{offset}); }}')
        ret.append('};')
        ret.append('')
    ret.append('template <>')
    ret.append(f'class TypeDetails<{name}> {{')
    ret.append('public:')
    ret.append('    static Type* getType() {')
    ret.append(f'        static Type* t = {name}::getType();')
    ret.append('        if (t->bytecount() != bytecount) {')
    ret.append('            throw std::runtime_error("somehow we have the wrong bytecount!");')
    ret.append('        }')
    ret.append('        return t;')
    ret.append('    }')
    ret.append('    static const uint64_t bytecount = sizeof(void*);')
    ret.append('};')
    ret.append('')
    return [e + '\n' for e in ret]


fwd_decls = set()


def gen_forward_declarations(element_types):
    ret = []
    for t in element_types:
        if t.__typed_python_category__ == 'Forward':
            forward_type_name = str(t)[8:-2]    # just for now!
            if forward_type_name not in fwd_decls:
                fwd_decls.add(forward_type_name)
                ret.append(f"class {forward_type_name};")
                ret.append("")
                ret.append('template <>')
                ret.append(f'class TypeDetails<{forward_type_name}*> {{')
                ret.append('public:')
                ret.append('    static Type* getType() {')
                ret.append('        static Type* t = PointerTo::Make(Int64::Make());  '
                           '// forward types are not actually constructed')
                ret.append('        return t;')
                ret.append('    }')
                ret.append('    static const uint64_t bytecount = sizeof(void*);')
                ret.append('};')
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
        return str(py_type)[8:-2] + '*'  # just for now! no forward resolution
    if cat == 'NamedTuple' or cat == 'Alternative':
        return cpp_type_mapping[py_type]
    return 'undefined_type'


def typed_python_codegen(**kwargs):
    ret = []
    for k, v in kwargs.items():
        if v.__typed_python_category__ == 'NamedTuple':
            ret += gen_named_tuple_type(k, **{n: cpp_type(t) for n, t in zip(v.ElementNames, v.ElementTypes)})
            cpp_type_mapping[v] = k
        elif v.__typed_python_category__ == 'Alternative':
            for e in v.__typed_python_alternatives__:
                ret += gen_forward_declarations(e.ElementType.ElementTypes)
            d = {nt.Name:
                 [(a, cpp_type(t)) for a, t in zip(nt.ElementType.ElementNames, nt.ElementType.ElementTypes)]
                 for nt in v.__typed_python_alternatives__ }
            ret += gen_alternative_type(k, d)
            cpp_type_mapping[v] = k
    return ret


def GenerateTestTypes(dest):
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
            A=Alternative( 'A', Sub1={'b': int, 'c': int}, Sub2={'d': str, 'e': str} ),
            Bexpress=Bexpress,
            NamedTupleTwoStrings=NamedTuple(X=str, Y=str),
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
            GenerateTestTypes(args.dest)
        except Exception:
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
