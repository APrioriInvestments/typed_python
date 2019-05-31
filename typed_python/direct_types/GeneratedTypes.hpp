#pragma once
// Generated Alternative A=
//     Sub1=(b=int64_t, c=int64_t)
//     Sub2=(d=String, e=String)

class A_Sub1;
class A_Sub2;

class A {
public:
    enum class kind { Sub1=0, Sub2=1 };

    static NamedTuple* Sub1_Type;
    static NamedTuple* Sub2_Type;

    static Alternative* getType() {
        PyObject* resolver = getOrSetTypeResolver();
        if (!resolver)
            throw std::runtime_error("A: no resolver");
        //std::cerr<<" " << Py_TYPE(resolver)->tp_name <<std::endl;
        PyObject* res = PyObject_CallMethod(resolver, "resolveTypeByName", "s", "typed_python.direct_types.generate_types.A");
        if (!res)
            throw std::runtime_error("typed_python.direct_types.generate_types.A: did not resolve");
        return (Alternative*)PyInstance::unwrapTypeArgToTypePtr(res);
    }
    static A fromPython(PyObject* p) {
        Alternative::layout* l = nullptr;
        PyInstance::copyConstructFromPythonInstance(getType(), (instance_ptr)&l, p, true);
        return A(l);
    }

    PyObject* toPython() {
        return PyInstance::extractPythonObject((instance_ptr)&mLayout, getType());
    }

    ~A() { getType()->destroy((instance_ptr)&mLayout); }
    A():mLayout(0) { getType()->constructor((instance_ptr)&mLayout); }
    A(kind k):mLayout(0) { ConcreteAlternative::Make(getType(), (int64_t)k)->constructor((instance_ptr)&mLayout); }
    A(const A& in) { getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&in.mLayout); }
    A& operator=(const A& other) { getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout); return *this; }

    static A Sub1(const int64_t& b, const int64_t& c);
    static A Sub2(const String& d, const String& e);

    kind which() const { return (kind)mLayout->which; }

    template <class F>
    auto check(const F& f) {
        if (isSub1()) { return f(*(A_Sub1*)this); }
        if (isSub2()) { return f(*(A_Sub2*)this); }
    }

    bool isSub1() const { return which() == kind::Sub1; }
    bool isSub2() const { return which() == kind::Sub2; }

    // Accessors for members
    int64_t b() const;
    int64_t c() const;
    String d() const;
    String e() const;

    Alternative::layout* getLayout() const { return mLayout; }
protected:
    explicit A(Alternative::layout* l): mLayout(l) {}
    Alternative::layout *mLayout;
};

NamedTuple* A::Sub1_Type = NamedTuple::Make(
    {TypeDetails<int64_t>::getType(), TypeDetails<int64_t>::getType()},
    {"b", "c"}
);

NamedTuple* A::Sub2_Type = NamedTuple::Make(
    {TypeDetails<String>::getType(), TypeDetails<String>::getType()},
    {"d", "e"}
);

template <>
class TypeDetails<A> {
public:
    static Type* getType() {
        static Type* t = A::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("A somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = sizeof(void*);
};

class A_Sub1 : public A {
public:
    static ConcreteAlternative* getType() {
        static ConcreteAlternative* t = ConcreteAlternative::Make(A::getType(), static_cast<int>(kind::Sub1));
        return t;
    }
    static Alternative* getAlternative() { return A::getType(); }

    A_Sub1():A(kind::Sub1) {}
    A_Sub1( const int64_t& b1,  const int64_t& c1):A(kind::Sub1) {
        b() = b1;
        c() = c1;
    }
    A_Sub1(const A_Sub1& other):A(kind::Sub1) {
        getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
    }
    A_Sub1& operator=(const A_Sub1& other) {
         getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
         return *this;
    }
    ~A_Sub1() {}

    int64_t& b() const { return *(int64_t*)(mLayout->data); }
    int64_t& c() const { return *(int64_t*)(mLayout->data + size1); }
private:
    static const int size1 = sizeof(int64_t);
};

A A::Sub1(const int64_t& b, const int64_t& c) {
    return A_Sub1(b, c);
}

class A_Sub2 : public A {
public:
    static ConcreteAlternative* getType() {
        static ConcreteAlternative* t = ConcreteAlternative::Make(A::getType(), static_cast<int>(kind::Sub2));
        return t;
    }
    static Alternative* getAlternative() { return A::getType(); }

    A_Sub2():A(kind::Sub2) {}
    A_Sub2( const String& d1,  const String& e1):A(kind::Sub2) {
        d() = d1;
        e() = e1;
    }
    A_Sub2(const A_Sub2& other):A(kind::Sub2) {
        getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
    }
    A_Sub2& operator=(const A_Sub2& other) {
         getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
         return *this;
    }
    ~A_Sub2() {}

    String& d() const { return *(String*)(mLayout->data); }
    String& e() const { return *(String*)(mLayout->data + size1); }
private:
    static const int size1 = sizeof(String);
};

A A::Sub2(const String& d, const String& e) {
    return A_Sub2(d, e);
}

int64_t A::b() const {
    if (isSub1())
        return ((A_Sub1*)this)->b();
    throw std::runtime_error("\"A\" subtype does not contain \"b\"");
}

int64_t A::c() const {
    if (isSub1())
        return ((A_Sub1*)this)->c();
    throw std::runtime_error("\"A\" subtype does not contain \"c\"");
}

String A::d() const {
    if (isSub2())
        return ((A_Sub2*)this)->d();
    throw std::runtime_error("\"A\" subtype does not contain \"d\"");
}

String A::e() const {
    if (isSub2())
        return ((A_Sub2*)this)->e();
    throw std::runtime_error("\"A\" subtype does not contain \"e\"");
}

// END Generated Alternative A

// Generated Alternative Overlap=
//     Sub1=(b=bool, c=int64_t)
//     Sub2=(b=String, c=TupleOf<String>)
//     Sub3=(b=int64_t)

class Overlap_Sub1;
class Overlap_Sub2;
class Overlap_Sub3;

class Overlap {
public:
    enum class kind { Sub1=0, Sub2=1, Sub3=2 };

    static NamedTuple* Sub1_Type;
    static NamedTuple* Sub2_Type;
    static NamedTuple* Sub3_Type;

    static Alternative* getType() {
        PyObject* resolver = getOrSetTypeResolver();
        if (!resolver)
            throw std::runtime_error("Overlap: no resolver");
        PyObject* res = PyObject_CallMethod(resolver, "resolveTypeByName", "s", "typed_python.direct_types.generate_types.Overlap");
        if (!res)
            throw std::runtime_error("typed_python.direct_types.generate_types.Overlap: did not resolve");
        return (Alternative*)PyInstance::unwrapTypeArgToTypePtr(res);
    }
    static Overlap fromPython(PyObject* p) {
        Alternative::layout* l = nullptr;
        PyInstance::copyConstructFromPythonInstance(getType(), (instance_ptr)&l, p, true);
        return Overlap(l);
    }

    PyObject* toPython() {
        return PyInstance::extractPythonObject((instance_ptr)&mLayout, getType());
    }

    ~Overlap() { getType()->destroy((instance_ptr)&mLayout); }
    Overlap():mLayout(0) { getType()->constructor((instance_ptr)&mLayout); }
    Overlap(kind k):mLayout(0) { ConcreteAlternative::Make(getType(), (int64_t)k)->constructor((instance_ptr)&mLayout); }
    Overlap(const Overlap& in) { getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&in.mLayout); }
    Overlap& operator=(const Overlap& other) { getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout); return *this; }

    static Overlap Sub1(const bool& b, const int64_t& c);
    static Overlap Sub2(const String& b, const TupleOf<String>& c);
    static Overlap Sub3(const int64_t& b);

    kind which() const { return (kind)mLayout->which; }

    template <class F>
    auto check(const F& f) {
        if (isSub1()) { return f(*(Overlap_Sub1*)this); }
        if (isSub2()) { return f(*(Overlap_Sub2*)this); }
        if (isSub3()) { return f(*(Overlap_Sub3*)this); }
    }

    bool isSub1() const { return which() == kind::Sub1; }
    bool isSub2() const { return which() == kind::Sub2; }
    bool isSub3() const { return which() == kind::Sub3; }

    // Accessors for members
    OneOf<bool,String,int64_t> b() const;
    OneOf<int64_t,TupleOf<String>> c() const;

    Alternative::layout* getLayout() const { return mLayout; }
protected:
    explicit Overlap(Alternative::layout* l): mLayout(l) {}
    Alternative::layout *mLayout;
};

NamedTuple* Overlap::Sub1_Type = NamedTuple::Make(
    {TypeDetails<bool>::getType(), TypeDetails<int64_t>::getType()},
    {"b", "c"}
);

NamedTuple* Overlap::Sub2_Type = NamedTuple::Make(
    {TypeDetails<String>::getType(), TypeDetails<TupleOf<String>>::getType()},
    {"b", "c"}
);

NamedTuple* Overlap::Sub3_Type = NamedTuple::Make(
    {TypeDetails<int64_t>::getType()},
    {"b"}
);

template <>
class TypeDetails<Overlap> {
public:
    static Type* getType() {
        static Type* t = Overlap::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("Overlap somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = sizeof(void*);
};

class Overlap_Sub1 : public Overlap {
public:
    static ConcreteAlternative* getType() {
        static ConcreteAlternative* t = ConcreteAlternative::Make(Overlap::getType(), static_cast<int>(kind::Sub1));
        return t;
    }
    static Alternative* getAlternative() { return Overlap::getType(); }

    Overlap_Sub1():Overlap(kind::Sub1) {}
    Overlap_Sub1( const bool& b1,  const int64_t& c1):Overlap(kind::Sub1) {
        b() = b1;
        c() = c1;
    }
    Overlap_Sub1(const Overlap_Sub1& other):Overlap(kind::Sub1) {
        getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
    }
    Overlap_Sub1& operator=(const Overlap_Sub1& other) {
         getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
         return *this;
    }
    ~Overlap_Sub1() {}

    bool& b() const { return *(bool*)(mLayout->data); }
    int64_t& c() const { return *(int64_t*)(mLayout->data + size1); }
private:
    static const int size1 = sizeof(bool);
};

Overlap Overlap::Sub1(const bool& b, const int64_t& c) {
    return Overlap_Sub1(b, c);
}

class Overlap_Sub2 : public Overlap {
public:
    static ConcreteAlternative* getType() {
        static ConcreteAlternative* t = ConcreteAlternative::Make(Overlap::getType(), static_cast<int>(kind::Sub2));
        return t;
    }
    static Alternative* getAlternative() { return Overlap::getType(); }

    Overlap_Sub2():Overlap(kind::Sub2) {}
    Overlap_Sub2( const String& b1,  const TupleOf<String>& c1):Overlap(kind::Sub2) {
        b() = b1;
        c() = c1;
    }
    Overlap_Sub2(const Overlap_Sub2& other):Overlap(kind::Sub2) {
        getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
    }
    Overlap_Sub2& operator=(const Overlap_Sub2& other) {
         getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
         return *this;
    }
    ~Overlap_Sub2() {}

    String& b() const { return *(String*)(mLayout->data); }
    TupleOf<String>& c() const { return *(TupleOf<String>*)(mLayout->data + size1); }
private:
    static const int size1 = sizeof(String);
};

Overlap Overlap::Sub2(const String& b, const TupleOf<String>& c) {
    return Overlap_Sub2(b, c);
}

class Overlap_Sub3 : public Overlap {
public:
    static ConcreteAlternative* getType() {
        static ConcreteAlternative* t = ConcreteAlternative::Make(Overlap::getType(), static_cast<int>(kind::Sub3));
        return t;
    }
    static Alternative* getAlternative() { return Overlap::getType(); }

    Overlap_Sub3():Overlap(kind::Sub3) {}
    Overlap_Sub3( const int64_t& b1):Overlap(kind::Sub3) {
        b() = b1;
    }
    Overlap_Sub3(const Overlap_Sub3& other):Overlap(kind::Sub3) {
        getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
    }
    Overlap_Sub3& operator=(const Overlap_Sub3& other) {
         getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
         return *this;
    }
    ~Overlap_Sub3() {}

    int64_t& b() const { return *(int64_t*)(mLayout->data); }
private:
};

Overlap Overlap::Sub3(const int64_t& b) {
    return Overlap_Sub3(b);
}

OneOf<bool,String,int64_t> Overlap::b() const {
    if (isSub1())
        return OneOf<bool,String,int64_t>(((Overlap_Sub1*)this)->b());
    if (isSub2())
        return OneOf<bool,String,int64_t>(((Overlap_Sub2*)this)->b());
    if (isSub3())
        return OneOf<bool,String,int64_t>(((Overlap_Sub3*)this)->b());
    throw std::runtime_error("\"Overlap\" subtype does not contain \"b\"");
}

OneOf<int64_t,TupleOf<String>> Overlap::c() const {
    if (isSub1())
        return OneOf<int64_t,TupleOf<String>>(((Overlap_Sub1*)this)->c());
    if (isSub2())
        return OneOf<int64_t,TupleOf<String>>(((Overlap_Sub2*)this)->c());
    throw std::runtime_error("\"Overlap\" subtype does not contain \"c\"");
}

// END Generated Alternative Overlap

// Generated NamedTuple NamedTupleTwoStrings
//    X=String
//    Y=String
class NamedTupleTwoStrings {
public:
    typedef String X_type;
    typedef String Y_type;
    X_type& X() const { return *(X_type*)(data); }
    Y_type& Y() const { return *(Y_type*)(data + size1); }
    static NamedTuple* getType() {
        static NamedTuple* t = NamedTuple::Make({
                TypeDetails<NamedTupleTwoStrings::X_type>::getType(),
                TypeDetails<NamedTupleTwoStrings::Y_type>::getType()
            },{
                "X",
                "Y"
            });
        return t;
        }

    static NamedTupleTwoStrings fromPython(PyObject* p) {
        NamedTupleTwoStrings l;
        PyInstance::copyConstructFromPythonInstance(getType(), (instance_ptr)&l, p, true);
        return l;
    }

    PyObject* toPython() {
        return PyInstance::extractPythonObject((instance_ptr)this, getType());
    }

    NamedTupleTwoStrings& operator = (const NamedTupleTwoStrings& other) {
        X() = other.X();
        Y() = other.Y();
        return *this;
    }

    NamedTupleTwoStrings(const NamedTupleTwoStrings& other) {
        new (&X()) X_type(other.X());
        new (&Y()) Y_type(other.Y());
    }

    ~NamedTupleTwoStrings() {
        Y().~Y_type();
        X().~X_type();
    }

    NamedTupleTwoStrings() {
        bool initX = false;
        bool initY = false;
        try {
            new (&X()) X_type();
            initX = true;
            new (&Y()) Y_type();
            initY = true;
        } catch(...) {
            try {
                if (initY) Y().~Y_type();
                if (initX) X().~X_type();
            } catch(...) {
            }
            throw;
        }
    }

    NamedTupleTwoStrings(const X_type& X_val, const Y_type& Y_val) {
        bool initX = false;
        bool initY = false;
        try {
            new (&X()) X_type(X_val);
            initX = true;
            new (&Y()) Y_type(Y_val);
            initY = true;
        } catch(...) {
            try {
                if (initY) Y().~Y_type();
                if (initX) X().~X_type();
            } catch(...) {
            }
            throw;
        }
    }
private:
    static const int size1 = sizeof(X_type);
    static const int size2 = sizeof(Y_type);
    uint8_t data[size1 + size2];
};

template <>
class TypeDetails<NamedTupleTwoStrings> {
public:
    static Type* getType() {
        static Type* t = NamedTupleTwoStrings::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("NamedTupleTwoStrings somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = 
        sizeof(NamedTupleTwoStrings::X_type) +
        sizeof(NamedTupleTwoStrings::Y_type);
};

// END Generated NamedTuple NamedTupleTwoStrings

// Generated NamedTuple NamedTupleBoolIntStr
//    b=bool
//    i=int64_t
//    s=String
class NamedTupleBoolIntStr {
public:
    typedef bool b_type;
    typedef int64_t i_type;
    typedef String s_type;
    b_type& b() const { return *(b_type*)(data); }
    i_type& i() const { return *(i_type*)(data + size1); }
    s_type& s() const { return *(s_type*)(data + size1 + size2); }
    static NamedTuple* getType() {
        static NamedTuple* t = NamedTuple::Make({
                TypeDetails<NamedTupleBoolIntStr::b_type>::getType(),
                TypeDetails<NamedTupleBoolIntStr::i_type>::getType(),
                TypeDetails<NamedTupleBoolIntStr::s_type>::getType()
            },{
                "b",
                "i",
                "s"
            });
        return t;
        }

    static NamedTupleBoolIntStr fromPython(PyObject* p) {
        NamedTupleBoolIntStr l;
        PyInstance::copyConstructFromPythonInstance(getType(), (instance_ptr)&l, p, true);
        return l;
    }

    PyObject* toPython() {
        return PyInstance::extractPythonObject((instance_ptr)this, getType());
    }

    NamedTupleBoolIntStr& operator = (const NamedTupleBoolIntStr& other) {
        b() = other.b();
        i() = other.i();
        s() = other.s();
        return *this;
    }

    NamedTupleBoolIntStr(const NamedTupleBoolIntStr& other) {
        new (&b()) b_type(other.b());
        new (&i()) i_type(other.i());
        new (&s()) s_type(other.s());
    }

    ~NamedTupleBoolIntStr() {
        s().~s_type();
        i().~i_type();
        b().~b_type();
    }

    NamedTupleBoolIntStr() {
        bool initb = false;
        bool initi = false;
        bool inits = false;
        try {
            new (&b()) b_type();
            initb = true;
            new (&i()) i_type();
            initi = true;
            new (&s()) s_type();
            inits = true;
        } catch(...) {
            try {
                if (inits) s().~s_type();
                if (initi) i().~i_type();
                if (initb) b().~b_type();
            } catch(...) {
            }
            throw;
        }
    }

    NamedTupleBoolIntStr(const b_type& b_val, const i_type& i_val, const s_type& s_val) {
        bool initb = false;
        bool initi = false;
        bool inits = false;
        try {
            new (&b()) b_type(b_val);
            initb = true;
            new (&i()) i_type(i_val);
            initi = true;
            new (&s()) s_type(s_val);
            inits = true;
        } catch(...) {
            try {
                if (inits) s().~s_type();
                if (initi) i().~i_type();
                if (initb) b().~b_type();
            } catch(...) {
            }
            throw;
        }
    }
private:
    static const int size1 = sizeof(b_type);
    static const int size2 = sizeof(i_type);
    static const int size3 = sizeof(s_type);
    uint8_t data[size1 + size2 + size3];
};

template <>
class TypeDetails<NamedTupleBoolIntStr> {
public:
    static Type* getType() {
        static Type* t = NamedTupleBoolIntStr::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("NamedTupleBoolIntStr somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = 
        sizeof(NamedTupleBoolIntStr::b_type) +
        sizeof(NamedTupleBoolIntStr::i_type) +
        sizeof(NamedTupleBoolIntStr::s_type);
};

// END Generated NamedTuple NamedTupleBoolIntStr

// Generated NamedTuple NamedTupleIntFloatDesc
//    a=OneOf<int64_t, double, bool>
//    b=double
//    desc=String
class NamedTupleIntFloatDesc {
public:
    typedef OneOf<int64_t, double, bool> a_type;
    typedef double b_type;
    typedef String desc_type;
    a_type& a() const { return *(a_type*)(data); }
    b_type& b() const { return *(b_type*)(data + size1); }
    desc_type& desc() const { return *(desc_type*)(data + size1 + size2); }
    static NamedTuple* getType() {
        static NamedTuple* t = NamedTuple::Make({
                TypeDetails<NamedTupleIntFloatDesc::a_type>::getType(),
                TypeDetails<NamedTupleIntFloatDesc::b_type>::getType(),
                TypeDetails<NamedTupleIntFloatDesc::desc_type>::getType()
            },{
                "a",
                "b",
                "desc"
            });
        return t;
        }

    static NamedTupleIntFloatDesc fromPython(PyObject* p) {
        NamedTupleIntFloatDesc l;
        PyInstance::copyConstructFromPythonInstance(getType(), (instance_ptr)&l, p, true);
        return l;
    }

    PyObject* toPython() {
        return PyInstance::extractPythonObject((instance_ptr)this, getType());
    }

    NamedTupleIntFloatDesc& operator = (const NamedTupleIntFloatDesc& other) {
        a() = other.a();
        b() = other.b();
        desc() = other.desc();
        return *this;
    }

    NamedTupleIntFloatDesc(const NamedTupleIntFloatDesc& other) {
        new (&a()) a_type(other.a());
        new (&b()) b_type(other.b());
        new (&desc()) desc_type(other.desc());
    }

    ~NamedTupleIntFloatDesc() {
        desc().~desc_type();
        b().~b_type();
        a().~a_type();
    }

    NamedTupleIntFloatDesc() {
        bool inita = false;
        bool initb = false;
        bool initdesc = false;
        try {
            new (&a()) a_type();
            inita = true;
            new (&b()) b_type();
            initb = true;
            new (&desc()) desc_type();
            initdesc = true;
        } catch(...) {
            try {
                if (initdesc) desc().~desc_type();
                if (initb) b().~b_type();
                if (inita) a().~a_type();
            } catch(...) {
            }
            throw;
        }
    }

    NamedTupleIntFloatDesc(const a_type& a_val, const b_type& b_val, const desc_type& desc_val) {
        bool inita = false;
        bool initb = false;
        bool initdesc = false;
        try {
            new (&a()) a_type(a_val);
            inita = true;
            new (&b()) b_type(b_val);
            initb = true;
            new (&desc()) desc_type(desc_val);
            initdesc = true;
        } catch(...) {
            try {
                if (initdesc) desc().~desc_type();
                if (initb) b().~b_type();
                if (inita) a().~a_type();
            } catch(...) {
            }
            throw;
        }
    }
private:
    static const int size1 = sizeof(a_type);
    static const int size2 = sizeof(b_type);
    static const int size3 = sizeof(desc_type);
    uint8_t data[size1 + size2 + size3];
};

template <>
class TypeDetails<NamedTupleIntFloatDesc> {
public:
    static Type* getType() {
        static Type* t = NamedTupleIntFloatDesc::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("NamedTupleIntFloatDesc somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = 
        sizeof(NamedTupleIntFloatDesc::a_type) +
        sizeof(NamedTupleIntFloatDesc::b_type) +
        sizeof(NamedTupleIntFloatDesc::desc_type);
};

// END Generated NamedTuple NamedTupleIntFloatDesc

// Generated NamedTuple NamedTupleBoolListOfInt
//    X=bool
//    Y=ListOf<int64_t>
class NamedTupleBoolListOfInt {
public:
    typedef bool X_type;
    typedef ListOf<int64_t> Y_type;
    X_type& X() const { return *(X_type*)(data); }
    Y_type& Y() const { return *(Y_type*)(data + size1); }
    static NamedTuple* getType() {
        static NamedTuple* t = NamedTuple::Make({
                TypeDetails<NamedTupleBoolListOfInt::X_type>::getType(),
                TypeDetails<NamedTupleBoolListOfInt::Y_type>::getType()
            },{
                "X",
                "Y"
            });
        return t;
        }

    static NamedTupleBoolListOfInt fromPython(PyObject* p) {
        NamedTupleBoolListOfInt l;
        PyInstance::copyConstructFromPythonInstance(getType(), (instance_ptr)&l, p, true);
        return l;
    }

    PyObject* toPython() {
        return PyInstance::extractPythonObject((instance_ptr)this, getType());
    }

    NamedTupleBoolListOfInt& operator = (const NamedTupleBoolListOfInt& other) {
        X() = other.X();
        Y() = other.Y();
        return *this;
    }

    NamedTupleBoolListOfInt(const NamedTupleBoolListOfInt& other) {
        new (&X()) X_type(other.X());
        new (&Y()) Y_type(other.Y());
    }

    ~NamedTupleBoolListOfInt() {
        Y().~Y_type();
        X().~X_type();
    }

    NamedTupleBoolListOfInt() {
        bool initX = false;
        bool initY = false;
        try {
            new (&X()) X_type();
            initX = true;
            new (&Y()) Y_type();
            initY = true;
        } catch(...) {
            try {
                if (initY) Y().~Y_type();
                if (initX) X().~X_type();
            } catch(...) {
            }
            throw;
        }
    }

    NamedTupleBoolListOfInt(const X_type& X_val, const Y_type& Y_val) {
        bool initX = false;
        bool initY = false;
        try {
            new (&X()) X_type(X_val);
            initX = true;
            new (&Y()) Y_type(Y_val);
            initY = true;
        } catch(...) {
            try {
                if (initY) Y().~Y_type();
                if (initX) X().~X_type();
            } catch(...) {
            }
            throw;
        }
    }
private:
    static const int size1 = sizeof(X_type);
    static const int size2 = sizeof(Y_type);
    uint8_t data[size1 + size2];
};

template <>
class TypeDetails<NamedTupleBoolListOfInt> {
public:
    static Type* getType() {
        static Type* t = NamedTupleBoolListOfInt::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("NamedTupleBoolListOfInt somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = 
        sizeof(NamedTupleBoolListOfInt::X_type) +
        sizeof(NamedTupleBoolListOfInt::Y_type);
};

// END Generated NamedTuple NamedTupleBoolListOfInt

// Generated NamedTuple NamedTupleAttrAndValues
//    attributes=TupleOf<String>
//    values=TupleOf<int64_t>
class NamedTupleAttrAndValues {
public:
    typedef TupleOf<String> attributes_type;
    typedef TupleOf<int64_t> values_type;
    attributes_type& attributes() const { return *(attributes_type*)(data); }
    values_type& values() const { return *(values_type*)(data + size1); }
    static NamedTuple* getType() {
        static NamedTuple* t = NamedTuple::Make({
                TypeDetails<NamedTupleAttrAndValues::attributes_type>::getType(),
                TypeDetails<NamedTupleAttrAndValues::values_type>::getType()
            },{
                "attributes",
                "values"
            });
        return t;
        }

    static NamedTupleAttrAndValues fromPython(PyObject* p) {
        NamedTupleAttrAndValues l;
        PyInstance::copyConstructFromPythonInstance(getType(), (instance_ptr)&l, p, true);
        return l;
    }

    PyObject* toPython() {
        return PyInstance::extractPythonObject((instance_ptr)this, getType());
    }

    NamedTupleAttrAndValues& operator = (const NamedTupleAttrAndValues& other) {
        attributes() = other.attributes();
        values() = other.values();
        return *this;
    }

    NamedTupleAttrAndValues(const NamedTupleAttrAndValues& other) {
        new (&attributes()) attributes_type(other.attributes());
        new (&values()) values_type(other.values());
    }

    ~NamedTupleAttrAndValues() {
        values().~values_type();
        attributes().~attributes_type();
    }

    NamedTupleAttrAndValues() {
        bool initattributes = false;
        bool initvalues = false;
        try {
            new (&attributes()) attributes_type();
            initattributes = true;
            new (&values()) values_type();
            initvalues = true;
        } catch(...) {
            try {
                if (initvalues) values().~values_type();
                if (initattributes) attributes().~attributes_type();
            } catch(...) {
            }
            throw;
        }
    }

    NamedTupleAttrAndValues(const attributes_type& attributes_val, const values_type& values_val) {
        bool initattributes = false;
        bool initvalues = false;
        try {
            new (&attributes()) attributes_type(attributes_val);
            initattributes = true;
            new (&values()) values_type(values_val);
            initvalues = true;
        } catch(...) {
            try {
                if (initvalues) values().~values_type();
                if (initattributes) attributes().~attributes_type();
            } catch(...) {
            }
            throw;
        }
    }
private:
    static const int size1 = sizeof(attributes_type);
    static const int size2 = sizeof(values_type);
    uint8_t data[size1 + size2];
};

template <>
class TypeDetails<NamedTupleAttrAndValues> {
public:
    static Type* getType() {
        static Type* t = NamedTupleAttrAndValues::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("NamedTupleAttrAndValues somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = 
        sizeof(NamedTupleAttrAndValues::attributes_type) +
        sizeof(NamedTupleAttrAndValues::values_type);
};

// END Generated NamedTuple NamedTupleAttrAndValues

// Generated Tuple Anon34726784
//    a0=int64_t
//    a1=int64_t
class Anon34726784 {
public:
    typedef int64_t a0_type;
    typedef int64_t a1_type;
    a0_type& a0() const { return *(a0_type*)(data); }
    a1_type& a1() const { return *(a1_type*)(data + size1); }
    static Tuple* getType() {
        static Tuple* t = Tuple::Make({
                TypeDetails<Anon34726784::a0_type>::getType(),
                TypeDetails<Anon34726784::a1_type>::getType()
            });
        return t;
        }

    static Anon34726784 fromPython(PyObject* p) {
        Anon34726784 l;
        PyInstance::copyConstructFromPythonInstance(getType(), (instance_ptr)&l, p, true);
        return l;
    }

    PyObject* toPython() {
        return PyInstance::extractPythonObject((instance_ptr)this, getType());
    }

    Anon34726784& operator = (const Anon34726784& other) {
        a0() = other.a0();
        a1() = other.a1();
        return *this;
    }

    Anon34726784(const Anon34726784& other) {
        new (&a0()) a0_type(other.a0());
        new (&a1()) a1_type(other.a1());
    }

    ~Anon34726784() {
        a1().~a1_type();
        a0().~a0_type();
    }

    Anon34726784() {
        bool inita0 = false;
        bool inita1 = false;
        try {
            new (&a0()) a0_type();
            inita0 = true;
            new (&a1()) a1_type();
            inita1 = true;
        } catch(...) {
            try {
                if (inita1) a1().~a1_type();
                if (inita0) a0().~a0_type();
            } catch(...) {
            }
            throw;
        }
    }

    Anon34726784(const a0_type& a0_val, const a1_type& a1_val) {
        bool inita0 = false;
        bool inita1 = false;
        try {
            new (&a0()) a0_type(a0_val);
            inita0 = true;
            new (&a1()) a1_type(a1_val);
            inita1 = true;
        } catch(...) {
            try {
                if (inita1) a1().~a1_type();
                if (inita0) a0().~a0_type();
            } catch(...) {
            }
            throw;
        }
    }
private:
    static const int size1 = sizeof(a0_type);
    static const int size2 = sizeof(a1_type);
    uint8_t data[size1 + size2];
};

template <>
class TypeDetails<Anon34726784> {
public:
    static Type* getType() {
        static Type* t = Anon34726784::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("Anon34726784 somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = 
        sizeof(Anon34726784::a0_type) +
        sizeof(Anon34726784::a1_type);
};

// END Generated Tuple Anon34726784

// Generated Tuple Anon34739280
//    a0=bool
//    a1=bool
class Anon34739280 {
public:
    typedef bool a0_type;
    typedef bool a1_type;
    a0_type& a0() const { return *(a0_type*)(data); }
    a1_type& a1() const { return *(a1_type*)(data + size1); }
    static Tuple* getType() {
        static Tuple* t = Tuple::Make({
                TypeDetails<Anon34739280::a0_type>::getType(),
                TypeDetails<Anon34739280::a1_type>::getType()
            });
        return t;
        }

    static Anon34739280 fromPython(PyObject* p) {
        Anon34739280 l;
        PyInstance::copyConstructFromPythonInstance(getType(), (instance_ptr)&l, p, true);
        return l;
    }

    PyObject* toPython() {
        return PyInstance::extractPythonObject((instance_ptr)this, getType());
    }

    Anon34739280& operator = (const Anon34739280& other) {
        a0() = other.a0();
        a1() = other.a1();
        return *this;
    }

    Anon34739280(const Anon34739280& other) {
        new (&a0()) a0_type(other.a0());
        new (&a1()) a1_type(other.a1());
    }

    ~Anon34739280() {
        a1().~a1_type();
        a0().~a0_type();
    }

    Anon34739280() {
        bool inita0 = false;
        bool inita1 = false;
        try {
            new (&a0()) a0_type();
            inita0 = true;
            new (&a1()) a1_type();
            inita1 = true;
        } catch(...) {
            try {
                if (inita1) a1().~a1_type();
                if (inita0) a0().~a0_type();
            } catch(...) {
            }
            throw;
        }
    }

    Anon34739280(const a0_type& a0_val, const a1_type& a1_val) {
        bool inita0 = false;
        bool inita1 = false;
        try {
            new (&a0()) a0_type(a0_val);
            inita0 = true;
            new (&a1()) a1_type(a1_val);
            inita1 = true;
        } catch(...) {
            try {
                if (inita1) a1().~a1_type();
                if (inita0) a0().~a0_type();
            } catch(...) {
            }
            throw;
        }
    }
private:
    static const int size1 = sizeof(a0_type);
    static const int size2 = sizeof(a1_type);
    uint8_t data[size1 + size2];
};

template <>
class TypeDetails<Anon34739280> {
public:
    static Type* getType() {
        static Type* t = Anon34739280::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("Anon34739280 somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = 
        sizeof(Anon34739280::a0_type) +
        sizeof(Anon34739280::a1_type);
};

// END Generated Tuple Anon34739280

// Generated NamedTuple Anon34754928
//    x=int64_t
//    y=int64_t
class Anon34754928 {
public:
    typedef int64_t x_type;
    typedef int64_t y_type;
    x_type& x() const { return *(x_type*)(data); }
    y_type& y() const { return *(y_type*)(data + size1); }
    static NamedTuple* getType() {
        static NamedTuple* t = NamedTuple::Make({
                TypeDetails<Anon34754928::x_type>::getType(),
                TypeDetails<Anon34754928::y_type>::getType()
            },{
                "x",
                "y"
            });
        return t;
        }

    static Anon34754928 fromPython(PyObject* p) {
        Anon34754928 l;
        PyInstance::copyConstructFromPythonInstance(getType(), (instance_ptr)&l, p, true);
        return l;
    }

    PyObject* toPython() {
        return PyInstance::extractPythonObject((instance_ptr)this, getType());
    }

    Anon34754928& operator = (const Anon34754928& other) {
        x() = other.x();
        y() = other.y();
        return *this;
    }

    Anon34754928(const Anon34754928& other) {
        new (&x()) x_type(other.x());
        new (&y()) y_type(other.y());
    }

    ~Anon34754928() {
        y().~y_type();
        x().~x_type();
    }

    Anon34754928() {
        bool initx = false;
        bool inity = false;
        try {
            new (&x()) x_type();
            initx = true;
            new (&y()) y_type();
            inity = true;
        } catch(...) {
            try {
                if (inity) y().~y_type();
                if (initx) x().~x_type();
            } catch(...) {
            }
            throw;
        }
    }

    Anon34754928(const x_type& x_val, const y_type& y_val) {
        bool initx = false;
        bool inity = false;
        try {
            new (&x()) x_type(x_val);
            initx = true;
            new (&y()) y_type(y_val);
            inity = true;
        } catch(...) {
            try {
                if (inity) y().~y_type();
                if (initx) x().~x_type();
            } catch(...) {
            }
            throw;
        }
    }
private:
    static const int size1 = sizeof(x_type);
    static const int size2 = sizeof(y_type);
    uint8_t data[size1 + size2];
};

template <>
class TypeDetails<Anon34754928> {
public:
    static Type* getType() {
        static Type* t = Anon34754928::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("Anon34754928 somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = 
        sizeof(Anon34754928::x_type) +
        sizeof(Anon34754928::y_type);
};

// END Generated NamedTuple Anon34754928

// Generated Tuple AnonTest
//    a0=Dict<Anon34726784, String>
//    a1=ConstDict<String, OneOf<bool, Anon34739280>>
//    a2=ListOf<Anon34726784>
//    a3=TupleOf<Anon34754928>
class AnonTest {
public:
    typedef Dict<Anon34726784, String> a0_type;
    typedef ConstDict<String, OneOf<bool, Anon34739280>> a1_type;
    typedef ListOf<Anon34726784> a2_type;
    typedef TupleOf<Anon34754928> a3_type;
    a0_type& a0() const { return *(a0_type*)(data); }
    a1_type& a1() const { return *(a1_type*)(data + size1); }
    a2_type& a2() const { return *(a2_type*)(data + size1 + size2); }
    a3_type& a3() const { return *(a3_type*)(data + size1 + size2 + size3); }
    static Tuple* getType() {
        static Tuple* t = Tuple::Make({
                TypeDetails<AnonTest::a0_type>::getType(),
                TypeDetails<AnonTest::a1_type>::getType(),
                TypeDetails<AnonTest::a2_type>::getType(),
                TypeDetails<AnonTest::a3_type>::getType()
            });
        return t;
        }

    static AnonTest fromPython(PyObject* p) {
        AnonTest l;
        PyInstance::copyConstructFromPythonInstance(getType(), (instance_ptr)&l, p, true);
        return l;
    }

    PyObject* toPython() {
        return PyInstance::extractPythonObject((instance_ptr)this, getType());
    }

    AnonTest& operator = (const AnonTest& other) {
        a0() = other.a0();
        a1() = other.a1();
        a2() = other.a2();
        a3() = other.a3();
        return *this;
    }

    AnonTest(const AnonTest& other) {
        new (&a0()) a0_type(other.a0());
        new (&a1()) a1_type(other.a1());
        new (&a2()) a2_type(other.a2());
        new (&a3()) a3_type(other.a3());
    }

    ~AnonTest() {
        a3().~a3_type();
        a2().~a2_type();
        a1().~a1_type();
        a0().~a0_type();
    }

    AnonTest() {
        bool inita0 = false;
        bool inita1 = false;
        bool inita2 = false;
        bool inita3 = false;
        try {
            new (&a0()) a0_type();
            inita0 = true;
            new (&a1()) a1_type();
            inita1 = true;
            new (&a2()) a2_type();
            inita2 = true;
            new (&a3()) a3_type();
            inita3 = true;
        } catch(...) {
            try {
                if (inita3) a3().~a3_type();
                if (inita2) a2().~a2_type();
                if (inita1) a1().~a1_type();
                if (inita0) a0().~a0_type();
            } catch(...) {
            }
            throw;
        }
    }

    AnonTest(const a0_type& a0_val, const a1_type& a1_val, const a2_type& a2_val, const a3_type& a3_val) {
        bool inita0 = false;
        bool inita1 = false;
        bool inita2 = false;
        bool inita3 = false;
        try {
            new (&a0()) a0_type(a0_val);
            inita0 = true;
            new (&a1()) a1_type(a1_val);
            inita1 = true;
            new (&a2()) a2_type(a2_val);
            inita2 = true;
            new (&a3()) a3_type(a3_val);
            inita3 = true;
        } catch(...) {
            try {
                if (inita3) a3().~a3_type();
                if (inita2) a2().~a2_type();
                if (inita1) a1().~a1_type();
                if (inita0) a0().~a0_type();
            } catch(...) {
            }
            throw;
        }
    }
private:
    static const int size1 = sizeof(a0_type);
    static const int size2 = sizeof(a1_type);
    static const int size3 = sizeof(a2_type);
    static const int size4 = sizeof(a3_type);
    uint8_t data[size1 + size2 + size3 + size4];
};

template <>
class TypeDetails<AnonTest> {
public:
    static Type* getType() {
        static Type* t = AnonTest::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("AnonTest somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = 
        sizeof(AnonTest::a0_type) +
        sizeof(AnonTest::a1_type) +
        sizeof(AnonTest::a2_type) +
        sizeof(AnonTest::a3_type);
};

// END Generated Tuple AnonTest

