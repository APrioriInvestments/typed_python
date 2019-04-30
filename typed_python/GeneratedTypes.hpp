// Generated Alternative A=
//     Sub1=(b=int64_t, c=int64_t)
//     Sub2=(d=String, e=String)

class A_Sub1;
class A_Sub2;

class A {
public:
    struct e {
        enum kind { Sub1=0, Sub2=1 };
    };

    static NamedTuple* Sub1_Type;
    static NamedTuple* Sub2_Type;

    static Alternative* getType();
    ~A() { getType()->destroy((instance_ptr)&mLayout); }
    A() { getType()->constructor((instance_ptr)&mLayout); }
    A(const A& in) { getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&in.mLayout); }
    A& operator=(const A& other) { getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout); return *this; }

    static A Sub1(const int64_t& b, const int64_t& c);
    static A Sub2(const String& d, const String& e);

    e::kind which() const { return (e::kind)mLayout->which; }

    template <class F>
    auto check(const F& f) {
        if (isSub1()) { return f(*(A_Sub1*)this); }
        if (isSub2()) { return f(*(A_Sub2*)this); }
    }

    bool isSub1() const { return which() == e::Sub1; }
    bool isSub2() const { return which() == e::Sub2; }

    // Accessors for members
    const int64_t& b() const;
    const int64_t& c() const;
    const String& d() const;
    const String& e() const;

protected:
    Alternative::layout *mLayout;
};

template <>
class TypeDetails<A*> {
public:
    static Type* getType() {
        static Type* t = new Forward(0, "A");
        return t;
    }
    static const uint64_t bytecount = sizeof(void*);
};

NamedTuple* A::Sub1_Type = NamedTuple::Make(
    {TypeDetails<int64_t>::getType(), TypeDetails<int64_t>::getType()},
    {"b", "c"}
);

NamedTuple* A::Sub2_Type = NamedTuple::Make(
    {TypeDetails<String>::getType(), TypeDetails<String>::getType()},
    {"d", "e"}
);

// static
Alternative* A::getType() {
    static Alternative* t = Alternative::Make("A", {
        {"Sub1", Sub1_Type},
        {"Sub2", Sub2_Type}
    }, {});
    return t;
}

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
        static ConcreteAlternative* t = ConcreteAlternative::Make(A::getType(), e::Sub1);
        return t;
    }
    static Alternative* getAlternative() { return A::getType(); }

    A_Sub1() { 
        getType()->constructor(
            (instance_ptr)&mLayout,
            [](instance_ptr p) {Sub1_Type->constructor(p);});
    }
    A_Sub1( const int64_t& b1,  const int64_t& c1) {
        A_Sub1(); 
        b() = b1;
        c() = c1;
    }
    A_Sub1(const A_Sub1& other) {
        getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
    }
    A_Sub1& operator=(const A_Sub1& other) {
         getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
         return *this;
    }
    ~A_Sub1() {
        getType()->destroy((instance_ptr)&mLayout);
    }

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
        static ConcreteAlternative* t = ConcreteAlternative::Make(A::getType(), e::Sub2);
        return t;
    }
    static Alternative* getAlternative() { return A::getType(); }

    A_Sub2() { 
        getType()->constructor(
            (instance_ptr)&mLayout,
            [](instance_ptr p) {Sub2_Type->constructor(p);});
    }
    A_Sub2( const String& d1,  const String& e1) {
        A_Sub2(); 
        d() = d1;
        e() = e1;
    }
    A_Sub2(const A_Sub2& other) {
        getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
    }
    A_Sub2& operator=(const A_Sub2& other) {
         getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
         return *this;
    }
    ~A_Sub2() {
        getType()->destroy((instance_ptr)&mLayout);
    }

    String& d() const { return *(String*)(mLayout->data); }
    String& e() const { return *(String*)(mLayout->data + size1); }
private:
    static const int size1 = sizeof(String);
};

A A::Sub2(const String& d, const String& e) {
    return A_Sub2(d, e);
}

const int64_t& A::b() const {
    if (isSub1())
        return ((A_Sub1*)this)->b();
    throw std::runtime_error("\"A\" subtype does not contain \"b\"");
}

const int64_t& A::c() const {
    if (isSub1())
        return ((A_Sub1*)this)->c();
    throw std::runtime_error("\"A\" subtype does not contain \"c\"");
}

const String& A::d() const {
    if (isSub2())
        return ((A_Sub2*)this)->d();
    throw std::runtime_error("\"A\" subtype does not contain \"d\"");
}

const String& A::e() const {
    if (isSub2())
        return ((A_Sub2*)this)->e();
    throw std::runtime_error("\"A\" subtype does not contain \"e\"");
}

// END Generated Alternative A

// Generated Alternative Bexpress=
//     BinOp=(left=Bexpress, op=String, right=Bexpress)
//     UnaryOp=(op=String, right=Bexpress)
//     Leaf=(value=bool)

class Bexpress_BinOp;
class Bexpress_UnaryOp;
class Bexpress_Leaf;

class Bexpress {
public:
    struct e {
        enum kind { BinOp=0, UnaryOp=1, Leaf=2 };
    };

    static NamedTuple* BinOp_Type;
    static NamedTuple* UnaryOp_Type;
    static NamedTuple* Leaf_Type;

    static Alternative* getType();
    ~Bexpress() { getType()->destroy((instance_ptr)&mLayout); }
    Bexpress() { getType()->constructor((instance_ptr)&mLayout); }
    Bexpress(const Bexpress& in) { getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&in.mLayout); }
    Bexpress& operator=(const Bexpress& other) { getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout); return *this; }

    static Bexpress BinOp(const Bexpress& left, const String& op, const Bexpress& right);
    static Bexpress UnaryOp(const String& op, const Bexpress& right);
    static Bexpress Leaf(const bool& value);

    e::kind which() const { return (e::kind)mLayout->which; }

    template <class F>
    auto check(const F& f) {
        if (isBinOp()) { return f(*(Bexpress_BinOp*)this); }
        if (isUnaryOp()) { return f(*(Bexpress_UnaryOp*)this); }
        if (isLeaf()) { return f(*(Bexpress_Leaf*)this); }
    }

    bool isBinOp() const { return which() == e::BinOp; }
    bool isUnaryOp() const { return which() == e::UnaryOp; }
    bool isLeaf() const { return which() == e::Leaf; }

    // Accessors for members
    const Bexpress& left() const;
    const String& op() const;
    const Bexpress& right() const;
    const bool& value() const;

protected:
    Alternative::layout *mLayout;
};

template <>
class TypeDetails<Bexpress*> {
public:
    static Type* getType() {
        static Type* t = new Forward(0, "Bexpress");
        return t;
    }
    static const uint64_t bytecount = sizeof(void*);
};

NamedTuple* Bexpress::BinOp_Type = NamedTuple::Make(
    {TypeDetails<Bexpress*>::getType(), TypeDetails<String>::getType(), TypeDetails<Bexpress*>::getType()},
    {"left", "op", "right"}
);

NamedTuple* Bexpress::UnaryOp_Type = NamedTuple::Make(
    {TypeDetails<String>::getType(), TypeDetails<Bexpress*>::getType()},
    {"op", "right"}
);

NamedTuple* Bexpress::Leaf_Type = NamedTuple::Make(
    {TypeDetails<bool>::getType()},
    {"value"}
);

// static
Alternative* Bexpress::getType() {
    static Alternative* t = Alternative::Make("Bexpress", {
        {"BinOp", BinOp_Type},
        {"UnaryOp", UnaryOp_Type},
        {"Leaf", Leaf_Type}
    }, {});
    BinOp_Type->directResolveForward(TypeDetails<Bexpress*>::getType(), t);
    UnaryOp_Type->directResolveForward(TypeDetails<Bexpress*>::getType(), t);
    return t;
}

template <>
class TypeDetails<Bexpress> {
public:
    static Type* getType() {
        static Type* t = Bexpress::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("Bexpress somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = sizeof(void*);
};

class Bexpress_BinOp : public Bexpress {
public:
    static ConcreteAlternative* getType() {
        static ConcreteAlternative* t = ConcreteAlternative::Make(A::getType(), e::BinOp);
        return t;
    }
    static Alternative* getAlternative() { return Bexpress::getType(); }

    Bexpress_BinOp() { 
        getType()->constructor(
            (instance_ptr)&mLayout,
            [](instance_ptr p) {BinOp_Type->constructor(p);});
    }
    Bexpress_BinOp( const Bexpress& left1,  const String& op1,  const Bexpress& right1) {
        Bexpress_BinOp(); 
        left() = left1;
        op() = op1;
        right() = right1;
    }
    Bexpress_BinOp(const Bexpress_BinOp& other) {
        getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
    }
    Bexpress_BinOp& operator=(const Bexpress_BinOp& other) {
         getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
         return *this;
    }
    ~Bexpress_BinOp() {
        getType()->destroy((instance_ptr)&mLayout);
    }

    Bexpress& left() const { return *(Bexpress*)(mLayout->data); }
    String& op() const { return *(String*)(mLayout->data + size1); }
    Bexpress& right() const { return *(Bexpress*)(mLayout->data + size1 + size2); }
private:
    static const int size1 = sizeof(Bexpress);
    static const int size2 = sizeof(String);
};

Bexpress Bexpress::BinOp(const Bexpress& left, const String& op, const Bexpress& right) {
    return Bexpress_BinOp(left, op, right);
}

class Bexpress_UnaryOp : public Bexpress {
public:
    static ConcreteAlternative* getType() {
        static ConcreteAlternative* t = ConcreteAlternative::Make(A::getType(), e::UnaryOp);
        return t;
    }
    static Alternative* getAlternative() { return Bexpress::getType(); }

    Bexpress_UnaryOp() { 
        getType()->constructor(
            (instance_ptr)&mLayout,
            [](instance_ptr p) {UnaryOp_Type->constructor(p);});
    }
    Bexpress_UnaryOp( const String& op1,  const Bexpress& right1) {
        Bexpress_UnaryOp(); 
        op() = op1;
        right() = right1;
    }
    Bexpress_UnaryOp(const Bexpress_UnaryOp& other) {
        getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
    }
    Bexpress_UnaryOp& operator=(const Bexpress_UnaryOp& other) {
         getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
         return *this;
    }
    ~Bexpress_UnaryOp() {
        getType()->destroy((instance_ptr)&mLayout);
    }

    String& op() const { return *(String*)(mLayout->data); }
    Bexpress& right() const { return *(Bexpress*)(mLayout->data + size1); }
private:
    static const int size1 = sizeof(String);
};

Bexpress Bexpress::UnaryOp(const String& op, const Bexpress& right) {
    return Bexpress_UnaryOp(op, right);
}

class Bexpress_Leaf : public Bexpress {
public:
    static ConcreteAlternative* getType() {
        static ConcreteAlternative* t = ConcreteAlternative::Make(A::getType(), e::Leaf);
        return t;
    }
    static Alternative* getAlternative() { return Bexpress::getType(); }

    Bexpress_Leaf() { 
        getType()->constructor(
            (instance_ptr)&mLayout,
            [](instance_ptr p) {Leaf_Type->constructor(p);});
    }
    Bexpress_Leaf( const bool& value1) {
        Bexpress_Leaf(); 
        value() = value1;
    }
    Bexpress_Leaf(const Bexpress_Leaf& other) {
        getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
    }
    Bexpress_Leaf& operator=(const Bexpress_Leaf& other) {
         getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
         return *this;
    }
    ~Bexpress_Leaf() {
        getType()->destroy((instance_ptr)&mLayout);
    }

    bool& value() const { return *(bool*)(mLayout->data); }
private:
};

Bexpress Bexpress::Leaf(const bool& value) {
    return Bexpress_Leaf(value);
}

const Bexpress& Bexpress::left() const {
    if (isBinOp())
        return ((Bexpress_BinOp*)this)->left();
    throw std::runtime_error("\"Bexpress\" subtype does not contain \"left\"");
}

const String& Bexpress::op() const {
    if (isBinOp())
        return ((Bexpress_BinOp*)this)->op();
    if (isUnaryOp())
        return ((Bexpress_UnaryOp*)this)->op();
    throw std::runtime_error("\"Bexpress\" subtype does not contain \"op\"");
}

const Bexpress& Bexpress::right() const {
    if (isBinOp())
        return ((Bexpress_BinOp*)this)->right();
    if (isUnaryOp())
        return ((Bexpress_UnaryOp*)this)->right();
    throw std::runtime_error("\"Bexpress\" subtype does not contain \"right\"");
}

const bool& Bexpress::value() const {
    if (isLeaf())
        return ((Bexpress_Leaf*)this)->value();
    throw std::runtime_error("\"Bexpress\" subtype does not contain \"value\"");
}

// END Generated Alternative Bexpress

// Generated NamedTuple NamedTupleTwoStrings
//    X=String
//    Y=String
class NamedTupleTwoStrings {
public:
    typedef String X_type;
    typedef String Y_type;
    X_type& X() const { return *(X_type*)(data); }
    Y_type& Y() const { return *(Y_type*)(data + size1); }
private:
    static const int size1 = sizeof(X_type);
    static const int size2 = sizeof(Y_type);
    uint8_t data[size1 + size2];
public:
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
private:
    static const int size1 = sizeof(b_type);
    static const int size2 = sizeof(i_type);
    static const int size3 = sizeof(s_type);
    uint8_t data[size1 + size2 + size3];
public:
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

// Generated NamedTuple Choice
//    A=NamedTupleTwoStrings
//    B=Bexpress
class Choice {
public:
    typedef NamedTupleTwoStrings A_type;
    typedef Bexpress B_type;
    A_type& A() const { return *(A_type*)(data); }
    B_type& B() const { return *(B_type*)(data + size1); }
private:
    static const int size1 = sizeof(A_type);
    static const int size2 = sizeof(B_type);
    uint8_t data[size1 + size2];
public:
    static NamedTuple* getType() {
        static NamedTuple* t = NamedTuple::Make({
                TypeDetails<Choice::A_type>::getType(),
                TypeDetails<Choice::B_type>::getType()
            },{
                "A",
                "B"
            });
        return t;
        }
    Choice& operator = (const Choice& other) {
        A() = other.A();
        B() = other.B();
        return *this;
    }

    Choice(const Choice& other) {
        new (&A()) A_type(other.A());
        new (&B()) B_type(other.B());
    }

    ~Choice() {
        B().~B_type();
        A().~A_type();
    }

    Choice() {
        bool initA = false;
        bool initB = false;
        try {
            new (&A()) A_type();
            initA = true;
            new (&B()) B_type();
            initB = true;
        } catch(...) {
            try {
                if (initB) B().~B_type();
                if (initA) A().~A_type();
            } catch(...) {
            }
            throw;
        }
    }

    Choice(const A_type& A_val, const B_type& B_val) {
        bool initA = false;
        bool initB = false;
        try {
            new (&A()) A_type(A_val);
            initA = true;
            new (&B()) B_type(B_val);
            initB = true;
        } catch(...) {
            try {
                if (initB) B().~B_type();
                if (initA) A().~A_type();
            } catch(...) {
            }
            throw;
        }
    }
};

template <>
class TypeDetails<Choice> {
public:
    static Type* getType() {
        static Type* t = Choice::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("Choice somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = 
        sizeof(Choice::A_type) +
        sizeof(Choice::B_type);
};

// END Generated NamedTuple Choice

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
private:
    static const int size1 = sizeof(a_type);
    static const int size2 = sizeof(b_type);
    static const int size3 = sizeof(desc_type);
    uint8_t data[size1 + size2 + size3];
public:
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
private:
    static const int size1 = sizeof(X_type);
    static const int size2 = sizeof(Y_type);
    uint8_t data[size1 + size2];
public:
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
private:
    static const int size1 = sizeof(attributes_type);
    static const int size2 = sizeof(values_type);
    uint8_t data[size1 + size2];
public:
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

