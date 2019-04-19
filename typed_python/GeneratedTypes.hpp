// Generated Alternative A
//    A=
//   Sub1=(b=int64_t, c=int64_t)
//   Sub2=(d=String, e=String)

class A_Sub1;
class A_Sub2;

class A {
public:
    struct e {
        enum kind { Sub1=0, Sub2=1 };
    };

    static NamedTuple* Sub1_Type;
    typedef int64_t Sub1_b_type;
    typedef int64_t Sub1_c_type;
    static const int Sub1_size1 = sizeof(Sub1_b_type);
    static const int Sub1_size2 = sizeof(Sub1_c_type);
    static NamedTuple* Sub2_Type;
    typedef String Sub2_d_type;
    typedef String Sub2_e_type;
    static const int Sub2_size1 = sizeof(Sub2_d_type);
    static const int Sub2_size2 = sizeof(Sub2_e_type);

    static Alternative* getType() {
        static Alternative* t = Alternative::Make("A", {
            {"Sub1", Sub1_Type},
            {"Sub2", Sub2_Type}
        }, {});
        return t;
    }
    ~A();
    A(); // only if the whole alternative is default initializable
    A(const A& in);
    A& operator=(const A& other);

    static A Sub1(Sub1_b_type b, Sub1_c_type c);
    static A Sub2(Sub2_d_type d, Sub2_e_type e);

    e::kind which() const { return (e::kind)mLayout->which; }

    template <class F>
    auto check(const F& f) {
        if (isSub1()) { return f(*(A_Sub1*)this); }
        if (isSub2()) { return f(*(A_Sub2*)this); }
    }

    bool isSub1() const { return which() == e::Sub1; }
    bool isSub2() const { return which() == e::Sub2; }

    // Accessors for elements of all subtypes here
    // But account for element name overlap and type differences...
    // const Sub1_b_type& b() const {}
    // const Sub1_c_type& c() const {}
    // const Sub2_d_type& d() const {}
    // const Sub2_e_type& e() const {}
    Alternative::layout* getLayout() const { return mLayout; }
private:
    Alternative::layout *mLayout;
};

NamedTuple* A::Sub1_Type = NamedTuple::Make(
    {TypeDetails<Sub1_b_type>::getType(), TypeDetails<Sub1_c_type>::getType()},
    {"b", "c"}
);
NamedTuple* A::Sub2_Type = NamedTuple::Make(
    {TypeDetails<Sub2_d_type>::getType(), TypeDetails<Sub2_e_type>::getType()},
    {"d", "e"}
);

class A_Sub1 : public A {
public:
    static ConcreteAlternative* getType() {
        static ConcreteAlternative* t = ConcreteAlternative::Make(A::getType(), e::Sub1);
        return t;
    }
    static Alternative* getAlternative() { return A::getType(); }

    A_Sub1() { 
        getType()->constructor(
            (instance_ptr)getLayout(),
            [](instance_ptr p) {Sub1_Type->constructor(p);});
    }
    A_Sub1(Sub1_b_type b1, Sub1_c_type c1) {
        A_Sub1(); 
        b() = b1;
        c() = c1;
     }
    A_Sub1(const A_Sub1& other) {
        getType()->copy_constructor((instance_ptr)getLayout(), (instance_ptr)other.getLayout());
    }
    A_Sub1& operator=(const A_Sub1& other) {
         getType()->assign((instance_ptr)getLayout(), (instance_ptr)other.getLayout());
    }
    ~A_Sub1() {
        getType()->destroy((instance_ptr)getLayout());
    }

    Sub1_b_type& b() const { return *(Sub1_b_type*)(getLayout()->data); }
    Sub1_c_type& c() const { return *(Sub1_c_type*)(getLayout()->data + Sub1_size1); }
};

class A_Sub2 : public A {
public:
    static ConcreteAlternative* getType() {
        static ConcreteAlternative* t = ConcreteAlternative::Make(A::getType(), e::Sub2);
        return t;
    }
    static Alternative* getAlternative() { return A::getType(); }

    A_Sub2() { 
        getType()->constructor(
            (instance_ptr)getLayout(),
            [](instance_ptr p) {Sub2_Type->constructor(p);});
    }
    A_Sub2(Sub2_d_type d1, Sub2_e_type e1) {
        A_Sub2(); 
        d() = d1;
        e() = e1;
     }
    A_Sub2(const A_Sub2& other) {
        getType()->copy_constructor((instance_ptr)getLayout(), (instance_ptr)other.getLayout());
    }
    A_Sub2& operator=(const A_Sub2& other) {
         getType()->assign((instance_ptr)getLayout(), (instance_ptr)other.getLayout());
    }
    ~A_Sub2() {
        getType()->destroy((instance_ptr)getLayout());
    }

    Sub2_d_type& d() const { return *(Sub2_d_type*)(getLayout()->data); }
    Sub2_e_type& e() const { return *(Sub2_e_type*)(getLayout()->data + Sub2_size1); }
};

template <>
class TypeDetails<A> {
public:
    static Type* getType() {
        static Type* t = A::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = sizeof(void*);
};

class Bexpress;

template <>
class TypeDetails<Bexpress*> {
public:
    static Type* getType() {
        static Type* t = PointerTo::Make(Int64::Make());  // forward types are not actually constructed
        return t;
    }
    static const uint64_t bytecount = sizeof(void*);
};

// Generated Alternative Bexpress
//    Bexpress=
//   BinOp=(left=Bexpress*, op=String, right=Bexpress*)
//   UnaryOp=(op=String, right=Bexpress*)
//   Leaf=(value=bool)

class Bexpress_BinOp;
class Bexpress_UnaryOp;
class Bexpress_Leaf;

class Bexpress {
public:
    struct e {
        enum kind { BinOp=0, UnaryOp=1, Leaf=2 };
    };

    static NamedTuple* BinOp_Type;
    typedef Bexpress* BinOp_left_type;
    typedef String BinOp_op_type;
    typedef Bexpress* BinOp_right_type;
    static const int BinOp_size1 = sizeof(BinOp_left_type);
    static const int BinOp_size2 = sizeof(BinOp_op_type);
    static const int BinOp_size3 = sizeof(BinOp_right_type);
    static NamedTuple* UnaryOp_Type;
    typedef String UnaryOp_op_type;
    typedef Bexpress* UnaryOp_right_type;
    static const int UnaryOp_size1 = sizeof(UnaryOp_op_type);
    static const int UnaryOp_size2 = sizeof(UnaryOp_right_type);
    static NamedTuple* Leaf_Type;
    typedef bool Leaf_value_type;
    static const int Leaf_size1 = sizeof(Leaf_value_type);

    static Alternative* getType() {
        static Alternative* t = Alternative::Make("Bexpress", {
            {"BinOp", BinOp_Type},
            {"UnaryOp", UnaryOp_Type},
            {"Leaf", Leaf_Type}
        }, {});
        return t;
    }
    ~Bexpress();
    Bexpress(); // only if the whole alternative is default initializable
    Bexpress(const Bexpress& in);
    Bexpress& operator=(const Bexpress& other);

    static Bexpress BinOp(BinOp_left_type left, BinOp_op_type op, BinOp_right_type right);
    static Bexpress UnaryOp(UnaryOp_op_type op, UnaryOp_right_type right);
    static Bexpress Leaf(Leaf_value_type value);

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

    // Accessors for elements of all subtypes here
    // But account for element name overlap and type differences...
    // const BinOp_left_type& left() const {}
    // const BinOp_op_type& op() const {}
    // const BinOp_right_type& right() const {}
    // const UnaryOp_op_type& op() const {}
    // const UnaryOp_right_type& right() const {}
    // const Leaf_value_type& value() const {}
    Alternative::layout* getLayout() const { return mLayout; }
private:
    Alternative::layout *mLayout;
};

NamedTuple* Bexpress::BinOp_Type = NamedTuple::Make(
    {TypeDetails<BinOp_left_type>::getType(), TypeDetails<BinOp_op_type>::getType(), TypeDetails<BinOp_right_type>::getType()},
    {"left", "op", "right"}
);
NamedTuple* Bexpress::UnaryOp_Type = NamedTuple::Make(
    {TypeDetails<UnaryOp_op_type>::getType(), TypeDetails<UnaryOp_right_type>::getType()},
    {"op", "right"}
);
NamedTuple* Bexpress::Leaf_Type = NamedTuple::Make(
    {TypeDetails<Leaf_value_type>::getType()},
    {"value"}
);

class Bexpress_BinOp : public Bexpress {
public:
    static ConcreteAlternative* getType() {
        static ConcreteAlternative* t = ConcreteAlternative::Make(A::getType(), e::BinOp);
        return t;
    }
    static Alternative* getAlternative() { return Bexpress::getType(); }

    Bexpress_BinOp() { 
        getType()->constructor(
            (instance_ptr)getLayout(),
            [](instance_ptr p) {BinOp_Type->constructor(p);});
    }
    Bexpress_BinOp(BinOp_left_type left1, BinOp_op_type op1, BinOp_right_type right1) {
        Bexpress_BinOp(); 
        left() = left1;
        op() = op1;
        right() = right1;
     }
    Bexpress_BinOp(const Bexpress_BinOp& other) {
        getType()->copy_constructor((instance_ptr)getLayout(), (instance_ptr)other.getLayout());
    }
    Bexpress_BinOp& operator=(const Bexpress_BinOp& other) {
         getType()->assign((instance_ptr)getLayout(), (instance_ptr)other.getLayout());
    }
    ~Bexpress_BinOp() {
        getType()->destroy((instance_ptr)getLayout());
    }

    BinOp_left_type& left() const { return *(BinOp_left_type*)(getLayout()->data); }
    BinOp_op_type& op() const { return *(BinOp_op_type*)(getLayout()->data + BinOp_size1); }
    BinOp_right_type& right() const { return *(BinOp_right_type*)(getLayout()->data + BinOp_size1 + BinOp_size2); }
};

class Bexpress_UnaryOp : public Bexpress {
public:
    static ConcreteAlternative* getType() {
        static ConcreteAlternative* t = ConcreteAlternative::Make(A::getType(), e::UnaryOp);
        return t;
    }
    static Alternative* getAlternative() { return Bexpress::getType(); }

    Bexpress_UnaryOp() { 
        getType()->constructor(
            (instance_ptr)getLayout(),
            [](instance_ptr p) {UnaryOp_Type->constructor(p);});
    }
    Bexpress_UnaryOp(UnaryOp_op_type op1, UnaryOp_right_type right1) {
        Bexpress_UnaryOp(); 
        op() = op1;
        right() = right1;
     }
    Bexpress_UnaryOp(const Bexpress_UnaryOp& other) {
        getType()->copy_constructor((instance_ptr)getLayout(), (instance_ptr)other.getLayout());
    }
    Bexpress_UnaryOp& operator=(const Bexpress_UnaryOp& other) {
         getType()->assign((instance_ptr)getLayout(), (instance_ptr)other.getLayout());
    }
    ~Bexpress_UnaryOp() {
        getType()->destroy((instance_ptr)getLayout());
    }

    UnaryOp_op_type& op() const { return *(UnaryOp_op_type*)(getLayout()->data); }
    UnaryOp_right_type& right() const { return *(UnaryOp_right_type*)(getLayout()->data + UnaryOp_size1); }
};

class Bexpress_Leaf : public Bexpress {
public:
    static ConcreteAlternative* getType() {
        static ConcreteAlternative* t = ConcreteAlternative::Make(A::getType(), e::Leaf);
        return t;
    }
    static Alternative* getAlternative() { return Bexpress::getType(); }

    Bexpress_Leaf() { 
        getType()->constructor(
            (instance_ptr)getLayout(),
            [](instance_ptr p) {Leaf_Type->constructor(p);});
    }
    Bexpress_Leaf(Leaf_value_type value1) {
        Bexpress_Leaf(); 
        value() = value1;
     }
    Bexpress_Leaf(const Bexpress_Leaf& other) {
        getType()->copy_constructor((instance_ptr)getLayout(), (instance_ptr)other.getLayout());
    }
    Bexpress_Leaf& operator=(const Bexpress_Leaf& other) {
         getType()->assign((instance_ptr)getLayout(), (instance_ptr)other.getLayout());
    }
    ~Bexpress_Leaf() {
        getType()->destroy((instance_ptr)getLayout());
    }

    Leaf_value_type& value() const { return *(Leaf_value_type*)(getLayout()->data); }
};

template <>
class TypeDetails<Bexpress> {
public:
    static Type* getType() {
        static Type* t = Bexpress::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = sizeof(void*);
};

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
};

template <>
class TypeDetails<NamedTupleTwoStrings> {
public:
    static Type* getType() {
        static Type* t = NamedTupleTwoStrings::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = 
        sizeof(NamedTupleTwoStrings::X_type) +
        sizeof(NamedTupleTwoStrings::Y_type);
};

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
};

template <>
class TypeDetails<Choice> {
public:
    static Type* getType() {
        static Type* t = Choice::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = 
        sizeof(Choice::A_type) +
        sizeof(Choice::B_type);
};

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
};

template <>
class TypeDetails<NamedTupleIntFloatDesc> {
public:
    static Type* getType() {
        static Type* t = NamedTupleIntFloatDesc::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = 
        sizeof(NamedTupleIntFloatDesc::a_type) +
        sizeof(NamedTupleIntFloatDesc::b_type) +
        sizeof(NamedTupleIntFloatDesc::desc_type);
};

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
};

template <>
class TypeDetails<NamedTupleBoolListOfInt> {
public:
    static Type* getType() {
        static Type* t = NamedTupleBoolListOfInt::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = 
        sizeof(NamedTupleBoolListOfInt::X_type) +
        sizeof(NamedTupleBoolListOfInt::Y_type);
};

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
};

template <>
class TypeDetails<NamedTupleAttrAndValues> {
public:
    static Type* getType() {
        static Type* t = NamedTupleAttrAndValues::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = 
        sizeof(NamedTupleAttrAndValues::attributes_type) +
        sizeof(NamedTupleAttrAndValues::values_type);
};

