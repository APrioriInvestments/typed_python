// Generated Tuple Anon39222656
//    a0=String
//    a1=Bytes
class Anon39222656 {
public:
    typedef String a0_type;
    typedef Bytes a1_type;
    a0_type& a0() const { return *(a0_type*)(data); }
    a1_type& a1() const { return *(a1_type*)(data + size1); }
private:
    static const int size1 = sizeof(a0_type);
    static const int size2 = sizeof(a1_type);
    uint8_t data[size1 + size2];
public:
    static Tuple* getType() {
        static Tuple* t = Tuple::Make({
                TypeDetails<Anon39222656::a0_type>::getType(),
                TypeDetails<Anon39222656::a1_type>::getType()
            });
        return t;
        }
    Anon39222656& operator = (const Anon39222656& other) {
        a0() = other.a0();
        a1() = other.a1();
        return *this;
    }

    Anon39222656(const Anon39222656& other) {
        new (&a0()) a0_type(other.a0());
        new (&a1()) a1_type(other.a1());
    }

    ~Anon39222656() {
        a1().~a1_type();
        a0().~a0_type();
    }

    Anon39222656() {
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

    Anon39222656(const a0_type& a0_val, const a1_type& a1_val) {
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
};

template <>
class TypeDetails<Anon39222656> {
public:
    static Type* getType() {
        static Type* t = Anon39222656::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("Anon39222656 somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = 
        sizeof(Anon39222656::a0_type) +
        sizeof(Anon39222656::a1_type);
};

// END Generated NamedTuple Anon39222656

// Generated Tuple Anon40676176
//    a0=int64_t
//    a1=int64_t
class Anon40676176 {
public:
    typedef int64_t a0_type;
    typedef int64_t a1_type;
    a0_type& a0() const { return *(a0_type*)(data); }
    a1_type& a1() const { return *(a1_type*)(data + size1); }
private:
    static const int size1 = sizeof(a0_type);
    static const int size2 = sizeof(a1_type);
    uint8_t data[size1 + size2];
public:
    static Tuple* getType() {
        static Tuple* t = Tuple::Make({
                TypeDetails<Anon40676176::a0_type>::getType(),
                TypeDetails<Anon40676176::a1_type>::getType()
            });
        return t;
        }
    Anon40676176& operator = (const Anon40676176& other) {
        a0() = other.a0();
        a1() = other.a1();
        return *this;
    }

    Anon40676176(const Anon40676176& other) {
        new (&a0()) a0_type(other.a0());
        new (&a1()) a1_type(other.a1());
    }

    ~Anon40676176() {
        a1().~a1_type();
        a0().~a0_type();
    }

    Anon40676176() {
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

    Anon40676176(const a0_type& a0_val, const a1_type& a1_val) {
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
};

template <>
class TypeDetails<Anon40676176> {
public:
    static Type* getType() {
        static Type* t = Anon40676176::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("Anon40676176 somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = 
        sizeof(Anon40676176::a0_type) +
        sizeof(Anon40676176::a1_type);
};

// END Generated NamedTuple Anon40676176

// Generated Tuple Anon40683760
//    a0=bool
//    a1=bool
class Anon40683760 {
public:
    typedef bool a0_type;
    typedef bool a1_type;
    a0_type& a0() const { return *(a0_type*)(data); }
    a1_type& a1() const { return *(a1_type*)(data + size1); }
private:
    static const int size1 = sizeof(a0_type);
    static const int size2 = sizeof(a1_type);
    uint8_t data[size1 + size2];
public:
    static Tuple* getType() {
        static Tuple* t = Tuple::Make({
                TypeDetails<Anon40683760::a0_type>::getType(),
                TypeDetails<Anon40683760::a1_type>::getType()
            });
        return t;
        }
    Anon40683760& operator = (const Anon40683760& other) {
        a0() = other.a0();
        a1() = other.a1();
        return *this;
    }

    Anon40683760(const Anon40683760& other) {
        new (&a0()) a0_type(other.a0());
        new (&a1()) a1_type(other.a1());
    }

    ~Anon40683760() {
        a1().~a1_type();
        a0().~a0_type();
    }

    Anon40683760() {
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

    Anon40683760(const a0_type& a0_val, const a1_type& a1_val) {
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
};

template <>
class TypeDetails<Anon40683760> {
public:
    static Type* getType() {
        static Type* t = Anon40683760::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("Anon40683760 somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = 
        sizeof(Anon40683760::a0_type) +
        sizeof(Anon40683760::a1_type);
};

// END Generated NamedTuple Anon40683760

// Generated NamedTuple Anon40698976
//    x=int64_t
//    y=int64_t
class Anon40698976 {
public:
    typedef int64_t x_type;
    typedef int64_t y_type;
    x_type& x() const { return *(x_type*)(data); }
    y_type& y() const { return *(y_type*)(data + size1); }
private:
    static const int size1 = sizeof(x_type);
    static const int size2 = sizeof(y_type);
    uint8_t data[size1 + size2];
public:
    static NamedTuple* getType() {
        static NamedTuple* t = NamedTuple::Make({
                TypeDetails<Anon40698976::x_type>::getType(),
                TypeDetails<Anon40698976::y_type>::getType()
            },{
                "x",
                "y"
            });
        return t;
        }
    Anon40698976& operator = (const Anon40698976& other) {
        x() = other.x();
        y() = other.y();
        return *this;
    }

    Anon40698976(const Anon40698976& other) {
        new (&x()) x_type(other.x());
        new (&y()) y_type(other.y());
    }

    ~Anon40698976() {
        y().~y_type();
        x().~x_type();
    }

    Anon40698976() {
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

    Anon40698976(const x_type& x_val, const y_type& y_val) {
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
};

template <>
class TypeDetails<Anon40698976> {
public:
    static Type* getType() {
        static Type* t = Anon40698976::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("Anon40698976 somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = 
        sizeof(Anon40698976::x_type) +
        sizeof(Anon40698976::y_type);
};

// END Generated NamedTuple Anon40698976

// Generated NamedTuple ObjectFieldId
//    objId=int64_t
//    fieldId=int64_t
//    isIndexValue=bool
class ObjectFieldId {
public:
    typedef int64_t objId_type;
    typedef int64_t fieldId_type;
    typedef bool isIndexValue_type;
    objId_type& objId() const { return *(objId_type*)(data); }
    fieldId_type& fieldId() const { return *(fieldId_type*)(data + size1); }
    isIndexValue_type& isIndexValue() const { return *(isIndexValue_type*)(data + size1 + size2); }
private:
    static const int size1 = sizeof(objId_type);
    static const int size2 = sizeof(fieldId_type);
    static const int size3 = sizeof(isIndexValue_type);
    uint8_t data[size1 + size2 + size3];
public:
    static NamedTuple* getType() {
        static NamedTuple* t = NamedTuple::Make({
                TypeDetails<ObjectFieldId::objId_type>::getType(),
                TypeDetails<ObjectFieldId::fieldId_type>::getType(),
                TypeDetails<ObjectFieldId::isIndexValue_type>::getType()
            },{
                "objId",
                "fieldId",
                "isIndexValue"
            });
        return t;
        }
    ObjectFieldId& operator = (const ObjectFieldId& other) {
        objId() = other.objId();
        fieldId() = other.fieldId();
        isIndexValue() = other.isIndexValue();
        return *this;
    }

    ObjectFieldId(const ObjectFieldId& other) {
        new (&objId()) objId_type(other.objId());
        new (&fieldId()) fieldId_type(other.fieldId());
        new (&isIndexValue()) isIndexValue_type(other.isIndexValue());
    }

    ~ObjectFieldId() {
        isIndexValue().~isIndexValue_type();
        fieldId().~fieldId_type();
        objId().~objId_type();
    }

    ObjectFieldId() {
        bool initobjId = false;
        bool initfieldId = false;
        bool initisIndexValue = false;
        try {
            new (&objId()) objId_type();
            initobjId = true;
            new (&fieldId()) fieldId_type();
            initfieldId = true;
            new (&isIndexValue()) isIndexValue_type();
            initisIndexValue = true;
        } catch(...) {
            try {
                if (initisIndexValue) isIndexValue().~isIndexValue_type();
                if (initfieldId) fieldId().~fieldId_type();
                if (initobjId) objId().~objId_type();
            } catch(...) {
            }
            throw;
        }
    }

    ObjectFieldId(const objId_type& objId_val, const fieldId_type& fieldId_val, const isIndexValue_type& isIndexValue_val) {
        bool initobjId = false;
        bool initfieldId = false;
        bool initisIndexValue = false;
        try {
            new (&objId()) objId_type(objId_val);
            initobjId = true;
            new (&fieldId()) fieldId_type(fieldId_val);
            initfieldId = true;
            new (&isIndexValue()) isIndexValue_type(isIndexValue_val);
            initisIndexValue = true;
        } catch(...) {
            try {
                if (initisIndexValue) isIndexValue().~isIndexValue_type();
                if (initfieldId) fieldId().~fieldId_type();
                if (initobjId) objId().~objId_type();
            } catch(...) {
            }
            throw;
        }
    }
};

template <>
class TypeDetails<ObjectFieldId> {
public:
    static Type* getType() {
        static Type* t = ObjectFieldId::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("ObjectFieldId somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = 
        sizeof(ObjectFieldId::objId_type) +
        sizeof(ObjectFieldId::fieldId_type) +
        sizeof(ObjectFieldId::isIndexValue_type);
};

// END Generated NamedTuple ObjectFieldId

// Generated NamedTuple IndexId
//    fieldId=int64_t
//    indexValue=Bytes
class IndexId {
public:
    typedef int64_t fieldId_type;
    typedef Bytes indexValue_type;
    fieldId_type& fieldId() const { return *(fieldId_type*)(data); }
    indexValue_type& indexValue() const { return *(indexValue_type*)(data + size1); }
private:
    static const int size1 = sizeof(fieldId_type);
    static const int size2 = sizeof(indexValue_type);
    uint8_t data[size1 + size2];
public:
    static NamedTuple* getType() {
        static NamedTuple* t = NamedTuple::Make({
                TypeDetails<IndexId::fieldId_type>::getType(),
                TypeDetails<IndexId::indexValue_type>::getType()
            },{
                "fieldId",
                "indexValue"
            });
        return t;
        }
    IndexId& operator = (const IndexId& other) {
        fieldId() = other.fieldId();
        indexValue() = other.indexValue();
        return *this;
    }

    IndexId(const IndexId& other) {
        new (&fieldId()) fieldId_type(other.fieldId());
        new (&indexValue()) indexValue_type(other.indexValue());
    }

    ~IndexId() {
        indexValue().~indexValue_type();
        fieldId().~fieldId_type();
    }

    IndexId() {
        bool initfieldId = false;
        bool initindexValue = false;
        try {
            new (&fieldId()) fieldId_type();
            initfieldId = true;
            new (&indexValue()) indexValue_type();
            initindexValue = true;
        } catch(...) {
            try {
                if (initindexValue) indexValue().~indexValue_type();
                if (initfieldId) fieldId().~fieldId_type();
            } catch(...) {
            }
            throw;
        }
    }

    IndexId(const fieldId_type& fieldId_val, const indexValue_type& indexValue_val) {
        bool initfieldId = false;
        bool initindexValue = false;
        try {
            new (&fieldId()) fieldId_type(fieldId_val);
            initfieldId = true;
            new (&indexValue()) indexValue_type(indexValue_val);
            initindexValue = true;
        } catch(...) {
            try {
                if (initindexValue) indexValue().~indexValue_type();
                if (initfieldId) fieldId().~fieldId_type();
            } catch(...) {
            }
            throw;
        }
    }
};

template <>
class TypeDetails<IndexId> {
public:
    static Type* getType() {
        static Type* t = IndexId::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("IndexId somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = 
        sizeof(IndexId::fieldId_type) +
        sizeof(IndexId::indexValue_type);
};

// END Generated NamedTuple IndexId

// Generated NamedTuple FieldDefinition
//    schema=String
//    typename0=String
//    fieldname=String
class FieldDefinition {
public:
    typedef String schema_type;
    typedef String typename0_type;
    typedef String fieldname_type;
    schema_type& schema() const { return *(schema_type*)(data); }
    typename0_type& typename0() const { return *(typename0_type*)(data + size1); }
    fieldname_type& fieldname() const { return *(fieldname_type*)(data + size1 + size2); }
private:
    static const int size1 = sizeof(schema_type);
    static const int size2 = sizeof(typename0_type);
    static const int size3 = sizeof(fieldname_type);
    uint8_t data[size1 + size2 + size3];
public:
    static NamedTuple* getType() {
        static NamedTuple* t = NamedTuple::Make({
                TypeDetails<FieldDefinition::schema_type>::getType(),
                TypeDetails<FieldDefinition::typename0_type>::getType(),
                TypeDetails<FieldDefinition::fieldname_type>::getType()
            },{
                "schema",
                "typename0",
                "fieldname"
            });
        return t;
        }
    FieldDefinition& operator = (const FieldDefinition& other) {
        schema() = other.schema();
        typename0() = other.typename0();
        fieldname() = other.fieldname();
        return *this;
    }

    FieldDefinition(const FieldDefinition& other) {
        new (&schema()) schema_type(other.schema());
        new (&typename0()) typename0_type(other.typename0());
        new (&fieldname()) fieldname_type(other.fieldname());
    }

    ~FieldDefinition() {
        fieldname().~fieldname_type();
        typename0().~typename0_type();
        schema().~schema_type();
    }

    FieldDefinition() {
        bool initschema = false;
        bool inittypename0 = false;
        bool initfieldname = false;
        try {
            new (&schema()) schema_type();
            initschema = true;
            new (&typename0()) typename0_type();
            inittypename0 = true;
            new (&fieldname()) fieldname_type();
            initfieldname = true;
        } catch(...) {
            try {
                if (initfieldname) fieldname().~fieldname_type();
                if (inittypename0) typename0().~typename0_type();
                if (initschema) schema().~schema_type();
            } catch(...) {
            }
            throw;
        }
    }

    FieldDefinition(const schema_type& schema_val, const typename0_type& typename0_val, const fieldname_type& fieldname_val) {
        bool initschema = false;
        bool inittypename0 = false;
        bool initfieldname = false;
        try {
            new (&schema()) schema_type(schema_val);
            initschema = true;
            new (&typename0()) typename0_type(typename0_val);
            inittypename0 = true;
            new (&fieldname()) fieldname_type(fieldname_val);
            initfieldname = true;
        } catch(...) {
            try {
                if (initfieldname) fieldname().~fieldname_type();
                if (inittypename0) typename0().~typename0_type();
                if (initschema) schema().~schema_type();
            } catch(...) {
            }
            throw;
        }
    }
};

template <>
class TypeDetails<FieldDefinition> {
public:
    static Type* getType() {
        static Type* t = FieldDefinition::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("FieldDefinition somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = 
        sizeof(FieldDefinition::schema_type) +
        sizeof(FieldDefinition::typename0_type) +
        sizeof(FieldDefinition::fieldname_type);
};

// END Generated NamedTuple FieldDefinition

// Generated NamedTuple TypeDefinition
//    fields=TupleOf<String>
//    indices=TupleOf<String>
class TypeDefinition {
public:
    typedef TupleOf<String> fields_type;
    typedef TupleOf<String> indices_type;
    fields_type& fields() const { return *(fields_type*)(data); }
    indices_type& indices() const { return *(indices_type*)(data + size1); }
private:
    static const int size1 = sizeof(fields_type);
    static const int size2 = sizeof(indices_type);
    uint8_t data[size1 + size2];
public:
    static NamedTuple* getType() {
        static NamedTuple* t = NamedTuple::Make({
                TypeDetails<TypeDefinition::fields_type>::getType(),
                TypeDetails<TypeDefinition::indices_type>::getType()
            },{
                "fields",
                "indices"
            });
        return t;
        }
    TypeDefinition& operator = (const TypeDefinition& other) {
        fields() = other.fields();
        indices() = other.indices();
        return *this;
    }

    TypeDefinition(const TypeDefinition& other) {
        new (&fields()) fields_type(other.fields());
        new (&indices()) indices_type(other.indices());
    }

    ~TypeDefinition() {
        indices().~indices_type();
        fields().~fields_type();
    }

    TypeDefinition() {
        bool initfields = false;
        bool initindices = false;
        try {
            new (&fields()) fields_type();
            initfields = true;
            new (&indices()) indices_type();
            initindices = true;
        } catch(...) {
            try {
                if (initindices) indices().~indices_type();
                if (initfields) fields().~fields_type();
            } catch(...) {
            }
            throw;
        }
    }

    TypeDefinition(const fields_type& fields_val, const indices_type& indices_val) {
        bool initfields = false;
        bool initindices = false;
        try {
            new (&fields()) fields_type(fields_val);
            initfields = true;
            new (&indices()) indices_type(indices_val);
            initindices = true;
        } catch(...) {
            try {
                if (initindices) indices().~indices_type();
                if (initfields) fields().~fields_type();
            } catch(...) {
            }
            throw;
        }
    }
};

template <>
class TypeDetails<TypeDefinition> {
public:
    static Type* getType() {
        static Type* t = TypeDefinition::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("TypeDefinition somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = 
        sizeof(TypeDefinition::fields_type) +
        sizeof(TypeDefinition::indices_type);
};

// END Generated NamedTuple TypeDefinition

// Generated Alternative ClientToServer=
//     TransactionData=(writes=ConstDict<ObjectFieldId, OneOf<None, Bytes>>, set_adds=ConstDict<IndexId, TupleOf<int64_t>>, set_removes=ConstDict<IndexId, TupleOf<int64_t>>, key_versions=TupleOf<ObjectFieldId>, index_versions=TupleOf<IndexId>, transaction_guid=int64_t)
//     CompleteTransaction=(as_of_version=int64_t, transaction_guid=int64_t)
//     Heartbeat=()
//     DefineSchema=(name=String, definition=ConstDict<String, TypeDefinition>)
//     LoadLazyObject=(schema=String, typename0=String, identity=int64_t)
//     Subscribe=(schema=String, typename0=OneOf<None, String>, fieldname_and_value=OneOf<None, Anon39222656>, isLazy=bool)
//     Flush=(guid=int64_t)
//     Authenticate=(token=String)

class ClientToServer_TransactionData;
class ClientToServer_CompleteTransaction;
class ClientToServer_Heartbeat;
class ClientToServer_DefineSchema;
class ClientToServer_LoadLazyObject;
class ClientToServer_Subscribe;
class ClientToServer_Flush;
class ClientToServer_Authenticate;

class ClientToServer {
public:
    enum class kind { TransactionData=0, CompleteTransaction=1, Heartbeat=2, DefineSchema=3, LoadLazyObject=4, Subscribe=5, Flush=6, Authenticate=7 };

    static NamedTuple* TransactionData_Type;
    static NamedTuple* CompleteTransaction_Type;
    static NamedTuple* Heartbeat_Type;
    static NamedTuple* DefineSchema_Type;
    static NamedTuple* LoadLazyObject_Type;
    static NamedTuple* Subscribe_Type;
    static NamedTuple* Flush_Type;
    static NamedTuple* Authenticate_Type;

    static Alternative* getType();
    ~ClientToServer() { getType()->destroy((instance_ptr)&mLayout); }
    ClientToServer():mLayout(0) { getType()->constructor((instance_ptr)&mLayout); }
    ClientToServer(kind k):mLayout(0) { ConcreteAlternative::Make(getType(), (int64_t)k)->constructor((instance_ptr)&mLayout); }
    ClientToServer(const ClientToServer& in) { getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&in.mLayout); }
    ClientToServer& operator=(const ClientToServer& other) { getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout); return *this; }

    static ClientToServer TransactionData(const ConstDict<ObjectFieldId, OneOf<None, Bytes>>& writes, const ConstDict<IndexId, TupleOf<int64_t>>& set_adds, const ConstDict<IndexId, TupleOf<int64_t>>& set_removes, const TupleOf<ObjectFieldId>& key_versions, const TupleOf<IndexId>& index_versions, const int64_t& transaction_guid);
    static ClientToServer CompleteTransaction(const int64_t& as_of_version, const int64_t& transaction_guid);
    static ClientToServer Heartbeat();
    static ClientToServer DefineSchema(const String& name, const ConstDict<String, TypeDefinition>& definition);
    static ClientToServer LoadLazyObject(const String& schema, const String& typename0, const int64_t& identity);
    static ClientToServer Subscribe(const String& schema, const OneOf<None, String>& typename0, const OneOf<None, Anon39222656>& fieldname_and_value, const bool& isLazy);
    static ClientToServer Flush(const int64_t& guid);
    static ClientToServer Authenticate(const String& token);

    kind which() const { return (kind)mLayout->which; }

    template <class F>
    auto check(const F& f) {
        if (isTransactionData()) { return f(*(ClientToServer_TransactionData*)this); }
        if (isCompleteTransaction()) { return f(*(ClientToServer_CompleteTransaction*)this); }
        if (isHeartbeat()) { return f(*(ClientToServer_Heartbeat*)this); }
        if (isDefineSchema()) { return f(*(ClientToServer_DefineSchema*)this); }
        if (isLoadLazyObject()) { return f(*(ClientToServer_LoadLazyObject*)this); }
        if (isSubscribe()) { return f(*(ClientToServer_Subscribe*)this); }
        if (isFlush()) { return f(*(ClientToServer_Flush*)this); }
        if (isAuthenticate()) { return f(*(ClientToServer_Authenticate*)this); }
    }

    bool isTransactionData() const { return which() == kind::TransactionData; }
    bool isCompleteTransaction() const { return which() == kind::CompleteTransaction; }
    bool isHeartbeat() const { return which() == kind::Heartbeat; }
    bool isDefineSchema() const { return which() == kind::DefineSchema; }
    bool isLoadLazyObject() const { return which() == kind::LoadLazyObject; }
    bool isSubscribe() const { return which() == kind::Subscribe; }
    bool isFlush() const { return which() == kind::Flush; }
    bool isAuthenticate() const { return which() == kind::Authenticate; }

    // Accessors for members
    ConstDict<ObjectFieldId, OneOf<None, Bytes>> writes() const;
    ConstDict<IndexId, TupleOf<int64_t>> set_adds() const;
    ConstDict<IndexId, TupleOf<int64_t>> set_removes() const;
    TupleOf<ObjectFieldId> key_versions() const;
    TupleOf<IndexId> index_versions() const;
    int64_t transaction_guid() const;
    int64_t as_of_version() const;
    String name() const;
    ConstDict<String, TypeDefinition> definition() const;
    String schema() const;
    OneOf<OneOf<None, String>,String> typename0() const;
    int64_t identity() const;
    OneOf<None, Anon39222656> fieldname_and_value() const;
    bool isLazy() const;
    int64_t guid() const;
    String token() const;

    Alternative::layout* getLayout() const { return mLayout; }
protected:
    Alternative::layout *mLayout;
};

template <>
class TypeDetails<ClientToServer*> {
public:
    static Forward* getType() {
        static Forward* t = new Forward(0, "ClientToServer");
        return t;
    }
    static const uint64_t bytecount = sizeof(void*);
};

NamedTuple* ClientToServer::TransactionData_Type = NamedTuple::Make(
    {TypeDetails<ConstDict<ObjectFieldId, OneOf<None, Bytes>>>::getType(), TypeDetails<ConstDict<IndexId, TupleOf<int64_t>>>::getType(), TypeDetails<ConstDict<IndexId, TupleOf<int64_t>>>::getType(), TypeDetails<TupleOf<ObjectFieldId>>::getType(), TypeDetails<TupleOf<IndexId>>::getType(), TypeDetails<int64_t>::getType()},
    {"writes", "set_adds", "set_removes", "key_versions", "index_versions", "transaction_guid"}
);

NamedTuple* ClientToServer::CompleteTransaction_Type = NamedTuple::Make(
    {TypeDetails<int64_t>::getType(), TypeDetails<int64_t>::getType()},
    {"as_of_version", "transaction_guid"}
);

NamedTuple* ClientToServer::Heartbeat_Type = NamedTuple::Make(
    {},
    {}
);

NamedTuple* ClientToServer::DefineSchema_Type = NamedTuple::Make(
    {TypeDetails<String>::getType(), TypeDetails<ConstDict<String, TypeDefinition>>::getType()},
    {"name", "definition"}
);

NamedTuple* ClientToServer::LoadLazyObject_Type = NamedTuple::Make(
    {TypeDetails<String>::getType(), TypeDetails<String>::getType(), TypeDetails<int64_t>::getType()},
    {"schema", "typename0", "identity"}
);

NamedTuple* ClientToServer::Subscribe_Type = NamedTuple::Make(
    {TypeDetails<String>::getType(), TypeDetails<OneOf<None, String>>::getType(), TypeDetails<OneOf<None, Anon39222656>>::getType(), TypeDetails<bool>::getType()},
    {"schema", "typename0", "fieldname_and_value", "isLazy"}
);

NamedTuple* ClientToServer::Flush_Type = NamedTuple::Make(
    {TypeDetails<int64_t>::getType()},
    {"guid"}
);

NamedTuple* ClientToServer::Authenticate_Type = NamedTuple::Make(
    {TypeDetails<String>::getType()},
    {"token"}
);

// static
Alternative* ClientToServer::getType() {
    static Alternative* t = Alternative::Make("ClientToServer", {
        {"TransactionData", TransactionData_Type},
        {"CompleteTransaction", CompleteTransaction_Type},
        {"Heartbeat", Heartbeat_Type},
        {"DefineSchema", DefineSchema_Type},
        {"LoadLazyObject", LoadLazyObject_Type},
        {"Subscribe", Subscribe_Type},
        {"Flush", Flush_Type},
        {"Authenticate", Authenticate_Type}
    }, {});
    static bool once = false;
    if (!once) {
        once = true;
        TypeDetails<ClientToServer*>::getType()->setTarget(t);
        t = (Alternative*)t->guaranteeForwardsResolved([](void* p) { return (Type*)0; });
    }
    return t;
}

template <>
class TypeDetails<ClientToServer> {
public:
    static Type* getType() {
        static Type* t = ClientToServer::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("ClientToServer somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = sizeof(void*);
};

class ClientToServer_TransactionData : public ClientToServer {
public:
    static ConcreteAlternative* getType() {
        static ConcreteAlternative* t = ConcreteAlternative::Make(ClientToServer::getType(), static_cast<int>(kind::TransactionData));
        return t;
    }
    static Alternative* getAlternative() { return ClientToServer::getType(); }

    ClientToServer_TransactionData():ClientToServer(kind::TransactionData) {}
    ClientToServer_TransactionData( const ConstDict<ObjectFieldId, OneOf<None, Bytes>>& writes1,  const ConstDict<IndexId, TupleOf<int64_t>>& set_adds1,  const ConstDict<IndexId, TupleOf<int64_t>>& set_removes1,  const TupleOf<ObjectFieldId>& key_versions1,  const TupleOf<IndexId>& index_versions1,  const int64_t& transaction_guid1):ClientToServer(kind::TransactionData) {
        writes() = writes1;
        set_adds() = set_adds1;
        set_removes() = set_removes1;
        key_versions() = key_versions1;
        index_versions() = index_versions1;
        transaction_guid() = transaction_guid1;
    }
    ClientToServer_TransactionData(const ClientToServer_TransactionData& other):ClientToServer(kind::TransactionData) {
        getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
    }
    ClientToServer_TransactionData& operator=(const ClientToServer_TransactionData& other) {
         getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
         return *this;
    }
    ~ClientToServer_TransactionData() {}

    ConstDict<ObjectFieldId, OneOf<None, Bytes>>& writes() const { return *(ConstDict<ObjectFieldId, OneOf<None, Bytes>>*)(mLayout->data); }
    ConstDict<IndexId, TupleOf<int64_t>>& set_adds() const { return *(ConstDict<IndexId, TupleOf<int64_t>>*)(mLayout->data + size1); }
    ConstDict<IndexId, TupleOf<int64_t>>& set_removes() const { return *(ConstDict<IndexId, TupleOf<int64_t>>*)(mLayout->data + size1 + size2); }
    TupleOf<ObjectFieldId>& key_versions() const { return *(TupleOf<ObjectFieldId>*)(mLayout->data + size1 + size2 + size3); }
    TupleOf<IndexId>& index_versions() const { return *(TupleOf<IndexId>*)(mLayout->data + size1 + size2 + size3 + size4); }
    int64_t& transaction_guid() const { return *(int64_t*)(mLayout->data + size1 + size2 + size3 + size4 + size5); }
private:
    static const int size1 = sizeof(ConstDict<ObjectFieldId, OneOf<None, Bytes>>);
    static const int size2 = sizeof(ConstDict<IndexId, TupleOf<int64_t>>);
    static const int size3 = sizeof(ConstDict<IndexId, TupleOf<int64_t>>);
    static const int size4 = sizeof(TupleOf<ObjectFieldId>);
    static const int size5 = sizeof(TupleOf<IndexId>);
};

ClientToServer ClientToServer::TransactionData(const ConstDict<ObjectFieldId, OneOf<None, Bytes>>& writes, const ConstDict<IndexId, TupleOf<int64_t>>& set_adds, const ConstDict<IndexId, TupleOf<int64_t>>& set_removes, const TupleOf<ObjectFieldId>& key_versions, const TupleOf<IndexId>& index_versions, const int64_t& transaction_guid) {
    return ClientToServer_TransactionData(writes, set_adds, set_removes, key_versions, index_versions, transaction_guid);
}

class ClientToServer_CompleteTransaction : public ClientToServer {
public:
    static ConcreteAlternative* getType() {
        static ConcreteAlternative* t = ConcreteAlternative::Make(ClientToServer::getType(), static_cast<int>(kind::CompleteTransaction));
        return t;
    }
    static Alternative* getAlternative() { return ClientToServer::getType(); }

    ClientToServer_CompleteTransaction():ClientToServer(kind::CompleteTransaction) {}
    ClientToServer_CompleteTransaction( const int64_t& as_of_version1,  const int64_t& transaction_guid1):ClientToServer(kind::CompleteTransaction) {
        as_of_version() = as_of_version1;
        transaction_guid() = transaction_guid1;
    }
    ClientToServer_CompleteTransaction(const ClientToServer_CompleteTransaction& other):ClientToServer(kind::CompleteTransaction) {
        getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
    }
    ClientToServer_CompleteTransaction& operator=(const ClientToServer_CompleteTransaction& other) {
         getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
         return *this;
    }
    ~ClientToServer_CompleteTransaction() {}

    int64_t& as_of_version() const { return *(int64_t*)(mLayout->data); }
    int64_t& transaction_guid() const { return *(int64_t*)(mLayout->data + size1); }
private:
    static const int size1 = sizeof(int64_t);
};

ClientToServer ClientToServer::CompleteTransaction(const int64_t& as_of_version, const int64_t& transaction_guid) {
    return ClientToServer_CompleteTransaction(as_of_version, transaction_guid);
}

class ClientToServer_Heartbeat : public ClientToServer {
public:
    static ConcreteAlternative* getType() {
        static ConcreteAlternative* t = ConcreteAlternative::Make(ClientToServer::getType(), static_cast<int>(kind::Heartbeat));
        return t;
    }
    static Alternative* getAlternative() { return ClientToServer::getType(); }

    ClientToServer_Heartbeat():ClientToServer(kind::Heartbeat) {}
    ClientToServer_Heartbeat(const ClientToServer_Heartbeat& other):ClientToServer(kind::Heartbeat) {
        getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
    }
    ClientToServer_Heartbeat& operator=(const ClientToServer_Heartbeat& other) {
         getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
         return *this;
    }
    ~ClientToServer_Heartbeat() {}

private:
};

ClientToServer ClientToServer::Heartbeat() {
    return ClientToServer_Heartbeat();
}

class ClientToServer_DefineSchema : public ClientToServer {
public:
    static ConcreteAlternative* getType() {
        static ConcreteAlternative* t = ConcreteAlternative::Make(ClientToServer::getType(), static_cast<int>(kind::DefineSchema));
        return t;
    }
    static Alternative* getAlternative() { return ClientToServer::getType(); }

    ClientToServer_DefineSchema():ClientToServer(kind::DefineSchema) {}
    ClientToServer_DefineSchema( const String& name1,  const ConstDict<String, TypeDefinition>& definition1):ClientToServer(kind::DefineSchema) {
        name() = name1;
        definition() = definition1;
    }
    ClientToServer_DefineSchema(const ClientToServer_DefineSchema& other):ClientToServer(kind::DefineSchema) {
        getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
    }
    ClientToServer_DefineSchema& operator=(const ClientToServer_DefineSchema& other) {
         getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
         return *this;
    }
    ~ClientToServer_DefineSchema() {}

    String& name() const { return *(String*)(mLayout->data); }
    ConstDict<String, TypeDefinition>& definition() const { return *(ConstDict<String, TypeDefinition>*)(mLayout->data + size1); }
private:
    static const int size1 = sizeof(String);
};

ClientToServer ClientToServer::DefineSchema(const String& name, const ConstDict<String, TypeDefinition>& definition) {
    return ClientToServer_DefineSchema(name, definition);
}

class ClientToServer_LoadLazyObject : public ClientToServer {
public:
    static ConcreteAlternative* getType() {
        static ConcreteAlternative* t = ConcreteAlternative::Make(ClientToServer::getType(), static_cast<int>(kind::LoadLazyObject));
        return t;
    }
    static Alternative* getAlternative() { return ClientToServer::getType(); }

    ClientToServer_LoadLazyObject():ClientToServer(kind::LoadLazyObject) {}
    ClientToServer_LoadLazyObject( const String& schema1,  const String& typename01,  const int64_t& identity1):ClientToServer(kind::LoadLazyObject) {
        schema() = schema1;
        typename0() = typename01;
        identity() = identity1;
    }
    ClientToServer_LoadLazyObject(const ClientToServer_LoadLazyObject& other):ClientToServer(kind::LoadLazyObject) {
        getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
    }
    ClientToServer_LoadLazyObject& operator=(const ClientToServer_LoadLazyObject& other) {
         getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
         return *this;
    }
    ~ClientToServer_LoadLazyObject() {}

    String& schema() const { return *(String*)(mLayout->data); }
    String& typename0() const { return *(String*)(mLayout->data + size1); }
    int64_t& identity() const { return *(int64_t*)(mLayout->data + size1 + size2); }
private:
    static const int size1 = sizeof(String);
    static const int size2 = sizeof(String);
};

ClientToServer ClientToServer::LoadLazyObject(const String& schema, const String& typename0, const int64_t& identity) {
    return ClientToServer_LoadLazyObject(schema, typename0, identity);
}

class ClientToServer_Subscribe : public ClientToServer {
public:
    static ConcreteAlternative* getType() {
        static ConcreteAlternative* t = ConcreteAlternative::Make(ClientToServer::getType(), static_cast<int>(kind::Subscribe));
        return t;
    }
    static Alternative* getAlternative() { return ClientToServer::getType(); }

    ClientToServer_Subscribe():ClientToServer(kind::Subscribe) {}
    ClientToServer_Subscribe( const String& schema1,  const OneOf<None, String>& typename01,  const OneOf<None, Anon39222656>& fieldname_and_value1,  const bool& isLazy1):ClientToServer(kind::Subscribe) {
        schema() = schema1;
        typename0() = typename01;
        fieldname_and_value() = fieldname_and_value1;
        isLazy() = isLazy1;
    }
    ClientToServer_Subscribe(const ClientToServer_Subscribe& other):ClientToServer(kind::Subscribe) {
        getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
    }
    ClientToServer_Subscribe& operator=(const ClientToServer_Subscribe& other) {
         getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
         return *this;
    }
    ~ClientToServer_Subscribe() {}

    String& schema() const { return *(String*)(mLayout->data); }
    OneOf<None, String>& typename0() const { return *(OneOf<None, String>*)(mLayout->data + size1); }
    OneOf<None, Anon39222656>& fieldname_and_value() const { return *(OneOf<None, Anon39222656>*)(mLayout->data + size1 + size2); }
    bool& isLazy() const { return *(bool*)(mLayout->data + size1 + size2 + size3); }
private:
    static const int size1 = sizeof(String);
    static const int size2 = sizeof(OneOf<None, String>);
    static const int size3 = sizeof(OneOf<None, Anon39222656>);
};

ClientToServer ClientToServer::Subscribe(const String& schema, const OneOf<None, String>& typename0, const OneOf<None, Anon39222656>& fieldname_and_value, const bool& isLazy) {
    return ClientToServer_Subscribe(schema, typename0, fieldname_and_value, isLazy);
}

class ClientToServer_Flush : public ClientToServer {
public:
    static ConcreteAlternative* getType() {
        static ConcreteAlternative* t = ConcreteAlternative::Make(ClientToServer::getType(), static_cast<int>(kind::Flush));
        return t;
    }
    static Alternative* getAlternative() { return ClientToServer::getType(); }

    ClientToServer_Flush():ClientToServer(kind::Flush) {}
    ClientToServer_Flush( const int64_t& guid1):ClientToServer(kind::Flush) {
        guid() = guid1;
    }
    ClientToServer_Flush(const ClientToServer_Flush& other):ClientToServer(kind::Flush) {
        getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
    }
    ClientToServer_Flush& operator=(const ClientToServer_Flush& other) {
         getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
         return *this;
    }
    ~ClientToServer_Flush() {}

    int64_t& guid() const { return *(int64_t*)(mLayout->data); }
private:
};

ClientToServer ClientToServer::Flush(const int64_t& guid) {
    return ClientToServer_Flush(guid);
}

class ClientToServer_Authenticate : public ClientToServer {
public:
    static ConcreteAlternative* getType() {
        static ConcreteAlternative* t = ConcreteAlternative::Make(ClientToServer::getType(), static_cast<int>(kind::Authenticate));
        return t;
    }
    static Alternative* getAlternative() { return ClientToServer::getType(); }

    ClientToServer_Authenticate():ClientToServer(kind::Authenticate) {}
    ClientToServer_Authenticate( const String& token1):ClientToServer(kind::Authenticate) {
        token() = token1;
    }
    ClientToServer_Authenticate(const ClientToServer_Authenticate& other):ClientToServer(kind::Authenticate) {
        getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
    }
    ClientToServer_Authenticate& operator=(const ClientToServer_Authenticate& other) {
         getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
         return *this;
    }
    ~ClientToServer_Authenticate() {}

    String& token() const { return *(String*)(mLayout->data); }
private:
};

ClientToServer ClientToServer::Authenticate(const String& token) {
    return ClientToServer_Authenticate(token);
}

ConstDict<ObjectFieldId, OneOf<None, Bytes>> ClientToServer::writes() const {
    if (isTransactionData())
        return ((ClientToServer_TransactionData*)this)->writes();
    throw std::runtime_error("\"ClientToServer\" subtype does not contain \"writes\"");
}

ConstDict<IndexId, TupleOf<int64_t>> ClientToServer::set_adds() const {
    if (isTransactionData())
        return ((ClientToServer_TransactionData*)this)->set_adds();
    throw std::runtime_error("\"ClientToServer\" subtype does not contain \"set_adds\"");
}

ConstDict<IndexId, TupleOf<int64_t>> ClientToServer::set_removes() const {
    if (isTransactionData())
        return ((ClientToServer_TransactionData*)this)->set_removes();
    throw std::runtime_error("\"ClientToServer\" subtype does not contain \"set_removes\"");
}

TupleOf<ObjectFieldId> ClientToServer::key_versions() const {
    if (isTransactionData())
        return ((ClientToServer_TransactionData*)this)->key_versions();
    throw std::runtime_error("\"ClientToServer\" subtype does not contain \"key_versions\"");
}

TupleOf<IndexId> ClientToServer::index_versions() const {
    if (isTransactionData())
        return ((ClientToServer_TransactionData*)this)->index_versions();
    throw std::runtime_error("\"ClientToServer\" subtype does not contain \"index_versions\"");
}

int64_t ClientToServer::transaction_guid() const {
    if (isTransactionData())
        return ((ClientToServer_TransactionData*)this)->transaction_guid();
    if (isCompleteTransaction())
        return ((ClientToServer_CompleteTransaction*)this)->transaction_guid();
    throw std::runtime_error("\"ClientToServer\" subtype does not contain \"transaction_guid\"");
}

int64_t ClientToServer::as_of_version() const {
    if (isCompleteTransaction())
        return ((ClientToServer_CompleteTransaction*)this)->as_of_version();
    throw std::runtime_error("\"ClientToServer\" subtype does not contain \"as_of_version\"");
}

String ClientToServer::name() const {
    if (isDefineSchema())
        return ((ClientToServer_DefineSchema*)this)->name();
    throw std::runtime_error("\"ClientToServer\" subtype does not contain \"name\"");
}

ConstDict<String, TypeDefinition> ClientToServer::definition() const {
    if (isDefineSchema())
        return ((ClientToServer_DefineSchema*)this)->definition();
    throw std::runtime_error("\"ClientToServer\" subtype does not contain \"definition\"");
}

String ClientToServer::schema() const {
    if (isLoadLazyObject())
        return ((ClientToServer_LoadLazyObject*)this)->schema();
    if (isSubscribe())
        return ((ClientToServer_Subscribe*)this)->schema();
    throw std::runtime_error("\"ClientToServer\" subtype does not contain \"schema\"");
}

OneOf<OneOf<None, String>,String> ClientToServer::typename0() const {
    if (isLoadLazyObject())
        return OneOf<OneOf<None, String>,String>(((ClientToServer_LoadLazyObject*)this)->typename0());
    if (isSubscribe())
        return OneOf<OneOf<None, String>,String>(((ClientToServer_Subscribe*)this)->typename0());
    throw std::runtime_error("\"ClientToServer\" subtype does not contain \"typename0\"");
}

int64_t ClientToServer::identity() const {
    if (isLoadLazyObject())
        return ((ClientToServer_LoadLazyObject*)this)->identity();
    throw std::runtime_error("\"ClientToServer\" subtype does not contain \"identity\"");
}

OneOf<None, Anon39222656> ClientToServer::fieldname_and_value() const {
    if (isSubscribe())
        return ((ClientToServer_Subscribe*)this)->fieldname_and_value();
    throw std::runtime_error("\"ClientToServer\" subtype does not contain \"fieldname_and_value\"");
}

bool ClientToServer::isLazy() const {
    if (isSubscribe())
        return ((ClientToServer_Subscribe*)this)->isLazy();
    throw std::runtime_error("\"ClientToServer\" subtype does not contain \"isLazy\"");
}

int64_t ClientToServer::guid() const {
    if (isFlush())
        return ((ClientToServer_Flush*)this)->guid();
    throw std::runtime_error("\"ClientToServer\" subtype does not contain \"guid\"");
}

String ClientToServer::token() const {
    if (isAuthenticate())
        return ((ClientToServer_Authenticate*)this)->token();
    throw std::runtime_error("\"ClientToServer\" subtype does not contain \"token\"");
}

// END Generated Alternative ClientToServer

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

    static Alternative* getType();
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
    Alternative::layout *mLayout;
};

template <>
class TypeDetails<A*> {
public:
    static Forward* getType() {
        static Forward* t = new Forward(0, "A");
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
    static bool once = false;
    if (!once) {
        once = true;
        TypeDetails<A*>::getType()->setTarget(t);
        t = (Alternative*)t->guaranteeForwardsResolved([](void* p) { return (Type*)0; });
    }
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

    static Alternative* getType();
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
    OneOf<TupleOf<String>,int64_t> c() const;

    Alternative::layout* getLayout() const { return mLayout; }
protected:
    Alternative::layout *mLayout;
};

template <>
class TypeDetails<Overlap*> {
public:
    static Forward* getType() {
        static Forward* t = new Forward(0, "Overlap");
        return t;
    }
    static const uint64_t bytecount = sizeof(void*);
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

// static
Alternative* Overlap::getType() {
    static Alternative* t = Alternative::Make("Overlap", {
        {"Sub1", Sub1_Type},
        {"Sub2", Sub2_Type},
        {"Sub3", Sub3_Type}
    }, {});
    static bool once = false;
    if (!once) {
        once = true;
        TypeDetails<Overlap*>::getType()->setTarget(t);
        t = (Alternative*)t->guaranteeForwardsResolved([](void* p) { return (Type*)0; });
    }
    return t;
}

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

OneOf<TupleOf<String>,int64_t> Overlap::c() const {
    if (isSub1())
        return OneOf<TupleOf<String>,int64_t>(((Overlap_Sub1*)this)->c());
    if (isSub2())
        return OneOf<TupleOf<String>,int64_t>(((Overlap_Sub2*)this)->c());
    throw std::runtime_error("\"Overlap\" subtype does not contain \"c\"");
}

// END Generated Alternative Overlap

// Generated Alternative Bexpress=
//     Leaf=(value=bool)
//     BinOp=(left=Bexpress, op=String, right=Bexpress)
//     UnaryOp=(op=String, right=Bexpress)

class Bexpress_Leaf;
class Bexpress_BinOp;
class Bexpress_UnaryOp;

class Bexpress {
public:
    enum class kind { Leaf=0, BinOp=1, UnaryOp=2 };

    static NamedTuple* Leaf_Type;
    static NamedTuple* BinOp_Type;
    static NamedTuple* UnaryOp_Type;

    static Alternative* getType();
    ~Bexpress() { getType()->destroy((instance_ptr)&mLayout); }
    Bexpress():mLayout(0) { getType()->constructor((instance_ptr)&mLayout); }
    Bexpress(kind k):mLayout(0) { ConcreteAlternative::Make(getType(), (int64_t)k)->constructor((instance_ptr)&mLayout); }
    Bexpress(const Bexpress& in) { getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&in.mLayout); }
    Bexpress& operator=(const Bexpress& other) { getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout); return *this; }

    static Bexpress Leaf(const bool& value);
    static Bexpress BinOp(const Bexpress& left, const String& op, const Bexpress& right);
    static Bexpress UnaryOp(const String& op, const Bexpress& right);

    kind which() const { return (kind)mLayout->which; }

    template <class F>
    auto check(const F& f) {
        if (isLeaf()) { return f(*(Bexpress_Leaf*)this); }
        if (isBinOp()) { return f(*(Bexpress_BinOp*)this); }
        if (isUnaryOp()) { return f(*(Bexpress_UnaryOp*)this); }
    }

    bool isLeaf() const { return which() == kind::Leaf; }
    bool isBinOp() const { return which() == kind::BinOp; }
    bool isUnaryOp() const { return which() == kind::UnaryOp; }

    // Accessors for members
    bool value() const;
    Bexpress left() const;
    String op() const;
    Bexpress right() const;

    Alternative::layout* getLayout() const { return mLayout; }
protected:
    Alternative::layout *mLayout;
};

template <>
class TypeDetails<Bexpress*> {
public:
    static Forward* getType() {
        static Forward* t = new Forward(0, "Bexpress");
        return t;
    }
    static const uint64_t bytecount = sizeof(void*);
};

NamedTuple* Bexpress::Leaf_Type = NamedTuple::Make(
    {TypeDetails<bool>::getType()},
    {"value"}
);

NamedTuple* Bexpress::BinOp_Type = NamedTuple::Make(
    {TypeDetails<Bexpress*>::getType(), TypeDetails<String>::getType(), TypeDetails<Bexpress*>::getType()},
    {"left", "op", "right"}
);

NamedTuple* Bexpress::UnaryOp_Type = NamedTuple::Make(
    {TypeDetails<String>::getType(), TypeDetails<Bexpress*>::getType()},
    {"op", "right"}
);

// static
Alternative* Bexpress::getType() {
    static Alternative* t = Alternative::Make("Bexpress", {
        {"Leaf", Leaf_Type},
        {"BinOp", BinOp_Type},
        {"UnaryOp", UnaryOp_Type}
    }, {});
    static bool once = false;
    if (!once) {
        once = true;
        TypeDetails<Bexpress*>::getType()->setTarget(t);
        t = (Alternative*)t->guaranteeForwardsResolved([](void* p) { return (Type*)0; });
    }
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

class Bexpress_Leaf : public Bexpress {
public:
    static ConcreteAlternative* getType() {
        static ConcreteAlternative* t = ConcreteAlternative::Make(Bexpress::getType(), static_cast<int>(kind::Leaf));
        return t;
    }
    static Alternative* getAlternative() { return Bexpress::getType(); }

    Bexpress_Leaf():Bexpress(kind::Leaf) {}
    Bexpress_Leaf( const bool& value1):Bexpress(kind::Leaf) {
        value() = value1;
    }
    Bexpress_Leaf(const Bexpress_Leaf& other):Bexpress(kind::Leaf) {
        getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
    }
    Bexpress_Leaf& operator=(const Bexpress_Leaf& other) {
         getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
         return *this;
    }
    ~Bexpress_Leaf() {}

    bool& value() const { return *(bool*)(mLayout->data); }
private:
};

Bexpress Bexpress::Leaf(const bool& value) {
    return Bexpress_Leaf(value);
}

class Bexpress_BinOp : public Bexpress {
public:
    static ConcreteAlternative* getType() {
        static ConcreteAlternative* t = ConcreteAlternative::Make(Bexpress::getType(), static_cast<int>(kind::BinOp));
        return t;
    }
    static Alternative* getAlternative() { return Bexpress::getType(); }

    Bexpress_BinOp():Bexpress(kind::BinOp) {}
    Bexpress_BinOp( const Bexpress& left1,  const String& op1,  const Bexpress& right1):Bexpress(kind::BinOp) {
        left() = left1;
        op() = op1;
        right() = right1;
    }
    Bexpress_BinOp(const Bexpress_BinOp& other):Bexpress(kind::BinOp) {
        getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
    }
    Bexpress_BinOp& operator=(const Bexpress_BinOp& other) {
         getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
         return *this;
    }
    ~Bexpress_BinOp() {}

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
        static ConcreteAlternative* t = ConcreteAlternative::Make(Bexpress::getType(), static_cast<int>(kind::UnaryOp));
        return t;
    }
    static Alternative* getAlternative() { return Bexpress::getType(); }

    Bexpress_UnaryOp():Bexpress(kind::UnaryOp) {}
    Bexpress_UnaryOp( const String& op1,  const Bexpress& right1):Bexpress(kind::UnaryOp) {
        op() = op1;
        right() = right1;
    }
    Bexpress_UnaryOp(const Bexpress_UnaryOp& other):Bexpress(kind::UnaryOp) {
        getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
    }
    Bexpress_UnaryOp& operator=(const Bexpress_UnaryOp& other) {
         getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout);
         return *this;
    }
    ~Bexpress_UnaryOp() {}

    String& op() const { return *(String*)(mLayout->data); }
    Bexpress& right() const { return *(Bexpress*)(mLayout->data + size1); }
private:
    static const int size1 = sizeof(String);
};

Bexpress Bexpress::UnaryOp(const String& op, const Bexpress& right) {
    return Bexpress_UnaryOp(op, right);
}

bool Bexpress::value() const {
    if (isLeaf())
        return ((Bexpress_Leaf*)this)->value();
    throw std::runtime_error("\"Bexpress\" subtype does not contain \"value\"");
}

Bexpress Bexpress::left() const {
    if (isBinOp())
        return ((Bexpress_BinOp*)this)->left();
    throw std::runtime_error("\"Bexpress\" subtype does not contain \"left\"");
}

String Bexpress::op() const {
    if (isBinOp())
        return ((Bexpress_BinOp*)this)->op();
    if (isUnaryOp())
        return ((Bexpress_UnaryOp*)this)->op();
    throw std::runtime_error("\"Bexpress\" subtype does not contain \"op\"");
}

Bexpress Bexpress::right() const {
    if (isBinOp())
        return ((Bexpress_BinOp*)this)->right();
    if (isUnaryOp())
        return ((Bexpress_UnaryOp*)this)->right();
    throw std::runtime_error("\"Bexpress\" subtype does not contain \"right\"");
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

// Generated Tuple AnonTest
//    a0=Dict<Anon40676176, String>
//    a1=ConstDict<String, OneOf<bool, Anon40683760>>
//    a2=ListOf<Anon40676176>
//    a3=TupleOf<Anon40698976>
class AnonTest {
public:
    typedef Dict<Anon40676176, String> a0_type;
    typedef ConstDict<String, OneOf<bool, Anon40683760>> a1_type;
    typedef ListOf<Anon40676176> a2_type;
    typedef TupleOf<Anon40698976> a3_type;
    a0_type& a0() const { return *(a0_type*)(data); }
    a1_type& a1() const { return *(a1_type*)(data + size1); }
    a2_type& a2() const { return *(a2_type*)(data + size1 + size2); }
    a3_type& a3() const { return *(a3_type*)(data + size1 + size2 + size3); }
private:
    static const int size1 = sizeof(a0_type);
    static const int size2 = sizeof(a1_type);
    static const int size3 = sizeof(a2_type);
    static const int size4 = sizeof(a3_type);
    uint8_t data[size1 + size2 + size3 + size4];
public:
    static Tuple* getType() {
        static Tuple* t = Tuple::Make({
                TypeDetails<AnonTest::a0_type>::getType(),
                TypeDetails<AnonTest::a1_type>::getType(),
                TypeDetails<AnonTest::a2_type>::getType(),
                TypeDetails<AnonTest::a3_type>::getType()
            });
        return t;
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

// END Generated NamedTuple AnonTest

