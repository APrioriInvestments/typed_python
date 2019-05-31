#pragma once
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

    static ObjectFieldId fromPython(PyObject* p) {
        ObjectFieldId l;
        PyInstance::copyConstructFromPythonInstance(getType(), (instance_ptr)&l, p, true);
        return l;
    }

    PyObject* toPython() {
        return PyInstance::extractPythonObject((instance_ptr)this, getType());
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
private:
    static const int size1 = sizeof(objId_type);
    static const int size2 = sizeof(fieldId_type);
    static const int size3 = sizeof(isIndexValue_type);
    uint8_t data[size1 + size2 + size3];
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

    static IndexId fromPython(PyObject* p) {
        IndexId l;
        PyInstance::copyConstructFromPythonInstance(getType(), (instance_ptr)&l, p, true);
        return l;
    }

    PyObject* toPython() {
        return PyInstance::extractPythonObject((instance_ptr)this, getType());
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
private:
    static const int size1 = sizeof(fieldId_type);
    static const int size2 = sizeof(indexValue_type);
    uint8_t data[size1 + size2];
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

// Generated NamedTuple ValueType
//    fields=TupleOf<String>
//    indices=TupleOf<String>
class ValueType {
public:
    typedef TupleOf<String> fields_type;
    typedef TupleOf<String> indices_type;
    fields_type& fields() const { return *(fields_type*)(data); }
    indices_type& indices() const { return *(indices_type*)(data + size1); }
    static NamedTuple* getType() {
        static NamedTuple* t = NamedTuple::Make({
                TypeDetails<ValueType::fields_type>::getType(),
                TypeDetails<ValueType::indices_type>::getType()
            },{
                "fields",
                "indices"
            });
        return t;
        }

    static ValueType fromPython(PyObject* p) {
        ValueType l;
        PyInstance::copyConstructFromPythonInstance(getType(), (instance_ptr)&l, p, true);
        return l;
    }

    PyObject* toPython() {
        return PyInstance::extractPythonObject((instance_ptr)this, getType());
    }

    ValueType& operator = (const ValueType& other) {
        fields() = other.fields();
        indices() = other.indices();
        return *this;
    }

    ValueType(const ValueType& other) {
        new (&fields()) fields_type(other.fields());
        new (&indices()) indices_type(other.indices());
    }

    ~ValueType() {
        indices().~indices_type();
        fields().~fields_type();
    }

    ValueType() {
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

    ValueType(const fields_type& fields_val, const indices_type& indices_val) {
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
private:
    static const int size1 = sizeof(fields_type);
    static const int size2 = sizeof(indices_type);
    uint8_t data[size1 + size2];
};

template <>
class TypeDetails<ValueType> {
public:
    static Type* getType() {
        static Type* t = ValueType::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("ValueType somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = 
        sizeof(ValueType::fields_type) +
        sizeof(ValueType::indices_type);
};

// END Generated NamedTuple ValueType

// Generated Tuple Anon41789424
//    a0=String
//    a1=Bytes
class Anon41789424 {
public:
    typedef String a0_type;
    typedef Bytes a1_type;
    a0_type& a0() const { return *(a0_type*)(data); }
    a1_type& a1() const { return *(a1_type*)(data + size1); }
    static Tuple* getType() {
        static Tuple* t = Tuple::Make({
                TypeDetails<Anon41789424::a0_type>::getType(),
                TypeDetails<Anon41789424::a1_type>::getType()
            });
        return t;
        }

    static Anon41789424 fromPython(PyObject* p) {
        Anon41789424 l;
        PyInstance::copyConstructFromPythonInstance(getType(), (instance_ptr)&l, p, true);
        return l;
    }

    PyObject* toPython() {
        return PyInstance::extractPythonObject((instance_ptr)this, getType());
    }

    Anon41789424& operator = (const Anon41789424& other) {
        a0() = other.a0();
        a1() = other.a1();
        return *this;
    }

    Anon41789424(const Anon41789424& other) {
        new (&a0()) a0_type(other.a0());
        new (&a1()) a1_type(other.a1());
    }

    ~Anon41789424() {
        a1().~a1_type();
        a0().~a0_type();
    }

    Anon41789424() {
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

    Anon41789424(const a0_type& a0_val, const a1_type& a1_val) {
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
class TypeDetails<Anon41789424> {
public:
    static Type* getType() {
        static Type* t = Anon41789424::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("Anon41789424 somehow we have the wrong bytecount!");
        }
        return t;
    }
    static const uint64_t bytecount = 
        sizeof(Anon41789424::a0_type) +
        sizeof(Anon41789424::a1_type);
};

// END Generated Tuple Anon41789424

// Generated Alternative ClientToServer=
//     TransactionData=(writes=ConstDict<ObjectFieldId, OneOf<None, Bytes>>, set_adds=ConstDict<IndexId, TupleOf<int64_t>>, set_removes=ConstDict<IndexId, TupleOf<int64_t>>, key_versions=TupleOf<ObjectFieldId>, index_versions=TupleOf<IndexId>, transaction_guid=int64_t)
//     CompleteTransaction=(as_of_version=int64_t, transaction_guid=int64_t)
//     Heartbeat=()
//     DefineSchema=(name=String, definition=ConstDict<String, ValueType>)
//     LoadLazyObject=(schema=String, typename0=String, identity=int64_t)
//     Subscribe=(schema=String, typename0=OneOf<None, String>, fieldname_and_value=OneOf<None, Anon41789424>, isLazy=bool)
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

    static Alternative* getType() {
        PyObject* resolver = getOrSetTypeResolver();
        if (!resolver)
            throw std::runtime_error("{name}: no resolver");
        PyObject* res = PyObject_CallMethod(resolver, "resolveTypeByName", "s", "object_database.database_connection.ClientToServer");
        if (!res)
            throw std::runtime_error("ClientToServer: did not resolve");
        return (Alternative*)PyInstance::unwrapTypeArgToTypePtr(res);
    }
    static ClientToServer fromPython(PyObject* p) {
        Alternative::layout* l = nullptr;
        PyInstance::copyConstructFromPythonInstance(getType(), (instance_ptr)&l, p, true);
        return ClientToServer(l);
    }

    PyObject* toPython() {
        return PyInstance::extractPythonObject((instance_ptr)&mLayout, getType());
    }

    ~ClientToServer() { getType()->destroy((instance_ptr)&mLayout); }
    ClientToServer():mLayout(0) { getType()->constructor((instance_ptr)&mLayout); }
    ClientToServer(kind k):mLayout(0) { ConcreteAlternative::Make(getType(), (int64_t)k)->constructor((instance_ptr)&mLayout); }
    ClientToServer(const ClientToServer& in) { getType()->copy_constructor((instance_ptr)&mLayout, (instance_ptr)&in.mLayout); }
    ClientToServer& operator=(const ClientToServer& other) { getType()->assign((instance_ptr)&mLayout, (instance_ptr)&other.mLayout); return *this; }

    static ClientToServer TransactionData(const ConstDict<ObjectFieldId, OneOf<None, Bytes>>& writes, const ConstDict<IndexId, TupleOf<int64_t>>& set_adds, const ConstDict<IndexId, TupleOf<int64_t>>& set_removes, const TupleOf<ObjectFieldId>& key_versions, const TupleOf<IndexId>& index_versions, const int64_t& transaction_guid);
    static ClientToServer CompleteTransaction(const int64_t& as_of_version, const int64_t& transaction_guid);
    static ClientToServer Heartbeat();
    static ClientToServer DefineSchema(const String& name, const ConstDict<String, ValueType>& definition);
    static ClientToServer LoadLazyObject(const String& schema, const String& typename0, const int64_t& identity);
    static ClientToServer Subscribe(const String& schema, const OneOf<None, String>& typename0, const OneOf<None, Anon41789424>& fieldname_and_value, const bool& isLazy);
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
    ConstDict<String, ValueType> definition() const;
    String schema() const;
    OneOf<String,OneOf<None, String>> typename0() const;
    int64_t identity() const;
    OneOf<None, Anon41789424> fieldname_and_value() const;
    bool isLazy() const;
    int64_t guid() const;
    String token() const;

    Alternative::layout* getLayout() const { return mLayout; }
protected:
    explicit ClientToServer(Alternative::layout* l): mLayout(l) {}
    Alternative::layout *mLayout;
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
    {TypeDetails<String>::getType(), TypeDetails<ConstDict<String, ValueType>>::getType()},
    {"name", "definition"}
);

NamedTuple* ClientToServer::LoadLazyObject_Type = NamedTuple::Make(
    {TypeDetails<String>::getType(), TypeDetails<String>::getType(), TypeDetails<int64_t>::getType()},
    {"schema", "typename0", "identity"}
);

NamedTuple* ClientToServer::Subscribe_Type = NamedTuple::Make(
    {TypeDetails<String>::getType(), TypeDetails<OneOf<None, String>>::getType(), TypeDetails<OneOf<None, Anon41789424>>::getType(), TypeDetails<bool>::getType()},
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
    ClientToServer_DefineSchema( const String& name1,  const ConstDict<String, ValueType>& definition1):ClientToServer(kind::DefineSchema) {
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
    ConstDict<String, ValueType>& definition() const { return *(ConstDict<String, ValueType>*)(mLayout->data + size1); }
private:
    static const int size1 = sizeof(String);
};

ClientToServer ClientToServer::DefineSchema(const String& name, const ConstDict<String, ValueType>& definition) {
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
    ClientToServer_Subscribe( const String& schema1,  const OneOf<None, String>& typename01,  const OneOf<None, Anon41789424>& fieldname_and_value1,  const bool& isLazy1):ClientToServer(kind::Subscribe) {
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
    OneOf<None, Anon41789424>& fieldname_and_value() const { return *(OneOf<None, Anon41789424>*)(mLayout->data + size1 + size2); }
    bool& isLazy() const { return *(bool*)(mLayout->data + size1 + size2 + size3); }
private:
    static const int size1 = sizeof(String);
    static const int size2 = sizeof(OneOf<None, String>);
    static const int size3 = sizeof(OneOf<None, Anon41789424>);
};

ClientToServer ClientToServer::Subscribe(const String& schema, const OneOf<None, String>& typename0, const OneOf<None, Anon41789424>& fieldname_and_value, const bool& isLazy) {
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

ConstDict<String, ValueType> ClientToServer::definition() const {
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

OneOf<String,OneOf<None, String>> ClientToServer::typename0() const {
    if (isLoadLazyObject())
        return OneOf<String,OneOf<None, String>>(((ClientToServer_LoadLazyObject*)this)->typename0());
    if (isSubscribe())
        return OneOf<String,OneOf<None, String>>(((ClientToServer_Subscribe*)this)->typename0());
    throw std::runtime_error("\"ClientToServer\" subtype does not contain \"typename0\"");
}

int64_t ClientToServer::identity() const {
    if (isLoadLazyObject())
        return ((ClientToServer_LoadLazyObject*)this)->identity();
    throw std::runtime_error("\"ClientToServer\" subtype does not contain \"identity\"");
}

OneOf<None, Anon41789424> ClientToServer::fieldname_and_value() const {
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

