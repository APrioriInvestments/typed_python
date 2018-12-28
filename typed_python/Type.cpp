#include "Type.hpp"

void Type::repr(instance_ptr self, std::ostringstream& out) {
    assertForwardsResolved();

    this->check([&](auto& subtype) {
        subtype.repr(self, out);
    });
}

char Type::cmp(instance_ptr left, instance_ptr right) {
    assertForwardsResolved();

    return this->check([&](auto& subtype) {
        return subtype.cmp(left, right);
    });
}

int32_t Type::hash32(instance_ptr left) {
    assertForwardsResolved();

    return this->check([&](auto& subtype) {
        return subtype.hash32(left);
    });
}

void Type::swap(instance_ptr left, instance_ptr right) {
    assertForwardsResolved();

    if (left == right) {
        return;
    }

    size_t remaining = m_size;
    while (remaining >= 8) {
        int64_t temp = *(int64_t*)left;
        *(int64_t*)left = *(int64_t*)right;
        *(int64_t*)right = temp;

        remaining -= 8;
        left += 8;
        right += 8;
    }

    while (remaining > 0) {
        int8_t temp = *(int8_t*)left;
        *(int8_t*)left = *(int8_t*)right;
        *(int8_t*)right = temp;

        remaining -= 1;
        left += 1;
        right += 1;
    }
}

// static
char Type::byteCompare(uint8_t* l, uint8_t* r, size_t count) {
    while (count >= 8 && *(uint64_t*)l == *(uint64_t*)r) {
        l += 8;
        r += 8;
        count -= 8;
    }

    for (long k = 0; k < count; k++) {
        if (l[k] < r[k]) {
            return -1;
        }
        if (l[k] > r[k]) {
            return 1;
        }
    }
    return 0;
}

void Type::constructor(instance_ptr self) {
    assertForwardsResolved();

    this->check([&](auto& subtype) { subtype.constructor(self); } );
}

void Type::destroy(instance_ptr self) {
    assertForwardsResolved();

    this->check([&](auto& subtype) { subtype.destroy(self); } );
}

void Type::forwardTypesMayHaveChanged() {
    m_references_unresolved_forwards = false;

    visitReferencedTypes([&](Type* t) {
        if (t->references_unresolved_forwards()) {
            m_references_unresolved_forwards = true;
        }
    });

    this->check([&](auto& subtype) {
        subtype._forwardTypesMayHaveChanged();
    });

    if (mTypeRep) {
        updateTypeRepForType(this, mTypeRep);
    }
}

void Type::copy_constructor(instance_ptr self, instance_ptr other) {
    assertForwardsResolved();

    this->check([&](auto& subtype) { subtype.copy_constructor(self, other); } );
}

void Type::assign(instance_ptr self, instance_ptr other) {
    assertForwardsResolved();

    this->check([&](auto& subtype) { subtype.assign(self, other); } );
}

bool Type::isBinaryCompatibleWith(Type* other) {
    if (other == this) {
        return true;
    }

    while (other->getTypeCategory() == TypeCategory::catPythonSubclass) {
        other = other->getBaseType();
    }

    auto it = mIsBinaryCompatible.find(other);
    if (it != mIsBinaryCompatible.end()) {
        return it->second != BinaryCompatibilityCategory::Incompatible;
    }

    //mark that we are recursing through this datastructure. we don't want to
    //loop indefinitely.
    mIsBinaryCompatible[other] = BinaryCompatibilityCategory::Checking;

    bool isCompatible = this->check([&](auto& subtype) {
        return subtype.isBinaryCompatibleWithConcrete(other);
    });

    mIsBinaryCompatible[other] = isCompatible ?
        BinaryCompatibilityCategory::Compatible :
        BinaryCompatibilityCategory::Incompatible
        ;

    return isCompatible;
}

bool OneOf::isBinaryCompatibleWithConcrete(Type* other) {
    if (other->getTypeCategory() != TypeCategory::catOneOf) {
        return false;
    }

    OneOf* otherO = (OneOf*)other;

    if (m_types.size() != otherO->m_types.size()) {
        return false;
    }

    for (long k = 0; k < m_types.size(); k++) {
        if (!m_types[k]->isBinaryCompatibleWith(otherO->m_types[k])) {
            return false;
        }
    }

    return true;
}

void OneOf::_forwardTypesMayHaveChanged() {
    m_size = computeBytecount();
    m_name = computeName();

    m_is_default_constructible = false;

    for (auto typePtr: m_types) {
        if (typePtr->is_default_constructible()) {
            m_is_default_constructible = true;
            break;
        }
    }
}

std::string OneOf::computeName() const {
    std::string res = "OneOf(";
    bool first = true;
    for (auto t: m_types) {
        if (first) {
            first = false;
        } else {
            res += ", ";
        }

        res += t->name();
    }

    res += ")";

    return res;
}

void OneOf::repr(instance_ptr self, std::ostringstream& stream) {
    m_types[*((uint8_t*)self)]->repr(self+1, stream);
}

int32_t OneOf::hash32(instance_ptr left) {
    Hash32Accumulator acc((int)getTypeCategory());

    acc.add(*(uint8_t*)left);
    acc.add(m_types[*((uint8_t*)left)]->hash32(left+1));

    return acc.get();
}

char OneOf::cmp(instance_ptr left, instance_ptr right) {
    if (((uint8_t*)left)[0] < ((uint8_t*)right)[0]) {
        return -1;
    }
    if (((uint8_t*)left)[0] > ((uint8_t*)right)[0]) {
        return 1;
    }

    return m_types[*((uint8_t*)left)]->cmp(left+1,right+1);
}

size_t OneOf::computeBytecount() const {
    size_t res = 0;

    for (auto t: m_types)
        res = std::max(res, t->bytecount());

    return res + 1;
}

void OneOf::constructor(instance_ptr self) {
    if (!m_is_default_constructible) {
        throw std::runtime_error(m_name + " is not default-constructible");
    }

    for (size_t k = 0; k < m_types.size(); k++) {
        if (m_types[k]->is_default_constructible()) {
            *(uint8_t*)self = k;
            m_types[k]->constructor(self+1);
            return;
        }
    }
}

void OneOf::destroy(instance_ptr self) {
    uint8_t which = *(uint8_t*)(self);
    m_types[which]->destroy(self+1);
}

void OneOf::copy_constructor(instance_ptr self, instance_ptr other) {
    uint8_t which = *(uint8_t*)self = *(uint8_t*)other;
    m_types[which]->copy_constructor(self+1, other+1);
}

void OneOf::assign(instance_ptr self, instance_ptr other) {
    uint8_t which = *(uint8_t*)self;
    if (which == *(uint8_t*)other) {
        m_types[which]->assign(self+1,other+1);
    } else {
        m_types[which]->destroy(self+1);

        uint8_t otherWhich = *(uint8_t*)other;
        *(uint8_t*)self = otherWhich;
        m_types[otherWhich]->copy_constructor(self+1,other+1);
    }
}

// static
OneOf* OneOf::Make(const std::vector<Type*>& types) {
    std::vector<Type*> flat_typelist;
    std::set<Type*> seen;

    //make sure we only get each type once and don't have any other 'OneOf' in there...
    std::function<void (const std::vector<Type*>)> visit = [&](const std::vector<Type*>& subvec) {
        for (auto t: subvec) {
            if (t->getTypeCategory() == catOneOf) {
                visit( ((OneOf*)t)->getTypes() );
            } else if (seen.find(t) == seen.end()) {
                flat_typelist.push_back(t);
                seen.insert(t);
            }
        }
    };

    visit(types);

    static std::mutex guard;

    std::lock_guard<std::mutex> lock(guard);

    typedef const std::vector<Type*> keytype;

    static std::map<keytype, OneOf*> m;

    auto it = m.find(flat_typelist);
    if (it == m.end()) {
        it = m.insert(std::make_pair(flat_typelist, new OneOf(flat_typelist))).first;
    }

    return it->second;
}

bool CompositeType::isBinaryCompatibleWithConcrete(Type* other) {
    if (other->getTypeCategory() != m_typeCategory) {
        return false;
    }

    CompositeType* otherO = (CompositeType*)other;

    if (m_types.size() != otherO->m_types.size()) {
        return false;
    }

    for (long k = 0; k < m_types.size(); k++) {
        if (!m_types[k]->isBinaryCompatibleWith(otherO->m_types[k])) {
            return false;
        }
    }

    return true;
}

void CompositeType::_forwardTypesMayHaveChanged() {
    m_is_default_constructible = true;
    m_size = 0;
    m_byte_offsets.clear();

    for (auto t: m_types) {
        m_byte_offsets.push_back(m_size);
        m_size += t->bytecount();
    }

    for (auto t: m_types) {
        if (!t->is_default_constructible()) {
            m_is_default_constructible = false;
        }
    }
}

char CompositeType::cmp(instance_ptr left, instance_ptr right) {
    for (long k = 0; k < m_types.size(); k++) {
        char res = m_types[k]->cmp(left + m_byte_offsets[k], right + m_byte_offsets[k]);
        if (res != 0) {
            return res;
        }
    }

    return 0;
}

void CompositeType::repr(instance_ptr self, std::ostringstream& stream) {
    stream << "(";

    for (long k = 0; k < getTypes().size();k++) {
        if (k > 0) {
            stream << ", ";
        }

        if (k < m_names.size()) {
            stream << m_names[k] << "=";
        }

        getTypes()[k]->repr(eltPtr(self,k),stream);
    }
    if (getTypes().size() == 1) {
        stream << ",";
    }

    stream << ")";
}

int32_t CompositeType::hash32(instance_ptr left) {
    Hash32Accumulator acc((int)getTypeCategory());

    for (long k = 0; k < getTypes().size();k++) {
        acc.add(getTypes()[k]->hash32(eltPtr(left,k)));
    }

    acc.add(getTypes().size());

    return acc.get();
}

void CompositeType::constructor(instance_ptr self) {
    if (!m_is_default_constructible) {
        throw std::runtime_error(m_name + " is not default-constructible");
    }

    for (size_t k = 0; k < m_types.size(); k++) {
        m_types[k]->constructor(self+m_byte_offsets[k]);
    }
}

void CompositeType::destroy(instance_ptr self) {
    for (long k = (long)m_types.size() - 1; k >= 0; k--) {
        m_types[k]->destroy(self+m_byte_offsets[k]);
    }
}

void CompositeType::copy_constructor(instance_ptr self, instance_ptr other) {
    for (long k = (long)m_types.size() - 1; k >= 0; k--) {
        m_types[k]->copy_constructor(self + m_byte_offsets[k], other+m_byte_offsets[k]);
    }
}

void CompositeType::assign(instance_ptr self, instance_ptr other) {
    for (long k = (long)m_types.size() - 1; k >= 0; k--) {
        m_types[k]->assign(self + m_byte_offsets[k], other+m_byte_offsets[k]);
    }
}

void NamedTuple::_forwardTypesMayHaveChanged() {
    ((CompositeType*)this)->_forwardTypesMayHaveChanged();

    std::string oldName = m_name;

    m_name = "NamedTuple(";
    for (long k = 0; k < m_types.size();k++) {
        if (k) {
            m_name += ", ";
        }
        m_name += m_names[k] + "=" + m_types[k]->name();
    }
    m_name += ")";
}

void Tuple::_forwardTypesMayHaveChanged() {
    ((CompositeType*)this)->_forwardTypesMayHaveChanged();

    m_name = "Tuple(";
    for (long k = 0; k < m_types.size();k++) {
        if (k) {
            m_name += ", ";
        }
        m_name += m_types[k]->name();
    }
    m_name += ")";
}

bool TupleOf::isBinaryCompatibleWithConcrete(Type* other) {
    if (other->getTypeCategory() != m_typeCategory) {
        return false;
    }

    TupleOf* otherO = (TupleOf*)other;

    return m_element_type->isBinaryCompatibleWith(otherO->m_element_type);
}

void TupleOf::repr(instance_ptr self, std::ostringstream& stream) {
    stream << "(";

    int32_t ct = count(self);

    for (long k = 0; k < ct;k++) {
        if (k > 0) {
            stream << ", ";
        }

        m_element_type->repr(eltPtr(self,k),stream);
    }

    stream << ")";
}

int32_t TupleOf::hash32(instance_ptr left) {
    if (!(*(layout**)left)) {
        return 0x123;
    }

    if ((*(layout**)left)->hash_cache == -1) {
        Hash32Accumulator acc((int)getTypeCategory());

        int32_t ct = count(left);
        acc.add(ct);

        for (long k = 0; k < ct;k++) {
            acc.add(m_element_type->hash32(eltPtr(left, k)));
        }

        (*(layout**)left)->hash_cache = acc.get();
        if ((*(layout**)left)->hash_cache == -1) {
            (*(layout**)left)->hash_cache = -2;
        }
    }

    return (*(layout**)left)->hash_cache;
}

char TupleOf::cmp(instance_ptr left, instance_ptr right) {
    if (!(*(layout**)left) && (*(layout**)right)) {
        return -1;
    }
    if (!(*(layout**)right) && (*(layout**)left)) {
        return 1;
    }
    if (!(*(layout**)right) && !(*(layout**)left)) {
        return 0;
    }
    layout& left_layout = **(layout**)left;
    layout& right_layout = **(layout**)right;

    if (&left_layout == &right_layout) {
        return 0;
    }

    size_t bytesPer = m_element_type->bytecount();

    for (long k = 0; k < left_layout.count && k < right_layout.count; k++) {
        char res = m_element_type->cmp(left_layout.data + bytesPer * k,
                                       right_layout.data + bytesPer * k);

        if (res != 0) {
            return res;
        }
    }

    if (left_layout.count < right_layout.count) {
        return -1;
    }

    if (left_layout.count > right_layout.count) {
        return 1;
    }

    return 0;
}

// static
TupleOf* TupleOf::Make(Type* elt) {
    static std::mutex guard;

    std::lock_guard<std::mutex> lock(guard);

    static std::map<Type*, TupleOf*> m;

    auto it = m.find(elt);
    if (it == m.end()) {
        it = m.insert(std::make_pair(elt, new TupleOf(elt))).first;
    }

    return it->second;
}

instance_ptr TupleOf::eltPtr(instance_ptr self, int64_t i) const {
    if (!(*(layout**)self)) {
        return self;
    }

    return (*(layout**)self)->data + i * m_element_type->bytecount();
}

int64_t TupleOf::count(instance_ptr self) const {
    if (!(*(layout**)self)) {
        return 0;
    }

    return (*(layout**)self)->count;
}

int64_t TupleOf::refcount(instance_ptr self) const {
    if (!(*(layout**)self)) {
        return 0;
    }

    return (*(layout**)self)->refcount;
}

void TupleOf::constructor(instance_ptr self) {
    constructor(self, 0, [](instance_ptr i, int64_t k) {});
}

void TupleOf::destroy(instance_ptr self) {
    if (!(*(layout**)self)) {
        return;
    }

    (*(layout**)self)->refcount--;
    if ((*(layout**)self)->refcount == 0) {
        m_element_type->destroy((*(layout**)self)->count, [&](int64_t k) {return eltPtr(self,k);});
        free((*(layout**)self));
    }
}

void TupleOf::copy_constructor(instance_ptr self, instance_ptr other) {
    (*(layout**)self) = (*(layout**)other);
    if (*(layout**)self) {
        (*(layout**)self)->refcount++;
    }
}

void TupleOf::assign(instance_ptr self, instance_ptr other) {
    layout* old = (*(layout**)self);

    (*(layout**)self) = (*(layout**)other);

    if (*(layout**)self) {
        (*(layout**)self)->refcount++;
    }

    destroy((instance_ptr)&old);
}

void ConstDict::_forwardTypesMayHaveChanged() {
    m_name = "ConstDict(" + m_key->name() + "->" + m_value->name() + ")";
    m_size = sizeof(void*);
    m_is_default_constructible = true;
    m_bytes_per_key = m_key->bytecount();
    m_bytes_per_key_value_pair = m_key->bytecount() + m_value->bytecount();
    m_bytes_per_key_subtree_pair = m_key->bytecount() + this->bytecount();
    m_key_value_pair_type = Tuple::Make({m_key, m_value});
}

bool ConstDict::isBinaryCompatibleWithConcrete(Type* other) {
    if (other->getTypeCategory() != m_typeCategory) {
        return false;
    }

    ConstDict* otherO = (ConstDict*)other;

    return m_key->isBinaryCompatibleWith(otherO->m_key) &&
        m_value->isBinaryCompatibleWith(otherO->m_value);
}

// static
ConstDict* ConstDict::Make(Type* key, Type* value) {
    static std::mutex guard;

    std::lock_guard<std::mutex> lock(guard);

    static std::map<std::pair<Type*, Type*>, ConstDict*> m;

    auto lookup_key = std::make_pair(key,value);

    auto it = m.find(lookup_key);
    if (it == m.end()) {
        it = m.insert(std::make_pair(lookup_key, new ConstDict(key, value))).first;
    }

    return it->second;
}

void ConstDict::repr(instance_ptr self, std::ostringstream& stream) {
    stream << "{";

    int32_t ct = count(self);

    for (long k = 0; k < ct;k++) {
        if (k > 0) {
            stream << ", ";
        }

        m_key->repr(kvPairPtrKey(self,k),stream);
        stream << ": ";
        m_value->repr(kvPairPtrValue(self,k),stream);
    }

    stream << "}";
}

int32_t ConstDict::hash32(instance_ptr left) {
    if (size(left) == 0) {
        return 0x123456;
    }

    if ((*(layout**)left)->hash_cache == -1) {
        Hash32Accumulator acc((int)getTypeCategory());

        int32_t count = size(left);
        acc.add(count);
        for (long k = 0; k < count;k++) {
            acc.add(m_key->hash32(kvPairPtrKey(left,k)));
            acc.add(m_value->hash32(kvPairPtrValue(left,k)));
        }

        (*(layout**)left)->hash_cache = acc.get();
        if ((*(layout**)left)->hash_cache == -1) {
            (*(layout**)left)->hash_cache = -2;
        }
    }

    return (*(layout**)left)->hash_cache;
}

//to make this fast(er), we do dict size comparison first, then keys, then values
char ConstDict::cmp(instance_ptr left, instance_ptr right) {
    if (size(left) < size(right)) {
        return -1;
    }
    if (size(left) > size(right)) {
        return 1;
    }

    if (*(layout**)left == *(layout**)right) {
        return 0;
    }

    int ct = count(left);
    for (long k = 0; k < ct; k++) {
        char res = m_key->cmp(kvPairPtrKey(left,k), kvPairPtrKey(right,k));
        if (res) {
            return res;
        }
    }

    for (long k = 0; k < ct; k++) {
        char res = m_value->cmp(
            kvPairPtrValue(left,k),
            kvPairPtrValue(right,k)
            );

        if (res) {
            return res;
        }
    }

    return 0;
}

void ConstDict::addDicts(instance_ptr lhs, instance_ptr rhs, instance_ptr output) const {
    std::vector<instance_ptr> keep;

    int64_t lhsCount = count(lhs);
    int64_t rhsCount = count(rhs);

    for (long k = 0; k < lhsCount; k++) {
        instance_ptr lhsVal = kvPairPtrKey(lhs, k);

        if (!lookupValueByKey(rhs, lhsVal)) {
            keep.push_back(lhsVal);
        }
    }

    constructor(output, rhsCount + keep.size(), false);

    for (long k = 0; k < rhsCount; k++) {
        m_key->copy_constructor(kvPairPtrKey(output,k), kvPairPtrKey(rhs, k));
        m_value->copy_constructor(kvPairPtrValue(output,k), kvPairPtrValue(rhs, k));
    }
    for (long k = 0; k < keep.size(); k++) {
        m_key->copy_constructor(kvPairPtrKey(output,k + rhsCount), keep[k]);
        m_value->copy_constructor(kvPairPtrValue(output,k + rhsCount), keep[k] + m_bytes_per_key);
    }
    incKvPairCount(output, keep.size() + rhsCount);

    sortKvPairs(output);
}

void ConstDict::subtractTupleOfKeysFromDict(instance_ptr lhs, instance_ptr rhs, instance_ptr output) const {
    TupleOf* tupleType = tupleOfKeysType();

    int64_t lhsCount = count(lhs);
    int64_t rhsCount = tupleType->count(rhs);

    std::set<int> remove;

    for (long k = 0; k < rhsCount; k++) {
        int64_t index = lookupIndexByKey(lhs, tupleType->eltPtr(rhs, k));
        if (index != -1) {
            remove.insert(index);
        }
    }

    constructor(output, lhsCount - remove.size(), false);

    long written = 0;
    for (long k = 0; k < lhsCount; k++) {
        if (remove.find(k) == remove.end()) {
            m_key->copy_constructor(kvPairPtrKey(output,written), kvPairPtrKey(lhs, k));
            m_value->copy_constructor(kvPairPtrValue(output,written), kvPairPtrValue(lhs, k));

            written++;
        }
    }

    incKvPairCount(output, written);
}

instance_ptr ConstDict::kvPairPtrKey(instance_ptr self, int64_t i) const {
    if (!(*(layout**)self)) {
        return self;
    }

    layout& record = **(layout**)self;

    return record.data + m_bytes_per_key_value_pair * i;
}

instance_ptr ConstDict::kvPairPtrValue(instance_ptr self, int64_t i) const {
    if (!(*(layout**)self)) {
        return self;
    }

    layout& record = **(layout**)self;

    return record.data + m_bytes_per_key_value_pair * i + m_bytes_per_key;
}

void ConstDict::incKvPairCount(instance_ptr self, int by) const {
    if (by == 0) {
        return;
    }

    layout& record = **(layout**)self;
    record.count += by;
}

void ConstDict::sortKvPairs(instance_ptr self) const {
    if (!*(layout**)self) {
        return;
    }

    layout& record = **(layout**)self;

    assert(!record.subpointers);

    if (record.count <= 1) {
        return;
    }
    else if (record.count == 2) {
        if (m_key->cmp(kvPairPtrKey(self, 0), kvPairPtrKey(self,1)) > 0) {
            m_key->swap(kvPairPtrKey(self,0), kvPairPtrKey(self,1));
            m_value->swap(kvPairPtrValue(self,0), kvPairPtrValue(self,1));
        }
        return;
    } else {
        std::vector<int> indices;
        for (long k=0;k<record.count;k++) {
            indices.push_back(k);
        }

        std::sort(indices.begin(), indices.end(), [&](int l, int r) {
            char res = m_key->cmp(kvPairPtrKey(self,l),kvPairPtrKey(self,r));
            return res < 0;
            });

        //create a temporary buffer
        std::vector<uint8_t> d;
        d.resize(m_bytes_per_key_value_pair * record.count);

        //final_lookup contains the location of each value in the original sort
        for (long k = 0; k < indices.size(); k++) {
            m_key->swap(kvPairPtrKey(self, indices[k]), &d[m_bytes_per_key_value_pair*k]);
            m_value->swap(kvPairPtrValue(self, indices[k]), &d[m_bytes_per_key_value_pair*k+m_bytes_per_key]);
        }

        //now move them back
        for (long k = 0; k < indices.size(); k++) {
            m_key->swap(kvPairPtrKey(self, k), &d[m_bytes_per_key_value_pair*k]);
            m_value->swap(kvPairPtrValue(self, k), &d[m_bytes_per_key_value_pair*k+m_bytes_per_key]);
        }
    }
}

instance_ptr ConstDict::keyTreePtr(instance_ptr self, int64_t i) const {
    if (!(*(layout**)self)) {
        return self;
    }

    layout& record = **(layout**)self;

    return record.data + m_bytes_per_key_subtree_pair * i;
}

bool ConstDict::instanceIsSubtrees(instance_ptr self) const {
    if (!(*(layout**)self)) {
        return self;
    }

    layout& record = **(layout**)self;

    return record.subpointers != 0;
}

int64_t ConstDict::refcount(instance_ptr self) const {
    if (!(*(layout**)self)) {
        return 0;
    }

    layout& record = **(layout**)self;

    return record.refcount;
}

int64_t ConstDict::count(instance_ptr self) const {
    if (!(*(layout**)self)) {
        return 0;
    }

    layout& record = **(layout**)self;

    if (record.subpointers) {
        return record.subpointers;
    }

    return record.count;
}

int64_t ConstDict::size(instance_ptr self) const {
    if (!(*(layout**)self)) {
        return 0;
    }

    return (*(layout**)self)->count;
}

int64_t ConstDict::lookupIndexByKey(instance_ptr self, instance_ptr key) const {
    if (!(*(layout**)self)) {
        return -1;
    }

    layout& record = **(layout**)self;

    assert(record.subpointers == 0); //this is not implemented yet

    long low = 0;
    long high = record.count;

    while (low < high) {
        long mid = (low+high)/2;
        char res = m_key->cmp(kvPairPtrKey(self, mid), key);

        if (res == 0) {
            return mid;
        } else if (res < 0) {
            low = mid+1;
        } else {
            high = mid;
        }
    }

    return -1;
}

instance_ptr ConstDict::lookupValueByKey(instance_ptr self, instance_ptr key) const {
    int64_t offset = lookupIndexByKey(self, key);
    if (offset == -1) {
        return 0;
    }
    return kvPairPtrValue(self, offset);
}

void ConstDict::constructor(instance_ptr self, int64_t space, bool isPointerTree) const {
    if (space == 0) {
        (*(layout**)self) = nullptr;
        return;
    }

    int bytesPer = isPointerTree ? m_bytes_per_key_subtree_pair : m_bytes_per_key_value_pair;

    (*(layout**)self) = (layout*)malloc(sizeof(layout) + bytesPer * space);

    layout& record = **(layout**)self;

    record.count = 0;
    record.subpointers = 0;
    record.refcount = 1;
    record.hash_cache = -1;
}

void ConstDict::constructor(instance_ptr self) {
    (*(layout**)self) = nullptr;
}

void ConstDict::destroy(instance_ptr self) {
    if (!(*(layout**)self)) {
        return;
    }

    layout& record = **(layout**)self;

    record.refcount--;
    if (record.refcount == 0) {
        if (record.subpointers == 0) {
            m_key->destroy(record.count, [&](long ix) {
                return record.data + m_bytes_per_key_value_pair * ix;
            });
            m_value->destroy(record.count, [&](long ix) {
                return record.data + m_bytes_per_key_value_pair * ix + m_bytes_per_key;
            });
        } else {
            m_key->destroy(record.subpointers, [&](long ix) {
                return record.data + m_bytes_per_key_subtree_pair * ix;
            });
            ((Type*)this)->destroy(record.subpointers, [&](long ix) {
                return record.data + m_bytes_per_key_subtree_pair * ix + m_bytes_per_key;
            });
        }

        free((*(layout**)self));
    }
}

void ConstDict::copy_constructor(instance_ptr self, instance_ptr other) {
    (*(layout**)self) = (*(layout**)other);
    if (*(layout**)self) {
        (*(layout**)self)->refcount++;
    }
}

void ConstDict::assign(instance_ptr self, instance_ptr other) {
    layout* old = (*(layout**)self);

    (*(layout**)self) = (*(layout**)other);

    if (*(layout**)self) {
        (*(layout**)self)->refcount++;
    }

    destroy((instance_ptr)&old);
}

int32_t String::hash32(instance_ptr left) {
    if (!(*(layout**)left)) {
        return 0x12345;
    }

    if ((*(layout**)left)->hash_cache == -1) {
        Hash32Accumulator acc((int)getTypeCategory());
        acc.addBytes(eltPtr(left, 0), bytes_per_codepoint(left) * count(left));
        (*(layout**)left)->hash_cache = acc.get();
        if ((*(layout**)left)->hash_cache == -1) {
            (*(layout**)left)->hash_cache = -2;
        }
    }

    return (*(layout**)left)->hash_cache;
}

char String::cmp(instance_ptr left, instance_ptr right) {
    if ( !(*(layout**)left) && !(*(layout**)right) ) {
        return 0;
    }
    if ( !(*(layout**)left) && (*(layout**)right) ) {
        return -1;
    }
    if ( (*(layout**)left) && !(*(layout**)right) ) {
        return 1;
    }

    if (bytes_per_codepoint(left) < bytes_per_codepoint(right)) {
        return -1;
    }

    if (bytes_per_codepoint(left) > bytes_per_codepoint(right)) {
        return 1;
    }

    int bytesPer = bytes_per_codepoint(right);

    char res = byteCompare(
        eltPtr(left, 0),
        eltPtr(right, 0),
        bytesPer * std::min(count(left), count(right))
        );

    if (res) {
        return res;
    }

    if (count(left) < count(right)) {
        return -1;
    }

    if (count(left) > count(right)) {
        return 1;
    }

    return 0;
}

void String::constructor(instance_ptr self, int64_t bytes_per_codepoint, int64_t count, const char* data) const {
    if (count == 0) {
        *(layout**)self = nullptr;
        return;
    }

    (*(layout**)self) = (layout*)malloc(sizeof(layout) + count * bytes_per_codepoint);

    (*(layout**)self)->bytes_per_codepoint = bytes_per_codepoint;
    (*(layout**)self)->pointcount = count;
    (*(layout**)self)->hash_cache = -1;
    (*(layout**)self)->refcount = 1;

    if (data) {
        ::memcpy((*(layout**)self)->data, data, count * bytes_per_codepoint);
    }
}

void String::repr(instance_ptr self, std::ostringstream& stream) {
    //as if it were bytes, which is totally wrong...
    stream << "\"";
    long bytes = count(self);
    uint8_t* base = eltPtr(self,0);

    static char hexDigits[] = "0123456789abcdef";

    for (long k = 0; k < bytes;k++) {
        if (base[k] == '"') {
            stream << "\\\"";
        } else if (base[k] == '\\') {
            stream << "\\\\";
        } else if (isprint(base[k])) {
            stream << base[k];
        } else {
            stream << "\\x" << hexDigits[base[k] / 16] << hexDigits[base[k] % 16];
        }
    }

    stream << "\"";
}

instance_ptr String::eltPtr(instance_ptr self, int64_t i) const {
    const static char* emptyPtr = "";

    if (*(layout**)self == nullptr) {
        return (instance_ptr)emptyPtr;
    }

    return (*(layout**)self)->eltPtr(i);
}

int64_t String::bytes_per_codepoint(instance_ptr self) const {
    if (*(layout**)self == nullptr) {
        return 1;
    }

    return (*(layout**)self)->bytes_per_codepoint;
}

int64_t String::count(instance_ptr self) const {
    if (*(layout**)self == nullptr) {
        return 0;
    }

    return (*(layout**)self)->pointcount;
}

void String::destroy(instance_ptr self) {
    if (!*(layout**)self) {
        return;
    }

    (*(layout**)self)->refcount--;

    if ((*(layout**)self)->refcount == 0) {
        free((*(layout**)self));
    }
}

void String::copy_constructor(instance_ptr self, instance_ptr other) {
    (*(layout**)self) = (*(layout**)other);
    if (*(layout**)self) {
        (*(layout**)self)->refcount++;
    }
}

void String::assign(instance_ptr self, instance_ptr other) {
    layout* old = (*(layout**)self);

    (*(layout**)self) = (*(layout**)other);

    if (*(layout**)self) {
        (*(layout**)self)->refcount++;
    }

    destroy((instance_ptr)&old);
}

bool Bytes::isBinaryCompatibleWithConcrete(Type* other) {
    if (other->getTypeCategory() != m_typeCategory) {
        return false;
    }

    return true;
}

void Bytes::repr(instance_ptr self, std::ostringstream& stream) {
    stream << "b" << "'";
    long bytes = count(self);
    uint8_t* base = eltPtr(self,0);

    static char hexDigits[] = "0123456789abcdef";

    for (long k = 0; k < bytes;k++) {
        if (base[k] == '\'') {
            stream << "\\'";
        } else if (base[k] == '\r') {
            stream << "\\r";
        } else if (base[k] == '\n') {
            stream << "\\n";
        } else if (base[k] == '\t') {
            stream << "\\t";
        } else if (base[k] == '\\') {
            stream << "\\\\";
        } else if (isprint(base[k])) {
            stream << base[k];
        } else {
            stream << "\\x" << hexDigits[base[k] / 16] << hexDigits[base[k] % 16];
        }
    }

    stream << "'";
}

int32_t Bytes::hash32(instance_ptr left) {
    Hash32Accumulator acc((int)getTypeCategory());

    if (!(*(layout**)left)) {
        return 0x1234;
    }

    if ((*(layout**)left)->hash_cache == -1) {
        Hash32Accumulator acc((int)getTypeCategory());

        acc.addBytes(eltPtr(left, 0), count(left));

        (*(layout**)left)->hash_cache = acc.get();
        if ((*(layout**)left)->hash_cache == -1) {
            (*(layout**)left)->hash_cache = -2;
        }
    }

    return (*(layout**)left)->hash_cache;
}

char Bytes::cmp(instance_ptr left, instance_ptr right) {
    if ( !(*(layout**)left) && !(*(layout**)right) ) {
        return 0;
    }
    if ( !(*(layout**)left) && (*(layout**)right) ) {
        return -1;
    }
    if ( (*(layout**)left) && !(*(layout**)right) ) {
        return 1;
    }
    if ( (*(layout**)left) == (*(layout**)right) ) {
        return 0;
    }

    char res = byteCompare(eltPtr(left, 0), eltPtr(right, 0), std::min(count(left), count(right)));

    if (res) {
        return res;
    }

    if (count(left) < count(right)) {
        return -1;
    }

    if (count(left) > count(right)) {
        return 1;
    }

    return 0;
}

void Bytes::constructor(instance_ptr self, int64_t count, const char* data) const {
    if (count == 0) {
        *(layout**)self = nullptr;
        return;
    }
    (*(layout**)self) = (layout*)malloc(sizeof(layout) + count);

    (*(layout**)self)->bytecount = count;
    (*(layout**)self)->refcount = 1;
    (*(layout**)self)->hash_cache = -1;

    if (data) {
        ::memcpy((*(layout**)self)->data, data, count);
    }
}

instance_ptr Bytes::eltPtr(instance_ptr self, int64_t i) const {
    //we don't want to have to return null here, but there is no actual memory to back this.
    if (*(layout**)self == nullptr) {
        return self;
    }

    return (*(layout**)self)->data + i;
}

int64_t Bytes::count(instance_ptr self) const {
    if (*(layout**)self == nullptr) {
        return 0;
    }

    return (*(layout**)self)->bytecount;
}

void Bytes::destroy(instance_ptr self) {
    if (!*(layout**)self) {
        return;
    }

    (*(layout**)self)->refcount--;

    if ((*(layout**)self)->refcount == 0) {
        free((*(layout**)self));
    }
}

void Bytes::copy_constructor(instance_ptr self, instance_ptr other) {
    (*(layout**)self) = (*(layout**)other);
    if (*(layout**)self) {
        (*(layout**)self)->refcount++;
    }
}

void Bytes::assign(instance_ptr self, instance_ptr other) {
    layout* old = (*(layout**)self);

    (*(layout**)self) = (*(layout**)other);

    if (*(layout**)self) {
        (*(layout**)self)->refcount++;
    }

    destroy((instance_ptr)&old);
}

bool Alternative::isBinaryCompatibleWithConcrete(Type* other) {
    if (other->getTypeCategory() == TypeCategory::catConcreteAlternative) {
        other = other->getBaseType();
    }

    if (other->getTypeCategory() != m_typeCategory) {
        return false;
    }

    Alternative* otherO = (Alternative*)other;

    if (m_subtypes.size() != otherO->m_subtypes.size()) {
        return false;
    }

    for (long k = 0; k < m_subtypes.size(); k++) {
        if (m_subtypes[k].first != otherO->m_subtypes[k].first) {
            return false;
        }
        if (!m_subtypes[k].second->isBinaryCompatibleWith(otherO->m_subtypes[k].second)) {
            return false;
        }
    }

    return true;
}

void Alternative::_forwardTypesMayHaveChanged() {
    m_size = sizeof(void*);

    m_is_default_constructible = false;
    m_all_alternatives_empty = true;
    m_arg_positions.clear();
    m_default_construction_ix = 0;
    m_default_construction_type = nullptr;

    for (auto& subtype_pair: m_subtypes) {
        if (subtype_pair.second->bytecount() > 0) {
            m_all_alternatives_empty = false;
        }

        if (m_arg_positions.find(subtype_pair.first) != m_arg_positions.end()) {
            throw std::runtime_error("Can't create an alternative with " +
                    subtype_pair.first + " defined twice.");
        }

        m_arg_positions[subtype_pair.first] = m_arg_positions.size();

        if (subtype_pair.second->is_default_constructible() && !m_is_default_constructible) {
            m_is_default_constructible = true;
            m_default_construction_ix = m_arg_positions[subtype_pair.first];
        }
    }

    m_size = (m_all_alternatives_empty ? 1 : sizeof(void*));
}

char Alternative::cmp(instance_ptr left, instance_ptr right) {
    if (m_all_alternatives_empty) {
        if (*(uint8_t*)left < *(uint8_t*)right) {
            return -1;
        }
        if (*(uint8_t*)left > *(uint8_t*)right) {
            return 1;
        }
        return 0;
    }

    layout& record_l = **(layout**)left;
    layout& record_r = **(layout**)right;

    if ( &record_l == &record_r ) {
        return 0;
    }

    if (record_l.which < record_r.which) {
        return -1;
    }
    if (record_l.which > record_r.which) {
        return 1;
    }

    return m_subtypes[record_l.which].second->cmp(record_l.data, record_r.data);
}

void Alternative::repr(instance_ptr self, std::ostringstream& stream) {
    stream << m_subtypes[which(self)].first;
    m_subtypes[which(self)].second->repr(eltPtr(self), stream);
}

int32_t Alternative::hash32(instance_ptr left) {
    Hash32Accumulator acc((int)TypeCategory::catAlternative);

    acc.add(which(left));
    acc.add(m_subtypes[which(left)].second->hash32(eltPtr(left)));

    return acc.get();
}

instance_ptr Alternative::eltPtr(instance_ptr self) const {
    if (m_all_alternatives_empty) {
        return self;
    }

    layout& record = **(layout**)self;

    return record.data;
}

int64_t Alternative::which(instance_ptr self) const {
    if (m_all_alternatives_empty) {
        return *(uint8_t*)self;
    }

    layout& record = **(layout**)self;

    return record.which;
}

void Alternative::destroy(instance_ptr self) {
    if (m_all_alternatives_empty) {
        return;
    }

    layout& record = **(layout**)self;

    record.refcount--;

    if (record.refcount == 0) {
        m_subtypes[record.which].second->destroy(record.data);
        free(*(layout**)self);
    }
}

void Alternative::copy_constructor(instance_ptr self, instance_ptr other) {
    if (m_all_alternatives_empty) {
        *(uint8_t*)self = *(uint8_t*)other;
        return;
    }

    (*(layout**)self) = (*(layout**)other);

    if (*(layout**)self) {
        (*(layout**)self)->refcount++;
    }
}

void Alternative::assign(instance_ptr self, instance_ptr other) {
    layout* old = (*(layout**)self);

    (*(layout**)self) = (*(layout**)other);

    if (*(layout**)self) {
        (*(layout**)self)->refcount++;
    }

    destroy((instance_ptr)&old);
}

// static
Alternative* Alternative::Make(std::string name,
                               const std::vector<std::pair<std::string, NamedTuple*> >& types,
                               const std::map<std::string, Function*>& methods //methods preclude us from being in the memo
                              ) {
    if (methods.size()) {
        return new Alternative(name, types, methods);
    }

    static std::mutex guard;

    std::lock_guard<std::mutex> lock(guard);

    typedef std::pair<std::string, std::vector<std::pair<std::string, NamedTuple*> > > keytype;

    static std::map<keytype, Alternative*> m;

    auto it = m.find(keytype(name, types));

    if (it == m.end()) {
        it = m.insert(std::make_pair(keytype(name, types), new Alternative(name, types, methods))).first;
    }

    return it->second;
}

Type* Alternative::pickConcreteSubclassConcrete(instance_ptr data) {
    uint8_t i = which(data);

    return ConcreteAlternative::Make(this, i);
}

void Alternative::constructor(instance_ptr self) {
    if (!m_default_construction_type) {
        m_default_construction_type = ConcreteAlternative::Make(this, m_default_construction_ix);
    }

    m_default_construction_type->constructor(self);
}

bool ConcreteAlternative::isBinaryCompatibleWithConcrete(Type* other) {
    if (other->getTypeCategory() == TypeCategory::catConcreteAlternative) {
        ConcreteAlternative* otherO = (ConcreteAlternative*)other;

        return otherO->m_alternative->isBinaryCompatibleWith(m_alternative) &&
            m_which == otherO->m_which;
    }

    if (other->getTypeCategory() == TypeCategory::catAlternative) {
        return m_alternative->isBinaryCompatibleWith(other);
    }

    return false;
}

void ConcreteAlternative::_forwardTypesMayHaveChanged() {
    m_base = m_alternative;
    m_name = m_alternative->name() + "." + m_alternative->subtypes()[m_which].first;
    m_size = m_alternative->bytecount();
    m_is_default_constructible = m_alternative->subtypes()[m_which].second->is_default_constructible();
}

void ConcreteAlternative::constructor(instance_ptr self) {
    if (m_alternative->all_alternatives_empty()) {
        *(uint8_t*)self = m_which;
    } else {
        constructor(self, [&](instance_ptr i) {
            m_alternative->subtypes()[m_which].second->constructor(i);
        });
    }
}

// static
ConcreteAlternative* ConcreteAlternative::Make(Alternative* alt, int64_t which) {
    static std::mutex guard;

    std::lock_guard<std::mutex> lock(guard);

    typedef std::pair<Alternative*, int64_t> keytype;

    static std::map<keytype, ConcreteAlternative*> m;

    auto it = m.find(keytype(alt ,which));

    if (it == m.end()) {
        it = m.insert(
            std::make_pair(keytype(alt,which), new ConcreteAlternative(alt,which))
            ).first;
    }

    return it->second;
}

bool PythonSubclass::isBinaryCompatibleWithConcrete(Type* other) {
    Type* nonPyBase = m_base;
    while (nonPyBase->getTypeCategory() == TypeCategory::catPythonSubclass) {
        nonPyBase = nonPyBase->getBaseType();
    }

    Type* otherNonPyBase = other;
    while (otherNonPyBase->getTypeCategory() == TypeCategory::catPythonSubclass) {
        otherNonPyBase = otherNonPyBase->getBaseType();
    }

    return nonPyBase->isBinaryCompatibleWith(otherNonPyBase);
}

// static
PythonSubclass* PythonSubclass::Make(Type* base, PyTypeObject* pyType) {
    static std::mutex guard;

    std::lock_guard<std::mutex> lock(guard);

    typedef std::pair<Type*, PyTypeObject*> keytype;

    static std::map<keytype, PythonSubclass*> m;

    auto it = m.find(keytype(base, pyType));

    if (it == m.end()) {
        it = m.insert(
            std::make_pair(keytype(base,pyType), new PythonSubclass(base, pyType))
            ).first;
    }

    return it->second;
}

void PythonObjectOfType::repr(instance_ptr self, std::ostringstream& stream) {
    PyObject* p = *(PyObject**)self;

    PyObject* o = PyObject_Repr(p);

    if (!o) {
        stream << "<EXCEPTION>";
        PyErr_Clear();
        return;
    }

    if (!PyUnicode_Check(o)) {
        stream << "<EXCEPTION>";
        Py_DECREF(o);
        return;
    }

    stream << PyUnicode_AsUTF8(o);

    Py_DECREF(o);
}

char PythonObjectOfType::cmp(instance_ptr left, instance_ptr right) {
    PyObject* l = *(PyObject**)left;
    PyObject* r = *(PyObject**)right;

    int res = PyObject_RichCompareBool(l, r, Py_EQ);
    if (res == -1) {
        PyErr_Clear();
        if (l < r) {
            return -1;
        }
        if (l > r) {
            return 1;
        }
        return 0;
    }

    if (res) {
        return 0;
    }

    res = PyObject_RichCompareBool(l, r, Py_LT);

    if (res == -1) {
        PyErr_Clear();
        if (l < r) {
            return -1;
        }
        if (l > r) {
            return 1;
        }
        return 0;
    }

    if (res) {
        return -1;
    }
    return 1;
}

// static
PythonObjectOfType* PythonObjectOfType::Make(PyTypeObject* pyType) {
    static std::mutex guard;

    std::lock_guard<std::mutex> lock(guard);

    typedef PyTypeObject* keytype;

    static std::map<keytype, PythonObjectOfType*> m;

    auto it = m.find(pyType);

    if (it == m.end()) {
        it = m.insert(
            std::make_pair(pyType, new PythonObjectOfType(pyType))
            ).first;
    }

    return it->second;
}

bool HeldClass::isBinaryCompatibleWithConcrete(Type* other) {
    if (other->getTypeCategory() != m_typeCategory) {
        return false;
    }

    HeldClass* otherO = (HeldClass*)other;

    if (m_members.size() != otherO->m_members.size()) {
        return false;
    }

    for (long k = 0; k < m_members.size(); k++) {
        if (std::get<0>(m_members[k]) != std::get<0>(otherO->m_members[k]) ||
                !std::get<1>(m_members[k])->isBinaryCompatibleWith(std::get<1>(otherO->m_members[k]))) {
            return false;
        }
    }

    return true;
}

void HeldClass::_forwardTypesMayHaveChanged() {
    m_is_default_constructible = true;
    m_byte_offsets.clear();

    //first m_members.size() bits (rounded up to nearest byte) contains the initialization flags.
    m_size = int((m_members.size() + 7) / 8); //round up to nearest byte

    for (auto t: m_members) {
        m_byte_offsets.push_back(m_size);
        m_size += std::get<1>(t)->bytecount();
    }
}

char HeldClass::cmp(instance_ptr left, instance_ptr right) {
    for (long k = 0; k < m_members.size(); k++) {
        bool leftInit = checkInitializationFlag(left,k);
        bool rightInit = checkInitializationFlag(right,k);

        if (leftInit && !rightInit) {
            return 1;
        }

        if (rightInit && !leftInit) {
            return -1;
        }

        if (leftInit && rightInit) {
            char res = std::get<1>(m_members[k])->cmp(left + m_byte_offsets[k], right + m_byte_offsets[k]);
            if (res != 0) {
                return res;
            }
        }
    }

    return 0;
}

void HeldClass::repr(instance_ptr self, std::ostringstream& stream) {
    stream << m_name << "(";

    for (long k = 0; k < m_members.size();k++) {
        if (k > 0) {
            stream << ", ";
        }

        if (checkInitializationFlag(self, k)) {
            stream << std::get<0>(m_members[k]) << "=";

            std::get<1>(m_members[k])->repr(eltPtr(self,k),stream);
        } else {
            stream << std::get<0>(m_members[k]) << " not initialized";
        }
    }
    if (m_members.size() == 1) {
        stream << ",";
    }

    stream << ")";
}

int32_t HeldClass::hash32(instance_ptr left) {
    Hash32Accumulator acc((int)getTypeCategory());

    for (long k = 0; k < m_members.size();k++) {
        if (checkInitializationFlag(left,k)) {
            acc.add(1);
            acc.add(std::get<1>(m_members[k])->hash32(eltPtr(left,k)));
        } else {
            acc.add(0);
        }
    }

    acc.add(m_members.size());

    return acc.get();
}

void HeldClass::setAttribute(instance_ptr self, int memberIndex, instance_ptr other) const {
	Type* member_t = std::get<1>(m_members[memberIndex]);
    if (checkInitializationFlag(self, memberIndex)) {
        member_t->assign(
            eltPtr(self, memberIndex),
            other
            );
    } else {
        member_t->copy_constructor(
            eltPtr(self, memberIndex),
            other
            );
        setInitializationFlag(self, memberIndex);
    }
}

void HeldClass::emptyConstructor(instance_ptr self) {
    //more efficient would be to just write over the bytes directly.
    for (size_t k = 0; k < m_members.size(); k++) {
        clearInitializationFlag(self, k);
    }
}

void HeldClass::constructor(instance_ptr self) {
    for (size_t k = 0; k < m_members.size(); k++) {
        Type* member_t = std::get<1>(m_members[k]);
        PyObject* member_val = std::get<2>(m_members[k]);
        Type* valType = native_instance_wrapper::extractTypeFrom(member_val->ob_type);
        if (wantsToDefaultConstruct(member_t)) {
            if (member_val != Py_None) {
                std::cout << "blabla!" << std::endl;

	        } else {
                member_t->constructor(self+m_byte_offsets[k]);
            }
            setInitializationFlag(self, k);
        } else {
            clearInitializationFlag(self, k);
        }
    }
}

void HeldClass::destroy(instance_ptr self) {
    for (long k = (long)m_members.size() - 1; k >= 0; k--) {
	    Type* member_t = std::get<1>(m_members[k]);
        if (checkInitializationFlag(self, k)) {
            member_t->destroy(self+m_byte_offsets[k]);
        }
    }
}

void HeldClass::copy_constructor(instance_ptr self, instance_ptr other) {
    for (long k = (long)m_members.size() - 1; k >= 0; k--) {
	    Type* member_t = std::get<1>(m_members[k]);
        if (checkInitializationFlag(other, k)) {
            member_t->copy_constructor(self+m_byte_offsets[k], other+m_byte_offsets[k]);
            setInitializationFlag(self, k);
        }
    }
}

void HeldClass::assign(instance_ptr self, instance_ptr other) {
    for (long k = (long)m_members.size() - 1; k >= 0; k--) {
        bool selfInit = checkInitializationFlag(self,k);
        bool otherInit = checkInitializationFlag(other,k);
        Type* member_t = std::get<1>(m_members[k]);
        if (selfInit && otherInit) {
            member_t->assign(self + m_byte_offsets[k], other+m_byte_offsets[k]);
        }
        else if (selfInit && !otherInit) {
            member_t->destroy(self+m_byte_offsets[k]);
            clearInitializationFlag(self, k);
        } else if (!selfInit && otherInit) {
            member_t->copy_constructor(self + m_byte_offsets[k], other+m_byte_offsets[k]);
            clearInitializationFlag(self, k);
        }
    }
}

void HeldClass::setInitializationFlag(instance_ptr self, int memberIndex) const {
    int byte = memberIndex / 8;
    int bit = memberIndex % 8;
    uint8_t mask = (1 << bit);
    ((uint8_t*)self)[byte] |= mask;
}

void HeldClass::clearInitializationFlag(instance_ptr self, int memberIndex) const {
    int byte = memberIndex / 8;
    int bit = memberIndex % 8;
    uint8_t mask = (1 << bit);
    ((uint8_t*)self)[byte] &= ~mask;
}

int HeldClass::memberNamed(const char* c) const {
    for (long k = 0; k < m_members.size(); k++) {
        if (std::get<0>(m_members[k]) == c) {
            return k;
        }
    }

    return -1;
}

bool Class::isBinaryCompatibleWithConcrete(Type* other) {
    if (other->getTypeCategory() != m_typeCategory) {
        return false;
    }

    Class* otherO = (Class*)other;

    return m_heldClass->isBinaryCompatibleWith(otherO->m_heldClass);
}

void Class::_forwardTypesMayHaveChanged() {
    m_is_default_constructible = m_heldClass->is_default_constructible();
    m_name = m_heldClass->name();
}

instance_ptr Class::eltPtr(instance_ptr self, int64_t ix) const {
    layout& l = **(layout**)self;
    return m_heldClass->eltPtr(l.data, ix);
}

void Class::setAttribute(instance_ptr self, int64_t ix, instance_ptr elt) const {
    layout& l = **(layout**)self;
    m_heldClass->setAttribute(l.data, ix, elt);
}

bool Class::checkInitializationFlag(instance_ptr self, int64_t ix) const {
    layout& l = **(layout**)self;
    return m_heldClass->checkInitializationFlag(l.data, ix);
}

char Class::cmp(instance_ptr left, instance_ptr right) {
    layout& l = **(layout**)left;
    layout& r = **(layout**)right;

    if ( &l == &r ) {
        return 0;
    }

    return m_heldClass->cmp(l.data,r.data);
}

void Class::repr(instance_ptr self, std::ostringstream& stream) {
    layout& l = **(layout**)self;
    m_heldClass->repr(l.data, stream);
}

int32_t Class::hash32(instance_ptr left) {
    layout& l = **(layout**)left;
    return m_heldClass->hash32(l.data);
}

void Class::emptyConstructor(instance_ptr self) {
    if (!m_is_default_constructible) {
        throw std::runtime_error(m_name + " is not default-constructible");
    }

    *(layout**)self = (layout*)malloc(sizeof(layout) + m_heldClass->bytecount());

    layout& l = **(layout**)self;
    l.refcount = 1;

    m_heldClass->emptyConstructor(l.data);
}

void Class::constructor(instance_ptr self) {
    if (!m_is_default_constructible) {
        throw std::runtime_error(m_name + " is not default-constructible");
    }

    *(layout**)self = (layout*)malloc(sizeof(layout) + m_heldClass->bytecount());

    layout& l = **(layout**)self;
    l.refcount = 1;

    m_heldClass->constructor(l.data);
}

int64_t Class::refcount(instance_ptr self) {
    layout& l = **(layout**)self;
    return l.refcount;
}

void Class::destroy(instance_ptr self) {
    layout& l = **(layout**)self;
    l.refcount--;

    if (l.refcount == 0) {
        m_heldClass->destroy(l.data);
        free(*(layout**)self);
    }
}

void Class::copy_constructor(instance_ptr self, instance_ptr other) {
    (*(layout**)self) = (*(layout**)other);
    (*(layout**)self)->refcount++;
}

void Class::assign(instance_ptr self, instance_ptr other) {
    layout* old = (*(layout**)self);

    (*(layout**)self) = (*(layout**)other);

    if (*(layout**)self) {
        (*(layout**)self)->refcount++;
    }

    destroy((instance_ptr)&old);
}
