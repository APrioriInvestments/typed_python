#pragma once

#include "Type.hpp"

class SetType : public Type {
  public:
    SetType(Type* eltype)
        : Type(TypeCategory::catSet)
        , m_key_type(eltype) {
        endOfConstructorInitialization();
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        visitor(m_key_type);
    }

    instance_ptr insertKey(instance_ptr self, instance_ptr key);
    instance_ptr lookupKey(instance_ptr self, instance_ptr key) const;
    bool discard(instance_ptr self, instance_ptr key);
    void clear(instance_ptr self);
    void constructor(instance_ptr self);
    void destroy(instance_ptr self);
    void copy_constructor(instance_ptr self, instance_ptr other);
    void repr(instance_ptr self, ReprAccumulator& stream);
    int64_t refcount(instance_ptr self) const;
    bool _updateAfterForwardTypesChanged();
    int64_t size(instance_ptr self) const;
    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp,
             bool suppressExceptions = false);
    Type* keyType() const { return m_key_type; }

    // hash_table_layout accessors
    int64_t slotCount(instance_ptr self) const;
    bool slotPopulated(instance_ptr self, size_t offset) const;
    instance_ptr keyAtSlot(instance_ptr self, size_t offset) const;

  public:
    static SetType* Make(Type* eltype);

  private:
    Type* m_key_type;
    size_t m_bytes_per_el;
};
