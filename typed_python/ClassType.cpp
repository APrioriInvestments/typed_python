#include "AllTypes.hpp"

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

void Class::repr(instance_ptr self, ReprAccumulator& stream) {
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
