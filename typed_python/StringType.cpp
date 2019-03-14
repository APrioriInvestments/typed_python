#include "AllTypes.hpp"

String::layout* String::upgradeCodePoints(layout* lhs, int32_t newBytesPerCodepoint) {
    if (!lhs) {
        return lhs;
    }

    if (newBytesPerCodepoint == lhs->bytes_per_codepoint) {
        lhs->refcount++;
        return lhs;
    }

    int64_t new_byteCount = sizeof(layout) + lhs->pointcount * newBytesPerCodepoint;

    layout* new_layout = (layout*)malloc(new_byteCount);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;
    new_layout->bytes_per_codepoint = newBytesPerCodepoint;
    new_layout->pointcount = lhs->pointcount;

    if (lhs->bytes_per_codepoint == 1 && new_layout->bytes_per_codepoint == 2) {
        for (long i = 0; i < lhs->pointcount; i++) {
            ((uint16_t*)new_layout->data)[i] = ((uint8_t*)lhs->data)[i];
        }
    }
    if (lhs->bytes_per_codepoint == 1 && new_layout->bytes_per_codepoint == 4) {
        for (long i = 0; i < lhs->pointcount; i++) {
            ((uint32_t*)new_layout->data)[i] = ((uint8_t*)lhs->data)[i];
        }
    }
    if (lhs->bytes_per_codepoint == 2 && new_layout->bytes_per_codepoint == 4) {
        for (long i = 0; i < lhs->pointcount; i++) {
            ((uint32_t*)new_layout->data)[i] = ((uint16_t*)lhs->data)[i];
        }
    }

    return new_layout;
}

String::layout* String::concatenate(layout* lhs, layout* rhs) {
    if (!rhs && !lhs) {
        return lhs;
    }
    if (!rhs) {
        lhs->refcount++;
        return lhs;
    }
    if (!lhs) {
        rhs->refcount++;
        return rhs;
    }

    if (lhs->bytes_per_codepoint < rhs->bytes_per_codepoint) {
        layout* newLayout = upgradeCodePoints(lhs, rhs->bytes_per_codepoint);

        layout* result = concatenate(newLayout, rhs);
        String::destroyStatic((instance_ptr)&newLayout);
        return result;
    }

    if (rhs->bytes_per_codepoint < lhs->bytes_per_codepoint) {
        layout* newLayout = upgradeCodePoints(rhs, lhs->bytes_per_codepoint);

        layout* result = concatenate(lhs, newLayout);
        String::destroyStatic((instance_ptr)&newLayout);
        return result;
    }

    //they're the same
    int64_t new_byteCount = sizeof(layout) + (rhs->pointcount + lhs->pointcount) * lhs->bytes_per_codepoint;

    layout* new_layout = (layout*)malloc(new_byteCount);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;
    new_layout->bytes_per_codepoint = lhs->bytes_per_codepoint;
    new_layout->pointcount = lhs->pointcount + rhs->pointcount;

    memcpy(new_layout->data, lhs->data, lhs->pointcount * lhs->bytes_per_codepoint);
    memcpy(new_layout->data + lhs->pointcount * lhs->bytes_per_codepoint,
        rhs->data, rhs->pointcount * lhs->bytes_per_codepoint);

    return new_layout;
}

String::layout* String::lower(layout *l) {
    if (!l) {
        return l;
    }

    int64_t new_byteCount = sizeof(layout) + l->pointcount * l->bytes_per_codepoint;
    layout* new_layout = (layout*)malloc(new_byteCount);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;
    new_layout->bytes_per_codepoint = l->bytes_per_codepoint;
    new_layout->pointcount = l->pointcount;

    if (l->bytes_per_codepoint == 1) {
        for (uint8_t *src = l->data, *dest = new_layout->data, *end = src + l->pointcount; src < end; ) {
            *dest++ = (uint8_t)tolower(*src++);
        }
    }
    else if (l->bytes_per_codepoint == 2) {
        for (uint16_t *src = (uint16_t *)l->data, *dest = (uint16_t *)new_layout->data, *end = src + l->pointcount; src < end; ) {
            *dest++ = (uint16_t)towlower(*src++);
        }
    }
    else if (l->bytes_per_codepoint == 4) {
        for (uint32_t *src = (uint32_t *)l->data, *dest = (uint32_t *)new_layout->data, *end = src + l->pointcount; src < end; ) {
            *dest++ = (uint32_t)towlower(*src++);
        }
    }

    return new_layout;
}

String::layout* String::getitem(layout* lhs, int64_t offset) {
    if (!lhs) {
        return lhs;
    }

    if (lhs->pointcount == 1) {
        lhs->refcount++;
        return lhs;
    }

    if (offset < 0) {
        offset += lhs->pointcount;
    }

    int64_t new_byteCount = sizeof(layout) + 1 * lhs->bytes_per_codepoint;

    layout* new_layout = (layout*)malloc(new_byteCount);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;
    new_layout->bytes_per_codepoint = lhs->bytes_per_codepoint;
    new_layout->pointcount = 1;

    //we could figure out if we can represent this with a smaller encoding.
    if (new_layout->bytes_per_codepoint == 1) {
        ((int8_t*)new_layout->data)[0] = ((int8_t*)lhs->data)[offset];
    }
    if (new_layout->bytes_per_codepoint == 2) {
        ((int16_t*)new_layout->data)[0] = ((int16_t*)lhs->data)[offset];
    }
    if (new_layout->bytes_per_codepoint == 4) {
        ((int32_t*)new_layout->data)[0] = ((int32_t*)lhs->data)[offset];
    }

    return new_layout;
}

int64_t String::bytesPerCodepointRequiredForUtf8(const uint8_t* utf8Str, int64_t length) {
    int64_t bytes_per_codepoint = 1;
    while (length > 0) {
        if (utf8Str[0] >> 7 == 0) {
            //one byte encoded here
            length -= 1;
            utf8Str++;
        }

        if (utf8Str[0] >> 5 == 0b110) {
            length -= 1;
            utf8Str+=2;
            bytes_per_codepoint = std::max<int64_t>(2, bytes_per_codepoint);
        }

        if (utf8Str[0] >> 4 == 0b1110) {
            length -= 1;
            utf8Str += 3;
            bytes_per_codepoint = std::max<int64_t>(2, bytes_per_codepoint);
        }

        if (utf8Str[0] >> 3 == 0b11110) {
            length -= 1;
            utf8Str+=4;
            bytes_per_codepoint = std::max<int64_t>(4, bytes_per_codepoint);
        }
    }
    return bytes_per_codepoint;
}

template<class T>
void decodeUtf8ToTyped(T* target, uint8_t* utf8Str, int64_t bytes_per_codepoint, int64_t length) {
    while (length > 0) {
        if ((utf8Str[0] >> 7) == 0) {
            //one byte encoded here
            target[0] = utf8Str[0];

            length -= 1;
            target++;
            utf8Str++;
        }
        else if ((utf8Str[0] >> 5) == 0b110) {
            target[0] = (uint32_t(utf8Str[0] & 0b11111) << 6) + uint32_t(utf8Str[1] & 0b111111);
            length -= 1;
            target++;
            utf8Str+=2;
        }
        else if ((utf8Str[0] >> 4) == 0b1110) {
            target[0] =
                (uint32_t(utf8Str[0] & 0b1111) << 12) +
                (uint32_t(utf8Str[1] & 0b111111) << 6) +
                 uint32_t(utf8Str[2] & 0b111111);
            length -= 1;
            target++;
            utf8Str+=3;
        }
        else if ((utf8Str[0] >> 3) == 0b11110) {
            target[0] =
                (uint32_t(utf8Str[0] & 0b111) << 18) +
                (uint32_t(utf8Str[1] & 0b111111) << 12) +
                (uint32_t(utf8Str[2] & 0b111111) << 6) +
                 uint32_t(utf8Str[3] & 0b111111);
            length -= 1;
            target++;
            utf8Str+=4;
        }
    }
}

void String::decodeUtf8To(uint8_t* target, uint8_t* utf8Str, int64_t bytes_per_codepoint, int64_t length) {
    if (bytes_per_codepoint == 1) {
        decodeUtf8ToTyped(target, utf8Str, bytes_per_codepoint, length);
    }
    if (bytes_per_codepoint == 2) {
        decodeUtf8ToTyped((uint16_t*)target, utf8Str, bytes_per_codepoint, length);
    }
    if (bytes_per_codepoint == 4) {
        decodeUtf8ToTyped((uint32_t*)target, utf8Str, bytes_per_codepoint, length);
    }
}

String::layout* String::createFromUtf8(const char* utfEncodedString, int64_t length) {
    int64_t bytes_per_codepoint = bytesPerCodepointRequiredForUtf8((uint8_t*)utfEncodedString, length);

    int64_t new_byteCount = sizeof(layout) + length * bytes_per_codepoint;

    layout* new_layout = (layout*)malloc(new_byteCount);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;
    new_layout->bytes_per_codepoint = bytes_per_codepoint;
    new_layout->pointcount = length;

    if (bytes_per_codepoint == 1) {
        memcpy(new_layout->data, utfEncodedString, length);
    } else {
        decodeUtf8To((uint8_t*)new_layout->data, (uint8_t*)utfEncodedString, bytes_per_codepoint, length);
    }

    return new_layout;
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

template<class T1, class T2>
char typedArrayCompare(T1* l, T2* r, size_t count) {
    for (size_t i = 0; i < count; i++) {
        uint32_t lv = l[i];
        uint32_t rv = r[i];
        if (lv < rv) {
            return -1;
        }
        if (lv > rv) {
            return 1;
        }
    }
    return 0;
}

bool String::cmp(instance_ptr left, instance_ptr right, int pyComparisonOp) {
    return cmpResultToBoolForPyOrdering(pyComparisonOp, cmpStatic(*(layout**)left, *(layout**)right));
}

char String::cmpStatic(layout* left, layout* right) {
    if ( !left && !right ) {
        return 0;
    }
    if ( !left && right ) {
        return -1;
    }
    if ( left && !right ) {
        return 1;
    }

    int bytesPerLeft = left->bytes_per_codepoint;
    int bytesPerRight = right->bytes_per_codepoint;
    int commonCount = std::min(left->pointcount, right->pointcount);

    char res = 0;

    if (bytesPerLeft == 1 && bytesPerRight == 1) {
        res = byteCompare(left->data, right->data, bytesPerLeft * commonCount);
    } else if (bytesPerLeft == 1 && bytesPerRight == 2) {
        res = typedArrayCompare((uint8_t*)left->data, (uint16_t*)right->data, commonCount);
    } else if (bytesPerLeft == 1 && bytesPerRight == 4) {
        res = typedArrayCompare((uint8_t*)left->data, (uint32_t*)right->data, commonCount);
    } else if (bytesPerLeft == 2 && bytesPerRight == 1) {
        res = typedArrayCompare((uint16_t*)left->data, (uint8_t*)right->data, commonCount);
    } else if (bytesPerLeft == 2 && bytesPerRight == 2) {
        res = typedArrayCompare((uint16_t*)left->data, (uint16_t*)right->data, commonCount);
    } else if (bytesPerLeft == 2 && bytesPerRight == 4) {
        res = typedArrayCompare((uint16_t*)left->data, (uint32_t*)right->data, commonCount);
    } else if (bytesPerLeft == 4 && bytesPerRight == 1) {
        res = typedArrayCompare((uint32_t*)left->data, (uint8_t*)right->data, commonCount);
    } else if (bytesPerLeft == 4 && bytesPerRight == 2) {
        res = typedArrayCompare((uint32_t*)left->data, (uint16_t*)right->data, commonCount);
    } else if (bytesPerLeft == 4 && bytesPerRight == 4) {
        res = typedArrayCompare((uint32_t*)left->data, (uint32_t*)right->data, commonCount);
    } else {
        throw std::runtime_error("Nonsensical bytes-per-codepoint");
    }

    if (res) {
        return res;
    }

    if (left->pointcount < right->pointcount) {
        return -1;
    }

    if (left->pointcount > right->pointcount) {
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

void String::repr(instance_ptr self, ReprAccumulator& stream) {
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
    destroyStatic(self);
}

void String::destroyStatic(instance_ptr self) {
    if (!*(layout**)self) {
        return;
    }

    if ((*(layout**)self)->refcount.fetch_sub(1) == 1) {
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

void Bytes::repr(instance_ptr self, ReprAccumulator& stream) {
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

