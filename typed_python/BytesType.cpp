/******************************************************************************
   Copyright 2017-2019 typed_python Authors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

#include "AllTypes.hpp"

typed_python_hash_type BytesType::hash(instance_ptr left) {
    HashAccumulator acc((int)getTypeCategory());

    if (!(*(layout**)left)) {
        return 0x1234;
    }

    if ((*(layout**)left)->hash_cache == -1) {
        HashAccumulator acc((int)getTypeCategory());

        acc.addBytes(eltPtr(left, 0), count(left));

        (*(layout**)left)->hash_cache = acc.get();
        if ((*(layout**)left)->hash_cache == -1) {
            (*(layout**)left)->hash_cache = -2;
        }
    }

    return (*(layout**)left)->hash_cache;
}

bool BytesType::cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions) {
    if ( !(*(layout**)left) && !(*(layout**)right) ) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, 0);
    }
    if ( !(*(layout**)left) && (*(layout**)right) ) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, -1);
    }
    if ( (*(layout**)left) && !(*(layout**)right) ) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
    }
    if ( (*(layout**)left) == (*(layout**)right) ) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, 0);
    }

    char res = byteCompare(eltPtr(left, 0), eltPtr(right, 0), std::min(count(left), count(right)));

    if (res) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, res);
    }

    if (count(left) < count(right)) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, -1);
    }

    if (count(left) > count(right)) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
    }

    return cmpResultToBoolForPyOrdering(pyComparisonOp, 0);
}

// static
char BytesType::cmpStatic(layout* left, layout* right) {
    if ( !left && !right ) {
        return 0;
    }
    if ( !left && right ) {
        return -1;
    }
    if ( left && !right ) {
        return 1;
    }

    int commonCount = std::min(left->bytecount, right->bytecount);
    char res = 0;
    res = byteCompare(left->data, right->data, commonCount);
    if (res) {
        return res;
    }
    if (left->bytecount < right->bytecount) {
        return -1;
    }
    if (left->bytecount > right->bytecount) {
        return 1;
    }

    return 0;
}

/* static */
void BytesType::split(ListOfType::layout *outList, layout* bytesLayout, layout* sep, int64_t max) {
    static ListOfType* listOfBytes = ListOfType::Make(StringType::Make());

    int64_t cur = 0;
    int64_t count = 0;

    listOfBytes->reserve((instance_ptr)&outList, 10);

    uint8_t* bytesData = (uint8_t*)bytesLayout->data;
    uint8_t sepChar = sep ? *(uint8_t*)sep->data : 0;
    uint8_t* sepDat = sep ? (uint8_t*)sep->data : 0;
    int64_t sepLen = sep ? sep->bytecount : 1;

    if (max == 0) {
        layout* remainder = createFromPtr((const char*)bytesData, bytesLayout->bytecount);
        listOfBytes->append((instance_ptr)&outList, (instance_ptr)&remainder);
        destroyStatic((instance_ptr)&remainder);
        return;
    }

    while (cur < bytesLayout->bytecount) {
        int64_t match = cur;

        if (sep) {
            if (sepLen == 1) {
                while (match < bytesLayout->bytecount && bytesData[match] != sepChar) {
                    match++;
                }
            } else {
                while (match + sepLen <= bytesLayout->bytecount && strncmp((const char*)bytesData, (const char*)sepDat, match)) {
                    match++;
                }
            }
        } else {
            while (match < bytesLayout->bytecount && (
                    bytesData[match] != '\n'
                &&  bytesData[match] != '\r'
                &&  bytesData[match] != '\t'
                &&  bytesData[match] != ' '
                &&  bytesData[match] != '\b'
                &&  bytesData[match] != '\f'
                )
            ) {
                match++;
            }
        }

        if (match + sepLen > bytesLayout->bytecount) {
            break;
        }

        layout* piece = createFromPtr((const char*)bytesData + cur, match - cur);

        if (outList->count == outList->reserved) {
            listOfBytes->reserve((instance_ptr)&outList, outList->reserved * 1.5);
        }

        ((layout**)outList->data)[outList->count++] = piece;

        cur = match + sepLen;

        count++;

        if (max >= 0 && count >= max)
            break;
    }
    layout* remainder = createFromPtr((const char*)bytesData + cur, bytesLayout->bytecount - cur);
    listOfBytes->append((instance_ptr)&outList, (instance_ptr)&remainder);
    destroyStatic((instance_ptr)&remainder);
}

BytesType::layout* BytesType::concatenate(layout* lhs, layout* rhs) {
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

    layout* new_layout = (layout*)malloc(sizeof(layout) + rhs->bytecount + lhs->bytecount);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;
    new_layout->bytecount = lhs->bytecount + rhs->bytecount;

    memcpy(new_layout->data, lhs->data, lhs->bytecount);
    memcpy(new_layout->data + lhs->bytecount, rhs->data, rhs->bytecount);

    return new_layout;
}

BytesType::layout* BytesType::createFromPtr(const char* data, int64_t length) {
    layout* new_layout = (layout*)malloc(sizeof(layout) + length);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;
    new_layout->bytecount = length;

    memcpy(new_layout->data, data, length);

    return new_layout;
}

void BytesType::constructor(instance_ptr self, int64_t count, const char* data) const {
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

instance_ptr BytesType::eltPtr(instance_ptr self, int64_t i) const {
    //we don't want to have to return null here, but there is no actual memory to back this.
    if (*(layout**)self == nullptr) {
        return self;
    }

    return (*(layout**)self)->data + i;
}

int64_t BytesType::count(instance_ptr self) const {
    if (*(layout**)self == nullptr) {
        return 0;
    }

    return (*(layout**)self)->bytecount;
}

void BytesType::destroy(instance_ptr self) {
    destroyStatic(self);
}

void BytesType::destroyStatic(instance_ptr self) {
    if (!*(layout**)self) {
        return;
    }

    if ((*(layout**)self)->refcount.fetch_sub(1) == 1) {
        free((*(layout**)self));
    }
}

void BytesType::copy_constructor(instance_ptr self, instance_ptr other) {
    (*(layout**)self) = (*(layout**)other);
    if (*(layout**)self) {
        (*(layout**)self)->refcount++;
    }
}

void BytesType::assign(instance_ptr self, instance_ptr other) {
    layout* old = (*(layout**)self);

    (*(layout**)self) = (*(layout**)other);

    if (*(layout**)self) {
        (*(layout**)self)->refcount++;
    }

    destroy((instance_ptr)&old);
}

bool BytesType::isBinaryCompatibleWithConcrete(Type* other) {
    if (other->getTypeCategory() != m_typeCategory) {
        return false;
    }

    return true;
}

void BytesType::repr(instance_ptr self, ReprAccumulator& stream, bool isStr) {
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

// static
bool BytesType::to_int64(BytesType::layout* b, int64_t *value) {
    enum State {left_space, sign, digit, underscore, right_space, failed} state = left_space;
    int64_t value_sign = 1;
    *value = 0;

    if (b) {
        size_t bytecount = b->bytecount;
        uint8_t* data = b->data;

        for (int64_t i = 0; i < bytecount; i++) {
            uint8_t c = data[i];
            if (uprops[c] & Uprops_SPACE) {
                if (state == underscore || state == sign) {
                    state = failed;
                    break;
                }
                else if (state == digit) {
                    state = right_space;
                }
            }
            else if (c == '+' || c == '-') {
                if (state != left_space) {
                    state = failed;
                    break;
                }
                if (c == '-') {
                    value_sign = -1;
                }
                state = sign;
            }
            else if (c < 128 && c >= '0' && c <= '9') {
                if (state == right_space) {
                    state = failed;
                    break;
                }
                *value *= 10;
                *value += c - '0';
                state = digit;
            }
            else if (c == '_') {
                if (state != digit) {
                    state = failed;
                    break;
                }
                state = underscore;
            }
            else {
                state = failed;
                break;
            }
        }
    }
    if (state == digit || state == right_space) {
        *value *= value_sign;
        return true;
    }
    else {
        *value = 0;
        return false;
    }
}

// static
bool BytesType::to_float64(BytesType::layout* b, double* value) {
    enum State {left_space, sign, whole, underscore, decimal, mantissa, underscore_mantissa,
                exp, expsign, exponent, underscore_exponent,
                right_space, identifier, identifier_right_space, failed} state = left_space;
    const int MAX_FLOAT_STR = 48;
    char buf[MAX_FLOAT_STR + 1];
    int cur = 0;
    *value = 0.0;

    if (b) {
        size_t bytecount = b->bytecount;
        uint8_t* data = b->data;

        for (int64_t i = 0; i < bytecount; i++) {
            bool accumulate = true;
            uint8_t c = data[i];
            if (uprops[c] & Uprops_SPACE) {
                accumulate = false;
                if (state == underscore || state == underscore_exponent || state == underscore_mantissa
                        || state == sign || state == exp || state == expsign) {
                    state = failed;
                }
                else if (state == identifier || state == identifier_right_space) {
                    state = identifier_right_space;
                }
                else if (state == right_space || state == whole || state == decimal || state == mantissa || state == exponent) {
                    state = right_space;
                }
            }
            else if (c == '+' || c == '-') {
                if (state == left_space) {
                    state = sign;
                }
                else if (state == exp) {
                    state = expsign;
                }
                else {
                    state = failed;
                }
            }
            else if (c < 128 && c >= '0' && c <= '9') {
                if (state == decimal) {
                    state = mantissa;
                }
                else if (state == exp) {
                    state = exponent;
                }
                else if (state == right_space || state == identifier || state == identifier_right_space) {
                    state = failed;
                }
                else if (state == underscore_mantissa) {
                    state = mantissa;
                }
                else if (state == underscore_exponent) {
                    state = exponent;
                }
                else {
                    state = whole;
                }
            }
            else if (c == '.') {
                if (state == left_space || state == sign || state == whole) {
                    state = decimal;
                }
                else {
                    state = failed;
                }
            }
            else if (c == 'e' || c == 'E') {
                if (state == whole || state == decimal || state == mantissa) {
                    state = exp;
                }
                else {
                    state = failed;
                }
            }
            else if (c == '_') {
                accumulate = false;
                if (state == whole) {
                    state = underscore;
                }
                else if (state == mantissa) {
                    state = underscore_mantissa;
                }
                else if (state == exponent) {
                    state = underscore_exponent;
                }
                else {
                    state = failed;
                }
            }
            else if (c < 128) {
                if (state == left_space || state == sign || state == identifier) {
                    state = identifier;
                }
                else {
                    state = failed;
                }
            }
            else {
                state = failed;
            }
            if (state == failed) {
                break;
            }
            if (accumulate && cur < MAX_FLOAT_STR) {
                buf[cur++] = c;
            }
        }
    }
    buf[cur] = 0;
    if (state == identifier || state == identifier_right_space) {
        char* start = buf;
        if (*start == '+' || *start == '-') {
            start++;
        }
        for (char* p = start; *p; p++) {
            *p = tolower(*p);
        }
        if (strcmp(start, "inf") && strcmp(start, "infinity") && strcmp(start, "nan")) {
            state = failed;
        }
    }
    if (state == whole || state == decimal || state == mantissa || state == exponent || state == right_space
            || state == identifier || state == identifier_right_space) {
        char* endptr;
        *value = strtod(buf, &endptr);
        return true;
    }
    else {
        *value = 0.0;
        return false;
    }
}

BytesType::layout* BytesType::lower(layout *l) {
    if (!l) {
        return l;
    }

    int64_t new_byteCount = sizeof(layout) + l->bytecount;
    layout* new_layout = (layout*)malloc(new_byteCount);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;
    new_layout->bytecount = l->bytecount;

    for (uint8_t *src = l->data, *dest = new_layout->data, *end = src + l->bytecount; src < end; ) {
        *dest++ = (uint8_t)tolower(*src++);
    }

    return new_layout;
}

BytesType::layout* BytesType::upper(layout *l) {
    if (!l) {
        return l;
    }

    int64_t new_byteCount = sizeof(layout) + l->bytecount;
    layout* new_layout = (layout*)malloc(new_byteCount);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;
    new_layout->bytecount = l->bytecount;

    for (uint8_t *src = l->data, *dest = new_layout->data, *end = src + l->bytecount; src < end; ) {
        *dest++ = (uint8_t)toupper(*src++);
    }

    return new_layout;
}


bool matchchar(uint8_t v, bool whiteSpace, BytesType::layout* values) {
    if (whiteSpace) return isspace(v);

    if (!values) return false;

    for (int i = 0; i < values->bytecount; i++) {
        if (v == values->data[i]) return true;
    }
    return false;
}


BytesType::layout* BytesType::strip(layout* l, bool whiteSpace, layout* values, bool fromLeft, bool fromRight) {
    if (!l) {
        return l;
    }

    int64_t leftPos = 0;
    int64_t rightPos = l->bytecount;

    uint8_t* dataPtr = l->data;

    if (fromLeft) {
        while (leftPos < rightPos && matchchar(dataPtr[leftPos], whiteSpace, values)) {
            leftPos++;
        }
    }

    if (fromRight) {
        while (leftPos < rightPos && matchchar(dataPtr[rightPos-1], whiteSpace, values)) {
            rightPos--;
        }
    }

    if (leftPos == rightPos) {
        return nullptr;
    }

    if (leftPos == 0 && rightPos == l->bytecount) {
        l->refcount++;
        return l;
    }

    size_t datalength = rightPos - leftPos;
    layout* new_layout = (layout*)malloc(sizeof(layout) + datalength);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;
    new_layout->bytecount = datalength;

    memcpy(new_layout->data, l->data + leftPos, datalength);

    return new_layout;
}
