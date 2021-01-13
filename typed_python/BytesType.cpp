/******************************************************************************
   Copyright 2017-2020 typed_python Authors

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
// assumes outList was initialized to an empty list before calling
// x.split() with no parameters is not the same thing as x.split(b' ')
// x.split() combines successive matches (of whitespace) into a single match
void BytesType::split(ListOfType::layout *outList, layout* in, layout* sep, int64_t max) {
    static ListOfType* listOfBytes = ListOfType::Make(BytesType::Make());

    int64_t cur = 0;
    int64_t count = 0;

    listOfBytes->reserve((instance_ptr)&outList, 10);

    uint8_t* inData = in ? (uint8_t*)in->data : nullptr;
    int64_t inLen = in ? in->bytecount : 0;
    uint8_t sepChar = sep ? *(uint8_t*)sep->data : 0;
    uint8_t* sepDat = sep ? (uint8_t*)sep->data : 0;
    int64_t sepLen = sep ? sep->bytecount : 1;

    if (max == 0) {
        if (!sep) {
            while (cur < inLen && std::isspace(inData[cur])) cur++;
        }
        if (cur != inLen) {
            layout* remainder = createFromPtr((const char*)inData + cur, inLen - cur);
            listOfBytes->append((instance_ptr)&outList, (instance_ptr)&remainder);
            destroyStatic((instance_ptr)&remainder);
        }
        return;
    }

    while (cur < inLen) {
        int64_t match = cur;

        if (sep) {
            if (sepLen == 1) {
                while (match < inLen && inData[match] != sepChar) {
                    match++;
                }
            } else {
                while (match + sepLen <= inLen && memcmp((const char*)inData + match, (const char*)sepDat, sepLen)) {
                    match++;
                }
            }
        } else {
            while (match < inLen && !isspace(inData[match])) {
                match++;
            }
        }

        if (match + sepLen > inLen) {
            break;
        }

        if (sep || match != cur) {
            layout* piece = createFromPtr((const char*)inData + cur, match - cur);

            if (outList->count == outList->reserved) {
                listOfBytes->reserve((instance_ptr)&outList, outList->reserved * 1.5);
            }

            ((layout**)outList->data)[outList->count++] = piece;
            cur = match + sepLen;
            count++;
            if (max >= 0 && count >= max)
                break;
        }
        else if (!sep) {
            cur++;
        }
    }
//    if (!sep) {
//        while (cur < inLen && std::isspace(inData[cur])) cur++;
//    }
    if (sep || inLen != cur) {
        layout* remainder = createFromPtr((const char*)inData + cur, inLen - cur);
        listOfBytes->append((instance_ptr)&outList, (instance_ptr)&remainder);
        destroyStatic((instance_ptr)&remainder);
    }
}

/* static */
// assumes outList was initialized to an empty list before calling
void BytesType::rsplit(ListOfType::layout *outList, layout* in, layout* sep, int64_t max) {
    static ListOfType* listOfBytes = ListOfType::Make(BytesType::Make());

    listOfBytes->reserve((instance_ptr)&outList, 10);

    uint8_t* inData = in ? (uint8_t*)in->data : nullptr;
    int64_t inLen = in ? in->bytecount : 0;
    uint8_t sepChar = sep ? *(uint8_t*)sep->data : 0;
    uint8_t* sepDat = sep ? (uint8_t*)sep->data : 0;
    int64_t sepLen = sep ? sep->bytecount : 1;

    int64_t cur = inLen - 1;
    int64_t count = 0;

    if (max == 0) {
        if (!sep) {
            while (cur >= 0 && std::isspace(inData[cur])) cur--;
        }
        if (cur >= 0) {
            layout* remainder = createFromPtr((const char*)inData, cur + 1);
            listOfBytes->append((instance_ptr)&outList, (instance_ptr)&remainder);
            destroyStatic((instance_ptr)&remainder);
        }
        return;
    }

    while (cur >= 0) {
        int64_t match = cur;

        if (sep) {
            if (sepLen == 1) {
                while (match >= 0 && inData[match] != sepChar) {
                    match--;
                }
            } else {
                while (match - sepLen + 1 >= 0 && memcmp((const char*)inData + match - sepLen + 1, (const char*)sepDat, sepLen)) {
                    match--;
                }
            }
        } else {
            while (match >= 0 && (
                    inData[match] != '\n'
                &&  inData[match] != '\r'
                &&  inData[match] != '\t'
                &&  inData[match] != ' '
                &&  inData[match] != '\x0B'  // note: \x0B is whitespace, but \b is not whitespace
                &&  inData[match] != '\f'
                )
            ) {
                match--;
            }
        }

        if (match - sepLen + 1 < 0) {
            break;
        }

        if (sep || match != cur) {
            layout* piece = createFromPtr((const char*)inData + match + 1, cur - match);

            if (outList->count == outList->reserved) {
                listOfBytes->reserve((instance_ptr)&outList, outList->reserved * 1.5);
            }

            ((layout**)outList->data)[outList->count++] = piece;
            cur = match - sepLen;
            count++;
            if (max >= 0 && count >= max) break;
        }
        else if (!sep) {
            cur--;
        }
    }
    if (sep || cur != -1) {
        layout* remainder = createFromPtr((const char*)inData, cur + 1);
        listOfBytes->append((instance_ptr)&outList, (instance_ptr)&remainder);
        destroyStatic((instance_ptr)&remainder);
    }
    listOfBytes->reverse((instance_ptr)&outList);
}

/* static */
// assumes outList was initialized to an empty list before calling
void BytesType::splitlines(ListOfType::layout *outList, layout* in, bool keepends) {
    static ListOfType* listOfBytes = ListOfType::Make(BytesType::Make());

    int64_t cur = 0;

    listOfBytes->reserve((instance_ptr)&outList, 10);

    uint8_t* inData = in ? (uint8_t*)in->data : nullptr;
    int64_t inLen = in ? in->bytecount : 0;
    int sepLen = 0;

    while (cur < inLen) {
        int64_t match = cur;

        while (match < inLen && inData[match] != '\n' &&  inData[match] != '\r') {
            match++;
        }
        sepLen = 0;
        if (match < inLen) {
            if (inData[match] == '\r' && match + 1 < inLen && inData[match+1] == '\n') {
                sepLen = 2;
            }
            else {
                sepLen = 1;
            }
        }

        if (match + sepLen > inLen) {
            break;
        }

        layout* piece = createFromPtr((const char*)inData + cur, match - cur + (keepends ? sepLen : 0));

        if (outList->count == outList->reserved) {
            listOfBytes->reserve((instance_ptr)&outList, outList->reserved * 1.5);
        }

        ((layout**)outList->data)[outList->count++] = piece;
        cur = match + sepLen;
    }
    if (inLen != cur) {
        layout* remainder = createFromPtr((const char*)inData + cur, inLen - cur);
        listOfBytes->append((instance_ptr)&outList, (instance_ptr)&remainder);
        destroyStatic((instance_ptr)&remainder);
    }
}

void BytesType::join(BytesType::layout **out, BytesType::layout *separator, ListOfType::layout *toJoin) {
    if (!out)
        throw std::invalid_argument("missing return argument");

    BytesType::Make()->constructor((instance_ptr)out);

    // return empty bytes when there is nothing to join
    if (toJoin->count == 0)
    {
        return;
    }

    static ListOfType *listOfBytesType = ListOfType::Make(BytesType::Make());
    int total_bytes = 0;

    for (int64_t i = 0; i < toJoin->count; i++)
    {
        instance_ptr item = listOfBytesType->eltPtr(toJoin, i);
        BytesType::layout** itemLayout = (BytesType::layout**)item;
        if (*itemLayout != nullptr) {
            total_bytes += (*itemLayout)->bytecount;
        }
    }

    // add the separators size
    if (separator != nullptr)
    {
        total_bytes += separator->bytecount * (toJoin->count - 1);
    }

    // add all the parts together
    *out = (layout*)tp_malloc(sizeof(layout) + total_bytes);
    (*out)->hash_cache = -1;
    (*out)->refcount = 1;
    (*out)->bytecount = total_bytes;

    // position in the output data array
    int position = 0;

    for (int64_t i = 0; i < toJoin->count; i++)
    {
        instance_ptr instptr_item = listOfBytesType->eltPtr(toJoin, i);
        BytesType::layout* item = *(BytesType::layout**)instptr_item;
        if (item != nullptr)
        {
            memcpy((*out)->data + position, item->data, item->bytecount);
            position += item->bytecount;
        }

        if (separator != nullptr && i != toJoin->count - 1)
        {
            memcpy((*out)->data + position, separator->data, separator->bytecount);
            position += separator->bytecount;
        }
    }
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

    layout* new_layout = (layout*)tp_malloc(sizeof(layout) + rhs->bytecount + lhs->bytecount);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;
    new_layout->bytecount = lhs->bytecount + rhs->bytecount;

    memcpy(new_layout->data, lhs->data, lhs->bytecount);
    memcpy(new_layout->data + lhs->bytecount, rhs->data, rhs->bytecount);

    return new_layout;
}

BytesType::layout* BytesType::createFromPtr(const char* data, int64_t length) {
    layout* new_layout = (layout*)tp_malloc(sizeof(layout) + length);
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
    (*(layout**)self) = (layout*)tp_malloc(sizeof(layout) + count);

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
        tp_free((*(layout**)self));
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
    layout* new_layout = (layout*)tp_malloc(new_byteCount);
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
    layout* new_layout = (layout*)tp_malloc(new_byteCount);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;
    new_layout->bytecount = l->bytecount;

    for (uint8_t *src = l->data, *dest = new_layout->data, *end = src + l->bytecount; src < end; ) {
        *dest++ = (uint8_t)toupper(*src++);
    }
    return new_layout;
}

BytesType::layout* BytesType::capitalize(layout *l) {
    if (!l) {
        return l;
    }

    int64_t new_byteCount = sizeof(layout) + l->bytecount;
    layout* new_layout = (layout*)tp_malloc(new_byteCount);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;
    new_layout->bytecount = l->bytecount;

    uint8_t *src = l->data, *dest = new_layout->data, *end = src + l->bytecount;
    if (l->bytecount) {
        *dest++ = (uint8_t)toupper(*src++);
        while (src < end) {
            *dest++ = (uint8_t)tolower(*src++);
        }
    }
    return new_layout;
}

BytesType::layout* BytesType::swapcase(layout *l) {
    if (!l) {
        return l;
    }

    int64_t new_byteCount = sizeof(layout) + l->bytecount;
    layout* new_layout = (layout*)tp_malloc(new_byteCount);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;
    new_layout->bytecount = l->bytecount;

    for (uint8_t *src = l->data, *dest = new_layout->data, *end = src + l->bytecount; src < end; ) {
        uint8_t c = *src++;
        if (islower(c))
            *dest++ = (uint8_t)toupper(c);
        else if (isupper(c))
            *dest++ = (uint8_t)tolower(c);
        else
            *dest++ = c;
    }
    return new_layout;
}

BytesType::layout* BytesType::title(layout *l) {
    if (!l) {
        return l;
    }

    int64_t new_byteCount = sizeof(layout) + l->bytecount;
    layout* new_layout = (layout*)tp_malloc(new_byteCount);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;
    new_layout->bytecount = l->bytecount;

    bool word = false;
    for (uint8_t *src = l->data, *dest = new_layout->data, *end = src + l->bytecount; src < end; ) {
        uint8_t c = *src++;
        if (word) {
            if (isalpha(c)) {
                *dest++ = (uint8_t)tolower(c);
            } else {
                *dest++ = c;
                word = false;
            }
        } else {
            if (isalpha(c)) {
                *dest++ = (uint8_t)toupper(c);
                word = true;
            } else {
                *dest++ = c;
            }
        }
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
    layout* new_layout = (layout*)tp_malloc(sizeof(layout) + datalength);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;
    new_layout->bytecount = datalength;

    memcpy(new_layout->data, l->data + leftPos, datalength);

    return new_layout;
}

BytesType::layout* BytesType::mult(layout* lhs, int64_t rhs) {
    if (!lhs) {
        return lhs;
    }
    if (rhs <= 0)
        return 0;
    int64_t new_length = lhs->bytecount * rhs;
    int64_t new_byteCount = sizeof(layout) + new_length;

    layout* new_layout = (layout*)tp_malloc(new_byteCount);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;
    new_layout->bytecount = new_length;

    for (size_t i = 0; i < rhs; i++) {
        memcpy(new_layout->data + i * lhs->bytecount, lhs->data, lhs->bytecount);
    }

    return new_layout;
}

// This 'replace' is slightly faster (5%) than 'bytes_replace' in bytes_wrapper.py.
// However, this 'replace' needs better memory management.
// Some notable special cases:
// 'aa'.replace('','z') = 'zazaz'
// 'aa'.replace('','z', 5) = 'aa'
// 'aa'.replace('','z', -1) = 'zazaz'
// ''.replace('','z') = 'z'
// ''.replace('','z', 5) = ''
// ''.replace('','z', -1) = 'z'
BytesType::layout* BytesType::replace(layout* l, layout* old, layout* repl, int64_t count) {
    if (!l) {
        if (old || !repl || count >= 0)
            return 0;
        layout *new_layout = (layout*)tp_malloc(sizeof(layout) + repl->bytecount);
        new_layout->refcount = 1;
        new_layout->hash_cache = -1;
        new_layout->bytecount = repl->bytecount;
        memcpy(new_layout->data, repl->data, repl->bytecount);
        return new_layout;
    }

    int64_t c = 0;
    size_t repl_len = repl ? repl->bytecount : 0;
    size_t old_len = old ? old->bytecount : 0;
    size_t max_matches = old_len ? l->bytecount / old_len : l->bytecount + 1;
    size_t max_increase = repl_len > old_len ? max_matches * (repl_len - old_len) : 0;
    size_t new_layout_size = sizeof(layout) + l->bytecount + max_increase;
    layout* new_layout = (layout*)tp_malloc(new_layout_size);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;
    new_layout->bytecount = l->bytecount;

    if ((!old_len && !repl_len) || old_len > l->bytecount || count == 0) {
         memcpy(new_layout->data, l->data, l->bytecount);
         return new_layout;
    }

    uint8_t* src = l->data;
    uint8_t* dst = new_layout->data;
    uint8_t* end_scan = l->data + l->bytecount - old_len + 1;
    uint8_t* end_src = l->data + l->bytecount;
    while (src < end_scan) {
        if (!old_len || 0 == memcmp(src, old->data, old_len)) {
            if (repl) {
                memcpy(dst, repl->data, repl_len);
                dst += repl_len;
            }
            if (!old_len) {
                *dst++ = *src++;
            }
            else {
                src += old_len;
            }
            new_layout->bytecount += repl_len - old_len;
            if (count > 0 && ++c >= count) break;
        }
        else {
            *dst++ = *src++;
        }
    }
    if (src < end_src) memcpy(dst, src, end_src - src);

    if (sizeof(layout) + new_layout->bytecount < new_layout_size) {
        new_layout = (layout*)tp_realloc(
            new_layout,
            new_layout_size,
            sizeof(layout) + new_layout->bytecount
        );
    }
    return new_layout;
}


BytesType::layout* BytesType::translate(layout* l, layout* table, layout* to_delete) {
    if (!l) return 0;

    if (table && table->bytecount != 256) {
        throw std::invalid_argument("translation table must be 256 characters long");
    }

    layout *new_layout = (layout*)tp_malloc(sizeof(layout) + l->bytecount);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;

    uint8_t* dst = new_layout->data;
    for (int i = 0; i < l->bytecount; i++) {
        uint8_t c = l->data[i];
        if (to_delete) {
            bool delete_this = false;
            for (int j = 0; j < to_delete->bytecount; j++)
                if (c == to_delete->data[j]) {
                    delete_this = true;
                    break;
                }
            if (delete_this) continue;
        }
        if (table)
            *dst++ = table->data[c];
        else
            *dst++ = c;
    }
    new_layout->bytecount = dst - new_layout->data;

    // might have shrunk
    if (new_layout->bytecount < l->bytecount) {
        new_layout = (layout*)tp_realloc(
            new_layout,
            sizeof(layout) + l->bytecount,
            sizeof(layout) + new_layout->bytecount
        );
    }
    return new_layout;
}

// assumes len(from) == len(to) checked before calling
BytesType::layout* BytesType::maketrans(layout* from, layout* to) {
    const int table_size = 256;
    layout* new_layout = (layout*)tp_malloc(sizeof(layout) + table_size);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;
    new_layout->bytecount = table_size;

    for (int i = 0; i < table_size; i++) {
        new_layout->data[i] = i;
    }

    for (int j = 0; j < (from ? from->bytecount : 0); j++) {
        new_layout->data[from->data[j]] = to->data[j];
    }

    return new_layout;
}
