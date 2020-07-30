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
#include  <iostream>
#include "UnicodeProps.hpp"
using namespace std;

StringType::layout* StringType::upgradeCodePoints(layout* lhs, int32_t newBytesPerCodepoint) {
    if (!lhs) {
        return lhs;
    }

    if (newBytesPerCodepoint == lhs->bytes_per_codepoint) {
        lhs->refcount++;
        return lhs;
    }

    int64_t new_byteCount = sizeof(layout) + lhs->pointcount * newBytesPerCodepoint;

    layout* new_layout = (layout*)tp_malloc(new_byteCount);
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

StringType::layout* StringType::concatenate(layout* lhs, layout* rhs) {
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
        StringType::destroyStatic((instance_ptr)&newLayout);
        return result;
    }

    if (rhs->bytes_per_codepoint < lhs->bytes_per_codepoint) {
        layout* newLayout = upgradeCodePoints(rhs, lhs->bytes_per_codepoint);

        layout* result = concatenate(lhs, newLayout);
        StringType::destroyStatic((instance_ptr)&newLayout);
        return result;
    }

    //they're the same
    int64_t new_byteCount = sizeof(layout) + (rhs->pointcount + lhs->pointcount) * lhs->bytes_per_codepoint;

    layout* new_layout = (layout*)tp_malloc(new_byteCount);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;
    new_layout->bytes_per_codepoint = lhs->bytes_per_codepoint;
    new_layout->pointcount = lhs->pointcount + rhs->pointcount;

    memcpy(new_layout->data, lhs->data, lhs->pointcount * lhs->bytes_per_codepoint);
    memcpy(new_layout->data + lhs->pointcount * lhs->bytes_per_codepoint,
        rhs->data, rhs->pointcount * lhs->bytes_per_codepoint);

    return new_layout;
}

StringType::layout* StringType::lower(layout *l) {
    if (!l) {
        return l;
    }

    int64_t new_byteCount = sizeof(layout) + l->pointcount * l->bytes_per_codepoint;
    layout* new_layout = (layout*)tp_malloc(new_byteCount);
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

StringType::layout* StringType::upper(layout *l) {
    if (!l) {
        return l;
    }

    int64_t new_byteCount = sizeof(layout) + l->pointcount * l->bytes_per_codepoint;
    layout* new_layout = (layout*)tp_malloc(new_byteCount);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;
    new_layout->bytes_per_codepoint = l->bytes_per_codepoint;
    new_layout->pointcount = l->pointcount;

    if (l->bytes_per_codepoint == 1) {
        for (uint8_t *src = l->data, *dest = new_layout->data, *end = src + l->pointcount; src < end; ) {
            *dest++ = (uint8_t)toupper(*src++);
        }
    }
    else if (l->bytes_per_codepoint == 2) {
        for (uint16_t *src = (uint16_t *)l->data, *dest = (uint16_t *)new_layout->data, *end = src + l->pointcount; src < end; ) {
            *dest++ = (uint16_t)towupper(*src++);
        }
    }
    else if (l->bytes_per_codepoint == 4) {
        for (uint32_t *src = (uint32_t *)l->data, *dest = (uint32_t *)new_layout->data, *end = src + l->pointcount; src < end; ) {
            *dest++ = (uint32_t)towupper(*src++);
        }
    }

    return new_layout;
}

uint32_t StringType::getpoint(layout *l, uint64_t i) {
    if (l->bytes_per_codepoint == 1)
        return ((uint8_t *)l->data)[i];
    else if (l->bytes_per_codepoint == 2)
        return ((uint16_t *)l->data)[i];
    else if (l->bytes_per_codepoint == 4)
        return ((uint32_t *)l->data)[i];
    return 0;
}

int64_t StringType::find(layout *l, layout *sub, int64_t start, int64_t stop) {
    if (!l || !l->pointcount) {
        if (!sub || !sub->pointcount)
            return start > 0 ? -1 : 0;
        return -1;
    }
    if (start < 0) {
        start += l->pointcount;
        if (start < 0) start = 0;
    }
    if (stop < 0) {
        stop += l->pointcount;
        if (stop < 0) stop = 0;
    }
    if (stop < start || start > l->pointcount)
        return -1;
    if (!sub || !sub->pointcount)
        return start >= 0 ? start : 0;

    if (stop > l->pointcount)
        stop = l->pointcount;

    stop -= (sub->pointcount - 1);

    if (start < 0 || stop < 0 || start >= stop || sub->pointcount > l->pointcount || start > l->pointcount - sub->pointcount)
        return -1;

    if (l->bytes_per_codepoint == 1 and sub->bytes_per_codepoint == 1 && sub->pointcount == 1) {
        const uint8_t* lPtr = (const uint8_t*)l->data;
        const uint8_t subChar = sub->data[0];

        for (int64_t i = start; i < stop; i++) {
            if (lPtr[i] == subChar) {
                return i;
            }
        }
        return -1;
    }

    for (int64_t i = start; i < stop; i++) {
        bool match = true;
        for (int64_t j = 0; j < sub->pointcount; j++) {
            if (getpoint(l, i+j) != getpoint(sub, j)) {
                match = false;
                break;
            }
        }
        if (match)
            return i;
    }

    return -1;
}

void StringType::split_3(ListOfType::layout* outList, layout* l, int64_t max) {
    if (!outList)
        throw std::invalid_argument("missing return argument");

    static ListOfType* listofstring = ListOfType::Make(StringType::Make());
    listofstring->resize((instance_ptr)&outList, 0, 0);

    if (!l || !l->pointcount)
        ;
    else if (max == 0) {
        // unexpected standard behavior:
        //   "   abc   ".split(maxsplit=0) = "abc   "  *** not "abc" nor "   abc   " ***
        int64_t cur = 0;
        while (cur < l->pointcount && uprops[getpoint(l, cur)] & Uprops_SPACE)
            cur++;
        layout* remainder = getsubstr(l, cur, l->pointcount);
        listofstring->append((instance_ptr)&outList, (instance_ptr)&remainder);
        destroyStatic((instance_ptr)&remainder);
    }
    else {
        int64_t cur = 0;
        int64_t count = 0;
        while (cur < l->pointcount) {
            int64_t match = cur;
            while (!(uprops[getpoint(l, match)] & Uprops_SPACE)) {
                match++;
                if (match >= l->pointcount) {
                    match = -1;
                    break;
                }
            }
            if (match < 0)
                break;
            if (cur != match) {
                layout* piece = getsubstr(l, cur, match);
                listofstring->append((instance_ptr)&outList, (instance_ptr)&piece);
                destroyStatic((instance_ptr)&piece);
                count++;
            }
            cur = match + 1;
            if (max >= 0 && count >= max)
                break;
        }
        while (cur < l->pointcount && uprops[getpoint(l, cur)] & Uprops_SPACE)
            cur++;
        if (cur < l->pointcount) {
            layout* remainder = getsubstr(l, cur, l->pointcount);
            listofstring->append((instance_ptr)&outList, (instance_ptr)&remainder);
            destroyStatic((instance_ptr)&remainder);
        }
    }
    // to force a refcount error, uncomment the line below
    //listofstring->copy_constructor((instance_ptr)&outList, (instance_ptr)&outList);
}

void StringType::split(ListOfType::layout* outList, layout* l, layout* sep, int64_t max) {
    if (!outList)
        throw std::invalid_argument("missing return argument");

    static ListOfType* listofstring = ListOfType::Make(StringType::Make());
    listofstring->resize((instance_ptr)&outList, 0, 0);

    if (!sep || !sep->pointcount) {
        throw std::invalid_argument("ValueError: empty separator");
    }
    else if (!l || !l->pointcount || max == 0) {
        listofstring->append((instance_ptr)&outList, (instance_ptr)&l);
    }
    else if (l->bytes_per_codepoint == 1 and sep->bytes_per_codepoint == 1 and sep->pointcount == 1) {
        int64_t cur = 0;
        int64_t count = 0;

        listofstring->reserve((instance_ptr)&outList, 10);

        uint8_t* lData = (uint8_t*)l->data;
        uint8_t sepChar = *(uint8_t*)sep->data;

        while (cur < l->pointcount) {
            int64_t match = cur;

            while (match < l->pointcount && lData[match] != sepChar) {
                match++;
            }

            if (match >= l->pointcount) {
                break;
            }

            layout* piece = getsubstr(l, cur, match);

            if (outList->count == outList->reserved) {
                listofstring->reserve((instance_ptr)&outList, outList->reserved * 1.5);
            }

            ((layout**)outList->data)[outList->count++] = piece;

            cur = match + 1;

            count++;

            if (max >= 0 && count >= max)
                break;
        }
        layout* remainder = getsubstr(l, cur, l->pointcount);
        listofstring->append((instance_ptr)&outList, (instance_ptr)&remainder);
        destroyStatic((instance_ptr)&remainder);
    }
    else {
        int64_t cur = 0;
        int64_t count = 0;

        listofstring->reserve((instance_ptr)&outList, 10);

        while (cur < l->pointcount) {
            int64_t match = find(l, sep, cur, l->pointcount);
            if (match < 0)
                break;

            layout* piece = getsubstr(l, cur, match);

            if (outList->count == outList->reserved) {
                listofstring->reserve((instance_ptr)&outList, outList->reserved * 1.5);
            }

            ((layout**)outList->data)[outList->count++] = piece;

            cur = match + sep->pointcount;
            count++;
            if (max >= 0 && count >= max)
                break;
        }
        layout* remainder = getsubstr(l, cur, l->pointcount);
        listofstring->append((instance_ptr)&outList, (instance_ptr)&remainder);
        destroyStatic((instance_ptr)&remainder);
    }
}

bool StringType::isalpha(layout *l) {
    if (!l || !l->pointcount)
        return false;
    for (int64_t i = 0; i < l->pointcount; i++)
        if (!(uprops[getpoint(l, i)] & Uprops_ALPHA))
            return false;
    return true;
}

bool StringType::isalnum(layout *l) {
    if (!l || !l->pointcount)
        return false;
    for (int64_t i = 0; i < l->pointcount; i++) {
        auto flags = uprops[getpoint(l, i)];
        if (!(flags & Uprops_ALPHA)
            && !(flags & Uprops_NUMERIC)
            && !(flags & Uprops_DECIMAL)
            && !(flags & Uprops_DIGIT)) // for completeness; looks to me like DECIMAL and DIGIT are subsets of NUMERIC
            return false;
    }
    return true;
}

bool StringType::isdecimal(layout *l) {
    if (!l || !l->pointcount)
        return false;
    for (int64_t i = 0; i < l->pointcount; i++)
        if (!(uprops[getpoint(l, i)] & Uprops_DECIMAL))
            return false;
    return true;
}

bool StringType::isdigit(layout *l) {
    if (!l || !l->pointcount)
        return false;
    for (int64_t i = 0; i < l->pointcount; i++) {
        auto flags = uprops[getpoint(l, i)];
        if (!(flags & Uprops_DECIMAL)
            && !(flags & Uprops_DIGIT)) // For completeness; looks to me like DECIMAL is a subset of DIGIT
            return false;
    }
    return true;
}

bool StringType::islower(layout *l) {
    if (!l || !l->pointcount)
        return false;
    bool found_one = false;
    for (int64_t i = 0; i < l->pointcount; i++) {
        auto flags = uprops[getpoint(l, i)];
        if (flags & Uprops_UPPER)
            return false;
        if (flags & Uprops_TITLE)
            return false;
        if (flags & Uprops_LOWER)
            found_one = true;
    }
    return found_one;
}

bool StringType::isnumeric(layout *l) {
    if (!l || !l->pointcount)
        return false;
    for (int64_t i = 0; i < l->pointcount; i++) {
        auto flags = uprops[getpoint(l, i)];
        if (!(flags & Uprops_DECIMAL)
            && !(flags & Uprops_DIGIT)
            && !(flags & Uprops_NUMERIC))
            return false;
    }
    return true;
}

bool StringType::isprintable(layout *l) {
    if (!l || !l->pointcount)
        return true;
    for (int64_t i = 0; i < l->pointcount; i++)
        if (!(uprops[getpoint(l, i)] & Uprops_PRINTABLE))
            return false;
    return true;
}

bool StringType::isspace(layout *l) {
    if (!l || !l->pointcount)
        return false;
    for (int64_t i = 0; i < l->pointcount; i++)
        if (!(uprops[getpoint(l, i)] & Uprops_SPACE))
            return false;
    return true;
}

bool StringType::istitle(layout *l) {
    if (!l || !l->pointcount)
        return false;
    bool last_cased = false;
    bool found_one = false;
    for (int64_t i = 0; i < l->pointcount; i++) {
        auto flags = uprops[getpoint(l, i)];
        bool upper = flags & Uprops_UPPER;
        bool lower = flags & Uprops_LOWER;
        bool title = flags & Uprops_TITLE;
        if (upper && last_cased)
            return false;
        if (lower && !last_cased)
            return false;
        if (title && last_cased)
            return false;
        last_cased = upper || lower || title;
        if (last_cased)
            found_one = true;
    }

    return found_one;
}

bool StringType::isupper(layout *l) {
    if (!l || !l->pointcount)
        return false;
    bool found_one = false;
    for (int64_t i = 0; i < l->pointcount; i++) {
        auto flags = uprops[getpoint(l, i)];
        if (flags & Uprops_LOWER)
            return false;
        if (flags & Uprops_TITLE)
            return false;
        if (flags & Uprops_UPPER)
            found_one = true;
    }
    return found_one;
}

StringType::layout* StringType::singleFromCodepoint(int64_t codePoint) {
    int bytesPerCodepoint;
    if (codePoint <= 0xFF) {
        bytesPerCodepoint = 1;
    } else if (codePoint <= 0xFFFF) {
        bytesPerCodepoint = 2;
    } else {
        bytesPerCodepoint = 4;
    }

    int64_t new_byteCount = sizeof(layout) + bytesPerCodepoint;

    layout* new_layout = (layout*)tp_malloc(new_byteCount);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;
    new_layout->bytes_per_codepoint = bytesPerCodepoint;
    new_layout->pointcount = 1;

    //we could figure out if we can represent this with a smaller encoding.
    if (bytesPerCodepoint == 1) {
        ((uint8_t*)new_layout->data)[0] = codePoint;
    }
    if (bytesPerCodepoint == 2) {
        ((uint16_t*)new_layout->data)[0] = codePoint;
    }
    if (bytesPerCodepoint == 4) {
        ((uint32_t*)new_layout->data)[0] = codePoint;
    }

    return new_layout;
}


uint32_t StringType::getord(layout* lhs) {
    if (!lhs) {
        return 0;
    }

    if (lhs->pointcount != 1) {
        return 0;
    }

    if (lhs->bytes_per_codepoint == 1) {
        return ((uint8_t*)lhs->data)[0];
    }
    if (lhs->bytes_per_codepoint == 2) {
        return ((uint16_t*)lhs->data)[0];
    }
    if (lhs->bytes_per_codepoint == 4) {
        return ((uint32_t*)lhs->data)[0];
    }

    return 0;
}

StringType::layout* StringType::getitem(layout* lhs, int64_t offset) {
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

    layout* new_layout = (layout*)tp_malloc(new_byteCount);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;
    new_layout->bytes_per_codepoint = lhs->bytes_per_codepoint;
    new_layout->pointcount = 1;

    //we could figure out if we can represent this with a smaller encoding.
    if (new_layout->bytes_per_codepoint == 1) {
        ((uint8_t*)new_layout->data)[0] = ((uint8_t*)lhs->data)[offset];
    }
    if (new_layout->bytes_per_codepoint == 2) {
        ((uint16_t*)new_layout->data)[0] = ((uint16_t*)lhs->data)[offset];
    }
    if (new_layout->bytes_per_codepoint == 4) {
        ((uint32_t*)new_layout->data)[0] = ((uint32_t*)lhs->data)[offset];
    }

    return new_layout;
}

StringType::layout* StringType::getsubstr(layout* l, int64_t start, int64_t stop) {
    if (!l) {
        return l;
    }

    if (start < 0) {
        start += l->pointcount;
        if (start < 0) start = 0;
    }
    if (stop < 0) {
        stop += l->pointcount;
        if (stop < 0) stop = 0;
    }
    if (stop < start || start > l->pointcount)
        stop = start = 0;

    if (stop > l->pointcount)
        stop = l->pointcount;

    if (start >= stop) {
        return nullptr;
    }

    size_t datalength = stop - start;
    size_t datasize = datalength * l->bytes_per_codepoint;
    int64_t new_byteCount = sizeof(layout) + datasize;

    layout* new_layout = (layout*)tp_malloc(new_byteCount);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;
    new_layout->bytes_per_codepoint = l->bytes_per_codepoint;
    new_layout->pointcount = datalength;

    memcpy(new_layout->data, l->data + start * l->bytes_per_codepoint, datalength);

    return new_layout;
}

// specialized form of the strip left/right algorithm based on the codepoint type.
template<class elt_type>
void stripInset(int64_t& ioLeftPos, int64_t& ioRightPos, elt_type* data, bool fromLeft, bool fromRight) {
    if (fromLeft) {
        while (ioLeftPos < ioRightPos && uprops[data[ioLeftPos]] & Uprops_SPACE) {
            ioLeftPos++;
        }
    }

    if (fromRight) {
        while (ioLeftPos < ioRightPos && uprops[data[ioRightPos-1]] & Uprops_SPACE) {
            ioRightPos--;
        }
    }
}

StringType::layout* StringType::strip(layout* l, bool fromLeft, bool fromRight) {
    if (!l) {
        return l;
    }

    int64_t leftPos = 0;
    int64_t rightPos = l->pointcount;

    uint8_t* dataPtr = l->data;

    if (l->bytes_per_codepoint == 1) {
        stripInset(leftPos, rightPos, (uint8_t*)dataPtr, fromLeft, fromRight);
    }
    else if (l->bytes_per_codepoint == 2) {
        stripInset(leftPos, rightPos, (uint16_t*)dataPtr, fromLeft, fromRight);
    }
    else if (l->bytes_per_codepoint == 4) {
        stripInset(leftPos, rightPos, (uint32_t*)dataPtr, fromLeft, fromRight);
    }

    if (leftPos == rightPos) {
        return nullptr;
    }

    if (leftPos == 0 && rightPos == l->pointcount) {
        l->refcount++;
        return l;
    }

    return getsubstr(l, leftPos, rightPos);
}

StringType::layout* StringType::lstrip(layout* l) {
    return strip(l, true, false);
}

StringType::layout* StringType::rstrip(layout* l) {
    return strip(l, false, true);
}


int64_t StringType::bytesPerCodepointRequiredForUtf8(const uint8_t* utf8Str, int64_t length) {
    int64_t bytes_per_codepoint = 1;
    while (length > 0) {
        if (utf8Str[0] >> 7 == 0) {
            //one byte encoded here
            length -= 1;
            utf8Str++;
        } else if (utf8Str[0] >> 5 == 0b110) {
            length -= 1;
            utf8Str+=2;
            bytes_per_codepoint = std::max<int64_t>(2, bytes_per_codepoint);
        } else if (utf8Str[0] >> 4 == 0b1110) {
            length -= 1;
            utf8Str += 3;
            bytes_per_codepoint = std::max<int64_t>(2, bytes_per_codepoint);
        } else if (utf8Str[0] >> 3 == 0b11110) {
            length -= 1;
            utf8Str+=4;
            bytes_per_codepoint = std::max<int64_t>(4, bytes_per_codepoint);
        } else {
            throw std::runtime_error("Improperly formatted unicode string.");
        }
    }
    return bytes_per_codepoint;
}

size_t StringType::countUtf8Codepoints(const uint8_t*utfData, size_t bytecount) {
    size_t result = 0;
    size_t k = 0;

    while (k < bytecount) {
        result += 1;

        if (utfData[k] >> 7 == 0) {
            k += 1;
        } else if ((utfData[k] >> 5) == 0b110) {
            k += 2;
        } else if ((utfData[k] >> 4) == 0b1110) {
            k += 3;
        } else if ((utfData[k] >> 3) == 0b11110) {
            k += 4;
        } else {
            throw std::runtime_error("corrupt utf8 data stream.");
        }
    }

    return result;
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

void StringType::decodeUtf8To(uint8_t* target, uint8_t* utf8Str, int64_t bytes_per_codepoint, int64_t length) {
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

StringType::layout* StringType::createFromUtf8(const char* utfEncodedString, int64_t length) {
    if (!length) {
        return nullptr;
    }

    int64_t bytes_per_codepoint = bytesPerCodepointRequiredForUtf8((uint8_t*)utfEncodedString, length);

    int64_t new_byteCount = sizeof(layout) + length * bytes_per_codepoint;

    layout* new_layout = (layout*)tp_malloc(new_byteCount);
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

bool StringType::cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions) {
    if (pyComparisonOp == Py_NE) {
        return !cmp(left, right, Py_EQ, suppressExceptions);
    }
    if (pyComparisonOp == Py_EQ) {
        return cmpStaticEq(*(layout**)left, *(layout**)right);
    }
    return cmpResultToBoolForPyOrdering(pyComparisonOp, cmpStatic(*(layout**)left, *(layout**)right));
}

bool StringType::cmpStaticEq(layout* left, layout* right) {
    if ( !left && !right ) {
        return true;
    }
    if ( !left && right ) {
        return false;
    }
    if ( left && !right ) {
        return false;
    }

    if (left->pointcount != right->pointcount) {
        return false;
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

    return res == 0;
}

char StringType::cmpStatic(layout* left, layout* right) {
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

void StringType::constructor(instance_ptr self, int64_t bytes_per_codepoint, int64_t count, const char* data) const {
    if (count == 0) {
        *(layout**)self = nullptr;
        return;
    }

    (*(layout**)self) = (layout*)tp_malloc(sizeof(layout) + count * bytes_per_codepoint);

    (*(layout**)self)->bytes_per_codepoint = bytes_per_codepoint;
    (*(layout**)self)->pointcount = count;
    (*(layout**)self)->hash_cache = -1;
    (*(layout**)self)->refcount = 1;

    if (data) {
        ::memcpy((*(layout**)self)->data, data, count * bytes_per_codepoint);
    }
}

void StringType::repr(instance_ptr self, ReprAccumulator& stream, bool isStr) {
    if (isStr) {
        // isStr=true should only come at the top-level of stringification. Composite
        // types always repr their internals, and so we should never see this input
        // because the interpreter would never need to call this function since it
        // would have an actual pystring instead.
        throw std::runtime_error("StringType::repr shouldn't ever get 'isStr=true'");
    }

    stream << "\"";

    //as if it were bytes, which is totally wrong...
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
            if (base[k] == '\n') {
                stream << "\\n";
            } else
            if (base[k] == '\r') {
                stream << "\\r";
            } else
            if (base[k] == '\t') {
                stream << "\\t";
            } else {
                stream << "\\x" << hexDigits[base[k] / 16] << hexDigits[base[k] % 16];
            }
        }
    }

    stream << "\"";
}

instance_ptr StringType::eltPtr(instance_ptr self, int64_t i) const {
    const static char* emptyPtr = "";

    if (*(layout**)self == nullptr) {
        return (instance_ptr)emptyPtr;
    }

    return (*(layout**)self)->eltPtr(i);
}

int64_t StringType::bytes_per_codepoint(instance_ptr self) const {
    if (*(layout**)self == nullptr) {
        return 1;
    }

    return (*(layout**)self)->bytes_per_codepoint;
}

int64_t StringType::count(instance_ptr self) const {
    return countStatic(*(layout**)self);
}

int64_t StringType::countStatic(StringType::layout* self) {
    if (self == nullptr) {
        return 0;
    }

    return self->pointcount;
}

void StringType::destroy(instance_ptr self) {
    destroyStatic(self);
}

void StringType::destroyStatic(instance_ptr self) {
    if (!*(layout**)self) {
        return;
    }

    if ((*(layout**)self)->refcount.fetch_sub(1) == 1) {
        tp_free((*(layout**)self));
    }
}

void StringType::copy_constructor(instance_ptr self, instance_ptr other) {
    (*(layout**)self) = (*(layout**)other);
    if (*(layout**)self) {
        (*(layout**)self)->refcount++;
    }
}

void StringType::assign(instance_ptr self, instance_ptr other) {
    layout* old = (*(layout**)self);

    (*(layout**)self) = (*(layout**)other);

    if (*(layout**)self) {
        (*(layout**)self)->refcount++;
    }

    destroy((instance_ptr)&old);
}

#define max(a,b) a < b ? b : a;

void StringType::join(StringType::layout **outString, StringType::layout *separator, ListOfType::layout *toJoin) {

    if (!outString)
        throw std::invalid_argument("missing return argument");

    StringType::Make()->constructor((instance_ptr)outString);

    // return empty string when there is nothing to join
    if(toJoin->count == 0)
    {
        return;
    }

    static ListOfType *listOfStrType = ListOfType::Make(StringType::Make());

    // find the max code point
    int32_t maxCodePoint = 0;

    // separator can be an empty string, then it's a nullptr
    if (separator != nullptr)
    {
        maxCodePoint = max(maxCodePoint, separator->bytes_per_codepoint);
    }

    for (int64_t i = 0; i < toJoin->count; i++)
    {
        instance_ptr item = listOfStrType->eltPtr(toJoin, i);
        if (item != nullptr)
        {
            maxCodePoint = max(maxCodePoint, StringType::Make()->bytes_per_codepoint(item));
        }
    }

    int resultCodepoints = 0;
    std::vector<StringType::layout*> itemsToJoin;
    StringType::layout* newSeparator;

    // convert to the max code point if needed
    for (int64_t i = 0; i < toJoin->count; i++)
    {
        instance_ptr item = listOfStrType->eltPtr(toJoin, i);
        StringType::layout** itemLayout = (StringType::layout**)item;
        StringType::layout* newLayout = StringType::Make()->upgradeCodePoints(*itemLayout, maxCodePoint);
        itemsToJoin.push_back(newLayout);
        if (*itemLayout != nullptr)
        {
            resultCodepoints += (*itemLayout)->pointcount;
        }
    }

    newSeparator = StringType::Make()->upgradeCodePoints((StringType::layout*)separator, maxCodePoint);

    // add the separators size
    if (separator != nullptr)
    {
        resultCodepoints += separator->pointcount * (toJoin->count - 1);
    }

    // add all the parts together
    *outString = (layout *) tp_malloc(sizeof(layout) + resultCodepoints * maxCodePoint);
    (*outString)->bytes_per_codepoint = maxCodePoint;
    (*outString)->hash_cache = -1;
    (*outString)->refcount = 1;
    (*outString)->pointcount = resultCodepoints;

    // position in the output data array
    int position = 0;

    for (int64_t i = 0; i < itemsToJoin.size(); i++)
    {

        StringType::layout* item = itemsToJoin[i];
        if (item != nullptr)
        {
            auto addingBytesCount = sizeof(uint8_t) * item->pointcount * maxCodePoint;
            memcpy((*outString)->data + position, item->data, addingBytesCount);
            position += addingBytesCount;
        }

        if (newSeparator != nullptr && i != toJoin->count - 1)
        {
            auto addingBytesCount = sizeof(uint8_t) * newSeparator->pointcount * maxCodePoint;
            memcpy((*outString)->data + position, newSeparator->data, addingBytesCount);
            position += addingBytesCount;
        }
    }

    // clean the temporary objects
    for(auto item: itemsToJoin)
    {
        StringType::Make()->destroy((instance_ptr)&item);
    }
    StringType::Make()->destroy((instance_ptr)&newSeparator);
}

// static
bool StringType::to_int64(StringType::layout* s, int64_t *value) {
    enum State {left_space, sign, digit, underscore, right_space, failed} state = left_space;
    int64_t value_sign = 1;
    *value = 0;

    if (s) {
        for (int64_t i = 0; i < s->pointcount; i++) {
            uint64_t c = StringType::getpoint(s, i);
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
bool StringType::to_float64(StringType::layout* s, double* value) {
    enum State {left_space, sign, whole, underscore, decimal, mantissa, underscore_mantissa,
                exp, expsign, exponent, underscore_exponent,
                right_space, identifier, identifier_right_space, failed} state = left_space;
    const int MAX_FLOAT_STR = 48;
    char buf[MAX_FLOAT_STR + 1];
    int cur = 0;
    *value = 0.0;

    if (s) {
        for (int64_t i = 0; i < s->pointcount; i++) {
            bool accumulate = true;
            uint64_t c = StringType::getpoint(s, i);
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
