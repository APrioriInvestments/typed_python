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

// Apply a PyUnicode_* function to each point of a string
template<class T, class pyunicode_fn>
inline StringType::layout* unicode_generic(pyunicode_fn& operation, T* data, StringType::layout* l)
{
    int32_t allocated_points = l->pointcount;
    int64_t new_byteCount = sizeof(StringType::layout) + allocated_points * l->bytes_per_codepoint;
    StringType::layout* new_layout = (StringType::layout*)tp_malloc(new_byteCount);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;
    new_layout->bytes_per_codepoint = l->bytes_per_codepoint;
    int32_t new_pointcount = l->pointcount;

    Py_UCS4 buf[7];
    int k = 0;
    int32_t increment = 8; // large enough to hold any result from a single character
    for (T* src = data, *end = src + l->pointcount; src < end; ) {
        int n = operation((Py_UCS4)(*src++), buf);
        if (n > 1) {
            new_pointcount += n - 1;
            if (new_pointcount > allocated_points) {
                allocated_points += increment;
                increment = (increment * 3) / 2;
                size_t priorBytecount = new_byteCount;
                new_byteCount = sizeof(StringType::layout) + allocated_points * l->bytes_per_codepoint;
                new_layout = (StringType::layout*)tp_realloc(
                    new_layout,
                    priorBytecount,
                    new_byteCount
                );
            }
        }
        for (int j = 0; j < n; j++) {
            ((T*)(new_layout->data))[k++] = (T)buf[j];
        }
    }
    if (allocated_points > new_pointcount) {
        allocated_points = new_pointcount;
        size_t priorBytecount = new_byteCount;
        new_byteCount = sizeof(StringType::layout) + allocated_points * l->bytes_per_codepoint;
        new_layout = (StringType::layout*)tp_realloc(new_layout, priorBytecount, new_byteCount);
    }
    new_layout->pointcount = new_pointcount;
    return new_layout;
}

StringType::layout* StringType::upper(layout *l) {
    if (!l) {
        return l;
    }

    if (l->bytes_per_codepoint == 1) {
        return unicode_generic(_PyUnicode_ToUpperFull, (uint8_t*)(l->data), l);
    }
    else if (l->bytes_per_codepoint == 2) {
        return unicode_generic(_PyUnicode_ToUpperFull, (uint16_t*)(l->data), l);
    }
    else if (l->bytes_per_codepoint == 4) {
        return unicode_generic(_PyUnicode_ToUpperFull, (uint32_t*)(l->data), l);
    }
    return 0; // error
}

StringType::layout* StringType::lower(layout *l) {
    if (!l) {
        return l;
    }

    if (l->bytes_per_codepoint == 1) {
        return unicode_generic(_PyUnicode_ToLowerFull, (uint8_t*)(l->data), l);
    }
    else if (l->bytes_per_codepoint == 2) {
        return unicode_generic(_PyUnicode_ToLowerFull, (uint16_t*)(l->data), l);
    }
    else if (l->bytes_per_codepoint == 4) {
        return unicode_generic(_PyUnicode_ToLowerFull, (uint32_t*)(l->data), l);
    }
    return 0; // error
}

StringType::layout* StringType::casefold(layout *l) {
    if (!l) {
        return l;
    }

    if (l->bytes_per_codepoint == 1) {
        return unicode_generic(_PyUnicode_ToFoldedFull, (uint8_t*)(l->data), l);
    }
    else if (l->bytes_per_codepoint == 2) {
        return unicode_generic(_PyUnicode_ToFoldedFull, (uint16_t*)(l->data), l);
    }
    else if (l->bytes_per_codepoint == 4) {
        return unicode_generic(_PyUnicode_ToFoldedFull, (uint32_t*)(l->data), l);
    }
    return 0; // error
}

template<class T>
inline StringType::layout* capitalize_generic(T* data, StringType::layout* l)
{
    int32_t allocated_points = l->pointcount;
    int64_t new_byteCount = sizeof(StringType::layout) + allocated_points * l->bytes_per_codepoint;
    StringType::layout* new_layout = (StringType::layout*)tp_malloc(new_byteCount);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;
    new_layout->bytes_per_codepoint = l->bytes_per_codepoint;
    int32_t new_pointcount = l->pointcount;

    Py_UCS4 buf[7];
    int k = 0;
    int32_t increment = 8; // large enough to hold any result from a single character
    for (T* src = data, *end = src + l->pointcount; src < end; ) {
        int n = (src == data)
                ? _PyUnicode_ToUpperFull((Py_UCS4)(*src++), buf)
                : _PyUnicode_ToLowerFull((Py_UCS4)(*src++), buf);
        if (n > 1) {
            new_pointcount += n - 1;
            if (new_pointcount > allocated_points) {
                allocated_points += increment;
                increment = (increment * 3) / 2;
                size_t priorBytecount = new_byteCount;
                new_byteCount = sizeof(StringType::layout) + allocated_points * l->bytes_per_codepoint;
                new_layout = (StringType::layout*)tp_realloc(new_layout, priorBytecount, new_byteCount);
            }
        }
        for (int j = 0; j < n; j++) {
            ((T*)(new_layout->data))[k++] = (T)buf[j];
        }
    }
    if (allocated_points > new_pointcount) {
        allocated_points = new_pointcount;
        size_t priorBytecount = new_byteCount;
        new_byteCount = sizeof(StringType::layout) + allocated_points * l->bytes_per_codepoint;
        new_layout = (StringType::layout*)tp_realloc(new_layout, priorBytecount, new_byteCount);
    }
    new_layout->pointcount = new_pointcount;
    return new_layout;
}

StringType::layout* StringType::capitalize(layout *l) {
    if (!l) {
        return l;
    }

    if (l->bytes_per_codepoint == 1) {
        return capitalize_generic((uint8_t*)(l->data), l);
    }
    else if (l->bytes_per_codepoint == 2) {
        return capitalize_generic((uint16_t*)(l->data), l);
    }
    else if (l->bytes_per_codepoint == 4) {
        return capitalize_generic((uint32_t*)(l->data), l);
    }
    return 0; // error
}

template<class T>
inline void swapcase_generic(T* src, uint32_t pointcount, T* dest) {
    for (T* end = src + pointcount; src < end; ) {
        T c = *src++;
        if (iswlower(c))
            *dest++ = (T)towupper(c);
        else if (iswupper(c))
            *dest++ = (T)towlower(c);
        else
            *dest++ = c;
    }
}

template<class T>
inline StringType::layout* swapcase_generic(T* data, StringType::layout* l)
{
    int32_t allocated_points = l->pointcount;
    int64_t new_byteCount = sizeof(StringType::layout) + allocated_points * l->bytes_per_codepoint;
    StringType::layout* new_layout = (StringType::layout*)tp_malloc(new_byteCount);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;
    new_layout->bytes_per_codepoint = l->bytes_per_codepoint;
    int32_t new_pointcount = l->pointcount;

    Py_UCS4 buf[7];
    int k = 0;
    int32_t increment = 8; // large enough to hold any result from a single character
    for (T* src = data, *end = src + l->pointcount; src < end; ) {
        bool make_upper = _PyUnicode_IsLowercase(*src);
        bool just_copy = !make_upper && !_PyUnicode_IsUppercase(*src);
        if (just_copy) {
            ((T*)(new_layout->data))[k++] = *src++;
            continue;
        }
        int n = make_upper
                ? _PyUnicode_ToUpperFull((Py_UCS4)(*src++), buf)
                : _PyUnicode_ToLowerFull((Py_UCS4)(*src++), buf);
        if (n > 1) {
            new_pointcount += n - 1;
            if (new_pointcount > allocated_points) {
                allocated_points += increment;
                increment = (increment * 3) / 2;
                size_t priorBytecount = new_byteCount;
                new_byteCount = sizeof(StringType::layout) + allocated_points * l->bytes_per_codepoint;
                new_layout = (StringType::layout*)tp_realloc(new_layout, priorBytecount, new_byteCount);
            }
        }
        for (int j = 0; j < n; j++) {
            ((T*)(new_layout->data))[k++] = (T)buf[j];
        }
    }
    if (allocated_points > new_pointcount) {
        allocated_points = new_pointcount;
        size_t origBytecount = new_byteCount;
        new_byteCount = sizeof(StringType::layout) + allocated_points * l->bytes_per_codepoint;
        new_layout = (StringType::layout*)tp_realloc(new_layout, origBytecount, new_byteCount);
    }
    new_layout->pointcount = new_pointcount;
    return new_layout;
}

StringType::layout* StringType::swapcase(layout *l) {
    if (!l) {
        return l;
    }

    if (l->bytes_per_codepoint == 1) {
        return swapcase_generic((uint8_t*)(l->data), l);
    }
    else if (l->bytes_per_codepoint == 2) {
        return swapcase_generic((uint16_t*)(l->data), l);
    }
    else if (l->bytes_per_codepoint == 4) {
        return swapcase_generic((uint32_t*)(l->data), l);
    }
    return 0; // error
}

template<class T>
inline void title_generic(T* src, uint32_t pointcount, T* dest) {
    bool word = false;
    for (T* end = src + pointcount; src < end; ) {
        T c = *src++;
        if (word) {
            if (iswalpha(c)) {
                *dest++ = (T)towlower(c);
            } else {
                *dest++ = c;
                word = false;
            }
        } else {
            if (iswalpha(c)) {
                *dest++ = (T)towupper(c);
                word = true;
            } else {
                *dest++ = c;
            }
        }
    }
}

template<class T>
inline StringType::layout* title_generic(T* data, StringType::layout* l)
{
    int32_t allocated_points = l->pointcount;
    int64_t new_byteCount = sizeof(StringType::layout) + allocated_points * l->bytes_per_codepoint;
    StringType::layout* new_layout = (StringType::layout*)tp_malloc(new_byteCount);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;
    new_layout->bytes_per_codepoint = l->bytes_per_codepoint;
    int32_t new_pointcount = l->pointcount;

    Py_UCS4 buf[7];
    int k = 0;
    bool word = false;
    int32_t increment = 8; // large enough to hold any result from a single character
    for (T* src = data, *end = src + l->pointcount; src < end; ) {
        Py_UCS4 c = (Py_UCS4)(*src);
        bool make_lower = false;
        bool make_title = false;
        if (word) {
            if (_PyUnicode_IsAlpha(c))  {
                make_lower = true;
            }
            else {
                word = false;
            }
        }
        else {
            if (_PyUnicode_IsAlpha(c))  {
                make_title = true;
                word = true;
            }
        }
        if (!make_lower && !make_title) {
            ((T*)(new_layout->data))[k++] = *src++;
            continue;
        }
        int n = make_title
                ? _PyUnicode_ToTitleFull((Py_UCS4)(*src++), buf)
                : _PyUnicode_ToLowerFull((Py_UCS4)(*src++), buf);
        if (n > 1) {
            new_pointcount += n - 1;
            if (new_pointcount > allocated_points) {
                allocated_points += increment;
                increment = (increment * 3) / 2;
                size_t priorBytecount = new_byteCount;
                new_byteCount = sizeof(StringType::layout) + allocated_points * l->bytes_per_codepoint;
                new_layout = (StringType::layout*)tp_realloc(new_layout, priorBytecount, new_byteCount);
            }
        }
        for (int j = 0; j < n; j++) {
            ((T*)(new_layout->data))[k++] = (T)buf[j];
        }
    }
    if (allocated_points > new_pointcount) {
        allocated_points = new_pointcount;
        size_t origBytecount = new_byteCount;
        new_byteCount = sizeof(StringType::layout) + allocated_points * l->bytes_per_codepoint;
        new_layout = (StringType::layout*)tp_realloc(new_layout, origBytecount, new_byteCount);
    }
    new_layout->pointcount = new_pointcount;
    return new_layout;
}

StringType::layout* StringType::title(layout *l) {
    if (!l) {
        return l;
    }

    if (l->bytes_per_codepoint == 1) {
        return title_generic((uint8_t*)(l->data), l);
    }
    else if (l->bytes_per_codepoint == 2) {
        return title_generic((uint16_t*)(l->data), l);
    }
    else if (l->bytes_per_codepoint == 4) {
        return title_generic((uint32_t*)(l->data), l);
    }
    return 0; // error
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

int64_t StringType::rfind(layout *l, layout *sub, int64_t start, int64_t stop) {
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
    if (stop > l->pointcount)
        stop = l->pointcount;
    if (!sub || !sub->pointcount)
        return stop;

    stop -= (sub->pointcount - 1);

    if (start < 0 || stop < 0 || start >= stop || sub->pointcount > l->pointcount || start > l->pointcount - sub->pointcount)
        return -1;

    if (l->bytes_per_codepoint == 1 and sub->bytes_per_codepoint == 1 && sub->pointcount == 1) {
        const uint8_t* lPtr = (const uint8_t*)l->data;
        const uint8_t subChar = sub->data[0];

        for (int64_t i = stop - 1; i >= start; i--) {
            if (lPtr[i] == subChar) {
                return i;
            }
        }
        return -1;
    }

    for (int64_t i = stop - 1; i >= start; i--) {
        bool match = true;
        for (int64_t j = 0; j < sub->pointcount; j++) {
            if (getpoint(l, i+j) != getpoint(sub, j)) {
                match = false;
                break;
            }
        }
        if (match) {
            return i;
            }
    }

    return -1;
}

int64_t StringType::count(layout *l, layout *sub, int64_t start, int64_t stop) {
    int64_t count = 0;

    if (!l || !l->pointcount) {
        return 0;
    }
    if (start < 0) {
        start += l->pointcount;
        if (start < 0) start = 0;
    }
    if (stop < 0) {
        stop += l->pointcount;
        if (stop < 0) stop = 0;
    }
    if (stop < start || start > l->pointcount) {
        return 0;
    }
    if (stop > l->pointcount) {
        stop = l->pointcount;
    }
    if (!sub || !sub->pointcount) {
        if (start > l->pointcount) return 0;
        count = stop - start + 1;
        return count >= 0 ? count : 0;
    }

    stop -= (sub->pointcount - 1);

    if (start < 0 || stop < 0 || start >= stop || sub->pointcount > l->pointcount || start > l->pointcount - sub->pointcount)
        return 0;

    if (l->bytes_per_codepoint == 1 and sub->bytes_per_codepoint == 1 && sub->pointcount == 1) {
        const uint8_t* lPtr = (const uint8_t*)l->data;
        const uint8_t subChar = sub->data[0];

        for (int64_t i = start; i < stop; i++) {
            if (lPtr[i] == subChar) {
                count++;
            }
        }
        return count;
    }

    for (int64_t i = start; i < stop; i++) {
        bool match = true;
        for (int64_t j = 0; j < sub->pointcount; j++) {
            if (getpoint(l, i+j) != getpoint(sub, j)) {
                match = false;
                break;
            }
        }
        if (match) {
            count++;
            i += sub->pointcount - 1;
        }
    }

    return count;
}

void StringType::split(ListOfType::layout* outList, layout* l, layout* sep, int64_t max) {
    if (!outList)
        throw std::invalid_argument("missing return argument");

    static ListOfType* listofstring = ListOfType::Make(StringType::Make());
    listofstring->resize((instance_ptr)&outList, 0, 0);

    int64_t sep_bytes_per_codepoint = sep ? sep->bytes_per_codepoint : 1;
    int64_t sep_pointcount = sep ? sep->pointcount : 1;

    if (!l || !l->pointcount) {
        if (sep && sep->pointcount) {
            listofstring->append((instance_ptr)&outList, (instance_ptr)&l);
        }
        return;
    }
    int64_t cur = 0;
    int64_t count = 0;

    listofstring->reserve((instance_ptr)&outList, 10);

    while ((count < max || max < 0) && cur < l->pointcount) {
        int64_t match = cur;

        if (sep && l->bytes_per_codepoint == 1 && sep_bytes_per_codepoint == 1 && sep_pointcount == 1) {
            while (match < l->pointcount && l->data[match] != sep->data[0]) match++;
        } else if (sep) {
            match = find(l, sep, cur, l->pointcount);
            if (match == -1) match = l->pointcount;
        } else {
            while (match < l->pointcount && !(uprops[getpoint(l, match)] & Uprops_SPACE)) match++;
        }
        if (match >= l->pointcount) break;

        if (sep || match != cur) {
            layout* piece = getsubstr(l, cur, match);
            if (outList->count == outList->reserved) {
                listofstring->reserve((instance_ptr)&outList, outList->reserved * 1.5);
            }
            ((layout**)outList->data)[outList->count++] = piece;

            cur = match + sep_pointcount;

            count++;
            if (max >= 0 && count >= max)
                break;
        }
        else if (!sep) {
            cur++;
        }
    }
    if (!sep) {
        while (cur < l->pointcount && (uprops[getpoint(l, cur)] & Uprops_SPACE)) {
            cur++;
        }
    }
    if (sep || cur < l->pointcount || max == 0) {
        layout* remainder = getsubstr(l, cur, l->pointcount);
        if (outList->count == outList->reserved) {
            listofstring->reserve((instance_ptr)&outList, outList->reserved + 1);
        }
        ((layout**)outList->data)[outList->count++] = remainder;
    }
}

void StringType::rsplit(ListOfType::layout* outList, layout* l, layout* sep, int64_t max) {
    if (!outList)
        throw std::invalid_argument("missing return argument");

    static ListOfType* listofstring = ListOfType::Make(StringType::Make());
    listofstring->resize((instance_ptr)&outList, 0, 0);

    int64_t sep_bytes_per_codepoint = sep ? sep->bytes_per_codepoint : 1;
    int64_t sep_pointcount = sep ? sep->pointcount : 1;

    if (!l || !l->pointcount) {
        if (sep && sep->pointcount) {
            listofstring->append((instance_ptr)&outList, (instance_ptr)&l);
        }
        return;
    }
    int64_t cur = l->pointcount - 1;
    int64_t count = 0;

    listofstring->reserve((instance_ptr)&outList, 10);

    while ((count < max || max < 0) && cur >= 0) {
        int64_t match = cur;

        if (sep && l->bytes_per_codepoint == 1 && sep_bytes_per_codepoint == 1 && sep_pointcount == 1) {
            while (match >= 0 && l->data[match] != sep->data[0]) match--;
        } else if (sep) {
            match = rfind(l, sep, 0, cur + 1);
            if (match != -1) match += sep_pointcount - 1;
        } else {
            while (match >= 0 && !(uprops[getpoint(l, match)] & Uprops_SPACE)) match--;
        }
        if (match < 0) break;

        if (sep || match != cur) {
            layout* piece = getsubstr(l, match + 1, cur + 1);
            if (outList->count == outList->reserved) {
                listofstring->reserve((instance_ptr)&outList, outList->reserved * 1.5);
            }
            ((layout**)outList->data)[outList->count++] = piece;

            cur = match - sep_pointcount;

            count++;
            if (max >= 0 && count >= max)
                break;
        }
        else if (!sep) {
            cur--;
        }
    }
    if (!sep) {
        while (cur >= 0 && (uprops[getpoint(l, cur)] & Uprops_SPACE)) {
            cur--;
        }
    }
    if (sep || cur >= 0 || max == 0) {
        layout* remainder = getsubstr(l, 0, cur + 1);
        if (outList->count == outList->reserved) {
            listofstring->reserve((instance_ptr)&outList, outList->reserved + 1);
        }
        ((layout**)outList->data)[outList->count++] = remainder;
    }
    listofstring->reverse((instance_ptr)&outList);
}

bool linebreak(int32_t c) {
    // the two-character line break '\r\n' must be handled outside of this function
    return c == '\n' || c == '\r'
    || c == 0x0B || c == 0x0C
    || c == 0x1C || c == 0x1D || c == 0x1E
    || c == 0x85 || c == 0x2028 || c == 0x2029;
}

/* static */
// assumes outList was initialized to an empty list before calling
void StringType::splitlines(ListOfType::layout *outList, layout* in, bool keepends) {
    static ListOfType* listOfString = ListOfType::Make(StringType::Make());

    int64_t cur = 0;

    listOfString->reserve((instance_ptr)&outList, 10);

    int64_t inLen = in ? in->pointcount : 0;
    int sepLen = 0;

    while (cur < inLen) {
        int64_t match = cur;

        while (match < inLen && !linebreak(getpoint(in, match))) {
            match++;
        }
        sepLen = 0;
        if (match < inLen) {
            if (getpoint(in, match) == '\r' && match + 1 < inLen && getpoint(in, match+1) == '\n') {
                sepLen = 2;
            }
            else {
                sepLen = 1;
            }
        }

        if (match + sepLen > inLen) {
            break;
        }

        layout* piece = getsubstr(in, cur, match + (keepends ? sepLen : 0));

        if (outList->count == outList->reserved) {
            listOfString->reserve((instance_ptr)&outList, outList->reserved * 1.5);
        }

        ((layout**)outList->data)[outList->count++] = piece;
        cur = match + sepLen;
    }
    if (inLen != cur) {
        layout* remainder = getsubstr(in, cur, inLen);
        if (outList->count == outList->reserved) {
            listOfString->reserve((instance_ptr)&outList, outList->reserved + 1);
        }
        listOfString->append((instance_ptr)&outList, (instance_ptr)&remainder);
        ((layout**)outList->data)[outList->count++] = remainder;
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

    memcpy(new_layout->data, l->data + start * l->bytes_per_codepoint, datasize);

    return new_layout;
}

// specialized form of the strip left/right algorithm based on the codepoint type.
template<class elt_type>
void stripInset_whitespace(int64_t& ioLeftPos, int64_t& ioRightPos, elt_type* data, bool fromLeft, bool fromRight) {
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

StringType::layout* StringType::strip_whitespace(layout* l, bool fromLeft, bool fromRight) {
    if (!l) {
        return l;
    }

    int64_t leftPos = 0;
    int64_t rightPos = l->pointcount;

    uint8_t* dataPtr = l->data;

    if (l->bytes_per_codepoint == 1) {
        stripInset_whitespace(leftPos, rightPos, (uint8_t*)dataPtr, fromLeft, fromRight);
    }
    else if (l->bytes_per_codepoint == 2) {
        stripInset_whitespace(leftPos, rightPos, (uint16_t*)dataPtr, fromLeft, fromRight);
    }
    else if (l->bytes_per_codepoint == 4) {
        stripInset_whitespace(leftPos, rightPos, (uint32_t*)dataPtr, fromLeft, fromRight);
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

StringType::layout* StringType::strip(layout* l, bool whiteSpace, layout* values, bool fromLeft, bool fromRight) {
    if (whiteSpace) {
        return strip_whitespace(l, fromLeft, fromRight);
    }

    if (!l || !values || values->pointcount == 0) {
        if (l) l->refcount++;
        return l;
    }

    int64_t leftPos = 0;
    int64_t rightPos = l->pointcount;

    if (fromLeft) {
        while (leftPos < rightPos) {
            uint32_t c = getpoint(l, leftPos);
            bool found = false;
            for (int i = 0; i < values->pointcount; i++) {
                if (c == getpoint(values, i)) {
                    leftPos++;
                    found = true;
                    break;
                }
            }
            if (!found) break;  // no matches
        }
    }
    if (fromRight) {
        while (leftPos < rightPos) {
            uint32_t c = getpoint(l, rightPos - 1);
            bool found = false;
            for (int i = 0; i < values->pointcount; i++) {
                if (c == getpoint(values, i)) {
                    rightPos--;
                    found = true;
                    break;
                }
            }
            if (!found) break;  // no matches
        }
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

StringType::layout* StringType::lstrip(layout* l, bool whiteSpace, StringType::layout* values) {
    return strip(l, whiteSpace, values, true, false);
}

StringType::layout* StringType::rstrip(layout* l, bool whiteSpace, StringType::layout* values) {
    return strip(l, whiteSpace, values, false, true);
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
            utf8Str += 2;
            bytes_per_codepoint = std::max<int64_t>(2, bytes_per_codepoint);
        } else if (utf8Str[0] >> 4 == 0b1110) {
            length -= 1;
            utf8Str += 3;
            bytes_per_codepoint = std::max<int64_t>(2, bytes_per_codepoint);
        } else if (utf8Str[0] >> 3 == 0b11110) {
            length -= 1;
            utf8Str += 4;
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

char StringType::cmpStatic(layout* left, uint8_t* right_data, int right_pointcount, int right_bytes_per_codepoint) {
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
    int bytesPerRight = right_bytes_per_codepoint;
    int commonCount = std::min(left->pointcount, right_pointcount);

    char res = 0;

    if (bytesPerLeft == 1 && bytesPerRight == 1) {
        res = byteCompare(left->data, right_data, bytesPerLeft * commonCount);
    } else if (bytesPerLeft == 1 && bytesPerRight == 2) {
        res = typedArrayCompare((uint8_t*)left->data, (uint16_t*)right_data, commonCount);
    } else if (bytesPerLeft == 1 && bytesPerRight == 4) {
        res = typedArrayCompare((uint8_t*)left->data, (uint32_t*)right_data, commonCount);
    } else if (bytesPerLeft == 2 && bytesPerRight == 1) {
        res = typedArrayCompare((uint16_t*)left->data, (uint8_t*)right_data, commonCount);
    } else if (bytesPerLeft == 2 && bytesPerRight == 2) {
        res = typedArrayCompare((uint16_t*)left->data, (uint16_t*)right_data, commonCount);
    } else if (bytesPerLeft == 2 && bytesPerRight == 4) {
        res = typedArrayCompare((uint16_t*)left->data, (uint32_t*)right_data, commonCount);
    } else if (bytesPerLeft == 4 && bytesPerRight == 1) {
        res = typedArrayCompare((uint32_t*)left->data, (uint8_t*)right_data, commonCount);
    } else if (bytesPerLeft == 4 && bytesPerRight == 2) {
        res = typedArrayCompare((uint32_t*)left->data, (uint16_t*)right_data, commonCount);
    } else if (bytesPerLeft == 4 && bytesPerRight == 4) {
        res = typedArrayCompare((uint32_t*)left->data, (uint32_t*)right_data, commonCount);
    } else {
        throw std::runtime_error("Nonsensical bytes-per-codepoint");
    }

    if (res) {
        return res;
    }

    if (left->pointcount < right_pointcount) {
        return -1;
    }

    if (left->pointcount > right_pointcount) {
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

    static char hexDigits[] = "0123456789abcdef";

    layout *l = *(layout**)self;
    for (int64_t i = 0; i < (l ? l->pointcount : 0); i++) {
        uint64_t c = StringType::getpoint(l, i);
        if (c == '"') {
            stream << "\\\"";
        } else if (c == '\\') {
            stream << "\\\\";
        } else if (c <= 0x1F || c == 0x7F) {
            if (c == '\n') {
                stream << "\\n";
            } else
            if (c == '\r') {
                stream << "\\r";
            } else
            if (c == '\t') {
                stream << "\\t";
            } else {
                stream << "\\x" << hexDigits[c / 16] << hexDigits[c % 16];
            }
        } else if (c < 0x80) {
            stream << (char)c;
        } else if (c < 0x800) {
            stream << (char)(0xC0 | (c >> 6));
            stream << (char)(0x80 | (c & 0x3F));
        } else if (c < 0x10000) {
            stream << (char)(0xE0 | (c >> 12));
            stream << (char)(0x80 | ((c >> 6) & 0x3F));
            stream << (char)(0x80 | (c & 0x3F));
        } else if (c < 0x110000) {
            stream << (char)(0xF0 | (c >> 18));
            stream << (char)(0x80 | ((c >> 12) & 0x3F));
            stream << (char)(0x80 | ((c >> 6) & 0x3F));
            stream << (char)(0x80 | (c & 0x3F));
        } else {
            const uint32_t rc = 0xFFFD;
            stream << (char)(0xF0 | (rc >> 18));
            stream << (char)(0x80 | ((rc >> 12) & 0x3F));
            stream << (char)(0x80 | ((rc >> 6) & 0x3F));
            stream << (char)(0x80 | (rc & 0x3F));
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

StringType::layout* StringType::mult(layout* lhs, int64_t rhs) {
    if (!lhs) {
        return lhs;
    }
    if (rhs <= 0)
        return 0;
    int64_t new_length = lhs->pointcount * rhs;
    int64_t new_byteCount = sizeof(layout) + new_length * lhs->bytes_per_codepoint;

    layout* new_layout = (layout*)tp_malloc(new_byteCount);
    new_layout->refcount = 1;
    new_layout->hash_cache = -1;
    new_layout->bytes_per_codepoint = lhs->bytes_per_codepoint;
    new_layout->pointcount = new_length;

    int64_t old_size = lhs->pointcount * lhs->bytes_per_codepoint;
    for (size_t i = 0; i < rhs; i++) {
        memcpy(new_layout->data + i * old_size, lhs->data, old_size);
    }

    return new_layout;
}
