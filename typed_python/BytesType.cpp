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
