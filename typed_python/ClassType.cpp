/******************************************************************************
   Copyright 2017-2019 Nativepython Authors

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

bool Class::isBinaryCompatibleWithConcrete(Type* other) {
    if (other->getTypeCategory() != m_typeCategory) {
        return false;
    }

    Class* otherO = (Class*)other;

    return m_heldClass->isBinaryCompatibleWith(otherO->m_heldClass);
}

bool Class::_updateAfterForwardTypesChanged() {
    bool is_default_constructible = m_heldClass->is_default_constructible();
    std::string name = m_heldClass->name();

    bool anyChanged = (m_name != name || m_is_default_constructible != is_default_constructible);

    m_name = name;
    m_is_default_constructible = is_default_constructible;

    return anyChanged;
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

bool Class::cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions) {
    const char* method = nullptr;
    switch (pyComparisonOp) {
        case Py_EQ:
            method = "__eq__";
            break;
        case Py_NE:
            method = "__ne__";
            break;
        case Py_LT:
            method = "__lt__";
            break;
        case Py_GT:
            method = "__gt__";
            break;
        case Py_LE:
            method = "__le__";
            break;
        case Py_GE:
            method = "__ge__";
            break;
    }

    auto it = m_heldClass->getMemberFunctions().find(method);

    if (it != m_heldClass->getMemberFunctions().end()) {
        //we found a user-defined method for this comparison function.
        PyObjectStealer leftAsPyObj(PyInstance::extractPythonObject(left, this));
        PyObjectStealer rightAsPyObj(PyInstance::extractPythonObject(right, this));

        std::pair<bool, PyObject*> res = PyFunctionInstance::tryToCall(
            it->second,
            leftAsPyObj,
            rightAsPyObj
            );

        if (res.first && !res.second) {
            throw PythonExceptionSet();
        }

        bool result = res.second == Py_True;
        decref(res.second);
        return result;
    }

    if (pyComparisonOp == Py_NE) {
        return !cmp(left, right, Py_EQ, suppressExceptions);
    }

    if (pyComparisonOp == Py_EQ) {
        //if these operators are not implemented, we defer to the class pointer
        uint64_t leftPtr = *(uint64_t*)left;
        uint64_t rightPtr = *(uint64_t*)right;

        return leftPtr == rightPtr;
    }

    PyErr_Format(
        PyExc_TypeError,
        "'%s' not defined between instances of '%s' and '%s'",
        pyComparisonOp == Py_EQ ? "==" :
        pyComparisonOp == Py_NE ? "!=" :
        pyComparisonOp == Py_LT ? "<" :
        pyComparisonOp == Py_LE ? "<=" :
        pyComparisonOp == Py_GT ? ">" :
        pyComparisonOp == Py_GE ? ">=" : "?",
        name().c_str(),
        name().c_str()
        );
    throw PythonExceptionSet();
}

void Class::repr(instance_ptr self, ReprAccumulator& stream) {
    auto it = m_heldClass->getMemberFunctions().find(stream.isStrCall() ? "__str__" : "__repr__");

    if (it != m_heldClass->getMemberFunctions().end()) {
        PyEnsureGilAcquired acquireTheGil;

        PyObjectStealer selfAsPyObj(PyInstance::extractPythonObject(self, this));

        std::pair<bool, PyObject*> res = PyFunctionInstance::tryToCall(
            it->second,
            selfAsPyObj
            );

        if (res.first) {
            if (!res.second) {
                throw PythonExceptionSet();
            }
            if (!PyUnicode_Check(res.second)) {
                decref(res.second);
                throw std::runtime_error(
                    stream.isStrCall() ? "__str__ returned a non-string" : "__repr__ returned a non-string"
                    );
            }

            stream << PyUnicode_AsUTF8(res.second);
            decref(res.second);

            return;
        }

        throw std::runtime_error(
            stream.isStrCall() ? "Found a __str__ method but failed to call it with 'self'"
                : "Found a __repr__ method but failed to call it with 'self'"
            );
    }


    layout& l = **(layout**)self;
    m_heldClass->repr(l.data, stream);
}

typed_python_hash_type Class::hash(instance_ptr left) {
    auto it = m_heldClass->getMemberFunctions().find("__hash__");

    if (it != m_heldClass->getMemberFunctions().end()) {
        PyEnsureGilAcquired acquireTheGil;

        PyObjectStealer leftAsPyObj(PyInstance::extractPythonObject(left, this));

        std::pair<bool, PyObject*> res = PyFunctionInstance::tryToCall(
            it->second,
            leftAsPyObj
            );
        if (res.first) {
            if (!res.second) {
                throw PythonExceptionSet();
            }
            if (!PyLong_Check(res.second)) {
                decref(res.second);
                throw std::runtime_error("__hash__ returned a non-int");
            }

            int32_t retval = PyLong_AsLong(res.second);
            decref(res.second);
            if (retval == -1) {
                retval = -2;
            }

            return retval;
        }

        throw std::runtime_error("Found a __hash__ method but failed to call it with 'self'");
    }

    layout& l = **(layout**)left;
    return m_heldClass->hash(l.data);
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

    if (l.refcount.fetch_sub(1) == 1) {
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
