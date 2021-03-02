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

bool Class::isBinaryCompatibleWithConcrete(Type* other) {
    if (other->getTypeCategory() != m_typeCategory) {
        return false;
    }

    Class* otherO = (Class*)other;

    return m_heldClass->isBinaryCompatibleWith(otherO->m_heldClass);
}

bool Class::_updateAfterForwardTypesChanged() {
    bool is_default_constructible = m_heldClass->is_default_constructible();

    bool anyChanged = m_is_default_constructible != is_default_constructible;

    m_is_default_constructible = is_default_constructible;

    return anyChanged;
}

instance_ptr Class::eltPtr(instance_ptr self, int64_t ix) const {
    layout& l = *instanceToLayout(self);
    return m_heldClass->eltPtr(l.data, ix);
}

void Class::setAttribute(instance_ptr self, int64_t ix, instance_ptr elt) const {
    layout& l = *instanceToLayout(self);
    m_heldClass->setAttribute(l.data, ix, elt);
}

void Class::delAttribute(instance_ptr self, int64_t ix) const {
    layout& l = *instanceToLayout(self);
    m_heldClass->delAttribute(l.data, ix);
}

bool Class::checkInitializationFlag(instance_ptr self, int64_t ix) const {
    layout& l = *instanceToLayout(self);
    return m_heldClass->checkInitializationFlag(l.data, ix);
}

//static
bool Class::cmpStatic(Class* t, instance_ptr left, instance_ptr right, int64_t pyComparisonOp) {
    // TODO: assert that t is a Class pointer
    return t->cmp(left, right, pyComparisonOp, false);
}

bool Class::cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions) {
    if (m_heldClass->hasAnyComparisonOperators()) {
        auto it = m_heldClass->getMemberFunctions().find(pyComparisonOpToMethodName(pyComparisonOp));

        if (it != m_heldClass->getMemberFunctions().end()) {
            //we found a user-defined method for this comparison function.
            PyObjectStealer leftAsPyObj(PyInstance::extractPythonObject(left, this));
            PyObjectStealer rightAsPyObj(PyInstance::extractPythonObject(right, this));

            std::pair<bool, PyObject*> res = PyFunctionInstance::tryToCall(
                it->second,
                nullptr,
                leftAsPyObj,
                rightAsPyObj
                );

            if (res.first && !res.second) {
                throw PythonExceptionSet();
            }

            int result = PyObject_IsTrue(res.second);
            decref(res.second);

            if (result == -1) {
                throw PythonExceptionSet();
            }

            return result != 0;
        }
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

void Class::repr(instance_ptr self, ReprAccumulator& stream, bool isStr) {
    auto it = m_heldClass->getMemberFunctions().find(isStr ? "__str__" : "__repr__");

    if (it != m_heldClass->getMemberFunctions().end()) {
        PyEnsureGilAcquired acquireTheGil;

        PyObjectStealer selfAsPyObj(PyInstance::extractPythonObject(self, this));

        std::pair<bool, PyObject*> res = PyFunctionInstance::tryToCall(
            it->second,
            nullptr,
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


    layout& l = *instanceToLayout(self);
    m_heldClass->repr(l.data, stream, isStr, true /* isClassNotHeldClass */);
}

typed_python_hash_type Class::hash(instance_ptr left) {
    layout& l = *instanceToLayout(left);

    auto it = m_heldClass->getMemberFunctions().find("__hash__");

    if (it != m_heldClass->getMemberFunctions().end()) {
        PyEnsureGilAcquired acquireTheGil;

        PyObjectStealer leftAsPyObj(PyInstance::extractPythonObject(left, this));

        std::pair<bool, PyObject*> res = PyFunctionInstance::tryToCall(
            it->second,
            nullptr,
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

    return m_heldClass->hash(l.data);
}

void Class::constructor(instance_ptr self, bool allowEmpty) {
    if (!m_is_default_constructible and !allowEmpty) {
        throw std::runtime_error(m_name + " is not default-constructible");
    }

    initializeInstance(self, (layout*)tp_malloc(sizeof(layout) + m_heldClass->bytecount()), 0);

    layout& l = *instanceToLayout(self);
    l.refcount = 1;
    l.vtable = m_heldClass->getVTable();

    m_heldClass->constructor(l.data, allowEmpty);
}

int64_t Class::refcount(instance_ptr self) {
    return instanceToLayout(self)->refcount;
}

void Class::destroy(instance_ptr self) {
    layout& l = *instanceToLayout(self);

    if (l.refcount.fetch_sub(1) == 1) {
        l.vtable->mType->destroy(l.data);
        tp_free(instanceToLayout(self));
    }
}

void Class::copy_constructor(instance_ptr self, instance_ptr other) {
    *(size_t*)self = *(size_t*)other;
    instanceToLayout(self)->refcount++;
}

void Class::assign(instance_ptr self, instance_ptr other) {
    layout* old = instanceToLayout(self);

    // bit-copy the instance, which includes both the classDispatch offset
    // and also the layout pointer.
    *(size_t*)self = *(size_t*)other;

    instanceToLayout(self)->refcount++;

    if (old->refcount.fetch_sub(1) == 1) {
        old->vtable->mType->destroy(old->data);
        tp_free(old);
    }
}
