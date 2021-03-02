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

#pragma once

#include "PyInstance.hpp"

class PyClassInstance : public PyInstance {
public:
    typedef Class modeled_type;

    Class* type();

    static void initializeClassWithDefaultArguments(
        Class* cls,
        uint8_t* data,
        PyObject* args,
        PyObject* kwargs
    );

    static int tpSetattrGeneric(
        PyObject* self,
        Type* t,
        instance_ptr data,
        PyObject* attrName,
        PyObject* attrVal
    );

    static bool pyValCouldBeOfTypeConcrete(Class* type, PyObject* pyRepresentation, ConversionLevel level);

    static PyObject* extractPythonObjectConcrete(Type* eltType, instance_ptr data);

    static void copyConstructFromPythonInstanceConcrete(Class* eltType, instance_ptr tgt, PyObject* pyRepresentation, ConversionLevel level);

    static void constructFromPythonArgumentsConcrete(Class* t, uint8_t* data, PyObject* args, PyObject* kwargs);

    std::pair<bool, PyObject*> callMemberFunction(
        const char* name,
        PyObject* arg0=nullptr,
        PyObject* arg1=nullptr,
        PyObject* arg2=nullptr
    );

    // generic form of 'callMemberFunction' which works on Class, HeldClass, and RefTo
    static std::pair<bool, PyObject*> callMemberFunctionGeneric(
        PyObject* self,
        Type* t,
        instance_ptr data,
        const char* name,
        PyObject* arg0=nullptr,
        PyObject* arg1=nullptr,
        PyObject* arg2=nullptr
    );

    int64_t tryCallHashMemberFunction();

    static bool compare_to_python_concrete(Class* t, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp);

    static instance_ptr getHeldClassData(Type* t, instance_ptr data);

    static HeldClass* getHeldClassType(Type* t);

    // generic form of 'tp_getattr' that works for Class, HeldClass, and RefTo
    static PyObject* tpGetattrGeneric(
        PyObject* self,
        Type* t,
        instance_ptr data,
        PyObject* pyAttrName,
        const char* attrName
    );

    PyObject* tp_getattr_concrete(PyObject* pyAttrName, const char* attrName);

    PyObject* tp_call_concrete(PyObject* args, PyObject* kwargs);

    static PyObject* tpCallConcreteGeneric(
        PyObject* self, Type* t, instance_ptr data, PyObject* args, PyObject* kwargs
    );

    static int sqContainsGeneric(PyObject* self, Type* t, instance_ptr data, PyObject* item);

    int sq_contains_concrete(PyObject* item);

    static Py_ssize_t mpAndSqLengthGeneric(PyObject* self, Type* t, instance_ptr data);

    Py_ssize_t mp_and_sq_length_concrete();

    static PyObject* mpSubscriptGeneric(PyObject* self, Type* t, instance_ptr data, PyObject* item);

    PyObject* mp_subscript_concrete(PyObject* item);

    static int mpAssignSubscriptGeneric(
        PyObject* self, Type* t, instance_ptr data, PyObject* item, PyObject* v
    );

    int mp_ass_subscript_concrete(PyObject* item, PyObject* v);

    static PyObject* tpIterGeneric(PyObject* self, Type* t, instance_ptr data);

    PyObject* tp_iter_concrete();

    static PyObject* tpIternextGeneric(PyObject* self, Type* t, instance_ptr data);

    PyObject* tp_iternext_concrete();

    static PyObject* pyUnaryOperatorConcreteGeneric(
        PyObject* self, Type* t, instance_ptr data, const char* op, const char* opErr
    );

    PyObject* pyUnaryOperatorConcrete(const char* op, const char* opErr);

    static PyObject* pyOperatorConcreteGeneric(
        PyObject* self, Type* t, instance_ptr data, PyObject* rhs, const char* op, const char* opErr
    );

    PyObject* pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr);

    static PyObject* pyOperatorConcreteReverseGeneric(
        PyObject* self, Type* t, instance_ptr data, PyObject* lhs, const char* op, const char* opErr
    );

    PyObject* pyOperatorConcreteReverse(PyObject* lhs, const char* op, const char* opErr);

    static PyObject* pyTernaryOperatorConcreteGeneric(
        PyObject* self, Type* t, instance_ptr data, PyObject* rhs,
        PyObject* ternaryArg, const char* op, const char* opErr
    );

    PyObject* pyTernaryOperatorConcrete(
        PyObject* rhs, PyObject* ternaryArg, const char* op, const char* opErr
    );

    static int pyInquiryGeneric(
        PyObject* self, Type* t, instance_ptr data, const char* op, const char* opErrRep
    );

    int pyInquiryConcrete(const char* op, const char* opErrRep);

    static void mirrorTypeInformationIntoPyTypeConcrete(
        Class* classT, PyTypeObject* pyType, bool asHeldClass=false
    );

    int tp_setattr_concrete(PyObject* attrName, PyObject* attrVal);

    static PyMethodDef* typeMethodsConcrete(Type* t);

    static PyObject* clsFormatGeneric(PyObject* o, Type* t, instance_ptr data, PyObject* args, PyObject* kwargs);
    static PyObject* clsBytesGeneric(PyObject* o, Type* t, instance_ptr data, PyObject* args, PyObject* kwargs);
    static PyObject* clsDirGeneric(PyObject* o, Type* t, instance_ptr data, PyObject* args, PyObject* kwargs);
    static PyObject* clsReversedGeneric(PyObject* o, Type* t, instance_ptr data, PyObject* args, PyObject* kwargs);
    static PyObject* clsComplexGeneric(PyObject* o, Type* t, instance_ptr data, PyObject* args, PyObject* kwargs);
    static PyObject* clsRoundGeneric(PyObject* o, Type* t, instance_ptr data, PyObject* args, PyObject* kwargs);
    static PyObject* clsTruncGeneric(PyObject* o, Type* t, instance_ptr data, PyObject* args, PyObject* kwargs);
    static PyObject* clsFloorGeneric(PyObject* o, Type* t, instance_ptr data, PyObject* args, PyObject* kwargs);
    static PyObject* clsCeilGeneric(PyObject* o, Type* t, instance_ptr data, PyObject* args, PyObject* kwargs);
    static PyObject* clsEnterGeneric(PyObject* o, Type* t, instance_ptr data, PyObject* args, PyObject* kwargs);
    static PyObject* clsExitGeneric(PyObject* o, Type* t, instance_ptr data, PyObject* args, PyObject* kwargs);

    static PyObject* clsFormat(PyObject* o, PyObject* args, PyObject* kwargs) {
        return clsFormatGeneric(o, ((PyInstance*)o)->type(), ((PyInstance*)o)->dataPtr(), args, kwargs);
    }
    static PyObject* clsBytes(PyObject* o, PyObject* args, PyObject* kwargs) {
        return clsBytesGeneric(o, ((PyInstance*)o)->type(), ((PyInstance*)o)->dataPtr(), args, kwargs);
    }
    static PyObject* clsDir(PyObject* o, PyObject* args, PyObject* kwargs) {
        return clsDirGeneric(o, ((PyInstance*)o)->type(), ((PyInstance*)o)->dataPtr(), args, kwargs);
    }
    static PyObject* clsReversed(PyObject* o, PyObject* args, PyObject* kwargs) {
        return clsReversedGeneric(o, ((PyInstance*)o)->type(), ((PyInstance*)o)->dataPtr(), args, kwargs);
    }
    static PyObject* clsComplex(PyObject* o, PyObject* args, PyObject* kwargs) {
        return clsComplexGeneric(o, ((PyInstance*)o)->type(), ((PyInstance*)o)->dataPtr(), args, kwargs);
    }
    static PyObject* clsRound(PyObject* o, PyObject* args, PyObject* kwargs) {
        return clsRoundGeneric(o, ((PyInstance*)o)->type(), ((PyInstance*)o)->dataPtr(), args, kwargs);
    }
    static PyObject* clsTrunc(PyObject* o, PyObject* args, PyObject* kwargs) {
        return clsTruncGeneric(o, ((PyInstance*)o)->type(), ((PyInstance*)o)->dataPtr(), args, kwargs);
    }
    static PyObject* clsFloor(PyObject* o, PyObject* args, PyObject* kwargs) {
        return clsFloorGeneric(o, ((PyInstance*)o)->type(), ((PyInstance*)o)->dataPtr(), args, kwargs);
    }
    static PyObject* clsCeil(PyObject* o, PyObject* args, PyObject* kwargs) {
        return clsCeilGeneric(o, ((PyInstance*)o)->type(), ((PyInstance*)o)->dataPtr(), args, kwargs);
    }
    static PyObject* clsEnter(PyObject* o, PyObject* args, PyObject* kwargs) {
        return clsEnterGeneric(o, ((PyInstance*)o)->type(), ((PyInstance*)o)->dataPtr(), args, kwargs);
    }
    static PyObject* clsExit(PyObject* o, PyObject* args, PyObject* kwargs) {
        return clsExitGeneric(o, ((PyInstance*)o)->type(), ((PyInstance*)o)->dataPtr(), args, kwargs);
    }
};
