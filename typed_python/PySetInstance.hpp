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

#pragma once
#include "PyInstance.hpp"
#include <functional>

class PySetInstance : public PyInstance {
  public:
    using modeled_type = SetType;

    // module definitions
    static PyMethodDef* typeMethodsConcrete(Type* t);
    static PyObject* setPop(PyObject* o, PyObject* args);
    static PyObject* setAdd(PyObject* o, PyObject* args);
    static PyObject* setContains(PyObject* o, PyObject* args);
    static PyObject* setDiscard(PyObject* o, PyObject* args);
    static PyObject* setRemove(PyObject* o, PyObject* args);
    static PyObject* setClear(PyObject* o, PyObject* args);
    static PyObject* setCopy(PyObject* o, PyObject* args);
    static PyObject* setUnion(PyObject* o, PyObject* args);
    static PyObject* setUpdate(PyObject* o, PyObject* args);
    static PyObject* setIntersection(PyObject* o, PyObject* args);
    static PyObject* setIntersectionUpdate(PyObject* o, PyObject* args);
    static PyObject* setDifference(PyObject* o, PyObject* args);
    static PyObject* setDifferenceUpdate(PyObject* o, PyObject* args);
    static PyObject* setSymmetricDifference(PyObject* o, PyObject* args);
    static PyObject* setSymmetricDifferenceUpdate(PyObject* o, PyObject* args);
    static PyObject* setIsSubset(PyObject* o, PyObject* args);
    static PyObject* setIsSuperset(PyObject* o, PyObject* args);
    static PyObject* setIsDisjoint(PyObject* o, PyObject* args);
    Py_ssize_t mp_and_sq_length_concrete();
    int sq_contains_concrete(PyObject* item);
    PyObject* tp_iter_concrete();
    PyObject* tp_iternext_concrete();

    static void copyConstructFromPythonInstanceConcrete(SetType* setType, instance_ptr tgt,
                                                        PyObject* pyRepresentation,
                                                        ConversionLevel level);
    static void constructFromPythonArgumentsConcrete(SetType* t, uint8_t* data, PyObject* args, PyObject* kwargs);
    static void mirrorTypeInformationIntoPyTypeConcrete(SetType* setType, PyTypeObject* pyType);
    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, ConversionLevel level);
    static bool compare_to_python_concrete(SetType* setT, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp);
    int pyInquiryConcrete(const char* op, const char* opErrRep);
    PyObject* pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr);
    PyObject* pyOperatorDifference(PyObject* rhs, const char* op, const char* opErr, bool reversed);
    PyObject* pyOperatorSymmetricDifference(PyObject* rhs, const char* op, const char* opErr, bool reversed);
    PyObject* pyOperatorUnion(PyObject* rhs, const char* op, const char* opErr, bool reversed);
    PyObject* pyOperatorIntersection(PyObject* rhs, const char* op, const char* opErr, bool reversed);

  private:
    static void insertKey(PySetInstance* self, PyObject* pyKey, instance_ptr key);
    static PyObject* try_remove(PyObject* o, PyObject* item, bool assertKeyError = false);
    static PyObject* try_add_if_not_found(PyObject* o, PySetInstance* to_be_added, PyObject* item);
    static void copy_elements(PyObject* dst, PyObject* src);
    static PyObject* set_union(PyObject* o, PyObject* other);
    static PyObject* set_intersection(PyObject* o, PyObject* other);
    static int set_intersection_update(PyObject* o, PyObject* other);
    static PyObject* set_difference(PyObject* o, PyObject* other);
    static int set_difference_update(PyObject* o, PyObject* other);
    static PyObject* set_symmetric_difference(PyObject* o, PyObject* other);
    static int set_symmetric_difference_update(PyObject* o, PyObject* other);
    static int set_is_subset(PyObject* o, PyObject* other);
    static int set_is_superset(PyObject* o, PyObject* other);
    static int set_is_disjoint(PyObject* o, PyObject* other);
    SetType* type();
    static void getDataFromNative(PySetInstance* src, std::function<void(instance_ptr)> func);
    static void getDataFromNative(PyTupleOrListOfInstance* src,
                                  std::function<void(instance_ptr)> func);
    static bool subset(SetType* setT, instance_ptr left, PyObject* right);
    static bool superset(SetType* setT, instance_ptr left, PyObject* right);
};
