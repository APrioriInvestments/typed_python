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

#pragma once
#include "PyInstance.hpp"
#include <functional>

class PySetInstance : public PyInstance {
  public:
    using modeled_type = SetType;

    // module definitions
    static PyMethodDef* typeMethodsConcrete(Type* t);
    static PyObject* setAdd(PyObject* o, PyObject* args);
    static PyObject* setContains(PyObject* o, PyObject* args);
    static PyObject* setDiscard(PyObject* o, PyObject* args);
    static PyObject* setRemove(PyObject* o, PyObject* args);
    static PyObject* setClear(PyObject* o, PyObject* args);
    static PyObject* setCopy(PyObject* o, PyObject* args);
    static PyObject* setUnion(PyObject* o, PyObject* args);
    static PyObject* setIntersection(PyObject* o, PyObject* args);
    static PyObject* setDifference(PyObject* o, PyObject* args);
    Py_ssize_t mp_and_sq_length_concrete();
    int sq_contains_concrete(PyObject* item);
    PyObject* tp_iter_concrete();
    PyObject* tp_iternext_concrete();

    static void copyConstructFromPythonInstanceConcrete(SetType* setType, instance_ptr tgt,
                                                        PyObject* pyRepresentation,
                                                        bool isExplicit);
    static void mirrorTypeInformationIntoPyTypeConcrete(SetType* setType, PyTypeObject* pyType);
    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation,
                                           bool isExplicit) {
        return true;
    }

  private:
    static int try_insert_key(PySetInstance* self, PyObject* pyKey, instance_ptr key);
    static PyObject* try_remove(PyObject* o, PyObject* item, bool assertKeyError = false);
    static void copy_elements(PyObject* dst, PyObject* src);
    static PyObject* set_intersection(PyObject* o, PyObject* other);
    static PyObject* set_difference(PyObject* o, PyObject* other);
    static int set_difference_update(PyObject* o, PyObject* other);
    SetType* type();
    static void getDataFromNative(PySetInstance* src, std::function<void(instance_ptr)> func);
    static void getDataFromNative(PyTupleOrListOfInstance* src,
                                  std::function<void(instance_ptr)> func);
};

