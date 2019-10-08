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

class PyOneOfInstance : public PyInstance {
public:
    typedef OneOfType modeled_type;

    static void copyConstructFromPythonInstanceConcrete(OneOfType* oneOf, instance_ptr tgt, PyObject* pyRepresentation, bool isExplicit) {
        auto tryToConvert = [&](bool usingExplicit) {
          for (long k = 0; k < oneOf->getTypes().size(); k++) {
              Type* subtype = oneOf->getTypes()[k];

              if (pyValCouldBeOfType(subtype, pyRepresentation, usingExplicit)) {
                  try {
                      copyConstructFromPythonInstance(subtype, tgt+1, pyRepresentation, usingExplicit);
                      *(uint8_t*)tgt = k;
                      return true;
                  } catch(PythonExceptionSet& e) {
                      PyErr_Clear();
                  } catch(...) {
                  }
              }
          }

          return false;
        };

        if (isExplicit) {
          //first try to convert it _not_ explicitly, so that direct conversions work
          if (tryToConvert(false)) {
            return;
          }

          //the allow agressive conversion
          if (tryToConvert(true)) {
            return;
          }
        } else {
          if (tryToConvert(false)) {
            return;
          }
        }

        throw std::logic_error("Can't initialize a " + oneOf->name() + " from an instance of " +
            std::string(pyRepresentation->ob_type->tp_name));
    }

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, bool isExplicit) {
        return true;
    }

    static PyObject* extractPythonObjectConcrete(modeled_type* oneofT, instance_ptr data) {
        std::pair<Type*, instance_ptr> child = oneofT->unwrap(data);
        return extractPythonObject(child.second, child.first);
    }

    static bool compare_to_python_concrete(OneOfType* t, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp) {
        std::pair<Type*, instance_ptr> child = t->unwrap(self);
        return compare_to_python(child.first, child.second, other, exact, pyComparisonOp);
    }

    static void mirrorTypeInformationIntoPyTypeConcrete(OneOfType* oneOfT, PyTypeObject* pyType) {
        PyObjectStealer types(PyTuple_New(oneOfT->getTypes().size()));

        for (long k = 0; k < oneOfT->getTypes().size(); k++) {
            PyTuple_SetItem(types, k, incref(typePtrToPyTypeRepresentation(oneOfT->getTypes()[k])));
        }

        //expose 'ElementType' as a member of the type object
        PyDict_SetItemString(pyType->tp_dict, "Types", types);
    }
};
