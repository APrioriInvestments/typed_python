/******************************************************************************
   Copyright 2017-2023 typed_python Authors

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

#include "PyInstance.hpp"
#include "ForwardType.hpp"

/* static */
Forward* Forward::MakeFromFunction(PyObject* funcObj) {
    if (!PyFunction_Check(funcObj)) {
        throw std::runtime_error(
            "Forwards can only be made from functions which return a single variable lookup."
        );
    }

    PyCodeObject* code = (PyCodeObject*)PyFunction_GetCode(funcObj);
    PyObject* closure = PyFunction_GetClosure(funcObj);

    if (!PyTuple_Check(code->co_names)) {
        throw std::runtime_error("Corrupt Forward lookup function: Code object co_names is not a tuple");
    }

    if (closure) {
        PyObjectStealer coFreevars(PyObject_GetAttrString(PyFunction_GetCode(funcObj), "co_freevars"));

        if (PyTuple_Size(code->co_names)) {
            throw std::runtime_error("Corrupt Forward lookup function: can't have more than one name");
        }

        if (!PyTuple_Check(coFreevars)) {
            throw std::runtime_error("Corrupt Forward lookup function: co_freevars was not a tuple");
        }

        if (PyTuple_Size(coFreevars) != PyTuple_Size(closure)) {
            throw std::runtime_error("Corrupt Forward lookup function: co_freevars not the same size as closure");
        }

        if (PyTuple_Size(coFreevars) != 1) {
            throw std::runtime_error("Corrupt Forward lookup function: can't have more than 1 free variable");
        }

        PyObject* varname = PyTuple_GetItem(coFreevars, 0);
        if (!PyUnicode_Check(varname)) {
            throw std::runtime_error("Corrupt Forward lookup function: closure varnames need to be strings");
        }

        std::string name = PyUnicode_AsUTF8(varname);

        if (!PyCell_Check(PyTuple_GetItem(closure, 0))) {
            throw std::runtime_error("Corrupt Forward lookup function: closure was not a cell");
        }

        return new Forward(name, PyTuple_GetItem(closure, 0));
    }

    if (PyTuple_Size(code->co_names) != 1) {
        throw std::runtime_error("Corrupt Forward lookup function: Code object co_names refers to more than 1 name");
    }

    if (!PyUnicode_Check(PyTuple_GetItem(code->co_names, 0))) {
        throw std::runtime_error("Corrupt Forward lookup function: Code object co_names refers to more than 1 name");
    }

    std::string name = PyUnicode_AsUTF8(PyTuple_GetItem(code->co_names, 0));
    return new Forward(name, PyFunction_GetGlobals(funcObj));
}

Type* Forward::lambdaDefinition() {
    if (!mCellOrDict) {
        return nullptr;
    }

    if (PyCell_Check(mCellOrDict) && PyCell_Get(mCellOrDict)) {
        Type* res = PyInstance::unwrapTypeArgToTypePtr(PyCell_Get(mCellOrDict));

        if (res == this) {
            return nullptr;
        }

        return res;
    }

    if (PyDict_Check(mCellOrDict)) {
        PyObject* item = PyDict_GetItemString(mCellOrDict, m_name.c_str());

        if (!item) {
            return nullptr;
        }

        Type* res = PyInstance::unwrapTypeArgToTypePtr(item);

        if (res == this) {
            return nullptr;
        }

        return res;
    }

    return nullptr;
}

bool Forward::lambdaDefinitionPopulated() {
    return lambdaDefinition();
}

void Forward::installDefinitionIfLambda() {
    if (!mCellOrDict) {
        return;
    }

    if (!m_forward_resolves_to) {
        return;
    }

    if (PyCell_Check(mCellOrDict) && PyCell_Get(mCellOrDict)) {
        if (PyType_Check(PyCell_Get(mCellOrDict))) {
            Type* cellContents = PyInstance::extractTypeFrom(
                (PyTypeObject*)PyCell_Get(mCellOrDict)
            );

            if (cellContents) {
                // see if cellContents resolves
                if (cellContents->isResolved()
                        && cellContents->forwardResolvesTo()
                            == m_forward_resolves_to) {
                    PyCell_Set(
                        mCellOrDict,
                        (PyObject*)PyInstance::typeObj(
                            m_forward_resolves_to
                        )
                    );
                }
            }
        }

        return;
    }

    if (PyDict_Check(mCellOrDict)) {
        PyObject* curContents = PyDict_GetItemString(mCellOrDict, m_name.c_str());

        if (curContents && PyType_Check(curContents)) {
            Type* curContentsType = PyInstance::extractTypeFrom(
                (PyTypeObject*)curContents
            );

            if (curContentsType && curContentsType->isResolved() &&
                    curContentsType->forwardResolvesTo()
                        == m_forward_resolves_to) {
                PyDict_SetItemString(
                    mCellOrDict,
                    m_name.c_str(),
                    (PyObject*)PyInstance::typeObj(
                        m_forward_resolves_to
                    )
                );
            }
        }
    }
}
