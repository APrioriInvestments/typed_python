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


Type* Forward::lambdaDefinition() {
    if (!mCellOrDict) {
        return nullptr;
    }

    if (PyCell_Check(mCellOrDict) && PyCell_Get(mCellOrDict)) {
        return PyInstance::unwrapTypeArgToTypePtr(PyCell_Get(mCellOrDict));
    }

    if (PyDict_Check(mCellOrDict)) {
        PyObject* item = PyDict_GetItemString(mCellOrDict, m_name.c_str());

        if (!item) {
            return nullptr;
        }

        return PyInstance::unwrapTypeArgToTypePtr(item);
    }

    return nullptr;
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
