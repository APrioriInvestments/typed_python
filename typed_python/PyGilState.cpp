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

#include "PyGilState.hpp"

PyEnsureGilReleased::PyEnsureGilReleased() :
    m_should_reaquire(false)
{
    if (curPyThreadState == nullptr) {
        curPyThreadState = PyEval_SaveThread();
        m_should_reaquire = true;
    }
}

PyEnsureGilReleased::~PyEnsureGilReleased() {
    if (m_should_reaquire) {
        PyEval_RestoreThread(curPyThreadState);
        curPyThreadState = nullptr;
    }
}

PyEnsureGilAcquired::PyEnsureGilAcquired() :
    m_should_rerelease(false)
{
    if (curPyThreadState) {
        PyEval_RestoreThread(curPyThreadState);
        m_should_rerelease = true;
        curPyThreadState = nullptr;
    }
}

PyEnsureGilAcquired::~PyEnsureGilAcquired() {
    if (m_should_rerelease) {
        curPyThreadState = PyEval_SaveThread();
    }
}

void assertHoldingTheGil() {
    if (curPyThreadState) {
        throw std::runtime_error("We're not holding the gil!");
    }
}
