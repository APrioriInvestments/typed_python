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

#include <Python.h>

//scoped object to ensure we're not holding the GIL. If we've
//already released it, this is a no-op. Upon destruction, we
//reacquire it.
class PyEnsureGilReleased {
public:
    PyEnsureGilReleased();

    ~PyEnsureGilReleased();

    static void gilReleaseThreadLoop();
private:
    bool m_should_reacquire;
};

//scoped object to ensure we're holding the GIL. If
//we released it in a thread above, this should reacquire it.
//if we already hold it, it should be a no-op
class PyEnsureGilAcquired {
public:
    PyEnsureGilAcquired();

    ~PyEnsureGilAcquired();

private:
    bool m_should_rerelease;
};


void assertHoldingTheGil();
