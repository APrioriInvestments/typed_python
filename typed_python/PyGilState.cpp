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

#include "PyGilState.hpp"


class ReleaseableThreadState;

// global mutex to coordinate acquirers and releasers
// note that we hold it in a pointer with a global reference to ensure
// that when the program is shut down we don't destroy the mutex object,
// as our background thread is not going to get released.
std::mutex* gilReleaseMutex = new std::mutex;

// the threadstate currently released by this thread
thread_local ReleaseableThreadState* curPyThreadState = 0;

ReleaseableThreadState* waitingPyThreadState = 0;

// is there anybody to wake us up?
int64_t gilReleaseThreadLoopActive = 0;


class ReleaseableThreadState {
public:
    // construct a releasable thread state. We must be
    // holding the GIL
    ReleaseableThreadState() {
        std::lock_guard<std::mutex> lock(*gilReleaseMutex);

        if (!gilReleaseThreadLoopActive) {
            // save our state immediately
            threadState = PyEval_SaveThread();
            isReleased = true;
        } else {
            threadState = PyThreadState_Get();
            isReleased = false;
            waitingPyThreadState = this;
        }
    }

    // release the current thread state.
    void release() {
        std::lock_guard<std::mutex> lock(*gilReleaseMutex);

        if (isReleased) {
            return;
        }

        release_();
    }

    // release the lock given that it's held and we're holding the gilReleaseMutex
    void release_() {
        if (PyThreadState_Get() != threadState) {
            std::cerr << "somehow another thread got the threadstate in ReleaseableThreadState" << std::endl;

            // this is a fatal error
            asm("int3");
        }

        // release the GIL. we're not holding it but it should be OK to release it from
        // this thread since the other thread is promising not to touch python anymore
        threadState = PyEval_SaveThread();
        isReleased = true;
    }

    // reacquire the current thread state. We must be holding the gilReleaseMutex
    void acquire() {
        {
            std::lock_guard<std::mutex> lock(*gilReleaseMutex);

            if (isReleased) {
                isReleased = false;
            } else {
                // we must be holding it
                if (waitingPyThreadState != this) {
                    std::cerr << "somehow we are not yet released, and yet not also waiting." << std::endl;

                    // this is a fatal error
                    asm("int3");
                }

                waitingPyThreadState = nullptr;

                // reacquire our own lock. we're holding the GIL at this point.
                return;
            }
        }

        PyEval_RestoreThread(threadState);
    }

    PyThreadState* threadState;

    bool isReleased;
};


PyEnsureGilReleased::PyEnsureGilReleased() :
    m_should_reacquire(false)
{
    if (curPyThreadState == nullptr) {
        curPyThreadState = new ReleaseableThreadState();
        m_should_reacquire = true;
    }
}

PyEnsureGilReleased::~PyEnsureGilReleased() {
    if (m_should_reacquire) {
        curPyThreadState->acquire();
        delete curPyThreadState;
        curPyThreadState = nullptr;
    }
}

PyEnsureGilAcquired::PyEnsureGilAcquired() :
    m_should_rerelease(false)
{
    if (curPyThreadState) {
        curPyThreadState->acquire();
        delete curPyThreadState;
        curPyThreadState = nullptr;

        m_should_rerelease = true;
    }
}

PyEnsureGilAcquired::~PyEnsureGilAcquired() {
    if (m_should_rerelease) {
        curPyThreadState = new ReleaseableThreadState();
    }
}

// check that we haven't released the GIL
void assertHoldingTheGil() {
    if (curPyThreadState) {
        std::cerr << "WARNING: assertHoldingTheGil() is failing.\n";
        asm("int3");
        throw std::runtime_error("We're not holding the gil!");
    }
}

// watch in a loop, waking up periodically to see if a thread
// wants to release its lock on the GIL
void PyEnsureGilReleased::gilReleaseThreadLoop() {
    {
        std::lock_guard<std::mutex> lock(*gilReleaseMutex);
        gilReleaseThreadLoopActive = 1;
    }

    while (true) {
        usleep(500);

        {
            std::lock_guard<std::mutex> lock(*gilReleaseMutex);
            if (waitingPyThreadState) {
                waitingPyThreadState->release_();
                waitingPyThreadState = 0;
            }
        }

    }
}
