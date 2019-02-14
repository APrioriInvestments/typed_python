#pragma once

#include <Python.h>

// thread-local counter for the currently released threadstate
extern thread_local PyThreadState* curPyThreadState;

//scoped object to ensure we're not holding the GIL. If we've
//already released it, this is a no-op. Upon destruction, we
//reacquire it.
class PyEnsureGilReleased {
public:
    PyEnsureGilReleased() :
        m_should_reaquire(false)
    {
        if (curPyThreadState == nullptr) {
            curPyThreadState = PyEval_SaveThread();
            m_should_reaquire = true;
        }
    }

    ~PyEnsureGilReleased() {
        if (m_should_reaquire) {
            PyEval_RestoreThread(curPyThreadState);
            curPyThreadState = nullptr;
        }
    }

private:
    bool m_should_reaquire;
};

//scoped object to ensure we're holding the GIL. If
//we released it in a thread above, this should reacquire it.
//if we already hold it, it should be a no-op
class PyEnsureGilAcquired {
public:
    PyEnsureGilAcquired() :
        m_should_rerelease(false)
    {
        if (curPyThreadState) {
            PyEval_RestoreThread(curPyThreadState);
            m_should_rerelease = true;
            curPyThreadState = nullptr;
        }
    }

    ~PyEnsureGilAcquired() {
        if (m_should_rerelease) {
            curPyThreadState = PyEval_SaveThread();
        }
    }

private:
    bool m_should_rerelease;
};

inline void assertHoldingTheGil() {
    if (curPyThreadState) {
        throw std::runtime_error("We're not holding the gil!");
    }
}
