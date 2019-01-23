#pragma once

#include <sstream>
#include <set>

class ReprAccumulator {
public:
    ReprAccumulator(std::ostringstream& stream) :
        m_stream(stream)
    {
    }

    //record that we have seen this object, and then return whether it was redundant.
    bool pushObject(void* ptr) {
        bool hasSeen = m_seen_objects[ptr] > 0;
        m_seen_objects[ptr]++;
        return !hasSeen;
    }

    void popObject(void* ptr) {
        m_seen_objects[ptr]--;
    }

    template<class T>
    ReprAccumulator& operator<<(const T& in) {
        m_stream << in;
        return *this;
    }

private:
    std::ostringstream& m_stream;

    std::map<void*, int64_t> m_seen_objects;
};

class PushReprState {
public:
    template<class T>
    PushReprState(ReprAccumulator& accumulator, T* ptr) :
        m_accumulator(accumulator),
        m_ptr((void*)ptr),
        m_was_new(false)
    {
        m_was_new = m_accumulator.pushObject(m_ptr);
    }

    ~PushReprState() {
        m_accumulator.popObject(m_ptr);
    }

    operator bool() const {
        return m_was_new;
    }

private:
    ReprAccumulator& m_accumulator;
    void* m_ptr;
    bool m_was_new;
};
