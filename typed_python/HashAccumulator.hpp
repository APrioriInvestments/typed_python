#pragma once

class Hash32Accumulator {
public:
    Hash32Accumulator(int32_t init) :
        m_state(init)
    {
    }

    void add(int32_t i) {
        m_state = (m_state * 1000003) ^ i;
    }

    void addBytes(uint8_t* bytes, int64_t count) {
        while (count >= 4) {
            add(*(int32_t*)bytes);
            bytes += 4;
            count -= 4;
        }
        while (count) {
            add((uint8_t)*bytes);
            bytes++;
            count--;
        }
    }

    int32_t get() const {
        return m_state;
    }

    void addRegister(bool i) { add(i ? 1:0); }
    void addRegister(uint8_t i) { add(i); }
    void addRegister(uint16_t i) { add(i); }
    void addRegister(uint32_t i) { add(i); }
    void addRegister(uint64_t i) { addBytes((uint8_t*)&i, sizeof(i)); }

    void addRegister(int8_t i) { add(i); }
    void addRegister(int16_t i) { add(i); }
    void addRegister(int32_t i) { add(i); }
    void addRegister(int64_t i) { addBytes((uint8_t*)&i, sizeof(i)); }

    void addRegister(float i) { addBytes((uint8_t*)&i, sizeof(i)); }
    void addRegister(double i) { addBytes((uint8_t*)&i, sizeof(i)); }

private:
    int32_t m_state;
};
