/******************************************************************************
   Copyright 2017-2022 typed_python Authors

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

#include "Type.hpp"

class ClosureVariableBindingStep {
    enum class BindingType {
        FUNCTION = 1,
        NAMED_FIELD = 2,
        INDEXED_FIELD = 3,
        ACCESS_CELL = 4
    };

    ClosureVariableBindingStep() :
        mKind(BindingType::ACCESS_CELL),
        mIndexedFieldToAccess(0),
        mFunctionToBind(nullptr)
    {}

public:
    ClosureVariableBindingStep(Type* bindFunction) :
        mKind(BindingType::FUNCTION),
        mIndexedFieldToAccess(0),
        mFunctionToBind(bindFunction)
    {}

    ClosureVariableBindingStep(std::string fieldAccess) :
        mKind(BindingType::NAMED_FIELD),
        mIndexedFieldToAccess(0),
        mFunctionToBind(nullptr),
        mNamedFieldToAccess(fieldAccess)
    {}

    ClosureVariableBindingStep(int elementAccess) :
        mKind(BindingType::INDEXED_FIELD),
        mFunctionToBind(nullptr),
        mIndexedFieldToAccess(elementAccess)
    {}

    template<class visitor_type>
    void _visitCompilerVisibleInternals(const visitor_type& v) {
        if (isFunction()) {
            v.visitHash(ShaHash(1));
            v.visitTopo(getFunction());
        } else
        if (isNamedField()) {
            v.visitHash(ShaHash(2) + ShaHash(getNamedField()));
        } else
        if (isIndexedField()) {
            v.visitHash(ShaHash(3) + ShaHash(getIndexedField()));
        } else
        if (isCellAccess()) {
            v.visitHash(ShaHash(4));
        } else {
            v.visitErr("Invalid ClosureVariableBindingStep found");
        }
    }

    static ClosureVariableBindingStep AccessCell() {
        ClosureVariableBindingStep step;
        return step;
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        if (mKind == BindingType::FUNCTION) {
            visitor(mFunctionToBind);
        }
    }

    bool isFunction() const {
        return mKind == BindingType::FUNCTION;
    }

    bool isNamedField() const {
        return mKind == BindingType::NAMED_FIELD;
    }

    bool isIndexedField() const {
        return mKind == BindingType::INDEXED_FIELD;
    }

    bool isCellAccess() const {
        return mKind == BindingType::ACCESS_CELL;
    }

    Type* getFunction() const {
        if (!isFunction()) {
            throw std::runtime_error("Binding is not a function");
        }

        return mFunctionToBind;
    }

    std::string getNamedField() const {
        if (!isNamedField()) {
            throw std::runtime_error("Binding is not a named field bindng");
        }

        return mNamedFieldToAccess;
    }

    int getIndexedField() const {
        if (!isIndexedField()) {
            throw std::runtime_error("Binding is not an index field bindng");
        }

        return mIndexedFieldToAccess;
    }

    bool operator<(const ClosureVariableBindingStep& step) const {
        if (mKind < step.mKind) {
            return true;
        }

        if (mKind > step.mKind) {
            return false;
        }

        if (mKind == BindingType::ACCESS_CELL) {
            return false;
        }

        if (mKind == BindingType::FUNCTION) {
            return mFunctionToBind < step.mFunctionToBind;
        }

        if (mKind == BindingType::NAMED_FIELD) {
            return mNamedFieldToAccess < step.mNamedFieldToAccess;
        }

        if (mKind == BindingType::INDEXED_FIELD) {
            return mIndexedFieldToAccess < step.mIndexedFieldToAccess;
        }

        return false;
    }

    template<class serialization_context_t, class buf_t>
    void serialize(serialization_context_t& context, buf_t& buffer, int fieldNumber) const {
        buffer.writeBeginCompound(fieldNumber);

        if (mKind == BindingType::ACCESS_CELL) {
            buffer.writeUnsignedVarintObject(0, 0);
        }
        else if (mKind == BindingType::FUNCTION) {
            buffer.writeUnsignedVarintObject(0, 1);
            context.serializeNativeType(mFunctionToBind, buffer, 1);
        }
        else if (mKind == BindingType::NAMED_FIELD) {
            buffer.writeUnsignedVarintObject(0, 2);
            buffer.writeStringObject(1, mNamedFieldToAccess);
        }
        else if (mKind == BindingType::INDEXED_FIELD) {
            buffer.writeUnsignedVarintObject(0, 3);
            buffer.writeUnsignedVarintObject(1, mIndexedFieldToAccess);
        }

        buffer.writeEndCompound();
    }

    template<class serialization_context_t, class buf_t>
    static ClosureVariableBindingStep deserialize(serialization_context_t& context, buf_t& buffer, int wireType) {
        ClosureVariableBindingStep out;

        int whichBinding = -1;

        buffer.consumeCompoundMessage(wireType, [&](size_t fieldNumber, size_t wireType) {
            if (fieldNumber == 0) {
                assertWireTypesEqual(wireType, WireType::VARINT);
                whichBinding = buffer.readUnsignedVarint();

                if (whichBinding == 0) {
                    out = ClosureVariableBindingStep::AccessCell();
                }
            }
            else if (fieldNumber == 1) {
                if (whichBinding == -1) {
                    throw std::runtime_error("Corrupt ClosureVariableBindingStep");
                }
                if (whichBinding == 1) {
                    out = ClosureVariableBindingStep(
                        context.deserializeNativeType(buffer, wireType)
                    );
                }
                else if (whichBinding == 2) {
                    assertWireTypesEqual(wireType, WireType::BYTES);
                    out = ClosureVariableBindingStep(buffer.readStringObject());
                }
                else if (whichBinding == 3) {
                    assertWireTypesEqual(wireType, WireType::VARINT);
                    out = ClosureVariableBindingStep(buffer.readUnsignedVarint());
                }
            } else {
                throw std::runtime_error("Corrupt ClosureVariableBindingStep");
            }
        });

        return out;
    }


private:
    BindingType mKind;

    // this can be a Function or a Forward that will become a function
    Type* mFunctionToBind;

    std::string mNamedFieldToAccess;

    int mIndexedFieldToAccess;
};


class ClosureVariableBinding {
public:
    ClosureVariableBinding() {}

    ClosureVariableBinding(const std::vector<ClosureVariableBindingStep>& steps) :
        mSteps(new std::vector<ClosureVariableBindingStep>(steps))
    {}

    ClosureVariableBinding(const std::vector<ClosureVariableBindingStep>& steps, ClosureVariableBindingStep step) :
        mSteps(new std::vector<ClosureVariableBindingStep>(steps))
    {
        mSteps->push_back(step);
    }

    ClosureVariableBinding(const ClosureVariableBinding& other) : mSteps(other.mSteps)
    {}

    template<class visitor_type>
    void _visitCompilerVisibleInternals(const visitor_type& v) {
        v.visitHash(ShaHash(mSteps->size()));

        for (auto step: *mSteps) {
            step._visitCompilerVisibleInternals(v);
        }
    }

    ClosureVariableBinding& operator=(const ClosureVariableBinding& other) {
        mSteps = other.mSteps;
        return *this;
    }

    ClosureVariableBinding operator+(ClosureVariableBindingStep step) {
        if (mSteps) {
            return ClosureVariableBinding(*mSteps, step);
        }

        return ClosureVariableBinding(std::vector<ClosureVariableBindingStep>(), step);
    }

    template<class serialization_context_t, class buf_t>
    void serialize(serialization_context_t& context, buf_t& buffer, int fieldNumber) const {
        buffer.writeBeginCompound(fieldNumber);
        for (long stepIx = 0; stepIx < size(); stepIx++) {
            (*this)[stepIx].serialize(context, buffer, stepIx);
        }
        buffer.writeEndCompound();
    }

    template<class serialization_context_t, class buf_t>
    static ClosureVariableBinding deserialize(serialization_context_t& context, buf_t& buffer, int wireType) {
        std::vector<ClosureVariableBindingStep> steps;
        buffer.consumeCompoundMessage(wireType, [&](size_t fieldNumber, size_t wireType) {
            steps.push_back(ClosureVariableBindingStep::deserialize(context, buffer, wireType));
        });

        return ClosureVariableBinding(steps);
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        if (!mSteps) {
            return;
        }

        for (auto& step: *mSteps) {
            step._visitReferencedTypes(visitor);
        }
    }

    ClosureVariableBinding withShiftedFrontBinding(long amount) const {
        if (!size()) {
            throw std::runtime_error("Empty Binding can't be shifted.");
        }

        if (!(*this)[0].isIndexedField()) {
            throw std::runtime_error("Shifting the first binding only makes sense if it's an indexed lookup");
        }

        std::vector<ClosureVariableBindingStep> steps;
        steps.push_back(ClosureVariableBindingStep((*this)[0].getIndexedField() + amount));

        for (long k = 1; k < size(); k++) {
            steps.push_back((*this)[k]);
        }

        return ClosureVariableBinding(steps);
    }

    size_t size() const {
        if (mSteps) {
            return mSteps->size();
        }

        return 0;
    }

    bool operator<(const ClosureVariableBinding& other) const {
        if (size() < other.size()) {
            return true;
        }
        if (size() > other.size()) {
            return false;
        }
        if (!size()) {
            return false;
        }

        return *mSteps < *other.mSteps;
    }

    ClosureVariableBindingStep operator[](int i) const {
        if (i < 0 || i >= size()) {
            throw std::runtime_error("ClosureVariableBinding index out of bounds");
        }

        return (*mSteps)[i];
    }

    Instance extractValueOrContainingClosure(Type* closureType, instance_ptr data);

private:
    std::shared_ptr<std::vector<ClosureVariableBindingStep> > mSteps;
};


inline ClosureVariableBinding operator+(const ClosureVariableBindingStep& step, const ClosureVariableBinding& binding) {
    std::vector<ClosureVariableBindingStep> steps;
    steps.push_back(step);
    for (long k = 0; k < binding.size(); k++) {
        steps.push_back(binding[k]);
    }
    return ClosureVariableBinding(steps);
}
