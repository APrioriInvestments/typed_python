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

#include "Type.hpp"
#include "TypedCellType.hpp"
#include "ReprAccumulator.hpp"
#include "Format.hpp"

class Function;


class ClosureVariableBindingStep {
    enum class BindingType {
        FUNCTION = 1,
        NAMED_FIELD = 2,
        INDEXED_FIELD = 3,
        ACCESS_CELL = 4
    };

    ClosureVariableBindingStep() :
        mKind(BindingType::ACCESS_CELL)
    {}

public:
    ClosureVariableBindingStep(Type* bindFunction) :
        mKind(BindingType::FUNCTION),
        mFunctionToBind(bindFunction)
    {}

    ClosureVariableBindingStep(std::string fieldAccess) :
        mKind(BindingType::NAMED_FIELD),
        mNamedFieldToAccess(fieldAccess)
    {}

    ClosureVariableBindingStep(int elementAccess) :
        mKind(BindingType::INDEXED_FIELD),
        mIndexedFieldToAccess(elementAccess)
    {}

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


class Function : public Type {
public:
    class FunctionArg {
    public:
        FunctionArg(std::string name, Type* typeFilterOrNull, PyObject* defaultValue, bool isStarArg, bool isKwarg) :
            m_name(name),
            m_typeFilter(typeFilterOrNull),
            m_defaultValue(defaultValue),
            m_isStarArg(isStarArg),
            m_isKwarg(isKwarg)
        {
            assert(!(isStarArg && isKwarg));
        }

        std::string getName() const {
            return m_name;
        }

        PyObject* getDefaultValue() const {
            return m_defaultValue;
        }

        Type* getTypeFilter() const {
            return m_typeFilter;
        }

        bool getIsStarArg() const {
            return m_isStarArg;
        }

        bool getIsKwarg() const {
            return m_isKwarg;
        }

        bool getIsNormalArg() const {
            return !m_isKwarg && !m_isStarArg;
        }

        template<class visitor_type>
        void _visitReferencedTypes(const visitor_type& visitor) {
            if (m_typeFilter) {
                visitor(m_typeFilter);
            }
        }

        bool operator<(const FunctionArg& other) const {
            if (m_name < other.m_name) {
                return true;
            }
            if (m_name > other.m_name) {
                return false;
            }
            if (m_typeFilter < other.m_typeFilter) {
                return true;
            }
            if (m_typeFilter > other.m_typeFilter) {
                return false;
            }
            if (m_defaultValue < other.m_defaultValue) {
                return true;
            }
            if (m_defaultValue > other.m_defaultValue) {
                return false;
            }
            if (m_isStarArg < other.m_isStarArg) {
                return true;
            }
            if (m_isStarArg > other.m_isStarArg) {
                return false;
            }
            if (m_isKwarg < other.m_isKwarg) {
                return true;
            }
            if (m_isKwarg > other.m_isKwarg) {
                return false;
            }

            return false;
        }

    private:
        std::string m_name;
        Type* m_typeFilter;
        PyObject* m_defaultValue;
        bool m_isStarArg;
        bool m_isKwarg;
    };

    class CompiledSpecialization {
    public:
        CompiledSpecialization(
                    compiled_code_entrypoint funcPtr,
                    Type* returnType,
                    const std::vector<Type*>& argTypes
                    ) :
            mFuncPtr(funcPtr),
            mReturnType(returnType),
            mArgTypes(argTypes)
        {}

        compiled_code_entrypoint getFuncPtr() const {
            return mFuncPtr;
        }

        Type* getReturnType() const {
            return mReturnType;
        }

        const std::vector<Type*>& getArgTypes() const {
            return mArgTypes;
        }

    private:
        compiled_code_entrypoint mFuncPtr;
        Type* mReturnType;
        std::vector<Type*> mArgTypes;
    };

    class Overload {
    public:
        Overload(
            PyObject* pyFuncCode,
            PyObject* pyFuncGlobals,
            PyObject* pyFuncDefaults,
            PyObject* pyFuncAnnotations,
            const std::map<std::string, PyObject*>& pyFuncGlobalsInCells,
            const std::vector<std::string>& pyFuncClosureVarnames,
            const std::map<std::string, ClosureVariableBinding> closureBindings,
            Type* returnType,
            const std::vector<FunctionArg>& args
            ) :
                mFunctionCode(incref(pyFuncCode)),
                mFunctionGlobals(incref(pyFuncGlobals)),
                mFunctionDefaults(incref(pyFuncDefaults)),
                mFunctionAnnotations(incref(pyFuncAnnotations)),
                mFunctionGlobalsInCells(pyFuncGlobalsInCells),
                mFunctionClosureVarnames(pyFuncClosureVarnames),
                mReturnType(returnType),
                mArgs(args),
                mCompiledCodePtr(nullptr),
                mHasKwarg(false),
                mHasStarArg(false),
                mMinPositionalArgs(0),
                mMaxPositionalArgs(-1),
                mClosureBindings(closureBindings),
                mCachedFunctionObj(nullptr)
        {
            long argsWithDefaults = 0;
            long argsDefinitelyConsuming = 0;

            for (auto arg: mArgs) {
                if (arg.getIsStarArg()) {
                    mHasStarArg = true;
                }
                else if (arg.getIsKwarg()) {
                    mHasKwarg = true;
                }
                else if (arg.getDefaultValue()) {
                    argsWithDefaults++;
                } else {
                    argsDefinitelyConsuming++;
                }
            }

            mMinPositionalArgs = argsDefinitelyConsuming;
            if (!mHasStarArg) {
                mMaxPositionalArgs = argsDefinitelyConsuming + argsWithDefaults;
            }
        }

        ~Overload() {
            if (mCachedFunctionObj) {
                decref(mCachedFunctionObj);
            }
        }

        Overload withShiftedFrontClosureBindings(long shiftAmount) const {
            std::map<std::string, ClosureVariableBinding> bindings;
            for (auto nameAndBinding: mClosureBindings) {
                bindings[nameAndBinding.first] = nameAndBinding.second.withShiftedFrontBinding(shiftAmount);
            }

            return Overload(
                mFunctionCode,
                mFunctionGlobals,
                mFunctionDefaults,
                mFunctionAnnotations,
                mFunctionGlobalsInCells,
                mFunctionClosureVarnames,
                bindings,
                mReturnType,
                mArgs
            );
        }

        Overload withClosureBindings(const std::map<std::string, ClosureVariableBinding> &bindings) const {
            return Overload(
                mFunctionCode,
                mFunctionGlobals,
                mFunctionDefaults,
                mFunctionAnnotations,
                mFunctionGlobalsInCells,
                mFunctionClosureVarnames,
                bindings,
                mReturnType,
                mArgs
            );
        }

        std::string toString() const {
            std::ostringstream str;

            str << "(";

            for (long k = 0; k < mArgs.size(); k++) {
                if (k) {
                    str << ", ";
                }

                if (mArgs[k].getIsStarArg()) {
                    str << "*";
                }

                if (mArgs[k].getIsKwarg()) {
                    str << "**";
                }

                str << mArgs[k].getName();

                if (mArgs[k].getDefaultValue()) {
                    str << "=...";
                }

                if (mArgs[k].getTypeFilter()) {
                    str << ": " << mArgs[k].getTypeFilter()->name();
                }
            }

            str << ")";

            if (mReturnType) {
                str << " -> " << mReturnType->name();
            }

            return str.str();
        }

        // return the FunctionArg* that a positional argument would map to, or 'nullptr' if
        // it wouldn't
        const FunctionArg* argForPositionalArgument(long argIx) const {
            if (argIx >= mArgs.size()) {
                return nullptr;
            }

            if (mArgs[argIx].getIsStarArg() || mArgs[argIx].getIsKwarg()) {
                return nullptr;
            }

            return &mArgs[argIx];
        }

        // can we possibly match 'argCount' positional arguments?
        bool couldMatchPositionalCount(long argCount) const {
            return argCount >= mMinPositionalArgs && argCount < mMaxPositionalArgs;
        }

        bool disjointFrom(const Overload& other) const {
            // we need to determine if all possible call signatures of these overloads
            // would route to one or the other unambiguously. we ignore keyword callsignatures
            // for the moment. For each possible positional argument, if we get disjointedness
            // then the whole set is disjoint.

            // if the set of numbers of arguments we can accept are disjoint, then we can't possibly
            // match the same queries.
            if (mMaxPositionalArgs < other.mMinPositionalArgs || other.mMaxPositionalArgs < mMinPositionalArgs) {
                return true;
            }

            // now check each positional argument
            for (long k = 0; k < mArgs.size() && k < other.mArgs.size(); k++) {
                const FunctionArg* arg1 = argForPositionalArgument(k);
                const FunctionArg* arg2 = other.argForPositionalArgument(k);

                if (arg1 && arg2 && !arg1->getDefaultValue() && !arg2->getDefaultValue() && arg1->getTypeFilter() && arg2->getTypeFilter()) {
                    if (arg1->getTypeFilter()->canConstructFrom(arg2->getTypeFilter(), false) == Maybe::False) {
                        return true;
                    }
                }
            }

            return false;
        }

        Type* getReturnType() const {
            return mReturnType;
        }

        const std::vector<FunctionArg>& getArgs() const {
            return mArgs;
        }

        template<class visitor_type>
        void _visitReferencedTypes(const visitor_type& visitor) {
            if (mReturnType) {
                visitor(mReturnType);
            }
            for (auto& a: mArgs) {
                a._visitReferencedTypes(visitor);
            }
            for (auto& varnameAndBinding: mClosureBindings) {
                varnameAndBinding.second._visitReferencedTypes(visitor);
            }
        }

        template<class visitor_type>
        void _visitContainedTypes(const visitor_type& visitor) {
        }

        const std::vector<CompiledSpecialization>& getCompiledSpecializations() const {
            return mCompiledSpecializations;
        }

        void addCompiledSpecialization(compiled_code_entrypoint e, Type* returnType, const std::vector<Type*>& argTypes) {
            mCompiledSpecializations.push_back(CompiledSpecialization(e,returnType,argTypes));
        }

        void touchCompiledSpecializations() {
            //force the memory for the compiled specializations to move.
            std::vector<CompiledSpecialization> other = mCompiledSpecializations;
            std::swap(mCompiledSpecializations, other);
        }

        bool operator<(const Overload& other) const {
            if (mFunctionCode < other.mFunctionCode) { return true; }
            if (mFunctionCode > other.mFunctionCode) { return false; }

            if (mFunctionGlobals < other.mFunctionGlobals) { return true; }
            if (mFunctionGlobals > other.mFunctionGlobals) { return false; }

            if (mFunctionGlobalsInCells < other.mFunctionGlobalsInCells) { return true; }
            if (mFunctionGlobalsInCells > other.mFunctionGlobalsInCells) { return false; }

            if (mClosureBindings < other.mClosureBindings) { return true; }
            if (mClosureBindings > other.mClosureBindings) { return false; }

            if (mReturnType < other.mReturnType) { return true; }
            if (mReturnType > other.mReturnType) { return false; }

            if (mArgs < other.mArgs) { return true; }
            if (mArgs > other.mArgs) { return false; }

            return false;
        }

        const std::map<std::string, ClosureVariableBinding>& getClosureVariableBindings() const {
            return mClosureBindings;
        }

        PyObject* getFunctionCode() const {
            return mFunctionCode;
        }

        PyObject* getFunctionGlobals() const {
            return mFunctionGlobals;
        }

        const std::map<std::string, PyObject*> getFunctionGlobalsInCells() const {
            return mFunctionGlobalsInCells;
        }

        // create a new function object for this closure (or cache it
        // if we have no closure)
        PyObject* buildFunctionObj(Type* closureType, instance_ptr closureData) const;

        Overload& operator=(const Overload& other) {
            mFunctionCode = incref(other.mFunctionCode);
            mFunctionGlobals = incref(other.mFunctionGlobals);
            mFunctionDefaults = incref(other.mFunctionDefaults);
            mFunctionAnnotations = incref(other.mFunctionAnnotations);
            mFunctionClosureVarnames = other.mFunctionClosureVarnames;

            mClosureBindings = other.mClosureBindings;
            mReturnType = other.mReturnType;
            mArgs = other.mArgs;
            mCompiledSpecializations = other.mCompiledSpecializations;
            mCompiledCodePtr = other.mCompiledCodePtr;

            mHasStarArg = other.mHasStarArg;
            mHasKwarg = other.mHasKwarg;
            mMinPositionalArgs = other.mMinPositionalArgs;
            mMaxPositionalArgs = other.mMaxPositionalArgs;

            for (auto nameAndOther: other.mFunctionGlobalsInCells) {
                mFunctionGlobalsInCells[nameAndOther.first] = incref(nameAndOther.second);
            }

            if (other.mCachedFunctionObj) {
                mCachedFunctionObj = incref(other.mCachedFunctionObj);
            }

            return *this;
        }

    private:
        PyObject* mFunctionCode;

        PyObject* mFunctionGlobals;

        // globals that are stored in cells. This happens when class objects
        // are defined inside of function scopes. We assume that anything in their
        // closure is global (and therefore constant) but it may not be defined yet,
        // so we can't just pull the value out and stick it in the function closure
        // itself. Each value in the map is guaranteed to be a 'cell' object.
        std::map<std::string, PyObject*> mFunctionGlobalsInCells;

        // the order (by name) of the variables in the __closure__ of the original
        // function. This is the order that the python code will expect.
        std::vector<std::string> mFunctionClosureVarnames;

        PyObject* mFunctionDefaults;

        PyObject* mFunctionAnnotations;

        mutable PyObject* mCachedFunctionObj;

        std::map<std::string, ClosureVariableBinding> mClosureBindings;

        Type* mReturnType;

        std::vector<FunctionArg> mArgs;

        // in compiled code, the closure arguments get passed in front of the
        // actual function arguments
        std::vector<CompiledSpecialization> mCompiledSpecializations;

        compiled_code_entrypoint mCompiledCodePtr; //accepts a pointer to packed arguments and another pointer with the return value

        bool mHasStarArg;
        bool mHasKwarg;
        size_t mMinPositionalArgs;
        size_t mMaxPositionalArgs;
    };

    Function(std::string inName,
            const std::vector<Overload>& overloads,
            Type* closureType,
            bool isEntrypoint
            ) :
        Type(catFunction),
        mOverloads(overloads),
        mIsEntrypoint(isEntrypoint),
        mRootName(inName)
    {
        m_is_simple = false;

        mClosureType = closureType;

        _updateAfterForwardTypesChanged();
        endOfConstructorInitialization(); // finish initializing the type object.
    }

    void _updateAfterForwardTypesChanged() {
        m_name = mRootName;

        m_size = mClosureType->bytecount();

        m_is_default_constructible = mClosureType->is_default_constructible();
    }

    static Function* Make(std::string inName, const std::vector<Overload>& overloads, Type* closureType, bool isEntrypoint) {
        static std::mutex guard;

        std::lock_guard<std::mutex> lock(guard);

        typedef std::tuple<const std::string, const std::vector<Overload>, Type*, bool> keytype;

        static std::map<keytype, Function*> *m = new std::map<keytype, Function*>();

        auto it = m->find(keytype(inName, overloads, closureType, isEntrypoint));
        if (it == m->end()) {
            it = m->insert(std::pair<keytype, Function*>(
                keytype(inName, overloads, closureType, isEntrypoint),
                new Function(inName, overloads, closureType, isEntrypoint)
            )).first;
        }

        return it->second;
    }

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
        visitor(mClosureType);
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        for (auto& o: mOverloads) {
            o._visitReferencedTypes(visitor);
        }
        visitor(mClosureType);
    }

    static Function* merge(Function* f1, Function* f2) {
        if (f1->getClosureType()->isTuple() && f2->getClosureType()->isTuple()) {
            std::vector<Type*> types;

            for (auto t: ((Tuple*)f1->getClosureType())->getTypes()) {
                types.push_back(t);
            }

            for (auto t: ((Tuple*)f2->getClosureType())->getTypes()) {
                types.push_back(t);
            }

            std::vector<Overload> overloads(f1->mOverloads);
            for (auto o: f2->mOverloads) {
                overloads.push_back(o.withShiftedFrontClosureBindings(((Tuple*)f1->getClosureType())->getTypes().size()));
            }

            return Function::Make(f1->mRootName, overloads, Tuple::Make(types), f1->isEntrypoint() || f2->isEntrypoint());
        }

        // in theory, we can merge any kinds of closures if we're careful about how to do it,
        // but the main use cases is for merging untyped closures, so that's all that's implemented now
        throw std::runtime_error("Can't merge function types that don't have simple closure types.");
    }

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions) {
        return mClosureType->cmp(left, right, pyComparisonOp, suppressExceptions);
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t wireType) {
        assertWireTypesEqual(wireType, WireType::EMPTY);
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber) {
        buffer.writeEmpty(fieldNumber);
    }

    void repr(instance_ptr self, ReprAccumulator& stream, bool isRepr) {
        stream << "<function " << m_name << ">";
    }

    typed_python_hash_type hash(instance_ptr left) {
        if (mClosureType->bytecount() == 0) {
            return 1;
        }

        return mClosureType->hash(left);
    }

    void constructor(instance_ptr self) {
        if (mClosureType->bytecount() == 0) {
            return;
        }

        mClosureType->constructor(self);
    }

    void destroy(instance_ptr self) {
        if (mClosureType->bytecount() == 0) {
            return;
        }

        mClosureType->destroy(self);
    }

    void copy_constructor(instance_ptr self, instance_ptr other) {
        if (mClosureType->bytecount() == 0) {
            return;
        }

        mClosureType->copy_constructor(self, other);
    }

    void assign(instance_ptr self, instance_ptr other) {
        if (mClosureType->bytecount() == 0) {
            return;
        }

        mClosureType->assign(self, other);
    }

    bool isPODConcrete() {
        return mClosureType->isPOD();
    }

    const std::vector<Overload>& getOverloads() const {
        return mOverloads;
    }

    void addCompiledSpecialization(
                    long whichOverload,
                    compiled_code_entrypoint entrypoint,
                    Type* returnType,
                    const std::vector<Type*>& argTypes
                    ) {
        if (whichOverload < 0 || whichOverload >= mOverloads.size()) {
            throw std::runtime_error("Invalid overload index.");
        }

        mOverloads[whichOverload].addCompiledSpecialization(entrypoint, returnType, argTypes);
    }

    // a test function to force the compiled specialization table to change memory
    // position
    void touchCompiledSpecializations(long whichOverload) {
        if (whichOverload < 0 || whichOverload >= mOverloads.size()) {
            throw std::runtime_error("Invalid overload index.");
        }

        mOverloads[whichOverload].touchCompiledSpecializations();
    }

    bool isEntrypoint() const {
        return mIsEntrypoint;
    }

    Function* withEntrypoint(bool isEntrypoint) {
        return Function::Make(mRootName, mOverloads, mClosureType, isEntrypoint);
    }

    Type* getClosureType() const {
        return mClosureType;
    }

    //returns 'true' if there are no actual values held in the closure.
    bool isEmptyClosure() const {
        return mClosureType->bytecount() == 0;
    }

    static bool reachesTypedClosureType(Type* t) {
        if (t->getTypeCategory() == Type::TypeCategory::catPyCell) {
            return false;
        }

        // we allow untyped closures to contain a TypedCell of NamedTuple of PyCells
        if (t->getTypeCategory() == Type::TypeCategory::catTypedCell) {
            return reachesTypedClosureType(((TypedCellType*)t)->getHeldType());
        }

        if (t->isComposite()) {
            CompositeType* tup = (CompositeType*)t;
            for (auto tupElt: tup->getTypes()) {
                if (reachesTypedClosureType(tupElt)) {
                    return true;
                }
            }
            return false;
        }

        // anything other than a tuple / named tuple of pycells is considered
        // a typed closure
        return true;
    }

    //returns 'false' if any of our closure args are not PyCells.
    bool isFullyUntypedClosure() const {
        if (!mClosureType->isTuple()) {
            return false;
        }

        for (auto subtype: ((Tuple*)mClosureType)->getTypes()) {
            if (!subtype->isNamedTuple()) {
                return false;

                for (auto shouldBeCell: ((NamedTuple*)subtype)->getTypes()) {
                    if (shouldBeCell->getTypeCategory() != Type::TypeCategory::catPyCell) {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    Function* replaceClosure(Type* closureType) {
        return Function::Make(mRootName, mOverloads, closureType, mIsEntrypoint);
    }

    Function* replaceOverloads(const std::vector<Overload>& overloads) {
        return Function::Make(mRootName, overloads, mClosureType, mIsEntrypoint);
    }

    Function* replaceOverloadVariableBindings(long index, const std::map<std::string, ClosureVariableBinding>& bindings) {
        std::vector<Overload> overloads(mOverloads);
        if (index < 0 || index >= mOverloads.size()) {
            throw std::runtime_error("Invalid index to replaceOverloadVariableBindings");
        }

        overloads[index] = overloads[index].withClosureBindings(bindings);

        return Function::Make(mRootName, overloads, mClosureType, mIsEntrypoint);
    }

private:
    std::vector<Overload> mOverloads;

    Type* mClosureType;

    bool mIsEntrypoint;

    std::string mRootName;
};
