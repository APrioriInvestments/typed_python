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

#pragma once

#include "Type.hpp"
#include "TypedCellType.hpp"
#include "ReprAccumulator.hpp"
#include "Format.hpp"
#include "SpecialModuleNames.hpp"
#include "PyInstance.hpp"
#include "ClosureVariableBinding.hpp"
#include "FunctionArg.hpp"
#include "FunctionOverload.hpp"

class Function;

PyDoc_STRVAR(Function_doc,
    "Function(f) -> typed function\n"
    "\n"
    "Converts function f to a typed function.\n"
    );

class Function : public Type {
public:
    Function() : Type(catFunction)
    {
    }

    const char* docConcrete() {
        return Function_doc;
    }

    Function(std::string inName,
            std::string qualname,
            std::string moduleName,
            const std::vector<FunctionOverload>& overloads,
            Type* closureType,
            bool isEntrypoint,
            bool isNocompile
            ) :
        Type(catFunction),
        mOverloads(overloads),
        mIsEntrypoint(isEntrypoint),
        mIsNocompile(isNocompile),
        mRootName(inName),
        mQualname(qualname),
        mModulename(moduleName),
        mClosureType(closureType)
    {
        m_is_forward_defined = true;
        m_name = mRootName;
    }

    void initializeDuringDeserialization(
        std::string inName,
        std::string qualname,
        std::string moduleName,
        std::vector<FunctionOverload> overloads,
        Type* closureType,
        bool isEntrypoint,
        bool isNocompile
    ) {
        mRootName = inName;
        mQualname = qualname;
        mModulename = moduleName;
        mOverloads = overloads;
        mClosureType = closureType;
        mIsEntrypoint = isEntrypoint;
        mIsNocompile = isNocompile;
    }

    std::string moduleNameConcrete() {
        if (mModulename.size() == 0) {
            return "builtins";
        }

        return mModulename;
    }

    std::string nameWithModuleConcrete() {
        return moduleNameConcrete() + "." + (mQualname.size() ? mQualname : mRootName);
    }

    // does this function have any of its globals that are not
    // resolved to actual values. if 'insistForwardsResolved' then we
    // return 'true' if the symbol resolves to a forward defined
    // type - the symbol is only considered resolved when there are no
    // visible forwards
    std::string firstUnresolvedSymbol(bool insistResolved) {
        for (auto& o: mOverloads) {
            std::string res = o.firstUnresolvedSymbol(insistResolved);
            if (res.size()) {
                return res;
            }
        }

        return "";
    }

    bool hasUnresolvedSymbols(bool insistResolved) {
        for (auto& o: mOverloads) {
            if (o.hasUnresolvedSymbols(insistResolved)) {
                return true;
            }
        }

        return false;
    }

    void autoresolveGlobals(const std::set<Type*>& resolvedForwards) {
        for (auto& o: mOverloads) {
            o.autoresolveGlobals(resolvedForwards);
        }
    }

    void finalizeTypeConcrete() {
        for (auto& o: mOverloads) {
            o.finalizeTypeConcrete();
        }
    }

    std::string computeRecursiveNameConcrete(TypeStack& typeStack) {
        return mRootName;
    }

    void postInitializeConcrete() {
        m_size = mClosureType->bytecount();
        m_is_default_constructible = mClosureType->is_default_constructible();
    }

    void initializeFromConcrete(Type* forwardDef) {
        Function* fwdFunc = (Function*)forwardDef;

        mClosureType = fwdFunc->mClosureType;
        mOverloads = fwdFunc->mOverloads;
        mIsEntrypoint = fwdFunc->mIsEntrypoint;
        mIsNocompile = fwdFunc->mIsNocompile;
        mRootName = fwdFunc->mRootName;
        mQualname = fwdFunc->mQualname;
        mModulename = fwdFunc->mModulename;
    }

    Type* cloneForForwardResolutionConcrete() {
        return new Function();
    }

    // replace any direct references to PyObject we're holding internally
    // with PyObjSnapshot references instead
    void internalizeConstants(
        std::unordered_map<PyObject*, PyObjSnapshot*>& constantMapCache,
        const std::map<Type*, Type*>& groupMap
    ) {
        for (auto& o: mOverloads) {
            o.internalizeConstants(constantMapCache, groupMap);
        }
    }

    void updateInternalTypePointersConcrete(const std::map<Type*, Type*>& groupMap) {
        updateTypeRefFromGroupMap(mClosureType, groupMap);

        for (auto& o: mOverloads) {
            o.updateInternalTypePointers(groupMap);
        }
    }

    static Function* Make(
        std::string inName,
        std::string qualname,
        std::string moduleName,
        std::vector<FunctionOverload> overloads,
        Type* closureType,
        bool isEntrypoint,
        bool isNocompile
    ) {
        bool anyFwd = false;

        if (closureType->isForwardDefined()) {
            anyFwd = true;
        }

        for (auto& o: overloads) {
            o._visitReferencedTypes([&](Type* t) {
                if (t->isForwardDefined()) {
                    anyFwd = true;
                }
            });

            if (o.hasUnresolvedSymbols(true)) {
                anyFwd = true;
            }
        }

        if (anyFwd) {
            return new Function(
                inName, qualname, moduleName, overloads, closureType, isEntrypoint, isNocompile
            );
        }

        PyEnsureGilAcquired getTheGil;

        typedef std::tuple<
            const std::string,
            const std::string,
            const std::string,
            const std::vector<FunctionOverload>,
            Type*,
            bool,
            bool
        > keytype;

        // allocate and leak the memo or else we will crash at process unload time
        static std::map<keytype, Function*> *memo = new std::map<keytype, Function*>();

        keytype key(
            inName, qualname, moduleName, overloads, closureType, isEntrypoint, isNocompile
        );

        auto it = memo->find(key);

        if (it != memo->end()) {
            return it->second;
        }

        Function* res = new Function(
            inName, qualname, moduleName, overloads, closureType, isEntrypoint, isNocompile
        );

        Function* concrete = (Function*)res->forwardResolvesTo();

        (*memo)[key] = concrete;
        return concrete;
    }

    template<class visitor_type>
    void _visitCompilerVisibleInternals(const visitor_type& v) {
        v.visitHash(
            ShaHash(1, m_typeCategory)
            + ShaHash(mIsNocompile ? 2 : 1)
            + ShaHash(mIsEntrypoint ? 2 : 1)
        );

        v.visitName(m_name);
        v.visitName(mRootName);
        v.visitName(mQualname);
        v.visitName(mModulename);

        v.visitTopo(mClosureType);

        v.visitHash(ShaHash(mOverloads.size()));

        for (auto o: mOverloads) {
            o._visitCompilerVisibleInternals(v);
        }
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

            std::vector<FunctionOverload> overloads(f1->mOverloads);
            for (auto o: f2->mOverloads) {
                overloads.push_back(o.withShiftedFrontClosureBindings(((Tuple*)f1->getClosureType())->getTypes().size()));
            }

            return Function::Make(
                f1->mRootName,
                f1->mQualname,
                f1->mModulename,
                overloads,
                Tuple::Make(types),
                f1->isEntrypoint() || f2->isEntrypoint(),
                f1->isNocompile() || f2->isNocompile()
            );
        }

        // in theory, we can merge any kinds of closures if we're careful about how to do it,
        // but the main use cases is for merging untyped closures, so that's all that's implemented now
        throw std::runtime_error("Can't merge function types that don't have simple closure types.");
    }

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions) {
        return mClosureType->cmp(left, right, pyComparisonOp, suppressExceptions);
    }

    void deepcopyConcrete(
        instance_ptr dest,
        instance_ptr src,
        DeepcopyContext& context
    ) {
        // we don't deepcopy into functions
        copy_constructor(dest, src);
    }

    size_t deepBytecountConcrete(instance_ptr instance, std::unordered_set<void*>& alreadyVisited, std::set<Slab*>* outSlabs) {
        // we explicitly don't count functions
        return 0;
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t wireType) {
        mClosureType->deserialize(self, buffer, wireType);
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber) {
        mClosureType->serialize(self, buffer, fieldNumber);
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

    const std::vector<FunctionOverload>& getOverloads() const {
        return mOverloads;
    }

    std::vector<FunctionOverload>& getOverloads() {
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

    bool isNocompile() const {
        return mIsNocompile;
    }

    Function* withMethodOf(Type* methodOf) {
        bool anyDifferent = false;
        for (auto& o: mOverloads) {
            if (o.getMethodOf() != methodOf) {
                anyDifferent = true;
                break;
            }
        }

        if (!anyDifferent) {
            return this;
        }

        std::vector<FunctionOverload> overloads;
        for (auto& o: mOverloads) {
            overloads.push_back(o.withMethodOf(methodOf));
        }

        return replaceOverloads(overloads);
    }

    Function* withEntrypoint(bool isEntrypoint) const {
        Function* f = Function::Make(
            mRootName, mQualname, mModulename, mOverloads, mClosureType, isEntrypoint, mIsNocompile
        );

        if (f->isForwardDefined() && !isForwardDefined()) {
            return (Function*)f->forwardResolvesTo();
        }
        return f;
    }

    Function* withNocompile(bool isNocompile) const {
        Function* f = Function::Make(
            mRootName, mQualname, mModulename, mOverloads, mClosureType, mIsEntrypoint, isNocompile
        );
        if (f->isForwardDefined() && !isForwardDefined()) {
            return (Function*)f->forwardResolvesTo();
        }
        return f;
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

    // //Function types can be instantiated even if forwards are not resolved
    // //in their annotations.
    // void assertForwardsResolvedSufficientlyToInstantiateConcrete() {
    //     mClosureType->assertForwardsResolvedSufficientlyToInstantiate();
    // }

    Function* replaceClosure(Type* closureType) const {
        return Function::Make(mRootName, mQualname, mModulename, mOverloads, closureType, mIsEntrypoint, mIsNocompile);
    }

    Function* replaceOverloads(const std::vector<FunctionOverload>& overloads) const {
        return Function::Make(mRootName, mQualname, mModulename, overloads, mClosureType, mIsEntrypoint, mIsNocompile);
    }

    Function* replaceOverloadVariableBindings(long index, const std::map<std::string, ClosureVariableBinding>& bindings) {
        std::vector<FunctionOverload> overloads(mOverloads);
        if (index < 0 || index >= mOverloads.size()) {
            throw std::runtime_error("Invalid index to replaceOverloadVariableBindings");
        }

        overloads[index] = overloads[index].withClosureBindings(bindings);

        return Function::Make(mRootName, mQualname, mModulename, overloads, mClosureType, mIsEntrypoint, mIsNocompile);
    }

    std::string qualname() const {
        if (mQualname.size()) {
            return mQualname;
        }

        return m_name;
    }

    std::string moduleName() const {
        return mModulename;
    }

private:
    std::vector<FunctionOverload> mOverloads;

    Type* mClosureType;

    bool mIsEntrypoint;

    bool mIsNocompile;

    std::string mRootName, mQualname, mModulename;
};
