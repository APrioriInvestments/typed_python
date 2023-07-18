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
#include "CompiledSpecialization.hpp"
#include "FunctionGlobal.hpp"


class FunctionOverload {
public:
    FunctionOverload(
        PyObject* pyFuncCode,
        PyObject* pyFuncDefaults,
        PyObject* pyFuncAnnotations,
        const std::map<std::string, FunctionGlobal>& inGlobals,
        const std::vector<std::string>& pyFuncClosureVarnames,
        const std::map<std::string, ClosureVariableBinding> closureBindings,
        Type* returnType,
        PyObject* pySignatureFunction,
        const std::vector<FunctionArg>& args,
        Type* methodOf
    ) :
            mFunctionCode(pyFuncCode),
            mFunctionDefaults(pyFuncDefaults),
            mFunctionAnnotations(pyFuncAnnotations),
            mGlobals(inGlobals),
            mFunctionClosureVarnames(pyFuncClosureVarnames),
            mReturnType(returnType),
            mSignatureFunction(pySignatureFunction),
            mArgs(args),
            mHasKwarg(false),
            mHasStarArg(false),
            mMinPositionalArgs(0),
            mMaxPositionalArgs(-1),
            mClosureBindings(closureBindings),
            mCachedFunctionObj(nullptr),
            mMethodOf(methodOf)
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

        increfAllPyObjects();
    }

    FunctionOverload(const FunctionOverload& other) {
        other.increfAllPyObjects();

        mFunctionCode = other.mFunctionCode;
        mGlobals = other.mGlobals;
        mFunctionDefaults = other.mFunctionDefaults;
        mFunctionAnnotations = other.mFunctionAnnotations;
        mSignatureFunction = other.mSignatureFunction;
        mMethodOf = other.mMethodOf;

        mFunctionClosureVarnames = other.mFunctionClosureVarnames;

        mClosureBindings = other.mClosureBindings;
        mReturnType = other.mReturnType;
        mArgs = other.mArgs;
        mCompiledSpecializations = other.mCompiledSpecializations;

        mHasStarArg = other.mHasStarArg;
        mHasKwarg = other.mHasKwarg;
        mMinPositionalArgs = other.mMinPositionalArgs;
        mMaxPositionalArgs = other.mMaxPositionalArgs;

        mCachedFunctionObj = other.mCachedFunctionObj;
    }

    ~FunctionOverload() {
        decrefAllPyObjects();
    }

    FunctionOverload withShiftedFrontClosureBindings(long shiftAmount) const {
        std::map<std::string, ClosureVariableBinding> bindings;
        for (auto nameAndBinding: mClosureBindings) {
            bindings[nameAndBinding.first] = nameAndBinding.second.withShiftedFrontBinding(shiftAmount);
        }

        return FunctionOverload(
            mFunctionCode,
            mFunctionDefaults,
            mFunctionAnnotations,
            mGlobals,
            mFunctionClosureVarnames,
            bindings,
            mReturnType,
            mSignatureFunction,
            mArgs,
            mMethodOf
        );
    }

    FunctionOverload withMethodOf(Type* methodOf) const {
        return FunctionOverload(
            mFunctionCode,
            mFunctionDefaults,
            mFunctionAnnotations,
            mGlobals,
            mFunctionClosureVarnames,
            mClosureBindings,
            mReturnType,
            mSignatureFunction,
            mArgs,
            methodOf
        );
    }

    FunctionOverload withClosureBindings(const std::map<std::string, ClosureVariableBinding> &bindings) const {
        return FunctionOverload(
            mFunctionCode,
            mFunctionDefaults,
            mFunctionAnnotations,
            mGlobals,
            mFunctionClosureVarnames,
            bindings,
            mReturnType,
            mSignatureFunction,
            mArgs,
            mMethodOf
        );
    }

    std::string toString() const {
        std::ostringstream str;

        str << "(";

        if (mMethodOf) {
            str << "method of " << mMethodOf->name() << ", ";
        }

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

    Type* getReturnType() const {
        return mReturnType;
    }

    const std::vector<std::string>& getFunctionClosureVarnames() const {
        return mFunctionClosureVarnames;
    }

    const std::vector<FunctionArg>& getArgs() const {
        return mArgs;
    }

    void finalizeTypeConcrete() {
        // ensure that the Function object we represent is in sync with
        // our actual return and argument types. It would be better to not have
        // any direct references to the python interpreter in this class and
        // rebuild the concrete function object on demand later...
        if (mReturnType) {
            if (mFunctionAnnotations
                && PyDict_Check(mFunctionAnnotations)
            ) {
                PyDict_SetItemString(
                    mFunctionAnnotations,
                    "return",
                    (PyObject*)PyInstance::typeObj(mReturnType)
                );
            }
        }
        for (auto& a: mArgs) {
            Type* typeFilter = a.getTypeFilter();

            if (typeFilter
                && mFunctionAnnotations
                && PyDict_Check(mFunctionAnnotations)
            ) {
                PyDict_SetItemString(
                    mFunctionAnnotations,
                    a.getName().c_str(),
                    (PyObject*)PyInstance::typeObj(a.getTypeFilter())
                );
            }
        }
    }

    void internalizeConstants(
        std::unordered_map<PyObject*, CompilerVisiblePyObj*>& constantMapCache,
        const std::map<Type*, Type*>& groupMap
    ) {
        for (auto& nameAndGlobal: mGlobals) {
            nameAndGlobal.second = nameAndGlobal.second.withConstantsInternalized(
                constantMapCache,
                groupMap
            );
        }
    }

    void updateInternalTypePointers(const std::map<Type*, Type*>& groupMap) {
        if (mReturnType) {
            Type::updateTypeRefFromGroupMap(mReturnType, groupMap);
        }

        for (auto& a: mArgs) {
            a._visitReferencedTypes([&](Type*& typePtr) {
                Type::updateTypeRefFromGroupMap(typePtr, groupMap);
            });
        }

        for (auto& varnameAndBinding: mClosureBindings) {
            varnameAndBinding.second._visitReferencedTypes([&](Type*& typePtr) {
                Type::updateTypeRefFromGroupMap(typePtr, groupMap);
            });
        }

        if (mMethodOf) {
            Type::updateTypeRefFromGroupMap(mMethodOf, groupMap);
        }

        for (auto& nameAndGlobal: mGlobals) {
            nameAndGlobal.second = nameAndGlobal.second.withUpdatedInternalTypePointers(groupMap);
        }
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        // we need to keep mArgs and mReturnType in sync with
        // mAnnotations, so if we change one of our types we
        // need to update the resulting dictionary as well.
        if (mReturnType) {
            visitor(mReturnType);
        }
        for (auto& a: mArgs) {
            a._visitReferencedTypes(visitor);
        }

        for (auto& varnameAndBinding: mClosureBindings) {
            varnameAndBinding.second._visitReferencedTypes(visitor);
        }

        if (mMethodOf) {
            visitor(mMethodOf);
        }

        for (auto& nameAndGlobal: mGlobals) {
            nameAndGlobal.second._visitReferencedTypes(visitor);
        }
    }

    template<class visitor_type>
    void _visitCompilerVisibleInternals(const visitor_type& visitor) {
        visitor.visitTopo(mFunctionCode);

        if (mFunctionAnnotations) {
            visitor.visitHash(ShaHash(2));

            if (PyDict_CheckExact(mFunctionAnnotations)) {
                PyObject *key, *value;
                Py_ssize_t pos = 0;

                while (PyDict_Next(mFunctionAnnotations, &pos, &key, &value)) {
                    visitor.visitTopo(key);
                    visitor.visitTopo(value);
                }
            }

            visitor.visitHash(ShaHash(2));
        }

        if (mSignatureFunction) {
            visitor.visitHash(ShaHash(1));
            visitor.visitTopo(mSignatureFunction);
        } else {
            visitor.visitHash(ShaHash(0));
        }

        if (mFunctionDefaults) {
            visitor.visitHash(ShaHash(2));

            if (PyDict_CheckExact(mFunctionDefaults)) {
                PyObject *key, *value;
                Py_ssize_t pos = 0;

                while (PyDict_Next(mFunctionDefaults, &pos, &key, &value)) {
                    visitor.visitTopo(key);
                    visitor.visitTopo(value);
                }
            }

            visitor.visitHash(ShaHash(2));
        }

        visitor.visitHash(ShaHash(mGlobals.size()));

        for (auto& nameAndGlobal: mGlobals) {
            nameAndGlobal.second._visitCompilerVisibleInternals(visitor);
        }

        if (mReturnType) {
            visitor.visitTopo(mReturnType);
        } else {
            visitor.visitHash(ShaHash());
        }

        if (mMethodOf) {
            visitor.visitTopo(mMethodOf);
        } else {
            visitor.visitHash(ShaHash());
        }

        visitor.visitHash(ShaHash(mClosureBindings.size()));
        for (auto nameAndClosure: mClosureBindings) {
            visitor.visitName(nameAndClosure.first);
            nameAndClosure.second._visitCompilerVisibleInternals(visitor);
        }

        visitor.visitHash(ShaHash(mArgs.size()));

        for (auto a: mArgs) {
            a._visitCompilerVisibleInternals(visitor);
        }

        visitor.visitHash(ShaHash(1));
        visitor.visitHash(ShaHash(mGlobals.size()));

        for (auto& nameAndGlobal: mGlobals) {
            visitor.visitName(nameAndGlobal.first);
            nameAndGlobal.second._visitCompilerVisibleInternals(visitor);
        }

        visitor.visitHash(ShaHash(1));
    }

    template<class visitor_type>
    static void visitCompilerVisibleGlobals(
        const visitor_type& visitor,
        PyCodeObject* code,
        PyObject* globals
    ) {
        std::vector<std::vector<PyObject*> > dotAccesses;

        extractDottedGlobalAccessesFromCode(code, dotAccesses);

        auto visitSequence = [&](const std::vector<PyObject*>& sequence) {
            PyObjectHolder curObj;
            std::string curName;

            for (PyObject* name: sequence) {
                std::string nameStr = PyUnicode_AsUTF8(name);

                if (isSpecialIgnorableName(nameStr)) {
                    break;
                }

                if (!curObj) {
                    curName = nameStr;
                } else {
                    curName = curName + "." + nameStr;
                }

                if (!curObj) {
                    // this is a lookup in the global dict
                    curObj.set(PyDict_GetItem(globals, name));

                    if (!curObj) {
                        // this is an invalid global lookup, which is OK. no need to hash anything.
                        PyErr_Clear();
                        return;
                    }
                } else {
                    // we're looking up an attribute of this object. We only want to look into modules.
                    if (PyModule_CheckExact(curObj) && PyObject_HasAttr(curObj, name)) {
                        PyObjectStealer moduleMember(PyObject_GetAttr(curObj, name));

                        curObj.steal(PyObject_GetAttr(curObj, name));
                        if (!curObj) {
                            // this is an invalid module member lookup. We can just bail.
                            PyErr_Clear();
                            return;
                        }
                    } else {
                        break;
                    }
                }
            }

            // also visit at the end of the sequence
            if (curObj) {
                visitor(curName, (PyObject*)curObj);
            }
        };

        for (auto& sequence: dotAccesses) {
            visitSequence(sequence);
        }
    }

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
    }

    const std::vector<CompiledSpecialization>& getCompiledSpecializations() const {
        return mCompiledSpecializations;
    }

    void addCompiledSpecialization(compiled_code_entrypoint e, Type* returnType, const std::vector<Type*>& argTypes) {
        CompiledSpecialization newSpec = CompiledSpecialization(e,returnType,argTypes);

        for (auto& spec: mCompiledSpecializations) {
            if (spec == newSpec) {
                return;
            }
        }

        mCompiledSpecializations.push_back(newSpec);
    }

    void touchCompiledSpecializations() {
        //force the memory for the compiled specializations to move.
        std::vector<CompiledSpecialization> other = mCompiledSpecializations;
        std::swap(mCompiledSpecializations, other);
    }

    bool operator<(const FunctionOverload& other) const {
        if (mFunctionCode < other.mFunctionCode) { return true; }
        if (mFunctionCode > other.mFunctionCode) { return false; }

        if (mGlobals < other.mGlobals) { return true; }
        if (mGlobals > other.mGlobals) { return false; }

        if (mClosureBindings < other.mClosureBindings) { return true; }
        if (mClosureBindings > other.mClosureBindings) { return false; }

        if (mReturnType < other.mReturnType) { return true; }
        if (mReturnType > other.mReturnType) { return false; }

        if (mArgs < other.mArgs) { return true; }
        if (mArgs > other.mArgs) { return false; }

        if (mMethodOf > other.mMethodOf) { return true; }
        if (mMethodOf < other.mMethodOf) { return false; }

        return false;
    }

    const std::map<std::string, ClosureVariableBinding>& getClosureVariableBindings() const {
        return mClosureBindings;
    }

    PyObject* getFunctionCode() const {
        return mFunctionCode;
    }

    PyObject* getFunctionDefaults() const {
        return mFunctionDefaults;
    }

    PyObject* getFunctionAnnotations() const {
        return mFunctionAnnotations;
    }

    Type* getMethodOf() const {
        return mMethodOf;
    }

    PyObject* getSignatureFunction() const {
        return mSignatureFunction;
    }

    const std::map<std::string, FunctionGlobal>& getGlobals() const {
        return mGlobals;
    }

    std::map<std::string, FunctionGlobal>& getGlobals() {
        return mGlobals;
    }

    /* walk over the opcodes in 'code' and extract all cases where we're accessing globals by name.

    In cases where we write something like 'x.y.z' the compiler shouldn't have a reference to 'x',
    just to whatever 'x.y.z' refers to.

    This transformation just figures out what the dotting sequences are.
    */
    static void extractDottedGlobalAccessesFromCode(PyCodeObject* code, std::vector<std::vector<PyObject*> >& outSequences) {
        uint8_t* bytes;
        Py_ssize_t bytecount;

        static PyObject* moduleHashName = PyUnicode_FromString("__module_hash__");
        outSequences.push_back(std::vector<PyObject*>({moduleHashName}));

        PyBytes_AsStringAndSize(((PyCodeObject*)code)->co_code, (char**)&bytes, &bytecount);

        long opcodeCount = bytecount / 2;

        // opcodes are encoded in the low byte
        auto opcodeFor = [&](int i) { return bytes[i * 2]; };

        // opcode targets are encoded in the high byte
        auto opcodeTargetFor = [&](int i) { return bytes[i * 2 + 1]; };

        const uint8_t LOAD_ATTR = 106;
        const uint8_t LOAD_GLOBAL = 116;
        const uint8_t DELETE_GLOBAL = 98;
        const uint8_t STORE_GLOBAL = 97;
        const uint8_t LOAD_METHOD = 160;


        std::vector<PyObject*> curDotSequence;
        for (long ix = 0; ix < opcodeCount; ix++) {
            // if we're loading an attr on an existing sequence, just make it bigger
            if ((opcodeFor(ix) == LOAD_ATTR || opcodeFor(ix) == LOAD_METHOD) && curDotSequence.size()) {
                curDotSequence.push_back(PyTuple_GetItem(code->co_names, opcodeTargetFor(ix)));
            } else if (curDotSequence.size()) {
                // any other operation should flush the buffer
                outSequences.push_back(curDotSequence);
                curDotSequence.clear();
            }

            // if we're loading a global, we start a new sequence
            if (opcodeFor(ix) == LOAD_GLOBAL) {
                curDotSequence.push_back(PyTuple_GetItem(code->co_names, opcodeTargetFor(ix)));
            } else if (
                opcodeFor(ix) == STORE_GLOBAL
                || opcodeFor(ix) == DELETE_GLOBAL
            ) {
                outSequences.push_back({PyTuple_GetItem(code->co_names, opcodeTargetFor(ix))});
            }
        }

        // flush the buffer if we have something
        if (curDotSequence.size()) {
            outSequences.push_back(curDotSequence);
        }

        // recurse into sub code objects
        iterate(code->co_consts, [&](PyObject* o) {
            if (PyCode_Check(o)) {
                extractDottedGlobalAccessesFromCode((PyCodeObject*)o, outSequences);
            }
        });
    }

    // get a list of all "global dot accesses" contained in 'code'.  This will pull out every
    // case where we have a sequence of opcodes that access a global variable by name and then
    // sequentially access members. So if you write 'x.y.z' and 'x' is a reference that will
    // be looked up using a 'LOAD_GLOBAL' then we will include 'x.y.z' in outAccesses.
    static void extractGlobalAccessesFromCode(PyCodeObject* code, std::set<std::string>& outAccesses) {
        uint8_t* bytes;
        Py_ssize_t bytecount;

        PyBytes_AsStringAndSize(((PyCodeObject*)code)->co_code, (char**)&bytes, &bytecount);

        long opcodeCount = bytecount / 2;

        // opcodes are encoded in the low byte
        auto opcodeFor = [&](int i) { return bytes[i * 2]; };

        // opcode targets are encoded in the high byte
        auto opcodeTargetFor = [&](int i) { return bytes[i * 2 + 1]; };

        const uint8_t LOAD_GLOBAL = 116;
        const uint8_t DELETE_GLOBAL = 98;
        const uint8_t STORE_GLOBAL = 97;

        for (long ix = 0; ix < opcodeCount; ix++) {
            // if we're loading a global, we start a new sequence
            if (opcodeFor(ix) == LOAD_GLOBAL) {
                PyObject* name = PyTuple_GetItem(code->co_names, opcodeTargetFor(ix));
                if (!PyUnicode_Check(name)) {
                    throw std::runtime_error("Function had a non-string object in co_names");
                }
                outAccesses.insert(PyUnicode_AsUTF8(name));
            } else if (
                opcodeFor(ix) == STORE_GLOBAL
                || opcodeFor(ix) == DELETE_GLOBAL
            ) {
                PyObject* name = PyTuple_GetItem(code->co_names, opcodeTargetFor(ix));
                if (!PyUnicode_Check(name)) {
                    throw std::runtime_error("Function had a non-string object in co_names");
                }
                outAccesses.insert(PyUnicode_AsUTF8(name));
            }
        }

        // recurse into sub code objects
        iterate(code->co_consts, [&](PyObject* o) {
            if (PyCode_Check(o)) {
                extractGlobalAccessesFromCode((PyCodeObject*)o, outAccesses);
            }
        });

        outAccesses.insert("__module_hash__");
    }

    void extractGlobalAccessesFromCodeIncludingCells(
        std::set<std::string>& outNames
    ) const {
        for (auto nameAndGlobal: mGlobals) {
            outNames.insert(nameAndGlobal.first);
        }

        outNames.insert("__module_hash__");

        extractGlobalAccessesFromCode((PyCodeObject*)mFunctionCode, outNames);
    }

    bool symbolIsUnresolved(std::string name, bool insistForwardsResolved) {
        if (mClosureBindings.find(name) != mClosureBindings.end()) {
            return false;
        }

        auto it = mGlobals.find(name);

        if (it == mGlobals.end()) {
            return true;
        }

        return it->second.isUnresolved(insistForwardsResolved);
    }

    void autoresolveGlobal(
        std::string name,
        const std::set<Type*>& resolvedForwards
    ) {
        auto it = mGlobals.find(name);

        if (it != mGlobals.end()) {
            it->second.autoresolveGlobal(resolvedForwards);
        }
    }

    void autoresolveGlobals(const std::set<Type*>& resolvedForwards) {
        std::set<std::string> allNames;
        extractGlobalAccessesFromCodeIncludingCells(allNames);

        for (auto nameStr: allNames) {
            if (nameStr != "__module_hash__") {
                autoresolveGlobal(nameStr, resolvedForwards);
            }
        }
    }

    bool hasUnresolvedSymbols(bool insistForwardsResolved) {
        std::set<std::string> allNames;
        extractGlobalAccessesFromCodeIncludingCells(allNames);

        for (auto nameStr: allNames) {
            if (nameStr != "__module_hash__" && symbolIsUnresolved(nameStr, insistForwardsResolved)) {
                return true;
            }
        }

        return false;
    }

    std::string firstUnresolvedSymbol(bool insistForwardsResolved) {
        std::set<std::string> allNames;
        extractGlobalAccessesFromCodeIncludingCells(allNames);

        for (auto nameStr: allNames) {
            if (nameStr != "__module_hash__" && symbolIsUnresolved(nameStr, insistForwardsResolved)) {
                return nameStr;
            }
        }

        return "";
    }

    static void buildInitialGlobalsDict(
        std::map<std::string, FunctionGlobal>& outGlobals,
        PyObject* inFuncGlobals,
        PyCodeObject* inFuncCode
    ) {
        std::set<std::string> allNamesString;
        extractGlobalAccessesFromCode(inFuncCode, allNamesString);

        for (auto nameStr: allNamesString) {
            if (nameStr != "__module_hash__" && outGlobals.find(nameStr) != outGlobals.end()) {
                throw std::runtime_error(
                    "Somehow we already a closure binding for " + nameStr
                    + " and somehow we want to register a global binding?"
                );
            }

            std::pair<std::string, FunctionGlobal> ref = FunctionGlobal::DottedGlobalsLookup(
                inFuncGlobals,
                nameStr
            );

            outGlobals[ref.first] = ref.second;

            std::cout << "have " << ref.first << " = " << ref.second.toString() << "\n";
        }
    }

    // create a new function object for this closure (or cache it
    // if we have no closure)
    PyObject* buildFunctionObj(Type* closureType, instance_ptr closureData);

    FunctionOverload& operator=(const FunctionOverload& other) {
        other.increfAllPyObjects();
        decrefAllPyObjects();

        mFunctionCode = other.mFunctionCode;
        mGlobals = other.mGlobals;
        mFunctionDefaults = other.mFunctionDefaults;
        mFunctionAnnotations = other.mFunctionAnnotations;
        mSignatureFunction = other.mSignatureFunction;

        mMethodOf = other.mMethodOf;

        mFunctionClosureVarnames = other.mFunctionClosureVarnames;

        mClosureBindings = other.mClosureBindings;
        mReturnType = other.mReturnType;
        mArgs = other.mArgs;
        mCompiledSpecializations = other.mCompiledSpecializations;

        mHasStarArg = other.mHasStarArg;
        mHasKwarg = other.mHasKwarg;
        mMinPositionalArgs = other.mMinPositionalArgs;
        mMaxPositionalArgs = other.mMaxPositionalArgs;

        mCachedFunctionObj = other.mCachedFunctionObj;

        return *this;
    }

    template<class serialization_context_t, class buf_t>
    void serialize(serialization_context_t& context, buf_t& buffer, int fieldNumber) const {
        buffer.writeBeginCompound(fieldNumber);

        context.serializePythonObject(mFunctionCode, buffer, 0);

        if (mFunctionDefaults) {
            context.serializePythonObject(mFunctionDefaults, buffer, 2);
        }

        if (mFunctionAnnotations) {
            context.serializePythonObject(mFunctionAnnotations, buffer, 3);
        }

        buffer.writeBeginCompound(4);
            int stringIx = 0;
            for (auto varname: mFunctionClosureVarnames) {
                buffer.writeStringObject(stringIx++, varname);
            }
        buffer.writeEndCompound();

        buffer.writeBeginCompound(5);
            int varIx = 0;
            for (auto& nameAndGlobal: mGlobals) {
                buffer.writeStringObject(varIx++, nameAndGlobal.first);
                nameAndGlobal.second.serialize(context, buffer, varIx++);
            }
        buffer.writeEndCompound();

        buffer.writeBeginCompound(6);
            int closureBindingIx = 0;
            for (auto nameAndBinding: mClosureBindings) {
                buffer.writeStringObject(closureBindingIx++, nameAndBinding.first);
                nameAndBinding.second.serialize(context, buffer, closureBindingIx++);
            }
        buffer.writeEndCompound();

        if (mReturnType) {
            context.serializeNativeType(mReturnType, buffer, 7);
        }

        buffer.writeBeginCompound(8);

            int argIx = 0;
            for (auto& arg: mArgs) {
                arg.serialize(context, buffer, argIx++);
            }

        buffer.writeEndCompound();

        if (mSignatureFunction) {
            context.serializePythonObject(mSignatureFunction, buffer, 9);
        }

        if (mMethodOf) {
            context.serializeNativeType(mMethodOf, buffer, 10);
        }

        buffer.writeEndCompound();
    }

    template<class serialization_context_t, class buf_t>
    static FunctionOverload deserialize(serialization_context_t& context, buf_t& buffer, int wireType) {
        PyObjectHolder functionCode;
        PyObjectHolder functionAnnotations;
        PyObjectHolder functionDefaults;
        PyObjectHolder functionSignature;
        std::vector<std::string> closureVarnames;
        std::map<std::string, FunctionGlobal> functionGlobals;
        std::map<std::string, ClosureVariableBinding> closureBindings;
        Type* returnType = nullptr;
        Type* methodOf = nullptr;
        std::vector<FunctionArg> args;

        buffer.consumeCompoundMessage(wireType, [&](size_t fieldNumber, size_t wireType) {
            if (fieldNumber == 0) {
                functionCode.steal(context.deserializePythonObject(buffer, wireType));
            }
            else if (fieldNumber == 2) {
                functionDefaults.steal(context.deserializePythonObject(buffer, wireType));
            }
            else if (fieldNumber == 3) {
                functionAnnotations.steal(context.deserializePythonObject(buffer, wireType));
            }
            else if (fieldNumber == 4) {
                buffer.consumeCompoundMessage(wireType, [&](size_t fieldNumber, size_t wireType) {
                    assertWireTypesEqual(wireType, WireType::BYTES);
                    closureVarnames.push_back(buffer.readStringObject());
                });
            }
            else if (fieldNumber == 5) {
                std::string last;
                buffer.consumeCompoundMessage(wireType, [&](size_t fieldNumber, size_t wireType) {
                    if (fieldNumber % 2 == 0) {
                        assertWireTypesEqual(wireType, WireType::BYTES);
                        last = buffer.readStringObject();
                    } else {
                        if (last == "") {
                            throw std::runtime_error("Corrupt Function closure encountered");
                        }
                        functionGlobals[last] = FunctionGlobal::deserialize(context, buffer, wireType);
                        last = "";
                    }
                });
            }
            else if (fieldNumber == 6) {
                std::string last;
                buffer.consumeCompoundMessage(wireType, [&](size_t fieldNumber, size_t wireType) {
                    if (fieldNumber % 2 == 0) {
                        assertWireTypesEqual(wireType, WireType::BYTES);
                        last = buffer.readStringObject();
                    } else {
                        closureBindings[last] = ClosureVariableBinding::deserialize(context, buffer, wireType);
                    }
                });
            }
            else if (fieldNumber == 7) {
                returnType = context.deserializeNativeType(buffer, wireType);
            }
            else if (fieldNumber == 8) {
                buffer.consumeCompoundMessage(wireType, [&](size_t fieldNumber, size_t wireType) {
                    args.push_back(FunctionArg::deserialize(context, buffer, wireType));
                });
            }
            else if (fieldNumber == 9) {
                functionSignature.steal(context.deserializePythonObject(buffer, wireType));
            }
            else if (fieldNumber == 10) {
                methodOf = context.deserializeNativeType(buffer, wireType);
            }
        });

        return FunctionOverload(
            functionCode,
            functionDefaults,
            functionAnnotations,
            functionGlobals,
            closureVarnames,
            closureBindings,
            returnType,
            functionSignature,
            args,
            methodOf
        );
    }

    void increfAllPyObjects() const {
        incref(mFunctionCode);
        incref(mFunctionDefaults);
        incref(mFunctionAnnotations);
        incref(mSignatureFunction);
    }

    void decrefAllPyObjects() {
        decref(mFunctionCode);
        decref(mFunctionDefaults);
        decref(mFunctionAnnotations);
        decref(mSignatureFunction);
    }

private:
    // every global we access not through our closure
    std::map<std::string, FunctionGlobal> mGlobals;

    PyObject* mFunctionCode;

    // the order (by name) of the variables in the __closure__ of the original
    // function. This is the order that the python code will expect.
    std::vector<std::string> mFunctionClosureVarnames;

    PyObject* mFunctionDefaults;

    PyObject* mFunctionAnnotations;

    PyObject* mSignatureFunction;

    // note that we are deliberately leaking this value because Overloads get
    // stashed in static std::map memos anyways.
    mutable PyObject* mCachedFunctionObj;

    std::map<std::string, ClosureVariableBinding> mClosureBindings;

    Type* mReturnType;

    // if we are a method of a class, what class?
    Type* mMethodOf;

    std::vector<FunctionArg> mArgs;

    // in compiled code, the closure arguments get passed in front of the
    // actual function arguments
    std::vector<CompiledSpecialization> mCompiledSpecializations;

    bool mHasStarArg;
    bool mHasKwarg;
    size_t mMinPositionalArgs;
    size_t mMaxPositionalArgs;
};
