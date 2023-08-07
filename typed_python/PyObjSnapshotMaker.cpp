#include "PyObjSnapshot.hpp"
#include "PyObjSnapshotMaker.hpp"


PyObjSnapshot* PyObjSnapshotMaker::internalize(const std::string& def) {
    PyObjSnapshot* res = new PyObjSnapshot(mGraph);

    res->becomeInternalizedOf(def, *this);

    return res;
}


PyObjSnapshot* PyObjSnapshotMaker::internalize(const FunctionArg& def) {
    PyObjSnapshot* res = new PyObjSnapshot(mGraph);

    res->becomeInternalizedOf(def, *this);

    return res;
}


PyObjSnapshot* PyObjSnapshotMaker::internalize(const ClosureVariableBinding& def) {
    PyObjSnapshot* res = new PyObjSnapshot(mGraph);

    res->becomeInternalizedOf(def, *this);

    return res;
}


PyObjSnapshot* PyObjSnapshotMaker::internalize(const ClosureVariableBindingStep& def) {
    PyObjSnapshot* res = new PyObjSnapshot(mGraph);

    res->becomeInternalizedOf(def, *this);

    return res;
}


PyObjSnapshot* PyObjSnapshotMaker::internalize(const MemberDefinition& def) {
    PyObjSnapshot* res = new PyObjSnapshot(mGraph);

    res->becomeInternalizedOf(def, *this);

    return res;
}

PyObjSnapshot* PyObjSnapshotMaker::internalize(const FunctionOverload& def) {
    PyObjSnapshot* res = new PyObjSnapshot(mGraph);

    res->becomeInternalizedOf(def, *this);

    return res;
}

PyObjSnapshot* PyObjSnapshotMaker::internalize(const std::vector<Type*>& def) {
    PyObjSnapshot* res = new PyObjSnapshot(mGraph);

    res->becomeBundleOf(def, *this);

    return res;
}

PyObjSnapshot* PyObjSnapshotMaker::internalize(const std::vector<HeldClass*>& def) {
    PyObjSnapshot* res = new PyObjSnapshot(mGraph);

    res->becomeBundleOf(def, *this);

    return res;
}

PyObjSnapshot* PyObjSnapshotMaker::internalize(const std::vector<FunctionArg>& def) {
    PyObjSnapshot* res = new PyObjSnapshot(mGraph);

    res->becomeBundleOf(def, *this);

    return res;
}

PyObjSnapshot* PyObjSnapshotMaker::internalize(const std::vector<std::string>& def) {
    PyObjSnapshot* res = new PyObjSnapshot(mGraph);

    res->becomeBundleOf(def, *this);

    return res;
}

PyObjSnapshot* PyObjSnapshotMaker::internalize(const std::vector<FunctionOverload>& def) {
    PyObjSnapshot* res = new PyObjSnapshot(mGraph);

    res->becomeBundleOf(def, *this);

    return res;
}

PyObjSnapshot* PyObjSnapshotMaker::internalize(const FunctionGlobal& def) {
    PyObjSnapshot* res = new PyObjSnapshot(mGraph);

    res->becomeInternalizedOf(def, *this);

    return res;
}

PyObjSnapshot* PyObjSnapshotMaker::internalize(const std::map<std::string, Function*>& inMethods) {
    PyObjSnapshot* res = new PyObjSnapshot(mGraph);

    res->becomeBundleOf(inMethods, *this);

    return res;
}

PyObjSnapshot* PyObjSnapshotMaker::internalize(const std::map<std::string, ClosureVariableBinding>& inMethods) {
    PyObjSnapshot* res = new PyObjSnapshot(mGraph);

    res->becomeBundleOf(inMethods, *this);

    return res;
}

PyObjSnapshot* PyObjSnapshotMaker::internalize(const std::map<std::string, FunctionGlobal>& inMethods) {
    PyObjSnapshot* res = new PyObjSnapshot(mGraph);

    res->becomeBundleOf(inMethods, *this);

    return res;
}

PyObjSnapshot* PyObjSnapshotMaker::internalize(const std::map<std::string, PyObject*>& inMethods) {
    PyObjSnapshot* res = new PyObjSnapshot(mGraph);

    res->becomeBundleOf(inMethods, *this);

    return res;
}

PyObjSnapshot* PyObjSnapshotMaker::internalize(const std::vector<MemberDefinition>& inMethods) {
    PyObjSnapshot* res = new PyObjSnapshot(mGraph);

    res->becomeBundleOf(inMethods, *this);

    return res;
}

/* static */
PyObjSnapshot* PyObjSnapshotMaker::internalize(PyObject* val)
{
    Type* t = PyInstance::extractTypeFrom(val, true);
    if (t) {
        return internalize(t);
    }

    ::Type* instanceType = PyInstance::extractTypeFrom(val->ob_type);
    if (instanceType) {
        return internalize(
            InstanceRef(
                ((PyInstance*)val)->dataPtr(),
                instanceType
            )
        );
    }

    auto it = mObjMapCache.find(val);

    if (it != mObjMapCache.end()) {
        return it->second;
    }

    mObjMapCache[val] = new PyObjSnapshot(mGraph);

    mObjMapCache[val]->becomeInternalizedOf(val, *this);

    return mObjMapCache[val];
}

/* static */
PyObjSnapshot* PyObjSnapshotMaker::internalize(Type* val)
{
    if (mLinkToInternal && val->getSnapshot()) {
        return val->getSnapshot();
    }

    auto it = mTypeMapCache.find(val);

    if (it != mTypeMapCache.end()) {
        return it->second;
    }

    mTypeMapCache[val] = new PyObjSnapshot(mGraph);

    mTypeMapCache[val]->becomeInternalizedOf(val, *this);

    return mTypeMapCache[val];
}

/* static */
PyObjSnapshot* PyObjSnapshotMaker::internalize(InstanceRef val)
{
    Type* t = val.extractType(true);
    if (t) {
        return internalize(t);
    }

    PyObject* o = val.extractPyobj();
    if (o) {
        return internalize(o);
    }

    auto it = mInstanceCache.find(val);

    if (it != mInstanceCache.end()) {
        return it->second;
    }

    mInstanceCache[val] = new PyObjSnapshot(mGraph);

    mInstanceCache[val]->becomeInternalizedOf(val, *this);

    return mInstanceCache[val];
}
