#include "PyObjSnapshot.hpp"
#include "PyObjSnapshotMaker.hpp"
#include "PyObjGraphSnapshot.hpp"
#include "PyObjRehydrator.hpp"



PyObjSnapshot::PyObjSnapshot(PyObjGraphSnapshot* inGraph) :
    mKind(Kind::Uninitialized),
    mType(nullptr),
    mPyObject(nullptr),
    mGraph(inGraph),
    mIndexInGraph(0)
{
    if (mGraph) {
        mIndexInGraph = mGraph->registerSnapshot(this);
    }
}

void PyObjSnapshot::rehydrate() {
    if (!needsHydration()) {
        return;
    }

    PyObjRehydrator rehydrator;

    rehydrator.start(this);
    rehydrator.finalize();
}

std::string PyObjSnapshot::kindAsString() const {
    if (mKind == Kind::Uninitialized) {
        return "Uninitialized";
    }
    if (mKind == Kind::String) {
        return "String";
    }
    if (mKind == Kind::NamedPyObject) {
        return "NamedPyObject";
    }
    if (mKind == Kind::Instance) {
        return "Instance";
    }
    if (mKind == Kind::PrimitiveType) {
        return "PrimitiveType";
    }
    if (mKind == Kind::ListOfType) {
        return "ListOfType";
    }
    if (mKind == Kind::TupleOfType) {
        return "TupleOfType";
    }
    if (mKind == Kind::TupleType) {
        return "TupleType";
    }
    if (mKind == Kind::NamedTupleType) {
        return "NamedTupleType";
    }
    if (mKind == Kind::OneOfType) {
        return "OneOfType";
    }
    if (mKind == Kind::ValueType) {
        return "ValueType";
    }
    if (mKind == Kind::DictType) {
        return "DictType";
    }
    if (mKind == Kind::ConstDictType) {
        return "ConstDictType";
    }
    if (mKind == Kind::SetType) {
        return "SetType";
    }
    if (mKind == Kind::PointerToType) {
        return "PointerToType";
    }
    if (mKind == Kind::RefToType) {
        return "RefToType";
    }
    if (mKind == Kind::AlternativeType) {
        return "AlternativeType";
    }
    if (mKind == Kind::ConcreteAlternativeType) {
        return "ConcreteAlternativeType";
    }
    if (mKind == Kind::AlternativeMatcherType) {
        return "AlternativeMatcherType";
    }
    if (mKind == Kind::PythonObjectOfTypeType) {
        return "PythonObjectOfTypeType";
    }
    if (mKind == Kind::SubclassOfTypeType) {
        return "SubclassOfTypeType";
    }
    if (mKind == Kind::ClassType) {
        return "ClassType";
    }
    if (mKind == Kind::HeldClassType) {
        return "HeldClassType";
    }
    if (mKind == Kind::FunctionType) {
        return "FunctionType";
    }
    if (mKind == Kind::FunctionOverload) {
        return "FunctionOverload";
    }
    if (mKind == Kind::FunctionGlobal) {
        return "FunctionGlobal";
    }
    if (mKind == Kind::FunctionArg) {
        return "FunctionArg";
    }
    if (mKind == Kind::FunctionClosureVariableBinding) {
        return "FunctionClosureVariableBinding";
    }
    if (mKind == Kind::FunctionClosureVariableBindingStep) {
        return "FunctionClosureVariableBindingStep";
    }
    if (mKind == Kind::ClassMemberDefinition) {
        return "ClassMemberDefinition";
    }
    if (mKind == Kind::BoundMethodType) {
        return "BoundMethodType";
    }
    if (mKind == Kind::ForwardType) {
        return "ForwardType";
    }
    if (mKind == Kind::TypedCellType) {
        return "TypedCellType";
    }
    if (mKind == Kind::PyList) {
        return "PyList";
    }
    if (mKind == Kind::PyDict) {
        return "PyDict";
    }
    if (mKind == Kind::PySet) {
        return "PySet";
    }
    if (mKind == Kind::PyTuple) {
        return "PyTuple";
    }
    if (mKind == Kind::PyClass) {
        return "PyClass";
    }
    if (mKind == Kind::PyObject) {
        return "PyObject";
    }
    if (mKind == Kind::PyFunction) {
        return "PyFunction";
    }
    if (mKind == Kind::PyCodeObject) {
        return "PyCodeObject";
    }
    if (mKind == Kind::PyCell) {
        return "PyCell";
    }
    if (mKind == Kind::PyModule) {
        return "PyModule";
    }
    if (mKind == Kind::PyModuleDict) {
        return "PyModuleDict";
    }
    if (mKind == Kind::PyClassDict) {
        return "PyClassDict";
    }
    if (mKind == Kind::PyStaticMethod) {
        return "PyStaticMethod";
    }
    if (mKind == Kind::PyClassMethod) {
        return "PyClassMethod";
    }
    if (mKind == Kind::PyBoundMethod) {
        return "PyBoundMethod";
    }
    if (mKind == Kind::PyProperty) {
        return "PyProperty";
    }
    if (mKind == Kind::ArbitraryPyObject) {
        return "ArbitraryPyObject";
    }
    if (mKind == Kind::InternalBundle) {
        return "InternalBundle";
    }

    throw std::runtime_error("Unknown PyObjSnapshot Kind: " + format((int)mKind));
}


void PyObjSnapshot::cloneFromSnapByHash(PyObjSnapshot* snap) {
    mKind = snap->mKind;
    mType = snap->mType;
    mInstance = snap->mInstance;
    mPyObject = incref(snap->mPyObject);
    mNamedInts = snap->mNamedInts;
    mNames = snap->mNames;
    mStringValue = snap->mStringValue;
    mModuleName = snap->mModuleName;
    mName = snap->mName;
    mQualname = snap->mQualname;

    auto intern = [&](PyObjSnapshot* s) {
        PyObjSnapshot* res = mGraph->snapshotForHash(
            s->mGraph->hashFor(s)
        );

        if (!res) {
            throw std::runtime_error("Somehow, when interning, a hash is missing");
        }

        return res;
    };

    for (auto e: snap->mElements) {
        mElements.push_back(intern(e));
    }

    for (auto e: snap->mKeys) {
        mKeys.push_back(intern(e));
    }

    for (auto nameAndE: snap->mNamedElements) {
        mNamedElements[nameAndE.first] = intern(nameAndE.second);
    }
}

void PyObjSnapshot::becomeInternalizedOf(
    Type* t,
    PyObjSnapshotMaker& maker
) {
    // we're always the internalized version of this object
    if (maker.linkBackToOriginalObject()) {
        mType = t;
    }

    mNamedInts["type_is_forward"] = t->isForwardDefined() ? 1 : 0;

    if (t->isListOf()) {
        mKind = Kind::ListOfType;
        mNamedElements["element_type"] = maker.internalize(((ListOfType*)t)->getEltType());
        return;
    }

    if (t->isTupleOf()) {
        mKind = Kind::TupleOfType;
        mNamedElements["element_type"] = maker.internalize(((TupleOfType*)t)->getEltType());
        return;
    }

    if (t->isTuple()) {
        mKind = Kind::TupleType;
        for (auto eltType: ((Tuple*)t)->getTypes()) {
            mElements.push_back(maker.internalize(eltType));
        }
        return;
    }

    if (t->isNamedTuple()) {
        mKind = Kind::NamedTupleType;
        for (auto eltType: ((NamedTuple*)t)->getTypes()) {
            mElements.push_back(maker.internalize(eltType));
        }
        for (auto name: ((NamedTuple*)t)->getNames()) {
            mNames.push_back(name);
        }
        return;
    }

    if (t->isOneOf()) {
        mKind = Kind::OneOfType;
        for (auto eltType: ((OneOfType*)t)->getTypes()) {
            mElements.push_back(maker.internalize(eltType));
        }
        return;
    }

    if (t->isValue()) {
        mKind = Kind::ValueType;
        mNamedElements["value_instance"] = maker.internalize(
            ((Value*)t)->value()
        );
        return;
    }

    if (t->isDict()) {
        mKind = Kind::DictType;
        mNamedElements["key_type"] = maker.internalize(
            ((DictType*)t)->keyType()
        );
        mNamedElements["value_type"] = maker.internalize(
            ((DictType*)t)->valueType()
        );
        return;
    }

    if (t->isConstDict()) {
        mKind = Kind::ConstDictType;
        mNamedElements["key_type"] = maker.internalize(
            ((ConstDictType*)t)->keyType()
        );
        mNamedElements["value_type"] = maker.internalize(
            ((ConstDictType*)t)->valueType()
        );
        return;
    }

    if (t->isSet()) {
        mKind = Kind::SetType;
        mNamedElements["key_type"] = maker.internalize(
            ((SetType*)t)->keyType()
        );
        return;
    }

    if (t->isPointerTo()) {
        mKind = Kind::PointerToType;
        mNamedElements["element_type"] = maker.internalize(
            ((PointerTo*)t)->getEltType()
        );
        return;
    }

    if (t->isRefTo()) {
        mKind = Kind::RefToType;
        mNamedElements["element_type"] = maker.internalize(
            ((RefTo*)t)->getEltType()
        );
        return;
    }

    if (t->isPythonObjectOfType()) {
        mKind = Kind::PythonObjectOfTypeType;
        mNamedElements["element_type"] = maker.internalize(
            (PyObject*)((PythonObjectOfType*)t)->pyType()
        );
        return;
    }

    if (t->isSubclassOf()) {
        mKind = Kind::SubclassOfTypeType;
        mNamedElements["element_type"] = maker.internalize(
            ((SubclassOfType*)t)->getSubclassOf()
        );
        return;
    }

    if (t->isTypedCell()) {
        mKind = Kind::TypedCellType;
        mNamedElements["element_type"] = maker.internalize(
            ((TypedCellType*)t)->getHeldType()
        );
        return;
    }

    if (t->isBoundMethod()) {
        mKind = Kind::BoundMethodType;
        mNamedElements["self_type"] = maker.internalize(
            ((BoundMethod*)t)->getFirstArgType()
        );
        mNames.push_back(((BoundMethod*)t)->getFuncName());
        return;
    }

    if (t->isAlternativeMatcher()) {
        mKind = Kind::AlternativeMatcherType;
        mNamedElements["alternative"] = maker.internalize(
            ((AlternativeMatcher*)t)->getAlternative()
        );
        return;
    }

    if (t->isConcreteAlternative()) {
        mKind = Kind::ConcreteAlternativeType;
        mNamedElements["alternative"] = maker.internalize(
            ((ConcreteAlternative*)t)->getAlternative()
        );
        mNamedInts["which"] = ((ConcreteAlternative*)t)->which();
        return;
    }

    if (t->isAlternative()) {
        mKind = Kind::AlternativeType;
        Alternative* alt = (Alternative*)t;

        mName = alt->name();
        mModuleName = alt->moduleName();

        for (long k = 0; k < alt->subtypes().size(); k++) {
            mElements.push_back(maker.internalize(alt->subtypes()[k].second));
            mNames.push_back(alt->subtypes()[k].first);
        }

        mNamedElements["alt_methods"] = maker.internalize(alt->getMethods());
        mNamedElements["alt_subtypes"] = maker.internalize(alt->getSubtypesConcrete());
        return;
    }

    if (t->isClass()) {
        mKind = Kind::ClassType;
        mName = ((Class*)t)->name();
        mModuleName = ((Class*)t)->moduleName();
        mNamedElements["held_class_type"] = maker.internalize(((Class*)t)->getHeldClass());
        return;
    }

    if (t->isHeldClass()) {
        mKind = Kind::HeldClassType;
        HeldClass* cls = (HeldClass*)t;

        mName = cls->name();
        mModuleName = cls->moduleName();

        mNamedElements["cls_type"] = maker.internalize(cls->getClassType());
        mNamedElements["cls_bases"] = maker.internalize(cls->getBases());

        mNamedElements["cls_own_methods"] = maker.internalize(cls->getOwnMemberFunctions());
        mNamedElements["cls_own_staticmethods"] = maker.internalize(cls->getOwnStaticFunctions());
        mNamedElements["cls_own_classmethods"] = maker.internalize(cls->getOwnClassMethods());
        mNamedElements["cls_own_properties"] = maker.internalize(cls->getOwnPropertyFunctions());
        mNamedElements["cls_own_classmembers"] = maker.internalize(cls->getOwnClassMembers());
        mNamedElements["cls_own_members"] = maker.internalize(cls->getOwnMembers());

        mNamedElements["cls_methods"] = maker.internalize(cls->getMemberFunctions());
        mNamedElements["cls_staticmethods"] = maker.internalize(cls->getStaticFunctions());
        mNamedElements["cls_classmethods"] = maker.internalize(cls->getClassMethods());
        mNamedElements["cls_properties"] = maker.internalize(cls->getPropertyFunctions());
        mNamedElements["cls_classmembers"] = maker.internalize(cls->getClassMembers());
        mNamedElements["cls_members"] = maker.internalize(cls->getMembers());

        mNamedInts["cls_is_final"] = cls->isFinal();
        return;
    }

    if (t->isFunction()) {
        mKind = Kind::FunctionType;
        Function* f = (Function*)t;

        mNamedElements["closure_type"] = maker.internalize(f->getClosureType());
        mNamedInts["is_entrypoint"] = f->isEntrypoint() ? 1 : 0;
        mNamedInts["is_nocompile"] = f->isNocompile() ? 1 : 0;

        mName = f->name();
        mQualname = f->qualname();
        mModuleName = f->moduleName();

        mNamedElements["func_overloads"] = maker.internalize(f->getOverloads());
        return;
    }

    if (t->isForward()) {
        mKind = Kind::ForwardType;
        Forward* f = (Forward*)t;

        if (f->getCellOrDict()) {
            mNamedElements["fwd_cell_or_dict"] = maker.internalize(f->getCellOrDict());

            // if we are a 'lambda' forward, we need to know what we resolve to, assuming
            // we do resolve.
            if (PyCell_Check(f->getCellOrDict()) && PyCell_Get(f->getCellOrDict())) {
                mNamedElements["fwd_cell_resolves_to"] = maker.internalize(PyCell_Get(f->getCellOrDict()));
            } else
            if (PyDict_Check(f->getCellOrDict())) {
                PyObject* o = PyDict_GetItemString(f->getCellOrDict(), f->name().c_str());
                if (o) {
                    mNamedElements["fwd_dict_resolves_to"] = maker.internalize(o);
                }
            }
        }

        if (f->getTarget()) {
            mNamedElements["fwd_target"] = maker.internalize(f->getTarget());
        }

        mName = f->name();
        return;
    }

    if (t->isRegister()
        || t->isString()
        || t->isBytes()
        || t->isPyCell()
        || t->isEmbeddedMessage()
        || t->isNone()
    ) {
        mKind = Kind::PrimitiveType;
        mType = t;
        return;
    }

    throw std::runtime_error("Type of category " + t->getTypeCategoryString() + " can't be snapshotted");
}



void PyObjSnapshot::becomeInternalizedOf(
    PyObject* val,
    PyObjSnapshotMaker& maker
) {
    if (PyInstance::extractTypeFrom(val, true)) {
        throw std::runtime_error("types should have been passed to the other pathway");
    }

    // we're always the internalized version of this object
    if (maker.linkBackToOriginalObject()) {
        mPyObject = incref(val);
    }

    static PyObject* lockType = staticPythonInstance("typed_python.internals", "lockType");

    if (val == lockType) {
        mKind = Kind::NamedPyObject;
        mName = "lockType";
        mModuleName = "typed_python.internals";
        return;
    }

    static PyObject* environType = staticPythonInstance("os", "_Environ");

    if (val->ob_type == (PyTypeObject*)environType) {
        mKind = Kind::NamedPyObject;
        mName = "_Environ";
        mModuleName = "os";
        return;
    }

    static PyObject* pyObj_object = staticPythonInstance("builtins", "object");

    if (val == pyObj_object) {
        mKind = Kind::NamedPyObject;
        mName = "object";
        mModuleName = "builtins";
        return;
    }

    static PyObject* pyObj_type = staticPythonInstance("builtins", "type");

    if (val == pyObj_type) {
        mKind = Kind::NamedPyObject;
        mName = "type";
        mModuleName = "builtins";
        return;
    }

    if (PyUnicode_Check(val)) {
        mKind = Kind::String;
        mStringValue = PyUnicode_AsUTF8(val);
        return;
    }

    if (PyTuple_Check(val)) {
        mKind = Kind::PyTuple;
        for (long i = 0; i < PyTuple_Size(val); i++) {
            mElements.push_back(maker.internalize(PyTuple_GetItem(val, i)));
        }
        return;
    }

    if (PyList_Check(val)) {
        mKind = Kind::PyList;
        for (long i = 0; i < PyList_Size(val); i++) {
            mElements.push_back(maker.internalize(PyList_GetItem(val, i)));
        }
        return;
    }

    if (PySet_Check(val)) {
        mKind = Kind::PySet;
        iterate(val, [&](PyObject* o) {
            mElements.push_back(maker.internalize(o));
        });
        return;
    }

    if (PyDict_Check(val)) {
        // see if this is a moduledict
        PyObject* mname = PyDict_GetItemString(val, "__name__");
        if (mname && PyUnicode_Check(mname)) {
            PyObject* sysModuleModules = staticPythonInstance("sys", "modules");
            PyObjectStealer moduleObj(PyObject_GetItem(sysModuleModules, mname));

            if (moduleObj) {
                PyObjectStealer moduleObjDict(PyObject_GenericGetDict(moduleObj, nullptr));
                if (!moduleObjDict) {
                    PyErr_Clear();
                } else
                if (moduleObjDict == val) {
                    mKind = Kind::PyModuleDict;
                    mName = PyUnicode_AsUTF8(mname);
                    mNamedElements["module_dict_of"] = maker.internalize(moduleObj);
                    return;
                }
            } else {
                PyErr_Clear();
            }
        }

        // see if this is a vanilla class dict
        PyObject* dictAccessor = PyDict_GetItemString(val, "__dict__");
        if (dictAccessor && dictAccessor->ob_type == &PyGetSetDescr_Type) {
            PyTypeObject* clsType = PyDescr_TYPE(dictAccessor);
            if (clsType) {
                if (clsType->tp_dict == val) {
                    if (isVanillaClassType(clsType)) {
                        mKind = Kind::PyClassDict;

                        mNamedElements["class_dict_of"] = maker.internalize(
                            (PyObject*)clsType
                        );

                        PyObject *key, *value;
                        Py_ssize_t pos = 0;

                        while (val && PyDict_Next(val, &pos, &key, &value)) {
                            if (PyUnicode_Check(key)
                                && PyUnicode_AsUTF8(key) != std::string("__dict__")
                                && PyUnicode_AsUTF8(key) != std::string("__weakref__")
                            ) {
                                mElements.push_back(maker.internalize(value));
                                mKeys.push_back(maker.internalize(key));
                            }
                        }

                        return;
                    }
                }
            }
        }

        // this is a vanilla dict
        mKind = Kind::PyDict;

        PyObject *key, *value;
        Py_ssize_t pos = 0;

        while (val && PyDict_Next(val, &pos, &key, &value)) {
            mElements.push_back(maker.internalize(value));
            mKeys.push_back(maker.internalize(key));
        }
        return;
    }

    if (PyCell_Check(val)) {
        mKind = Kind::PyCell;

        if (PyCell_Get(val)) {
            mNamedElements["cell_contents"] = maker.internalize(PyCell_Get(val));
        }

        return;
    }

    if (PyType_Check(val)) {
        PyTypeObject* tp = (PyTypeObject*)val;

        if (isVanillaClassType(tp)) {
            mKind = Kind::PyClass;

            mName = tp->tp_name;
            mModuleName = dictGetStringOrEmpty(tp->tp_dict, "__module__");

            if (tp->tp_dict) {
                mNamedElements["cls_dict"] = maker.internalize(tp->tp_dict);
            }
            if (tp->tp_bases) {
                mNamedElements["cls_bases"] = maker.internalize(tp->tp_bases);
            }

            return;
        }
    }

    if (PyFunction_Check(val)) {
        mKind = Kind::PyFunction;

        PyFunctionObject* f = (PyFunctionObject*)val;

        mName = stringOrEmpty(f->func_name);
        mModuleName = stringOrEmpty(f->func_module);

        if (f->func_name) {
            mNamedElements["func_name"] = maker.internalize(f->func_name);
        }
        if (f->func_module) {
            mNamedElements["func_module"] = maker.internalize(f->func_module);
        }
        if (f->func_qualname) {
            mNamedElements["func_qualname"] = maker.internalize(f->func_qualname);
        }
        if (PyFunction_GetClosure(val)) {
            mNamedElements["func_closure"] = maker.internalize(PyFunction_GetClosure(val));
        }
        if (PyFunction_GetCode(val)) {
            mNamedElements["func_code"] = maker.internalize(PyFunction_GetCode(val));
        }
        if (PyFunction_GetModule(val)) {
            mNamedElements["func_module"] = maker.internalize(PyFunction_GetModule(val));
        }
        if (PyFunction_GetAnnotations(val)) {
            mNamedElements["func_annotations"] = maker.internalize(PyFunction_GetAnnotations(val));
        }
        if (PyFunction_GetDefaults(val)) {
            mNamedElements["func_defaults"] = maker.internalize(PyFunction_GetDefaults(val));
        }
        if (PyFunction_GetKwDefaults(val)) {
            mNamedElements["func_kwdefaults"] = maker.internalize(PyFunction_GetKwDefaults(val));
        }
        if (PyFunction_GetGlobals(val)) {
            mNamedElements["func_globals"] = maker.internalize(PyFunction_GetGlobals(val));
        }
        return;
    }

    if (PyCode_Check(val)) {
        mKind = Kind::PyCodeObject;

        PyCodeObject* co = (PyCodeObject*)val;

        mNamedInts["co_argcount"] = co->co_argcount;
        mNamedInts["co_kwonlyargcount"] = co->co_kwonlyargcount;
        mNamedInts["co_nlocals"] = co->co_nlocals;
        mNamedInts["co_stacksize"] = co->co_stacksize;
        mNamedInts["co_firstlineno"] = co->co_firstlineno;
        mNamedInts["co_posonlyargcount"] = co->co_posonlyargcount;
        mNamedInts["co_flags"] = co->co_flags;

        mNamedElements["co_code"] = maker.internalize(co->co_code);
        mNamedElements["co_consts"] = maker.internalize(co->co_consts);
        mNamedElements["co_names"] = maker.internalize(co->co_names);
        mNamedElements["co_varnames"] = maker.internalize(co->co_varnames);
        mNamedElements["co_freevars"] = maker.internalize(co->co_freevars);
        mNamedElements["co_cellvars"] = maker.internalize(co->co_cellvars);
        mNamedElements["co_name"] = maker.internalize(co->co_name);
        mNamedElements["co_filename"] = maker.internalize(co->co_filename);

#           if PY_MINOR_VERSION >= 10
            mNamedElements["co_linetable"] = maker.internalize(co->co_linetable);
#           else
            mNamedElements["co_lnotab"] = maker.internalize(co->co_lnotab);
#           endif

        return;
    }

    ::Type* instanceType = PyInstance::extractTypeFrom(val->ob_type);
    if (instanceType) {
        throw std::runtime_error(
            "PyObjSnapshotMaker::internalize should already have cast this to an instance call"
        );
        return;
    }

    if (val == Py_None) {
        mKind = Kind::Instance;
        return;
    }

    if (PyBool_Check(val)) {
        mKind = Kind::Instance;
        mInstance = Instance::create(val == Py_True);
        return;
    }

    if (PyLong_Check(val)) {
        mKind = Kind::Instance;

        try {
            mInstance = Instance::create((int64_t)PyLong_AsLongLong(val));
        }
        catch(...) {
            mInstance = Instance::create((uint64_t)PyLong_AsUnsignedLongLong(val));
        }

        return;
    }

    if (PyFloat_Check(val)) {
        mKind = Kind::Instance;
        mInstance = Instance::create(PyFloat_AsDouble(val));
        return;
    }

    if (PyModule_Check(val)) {
        PyObject* sysModuleModules = staticPythonInstance("sys", "modules");

        PyObjectStealer name(PyObject_GetAttrString(val, "__name__"));
        if (name) {
            if (PyUnicode_Check(name)) {
                PyObjectStealer moduleObject(PyObject_GetItem(sysModuleModules, name));
                if (moduleObject) {
                    if (moduleObject == val) {
                        mKind = Kind::PyModule;
                        mName = PyUnicode_AsUTF8(name);
                        return;
                    }
                } else {
                    PyErr_Clear();
                }
            }
        } else {
            PyErr_Clear();
        }
    }

    if (PyBytes_Check(val)) {
        mKind = Kind::Instance;
        mInstance = Instance::createAndInitialize(
            BytesType::Make(),
            [&](instance_ptr i) {
                BytesType::Make()->constructor(
                    i,
                    PyBytes_GET_SIZE(val),
                    PyBytes_AsString(val)
                );
            }
        );
        return;
    }

    if (isVanillaClassType(val->ob_type)) {
        mKind = Kind::PyObject;

        mNamedElements["inst_type"] = maker.internalize((PyObject*)val->ob_type);

        PyObjectStealer dict(PyObject_GenericGetDict(val, nullptr));
        if (dict) {
            mNamedElements["inst_dict"] = maker.internalize(dict);
        }
        return;
    }

    if (val->ob_type == &PyStaticMethod_Type || val->ob_type == &PyClassMethod_Type) {
        if (val->ob_type == &PyStaticMethod_Type) {
            mKind = Kind::PyStaticMethod;
        } else {
            mKind = Kind::PyClassMethod;
        }

        PyObjectStealer funcObj(PyObject_GetAttrString(val, "__func__"));

        mNamedElements["meth_func"] = maker.internalize(funcObj);

        return;
    }

    if (val->ob_type == &PyProperty_Type) {
        mKind = Kind::PyProperty;

        JustLikeAPropertyObject* prop = (JustLikeAPropertyObject*)val;

        if (prop->prop_get) {
            mNamedElements["prop_get"] = maker.internalize(prop->prop_get);
        }
        if (prop->prop_set) {
            mNamedElements["prop_set"] = maker.internalize(prop->prop_set);
        }
        if (prop->prop_del) {
            mNamedElements["prop_del"] = maker.internalize(prop->prop_del);
        }
        if (prop->prop_doc) {
            mNamedElements["prop_doc"] = maker.internalize(prop->prop_doc);
        }

        #if PY_MINOR_VERSION >= 10
        if (prop->prop_name) {
            mNamedElements["prop_name"] = maker.internalize(prop->prop_name);
        }
        #endif

        return;
    }

    if (val->ob_type == &PyMethod_Type) {
        mKind = Kind::PyBoundMethod;

        PyObjectStealer fself(PyObject_GetAttrString(val, "__self__"));
        PyObjectStealer ffunc(PyObject_GetAttrString(val, "__func__"));

        mNamedElements["meth_self"] = maker.internalize(fself);
        mNamedElements["meth_func"] = maker.internalize(ffunc);
        return;
    }

    if (isPyObjectGloballyIdentifiable(val)) {
        mKind = Kind::NamedPyObject;

        // no checks are necessary because isPyObjectGloballyIdentifiable
        // confirms that this is OK.
        PyObjectStealer moduleName(PyObject_GetAttrString(val, "__module__"));
        PyObjectStealer clsName(PyObject_GetAttrString(val, "__name__"));

        mModuleName = PyUnicode_AsUTF8(moduleName);
        mName = PyUnicode_AsUTF8(clsName);
        return;
    }

    mKind = Kind::ArbitraryPyObject;
    mPyObject = incref(val);
}

void PyObjSnapshot::becomeInternalizedOf(
    InstanceRef val,
    PyObjSnapshotMaker& maker
) {
    mKind = Kind::Instance;
    mInstance = val;
}

void PyObjSnapshot::becomeInternalizedOf(
    const MemberDefinition& val,
    PyObjSnapshotMaker& maker
) {
    mKind = Kind::ClassMemberDefinition;
    mName = val.getName();
    mNamedElements["type"] = maker.internalize(val.getType());
    mNamedElements["defaultValue"] = maker.internalize(val.getDefaultValue());
    mNamedInts["isNonempty"] = val.getIsNonempty() ? 1 : 0;
}

void PyObjSnapshot::becomeInternalizedOf(
    const FunctionGlobal& val,
    PyObjSnapshotMaker& maker
) {
    mKind = Kind::FunctionGlobal;

    if (val.isUnbound()) {
        return;
    }

    if (val.isNamedModuleMember()) {
        mModuleName = val.getModuleName();
        mName = val.getName();
        mNamedElements["moduleDict"] = maker.internalize(val.getModuleDictOrCell());

        // TODO: ultimately we should get rid of this. It shouldn't be necessary to
        // establish a valid 'identity' and its a little arbitrary that we're only recursing
        // into types instead of all python objects. For the moment, we need it to keep everything
        // working.
        PyObject* o = val.extractGlobalRefFromDictOrCell();
        if (o && PyType_Check(o) && PyInstance::extractTypeFrom(o, true)) {
            mNamedElements["resolve_to"] = maker.internalize(
                PyInstance::extractTypeFrom(o, true)
            );
        }

        return;
    }

    if (val.isGlobalInCell()) {
        mNamedElements["cell"] = maker.internalize(val.getModuleDictOrCell());
        return;
    }

    if (val.isGlobalInDict()) {
        mNamedElements["moduleDict"] = maker.internalize(val.getModuleDictOrCell());
        mName = val.getName();
        return;
    }

    if (val.isConstant()) {
        mNamedElements["constant"] = maker.internalize(val.getConstant());
        return;
    }
}

void PyObjSnapshot::becomeInternalizedOf(
    const FunctionOverload& val,
    PyObjSnapshotMaker& maker
) {
    mKind = Kind::FunctionOverload;

    mNamedElements["func_globals"] = maker.internalize(val.getGlobals());
    mNamedElements["func_code"] = maker.internalize(val.getFunctionCode());
    if (val.getFunctionDefaults()) {
        mNamedElements["func_defaults"] = maker.internalize(val.getFunctionDefaults());
    }
    if (val.getFunctionAnnotations()) {
        mNamedElements["func_annotations"] = maker.internalize(val.getFunctionAnnotations());
    }
    mNamedElements["func_args"] = maker.internalize(val.getArgs());
    mNamedElements["func_closure_varnames"] = maker.internalize(
        val.getFunctionClosureVarnames()
    );
    mNamedElements["func_globals_in_closure_varnames"] = maker.internalize(
        val.getFunctionGlobalsInClosureVarnames()
    );

    if (val.getMethodOf()) {
        mNamedElements["func_method_of"] = maker.internalize(val.getMethodOf());
    }

    if (val.getSignatureFunction()) {
        mNamedElements["func_signature_func"] = maker.internalize(val.getSignatureFunction());
    }

    mNamedElements["func_closure_variable_bindings"] = maker.internalize(
        val.getClosureVariableBindings()
    );

    if (val.getReturnType()) {
        mNamedElements["func_ret_type"] = maker.internalize(val.getReturnType());
    }
}


void PyObjSnapshot::becomeInternalizedOf(
    const FunctionArg& arg,
    PyObjSnapshotMaker& maker
) {
    mKind = Kind::FunctionArg;

    mName = arg.getName();
    if (arg.getTypeFilter()) {
        mNamedElements["arg_type_filter"] = maker.internalize(arg.getTypeFilter());
    }

    if (arg.getDefaultValue()) {
        mNamedElements["arg_default_value"] = maker.internalize(arg.getDefaultValue());
    }

    mNamedInts["arg_is_star_arg"] = arg.getIsStarArg() ? 1 : 0;
    mNamedInts["arg_is_kwarg"] = arg.getIsKwarg() ? 1 : 0;
}

void PyObjSnapshot::becomeInternalizedOf(
    const ClosureVariableBinding& val,
    PyObjSnapshotMaker& maker
) {
    mKind = Kind::FunctionClosureVariableBinding;

    for (long k = 0; k < val.size(); k++) {
        mElements.push_back(maker.internalize(val[k]));
    }
}

void PyObjSnapshot::becomeInternalizedOf(
    const ClosureVariableBindingStep& step,
    PyObjSnapshotMaker& maker
) {
    mKind = Kind::FunctionClosureVariableBindingStep;

    if (step.isFunction()) {
        mNamedElements["step_func"] = maker.internalize(step.getFunction());
    }
    if (step.isNamedField()) {
        mNames.push_back(step.getNamedField());
    }
    if (step.isIndexedField()) {
        mNamedInts["step_index"] = step.getIndexedField();
    }
}

void PyObjSnapshot::becomeInternalizedOf(
    const std::string& val,
    PyObjSnapshotMaker& maker
) {
    mKind = Kind::String;
    mStringValue = val;
}

PyObjSnapshot* PyObjSnapshot::computeForwardTargetTransitive() {
    PyObjSnapshot* source = this;
    int steps = 0;

    while (true) {
        steps += 1;
        if (steps > mGraph->getObjects().size()) {
            throw std::runtime_error("Forward cycle detected");
        }

        if (source->mKind != Kind::ForwardType) {
            // TODO - what if we're not a type?
            if (!source->willBeATpType()) {
                throw std::runtime_error("Should we be converting this to a Value type?");
            }

            return source;
        }

        source = source->computeForwardTarget();
        if (!source) {
            return nullptr;
        }
    }
}

