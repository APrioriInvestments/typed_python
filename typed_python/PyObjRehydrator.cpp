#include "PyObjRehydrator.hpp"
#include "PyObjSnapshot.hpp"

void PyObjRehydrator::start(PyObjSnapshot* snapshot) {
    std::function<void (PyObjSnapshot*)> visit = [&](PyObjSnapshot* snap) {
        if (mSnapshots.find(snap) != mSnapshots.end()) {
            return;
        }

        if (!snap->isUncacheable()) {
            if (!snap->needsHydration()) {
                return;
            }

            mSnapshots.insert(snap);
        }

        for (auto elt: snap->mElements) {
            visit(elt);
        }

        for (auto elt: snap->mKeys) {
            visit(elt);
        }

        for (auto& nameAndElt: snap->mNamedElements) {
            visit(nameAndElt.second);
        }
    };

    visit(snapshot);
}

void PyObjRehydrator::buildPyobjForTpType(PyObjSnapshot* s) {
    if (!s->mType) {
        throw std::runtime_error("Somehow this Snapshot doesn't have a Type.");
    }
    if (s->mPyObject) {
        return;
    }

    // these objects don't actually build a python representation for the type - we always
    // use the actual type
    if (s->mKind == PyObjSnapshot::Kind::PythonObjectOfTypeType) {
        return;
    }

    s->mPyObject = incref((PyObject*)PyInstance::typeObj(s->mType));

    if (!s->mPyObject) {
        throw std::runtime_error("Somehow we don't have a PyObject for our type.");
    }
}


void PyObjRehydrator::rehydrateAll() {
    // first, do a pass building the skeletons for all of our objects.
    // after this, there will be a Type* allocated for every new Type object with the
    // correct name, which is enough to construct the PyTypeObject for it.
    for (auto s: mSnapshots) {
        if (s->willBeATpType()) {
            buildTypeSkeleton(s);
        }
    }

    // now that every TP type knows its name, we can instantiate it
    for (auto s: mSnapshots) {
        if (s->willBeATpType()) {
            buildPyobjForTpType(s);
        }
    }

    // then actually initializethe type objects. these will build PyObjects as needed
    for (auto s: mSnapshots) {
        if (s->willBeATpType()) {
            rehydrateTpType(s);
        }
    }

    for (auto s: mSnapshots) {
        rehydrate(s);
    }

    for (auto n: mSnapshots) {
        finalizeRehydration(n);
    }

    std::vector<Type*> types;
    for (auto o: mSnapshots) {
        if (o->mType) {
            types.push_back(o->mType);
        }
    }

    std::sort(types.begin(), types.end(), [&](Type* l, Type* r) {
        if (l->typeLevel() < r->typeLevel()) {
            return true;
        }
        if (l->typeLevel() > r->typeLevel()) {
            return false;
        }
        return l < r;
    });

    bool anyUpdated = true;
    size_t passCt = 0;
    while (anyUpdated) {
        anyUpdated = false;
        for (auto t: types) {
            if (t->postInitialize()) {
                anyUpdated = true;
            }
        }
        passCt += 1;

        // we can run this algorithm until all type sizes have stabilized. Conceivably we
        // could introduce an error that would cause this to not converge - this should
        // detect that.
        if (passCt > types.size() * 2 + 10) {
            throw std::runtime_error("Type size graph is not stabilizing.");
        }
    }

    // let each type update any internal caches it might need before it gets instantiated
    for (auto t: types) {
        t->finalizeType();
    }

    for (auto t: types) {
        t->typeFinishedBeingDeserializedPhase1();
    }

    // now that we've set function globals, we need to go over all the classes and
    // held classes and make sure the versions of the function types they're actually
    // using have the new definitions. Unfortunately, Overload objects are not pointers
    // (maybe they should be?) and so when we merge Overloads from base/child classes
    // they don't get their globals replaced with the appropriate values
    for (auto& t: types) {
        if (t && t->isHeldClass()) {
            ((HeldClass*)t)->mergeOwnFunctionsIntoInheritanceTree();
        }
    }

    for (auto t: types) {
        t->typeFinishedBeingDeserializedPhase2();
    }

    for (auto obj: mSnapshots) {
        if (obj->mType) {
            obj->mPyObject = incref((PyObject*)PyInstance::typeObj(obj->mType));
        }
    }
}

Type* PyObjRehydrator::typeFor(PyObjSnapshot* snapshot) {
    if (!snapshot->needsHydration()) {
        return snapshot->getType();
    }

    if (snapshot->willBeATpType()) {
        rehydrate(snapshot);
        return snapshot->mType;
    }

    return nullptr;
}


PyObject* PyObjRehydrator::pyobjFor(PyObjSnapshot* snapshot) {
    if (!snapshot->needsHydration()) {
        return snapshot->getPyObj();
    }

    rehydrate(snapshot);

    return snapshot->mPyObject;
}

void PyObjRehydrator::getFrom(
    PyObjSnapshot* snapshot,
    Type*& out
) {
    out = typeFor(snapshot);
    if (!out) {
        throw std::runtime_error("Corrupt PyObjSnapshot.InternalBundle - expected a type");
    }
}

void PyObjRehydrator::getFrom(
    PyObjSnapshot* snapshot,
    PyObject*& out
) {
    out = pyobjFor(snapshot);
    if (!out) {
        throw std::runtime_error("Corrupt PyObjSnapshot.InternalBundle - expected a pyobj");
    }
}

void PyObjRehydrator::getFrom(
    PyObjSnapshot* snapshot,
    HeldClass*& out
) {
    Type* t = typeFor(snapshot);
    if (!t->isHeldClass()) {
        throw std::runtime_error("Corrupt PyObjSnapshot - expected a HeldClass");
    }
    out = (HeldClass*)t;
}

void PyObjRehydrator::getFrom(
    PyObjSnapshot* snapshot,
    Function*& out
) {
    Type* t = typeFor(snapshot);
    if (!t->isFunction()) {
        throw std::runtime_error("Corrupt PyObjSnapshot - expected a Function");
    }
    out = (Function*)t;
}

void PyObjRehydrator::getFrom(
    PyObjSnapshot* e,
    MemberDefinition& out
) {
    out = MemberDefinition(
        e->mName,
        getNamedElementType(e, "type"),
        getNamedElementInstance(e, "defaultValue"),
        e->mNamedInts["isNonempty"]
    );
}

void PyObjRehydrator::getFrom(
    PyObjSnapshot* snapshot,
    FunctionOverload& out
) {
    std::map<std::string, FunctionGlobal> globals;
    std::vector<std::string> pyFuncClosureVarnames;
    std::vector<std::string> globalsInClosureVarnames;
    std::map<std::string, ClosureVariableBinding> closureBindings;
    std::vector<FunctionArg> args;

    getNamedBundle(snapshot, "func_globals", globals);
    getNamedBundle(snapshot, "func_closure_varnames", pyFuncClosureVarnames);
    getNamedBundle(snapshot, "func_globals_in_closure_varnames", globalsInClosureVarnames);
    getNamedBundle(snapshot, "func_closure_variable_bindings", closureBindings);
    getNamedBundle(snapshot, "func_args", args);

    out = FunctionOverload(
        getNamedElementPyobj(snapshot, "func_code"),
        getNamedElementPyobj(snapshot, "func_defaults", true),
        getNamedElementPyobj(snapshot, "func_annotations", true),
        globals,
        pyFuncClosureVarnames,
        globalsInClosureVarnames,
        closureBindings,
        getNamedElementType(snapshot, "func_ret_type", true),
        getNamedElementPyobj(snapshot, "func_signature_func", true),
        args,
        getNamedElementType(snapshot, "func_method_of", true)
    );
}

void PyObjRehydrator::getFrom(
    PyObjSnapshot* snapshot,
    std::string& out
) {
    if (!snapshot || snapshot->mKind != PyObjSnapshot::Kind::String) {
        throw std::runtime_error("Corrupt PyObjSnapshot.String");
    }

    out = snapshot->getStringValue();
}

void PyObjRehydrator::getFrom(
    PyObjSnapshot* snapshot,
    FunctionGlobal& out
) {
    if (snapshot->mNamedElements.find("moduleDict") != snapshot->mNamedElements.end() && snapshot->mModuleName.size()) {
        out = FunctionGlobal::NamedModuleMember(
            getNamedElementPyobj(snapshot, "moduleDict"),
            snapshot->mModuleName,
            snapshot->mName
        );
        return;
    }

    if (snapshot->mNamedElements.find("moduleDict") != snapshot->mNamedElements.end()) {
        out = FunctionGlobal::GlobalInDict(
            getNamedElementPyobj(snapshot, "moduleDict"),
            snapshot->mName
        );
        return;
    }

    if (snapshot->mNamedElements.find("cell") != snapshot->mNamedElements.end()) {
        out = FunctionGlobal::GlobalInCell(
            getNamedElementPyobj(snapshot, "cell")
        );
        return;
    }
}

void PyObjRehydrator::getFrom(
    PyObjSnapshot* snapshot,
    FunctionArg& out
) {
    Type* filter = getNamedElementType(snapshot, "arg_type_filter", true);
    PyObject* defaultVal = getNamedElementPyobj(snapshot, "arg_default_value", true);
    bool isStarArg = snapshot->mNamedInts["arg_is_star_arg"];
    bool isKwarg = snapshot->mNamedInts["arg_is_kwarg"];

    out = FunctionArg(
        snapshot->mName,
        filter,
        defaultVal,
        isStarArg,
        isKwarg
    );
}

void PyObjRehydrator::getFrom(
    PyObjSnapshot* snapshot,
    ClosureVariableBinding& out
) {
    std::vector<ClosureVariableBindingStep> steps;

    for (auto e: snapshot->mElements) {
        ClosureVariableBindingStep step;
        getFrom(e, step);
        steps.push_back(step);
    }

    out = ClosureVariableBinding(steps);
}

void PyObjRehydrator::getFrom(
    PyObjSnapshot* snapshot,
    ClosureVariableBindingStep& out
) {
    if (snapshot->mNamedElements.find("step_func") != snapshot->mNamedElements.end()) {
        Function* f;
        getFrom(snapshot->mNamedElements.find("step_func")->second, f);
        out = ClosureVariableBindingStep(f);
        return;
    }

    if (snapshot->mNames.size()) {
        out = ClosureVariableBindingStep(snapshot->mNames[0]);
        return;
    }

    if (snapshot->mNamedInts.find("step_index") != snapshot->mNamedInts.end()) {
        out = ClosureVariableBindingStep(snapshot->mNamedInts["step_index"]);
        return;
    }
}

Instance PyObjRehydrator::getNamedElementInstance(
    PyObjSnapshot* snapshot,
    std::string name,
    bool allowEmpty
) {
    // TODO: this is just wrong
    return snapshot->mInstance;
}

PyObject* PyObjRehydrator::getNamedElementPyobj(
    PyObjSnapshot* snapshot,
    std::string name,
    bool allowEmpty
) {
    auto it = snapshot->mNamedElements.find(name);
    if (it == snapshot->mNamedElements.end()) {
        if (allowEmpty) {
            return nullptr;
        }
        throw std::runtime_error(
            "Corrupt PyObjSnapshot." + snapshot->kindAsString() + ". missing " + name
        );
    }

    if (!it->second) {
        throw std::runtime_error("Corrupt PyObjSnapshot: nullptr in mNamedElements");
    }

    PyObject* o = pyobjFor(it->second);

    if (!o && !allowEmpty) {
        throw std::runtime_error("Corrupt " + snapshot->toString()
            + ": no pyobj for elt " + name + " which is " + it->second->toString()
        );
    }

    return pyobjFor(it->second);
}

Type* PyObjRehydrator::getNamedElementType(
    PyObjSnapshot* snapshot,
    std::string name,
    bool allowEmpty
) {
    auto it = snapshot->mNamedElements.find(name);
    if (it == snapshot->mNamedElements.end()) {
        if (allowEmpty) {
            return nullptr;
        }
        throw std::runtime_error(
            "Corrupt PyObjSnapshot." + snapshot->kindAsString() + ". missing " + name
        );
    }

    return typeFor(it->second);
}

void PyObjRehydrator::buildTypeSkeleton(PyObjSnapshot* snap) {
    if (snap->mKind == PyObjSnapshot::Kind::PrimitiveType) {
        // nothing to do
        return;
    }

    if (snap->mType) {
        throw std::runtime_error("Somehow this snapshot already has a Type?");
    }

    if (snap->mKind == PyObjSnapshot::Kind::ListOfType) {
        snap->mType = new ListOfType();
    } else
    if (snap->mKind == PyObjSnapshot::Kind::TupleOfType) {
        snap->mType = new TupleOfType();
    } else
    if (snap->mKind == PyObjSnapshot::Kind::DictType) {
        snap->mType = new DictType();
    } else
    if (snap->mKind == PyObjSnapshot::Kind::ConstDictType) {
        snap->mType = new ConstDictType();
    } else
    if (snap->mKind == PyObjSnapshot::Kind::SetType) {
        snap->mType = new SetType();
    } else
    if (snap->mKind == PyObjSnapshot::Kind::PointerToType) {
        snap->mType = new PointerTo();
    } else
    if (snap->mKind == PyObjSnapshot::Kind::RefToType) {
        snap->mType = new RefTo();
    } else
    if (snap->mKind == PyObjSnapshot::Kind::ValueType) {
        snap->mType = new Value();
    } else
    if (snap->mKind == PyObjSnapshot::Kind::OneOfType) {
        snap->mType = new OneOfType();
    } else
    if (snap->mKind == PyObjSnapshot::Kind::PythonObjectOfTypeType) {
        snap->mType = new PythonObjectOfType();
    } else
    if (snap->mKind == PyObjSnapshot::Kind::TupleType) {
        snap->mType = new Tuple();
    } else
    if (snap->mKind == PyObjSnapshot::Kind::NamedTupleType) {
        snap->mType = new NamedTuple();
    } else
    if (snap->mKind == PyObjSnapshot::Kind::BoundMethodType) {
        snap->mType = new BoundMethod();
    } else
    if (snap->mKind == PyObjSnapshot::Kind::TypedCellType) {
        snap->mType = new TypedCellType();
    } else
    if (snap->mKind == PyObjSnapshot::Kind::ForwardType) {
        snap->mType = new Forward(snap->mName);
    } else
    if (snap->mKind == PyObjSnapshot::Kind::ConcreteAlternativeType) {
        snap->mType = new ConcreteAlternative();
    } else
    if (snap->mKind == PyObjSnapshot::Kind::AlternativeMatcherType) {
        snap->mType = new AlternativeMatcher();
    } else
    if (snap->mKind == PyObjSnapshot::Kind::AlternativeType) {
        snap->mType = new Alternative();
    } else
    if (snap->mKind == PyObjSnapshot::Kind::ClassType) {
        snap->mType = new Class();
    } else
    if (snap->mKind == PyObjSnapshot::Kind::HeldClassType) {
        snap->mType = new HeldClass();
    } else
    if (snap->mKind == PyObjSnapshot::Kind::FunctionType) {
        snap->mType = new Function();
    } else {
        throw std::runtime_error("Can't rehydrate a PyObjSnapshot of kind " + snap->kindAsString());
    }

    snap->mType->markActivelyBeingDeserialized(
        snap->mNamedInts["type_is_forward"],
        snap->mModuleName,
        snap->mName
    );

    return;
}

void PyObjRehydrator::rehydrateTpType(PyObjSnapshot* snap) {
    if (!snap->mType) {
        return;
    }

    if (snap->mKind == PyObjSnapshot::Kind::ListOfType) {
        ((ListOfType*)snap->mType)->initializeDuringDeserialization(getNamedElementType(snap, "element_type"));
        return;
    }

    if (snap->mKind == PyObjSnapshot::Kind::TupleOfType) {
        ((TupleOfType*)snap->mType)->initializeDuringDeserialization(getNamedElementType(snap, "element_type"));
        return;
    }

    if (snap->mKind == PyObjSnapshot::Kind::DictType) {
        ((DictType*)snap->mType)->initializeDuringDeserialization(
            getNamedElementType(snap, "key_type"),
            getNamedElementType(snap, "value_type")
        );
        return;
    }

    if (snap->mKind == PyObjSnapshot::Kind::ConstDictType) {
        ((ConstDictType*)snap->mType)->initializeDuringDeserialization(
            getNamedElementType(snap, "key_type"),
            getNamedElementType(snap, "value_type")
        );
        return;
    }

    if (snap->mKind == PyObjSnapshot::Kind::SetType) {
        ((SetType*)snap->mType)->initializeDuringDeserialization(
            getNamedElementType(snap, "key_type")
        );
        return;
    }

    if (snap->mKind == PyObjSnapshot::Kind::PointerToType) {
        ((PointerTo*)snap->mType)->initializeDuringDeserialization(
            getNamedElementType(snap, "element_type")
        );
        return;
    }

    if (snap->mKind == PyObjSnapshot::Kind::RefToType) {
        ((RefTo*)snap->mType)->initializeDuringDeserialization(
            getNamedElementType(snap, "element_type")
        );
        return;
    }

    if (snap->mKind == PyObjSnapshot::Kind::ValueType) {
        if (snap->mNamedElements.find("value_instance") == snap->mNamedElements.end()) {
            throw std::runtime_error("Corrupt PyObjSnapshot::Value");
        }

        // TODO: this needs to be a proper rehydration when we are
        // able to fully model 'instance' objects.
        // rehydrate(snap->mNamedElements["value_instance"]);

        ((Value*)snap->mType)->initializeDuringDeserialization(
            snap->mNamedElements["value_instance"]->mInstance
        );

        return;
    }

    if (snap->mKind == PyObjSnapshot::Kind::OneOfType) {
        std::vector<Type*> types;
        for (auto s: snap->mElements) {
            types.push_back(typeFor(s));
        }

        ((OneOfType*)snap->mType)->initializeDuringDeserialization(types);

        return;
    }

    if (snap->mKind == PyObjSnapshot::Kind::PythonObjectOfTypeType) {
        PyObject* obType = getNamedElementPyobj(snap, "element_type", false);

        if (!PyType_Check(obType)) {
            throw std::runtime_error("Can't rehydrate a PythonObjectOfType without a subclass of type");
        };

        ((PythonObjectOfType*)snap->mType)->initializeDuringDeserialization((PyTypeObject*)obType);
        return;
    }

    if (snap->mKind == PyObjSnapshot::Kind::TupleType) {
        std::vector<Type*> types;
        for (auto s: snap->mElements) {
            types.push_back(typeFor(s));
        }

        ((Tuple*)snap->mType)->initializeDuringDeserialization(types);

        return;
    }

    if (snap->mKind == PyObjSnapshot::Kind::NamedTupleType) {
        std::vector<Type*> types;
        for (auto s: snap->mElements) {
            types.push_back(typeFor(s));
        }

        ((NamedTuple*)snap->mType)->initializeDuringDeserialization(types, snap->mNames);

        return;
    }

    if (snap->mKind == PyObjSnapshot::Kind::BoundMethodType) {
        if (snap->mNames.size() != 1) {
            throw std::runtime_error("Corrupt PyObjSnapshot::BoundMethod");
        }

        ((BoundMethod*)snap->mType)->initializeDuringDeserialization(
            snap->mNames[0],
            getNamedElementType(snap, "self_type")
        );

        return;
    }

    if (snap->mKind == PyObjSnapshot::Kind::TypedCellType) {
        ((TypedCellType*)snap->mType)->initializeDuringDeserialization(
            getNamedElementType(snap, "element_type")
        );

        return;
    }

    if (snap->mKind == PyObjSnapshot::Kind::ForwardType) {
        if (snap->mNamedElements.find("fwd_cell_or_dict") != snap->mNamedElements.end()) {
            ((Forward*)snap->mType)->setCellOrDict(
                getNamedElementPyobj(snap, "fwd_cell_or_dict")
            );
        }

        if (snap->mNamedElements.find("fwd_target") != snap->mNamedElements.end()) {
            ((Forward*)snap->mType)->define(
                getNamedElementType(snap, "fwd_target")
            );
        }

        return;
    }

    if (snap->mKind == PyObjSnapshot::Kind::ConcreteAlternativeType) {
        if (!getNamedElementType(snap, "alternative")->isAlternative()){
            throw std::runtime_error("Corrupt PyObjSnapshot.Alternative");
        }

        ((ConcreteAlternative*)snap->mType)->initializeDuringDeserialization(
            snap->mNamedInts["which"],
            (Alternative*)getNamedElementType(snap, "alternative")
        );

        return;
    }

    if (snap->mKind == PyObjSnapshot::Kind::AlternativeMatcherType) {
        ((AlternativeMatcher*)snap->mType)->initializeDuringDeserialization(
            getNamedElementType(snap, "alternative")
        );

        return;
    }

    if (snap->mKind == PyObjSnapshot::Kind::AlternativeType) {
        std::vector<std::pair<std::string, NamedTuple*> > types;
        std::map<std::string, Function*> methods;
        std::vector<Type*> subtypesConcrete;

        for (long k = 0; k < snap->mElements.size() && k < snap->mNames.size(); k++) {
            Type* nt = typeFor(snap->mElements[k]);

            if (!nt->isNamedTuple()) {
                throw std::runtime_error("Corrupt PyObjSnapshot.Alternative");
            }

            types.push_back(
                std::make_pair(
                    snap->mNames[k],
                    (NamedTuple*)nt
                )
            );
        }

        getNamedBundle(snap, "alt_methods", methods);
        getNamedBundle(snap, "alt_subtypes", subtypesConcrete);

        ((Alternative*)snap->mType)->initializeDuringDeserialization(
            snap->mName,
            snap->mModuleName,
            types,
            methods,
            subtypesConcrete
        );

        return;
    }

    if (snap->mKind == PyObjSnapshot::Kind::ClassType) {
        Type* heldClass = getNamedElementType(snap, "held_class_type");

        if (!heldClass->isHeldClass()) {
            throw std::runtime_error("Corrupt PyObjSnapshot.Class");
        }

        ((Class*)snap->mType)->initializeDuringDeserialization(
            snap->mName,
            (HeldClass*)heldClass
        );
        return;
    }

    if (snap->mKind == PyObjSnapshot::Kind::HeldClassType) {
        std::vector<HeldClass*> bases;
        std::vector<MemberDefinition> members;
        std::map<std::string, Function*> memberFunctions;
        std::map<std::string, Function*> staticFunctions;
        std::map<std::string, Function*> propertyFunctions;
        std::map<std::string, PyObject*> classMembers;
        std::map<std::string, Function*> classMethods;

        getNamedBundle(snap, "cls_bases", bases);
        getNamedBundle(snap, "cls_members", members);
        getNamedBundle(snap, "cls_methods", memberFunctions);
        getNamedBundle(snap, "cls_staticmethods", staticFunctions);
        getNamedBundle(snap, "cls_properties", propertyFunctions);
        getNamedBundle(snap, "cls_classmembers", classMembers);
        getNamedBundle(snap, "cls_classmethods", classMethods);

        std::vector<MemberDefinition> own_members;
        std::map<std::string, Function*> own_memberFunctions;
        std::map<std::string, Function*> own_staticFunctions;
        std::map<std::string, Function*> own_propertyFunctions;
        std::map<std::string, PyObject*> own_classMembers;
        std::map<std::string, Function*> own_classMethods;

        getNamedBundle(snap, "cls_own_members", own_members);
        getNamedBundle(snap, "cls_own_methods", own_memberFunctions);
        getNamedBundle(snap, "cls_own_staticmethods", own_staticFunctions);
        getNamedBundle(snap, "cls_own_properties", own_propertyFunctions);
        getNamedBundle(snap, "cls_own_classmembers", own_classMembers);
        getNamedBundle(snap, "cls_own_classmethods", own_classMethods);

        Type* clsType = getNamedElementType(snap, "cls_type");
        if (!clsType->isClass()) {
            throw std::runtime_error("Corrupt PyObjSnapshot.HeldClass");
        }

        ((HeldClass*)snap->mType)->initializeDuringDeserialization(
            snap->mName,
            bases,
            snap->mNamedInts["cls_is_final"],
            members,
            memberFunctions,
            staticFunctions,
            propertyFunctions,
            classMembers,
            classMethods,

            own_members,
            own_memberFunctions,
            own_staticFunctions,
            own_propertyFunctions,
            own_classMembers,
            own_classMethods,

            (Class*)clsType
        );
        return;
    }

    if (snap->mKind == PyObjSnapshot::Kind::FunctionType) {
        std::string name = snap->mName;
        std::string qualname = snap->mQualname;
        std::string moduleName = snap->mModuleName;
        std::vector<FunctionOverload> overloads;
        Type* closureType = getNamedElementType(snap, "closure_type");
        bool isEntrypoint = snap->mNamedInts["is_nocompile"];
        bool isNocompile = snap->mNamedInts["is_nocompile"];

        getNamedBundle(snap, "func_overloads", overloads);

        ((Function*)snap->mType)->initializeDuringDeserialization(
            name,
            qualname,
            moduleName,
            overloads,
            closureType,
            isEntrypoint,
            isNocompile
        );
        return;
    }

    throw std::runtime_error("Can't rehydrate a PyObjSnapshot of kind " + snap->kindAsString());
}

void PyObjRehydrator::rehydrate(PyObjSnapshot* obj) {
    if (mSnapshots.find(obj) == mSnapshots.end()) {
        throw std::runtime_error(
            "PyObjRehydrator walk didn't pick up snap of type "
            + obj->kindAsString()
        );
    }

    PyObjSnapshot::Kind kind = obj->mKind;
    PyObject*& pyObject(obj->mPyObject);

    // already rehydrated
    if (pyObject) {
        return;
    }

    static PyObject* sysModuleModules = staticPythonInstance("sys", "modules");

    if (kind == PyObjSnapshot::Kind::PythonObjectOfTypeType) {
        // PythonObjectOfType is special because its Type* doesn't have a builtin
        // representation of the Type as a PythonTypeObject since we just use the real thing.
        // this is similar to how we have a type OneOf(T1, T2,...) but never see an instance
        // of it.
        pyObject = getNamedElementPyobj(obj, "element_type", false);
    } else
    if (kind == PyObjSnapshot::Kind::ArbitraryPyObject) {
        throw std::runtime_error("Corrupt PyObjSnapshot.ArbitraryPyObject: missing pyObject");
    } else
if (kind == PyObjSnapshot::Kind::PrimitiveType) {
        pyObject = (PyObject*)PyInstance::typeObj(obj->mType);
    } else
if (kind == PyObjSnapshot::Kind::String) {
        pyObject = PyUnicode_FromString(obj->mStringValue.c_str());
    } else
if (kind == PyObjSnapshot::Kind::PrimitiveInstance) {
        pyObject = PyInstance::extractPythonObject(obj->mInstance);
    } else
if (kind == PyObjSnapshot::Kind::NamedPyObject) {
        PyObjectStealer nameAsStr(PyUnicode_FromString(obj->mModuleName.c_str()));
        PyObjectStealer moduleObject(
            PyObject_GetItem(sysModuleModules, nameAsStr)
        );

        if (!moduleObject) {
            PyErr_Clear();
            // TODO: should we be importing here? that seems dangerous...
            throw std::runtime_error("Somehow module " + obj->mModuleName + " is not loaded!");
        }
        PyObjectStealer inst(PyObject_GetAttrString(moduleObject, obj->mName.c_str()));
        if (!inst) {
            PyErr_Clear();

            // TODO: should we be importing here? that seems dangerous...
            throw std::runtime_error("Somehow module " + obj->mModuleName + " is missing member " + obj->mName);
        }

        pyObject = incref(inst);
    } else
if (kind == PyObjSnapshot::Kind::PyDict || kind == PyObjSnapshot::Kind::PyClassDict) {
        pyObject = PyDict_New();
    } else
if (kind == PyObjSnapshot::Kind::PyList) {
        pyObject = PyList_New(0);

        for (long k = 0; k < obj->mElements.size(); k++) {
            PyList_Append(pyObject, pyobjFor(obj->mElements[k]));
        }
    } else
if (kind == PyObjSnapshot::Kind::PyTuple) {
        pyObject = PyTuple_New(obj->mElements.size());

        // first initialize it in case we throw somehow
        for (long k = 0; k < obj->mElements.size(); k++) {
            PyTuple_SetItem(pyObject, k, incref(Py_None));
        }

        for (long k = 0; k < obj->mElements.size(); k++) {
            PyTuple_SetItem(pyObject, k, incref(pyobjFor(obj->mElements[k])));
        }
    } else
if (kind == PyObjSnapshot::Kind::PySet) {
        pyObject = PySet_New(nullptr);

        for (long k = 0; k < obj->mElements.size(); k++) {
            PySet_Add(pyObject, incref(pyobjFor(obj->mElements[k])));
        }
    } else
if (kind == PyObjSnapshot::Kind::PyClass) {
        PyObjectStealer argTup(PyTuple_New(3));
        PyTuple_SetItem(argTup, 0, PyUnicode_FromString(obj->mName.c_str()));
        PyTuple_SetItem(argTup, 1, incref(getNamedElementPyobj(obj, "cls_bases")));
        PyTuple_SetItem(argTup, 2, incref(getNamedElementPyobj(obj, "cls_dict")));

        pyObject = PyType_Type.tp_new(&PyType_Type, argTup, nullptr);

        if (!pyObject) {
            throw PythonExceptionSet();
        }
    } else
if (kind == PyObjSnapshot::Kind::PyModule) {
        PyObjectStealer nameAsStr(PyUnicode_FromString(obj->mName.c_str()));
        PyObjectStealer moduleObject(
            PyObject_GetItem(sysModuleModules, nameAsStr)
        );

        if (!moduleObject) {
            PyErr_Clear();
            // TODO: should we be importing here? that seems dangerous...
            throw std::runtime_error("Somehow module " + obj->mName + " is not loaded!");
        }

        pyObject = incref(moduleObject);
    } else
if (kind == PyObjSnapshot::Kind::PyModuleDict) {
        pyObject = PyObject_GenericGetDict(getNamedElementPyobj(obj, "module_dict_of"), nullptr);
        if (!pyObject) {
            throw PythonExceptionSet();
        }
    } else
if (kind == PyObjSnapshot::Kind::PyCell) {
        pyObject = PyCell_New(nullptr);
    } else
if (kind == PyObjSnapshot::Kind::PyObject) {
        PyObject* t = getNamedElementPyobj(obj, "inst_type");
        PyObject* d = getNamedElementPyobj(obj, "inst_dict");

        static PyObject* emptyTuple = PyTuple_Pack(0);

        pyObject = ((PyTypeObject*)t)->tp_new(
            ((PyTypeObject*)t),
            emptyTuple,
            nullptr
        );

        if (PyObject_GenericSetDict(pyObject, d, nullptr)) {
            throw PythonExceptionSet();
        }
    } else
if (kind == PyObjSnapshot::Kind::PyFunction) {
        pyObject = PyFunction_New(
            getNamedElementPyobj(obj, "func_code"),
            getNamedElementPyobj(obj, "func_globals")
        );

        if (obj->mNamedElements.find("func_closure") != obj->mNamedElements.end()) {
            PyFunction_SetClosure(
                pyObject,
                getNamedElementPyobj(obj, "func_closure")
            );
        }

        if (obj->mNamedElements.find("func_annotations") != obj->mNamedElements.end()) {
            PyFunction_SetAnnotations(
                pyObject,
                getNamedElementPyobj(obj, "func_annotations")
            );
        }

        if (obj->mNamedElements.find("func_defaults") != obj->mNamedElements.end()) {
            PyFunction_SetAnnotations(
                pyObject,
                getNamedElementPyobj(obj, "func_defaults")
            );
        }

        if (obj->mNamedElements.find("func_qualname") != obj->mNamedElements.end()) {
            PyObject_SetAttrString(
                pyObject,
                "__qualname__",
                getNamedElementPyobj(obj, "func_qualname")
            );
        }

        if (obj->mNamedElements.find("func_kwdefaults") != obj->mNamedElements.end()) {
            PyObject_SetAttrString(
                pyObject,
                "__kwdefaults__",
                getNamedElementPyobj(obj, "func_kwdefaults")
            );
        }

        if (obj->mNamedElements.find("func_name") != obj->mNamedElements.end()) {
            PyObject_SetAttrString(
                pyObject,
                "__name__",
                getNamedElementPyobj(obj, "func_name")
            );
        }
    } else
if (kind == PyObjSnapshot::Kind::PyCodeObject) {
#if PY_MINOR_VERSION < 8
        pyObject = (PyObject*)PyCode_New(
#else
        pyObject = (PyObject*)PyCode_NewWithPosOnlyArgs(
#endif
            obj->mNamedInts["co_argcount"],
#if PY_MINOR_VERSION >= 8
            obj->mNamedInts["co_posonlyargcount"],
#endif
            obj->mNamedInts["co_kwonlyargcount"],
            obj->mNamedInts["co_nlocals"],
            obj->mNamedInts["co_stacksize"],
            obj->mNamedInts["co_flags"],
            getNamedElementPyobj(obj, "co_code"),
            getNamedElementPyobj(obj, "co_consts"),
            getNamedElementPyobj(obj, "co_names"),
            getNamedElementPyobj(obj, "co_varnames"),
            getNamedElementPyobj(obj, "co_freevars"),
            getNamedElementPyobj(obj, "co_cellvars"),
            getNamedElementPyobj(obj, "co_filename"),
            getNamedElementPyobj(obj, "co_name"),
            obj->mNamedInts["co_firstlineno"],
            getNamedElementPyobj(obj,
#if PY_MINOR_VERSION >= 10
                "co_linetable"
#else
                "co_lnotab"
#endif
            )
        );
    } else
if (kind == PyObjSnapshot::Kind::PyStaticMethod) {
        pyObject = PyStaticMethod_New(Py_None);

        JustLikeAClassOrStaticmethod* method = (JustLikeAClassOrStaticmethod*)pyObject;
        decref(method->cm_callable);
        method->cm_callable = incref(getNamedElementPyobj(obj, "meth_func"));
    } else
if (kind == PyObjSnapshot::Kind::PyClassMethod) {
        pyObject = PyClassMethod_New(Py_None);

        JustLikeAClassOrStaticmethod* method = (JustLikeAClassOrStaticmethod*)pyObject;
        decref(method->cm_callable);
        method->cm_callable = incref(getNamedElementPyobj(obj, "meth_func"));
    } else
if (kind == PyObjSnapshot::Kind::PyProperty) {
        static PyObject* nones = PyTuple_Pack(3, Py_None, Py_None, Py_None);

        pyObject = PyObject_CallObject((PyObject*)&PyProperty_Type, nones);

        JustLikeAPropertyObject* dest = (JustLikeAPropertyObject*)pyObject;

        decref(dest->prop_get);
        decref(dest->prop_set);
        decref(dest->prop_del);
        decref(dest->prop_doc);

        dest->prop_get = nullptr;
        dest->prop_set = nullptr;
        dest->prop_del = nullptr;
        dest->prop_doc = nullptr;

        #if PY_MINOR_VERSION >= 10
        decref(dest->prop_name);
        dest->prop_name = nullptr;
        #endif

        dest->prop_get = incref(getNamedElementPyobj(obj, "prop_get", true));
        dest->prop_set = incref(getNamedElementPyobj(obj, "prop_set", true));
        dest->prop_del = incref(getNamedElementPyobj(obj, "prop_del", true));
        dest->prop_doc = incref(getNamedElementPyobj(obj, "prop_doc", true));

        #if PY_MINOR_VERSION >= 10
        decref(dest->prop_name);
        dest->prop_name = incref(getNamedElementPyobj(obj, "prop_name", true));
        #endif
    } else
if (kind == PyObjSnapshot::Kind::PyBoundMethod) {
        pyObject = PyMethod_New(Py_None, Py_None);
        PyMethodObject* method = (PyMethodObject*)pyObject;
        decref(method->im_func);
        decref(method->im_self);
        method->im_func = nullptr;
        method->im_self = nullptr;

        method->im_func = incref(getNamedElementPyobj(obj, "meth_func"));
        method->im_self = incref(getNamedElementPyobj(obj, "meth_self"));
    } else {
        if (obj->willBeATpType()) {
            asm("int3");
            throw std::runtime_error("Expected PyObjSnapshot of kind " + obj->kindAsString() + " to already have a pyobj.");
        }

        throw std::runtime_error(
            "Can't make a python object representation for a PyObjSnapshot of kind " +
            obj->kindAsString()
        );
    }
}

void PyObjRehydrator::finalizeRehydration(PyObjSnapshot* obj) {
    if (obj->mKind == PyObjSnapshot::Kind::PyClass) {
        if (obj->mNamedElements.find("cls_dict") == obj->mNamedElements.end()) {
            throw std::runtime_error("Corrupt PyClass - no cls_dict");
        }

        PyObjSnapshot* clsDictPyObj = obj->mNamedElements["cls_dict"];

        for (long k = 0; k < clsDictPyObj->elements().size()
                        && k < clsDictPyObj->keys().size(); k++) {
            if (clsDictPyObj->keys()[k]->isString()) {
                PyObject_SetAttrString(
                    obj->mPyObject,
                    clsDictPyObj->keys()[k]->getStringValue().c_str(),
                    clsDictPyObj->elements()[k]->getPyObj()
                );
            }
        }
    } else
    if (obj->mKind == PyObjSnapshot::Kind::PyDict || obj->mKind == PyObjSnapshot::Kind::PyClassDict) {
        for (long k = 0; k < obj->mElements.size()
                && k < obj->mKeys.size(); k++) {
            PyDict_SetItem(
                obj->mPyObject,
                obj->mKeys[k]->getPyObj(),
                obj->mElements[k]->getPyObj()
            );
        }
    } else
    if (obj->mKind == PyObjSnapshot::Kind::PyCell) {
        auto it = obj->mNamedElements.find("cell_contents");
        if (it != obj->mNamedElements.end()) {
            PyCell_Set(
                obj->mPyObject,
                it->second->getPyObj()
            );
        }
    }
}
