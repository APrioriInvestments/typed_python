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

#include "AllTypes.hpp"

Value::Value() :
    Type(TypeCategory::catValue),
    mValueAsPyobj(nullptr)
{
    m_size = 0;
    m_is_default_constructible = true;
}


Value::Value(const Instance& instance) :
    Type(TypeCategory::catValue),
    mInstance(instance),
    mValueAsPyobj(nullptr)
{
    m_size = 0;
    m_is_default_constructible = true;

    mValueAsPyobj = PyInstance::extractPythonObject(mInstance);
    m_is_forward_defined = true;
}

Type* Value::Make(Instance i) {
    PyEnsureGilAcquired getTheGil;

    // if its a forward declared Type then we need to behave differently
    if (i.type()->isPythonObjectOfType() &&
        PyType_Check(
            PyObjectHandleTypeBase::getPyObj(i.data())
        )
    ) {
        PyObject* obj = PyObjectHandleTypeBase::getPyObj(i.data());
        Type* t = PyInstance::extractTypeFrom((PyTypeObject*)obj, true);

        if (t) {
            static std::map<Type*, Value*> typeMemo;

            auto it = typeMemo.find(t);

            if (it != typeMemo.end()) {
                return it->second;
            }

            if (t->isForwardDefined()) {
                typeMemo[t] = new Value(i);
            } else {
                typeMemo[t] = (Value*)(new Value(i))->forwardResolvesTo();
            }

            return typeMemo[t];
        }
    }

    static std::map<Instance, Value*> instanceMemo;

    auto it = instanceMemo.find(i);

    if (it == instanceMemo.end()) {
        Value* resolved = (Value*)((new Value(i))->forwardResolvesTo());
        it = instanceMemo.insert(std::make_pair(i, resolved)).first;
    }

    return it->second;
}

std::string Value::computeRecursiveNameConcrete(TypeStack& typeStack)
{
    if (
        mInstance.type()->isPythonObjectOfType() &&
        PyType_Check(
            PyObjectHandleTypeBase::getPyObj(mInstance.data())
        )
    ) {
        PyObject* obj = PyObjectHandleTypeBase::getPyObj(mInstance.data());

        Type* t = PyInstance::extractTypeFrom((PyTypeObject*)obj, true);

        std::string name;
        if (t) {
            name = t->computeRecursiveName(typeStack);
        } else {
            name = ((PyTypeObject*)obj)->tp_name;
        }

        auto ix = name.rfind('.');

        if (ix == std::string::npos) {
            ix = 0;
        } else {
            ix += 1;
        }

        return "Value(" + name.substr(ix) + ")";
    } else {
        return mInstance.repr();
    }
}

void Value::updateInternalTypePointersConcrete(
    const std::map<Type*, Type*>& groupMap
) {
    Type* ownType = mInstance.extractType(true);

    auto it = groupMap.find(ownType);

    if (it != groupMap.end()) {
        mInstance = Instance::create(
            (PyObject*)PyInstance::typeObj(it->second)
        );
        mValueAsPyobj = PyInstance::extractPythonObject(mInstance);
    }
}

void Value::initializeFromConcrete(Type* forwardDefinitionOfSelf) {
    mInstance = ((Value*)forwardDefinitionOfSelf)->mInstance.clone();
    mValueAsPyobj = PyInstance::extractPythonObject(mInstance);
}
