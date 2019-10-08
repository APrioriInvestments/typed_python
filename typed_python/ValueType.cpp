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

#include "AllTypes.hpp"

Type* Value::MakeInt64(int64_t i) {
    return Make(Instance::create(Int64::Make(), (instance_ptr)&i));
}

Type* Value::MakeFloat64(double i) {
    return Make(Instance::create(Float64::Make(), (instance_ptr)&i));
}

Type* Value::MakeBool(bool i) {
    return Make(Instance::create(Bool::Make(), (instance_ptr)&i));
}

Type* Value::MakeBytes(char* data, size_t count) {
    return Make(Instance::createAndInitialize(BytesType::Make(), [&](instance_ptr i) {
        BytesType::Make()->constructor(i, count, data);
    }));
}

Type* Value::MakeString(size_t bytesPerCodepoint, size_t count, char* data) {
    return Make(Instance::createAndInitialize(StringType::Make(), [&](instance_ptr i) {
        StringType::Make()->constructor(i, bytesPerCodepoint, count, data);
    }));
}

