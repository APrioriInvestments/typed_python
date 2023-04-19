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


enum class VisibilityType {
    // compute the identity of the object as far as the interpreter is concerned,
    // excluding any notion of pointer identity.  This is used to memoize Type instances.
    // We don't look into module-level members when computing this value, so this is valid
    // as soon as a type object is constructed
    Identity = 1,
    // compute the identity of the object as far as the compiler itself is concerned.
    // this is stronger than the notion of the Identity hash because the
    Compiler = 2,
};

inline std::string visibilityTypeToStr(VisibilityType v) {
  if (v == VisibilityType::Identity) {
    return "Identity";
  }
  if (v == VisibilityType::Compiler) {
    return "Compiler";
  }
  return "Unknown";
}
