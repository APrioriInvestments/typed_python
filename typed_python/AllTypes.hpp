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

//the pattern of using 'check' to explicitly enumerate all of
//our subclasses by TypeCategory requires that if you include
//want to use any Type, you need to know about all of them.
#include "Type.hpp"
#include "TypeDetails.hpp"
#include "RegisterTypes.hpp"
#include "ForwardType.hpp"
#include "OneOfType.hpp"
#include "CompositeType.hpp"
#include "PointerToType.hpp"
#include "RefToType.hpp"
#include "TupleOrListOfType.hpp"
#include "ConstDictType.hpp"
#include "DictType.hpp"
#include "NoneType.hpp"
#include "Instance.hpp"
#include "StringType.hpp"
#include "BytesType.hpp"
#include "ValueType.hpp"
#include "AlternativeType.hpp"
#include "ConcreteAlternativeType.hpp"
#include "PythonSubclassType.hpp"
#include "PythonObjectOfTypeType.hpp"
#include "FunctionType.hpp"
#include "HeldClassType.hpp"
#include "ClassType.hpp"
#include "BoundMethodType.hpp"
#include "AlternativeMatcherType.hpp"
#include "EmbeddedMessageType.hpp"
#include "SetType.hpp"
#include "PyCellType.hpp"
#include "TypedCellType.hpp"
#include "SubclassOfType.hpp"
#include "direct_types/Bytes.hpp"
