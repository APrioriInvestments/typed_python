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

/*****
a .cpp file that includes all the other .cpp files associated
with the project implementation.

So much code overlaps between the various translation units
that there's no point splitting them up - its faster to just
compile the entire group all at once.
******/

#include "_types.cpp"
#include "_runtime.cpp"
#include "Instance.cpp"
#include "util.cpp"

#include "PyInstance.cpp"
#include "PyConstDictInstance.cpp"
#include "PyDictInstance.cpp"
#include "PyTupleOrListOfInstance.cpp"
#include "PyPointerToInstance.cpp"
#include "PyRefToInstance.cpp"
#include "PyCompositeTypeInstance.cpp"
#include "PyClassInstance.cpp"
#include "PyAlternativeInstance.cpp"
#include "PyFunctionInstance.cpp"
#include "PyBoundMethodInstance.cpp"
#include "PyGilState.cpp"
#include "PySetInstance.cpp"

#include "SubclassOfType.cpp"
#include "SetType.cpp"
#include "AlternativeType.cpp"
#include "BytesType.cpp"
#include "ClassType.cpp"
#include "CompositeType.cpp"
#include "ConcreteAlternativeType.cpp"
#include "DictType.cpp"
#include "ConstDictType.cpp"
#include "HeldClassType.cpp"
#include "OneOfType.cpp"
#include "PythonObjectOfTypeType.cpp"
#include "PythonSerializationContext.cpp"
#include "PythonSerializationContext_serialization.cpp"
#include "PythonSerializationContext_deserialization.cpp"
#include "PythonSubclassType.cpp"
#include "StringType.cpp"
#include "TupleOrListOfType.cpp"
#include "Type.cpp"
#include "FunctionType.cpp"

#include "SerializationBuffer.cpp"
#include "DeserializationBuffer.cpp"
#include "Sha1.cpp"
#include "MutuallyRecursiveTypeGroup.cpp"
#include "TypeOrPyobj.cpp"
#include "Memory.cpp"
#include "PySlab.cpp"
#include "PyModuleRepresentation.cpp"
#include "Slab.cpp"
#include "PyTemporaryReferenceTracer.cpp"

#include "lz4.c"
#include "lz4frame.c"
#include "lz4hc.c"
#include "xxhash.c"
