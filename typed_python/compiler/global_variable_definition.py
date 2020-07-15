#   Copyright 2020 typed_python Authors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from typed_python import Alternative


GlobalVariableMetadata = Alternative(
    "GlobalVariableMetadata",
    StringConstant=dict(value=str),
    BytesConstant=dict(value=bytes),
    IntegerConstant=dict(value=int),
    # a pointer to a PythonObjectOfType (e.g. our version, with
    # refcounts that are not in the GIL)
    PointerToPyObject=dict(value=object),
    # the raw ID of a global python object, like a builtin or
    # an exception
    IdOfPyObject=dict(value=object),
    # a pointer to the Type* in the underlying
    RawTypePointer=dict(value=type),
    PointerToTypedPythonObject=dict(value=object, type=type),
    # a typed python object at module scope (and therefore truly global)
    PointerToTypedPythonObjectAsMemberOfDict=dict(
        sourceDict=object, name=str, type=type
    ),
    ClassVtable=dict(value=type),
    ClassMethodDispatchSlot=dict(
        clsType=object,
        methodName=str,
        retType=object,
        argTupleType=object,
        kwargTupleType=object
    )
)


class GlobalVariableDefinition:
    """Representation for a single globally defined value.

    Each such value has a formal name (which should be unique across
    all possible compiled value sets, so usually its a hash), a type,
    and some metadata indicating to the calling context what its for.
    """
    def __init__(self, name, typ, metadata):
        """Initialize a GlobalVariableDefinition.

        Args:
            name - a string uniquely identifying the global variable
            typ - a native_ast type
            metadata - any 'value-like' python object we can use
                to identify the variable.
        """
        self.name = name
        self.type = typ
        self.metadata = metadata
