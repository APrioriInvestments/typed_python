#   Copyright 2017-2019 typed_python Authors
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
from typed_python import Class, Type
from typed_python.compiler import native_ast
from typed_python.compiler.type_wrappers.wrapper import Wrapper


def cannotBeSubclass(t1, t2):
    """Determine if 't1' cannot be a subclass of 't2'.

    In particular, most typed python types do not admit subclassing, and the only way
    two types can have a common child is if they are both Class objects and one has
    no members.
    """
    # if one is a child of the other then this is obviously false
    if issubclass(t1, t2) or issubclass(t2, t1):
        return False

    # if either is a typed python type, but if neither is a Class object,
    # then its impossible for one to be a subclass of the other.
    if (
        (issubclass(t1, Type) or issubclass(t2, Type))
        and (not issubclass(t1, Class) or not issubclass(t2, Class))
    ):
        return True

    if issubclass(t1, Class) and issubclass(t2, Class) and t1.MemberTypes and t2.MemberTypes:
        return True

    return False


class IsinstanceWrapper(Wrapper):
    is_pod = True
    is_empty = False
    is_pass_by_ref = False

    def __init__(self):
        super().__init__(isinstance)

    def getNativeLayoutType(self):
        return native_ast.Type.Void()

    @Wrapper.unwrapOneOfAndValue
    def convert_call(self, context, expr, args, kwargs):
        if len(args) == 2 and not kwargs:
            instance = args[0]
            typeObj = args[1]
            instanceType = instance.convert_typeof()

            return instanceType.convert_issubclass(typeObj, False)

        return super().convert_call(context, expr, args, kwargs)
