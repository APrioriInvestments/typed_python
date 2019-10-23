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

from typed_python._types import Forward


class Module:
    """Module - a collection of typed_python types.

    A Module makes it easier to define a set of related recursive types,
    by automatically allocating Forward types for internal members.

    If you access a member of the module (and it's not frozen yet), you
    will implicitly allocate a new Forward type. If you assign to a member
    of the module and you allocated a Forward type for that member, assignment
    will define the Forward type. This makes it possible to easily make
    recursive class definitions.

    Module member names must start with a capital letter.

    Usage:
        m = Module("mymodule")

        # assign types directly
        m.AnInt = int

        # use it to define recursive types
        m.IntOrTupleOfSelf = OneOf(int, TupleOf(m.IntOrTupleOfSelf))

        # or use it to define mutually recursive classes
        @m.define
        class X:
            y = Member(OneOf(None, m.Y))

        @m.define
        class Y:
            x = Member(OneOf(None, m.X))

        # and finally freeze it so that you can't modify it.
        m.freeze()
    """
    def __init__(self, name):
        self._items = {}
        self.frozen = False
        self.name = name

    def __getattr__(self, name):
        if name[:1] == "_" or not name[:1].isupper():
            raise AttributeError(name)

        if name not in self._items:
            self._items[name] = Forward(self.name + "." + name)
        return self._items[name]

    def __setattr__(self, name, typ):
        if name in ['_items', 'frozen', 'name']:
            self.__dict__[name] = typ
            return

        if name[:1] == "_" or not name[:1].isupper():
            raise AttributeError(name)

        if self.frozen:
            raise Exception(f"Module {self.name} is frozen, so you can't define {name} in it.")
        if name not in self._items:
            self._items[name] = typ
        else:
            self._items[name] = self._items[name].define(typ)

    def define(self, cls):
        self.__setattr__(cls.__name__, cls)
        return self.__getattr__(cls.__name__)

    def freeze(self):
        for moduleMember, typ in self._items.items():
            if getattr(typ, "__typed_python_category__", "") == "Forward":
                raise Exception(f"{self.name}.{moduleMember} is not defined yet.")

        self.frozen = True
