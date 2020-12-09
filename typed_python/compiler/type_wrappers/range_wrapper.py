#   Copyright 2017-2020 typed_python Authors
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

from typed_python.compiler.type_wrappers.wrapper import Wrapper
from typed_python import Class, Final, Member, PointerTo, NamedTuple
from typed_python import pointerTo
import typed_python.compiler.native_ast as native_ast


class Range(Class, Final, __name__='range'):
    start = Member(int)
    stop = Member(int)
    step = Member(int)

    def __repr__(self):
        if self.step == 1:
            return f"range({self.start}, {self.stop})"

        return f"range({self.start}, {self.stop}, {self.step})"

    def __init__(self, stop):
        self.start = 0
        self.stop = stop
        self.step = 1

    def __init__(self, start, stop):
        self.start = start
        self.stop = stop
        self.step = 1

    def __init__(self, start, stop, step):
        self.start = start
        self.stop = stop
        self.step = step

    def __iter__(self):
        return RangeIterator(start=self.start - self.step, stop=self.stop, step=self.step)


class RangeIterator(NamedTuple(start=int, stop=int, step=int)):
    def __fastnext__(self) -> PointerTo(int):
        startPtr = pointerTo(self).start
        stopPtr = pointerTo(self).stop
        stepPtr = pointerTo(self).step

        startPtr.set(startPtr.get() + stepPtr.get())

        if stepPtr.get() > 0:
            if startPtr.get() < stopPtr.get():
                return startPtr
        else:
            if startPtr.get() > stopPtr.get():
                return startPtr

        return PointerTo(int)()
