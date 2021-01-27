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

from typed_python import NamedTuple, Class, Final, PointerTo
from typed_python import pointerTo


class Range(Class, Final, __name__='range'):
    def __repr__(self):
        return "<class 'range'>"

    def __str__(self):
        return "<class 'range'>"

    def __call__(self, stop):
        return RangeCls(start=0, stop=stop, step=1)

    def __call__(self, start, stop):  # noqa
        return RangeCls(start=start, stop=stop, step=1)

    def __call__(self, start, stop, step):  # noqa
        return RangeCls(start=start, stop=stop, step=step)


range = Range()


class RangeCls(NamedTuple(start=int, stop=int, step=int)):
    def __str__(self):
        return repr(self)

    def __repr__(self):
        if self.step == 1:
            return f"range({self.start}, {self.stop})"

        return f"range({self.start}, {self.stop}, {self.step})"

    def __iter__(self):
        return RangeIterator(start=self.start - self.step, stop=self.stop, step=self.step)

    def __typed_python_int_iter_size__(self):
        if self.step == 1:
            return self.stop - self.start

        if self.step > 0:
            return ((self.stop - self.start - 1) // self.step) + 1
        else:
            return ((self.start - self.stop - 1) // (-self.step)) + 1

    def __typed_python_int_iter_value__(self, x):
        return self.start + self.step * x


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
