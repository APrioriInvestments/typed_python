#   Copyright 2017 Braxton Mckee
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

import nativepython.runtime as runtime
import nativepython.util as util
import nativepython.type_model as type_model

import unittest
import ast
import time

class Counter(type_model.cls):
    def __types__(cls):
        cls.types.count = int

    def __init__(self):
        self.count = 0

class CounterInc(type_model.cls):
    def __types__(cls):
        cls.types.target = Counter.pointer

    def __init__(self, target):
        self.target = target
        if self.target:
            self.target.count += 1

    def __copy_constructor__(self, other):
        self.target = other.target
        if self.target:
            self.target.count += 1

    def __assign__(self, other):
        if self.target:
            self.target.count -= 1
        self.target = other.target
        if self.target:
            self.target.count += 1

    def __destructor__(self):
        if self.target:
            self.target.count -= 1
