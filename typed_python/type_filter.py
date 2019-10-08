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


class TypeFilterBase(type):
    def __instancecheck__(self, other):
        if not isinstance(other, self.base_type):
            return False

        try:
            if not self.filter_function(other):
                return False
        except Exception:
            return False

        return True


def TypeFilter(base_type, filter_function):
    """TypeFilter(base_type, filter_function)

    Produce a 'type object' that can be used in typed python to filter objects by
    arbitrary criteria.
    """
    class TypeFilter(metaclass=TypeFilterBase):
        pass

    TypeFilter.base_type = base_type
    TypeFilter.filter_function = filter_function

    return TypeFilter
