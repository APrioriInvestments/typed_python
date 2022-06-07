# Copyright 2017-2020 typed_python Authors
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

import unittest
from typed_python.lib.datetime.timezone_offset import is_tz_offset


class TestTimezoneOffset(unittest.TestCase):

    def test_is_tz_offset_valid(self):
        tz_offsets = [
            (-12, 0, 0),  # eastmost
            (14, 0, 0),   # westmost
            (10, 4, 4),   # random
        ]

        for tz_offset in tz_offsets:
            assert is_tz_offset(tz_offset[0], tz_offset[1]), tz_offset

    def test_is_tz_offset_invalid(self):
        tz_offsets = [
            (-13, 0, 0),  # out of range
            (-12, 1, 0),  # out of range
            (14, 1, 0),   # out of range
            (10, 60, 4),  # min > 59
            (10, -1, 4),  # min < 0
        ]

        for tz_offset in tz_offsets:
            assert not is_tz_offset(tz_offset[0], tz_offset[1]), tz_offset
