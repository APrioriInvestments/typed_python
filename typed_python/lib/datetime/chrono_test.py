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
from typed_python.lib.datetime.chrono import Chrono


class TestChrono(unittest.TestCase):
    def test_is_leap_year_valid(self):
        leap_years = [
            2000,
            2004,
            2008,
            2012,
            2016,
            2020,
            2024,
            2028,
            2032,
            2036,
            2040,
            2044,
            2048,
        ]

        for year in leap_years:
            assert Chrono.is_leap_year(year), year

    def test_is_leap_year_invalid(self):
        not_leap_years = [1700, 1800, 1900, 1997, 1999, 2100, 2022]

        for year in not_leap_years:
            assert not Chrono.is_leap_year(year), year

    def test_is_date_valid(self):
        # y, m, d
        dates = [(1997, 1, 1), (2020, 2, 29)]  # random date  # Feb 29 on leap year

        for date in dates:
            assert Chrono.is_valid_date(date[0], date[1], date[2]), date

    def test_is_date_invalid(self):
        # y, m, d
        dates = [
            (1997, 0, 1),  # Month < 1
            (1997, 13, 1),  # Month > 12
            (1997, 1, 0),  # Day < 1
            (1997, 1, 32),  # Day > 31 in Jan
            (1997, 2, 29),  # Day > 28 in non-leap-year Feb,
            (2100, 2, 29),  # Day > 28 in non-leap-year Feb,
            (1997, 0, 25),  # Month < 1
            (2020, 2, 30),  # Day > 29 in Feb (leap year)
            (2020, 4, 31),  # Day > 30 in Apr (leap year)
            (2020, 6, 31),  # Day > 30 in June (leap year)
            (2020, 9, 31),  # Day > 30 in Sept (leap year)
            (2020, 11, 31),  # Day > 30 in Nov (leap year)
        ]

        for date in dates:
            assert not Chrono.is_valid_date(date[0], date[1], date[2]), date

    def test_is_time_valid(self):
        # h, m, s
        times = [
            (0, 0, 0),  # 00:00:00
            (24, 0, 0),  # 24:00:00
            (1, 1, 1),  # random time
            (12, 59, 59),  # random time
        ]
        for time in times:
            assert Chrono.is_valid_time(time[0], time[1], time[2]), time

    def test_is_time_invalid(self):
        # h, m, s
        times = [
            (24, 1, 0),  # m and s must be 0 if hour is 24
            (25, 0, 0),  # hour greater than 24
            (-1, 0, 0),  # hour less than 0
            (1, 0, -1),  # second < 1
            (1, -1, 0),  # min < 1
            (1, 0, 60),  # second > 59
            (1, 60, 0),  # min > 59
        ]
        for time in times:
            assert not Chrono.is_valid_time(time[0], time[1], time[2]), time

    def test_days_from_civil(self):
        days = Chrono.days_from_civil(1999, 2, 15)
        res = Chrono.civil_from_days(days)
        assert res.year == 1999
        assert res.month == 2
        assert res.day == 15

        days = Chrono.days_from_civil(2022, 12, 23)
        res = Chrono.civil_from_days(days)
        assert res.year == 2022
        assert res.month == 12
        assert res.day == 23
