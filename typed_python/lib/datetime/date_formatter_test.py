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
import time
from datetime import datetime, timedelta
from typed_python.lib.datetime.date_formatter import DateFormatter
import pytz


def get_datetimes_in_range(start, end, step):
    d = start
    dates = [start]

    while d < end:
        if step == 'days':
            d += timedelta(days=1)
        elif step == 'hours':
            d += timedelta(hours=1)
        elif step == 'minutes':
            d += timedelta(minutes=1)
        elif step == 'seconds':
            d += timedelta(seconds=1)
        else:
            raise ValueError('Unsupported step: ' + step)
        dates.append(d)
    return dates


def get_years_in_range(start, end):
    dates = []
    for i in range(start, end):
        dates.append(datetime(i, 1, 1, 0, 0, 0, 0, pytz.UTC))

    return dates


class TestDateFormatter(unittest.TestCase):
    def test_isoformat(self):
        seconds = get_datetimes_in_range(start=datetime(2020, 2, 29, 13, 17, 0, 0, pytz.UTC),
                                         end=datetime(2020, 2, 29, 13, 19, 0, 0, pytz.UTC),
                                         step='seconds')
        for second in seconds:
            assert DateFormatter.isoformat(datetime.timestamp(second), 0) == second.strftime(
                '%Y-%m-%dT%H:%M:%S'), second.strftime('%Y-%m-%dT%H:%M:%S')

    def test_format_directives(self):
        seconds = get_datetimes_in_range(start=datetime(2020, 2, 29, 13, 17, 0, 0, pytz.UTC),
                                         end=datetime(2020, 2, 29, 13, 19, 0, 0, pytz.UTC),
                                         step='seconds')
        for second in seconds:
            assert DateFormatter.format(datetime.timestamp(
                second), 0, '%Y-%m-%dT%H:%M:%S') == second.strftime('%Y-%m-%dT%H:%M:%S'), second.strftime('%Y-%m-%dT%H:%M:%S')

    def test_format_directive_a(self):
        days = get_datetimes_in_range(start=datetime(2019, 1, 1, 0, 0, 0, 0, pytz.UTC),
                                      end=datetime(2020, 1, 31, 0, 0, 0, 0, pytz.UTC),
                                      step='days')

        for day in days:
            assert DateFormatter.format(datetime.timestamp(day), 0, '%a') == day.strftime('%a'), day.strftime('%Y-%m-%d')

    def test_format_directive_A(self):
        days = get_datetimes_in_range(start=datetime(2019, 1, 1, 0, 0, 0, 0, pytz.UTC),
                                      end=datetime(2020, 1, 31, 0, 0, 0, 0, pytz.UTC),
                                      step='days')

        for day in days:
            assert DateFormatter.format(datetime.timestamp(day), 0, '%A') == day.strftime('%A'), day.strftime('%Y-%m-%d')

    def test_format_directive_w(self):
        days = get_datetimes_in_range(start=datetime(2019, 1, 1, 0, 0, 0, 0, pytz.UTC),
                                      end=datetime(2020, 1, 31, 0, 0, 0, 0, pytz.UTC),
                                      step='days')

        for day in days:
            assert DateFormatter.format(datetime.timestamp(day), 0, '%w') == day.strftime('%w'), day.strftime('%Y-%m-%d')

    def test_format_directive_d(self):
        days = get_datetimes_in_range(start=datetime(2019, 1, 1, 0, 0, 0, 0, pytz.UTC),
                                      end=datetime(2020, 1, 31, 0, 0, 0, 0, pytz.UTC),
                                      step='days')

        for day in days:
            assert DateFormatter.format(datetime.timestamp(day), 0, '%d') == day.strftime('%d'), day.strftime('%Y-%m-%d')

    def test_format_directive_b(self):
        days = get_datetimes_in_range(start=datetime(2019, 1, 1, 0, 0, 0, 0, pytz.UTC),
                                      end=datetime(2020, 1, 31, 0, 0, 0, 0, pytz.UTC),
                                      step='days')

        for day in days:
            assert DateFormatter.format(datetime.timestamp(day), 0, '%b') == day.strftime('%b'), day.strftime('%Y-%m-%d')

    def test_format_directive_B(self):
        days = get_datetimes_in_range(start=datetime(2019, 1, 1, 0, 0, 0, 0, pytz.UTC),
                                      end=datetime(2020, 1, 31, 0, 0, 0, 0, pytz.UTC),
                                      step='days')

        for day in days:
            assert DateFormatter.format(datetime.timestamp(day), 0, '%B') == day.strftime('%B'), day.strftime('%Y-%m-%d')

    def test_format_directive_m(self):
        days = get_datetimes_in_range(start=datetime(2019, 1, 1, 0, 0, 0, 0, pytz.UTC),
                                      end=datetime(2020, 1, 31, 0, 0, 0, 0, pytz.UTC),
                                      step='days')

        for day in days:
            assert DateFormatter.format(datetime.timestamp(day), 0, '%m') == day.strftime('%m'), day.strftime('%Y-%m-%d')

    def test_format_directive_y(self):
        years = get_years_in_range(1999, 2022)
        for year in years:
            assert DateFormatter.format(datetime.timestamp(year), 0, '%y') == year.strftime('%y'), year.strftime('%Y-%m-%d')

    def test_format_directive_H(self):
        minutes = get_datetimes_in_range(start=datetime(2020, 2, 29, 0, 17, 0, 0, pytz.UTC),
                                         end=datetime(2020, 2, 29, 23, 59, 0, 0, pytz.UTC),
                                         step='minutes')
        for minute in minutes:
            assert DateFormatter.format(datetime.timestamp(minute), 0, '%H') == minute.strftime(
                '%H'), minute.strftime('%Y-%m-%dT%H:%M:%S')

    def test_format_directive_I(self):
        minutes = get_datetimes_in_range(start=datetime(2020, 2, 29, 0, 17, 0, 0, pytz.UTC),
                                         end=datetime(2020, 2, 29, 23, 59, 0, 0, pytz.UTC),
                                         step='minutes')
        for minute in minutes:
            assert DateFormatter.format(datetime.timestamp(minute), 0, '%I') == minute.strftime('%I'), minute.strftime('%Y-%m-%dT%H:%M:%S')

        unixtime = time.time()
        dt = datetime.fromtimestamp(unixtime)
        assert dt.strftime('%I') == DateFormatter.format(unixtime, time.localtime().tm_gmtoff, '%I')

    def test_format_directive_p(self):
        minutes = get_datetimes_in_range(start=datetime(2020, 2, 29, 0, 17, 0, 0, pytz.UTC),
                                         end=datetime(2020, 2, 29, 23, 59, 0, 0, pytz.UTC),
                                         step='minutes')
        for minute in minutes:
            assert DateFormatter.format(datetime.timestamp(minute), 0, '%p') == minute.strftime('%p'), minute.strftime('%Y-%m-%dT%H:%M:%S')

    def test_format_directive_M(self):
        minutes = get_datetimes_in_range(start=datetime(2020, 2, 29, 10, 17, 0, 0, pytz.UTC),
                                         end=datetime(2020, 2, 29, 12, 19, 0, 0, pytz.UTC),
                                         step='minutes')
        for minute in minutes:
            assert DateFormatter.format(datetime.timestamp(minute), 0, '%M') == minute.strftime('%M'), minute.strftime('%Y-%m-%dT%H:%M:%S')

    def test_format_directive_S(self):
        seconds = get_datetimes_in_range(start=datetime(2020, 2, 29, 13, 17, 0, 0, pytz.UTC),
                                         end=datetime(2020, 2, 29, 13, 19, 0, 0, pytz.UTC),
                                         step='seconds')
        for second in seconds:
            assert DateFormatter.format(datetime.timestamp(second), 0, '%S') == second.strftime('%S'), second.strftime('%Y-%m-%dT%H:%M:%S')

    def test_format_directive_Z(self):
        days = get_datetimes_in_range(start=datetime(2019, 1, 1, 0, 0, 0, 0, pytz.UTC),
                                      end=datetime(2020, 1, 31, 0, 0, 0, 0, pytz.UTC),
                                      step='days')

        for day in days:
            assert DateFormatter.format(datetime.timestamp(day), 0, '%Z') == 'UTC', day.strftime('%Y-%m-%d')

    def test_format_directive_z(self):
        days = get_datetimes_in_range(start=datetime(2019, 1, 1, 0, 0, 0, 0, pytz.UTC),
                                      end=datetime(2020, 1, 31, 0, 0, 0, 0, pytz.UTC),
                                      step='days')

        for day in days:
            assert DateFormatter.format(datetime.timestamp(day), 0, '%z') == '+0000', day.strftime('%Y-%m-%d')

    def test_format_directive_C(self):
        years = get_years_in_range(1999, 2022)

        for year in years:
            assert DateFormatter.format(datetime.timestamp(year), 0, '%C') == year.strftime('%C'), year.strftime('%Y')

    def test_format_directive_Y(self):
        years = get_years_in_range(1999, 2022)
        for year in years:
            assert DateFormatter.format(datetime.timestamp(year), 0, '%Y') == year.strftime('%Y'), year.strftime('%Y-%m-%d')

    def test_format_directive_u(self):
        days = get_datetimes_in_range(start=datetime(2019, 1, 1, 0, 0, 0, 0, pytz.UTC),
                                      end=datetime(2020, 1, 31, 0, 0, 0, 0, pytz.UTC),
                                      step='days')

        for day in days:
            assert DateFormatter.format(datetime.timestamp(day), 0, '%u') == day.strftime('%u'), day.strftime('%Y-%m-%d')

    def test_format_directive_percent(self):
        days = get_datetimes_in_range(start=datetime(2019, 1, 1, 0, 0, 0, 0, pytz.UTC),
                                      end=datetime(2020, 1, 31, 0, 0, 0, 0, pytz.UTC),
                                      step='days')

        for day in days:
            assert DateFormatter.format(datetime.timestamp(day), 0, '%%') == day.strftime('%%'), day.strftime('%Y-%m-%d')

    def test_format_directive_doy(self):
        days = get_datetimes_in_range(start=datetime(2019, 1, 1, 0, 0, 0, 0, pytz.UTC),
                                      end=datetime(2020, 1, 31, 0, 0, 0, 0, pytz.UTC),
                                      step='days')

        for day in days:
            assert DateFormatter.format(datetime.timestamp(day), 0, '%j') == day.strftime('%j'), day.strftime('%Y-%m-%d')
