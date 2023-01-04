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
from typed_python.lib.datetime.DateTime import UTC, FixedOffsetTimezone
import pytz


def get_datetimes_in_range(start, end, step):
    d = start
    dates = [start]

    while d < end:
        if step == "days":
            d += timedelta(days=1)
        elif step == "hours":
            d += timedelta(hours=1)
        elif step == "minutes":
            d += timedelta(minutes=1)
        elif step == "seconds":
            d += timedelta(seconds=1)
        else:
            raise ValueError("Unsupported step: " + step)
        dates.append(d)
    return dates


def get_years_in_range(start, end):
    dates = []
    for i in range(start, end):
        dates.append(datetime(i, 1, 1, 0, 0, 0, 0, pytz.UTC))

    return dates


class TestDateFormatter(unittest.TestCase):
    def test_isoformat(self):
        seconds = get_datetimes_in_range(
            start=datetime(2020, 2, 29, 13, 17, 0, 0, pytz.UTC),
            end=datetime(2020, 2, 29, 13, 19, 0, 0, pytz.UTC),
            step="seconds",
        )
        for second in seconds:
            assert DateFormatter.isoformat(
                datetime.timestamp(second), UTC
            ) == second.strftime("%Y-%m-%dT%H:%M:%S"), second.strftime(
                "%Y-%m-%dT%H:%M:%S"
            )

    def test_format_directives(self):
        seconds = get_datetimes_in_range(
            start=datetime(2020, 2, 29, 13, 17, 0, 0, pytz.UTC),
            end=datetime(2020, 2, 29, 13, 19, 0, 0, pytz.UTC),
            step="seconds",
        )
        for second in seconds:
            assert DateFormatter.format(
                datetime.timestamp(second), UTC, "%Y-%m-%dT%H:%M:%S"
            ) == second.strftime("%Y-%m-%dT%H:%M:%S"), second.strftime(
                "%Y-%m-%dT%H:%M:%S"
            )

    def test_format_directive_a(self):
        days = get_datetimes_in_range(
            start=datetime(2019, 1, 1, 0, 0, 0, 0, pytz.UTC),
            end=datetime(2020, 1, 31, 0, 0, 0, 0, pytz.UTC),
            step="days",
        )

        for day in days:
            assert DateFormatter.format(
                datetime.timestamp(day), UTC, "%a"
            ) == day.strftime("%a"), day.strftime("%Y-%m-%d")

    def test_format_directive_A(self):
        days = get_datetimes_in_range(
            start=datetime(2019, 1, 1, 0, 0, 0, 0, pytz.UTC),
            end=datetime(2020, 1, 31, 0, 0, 0, 0, pytz.UTC),
            step="days",
        )

        for day in days:
            assert DateFormatter.format(
                datetime.timestamp(day), UTC, "%A"
            ) == day.strftime("%A"), day.strftime("%Y-%m-%d")

    def test_format_directive_w(self):
        days = get_datetimes_in_range(
            start=datetime(2019, 1, 1, 0, 0, 0, 0, pytz.UTC),
            end=datetime(2020, 1, 31, 0, 0, 0, 0, pytz.UTC),
            step="days",
        )

        for day in days:
            assert DateFormatter.format(
                datetime.timestamp(day), UTC, "%w"
            ) == day.strftime("%w"), day.strftime("%Y-%m-%d")

    def test_format_directive_d(self):
        days = get_datetimes_in_range(
            start=datetime(2019, 1, 1, 0, 0, 0, 0, pytz.UTC),
            end=datetime(2020, 1, 31, 0, 0, 0, 0, pytz.UTC),
            step="days",
        )

        for day in days:
            assert DateFormatter.format(
                datetime.timestamp(day), UTC, "%d"
            ) == day.strftime("%d"), day.strftime("%Y-%m-%d")

    def test_format_directive_b(self):
        days = get_datetimes_in_range(
            start=datetime(2019, 1, 1, 0, 0, 0, 0, pytz.UTC),
            end=datetime(2020, 1, 31, 0, 0, 0, 0, pytz.UTC),
            step="days",
        )

        for day in days:
            assert DateFormatter.format(
                datetime.timestamp(day), UTC, "%b"
            ) == day.strftime("%b"), day.strftime("%Y-%m-%d")

    def test_format_directive_B(self):
        days = get_datetimes_in_range(
            start=datetime(2019, 1, 1, 0, 0, 0, 0, pytz.UTC),
            end=datetime(2020, 1, 31, 0, 0, 0, 0, pytz.UTC),
            step="days",
        )

        for day in days:
            assert DateFormatter.format(
                datetime.timestamp(day), UTC, "%B"
            ) == day.strftime("%B"), day.strftime("%Y-%m-%d")

    def test_format_directive_m(self):
        days = get_datetimes_in_range(
            start=datetime(2019, 1, 1, 0, 0, 0, 0, pytz.UTC),
            end=datetime(2020, 1, 31, 0, 0, 0, 0, pytz.UTC),
            step="days",
        )

        for day in days:
            assert DateFormatter.format(
                datetime.timestamp(day), UTC, "%m"
            ) == day.strftime("%m"), day.strftime("%Y-%m-%d")

    def test_format_directive_y(self):
        years = get_years_in_range(1999, 2022)
        for year in years:
            assert DateFormatter.format(
                datetime.timestamp(year), UTC, "%y"
            ) == year.strftime("%y"), year.strftime("%Y-%m-%d")

    def test_format_directive_H(self):
        minutes = get_datetimes_in_range(
            start=datetime(2020, 2, 29, 0, 17, 0, 0, pytz.UTC),
            end=datetime(2020, 2, 29, 23, 59, 0, 0, pytz.UTC),
            step="minutes",
        )
        for minute in minutes:
            assert DateFormatter.format(
                datetime.timestamp(minute), UTC, "%H"
            ) == minute.strftime("%H"), minute.strftime("%Y-%m-%dT%H:%M:%S")

    def test_format_directive_I(self):
        minutes = get_datetimes_in_range(
            start=datetime(2020, 2, 29, 0, 17, 0, 0, pytz.UTC),
            end=datetime(2020, 2, 29, 23, 59, 0, 0, pytz.UTC),
            step="minutes",
        )
        for minute in minutes:
            assert DateFormatter.format(
                datetime.timestamp(minute), UTC, "%I"
            ) == minute.strftime("%I"), minute.strftime("%Y-%m-%dT%H:%M:%S")

        unixtime = time.time()
        dt = datetime.fromtimestamp(unixtime)
        timezone = FixedOffsetTimezone(offset_hours=-time.localtime().tm_gmtoff)
        assert dt.strftime("%I") == DateFormatter.format(unixtime, timezone, "%I")

    def test_format_directive_p(self):
        minutes = get_datetimes_in_range(
            start=datetime(2020, 2, 29, 0, 17, 0, 0, pytz.UTC),
            end=datetime(2020, 2, 29, 23, 59, 0, 0, pytz.UTC),
            step="minutes",
        )
        for minute in minutes:
            assert DateFormatter.format(
                datetime.timestamp(minute), UTC, "%p"
            ) == minute.strftime("%p"), minute.strftime("%Y-%m-%dT%H:%M:%S")

    def test_format_directive_M(self):
        minutes = get_datetimes_in_range(
            start=datetime(2020, 2, 29, 10, 17, 0, 0, pytz.UTC),
            end=datetime(2020, 2, 29, 12, 19, 0, 0, pytz.UTC),
            step="minutes",
        )
        for minute in minutes:
            assert DateFormatter.format(
                datetime.timestamp(minute), UTC, "%M"
            ) == minute.strftime("%M"), minute.strftime("%Y-%m-%dT%H:%M:%S")

    def test_format_directive_S(self):
        seconds = get_datetimes_in_range(
            start=datetime(2020, 2, 29, 13, 17, 0, 0, pytz.UTC),
            end=datetime(2020, 2, 29, 13, 19, 0, 0, pytz.UTC),
            step="seconds",
        )
        for second in seconds:
            assert DateFormatter.format(
                datetime.timestamp(second), UTC, "%S"
            ) == second.strftime("%S"), second.strftime("%Y-%m-%dT%H:%M:%S")

    def test_format_directive_Z(self):
        days = get_datetimes_in_range(
            start=datetime(2019, 1, 1, 0, 0, 0, 0, pytz.UTC),
            end=datetime(2020, 1, 31, 0, 0, 0, 0, pytz.UTC),
            step="days",
        )

        for day in days:
            assert (
                DateFormatter.format(datetime.timestamp(day), UTC, "%Z") == "UTC"
            ), day.strftime("%Y-%m-%d")

    def test_format_directive_z(self):
        days = get_datetimes_in_range(
            start=datetime(2019, 1, 1, 0, 0, 0, 0, pytz.UTC),
            end=datetime(2020, 1, 31, 0, 0, 0, 0, pytz.UTC),
            step="days",
        )

        for day in days:
            assert (
                DateFormatter.format(datetime.timestamp(day), UTC, "%z") == "+0000"
            ), day.strftime("%Y-%m-%d")

    def test_format_directive_C(self):
        years = get_years_in_range(1999, 2022)

        for year in years:
            assert DateFormatter.format(
                datetime.timestamp(year), UTC, "%C"
            ) == year.strftime("%C"), year.strftime("%Y")

    def test_format_directive_Y(self):
        years = get_years_in_range(1999, 2022)
        for year in years:
            assert DateFormatter.format(
                datetime.timestamp(year), UTC, "%Y"
            ) == year.strftime("%Y"), year.strftime("%Y-%m-%d")

    def test_format_directive_u(self):
        days = get_datetimes_in_range(
            start=datetime(2019, 1, 1, 0, 0, 0, 0, pytz.UTC),
            end=datetime(2020, 1, 31, 0, 0, 0, 0, pytz.UTC),
            step="days",
        )

        for day in days:
            assert DateFormatter.format(
                datetime.timestamp(day), UTC, "%u"
            ) == day.strftime("%u"), day.strftime("%Y-%m-%d")

    def test_format_directive_percent(self):
        days = get_datetimes_in_range(
            start=datetime(2019, 1, 1, 0, 0, 0, 0, pytz.UTC),
            end=datetime(2020, 1, 31, 0, 0, 0, 0, pytz.UTC),
            step="days",
        )

        for day in days:
            assert DateFormatter.format(
                datetime.timestamp(day), UTC, "%%"
            ) == day.strftime("%%"), day.strftime("%Y-%m-%d")

    def test_format_directive_doy(self):
        days = get_datetimes_in_range(
            start=datetime(2019, 1, 1, 0, 0, 0, 0, pytz.UTC),
            end=datetime(2020, 1, 31, 0, 0, 0, 0, pytz.UTC),
            step="days",
        )

        for day in days:
            assert DateFormatter.format(
                datetime.timestamp(day), UTC, "%j"
            ) == day.strftime("%j"), day.strftime("%Y-%m-%d")

    def test_format_string_Ymd(self):
        days = get_datetimes_in_range(
            start=datetime(2019, 1, 1, 0, 0, 0, 0, pytz.UTC),
            end=datetime(2020, 1, 31, 0, 0, 0, 0, pytz.UTC),
            step="days",
        )

        for day in days:
            assert DateFormatter.format(
                datetime.timestamp(day), UTC, "%Y-%m-%d"
            ) == day.strftime("%Y-%m-%d"), day.strftime("%Y-%m-%d")

    def test_format_string_ymd(self):
        days = get_datetimes_in_range(
            start=datetime(2019, 1, 1, 0, 0, 0, 0, pytz.UTC),
            end=datetime(2020, 1, 31, 0, 0, 0, 0, pytz.UTC),
            step="days",
        )

        for day in days:
            assert DateFormatter.format(
                datetime.timestamp(day), UTC, "%y-%m-%d"
            ) == day.strftime("%y-%m-%d"), day.strftime("%Y-%m-%d")

    def test_format_string_abdY(self):
        days = get_datetimes_in_range(
            start=datetime(2019, 1, 1, 0, 0, 0, 0, pytz.UTC),
            end=datetime(2020, 1, 31, 0, 0, 0, 0, pytz.UTC),
            step="days",
        )

        for day in days:
            assert DateFormatter.format(
                datetime.timestamp(day), UTC, "%a %b %d, %Y"
            ) == day.strftime("%a %b %d, %Y"), day.strftime("%Y-%m-%d")

    def test_format_string_YmdHMS(self):
        minutes = get_datetimes_in_range(
            start=datetime(2020, 2, 29, 10, 17, 0, 0, pytz.UTC),
            end=datetime(2020, 2, 29, 12, 19, 0, 0, pytz.UTC),
            step="minutes",
        )
        for minute in minutes:
            assert DateFormatter.format(
                datetime.timestamp(minute), UTC, "%Y-%m-%d %H:%M:%S"
            ) == minute.strftime("%Y-%m-%d %H:%M:%S"), minute.strftime(
                "%Y-%m-%dT%H:%M:%S"
            )

    def test_format_string_YmdTHMS(self):
        minutes = get_datetimes_in_range(
            start=datetime(2020, 2, 29, 10, 17, 0, 0, pytz.UTC),
            end=datetime(2020, 2, 29, 12, 19, 0, 0, pytz.UTC),
            step="minutes",
        )
        for minute in minutes:
            assert DateFormatter.format(
                datetime.timestamp(minute), UTC, "%Y-%m-%dT%H:%M:%S"
            ) == minute.strftime("%Y-%m-%dT%H:%M:%S"), minute.strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
