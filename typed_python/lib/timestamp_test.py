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

import pytz
import time
import unittest

from typed_python.compiler.runtime import Entrypoint, PrintNewFunctionVisitor

from typed_python.lib.timestamp import Timestamp
from datetime import datetime, timezone
from typed_python import ListOf


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *args):
        self.duration = time.time() - self.t0
        print(self.duration)

    def getDuration(self):
        return self.duration


@Entrypoint
def make_list_of_iso_datestrings(n):
    res = ListOf(str)()
    for i in range(n):
        dt = datetime.fromtimestamp(i)
        res.append(dt.isoformat())
    return res


@Entrypoint
def make_list_of_datetimes(n):
    res = ListOf(datetime)()
    for i in range(n):
        dt = datetime.fromtimestamp(i)
        res.append(dt)
    return res


@Entrypoint
def listOfTimestamps(N):
    res = ListOf(Timestamp)()
    for unixtime in range(N):
        res.append(Timestamp.make(unixtime))

    return res


@Entrypoint
def listOfDatetimes(N):
    res = ListOf(datetime)()
    for unixtime in range(N):
        res.append(datetime.fromtimestamp(unixtime))

    return res


@Entrypoint
def parseNycTimestamps(strings: ListOf(str)):
    res = ListOf(Timestamp)()
    for string in strings:
        res.append(Timestamp.parse_nyc(string))
    return res


@Entrypoint
def parseTimestamps(strings: ListOf(str)):
    res = ListOf(Timestamp)()
    for string in strings:
        res.append(Timestamp.parse(string))
    return res


@Entrypoint
def parseDatetimes(strings: ListOf(str)):
    res = ListOf(datetime)()
    for string in strings:
        res.append(datetime.strptime(string, "%Y-%m-%dT%H:%M:%S"))
    return res


@Entrypoint
def formatTimestamps(timestamps: ListOf(Timestamp)):
    res = ListOf(str)()
    for timestamp in timestamps:
        res.append(timestamp.format(format="%Y-%m-%d"))

    return res


@Entrypoint
def formatDatetimes(datetimes: ListOf(datetime)):
    res = ListOf(str)()
    for dt in datetimes:
        res.append(dt.strftime("%Y-%m-%d"))
    return res


class TestTimestamp(unittest.TestCase):
    def test_demo_usage(self):

        # create timestamp from unixtime
        Timestamp.make(time.time())
        Timestamp.make(ts=time.time())

        # create timestamp from iso 8601 date string
        Timestamp.parse("2022-10-22T06:39")

        # create timestamp from iso8601ish string (space instead of T separator)
        Timestamp.parse("2022-10-22 06:39")

        # create timestamp from non iso date string
        Timestamp.parse("Oct 22, 2022 06:39")

        # with relative tz (offset changes with dst)
        Timestamp.parse("2022-10-22T06:39NYC")

        # with fixed offset tz
        Timestamp.parse("2022-10-22T06:39UTC")

        # with fixed offset tz
        Timestamp.parse("2022-10-22T06:39EST")

        # get date string from timestamp as YYYY-MM-DD
        ts = Timestamp.make(time.time())
        ts.format(format="%Y-%m-%d")

    def test_eq(self):
        # The following commented block of code sometimes unexpectedly errors with something like
        # AssertionError: assert Held(Timestamp)(ts=2,) == Held(Timestamp)(ts=2,)
        #
        # ts1 = Timestamp.make(2)
        # ts2 = Timestamp.make(2)
        # assert ts1 == ts2
        #
        # Added an otherwise unnecessary Entrypointed inner() function as a workaround
        #
        # See: https://github.com/APrioriInvestments/typed_python/issues/404 for bug details
        @Entrypoint
        def inner():
            assert Timestamp.make(2) == Timestamp.make(2)

        inner()

    def test_ge(self):
        # The following commented block of unexpectedly errors with something like
        # AssertionError: assert Held(Timestamp)(ts=1.6694e+09,) >= Held(Timestamp)(ts=1.6694e+09,)
        #
        # Added an otherwise unnecessary Entrypointed inner() function as a workaround
        #
        # See: https://github.com/APrioriInvestments/typed_python/issues/404 for bug details
        @Entrypoint
        def inner():
            unixtime = time.time()
            ts1 = Timestamp.make(unixtime)
            ts2 = Timestamp.make(unixtime)
            ts3 = Timestamp.make(unixtime - 1)
            assert ts1 >= ts2
            assert ts1 >= ts3

        inner()

    def test_gt(self):
        unixtime = time.time()
        ts1 = Timestamp.make(unixtime)
        ts2 = Timestamp.make(unixtime - 1)
        assert ts1 > ts2

    def test_le(self):
        # See comments in test_ge above
        @Entrypoint
        def inner():
            unixtime = time.time()
            ts1 = Timestamp.make(unixtime)
            ts2 = Timestamp.make(unixtime)
            ts3 = Timestamp.make(unixtime + 1)
            assert ts1 <= ts2
            assert ts1 <= ts3

        inner()

    def test_lt(self):
        unixtime = time.time()
        ts1 = Timestamp.make(unixtime)
        ts2 = Timestamp.make(unixtime + 1)

        assert ts1 < ts2

    def test_ne(self):
        unixtime = time.time()
        ts1 = Timestamp.make(unixtime)
        ts2 = Timestamp.make(unixtime + 1)
        ts3 = Timestamp.make(unixtime - 1)

        assert ts1 != ts2
        assert ts1 != ts3

    def test_add(self):
        unixtime = time.time()
        ts1 = Timestamp.make(unixtime)
        ts2 = Timestamp.make(5)
        ts3 = ts1 + ts2.ts
        assert ts3.ts == unixtime + 5

    def test_sub(self):
        unixtime = time.time()
        ts1 = Timestamp.make(unixtime)
        ts2 = Timestamp.make(5)
        ts3 = ts1 - ts2
        assert ts3 == unixtime - 5

    def test_format_default(self):
        # Just a superficial test. format proxies to DateFormatter.format
        # which has more extensive testing
        unixtime = time.time()
        timestamp = Timestamp.make(unixtime + time.localtime().tm_gmtoff)
        dt = datetime.fromtimestamp(unixtime)
        assert dt.isoformat(timespec="seconds").replace("T", " ") == timestamp.format()

    def test_format(self):
        # Just a superficial test. format proxies to  DateFormatter.format
        # which has more extensive testing
        unixtime = time.time()
        timestamp = Timestamp.make(unixtime + time.localtime().tm_gmtoff)
        dt = datetime.fromtimestamp(unixtime)
        assert dt.isoformat(timespec="seconds") == timestamp.format(
            format="%Y-%m-%dT%H:%M:%S"
        )

    def test_parse(self):
        unixtime = time.time()
        timestamp = Timestamp.make(unixtime)
        dt = datetime.fromtimestamp(unixtime, tz=timezone.utc).timetuple()
        date_str = f"{dt.tm_year}-{dt.tm_mon:02d}-{dt.tm_mday:02d} {dt.tm_hour:02d}:{dt.tm_min:02d}:{dt.tm_sec:02d}"
        parsed_timestamp = Timestamp.parse(date_str)

        assert int(timestamp) == int(parsed_timestamp)

    def test_parse_ampm(self):
        res = Timestamp.parse_nyc('2019/08/04 6:59 PM').ts
        expected = pytz.timezone('America/New_York').localize(datetime(2019, 8, 4, 18, 59, 0, 0)).timestamp()
        assert res == expected

    def test_parse_single_digit_day(self):
        res = Timestamp.parse_nyc('2020/12/1  14:15').ts
        expected = pytz.timezone('America/New_York').localize(datetime(2020, 12, 1, 14, 15, 0)).timestamp()
        assert res == expected

    def test_compare_timestamp_datetime_from_unixtime(self):
        runs = 10000000

        Timestamp.make(1)

        start = time.time()
        listOfTimestamps(runs)
        tsTime = time.time() - start

        start = time.time()
        listOfDatetimes(runs)
        dtTime = time.time() - start

        speedup = dtTime / tsTime

        print(
            "Timestamp.make ("
            + str(tsTime)
            + ") is "
            + str(speedup)
            + " faster than datetime.fromtimestamp ("
            + str(dtTime)
            + ")"
        )

        # assert speedup > 30 and speedup < 40, speedup

    def test_compare_timestamp_datetime_from_string(self):
        runs = 100000
        date_strings = make_list_of_iso_datestrings(runs)

        with PrintNewFunctionVisitor():
            Timestamp.parse("1997")

        start = time.time()
        parseTimestamps(date_strings)
        tsTime = time.time() - start

        start = time.time()
        parseDatetimes(date_strings)
        dtTime = time.time() - start

        if dtTime > tsTime:
            speedup = dtTime / tsTime
            compare = "x faster"
        else:
            speedup = tsTime / dtTime
            compare = "x slower"

        print(
            "Timestamp.parse ("
            + str(tsTime)
            + ") is "
            + str("{:.2f}".format(speedup))
            + compare
            + " than datetime.strptime ("
            + str(dtTime)
            + ")"
        )
        # assert speedup > 7 and speedup < 8

    def test_compare_timestamp_datetime_format(self):
        runs = 1000000
        timestamps = listOfTimestamps(runs)
        datetimes = listOfDatetimes(runs)

        with PrintNewFunctionVisitor():
            ts = Timestamp.make(2)
            ts.format(format="%Y-%m-%d")

        start = time.time()
        formatTimestamps(timestamps)
        tsTime = time.time() - start

        start = time.time()
        formatDatetimes(datetimes)
        dtTime = time.time() - start

        if dtTime > tsTime:
            speedup = dtTime / tsTime
            compare = "x faster"
        else:
            speedup = tsTime / dtTime
            compare = "x slower"

        print(
            "Timestamp.format ("
            + str(tsTime)
            + ") is "
            + str("{:.2f}".format(speedup))
            + compare
            + " than datetime.strformat ("
            + str(dtTime)
            + ")"
        )

        assert dtTime > tsTime and (speedup > 1 and speedup <= 4)

    def test_compare_timestamp_nyc_datetime_from_string(self):
        runs = 100000
        date_strings = make_list_of_iso_datestrings(runs)

        with PrintNewFunctionVisitor():
            Timestamp.parse_nyc("1997")

        start = time.time()
        parseNycTimestamps(date_strings)
        tsTime = time.time() - start

        start = time.time()
        parseDatetimes(date_strings)
        dtTime = time.time() - start

        if dtTime > tsTime:
            speedup = dtTime / tsTime
            compare = "x faster"
        else:
            speedup = tsTime / dtTime
            compare = "x slower"

        print(
            "Timestamp.parse ("
            + str(tsTime)
            + ") is "
            + str("{:.2f}".format(speedup))
            + compare
            + " than datetime.strptime ("
            + str(dtTime)
            + ")"
        )
        # assert speedup > 7 and speedup < 8
