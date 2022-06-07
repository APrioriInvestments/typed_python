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
    res = ListOf(datetime)([])
    for unixtime in range(N):
        res.append(datetime.fromtimestamp(unixtime))

    return res


@Entrypoint
def parseTimestamps(strings: ListOf(str)):
    res = ListOf(Timestamp)([])
    for string in strings:
        res.append(Timestamp.parse(string))
    return res


@Entrypoint
def parseDatetimes(strings: ListOf(str)):
    res = ListOf(datetime)([])
    for string in strings:
        res.append(datetime.strptime(string, '%Y-%m-%dT%H:%M:%S'))
    return res


@Entrypoint
def formatTimestamps(timestamps: ListOf(Timestamp)):
    res = ListOf(str)()
    for timestamp in timestamps:
        res.append(timestamp.format(format='%Y-%m-%d'))

    return res


@Entrypoint
def formatDatetimes(datetimes: ListOf(datetime)):
    res = ListOf(str)()
    for dt in datetimes:
        res.append(dt.strftime('%Y-%m-%d'))
    return res


class TestTimestamp(unittest.TestCase):

    def test_demo_usage(self):

        # create timestamp from unixtime
        Timestamp.make(time.time())
        Timestamp.make(ts=time.time())

        # create timestamp from iso 8601 date string
        Timestamp.parse('2022-10-22T06:39')

        # create timestamp from non iso date string
        Timestamp.parse('Oct 22, 2022 06:39')

        # with relative tz (offset changes with dst)
        Timestamp.parse('2022-10-22T06:39NYC')

        # with relative tz (offset changes with dst)
        Timestamp.parse('2022-10-22T06:39ET')

        # with fixed offset tz
        Timestamp.parse('2022-10-22T06:39UTC')

        # with fixed offset tz
        Timestamp.parse('2022-10-22T06:39EST')

        # with fixed offset tz
        Timestamp.parse('2022-10-22T06:39EDT')

        # with fixed offset tz
        Timestamp.parse('2022-10-22T06:39EDT')

    def test_eq(self):
        with PrintNewFunctionVisitor():
            unixtime = time.time()
            ts1 = Timestamp.make(unixtime)
            ts2 = Timestamp.make(unixtime)
            assert ts1.eq(ts2)

    def test_ge(self):
        unixtime = time.time()
        ts1 = Timestamp.make(unixtime)
        ts2 = Timestamp.make(unixtime)
        ts3 = Timestamp.make(unixtime - 1)

        assert ts1.ge(ts2)
        assert ts1.ge(ts3)

    def test_gt(self):
        unixtime = time.time()
        ts1 = Timestamp.make(unixtime)
        ts2 = Timestamp.make(unixtime - 1)
        assert ts1.gt(ts2)

    def test_le(self):
        unixtime = time.time()
        ts1 = Timestamp.make(unixtime)
        ts2 = Timestamp.make(unixtime)
        ts3 = Timestamp.make(unixtime + 1)

        assert ts1.le(ts2)
        assert ts1.le(ts3)

    def test_lt(self):
        unixtime = time.time()
        ts1 = Timestamp.make(unixtime)
        ts2 = Timestamp.make(unixtime + 1)

        assert ts1.lt(ts2)

    def test_ne(self):
        unixtime = time.time()
        ts1 = Timestamp.make(unixtime)
        ts2 = Timestamp.make(unixtime + 1)
        ts3 = Timestamp.make(unixtime - 1)

        assert ts1.ne(ts2)
        assert ts1.ne(ts3)

    def test_add(self):
        unixtime = time.time()
        ts = Timestamp.make(unixtime)
        ts1 = ts.add(5)
        unixtime = time.time()
        ts = Timestamp.make(unixtime)
        ts1 = ts.sub(5)

        assert float(ts1) == unixtime - 5

    def test_to_string_default(self):
        # just a superficial test here.
        # to_string calls DateFormatter.to_string which has more robust tests
        unixtime = time.time()
        timestamp = Timestamp.make(unixtime + time.localtime().tm_gmtoff)
        dt = datetime.fromtimestamp(unixtime)
        assert dt.isoformat(timespec='seconds') == timestamp.format()

    def test_to_string(self):
        # just a superficial test here.
        # to_string calls DateFormatter.to_string which has more robust tests
        unixtime = time.time()
        timestamp = Timestamp.make(unixtime + time.localtime().tm_gmtoff)
        dt = datetime.fromtimestamp(unixtime)
        assert dt.isoformat(timespec='seconds') == timestamp.format(format="%Y-%m-%dT%H:%M:%S")

    def test_from_date(self):
        unixtime = time.time()
        dt_tuple = datetime.fromtimestamp(unixtime, tz=timezone.utc).timetuple()

        timestamp = Timestamp.from_date(year=dt_tuple.tm_year,
                                        month=dt_tuple.tm_mon,
                                        day=dt_tuple.tm_mday,
                                        hour=dt_tuple.tm_hour,
                                        minute=dt_tuple.tm_min,
                                        second=dt_tuple.tm_sec)
        assert int(unixtime) == int(timestamp)

    def test_parse(self):
        unixtime = time.time()
        timestamp = Timestamp.make(unixtime)
        dt = datetime.fromtimestamp(unixtime, tz=timezone.utc).timetuple()
        date_str = f"{dt.tm_year}-{dt.tm_mon:02d}-{dt.tm_mday:02d} {dt.tm_hour:02d}:{dt.tm_min:02d}:{dt.tm_sec:02d}"
        parsed_timestamp = Timestamp.parse(date_str)

        assert ((int(timestamp) == int(parsed_timestamp)))

    def test_timestamp_is_held_class(self):
        """ This is a temporary test intended to exhibit the Held class semantics.

        Moral of the story: a Class decorated with 'Held' isn't supposed to have
        pointers to its instances. Instead, its instances get cloned when otherwise
        a new pointer would be created (e.g. insertion to a list, access from a list).
        This is how python constants behave already. However, unlike python constants,
        Held Classes may have modifiable state (Members). This can be confusing if
        the user expects state to be shared between clones (it isn't). For the most
        part, the user should probably not modify their state - in particular, use
        the default Class constructor rather than writing __init__ methods.

        Why do this?
        In the case of Timestamp - it allows us to inject some type information (the
        statement that certain floats represent timestamps, have access to certain
        methods etc.) but maintain the performance of working directly with the
        floats (values live on the stack).
        """
        ts0 = Timestamp.make(0.0)

        timestamps = ListOf(Timestamp)([ts0])

        # putting ts0 in the ListOf cloned it, so:

        # (1) modifications to ts0 are not seen by the ListOf
        ts0.ts = 1.0
        assert ts0.ts == 1.0
        assert timestamps[0].ts == 0.0

        # (2) modifications to timestamps[0] are not seen by ts0
        timestamps[0].ts = 2.0
        assert ts0.ts == 1.0
        assert timestamps[0].ts == 2.0

        # 'access-to-modify' and 'access-to-assign' diverge when dealing
        # with held classes (this is a major departure from python semantics), i.e.:

        # access-to-assign: ts1 is a clone of timestamps[0]
        ts1 = timestamps[0]
        assert ts1.ts == timestamps[0].ts == 2.0

        # (3) modifications changes to ts1 are not seen by timestamps[0]
        ts1.ts = 3.0
        assert ts1.ts == 3.0
        assert timestamps[0].ts == 2.0

        # (4) 'access-to-modify': modifications to timestamps[0] are not seen by ts1
        timestamps[0].ts = 4.0
        assert ts1.ts == 3.0
        assert timestamps[0].ts == 4.0

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

        print('Timestamp.make (' + str(tsTime) + ') is ' + str(speedup) +
              ' faster than datetime.fromtimestamp (' + str(dtTime) + ')')

        # assert speedup > 30 and speedup < 40, speedup

    def test_compare_timestamp_datetime_from_string(self):
        runs = 100000
        date_strings = make_list_of_iso_datestrings(runs)

        with PrintNewFunctionVisitor():
            Timestamp.parse('1997')

        start = time.time()
        parseTimestamps(date_strings)
        tsTime = time.time() - start

        start = time.time()
        parseDatetimes(date_strings)
        dtTime = time.time() - start

        if dtTime > tsTime:
            speedup = dtTime / tsTime
            compare = 'x faster'
        else:
            speedup = tsTime / dtTime
            compare = 'x slower'

        print('Timestamp.parse (' + str(tsTime) + ') is ' +
              str("{:.2f}".format(speedup)) + compare + ' than datetime.strptime (' + str(dtTime) + ')')
        # assert speedup > 7 and speedup < 8

    def test_compare_timestamp_datetime_format(self):
        runs = 1000000
        timestamps = listOfTimestamps(runs)
        datetimes = listOfDatetimes(runs)

        with PrintNewFunctionVisitor():
            ts = Timestamp.make(2)
            ts.format(format='%Y-%m-%d')
            (Timestamp.make(2)).format(format='%Y-%m-%d')

        start = time.time()
        formatTimestamps(timestamps)
        tsTime = time.time() - start

        start = time.time()
        formatDatetimes(datetimes)
        dtTime = time.time() - start

        if dtTime > tsTime:
            speedup = dtTime / tsTime
            compare = 'x faster'
        else:
            speedup = tsTime / dtTime
            compare = 'x slower'

        print('Timestamp.format (' + str(tsTime) + ') is ' +
              str("{:.2f}".format(speedup)) + compare + ' than datetime.strformat (' + str(dtTime) + ')')

        # assert speedup > 1 and speedup < 1.5
