import time
import unittest
from typed_python.lib.timestamp import Timestamp
from datetime import datetime, timezone
from typed_python import ListOf
import pytz


class TestTimestamp(unittest.TestCase):
    def test_date_no_offset(self):
        unixtime = time.time()
        timestamp = Timestamp.make(unixtime)
        dt_tuple = datetime.fromtimestamp(unixtime, tz=timezone.utc).timetuple()
        date = timestamp.date()

        assert dt_tuple.tm_year == date.year
        assert dt_tuple.tm_mon == date.month
        assert dt_tuple.tm_mday == date.day
        assert dt_tuple.tm_hour == date.hour
        assert dt_tuple.tm_min == date.minute
        assert dt_tuple.tm_sec == date.second

    def test_date_string_offset(self):
        unixtime = time.time()
        timestamp = Timestamp.make(unixtime)
        dt_tuple = datetime.fromtimestamp(unixtime, tz=timezone.utc).timetuple()
        date = timestamp.date('+00:00')

        assert dt_tuple.tm_year == date.year
        assert dt_tuple.tm_mon == date.month
        assert dt_tuple.tm_mday == date.day
        assert dt_tuple.tm_hour == date.hour
        assert dt_tuple.tm_min == date.minute
        assert dt_tuple.tm_sec == date.second

    def test_date_with_numeric_utc_offset_localtime(self):
        unixtime = time.time()
        timestamp = Timestamp.make(unixtime)
        dt_tuple = datetime.fromtimestamp(unixtime).timetuple()

        utc_offset = time.localtime().tm_gmtoff
        date = timestamp.date(utc_offset)

        assert dt_tuple.tm_year == date.year
        assert dt_tuple.tm_mon == date.month
        assert dt_tuple.tm_mday == date.day
        assert dt_tuple.tm_hour == date.hour
        assert dt_tuple.tm_min == date.minute
        assert dt_tuple.tm_sec == date.second

    def test_date_with_numeric_utc_offset(self):
        tz = pytz.timezone('America/New_York')

        ny_now = datetime.now(tz)
        ny_dt_tuple = ny_now.timetuple()
        unixtime = datetime.timestamp(ny_now)

        timestamp = Timestamp.make(unixtime)
        date = timestamp.date(ny_now.utcoffset().total_seconds())

        assert ny_dt_tuple.tm_year == date.year
        assert ny_dt_tuple.tm_mon == date.month
        assert ny_dt_tuple.tm_mday == date.day
        assert ny_dt_tuple.tm_hour == date.hour
        assert ny_dt_tuple.tm_min == date.minute
        assert ny_dt_tuple.tm_sec == date.second

    def test_date_with_string_utc_offset_localtime(self):
        unixtime = time.time()
        timestamp = Timestamp.make(unixtime)
        dt = datetime.fromtimestamp(unixtime)
        dt_tuple = dt.timetuple()

        date = timestamp.date(time.strftime("%z"))

        assert dt_tuple.tm_year == date.year
        assert dt_tuple.tm_mon == date.month
        assert dt_tuple.tm_mday == date.day
        assert dt_tuple.tm_hour == date.hour
        assert dt_tuple.tm_min == date.minute
        assert dt_tuple.tm_sec == date.second

    def test_date_default_offset(self):
        unixtime = time.time()
        timestamp = Timestamp.make(unixtime)
        dt_tuple = datetime.fromtimestamp(unixtime, tz=timezone.utc).timetuple()
        date = timestamp.date()

        assert dt_tuple.tm_year == date.year
        assert dt_tuple.tm_mon == date.month
        assert dt_tuple.tm_mday == date.day
        assert dt_tuple.tm_hour == date.hour
        assert dt_tuple.tm_min == date.minute
        assert dt_tuple.tm_sec == date.second

    def test_timetuple_with_offset(self):
        unixtime = time.time()
        timestamp = Timestamp.make(unixtime)
        dt_tuple = datetime.fromtimestamp(unixtime, tz=timezone.utc).timetuple()
        date = timestamp.date()

        assert dt_tuple.tm_year == date.year
        assert dt_tuple.tm_mon == date.month
        assert dt_tuple.tm_mday == date.day
        assert dt_tuple.tm_hour == date.hour
        assert dt_tuple.tm_min == date.minute
        assert dt_tuple.tm_sec == date.second

    def test_to_string(self):
        unixtime = time.time()
        timestamp = Timestamp.make(unixtime)
        dt = datetime.fromtimestamp(unixtime)

        assert dt.isoformat(timespec='seconds') == timestamp.to_string(time.localtime().tm_gmtoff, format="%Y-%m-%dT%H:%M:%S")

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
