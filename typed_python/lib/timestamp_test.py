import time
import unittest
from typed_python.lib.timestamp import Timestamp
from datetime import datetime, timezone
from typed_python import ListOf


class TestTimestamp(unittest.TestCase):
    def test_timetuple_default_offset(self):
        unixtime = time.time()
        timestamp = Timestamp.make(unixtime)
        dt_tuple = datetime.fromtimestamp(unixtime, tz=timezone.utc).timetuple()
        tstuple = timestamp.timetuple()

        assert dt_tuple.tm_year == tstuple.tm_year
        assert dt_tuple.tm_mon == tstuple.tm_mon
        assert dt_tuple.tm_mday == tstuple.tm_mday
        assert dt_tuple.tm_hour == tstuple.tm_hour
        assert dt_tuple.tm_min == tstuple.tm_min
        assert dt_tuple.tm_sec == tstuple.tm_sec

    def test_timetuple_with_offset(self):
        unixtime = time.time()
        timestamp = Timestamp.make(unixtime)
        dt_tuple = datetime.fromtimestamp(unixtime, tz=timezone.utc).timetuple()
        tstuple = timestamp.timetuple()

        assert dt_tuple.tm_year == tstuple.tm_year
        assert dt_tuple.tm_mon == tstuple.tm_mon
        assert dt_tuple.tm_mday == tstuple.tm_mday
        assert dt_tuple.tm_hour == tstuple.tm_hour
        assert dt_tuple.tm_min == tstuple.tm_min
        assert dt_tuple.tm_sec == tstuple.tm_sec

    def test_isoformat_local_offset(self):
        unixtime = time.time()
        timestamp = Timestamp.make(unixtime)
        dt = datetime.fromtimestamp(unixtime)
        assert dt.isoformat(timespec='seconds') == timestamp.isoformat('-04:00')

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
