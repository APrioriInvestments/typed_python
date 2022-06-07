import time
import unittest
from typed_python.lib.timestamp import Timestamp
from datetime import datetime, timezone


class TestTimestamp(unittest.TestCase):
    def test_timetuple_default_offset(self):
        unixtime = time.time()
        timestamp = Timestamp(unixtime)
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
        timestamp = Timestamp(unixtime)
        dt_tuple = datetime.fromtimestamp(unixtime, tz=timezone.utc).timetuple()
        tstuple = timestamp.timetuple('+00:00')

        assert dt_tuple.tm_year == tstuple.tm_year
        assert dt_tuple.tm_mon == tstuple.tm_mon
        assert dt_tuple.tm_mday == tstuple.tm_mday
        assert dt_tuple.tm_hour == tstuple.tm_hour
        assert dt_tuple.tm_min == tstuple.tm_min
        assert dt_tuple.tm_sec == tstuple.tm_sec

    def test_isoformat_local_offset(self):
        unixtime = time.time()
        utc_offset = time.strftime('%z')

        utc_offset_with_colon = utc_offset[0:3] + ':' + utc_offset[3:5]
        timestamp = Timestamp(unixtime)
        dt = datetime.fromtimestamp(unixtime)

        assert dt.isoformat(timespec='seconds') == timestamp.isoformat(utc_offset)
        assert dt.isoformat(timespec='seconds') == timestamp.isoformat(utc_offset_with_colon)
