import pytz
import datetime
from typed_python.lib.datetime.DateTime import DateTime, Date, TimeOfDay, TimeZone, DaylightSavingsTimezone

Nyc = DaylightSavingsTimezone(dst_offset_hours=4, st_offset_hours=5)

def test_DateTime():
    year, month, day = 2022, 12, 23
    hour, minute, second = 18, 40, 36
    date = Date(year=year, month=month, day=day)
    timeOfDay = TimeOfDay(hour=hour, minute=minute, second=second)
    dateTime = DateTime(date = date, timeOfDay=timeOfDay)

    timestamp = Nyc.timestamp(dateTime)

    tz = pytz.timezone("America/New_York")
    ts = tz.localize(datetime.datetime(year, month, day, hour, minute, second)).timestamp()

    assert timestamp == ts


