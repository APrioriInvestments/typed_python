import pytest
import pytz
import datetime
from typed_python.lib.datetime.DateTime import (
    DateTime,
    Date,
    TimeOfDay,
    TimeZone,
    DaylightSavingsTimezone,
    NonexistentDateTime,
    FixedOffsetTimezone,
)

Nyc = DaylightSavingsTimezone(dst_offset_hours=4, st_offset_hours=5)
Utc = FixedOffsetTimezone(offset_hours=0)


def test_DateTime_to_timestamp():
    year, month, day = 2022, 12, 23
    hour, minute, second = 18, 40, 36
    date = Date(year=year, month=month, day=day)
    timeOfDay = TimeOfDay(hour=hour, minute=minute, second=second)
    dateTime = DateTime(date=date, timeOfDay=timeOfDay)

    timestamp = Nyc.timestamp(dateTime)

    tz = pytz.timezone("America/New_York")
    ts = tz.localize(datetime.datetime(year, month, day, hour, minute, second)).timestamp()

    assert timestamp == ts


def test_DateTime_nonexistent_DateTime():
    year, month, day = 2022, 3, 13
    hour, minute, second = 2, 30, 0
    date = Date(year=year, month=month, day=day)
    timeOfDay = TimeOfDay(hour=hour, minute=minute, second=second)
    dateTime = DateTime(date=date, timeOfDay=timeOfDay)

    with pytest.raises(NonexistentDateTime):
        Nyc.timestamp(dateTime)


def test_DateTime_to_timestamp_daylight_savings():
    year, month, day = 2022, 11, 6
    hour, minute, second = 1, 30, 0
    date = Date(year=year, month=month, day=day)
    timeOfDay = TimeOfDay(hour=hour, minute=minute, second=second)
    dateTime = DateTime(date=date, timeOfDay=timeOfDay)

    timestamp1 = Nyc.timestamp(dateTime)
    timestamp2 = Nyc.timestamp(dateTime, afterFold=True)

    tz = pytz.timezone("America/New_York")
    ts = tz.localize(datetime.datetime(year, month, day, hour, minute, second)).timestamp()

    assert timestamp2 == ts


def test_DateTime_from_timestamp():
    year, month, day = 2022, 11, 6
    hour, minute, second = 1, 30, 0
    date = Date(year=year, month=month, day=day)
    timeOfDay = TimeOfDay(hour=hour, minute=minute, second=second)
    dateTime = DateTime(date=date, timeOfDay=timeOfDay)

    timestamp = Utc.timestamp(dateTime)
    newDateTime = Utc.datetime(timestamp)
    assert newDateTime == dateTime
