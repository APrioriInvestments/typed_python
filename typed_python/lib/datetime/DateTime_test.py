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
Awt = FixedOffsetTimezone(offset_hours=-8)


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


def test_DateTime_to_timestamp_daylight_savings():
    year, month, day = 2022, 11, 6
    hour, minute, second = 5, 30, 0
    date = Date(year=year, month=month, day=day)
    timeOfDay = TimeOfDay(hour=hour, minute=minute, second=second)
    utcDateTime = DateTime(date=date, timeOfDay=timeOfDay)

    oneThirtyAmNycFirstFold = Utc.timestamp(utcDateTime)

    year, month, day = 2022, 11, 6
    hour, minute, second = 6, 30, 0
    date = Date(year=year, month=month, day=day)
    timeOfDay = TimeOfDay(hour=hour, minute=minute, second=second)
    utcDateTime = DateTime(date=date, timeOfDay=timeOfDay)

    oneThirtyAmNycSecondFold = Utc.timestamp(utcDateTime)

    assert oneThirtyAmNycSecondFold - oneThirtyAmNycFirstFold == 3600
    assert Nyc.datetime(oneThirtyAmNycFirstFold) == Nyc.datetime(oneThirtyAmNycSecondFold)


def test_fixed_offset():
    year, month, day = 2022, 11, 6
    hour, minute, second = 5, 30, 0
    date = Date(year=year, month=month, day=day)
    timeOfDay = TimeOfDay(hour=hour, minute=minute, second=second)
    dateTime = DateTime(date=date, timeOfDay=timeOfDay)

    ts = pytz.timezone('Asia/Hong_Kong').localize(datetime.datetime(year, month, day, hour, minute, second)).timestamp()

    res = FixedOffsetTimezone(offset_hours=-8).timestamp(dateTime)
    assert ts == res


