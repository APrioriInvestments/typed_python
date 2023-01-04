import pytest
import pytz
import datetime
from typed_python.lib.datetime.DateTime import (
    DateTime,
    Date,
    TimeOfDay,
    NonexistentDateTime,
    FixedOffsetTimezone,
    EST,
    NYC,
    UTC,
)
from typed_python.lib.timestamp import Timestamp


def test_DateTime_to_timestamp():
    year, month, day = 2022, 12, 23
    hour, minute, second = 18, 40, 36
    date = Date(year=year, month=month, day=day)
    timeOfDay = TimeOfDay(hour=hour, minute=minute, second=second)
    dateTime = DateTime(date=date, timeOfDay=timeOfDay)

    timestamp = NYC.timestamp(dateTime)

    tz = pytz.timezone("America/New_York")
    ts = tz.localize(
        datetime.datetime(year, month, day, hour, minute, second)
    ).timestamp()

    assert timestamp == ts


def test_DateTime_nonexistent_DateTime():
    year, month, day = 2022, 3, 13
    hour, minute, second = 2, 30, 0
    date = Date(year=year, month=month, day=day)
    timeOfDay = TimeOfDay(hour=hour, minute=minute, second=second)
    dateTime = DateTime(date=date, timeOfDay=timeOfDay)

    with pytest.raises(NonexistentDateTime):
        NYC.timestamp(dateTime)


def test_DateTime_from_timestamp():
    year, month, day = 2022, 11, 6
    hour, minute, second = 1, 30, 0
    date = Date(year=year, month=month, day=day)
    timeOfDay = TimeOfDay(hour=hour, minute=minute, second=second)
    dateTime = DateTime(date=date, timeOfDay=timeOfDay)

    timestamp = UTC.timestamp(dateTime)
    newDateTime = UTC.datetime(timestamp)
    assert newDateTime == dateTime


def test_DateTime_to_timestamp_daylight_savings():
    year, month, day = 2022, 11, 6
    hour, minute, second = 5, 30, 0
    date = Date(year=year, month=month, day=day)
    timeOfDay = TimeOfDay(hour=hour, minute=minute, second=second)
    utcDateTime = DateTime(date=date, timeOfDay=timeOfDay)

    oneThirtyAmNycFirstFold = UTC.timestamp(utcDateTime)

    year, month, day = 2022, 11, 6
    hour, minute, second = 6, 30, 0
    date = Date(year=year, month=month, day=day)
    timeOfDay = TimeOfDay(hour=hour, minute=minute, second=second)
    utcDateTime = DateTime(date=date, timeOfDay=timeOfDay)

    oneThirtyAmNycSecondFold = UTC.timestamp(utcDateTime)

    assert oneThirtyAmNycSecondFold - oneThirtyAmNycFirstFold == 3600
    assert NYC.datetime(oneThirtyAmNycFirstFold) == NYC.datetime(
        oneThirtyAmNycSecondFold
    )


def test_fixed_offset():
    year, month, day = 2022, 11, 6
    hour, minute, second = 5, 30, 0
    date = Date(year=year, month=month, day=day)
    timeOfDay = TimeOfDay(hour=hour, minute=minute, second=second)
    dateTime = DateTime(date=date, timeOfDay=timeOfDay)

    ts = (
        pytz.timezone("Asia/Hong_Kong")
        .localize(datetime.datetime(year, month, day, hour, minute, second))
        .timestamp()
    )

    res = FixedOffsetTimezone(offset_hours=+8).timestamp(dateTime)
    assert ts == res


def test_datetime_to_timestamp_and_back():
    for tz in [NYC, EST, UTC]:
        date = Date(year=2022, month=11, day=6)
        timeOfDay = TimeOfDay(hour=1, minute=30, second=0)
        dtime = DateTime(date=date, timeOfDay=timeOfDay)
        ts = tz.timestamp(dtime)
        dtime2 = tz.datetime(ts)
        assert dtime == dtime2


def test_EST_against_datetime():
    tz = pytz.timezone("America/Atikokan")
    externalTimestamp = tz.localize(
        datetime.datetime(2022, 11, 6, 1, 30, 0)
    ).timestamp()
    internalTimestamp = Timestamp.parse("2022-11-06 01:30:00est")
    assert externalTimestamp == internalTimestamp.ts, (
        externalTimestamp - internalTimestamp.ts
    )


def test_timestamp_parse_around_daylight_savings_switch():
    nycDateStringsToUtcDateStrings = {
        "2022-11-05 00:30:00nyc": "2022-11-05 04:30:00",
        "2022-11-05 01:30:00nyc": "2022-11-05 05:30:00",
        "2022-11-05 02:30:00nyc": "2022-11-05 06:30:00",
        "2022-11-05 03:30:00nyc": "2022-11-05 07:30:00",
        "2022-11-05 04:30:00nyc": "2022-11-05 08:30:00",
        "2022-11-05 05:30:00nyc": "2022-11-05 09:30:00",
        "2022-11-05 06:30:00nyc": "2022-11-05 10:30:00",
        "2022-11-05 07:30:00nyc": "2022-11-05 11:30:00",
        "2022-11-05 08:30:00nyc": "2022-11-05 12:30:00",
        "2022-11-05 09:30:00nyc": "2022-11-05 13:30:00",
        "2022-11-05 10:30:00nyc": "2022-11-05 14:30:00",
        "2022-11-05 11:30:00nyc": "2022-11-05 15:30:00",
        "2022-11-05 12:30:00nyc": "2022-11-05 16:30:00",
        "2022-11-05 13:30:00nyc": "2022-11-05 17:30:00",
        "2022-11-05 14:30:00nyc": "2022-11-05 18:30:00",
        "2022-11-05 15:30:00nyc": "2022-11-05 19:30:00",
        "2022-11-05 16:30:00nyc": "2022-11-05 20:30:00",
        "2022-11-05 17:30:00nyc": "2022-11-05 21:30:00",
        "2022-11-05 18:30:00nyc": "2022-11-05 22:30:00",
        "2022-11-05 19:30:00nyc": "2022-11-05 23:30:00",
        "2022-11-05 20:30:00nyc": "2022-11-06 00:30:00",
        "2022-11-05 21:30:00nyc": "2022-11-06 01:30:00",
        "2022-11-05 22:30:00nyc": "2022-11-06 02:30:00",
        "2022-11-05 23:30:00nyc": "2022-11-06 03:30:00",
        # next day
        "2022-11-06 00:30:00nyc": "2022-11-06 04:30:00",
        "2022-11-06 01:30:00nyc": "2022-11-06 05:30:00",  # this nyc datetime string corresponds to
        # two different timestamps-- by default, we pick the first.
        # daylight savings fall
        "2022-11-06 02:30:00nyc": "2022-11-06 07:30:00",
        "2022-11-06 03:30:00nyc": "2022-11-06 08:30:00",
        "2022-11-06 04:30:00nyc": "2022-11-06 09:30:00",
        "2022-11-06 05:30:00nyc": "2022-11-06 10:30:00",
        "2022-11-06 06:30:00nyc": "2022-11-06 11:30:00",
        "2022-11-06 07:30:00nyc": "2022-11-06 12:30:00",
        "2022-11-06 08:30:00nyc": "2022-11-06 13:30:00",
        "2022-11-06 09:30:00nyc": "2022-11-06 14:30:00",
        "2022-11-06 10:30:00nyc": "2022-11-06 15:30:00",
        "2022-11-06 11:30:00nyc": "2022-11-06 16:30:00",
        "2022-11-06 12:30:00nyc": "2022-11-06 17:30:00",
        "2022-11-06 13:30:00nyc": "2022-11-06 18:30:00",
        "2022-11-06 14:30:00nyc": "2022-11-06 19:30:00",
        "2022-11-06 15:30:00nyc": "2022-11-06 20:30:00",
        "2022-11-06 16:30:00nyc": "2022-11-06 21:30:00",
        "2022-11-06 17:30:00nyc": "2022-11-06 22:30:00",
        "2022-11-06 18:30:00nyc": "2022-11-06 23:30:00",
        "2022-11-06 19:30:00nyc": "2022-11-07 00:30:00",
        "2022-11-06 20:30:00nyc": "2022-11-07 01:30:00",
        "2022-11-06 21:30:00nyc": "2022-11-07 02:30:00",
        "2022-11-06 22:30:00nyc": "2022-11-07 03:30:00",
        "2022-11-06 23:30:00nyc": "2022-11-07 04:30:00",
    }

    for k, expected in nycDateStringsToUtcDateStrings.items():
        res = Timestamp.parse(k).format()
        assert res == expected, (res, expected)

    nycDateStringsToUtcDateStrings = {
        "2022-03-12 00:30:00nyc": "2022-03-12 05:30:00",
        "2022-03-12 01:30:00nyc": "2022-03-12 06:30:00",
        "2022-03-12 02:30:00nyc": "2022-03-12 07:30:00",
        "2022-03-12 03:30:00nyc": "2022-03-12 08:30:00",
        "2022-03-12 04:30:00nyc": "2022-03-12 09:30:00",
        "2022-03-12 05:30:00nyc": "2022-03-12 10:30:00",
        "2022-03-12 06:30:00nyc": "2022-03-12 11:30:00",
        "2022-03-12 07:30:00nyc": "2022-03-12 12:30:00",
        "2022-03-12 08:30:00nyc": "2022-03-12 13:30:00",
        "2022-03-12 09:30:00nyc": "2022-03-12 14:30:00",
        "2022-03-12 10:30:00nyc": "2022-03-12 15:30:00",
        "2022-03-12 11:30:00nyc": "2022-03-12 16:30:00",
        "2022-03-12 12:30:00nyc": "2022-03-12 17:30:00",
        "2022-03-12 13:30:00nyc": "2022-03-12 18:30:00",
        "2022-03-12 14:30:00nyc": "2022-03-12 19:30:00",
        "2022-03-12 15:30:00nyc": "2022-03-12 20:30:00",
        "2022-03-12 16:30:00nyc": "2022-03-12 21:30:00",
        "2022-03-12 17:30:00nyc": "2022-03-12 22:30:00",
        "2022-03-12 18:30:00nyc": "2022-03-12 23:30:00",
        "2022-03-12 19:30:00nyc": "2022-03-13 00:30:00",
        "2022-03-12 20:30:00nyc": "2022-03-13 01:30:00",
        "2022-03-12 21:30:00nyc": "2022-03-13 02:30:00",
        "2022-03-12 22:30:00nyc": "2022-03-13 03:30:00",
        "2022-03-12 23:30:00nyc": "2022-03-13 04:30:00",
        # next day
        "2022-03-13 00:30:00nyc": "2022-03-13 05:30:00",
        "2022-03-13 01:30:00nyc": "2022-03-13 06:30:00",
        # daylight3savings spring
        "2022-03-13 02:30:00nyc": "2022-03-13 06:30:00",  # this nyc datetime
        # string should not exist!
        "2022-03-13 03:30:00nyc": "2022-03-13 07:30:00",
        "2022-03-13 04:30:00nyc": "2022-03-13 08:30:00",
        "2022-03-13 05:30:00nyc": "2022-03-13 09:30:00",
        "2022-03-13 06:30:00nyc": "2022-03-13 10:30:00",
        "2022-03-13 07:30:00nyc": "2022-03-13 11:30:00",
        "2022-03-13 08:30:00nyc": "2022-03-13 12:30:00",
        "2022-03-13 09:30:00nyc": "2022-03-13 13:30:00",
        "2022-03-13 10:30:00nyc": "2022-03-13 14:30:00",
        "2022-03-13 11:30:00nyc": "2022-03-13 15:30:00",
        "2022-03-13 12:30:00nyc": "2022-03-13 16:30:00",
        "2022-03-13 13:30:00nyc": "2022-03-13 17:30:00",
        "2022-03-13 14:30:00nyc": "2022-03-13 18:30:00",
        "2022-03-13 15:30:00nyc": "2022-03-13 19:30:00",
        "2022-03-13 16:30:00nyc": "2022-03-13 20:30:00",
        "2022-03-13 17:30:00nyc": "2022-03-13 21:30:00",
        "2022-03-13 18:30:00nyc": "2022-03-13 22:30:00",
        "2022-03-13 19:30:00nyc": "2022-03-13 23:30:00",
        "2022-03-13 20:30:00nyc": "2022-03-14 00:30:00",
        "2022-03-13 21:30:00nyc": "2022-03-14 01:30:00",
        "2022-03-13 22:30:00nyc": "2022-03-14 02:30:00",
        "2022-03-13 23:30:00nyc": "2022-03-14 03:30:00",
    }

    for k, expected in nycDateStringsToUtcDateStrings.items():
        try:
            res = Timestamp.parse(k).format()
            assert res == expected, (res, expected)
        except NonexistentDateTime:
            assert k == "2022-03-13 02:30:00nyc"
