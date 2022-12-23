import pytest
import pytz
import datetime
from typed_python.lib.datetime.date_time import (
    Date,
    DateTime,
    TimeOfDay,
    NonexistentDateTime,
    FixedOffsetTimezone,
    EST,
    NYC,
    CHI,
    UTC,
    last_weekday_of_month,
    DaylightSavingsTimezone,
    SwitchOffsetTimezone,
    OneFoldOnlyError,
)
from typed_python import Timestamp
from typed_python.lib.datetime.date_parser_test import get_datetimes_in_range


def test_last_weekday_of_month():
    assert last_weekday_of_month(2023, 1, 2) == Date(2023, 1, 31)
    assert last_weekday_of_month(2023, 1, 1) == Date(2023, 1, 30)
    assert last_weekday_of_month(2023, 1, 0) == Date(2023, 1, 29)
    assert last_weekday_of_month(2023, 1, 6) == Date(2023, 1, 28)
    assert last_weekday_of_month(2023, 1, 5) == Date(2023, 1, 27)
    assert last_weekday_of_month(2023, 1, 4) == Date(2023, 1, 26)
    assert last_weekday_of_month(2023, 1, 3) == Date(2023, 1, 25)

    assert last_weekday_of_month(1965, 2, 0) == Date(1965, 2, 28)
    assert last_weekday_of_month(1965, 2, 6) == Date(1965, 2, 27)
    assert last_weekday_of_month(1965, 2, 5) == Date(1965, 2, 26)
    assert last_weekday_of_month(1965, 2, 4) == Date(1965, 2, 25)
    assert last_weekday_of_month(1965, 2, 3) == Date(1965, 2, 24)
    assert last_weekday_of_month(1965, 2, 2) == Date(1965, 2, 23)
    assert last_weekday_of_month(1965, 2, 1) == Date(1965, 2, 22)


def test_DateTime_str():
    assert str(DateTime(2022, 1, 10, 10, 7, 45.12432)) == "2022-01-10 10:07:45"


def test_DateTime_to_timestamp():
    ymdhms = (2022, 12, 23, 18, 40, 46)
    dateTime = DateTime(*ymdhms)

    timestamp = NYC.timestamp(dateTime)

    tz = pytz.timezone("America/New_York")
    ts = tz.localize(datetime.datetime(*ymdhms)).timestamp()

    assert timestamp == ts


def test_DateTime_nonexistent_DateTime():
    dateTime = DateTime(2022, 3, 13, 2, 30, 0)

    with pytest.raises(NonexistentDateTime):
        NYC.timestamp(dateTime)


def test_DateTime_from_timestamp():
    dateTime = DateTime(2022, 11, 6, 1, 30, 0)

    timestamp = UTC.timestamp(dateTime)
    newDateTime = UTC.datetime(timestamp)
    assert newDateTime == dateTime


def test_DateTime_to_timestamp_daylight_savings():
    utcDateTime = DateTime(2022, 11, 6, 5, 30, 0)
    oneThirtyAmNycFirstFold = UTC.timestamp(utcDateTime)

    utcDateTime = DateTime(2022, 11, 6, 6, 30, 0)
    oneThirtyAmNycSecondFold = UTC.timestamp(utcDateTime)

    assert oneThirtyAmNycSecondFold - oneThirtyAmNycFirstFold == 3600
    assert NYC.datetime(oneThirtyAmNycFirstFold) == NYC.datetime(
        oneThirtyAmNycSecondFold
    )


def test_fixed_offset():
    ymdhms = (2022, 11, 6, 5, 30, 0)

    ts = (
        pytz.timezone("Asia/Hong_Kong").localize(datetime.datetime(*ymdhms)).timestamp()
    )

    dateTime = DateTime(*ymdhms)
    res = FixedOffsetTimezone(offset_hours=+8).timestamp(dateTime)
    assert ts == res


def test_datetime_to_timestamp_and_back():
    for tz in [NYC, EST, UTC]:
        dtime = DateTime(2022, 11, 6, 1, 30, 0)
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


def test_TimeOfDay():
    assert TimeOfDay(12, 3, 30) < TimeOfDay(12, 3, 31)
    assert TimeOfDay(11, 10, 30) < TimeOfDay(12, 3, 31)
    assert TimeOfDay(13, 10, 30) > TimeOfDay(12, 3, 31)
    assert str(TimeOfDay(13, 10, 30)) == "13:10:30"


def test_afterFold_dst_end():
    with pytest.raises(OneFoldOnlyError):
        NYC.timestamp(DateTime(2022, 11, 2, 1, 30, 0), afterFold=True)

    ymdhms = (2022, 11, 6, 1, 30, 0)
    tsSecondFold = (
        pytz.timezone("America/New_York")
        .localize(datetime.datetime(*ymdhms))
        .timestamp()
    )
    assert NYC.timestamp(DateTime(*ymdhms), afterFold=False) == tsSecondFold - 3600
    assert NYC.timestamp(DateTime(*ymdhms), afterFold=True) == tsSecondFold


def test_nyc_1918_10_27():
    nycDateStringsToUtcDateStrings = {
        "1918-10-27 00:30:00nyc": "1918-10-27 04:30:00",
        "1918-10-27 01:30:00nyc": "1918-10-27 05:30:00",
        # daylight savings fall back
        "1918-10-27 02:30:00nyc": "1918-10-27 07:30:00",
        "1918-10-27 03:30:00nyc": "1918-10-27 08:30:00",
        "1918-10-27 04:30:00nyc": "1918-10-27 09:30:00",
    }

    for k, expected in nycDateStringsToUtcDateStrings.items():
        res = Timestamp.parse(k).format()
        assert res == expected, (res, expected)


def test_nyc_since_1902():
    nyc = pytz.timezone("America/New_York")
    datetimes = get_datetimes_in_range(
        start=datetime.datetime(1902, 1, 1, 20, 0, 1, 0),
        end=datetime.datetime(2030, 1, 1, 20, 0, 1, 0),
        step="hours",
    )

    lastYear = None
    for dt in datetimes:
        if dt.year != lastYear:
            print(f"Checking year {dt.year}")
        lastYear = dt.year

        them = nyc.localize(dt).timestamp()

        try:
            dateTime = DateTime(
                dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second
            )
            us = NYC.timestamp(dateTime)

        except NonexistentDateTime:
            continue

        try:
            assert us == them, f"Difference for {str(dt)} is {us - them} seconds"

        except AssertionError:
            tz = NYC.chooseTimezone(dt.year)
            if isinstance(tz, DaylightSavingsTimezone):
                dst_end = tz.dst_boundaries.getDaylightSavingsEnd(dt.year)
                if (
                    dst_end.date == dateTime.date
                    and dst_end.timeOfDay.hour - 1 == dateTime.timeOfDay.hour
                ):
                    if (
                        us - them == -3600
                    ):  # we choose the first fold and they choose the second.
                        continue
            elif isinstance(tz, SwitchOffsetTimezone):
                switch = tz.switch_datetime
                if (
                    switch.date == dateTime.date
                    and switch.timeOfDay.hour - 1 == dateTime.timeOfDay.hour
                ):
                    if (
                        us - them == -3600
                    ):  # we choose the first fold and they choose the second.
                        continue

            raise


def test_nyc_vs_chi():
    ts_chi = CHI.timestamp(DateTime(2019, 7, 2, 8, 30, 0))
    ts_nyc = NYC.timestamp(DateTime(2019, 7, 2, 8, 30, 0))
    assert ts_nyc - ts_chi == -3600


def test_Date_methods():
    assert Date(1970, 1, 1).daysSinceEpoch() == 0
    assert Date(1970, 1, 2).daysSinceEpoch() == 1
    assert Date(1970, 1, 1) - Date(1970, 1, 2) == -1
    assert Date(2000, 12, 31).dayOfYear() == 366  # leap year
    assert Date(2000, 12, 31).nextMonthStart() == Date(2001, 1, 1)
    assert Date(2000, 12, 30).nextMonthStart() == Date(2001, 1, 1)
