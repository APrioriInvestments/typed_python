from apriori.security_util.holidays.date_string_util import userStringToTimestamp 
import datetime
import pytz

from typed_python.lib.timestamp import Timestamp
from typed_python.lib.datetime.timezone import Timezone



def test_timestamps():

    oneThirtyAm = userStringToTimestamp('2022-11-06 01:30:00')

    oneThirtyAmTimestamp = Timestamp.make(oneThirtyAm)

    print(oneThirtyAmTimestamp.format())
    import pdb; pdb.set_trace()

    oneThirtyAmTimestamp = Timestamp.make(oneThirtyAm)


def test_EDT():

    # the utc timestamp corresponding to the time of day 
    # at 1:30 am on 6 nov 22 in EDT (whose offset is +14400)
    tz_str = 'edt'
    ts = Timestamp.parse('2022-11-06 01:30:00'+tz_str)


    offset = Timezone.TZ_STR_TO_OFFSET[tz_str].get_offset(0.0)
    print(offset)
    print(ts.format(-offset))


def test_EST():
    tz = pytz.timezone("America/Atikokan")
    externalTimestamp = tz.localize(datetime.datetime(2022, 11, 6, 1, 30,0)).timestamp()
    internalTimestamp = Timestamp.parse('2022-11-06 01:30:00est')
    assert externalTimestamp == internalTimestamp.ts, externalTimestamp - internalTimestamp.ts


def test_timestamp_parse_around_daylight_savings_switch():
    nycDateStringsToUtcDateStrings= {
        '2022-11-05 00:30:00nyc': '2022-11-05 04:30:00',
        '2022-11-05 01:30:00nyc': '2022-11-05 05:30:00',
        '2022-11-05 02:30:00nyc': '2022-11-05 06:30:00',
        '2022-11-05 03:30:00nyc': '2022-11-05 07:30:00',
        '2022-11-05 04:30:00nyc': '2022-11-05 08:30:00',
        '2022-11-05 05:30:00nyc': '2022-11-05 09:30:00',
        '2022-11-05 06:30:00nyc': '2022-11-05 10:30:00',
        '2022-11-05 07:30:00nyc': '2022-11-05 11:30:00',
        '2022-11-05 08:30:00nyc': '2022-11-05 12:30:00',
        '2022-11-05 09:30:00nyc': '2022-11-05 13:30:00',
        '2022-11-05 10:30:00nyc': '2022-11-05 14:30:00',
        '2022-11-05 11:30:00nyc': '2022-11-05 15:30:00',
        '2022-11-05 12:30:00nyc': '2022-11-05 16:30:00',
        '2022-11-05 13:30:00nyc': '2022-11-05 17:30:00',
        '2022-11-05 14:30:00nyc': '2022-11-05 18:30:00',
        '2022-11-05 15:30:00nyc': '2022-11-05 19:30:00',
        '2022-11-05 16:30:00nyc': '2022-11-05 20:30:00',
        '2022-11-05 17:30:00nyc': '2022-11-05 21:30:00',
        '2022-11-05 18:30:00nyc': '2022-11-05 22:30:00',
        '2022-11-05 19:30:00nyc': '2022-11-05 23:30:00',
        '2022-11-05 20:30:00nyc': '2022-11-06 00:30:00',
        '2022-11-05 21:30:00nyc': '2022-11-06 01:30:00',
        '2022-11-05 22:30:00nyc': '2022-11-06 02:30:00',
        '2022-11-05 23:30:00nyc': '2022-11-06 03:30:00',
        # next day
        '2022-11-06 00:30:00nyc': '2022-11-06 04:30:00',
        '2022-11-06 01:30:00nyc': '2022-11-06 05:30:00', # this nyc datetime string corresponds to two different timestamps-- it looks like we pick the first.
        # daylight savings fall
        '2022-11-06 02:30:00nyc': '2022-11-06 07:30:00',
        '2022-11-06 03:30:00nyc': '2022-11-06 08:30:00',
        '2022-11-06 04:30:00nyc': '2022-11-06 09:30:00',
        '2022-11-06 05:30:00nyc': '2022-11-06 10:30:00',
        '2022-11-06 06:30:00nyc': '2022-11-06 11:30:00',
        '2022-11-06 07:30:00nyc': '2022-11-06 12:30:00',
        '2022-11-06 08:30:00nyc': '2022-11-06 13:30:00',
        '2022-11-06 09:30:00nyc': '2022-11-06 14:30:00',
        '2022-11-06 10:30:00nyc': '2022-11-06 15:30:00',
        '2022-11-06 11:30:00nyc': '2022-11-06 16:30:00',
        '2022-11-06 12:30:00nyc': '2022-11-06 17:30:00',
        '2022-11-06 13:30:00nyc': '2022-11-06 18:30:00',
        '2022-11-06 14:30:00nyc': '2022-11-06 19:30:00',
        '2022-11-06 15:30:00nyc': '2022-11-06 20:30:00',
        '2022-11-06 16:30:00nyc': '2022-11-06 21:30:00',
        '2022-11-06 17:30:00nyc': '2022-11-06 22:30:00',
        '2022-11-06 18:30:00nyc': '2022-11-06 23:30:00',
        '2022-11-06 19:30:00nyc': '2022-11-07 00:30:00',
        '2022-11-06 20:30:00nyc': '2022-11-07 01:30:00',
        '2022-11-06 21:30:00nyc': '2022-11-07 02:30:00',
        '2022-11-06 22:30:00nyc': '2022-11-07 03:30:00',
        '2022-11-06 23:30:00nyc': '2022-11-07 04:30:00',
    }

    for k, expected in nycDateStringsToUtcDateStrings.items():
        res = Timestamp.parse(k).format()
        assert res == expected, (res, expected)

    nycDateStringsToUtcDateStrings= {
        '2022-03-12 00:30:00nyc': '2022-03-12 05:30:00',
        '2022-03-12 01:30:00nyc': '2022-03-12 06:30:00',
        '2022-03-12 02:30:00nyc': '2022-03-12 07:30:00',
        '2022-03-12 03:30:00nyc': '2022-03-12 08:30:00',
        '2022-03-12 04:30:00nyc': '2022-03-12 09:30:00',
        '2022-03-12 05:30:00nyc': '2022-03-12 10:30:00',
        '2022-03-12 06:30:00nyc': '2022-03-12 11:30:00',
        '2022-03-12 07:30:00nyc': '2022-03-12 12:30:00',
        '2022-03-12 08:30:00nyc': '2022-03-12 13:30:00',
        '2022-03-12 09:30:00nyc': '2022-03-12 14:30:00',
        '2022-03-12 10:30:00nyc': '2022-03-12 15:30:00',
        '2022-03-12 11:30:00nyc': '2022-03-12 16:30:00',
        '2022-03-12 12:30:00nyc': '2022-03-12 17:30:00',
        '2022-03-12 13:30:00nyc': '2022-03-12 18:30:00',
        '2022-03-12 14:30:00nyc': '2022-03-12 19:30:00',
        '2022-03-12 15:30:00nyc': '2022-03-12 20:30:00',
        '2022-03-12 16:30:00nyc': '2022-03-12 21:30:00',
        '2022-03-12 17:30:00nyc': '2022-03-12 22:30:00',
        '2022-03-12 18:30:00nyc': '2022-03-12 23:30:00',
        '2022-03-12 19:30:00nyc': '2022-03-13 00:30:00',
        '2022-03-12 20:30:00nyc': '2022-03-13 01:30:00',
        '2022-03-12 21:30:00nyc': '2022-03-13 02:30:00',
        '2022-03-12 22:30:00nyc': '2022-03-13 03:30:00',
        '2022-03-12 23:30:00nyc': '2022-03-13 04:30:00',
        # next day
        '2022-03-13 00:30:00nyc': '2022-03-13 05:30:00',
        '2022-03-13 01:30:00nyc': '2022-03-13 06:30:00',
        # daylight3savings spring
        '2022-03-13 02:30:00nyc': '2022-03-13 06:30:00', # this nyc datetime string should not exist!
        '2022-03-13 03:30:00nyc': '2022-03-13 07:30:00',
        '2022-03-13 04:30:00nyc': '2022-03-13 08:30:00',
        '2022-03-13 05:30:00nyc': '2022-03-13 09:30:00',
        '2022-03-13 06:30:00nyc': '2022-03-13 10:30:00',
        '2022-03-13 07:30:00nyc': '2022-03-13 11:30:00',
        '2022-03-13 08:30:00nyc': '2022-03-13 12:30:00',
        '2022-03-13 09:30:00nyc': '2022-03-13 13:30:00',
        '2022-03-13 10:30:00nyc': '2022-03-13 14:30:00',
        '2022-03-13 11:30:00nyc': '2022-03-13 15:30:00',
        '2022-03-13 12:30:00nyc': '2022-03-13 16:30:00',
        '2022-03-13 13:30:00nyc': '2022-03-13 17:30:00',
        '2022-03-13 14:30:00nyc': '2022-03-13 18:30:00',
        '2022-03-13 15:30:00nyc': '2022-03-13 19:30:00',
        '2022-03-13 16:30:00nyc': '2022-03-13 20:30:00',
        '2022-03-13 17:30:00nyc': '2022-03-13 21:30:00',
        '2022-03-13 18:30:00nyc': '2022-03-13 22:30:00',
        '2022-03-13 19:30:00nyc': '2022-03-13 23:30:00',
        '2022-03-13 20:30:00nyc': '2022-03-14 00:30:00',
        '2022-03-13 21:30:00nyc': '2022-03-14 01:30:00',
        '2022-03-13 22:30:00nyc': '2022-03-14 02:30:00',
        '2022-03-13 23:30:00nyc': '2022-03-14 03:30:00',
    }

    for k, expected in nycDateStringsToUtcDateStrings.items():
        res = Timestamp.parse(k).format()
        assert res == expected, (res, expected)


def test_non_existent_datestrings():
    tz = pytz.timezone("America/New_York")
    one= tz.localize(datetime.datetime(2022, 3, 13, 1, 0,0))
    oneOOne= tz.localize(datetime.datetime(2022, 3, 13, 1, 1,0))
    oneThirty = tz.localize(datetime.datetime(2022, 3, 13, 1, 30,0))
    two = tz.localize(datetime.datetime(2022, 3, 13, 2, 0,0))
    twoOOne = tz.localize(datetime.datetime(2022, 3, 13, 2, 1,0))
    twoThirty = tz.localize(datetime.datetime(2022, 3, 13, 2, 0,0))
    twoFortyFive = tz.localize(datetime.datetime(2022, 3, 13, 2, 45,0))
    three = tz.localize(datetime.datetime(2022, 3, 13, 3, 0, 0))
    threeThirty = tz.localize(datetime.datetime(2022, 3, 13, 3, 30,0))


    twoOOne.timestamp() - oneOOne.timestamp()

    twoThirty.timestamp() - oneThirty.timestamp()
    threeThirty.timestamp() - twoThirty.timestamp()
    twoFortyFive.timestamp() - twoThirty.timestamp()
    three.timestamp() - twoFortyFive.timestamp()
    two.timestamp() - one.timestamp()
    three.timestamp() - two.timestamp()


    twoThirty.timestamp() - oneThirty.timestamp()
    threeThirty.timestamp() - twoThirty.timestamp()



    import pdb; pdb.set_trace()
    print(f"dtime:\n{dtime}")




