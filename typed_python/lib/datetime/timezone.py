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

from typed_python import Class, Dict, Entrypoint, Member, Final
from typed_python.lib.datetime.chrono import Chrono


class Offset(Class):
    @Entrypoint
    def get_offset(self, _: int) -> int:
        raise NotImplementedError


class FixedOffset(Offset, Final):
    offset = Member(int)

    @Entrypoint
    def __init__(self, offset: int):
        self.offset = offset

    @Entrypoint
    def get_offset(self, _: int) -> int:
        return self.offset

# Would like to give RelativeOffset* better names but...
#    1) RelativeOffset_BS_BM_CA_MX_US.... is clunky
#    2) can't easily name start/end of dst, because other regions have same start dates but different times
#
# For nowe we'll just add new 'regions' as necessary. E.g RelativeOffset3: Cuba, RelativeOffset3: EU + lots
# more europe


class RelativeOffset1(Offset, Final):
    """
    Timezones where the offset from UTC changes with DST beginning on the second Sunday in March at 2:00
    and ending on the first Sunday in November at 2:00.

    Covers: Bahamas, Bermuda, parts of Canada, parts of Greenland, Haiti, parts of Mexico, Saint Pierre and
    Miquelon, Turks and Caicos Islands, parts of the United States
    """

    st_offset = Member(int)
    dt_offset = Member(int)

    @Entrypoint
    def __init__(self, st_offset: int, dt_offset: int):
        self.st_offset = st_offset
        self.dt_offset = dt_offset

    @Entrypoint
    def get_offset(self, ts: int) -> int:
        return self.dt_offset if self.is_dst(ts) else self.st_offset

    @Entrypoint
    def is_dst(self, ts: float) -> bool:
        year = Chrono.year_from_ts(ts)

        # 2:00am second Sunday march
        ds_start = Chrono.get_nth_dow_of_month_ts(2, 0, 3, year) + 7200

        # 2:00 am first sunday in november
        ds_end = Chrono.get_nth_dow_of_month_ts(1, 0, 11, year) + 7200

        return ts >= ds_start and ts <= ds_end


class Timezone(Class, Final):
    AT = RelativeOffset1(st_offset=21600, dt_offset=18000)

    ADT = FixedOffset(offset=18000)
    AST = FixedOffset(offset=21600)

    CT = RelativeOffset1(st_offset=21600, dt_offset=18000)
    CDT = FixedOffset(offset=18000)
    CST = FixedOffset(offset=21600)

    ET = RelativeOffset1(st_offset=18000, dt_offset=14400)
    EDT = FixedOffset(offset=14400)
    EST = FixedOffset(offset=18000)

    MT = RelativeOffset1(st_offset=25200, dt_offset=21600)
    MDT = FixedOffset(offset=21600)
    MST = FixedOffset(offset=25200)

    PT = RelativeOffset1(st_offset=28800, dt_offset=25200)
    PDT = FixedOffset(offset=25200)
    PST = FixedOffset(offset=28800)

    UTC = FixedOffset(offset=0)

    TZ_STR_TO_OFFSET = Dict(str, Offset)({
        '': UTC,
        '+0000': UTC,
        'cdt': CDT,
        'cst': CST,
        'ct': CT,
        'edt': EDT,
        'est': EST,
        'et': ET,
        'gmt': UTC,
        'mdt': MDT,
        'mst': MST,
        'mt': MT,
        'nyc': ET,
        'pdt': PDT,
        'pst': PST,
        'pt': PT,
        'utc': UTC,
        'z': UTC,
    })

    @Entrypoint
    @staticmethod
    def tz_str_to_utc_offset(tz_str: str, unixtime: int) -> int:
        '''
        Get utc offset by timezone abbreviation
          Parameters:
            tz_abbr(string): a timezone indicator. examples: 'ET', 'EST', 'NYC'
          Returns:
            (int): The utc offset in seconds
        '''
        return Timezone.TZ_STR_TO_OFFSET[tz_str.lower()].get_offset(unixtime)

    @Entrypoint
    @staticmethod
    def ts_to_utc(ts: float, tz_str: str = '') -> float:
        '''
        Converts a timestamp to its equivalent in UTC
          Parameters:
            ts (float): A unix timetamp
            tz_str(str): A timezone abbreviation (e.g. est, edt, nyc) or an ISO 8601 timezone offset (e.g. +0000 or -0101)
          Returns:
            (int): The utc offset in seconds
        '''
        tz_str = tz_str.lower()

        if tz_str in Timezone.TZ_STR_TO_OFFSET:
            return ts + Timezone.TZ_STR_TO_OFFSET[tz_str].get_offset(ts)

        return ts + Timezone.parse_tz_offset(tz_str)

    @Entrypoint
    @staticmethod
    def is_valid_tz_string(tz_str: str):
        '''
        Tests if a string represents a supported tz
        '''
        return tz_str.lower() in Timezone.TZ_STR_TO_OFFSET

    @Entrypoint
    @staticmethod
    def is_valid_tz_offset(hour: int, min: int, second: float = 0.0) -> bool:
        '''
        Tests if an hour,min combination is a valid offset from UTC
        Parameters:
          hour(int): The hour
          min(int): The minute
        Returns:
          True if the inputs are in the range UTC-12:00 to UTC+14
        '''
        if hour > 14 or hour < -12:
            return False

        if (hour == 14 or hour == -12) and min > 0:
            return False

        if min < 0 or min + second >= 60:
            return False

        return True

    @Entrypoint
    @staticmethod
    def parse_tz_offset(offset: str) -> int:
        '''
        Converts a set of tokens representing a timezone offset to seconds.
        Parameters:
            tokens (ListOf(str)): A set of string tokens representing a timezone. E.g. ['Z'] or ['+', '02', ':', '23']
        Returns:
            (int): The offset in seconds
        '''

        if offset[0] != '+' and offset[0] != '-':
            raise ValueError("tz offset must begin with '+' or '-'", offset)

        sign = offset[0]
        value = offset[1:]

        hour, min, second = 0, 0, 0

        # [+|-]HH or [+|-]HHMM
        if len(value) == 2:
            hour, min, second = int(value), 0, 0.0
        elif len(value) == 4:
            hour, min, second = int(value[:2]), int(value[2:4]), 0.0
        elif len(value) >= 6:
            hour, min, second = int(value[:2]), int(value[2:4]), float(value[6:])

        hour = hour * -1 if sign == '-' else hour

        if Timezone.is_valid_tz_offset(hour, min, second):
            return hour * 3600 + (min * 60 if hour > 0 else min * -60) + (second if hour > 0 else second * -1)
        else:
            raise ValueError('Invalid tz offset: ')
