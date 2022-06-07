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

from typed_python import Entrypoint, Dict, Class
from typed_python.lib.datetime.chrono import get_nth_dow_of_month_unixtime, get_year_from_unixtime


@Entrypoint
def is_us_ca_dst(ts: float) -> bool:
    '''
        Tests if a timestamp falls within Daylight Saving Time or Standard Time in the United States or Canada
        Parameters:
            ts(float): the timestamp
        Returns:
            (boolean): True if the ts is in DST, false otherwise
    '''
    year = get_year_from_unixtime(ts)

    # 2:00am second Sunday march
    ds_start = get_nth_dow_of_month_unixtime(2, 0, 3, year) + 7200

    # 2:00 am first sunday in november
    ds_end = get_nth_dow_of_month_unixtime(1, 0, 11, year) + 7200

    return ts >= ds_start and ts <= ds_end


class Timezone(Class):
    @Entrypoint
    @staticmethod
    def get_offset(_: int) -> int:
        raise NotImplementedError


class UTC(Timezone):
    @Entrypoint
    @staticmethod
    def get_offset(_: int = 0):
        return 0


class UTC_MINUS_0400(Timezone):
    @Entrypoint
    @staticmethod
    def get_offset(_: int = 0):
        return 14400


class UTC_MINUS_0500(Timezone):
    @Entrypoint
    @staticmethod
    def get_offset(_: int = 0):
        return 18000


class UTC_MINUS_0600(Timezone):
    @Entrypoint
    @staticmethod
    def get_offset(_: int = 0):
        return 21600


class UTC_MINUS_0700(Timezone):
    @Entrypoint
    @staticmethod
    def get_offset(_: int = 0):
        return 25200


class UTC_MINUS_0800(Timezone):
    @Entrypoint
    @staticmethod
    def get_offset(_: int = 0):
        return 28800


class CT(Timezone):
    @Entrypoint
    @staticmethod
    def get_offset(ts):
        return UTC_MINUS_0500.get_offset() if is_us_ca_dst(ts) else UTC_MINUS_0600.get_offset()


class ET(Timezone):
    @Entrypoint
    @staticmethod
    def get_offset(ts):
        return UTC_MINUS_0400.get_offset() if is_us_ca_dst(ts) else UTC_MINUS_0500.get_offset()


class MT(Timezone):
    @Entrypoint
    @staticmethod
    def get_offset(ts):
        return UTC_MINUS_0600.get_offset() if is_us_ca_dst(ts) else UTC_MINUS_0700.get_offset()


class PT(Timezone):
    @Entrypoint
    @staticmethod
    def get_offset(ts):
        return UTC_MINUS_0700.get_offset() if is_us_ca_dst(ts) else UTC_MINUS_0800.get_offset()


TZ_ABBR_TO_TIMEZONE = Dict(str, Timezone)({
    '': UTC(),
    '+0000': UTC(),
    'cdt': UTC_MINUS_0500(),
    'cst': UTC_MINUS_0600(),
    'ct': CT(),
    'edt': UTC_MINUS_0400(),
    'est': UTC_MINUS_0500(),
    'et': ET(),
    'gmt': UTC(),
    'mdt': UTC_MINUS_0600(),
    'mst': UTC_MINUS_0700(),
    'mt': MT(),
    'nyc': ET(),
    'pdt': UTC_MINUS_0700(),
    'pst': UTC_MINUS_0800(),
    'pt': PT(),
    'utc': UTC(),
    'z': UTC(),
})


@Entrypoint
def tz_abbr_to_utc_offset(tz_abbr: str, unixtime: int) -> int:
    '''
       Get utc offset by timezone abbreviation
       Parameters:
            tz_abbr(string): a timezone indicator. examples: 'ET', 'EST', 'NYC'
       Returns:
            (int): The utc offset in seconds
    '''
    return TZ_ABBR_TO_TIMEZONE[tz_abbr.lower()].get_offset(unixtime)


@Entrypoint
def tz_string_to_utc_offset(tz_str: str, unixtime: int) -> int:
    '''
       Get utc offset by timezone abbreviation
       Parameters:
            tz_abbr(string): a timezone indicator. examples: 'ET', 'EST', 'NYC'
       Returns:
            (int): The utc offset in seconds
    '''
    return TZ_ABBR_TO_TIMEZONE[tz_str.lower()].get_offset(unixtime)


@Entrypoint
def is_supported_tz_abbr(tz_abbr: str):
    return tz_abbr.lower() in TZ_ABBR_TO_TIMEZONE
