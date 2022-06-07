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

from typed_python import Entrypoint, Dict
from typed_python.lib.datetime.chrono import get_nth_dow_of_month_unixtime, get_year_from_unixtime


@Entrypoint
def is_tz_offset(hour: int, min: int, second: float = 0.0) -> bool:
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
def is_us_ca_dst(ts: float) -> bool:
    '''
        Checks if a timestamp falls within DST or ST in The US or Canada
        Parameters:
            ts: a float
        Returns:
            (boolean): True if the ts is in DST, false otherwise
    '''
    year = get_year_from_unixtime(ts)

    # 2:00am second Sunday march
    ds_start = get_nth_dow_of_month_unixtime(2, 0, 3, year) + (2 * 60 * 60)

    # 2:00 am first sunday in november
    ds_end = get_nth_dow_of_month_unixtime(1, 0, 11, year) + (2 * 60 * 60)

    return ts >= ds_start and ts <= ds_end


@Entrypoint
def et(ts: float) -> int:
    '''
        Get utc offset for the given Eastern Time timestamp
        Parameters:
            ts: a float
        Returns:
            (int): The utc offset in seconds
    '''
    return 14400 if is_us_ca_dst(ts) else 18000


@Entrypoint
def pt(ts: float) -> int:
    '''
        Get utc offset for the given Pacific Time timestamp
        Parameters:
            ts: a float
        Returns:
            (int): The utc offset in seconds
    '''
    return 25200 if is_us_ca_dst(ts) else 28800


relative_offsets = Dict(str, object)({
    'et': et,
    'nyc': et,
    'pt': pt,
})

offsets = Dict(str, int)({
    'edt': 18000,
    'est': 14400,
    'gmt': 0,
    'pdt': 25200,
    'pst': 28800,
    'utc': 0,
    'z': 0
})


@Entrypoint
def utc_offset_by_tz_abbreviation(tzname: str, unixtime: int):
    """Get utc offset by timezone abbreviation"""
    tz = tzname.lower()
    return offsets[tz] if tz in offsets else relative_offsets[tz](unixtime)
