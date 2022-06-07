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

from typed_python import Entrypoint

# This file implements some useful low level algorithms for processing dates and times.
# Many of the algorithms are described here https://howardhinnant.github.io/date_algorithms.html


@Entrypoint
def days_from_civil(year: int = 0, month: int = 0, day: int = 0) -> int:
    '''
    Creates a unix timestamp from date values.
    Parameters:
        year (int): The year
        month (int): The month. January: 1, February: 2, ....
        day (int): The day
    Returns:
        seconds(float): The number of seconds

    Implements the low level days_from_civil algorithm
    '''
    year -= month <= 2
    era = (year if year >= 0 else year - 399) // 400
    yoe = (year - era * 400)
    doy = (153 * ( month - 3 if month > 2 else month + 9) + 2) // 5 + day - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    days = era * 146097 + doe - 719468

    return days


@Entrypoint
def date_to_seconds(year: int = 0, month: int = 0, day: int = 0) -> float:
    '''
        Creates a unix timestamp from date values.
        Parameters:
            year (int): The year
            month (int): The month. January: 1, February: 2, ....
            day (int): The day
        Returns:
            seconds(float): The number of seconds

    '''
    return days_from_civil(year, month, day) * 86400


@Entrypoint
def time_to_seconds(hour: int = 0, minute: int = 0, second: float = 0) -> float:
    '''
        Converts and hour, min, second combination into seconds
        Parameters:
            hour (int): The hour (0-23)
            minute (int): The minute
            second (int): The second
        Returns:
           (float) the number of seconds
    '''
    return (hour * 3600) + (minute * 60) + second


@Entrypoint
def weekday_difference(x: int, y: int) -> int:
    '''
        Gets the difference in days between two weekdays
        Parameters:
            x (int): The first day
            y (int): The second day

        Returns:
           (int) the difference between the two weekdays
    '''
    x -= y
    return x if x >= 0 and x <= 6 else x + 7


@Entrypoint
def weekday_from_days(z: int) -> int:
    '''
        Gets the day of week given the number of days from the unix epoch
        Parameters:
            z (int): The number of days from the epoch

        Returns:
           (int) the weekday (0-6)
    '''
    return (z + 4) % 7 if z >= -4 else (z + 5) % 7 + 6


@Entrypoint
def get_nth_dow_of_month_unixtime(n: int, wd: int, month: int, year: int) -> int:
    '''
        Gets the date of the nth day of the month for a given year. E.g. get 2nd Sat in July 2022
        Parameters:
            n (int): nth day of week (1-4).
            wd (int): the weekday (0-6) where 0 => Sunday
            month (int): the month (1-12)
            year (int): the year

        Returns:
           (int): The nth day of the month in unixtime
    '''
    if n < 1 or n > 4:
        raise ValueError('n should be 1-4')
    if wd < 0 or wd > 6:
        raise ValueError('wd should be 0-6')
    if month < 1 or month > 12:
        raise ValueError('invalid month')

    wd_1st = weekday_from_days(days_from_civil(year, month, 1))

    return date_to_seconds(year=year,
                           month=month,
                           day=weekday_difference(wd, wd_1st) + 1 + (n - 1) * 7)


@Entrypoint
def get_year_from_unixtime(ts: float) -> int:
    '''
        Gets the year from a unixtime
        Parameters:
            ts (float): the unix timestamp
        Returns:
           (int): The year
    '''
    z = int(ts // 86400 + 719468)
    era = (z if z >= 0 else z - 146096) // 146097
    doe = z - era * 146097
    yoe = (doe - (doe // 1460) + (doe // 36524) - (doe // 146096)) // 365
    y = yoe + era * 400
    doy = doe - ((365 * yoe) + (yoe // 4) - (yoe // 100))
    mp = (5 * doy + 2) // 153
    m = mp + (3 if mp < 10 else -9)
    y += (m <= 2)
    return y


@Entrypoint
def is_leap_year(year: int):
    '''
    Tests if a year is a leap year.
    Parameters:
        year(int): The year
    Returns:
        True if the year is a leap year, False otherwise
    '''
    return (year % 4 == 0 and year % 100 != 0) or year % 400 == 0


@Entrypoint
def convert_to_12h(hour: int):
    return 12 if (hour == 0 or hour == 12 or hour == 24) else (hour if hour < 12 else hour - 12)


@Entrypoint
def is_date(year: int, month: int, day: int) -> bool:
    '''
    Tests if a year, month, day combination is a valid date. Year is required.
    Month and day are optional. If day is present, month is required.
    Parameters:
        year (int): The year
        month (int): The month (January=1)
        day (int): The day of the month
    Returns:
        True if the date is valid, False otherwise
    '''
    hasYear, hasMonth, hasDay = year > -1, month > -1, day > -1

    if not hasYear:
        return False
    if hasMonth and not hasYear:
        return False
    if hasDay and not hasMonth:
        return False
    if hasMonth and (month < 1 or month > 12):
        return False

    if hasDay:
        if day < 1:
            return False
        elif month == 1 or month == 3 or month == 5 or month == 7 or month == 8 or month == 10 or month == 12:
            return day < 32
        elif month == 4 or month == 6 or month == 9 or month == 11:
            return day < 31
        elif month == 2:
            if ((year % 4 == 0 and year % 100 != 0) or year % 400 == 0):
                return day < 30
            return day < 29

    return True


@Entrypoint
def is_time(hour: int, min: int, sec: float) -> bool:
    '''
    Tests if a hour, min, sec combination is a valid time.
    Parameters:
        hour(int): The hour
        min(int): The min
        sec(float): The second
    Returns:
        True if the time is valid, False otherwise
    '''
    # '24' is valid alternative to '0' but only when min and sec are both 0
    if hour < 0 or hour > 24 or (hour == 24 and (min != 0 or sec != 0)):
        return False
    elif min < 0 or min > 59 or sec < 0 or sec >= 60:
        return False
    return True


@Entrypoint
def is_datetime(year: int, month: int, day: int, hour: float, min: float, sec: float) -> bool:
    '''
    Tests if a year, month, day hour, min, sec combination is a valid date time.
    Parameters:
        year (int): The year
        month (int): The month (January=>1)
        day (int): The day of the month
        hour(int): The hour
        min(int): The min
        sec(float): The second
    Returns:
        True if the datetime is valid, False otherwise
    '''
    return is_date(year, month, day) and is_time(hour, min, sec)
