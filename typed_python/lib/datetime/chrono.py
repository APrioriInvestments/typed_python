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

from typed_python import Class, Entrypoint, Final


class Chrono(Class, Final):
    """
      Implements a number of useful algorithms for working with dates and times.
    """

    @Entrypoint
    @staticmethod
    def days_from_civil(year: int = 0, month: int = 0, day: int = 0) -> int:
        """
        Creates a unix timestamp from year, month, day values.
        Parameters:
            year (int): The year
            month (int): The month. January: 1, February: 2, ....
            day (int): The day
        Returns:
            seconds(float): The number of seconds
        """
        # Implements the days_from_civil algorithm described here:
        # https://howardhinnant.github.io/date_algorithms.html#days_from_civil

        year -= month <= 2
        era = (year if year >= 0 else year - 399) // 400
        # year of the era
        yoe = year - era * 400

        # day of the year
        doy = (153 * (month - 3 if month > 2 else month + 9) + 2) // 5 + day - 1

        # day of the era
        doe = yoe * 365 + yoe // 4 - yoe // 100 + doy

        # number of days since epch
        days = era * 146097 + doe - 719468

        return days

    @Entrypoint
    @staticmethod
    def date_to_seconds(year: int = 0, month: int = 0, day: int = 0) -> float:
        """
            Creates a unix timestamp from date values.
            Parameters:
                year (int): The year
                month (int): The month. January: 1, February: 2, ....
                day (int): The day
            Returns:
                seconds(float): The number of seconds

        """
        return Chrono.days_from_civil(year, month, day) * 86400

    @Entrypoint
    @staticmethod
    def time_to_seconds(hour: int = 0, minute: int = 0, second: float = 0) -> float:
        """
            Converts and hour, min, second combination into seconds
            Parameters:
                hour (int): The hour (0-23)
                minute (int): The minute
                second (float): The second
            Returns:
                (float) the number of seconds
        """
        return (hour * 3600) + (minute * 60) + second

    @Entrypoint
    @staticmethod
    def weekday_difference(day1: int, day2: int) -> int:
        """
            Gets the difference in days between two weekdays
            Parameters:
                day1 (int): The first day
                day2 (int): The second day

            Returns:
               (int) the difference between the two weekdays
        """
        day1 -= day2
        return day1 if day1 >= 0 and day1 <= 6 else day1 + 7

    @Entrypoint
    @staticmethod
    def weekday_from_days(days_from_epoch: int) -> int:
        """
            Gets the weekday of the day a given the number of days from the unix epoch
            Parameters:
                days_from_epoch (int): The number of days from the unix epoch

            Returns:
            (int) the weekday (0-6)
        """
        return (
            (days_from_epoch + 4) % 7
            if days_from_epoch >= -4
            else (days_from_epoch + 5) % 7 + 6
        )

    @Entrypoint
    @staticmethod
    def get_nth_dow_of_month_ts(n: int, dow: int, month: int, year: int) -> int:
        """
            Gets the timestamp for the nth day-of-week for the given month/year. E.g. get 2nd Sat in July 2022
            Parameters:
                n (int): nth day of week (1-4).
                dow (int): The day of the week (0-6) where 0 => Sunday ... 6 => Saturday
                month (int): the month (1-12)
                year (int): the year

            Returns:
               (int): The nth day of the month in unixtime
        """
        # Note: Some months will have a 5th dow. i.e. Some months may have 5 Saturdays, for e.g.
        # We, however, restrict this to the 4th dow for reliablity and predictability
        if n < 1 or n > 4:
            raise ValueError("n should be 1-4:", n)
        if dow < 0 or dow > 6:
            raise ValueError("dow should be 0-6:", dow)
        if month < 1 or month > 12:
            raise ValueError("invalid month:", month)

        weekday = Chrono.weekday_from_days(Chrono.days_from_civil(year, month, 1))

        return Chrono.date_to_seconds(
            year=year,
            month=month,
            day=Chrono.weekday_difference(dow, weekday) + 1 + (n - 1) * 7,
        )

    @Entrypoint
    @staticmethod
    def year_from_ts(ts: float) -> int:
        """
            Gets the year from a unix timestamp
            Parameters:
                ts (float): the unix timestamp
            Returns:
                (int): The year
        """
        # Based on the days_from_civil algorithm described here:
        # https://howardhinnant.github.io/date_algorithms.html#civil_from_days
        z = int(ts // 86400 + 719468)
        era = (z if z >= 0 else z - 146096) // 146097
        doe = z - era * 146097
        yoe = (doe - (doe // 1460) + (doe // 36524) - (doe // 146096)) // 365
        year = yoe + era * 400
        doy = doe - ((365 * yoe) + (yoe // 4) - (yoe // 100))
        mp = (5 * doy + 2) // 153
        m = mp + (3 if mp < 10 else -9)
        year += m <= 2
        return year

    @Entrypoint
    @staticmethod
    def is_leap_year(year: int):
        """
        Tests if a year is a leap year.
        Parameters:
            year(int): The year
        Returns:
            True if the year is a leap year, False otherwise
        """
        return (year % 4 == 0 and year % 100 != 0) or year % 400 == 0

    @Entrypoint
    @staticmethod
    def is_valid_time(hour: int, minute: int, second: float) -> bool:
        """
        Tests if a hour, min, sec combination is a valid time.
        Parameters:
            hour(int): The hour
            min(int): The min
            sec(float): The second
        Returns:
            True if the time is valid, False otherwise
        """
        # '24' is valid alternative to '0' but only when min and sec are both 0
        if hour < 0 or hour > 24 or (hour == 24 and (minute != 0 or second != 0)):
            return False
        elif minute < 0 or minute > 59 or second < 0 or second >= 60:
            return False
        return True

    @Entrypoint
    @staticmethod
    def is_valid_date(year: int, month: int, day: int) -> bool:
        """
        Tests if a year, month, day combination is a valid date. Year is required.
        Month and day are optional. If day is present, month is required.
        Parameters:
            year (int): The year
            month (int): The month (January=1)
            day (int): The day of the month
        Returns:
            True if the date is valid, False otherwise
        """
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
            elif (
                month == 1
                or month == 3
                or month == 5
                or month == 7
                or month == 8
                or month == 10
                or month == 12
            ):
                return day < 32
            elif month == 4 or month == 6 or month == 9 or month == 11:
                return day < 31
            elif month == 2:
                if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
                    return day < 30
                return day < 29

        return True
