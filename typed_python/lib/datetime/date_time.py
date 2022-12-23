from typed_python import Class, Final, Member, Entrypoint, ConstDict, Held, TypeFunction
from typed_python.lib.datetime.chrono import Chrono


class NonexistentDateTime(Exception):
    pass


class OneFoldOnlyError(Exception):
    pass


@Held
class TimeOfDay(Class, Final):
    # Models a naive, timezone-unaware time of day.
    hour = Member(int)
    minute = Member(int)
    second = Member(float)

    def __init__(self, hour: int, minute: int, second: float):
        self.hour = hour
        self.minute = minute
        self.second = second

    def __str__(self):
        return "%02.f:%02.f:%02.f" % (self.hour, self.minute, self.second)

    def __repr__(self):
        return (
            f"TimeOfDay(hour={self.hour}, minute={self.minute}, second={self.second})"
        )

    @Entrypoint
    def secondsSinceMidnight(self) -> float:
        """Returns the number of seconds that have elapsed since midnight.

        Does not account for timezone offset switches, e.g., during daylight
        savings time.
        """
        return self.second + self.minute * 60 + self.hour * 3600

    @Entrypoint
    def __eq__(self, other):
        return (
            self.hour == other.hour
            and self.minute == other.minute
            and self.second == other.second
        )

    @Entrypoint
    def __lt__(self, other):
        if self.hour != other.hour:
            return self.hour < other.hour

        if self.minute != other.minute:
            return self.minute < other.minute

        if self.second != other.second:
            return self.second < other.second

        return False

    @Entrypoint
    def __gt__(self, other):
        if other.hour != self.hour:
            return other.hour < self.hour

        if other.minute != self.minute:
            return other.minute < self.minute

        if other.second != self.second:
            return other.second < self.second

        return False

    @Entrypoint
    def __le__(self, other):
        return self == other or self < other

    @Entrypoint
    def __ge__(self, other):
        return self == other or self > other


@Held
class Date(Class, Final):
    # Models a date in the Gregorian calendar.
    # For example, December 22, 1975 would be constructed
    # as `Date(1975, 12, 22)`

    year = Member(int)
    month = Member(int)
    day = Member(int)

    def __init__(self, year: int, month: int, day: int):
        self.year = year
        self.month = month
        self.day = day

    def __str__(self):
        return "%02.f-%02.f-%02.f" % (self.year, self.month, self.day)

    def __repr__(self):
        return f"Date(year={self.year}, month={self.month}, day={self.day})"

    @Entrypoint
    def __eq__(self, other):
        return (
            self.year == other.year
            and self.month == other.month
            and self.day == other.day
        )

    @Entrypoint
    def __lt__(self, other):
        if self.year != other.year:
            return self.year < other.year

        if self.month != other.month:
            return self.month < other.month

        if self.day != other.day:
            return self.day < other.day

        return False

    @Entrypoint
    def __gt__(self, other):
        if other.year != self.year:
            return other.year < self.year

        if other.month != self.month:
            return other.month < self.month

        if other.day != self.day:
            return other.day < self.day

        return False

    @Entrypoint
    def __le__(self, other):
        return self == other or self < other

    @Entrypoint
    def __ge__(self, other):
        return self == other or self > other

    @Entrypoint
    def __sub__(self, other) -> int:
        """Returns the number of days separating `self` from `other`.

        Parameters
        ----------
        other : Date

        Returns
        -------
        int - the number of days separating `self` from `other`.
            Negative if `other` comes after `self`.
        """
        return self.daysSinceEpoch() - other.daysSinceEpoch()

    @Entrypoint
    def daysSinceEpoch(self) -> int:
        """Returns the number of days since 1970 January 1."""
        return Chrono.days_from_civil(self.year, self.month, self.day)

    @Entrypoint
    def weekday(self) -> int:
        """Returns an integer [0, 6] indicating the day of the week.
        0 => Sunday
        1 => Monday
        2 => Tuesday
        3 => Wednesday
        4 => Thursday
        6 => Friday
        """
        daysSinceEpoch = self.daysSinceEpoch()
        return Chrono.weekday_from_days(daysSinceEpoch)

    @Entrypoint
    def dayOfYear(self) -> int:
        """Returns an integer [1, 366] indicating the day of the year."""
        return Chrono.day_of_year(self.year, self.month, self.day)

    @Entrypoint
    def nextMonthStart(self):
        """Returns a Date indicating when the following month begins."""
        if self.month == 12:
            year = self.year + 1
            month = 1
        else:
            year = self.year
            month = self.month + 1

        return Date(year, month, 1)

    @Entrypoint
    def lastDayOfMonth(self):
        """Returns the last Date of the month of `self`."""
        return Date.fromDaysSinceEpoch(self.nextMonthStart().daysSinceEpoch() - 1)

    @Entrypoint
    def daysUntilEndOfMonth(self) -> int:
        return self.lastDayOfMonth() - self

    @Entrypoint
    def daysUntilEndOfYear(self) -> int:
        return self.lastDayOfYear() - self

    @Entrypoint
    def lastDayOfYear(self):
        return Date(self.year, 12, 31)

    @Entrypoint
    @staticmethod
    def fromDaysSinceEpoch(daysSinceEpoch):
        """Returns a Date from the number of days that have elapsed since
        1970 January 1."""
        dt = Chrono.civil_from_days(daysSinceEpoch)
        return Date(dt.year, dt.month, dt.day)

    @Entrypoint
    def daysInMonth(self) -> int:
        """Returns the number of days in the month."""
        return self.lastDayOfMonth() - Date(self.year, self.month, 1)

    @Entrypoint
    def firstOfMonth(self):
        """Returns the first of the month"""
        return Date(self.year, self.month, 1)

    @Entrypoint
    def quarterOfYear(self):
        return (self.date.month - 1) // 3 + 1


@Held
class DateTime(Class, Final):
    # Models a naive, timezone-unaware date and time in the Gregorian calendar.
    date = Member(Date)
    timeOfDay = Member(TimeOfDay)

    def __init__(
        self, year: int, month: int, day: int, hour: int, minute: int, second: float
    ):
        self.date = Date(year, month, day)
        self.timeOfDay = TimeOfDay(hour, minute, second)

    def __init__(self, date: Date, timeOfDay: TimeOfDay):  # noqa: F811
        self.date = date
        self.timeOfDay = timeOfDay

    @Entrypoint
    def __eq__(self, other):
        return self.date == other.date and self.timeOfDay == other.timeOfDay

    @Entrypoint
    def __lt__(self, other):
        if self.date != other.date:
            return self.date < other.date

        if self.timeOfDay != other.timeOfDay:
            return self.timeOfDay < other.timeOfDay

        return False

    @Entrypoint
    def __gt__(self, other):
        if other.date != self.date:
            return other.date < self.date

        if other.timeOfDay != self.timeOfDay:
            return other.timeOfDay < self.timeOfDay

        return False

    @Entrypoint
    def __le__(self, other):
        return self == other or self < other

    @Entrypoint
    def __ge__(self, other):
        return self == other or self > other

    def __str__(self):
        return str(self.date) + " " + str(self.timeOfDay)

    def __repr__(self):
        return f"DateTime(date={repr(self.date)}, timeOfDay={repr(self.timeOfDay)})"


class Timezone(Class):
    # An interface for mapping DateTimes to UTC timestamps (i.e.,
    # unequivocal instants in time and vice versa.

    @Entrypoint
    def timestamp(self, dateTime: DateTime, afterFold: bool = False) -> float:
        """Return a UTC timestamp at the instant a locale's clocks would
        show the inputted `DateTime`.

        Parameters
        ----------
        dateTime : DateTime
            The date and time a clock would read in a given locale.
        afterFold : bool
            Sometimes, the same DateTime corresponds to two different
            UTC timestamps, e.g., when clocks are rolled backwards in the
            Autumn for Daylight Savings time. In these cases, 1am to 2am
            hour repeats. When `afterFold` is True, we return the second
            occurence. Otherwise we return the first occurence.

        Returns
        -------
        float

        """
        raise NotImplementedError("Subclasses implement.")

    @Entrypoint
    def datetime(self, timestamp: float) -> DateTime:
        """Return the DateTime a locale's clocks would read at the inputted
        UTC timestamp.

        Parameters
        ----------
        timestamp : float
            timestamp

        """
        raise NotImplementedError("Subclasses implement.")

    @Entrypoint
    def _datetimeFromTimestampAndOffset(
        self, timestamp: float, offset_hours: float
    ) -> DateTime:
        """Returns a DateTime from a UTC timestamp and an offset (in hours). This
        returns the clock in a locale offset from UTC by `offset_hours`.

        Parameters
        ----------
        timestamp : float
            a UTC timestamp.
        offset_hours : float
            the number of hours ahead or behind UTC for a given locale. For example,
            eastern standard time, EST, has offset_hours = -5
        """
        ts = timestamp + offset_hours * 3600
        day = ts // 86400
        secondsSinceMidnight = ts % 86400

        date = Chrono.civil_from_days(day)

        hour = secondsSinceMidnight // 3600
        seconds = secondsSinceMidnight % 3600
        minute = seconds // 60
        second = seconds % 60

        return DateTime(date.year, date.month, date.day, hour, minute, second)


class DaylightSavingsBoundaryRule(Class):
    # An interface for generating DateTimes in a given year corresponding
    # to the beginning and end of a DaylightSavingsTime period in a given
    # locale.
    @Entrypoint
    def getDaylightSavingsStart(self, year: int) -> DateTime:
        raise NotImplementedError("Subclasses implement")

    @Entrypoint
    def getDaylightSavingsEnd(self, year: int) -> DateTime:
        raise NotImplementedError("Subclasses implement")


class NthWeekdayRule(DaylightSavingsBoundaryRule, Final):
    # A DaylightSavingsBoundaryRule that returns, for example, the second
    # Sunday in March and the last Tuesday in November of a given year.
    nStart = Member(int)
    weekdayStart = Member(int)
    monthStart = Member(int)
    nEnd = Member(int)
    weekdayEnd = Member(int)
    monthEnd = Member(int)

    def __init__(
        self,
        nStart: int,
        weekdayStart: int,
        monthStart: int,
        nEnd: int,
        weekdayEnd: int,
        monthEnd: int,
    ):
        """
        Parameters
        ----------
        nStart/nEnd : int
            The ordinal corresponding to which occurence of a weekday we consider
            to be the start/end.
        weekdayStart/weekdayEnd : int
            The day of the week (0 => Sunday)
        monthStart/monthEnd : int
            The month of the year (1 => January, etc.)

        For example
            nStart = 1
            weekdayStart = 2
            monthStart = 3
            nEnd = 4
            weekdayEnd = 5
            monthEnd = 6

        models an interval that runs from the first Tuesday of March
        to the 4th Friday of June.
        """
        self.nStart = nStart
        self.weekdayStart = weekdayStart
        self.monthStart = monthStart
        self.nEnd = nEnd
        self.weekdayEnd = weekdayEnd
        self.monthEnd = monthEnd

    @Entrypoint
    def getDaylightSavingsStart(self, year: int) -> DateTime:
        date = Date.fromDaysSinceEpoch(
            Chrono.get_nth_dow_of_month(
                self.nStart, self.weekdayStart, self.monthStart, year
            )
        )
        return DateTime(date=date, timeOfDay=TimeOfDay(2, 0, 0))

    @Entrypoint
    def getDaylightSavingsEnd(self, year: int) -> DateTime:
        date = Date.fromDaysSinceEpoch(
            Chrono.get_nth_dow_of_month(self.nEnd, self.weekdayEnd, self.monthEnd, year)
        )
        return DateTime(date=date, timeOfDay=TimeOfDay(2, 0, 0))


class LastWeekdayRule(DaylightSavingsBoundaryRule, Final):
    # A DaylightSavingsBoundaryRule that returns, for example, the last
    # Tuesday in February and the last Thursday in October of a given year"""
    weekdayStart = Member(int)
    monthStart = Member(int)
    weekdayEnd = Member(int)
    monthEnd = Member(int)

    def __init__(
        self, weekdayStart: int, monthStart: int, weekdayEnd: int, monthEnd: int
    ):
        """
        Parameters
        ----------
        weekdayStart/weekdayEnd : int
            The day of the week (0 => Sunday)
        monthStart/monthEnd : int
            The month of the year (1 => January, etc.)
        """
        self.weekdayStart = weekdayStart
        self.monthStart = monthStart
        self.weekdayEnd = weekdayEnd
        self.monthEnd = monthEnd

    @Entrypoint
    def getDaylightSavingsStart(self, year: int) -> DateTime:
        date = last_weekday_of_month(year, self.monthStart, self.weekdayStart)
        return DateTime(date=date, timeOfDay=TimeOfDay(2, 0, 0))

    @Entrypoint
    def getDaylightSavingsEnd(self, year: int) -> DateTime:
        date = last_weekday_of_month(year, self.monthEnd, self.weekdayEnd)
        return DateTime(date=date, timeOfDay=TimeOfDay(2, 0, 0))


class NthToLastWeekdayRule(DaylightSavingsBoundaryRule, Final):
    # A DaylightSavingsBoundaryRule that returns, for example, the 2nd
    # Tuesday in February and the last Thursday in October of a given year"""
    nStart = Member(int)
    weekdayStart = Member(int)
    monthStart = Member(int)
    weekdayEnd = Member(int)
    monthEnd = Member(int)

    def __init__(
        self,
        nStart: int,
        weekdayStart: int,
        monthStart: int,
        weekdayEnd: int,
        monthEnd: int,
    ):
        """
        Parameters
        ----------
        nStart : int
            The ordinal corresponding to which occurence of a weekday we consider
            to be the start.
        weekdayStart/weekdayEnd : int
            The day of the week (0 => Sunday)
        monthStart/monthEnd : int
            The month of the year (1 => January, etc.)
        """
        self.nStart = nStart
        self.weekdayStart = weekdayStart
        self.monthStart = monthStart
        self.weekdayEnd = weekdayEnd
        self.monthEnd = monthEnd

    @Entrypoint
    def getDaylightSavingsStart(self, year: int) -> DateTime:
        date = Date.fromDaysSinceEpoch(
            Chrono.get_nth_dow_of_month(
                self.nStart, self.weekdayStart, self.monthStart, year
            )
        )
        return DateTime(date=date, timeOfDay=TimeOfDay(2, 0, 0))

    @Entrypoint
    def getDaylightSavingsEnd(self, year: int) -> DateTime:
        date = last_weekday_of_month(year, self.monthEnd, self.weekdayEnd)
        return DateTime(date=date, timeOfDay=TimeOfDay(2, 0, 0))


class DateTimeRule(DaylightSavingsBoundaryRule, Final):
    # A DaylightSavingsBoundaryRule that returns two explicit DateTimes.
    dateTimeStart = Member(DateTime)
    dateTimeEnd = Member(DateTime)

    def __init__(self, dateTimeStart: DateTime, dateTimeEnd: DateTime):
        self.dateTimeStart = dateTimeStart
        self.dateTimeEnd = dateTimeEnd

    @Entrypoint
    def getDaylightSavingsStart(self, year: int) -> DateTime:
        if year != self.dateTimeStart.date.year:
            raise Exception("You are probably using the wrong rule for this timezone.")

        return self.dateTimeStart

    @Entrypoint
    def getDaylightSavingsEnd(self, year: int) -> DateTime:
        if year != self.dateTimeEnd.date.year:
            raise Exception("You are probably using the wrong rule for this timezone.")

        return self.dateTimeEnd


class DaylightSavingsTimezone(Timezone, Final):
    # A Timezone that has two different offsets in a given year,
    # st_offset_hours, which applies before and after daylight savings time,
    # dst_offset_hours, which applies during daylight savings time, and a rule
    # for computing when daylight savings time applies.

    dst_offset_hours = Member(float)
    st_offset_hours = Member(float)
    dst_boundaries = Member(DaylightSavingsBoundaryRule)

    def __init__(
        self,
        dst_offset_hours: float,
        st_offset_hours: float,
        dst_boundaries: DaylightSavingsBoundaryRule,
    ):
        """
        Parameters
        ----------
        dst_offset_hours : float
            The number of hours to offset from UTC during daylight savings.
        st_offset_hours : float
            The number of hours to offset from UTC outside daylight savings.
        dst_boundaries : DaylightSavingsBoundaryRule
            A rule for computing when daylight savings begins and ends in each
            year.
        """
        assert st_offset_hours < dst_offset_hours
        self.dst_offset_hours = dst_offset_hours
        self.st_offset_hours = st_offset_hours
        self.dst_boundaries = dst_boundaries

    @Entrypoint
    def timestamp(self, dateTime: DateTime, afterFold: bool = False) -> float:
        year = dateTime.date.year

        # second sunday of march
        ds_start = self.dst_boundaries.getDaylightSavingsStart(year)

        # first sunday of november
        ds_end = self.dst_boundaries.getDaylightSavingsEnd(year)

        is_daylight_savings = not (
            (dateTime > ds_end or dateTime < ds_start) or afterFold
        )

        if afterFold and (dateTime.date != ds_end.date or dateTime > ds_end):
            raise OneFoldOnlyError("There is only one fold.")

        if (
            dateTime.date == ds_start.date
            and dateTime.timeOfDay.hour == ds_start.timeOfDay.hour
        ):
            raise NonexistentDateTime(dateTime)

        offset_hours = (
            self.dst_offset_hours if is_daylight_savings else self.st_offset_hours
        )

        return (
            dateTime.date.daysSinceEpoch() * 86400
            - offset_hours * 3600
            + dateTime.timeOfDay.secondsSinceMidnight()
        )

    @Entrypoint
    def datetime(self, timestamp) -> DateTime:
        # Figure out if this timestamp falls within daylight savings or not.
        # Technically this could be off by a day (on Dec 31 or Jan1),
        # and in these cases, we're not in daylight savings time anyway,
        # so it gives the right answer.
        year = Chrono.civil_from_days(timestamp // 86400).year

        startDateTime = self.dst_boundaries.getDaylightSavingsStart(year)
        endDateTime = self.dst_boundaries.getDaylightSavingsEnd(year)

        ts_start = (
            startDateTime.date.daysSinceEpoch() * 86400 - self.st_offset_hours * 3600
        ) + startDateTime.timeOfDay.secondsSinceMidnight()

        ts_end = (
            endDateTime.date.daysSinceEpoch() * 86400 - self.dst_offset_hours * 3600
        ) + endDateTime.timeOfDay.secondsSinceMidnight()

        # get offset
        offset_hours = (
            self.st_offset_hours
            if timestamp < ts_start or timestamp > ts_end
            else self.dst_offset_hours
        )

        return self._datetimeFromTimestampAndOffset(timestamp, offset_hours)


class FixedOffsetTimezone(Timezone, Final):
    """Returns a Timezone that is offset a fixed number of hours from UTC"""

    offset_hours = Member(float)

    @Entrypoint
    def timestamp(self, dateTime: DateTime, afterFold: bool = False) -> float:
        return (
            dateTime.date.daysSinceEpoch() * 86400
            + dateTime.timeOfDay.secondsSinceMidnight()
            - self.offset_hours * 3600
        )

    @Entrypoint
    def datetime(self, timestamp: float) -> DateTime:
        return self._datetimeFromTimestampAndOffset(timestamp, self.offset_hours)


class SwitchOffsetTimezone(Timezone, Final):
    # Occassionally, the offset will change exactly one time in a given year. This happened
    # in the United States in 1945
    offset_hours_before = Member(float)
    offset_hours_after = Member(float)
    switch_datetime = Member(DateTime)

    def __init__(
        self,
        offset_hours_before: float,
        offset_hours_after: float,
        switch_datetime: DateTime,
    ):
        assert (
            offset_hours_before != offset_hours_after
        ), "If the offsets are the same, there is no switch."

        self.offset_hours_before = offset_hours_before
        self.offset_hours_after = offset_hours_after
        self.switch_datetime = switch_datetime

    @Entrypoint
    def timestamp(self, dateTime: DateTime, afterFold: bool = False) -> float:
        switch = self.switch_datetime

        is_after = dateTime > switch or afterFold

        jumps_forward = self.offset_hours_after > self.offset_hours_before
        if afterFold and (
            dateTime.date != switch.date or dateTime > switch or jumps_forward
        ):
            raise OneFoldOnlyError("There is only one fold.")

        if jumps_forward and dateTime.timeOfDay.hour == switch.timeOfDay.hour:
            raise NonexistentDateTime(dateTime)

        offset_hours = self.offset_hours_after if is_after else self.offset_hours_before

        return (
            dateTime.date.daysSinceEpoch() * 86400
            - offset_hours * 3600
            + dateTime.timeOfDay.secondsSinceMidnight()
        )

    @Entrypoint
    def datetime(self, timestamp: float) -> DateTime:
        ts_switch = (
            self.switch_datetime.date.daysSinceEpoch() * 86400
            - self.offset_hours_before * 3600
        ) + self.switch_datetime.timeOfDay.secondsSinceMidnight()

        # get offset
        offset_hours = (
            self.offset_hours_before
            if timestamp < ts_switch
            else self.offset_hours_after
        )

        return self._datetimeFromTimestampAndOffset(timestamp, offset_hours)


@TypeFunction
def UsTimezone(st_offset_hours, dst_offset_hours):
    class UsTimezone_(Timezone, Final):
        st_offset = st_offset_hours
        dst_offset = dst_offset_hours
        TIMEZONES_BY_START_YEAR = ConstDict(int, Timezone)(
            {
                2007: DaylightSavingsTimezone(
                    dst_offset_hours=dst_offset_hours,
                    st_offset_hours=st_offset_hours,
                    dst_boundaries=NthWeekdayRule(2, 0, 3, 1, 0, 11),
                ),
                1987: DaylightSavingsTimezone(
                    dst_offset_hours=dst_offset_hours,
                    st_offset_hours=st_offset_hours,
                    dst_boundaries=NthToLastWeekdayRule(1, 0, 4, 0, 10),
                ),
                1975: DaylightSavingsTimezone(
                    dst_offset_hours=dst_offset_hours,
                    st_offset_hours=st_offset_hours,
                    dst_boundaries=DateTimeRule(
                        DateTime(1975, 2, 23, 2, 0, 0), DateTime(1975, 10, 26, 2, 0, 0)
                    ),
                ),
                1974: DaylightSavingsTimezone(
                    dst_offset_hours=dst_offset_hours,
                    st_offset_hours=st_offset_hours,
                    dst_boundaries=DateTimeRule(
                        DateTime(1974, 1, 6, 2, 0, 0), DateTime(1974, 10, 27, 2, 0, 0)
                    ),
                ),
                1955: DaylightSavingsTimezone(
                    dst_offset_hours=dst_offset_hours,
                    st_offset_hours=st_offset_hours,
                    dst_boundaries=LastWeekdayRule(0, 4, 0, 10),
                ),
                1946: DaylightSavingsTimezone(
                    dst_offset_hours=dst_offset_hours,
                    st_offset_hours=st_offset_hours,
                    dst_boundaries=LastWeekdayRule(0, 4, 0, 9),
                ),
                1945: SwitchOffsetTimezone(
                    offset_hours_before=dst_offset_hours,
                    offset_hours_after=st_offset_hours,
                    switch_datetime=DateTime(1945, 9, 30, 2, 0, 0),
                ),
                1943: FixedOffsetTimezone(offset_hours=dst_offset_hours),
                1942: SwitchOffsetTimezone(
                    offset_hours_before=st_offset_hours,
                    offset_hours_after=dst_offset_hours,
                    switch_datetime=DateTime(1942, 2, 9, 2, 0, 0),
                ),
                1921: DaylightSavingsTimezone(
                    dst_offset_hours=dst_offset_hours,
                    st_offset_hours=st_offset_hours,
                    dst_boundaries=LastWeekdayRule(0, 4, 0, 9),
                ),
                1918: DaylightSavingsTimezone(
                    dst_offset_hours=dst_offset_hours,
                    st_offset_hours=st_offset_hours,
                    dst_boundaries=LastWeekdayRule(0, 3, 0, 10),
                ),
                1884: FixedOffsetTimezone(offset_hours=st_offset_hours),
                1883: SwitchOffsetTimezone(
                    offset_hours_before=-17762 / 3600,
                    offset_hours_after=-10858 / 3600,
                    switch_datetime=DateTime(1883, 11, 18, 12, 3, 0),
                ),
                1776: FixedOffsetTimezone(offset_hours=-17762 / 3600),
            }
        )

        @Entrypoint
        def timestamp(self, dateTime: DateTime, afterFold: bool = False) -> float:
            return self.chooseTimezone(dateTime.date.year).timestamp(
                dateTime, afterFold
            )

        @Entrypoint
        def datetime(self, timestamp: float) -> DateTime:
            year = Chrono.civil_from_days(timestamp // 86400).year
            return self.chooseTimezone(year).datetime(timestamp)

        @Entrypoint
        def chooseTimezone(self, year: int):
            if year >= 2007:
                return self.TIMEZONES_BY_START_YEAR[2007]

            if year >= 1987:
                return self.TIMEZONES_BY_START_YEAR[1987]

            if year == 1975:
                return self.TIMEZONES_BY_START_YEAR[1975]

            if year == 1974:
                return self.TIMEZONES_BY_START_YEAR[1974]

            if year >= 1955:
                return self.TIMEZONES_BY_START_YEAR[1955]

            if year >= 1946:
                return self.TIMEZONES_BY_START_YEAR[1946]

            if year == 1945:
                return self.TIMEZONES_BY_START_YEAR[1945]

            if year >= 1943:
                return self.TIMEZONES_BY_START_YEAR[1943]

            if year == 1942:
                return self.TIMEZONES_BY_START_YEAR[1942]

            if year >= 1921:
                return self.TIMEZONES_BY_START_YEAR[1921]

            if year >= 1918:
                return self.TIMEZONES_BY_START_YEAR[1918]

            if year >= 1884:
                return self.TIMEZONES_BY_START_YEAR[1884]

            if year == 1883:
                return self.TIMEZONES_BY_START_YEAR[1883]

            else:
                return self.TIMEZONES_BY_START_YEAR[1776]

    return UsTimezone_


NYC = UsTimezone(-5, -4)()
CHI = UsTimezone(-6, -5)()
UTC = FixedOffsetTimezone(offset_hours=0)
EST = FixedOffsetTimezone(offset_hours=-5)
IST = FixedOffsetTimezone(offset_hours=2)


class TimezoneChecker(Class, Final):
    TIMEZONES = ConstDict(str, Timezone)(
        {
            "": UTC,
            "+0000": UTC,
            "nyc": NYC,
            "utc": UTC,
            "z": UTC,
            "est": EST,
            "edt": NYC,
            "ist": IST,
            "chi": CHI,
        }
    )

    @classmethod
    @Entrypoint
    def isValidTimezone(cls, timeZoneString: str) -> bool:
        return timeZoneString.lower() in cls.TIMEZONES


@Entrypoint
def last_weekday_of_month(year: int, month: int, weekday: int) -> Date:
    """Return the last weekday in a given month.

    Parameters
    ----------
    year : int
        The year in the Gregorian calendar.
    month : int
        The day of the month.
    weekday : int
        The day of the week (0=>Sunday).
    """
    monthEnd = Date(year, month, 1).lastDayOfMonth()
    monthEndWeekday = monthEnd.weekday()
    return Date.fromDaysSinceEpoch(
        monthEnd.daysSinceEpoch() - (monthEndWeekday - weekday) % 7
    )
