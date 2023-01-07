from typed_python import Class, Final, Member, Entrypoint, ConstDict, Held, TypeFunction
from typed_python.lib.datetime.chrono import Chrono


class NonexistentDateTime(Exception):
    pass


class OneFoldOnlyError(Exception):
    pass


@Held
class TimeOfDay(Class, Final):
    hour = Member(int)
    minute = Member(int)
    second = Member(float)

    @Entrypoint
    def __init__(self, hour: int, minute: int, second: float):
        self.hour = hour
        self.minute = minute
        self.second = second

    @Entrypoint
    def secondsSinceMidnight(self) -> float:
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


@Held
class Date(Class, Final):
    year = Member(int)
    month = Member(int)
    day = Member(int)

    @Entrypoint
    def __init__(self, year: int, month: int, day: int):
        self.year = year
        self.month = month
        self.day = day

    @Entrypoint
    def __eq__(self, other):
        return self.year == other.year and self.month == other.month and self.day == other.day

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
    def __sub__(self, other) -> int:
        return self.daysSinceEpoch() - other.daysSinceEpoch()

    @Entrypoint
    def daysSinceEpoch(self):
        return Chrono.days_from_civil(self.year, self.month, self.day)

    @Entrypoint
    def weekday(self):
        """Returns an integer: 0 for Monday, 1 for Tuesday, ..., 6 for Sunday"""
        daysSinceEpoch = self.daysSinceEpoch()
        return Chrono.weekday_from_days(daysSinceEpoch)

    @Entrypoint
    def dayOfYear(self):
        return Chrono.day_of_year(self.year, self.month, self.day)

    @Entrypoint
    def nextMonthStart(self):
        if self.month == 12:
            year = self.year + 1
            month = 1
        else:
            year = self.year
            month = self.month + 1

        return Date(year, month, 1)

    @Entrypoint
    def lastDayOfMonth(self):
        return Date.fromDaysSinceEpoch(self.nextMonthStart().daysSinceEpoch() - 1)

    @Entrypoint
    @staticmethod
    def fromDaysSinceEpoch(daysSinceEpoch):
        dt = Chrono.civil_from_days(daysSinceEpoch)
        return Date(dt.year, dt.month, dt.day)

    @Entrypoint
    def daysInMonth(self):
        return self.lastDayOfMonth() - Date(self.year, self.month, 1)

    @Entrypoint
    def firstOfMonth(self):
        return Date(self.year, self.month, 1)

    @Entrypoint
    def lastWeekdayOfMonth(self, weekday: int):
        self.firstOfMonth().weekday()


class DateTime(Class, Final):
    date = Member(Date)
    timeOfDay = Member(TimeOfDay)

    @Entrypoint
    def __init__(self, year: int, month: int, day: int, hour: int, minute: int, second: float):
        self.date = Date(year, month, day)
        self.timeOfDay = TimeOfDay(hour, minute, second)

    @Entrypoint
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

    def __str__(self):

        return (
            "-".join(
                [
                    str(self.date.year),
                    str(self.date.month).zfill(2),
                    str(self.date.day).zfill(2),
                ]
            )
            + " "
            + ":".join(
                [
                    str(self.timeOfDay.hour).zfill(2),
                    str(self.timeOfDay.minute).zfill(2),
                    str(int(self.timeOfDay.second)).zfill(2),
                ]
            )
        )


class TimeZone(Class):
    @Entrypoint
    def timestamp(self, dateTime: DateTime, afterFold: bool = False) -> float:
        raise NotImplementedError("Subclasses implement.")

    @Entrypoint
    def datetime(self, timestamp: float) -> DateTime:
        raise NotImplementedError("Subclasses implement.")

    @Entrypoint
    def _datetimeFromTimestampAndOffset(self, timestamp: float, offset_hours: int) -> DateTime:
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
    @Entrypoint
    def getDaylightSavingsStart(self, year: int) -> DateTime:
        raise NotImplementedError("Subclasses implement")

    @Entrypoint
    def getDaylightSavingsEnd(self, year: int) -> DateTime:
        raise NotImplementedError("Subclasses implement")


class NthWeekdayRule(DaylightSavingsBoundaryRule, Final):
    nStart = Member(int)
    weekdayStart = Member(int)
    monthStart = Member(int)
    nEnd = Member(int)
    weekdayEnd = Member(int)
    monthEnd = Member(int)

    @Entrypoint
    def __init__(self, nStart, weekdayStart, monthStart, nEnd, weekdayEnd, monthEnd):
        self.nStart = nStart
        self.weekdayStart = weekdayStart
        self.monthStart = monthStart
        self.nEnd = nEnd
        self.weekdayEnd = weekdayEnd
        self.monthEnd = monthEnd

    @Entrypoint
    def getDaylightSavingsStart(self, year: int) -> DateTime:
        date = Date.fromDaysSinceEpoch(
            Chrono.get_nth_dow_of_month(self.nStart, self.weekdayStart, self.monthStart, year)
        )
        return DateTime(date=date, timeOfDay=TimeOfDay(2, 0, 0))

    @Entrypoint
    def getDaylightSavingsEnd(self, year: int) -> DateTime:
        date = Date.fromDaysSinceEpoch(
            Chrono.get_nth_dow_of_month(self.nEnd, self.weekdayEnd, self.monthEnd, year)
        )
        return DateTime(date=date, timeOfDay=TimeOfDay(2, 0, 0))


class LastWeekdayRule(DaylightSavingsBoundaryRule, Final):
    weekdayStart = Member(int)
    monthStart = Member(int)
    weekdayEnd = Member(int)
    monthEnd = Member(int)

    @Entrypoint
    def __init__(self, weekdayStart: int, monthStart: int, weekdayEnd: int, monthEnd: int):
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
    nStart = Member(int)
    weekdayStart = Member(int)
    monthStart = Member(int)
    weekdayEnd = Member(int)
    monthEnd = Member(int)

    @Entrypoint
    def __init__(
        self, nStart: int, weekdayStart: int, monthStart: int, weekdayEnd: int, monthEnd: int
    ):
        self.nStart = nStart
        self.weekdayStart = weekdayStart
        self.monthStart = monthStart
        self.weekdayEnd = weekdayEnd
        self.monthEnd = monthEnd

    @Entrypoint
    def getDaylightSavingsStart(self, year: int) -> DateTime:
        date = Date.fromDaysSinceEpoch(
            Chrono.get_nth_dow_of_month(self.nStart, self.weekdayStart, self.monthStart, year)
        )
        return DateTime(date=date, timeOfDay=TimeOfDay(2, 0, 0))

    @Entrypoint
    def getDaylightSavingsEnd(self, year: int) -> DateTime:
        date = last_weekday_of_month(year, self.monthEnd, self.weekdayEnd)
        return DateTime(date=date, timeOfDay=TimeOfDay(2, 0, 0))


class DateTimeRule(DaylightSavingsBoundaryRule, Final):
    dateTimeStart = Member(DateTime)
    dateTimeEnd = Member(DateTime)

    @Entrypoint
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


class DaylightSavingsTimezone(TimeZone, Final):
    dst_offset_hours = Member(float)
    st_offset_hours = Member(float)
    dst_boundaries = Member(DaylightSavingsBoundaryRule)

    @Entrypoint
    def __init__(
        self,
        dst_offset_hours: float,
        st_offset_hours: float,
        dst_boundaries: DaylightSavingsBoundaryRule,
    ):
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

        is_daylight_savings = not ((dateTime > ds_end or dateTime < ds_start) or afterFold)

        if afterFold and (dateTime.date != ds_end.date or dateTime > ds_end):
            raise OneFoldOnlyError("There is only one fold.")

        if (
            dateTime.date == ds_start.date
            and dateTime.timeOfDay.hour == ds_start.timeOfDay.hour
        ):
            raise NonexistentDateTime(dateTime)

        offset_hours = self.dst_offset_hours if is_daylight_savings else self.st_offset_hours

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


class FixedOffsetTimezone(TimeZone, Final):
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


class NycTimezone(TimeZone, Final):
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


class SwitchOffsetTimezone(TimeZone, Final):
    offset_hours_before = Member(float)
    offset_hours_after = Member(float)
    switch_datetime = Member(DateTime)

    @Entrypoint
    def __init__(
        self, offset_hours_before: float, offset_hours_after: float, switch_datetime: DateTime
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
        if afterFold and (dateTime.date != switch.date or dateTime > switch or is_start_like):
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
            self.offset_hours_before if timestamp < ts_switch else self.offset_hours_after
        )

        return self._datetimeFromTimestampAndOffset(timestamp, offset_hours)


@TypeFunction
def UsTimeZone(st_offset_hours, dst_offset_hours):
    class UsTimeZone_(TimeZone, Final):
        TIMEZONES_BY_START_YEAR = ConstDict(int, TimeZone)(
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
            # print(f"dateTime:\n{dateTime}")
            return self.chooseTimezone(dateTime.date.year).timestamp(dateTime, afterFold)

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

    return UsTimeZone_


NYC = UsTimeZone(-5, -4)()
UTC = FixedOffsetTimezone(offset_hours=0)
EST = FixedOffsetTimezone(offset_hours=-5)
IST = FixedOffsetTimezone(offset_hours=2)


class TimeZoneChecker(Class, Final):
    TIMEZONES = ConstDict(str, TimeZone)(
        {
            "": UTC,
            "+0000": UTC,
            "nyc": NYC,
            "utc": UTC,
            "z": UTC,
            "est": EST,
            "edt": NYC,
            "ist": IST,
        }
    )

    @classmethod
    @Entrypoint
    def isValidTimezone(cls, timeZoneString: str) -> bool:
        return timeZoneString.lower() in cls.TIMEZONES


@Entrypoint
def last_weekday_of_month(year: int, month: int, weekday: int) -> Date:
    """ weekday = 0 is Sunday """
    monthEnd = Date(year, month, 1).lastDayOfMonth()
    monthEndWeekday = monthEnd.weekday()
    return Date.fromDaysSinceEpoch(monthEnd.daysSinceEpoch() - (monthEndWeekday - weekday) % 7)
