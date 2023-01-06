from typed_python import Class, Final, Member, Entrypoint, ConstDict, Held
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
    def secondsSinceMidnight(self, afterFold: bool = False) -> float:
        return (self.second + self.minute * 60 + self.hour * 3600) + afterFold * 3600

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
    def __gt__(self, other):
        return (
            self.year == other.year
            and self.month == other.month
            and self.day == other.day
        )


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
        return (Chrono.weekday_from_days(daysSinceEpoch) - 1) % 7

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

    def __eq__(self, other):
        return self.date == other.date and self.timeOfDay == other.timeOfDay

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




class TimeZone(Class):
    @Entrypoint
    def timestamp(self, dateTime: DateTime, afterFold: bool = False) -> float:
        raise NotImplementedError("Subclasses implement.")

    @Entrypoint
    def datetime(self, timestamp: float) -> DateTime:
        raise NotImplementedError("Subclasses implement.")

    @Entrypoint
    def _datetimeFromTimestampAndOffset(
        self, timestamp: float, offset_hours: int
    ) -> DateTime:
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
        raise NotImplementedError('Subclasses implement')

    @Entrypoint
    def getDaylightSavingsEnd(self, year: int) -> DateTime:
        raise NotImplementedError('Subclasses implement')

class NycRule2007(DaylightSavingsBoundaryRule, Final):
    @Entrypoint
    def getDaylightSavingsStart(self, year: int) -> DateTime:
        date = Date.fromDaysSinceEpoch(Chrono.get_nth_dow_of_month(2, 0, 3, year))
        return DateTime(date=date, timeOfDay=TimeOfDay(2, 0, 0))

    @Entrypoint
    def getDaylightSavingsEnd(self, year: int) -> DateTime:
        date = Date.fromDaysSinceEpoch(Chrono.get_nth_dow_of_month(1, 0, 11, year))
        return DateTime(date=date, timeOfDay=TimeOfDay(2, 0, 0))



class DaylightSavingsTimezone(TimeZone, Final):
    dst_offset_hours = Member(int)
    st_offset_hours = Member(int)
    dst_boundaries = Member(DaylightSavingsBoundaryRule)

    @Entrypoint
    def timestamp(self, dateTime: DateTime, afterFold: bool = False) -> float:

        year = dateTime.date.year

        # second sunday of march
        ds_start = self.dst_boundaries.getDaylightSavingsStart(year)

        # first sunday of november
        ds_end = self.dst_boundaries.getDaylightSavingsEnd(year)


        is_daylight_savings = True

        if dateTime < ds_start or dateTime > ds_end:
            is_daylight_savings = False
            if afterFold:
                raise OneFoldOnlyError("There is only one fold.")

        if dateTime.date == ds_start.date:
            if dateTime.timeOfDay.hour == ds_start.timeOfDay.hour:
                raise NonexistentDateTime("This date time does not exist on this date.")

            if afterFold:
                raise OneFoldOnlyError("There is only one fold.")

            is_daylight_savings = dateTime.timeOfDay.hour > 2

        if dateTime.date == ds_end.date:
            if dateTime.timeOfDay.hour > ds_end.timeOfDay.hour:
                afterFold = True
                is_daylight_savings = True
            else:
                is_daylight_savings = not afterFold and dateTime.timeOfDay < ds_end.timeOfDay

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
            startDateTime.date.daysSinceEpoch() * 86400
            - self.st_offset_hours * 3600
        ) + startDateTime.timeOfDay.secondsSinceMidnight()

        ts_end = (
            endDateTime.date.daysSinceEpoch() * 86400
            - self.dst_offset_hours * 3600
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
            + dateTime.timeOfDay.secondsSinceMidnight(afterFold)
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
            + dateTime.timeOfDay.secondsSinceMidnight(afterFold)
            - self.offset_hours * 3600
        )

    @Entrypoint
    def datetime(self, timestamp: float) -> DateTime:
        return self._datetimeFromTimestampAndOffset(timestamp, self.offset_hours)


NYC = DaylightSavingsTimezone(dst_offset_hours=-4, st_offset_hours=-5, dst_boundaries=NycRule2007())
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
    monthEnd = Date(year, month, 1).lastDayOfMonth()
    monthEndWeekday = monthEnd.weekday()
    return Date.fromDaysSinceEpoch(monthEnd.daysSinceEpoch() - (monthEndWeekday - weekday) % 7)
