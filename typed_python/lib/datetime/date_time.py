from typed_python import Class, Final, Member, Entrypoint, ConstDict, Held
from typed_python.lib.datetime.chrono import Chrono


class NonexistentDateTime(Exception):
    pass


class OneFold(Exception):
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
    def daysSinceEpoch(self):
        return Chrono.days_from_civil(self.year, self.month, self.day)

    @Entrypoint
    def weekday(self):
        daysSinceEpoch = self.daysSinceEpoch()
        return Chrono.weekday_from_days(daysSinceEpoch)

    @Entrypoint
    def dayOfYear(self):
        return Chrono.day_of_year(self.year, self.month, self.day)


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
        tod = TimeOfDay(hour=hour, minute=minute, second=second)

        return DateTime(
            date=Date(year=date.year, month=date.month, day=date.day), timeOfDay=tod
        )


class DaylightSavingsTimezone(TimeZone, Final):
    dst_offset_hours = Member(int)
    st_offset_hours = Member(int)

    @Entrypoint
    def timestamp(self, dateTime: DateTime, afterFold: bool = False) -> float:

        year = dateTime.date.year

        # second sunday of march
        ds_start = Chrono.get_nth_dow_of_month(2, 0, 3, year)

        # first sunday of november
        ds_end = Chrono.get_nth_dow_of_month(1, 0, 11, year)

        day = dateTime.date.daysSinceEpoch()

        is_daylight_savings = True

        if day < ds_start:
            is_daylight_savings = False
            if afterFold:
                raise OneFold("There is only one fold.")

        if day > ds_end:
            is_daylight_savings = False
            if afterFold:
                raise OneFold("There is only one fold.")

        if day == ds_start:
            if dateTime.timeOfDay.hour == 2:
                raise NonexistentDateTime("This date time does not exist on this date.")

            if afterFold:
                raise OneFold("There is only one fold.")

            is_daylight_savings = dateTime.timeOfDay.hour > 2

        if day == ds_end:
            if dateTime.timeOfDay.hour > 2:
                afterFold = True
                is_daylight_savings = True
            else:
                is_daylight_savings = afterFold < 2 and dateTime.timeOfDay.hour < 2

        offset_hours = (
            self.dst_offset_hours if is_daylight_savings else self.st_offset_hours
        )

        return (
            day * 86400
            - offset_hours * 3600
            + dateTime.timeOfDay.secondsSinceMidnight(afterFold)
        )

    @Entrypoint
    def datetime(self, timestamp) -> DateTime:
        # Figure out if this timestamp falls within daylight savings or not.
        # Technically this could be off by a day (on Dec 31 or Jan1),
        # and in these cases, we're not in daylight savings time anyway,
        # so it gives the right answer.
        year = Chrono.civil_from_days(timestamp // 86400).year

        # second sunday of march
        ts_start = (
            Chrono.get_nth_dow_of_month(2, 0, 3, year) * 86400
            - self.st_offset_hours * 3600
        ) + 7200

        # first sunday of november
        ts_end = (
            Chrono.get_nth_dow_of_month(1, 0, 11, year) * 86400
            - self.dst_offset_hours * 3600
        ) + 7200

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


NYC = DaylightSavingsTimezone(dst_offset_hours=-4, st_offset_hours=-5)
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
