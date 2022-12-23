class TimeOfDay(Class, Final):
    hour = Member(int)
    minute = Member(int)
    second = Member(float)

class Date(Class, Final):
    year = Member(int)
    month = Member(int)
    day = Member(int)

class DateTime(Class, Final):
    date = Member(Date)
    timeOfDay = Member(TimeOfDay)

    @Entrypoint
    @property
    def year(self):
        return self.date.year

    @Entrypoint
    @property
    def month(self):
        return self.date.month

    @Entrypoint
    @property
    def day(self):
        return self.date.day

    @Entrypoint
    @property
    def hour(self):
        return self.timeOfDay.hour

    @Entrypoint
    @property
    def minute(self):
        return self.timeOfDay.minute

    @Entrypoint
    @property
    def second(self):
        return self.timeOfDay.second


class TimeZone(Class):
    @Entrypoint
    @statimethod
    def getConcurrentUtcDateTime(dateTime: DateTime, fold :int = 1) -> DateTime:
        raise NotImplementedError


class DaylightSavingsTimezone(TimeZone, Final):
    dst_offset = Member(int)
    st_offset = Member(int)

    @Entrypoint
    @staticmethod
    def _get_offset_to_utc(dateTime: DateTime, fold: int = 1) -> bool:
        ds_start = Chrono.get_neth_dow_of_month_ts(2, 0, 3, dateTime.year) + 7200
        ds_end = Chrono.get_nth_dow_of_month_ts(1, 0, 11, dateTime.year) + 7200

        return ts >= ds_start and ts <= ds_end

class FixedOffsetTimezone(TimeZone, Final):
    offset = Member(Offset)

    @Entrypoint
    @statimethod
    def getConcurrentUtcDateTime(dateTime: DateTime, fold :int = 1) -> DateTime:


        date = Date(year, month, day)
        timeOfDay = TimeOfDay(hour, minute, second)
        return DateTime(date=date, timeOfDay=timeOfDay)

class Offset(Class):
    def fromDatetime(dateTime: DateTime) -> DateTime:
        raise NotImplementedError

class FixedOffset(Offset, Final):





