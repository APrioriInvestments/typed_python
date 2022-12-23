from typed_python import Class, Final, Member, Entrypoint


class TimeOfDay(Class, Final):
    hour = Member(int)
    minute = Member(int)
    second = Member(float)


class Date(Class, Final):
    year = Member(int)
    month = Member(int)
    day = Member(int)

    @Entrypoint
    def daysSinceEpoch(self):
        return Chrono.days_from_civil(self.year, self.month, self.day)

    @Entrypoint
    def weekday(self):
        daysSinceEpoch = self.daysSinceEpoch()
        return Chrono.weekday_from_day(daysSinceEpoch)

    @Entrypoint
    @staticmethod
    def fromDaysSinceEpoch(daysSinceEpoch: int) -> Date:
        dateTuple = Chrono.civil_from_days(daysSinceEpoch)
        return Date(year=dateTuple.year, month=dateTuple.month, day=dateTuple.day)


class DateTime(Class, Final):
    date = Member(Date)
    timeOfDay = Member(TimeOfDay)


class TimeZone(Class):
    @Entrypoint
    @staticmethod
    def getConcurrentUtcDateTime(dateTime: DateTime, fold: int = 1) -> DateTime:
        raise NotImplementedError("Subclasses implement.")

    @Entrypoint
    @staticmethod
    def offsetDateTime(dateTime: DateTime, offset_hours: int) -> DateTime:
        hours = dateTime.timeOfDay.hour + offset_hours
        daysToAdd, hours = divmod(hours, 24)

        if daysToAdd != 0:
            days = dateTime.date.daysSinceEpoch() + daysToAdd
            date = Date.fromDaysSinceEpoch(days)
        else:
            date = dateTime.date

        return DateTime(
            date=date,
            timeOfDay=TimeOfDay(hour=hours, minute=dateTime.timeOfDay.minute, second=dateTime.timeOfDay.second),
        )


class DaylightSavingsTimezone(TimeZone, Final):
    dst_offset_hours = Member(int)
    st_offset_hours = Member(int)

    @Entrypoint
    @staticmethod
    def _is_daylight_savings(dateTime, occurence: int=1) -> bool:
        year = dateTime.date.year

        # second sunday of march 
        ds_start = Chrono.get_nth_dow_of_month(2, 0, 3, year)

        # first sunday of november
        ds_end = Chrono.get_nth_dow_of_month(1, 0, 11, year)

        day = dateTime.date.daysSinceEpoch()

        if day < ds_start:
            return False

        if day > ds_end:
            return False

        if day == ds_start:
            return dateTime.timeOfDay.hour>2

        if day == ds_end:
            return occurrence < 2 and dateTime.timeOfDay.hour<2

        return True


    @Entrypoint
    @staticmethod
    def getConcurrentUtcDateTime(self, dateTime: DateTime, occurence: int = 1) -> bool:
        offset_hours = self.dst_offset_hours if self._is_daylight_savings(dateTime, occurence) else self.st_offset_hours
        return self.offsetDateTime(dateTime, offset_hours)


class FixedOffsetTimezone(TimeZone, Final):
    offset_hours = Member(int)

    @Entrypoint
    def getConcurrentUtcDateTime(self, dateTime: DateTime, occurence: int = 1) -> DateTime:
        return self.offsetDateTime(dateTime, self.offset_hours)
