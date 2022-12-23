from typed_python import Class, Final, Member, Entrypoint


class TimeOfDay(Class, Final):
    hour = Member(int)
    minute = Member(int)
    second = Member(float)

    def secondsSinceMidnight(self, occurence=1: int):
        return occurence * (second + minute * 60 + hour * 3600)


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
    def timestamp(dateTime: DateTime, occurence: int = 1) -> float:
        raise NotImplementedError("Subclasses implement.")


class DaylightSavingsTimezone(TimeZone, Final):
    dst_offset_hours = Member(int)
    st_offset_hours = Member(int)

    @Entrypoint
    @staticmethod
    def timestamp(self, dateTime: DateTime, occurence: int = 1) -> bool:

        year = dateTime.date.year

        # second sunday of march 
        ds_start = Chrono.get_nth_dow_of_month(2, 0, 3, year)

        # first sunday of november
        ds_end = Chrono.get_nth_dow_of_month(1, 0, 11, year)

        day = dateTime.date.daysSinceEpoch()

        is_daylight_savings = True

        if day < ds_start:
            is_daylight_savings = False

        if day > ds_end:
            is_daylight_savings = False

        if day == ds_start:
            is_daylight_savings = dateTime.timeOfDay.hour>2

        if day == ds_end:
            is_daylight_savings = occurrence < 2 and dateTime.timeOfDay.hour<2

        offset_hours = self.dst_offset_hours if is_daylight_savings else self.st_offset_hours

        return days * 86400 + offset_hours * 3600 + dateTime.timeOfDay.secondsSinceMidnight(occurence)



class FixedOffsetTimezone(TimeZone, Final):
    offset_hours = Member(int)

    @Entrypoint
    def timestamp(self, dateTime: DateTime, occurence: int = 1) -> DateTime:
        return dateTime.date.daysSinceEpoch() * 86400 + dateTime.timeOfDay.secondsSinceMidnight(occurence) + self.offset_hours * 3600
