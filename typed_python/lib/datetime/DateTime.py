from typed_python import Class, Final, Member, Entrypoint
from typed_python.lib.datetime.chrono import Chrono

class NonexistentDateTime(Exception):
    pass

class TimeOfDay(Class, Final):
    hour = Member(int)
    minute = Member(int)
    second = Member(float)

    def secondsSinceMidnight(self, afterFold: bool=False) -> float:
        return (self.second + self.minute * 60 + self.hour * 3600) + afterFold * 3600

    @Entrypoint
    def __eq__(self, other):
        return self.hour == other.hour and self.minute == other.minute and self.second == other.second


class Date(Class, Final):
    year = Member(int)
    month = Member(int)
    day = Member(int)

    @Entrypoint
    def __eq__(self, other):
        return self.year == other.year and self.month == other.month and self.day == other.day

    @Entrypoint
    def daysSinceEpoch(self):
        return Chrono.days_from_civil(self.year, self.month, self.day)

    @Entrypoint
    def weekday(self):
        daysSinceEpoch = self.daysSinceEpoch()
        return Chrono.weekday_from_day(daysSinceEpoch)

    @Entrypoint
    @staticmethod
    def fromDaysSinceEpoch(daysSinceEpoch: int):
        dateTuple = Chrono.civil_from_days(daysSinceEpoch)
        return Date(year=dateTuple.year, month=dateTuple.month, day=dateTuple.day)


class DateTime(Class, Final):
    date = Member(Date)
    timeOfDay = Member(TimeOfDay)

    def __eq__(self, other):
        return self.date == other.date and self.timeOfDay == other.timeOfDay



class TimeZone(Class):
    @Entrypoint
    def timestamp(dateTime: DateTime, afterFold: bool = False) -> float:
        raise NotImplementedError("Subclasses implement.")


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

        if day > ds_end:
            is_daylight_savings = False

        if day == ds_start:
            if dateTime.timeOfDay.hour == 2:
                raise NonexistentDateTime("This date time does not exist on this date.")

            is_daylight_savings = dateTime.timeOfDay.hour > 2

        if day == ds_end:
            if dateTime.timeOfDay.hour > 2:
                afterFold = True
                is_daylight_savings = True
            else:
                is_daylight_savings = afterFold < 2 and dateTime.timeOfDay.hour < 2

        offset_hours = self.dst_offset_hours if is_daylight_savings else self.st_offset_hours

        return day * 86400 + offset_hours * 3600 + dateTime.timeOfDay.secondsSinceMidnight(afterFold)

    @Entrypoint
    def datetime(self, timestamp) -> DateTime:
        daysSinceEpoch, secondsSinceMidnight = divmod(timestamp, 86400)
        utcDateNt = Chrono.civil_from_days(daysSinceEpoch)

        # second sunday of march 
        ds_start = Chrono.get_nth_dow_of_month(2, 0, 3, utcDateNt.year) + self.st_offset_hours * 3600

        # first sunday of november
        ds_end = Chrono.get_nth_dow_of_month(1, 0, 11, utcDateNt.year) + self.dst_offset_hours * 3600

        # get offset
        offset_hours = self.dst_offset_hours if timestamp < ds_start and timestamp > ds_end else self.st_offset_hours

        # adjust seconds since midnight for timezone and re-compute into datetime.
        secondsSinceMidnight += offset_hours * 3600
        daysToAdd, secondsSinceMidnight = divmod(secondsSinceMidnight, 86400)
        hour, seconds = divmod(secondsSinceMidnight, 3600)
        minute, second = divmod(seconds, 60)

        date = Chrono.civil_from_days(daysSinceEpoch + daysToAdd) if daysToAdd else utcDateNt
        tod = TimeOfDay(hour=hour, minute=minute, second=second)
        return DateTime(date=Date(year=date.year, month=date.month, day=date.day), timeOfDay = tod)




class FixedOffsetTimezone(TimeZone, Final):
    offset_hours = Member(int)

    @Entrypoint
    def timestamp(self, dateTime: DateTime, afterFold: bool = False) -> float:
        return dateTime.date.daysSinceEpoch() * 86400 + dateTime.timeOfDay.secondsSinceMidnight(afterFold) + self.offset_hours * 3600

    @Entrypoint
    def datetime(self, timestamp) -> DateTime:
        daysSinceEpoch, secondsSinceMidnight = divmod(timestamp, 86400)
        utcDateNt = Chrono.civil_from_days(daysSinceEpoch)

        offset_hours = self.offset_hours

        # adjust seconds since midnight for timezone and re-compute into datetime.
        secondsSinceMidnight += offset_hours * 3600
        daysToAdd, secondsSinceMidnight = divmod(secondsSinceMidnight, 86400)
        hour, seconds = divmod(secondsSinceMidnight, 3600)
        minute, second = divmod(seconds, 60)

        date = Chrono.civil_from_days(daysSinceEpoch + daysToAdd) if daysToAdd else utcDateNt
        tod = TimeOfDay(hour=hour, minute=minute, second=second)
        return DateTime(date=Date(year=date.year, month=date.month, day=date.day), timeOfDay = tod)

