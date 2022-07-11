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

from typed_python.compiler.runtime import Entrypoint
from typed_python._types import civilfromdays, secondsfromcivil
from typed_python import Class, Final, Member, NamedTuple, Held
from time import strptime, strftime

Date = NamedTuple(year=int, month=int, day=int, hour=int, minute=int, second=int, ms=int, us=int, ns=int, weekday=int, doy=int)

DEFAULT_UTC_OFFSET = 0


def string_offset_to_seconds(utc_offset: str) -> int:
    offset = ''.join(utc_offset.split(':'))
    hrs = int(offset[0:3])
    mins = int(offset[3:5])
    return hrs * 3600 + (mins * 60 if hrs > 0 else mins * -60)


@Held
class Timestamp(Class, Final):
    ts = Member(float)

    @staticmethod
    def make(ts):
        return Timestamp(ts=ts)

    def __int__(self):
        return int(self.ts)

    def __float__(self):
        return self.ts

    def __str__(self):
        return self.to_string()

    @Entrypoint
    def date(self, utc_offset: int = DEFAULT_UTC_OFFSET) -> Date:
        '''
        Creates a Date tuple from this timestamp representing a date
        Parameters:
            utc_offset (int): The offset fromt UTC in seconds

        Returns:
            date (Date): a Date tuple. E.g. Date(year=2022, month=6, day=1, hour=5, minute=45, second=30, ms=1, us=4, ns=33)
        '''
        utc_offset = DEFAULT_UTC_OFFSET if utc_offset is None else utc_offset
        ts = self.ts + utc_offset

        cfd = civilfromdays(ts)

        return Date(
            year=cfd[0],
            month=cfd[1],
            day=cfd[2],
            hour=cfd[3],
            minute=cfd[4],
            second=cfd[5],
            ms=cfd[6],
            us=cfd[7],
            ns=cfd[8],
            weekday=cfd[9],
            doy=cfd[10]
        )

    @Entrypoint  # noqa : F811
    def date(self, utc_offset: str = DEFAULT_UTC_OFFSET) -> Date:
        '''
        Creates a Date tuple from this timestamp representing a date
        Parameters:
            utc_offset (string): The offset from UTC as a string. E.g. '+0200' or '+02:00'

        Returns:
            date (Date): a Date tuple. E.g. Date(year=2022, month=6, day=1, hour=5, minute=45, second=30, ms=1, us=4, ns=33)
        '''
        return self.date(string_offset_to_seconds(utc_offset))

    @Entrypoint
    def to_string(self, utc_offset: int = DEFAULT_UTC_OFFSET, format: str = "%Y-%m-%d %H:%M:%S") -> str:
        '''
        Converts a Timestamp to a string in a given format
        Parameters:
            utc_offset (int): The offset from UTC in seconds
            format (str): A string specifying formatting directives. E.g. '%Y-%m-%d %H:%M:%S'
        Returns:
            date_str(str): A string represening the date in the specified format. E.g. "Mon January 2, 2021"
        '''
        d = self.date(utc_offset)
        return strftime(format, (d.year, d.month, d.day, d.hour, d.minute, d.second, d.weekday, d.doy, 0))

    @ Entrypoint  # noqa : F811
    def to_string(self, utc_offset: str = DEFAULT_UTC_OFFSET, format: str = "%Y-%m-%d %H:%M:%S") -> str:
        '''
        Converts a Timestamp to a string in a given format
        Parameters:
            utc_offset (string): The offset from UTC as a string. E.g. '+0200' or '+02:00'
            format (str): A string specifying formatting directives. E.g. '%Y-%m-%d %H:%M:%S'
        Returns:
            date_str(str): A string represening the date in the specified format. E.g. "Mon January 2, 2021"
        '''
        date = self.date(utc_offset, format)
        return f"{date.year}-{date.month:02d}-{date.day:02d} {date.hour:02d}:{date.minute:02d}:{date.second:02d}"

    @ Entrypoint
    @ staticmethod
    def from_date(year=0, month=0, day=0, hour=0, minute=0, second=0, ms=0, us=0, ns=0):
        '''
        Creates a Timestamp from date values.
        Parameters:
            year (int): The year
            month (int): The month. January: 1, February: 2, ....
            day (int): The day
            hour (int): The hour (0-23)
            minute (int): The minute
            second (int): The second
            ms (int): The millisecond
            us (int): The microsecond
            ns(int): The nanosecond
        Returns:
            timestamp (Timestamp): A Timestamp
        '''

        ts = secondsfromcivil(
            year,
            month,
            day,
            hour,
            minute,
            second,
            ms,
            us,
            ns
        )
        return Timestamp(ts=ts)

    @ Entrypoint  # noqa : F811
    @ staticmethod
    def parse(date_str: str, format: str = "%Y-%m-%d %H:%M:%S"):
        '''
        Creates a Timestamp from date values.
        Parameters:
            date_str (str): A date string
            format (str): a format string. e.g. "%Y-%m-%d %H:%M:%S". defaults to "%Y-%m-%d %H:%M:%S"
        Returns:
            timestamp (Timestamp): A Timestamp
        '''
        # doesn't do sub seconds
        # consider https://github.com/closeio/ciso8601
        # or could preprocess the format string, grab the subsecond part and add it to the timestamp

        time_struct = strptime(date_str, format)
        return Timestamp.from_date(year=time_struct.tm_year,
                                   month=time_struct.tm_mon,
                                   day=time_struct.tm_mday,
                                   hour=time_struct.tm_hour,
                                   minute=time_struct.tm_min,
                                   second=time_struct.tm_sec)
