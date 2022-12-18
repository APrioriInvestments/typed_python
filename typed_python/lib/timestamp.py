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
from typed_python import Class, Final, Member, Held
from typed_python.lib.datetime.date_parser import DateParser
from typed_python.lib.datetime.date_formatter import DateFormatter
from typed_python.lib.datetime.chrono import Chrono


@Held
class Timestamp(Class, Final):
    """A wrapper around a unix timestamp that adds functionality for parsing and string formatting"""
    ts = Member(float)

    @Entrypoint
    def __int__(self):
        return int(self.ts)

    @Entrypoint
    def __float__(self):
        return self.ts

    @Entrypoint
    def __str__(self):
        return self.format()

    @Entrypoint
    def __eq__(self, other) -> bool:
        return self.ts == other.ts

    @Entrypoint
    def __ne__(self, other) -> bool:
        return self.ts != other.ts

    @Entrypoint
    def __ge__(self, other) -> bool:
        return self.ts >= other.ts

    @Entrypoint
    def __gt__(self, other) -> bool:
        return self.ts > other.ts

    @Entrypoint
    def __lt__(self, other) -> bool:
        return self.ts < other.ts

    @Entrypoint
    def __le__(self, other) -> bool:
        return self.ts <= other.ts

    @Entrypoint
    def __add__(self, other):
        return Timestamp(ts=self.ts + other.ts)

    @Entrypoint
    def __sub__(self, other):
        return Timestamp(ts=self.ts - other.ts)

    @Entrypoint
    @staticmethod
    def make(ts: float):
        '''
        Creates a Timestamp from a float
        Parameters:
            ts: a float
        Returns:
            timestamp (Timestamp): A Timestamp
        '''
        return Timestamp(ts=ts)

    @Entrypoint
    def __init__(self, ts: float):
        self.ts = ts

    @Entrypoint
    @staticmethod
    def parse(date_str: str):
        '''
        Creates a Timestamp from date strings.
        Parameters:
            date_str (str): A date string. E.g 2022-07-30 17:56:46
        Returns:
            timestamp (Timestamp): A Timestamp
        '''
        return Timestamp(ts=DateParser.parse(date_str))

    @Entrypoint
    def format(self, utc_offset: int = 0, format: str = "%Y-%m-%d %H:%M:%S") -> str:
        '''
        Converts a Timestamp to a string in a given format
        Parameters:
            utc_offset (int): The offset from UTC in seconds
            format (str): A string specifying formatting directives. E.g. '%Y-%m-%dT%H:%M:%S'
        Returns:
            date_str(str): A string representing the date in the specified format. E.g. "Mon January 2, 2021"
        '''
        return DateFormatter.format(self.ts, utc_offset, format)

    @Entrypoint
    @staticmethod
    def from_date(year=0, month=0, day=0, hour=0, minute=0, second=0):
        '''
        Creates a Timestamp from date values.
        Parameters:
            year (int): The year
            month (int): The month. January: 1, February: 2, ....
            day (int): The day
            hour (int): The hour (0-23)
            minute (int): The minute
            second (float): The second.
        Returns:
            timestamp (Timestamp): A Timestamp
        '''
        return Timestamp(ts=Chrono.date_to_seconds(year, month, day) + Chrono.time_to_seconds(hour, minute, second))
