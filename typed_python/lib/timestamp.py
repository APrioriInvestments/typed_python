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


from typed_python import Class, Final, Member, Held
from typed_python.lib.datetime.date_parser import DateParser
from typed_python.lib.datetime.date_formatter import DateFormatter
from typed_python.lib.datetime.date_time import UTC, NYC, Timezone


@Held
class Timestamp(Class, Final):
    """A wrapper around a unix timestamp that adds functionality for parsing and string formatting"""

    ts = Member(float)

    def __int__(self):
        return int(self.ts)

    def __float__(self):
        return self.ts

    def __str__(self):
        return str(self.ts)

    def __eq__(self, other) -> bool:
        return self.ts == other.ts

    def __ne__(self, other) -> bool:
        return self.ts != other.ts

    def __ge__(self, other) -> bool:
        return self.ts >= other.ts

    def __gt__(self, other) -> bool:
        return self.ts > other.ts

    def __lt__(self, other) -> bool:
        return self.ts < other.ts

    def __le__(self, other) -> bool:
        return self.ts <= other.ts

    def __add__(self, other: float):
        return Timestamp(ts=self.ts + other)

    def __sub__(self, other) -> float:
        return self.ts - other.ts

    def midpoint(self, other):
        return self + (other - self) / 2

    @staticmethod
    def make(ts: float):
        return Timestamp(ts=ts)

    def __init__(self, ts: float):
        self.ts = ts

    @staticmethod
    def parse(date_str: str, timezone: Timezone):  # noqa: F811
        return Timestamp(ts=DateParser.parse_with_timezone(date_str, timezone))

    @staticmethod
    def parse_nyc(date_str: str):
        return Timestamp(ts=DateParser.parse_with_timezone(date_str, NYC))

    @staticmethod
    def parse(date_str: str):  # noqa: F811
        return Timestamp(ts=DateParser.parse(date_str))

    def format(self, timezone: Timezone = UTC, format: str = "%Y-%m-%d %H:%M:%S") -> str:
        """ Converts a Timestamp to a formatted string in a given timezone

        Parameters
        ----------
        timezone : Timezone
            The timezone in which to compute formatted string of the UTC timestamp.
        format : str
            The format for the string.
        """
        return DateFormatter.format(self.ts, timezone, format)

    def format_nyc(self, format: str = "%Y-%m-%d %H:%M:%S") -> str:
        """ Converts a Timestamp to a formatted string in NYC

        Parameters
        ----------
        format : str
            The format for the string.
        """
        return DateFormatter.format(self.ts, NYC, format)

    def weekday(self, timezone: Timezone) -> int:
        """The day of the week of the timestamp in a given timezone.
            0 => Sunday,  1 => Monday

        Parameters
        ----------
        timezone : Timezone
            The timezone in which to compute the weekday.

        """
        return timezone.datetime(self.ts).date.weekday()

    def dayOfYear(self, timezone: Timezone) -> int:
        """Get the day of the year for the timestamp in a given timezone..

        Parameters
        ----------
        timezone : Timezone
            The timezone in which to compute the day of the year.

        """
        return timezone.datetime(self.ts).date.dayOfYear()

    def quarterOfYear(self, timezone: Timezone) -> int:
        """Get the quarter of the year for the timestamp in a given timezone..

        Parameters
        ----------
        timezone : Timezone
            The timezone in which to compute the day of the year.

        """
        return timezone.datetime(self.ts).date.quarterOfYear()
