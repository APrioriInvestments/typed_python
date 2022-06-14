from typed_python.compiler.runtime import Entrypoint
from typed_python import Class, Final, Member, NamedTuple, Held
from calendar import day_abbr, day_name, month_abbr, month_name

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


Date = NamedTuple(year=int, month=int, day=int, hour=int, minute=int, second=int, ms=int, weekday=int)

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
        # Implements the low level civil_from_days algorithm described here
        # http://howardhinnant.github.io/date_algorithms.html#civil_from_days
        utc_offset = DEFAULT_UTC_OFFSET if utc_offset is None else utc_offset
        ts = self.ts + utc_offset
        z = ts // 86400 + 719468
        era = (z if z >= 0 else z - 146096) // 146097
        doe = z - era * 146097
        yoe = (doe - (doe // 1460) + (doe // 36524) - (doe // 146096)) // 365
        y = yoe + era * 400
        doy = doe - (365 * yoe + (yoe // 4) - (yoe // 100))
        mp = (5 * doy + 2) // 153
        d = doy - (153 * mp + 2) // 5 + 1
        m = mp + (3 if mp < 10 else -9)
        y += (m <= 2)

        h = (ts // 3600) % 24
        min = (ts // (3600 / 60)) % 60
        s = (ts // (3600 / 60 / 60)) % (60)

        days = ts // 86400

        weekday = (days + 4) % 7 if days >= -4 else (days + 5) % 7 + 6

        return Date(year=y, month=m, day=d, hour=h, minute=min, weekday=weekday, second=s)

    @Entrypoint  # noqa : F811
    def date(self, utc_offset: str = DEFAULT_UTC_OFFSET, fmt: str = "%Y-%m-%d %H:%M:%S") -> Date:
        return self.date(string_offset_to_seconds(utc_offset))

    @Entrypoint
    def to_string(self, utc_offset: int = DEFAULT_UTC_OFFSET, format: str = "%Y-%m-%d %H:%M:%S") -> str:
        date = self.date(utc_offset)
        # reduce .replace calls for default format
        if format is None or format == "%Y-%m-%d %H:%M:%S":
            return f"{date.year}-{date.month:02d}-{date.day:02d} {date.hour:02d}:{date.minute:02d}:{date.second:02d}"

        return (
            format
            .replace("%Y", str(date.year))
            .replace("%y", str(date.year % 100))
            .replace("%b", str(month_abbr[date.month]))
            .replace("%B", str(month_name[date.month]))
            .replace("%a", str(day_abbr[date.weekday - 1]))
            .replace("%A", str(day_name[date.weekday - 1]))
            .replace('%m', f'{date.month:02d}')
            .replace('%d', f'{date.day:02d}')
            .replace('%H', f'{date.hour:02d}')
            .replace('%M', f'{date.minute:02d}')
            .replace('%S', f'{date.second:02d}')
        )

    @Entrypoint  # noqa : F811
    def to_string(self, utc_offset: str = DEFAULT_UTC_OFFSET, format: str = "%Y-%m-%d %H:%M:%S") -> str:
        date = self.date(utc_offset, format)
        return f"{date.year}-{date.month:02d}-{date.day:02d} {date.hour:02d}:{date.minute:02d}:{date.second:02d}"

    @Entrypoint
    @staticmethod
    def from_date(year=0, month=0, day=0, hour=0, minute=0, second=0, ms=0, us=0, ns=0):
        # Implements the low level days_from_civil algorithm described here
        # http://howardhinnant.github.io/date_algorithms.html#civil_from_days
        year -= month <= 2
        era = (year if year >= 0 else year - 399) // 400
        yoe = (year - era * 400)
        doy = (153 * ( month - 3 if month > 2 else month + 9) + 2) // 5 + day - 1
        doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
        days = era * 146097 + doe - 719468

        ts = (days * 86400) + (hour * 3600) + (minute * 60) + second + (ms // 1000) + (us / 1000000) + (ns / 1000000000)
        return Timestamp(ts=ts)

    @Entrypoint  # noqa : F811
    @staticmethod
    def from_date(date_str: str, format: str):
        # parse a datestring and return a timestamp
        return 0

    @Entrypoint
    def timefrstr(self, fmt: str = "%Y-%m-%d %H:%M:%S") -> None:
        return self.make(0)
