#   Copyright 2017-2020 typed_python Authors
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

from typed_python import Class, Final, Member, NamedTuple, Held
from typed_python.compiler.runtime import Entrypoint

Date = NamedTuple(tm_year=int, tm_mon=int, tm_mday=int, tm_hour=int, tm_min=int, tm_sec=int, tm_ms=int)

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
        return self.isoformat()

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

        return Date(tm_year=y, tm_mon=m, tm_mday=d, tm_hour=h, tm_min=min, tm_sec=s)

    @Entrypoint
    def _date(self, utc_offset: str = DEFAULT_UTC_OFFSET, fmt: str = "%Y-%m-%d %H:%M:%S") -> Date:
        return self.date(string_offset_to_seconds(utc_offset))

    @Entrypoint
    def strfrtime(self, utc_offset: int = DEFAULT_UTC_OFFSET, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
        # %Y-%m-%d %H:%M:%S"
        date = self.date(utc_offset)
        return f"{date.tm_year}-{date.tm_mon:02d}-{date.tm_mday:02d}T{date.tm_hour:02d}:{date.tm_min:02d}:{date.tm_sec:02d}"

    @Entrypoint
    def _strfrtime(self, utc_offset: str = DEFAULT_UTC_OFFSET) -> str:
        date = self.date(utc_offset)
        return f"{date.tm_year}-{date.tm_mon:02d}-{date.tm_mday:02d}T{date.tm_hour:02d}:{date.tm_min:02d}:{date.tm_sec:02d}"

    @Entrypoint
    @staticmethod
    def fromdate(year=0, mon=0, day=0, hr=0, min=0, sec=0, msec=0):
        # Implements the low level days_from_civil algorithm described here
        # http://howardhinnant.github.io/date_algorithms.html#civil_from_days
        year -= mon <= 2
        era = (year if year >= 0 else year - 399) // 400
        yoe = (year - era * 400)
        doy = (153 * ( mon - 3 if mon > 2 else mon + 9) + 2) // 5 + day - 1
        doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
        days = era * 146097 + doe - 719468

        ts = (days * 86400) + (hr * 3600) + (min * 60) + sec + (msec // 1000)
        return Timestamp(ts=ts)

    @Entrypoint
    def timefrstr(self, fmt: str = "%Y-%m-%d %H:%M:%S") -> None:
        return self.make(0)
