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

from typed_python import Class, Final, Member, NamedTuple
from typed_python.compiler.runtime import Entrypoint

TimeTuple = NamedTuple(tm_year=int, tm_mon=int, tm_mday=int, tm_hour=int, tm_min=int, tm_sec=int)

DEFAULT_UTC_OFFSET = '+00:00'


def offset_to_seconds(utc_offset: str) -> int:
    segments = utc_offset.split(':')
    hrs = int(segments[0])
    mins = int(segments[1])

    seconds = hrs * 3600 + (mins * 60 if hrs > 0 else mins * -60)
    return seconds


class Timestamp(Class, Final):
    ts = Member(float)

    def __int__(self):
        return int(self.ts)

    def __float__(self):
        return self.ts

    def __str__(self):
        return self.isoformat()

    @Entrypoint
    def __init__(self, ts: float) -> None:
        self.ts = ts

    @Entrypoint
    def timetuple(self, utc_offset: str = DEFAULT_UTC_OFFSET) -> TimeTuple:
        # Implements the low level civil_from_days algorithm described here
        # http://howardhinnant.github.io/date_algorithms.html#civil_from_days

        utc_offset = DEFAULT_UTC_OFFSET if not utc_offset else utc_offset

        ts = self.ts + offset_to_seconds(utc_offset)
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

        return TimeTuple(tm_year=y, tm_mon=m, tm_mday=d, tm_hour=h, tm_min=min, tm_sec=s)

    @Entrypoint
    def isoformat(self, utc_offset: str = DEFAULT_UTC_OFFSET) -> str:
        tup = self.timetuple(utc_offset)
        return f"{tup.tm_year}-{tup.tm_mon:02d}-{tup.tm_mday:02d}T{tup.tm_hour:02d}:{tup.tm_min:02d}:{tup.tm_sec:02d}"
