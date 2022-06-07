#   Copyright 2017-2021 typed_python Authors
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

from typed_python import (
    civilfromdays, secondsfromcivil
)
from datetime import datetime, timezone
import time


def test_civilfromdays():
    unixtime = 1656879814  # 2022-07-03 20:23:34
    cfd = civilfromdays(unixtime)

    assert 2022 == cfd[0]
    assert 7 == cfd[1]
    assert 3 == cfd[2]
    assert 20 == cfd[3]
    assert 23 == cfd[4]
    assert 34 == cfd[5]


def test_civilfromdays_current():
    unixtime = time.time()
    dt_tuple = datetime.fromtimestamp(unixtime, tz=timezone.utc).timetuple()
    cfd = civilfromdays(unixtime)

    assert dt_tuple.tm_year == cfd[0]
    assert dt_tuple.tm_mon == cfd[1]
    assert dt_tuple.tm_mday == cfd[2]
    assert dt_tuple.tm_hour == cfd[3]
    assert dt_tuple.tm_min == cfd[4]
    assert dt_tuple.tm_sec == cfd[5]


def test_secondsfromcivil():
    unixtime = time.time()
    dt_tuple = datetime.fromtimestamp(unixtime, tz=timezone.utc).timetuple()

    dfc = secondsfromcivil(
        dt_tuple.tm_year,
        dt_tuple.tm_mon,
        dt_tuple.tm_mday,
        dt_tuple.tm_hour,
        dt_tuple.tm_min,
        dt_tuple.tm_sec,
        0,
        0,
        0,
    )

    print(dfc)
