#   Copyright 2017-2023 typed_python Authors
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
    Entrypoint
)
import time


@pytest.mark.group_one
def test_knows_its_a_float():
    @Entrypoint
    def callTime():
        return time.time()

    assert callTime.resultTypeFor().typeRepresentation is float
    callTime()

    assert abs(callTime() - time.time()) < 0.001


@pytest.mark.group_one
def test_call_perf():
    @Entrypoint
    def callTimeNTimes(times):
        elapsed = 0.0
        t0 = time.time()

        for _ in range(times):
            t1 = time.time()

            elapsed += t1 - t0
            t0 = t1

        return elapsed

    callTimeNTimes(1)

    t0 = time.time()
    estElapsed = callTimeNTimes(10000000)
    t1 = time.time()

    elapsedAct = t1 - t0

    print(estElapsed, elapsedAct)

    assert .8 < estElapsed / elapsedAct < 1.2
