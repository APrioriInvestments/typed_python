#   Coyright 2017-2019 Nativepython Authors
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

import time
from datetime import datetime
import pytz

from object_database.web import cells as cells
from object_database.web.CellsTestPage import CellsTestPage


class CurrentTimestamp(CellsTestPage):
    def cell(self):
        displayTime = time.time()
        return cells.Timestamp(displayTime)

    def text(self):
        return "You should see the current time and zone nicely displayed."


class DefinedTimestamp(CellsTestPage):
    def cell(self):
        nyc = pytz.timezone("America/New_York")
        timestamp = nyc.localize(datetime(2019, 7, 16, 0, 0, 0)).timestamp()

        return cells.Sequence([
            cells.Timestamp(timestamp + 3600 * i)
            for i in range(24)]
        )

    def text(self):
        return "You should see 24 hours of timestamps all correctly displayed with their hours."


class Milliseconds(CellsTestPage):
    def cell(self):
        nyc = pytz.timezone("America/New_York")
        timestamp = nyc.localize(datetime(2019, 7, 16, 8, 0, 0)).timestamp()

        return cells.Sequence([
            cells.Timestamp(timestamp + i / 1000)
            for i in [1, 2, 10, 20, 100, 200]]
        )

    def text(self):
        return "You should see '2019-07-16 08:00:00' followed by 1, 2, 10, 20, 100, 200 milliseconds."
