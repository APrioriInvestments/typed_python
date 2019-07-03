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
import datetime

from object_database.web import cells as cells
from object_database.web.CellsTestPage import CellsTestPage


class BasicDateTimePicker(CellsTestPage):
    def cell(self):
        # year, month, day, hour, minute, second
        aSlot = cells.Slot(datetime.datetime(2011, 10, 21, 12, 20, 15).timestamp())

        return cells.Card(
            cells.DateTimePicker(slot=aSlot) +
            cells.Subscribed(lambda: f"You picked {aSlot.get()} seconds") +
            cells.Subscribed(lambda: cells.Timestamp(aSlot.get())),
            padding=4
        )

    def text(self):
        return "You should see a datetime picker."
