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

from object_database.web import cells
from object_database.web.CellsTestPage import CellsTestPage
from object_database.web.cells.util import Padding, Margin


class BasicPadding(CellsTestPage):

    def cell(self):
        buttons = [
            Padding(10, cells.Button("First", lambda: None)),
            Padding(10, cells.Button("Second", lambda: None)),
            Padding(10, cells.Button("Third", lambda: None))
        ]
        return cells.HorizontalSequence(buttons)

    def text(self):
        return "Should see a horizontal sequence of three buttons each with 10px padding"


class BasicMargin(CellsTestPage):
    def cell(self):
        buttons = [
            Margin(10, cells.Button("First", lambda: None)),
            Margin(10, cells.Button("Second", lambda: None)),
            Margin(10, cells.Button("Third", lambda: None))
        ]
        return cells.HorizontalSequence(buttons)

    def text(self):
        return "Should see a horizontal sequence of three buttons each with 10px margin"
