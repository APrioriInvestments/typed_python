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

from object_database.web import cells as cells
from object_database.web.CellsTestPage import CellsTestPage


class basicScrollable(CellsTestPage):
    def cell(self):
        return cells.Scrollable(
            cells.Sequence([
                cells.Card("This is a card", padding=2) for index in range(20)
            ]), height="200px"
        )

    def text(self):
        return "You should see some scrollable content."


class multiScrollable(CellsTestPage):
    def cell(self):
        return cells.SplitView([
            (cells.Card(
                cells.Scrollable(cells.Sequence([
                    cells.Card("Row %s of 20" % (item + 1)) for item in range(20)
                ]), height="200px"),
                padding=10), 1),
            (cells.Card(
                cells.Scrollable(cells.Sequence([
                    cells.Card("Row %s of 10" % (item + 1)) for item in range(10)
                ]), height="200px"),
                padding=10), 10),
        ])

    def text(self):
        return "You should see some scrollable content."


class basicNonScrollable(CellsTestPage):
    def cell(self):
        return cells.Scrollable(
            cells.Sequence([
                cells.Card("This is a card", padding=2) for index in range(20)
            ]), height=None
        )

    def text(self):
        return ("This content doesn't scroll; change the 'height' argument to "
                "a css height value (example: '100px').")
