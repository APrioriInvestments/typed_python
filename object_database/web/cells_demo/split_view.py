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


class BasicColumn(CellsTestPage):
    def cell(self):
        return cells.SplitView([
            (cells.Card(cells.Text('First Element')), 1),
            (cells.Card(cells.Text('Second Element')), 2)
        ])

    def text(self):
        return "You should see two cards, the second twice as wide as the first"


class BasicRow(CellsTestPage):
    def cell(self):
        return cells.SplitView([
            (cells.Card(cells.Text('First Element')), 2),
            (cells.Card(cells.Text('Second Element')), 1)
        ], split="horizontal")

    def text(self):
        return "You should see two Cards on top of each other, the first twice as long as the second"


class NestedRowToColumn(CellsTestPage):
    def cell(self):
        return cells.SplitView([
            (cells.Card(
                cells.SplitView([
                    (cells.Card("Row 1 Column 1"), 1),
                    (cells.Card("Row 2 Column 1"), 1)
                ], split="horizontal"),
                padding=10), 2),
            (cells.Card(
                cells.SplitView([
                    (cells.Card("Row 1 Column 2"), 4),
                    (cells.Card("Row 2 Column 2"), 1)
                ], split="horizontal"),
                padding=10), 1)
        ])

    def text(self):
        return "You should see a SplitView row of two SplitView columns"
