#   Copyright 2017-2019 Nativepython Authors
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


class BasicPanel(CellsTestPage):
    def cell(self):
        return cells.Panel(cells.Text("Button in a Panel") + cells.Button("A Button", lambda: None))

    def text(self):
        return "Should see Text and Button Vert Sequence inside of a bordered panel, taking up greedy space in both dimensions"


class BasicPanelInFlexSequence(CellsTestPage):
    def cell(self):
        outer = cells.Sequence([
            cells.Button("Top Button", lambda: None),
            cells.Flex(cells.Panel(
                cells.Text("A Flexing Panel"))),
            cells.Button("Bottom Button", lambda: None)
        ])
        return outer

    def text(self):
        return "Should see a flex parent vert sequence where the Panel is flexing as the child, no longer taking up 100% in both dimensions (since we are flexing)"
