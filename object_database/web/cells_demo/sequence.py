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


class VerticalSplitSequence(CellsTestPage):
    def cell(self):
        return cells.Sequence(
            [cells.Text("item 1", text_color="red"),
             cells.Text("item 2", text_color="blue")]
        )

    def text(self):
        return "You should see a vertically split sequence of text."


class HorizontalSplitSequence(CellsTestPage):
    def cell(self):
        return cells.Sequence(
            [cells.Text("item 1", text_color="red"),
             cells.Text("item 2", text_color="blue")],
            split="horizontal"
        )

    def text(self):
        return "You should see a horizontally split sequence of text."