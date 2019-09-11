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
from object_database.web.cells.util import Flex


class VerticalSubscribedSequenceNotFlex(CellsTestPage):
    def cell(self):
        x = cells.Slot(())
        top_button = cells.Button("Add Item", lambda: x.set(x.get() + (len(x.get()),)))
        bottom_button = cells.Button("This does nothing", lambda: None)
        sub_sequence = cells.SubscribedSequence(
            lambda: x.get(),
            lambda item: cells.Text("Item: {}".format(item))
        )
        return (top_button + sub_sequence + bottom_button)

    def text(self):
        return "Should display a vertical sequence that is NOT flexed"


class VerticalSubscribedSequenceFlexed(CellsTestPage):
    def cell(self):
        x = cells.Slot(())
        top_button = cells.Button("Add Item", lambda: x.set(x.get() + (len(x.get()),)))
        bottom_button = cells.Button("This does nothing", lambda: None)
        sub_sequence = cells.SubscribedSequence(
            lambda: x.get(),
            lambda item: cells.Text("Item: {}".format(item))
        )
        return (top_button + Flex(sub_sequence) + bottom_button)

    def text(self):
        return "Should display a vertical sequence that is NOT flexed"
