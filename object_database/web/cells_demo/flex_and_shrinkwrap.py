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
from object_database.web.cells.util import Flex, ShrinkWrap


class VertSequenceWithFlex(CellsTestPage):
    def cell(self):
        textItems = [cells.Text("Item %s" % i) for i in range(50)]
        middleCard = cells.Card(cells.Sequence(textItems))
        firstButton = cells.Button("First Button", lambda: None)
        secondButton = cells.Button("Another Button", lambda: None)
        return (firstButton + Flex(middleCard) + secondButton)

    def text(self):
        return "The vertical sequence should display two shrinkwrapped buttons sandwiching a flexed-out card containing a scrollable list of items"


class VertSequenceWithMultiFlex(CellsTestPage):
    def cell(self):
        firstTextItems = [cells.Text("Item %s" % i) for i in range(50)]
        firstCard = cells.Card(cells.Sequence(firstTextItems))
        secondTextItems = [cells.Text("Item %s" % i) for i in range(60)]
        secondCard = cells.Card(cells.Sequence(secondTextItems))
        firstButton = cells.Button("First Button", lambda: None)
        secondButton = cells.Button("Another Button", lambda: None)
        return (firstButton + Flex(firstCard) + Flex(secondCard) + secondButton)

    def text(self):
        return "The vertical sequence should display two shrinkwrapped buttons sandwiching *two* flexed-out cards of equal, greedy size containing a scrollable list of items"


class VertSequenceWithoutFlex(CellsTestPage):
    def cell(self):
        textItems = [cells.Text("Item %s" % i) for i in range(50)]
        middleCard = cells.Card(cells.Sequence(textItems))
        firstButton = cells.Button("First Button", lambda: None)
        secondButton = cells.Button("Another Button", lambda: None)
        return (firstButton + middleCard + secondButton)

    def text(self):
        return "The vertical sequence should not be a flex container and should display all of its contents using as much space as it needs"
