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


class VertSequenceBasicConcat(CellsTestPage):
    def cell(self):
        return (cells.Text("Hello") + cells.Text("There") + cells.Text("This Is A") + cells.Button("Button", lambda: None))

    def text(self):
        return "Should use plus operator to produce a non-flexed vertical sequence"

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


class VertSequenceWithoutFlexNestedSeq(CellsTestPage):
    def cell(self):
        letteredTextItems = [cells.Text("Inner %s" % letter) for letter in ['A', 'B', 'C', 'D', 'E']]
        numberedTextItems = [cells.Text("Item %s" % i) for i in range(50)]
        numberedTextItems.append(cells.Sequence(letteredTextItems))

        firstButton = cells.Button("Top Button", lambda: None)
        lastButton = cells.Button("Bottom Button", lambda: None)

        return (firstButton + cells.Sequence(numberedTextItems) + lastButton)

    def text(self):
        return "Non Flex-Parent Sequences should flatten any nested Sequences and overflow normally"


class HorizSequenceWithFlex(CellsTestPage):
    def cell(self):
        textItems = [cells.Button("Inner Button %s" % i, lambda: None) for i in range(50)]
        middleArea = cells.HorizontalSequence(textItems)
        firstButton = cells.Button("First Button", lambda: None)
        secondButton = cells.Button("Another Button", lambda: None)
        return (firstButton >> Flex(middleArea) >> secondButton)

    def text(self):
        return "The vertical sequence should display two shrinkwrapped buttons sandwiching a flexed-out sequence containing a scrollable list of inner buttons"


class HorizSequenceWithoutFlex(CellsTestPage):
    def cell(self):
        buttonItems = [cells.Button("Inner Button %s" % i, lambda: None) for i in range(50)]
        firstButton = cells.Button("First Button", lambda: None)
        secondButton = cells.Button("Another Button", lambda: None)
        return cells.HorizontalSequence([firstButton] + buttonItems + [secondButton])

    def text(self):
        return "The vertical sequence should not be a flex container and should display all of its contents using as much space as it needs"

class HorizSequenceBasicConcat(CellsTestPage):
    def cell(self):
        result = (cells.Text("Hi") >> cells.Text("Bye") >> cells.Text("Go Away") >> cells.Button("Away", lambda: None))
        return result

    def text(self):
        return "Should produce a non-flexed HorizontalSequence of the 4 elements"


class MixedSequenceComposition(CellsTestPage):
    def cell(self):
        result = (cells.Text("Hi") >> cells.Text("Bye")) + (cells.Text("Go Away") >> cells.Button("Away", lambda: None))
        return result

    def text(self):
        return "You should see a complex composition of both kinds of Sequences"
