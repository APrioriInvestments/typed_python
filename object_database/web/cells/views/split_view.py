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
"""SplitView Cell

This module contains a Cell that is designed
as a view that can split its children proportionally
in either the horizontal or vertical axes
"""
from ..cells import Cell


class SplitView(Cell):
    """SplitView Cell

    I am a kind of of display wrapper around my
    passed in children (tuples) that displays them
    proportionally in either along the vertical or
    horizontal axis

    Example
    -------
    We can easily create a row where three children
    will be displayed with the middle child
    proportionally three times as wide as the
    smallest child::
        SplitView([
            (Card(Text("First Item")), 1),
            (Card(Text("Second Item")), 3),
            (Card(text("Third Item")), 1)
        ])
    """
    def __init__(self, childrenTuples=[], split="vertical"):
        """
        Parameters:
        -----------
        childrenTuples: list of tuples
            A list of size 2 tuples in which
            the first element is a Cell instance
            and the second is an integer representing
            the ratio of the width/height that the cell
            area should take up.
        split: str
            The split axis of the  view. Can
            be either 'horizontal' or 'vertical'. Defaults
            to 'vertical'.
        """
        super().__init__()
        self.childrenTuples = childrenTuples
        self.split = split

        self.updateReplacements()
        self.updateNamedChildren()
        self.updateProportionInfo()

    def recalculate(self):
        self.updateReplacements()
        self.updateNamedChildren()
        self.updateProportionInfo()

        # For now, we have to put something in contents
        # to avoid the Cell/Cells infrastructure
        # from throwing an error about replacements
        # not being used.
        self.contents = "\n".join(self.children.keys())

    def updateProportionInfo(self):
        """
        Cycles through the child tuples
        and pulls out the ratio/proportion
        values (second index), adding them as
        a separate list to exportData.
        Also sets the split axis property export.
        """
        proportions = []
        for childTuple in self.childrenTuples:
            proportions.append(childTuple[1])
        self.exportData['proportions'] = proportions
        self.exportData['split'] = self.split

    def updateReplacements(self):
        for i in range(len(self.childrenTuples)):
            childTuple = self.childrenTuples[i]
            self.children["____element_{}__".format(i)] = Cell.makeCell(childTuple[0])

    def updateNamedChildren(self):
        newChildren = []
        for i in range(len(self.childrenTuples)):
            childTuple = self.childrenTuples[i]
            newChildren.append(Cell.makeCell(childTuple[0]))
        self.namedChildren['elements'] = newChildren
