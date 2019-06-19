"""ResizablePanel Cell

This Cell takes two child Cells as arguments and will
display them in a split view that has an adjustable
bar along the provided split axis
"""
from ..cells import Cell


class ResizablePanel(Cell):
    def __init__(self, first, second, split="vertical"):
        """
        Parameters
        ----------
        first (Cell) - The first child Cell to display,
                left or top depending on axis
        second (Cell) - The second child Cell to
                display, right or bottom dep. on axis
        split - Along which axis to split the view.
                defaults to vertical
        """
        super().__init__()
        self.first = first
        self.second = second
        self.split = split
        self.exportData['split'] = split
        self.updateReplacements()
        self.updateNamedChildren()
        self.contents = "____first__ ____second__"

    def recalculate(self):
        self.updateNamedChildren()
        self.updateReplacements()
        self.exportData['split'] = self.split

    def updateReplacements(self):
        self.children = {
            '____first__': Cell.makeCell(self.first),
            '____second__': Cell.makeCell(self.second)
        }

    def updateNamedChildren(self):
        self.namedChildren = {
            'first': Cell.makeCell(self.first),
            'second': Cell.makeCell(self.second)
        }
