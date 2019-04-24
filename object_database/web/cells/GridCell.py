"""
Basic Grid Cells that incoporate
a GridLayout
"""
from object_database.web.cells.layouts.GridLayout import GridLayout
from object_database.web.cells import Cell
from object_database.web.html.html_gen import HTMLElement, HTMLTextContent


class BasicGrid(Cell):
    def __init__(self, children, num_columns=2, num_rows=2):
        super().__init__()
        self.layout = GridLayout(num_columns, num_rows)
        self.children = {}
        for i in range(0, len(children)):
            self.children['____content_child_%s__' % i] = Cell.makeCell(children[i])

    def recalculate(self):
        child_placeholders = []
        for child_key, _ in self.children.items():
            child_placeholders.append(HTMLTextContent(child_key))

        self.contents = str(
            HTMLElement.div()
            .set_attribute('style', self.layout.get_style_inline())
            .add_classes(['cell', 'container-cell'])
            .add_children(child_placeholders)
        )


class ColorCell(Cell):
    def __init__(self, color):
        super().__init__()
        self.color = color

    def recalculate(self):
        return str(
            HTMLElement.div()
            .set_attribute('style', 'background-color:%s;' % self.color)
            .add_classes(['cell', 'color-cell'])
        )
