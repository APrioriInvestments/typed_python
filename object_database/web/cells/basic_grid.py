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

"""Example consumer of GridLayout"""
from object_database.web.cells import Cell
from object_database.web.html.html_gen import HTMLElement, HTMLTextContent
from object_database.web.html.styles import StyleAttributes
from object_database.web.cells.layouts import GridLayout


class GridView(Cell):
    def __init__(self, children, num_columns=2, num_rows=2):
        super().__init__()
        self.layout = GridLayout(num_columns, num_rows)
        self.layout.grid_gap = '1em'
        self.children = {}
        for i in range(0, len(children)):
            self.children['____content_child_%s__' % i] = Cell.makeCell(children[i])

    def recalculate(self):
        child_placeholders = []
        for child_key, _ in self.children.items():
            child_placeholders.append(HTMLTextContent(child_key))

        self.contents = str(
            HTMLElement.div()
            .set_attribute('style', self.style().as_string())
            .add_classes(['cell', 'container-cell', 'grid-container'])
            .add_children(child_placeholders)
        )

    def style(self):
        layout_style = self.layoutStyle()
        in_parent_style = self.styleInParentLayout()
        return layout_style + in_parent_style

    def styleInParentLayout(self):
        if isinstance(self.parent.layout, GridLayout):
            return None
        return StyleAttributes(height='80vh', width='80vw')

class ColorCell(Cell):
    """Just an example. Not for prod."""
    def __init__(self, color, children=[]):
        super().__init__()
        self.color = color
        for i in range(len(children)):
            self.children['____content_child_%s__' % i] = Cell.makeCell(children[i])

    def recalculate(self):
        child_placeholders = []
        for child_key, _ in self.children.items():
            child_placeholders.append(HTMLTextContent(child_key))

        self.contents = str(
            HTMLElement.div()
            .set_attribute('style', self.style().as_string())
            .add_classes(['cell', 'color-cell'])
            .add_children(child_placeholders)
        )

    def style(self):
        layout_style = self.layoutStyle()
        self.baseStyles.add_style('background-color', self.color)
        in_parent_style = self.styleInParentLayout()
        return self.baseStyles + layout_style + in_parent_style
