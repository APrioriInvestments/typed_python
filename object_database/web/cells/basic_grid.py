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
from object_database.web.cells.layouts import GridChildStyler

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

class GridViewWithSidebar(Cell):
    def __init__(self, sidebarCell, contentCell, footerCell=None):
        super().__init__()
        layout_settings = {
            'num_columns': 2,
            'num_rows': 2,
            'col_mapping': ['200px', '1fr'],
            'row_mapping': ['1fr', '1fr']
        }
        self.sidebarCell = Cell.makeCell(sidebarCell)
        self.contentCell = Cell.makeCell(contentCell)
        self.footerCell = None

        # If we have a footer, we need to
        # add an extra row at the bottom.
        if footerCell:
            layout_settings['num_rows'] = 3
            layout_settings['row_mapping'].append('200px')
            self.footerCell = Cell.makeCell(footerCell)
        self.layout = GridLayout(**layout_settings)
        self._styleGridChildren()
        self.children = {
            '____child_sidebar__': self.sidebarCell,
            '____child_content__': self.contentCell
        }
        if self.footerCell:
            self.children['____child_footer__'] = self.footerCell

    def recalculate(self):
        element = (
            HTMLElement.div()
            .set_attribute('style', self.style().as_string())
            .add_classes(['cell', 'grid-view', 'grid-view-sidebar'])
            .with_children(
                HTMLTextContent('____child_sidebar__'),
                HTMLTextContent('____child_content__')
            )
        )
        if self.footerCell:
            element.add_child(HTMLTextContent('____child_footer__'))
        self.contents = str(element)

    def style(self):
        layout_style = self.layoutStyle()
        in_parent_style = self.styleInParentLayout()
        return layout_style + in_parent_style + self.baseStyles

    def _styleGridChildren(self):
        sidebar_styler = GridChildStyler(row_span=2)
        content_styler = GridChildStyler(row_span=2)
        if self.footerCell:
            footer_styler = GridChildStyler(column_span=2)
            self.footerCell.baseStyles.append(footer_styler.get_styles())
        self.sidebarCell.baseStyles.append(sidebar_styler.get_styles())
        self.contentCell.baseStyles.append(content_styler.get_styles())


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
