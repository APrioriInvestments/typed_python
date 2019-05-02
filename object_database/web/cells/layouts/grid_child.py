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

from object_database.web.html.styles import StyleAttributes
from object_database.web.cells.layouts.styler import Styler


class GridChildStyler(Styler):
    """A Styler for generating styles for children of GridLayouts.

        This class represents a partial implementation of the
        CSS Grid Specification as it pertains to child elements
        of CSS Grid parent elements. For a more in-depth
        explanation, see the MDN documentation at
        https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Grid_Layout
    """
    def __init__(self, row_start=None, row_end=None,
                 column_start=None, column_end=None, **kwargs):
        """
        Parameters
        ----------
        row_start: int
            The row grid-line number at which this
            child should start. Defaults to None.
        row_end: int
            The row grid-line number at which this
            child should end. Defaults to None.
        column_start: int
            The column grid-line number at which
            this child should start. Defaults to None.
        column_end: int
            The column grid-line number at which
            this child should end. Defaults to None.
        row_span: int
            An alternative to row_end. Determines
            how many cells to 'span' from the start
            on the row axis. Will override any set
            row_end value. Defaults to None.
        column_span: int
            An alternative to column_end. Determines
            how many cells to 'span' from the start
            on the column axis. Will override any
            set column_end value. Defaults to None.
        """
        self.row_start = row_start
        self.row_end = row_end
        self.column_start = column_start
        self.column_end = column_end
        self.row_span = None
        self.column_span = None
        if 'row_span' in kwargs:
            self.row_span = kwargs['row_span']
        if 'column_span' in kwargs:
            self.column_span = kwargs['column_span']

    def get_styles(self):
        """Returns a configured StyleAttributes instance.

        Returns
        -------
        StyleAttributes
            A StyleAttributes instance configured
            with the appropriate values for a
            grid element child.
        """
        column = self._make_column_styles()
        row = self._make_row_styles()
        return column + row

    def get_style_inline(self):
        """Returns a string of inline-formatted styling
        suited for inline HTML elements.

        Returns
        -------
        String
            An HTML inline formatted style string.
        """
        return self.get_styles().as_string()

    def _make_column_styles(self):
        """Returns a StyleAttributes instance for
        the columns as configured"""
        styles = StyleAttributes()
        if self.column_end:
            styles.add_style('grid-column-end', str(self.column_end))

        if self.column_span:
            styles.add_style('grid-column-end',
                             "span {}".format(self.column_span))
        if self.column_start:
            styles.add_style('grid-column-start', str(self.column_start))
        return styles

    def _make_row_styles(self):
        """Returns a StyleAttributes instance for
        the rows as configured"""
        styles = StyleAttributes()
        if self.row_end:
            styles.add_style('grid-row-end', str(self.row_end))

        if self.row_span:
            styles.add_style('grid-row-end',
                             "span {}".format(self.row_span))
        if self.row_start:
            styles.add_style('grid-row-start', str(self.row_start))
        return styles
