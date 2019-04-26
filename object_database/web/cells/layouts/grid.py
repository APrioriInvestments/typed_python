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

"""A Layout properties object that represents a CSS Grid.

GridLayout represents the parent element container of any
CSS grid structure, and handles all appropriate mapping
of rows at instantiation.

Like other Layout objects, GridLayout interfaces with
StyleAttributes objects and can return these to any
consumers of its behaviors.
"""
from object_database.web.html.styles import StyleAttributes
from io import StringIO


class GridLayout():
    def __init__(self, num_rows=2, num_columns=2, row_mapping=None, col_mapping=None):
        """
        Parameters
        ----------
        num_rows: int
            The number of rows that the grid will have.
        num_columns: int
            The number of columns that the grid will have.
        row_mapping: list
            A list of CSS dimension values for each item in the
            row. If nothing is passed, defaults to `1fr` (a
            relative measurement) for each item. List size should
            match the number of rows passed. Any valid CSS measurement
            property can be passed as a list item.
        col_mapping: list
            A list of CSS dimension values for each item in the
            row. If nothing is passed, defaults to `1fr` (a
            relative measurement) for each item. List size should
            match the number of columns passed. Any valid CSS
            measurment property can be passed as a list item.
        """
        self.update_rows(num_rows, row_mapping)
        self.update_columns(num_columns, col_mapping)

        """Grids are either inline-grid or just grid"""
        self.inline = False

        """A CSS value for gap between cells"""
        self.grid_gap = None


    def update_rows(self, num_rows, row_mapping):
        """Updates the mapping for the grid rows.

        Implementors can also pass a new number of rows
        as well as a new mapping.

        If `row_mapping` is None, we assume
        a measurment of `1fr` for each item of
        `num_rows` in a new list.

        Parameters
        ----------
        num_rows: int
            The new number of rows in the grid.
        row_mapping: list
            A list of CSS measurments for each item
            in the row. If `None`, we assume a list
            of `1fr` items of `num_rows` length
        """
        if not row_mapping:
            self.row_mapping = ['1fr' for num in range(num_rows)]
        elif num_rows != len(row_mapping):
            raise Exception("Number of GridLayout rows does not match provided mapping!")
        else:
            self.row_mapping = row_mapping
        self.num_rows = num_rows

    def update_columns(self, num_columns, col_mapping):
        """Updates the mapping for grid columns.

        Implementors can also pass a new number of columns
        as well as a new mapping.

        If `col_mapping` is `None`, we assume
        a measurment of `1fr` for each item in
        the list, which will be of length `num_columns`

        Parameters
        ----------
        num_columns: int
            The new number of columns for the grid.
        col_mapping: list
            A list of CSS measurments for each item
            in the row. If `None`, we assume a list
            of `1fr` for items of `num_columns` length
        """
        if not col_mapping:
            self.col_mapping = ['1fr' for num in range(num_columns)]
        elif num_columns != len(col_mapping):
            raise Exception("Number of GridLayout columns does not match provided mapping!")
        else:
            self.col_mapping = col_mapping
        self.num_columns = num_columns

    def get_style_inline(self):
        """Returns this layout's CSS style for inline HTML.

        Designed to return a string that can be inserted
        directly as an HTML Element's `style=` attribute.

        Returns
        -------
        string
            A formatted CSS style string ready
            to be applied to an element's inline
            `style` attribute.
        """
        styles = self._make_styles()
        return styles.as_string()

    def get_styles(self):
        """Returns the `StyleAttributes` instance for the layout.

        This method will re-calculate all of the internal values
        and settings when called, providing a way to grab an
        up to date `StyleAttributes` object that represents
        the layout

        Returns
        -------
        StyleAttributes
            A `StyleAttributes` instance that represents
            the CSS properties corresponding to the layout.
        """
        return self._make_styles()

    def _make_styles(self):
        styles = StyleAttributes()
        if self.inline:
            styles.add_style('display', 'inline-grid')
        else:
            styles.add_style('display', 'grid')

        if self.grid_gap:
            styles.add_style('grid-gap', self.grid_gap)

        self._make_row_style_on(styles)
        self._make_column_style_on(styles)
        return styles

    def _make_row_style_on(self, styles):
        """Formats and adds row settings/styles"""
        stream = StringIO()
        counted_list = self._counted_list_for(self.row_mapping)
        for track_val in counted_list:
            if track_val[1] > 1:
                stream.write("repeat({}, {}) ".format(track_val[1], track_val[0]))
            else:
                stream.write("{} ".format(track_val[0]))
        styles.add_style('grid-template-rows', stream.getvalue().rstrip())

    def _make_column_style_on(self, styles):
        """Formats and adds column settings/styles"""
        stream = StringIO()
        counted_list = self._counted_list_for(self.col_mapping)
        for track_val in counted_list:
            if track_val[1] > 1:
                stream.write("repeat({}, {}) ".format(track_val[1], track_val[0]))
            else:
                stream.write("{} ".format(track_val[0]))
        styles.add_style('grid-template-columns', stream.getvalue().rstrip())

    @staticmethod
    def _counted_list_for(main_list):
        """Returns a list of tupled representing ordered counts.

        Takes an input list of items and returns a list of
        tuples that represent the consecutive counts of each item.
        Note that this is different from a `Counter`, because it
        counts *consecutive* occurences of the same item.

        We use this to properly format consecutive row/column
        measurment values which are displayed using `repeat()`
        CSS function.

        Example
        -------
            ::
            input = [1, 2, 2, 2, 3, 3, 4, 5, 7, 7, 2, 2, 2, 9]
            output = self._counted_list_for(input)
            # [(1, 1), (2, 3), (3, 2), (4, 1), (5, 1), (7, 2), (2, 3), (9, 1)]
        """
        prev = None

        count = 1
        new_list = []
        prev = main_list[0]
        for i in range(1, len(main_list)):
            next_val = main_list[i]
            if next_val == prev:
                count += 1
            elif count > 1:
                new_list.append((prev, count))
                count = 1
            else:
                new_list.append((prev, 1))
            prev = next_val
        new_list.append((prev, count))
        return new_list
