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

"""A ChildLayout object that represents styling for children of Grid containers.
"""
from object_database.web.html.styles import StyleAttributes

class GridChildStyler():
    def __init__(self, row_start=None, row_end=None,
                 column_start=None, column_end=None, **kwargs):
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
        column = self._make_column_styles()
        row = self._make_row_styles()
        return column + row


    def _make_column_styles(self):
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
        styles = StyleAttributes()
        if self.row_end:
            styles.add_style('grid-row-end', str(self.row_end))

        if self.row_span:
            styles.add_style('grid-row-end',
                             "span {}".format(self.row_span))
        if self.row_start:
            styles.add_style('grid-row-start', str(self.row_start))
        return styles
