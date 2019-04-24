from object_database.web.html.styles import StyleAttributes
from io import StringIO


class GridLayout():
    def __init__(self, num_rows=2, num_columns=2, row_mapping=None, col_mapping=None):
        self.update_rows(num_rows, row_mapping)
        self.update_columns(num_columns, col_mapping)

        # By default, we don't make
        # the grid inline
        self.inline = False


    def update_rows(self, num_rows, row_mapping):
        if not row_mapping:
            self.row_mapping = ['1fr' for num in range(num_rows - 1)]
        elif num_rows != len(row_mapping):
            raise Exception("Number of GridLayout rows does not match provided mapping!")
        else:
            self.row_mapping = row_mapping
        self.num_rows = num_rows

    def update_columns(self, num_columns, col_mapping):
        if not col_mapping:
            self.col_mapping = ['1fr' for num in range(num_columns - 1)]
        elif num_columns != len(col_mapping):
            raise Exception("Number of GridLayout columns does not match provided mapping!")
        else:
            self.col_mapping = col_mapping
        self.num_columns = num_columns

    def get_style_inline(self):
        styles = self._make_styles()
        return styles.as_string()

    def _make_styles(self):
        styles = StyleAttributes()
        if self.inline:
            styles.add_style('display', 'inline-grid')
        else:
            styles.add_style('display', 'grid')

        self._make_row_style_on(styles)
        return styles

    def _make_row_style_on(self, styles):
        stream = StringIO()
        counted_list = self._counted_list_for(self.row_mapping)
        for track_val in counted_list:
            if track_val[1] > 1:
                stream.write("repeat({}, {}) ".format(track_val[1], track_val[0]))
            else:
                stream.write("{} ".format(track_val[0]))
        styles.add_style('grid-template-rows', stream.getvalue().rstrip())

    def _make_column_style_on(self, styles):
        stream = StringIO()
        counted_list = self._counted_list_for(self.col_mapping)
        for track_val in counted_list:
            if track_val[1] > 1:
                stream.write("repeat({}, {}) ".format(track_val[1], track_val[0]))
            else:
                stream.write("{} ".format(track_val[0]))
        styles.add_style('grid-template-columns')

    @staticmethod
    def _counted_list_for(main_list):
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
