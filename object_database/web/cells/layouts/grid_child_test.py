import unittest
from object_database.web.cells.layouts.grid_child import GridChildStyler


class GridChildStylerTests(unittest.TestCase):
    def setUp(self):
        pass

    def test__make_row_start(self):
        grid_child = GridChildStyler(row_start=3)
        expected = '3'
        result_styles = grid_child._make_row_styles()
        result = result_styles.get_style('grid-row-start')
        self.assertEqual(expected, result)

    def test__make_row_end(self):
        grid_child = GridChildStyler(row_end=5)
        expected = '5'
        result_styles = grid_child._make_row_styles()
        result = result_styles.get_style('grid-row-end')
        self.assertEqual(expected, result)

    def test__make_row_span(self):
        grid_child = GridChildStyler(row_span=6)
        expected = 'span 6'
        result_styles = grid_child._make_row_styles()
        result = result_styles.get_style('grid-row-end')
        self.assertEqual(expected, result)

    def test__make_row_ignores_missing_start(self):
        grid_child = GridChildStyler(row_end=5)
        result = grid_child._make_row_styles()
        self.assertFalse('grid-row-start' in result.as_dict())

    def test_row_span_takes_precedent(self):
        grid_child = GridChildStyler(row_start=1, row_end=4, row_span=6)
        expected = 'span 6'
        result = grid_child.get_styles().get_style('grid-row-end')
        self.assertEqual(expected, result)

    def test__make_column_start(self):
        grid_child = GridChildStyler(column_start=3)
        expected = '3'
        result_styles = grid_child._make_column_styles()
        result = result_styles.get_style('grid-column-start')
        self.assertEqual(expected, result)

    def test__make_column_end(self):
        grid_child = GridChildStyler(column_end=5)
        expected = '5'
        result_styles = grid_child._make_column_styles()
        result = result_styles.get_style('grid-column-end')
        self.assertEqual(expected, result)

    def test__make_column_span(self):
        grid_child = GridChildStyler(column_span=6)
        expected = 'span 6'
        result_styles = grid_child._make_column_styles()
        result = result_styles.get_style('grid-column-end')
        self.assertEqual(expected, result)

    def test__make_column_ignores_missing_start(self):
        grid_child = GridChildStyler(column_end=5)
        result = grid_child._make_column_styles()
        self.assertFalse('grid-column-start' in result.as_dict())

    def test_column_span_takes_precedent(self):
        grid_child = GridChildStyler(column_start=1, column_end=4, column_span=6)
        expected = 'span 6'
        result = grid_child.get_styles().get_style('grid-column-end')
        self.assertEqual(expected, result)
