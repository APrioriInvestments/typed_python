import unittest
from object_database.web.cells.layouts import GridLayout
from object_database.web.html.styles import StyleAttributes


class GridLayoutTests(unittest.TestCase):
    def setUp(self):
        pass

    def test__counted_list_for(self):
        before = [1, 2, 2, 2, 3, 4, 5, 5, 6, 2, 2, 7, 7]
        expected = [
            (1, 1),
            (2, 3),
            (3, 1),
            (4, 1),
            (5, 2),
            (6, 1),
            (2, 2),
            (7, 2)
        ]
        layout = GridLayout()
        result = layout._counted_list_for(before)
        self.assertEqual(expected, result)

    def test__make_row_style_on(self):
        layout = GridLayout(num_rows=5, row_mapping=['2fr', '1fr', '1fr', '3fr', '1fr'])
        styles = StyleAttributes()
        layout._make_row_style_on(styles)
        expected = '2fr repeat(2, 1fr) 3fr 1fr'
        result = styles.get_style('grid-template-rows')
        self.assertEqual(expected, result)
