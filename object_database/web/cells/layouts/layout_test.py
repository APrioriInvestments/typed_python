import unittest
from object_database.web.cells.layouts.grid import GridLayout
from object_database.web.cells.layouts.flex import FlexLayout
from object_database.web.cells.layouts.flex_child import FlexChildStyler
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


class FlexLayoutTests(unittest.TestCase):
    def setUp(self):
        pass

    def test_row_style(self):
        layout = FlexLayout()
        layout._make_styles()
        style = layout.get_styles()
        result = style.get_style('display')
        expected = 'flex'
        result = style.get_style('display')
        self.assertEqual(expected, result)
        result = style.get_style('flex-direction')
        expected = 'row'
        self.assertEqual(expected, result)

    def test_col_style(self):
        layout = FlexLayout(direction="column")
        layout._make_styles()
        style = layout.get_styles()
        result = style.get_style('display')
        expected = 'flex'
        result = style.get_style('display')
        self.assertEqual(expected, result)
        result = style.get_style('flex-direction')
        expected = 'column'
        self.assertEqual(expected, result)

    def test_reversed_style(self):
        layout = FlexLayout(direction="column", reverse=True)
        layout._make_styles()
        style = layout.get_styles()
        result = style.get_style('display')
        expected = 'flex'
        result = style.get_style('display')
        self.assertEqual(expected, result)
        result = style.get_style('flex-direction')
        expected = 'column-reverse'
        self.assertEqual(expected, result)

    def test_align_options_style(self):
        layout = FlexLayout(wrap="wrap", align_items="stretch")
        layout._make_styles()
        style = layout.get_styles()
        result = style.get_style('display')
        expected = 'flex'
        result = style.get_style('display')
        self.assertEqual(expected, result)
        result = style.get_style('flex-wrap')
        expected = 'wrap'
        self.assertEqual(expected, result)
        result = style.get_style('align-items')
        expected = 'stretch'
        self.assertEqual(expected, result)

    def test_align_options_assert(self):
        with self.assertRaises(Exception):
            FlexLayout(wrap="BADINPUT")
        with self.assertRaises(Exception):
            FlexLayout(justify_content="BADINPUT")


class FlexChildStylerTests(unittest.TestCase):
    def setUp(self):
        pass

    def test_order(self):
        layout = FlexChildStyler()
        layout._make_styles()
        style = layout.get_styles()
        result = style.get_style('order')
        expected = '0'
        self.assertEqual(expected, result)
        layout = FlexChildStyler(order=-1)
        style = layout.get_styles()
        result = style.get_style('order')
        expected = '-1'
        self.assertEqual(expected, result)

    def test_flex_grow(self):
        layout = FlexChildStyler()
        layout._make_styles()
        style = layout.get_styles()
        result = style.get_style('flex-grow')
        expected = '0'
        self.assertEqual(expected, result)
        layout = FlexChildStyler(flex_grow=-1)
        style = layout.get_styles()
        result = style.get_style('flex-grow')
        expected = '-1'
        self.assertEqual(expected, result)

    def test_flex_shrink(self):
        layout = FlexChildStyler()
        layout._make_styles()
        style = layout.get_styles()
        result = style.get_style('flex-shrink')
        expected = '1'
        self.assertEqual(expected, result)
        layout = FlexChildStyler(flex_shrink=-1)
        style = layout.get_styles()
        result = style.get_style('flex-shrink')
        expected = '-1'
        self.assertEqual(expected, result)

    def test_flex_basis(self):
        layout = FlexChildStyler()
        layout._make_styles()
        style = layout.get_styles()
        result = style.get_style('flex-basis')
        expected = 'auto'
        self.assertEqual(expected, result)
        layout = FlexChildStyler(flex_basis="25%")
        style = layout.get_styles()
        result = style.get_style('flex-basis')
        expected = '25%'
        self.assertEqual(expected, result)

    def test_align_self(self):
        layout = FlexChildStyler()
        layout._make_styles()
        style = layout.get_styles()
        result = style.get_style('align-self')
        expected = 'auto'
        self.assertEqual(expected, result)
        layout = FlexChildStyler(align_self="flex-start")
        style = layout.get_styles()
        result = style.get_style('align-self')
        expected = 'flex-start'
        self.assertEqual(expected, result)

    def test_align_options_assert(self):
        with self.assertRaises(Exception):
            FlexLayout(order="BADINPUT")
        with self.assertRaises(Exception):
            FlexLayout(flex_basis="BADINPUT")
