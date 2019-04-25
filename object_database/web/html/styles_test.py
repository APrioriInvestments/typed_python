import unittest
from object_database.web.html.styles import StyleAttributes

class StyleAttributesTests(unittest.TestCase):
    def setUp(self):
        pass

    def test_add_styles(self):
        initial_styles = {
            'top': 0,
            'display': 'none',
            'border-right': '1px solid green'
        }
        styles_to_add = {
            'display': 'block',
            'top': '50px',
            'left': '25px'
        }
        styles = StyleAttributes(**initial_styles)
        styles.add_styles(styles_to_add)
        self.assertTrue('left' in styles._attributes)
        self.assertTrue(styles._attributes['top'] == '50px')
        self.assertTrue(styles._attributes['display'] == 'block')
        self.assertTrue('border-right' in styles._attributes)

    def test_add_style_unique(self):
        first = StyleAttributes(top='0px', left='50px')
        second = StyleAttributes(top='100px', display='block')
        result = first + second
        self.assertFalse(first is result)
        self.assertFalse(second is result)
        self.assertTrue('display' in result._attributes)

    def test_add_style_new(self):
        initial_styles = {
            'top': '0px',
            'left': '50px'
        }
        styles = StyleAttributes(**initial_styles)
        styles.add_style('display', 'block')
        self.assertTrue('display' in styles._attributes)
        self.assertTrue(styles._attributes['display'] == 'block')

    def test_plus_style_new(self):
        initial_styles = {
            'top': '0px',
            'left': '50px'
        }
        styles = StyleAttributes(**initial_styles)
        other_styles = StyleAttributes(display='block')
        new_styles = styles + other_styles
        self.assertTrue('display' in new_styles._attributes)
        self.assertTrue(new_styles._attributes['display'] == 'block')

    def test_add_style_update(self):
        initial_styles = {
            'top': '0px',
            'left': '50px',
            'display': 'none'
        }
        styles = StyleAttributes(**initial_styles)
        styles.add_style('display', 'block')
        self.assertTrue(styles._attributes['display'] == 'block')

    def test_get_style(self):
        styles = StyleAttributes(top='0px', left='50px')
        result = styles.get_style('top')
        self.assertEqual('0px', result)

    def test_remove_style(self):
        styles = StyleAttributes(top="0", left="50px", display="block")
        styles.remove_style('top')
        self.assertFalse('top' in styles._attributes)

    def test_as_dict(self):
        styles = StyleAttributes(top="0px", left="50px", display="block")
        result = styles.as_dict()
        self.assertEqual(result, styles._attributes)

    def test_as_string(self):
        initial_styles = {
            'top': '0px',
            'left': '50px',
            'display': 'none'
        }
        expected = 'top: 0px; left: 50px; display: none;'
        styles = StyleAttributes(**initial_styles)
        result = styles.as_string()
        self.assertEqual(result, expected)

    def test_is_empty(self):
        empty = StyleAttributes()
        not_empty = StyleAttributes(display='none')
        self.assertTrue(empty.is_empty())
        self.assertFalse(not_empty.is_empty())

    def test_class_inline_string_from(self):
        styles = StyleAttributes(top="0", left="0", bottom="5px")
        expected = "top: 0; left: 0; bottom: 5px;"
        result = StyleAttributes.inline_string_from(styles)
        self.assertEqual(expected, result)
