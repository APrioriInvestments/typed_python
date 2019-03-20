#!/usr/bin/env python3

import re
import unittest

from html_gen import HTMLElement, HTML_TAG_CONFIG, HTMLElementChildrenError
from io import StringIO


class HTMLGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.current_tag_names = [item["tag_name"] for item in HTML_TAG_CONFIG]

    def test_add_child(self):
        test_child = HTMLElement()
        test_parent = HTMLElement()
        test_parent.add_child(test_child)
        self.assertTrue(test_child in test_parent.children)

    def test_add_child_self_closing(self):
        test_child = HTMLElement()
        test_parent = HTMLElement(is_self_closing=True)

        self.assertRaises(HTMLElementChildrenError, test_parent.add_child,
                          test_child)

    def test_add_children(self):
        kids = [HTMLElement() for i in range(0, 10)]
        test_parent = HTMLElement()
        test_parent.add_children(kids)
        all_present = True
        for kid in kids:
            if kid not in test_parent.children:
                all_present = False
        self.assertTrue(all_present)

    def test_print_on_basic(self):
        stream = StringIO()
        element = HTMLElement('div')
        element.attributes['class'] = 'column-4 medium'
        element.print_on(stream, newlines=False)
        output = stream.getvalue()
        self.assertEqual('<div class="column-4 medium"></div>', output)

    def test_print_on_nested(self):
        stream = StringIO()
        child = HTMLElement('p')
        child.attributes['class'] = 'column-4 medium'
        parent = HTMLElement('div', children=[child])
        parent.attributes['class'] = 'column-4 medium'
        parent.print_on(stream, newlines=False)
        output = stream.getvalue()
        test_out =('<div class="column-4 medium"><p class="column-4 medium">' +
                  '</p></div>')
        # we don't care about white spaces or new linesso much
        output = re.sub('\s{2,}', '', output)
        output = re.sub(r'\n', '', output)
        self.assertEqual(test_out, output)

    def test_print_on_set_boundmethod(self):
        stream = StringIO()
        element = HTMLElement.div()
        element.attributes['class'] = 'column-4 medium'
        element.print_on(stream, newlines=False)
        output = stream.getvalue()
        self.assertEqual('<div class="column-4 medium"></div>', output)

    def test_list_methods(self):
        method_names = HTMLElement.all_methods
        for name in self.current_tag_names:
            self.assertIn(name, method_names)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
