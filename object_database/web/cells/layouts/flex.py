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

from object_database.web.cells.layouts.layout import BaseLayout
from object_database.web.html.styles import StyleAttributes


class FlexLayout(BaseLayout):
    """A Layout properties object that represents a CSS Flex.

    FlexLayout represents the parent element container of any
    CSS flex structure, and handles all appropriate mapping
    of rows at instantiation.

    Like other Layout objects, FlexLayout interfaces with
    StyleAttributes objects and can return these to any
    consumers of its behaviors.
    """
    def __init__(self, direction="row", reverse=False, wrap="nowrap",
                 justify_content="flex-start", align_items="stretch",
                 align_content="flex-start"):
        """
        Parameters
        ----------
        direction: Str
            Flex direction; either "row" or "column".
        reverse: Bool
            Reverses the natural order of either rows or
            columns.
        wrap: Str
            "nowrap", "wrap" or "wrap-reverse"
        justify_content : Str
            "flex-start", "flex-end", "center", "space-between",
            "space-around", "space-evenly"
        align_items: Str
            "stretch", "flex-start", "flex-end", "center", "baseline"
        align_content: Str
            "flex-start", "flex-end", "center", "space-between",
            "space-around", "stretch"

        """
        dir_ops = ["row", "column"]
        assert direction in dir_ops, ("flex direction must be one of %s."
                                      % ", ".join(dir_ops))
        wrap_ops = ["nowrap", "wrap", "wrap-reverse"]
        assert wrap in wrap_ops, ("flex wrap must be one of must be one of %s."
                                  % ", ".join(wrap_ops))
        jc_ops = ["flex-start", "flex-end", "center", "space-between",
                  "space-around", "space-evenly"]
        assert justify_content in jc_ops, ("flex justify-contentmust be one"
                                           " of must be one of %s."
                                           % ", ".join(wrap_ops))
        ai_ops = ["stretch", "flex-start", "flex-end", "center", "baseline"]
        assert align_items in ai_ops, ("flex align-items must be on of %s."
                                       % ", ".join(ai_ops))
        ac_ops = ["flex-start", "flex-end", "center", "space-between",
                  "space-around", "stretch"]
        assert align_content in ac_ops, ("flex align-items must be on of %s." %
                                         ", ".join(ac_ops))
        self.direction = direction
        self.reverse = reverse
        self.wrap = wrap
        self.justify_content = justify_content
        self.align_items = align_items
        self.align_content = align_content

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
        styles.add_style('display', 'flex')

        if self.reverse:
            reverse = "-reverse"
        else:
            reverse = ""

        styles.add_style('flex-direction', self.direction + reverse)
        styles.add_style('flex-wrap', self.wrap)
        styles.add_style('justify-content', self.justify_content)
        styles.add_style('align-items', self.align_items)
        styles.add_style('align-content', self.align_content)

        return styles
