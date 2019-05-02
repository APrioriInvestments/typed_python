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


class FlexChildStyler(Styler):
    """A Styler for generating styles for children of FlexLayouts.

        This class represents a partial implementation of the
        CSS Flex Specification as it pertains to child elements
        of CSS Flex parent elements. For a more in-depth
        explanation, see the MDN documentation at
        https://developer.mozilla.org/en-US/docs/Learn/CSS/CSS_layout/Flexbox
    """
    def __init__(self, order=0, flex_grow=0, flex_shrink=1, flex_basis="auto",
                 align_self="auto"):
        """
        Parameters
        ----------
        order: int
            Order in which this child element is laid out.
        flex_grow: int
            If set to 1 to one in all children, remaining space in the
            container will be distributed equally. If set to 2 on a given child
            it will (attempt) take take up twice of remaining space, etc.
        flex_shink: int
            Similar to flex_grow above, but allows the item to shrink if
            necessary.
        flex_basis: str
            Either "auto" or a length argument ("5%", "10rem" etc). Defines the
            default size before remaining space is distributed.
        align_self: str
            This allows the default "align-items" set on the parent to be
            overriden by a child. Options include "auto", "flex-start", "flex-end
           ", "center", "baseline", "stretch".
        """
        as_ops = ["auto", "flex-start", "flex-end", "center", "baseline",
                  "stretch"]
        assert align_self in as_ops, ("align_self must be on of %s." %
                                      ", ".join(as_ops))
        assert isinstance(order, int), ("order must be an int.")
        assert isinstance(flex_grow, int), ("flex_grow must be an int.")
        assert isinstance(flex_shrink, int), ("flex_shrink must be an int.")

        self.order = order
        self.flex_grow = flex_grow
        self.flex_shrink = flex_shrink
        self.flex_basis = flex_basis
        self.align_self = align_self

    def get_styles(self):
        """Returns a configured StyleAttributes instance.

        Returns
        -------
        StyleAttributes
            A StyleAttributes instance configured
            with the appropriate values for a
            grid element child.
        """
        return self._make_styles()

    def get_style_inline(self):
        """Returns a string of inline-formatted styling
        suited for inline HTML elements.

        Returns
        -------
        String
            An HTML inline formatted style string.
        """
        return self.get_styles().as_string()

    def _make_styles(self):
        """Returns a StyleAttributes instance for
        the columns as configured"""
        styles = StyleAttributes()
        styles.add_style('order', str(self.order))
        styles.add_style('flex-grow', str(self.flex_grow))
        styles.add_style('flex-shrink', str(self.flex_shrink))
        styles.add_style('flex-basis', self.flex_basis)
        styles.add_style('align-self', self.align_self)

        return styles

