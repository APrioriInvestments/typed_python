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

import abc


class BaseLayout(abc.ABC):
    """A Layout base ABC class.

    Not to to be used directly.

    """

    @abc.abstractmethod
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

    @abc.abstractmethod
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
