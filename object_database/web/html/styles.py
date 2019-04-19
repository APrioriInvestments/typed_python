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

"""HTML Style Attributes and String Generation

This module contains classes that deal specifically
with mapping CSS styles and generating dictionaries
and inline HTML strings from those values
"""
from io import StringIO

class StyleAttributes():
    """CSS Style Attributes class.

    This class deals with all operations
    that concern CSS Style key/value pairs
    that will ideally be linked to or displayed
    inline within HTMLElements.

    Its primary responsibility is to update and
    present dictionary and inline HTML string
    representations of CSS Style information.

    Its primary consumers should be HTMLElement
    and and Layout classes"""
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        kwargs: dict
            A dictionary of key value pairs
            corresponding to CSS Style values.
            We use the **kwargs form as a convenience
        """
        self._attributes = kwargs

    def add_styles(self, aDictionary):
        """Merges a passed Dictionary of new
        style key value pairs into the current
        values in this instance.

        Parameters
        ----------
        aDictionary: dict
            A dictionary of CSS style key/val
            pairs.
        """
        self._attributes.update(aDictionary)

    def add_style(self, key, val):
        """Adds or updates a single style to
        setting to the passed value.

        Parameters
        ----------
        key: string
            The name of the CSS style attribute
            to add or update
        val: string
            The value that will be updated or
            added
        """
        self._attributes[key] = val

    def get_style(self, style_name):
        """Returns the value of the style at
        the passed in key.

        Parameters
        ----------
        style_name: string
            The name of the CSS style attribute
            for which we will retrieve the value.

        Returns
        -------
        string
            The value of the requested style.
            Will return None if the style
            is not present in this instance
        """
        if style_name in self._attributes:
            return self._attributes[style_name]
        return None

    def remove_style(self, style_name):
        """Removes the specified style from
        the intance, if present.

        Parameters
        ----------
        style_name: string
            The name of the style to remove
        """
        if style_name in self._attributes:
            del self._attributes[style_name]

    def as_dict(self):
        """Returs a dicitonary representation of the styles.

        Returns
        -------
        dict
            A dictionary representation of the
            styles and their values
        """
        return dict(self._attributes)

    def as_string(self):
        """Returns a string representation of the
        styles and their values formatted for
        display inline in HTML elements.

        Example
        -------
        Example::
            styles = StyleAttributes(top='0px', left='0px')
            result = styles.as_string()
            # 'top: 0px; left: 0px;'
            # Which can be used on an element:
            # <div style='top: 0px; left: 0px;'></div>

        Returns
        -------
        string
            A string representation of the styles
            and their values, designed for use
            inline in HTML elements.
        """
        stream = StringIO()
        for key, val in self._attributes.items():
            stream.write("{}: {}; ".format(key, val))
        return stream.getvalue().rstrip()

    def is_empty(self):
        """Returns true only if there are no styles
        set at all

        Returns
        -------
        Boolean
            True if no styles are set
        """
        return len(self._attributes.items()) == 0

    @classmethod
    def inline_string_from(cls, aStyleAttributes):
        """Convenience method that returns a string
        representation of a passed-in StyleAttributes
        instance, formatted for inline HTML elements.

        Note that this class method creates a new temporary
        StyleAttributes object, copies its internals, then
        calls the as_string method and returns the results.

        Parameters
        ----------
        aStyleAttributes: StyleAttributes
            A style attributes object for which
            this method will generate a string.

        Returns
        -------
        string
            A string representation of the passed
            in StyleAttributes object that is
            formatted for inline HTML display.
        """
        temp = cls(**aStyleAttributes.as_dict())
        return temp.as_string()

    def __str__(self):
        stream = StringIO()
        stream.write("({}\n{})".format(self.__class__.__name__, str(self._attributes)))
        return stream.getvalue()
