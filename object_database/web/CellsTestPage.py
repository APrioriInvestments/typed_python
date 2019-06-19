#   Coyright 2017-2019 Nativepython Authors
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


class CellsTestPage(object):
    """Base class for each kind of cells test page.

    We find all subclasses of these in object_database.web.cells_demo and
    show them in a hierarchical view.
    """
    def category(self):
        """Return a string we'll use to group this."""
        return type(self).__module__.split(".")[-1]

    def name(self):
        return type(self).__name__

    def cell(self):
        """Return a 'cell' that this displays."""
        raise NotImplementedError("Subclasses implement")

    def text(self):
        """Return the canonical description of what this is supposed to show.
        """
        raise NotImplementedError("Subclasses implement")
