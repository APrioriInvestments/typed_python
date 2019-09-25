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

import traceback

from object_database.web import cells as cells
from object_database.web.CellsTestPage import CellsTestPage


class CodeEditorDemo(CellsTestPage):
    def cell(self):
        isShown = cells.Slot(False)

        return cells.Button("Toggle the editor", lambda: isShown.set(not isShown.get())) + cells.Subscribed(
            lambda: cells.CodeEditor() if isShown.get() else None
        )

    def text(self):
        return "You should see a button that lets you see a text editor."
