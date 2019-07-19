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

from object_database.web import cells as cells
from object_database.web.CellsTestPage import CellsTestPage


class MultiPageTable(CellsTestPage):
    def cell(self):
        return cells.Table(
            colFun=lambda: ["Col 1", "Col 2"],
            rowFun=lambda: list(range(100)),
            headerFun=lambda x: x,
            rendererFun=lambda w, field: "hi",
            maxRowsPerPage=50
        )

    def text(self):
        return "You should see a table with two columns, two pages of 50 rows and all fields saying 'hi'"
