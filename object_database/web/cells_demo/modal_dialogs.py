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


class BasicModal(CellsTestPage):
    def cell(self):
        isShowing = cells.Slot(False)

        def buttonCallback():
            isShowing.set(not isShowing.get())

        button = cells.Button("Toggle Modal", buttonCallback)
        modal = cells.Modal(
            "Basic Modal",
            cells.Text("Modal Body"),
            isShowing,
            Close=buttonCallback
        )
        return cells.Card(button + modal)

    def text(self):
        return "When you click Toggle, you should see a basic modal appear and it should be closable"


class ModalWithUpdateField(CellsTestPage):
    def cell(self):
        isShowing = cells.Slot(False)
        sharedContent = cells.Slot("Some Text")

        def buttonCallback():
            isShowing.set(not isShowing.get())

        button = cells.Button("Open Modal", buttonCallback)
        textDisplay = cells.Subscribed(lambda: cells.Text(sharedContent.get()))
        modal = cells.Modal(
            "Text Updater",
            cells.SingleLineTextBox(sharedContent),
            show=isShowing,
            Close=buttonCallback
        )
        return cells.Card(
            button + textDisplay + modal)

    def text(self):
        return "You should see a button that lets you edit the 'Some Text' text in a modal popup."
