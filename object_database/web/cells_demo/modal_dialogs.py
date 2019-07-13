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


class ModalDialogBox(CellsTestPage):
    def cell(self):
        slot = cells.Slot("Some Text")
        isEditing = cells.Slot(False)

        def makeModal():
            editSlot = Slot(slot.get())
            def onCancel():
                isEditing.set(False)
            def onOk():
                slot.set(editSlot.get())

            return cells.Modal(
                "A Modal",
                cells.SingleLineTextBox(editSlot),
                Cancel=onCancel,
                OK=onOk
            )

        return (
            cells.Card(
                cells.Text("Here is some text: you should be able to click a button to edit it in a modal.") +
                cells.Subscribed(lambda: slot.get()) +
                cells.Button("Edit", lambda: isEditing.set(True))
            ) +
            cells.Subscribed(makeModal()) if isEditing.get() else None
        )

    def text(self):
        return "You should see a button that lets you edit the 'Some Text' text in a modal popup."
