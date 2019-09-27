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

class CodeEditorInHorizSequence(CellsTestPage):
    def cell(self):
        editorShown = cells.Slot(False)
        contentsShown = cells.Slot(False)

        contents = cells.Slot("")

        def onTextChange(buffer, selection):
            contents.set(buffer)

        def toggle(aSlot):
            print("Toggling")
            print(aSlot.get())
            aSlot.toggle()
            print(aSlot.get())

        return (
            cells.Button("Show the editor", lambda: toggle(editorShown)) +
            cells.Button("Show the editor's contents", lambda: toggle(contentsShown)) +
            cells.HorizontalSubscribedSequence(lambda:
                (["Ed"] if editorShown.get() else []) +
                (["Contents"] if contentsShown.get() else []),
                lambda which:
                    cells.CodeEditor(onTextChange=onTextChange) if which == "Ed" else
                    cells.Subscribed(lambda: contents.get()) if which == "Contents" else
                    None
            )
        )

    def text(self):
        return "You should see two buttons that let you turn the editor on and off, and also see its contents."


class CodeEditorBasicHorizSequence(CellsTestPage):

    def cell(self):
        contents = cells.Slot("No Text Entered Yet!")

        def onTextChange(content, selection):
            contents.set(content)

        return cells.HorizontalSequence([
            cells.CodeEditor(onTextChange=onTextChange),
            cells.Panel(
                cells.Subscribed(contents.get)
            )
        ])

    def text(self):
        return "Should see a CodeEditor and its content (in panel) in a HorizontalSequence that is not a flex parent"

class CodeEditorInSplitView(CellsTestPage):
    def cell(self):
        contents = cells.Slot("")

        def onTextChange(buffer, selection):
            contents.set(buffer)

        return (
            cells.ResizablePanel(
                cells.CodeEditor(onTextChange=onTextChange),
                cells.Subscribed(lambda: cells.Code(contents.get())),
            )
        )

    def text(self):
        return "You should see a code editor and a mirror of its contents."

class CodeEditorInSplitViewWithHeader(CellsTestPage):
    def cell(self):
        contents = cells.Slot("")

        def onTextChange(buffer, selection):
            contents.set(buffer)

        return (
            cells.ResizablePanel(
                cells.Text("This is an editor:") + cells.CodeEditor(onTextChange=onTextChange),
                cells.Text("This should show what's in the editor") +
                    cells.Subscribed(lambda: cells.Code(contents.get())),
            )
        )

    def text(self):
        return "You should see a code editor and a mirror of its contents."
