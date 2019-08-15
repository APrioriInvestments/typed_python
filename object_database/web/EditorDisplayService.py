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
import textwrap
import urllib
import sys

from typed_python.inspect_override import getsourcelines
from object_database.service_manager.ServiceBase import ServiceBase
from typed_python.Codebase import Codebase

import object_database.web.cells as cells
import object_database as object_database
from object_database.web.CellsTestPage import CellsTestPage
from object_database import Schema

schema = Schema("core.web.EditorDisplayService")


@schema.define
class Counter:
    value = int


class EditorDisplayService(ServiceBase):
    gbRamUsed = 0
    coresUsed = 0

    @staticmethod
    def serviceDisplay(serviceObject, instance=None, objType=None, queryArgs=None):
        page = None

        #######

        # INPUT AREA
        def onEnter(buffer, selection):
            contentsOverride.set(buffer)

        ed = cells.CodeEditor(keybindings={'Enter': onEnter},
                              noScroll=True, minLines=20,
                              onTextChange=lambda *args: None)

        inputArea = cells.SplitView(
            [
                (cells.Button(cells.Octicon("sync"), reload), 1),
                (ed, 25)
            ], split="horizontal"
        )

        contentsOverride = cells.Slot()

        def actualDisplay():
            if contentsOverride.get() is not None:
                try:
                    locals = {}
                    exec(contentsOverride.get(), sys.modules[
                        type(page).__module__].__dict__, locals)
                    return locals['cell'](page)
                except Exception:
                    return cells.Traceback(traceback.format_exc())
            else:
                return cells.Sequence(
                    [cells.Card(cells.Text("Row %s of 20" % (item + 1))) for item in
                     range(20)]
                )

        # DISPLAY AREA
        displayArea = cells.Card(
            cells.Subscribed(actualDisplay)
        )

        return cells.SplitView([
            (displayArea, 3),
            (inputArea, 2)
        ], split="vertical")

    def doWork(self, shouldStop):
        while not shouldStop.is_set():
            shouldStop.wait(100.0)


def reload():
    """Force the process to kill itself. When you refresh,
    it'll be the new code."""
    import os
    os._exit(0)
