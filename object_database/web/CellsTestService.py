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

schema = Schema("core.web.CellsTestService")


@schema.define
class Counter:
    value = int


_pagesCache = {}


def getPages():
    if _pagesCache:
        return _pagesCache

    # force us to actually import everything in object database
    odbCodebase = Codebase.FromRootlevelModule(object_database)
    # these are all the cell_demo cells
    for name, value in odbCodebase.allModuleLevelValues():
        if isinstance(value, type) and issubclass(value, CellsTestPage) and \
           value is not CellsTestPage:
            try:
                instance = value()
                _pagesCache.setdefault(instance.category(), {})[
                    value.__name__] = instance
            except Exception:
                traceback.print_exc()

    return _pagesCache


class CellsTestService(ServiceBase):
    gbRamUsed = 0
    coresUsed = 0

    @staticmethod
    def serviceDisplay(serviceObject, instance=None, objType=None,
                       queryArgs=None):
        queryArgs = queryArgs or {}

        # general sizing
        resultAreaHeight = '45vh'
        inputAreaHeight = '45vh'
        descriptionAreaHeight = '5vh'
        if 'category' in queryArgs and 'name' in queryArgs:
            page = getPages()[queryArgs['category']][queryArgs['name']]
            contentsOverride = cells.Slot()

            pageSource = textwrap.dedent(
                "".join(getsourcelines(page.cell.__func__)[0]))

            def actualDisplay():
                if contentsOverride.get() is not None:
                    try:
                        locals = {}
                        exec(contentsOverride.get(), sys.modules[
                            type(page).__module__].__dict__, locals)
                        return locals['cell'](page)
                    except Exception:
                        return cells.Traceback(traceback.format_exc())

                return page.cell()

            def onEnter(buffer, selection):
                contentsOverride.set(buffer)

            ed = cells.CodeEditor(keybindings={'Enter': onEnter},
                                  noScroll=True, minLines=20,
                                  onTextChange=lambda *args: None).height(
                                      "100%")
            ed.setContents(pageSource)

            description = page.text()

        else:
            page = None
            description = ""
            ed = cells.Card("pick something").height("100%")

            def actualDisplay():
                return cells.Text("nothing to display")

        resultArea = cells.SplitView([
            (cells.Card(
                cells.Subscribed(actualDisplay),
                padding=4).height(resultAreaHeight), 2),
            (cells.Card(cells.Text(description),
                        padding=2).height(descriptionAreaHeight),
             1)], split="horizontal")

        inputArea = cells.Card(
            cells.SplitView(
                [(selectionPanel(page, inputAreaHeight), 1), (ed, 8)]
            ), padding=2
        ).height(inputAreaHeight)

        return cells.SplitView([
            (resultArea, 1),
            (inputArea, 1)
        ], split="horizontal")

    def doWork(self, shouldStop):
        while not shouldStop.is_set():
            shouldStop.wait(100.0)


def reload():
    """Force the process to kill itself. When you refresh,
    it'll be the new code."""
    import os
    os._exit(0)


def selectionPanel(page, height):
    availableCells = cells.Sequence(
        [cells.Clickable(
            x.category() + "." + x.name(),
            "CellsTestService?" + urllib.parse.urlencode(
                dict(category=x.category(), name=x.name())),
            makeBold=x is page)
            for perCategory in getPages().values()
            for x in perCategory.values()],
        split="horizontal"
    )
    return cells.Card(
        cells.SplitView([
            (cells.Button(cells.Octicon("sync"), reload), 1),
            (availableCells, 6)
        ], split="horizontal")
    ).background_color(
        "#FAFAFA").height("100%")
