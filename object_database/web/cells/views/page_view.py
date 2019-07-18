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
"""PageView Cell

This cell is a view that fills the entire page with optional
header and footer areas and a responsive middle main content
area.
"""
from ..cells import Cell


class PageView(Cell):
    def __init__(self, main, header=None, footer=None):
        super().__init__()
        self.main = main
        self.header = header
        self.footer = footer
        self.updateChildren()

    def updateChildren(self):
        self.children['____main__'] = self.main
        self.namedChildren['main'] = self.main
        if self.header:
            self.children['____header__'] = self.header
            self.namedChildren['header'] = self.header
        if self.footer:
            self.children['____footer__'] = self.footer
            self.namedChildren['footer'] = self.footer
