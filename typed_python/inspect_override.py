#   Copyright 2017 Braxton Mckee
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

import imp
import inspect
import linecache
import os
import re

from inspect import *


class InspectError(Exception):
    pass


pathExistsOnDiskCache_ = {}
linesCache_ = {}


def getlines(path):
    """return a list of lines for a given path.

    This override is also careful to map "<stdin>" to the full contents of
    the readline buffer.
    """
    if path in linesCache_:
        return linesCache_[path]

    if path not in pathExistsOnDiskCache_:
        pathExistsOnDiskCache_[path] = os.path.exists(path)

    if pathExistsOnDiskCache_[path]:
        with open(path, "r") as f:
            linesCache_[path] = f.readlines()
        return linesCache_[path]
    elif path in linecache.cache:
        return linecache.cache[path][2]
    else:
        return None


def getfile(pyObject):
    try:
        return inspect.getfile(pyObject)
    except TypeError:
        if isclass(pyObject):
            return _try_getfile_class(pyObject)
        raise


def _try_getfile_class(pyObject):
    members = getmembers(
        pyObject,
        lambda _: ismethod(_) or isfunction(_)
    )

    if len(members) == 0:
        raise InspectError(
            "can't get source code for class %s" % pyObject
        )

    # members is a list of tuples: (name, func)
    elt0 = members[0][1]

    if isfunction(elt0):
        func = elt0
    else:
        # must be a method
        func = elt0

    return inspect.getfile(func)


def getsourcefile(pyObject):
    """Return the filename that can be used to locate an object's source.
    Return None if no way can be identified to get the source.
    """
    filename = getfile(pyObject)

    if filename == "<stdin>":
        return filename

    if filename[-4:].lower() in ('.pyc', '.pyo'):
        filename = filename[:-4] + '.py'
    for suffix, mode, _ in imp.get_suffixes():
        if 'b' in mode and filename[-len(suffix):].lower() == suffix:
            # Looks like a binary file.  We want to only return a text file.
            return None

    if filename not in pathExistsOnDiskCache_:
        pathExistsOnDiskCache_[filename] = os.path.exists(filename)

    if pathExistsOnDiskCache_[filename]:
        return filename

    # only return a non-existent filename if the module has a PEP 302 loader
    if hasattr(getmodule(pyObject, filename), '__loader__'):
        return filename
    # or it is in the linecache
    if filename in linecache.cache:
        return filename


def findsource(pyObject):
    """Return the entire source file and starting line number for an object.

    The argument may be a module, class, method, function, traceback, frame,
    or code object.  The source code is returned as a list of all the lines
    in the file and the line number indexes a line in that list.  An IOError
    is raised if the source code cannot be retrieved."""

    pyFile = getfile(pyObject)
    sourcefile = getsourcefile(pyObject)

    if not sourcefile and pyFile[:1] + pyFile[-1:] != '<>':
        raise IOError('source code not available')

    pyFile = sourcefile if sourcefile else pyFile

    lines = getlines(pyFile)

    if not lines:
        raise IOError('could not get source code')

    if ismodule(pyObject):
        return lines, 0

    if isclass(pyObject):
        name = pyObject.__name__
        pat = re.compile(r'^(\s*)class\s*' + name + r'\b')
        # find all matching class definitions and if more than one
        # is found, raise a InspectError
        candidates = []
        for i in range(len(lines)):
            match = pat.match(lines[i])
            if match:
                # add to candidate list
                candidates.append(i)
        if not candidates:
            raise IOError('could not find class definition for %s' % pyObject)
        elif len(candidates) > 1:
            raise InspectError('could not find class unequivocally: class '
                               + name)
        else:
            # classes may have decorators and the decorator is considered part
            # of the class definition
            lnum = candidates[0]
            pat = re.compile(r'^(\s*)@\w+')
            while lnum > 0 and pat.match(lines[lnum-1]):
                lnum -= 1
            return lines, lnum

    if ismethod(pyObject):
        pyObject = pyObject
    if isfunction(pyObject):
        try:
            pyObject = pyObject.__code__
        except Exception:
            print(pyObject)
            print(dir(pyObject))
            raise
    if istraceback(pyObject):
        pyObject = pyObject.tb_frame
    if isframe(pyObject):
        pyObject = pyObject.f_code
    if iscode(pyObject):
        if not hasattr(pyObject, 'co_firstlineno'):
            raise IOError('could not find function definition')
        lnum = pyObject.co_firstlineno - 1
        pat = re.compile(r'^(\s*def\s)|(.*(?<!\w)lambda(:|\s))|^(\s*@)')
        while lnum > 0:
            if pat.match(lines[lnum]):
                break
            lnum = lnum - 1
        return lines, lnum
    raise IOError('could not find code object')


def getsourcelines(pyObject):
    """Return a list of source lines and starting line number for an object.

    The argument may be a module, class, method, function, traceback, frame,
    or code object.  The source code is returned as a list of the lines
    corresponding to the object and the line number indicates where in the
    original source file the first line of code was found.  An IOError is
    raised if the source code cannot be retrieved."""
    lines, lnum = findsource(pyObject)

    if ismodule(pyObject):
        return lines, 0
    else:
        return getblock(lines[lnum:]), lnum + 1


def getsource(pyObject):
    """Return the text of the source code for an object.

    The argument may be a module, class, method, function, traceback, frame,
    or code object.  The source code is returned as a single string.  An
    IOError is raised if the source code cannot be retrieved."""
    lines, _ = getsourcelines(pyObject)
    return ''.join(lines)
