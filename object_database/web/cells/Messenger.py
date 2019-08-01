"""Messenger: module for formatting Cells socket messages

All messages that are sent over the socket to the UI
should be formatted using functions in this module
"""


def oldCellUpdated(cell, replaceDict={}):
    """A lifecycle message formatter
    to be used when a Cell is created or
    updated.

    Parameters
    ----------
    cell: Cell
          An instance of a Cell whose view
          should be updated in the UI
    replaceDict: dict
          A dictionary containing child cell ids
          mapped to replacement string names

    Returns
    -------
    A JSON parsable dictionary that can
    be sent over a websocket
    """
    res = {
        'channel': '#main',
        'type': '#cellUpdated',
        'id': cell.identity,
        'replacements': replaceDict,
        'shouldDisplay': cell.shouldDisplay,
        'componentName': cell.__class__.__name__,
        'replacementKeys': [k for k in cell.children.keys()],
        'extraData': cell.exportData
    }
    if cell.postscript:
        res['postscript'] = cell.postscript

    return res


def cellUpdated(cell, replaceDict):
    structure = getStructure(
        cell.parent.identity,
        cell,
        None,
        expand=True)
    envelope = {
        "channel": "#main",
        "type": "#cellUpdated",
        "shouldDisplay": cell.shouldDisplay,
        "extraData": cell.exportData
    }
    structure.update(envelope)
    if cell.postscript:
        structure['postscript'] = cell.postscript

    return structure


def cellDiscarded(cell):
    """A lifecycle message formatter
    to be used when a Cell is discarded
    and removed from the session.

    Parameters
    ----------
    cell: Cell
          The Cell instance that is being discarded

    Returns
    -------
    A JSON parsable dictionary that can
    be sent over a websocket
    """
    return {
        'channel': '#main',
        'type': '#cellDiscarded',
        'id': cell.identity
    }


def appendPostscript(jsString):
    """A lifecycle message formatter
    to be used when we are appending a
    postscript just by itself

    Parameters
    ----------
    jsString: str
              An appropriately escaped string
              of Javascript to append to the
              current postscripts in the UI

    Returns
    -------
    A JSON parsable dictionary that can
    be sent over a websocket
    """
    return {
        'channel': '#main',
        'type': '#appendPostscript',
        'script': jsString
    }


def getStructure(parent_id, cell, name_in_parent, expand=False):
    """Responds with a dict structure representative of the
    passed in cell that will be suitable for JSON parsing.

    Notes
    -----
    There are two ways to use this function: expanded
    or not.
    Expanded will return a recursive dict structure where
    each named child is represented also as a complete dict
    along with all of its own namedChildren, and so on.
    Unexpanded will return just the given Cell's structure,
    and it's namedChildren structure will all resolve to
    Cell IDs (rather than expanded dicts)

    Parameters
    ----------
    parent_id: str|integer
        The Cell identity of the passed-in Cell's parent
    cell: Cell
        The target Cell whose structure we will map to
        a dictionary.
    name_in_parent: str
        If the passed-in Cell is a namedChild of another cell,
        we provide that name as this argument
    expand: boolean
        Whether or not to return an 'expanded' dictionary
        meaning all named children of the current cell will
        also have their own dict structures parsed out.
        See the Notes above.
        Defaults to False

    Returns
    -------
    dict: A dictionary representing the Cell structure,
          expanded or otherwise, that can be parsed
          into JSON
    """
    if expand:
        return _getExpandedStructure(parent_id, cell, name_in_parent)
    return _getFlatStructure(parent_id, cell, name_in_parent)


"""Helper Functions"""


def _getFlatStructure(parent_id, cell, name_in_parent):
    own_children = _getFlatChildren(cell)
    return {
        "id": cell.identity,
        "cellType": cell.__class__.__name__,
        "nameInParent": name_in_parent,
        "parentId": parent_id,
        "namedChildren": own_children
    }


def _getFlatChildren(cell):
    own_children = {}
    for child_name, child in cell.namedChildren.items():
        own_children[child_name] = _resolveFlatChild(child)

    return own_children


def _resolveFlatChild(cell_or_list):
    if isinstance(cell_or_list, list):
        return [_resolveFlatChild(cell) for cell in cell_or_list]
    return cell_or_list.identity


def _getExpandedStructure(parent_id, cell, name_in_parent):
    own_children = _getExpandedChildren(cell)
    return {
        "id": cell.identity,
        "cellType": cell.__class__.__name__,
        "properties": cell.exportData,
        "nameInParent": name_in_parent,
        "parentId": parent_id,
        "namedChildren": own_children
    }


def _getExpandedChildren(cell):
    own_children = {}
    for child_name, child in cell.namedChildren.items():
        own_children[child_name] = _resolveExpandedChild(cell.identity, child,
                                                         child_name)
    return own_children


def _resolveExpandedChild(parent_id, cell_or_list, name_in_parent):
    if isinstance(cell_or_list, list):
        return [_resolveExpandedChild(parent_id, cell, name_in_parent) for
                cell in cell_or_list]
    return _getExpandedStructure(parent_id, cell_or_list, name_in_parent)
