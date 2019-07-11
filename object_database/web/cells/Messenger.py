"""Messenger: module for formatting Cells socket messages

All messages that are sent over the socket to the UI
should be formatted using functions in this module
"""


def cellUpdated(cell, replaceDict={}):
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
    res =  {
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
