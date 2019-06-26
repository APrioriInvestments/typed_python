"""KeyAction Cell

Listens key-combo press events from the UI"""
from ..cells import Cell

"""A class for adding Key combination event listeners

to the frontend UI.

Notes
-----
There are two types of keys you can register.
(1) Single alphanumeric and typed symbolic keys,
    such as:
        `i`, `2`, `K`, `?`, etc
(2) Modifier key plus (1) key values pressed
    together, like pressing `Control` and `K`, etc.

The possible modifier keys are:
    `Alt`, `Control`, `Meta`, and `Shift`

The way to register a `keyCmd` in the constructor for this
class is to provide just the single character corresponding
to the key, or, if it's a combo, to format the combo like:
    `<modifier>+<alphanum/symbol-character>`
For example:
    `Alt+1`, `Control+i`, `Meta+?` etc

Note also that `Ctrl` and `Control` will both work for
the control modifier key.

"""


class KeyAction(Cell):
    """
    Parameters
    __________
    keyCmd str: A string of a key combo like `Alt+i`, `Meta+D`
                or event single keys like `X`.
    callback func: A function that will be called with the
                event response dictionary data as the sole arg.
    wantedInfo list of str: A list of keys on the UI keydown event object
                whose values should be sent back to the Cell in the response
                message. Optional; defaults to None.
    priority int: The priority level of the event response.
    stopsPropagation bool: Whether or not this KeyAction should fire and then
                stop other KeyActions with the same keyCmd from firing

    Notes
    -----
    For an explanation of possible `keyCmd` values, see the class comment.

    For possible `wantedInfo` key names (which map to JS event data on the
    front end), see the MDN docs here:
    https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent
    """
    def __init__(self, keyCmd, callback, wantedInfo=None, priority=4, stopsPropagation=False):
        super().__init__()
        self.shouldDisplay = False
        self.keyCmd = keyCmd
        self.callback = callback
        self.wantedInfo = wantedInfo
        self.priority = priority
        self.stopsPropagation = stopsPropagation

        assert 1 <= self.priority <= 4

    def recalculate(self):
        self.exportData = {
            'keyCombo': self.keyCmd,
            'wantedEventKeys': self.wantedInfo,
            'priority': self.priority,
            'stopsPropagation': self.stopsPropagation
        }

    def onMessage(self, messageFrame):
        messageFrame['data']['keyCmd'] = self.keyCmd
        self.callback(messageFrame['data'])
