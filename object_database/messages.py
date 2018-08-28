from typed_python import *

_heartbeatInterval = [5.0]
def setHeartbeatInterval(newInterval):
    _heartbeatInterval[0] = newInterval

def getHeartbeatInterval():
    return _heartbeatInterval[0]

ClientToServer = Alternative(
    "ClientToServer",
    NewTransaction = {
        "writes": ConstDict(str, OneOf(None, str)),
        "set_adds": ConstDict(str, TupleOf(str)),
        "set_removes": ConstDict(str, TupleOf(str)),
        "key_versions": TupleOf(str),
        "index_versions": TupleOf(str),
        "as_of_version": int,
        "transaction_guid": str
        },
    SendValues = {
        "keys": TupleOf(str)
        },
    SendSets = {
        "keys": TupleOf(str)
        },
    Heartbeat = {}
    )

ServerToClient = Alternative(
    "ServerToClient",
    Initialize = {'transaction_num': int, 'connIdentity': str},
    TransactionResult = {'transaction_guid': str, 'success': bool},
    KeyInfo = {'key': str, 'data': OneOf(str, None, TupleOf(str)), 'transaction_id': int},
    Disconnected = {},
    Transaction = {
        "writes": ConstDict(str, OneOf(str, None)),
        "set_adds": ConstDict(str, TupleOf(str)),
        "set_removes": ConstDict(str, TupleOf(str)),
        "transaction_id": int
        }
    )
