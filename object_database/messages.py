from typed_python import *

ClientToServer = Alternative(
    "ClientToServer",
    NewTransaction = {
        "writes": ConstDict(str, str),
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
        }
    )

ServerToClient = Alternative(
    "ServerToClient",
    Initialize = {'transaction_num': int, 'connIdentity': str},
    TransactionResult = {'transaction_guid': str, 'success': bool},
    KeyInfo = {'key': str, 'data': str, 'is_set': bool, 'transaction_id': int},
    Disconnected = {},
    Transaction = {
        "writes": ConstDict(str, str),
        "set_adds": ConstDict(str, TupleOf(str)),
        "set_removes": ConstDict(str, TupleOf(str)),
        "transaction_id": int
        }
    )
