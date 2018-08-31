from typed_python import *

from object_database.schema import SchemaDefinition

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
    Heartbeat = {},
    DefineSchema = { 'name': str, 'definition': SchemaDefinition },
    Subscribe = { 
        'schema': str, 
        'typename': OneOf(None, str), 
        'fieldname_and_value': OneOf(None, Tuple(str,str)) 
        },
    Flush = {'guid': str}
    )

ServerToClient = Alternative(
    "ServerToClient",
    Initialize = {'transaction_num': int, 'connIdentity': str},
    TransactionResult = {'transaction_guid': str, 'success': bool, 'badKey': OneOf(None, str) },
    FlushResponse = {'guid': str},
    SubscriptionData = {
        'schema': str, 
        'typename': OneOf(None, str),
        'fieldname_and_value': OneOf(None, Tuple(str,str)),
        'values': ConstDict(str, OneOf(str, None)), #value
        'index_values': ConstDict(str, OneOf(str, None)),
        'identities': OneOf(None, TupleOf(str)), #the identities in play if this is an index-level subscription
        },
    SubscriptionComplete = {
        'schema': str, 
        'typename': OneOf(None, str),
        'fieldname_and_value': OneOf(None, Tuple(str,str)),
        'tid': int #marker transaction id
        },
    SubscriptionIncrease = {
        'schema': str, 
        'typename': str,
        'fieldname_and_value': Tuple(str,str),
        'identities': TupleOf(str), #the identities in play if this is an index-level subscription
        },
    Disconnected = {},
    Transaction = {
        "writes": ConstDict(str, OneOf(str, None)),
        "set_adds": ConstDict(str, TupleOf(str)),
        "set_removes": ConstDict(str, TupleOf(str)),
        "transaction_id": int
        }
    )
