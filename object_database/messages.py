from typed_python import *
from object_database.schema import SchemaDefinition
from object_database.algebraic_to_json import Encoder

_encoder = Encoder()

_heartbeatInterval = [5.0]
def setHeartbeatInterval(newInterval):
    _heartbeatInterval[0] = newInterval

def getHeartbeatInterval():
    return _heartbeatInterval[0]

USE_SLOWER_BUT_STRONGLY_TYPED_MESSAGES = False

if USE_SLOWER_BUT_STRONGLY_TYPED_MESSAGES:
    ClientToServer = Alternative(
        "ClientToServer",
        TransactionData = {
            "writes": ConstDict(str, OneOf(None, str)),
            "set_adds": ConstDict(str, TupleOf(str)),
            "set_removes": ConstDict(str, TupleOf(str)),
            "key_versions": TupleOf(str),
            "index_versions": TupleOf(str),
            "transaction_guid": str
            },
        CompleteTransaction = {
            "as_of_version": int,
            "transaction_guid": str
            },
        Heartbeat = {},
        DefineSchema = { 'name': str, 'definition': SchemaDefinition },
        LoadLazyObject = { 'schema': str, 'typename': str, 'identity': str },
        Subscribe = { 
            'schema': str, 
            'typename': OneOf(None, str), 
            'fieldname_and_value': OneOf(None, Tuple(str,str)),
            'isLazy': bool #load values when we first request them, instead of blocking on all the data.
            },
        Flush = {'guid': str}
        )

    ServerToClient = Alternative(
        "ServerToClient",
        Initialize = {'transaction_num': int, 'connIdentity': str, 'identity_root': int},
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
        LazyTransactionPriors = { 'writes': ConstDict(str, OneOf(str, None)) },
        LazyLoadResponse = { 'identity': str, 'values': ConstDict(str, OneOf(str, None)) },
        LazySubscriptionData = {
            'schema': str, 
            'typename': OneOf(None, str),
            'fieldname_and_value': OneOf(None, Tuple(str,str)),
            'identities': TupleOf(str),
            'index_values': ConstDict(str, OneOf(str, None))
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
            'identities': TupleOf(str)
            },
        Disconnected = {},
        Transaction = {
            "writes": ConstDict(str, OneOf(str, None)),
            "set_adds": ConstDict(str, TupleOf(str)),
            "set_removes": ConstDict(str, TupleOf(str)),
            "transaction_id": int
            }
        )
else:
    class ClientToServer:
        def __init__(self, **kwargs):
            kwargs = dict(kwargs)

            if 'definition' in kwargs:
                if not isinstance(kwargs['definition'], SchemaDefinition):
                    kwargs['definition'] = _encoder.from_json(kwargs['definition'], SchemaDefinition)
                    
            self.__dict__ = kwargs

        @staticmethod
        def from_json(msg):
            return ClientToServer(**msg)

        def to_json(self):
            d = dict(self.__dict__)
            if 'definition' in d:                        
                d['definition'] = _encoder.to_json(SchemaDefinition, d['definition'])

            return d

        @staticmethod
        def TransactionData(writes, set_adds, set_removes, key_versions, index_versions, transaction_guid):
            return ClientToServer(
                type='TransactionData', 
                writes=writes, 
                set_adds=set_adds, 
                set_removes=set_removes, 
                key_versions=key_versions, 
                index_versions=index_versions,
                transaction_guid=transaction_guid
                )

        @staticmethod
        def CompleteTransaction(as_of_version, transaction_guid):
            return ClientToServer(
                type='CompleteTransaction', 
                as_of_version=as_of_version, 
                transaction_guid=transaction_guid
                )

        @staticmethod
        def Heartbeat():
            return ClientToServer(type="Heartbeat")

        @staticmethod
        def LoadLazyObject(schema, typename, identity):
            return ClientToServer(type="LoadLazyObject", schema=schema, typename=typename, identity=identity)

        @staticmethod
        def DefineSchema(name, definition):
            return ClientToServer(type="DefineSchema", name=name, definition=definition)

        @staticmethod
        def Subscribe(schema, typename, fieldname_and_value, isLazy):
            return ClientToServer(type='Subscribe', schema=schema, typename=typename, fieldname_and_value=fieldname_and_value, isLazy=isLazy)

        @staticmethod
        def Flush(guid):
            return ClientToServer(type='Flush', guid=guid)

        @property
        def matches(self):
            class M:
                def __getattr__(_, x):
                    return x == self.type
            return M()


    class ServerToClient:
        def __init__(self, **kwargs):
            self.__dict__ = dict(kwargs)
            if isinstance(self.__dict__.get('fieldname_and_value'), list):
                self.__dict__['fieldname_and_value'] = tuple(self.__dict__['fieldname_and_value'])
            if isinstance(self.__dict__.get('identities'), list):
                self.__dict__['identities'] = tuple(self.__dict__['identities'])

        def __str__(self):
            return "ServerToClient." + self.type

        @staticmethod
        def from_json(msg):
            return ServerToClient(**msg)

        def to_json(self):
            return dict(self.__dict__)

        @staticmethod
        def Initialize(transaction_num, connIdentity, identity_root):
            return ServerToClient(type="Initialize", transaction_num=transaction_num,connIdentity=connIdentity, identity_root=identity_root)

        @staticmethod
        def TransactionResult(transaction_guid, success, badKey):
            return ServerToClient(type='TransactionResult', 
                transaction_guid=transaction_guid,
                success=success,
                badKey=badKey
                )
        
        @staticmethod
        def FlushResponse(guid):
            return ServerToClient(type='FlushResponse', guid=guid)
        
        @staticmethod
        def SubscriptionData(schema, typename, fieldname_and_value, values, index_values, identities):
            return ServerToClient(type='SubscriptionData',
                schema=schema,
                typename=typename,
                fieldname_and_value=fieldname_and_value,
                values=values,
                index_values=index_values,
                identities=tuple(identities) if identities is not None else None
                )
            
        @staticmethod
        def LazyTransactionPriors(writes):
            return ServerToClient(type='LazyTransactionPriors', writes=writes)

        @staticmethod
        def LazyLoadResponse(identity, values):
            return ServerToClient(type='LazyLoadResponse', 
                identity=identity,
                values=values
                )

        @staticmethod
        def LazySubscriptionData(schema, typename, fieldname_and_value, identities, index_values):
            return ServerToClient(type='LazySubscriptionData',
                schema=schema,
                typename=typename,
                fieldname_and_value=fieldname_and_value,
                identities=tuple(identities) if identities is not None else None,
                index_values=index_values
                )
        
        @staticmethod
        def SubscriptionComplete(schema, typename, fieldname_and_value, tid):
            return ServerToClient(type='SubscriptionComplete',
                schema=schema,
                typename=typename,
                fieldname_and_value=fieldname_and_value,
                tid=tid
                )
        
        @staticmethod
        def SubscriptionIncrease(schema, typename, fieldname_and_value, identities):
            return ServerToClient(
                type='SubscriptionIncrease',
                schema=schema,
                typename=typename,
                fieldname_and_value=fieldname_and_value,
                identities=tuple(identities) if identities is not None else None,
                )
        
        @staticmethod
        def Disconnected():
            return ServerToClient(type="Disconnected")
        
        @staticmethod
        def Transaction(writes, set_adds, set_removes, transaction_id):
            return ServerToClient(
                type='Transaction', 
                writes=writes,
                set_adds={k:tuple(v) for k,v in set_adds.items()},
                set_removes={k:tuple(v) for k,v in set_removes.items()},
                transaction_id=transaction_id
                )

        @property
        def matches(self):
            class M:
                def __getattr__(_, x):
                    return x == self.type
            return M()
