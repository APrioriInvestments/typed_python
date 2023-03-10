# from typed_python import Entrypoint
# @Entrypoint
# def callSin(x):
#     return numpy.sin(x)



# assert isinstance(callSin(numpy.ones(10).cumsum()), numpy.ndarray)

# @Entrypoint
# def callF(f, x):
#     return f(x)

# assert isinstance(callF(numpy.sin, numpy.ones(10).cumsum()), numpy.ndarray)

import numpy 
import pickle
from typed_python import SerializationContext

object_to_deserialise = numpy.sin
print('pre-serialise')
print(object_to_deserialise)
print(type(object_to_deserialise))

res = SerializationContext().serialize(object_to_deserialise)

# res = SerializationContext().serialize({'test': [1,2,3]})
print(res)


unserialized = SerializationContext().deserialize(res)

print('after round trip')
print(unserialized)
print(type(unserialized))

print(unserialized is object_to_deserialise)
"""
serialize
    nameForObject -> None
    representationFor -> (tuplke)  - goes into the c code.

should be tuple (factory, args, representation).
"""