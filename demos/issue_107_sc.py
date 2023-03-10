# test whether we can serialize numpy ufuncs


import numpy
from typed_python import SerializationContext
sc = SerializationContext()

sc.serialize(numpy.array) # throws
sc.serialize(numpy.array([1,2,3])) # OK
sc.serialize(numpy.array([1,2,3]).max) # throws
sc.serialize(numpy.max) # throws

f = sc.deserialize(sc.serialize(numpy.max))

import pdb; pdb.set_trace()
print(f)
