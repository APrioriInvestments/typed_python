import sys

from typed_python import Class, SerializationContext, Held, isForwardDefined
from typed_python._types import typeWalkRecord, recursiveTypeGroupRepr


def writer():
	@Held
	class C(Class):
		def f(self):
			return C

	assert not isForwardDefined(C)

	print("group is:")
	print(typeWalkRecord(type(C.f).overloads[0].funcGlobalsInCells['C'], 'identity'))
	print(typeWalkRecord(type(C.f).overloads[0].funcGlobalsInCells['C'], 'compiler'))

	# with open("a.dat", "wb") as f:
		# f.write(SerializationContext().serialize(C))


def reader():
	with open("a.dat", "rb") as f:
		C = SerializationContext().deserialize(f.read())

	print(C)
	#assert C().f() is C

if sys.argv[1:] == ['r']:
	reader()
else:
	writer()




