from typed_python import SerializationContext
from typed_python.type_function import isTypeFunctionType

from typed_python.min_repro.python_type import PythonType
PythonType()

with open("/home/ubuntu/code/Platform/typed_python/typed_python/min_repro/T", "rb") as file:
    T_read = SerializationContext().deserialize(file.read())


print(isTypeFunctionType(T_read))
print(isTypeFunctionType(PythonType()))
