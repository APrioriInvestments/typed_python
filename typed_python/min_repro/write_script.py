from typed_python import SerializationContext

from typed_python.min_repro.python_type import PythonType


with open("/home/ubuntu/code/Platform/typed_python/typed_python/min_repro/T", "wb") as file:
    file.write(SerializationContext().serialize(PythonType()))
