from typed_python import SerializationContext
from typed_python.type_function import isTypeFunctionType

import typed_python.min_repro.write_script_1
with open("/home/ubuntu/code/Platform/typed_python/typed_python/min_repro/T_1", "rb") as file:
    T_read = SerializationContext().deserialize(file.read())


print(isTypeFunctionType(T_read))
