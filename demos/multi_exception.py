from typed_python import Entrypoint, SerializationContext


# # check that a serialized then deserialized exception is still an exception
# # assert SerializationContext().deserialize(SerializationContext().serialize(ValueError())) == ValueError()
# v = ValueError()

# c_v = SerializationContext().deserialize(SerializationContext().serialize(v))
# print(repr(v))
# print(repr(c_v))
# assert v == c_v
# # assert SerializationContext().



# @Entrypoint
def catcher(toRaise):
    try:
        raise toRaise()
    except (ValueError, TypeError):
        return "OK"

try:
    c = Entrypoint(catcher)
    c(ValueError)
except ValueError:
    print("caught ValueError")


print('running pyfunc')
c.extractPyFun(0)(ValueError)
print('running original')
c(ValueError)
# val = c(ValueError)


from typed_python import Runtime

r = Runtime.singleton()
native_converter = r.converter
compiler = r.llvm_compiler
converter = compiler.converter

# # import pdb; pdb.set_trace()

# assert catcher(ValueError) == "OK"
# assert catcher(TypeError) == "OK"
