from typed_python import SerializationContext, TypeFunction


@TypeFunction
def PythonType(T):
    class Res:
        pass

    return Res


class T:
    pass


with open("/home/ubuntu/code/Platform/typed_python/typed_python/min_repro/T_1", "wb") as file:
    file.write(SerializationContext().serialize(PythonType(T)))
