
from typed_python import SerializationContext, Value, Entrypoint  #, Class

def f():
    class A:
        # def __enter__():
        pass

    class B:
        pass

    return Value(A), Value(B)

def clone(obj):
return sc.deserialize(sc.serialize(obj) )

if __name__ == '__main__':
    sc = SerializationContext()  # nameToObjectOverride={'A': A_v, 'B': B_v})

    c_f = Entrypoint(f)
    c_f()
    A_v, B_v = f()




    cA_v = clone(A_v)
    cB_v = clone(B_v)

    # print()
    # print()
    # print()
    # print()
    print(cA_v)
    print(cB_v)

    assert A_v == cA_v
    assert B_v == cB_v, str(cB_v)
