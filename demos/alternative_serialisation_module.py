from typed_python import Alternative, SerializationContext, Entrypoint

def clone(x):
    return SerializationContext().deserialize(SerializationContext().serialize(x))

# class F:
@Entrypoint
def test_alternative_hashing():
    A = Alternative("A", A=dict(a=int))
    print('A from module 1', A, 'clones to', clone(A))
    return A

test_alternative_hashing()
