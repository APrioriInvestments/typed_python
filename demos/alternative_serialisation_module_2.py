from typed_python import Alternative, SerializationContext, Entrypoint

from alternative_serialisation_module import clone

# class G:
@Entrypoint
def test_alternative_hashing_2():
    A = Alternative("A", A=dict(a=int))
    print('A from module 2', A, 'clones to', clone(A))
    return A

test_alternative_hashing_2()
