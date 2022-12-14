from typed_python import NamedTuple, Dict, ListOf, Class, Member


# nt = hash(NamedTuple(x=list)(x=[]))
# nt = hash(NamedTuple(x=list)(x=[]))

hash(ListOf(int)())


class A(Class):
    x = Member(list)


class A_vanilla:
    def __init__(self, x) -> None:
        self.x = x


a = A(x=[])
print(hash(a))

a_v = A_vanilla(x=[1])
print(hash(a_v))


class_with_mutable_attribute = hash(A(x=[]))

print(class_with_mutable_attribute)


class A(Class):
    x = Member(Dict(int, int))


hash(A(x=Dict(int, int)()))


hash(Dict(int, int)())
# print(nt)
