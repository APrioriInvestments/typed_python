from typed_python import NamedTuple

Nt1 = NamedTuple(x=int)
Nt2 = NamedTuple(x=int, y=str)

i1 = Nt1()
i11 = Nt1(i1)  # this works, as it should

i1 == i11
i1 is i11
i2 = Nt2(i1)  # this works, as it should


class WrappedNt1(Nt1):
    pass


class WrappedNt2(Nt2):
    pass


wi10 = WrappedNt1(i1)  # this works, as it should
wi11 = WrappedNt1(wi10)  # this works, as it should

wi10 == wi11
wi10 is wi11

wi20 = WrappedNt2(i1)  # this works, as it should
wi21 = WrappedNt2(wi20)  # this works, as it should

wi20 == wi21
wi20 is wi21

wi22 = WrappedNt2(wi10)  # this should work but it doesn't
