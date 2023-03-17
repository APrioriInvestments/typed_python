from typed_python.compiler.merge_type_wrappers import mergeTypes
from typed_python import OneOf, Value, Class


class Base(Class):
    pass


class Child(Base):
    pass


def test_merge_types():
    assert mergeTypes([float, int]) == OneOf(float, int)
    assert mergeTypes([float, Value(1)]) == OneOf(1, float)
    assert mergeTypes([float, Value(1.5)]) == float
    assert mergeTypes([OneOf(1, float), int]) == OneOf(float, int)
    assert mergeTypes([OneOf(float, Child), OneOf(int, Base)]) == OneOf(Base, float, int)
    assert mergeTypes([object, str]) == object
    assert mergeTypes([OneOf(str, None), object]) == object
