"""test_module.py"""

from typed_python import Entrypoint

# Complied function implementing str.decref
@Entrypoint
def string_test(x: str) -> str:
    # y = x + ' hi'
    z = x  # + 'bla'
    # del
    return z


# Complied function implementing str.decref
@Entrypoint
def depends_on_string_test_differently(x: str) -> str:
    y = string_test(x)  # + ' hi'

    return y


# string_test('bla')
depends_on_string_test_differently("bla")


# Complied function implementing str.decref
@Entrypoint
def string_test_two(x: str) -> str:
    y = x + " hi2"
    return x
    # return y
