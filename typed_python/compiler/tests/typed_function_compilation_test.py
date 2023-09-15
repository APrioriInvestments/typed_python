import unittest
from typed_python import Function, Entrypoint


class TestCompileTypedFunctions(unittest.TestCase):
    @pytest.mark.group_one
    def test_compile_function_with_no_ret_type(self):
        # this function definitely throws, and so it doesn't
        # have a return type.
        @Function
        def f(x: int):
            raise Exception("an exception")

        @Entrypoint
        def callF(x: object):
            return f(x)

        with self.assertRaisesRegex(Exception, "an exception"):
            callF(10)
