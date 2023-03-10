from typed_python import NamedTuple


function_one = lambda row: row.a + row.b +row.c

function_three = lambda a: a ** 2

function_two = lambda a, b, c: a + b +  c


def discriminate(func: callable):
    """discriminate between the possible values of `func` by checking if the args of func are a NamedTuple or not"""
    if len(func.__code__.co_varnames) == 1:
        print('function_one')
    else:
        print('function_two')

if __name__ == '__main__':
    discriminate(function_one)
    discriminate(function_two)
