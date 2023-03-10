from typed_python import Entrypoint

@Entrypoint
def f(x):
    import os.path

    return os.path.basename(x)

f('test/test2/test3.py')
