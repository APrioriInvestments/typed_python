import os
# this has to be set prior to be TP import
os.environ["TP_COMPILER_CACHE"] = "compiler_cache"


from typed_python import Entrypoint

@Entrypoint
def g():
    import test_import_lock_2

g()
