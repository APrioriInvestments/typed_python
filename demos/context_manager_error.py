
# import os 
# os.environ["TP_COMPILER_CACHE"] = "compiler_cache"

# from typed_python import Entrypoint



# class ConMan1():
#     def __enter__(self):
#         return 1

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         return True

# class ConMan2():
#     def __enter__(self):
#         return 2

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         raise NotImplementedError('ConMan2')

# def f():
#     with ConMan1() as x, ConMan2() as y:
#         result = x + y
#     return result

# # # GENERIC WALKER

# # # CLONE METHOD SERIALIZE&DESERIALISE - CHECK DEV

# # # EXPRS_TR TRUNCATING AFTER SIX CHARS - USEFUL OR NOT?



# # """
# # I believe the culprit to be a deserialisation error.
# # """
# from typed_python import SerializationContext
# def clone(obj):
#     return SerializationContext().deserialize(SerializationContext().serialize(obj))


# cf = Entrypoint(f)
# # i believe clone(obj) != obj

# cf()

# assert clone(f) == f
# assert clone(cf) == cf

# cf()

# assert clone(cf) == cf



# class ConMan1():
#     def __enter__(self):
#         return 1
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         return True
# class ConMan2():
#     def __enter__(self):
#         return 2
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         raise NotImplementedError('ConMan2')
# def f():
#     with ConMan1() as x, ConMan2() as y:
#         result = x + y
#     return result
# def check_f():
#     assert Entrypoint(f)() == f()


# c_f = 


# check_f()
# check_f()



from typed_python import Value, Entrypoint, SerializationContext

sc = SerializationContext()
def clone(obj):
    return sc.deserialize(sc.serialize(obj))

class A:
    pass

class B:
    pass


A_v = Value(A)

B_v = Value(B)



A_cloned = clone(A_v)

B_cloned = clone(B_v)

print(A_cloned)
print(B_cloned)