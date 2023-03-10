# from typed_python import Value, Entrypoint, SerializationContext, Class

# sc = SerializationContext()
# def clone(obj):
#     return sc.deserialize(sc.serialize(obj))

# from typed_python import SerializationContext

# SerializationContext()

# class A:
#     pass


# class B:
#     pass

# class C:
#     pass

# A_v = Value(A)
# B_v = Value(B)
# C_v = Value(B)




# A_cloned = clone(A_v)

# B_cloned = clone(B_v)
# C_cloned = clone(C_v)

# print()


# print(A)
# # print(clone(A))
# print(A_v)
# # print(dir(A_v))
# print(A_cloned)
# print()
# print(B)
# # print(clone(B))
# print(B_v)
# print(B_cloned)


from typed_python import SerializationContext, Value

class A:
    pass

class B:
    pass

A_v = Value(A)
B_v = Value(B)

sc = SerializationContext()

def clone(obj):
    return sc.deserialize(sc.serialize(obj) )

A_cloned = clone(A_v)
B_cloned = clone(A_v)

print(A_v)
print(A_cloned)
print()
print(B_v)
print(B_cloned)
print()