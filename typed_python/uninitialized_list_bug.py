from typed_python import ListOf, Entrypoint


N = 50
A = ListOf(bool)()
A.reserve(N)
A.setSizeUnsafe(N)


@Entrypoint
def countTrueInUninitializedList(A):
    total = 0
    for i in range(len(A)):
        if A[i]:
            total += 1

    return total, sum(A)


print(countTrueInUninitializedList(A))


@Entrypoint
def sameButWithAddedStatements(A):
    indices = ListOf(int)()
    total = 0
    for i in range(len(A)):
        if A[i]:
            indices.append(i)
            total += 1

    return total, sum(A)


print(sameButWithAddedStatements(A))
