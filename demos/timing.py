from itertools import zip_longest
from typed_python import Entrypoint, ListOf
import cProfile
import timeit
import dis


def slow_sum(someList, zero):
    for x in someList:
        zero += x
    return zero


speedy_sum = Entrypoint(slow_sum)


list_one = list(range(10000))
list_two = ListOf(int)(range(10000))
list_two_float = ListOf(float)(range(10000))


def main():
    list_one = list(range(10000))
    slow_sum(list_one, zero=0)
    speedy_sum(list_one, zero=0)


if __name__ == "__main__":
    # cProfile.run('main()')

    # print("normal python list")
    # print("vanilla sum")
    # print(timeit.timeit("slow_sum(list_one, zero=0)", setup="from __main__ import slow_sum, list_one", number=int(1e3)))
    # print("compiled attempt")
    # print(timeit.timeit("speedy_sum(list_one, zero=0)", setup="from __main__ import speedy_sum, list_one",  number=int(1e3)))

    # print("typed python list")
    # print("vanilla sum")
    # print(timeit.timeit("slow_sum(list_two, zero=0)", setup="from __main__ import slow_sum, list_two", number=int(1e3)))
    # print("compiled attempt")
    # print(timeit.timeit("speedy_sum(list_two, zero=0)", setup="from __main__ import speedy_sum, list_two",  number=int(1e3)))
    # print("compiled attempt float")
    # print(timeit.timeit("speedy_sum(list_two_float, zero=0.0)", setup="from __main__ import speedy_sum, list_two_float",  number=int(1e3)))
    for i in range(10):
        x = speedy_sum(list_two, zero=0)

    # print("SLOW SYM:")
    # print(dis.dis(slow_sum))
    # print("SPEEDY SUM:")
    # print(dis.dis(speedy_sum))
    # fails
    # print(timeit.timeit("speedy_sum(list_two_float, zero=0)", setup="from __main__ import speedy_sum, list_two_float",  number=int(1e3)))

    # l = list()
    # l2 = ListOf(int)([])

    # print(dir(1))
    # print(dir(l2))

    # print("\n".join(list("\t\t\t".join([x,y]) for x,y in zip_longest(dir(l), dir(l2),fillvalue="- "))))
