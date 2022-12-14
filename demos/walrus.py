"""Example of walrus operator usage"""

import typed_python

from typing import List


def cubes(list_length: int, threshold: int):
    # def cubes(list_length: int, threshold: int) -> List[int]:
    cube_list = [y for x in range(list_length) if (y := x**3) < threshold]
    # cube_list = [x**3  for x in range(list_length) if  x**3 < threshold]
    return typed_python.ListOf(int)(cube_list)


if __name__ == "__main__":

    # classic list comp example
    print(cubes(list_length=10, threshold=20))

    cube_speedy = typed_python.Entrypoint(cubes)

    print(cube_speedy(list_length=10, threshold=20))
