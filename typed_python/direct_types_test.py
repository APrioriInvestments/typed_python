import sys
import argparse
# from typed_python._types import resolveForwards
from typed_python._types import cpp_tests


def main(argv):
    parser = argparse.ArgumentParser(description='Test cpp code in module')
    parser.add_argument('-t', '--test', action='store_true')
    args = parser.parse_args()
    ret = 0

    if args.test:
        try:
            ret = cpp_tests()
        except Exception:
            return 1
    return ret


if __name__ == '__main__':
    sys.exit(main(sys.argv))
