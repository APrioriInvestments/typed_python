#!/usr/bin/env python3
import sys
from typed_python import generate_types

if len(sys.argv) < 2:
    print(f"usage: {sys.argv[0]} destination")
else:
    generate_types.main(sys.argv)
