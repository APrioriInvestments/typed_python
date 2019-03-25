import sys

Uprops_ALPHA = 0x01
Uprops_LOWER = 0x02
Uprops_UPPER = 0x04
Uprops_DECIMAL = 0x08
Uprops_DIGIT = 0x10
Uprops_NUMERIC = 0x20
Uprops_SPACE = 0x40
Uprops_PRINTABLE = 0x80
Uprops_TITLE = 0x100

first = 0
last = 0x10ffff
print("#pragma once")
print("")
print("// Generated from:")
print("/* ", sys.version, " */")
print("")
print("#define Uprops_ALPHA 0x{:04x}".format(Uprops_ALPHA))
print("#define Uprops_LOWER 0x{:04x}".format(Uprops_LOWER))
print("#define Uprops_UPPER 0x{:04x}".format(Uprops_UPPER))
print("#define Uprops_DECIMAL 0x{:04x}".format(Uprops_DECIMAL))
print("#define Uprops_DIGIT 0x{:04x}".format(Uprops_DIGIT))
print("#define Uprops_NUMERIC 0x{:04x}".format(Uprops_NUMERIC))
print("#define Uprops_SPACE 0x{:04x}".format(Uprops_SPACE))
print("#define Uprops_PRINTABLE 0x{:04x}".format(Uprops_PRINTABLE))
print("#define Uprops_TITLE 0x{:04x}".format(Uprops_TITLE))
print("uint16_t uprops[0x{:x}] = {{".format(last+1), end="")
for i in range(first, last + 1):
    if i % 16 == 0:
        print()
    c = chr(i)
    flags = Uprops_ALPHA if c.isalpha() else 0
    flags += Uprops_LOWER if c.islower() else 0
    flags += Uprops_UPPER if c.isupper() else 0
    flags += Uprops_DECIMAL if c.isdecimal() else 0
    flags += Uprops_DIGIT if c.isdigit() else 0
    flags += Uprops_NUMERIC if c.isnumeric() else 0
    flags += Uprops_SPACE if c.isspace() else 0
    flags += Uprops_PRINTABLE if c.isprintable() else 0
    flags += Uprops_TITLE if c.istitle() and not c.isupper() else 0
    print("0x{:x}{}".format(flags, "," if i != last else ""), end="")
print("\n};")
