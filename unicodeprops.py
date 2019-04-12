#   Copyright 2017-2019 Nativepython Authors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

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
print("static uint16_t uprops_runlength[] = {{".format(last+1))

prev = None
count = 0
entries = 0
sum = 0
for i in range(first, last + 1):
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
    if flags != prev:
        if prev is not None:
            print("0x{:x},{}, ".format(prev, count), end="")
            sum += count
            entries += 1
            if entries % 16 == 0:
                print()
        prev = flags
        count = 1
    else:
        count += 1
        if count == 0xFFFF:
            print("0x{:x},{}, ".format(prev, count), end="")
            sum += count
            entries += 1
            if entries % 16 == 0:
                print()
            count = 0
print("0x{:x},{}\n}};\n".format(prev, count))
sum += count
entries += 1

print("// check 0x{:x} == 0x{:x}\n".format(sum, last+1))

print("static uint16_t uprops[0x{:x}];\n".format(last+1))

print("void initialize_uprops(void) {")
print("    uint16_t *cur = uprops;")
print("    for (long k = 0; k < {}; k++)".format(entries))
print("        for (long j = 0; j < uprops_runlength[k*2+1]; j++)")
print("            *cur++ = uprops_runlength[k*2];")
print("}")
