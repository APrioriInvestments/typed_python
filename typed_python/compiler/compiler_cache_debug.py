import os
from typed_python import ListOf, Dict, SerializationContext


def displayHash(fullpath):
    lines = []
    disp = lines.append

    disp(f"IN DIRECTORY {fullpath}:")

    def display(filename, serializeType=object, prettyPrinter=None):
        disp(f" * {filename}:")

        with open(os.path.join(fullpath, filename), "rb") as f:
            data = SerializationContext().deserialize(f.read(), serializeType)
            if prettyPrinter is None:
                for k, v in data.items():
                    disp(f"     {k} --> {v}")
            else:
                prettyPrinter(data, disp)

        disp("")

    display("type_manifest.dat")
    display("native_type_manifest.dat")
    display("name_manifest.dat", serializeType=Dict(str, str))
    display("globals_manifest.dat")
    display(
        "submodules.dat",
        serializeType=ListOf(str),
        prettyPrinter=lambda data, disp: disp("\n".join((f"     {val}" for val in data)))
    )

    print("\n".join(lines))
