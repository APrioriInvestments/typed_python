from typed_python import SerializationContext
import pdb

INPUT_PATH = "/home/wgrant/Dev/typed_python/demos/python_s_slow_u_slice_u_3/logfile_sanitised.bytes"


with open(INPUT_PATH, "rb") as flines:
    data = SerializationContext().deserialize(flines.read())


import json

with open("failure_dict.json", "w") as outflines:
    json.dump(data, outflines, indent=2)
