from typed_python import SerializationContext
import pdb

# INPUT_PATH = '/home/wgrant/Dev/typed_python/demos/python_s_slow_u_slice_u_3/logfile.bytes'
INPUT_PATH = "/home/wgrant/Dev/typed_python/demos/logfile.bytes"
# OUTPUT_PATH = '/home/wgrant/Dev/typed_python/demos/logfile_cleaned.bytes'

# def stringify_serialized_data(input_data):
#     output_data = {}
#     output_data['externallyDefinedFunctionTypes'] = [(x,tuple(y.keys())) for x, y in input_data['externallyDefinedFunctionTypes']]

#     output_data['namedCallTargetToLLVM'] = [(x, {'name': y.name, 'external': y.external}) for x,y in input_data['namedCallTargetToLLVM']]

#     output_data['function_definitions'] = [(x) for x, y in input_data['function_definitions']]

#     return output_data


with open(INPUT_PATH, "rb") as flines:
    data = SerializationContext().deserialize(flines.read())


for key, value in data.items():
    print(key)
    # for vals in value:
    #     print("\t", vals)
    print(value)
    print()


# output = stringify_serialized_data(data)

# for key, value in output.items():
#     print(key)
#     print(value)
#     print()

# for x, y in data['function_definitions']:
#     print(y)
#     print(type(y))
#     print(dir(y))

#     for val in y:
#         for val2 in val:
#             print(val2)
#     soidfj
# # # pdb.set_trace()
