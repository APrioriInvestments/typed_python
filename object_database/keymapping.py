from typed_python import sha_hash


def data_key(obj_type, identity, field_name):
    return data_key_from_names(obj_type.__schema__.name, obj_type.__qualname__, identity, field_name)


def split_data_key(key):
    schema_name, typename, identity, field_name = key.split(":")
    return schema_name, typename, identity, field_name


def data_key_from_names(schema_name, typename, identity, field_name):
    return schema_name + ":" + typename + ":" + identity + ":" + field_name


def index_value_to_hash(value):
    if isinstance(value, int):
        return "int_" + str(value)
    if isinstance(value, str) and len(value) < 37:
        return "str_" + value
    return sha_hash(value).hexdigest


def index_key_from_names(schema_name, typename, field_name, value):
    return schema_name + ":" + typename + ": ix:" + field_name + ":" + index_value_to_hash(value)


def index_key_from_names_encoded(schema_name, typename, field_name, value_encoded):
    return schema_name + ":" + typename + ": ix:" + field_name + ":" + value_encoded


def data_reverse_index_key(schema_name, typename, identity, field_name):
    return schema_name + ":" + typename + ":" + identity + ": ixval:" + field_name


def split_data_reverse_index_key(ik):
    schema_name, typename, identity, _, field_name = ik.split(":")
    return schema_name, typename, identity, field_name


def split_index_key(key):
    assert isinstance(key, str), key
    schema, typename, _, field_name, valhash = key.rsplit(":")
    return schema + ":" + typename + ": ix:" + field_name, valhash


def split_index_key_full(key):
    assert isinstance(key, str), key
    schema, typename, _, field_name, valhash = key.rsplit(":")
    return schema, typename, field_name, valhash


def index_group(schema_name, typename, field_name):
    return schema_name + ":" + typename + ": ix:" + field_name


def index_group_and_hashval_to_index_key(index_group, hashval):
    return index_group + ":" + hashval


def index_key(obj_type, field_name, value):
    return index_key_from_names(obj_type.__schema__.name, obj_type.__qualname__, field_name, value)


def isIndexKey(key):
    return ': ix:' in key
