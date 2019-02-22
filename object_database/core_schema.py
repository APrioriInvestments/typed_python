from object_database.schema import Schema

core_schema = Schema("core")


@core_schema.define
class Connection:
    pass
