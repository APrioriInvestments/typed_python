from pydantic import BaseModel


class Thing(BaseModel):
    list_of_ints: list[int]


thing = Thing(list_of_ints=[1, 2, 3, 4])
thing = Thing(list_of_ints=[1, 2, 3, "bla"])
