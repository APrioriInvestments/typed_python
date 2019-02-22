
class TypeFilterBase(type):
    def __instancecheck__(self, other):
        if not isinstance(other, self.base_type):
            return False

        try:
            if not self.filter_function(other):
                return False
        except Exception:
            return False

        return True

def TypeFilter(base_type, filter_function):
    """TypeFilter(base_type, filter_function)

    Produce a 'type object' that can be used in typed python to filter objects by
    arbitrary criteria.
    """
    class TypeFilter(metaclass=TypeFilterBase):
        pass

    TypeFilter.base_type = base_type
    TypeFilter.filter_function = filter_function

    return TypeFilter
