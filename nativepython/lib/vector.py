import nativepython
import nativepython.type_model as type_model
import nativepython.util as util

addr = util.addr

@util.typefun
def Vector(T):
    assert isinstance(T, type_model.Type)

    class Iterator(type_model.cls):
        def __types__(cls):
            cls.types._vec_ptr = Vector.pointer
            cls.types._i = int

        def __init__(self, vec_ptr):
            self._vec_ptr.__init__(vec_ptr)

        def has_next(self):
            return self._i < len(self._vec_ptr[0])

        def next(self):
            old_i = self._i
            self._i += 1
            return util.ref((self._vec_ptr[0])[old_i])

    class Vector(type_model.cls):
        def __types__(cls):
            cls.types._ptr = T.pointer
            cls.types._reserved = int
            cls.types._size = int

        def __destructor__(self):
            self._teardown()

        def __assign__(self, other):
            self._teardown()
            self._become(other)

        def __copy_constructor__(self, other):
            self._become(other)

        def _become(self, other):
            if other._ptr:
                self._ptr = T.pointer(util.malloc(T.sizeof * other._reserved))
                self._reserved = other._reserved
                self._size = other._size

                for i in range(self._size):
                    util.in_place_new(self._ptr + i, other._ptr[i])
            else:
                self._ptr = T.pointer(0)
                self._reserved = 0
                self._size = 0
            
        def _teardown(self):
            if self._ptr:
                for i in range(self._size):
                    util.in_place_destroy(self._ptr + i)

                util.free(self._ptr)

        def __len__(self):
            return self._size
            
        def __getitem__(self, index):
            return util.ref(self._ptr[index])

        def __setitem__(self, index, value):
            self._ptr[index] = value

        def append(self, value):
            if self._reserved <= self._size:
                self.reserve(self._size * 2 + 1)

            util.in_place_new(self._ptr + self._size, value)

            self._size += 1

        def reserve(self, count):
            if count < self._size:
                count = self._size

            if count == self._reserved:
                return

            new_ptr = T.pointer(util.malloc(T.sizeof * count))
                
            for i in range(self._size):
                util.in_place_new(new_ptr + i, self._ptr[i])
                util.in_place_destroy(self._ptr + i)

            if self._ptr:
                util.free(self._ptr)

            self._ptr = new_ptr
            self._reserved = count

        def resize(self, count):
            count = int(count)
            
            if count > self._reserved:
                self.reserve(count)
            
            if count < 0:
                count = 0

            while count < self._size:
                self._size -= 1
                util.in_place_destroy(self._ptr + self._size)

            while count > self._size:
                util.in_place_new(self._ptr + self._size, T())
                self._size += 1

        def __iter__(self):
            return Iterator(util.addr(self))
    
    return Vector




