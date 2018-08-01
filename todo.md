* typed_python
    * implement checks for internal objects that hold themselves
    * function type filters are not expressive enough to match f(T, T) yet
    * members with defaults - what should this look like?
    * implement interfaces
    * implement subclassing
    * defaults for member variables
    * typing model for float32, float64, int8/16/32 etc (should wrap numpy)
    * typing model for specifying that something is a numpy array
    * forward types, so we can declare recursive unions
        * alternatives already do this, but it would be good to generalize
        * otherwise, you can't define 
            class List:
                head = int
                tail = OneOf(List | None)
          because 'List' isn't defined yet