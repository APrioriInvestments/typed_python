class ConversionLevel:
    """Describe the various ways we can convert between two typed python Types."""

    def __lt__(self, other):
        return self.LEVEL < other.LEVEL

    def __le__(self, other):
        return self.LEVEL <= other.LEVEL

    def __gt__(self, other):
        return self.LEVEL > other.LEVEL

    def __ge__(self, other):
        return self.LEVEL >= other.LEVEL

    def __eq__(self, other):
        if not isinstance(other, ConversionLevel):
            return False

        return self.LEVEL == other.LEVEL

    def __hash__(self):
        return hash(self.LEVEL)

    def isNewOrHigher(self):
        return self.LEVEL >= New.LEVEL

    def isImplicitContainersOrHigher(self):
        return self.LEVEL >= ImplicitContainers.LEVEL

    def __str__(self):
        return type(self).__name__

    @staticmethod
    def fromIntegerLevel(intLevel):
        for level in [
            ConversionLevel.Signature,
            ConversionLevel.Upcast,
            ConversionLevel.UpcastContainers,
            ConversionLevel.Implicit,
            ConversionLevel.ImplicitContainers,
            ConversionLevel.New
        ]:
            if level.LEVEL == intLevel:
                return level

        raise Exception(f"Invalid integer level: {intLevel}")

    @staticmethod
    def functionConversionSequence():
        """Return the sequence of levels we'll try to convert function arguments using."""
        return [
            ConversionLevel.Signature,
            ConversionLevel.Upcast,
            ConversionLevel.UpcastContainers,
            ConversionLevel.Implicit,
            ConversionLevel.ImplicitContainers
        ]


class Signature(ConversionLevel):
    """Allow only changes at the type representation level. For instance,
    a OneOf(T1, T2) can become a T1 if that's the actual value, or a
    class can become a sub/superclass.  No new objects will be created,
    and the type of any concrete instance cannot be changed. We're simply
    changing the way we view the actual instance.
    """
    LEVEL = 0


class Upcast(ConversionLevel):
    """Allow all casts from Signature, and also allow integer/float upcasting,
    and typed Tuple/NamedTuple upcasting.

    Specifically, this means a Tuple(int, int) can match a Tuple(int, float).

    However, a TupleOf(int) won't match a TupleOf(float) because it requires walking
    an object of unknown size.
    """
    LEVEL = 1


class UpcastContainers(ConversionLevel):
    """Allow upcasts, and also allow conversion of untyped containers to
    typed containers as long as no new mutables are created, and upcasting
    of typed tuples if it's allowed by the rules on the interior types.

    At this level, we don't pay attention to the interior type of TupleOf
    containers, so a TupleOf(int) and an untyped tuple will both allow
    a full walk of their contents to determine whether they can convert.

    This means that an empty TupleOf(T1) and TupleOf(T2) are essentially equivalent,

    This is the cast level used by dictionary key lookups, and so
    Dict(TupleOf(int), int) allows lookup from untyped (1, 2, 3). Similarly
    for set values.
    """
    LEVEL = 2


class Implicit(ConversionLevel):
    """Just like UpcastContainers, except that all integer / float conversions
    are allowed.

    This is the standard conversion level when converting the contents of
    a container triggered by ImplicitContainers or New.
    """
    LEVEL = 3


class ImplicitContainers(ConversionLevel):
    """Just like Implicit, but also allow container conversions.

    This is the standard conversion level applied by New on interior elements
    of structured types (like NamedTuple) where we want to allow container
    conversion, but not conversions like -> str.

    It also applies to setting dictionary values, list values,
    and class members.

    Container types apply this to their interior elements as well, which
    ensures that we can use idiomatic expressions like

        ListOf(ListOf(int))([[]])

    And have them work.
    """
    LEVEL = 4


class New(ConversionLevel):
    """Invoked by an explicit call to a type constructor. Allows the construction of a
    new instance of the type itself. Internal conversions are done as 'ImplicitContainers' calls.
    """
    LEVEL = 5


ConversionLevel.Signature = Signature()
ConversionLevel.Upcast = Upcast()
ConversionLevel.UpcastContainers = UpcastContainers()
ConversionLevel.Implicit = Implicit()
ConversionLevel.ImplicitContainers = ImplicitContainers()
ConversionLevel.New = New()
