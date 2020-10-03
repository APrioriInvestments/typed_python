#pragma once

/*******
Describes the different ways in which we can convert objects from one
type to another during execution.


Our goal is to support natural type-conversion operations that happen
in four different places:

    * function argument conversion (e.g. call f(x: int) with a bool)
    * construction of new objects through a type call (e.g. ListOf(int)([1, 2, 3]))
    * conversion of arguments to dictionary.getitem, array indices, etc.
    * conversion of rhs of slice assignments and class member assignments.

In each case we want to support natural uses, while throwing exceptions
in cases where conversion could really be masking an error.

Generally speaking, our rules are

    1. only the outer call of T(x) may fundamentally change the nature
        of an object, in the sense that "int" -> "float" retains
        the numeric character of an argument, but "int" -> "str" changes
        the character of the argument completely. Specifically, this means that
        a call to ListOf(str)(x) should proceed for any iterable x
        regardless of its type, but the conversion of the interior
        elements of 'x' should not allow them to become strings if
        they are integers.  Modification from an untyped tuple to
        a TupleOf, or an untyped tuple to Tuple or NamedTuple doesn't
        constitute a fundamental change. Creating a new mutable object
        (such as a ListOf, Dict, or Set) does, because the new object
        has a separate state from the old one.

        Specifically, this means you can't assign a ListOf(int) or a list to a
        class member expecting a ListOf(float), because it will create
        a new list underneath you. But you can assign a TupleOf(int) to
        a TupleOf(float) because the tuple is not mutable.

    2. we should generally be fairly willing to cast integer and float
        types to each other.  In common python use, it's standard to not
        worry about writing 0.0 when you mean '0', and generating exceptions
        when a user attempts to assign a float to an integer class member
        is overly prescriptive.

        An exception is that we shouldn't downcast float->int when we're looking
        up in a list or dictionary.

    3. when we are converting to a OneOf or performing argument matching,
        we scan over the arguments first looking for a direct match. We then
        allow do a second pass allowing upcasting. We then do a final pass allowing
        any integer/float conversion, and conversion of container types.

        We extend these rules to containers: when looking
        for a direct match, we won't allow 'tuple' to match TupleOf(T), and
        we won't allow TupleOf(int) to match TupleOf(float). In the next
        pass, when we allow upcasting, we'll allow TupleOf(UInt16) to match
        TupleOf(UInt32), or TupleOf(OneOf(UInt8, float)), but not TupleOf(Int8).

        Upcasts must be strict - more bits, more signedness.

        Note that this can make conversion semantics dependent on the order
        in which functions are defined. Unlike C++ which attempts to statically
        determine a non-ambiguous order, and which fails compilation if it cant.

*******/
enum class ConversionLevel {
    // Allow only changes at the type representation level. For instance,
    // a OneOf(T1, T2) can become a T1 if that's the actual value, or a
    // class can become a sub/superclass.  No new objects will be created,
    // and the type of any concrete instance cannot be changed. We're simply
    // changing the way we view the actual instance.
    Signature,

    // Allow all casts from Signature, and also allow integer/float upcasting,
    // and typed Tuple/TupleOf/NamedTuple upcasting.
    Upcast,

    // Allow upcasts, and also allow conversion of untyped containers to
    // typed containers as long as no new mutables are created, and upcasting
    // of typed tuples if it's allowed by the rules on the interior types.

    // This is the cast level used by dictionary key lookups, and so
    // Dict(TupleOf(int), int) allows lookup from untyped (1, 2, 3)
    UpcastContainers,

    // Just like UpcastContainers, except that all integer / float conversions
    // are allowed.

    // This is the standard conversion level for adding items to mutable containers
    // and to setting class members.
    Implicit,

    // Like implicit, but also allow container type changes. This is the level
    // invoked on interior types by New operations on Tuple/NamedTuple/Alternatives
    // and its intended to allow containers to convert, but not to allow things
    // to silently become strings.
    ImplicitContainers,

    // Invoked by an explicit call to a type constructor. Allows the construction of a
    // new instance of the type itself. Internal conversions are done as 'Implicit' calls.
    // This means a ListOf(ListOf(int))([[1]]) will fail because the inner list is
    // a mutable instance.
    New,

    // Invoked by a call to a type's 'convert' function. Allows the construction of a
    // new instance of the type itself and recursively converts child elements with
    // 'DeepNew'.
    // This means a ListOf(ListOf(int))([[1]]) will succeed, and will create new.
    // lists.
    DeepNew
};

inline int conversionLevelToInt(ConversionLevel level) {
    if (level == ConversionLevel::Signature) {
        return 0;
    }
    if (level == ConversionLevel::Upcast) {
        return 1;
    }
    if (level == ConversionLevel::UpcastContainers) {
        return 2;
    }
    if (level == ConversionLevel::Implicit) {
        return 3;
    }
    if (level == ConversionLevel::ImplicitContainers) {
        return 4;
    }
    if (level == ConversionLevel::New) {
        return 5;
    }
    if (level == ConversionLevel::DeepNew) {
        return 6;
    }

    throw std::runtime_error("Invalid ConversionLevel");
}

inline std::string conversionLevelToString(ConversionLevel level) {
    if (level == ConversionLevel::Signature) {
        return "Signature";
    }
    if (level == ConversionLevel::Upcast) {
        return "Upcast";
    }
    if (level == ConversionLevel::UpcastContainers) {
        return "UpcastContainers";
    }
    if (level == ConversionLevel::Implicit) {
        return "Implicit";
    }
    if (level == ConversionLevel::ImplicitContainers) {
        return "ImplicitContainers";
    }
    if (level == ConversionLevel::New) {
        return "New";
    }
    if (level == ConversionLevel::DeepNew) {
        return "DeepNew";
    }

    throw std::runtime_error("Invalid ConversionLevel");
}

inline ConversionLevel intToConversionLevel(int level) {
    if (level == 0) {
        return ConversionLevel::Signature;
    }
    if (level == 1) {
        return ConversionLevel::Upcast;
    }
    if (level == 2) {
        return ConversionLevel::UpcastContainers;
    }
    if (level == 3) {
        return ConversionLevel::Implicit;
    }
    if (level == 4) {
        return ConversionLevel::ImplicitContainers;
    }
    if (level == 5) {
        return ConversionLevel::New;
    }
    if (level == 6) {
        return ConversionLevel::DeepNew;
    }

    throw std::runtime_error("Invalid ConversionLevel");
}

inline bool operator<(ConversionLevel lhs, ConversionLevel rhs) {
    return conversionLevelToInt(lhs) < conversionLevelToInt(rhs);
}

inline bool operator>(ConversionLevel lhs, ConversionLevel rhs) {
    return conversionLevelToInt(lhs) > conversionLevelToInt(rhs);
}

inline bool operator<=(ConversionLevel lhs, ConversionLevel rhs) {
    return conversionLevelToInt(lhs) <= conversionLevelToInt(rhs);
}

inline bool operator>=(ConversionLevel lhs, ConversionLevel rhs) {
    return conversionLevelToInt(lhs) >= conversionLevelToInt(rhs);
}
