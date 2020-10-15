#pragma once

/*******
Describes the different ways in which we can convert objects from one
type to another during execution.

See conversion_level.py for more.
*******/
enum class ConversionLevel {
    Signature,
    Upcast,
    UpcastContainers,
    Implicit,
    ImplicitContainers,
    New,
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
