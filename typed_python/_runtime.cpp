#include <stdint.h>
#include <stdarg.h>
#include <cmath>
#include <Python.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "AllTypes.hpp"
#include "StringType.hpp"
#include "BytesType.hpp"
#include "hash_table_layout.hpp"
#include "PyInstance.hpp"

#include <pythread.h>

PyObject* getRuntimeSingleton() {
    assertHoldingTheGil();

    static PyObject* pyRuntimeModule = runtimeModule();

    if (!pyRuntimeModule) {
        if (!PyErr_Occurred()) {
            PyErr_Format(PyExc_RuntimeError, "Internal error: couldn't find typed_python.compiler.runtime");
        }
        throw PythonExceptionSet();
    }

    static PyObject* runtimeClass = PyObject_GetAttrString(pyRuntimeModule, "Runtime");

    if (!runtimeClass) {
        if (!PyErr_Occurred()) {
            PyErr_Format(PyExc_RuntimeError, "Internal error: couldn't find typed_python.compiler.runtime.Runtime");
        }
        throw PythonExceptionSet();
    }

    static PyObject* singleton = PyObject_CallMethod(runtimeClass, "singleton", "");

    if (!singleton) {
        if (!PyErr_Occurred()) {
            PyErr_Format(
                PyExc_RuntimeError,
                "Internal error: couldn't call typed_python.compiler.runtime.Runtime.singleton"
            );
        }
        throw PythonExceptionSet();
    }

    return singleton;
}

// Note: extern C identifiers are distinguished only up to 32 characters
// nativepython_runtime_12345678901
extern "C" {
    void np_compileClassDispatch(ClassDispatchTable* classDispatchTable, int slot) {
        PyEnsureGilAcquired getTheGil;

        // check if there is an error already in place
        PyObject *existingErrorTypePtr, *existingErrorValuePtr, *existingErrorTracebackPtr;
        PyErr_Fetch(&existingErrorTypePtr, &existingErrorValuePtr, &existingErrorTracebackPtr);
        PyObjectHolder existingErrorType, existingErrorValue, existingErrorTraceback;

        existingErrorType.steal(existingErrorTypePtr);
        existingErrorValue.steal(existingErrorValuePtr);
        existingErrorTraceback.steal(existingErrorTracebackPtr);

        PyObject* singleton = getRuntimeSingleton();

        PyObjectStealer res(
            PyObject_CallMethod(
                singleton,
                "compileClassDispatch",
                "OOl",
                PyInstance::typeObj(classDispatchTable->getInterfaceClass()->getClassType()),
                PyInstance::typeObj(classDispatchTable->getImplementingClass()->getClassType()),
                (uint64_t)slot
            )
        );

        if (!res) {
            if (existingErrorType) {
                // we were unwinding an exception already. Now we'll replace it with the new one.
                return;
            }

            throw PythonExceptionSet();
        }

        if (!classDispatchTable->get(slot)) {
            // here, we have to throw so we don't try to call the empty pointer.
            PyErr_Format(
                PyExc_TypeError,
                "Failed to populate the classDispatchTable"
            );
            throw PythonExceptionSet();
        }

        if (existingErrorType) {
            //reset the error code
            existingErrorType.extract();
            existingErrorValue.extract();
            existingErrorTraceback.extract();

            PyErr_Restore(existingErrorTypePtr, existingErrorValuePtr, existingErrorTracebackPtr);
        }
    }

    Type* np_classTypeAsPointer(VTable* vtable) {
        return vtable->mType->getClassType();
    }

    bool np_typePtrIsSubclass(Type* derived, Type* super) {
        if (Type::typesEquivalent(derived, super)) {
            return true;
        }

        return derived->isSubclassOf(super);
    }

    PythonObjectOfType::layout_type* np_convertTypePtrToTypeObj(Type* p) {
        PyEnsureGilAcquired getTheGil;

        return  PythonObjectOfType::createLayout((PyObject*)PyInstance::typeObj(p));
    }

    // downcast the class instance pointer (classPtr) to targetType and write the downcast version
    // into 'outClassPtr' with an incref if possible, otherwise return false.
    bool np_classObjectDowncast(void* classPtr, void** outClassPtr, Class* targetType) {
        Class::layout* layout = Class::instanceToLayout((instance_ptr)&classPtr);
        Class* actualClass = Class::actualTypeForLayout((instance_ptr)&classPtr);

        int mroIndex = actualClass->getHeldClass()->getMroIndex(targetType->getHeldClass());

        if (mroIndex < 0) {
            // 'actualClass' doesn't have 'targetType' as a base class, so we can't
            // masquerade as it
            return false;
        }

        Class::initializeInstance((instance_ptr)outClassPtr, layout, mroIndex);
        layout->refcount += 1;

        return true;
    }

    void np_compileClassDestructor(VTable* vtable) {
        PyEnsureGilAcquired getTheGil;

        // check if there is an error already in place
        PyObject *existingErrorTypePtr, *existingErrorValuePtr, *existingErrorTracebackPtr;
        PyErr_Fetch(&existingErrorTypePtr, &existingErrorValuePtr, &existingErrorTracebackPtr);
        PyObjectHolder existingErrorType, existingErrorValue, existingErrorTraceback;

        existingErrorType.steal(existingErrorTypePtr);
        existingErrorValue.steal(existingErrorValuePtr);
        existingErrorTraceback.steal(existingErrorTracebackPtr);

        PyObject* singleton = getRuntimeSingleton();

        PyObjectStealer res(
            PyObject_CallMethod(
                singleton,
                "compileClassDestructor",
                "O",
                PyInstance::typeObj(vtable->mType->getClassType())
            )
        );

        if (!res) {
            if (existingErrorType) {
                // we were unwinding an exception already. Now we'll replace it with the new one.
                return;
            }

            throw PythonExceptionSet();
        }

        if (!vtable->mCompiledDestructorFun) {
            PyErr_Format(
                PyExc_TypeError,
                "Failed to populate the destructor"
            );
            throw PythonExceptionSet();
        }

        if (existingErrorType) {
            //reset the error code
            existingErrorType.extract();
            existingErrorValue.extract();
            existingErrorTraceback.extract();

            PyErr_Restore(existingErrorTypePtr, existingErrorValuePtr, existingErrorTracebackPtr);
        }
    }

    void np_raiseAttributeErr(uint8_t* attributeName) {
        PyEnsureGilAcquired getTheGil;

        PyErr_Format(
            PyExc_AttributeError,
            "Attribute '%s' is not initialized",
            attributeName
        );
        throw PythonExceptionSet();
    }

    ClassDispatchTable* computeTypeClassDispatchTable(Type* concreteType, Type* knownAsType) {
        if (concreteType->isClass()) {
            concreteType = ((Class*)concreteType)->getHeldClass();
        }

        if (knownAsType->isClass()) {
            knownAsType = ((Class*)knownAsType)->getHeldClass();
        }

        HeldClass* concreteCls = (HeldClass*)concreteType;
        HeldClass* knownAsCls = (HeldClass*)knownAsType;

        VTable* table = concreteCls->getVTable();
        ClassDispatchTable* dispatchTable = table->mDispatchTables;
        while (true) {
            if (dispatchTable->getInterfaceClass() == knownAsCls) {
                return dispatchTable;
            }

            dispatchTable++;
        }
    }


    // START math functions
    // parameters are checked before calling these functions

    // In general, for math functions, if the argument is finite, overflow errors are raised as exceptions.
    // But if the argument is inf, -inf, or nan, the function is calculated without raising an exception,
    // and the result _may_ be inf, -inf, or nan.
    // Fun fact: math.atan2(math.inf, math.inf) = .7853...
    inline void raise_if_inf(double a, double ret) {
        if (std::isfinite(a) && std::isinf(ret)) {
            PyEnsureGilAcquired getTheGil;
            PyErr_Format(PyExc_OverflowError, "math range error");
            throw PythonExceptionSet();
        }
    }

    double np_acos_float64(double d) {
        double ret = std::acos(d);
        raise_if_inf(d, ret);
        return ret;
    }

    double np_acosh_float64(double d) {
        double ret =  std::acosh(d);
        raise_if_inf(d, ret);
        return ret;
    }

    double np_asin_float64(double d) {
        double ret = std::asin(d);
        raise_if_inf(d, ret);
        return ret;
    }

    double np_asinh_float64(double d) {
        double ret = std::asinh(d);
        raise_if_inf(d, ret);
        return ret;
    }

    double np_atan_float64(double d) {
        double ret = std::atan(d);
        raise_if_inf(d, ret);
        return ret;
    }

    double np_atan2_float64(double d1, double d2) {
        double ret = std::atan2(d1, d2);
        // If one of d1 or d2 is not finite, don't raise exception.
        // And d1+d2 is not finite if one of d1 or d2 is not finite.
        raise_if_inf(d1+d2, ret);
        return ret;
    }

    double np_atanh_float64(double d) {
        double ret = std::atanh(d);
        raise_if_inf(d, ret);
        return ret;
    }

    double np_cosh_float64(double d) {
        double ret = std::cosh(d);
        raise_if_inf(d, ret);
        return ret;
    }

    double np_erf_float64(double d) {
        double ret = std::erf(d);
        raise_if_inf(d, ret);
        return ret;
    }

    double np_erfc_float64(double d) {
        double ret = std::erfc(d);
        raise_if_inf(d, ret);
        return ret;
    }

    double np_expm1_float64(double d) {
        double ret = std::expm1(d);
        raise_if_inf(d, ret);
        return ret;
    }

    // d = 171.0 will overflow 64-bit float, so could replace this calculation with a table lookup
    // from 0 to 170.
    // This also would avoid some accumulated errors in the multiplication.
    double np_factorial64(double d) {
        if (d >= 171.0) {
            PyErr_Format(PyExc_OverflowError, "math range error");
            throw PythonExceptionSet();
        }

        double ret = 1;
        double d1 = 1.0;
        while (d1 < d) {
            ret *= ++d1;
        }
        return ret;
    }

    // As for all of these, parameter checks have already occurred.
    // n = 21 will overflow 64-bit integer, so could replace this calculation with a table lookup
    // from 0 to 20.
    int64_t np_factorial(int64_t n) {
        if (n >= 21) {
            PyErr_Format(PyExc_OverflowError, "math range error");
            throw PythonExceptionSet();
        }

        int64_t ret = 1;
        int64_t i = 1;
        while (i < n) {
            ret *= ++i;
        }
        return ret;
    }

    double np_fmod_float64(double d1, double d2) {
        return std::fmod(d1, d2);
    }

    void np_frexp_float64(double d, instance_ptr ret) {
        int exp;
        double man = frexp(d, &exp);
        static Tuple* tupleT = Tuple::Make({Float64::Make(), Int64::Make()});

        tupleT->constructor(ret,
            [&](uint8_t* eltPtr, int64_t k) {
                if (k == 0)
                    *(double*)eltPtr = man;
                else
                    *(int64_t*)eltPtr = (int64_t)exp;
                }
            );
    }

    double np_gamma_float64(double d) {
        double ret = std::tgamma(d);
        raise_if_inf(d, ret);
        return ret;
    }

    int64_t np_gcd(int64_t i1, int64_t i2) {
        if (i1 < 0) i1 = -i1;
        if (i2 < 0) i2 = -i2;
        if (i1 == 0) {
            return i2;
        }

        while (i2 != 0) {
            if (i1 % i2 == 0) {
                return i2;
            }
            uint64_t i1t = i1;
            i1 = i2;
            i2 = i1t % i2;
        }
        return i1;
    }

    bool np_isclose_float64(double d1, double d2, double rel_tol, double abs_tol) {
        double m = fmax(fabs(d1), fabs(d2));
        return fabs(d1 - d2) <= fmax(rel_tol * m, abs_tol);
    }

    double np_ldexp_float64(double d, int i) {
        double ret = std::ldexp(d, i);
        raise_if_inf(d, ret);
        return ret;
    }

    double np_lgamma_float64(double d) {
        double ret = std::lgamma(d);
        raise_if_inf(d, ret);
        return ret;
    }

    double np_log1p_float64(double d) {
        double ret = std::log1p(d);
        raise_if_inf(d, ret);
        return ret;
    }

    void np_modf_float64(double d, instance_ptr ret) {
        double integer;
        double frac = modf(d, &integer);
        static Tuple* tupleT = Tuple::Make({Float64::Make(), Float64::Make()});

        tupleT->constructor(ret,
            [&](uint8_t* eltPtr, int64_t k) { *(double*)eltPtr = k == 0 ? frac : integer; }
        );
    }

    double np_sinh_float64(double d) {
        double ret = std::sinh(d);
        raise_if_inf(d, ret);
        return ret;
    }

    double np_tan_float64(double d) {
        double ret = std::tan(d);
        if (std::isinf(ret)) {
            PyErr_Format(PyExc_OverflowError, "math range error");
            throw PythonExceptionSet();
        }
        return ret;
    }

    double np_tanh_float64(double d) {
        double ret = std::tanh(d);
        raise_if_inf(d, ret);
        return ret;
    }

    // END math functions

    bool nativepython_runtime_string_eq(StringType::layout* lhs, StringType::layout* rhs) {
        if (lhs == rhs) {
            return true;
        }

        return StringType::cmpStaticEq(lhs, rhs);
    }

    int64_t nativepython_runtime_string_cmp(StringType::layout* lhs, StringType::layout* rhs) {
        return StringType::cmpStatic(lhs, rhs);
    }

    bool np_runtime_alternative_cmp(Alternative* tp, instance_ptr lhs, instance_ptr rhs, int64_t pyComparisonOp) {
        return Alternative::cmpStatic(tp, lhs, rhs, pyComparisonOp);
    }

    bool np_runtime_class_cmp(Class* tp, instance_ptr lhs, instance_ptr rhs, int64_t pyComparisonOp) {
        PyEnsureGilAcquired acquireTheGil;
        return Class::cmpStatic(tp, lhs, rhs, pyComparisonOp);
    }

    StringType::layout* nativepython_runtime_string_concat(StringType::layout* lhs, StringType::layout* rhs) {
        return StringType::concatenate(lhs, rhs);
    }

    StringType::layout* nativepython_runtime_string_lower(StringType::layout* l) {
        return StringType::lower(l);
    }

    StringType::layout* nativepython_runtime_string_upper(StringType::layout* l) {
        return StringType::upper(l);
    }

    StringType::layout* nativepython_runtime_string_capitalize(StringType::layout* l) {
        return StringType::capitalize(l);
    }

    StringType::layout* nativepython_runtime_string_casefold(StringType::layout* l) {
        return StringType::casefold(l);
    }

    StringType::layout* nativepython_runtime_string_swapcase(StringType::layout* l) {
        return StringType::swapcase(l);
    }

    StringType::layout* nativepython_runtime_string_title(StringType::layout* l) {
        return StringType::title(l);
    }

    StringType::layout* nativepython_runtime_string_strip(StringType::layout* l, bool whitespace, StringType::layout* values, bool fromLeft, bool fromRight) {
        return StringType::strip(l, whitespace, values, fromLeft, fromRight);
    }

    int64_t nativepython_runtime_string_find(StringType::layout* l, StringType::layout* sub, int64_t start, int64_t end) {
        return StringType::find(l, sub, start, end);
    }

    int64_t nativepython_runtime_string_rfind(StringType::layout* l, StringType::layout* sub, int64_t start, int64_t end) {
        return StringType::rfind(l, sub, start, end);
    }

    int64_t nativepython_runtime_string_count(StringType::layout* l, StringType::layout* sub, int64_t start, int64_t end) {
        return StringType::count(l, sub, start, end);
    }

    int64_t nativepython_runtime_string_index(StringType::layout* l, StringType::layout* sub, int64_t start, int64_t end) {
        int64_t ret =  StringType::find(l, sub, start, end);
        if (ret == -1) {
            PyEnsureGilAcquired acquireTheGil;
            PyErr_Format(
                PyExc_ValueError, "substring not found"
            );
            throw PythonExceptionSet();
        }
        return ret;
    }

    int64_t nativepython_runtime_string_rindex(StringType::layout* l, StringType::layout* sub, int64_t start, int64_t end) {
        int64_t ret =  StringType::rfind(l, sub, start, end);
        if (ret == -1) {
            PyEnsureGilAcquired acquireTheGil;
            PyErr_Format(
                PyExc_ValueError, "substring not found"
            );
            throw PythonExceptionSet();
        }
        return ret;
    }

    void nativepython_runtime_bytes_join(BytesType::layout** out, BytesType::layout* separator, ListOfType::layout* toJoin) {
        BytesType::join(out, separator, toJoin);
    }

    void nativepython_runtime_string_join(StringType::layout** out, StringType::layout* separator, ListOfType::layout* toJoin) {
        StringType::join(out, separator, toJoin);
    }

    ListOfType::layout* nativepython_runtime_bytes_split(BytesType::layout* l, BytesType::layout* sep, int64_t max) {
        static ListOfType* listOfBytesT = ListOfType::Make(BytesType::Make());

        ListOfType::layout* outList;

        listOfBytesT->constructor((instance_ptr)&outList);

        BytesType::split(outList, l, sep, max);

        return outList;
    }

    ListOfType::layout* nativepython_runtime_bytes_rsplit(BytesType::layout* l, BytesType::layout* sep, int64_t max) {
        static ListOfType* listOfBytesT = ListOfType::Make(BytesType::Make());

        ListOfType::layout* outList;

        listOfBytesT->constructor((instance_ptr)&outList);

        BytesType::rsplit(outList, l, sep, max);

        return outList;
    }

    ListOfType::layout* nativepython_runtime_bytes_splitlines(BytesType::layout* l, bool keepends) {
        static ListOfType* listOfBytesT = ListOfType::Make(BytesType::Make());

        ListOfType::layout* outList;

        listOfBytesT->constructor((instance_ptr)&outList);

        BytesType::splitlines(outList, l, keepends);

        return outList;
    }

    ListOfType::layout* nativepython_runtime_string_split(StringType::layout* l, StringType::layout* sep, int64_t max) {
        static ListOfType* listOfStringT = ListOfType::Make(StringType::Make());

        ListOfType::layout* outList;

        listOfStringT->constructor((instance_ptr)&outList);
        StringType::split(outList, l, sep, max);

        return outList;
    }

    ListOfType::layout* nativepython_runtime_string_rsplit(StringType::layout* l, StringType::layout* sep, int64_t max) {
        static ListOfType* listOfStringT = ListOfType::Make(StringType::Make());

        ListOfType::layout* outList;

        listOfStringT->constructor((instance_ptr)&outList);

        StringType::rsplit(outList, l, sep, max);

        return outList;
    }

    ListOfType::layout* nativepython_runtime_string_splitlines(StringType::layout* l, bool keepends) {
        static ListOfType* listOfStringT = ListOfType::Make(StringType::Make());

        ListOfType::layout* outList;

        listOfStringT->constructor((instance_ptr)&outList);

        StringType::splitlines(outList, l, keepends);

        return outList;
    }

    bool nativepython_runtime_string_isalpha(StringType::layout* l) {
        return StringType::isalpha(l);
    }

    bool nativepython_runtime_string_isalnum(StringType::layout* l) {
        return StringType::isalnum(l);
    }

    bool nativepython_runtime_string_isdecimal(StringType::layout* l) {
        return StringType::isdecimal(l);
    }

    bool nativepython_runtime_string_isdigit(StringType::layout* l) {
        return StringType::isdigit(l);
    }

    bool nativepython_runtime_string_isidentifier(StringType::layout* l) {
        // Not bothering to implement this myself...
        PyEnsureGilAcquired getTheGil;
        PyObject* s = PyInstance::extractPythonObject((instance_ptr)&l, StringType::Make());
        if (!s) {
            throw PythonExceptionSet();
        }

        int ret = PyUnicode_IsIdentifier(s);
        decref(s);
        if (ret == -1) {
            throw PythonExceptionSet();
        }

        return ret;
    }

    bool nativepython_runtime_string_islower(StringType::layout* l) {
        return StringType::islower(l);
    }

    bool nativepython_runtime_string_isnumeric(StringType::layout* l) {
        return StringType::isnumeric(l);
    }

    bool nativepython_runtime_string_isprintable(StringType::layout* l) {
        return StringType::isprintable(l);
    }

    bool nativepython_runtime_string_isspace(StringType::layout* l) {
        return StringType::isspace(l);
    }

    bool nativepython_runtime_string_istitle(StringType::layout* l) {
        return StringType::istitle(l);
    }

    bool nativepython_runtime_string_isupper(StringType::layout* l) {
        return StringType::isupper(l);
    }

    StringType::layout* nativepython_runtime_string_getitem_int64(StringType::layout* lhs, int64_t index) {
        return StringType::getitem(lhs, index);
    }

    StringType::layout* nativepython_runtime_string_mult(StringType::layout* lhs, int64_t rhs) {
        return StringType::mult(lhs, rhs);
    }

    StringType::layout* nativepython_runtime_string_chr(int64_t code) {
        if (code < 0 || code > 0x10ffff) {
            PyEnsureGilAcquired getTheGil;

            PyErr_Format(
                PyExc_ValueError, "chr() arg not in range(0x10ffff)"
            );

            throw PythonExceptionSet();
        }

        return StringType::singleFromCodepoint(code);
    }

    int64_t nativepython_runtime_string_ord(StringType::layout* lhs) {
        if (StringType::countStatic(lhs) != 1) {
            PyEnsureGilAcquired getTheGil;
            PyErr_Format(
                PyExc_TypeError, "ord() expected a character, but string of length %d found",
                StringType::countStatic(lhs)
            );

            throw PythonExceptionSet();
        }

        return StringType::getord(lhs);
    }

    StringType::layout* nativepython_runtime_string_getslice_int64(StringType::layout* lhs, int64_t start, int64_t stop) {
        return StringType::getsubstr(lhs, start, stop);
    }

    StringType::layout* nativepython_runtime_string_from_utf8_and_len(const char* utf8_str, int64_t len) {
        return StringType::createFromUtf8(utf8_str, len);
    }

    BytesType::layout* nativepython_runtime_bytes_getslice_int64(BytesType::layout* lhs, int64_t start, int64_t stop) {
        if (!lhs) {
            return nullptr;
        }
        int32_t len = lhs->bytecount;

        if (start < 0) {
            start += len;
        }
        if (stop < 0) {
            stop += len;
        }
        if (start < 0) {
            start = 0;
        }
        if (stop < 0) {
            stop = 0;
        }
        if (start > len) {
            start = len;
        }
        if (stop > len) {
            stop = len;
        }
        if (start >= stop) {
            return nullptr;
        }

        return BytesType::createFromPtr((char*)lhs->data + start, stop - start);
    }

    int64_t nativepython_runtime_bytes_cmp(BytesType::layout* lhs, BytesType::layout* rhs) {
        return BytesType::cmpStatic(lhs, rhs);
    }

    BytesType::layout* nativepython_runtime_bytes_concat(BytesType::layout* lhs, BytesType::layout* rhs) {
        return BytesType::concatenate(lhs, rhs);
    }

    BytesType::layout* nativepython_runtime_bytes_from_ptr_and_len(const char* utf8_str, int64_t len) {
        return BytesType::createFromPtr(utf8_str, len);
    }

    BytesType::layout* nativepython_runtime_bytes_lower(BytesType::layout* l) {
        return BytesType::lower(l);
    }

    BytesType::layout* nativepython_runtime_bytes_upper(BytesType::layout* l) {
        return BytesType::upper(l);
    }

    BytesType::layout* nativepython_runtime_bytes_capitalize(BytesType::layout* l) {
        return BytesType::capitalize(l);
    }

    BytesType::layout* nativepython_runtime_bytes_swapcase(BytesType::layout* l) {
        return BytesType::swapcase(l);
    }

    BytesType::layout* nativepython_runtime_bytes_title(BytesType::layout* l) {
        return BytesType::title(l);
    }

    BytesType::layout* nativepython_runtime_bytes_strip(BytesType::layout* l, bool fromLeft, bool fromRight) {
        return BytesType::strip(l, true, nullptr, fromLeft, fromRight);
    }

    BytesType::layout* nativepython_runtime_bytes_strip2(BytesType::layout* l, BytesType::layout* values, bool fromLeft, bool fromRight) {
        return BytesType::strip(l, false, values, fromLeft, fromRight);
    }

    BytesType::layout* nativepython_runtime_bytes_mult(BytesType::layout* lhs, int64_t rhs) {
        return BytesType::mult(lhs, rhs);
    }

    BytesType::layout* nativepython_runtime_bytes_replace(
            BytesType::layout* l,
            BytesType::layout* old,
            BytesType::layout* the_new,
            int64_t count
    ) {
        return BytesType::replace(l, old, the_new, count);
    }

    enum Codec { CODEC_UNKNOWN = 0, CODEC_UTF8 };
    Codec CodecFromStr(const char *s) {
        if (!s || !strcmp(s, "utf-8")
                || !strcmp(s, "utf_8")
                || !strcmp(s, "utf8")) {
            return CODEC_UTF8;
        }
        return CODEC_UNKNOWN;
    }

    enum ErrHandler { ERH_UNKNOWN = 0, ERH_STRICT = 1, ERH_IGNORE = 2, ERH_REPLACE = 3 };
    ErrHandler ErrHandlerFromStr(const char *s) {
        if (!s || !strcmp(s, "strict")) {
            return ERH_STRICT;
        } else if (!strcmp(s, "ignore")) {
            return ERH_IGNORE;
        } else if (!strcmp(s, "replace")) {
            return ERH_IGNORE;
        } else {
            return ERH_UNKNOWN;
        }
    }

    StringType::layout* nativepython_runtime_bytes_decode(
            BytesType::layout* l,
            StringType::layout* encoding,
            StringType::layout* errors
    ) {
        const char* c_encoding = 0;
        std::string sEncoding;
        if (encoding) {
            sEncoding = StringType::Make()->toUtf8String((instance_ptr)&encoding);
            c_encoding = sEncoding.c_str();
        }

        const char* c_errors = 0;
        std::string sErrors;
        if (errors) {
            sErrors = StringType::Make()->toUtf8String((instance_ptr)&errors);
            c_errors = sErrors.c_str();
        }

        Codec codec = CodecFromStr(c_encoding);
        if (codec) {
            ErrHandler errhandler = ErrHandlerFromStr(c_errors);
            if (errhandler) {
                if (codec == CODEC_UTF8) {
                    uint8_t* data = l ? (uint8_t*)(l->data) : 0;
                    // TODO: combine countUtf8Codepoints and createFromUtf8 into a single function with a single loop
                    size_t pointCount = l ? StringType::countUtf8Codepoints(data, l->bytecount) : 0;
                    return StringType::createFromUtf8((const char*)data, pointCount);
                }
            }
        }

        PyEnsureGilAcquired getTheGil;

        PyObject* b = PyInstance::extractPythonObject((instance_ptr)&l, BytesType::Make());
        if (!b) {
            throw PythonExceptionSet();
        }

        PyObject* s = PyUnicode_FromEncodedObject(b, c_encoding, c_errors);
        decref(b);
        if (!s) {
            throw PythonExceptionSet();
        }

        StringType::layout* ret = 0;
        PyInstance::copyConstructFromPythonInstance(StringType::Make(), (instance_ptr)&ret, s, ConversionLevel::New);
        decref(s);

        return ret;
    }


    BytesType::layout* encodeUtf8(StringType::layout *l, ErrHandler err) {
        int64_t max_ret_bytes_per_codepoint = l ? (l->bytes_per_codepoint < 4 ? l->bytes_per_codepoint + 1 : 4) : 0;
        int64_t max_ret_size = l ? l->pointcount * max_ret_bytes_per_codepoint : 0;

        BytesType::layout* out = (BytesType::layout*)tp_malloc(sizeof(BytesType::layout) + max_ret_size);
        out->refcount = 1;
        out->hash_cache = -1;
        out->bytecount = 0;
        if (!l || !l->pointcount) return out;

        int64_t cur = 0;
        for (int64_t i = 0; i < l->pointcount; i++) {
            uint32_t c = StringType::getpoint(l, i);
            if (c < 0x80) {
                out->data[cur++] = (uint8_t)c;
            } else if (c < 0x800) {
                out->data[cur++] = 0xC0 | (c >> 6);
                out->data[cur++] = 0x80 | (c & 0x3F);
            } else if (c < 0x10000) {
                out->data[cur++] = 0xE0 | (c >> 12);
                out->data[cur++] = 0x80 | ((c >> 6) & 0x3F);
                out->data[cur++] = 0x80 | (c & 0x3F);
            } else if (c < 0x110000) {
                out->data[cur++] = 0xF0 | (c >> 18);
                out->data[cur++] = 0x80 | ((c >> 12) & 0x3F);
                out->data[cur++] = 0x80 | ((c >> 6) & 0x3F);
                out->data[cur++] = 0x80 | (c & 0x3F);
            } else if (err == ERH_REPLACE) {
                const uint32_t rc = 0xFFFD;
                out->data[cur++] = 0xE0 | (rc >> 12);
                out->data[cur++] = 0x80 | ((rc >> 6) & 0x3F);
                out->data[cur++] = 0x80 | (rc & 0x3F);
            } else if (err == ERH_STRICT) {
                PyEnsureGilAcquired getTheGil;
                PyErr_Format(
                    PyExc_UnicodeError,
                    "illegal Unicode character"
                    );
                throw PythonExceptionSet();
            } // else err == ERH_IGNORE
        }

        out->bytecount = cur;
        if (out->bytecount < max_ret_size) {
            out = (BytesType::layout*)tp_realloc(
                out,
                sizeof(BytesType::layout) + max_ret_size,
                sizeof(BytesType::layout) + out->bytecount
            );
        }

        return out;
    }

    BytesType::layout* nativepython_runtime_str_encode(
            StringType::layout* l,
            StringType::layout* encoding,
            StringType::layout* errors
    ) {
        const char* c_encoding = 0;
        std::string sEncoding;
        if (encoding) {
            sEncoding = StringType::Make()->toUtf8String((instance_ptr)&encoding);
            c_encoding = sEncoding.c_str();
        }

        const char* c_errors = 0;
        std::string sErrors;
        if (errors) {
            sErrors = StringType::Make()->toUtf8String((instance_ptr)&errors);
            c_errors = sErrors.c_str();
        }

        Codec codec = CodecFromStr(c_encoding);
        if (codec) {
            ErrHandler errhandler = ErrHandlerFromStr(c_errors);
            if (errhandler) {
                if (codec == CODEC_UTF8) {
                    return encodeUtf8(l, errhandler);
                }
            }
        }

        PyEnsureGilAcquired getTheGil;

        PyObject* s = PyInstance::extractPythonObject((instance_ptr)&l, StringType::Make());
        if (!s) {
            throw PythonExceptionSet();
        }

        PyObject* b = PyUnicode_AsEncodedString(s, c_encoding, c_errors);
        decref(s);
        if (!b) {
            throw PythonExceptionSet();
        }

        BytesType::layout* ret = 0;
        PyInstance::copyConstructFromPythonInstance(BytesType::Make(), (instance_ptr)&ret, b, ConversionLevel::New);
        decref(b);

        return ret;
    }

    BytesType::layout* nativepython_runtime_bytes_translate(
            BytesType::layout* l,
            BytesType::layout* table,
            BytesType::layout* to_delete
    ) {
        return BytesType::translate(l, table, to_delete);
    }

    BytesType::layout* nativepython_runtime_bytes_maketrans(
            BytesType::layout* from,
            BytesType::layout* to
    ) {
        return BytesType::maketrans(from, to);
    }

    PythonObjectOfType::layout_type* nativepython_runtime_create_pyobj(PyObject* p) {
        PyEnsureGilAcquired getTheGil;
        return PythonObjectOfType::createLayout(p);
    }

    void np_initialize_exception(PythonObjectOfType::layout_type* layout) {
        PyEnsureGilAcquired getTheGil;

        PyObject* prevType;
        PyObject* prevValue;
        PyObject* prevTraceback;
        PyErr_GetExcInfo(&prevType, &prevValue, &prevTraceback);

        if (layout) {
            PyTypeObject* tp = layout->pyObj->ob_type;
            bool hasBaseE = false;

            while (tp) {
                if (tp == (PyTypeObject*)PyExc_BaseException) {
                    hasBaseE = true;
                }
                tp = tp->tp_base;
            }

            if (!hasBaseE) {
                PyErr_Format(
                    PyExc_TypeError,
                    "exceptions must derive from BaseException, not %S",
                    (PyObject*)layout->pyObj->ob_type
                );

                return;
            }

            if (prevValue) {
                PyException_SetContext(layout->pyObj, prevValue);
            }
            decref(prevType);
            decref(prevTraceback);

            PyErr_SetObject(
                (PyObject*)layout->pyObj->ob_type,
                layout->pyObj
            );
        }
        else {
            if (!prevValue) {
                decref(prevType);
                decref(prevValue);
                decref(prevTraceback);
                PyErr_SetString(PyExc_RuntimeError, "No active exception to reraise");
                throw PythonExceptionSet();
            }
            PyErr_Restore(prevType, prevValue, prevTraceback);
        }
    }

    void np_initialize_exception_w_cause(
            PythonObjectOfType::layout_type* layoutExc,
            PythonObjectOfType::layout_type* layoutCause
            ) {
        PyEnsureGilAcquired getTheGil;

        if (layoutExc) {
            PyTypeObject* tp = layoutExc->pyObj->ob_type;
            bool hasBaseE = false;

            while (tp) {
                if (tp == (PyTypeObject*)PyExc_BaseException) {
                    hasBaseE = true;
                }
                tp = tp->tp_base;
            }

            if (!hasBaseE) {
                PyErr_Format(
                    PyExc_TypeError,
                    "exceptions must derive from BaseException, not %S",
                    (PyObject*)layoutExc->pyObj->ob_type
                );

                return;
            }

            PyException_SetCause(layoutExc->pyObj, layoutCause ? incref(layoutCause->pyObj) : NULL);
            PyErr_Restore((PyObject*)incref(layoutExc->pyObj->ob_type), incref(layoutExc->pyObj), nullptr);
        }
        else {
            PyObject* prevType;
            PyObject* prevValue;
            PyObject* prevTraceback;
            PyErr_GetExcInfo(&prevType, &prevValue, &prevTraceback);

            if (!prevValue) {
                decref(prevType);
                decref(prevValue);
                decref(prevTraceback);
                PyErr_SetString(PyExc_RuntimeError, "No active exception to reraise");
                throw PythonExceptionSet();
            }
            PyException_SetCause(prevValue, layoutCause ? incref(layoutCause->pyObj) : NULL);
            PyErr_Restore(prevType, prevValue, prevTraceback);
        }
    }

    void np_clear_exception() {
        PyEnsureGilAcquired getTheGil;
        PyErr_Clear();
    }

    void np_clear_exc_info() {
        PyEnsureGilAcquired getTheGil;
        PyErr_SetExcInfo(NULL, NULL, NULL);
    }

    void np_fetch_exception_tuple(instance_ptr inst) {
        PyEnsureGilAcquired getTheGil;

        static Type* return_type = Tuple::Make({
            PythonObjectOfType::AnyPyObject(),
            PythonObjectOfType::AnyPyObject(),
            PythonObjectOfType::AnyPyObject()
        });

        PyObject* type;
        PyObject* value;
        PyObject* traceback;
        PyErr_Fetch(&type, &value, &traceback);

        if (type && value) {
            PyErr_NormalizeException(&type, &value, &traceback);
            if (traceback) {
                PyException_SetTraceback(value, traceback);
            }
        }

        PyObjectStealer p(PyTuple_New(3));
        PyTuple_SetItem(p, 0, type ? type : incref(Py_None));
        PyTuple_SetItem(p, 1, value ? value : incref(Py_None));
        PyTuple_SetItem(p, 2, traceback ? traceback : incref(Py_None));

        PyInstance::copyConstructFromPythonInstance(return_type, inst, p, ConversionLevel::New);

        // Since we've caught it, need to save it as the most recently caught exception.
        PyErr_SetExcInfo(type, value, traceback);
    }

    void np_raise_exception_tuple(
        // this should be a pointer to a Tuple(object, object, object), as returned by
        // fetch_exception_tuple
        PythonObjectOfType::layout_type** tuple
    ) {
        PyEnsureGilAcquired getTheGil;

        PyErr_Restore(
            incref(tuple[0]->pyObj),
            incref(tuple[1]->pyObj),
            incref(tuple[2]->pyObj)
        );

        throw PythonExceptionSet();
    }

    bool np_match_exception(PyObject* exc) {
        PyEnsureGilAcquired getTheGil;
        return PyErr_ExceptionMatches(exc);
    }

    bool np_match_given_exception(PythonObjectOfType::layout_type* given, PyObject* exc) {
        PyEnsureGilAcquired getTheGil;
        return PyErr_GivenExceptionMatches(given->pyObj, exc);
    }

    // fetch = catch + return the caught exception
    // This should be only called within an exception handler, so we know there
    // is an exception waiting for us to fetch.
    PythonObjectOfType::layout_type* np_fetch_exception() {
        PyEnsureGilAcquired getTheGil;

        PyObject* type;
        PyObject* value;
        PyObject* traceback;

        PyErr_Fetch(&type, &value, &traceback);
        PyErr_NormalizeException(&type, &value, &traceback);
        if (traceback) {
            PyException_SetTraceback(value, traceback);
        }

        PythonObjectOfType::layout_type* ret = PythonObjectOfType::createLayout(value);

        // Since we've caught it, need to save it as the most recently caught exception.
        PyErr_SetExcInfo(type, value, traceback);

        return ret;
    }

    // This should be only called within an exception handler, so we know there
    // is an exception waiting for us to fetch.
    void np_catch_exception() {
        PyEnsureGilAcquired getTheGil;

        PyObject* type;
        PyObject* value;
        PyObject* traceback;

        PyErr_Fetch(&type, &value, &traceback);
        PyErr_NormalizeException(&type, &value, &traceback);
        if (traceback) {
            PyException_SetTraceback(value, traceback);
        }

        // Since we've caught it, need to save it as the most recently caught exception.
        PyErr_SetExcInfo(type, value, traceback);
    }

    void np_add_traceback(const char* funcname, const char* filename, int lineno) {
        PyEnsureGilAcquired getTheGil;
        _PyTraceback_Add(funcname, filename, lineno);
    }

    PythonObjectOfType::layout_type* np_builtin_pyobj_by_name(const char* utf8_name) {
        PyEnsureGilAcquired getTheGil;

        static PyObject* module = builtinsModule();

        return PythonObjectOfType::createLayout(PyObject_GetAttrString(module, utf8_name));
    }

    PythonObjectOfType::layout_type* nativepython_runtime_get_pyobj_None() {
        PyEnsureGilAcquired acquireTheGil;

        return PythonObjectOfType::createLayout(Py_None);
    }

    StringType::layout* nativepython_runtime_repr(instance_ptr inst, Type* tp) {
        PyEnsureGilAcquired getTheGil;

        PyObjectStealer o(PyInstance::extractPythonObject(inst, tp));
        if (!o) {
            throw PythonExceptionSet();
        }
        PyObject *r = PyObject_Repr(o);
        if (!r) {
            throw PythonExceptionSet();
        }
        Py_ssize_t s;
        const char* c = PyUnicode_AsUTF8AndSize(r, &s);
        StringType::layout *ret = StringType::createFromUtf8(c, s);
        decref(r);
        return ret;
    }


    // attempt to convert object in 'inst' of type 'tp' to a string using the interpreter
    // on success, return True, and outStr points to a string. On failure, return false,
    // and outStr remains uninitialized.
    bool np_try_pyobj_to_str(instance_ptr inst, StringType::layout** outStr, Type* tp) {
        PyEnsureGilAcquired getTheGil;

        PyObjectStealer o(PyInstance::extractPythonObject(inst, tp));
        if (!o) {
            PyErr_Clear();
            return false;
        }
        PyObject *r = PyObject_Str(o);
        if (!r) {
            PyErr_Clear();
            return false;
        }

        Py_ssize_t s;
        const char* c = PyUnicode_AsUTF8AndSize(r, &s);
        *outStr = StringType::createFromUtf8(c, s);
        decref(r);

        return true;
    }

    uint64_t nativepython_pyobj_len(PythonObjectOfType::layout_type* layout) {
        PyEnsureGilAcquired getTheGil;

        int ret = PyObject_Length(layout->pyObj);
        if (ret == -1) {
            throw PythonExceptionSet();
        }
        return ret;
    }

    // call a Function object from the interpreter
    PythonObjectOfType::layout_type* nativepython_runtime_call_func_as_pyobj(
        Function* func,
        instance_ptr closure,
        int argCount,
        int kwargCount,
        ...
    ) {
        PyEnsureGilAcquired getTheGil;

        if (func->getOverloads().size() != 1) {
            PyErr_SetString(PyExc_TypeError, "can't call a nocompile function with more than 1 overload");
            throw PythonExceptionSet();
        }

        PyObjectStealer funcObj(func->getOverloads()[0].buildFunctionObj(func->getClosureType(), closure));

        if (!funcObj) {
            throw PythonExceptionSet();
        }

        // each of 'argCount' arguments is a PyObject* followed by a const char*
        va_list va_args;
        va_start(va_args, kwargCount);

        PyObjectStealer args(PyTuple_New(argCount));
        PyObjectStealer kwargs(PyDict_New());

        for (int i = 0; i < argCount; ++i) {
            PyTuple_SetItem((PyObject*)args, i, incref(va_arg(va_args, PythonObjectOfType::layout_type*)->pyObj));
        }

        for (int i = 0; i < kwargCount; ++i) {
            PyObject* kwargVal = va_arg(va_args, PythonObjectOfType::layout_type*)->pyObj;
            const char* kwargName = va_arg(va_args, const char*);

            PyDict_SetItemString((PyObject*)kwargs, kwargName, kwargVal);
        }

        va_end(va_args);

        PyObject* res = PyObject_Call((PyObject*)funcObj, args, kwargs);

        if (!res) {
            throw PythonExceptionSet();
        }

        return PythonObjectOfType::stealToCreateLayout(res);
    }

    PythonObjectOfType::layout_type* nativepython_runtime_call_pyobj(PythonObjectOfType::layout_type* toCall, int argCount, int kwargCount, ...) {
        PyEnsureGilAcquired getTheGil;

        // each of 'argCount' arguments is a PyObject* followed by a const char*
        va_list va_args;
        va_start(va_args, kwargCount);

        PyObjectStealer args(PyTuple_New(argCount));
        PyObjectStealer kwargs(PyDict_New());

        for (int i = 0; i < argCount; ++i) {
            PyTuple_SetItem((PyObject*)args, i, incref(va_arg(va_args, PythonObjectOfType::layout_type*)->pyObj));
        }

        for (int i = 0; i < kwargCount; ++i) {
            PyObject* kwargVal = va_arg(va_args, PythonObjectOfType::layout_type*)->pyObj;
            const char* kwargName = va_arg(va_args, const char*);

            PyDict_SetItemString((PyObject*)kwargs, kwargName, kwargVal);
        }

        va_end(va_args);

        PyObject* res = PyObject_Call(toCall->pyObj, args, kwargs);

        if (!res) {
            throw PythonExceptionSet();
        }

        return PythonObjectOfType::stealToCreateLayout(res);
    }

    PythonObjectOfType::layout_type* nativepython_runtime_getattr_pyobj(PythonObjectOfType::layout_type* p, const char* a) {
        PyEnsureGilAcquired getTheGil;

        PyObject* res = PyObject_GetAttrString(p->pyObj, a);

        if (!res) {
            throw PythonExceptionSet();
        }

        return PythonObjectOfType::stealToCreateLayout(res);
    }

    PythonObjectOfType::layout_type* nativepython_runtime_getitem_pyobj(PythonObjectOfType::layout_type* p, PythonObjectOfType::layout_type* a) {
        PyEnsureGilAcquired getTheGil;

        PyObject* res = PyObject_GetItem(p->pyObj, a->pyObj);

        if (!res) {
            throw PythonExceptionSet();
        }

        return PythonObjectOfType::stealToCreateLayout(res);
    }

    void nativepython_runtime_delitem_pyobj(PythonObjectOfType::layout_type* p, PythonObjectOfType::layout_type* a) {
        PyEnsureGilAcquired getTheGil;

        int success = PyObject_DelItem(p->pyObj, a->pyObj);

        if (success != 0) {
            throw PythonExceptionSet();
        }
    }

    void nativepython_runtime_setitem_pyobj(
            PythonObjectOfType::layout_type* p,
            PythonObjectOfType::layout_type* index,
            PythonObjectOfType::layout_type* value
    ) {
        PyEnsureGilAcquired getTheGil;

        int res = PyObject_SetItem(p->pyObj, index->pyObj, value->pyObj);

        if (res) {
            throw PythonExceptionSet();
        }
    }

    void nativepython_runtime_setattr_pyobj(PythonObjectOfType::layout_type* p, const char* a, PythonObjectOfType::layout_type* val) {
        PyEnsureGilAcquired getTheGil;

        int res = PyObject_SetAttrString(p->pyObj, a, val->pyObj);

        if (res) {
            throw PythonExceptionSet();
        }
    }

    void np_destroy_pyobj_handle(PythonObjectOfType::layout_type* p) {
        PythonObjectOfType::destroyLayoutIfRefcountIsZero(p);
    }

    int64_t nativepython_runtime_mod_int64_int64(int64_t l, int64_t r) {
        if (r == 1 || r == -1 || r == 0 || l == 0) {
            return 0;
        }

        if (r < 0) {
            if (l < 0) {
                return -((-l) % (-r));
            }
            return - (-r - ((l-1) % (-r) + 1) );
        }

        if (l < 0) {
            return r - ((-l-1) % r + 1);
        }

        return l % r;
    }

    int64_t nativepython_runtime_mod_uint64_uint64(uint64_t l, uint64_t r) {
        if (r == 1 || r == 0 || l == 0) {
            return 0;
        }

        return l % r;
    }

    double nativepython_runtime_mod_float64_float64(double l, double r) {
        if (std::isnan(r) or std::isnan(l)) {
            return NAN;
        }

        if (l == 0.0) {
            return 0.0;
        }

        if (r == 0.0) {
            PyEnsureGilAcquired acquireTheGil;

            PyErr_Format(PyExc_ZeroDivisionError, "float modulo");
            throw PythonExceptionSet();
        }

        double mod = fmod(l, r);
        if (mod) {
            if ((r < 0) != (mod < 0))
                mod += r;
        }
        return mod;
    }

    double nativepython_runtime_pow_float64_float64(double l, double r) {
        if (l == 0.0 && r < 0.0)
        {
            PyEnsureGilAcquired acquireTheGil;

            PyErr_Format(PyExc_ZeroDivisionError, "0.0 cannot be raised to a negative power");
            throw PythonExceptionSet();
        }

        double result = std::pow(l, r);

        if (l < 0.0 && r > 0.0 && nativepython_runtime_mod_float64_float64(r, 2.0) == 1.0 && result > 0.0)
            return -result;

        return result;
    }

    double nativepython_runtime_pow_int64_int64(int64_t l, int64_t r) {
        if (l == 0 && r < 0) {
            PyEnsureGilAcquired acquireTheGil;
            PyErr_Format(PyExc_ZeroDivisionError, "0.0 cannot be raised to a negative power");
            throw PythonExceptionSet();
        }

        double result = std::pow(l, r);

        if (l < 0 && r > 0 && r % 2 && result > 0)
            return -result;

        return result;
    }

    double nativepython_runtime_pow_uint64_uint64(uint64_t l, uint64_t r) {
        return std::pow(l,r);
    }

    // should match corresponding function in PyRegisterTypeInstance.hpp
    int64_t nativepython_runtime_lshift_int64_int64(int64_t l, int64_t r) {
        if (r < 0) {
            PyEnsureGilAcquired acquireTheGil;
            PyErr_Format(PyExc_ValueError, "negative shift count");
            throw PythonExceptionSet();
        }

        if (PY_MINOR_VERSION > 6 && l == 0) {
            return 0;
        }

        if ((l == 0 && r > SSIZE_MAX) || (l != 0 && r >= 1024)) { // 1024 is arbitrary
            PyEnsureGilAcquired acquireTheGil;
            PyErr_Format(PyExc_OverflowError, "shift count too large");
            throw PythonExceptionSet();
        }

        return (l >= 0) ? l << r : -((-l) << r);
    }

    // should match corresponding function in PyRegisterTypeInstance.hpp
    uint64_t nativepython_runtime_lshift_uint64_uint64(uint64_t l, uint64_t r) {
        if ((l == 0 && r > SSIZE_MAX) || (l != 0 && r >= 1024)) { // 1024 is arbitrary
            PyEnsureGilAcquired acquireTheGil;
            PyErr_Format(PyExc_OverflowError, "shift count too large");
            throw PythonExceptionSet();
        }
        return l << r;
    }

    // should match corresponding function in PyRegisterTypeInstance.hpp
    uint64_t nativepython_runtime_rshift_uint64_uint64(uint64_t l, uint64_t r) {
        if (r > SSIZE_MAX) {
            PyEnsureGilAcquired acquireTheGil;
            PyErr_Format(PyExc_OverflowError, "shift count too large");
            throw PythonExceptionSet();
        }
        if (r == 0)
            return l;
        if (r >= 64)
            return 0;
        return l >> r;
    }

    // should match corresponding function in PyRegisterTypeInstance.hpp
    int64_t nativepython_runtime_rshift_int64_int64(int64_t l, int64_t r) {
        if (r < 0) {
            PyEnsureGilAcquired acquireTheGil;
            PyErr_Format(PyExc_ValueError, "negative shift count");
            throw PythonExceptionSet();
        }
        if (r > SSIZE_MAX) {
            PyEnsureGilAcquired acquireTheGil;
            PyErr_Format(PyExc_OverflowError, "shift count too large");
            throw PythonExceptionSet();
        }
        if (r == 0)
            return l;
        if (l >= 0)
            return l >> r;
        int64_t ret = (-l) >> r;
        if (ret == 0)
            return -1;
        if (l == -l)  // int64_min case
            return ret;
        return -ret;
    }

    // should match corresponding function in PyRegisterTypeInstance.hpp
    int64_t nativepython_runtime_floordiv_int64_int64(int64_t l, int64_t r) {
        if (r == 0) {
            PyEnsureGilAcquired acquireTheGil;
            PyErr_Format(PyExc_ZeroDivisionError, "integer floordiv by zero");
            throw PythonExceptionSet();
        }
        if (l < 0 && l == -l && r == -1) {
            // overflow because int64_min / -1 > int64_max
            return 1;
        }

        if ((l>0 && r>0) || (l<0 && r<0)) { //same signs
            return l / r;
        }
        // opposite signs
        return (l % r) ? l / r - 1 : l / r;
    }

    // should match corresponding function in PyRegisterTypeInstance.hpp
    double nativepython_runtime_floordiv_float64_float64(double l, double r) {
        if (r == 0.0) {
            PyEnsureGilAcquired acquireTheGil;
            PyErr_Format(PyExc_ZeroDivisionError, "floating point floordiv by zero");
            throw PythonExceptionSet();
        }

        double result = (l - nativepython_runtime_mod_float64_float64(l, r))/r;

        double floorresult = std::floor(result);

        if (result - floorresult > 0.5)
            floorresult += 1.0;

        return floorresult;
    }

    void np_throwNullPtr() {
        throw (void*)nullptr;
    }

    // attempt to initialize 'tgt' of type 'tp' with data from 'obj'. Returns true if we
    // are able to do so, false otherwise. If 'canThrow', then allow an exception to propagate
    // if we can't convert.
    bool np_runtime_pyobj_to_typed(
        PythonObjectOfType::layout_type *layout,
        instance_ptr tgt,
        Type* tp,
        int64_t conversionLevel,
        int64_t canThrow
    ) {
        PyEnsureGilAcquired acquireTheGil;

        try {
            ConversionLevel level = intToConversionLevel(conversionLevel);

            if (!PyInstance::pyValCouldBeOfType(tp, layout->pyObj, level) && !canThrow) {
                return false;
            }

            PyInstance::copyConstructFromPythonInstance(
                tp,
                tgt,
                layout->pyObj,
                level
            );

            return true;
        } catch (PythonExceptionSet&) {
            if (canThrow) {
                throw;
            }

            PyErr_Clear();
            return false;
        } catch(std::exception& e) {
            if (canThrow) {
                PyErr_SetString(PyExc_TypeError, e.what());
                throw PythonExceptionSet();
            }
            return false;
        }
    }

    PythonObjectOfType::layout_type* np_runtime_to_pyobj(instance_ptr obj, Type* tp) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyInstance::extractPythonObject(obj, tp);

        if (!res) {
            throw PythonExceptionSet();
        }

        return PythonObjectOfType::stealToCreateLayout(res);
    }

    void np_print_bytes(uint8_t* bytes) {
        std::cout << bytes << std::flush;
    }

    void nativepython_print_string(StringType::layout* layout) {
        std::cout << StringType::Make()->toUtf8String((instance_ptr)&layout) << std::flush;
    }

    /* convert a float to an int, returning true if conversion is successful.

    This function writes the value into 'out' if successful. if 'canThrow', then
    raise an exception on failure.

    Type* determines what kind of instance we're writing into.
    */
    bool nativepython_float32_to_int(void* out, float f, bool canThrow, Type* targetType) {
        if (!std::isfinite(f)) {
            if (!canThrow) {
                return false;
            }

            if (std::isnan(f)) {
                PyEnsureGilAcquired acquireTheGil;

                PyErr_Format(PyExc_ValueError, "Cannot convert float NaN to integer");
                throw PythonExceptionSet();
            }
            if (std::isinf(f)) {
                PyEnsureGilAcquired acquireTheGil;

                PyErr_Format(PyExc_ValueError, "Cannot convert float infinity to integer");
                throw PythonExceptionSet();
            }
        }

        Type::TypeCategory cat = targetType->getTypeCategory();

        RegisterTypeProperties::assign((instance_ptr)out, cat, f);

        return true;
    }

    /* convert a double to an int, returning true if conversion is successful.

    This function writes the value into 'out' if successful. if 'canThrow', then
    raise an exception on failure.

    Type* determines what kind of instance we're writing into.
    */
    bool nativepython_float64_to_int(void* out, double f, bool canThrow, Type* targetType) {
        if (!std::isfinite(f)) {
            if (!canThrow) {
                return false;
            }

            if (std::isnan(f)) {
                PyEnsureGilAcquired acquireTheGil;

                PyErr_Format(PyExc_ValueError, "Cannot convert float NaN to integer");
                throw PythonExceptionSet();
            }
            if (std::isinf(f)) {
                PyEnsureGilAcquired acquireTheGil;

                PyErr_Format(PyExc_ValueError, "Cannot convert float infinity to integer");
                throw PythonExceptionSet();
            }
        }

        Type::TypeCategory cat = targetType->getTypeCategory();

        RegisterTypeProperties::assign((instance_ptr)out, cat, f);

        return true;
    }

    StringType::layout* nativepython_int64_to_string(int64_t i) {
        char data[21];

        int64_t count = sprintf((char*)data, "%ld", i);
        return StringType::createFromUtf8(data, count);
    }

    StringType::layout* nativepython_uint64_to_string(uint64_t u) {
        char data[24];

        int64_t count = sprintf((char*)data, "%luu64", u);
        return StringType::createFromUtf8(data, count);
    }


    StringType::layout* nativepython_float64_to_string(double f) {
        char buf[32] = "";
        double a = fabs(f);

        if (a >= 1e16) sprintf(buf, "%.16e", f);
        else if (a >= 1e15 || a == 0.0) sprintf(buf, "%.1f", f);
        else if (a >= 1e14) sprintf(buf, "%.2f", f);
        else if (a >= 1e13) sprintf(buf, "%.3f", f);
        else if (a >= 1e12) sprintf(buf, "%.4f", f);
        else if (a >= 1e11) sprintf(buf, "%.5f", f);
        else if (a >= 1e10) sprintf(buf, "%.6f", f);
        else if (a >= 1e9) sprintf(buf, "%.7f", f);
        else if (a >= 1e8) sprintf(buf, "%.8f", f);
        else if (a >= 1e7) sprintf(buf, "%.9f", f);
        else if (a >= 1e6) sprintf(buf, "%.10f", f);
        else if (a >= 1e5) sprintf(buf, "%.11f", f);
        else if (a >= 1e4) sprintf(buf, "%.12f", f);
        else if (a >= 1e3) sprintf(buf, "%.13f", f);
        else if (a >= 1e2) sprintf(buf, "%.14f", f);
        else if (a >= 10) sprintf(buf, "%.15f", f);
        else if (a >= 1) sprintf(buf, "%.16f", f);
        else if (a >= 0.1) sprintf(buf, "%.17f", f);
        else if (a >= 0.01) sprintf(buf, "%.18f", f);
        else if (a >= 0.001) sprintf(buf, "%.19f", f);
        else if (a >= 0.0001) sprintf(buf, "%.20f", f);
        else sprintf(buf, "%.16e", f);

        remove_trailing_zeros_pystyle(buf);

        return StringType::createFromUtf8(&buf[0], strlen(buf));
    }

    StringType::layout* nativepython_float32_to_string(float f) {
        std::ostringstream s;

        s << f << "f32";

        std::string rep = s.str();

        return StringType::createFromUtf8(&rep[0], rep.size());
    }

    StringType::layout* nativepython_bool_to_string(bool b) {
        if (b)
            return StringType::createFromUtf8("True", 4);
        else
            return StringType::createFromUtf8("False", 5);
    }

    hash_table_layout* nativepython_tableCreate() {
        hash_table_layout* result;

        result = (hash_table_layout*)tp_malloc(sizeof(hash_table_layout));

        new (result) hash_table_layout();

        result->refcount += 1;

        return result;
    }

    int32_t nativepython_tableAllocateNewSlot(hash_table_layout* layout, size_t kvPairSize) {
        return layout->allocateNewSlot(kvPairSize);
    }

    hash_table_layout* nativepython_tableCopy(hash_table_layout* layout, Type* tp) {
        Type* itemType = 0;

        if (tp->getTypeCategory() == Type::TypeCategory::catSet) {
            SetType* setT = (SetType*)tp;
            itemType = setT->keyType();
        }
        else if (tp->getTypeCategory() == Type::TypeCategory::catDict) {
            DictType* dictT = (DictType*)tp;
            itemType = Tuple::Make({dictT->keyType(), dictT->valueType()});
        }
        else {
            PyErr_Format(
                PyExc_TypeError,
                "tableCopy of type '%s' not supported",
                tp->name().c_str()
            );
            return 0;
        }

        return layout->copyTable(
            itemType->bytecount(),
            itemType->isPOD(),
            [&](instance_ptr self, instance_ptr other) {itemType->copy_constructor(self, other);}
        );
    }

    void nativepython_tableResize(hash_table_layout* layout) {
        layout->resizeTable();
    }

    void nativepython_tableCompress(hash_table_layout* layout, size_t kvPairSize) {
        layout->compressItemTable(kvPairSize);
    }

    int32_t nativepython_hash_float32(float val) {
        HashAccumulator acc;

        acc.addRegister(val);

        return acc.get();
    }

    int32_t nativepython_hash_float64(double val) {
        HashAccumulator acc;

        acc.addRegister(val);

        return acc.get();
    }

    int32_t nativepython_hash_int64(int64_t val) {
        HashAccumulator acc;

        acc.addRegister(val);

        return acc.get();
    }

    int32_t nativepython_hash_uint64(uint64_t val) {
        HashAccumulator acc;

        acc.addRegister(val);

        return acc.get();
    }

    int32_t nativepython_hash_string(StringType::layout* s) {
        return StringType::hash_static((instance_ptr)&s);
    }

    int32_t nativepython_hash_bytes(BytesType::layout* s) {
        return BytesType::Make()->hash((instance_ptr)&s);
    }

    int32_t nativepython_hash_alternative(Alternative::layout* s, Alternative* tp) {
        // TODO: assert tp is an Alternative
        //if (tp->getTypeCategory() != Type::TypeCategory::catAlternative)
        //    throw std::logic_error("Called hash_alternative with a non-Alternative type");
        return tp->hash((instance_ptr)&s);
    }

    int32_t nativepython_hash_class(Class::layout* s, Class* tp) {
        // TODO: assert tp is a Class
        //if (tp->getTypeCategory() != Type::TypeCategory::catClass)
        //    throw std::logic_error("Called hash_class with a non-Class type");
        return tp->hash((instance_ptr)&s);
    }

    bool nativepython_isinf_float32(float f) { return std::isinf(f); }

    bool nativepython_isnan_float32(float f) { return std::isnan(f); }

    bool nativepython_isfinite_float32(float f) { return std::isfinite(f); }

    bool nativepython_isinf_float64(double f) { return std::isinf(f); }

    bool nativepython_isnan_float64(double f) { return std::isnan(f); }

    bool nativepython_isfinite_float64(double f) { return std::isfinite(f); }

    double nativepython_runtime_round_float64(double l, int64_t n) {
        double ret;
        int64_t m = 1;
        int64_t d = 1;
        for (int64_t i = 0; i < n; i++)
            m *= 10;
        for (int64_t i = n; i < 0 ; i++)
            d *= 10;
        if (m > 1)
            l *= m;
        else if (d > 1)
            l /= d;
        ret = round(l);
        int64_t isodd = int64_t(ret) % 2;
        if (fabs(l - ret) == 0.5 && isodd)
            ret -= isodd;
        if (m > 1)
            return ret / m;
        else if (d > 1) {
            return ret * d;
        }
        else
            return ret;
    }

    double nativepython_runtime_trunc_float64(double l) {
        return trunc(l);
    }

    double nativepython_runtime_floor_float64(double l) {
        return floor(l);
    }

    double nativepython_runtime_ceil_float64(double l) {
        return ceil(l);
    }

    ListOfType::layout* nativepython_runtime_dir(instance_ptr i, Type* tp) {
        PyEnsureGilAcquired acquireTheGil;

        PyObjectStealer o(PyInstance::extractPythonObject(i, tp));
        PyObjectStealer dir(PyObject_Dir(o));
        ListOfType *retType = ListOfType::Make(StringType::Make());
        ListOfType::layout* ret = 0;

        PyInstance::copyConstructFromPythonInstance(retType, (instance_ptr)&ret, dir, ConversionLevel::New);

        return ret;
    }

    int64_t np_pyobj_pynumber_index(PythonObjectOfType::layout_type* lhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_Index(lhs->pyObj);

        if (!res) {
            throw PythonExceptionSet();
        }

        if (!PyLong_Check(res)) {
            PyErr_Format(PyExc_TypeError, "__index__ returned non-int (type %s)", res->ob_type->tp_name);
            throw PythonExceptionSet();
        }

        return PyLong_AsLong(res);
    }

    PythonObjectOfType::layout_type* np_pyobj_typeof(PythonObjectOfType::layout_type* lhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = incref((PyObject*)lhs->pyObj->ob_type);

        if (!res) {
            throw PythonExceptionSet();
        }

        return PythonObjectOfType::stealToCreateLayout(res);
    }

    PythonObjectOfType::layout_type* np_pyobj_Add(PythonObjectOfType::layout_type* lhs, PythonObjectOfType::layout_type* rhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_Add(lhs->pyObj, rhs->pyObj);

        if (!res) {
            throw PythonExceptionSet();
        }

        return PythonObjectOfType::stealToCreateLayout(res);
    }

    PythonObjectOfType::layout_type* np_pyobj_Subtract(PythonObjectOfType::layout_type* lhs, PythonObjectOfType::layout_type* rhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_Subtract(lhs->pyObj, rhs->pyObj);

        if (!res) {
            throw PythonExceptionSet();
        }

        return PythonObjectOfType::stealToCreateLayout(res);
    }

    PythonObjectOfType::layout_type* np_pyobj_Multiply(PythonObjectOfType::layout_type* lhs, PythonObjectOfType::layout_type* rhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_Multiply(lhs->pyObj, rhs->pyObj);

        if (!res) {
            throw PythonExceptionSet();
        }

        return PythonObjectOfType::stealToCreateLayout(res);
    }

    PythonObjectOfType::layout_type* np_pyobj_Pow(PythonObjectOfType::layout_type* lhs, PythonObjectOfType::layout_type* rhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_Power(lhs->pyObj, rhs->pyObj, Py_None);

        if (!res) {
            throw PythonExceptionSet();
        }

        return PythonObjectOfType::stealToCreateLayout(res);
    }

    PythonObjectOfType::layout_type* np_pyobj_MatrixMultiply(PythonObjectOfType::layout_type* lhs, PythonObjectOfType::layout_type* rhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_MatrixMultiply(lhs->pyObj, rhs->pyObj);

        if (!res) {
            throw PythonExceptionSet();
        }

        return PythonObjectOfType::stealToCreateLayout(res);
    }

    PythonObjectOfType::layout_type* np_pyobj_TrueDivide(PythonObjectOfType::layout_type* lhs, PythonObjectOfType::layout_type* rhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_TrueDivide(lhs->pyObj, rhs->pyObj);

        if (!res) {
            throw PythonExceptionSet();
        }

        return PythonObjectOfType::stealToCreateLayout(res);
    }

    PythonObjectOfType::layout_type* np_pyobj_FloorDivide(PythonObjectOfType::layout_type* lhs, PythonObjectOfType::layout_type* rhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_FloorDivide(lhs->pyObj, rhs->pyObj);

        if (!res) {
            throw PythonExceptionSet();
        }

        return PythonObjectOfType::stealToCreateLayout(res);
    }

    PythonObjectOfType::layout_type* np_pyobj_Remainder(PythonObjectOfType::layout_type* lhs, PythonObjectOfType::layout_type* rhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_Remainder(lhs->pyObj, rhs->pyObj);

        if (!res) {
            throw PythonExceptionSet();
        }

        return PythonObjectOfType::stealToCreateLayout(res);
    }

    PythonObjectOfType::layout_type* np_pyobj_Lshift(PythonObjectOfType::layout_type* lhs, PythonObjectOfType::layout_type* rhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_Lshift(lhs->pyObj, rhs->pyObj);

        if (!res) {
            throw PythonExceptionSet();
        }

        return PythonObjectOfType::stealToCreateLayout(res);
    }

    PythonObjectOfType::layout_type* np_pyobj_Rshift(PythonObjectOfType::layout_type* lhs, PythonObjectOfType::layout_type* rhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_Rshift(lhs->pyObj, rhs->pyObj);

        if (!res) {
            throw PythonExceptionSet();
        }

        return PythonObjectOfType::stealToCreateLayout(res);
    }

    PythonObjectOfType::layout_type* np_pyobj_Or(PythonObjectOfType::layout_type* lhs, PythonObjectOfType::layout_type* rhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_Or(lhs->pyObj, rhs->pyObj);

        if (!res) {
            throw PythonExceptionSet();
        }

        return PythonObjectOfType::stealToCreateLayout(res);
    }

    PythonObjectOfType::layout_type* np_pyobj_Xor(PythonObjectOfType::layout_type* lhs, PythonObjectOfType::layout_type* rhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_Xor(lhs->pyObj, rhs->pyObj);

        if (!res) {
            throw PythonExceptionSet();
        }

        return PythonObjectOfType::stealToCreateLayout(res);
    }

    PythonObjectOfType::layout_type* np_pyobj_And(PythonObjectOfType::layout_type* lhs, PythonObjectOfType::layout_type* rhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_And(lhs->pyObj, rhs->pyObj);

        if (!res) {
            throw PythonExceptionSet();
        }

        return PythonObjectOfType::stealToCreateLayout(res);
    }

    PythonObjectOfType::layout_type* np_pyobj_compare(PythonObjectOfType::layout_type* lhs, PythonObjectOfType::layout_type* rhs, int comparisonOp) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyObject_RichCompare(lhs->pyObj, rhs->pyObj, comparisonOp);

        if (!res) {
            throw PythonExceptionSet();
        }

        return PythonObjectOfType::stealToCreateLayout(res);
    }

    bool np_pyobj_issubclass(PythonObjectOfType::layout_type* subclass, PythonObjectOfType::layout_type* superclass, int comparisonOp) {
        PyEnsureGilAcquired acquireTheGil;

        int res = PyObject_IsSubclass(subclass->pyObj, superclass->pyObj);

        if (res == -1) {
            throw PythonExceptionSet();
        }

        return res;
    }

    PythonObjectOfType::layout_type* np_pyobj_EQ(PythonObjectOfType::layout_type* lhs, PythonObjectOfType::layout_type* rhs) {
        return np_pyobj_compare(lhs, rhs, Py_EQ);
    }

    PythonObjectOfType::layout_type* np_pyobj_NE(PythonObjectOfType::layout_type* lhs, PythonObjectOfType::layout_type* rhs) {
        return np_pyobj_compare(lhs, rhs, Py_NE);
    }

    PythonObjectOfType::layout_type* np_pyobj_LT(PythonObjectOfType::layout_type* lhs, PythonObjectOfType::layout_type* rhs) {
        return np_pyobj_compare(lhs, rhs, Py_LT);
    }

    PythonObjectOfType::layout_type* np_pyobj_GT(PythonObjectOfType::layout_type* lhs, PythonObjectOfType::layout_type* rhs) {
        return np_pyobj_compare(lhs, rhs, Py_GT);
    }

    PythonObjectOfType::layout_type* np_pyobj_LE(PythonObjectOfType::layout_type* lhs, PythonObjectOfType::layout_type* rhs) {
        return np_pyobj_compare(lhs, rhs, Py_LE);
    }

    PythonObjectOfType::layout_type* np_pyobj_GE(PythonObjectOfType::layout_type* lhs, PythonObjectOfType::layout_type* rhs) {
        return np_pyobj_compare(lhs, rhs, Py_GE);
    }

    PythonObjectOfType::layout_type* np_pyobj_In(PythonObjectOfType::layout_type* lhs, PythonObjectOfType::layout_type* rhs) {
        PyEnsureGilAcquired acquireTheGil;

        int res = PySequence_Contains(rhs->pyObj, lhs->pyObj);

        if (res == -1) {
            throw PythonExceptionSet();
        }

        return PythonObjectOfType::createLayout(res == 1 ? Py_True : Py_False);
    }

    PythonObjectOfType::layout_type* np_pyobj_NotIn(PythonObjectOfType::layout_type* lhs, PythonObjectOfType::layout_type* rhs) {
        PyEnsureGilAcquired acquireTheGil;

        int res = PySequence_Contains(rhs->pyObj, lhs->pyObj);

        if (res == -1) {
            throw PythonExceptionSet();
        }

        return PythonObjectOfType::createLayout(res == 1 ? Py_False : Py_True);
    }

    PythonObjectOfType::layout_type* np_pyobj_Invert(PythonObjectOfType::layout_type* lhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_Invert(lhs->pyObj);

        if (!res) {
            throw PythonExceptionSet();
        }

        return PythonObjectOfType::stealToCreateLayout(res);
    }

    PythonObjectOfType::layout_type* np_pyobj_Positive(PythonObjectOfType::layout_type* lhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_Positive(lhs->pyObj);

        if (!res) {
            throw PythonExceptionSet();
        }

        return PythonObjectOfType::stealToCreateLayout(res);
    }

    PythonObjectOfType::layout_type* np_pyobj_Negative(PythonObjectOfType::layout_type* lhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_Negative(lhs->pyObj);

        if (!res) {
            throw PythonExceptionSet();
        }

        return PythonObjectOfType::stealToCreateLayout(res);
    }

    bool np_pyobj_Not(PythonObjectOfType::layout_type* lhs) {
        PyEnsureGilAcquired acquireTheGil;

        int64_t res = PyObject_Not(lhs->pyObj);
        if (res == -1) {
            throw PythonExceptionSet();
        }

        return res;
    }

    // this struct is defined in threadmodule.c - so it's internal to python itself,
    // but because we want to bypass the GIL, we'll just read from the corresponding
    // PyObject* knowing its structure.
    typedef struct {
        PyObject_HEAD
        PyThread_type_lock lock_lock;
        PyObject *in_weakreflist;
        char locked; /* for sanity checking */
    } np_lockobject_equivalent;

    bool np_pyobj_locktype_lock(PythonObjectOfType::layout_type* lockPtr) {
        PyThread_acquire_lock_timed(
            ((np_lockobject_equivalent*)(lockPtr->pyObj))->lock_lock,
            -1,
            0
        );

        ((np_lockobject_equivalent*)(lockPtr->pyObj))->locked = true;

        return true;
    }

    bool np_pyobj_locktype_unlock(PythonObjectOfType::layout_type* lockPtr) {
        // if (!((np_lockobject_equivalent*)(lockPtr->pyObj))->locked) {
        //     PyEnsureGilAcquired getTheGil;
        //     PyErr_SetString(PyExc_RuntimeError, "release unlocked lock");
        //     throw PythonExceptionSet();
        // }

        // reset the sanity check
        ((np_lockobject_equivalent*)(lockPtr->pyObj))->locked = false;

        PyThread_release_lock(
            ((np_lockobject_equivalent*)(lockPtr->pyObj))->lock_lock
        );

        return false; // __exit__ returning false means don't suppress exceptions
    }

    typedef struct {
        PyObject_HEAD
        PyThread_type_lock rlock_lock;
        unsigned long rlock_owner;
        unsigned long rlock_count;
        PyObject *in_weakreflist;
    } np_lockobject_rlockobject;

    bool np_pyobj_rlocktype_lock(PythonObjectOfType::layout_type* lockPtr) {
        np_lockobject_rlockobject* lockObj = (np_lockobject_rlockobject*)(lockPtr->pyObj);

        unsigned long tid = PyThread_get_thread_ident();

        if (lockObj->rlock_count > 0 && tid == lockObj->rlock_owner) {
            unsigned long count = lockObj->rlock_count + 1;
            if (count <= lockObj->rlock_count) {
                PyEnsureGilAcquired getTheGil;

                PyErr_SetString(PyExc_OverflowError,
                                "Internal lock count overflowed");
                throw PythonExceptionSet();
            }

            lockObj->rlock_count = count;

            return true;
        }

        int r = PyThread_acquire_lock_timed(lockObj->rlock_lock, -1, 0);

        if (r == PY_LOCK_ACQUIRED) {
            assert(lockObj->rlock_count == 0);
            lockObj->rlock_owner = tid;
            lockObj->rlock_count = 1;
        }
        else if (r == PY_LOCK_INTR) {
            PyEnsureGilAcquired getTheGil;

            PyErr_SetString(PyExc_RuntimeError,
                            "Expected we'd get the lock but we didn't.");
            throw PythonExceptionSet();
        }

        return true;
    }

    bool np_pyobj_rlocktype_unlock(PythonObjectOfType::layout_type* lockPtr) {
        np_lockobject_rlockobject* lockObj = (np_lockobject_rlockobject*)(lockPtr->pyObj);

        unsigned long tid = PyThread_get_thread_ident();

        if (lockObj->rlock_count == 0 || lockObj->rlock_owner != tid) {
            PyEnsureGilAcquired getTheGil;
            PyErr_SetString(PyExc_RuntimeError,
                            "cannot release un-acquired lock");
            throw PythonExceptionSet();
        }
        if (--lockObj->rlock_count == 0) {
            lockObj->rlock_owner = 0;
            PyThread_release_lock(lockObj->rlock_lock);
        }
        return false; // __exit__ returning false means don't suppress exceptions
    }

    double np_pyobj_ceil(PythonObjectOfType::layout_type* obj) {
        PyEnsureGilAcquired acquireTheGil;

        if (!PyObject_HasAttrString(obj->pyObj, "__ceil__")) {
            double val = PyFloat_AsDouble(obj->pyObj);
            if (val == -1.0 && PyErr_Occurred()) {
                throw PythonExceptionSet();
            }
            return ceil(val);
        }

        PyObjectHolder retObj(PyObject_CallMethod(obj->pyObj, "__ceil__", NULL));
        double ret;

        // This is an additional condition.  Python does not have this condition (__ceil__ does not have to return a number).
        if (PyFloat_Check(retObj)) {
            ret = PyFloat_AsDouble(retObj);
        }
        else if (PyLong_Check(retObj)) {
             ret = PyLong_AsDouble(retObj);
        }
        else {
            PyObjectHolder floatObj(PyObject_CallMethod(retObj, "__float__", NULL));
            if (floatObj) {
                ret = PyFloat_AsDouble(floatObj);
            }
            else {
                PyErr_SetString(PyExc_TypeError, "__ceil__ returned non-number");
                throw PythonExceptionSet();
            }
        }
        if (ret == -1.0 && PyErr_Occurred()) {
            throw PythonExceptionSet();
        }
        return ret;
    }

    double np_pyobj_floor(PythonObjectOfType::layout_type* obj) {
        PyEnsureGilAcquired acquireTheGil;

        if (!PyObject_HasAttrString(obj->pyObj, "__floor__")) {
            double val = PyFloat_AsDouble(obj->pyObj);
            if (val == -1.0 && PyErr_Occurred()) {
                throw PythonExceptionSet();
            }
            return floor(val);
        }

        PyObjectHolder retObj(PyObject_CallMethod(obj->pyObj, "__floor__", NULL));
        double ret;

        // This is an additional condition.  Python does not have this condition (__floor__ does not have to return a number).
        if (PyFloat_Check(retObj)) {
            ret = PyFloat_AsDouble(retObj);
        }
        else if (PyLong_Check(retObj)) {
            ret = PyLong_AsDouble(retObj);
        }
        else {
            PyObjectHolder floatObj(PyObject_CallMethod(retObj, "__float__", NULL));
            if (floatObj) {
                ret = PyFloat_AsDouble(floatObj);
            }
            else {
                PyErr_SetString(PyExc_TypeError, "__floor__ returned non-number");
                throw PythonExceptionSet();
            }
        }
        if (ret == -1.0 && PyErr_Occurred()) {
            throw PythonExceptionSet();
        }
        return ret;
    }

    double np_pyobj_trunc(PythonObjectOfType::layout_type* obj) {
        PyEnsureGilAcquired acquireTheGil;
        if (PyFloat_Check(obj->pyObj)) {
            double val = PyFloat_AsDouble(obj->pyObj);
            if (val == -1.0 && PyErr_Occurred()) {
                throw PythonExceptionSet();
            }
            return trunc(val);
        }

        if (PyLong_Check(obj->pyObj)) {
            long val = PyLong_AsLong(obj->pyObj);
            if (val == -1 && PyErr_Occurred()) {
                throw PythonExceptionSet();
            }
            return double(val);
        }


        if (!PyObject_HasAttrString(obj->pyObj, "__trunc__")) {
            // unlike the others, __trunc__ doesn't devolve to __float__
            PyErr_Format(PyExc_TypeError, "type %s doesn't define __trunc__ method", Py_TYPE(obj->pyObj)->tp_name);
            throw PythonExceptionSet();
        }

        PyObjectHolder retObj(PyObject_CallMethod(obj->pyObj, "__trunc__", NULL));
        double ret;

        // This is an additional condition.  Python does not have this condition (__trunc__ does not have to return a number).
        if (PyFloat_Check(retObj)) {
            ret = PyFloat_AsDouble(retObj);
        }
        else if (PyLong_Check(retObj)) {
             ret = PyLong_AsDouble(retObj);
        }
        else {
            PyObjectHolder floatObj(PyObject_CallMethod(retObj, "__float__", NULL));
            if (floatObj) {
                ret = PyFloat_AsDouble(floatObj);
            }
            else {
                PyErr_SetString(PyExc_TypeError, "__trunc__ returned non-number");
                throw PythonExceptionSet();
            }
        }
        if (ret == -1.0 && PyErr_Occurred()) {
            throw PythonExceptionSet();
        }
        return ret;
    }

    int64_t np_str_to_int64(StringType::layout* s) {
        int64_t ret = 0;
        if (StringType::to_int64(s, &ret)) {
            return ret;
        }
        else {
            PyEnsureGilAcquired getTheGil;
            PyErr_SetString(PyExc_ValueError, "'invalid literal for int() with base 10");
            throw PythonExceptionSet();
        }
    }

    double np_str_to_float64(StringType::layout* s) {
        double ret = 0;
        if (StringType::to_float64(s, &ret)) {
            return ret;
        }
        else {
            PyEnsureGilAcquired getTheGil;
            PyErr_Format(PyExc_ValueError, "could not convert string to float: '%s'",
                StringType::Make()->toUtf8String((instance_ptr)&s).c_str());
            throw PythonExceptionSet();
        }
    }

    int64_t np_bytes_to_int64(BytesType::layout* s) {
        int64_t ret = 0;
        if (BytesType::to_int64(s, &ret)) {
            return ret;
        }
        else {
            PyEnsureGilAcquired getTheGil;
            PyErr_SetString(PyExc_ValueError, "'invalid literal for int() with base 10");
            throw PythonExceptionSet();
        }
    }

    double np_bytes_to_float64(BytesType::layout* s) {
        double ret = 0;
        if (BytesType::to_float64(s, &ret)) {
            return ret;
        }
        else {
            PyEnsureGilAcquired getTheGil;
            std::string asString(s->data, s->data + s->bytecount);
            // To match interpreter behavior, the error string reads 'could not convert string to float'
            // instead of 'could not convert bytes to float'.
            PyErr_Format(PyExc_ValueError, "could not convert string to float: b'%s'", asString.c_str());
            throw PythonExceptionSet();
        }
    }

    double np_pyobj_to_float64(PythonObjectOfType::layout_type* obj) {
    // This conversion matches how the python math module converts arguments to float
        PyEnsureGilAcquired getTheGil;

        double res = PyFloat_AsDouble(obj->pyObj);
        if (res == -1.0 && PyErr_Occurred()) {
            throw PythonExceptionSet();
        }
        return res;
    }

    int64_t np_pyobj_to_int64(PythonObjectOfType::layout_type* obj) {
        PyEnsureGilAcquired getTheGil;

        int64_t res = PyLong_AsLong(obj->pyObj);
        if (res == -1 && PyErr_Occurred()) {
            throw PythonExceptionSet();
        }
        return res;
    }

    BytesType::layout* tp_list_or_tuple_of_to_bytes(TupleOrListOfType::layout* obj, Type* typeObj) {
        if (!typeObj->isTupleOrListOf() || !((TupleOrListOfType*)typeObj)->getEltType()->isPOD()) {
            PyEnsureGilAcquired getTheGil;
            PyErr_Format(PyExc_TypeError, "Expected a POD Tuple or List. Got %s", typeObj->name().c_str());
            throw PythonExceptionSet();
        }

        TupleOrListOfType* lstType = (TupleOrListOfType*)typeObj;

        return BytesType::createFromPtr(
            (const char*)obj->data,
            obj->count * lstType->getEltType()->bytecount()
        );
    }

    TupleOrListOfType::layout* tp_list_or_tuple_of_from_bytes(BytesType::layout* bytes, Type* typeObj) {
        if (!typeObj->isTupleOrListOf() || !((TupleOrListOfType*)typeObj)->getEltType()->isPOD()) {
            PyEnsureGilAcquired getTheGil;
            PyErr_Format(PyExc_TypeError, "Expected a POD Tuple or List. Got %s", typeObj->name().c_str());
            throw PythonExceptionSet();
        }

        TupleOrListOfType* tupT = (TupleOrListOfType*)typeObj;

        size_t bytecount = tupT->getEltType()->bytecount();

        if ((bytecount == 0 && bytes->bytecount) || bytes->bytecount % bytecount) {
            PyErr_Format(PyExc_ValueError, "Byte array must be an integer multiple of underlying type.");
            throw PythonExceptionSet();
        }

        size_t eltCount = (bytecount == 0 ? 0 : bytes->bytecount / bytecount);

        TupleOrListOfType::layout* res;

        tupT->constructor((instance_ptr)&res);
        tupT->reserve((instance_ptr)&res, eltCount);

        memcpy(res->data, bytes->data, bytes->bytecount);

        tupT->setSizeUnsafe((instance_ptr)&res, eltCount);
        return res;
    }

    bool np_pyobj_to_bool(PythonObjectOfType::layout_type* obj) {
        PyEnsureGilAcquired getTheGil;

        bool res = PyObject_IsTrue(obj->pyObj);

        if (PyErr_Occurred()) {
            throw PythonExceptionSet();
        }

        return res;
    }

    /*****
     call 'PyIter_Next', but adapt the result to meet our runtime requirements.

     PyIter_Next returns NULL if there are no remaining items, but can also throw an exception.

     This function retains the convention that upon exhausting the container we return 'nullptr',
     but raises the exception if its present.
    ******/
    PythonObjectOfType::layout_type* np_pyobj_iter_next(PythonObjectOfType::layout_type* toIterate) {
        PyEnsureGilAcquired getTheGil;

        if (!PyIter_Check(toIterate->pyObj)) {
            PyErr_Format(PyExc_TypeError, "iter() returned non-iterator of type '%s'", toIterate->pyObj->ob_type->tp_name);
            throw PythonExceptionSet();
        }

        PyObject* res = PyIter_Next(toIterate->pyObj);

        if (!res) {
            if (PyErr_Occurred()) {
                throw PythonExceptionSet();
            }

            return nullptr;
        }

        return PythonObjectOfType::stealToCreateLayout(res);
    }

    // set the python exception state, but don't actually throw.
    void np_raise_exception_fastpath(const char* message, const char* exceptionTypeName) {
        PyEnsureGilAcquired getTheGil;

        PyObject* module = builtinsModule();

        PyObject* excType = PyObject_GetAttrString(module, exceptionTypeName);
        if (!excType) {
            return;
        }

        PyErr_SetString(excType, message);
    }
}
