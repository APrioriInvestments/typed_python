#include <stdint.h>
#include <cmath>
#include <Python.h>
#include <iostream>
#include "AllTypes.hpp"
#include "StringType.hpp"
#include "BytesType.hpp"
#include "hash_table_layout.hpp"
#include "PyInstance.hpp"

thread_local const char* nativepython_cur_exception_value = nullptr;

const char* nativepython_runtime_get_stashed_exception() {
    return nativepython_cur_exception_value;
}

bool nativepython_runtime_get_stashed_exception_is_python_exception_set() {
    return nativepython_cur_exception_value == (const char*)-1;
}

extern "C" {

    bool nativepython_runtime_string_eq(StringType::layout* lhs, StringType::layout* rhs) {
        if (lhs == rhs) {
            return true;
        }

        return StringType::cmpStaticEq(lhs, rhs);
    }

    void nativepython_runtime_throw_python_exception_set() {
        nativepython_cur_exception_value = (const char*)-1;
        throw PythonExceptionSet();
    }

    int64_t nativepython_runtime_string_cmp(StringType::layout* lhs, StringType::layout* rhs) {
        return StringType::cmpStatic(lhs, rhs);
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

    StringType::layout* nativepython_runtime_string_strip(StringType::layout* l, bool fromLeft, bool fromRight) {
        return StringType::strip(l, fromLeft, fromRight);
    }

    int64_t nativepython_runtime_string_find(StringType::layout* l, StringType::layout* sub, int64_t start, int64_t end) {
        return StringType::find(l, sub, start, end);
    }

    int64_t nativepython_runtime_string_find_2(StringType::layout* l, StringType::layout* sub) {
        return StringType::find(l, sub, 0, l ? l->pointcount : 0);
    }

    int64_t nativepython_runtime_string_find_3(StringType::layout* l, StringType::layout* sub, int64_t start) {
        return StringType::find(l, sub, start, l ? l->pointcount : 0);
    }

    void nativepython_runtime_string_join(StringType::layout** outString, StringType::layout* separator, ListOfType::layout* toJoin) {
        StringType::join(outString, separator, toJoin);
    }

    void nativepython_runtime_string_split(ListOfType::layout* outList, StringType::layout* l, StringType::layout* sep, int64_t max) {
        StringType::split(outList, l, sep, max);
    }

    void nativepython_runtime_string_split_2(ListOfType::layout* outList, StringType::layout* l) {
        StringType::split_3(outList, l, -1);
    }

    void nativepython_runtime_string_split_3(ListOfType::layout* outList, StringType::layout* l, StringType::layout* sep) {
        StringType::split(outList, l, sep, -1);
    }

    void nativepython_runtime_string_split_3max(ListOfType::layout* outList, StringType::layout* l, int64_t max) {
        StringType::split_3(outList, l, max);
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

    StringType::layout* nativepython_runtime_string_from_utf8_and_len(const char* utf8_str, int64_t len) {
        return StringType::createFromUtf8(utf8_str, len);
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

    //a temporary kluge to allow us to communicate between exception throw sites and
    //the native-code invoker until we have a more complete exception model built out.
    void nativepython_runtime_stash_const_char_ptr_for_exception(const char* m) {
        nativepython_cur_exception_value = m;
    }

    void nativepython_runtime_incref_pyobj(PyObject* p) {
        PyEnsureGilAcquired getTheGil;

        incref(p);
    }

    PyObject* nativepython_runtime_get_pyobj_None() {
        return Py_None;
    }

    StringType::layout* nativepython_runtime_repr(instance_ptr inst, Type* tp) {
        PyEnsureGilAcquired getTheGil;

        PyObject* o = PyInstance::extractPythonObject(inst, tp);
        if (!o) {
            PyErr_PrintEx(0);
            throw std::runtime_error("failed to extract python object");
        }
        PyObject *r = PyObject_Repr(o);
        if (!r) {
            PyErr_PrintEx(0);
            throw std::runtime_error("PyObject_Repr returned 0");
        }
        Py_ssize_t s;
        const char* c = PyUnicode_AsUTF8AndSize(r, &s);
        return StringType::createFromUtf8(c, s);
    }

    StringType::layout* nativepython_runtime_str(instance_ptr inst, Type* tp) {
        PyEnsureGilAcquired getTheGil;

        PyObject* o = PyInstance::extractPythonObject(inst, tp);
        if (!o) {
            PyErr_PrintEx(0);
            throw std::runtime_error("failed to extract python object");
        }
        PyObject *r = PyObject_Str(o);
        if (!r) {
            PyErr_PrintEx(0);
            throw std::runtime_error("PyObject_Str returned 0");
        }
        Py_ssize_t s;
        const char* c = PyUnicode_AsUTF8AndSize(r, &s);
        return StringType::createFromUtf8(c, s);
    }

    PyObject* nativepython_runtime_getattr_pyobj(PyObject* p, const char* a) {
        PyEnsureGilAcquired getTheGil;

        PyObject* res = PyObject_GetAttrString(p, a);

        if (!res) {
            nativepython_runtime_throw_python_exception_set();
        }

        return res;
    }

    PyObject* nativepython_runtime_getitem_pyobj(PyObject* p, PyObject* a) {
        PyEnsureGilAcquired getTheGil;

        PyObject* res = PyObject_GetItem(p, a);

        if (!res) {
            nativepython_runtime_throw_python_exception_set();
        }

        return res;
    }

    void nativepython_runtime_setattr_pyobj(PyObject* p, const char* a, PyObject* val) {
        PyEnsureGilAcquired getTheGil;

        int res = PyObject_SetAttrString(p, a, val);

        if (res) {
            nativepython_runtime_throw_python_exception_set();
        }
    }

    void nativepython_runtime_decref_pyobj(PyObject* p) {
        PyEnsureGilAcquired getTheGil;

        decref(p);
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
        if (l == 0.0) {
            return 0.0;
        }
        if (r == 0.0) {
            throw std::runtime_error("mod by 0.0");
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
            throw std::runtime_error("0**-x err");
        double result = std::pow(l, r);
        if (l < 0.0 && r > 0.0 && nativepython_runtime_mod_float64_float64(r, 2.0) == 1.0 && result > 0.0)
            return -result;
        return result;
    }

    double nativepython_runtime_pow_int64_int64(int64_t l, int64_t r) {
        if (l == 0 && r < 0)
            throw std::runtime_error("0**-x err");
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
            throw std::runtime_error("negative shift count");
        }
        if ((l == 0 && r > SSIZE_MAX) || (l != 0 && r >= 1024)) { // 1024 is arbitrary
            throw std::runtime_error("shift count too large");
        }
        return (l >= 0) ? l << r : -((-l) << r);
    }

    // should match corresponding function in PyRegisterTypeInstance.hpp
    uint64_t nativepython_runtime_lshift_uint64_uint64(uint64_t l, uint64_t r) {
        if ((l == 0 && r > SSIZE_MAX) || (l != 0 && r >= 1024)) { // 1024 is arbitrary
            throw std::runtime_error("shift count too large");
        }
        return l << r;
    }

    // should match corresponding function in PyRegisterTypeInstance.hpp
    uint64_t nativepython_runtime_rshift_uint64_uint64(uint64_t l, uint64_t r) {
        if (r > SSIZE_MAX) {
            throw std::runtime_error("shift count too large");
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
            throw std::runtime_error("negative shift count");
        }
        if (r > SSIZE_MAX) {
            throw std::runtime_error("shift count too large");
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
            throw std::runtime_error("floordiv by 0");
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
            throw std::runtime_error("floordiv by 0.0");
        }
        double result = (l - nativepython_runtime_mod_float64_float64(l, r))/r;
        double floorresult = std::floor(result);
        if (result - floorresult > 0.5)
            floorresult += 1.0;
        return floorresult;
    }

    // attempt to initialize 'tgt' of type 'tp' with data from 'obj'. Returns true if we
    // are able to do so.
    bool np_runtime_pyobj_to_typed(PyObject *obj, instance_ptr tgt, Type* tp, bool isExplicit) {
        PyEnsureGilAcquired acquireTheGil;
        try {
            if (!PyInstance::pyValCouldBeOfType(tp, obj,  isExplicit)) {
                return false;
            }

            PyInstance::copyConstructFromPythonInstance(tp, tgt, obj, isExplicit);

            return true;
        } catch(PythonExceptionSet&) {
            PyErr_Clear();
            return false;
        } catch(...) {
            return false;
        }
    }

    PyObject* np_runtime_to_pyobj(instance_ptr obj, Type* tp) {
        PyEnsureGilAcquired acquireTheGil;
        return PyInstance::extractPythonObject(obj, tp);
    }

    PyObject* np_runtime_int64_to_pyobj(int64_t i) {
        return PyLong_FromLongLong(i);
    }

    PyObject* np_runtime_int32_to_pyobj(int32_t i) {
        return PyLong_FromLong(i);
    }

    PyObject* np_runtime_int16_to_pyobj(int16_t i) {
        return PyLong_FromLong(i);
    }

    PyObject* np_runtime_int8_to_pyobj(int8_t i) {
        return PyLong_FromLong(i);
    }

    PyObject* np_runtime_uint64_to_pyobj(uint64_t u) {
        return PyLong_FromUnsignedLongLong(u);
    }

    PyObject* np_runtime_uint32_to_pyobj(uint32_t u) {
        return PyLong_FromUnsignedLong(u);
    }

    PyObject* np_runtime_uint16_to_pyobj(uint16_t u) {
        return PyLong_FromUnsignedLong(u);
    }

    PyObject* np_runtime_uint8_to_pyobj(uint8_t u) {
        return PyLong_FromUnsignedLong(u);
    }

    PyObject* np_runtime_float64_to_pyobj(double f) {
        return PyFloat_FromDouble(f);
    }

    PyObject* np_runtime_float32_to_pyobj(float f) {
        return PyFloat_FromDouble((double)f);
    }

    // C identifiers can ignore character 32 and onward, so shorten prefix to just "np_"
    int64_t np_runtime_pyobj_to_int64(PyObject* i) {
        if (PyLong_Check(i)) {
            return PyLong_AsLong(i);
        }

        throw std::runtime_error("Couldn't convert to int64.");
    }

    int32_t np_runtime_pyobj_to_int32(PyObject* i) {
        if (PyLong_Check(i)) {
            return PyLong_AsLong(i);
        }

        throw std::runtime_error("Couldn't convert to int32.");
    }

    int16_t np_runtime_pyobj_to_int16(PyObject* i) {
        if (PyLong_Check(i)) {
            return PyLong_AsLong(i);
        }

        throw std::runtime_error("Couldn't convert to int16.");
    }

    int8_t np_runtime_pyobj_to_int8(PyObject* i) {
        if (PyLong_Check(i)) {
            return PyLong_AsLong(i);
        }

        throw std::runtime_error("Couldn't convert to int8.");
    }

    uint64_t np_runtime_pyobj_to_uint64(PyObject* i) {
        if (PyLong_Check(i)) {
            return PyLong_AsUnsignedLong(i);
        }

        throw std::runtime_error("Couldn't convert to uint64.");
    }

    uint32_t np_runtime_pyobj_to_uint32(PyObject* i) {
        if (PyLong_Check(i)) {
            return PyLong_AsUnsignedLong(i);
        }

        throw std::runtime_error("Couldn't convert to uint32.");
    }

    uint16_t np_runtime_pyobj_to_uint16(PyObject* i) {
        if (PyLong_Check(i)) {
            return PyLong_AsUnsignedLong(i);
        }

        throw std::runtime_error("Couldn't convert to uint16.");
    }

    uint8_t np_runtime_pyobj_to_uint8(PyObject* i) {
        if (PyLong_Check(i)) {
            return PyLong_AsUnsignedLong(i);
        }

        throw std::runtime_error("Couldn't convert to uint8.");
    }

    double np_runtime_pyobj_to_float64(PyObject* o) {
        if (PyFloat_Check(o)) {
            return PyFloat_AsDouble(o);
        }

        throw std::runtime_error("Couldn't convert to float64.");
    }

    float np_runtime_pyobj_to_float32(PyObject* o) {
        if (PyFloat_Check(o)) {
            return (float)PyFloat_AsDouble(o);
        }

        throw std::runtime_error("Couldn't convert to float32.");
    }

    void nativepython_print_int64(int64_t t) {
        std::cout << "function pointer " << t << std::endl;
    }

    void nativepython_print_string(StringType::layout* layout) {
        std::cout << StringType::Make()->toUtf8String((instance_ptr)&layout) << std::flush;
    }

    StringType::layout* nativepython_int64_to_string(int64_t i) {
        char data[20];

        int64_t count = sprintf((char*)data, "%ld", i);

        return StringType::createFromUtf8(data, count);
    }

    StringType::layout* nativepython_float64_to_string(double i) {
        std::ostringstream s;
        ReprAccumulator acc(s);
        acc << i;

        std::string rep = s.str();
        return StringType::createFromUtf8(&rep[0], rep.size());
    }

    StringType::layout* nativepython_float32_to_string(float i) {
        std::ostringstream s;

        s << i << "f32";

        std::string rep = s.str();

        return StringType::createFromUtf8(&rep[0], rep.size());
    }

    hash_table_layout* nativepython_dict_create() {
        hash_table_layout* result;

        result = (hash_table_layout*)malloc(sizeof(hash_table_layout));

        new (result) hash_table_layout();

        result->refcount += 1;

        return result;
    }

    int32_t nativepython_dict_allocateNewSlot(hash_table_layout* layout, size_t kvPairSize) {
        return layout->allocateNewSlot(kvPairSize);
    }

    void nativepython_dict_resizeTable(hash_table_layout* layout) {
        layout->resizeTable();
    }

    void nativepython_dict_compressItemTable(hash_table_layout* layout, size_t kvPairSize) {
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

    bool nativepython_isinf_float32(float f) { return std::isinf(f); }

    bool nativepython_isnan_float32(float f) { return std::isnan(f); }

    bool nativepython_isfinite_float32(float f) { return std::isfinite(f); }

    bool nativepython_isinf_float64(double f) { return std::isinf(f); }

    bool nativepython_isnan_float64(double f) { return std::isnan(f); }

    bool nativepython_isfinite_float64(double f) { return std::isfinite(f); }
}

