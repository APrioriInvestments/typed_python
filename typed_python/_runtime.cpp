#include <stdint.h>
#include <stdarg.h>
#include <cmath>
#include <Python.h>
#include <iostream>
#include <iomanip>
#include "AllTypes.hpp"
#include "StringType.hpp"
#include "BytesType.hpp"
#include "hash_table_layout.hpp"
#include "PyInstance.hpp"

// Note: extern C identifiers are distinguished only up to 32 characters
// nativepython_runtime_12345678901
extern "C" {

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

    void np_initialize_exception(PyObject* o) {
        PyEnsureGilAcquired getTheGil;

        PyTypeObject* tp = o->ob_type;
        bool hasBaseE = false;

        while (tp) {
            if (tp == (PyTypeObject*)PyExc_BaseException) {
                hasBaseE = true;
            }
            tp = tp->tp_base;
        }

        if (!hasBaseE) {
            PyErr_Format(PyExc_TypeError, "exceptions must derive from BaseException, not %S", (PyObject*)o->ob_type);
            return;
        }

        PyErr_Restore((PyObject*)incref(o->ob_type), incref(o), nullptr);
    }

    void np_add_traceback(const char* funcname, const char* filename, int lineno) {
        PyEnsureGilAcquired getTheGil;

        _PyTraceback_Add(funcname, filename, lineno);
    }

    PyObject* np_builtin_pyobj_by_name(const char* utf8_name) {
        PyEnsureGilAcquired getTheGil;

        static PyObject* module = PyImport_ImportModule("builtins");

        return PyObject_GetAttrString(module, utf8_name);
    }

    void nativepython_runtime_incref_pyobj(PyObject* p) {
        PyEnsureGilAcquired getTheGil;

        incref(p);
    }

    PyObject* nativepython_runtime_get_pyobj_None() {
        PyEnsureGilAcquired acquireTheGil;

        return incref(Py_None);
    }

    StringType::layout* nativepython_runtime_repr(instance_ptr inst, Type* tp) {
        PyEnsureGilAcquired getTheGil;

        PyObjectStealer o(PyInstance::extractPythonObject(inst, tp));
        if (!o) {
            PyErr_PrintEx(0);
            throw std::runtime_error("nativepython_runtime_repr: failed to extract python object");
        }
        PyObject *r = PyObject_Repr(o);
        if (!r) {
            PyErr_PrintEx(0);
            throw std::runtime_error("nativepython_runtime_repr: PyObject_Repr returned 0");
        }
        Py_ssize_t s;
        const char* c = PyUnicode_AsUTF8AndSize(r, &s);
        StringType::layout *ret = StringType::createFromUtf8(c, s);
        decref(r);
        return ret;
    }

    StringType::layout* nativepython_runtime_str(instance_ptr inst, Type* tp) {
        PyEnsureGilAcquired getTheGil;

        PyObjectStealer o(PyInstance::extractPythonObject(inst, tp));
        if (!o) {
            PyErr_PrintEx(0);
            throw std::runtime_error("nativepython_runtime_str: failed to extract python object");
        }
        PyObject *r = PyObject_Str(o);
        if (!r) {
            PyErr_PrintEx(0);
            throw std::runtime_error("nativepython_runtime_str: PyObject_Str returned 0");
        }
        Py_ssize_t s;
        const char* c = PyUnicode_AsUTF8AndSize(r, &s);
        StringType::layout* ret =StringType::createFromUtf8(c, s);
        decref(r);
        return ret;
    }

    uint64_t nativepython_runtime_len(instance_ptr inst, Type* tp) {
        PyEnsureGilAcquired getTheGil;

        PyObjectStealer o(PyInstance::extractPythonObject(inst, tp));
        if (!o) {
            PyErr_PrintEx(0);
            throw std::runtime_error("nativepython_runtime_len: failed to extract python object");
        }
        Py_ssize_t len = PyObject_Length(o);
        if (len == -1) {
            PyErr_PrintEx(0);
            throw std::runtime_error("nativepython_runtime_str: PyObject_Length returned -1");
        }
        return len;
    }

    uint64_t nativepython_pyobj_len(PyObject* pyobj) {
        PyEnsureGilAcquired getTheGil;

        return PyObject_Length(pyobj);
    }

    bool nativepython_runtime_contains(instance_ptr self, Type* self_tp, instance_ptr item, Type* item_tp) {
        PyEnsureGilAcquired getTheGil;

        PyObjectStealer o(PyInstance::extractPythonObject(self, self_tp));
        if (!o) {
            PyErr_PrintEx(0);
            throw std::runtime_error("nativepython_runtime_contains: failed to extract python object");
        }
        PyObjectStealer i(PyInstance::extractPythonObject(item, item_tp));
        if (!i) {
            PyErr_PrintEx(0);
            throw std::runtime_error("nativepython_runtime_contains: failed to extract item python object");
        }
        PyObject* contains = PyObject_GetAttrString(o, "__contains__");
        if (!contains) {
            return 0;
        }
        int ret = PyObject_IsTrue(contains);
        decref(contains);
        return ret;
    }

    PyObject* nativepython_runtime_call_pyobj(PyObject* toCall, int argCount, int kwargCount, ...) {
        PyEnsureGilAcquired getTheGil;

        // each of 'argCount' arguments is a PyObject* followed by a const char*
        va_list va_args;
        va_start(va_args, kwargCount);

        PyObjectStealer args(PyTuple_New(argCount));
        PyObjectStealer kwargs(PyDict_New());

        for (int i = 0; i < argCount; ++i) {
            PyTuple_SetItem((PyObject*)args, i, incref(va_arg(va_args, PyObject*)));
        }

        for (int i = 0; i < kwargCount; ++i) {
            PyObject* kwargVal = va_arg(va_args, PyObject*);
            const char* kwargName = va_arg(va_args, const char*);

            PyDict_SetItemString((PyObject*)kwargs, kwargName, kwargVal);
        }

        va_end(va_args);

        PyObject* res = PyObject_Call(toCall, args, kwargs);

        if (!res) {
            throw PythonExceptionSet();
        }

        return res;
    }

    PyObject* nativepython_runtime_getattr_pyobj(PyObject* p, const char* a) {
        PyEnsureGilAcquired getTheGil;

        PyObject* res = PyObject_GetAttrString(p, a);

        if (!res) {
            throw PythonExceptionSet();
        }

        return res;
    }

    PyObject* nativepython_runtime_getitem_pyobj(PyObject* p, PyObject* a) {
        PyEnsureGilAcquired getTheGil;

        PyObject* res = PyObject_GetItem(p, a);

        if (!res) {
            throw PythonExceptionSet();
        }

        return res;
    }

    void nativepython_runtime_delitem_pyobj(PyObject* p, PyObject* a) {
        PyEnsureGilAcquired getTheGil;

        int success = PyObject_DelItem(p, a);

        if (success != 0) {
            throw PythonExceptionSet();
        }
    }

    void nativepython_runtime_setitem_pyobj(PyObject* p, PyObject* index, PyObject* value) {
        PyEnsureGilAcquired getTheGil;

        int res = PyObject_SetItem(p, index, value);

        if (res) {
            throw PythonExceptionSet();
        }
    }

    void nativepython_runtime_setattr_pyobj(PyObject* p, const char* a, PyObject* val) {
        PyEnsureGilAcquired getTheGil;

        int res = PyObject_SetAttrString(p, a, val);

        if (res) {
            throw PythonExceptionSet();
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

    bool np_runtime_instance_to_bool(instance_ptr i, Type* tp) {
        PyEnsureGilAcquired getTheGil;

        PyObjectStealer o(PyInstance::extractPythonObject(i, tp));

        int r = PyObject_IsTrue(o);

        if (r == -1) {
            throw PythonExceptionSet();
        }

        return r;
    }

    PyObject* np_runtime_to_pyobj(instance_ptr obj, Type* tp) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyInstance::extractPythonObject(obj, tp);

        if (!res) {
            throw PythonExceptionSet();
        }

        return res;
    }

    void nativepython_print_string(StringType::layout* layout) {
        std::cout << StringType::Make()->toUtf8String((instance_ptr)&layout) << std::flush;
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

    int32_t nativepython_hash_alternative(Alternative::layout* s, Alternative* tp) {
        if (tp->getTypeCategory() != Type::TypeCategory::catAlternative)
            throw std::logic_error("Called hash_alternative with a non-Alternative type");

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

    PyObject* nativepython_runtime_complex(double r, double i) {
        PyEnsureGilAcquired acquireTheGil;

        return PyComplex_FromDoubles(r, i);
    }

    ListOfType::layout* nativepython_runtime_dir(instance_ptr i, Type* tp) {
        PyEnsureGilAcquired acquireTheGil;

        PyObjectStealer o(PyInstance::extractPythonObject(i, tp));
        PyObjectStealer dir(PyObject_Dir(o));
        ListOfType *retType = ListOfType::Make(StringType::Make());
        ListOfType::layout* ret = 0;

        PyInstance::copyConstructFromPythonInstance(retType, (instance_ptr)&ret, dir, true);

        return ret;
    }

    PyObject* np_pyobj_Add(PyObject* lhs, PyObject* rhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_Add(lhs, rhs);

        if (!res) {
            throw PythonExceptionSet();
        }

        return res;
    }

    PyObject* np_pyobj_Subtract(PyObject* lhs, PyObject* rhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_Subtract(lhs, rhs);

        if (!res) {
            throw PythonExceptionSet();
        }

        return res;
    }

    PyObject* np_pyobj_Multiply(PyObject* lhs, PyObject* rhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_Multiply(lhs, rhs);

        if (!res) {
            throw PythonExceptionSet();
        }

        return res;
    }

    PyObject* np_pyobj_Pow(PyObject* lhs, PyObject* rhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_Power(lhs, rhs, Py_None);

        if (!res) {
            throw PythonExceptionSet();
        }

        return res;
    }

    PyObject* np_pyobj_MatrixMultiply(PyObject* lhs, PyObject* rhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_MatrixMultiply(lhs, rhs);

        if (!res) {
            throw PythonExceptionSet();
        }

        return res;
    }

    PyObject* np_pyobj_TrueDivide(PyObject* lhs, PyObject* rhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_TrueDivide(lhs, rhs);

        if (!res) {
            throw PythonExceptionSet();
        }

        return res;
    }

    PyObject* np_pyobj_FloorDivide(PyObject* lhs, PyObject* rhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_FloorDivide(lhs, rhs);

        if (!res) {
            throw PythonExceptionSet();
        }

        return res;
    }

    PyObject* np_pyobj_Remainder(PyObject* lhs, PyObject* rhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_Remainder(lhs, rhs);

        if (!res) {
            throw PythonExceptionSet();
        }

        return res;
    }

    PyObject* np_pyobj_Lshift(PyObject* lhs, PyObject* rhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_Lshift(lhs, rhs);

        if (!res) {
            throw PythonExceptionSet();
        }

        return res;
    }

    PyObject* np_pyobj_Rshift(PyObject* lhs, PyObject* rhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_Rshift(lhs, rhs);

        if (!res) {
            throw PythonExceptionSet();
        }

        return res;
    }

    PyObject* np_pyobj_Or(PyObject* lhs, PyObject* rhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_Or(lhs, rhs);

        if (!res) {
            throw PythonExceptionSet();
        }

        return res;
    }

    PyObject* np_pyobj_Xor(PyObject* lhs, PyObject* rhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_Xor(lhs, rhs);

        if (!res) {
            throw PythonExceptionSet();
        }

        return res;
    }

    PyObject* np_pyobj_And(PyObject* lhs, PyObject* rhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_And(lhs, rhs);

        if (!res) {
            throw PythonExceptionSet();
        }

        return res;
    }

    PyObject* np_pyobj_Invert(PyObject* lhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_Invert(lhs);

        if (!res) {
            throw PythonExceptionSet();
        }

        return res;
    }

    PyObject* np_pyobj_Positive(PyObject* lhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_Positive(lhs);

        if (!res) {
            throw PythonExceptionSet();
        }

        return res;
    }

    PyObject* np_pyobj_Negative(PyObject* lhs) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyNumber_Negative(lhs);

        if (!res) {
            throw PythonExceptionSet();
        }

        return res;
    }

    bool np_pyobj_Not(PyObject* lhs) {
        PyEnsureGilAcquired acquireTheGil;

        int64_t res = PyObject_Not(lhs);
        if (res == -1) {
            throw PythonExceptionSet();
        }

        return res;
    }

}

