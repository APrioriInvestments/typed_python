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

#include <pythread.h>

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

    PythonObjectOfType::layout_type* nativepython_runtime_create_pyobj(PyObject* p) {
        PyEnsureGilAcquired getTheGil;
        return PythonObjectOfType::createLayout(p);
    }

    void np_initialize_exception(PythonObjectOfType::layout_type* layout) {
        PyEnsureGilAcquired getTheGil;

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

        PyErr_Restore((PyObject*)incref(layout->pyObj->ob_type), incref(layout->pyObj), nullptr);
    }

    void np_add_traceback(const char* funcname, const char* filename, int lineno) {
        PyEnsureGilAcquired getTheGil;

        _PyTraceback_Add(funcname, filename, lineno);
    }

    PythonObjectOfType::layout_type* np_builtin_pyobj_by_name(const char* utf8_name) {
        PyEnsureGilAcquired getTheGil;

        static PyObject* module = PyImport_ImportModule("builtins");

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

    uint64_t nativepython_pyobj_len(PythonObjectOfType::layout_type* layout) {
        PyEnsureGilAcquired getTheGil;

        return PyObject_Length(layout->pyObj);
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
        PythonObjectOfType::decrefLayoutWithoutHoldingTheGil(p);
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

        if (PY_MINOR_VERSION > 6 && l == 0) {
            return 0;
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
    bool np_runtime_pyobj_to_typed(PythonObjectOfType::layout_type *layout, instance_ptr tgt, Type* tp, bool isExplicit) {
        PyEnsureGilAcquired acquireTheGil;

        try {
            if (!PyInstance::pyValCouldBeOfType(tp, layout->pyObj,  isExplicit)) {
                return false;
            }

            PyInstance::copyConstructFromPythonInstance(tp, tgt, layout->pyObj, isExplicit);

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

    PythonObjectOfType::layout_type* np_runtime_to_pyobj(instance_ptr obj, Type* tp) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyInstance::extractPythonObject(obj, tp);

        if (!res) {
            throw PythonExceptionSet();
        }

        return PythonObjectOfType::stealToCreateLayout(res);
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

        PyInstance::copyConstructFromPythonInstance(retType, (instance_ptr)&ret, dir, true);

        return ret;
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

    bool np_pyobj_compare(PythonObjectOfType::layout_type* lhs, PythonObjectOfType::layout_type* rhs, int comparisonOp) {
        PyEnsureGilAcquired acquireTheGil;

        PyObject* res = PyObject_RichCompare(lhs->pyObj, rhs->pyObj, comparisonOp);

        if (!res) {
            throw PythonExceptionSet();
        }

        return PythonObjectOfType::stealToCreateLayout(res);
    }

    bool np_pyobj_EQ(PythonObjectOfType::layout_type* lhs, PythonObjectOfType::layout_type* rhs) {
        return np_pyobj_compare(lhs, rhs, Py_EQ);
    }

    bool np_pyobj_NE(PythonObjectOfType::layout_type* lhs, PythonObjectOfType::layout_type* rhs) {
        return np_pyobj_compare(lhs, rhs, Py_NE);
    }

    bool np_pyobj_LT(PythonObjectOfType::layout_type* lhs, PythonObjectOfType::layout_type* rhs) {
        return np_pyobj_compare(lhs, rhs, Py_LT);
    }

    bool np_pyobj_GT(PythonObjectOfType::layout_type* lhs, PythonObjectOfType::layout_type* rhs) {
        return np_pyobj_compare(lhs, rhs, Py_GT);
    }

    bool np_pyobj_LE(PythonObjectOfType::layout_type* lhs, PythonObjectOfType::layout_type* rhs) {
        return np_pyobj_compare(lhs, rhs, Py_LE);
    }

    bool np_pyobj_GE(PythonObjectOfType::layout_type* lhs, PythonObjectOfType::layout_type* rhs) {
        return np_pyobj_compare(lhs, rhs, Py_GE);
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

        return true;
    }

    void np_pyobj_locktype_unlock(PythonObjectOfType::layout_type* lockPtr) {
        PyThread_release_lock(
            ((np_lockobject_equivalent*)(lockPtr->pyObj))->lock_lock
        );
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

    void np_pyobj_rlocktype_unlock(PythonObjectOfType::layout_type* lockPtr) {
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
    }

    int64_t np_str_to_int64(StringType::layout* s) {
        PyEnsureGilAcquired getTheGil; // since to_int64 can raise ValueErr

        return StringType::to_int64(s);
    }

    double np_str_to_float64(StringType::layout* s) {
        PyEnsureGilAcquired getTheGil;

        PyObject* str_obj = PyInstance::extractPythonObject((instance_ptr)&s, StringType::Make());
        if (!str_obj) {
            throw PythonExceptionSet();
        }
        PyObject *float_obj = PyFloat_FromString(str_obj);
        decref(str_obj);
        if (!float_obj) {
            throw PythonExceptionSet();
        }
        double ret = PyFloat_AsDouble(float_obj);
        decref(float_obj);
        return ret;
    }

    int64_t np_bytes_to_int64(BytesType::layout* l) {
        PyEnsureGilAcquired getTheGil;

        PyObject* str_obj = PyUnicode_FromKindAndData(PyUnicode_1BYTE_KIND, l->data, l->bytecount);
        if (!str_obj) {
            throw PythonExceptionSet();
        }
        PyObject *long_obj = PyLong_FromUnicodeObject(str_obj, 10);
        decref(str_obj);
        if (!long_obj) {
            throw PythonExceptionSet();
        }
        long ret = PyLong_AsLong(long_obj);
        decref(long_obj);
        return ret;
    }

    double np_bytes_to_float64(BytesType::layout* l) {
        PyEnsureGilAcquired getTheGil;

        PyObject* str_obj = PyUnicode_FromKindAndData(PyUnicode_1BYTE_KIND, l->data, l->bytecount);
        if (!str_obj) {
            throw PythonExceptionSet();
        }
        PyObject *float_obj = PyFloat_FromString(str_obj);
        decref(str_obj);
        if (!float_obj) {
            throw PythonExceptionSet();
        }
        double ret = PyFloat_AsDouble(float_obj);
        decref(float_obj);
        return ret;
    }
}
