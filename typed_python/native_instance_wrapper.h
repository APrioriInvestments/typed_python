#pragma once

#include "util.hpp"

//extension of PyTypeObject that adds a Type* at the end.
struct NativeTypeWrapper {
    PyTypeObject typeObj;
    Type* mType;
};

class InternalPyException {};

//throw to indicate we set a python error already.
class PythonExceptionSet {};

struct native_instance_wrapper {
    PyObject_HEAD

    bool mIsInitialized;
    bool mIsMatcher;
    char mIteratorFlag; //0 is keys, 1 is values, 2 is pairs
    int64_t mIteratorOffset; //-1 if we're not an iterator

    Instance mContainingInstance;

    static bool guaranteeForwardsResolved(Type* t);

    static void guaranteeForwardsResolvedOrThrow(Type* t);

    //return the standard python representation of an object of type 'eltType'
    template<class init_func>
    static PyObject* initializePythonRepresentation(Type* eltType, const init_func& f) {
        if (!guaranteeForwardsResolved(eltType)) {
            return nullptr;
        }

        Instance instance(eltType, f);

        return extractPythonObject(instance.data(), instance.type());
    }

    //initialize a native_instance_wrapper for 'eltType'. For ints, floats, etc, with
    //actual native representations, this will produce a wrapper object (maybe not what you want)
    //rather than the standard python representation.
    template<class init_func>
    static PyObject* initialize(Type* eltType, const init_func& f) {
        if (!guaranteeForwardsResolved(eltType)) {
            return nullptr;
        }

        native_instance_wrapper* self =
            (native_instance_wrapper*)typeObj(eltType)->tp_alloc(typeObj(eltType), 0);

        try {
            self->initialize(f);

            return (PyObject*)self;
        } catch(...) {
            typeObj(eltType)->tp_dealloc((PyObject*)self);
            throw;
        }
    }

    template<class init_func>
    void initialize(const init_func& i) {
        Type* type = extractTypeFrom(((PyObject*)this)->ob_type);
        guaranteeForwardsResolvedOrThrow(type);

        mIsInitialized = false;
        new (&mContainingInstance) Instance( type, i );
        mIsInitialized = true;
    }

    instance_ptr dataPtr();

    static PyObject* constDictItems(PyObject *o);

    static PyObject* constDictKeys(PyObject *o);

    static PyObject* constDictValues(PyObject *o);

    static PyObject* constDictGet(PyObject* o, PyObject* args);

    static PyMethodDef* typeMethods(Type* t);

    static void tp_dealloc(PyObject* self);

    static bool pyValCouldBeOfType(Type* t, PyObject* pyRepresentation);

    static void copy_constructor(Type* eltType, instance_ptr tgt, PyObject* pyRepresentation);

    static void initializeClassWithDefaultArguments(Class* cls, uint8_t* data, PyObject* args, PyObject* kwargs);

    static void constructFromPythonArguments(uint8_t* data, Type* t, PyObject* args, PyObject* kwargs);

    //produce the pythonic representation of this object. for things like integers, string, etc,
    //convert them back to their python-native form. otherwise, a pointer back into a native python
    //structure
    static PyObject* extractPythonObject(instance_ptr data, Type* eltType);

    static PyObject *tp_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds);

    static Py_ssize_t sq_length(PyObject* o);

    static PyObject* nb_rshift(PyObject* lhs, PyObject* rhs);

    static std::pair<bool, PyObject*> checkForPyOperator(PyObject* lhs, PyObject* rhs, const char* op);

    static PyObject* nb_add(PyObject* lhs, PyObject* rhs);

    static PyObject* nb_subtract(PyObject* lhs, PyObject* rhs);

    static PyObject* sq_concat(PyObject* lhs, PyObject* rhs);

    static PyObject* sq_item(PyObject* o, Py_ssize_t ix);

    static PyTypeObject* typeObj(Type* inType);

    static PySequenceMethods* sequenceMethodsFor(Type* t);

    static PyNumberMethods* numberMethods(Type* t);

    static Py_ssize_t mp_length(PyObject* o);

    static int sq_contains(PyObject* o, PyObject* item);

    static PyObject* mp_subscript(PyObject* o, PyObject* item);

    static PyMappingMethods* mappingMethods(Type* t);

    static bool isSubclassOfNativeType(PyTypeObject* typeObj);

    static bool isNativeType(PyTypeObject* typeObj);

    static Type* extractTypeFrom(PyTypeObject* typeObj, bool exact=false);

    static int classInstanceSetAttributeFromPyObject(Class* cls, uint8_t* data, PyObject* attrName, PyObject* attrVal);

    static int tp_setattro(PyObject *o, PyObject* attrName, PyObject* attrVal);

    static std::pair<bool, PyObject*> tryToCallOverload(const Function::Overload& f, PyObject* self, PyObject* args, PyObject* kwargs);

    static PyObject* dispatchFunctionCallToNative(const Function::Overload& overload, PyObject* argTuple, PyObject *kwargs);

    static PyObject* tp_call(PyObject* o, PyObject* args, PyObject* kwargs);

    static PyObject* tp_getattro(PyObject *o, PyObject* attrName);

    static PyObject* getattr(Type* type, instance_ptr data, char* attr_name);

    static Py_hash_t tp_hash(PyObject *o);

    static char compare_to_python(Type* t, instance_ptr self, PyObject* other, bool exact);

    static PyObject *tp_richcompare(PyObject *a, PyObject *b, int op);

    static PyObject* tp_iter(PyObject *o);

    static PyObject* tp_iternext(PyObject *o);

    static PyObject* tp_repr(PyObject *o);

    static PyObject* tp_str(PyObject *o);

    static bool typeCanBeSubclassed(Type* t);

    static PyBufferProcs* bufferProcs();

    /**
         Maintains a symbol-table and returns a PyTypeObject* for the given Type*
    */
    static PyTypeObject* typeObjInternal(Type* inType);

    static void mirrorTypeInformationIntoPyType(Type* inType, PyTypeObject* pyType);

    static PyTypeObject* getObjectAsTypeObject();

    static PyObject* createOverloadPyRepresentation(Function* f);

    static Type* pyFunctionToForward(PyObject* arg);

    static Type* tryUnwrapPyInstanceToType(PyObject* arg);

    static PyObject* categoryToPyString(Type::TypeCategory cat);

    static Type* tryUnwrapPyInstanceToValueType(PyObject* typearg);

    static PyObject* typePtrToPyTypeRepresentation(Type* t);

    static Type* unwrapTypeArgToTypePtr(PyObject* typearg);
};
