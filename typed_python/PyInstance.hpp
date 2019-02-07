#pragma once

#include "util.hpp"
#include "AllTypes.hpp"

//extension of PyTypeObject that adds a Type* at the end.
struct NativeTypeWrapper {
    PyTypeObject typeObj;
    Type* mType;
};

class InternalPyException {};

//throw to indicate we set a python error already.
class PyClassInstance;
class PyHeldClassInstance;
class PyListOfInstance;
class PyTupleOfInstance;
class PyConstDictInstance;
class PyPointerToInstance;
class PyCompositeTypeInstance;
class PyTupleInstance;
class PyNamedTupleInstance;
class PyAlternativeInstance;
class PyConcreteAlternativeInstance;
class PyStringInstance;
class PyBytesInstance;
class PyFunctionInstance;
class PyBoundMethodInstance;
class PyNoneInstance;
class PyValueInstance;
class PyPythonSubclassInstance;
class PyPythonObjectOfTypeInstance;
class PyOneOfInstance;
class PyForwardInstance;

template<class T>
class PyRegisterTypeInstance;

class PyInstance {
public:
    PyObject_HEAD

    bool mIsInitialized;
    bool mIsMatcher;
    char mIteratorFlag; //0 is keys, 1 is values, 2 is pairs
    int64_t mIteratorOffset; //-1 if we're not an iterator

    Instance mContainingInstance;

    template<class T>
    static auto specialize(PyObject* obj, const T& f) {
        switch (extractTypeFrom(obj->ob_type)->getTypeCategory()) {
            case Type::TypeCategory::catBool:
                return f(*(PyRegisterTypeInstance<bool>*)obj);
            case Type::TypeCategory::catUInt8:
                return f(*(PyRegisterTypeInstance<uint8_t>*)obj);
            case Type::TypeCategory::catUInt16:
                return f(*(PyRegisterTypeInstance<uint16_t>*)obj);
            case Type::TypeCategory::catUInt32:
                return f(*(PyRegisterTypeInstance<uint32_t>*)obj);
            case Type::TypeCategory::catUInt64:
                return f(*(PyRegisterTypeInstance<uint64_t>*)obj);
            case Type::TypeCategory::catInt8:
                return f(*(PyRegisterTypeInstance<int8_t>*)obj);
            case Type::TypeCategory::catInt16:
                return f(*(PyRegisterTypeInstance<int16_t>*)obj);
            case Type::TypeCategory::catInt32:
                return f(*(PyRegisterTypeInstance<int32_t>*)obj);
            case Type::TypeCategory::catFloat32:
                return f(*(PyRegisterTypeInstance<float>*)obj);
            case Type::TypeCategory::catValue:
               return f(*(PyValueInstance*)obj);
            case Type::TypeCategory::catTupleOf:
                return f(*(PyTupleOfInstance*)obj);
            case Type::TypeCategory::catPointerTo:
                 return f(*(PyPointerToInstance*)obj);
            case Type::TypeCategory::catListOf:
                return f(*(PyListOfInstance*)obj);
            case Type::TypeCategory::catNamedTuple:
                return f(*(PyNamedTupleInstance*)obj);
            case Type::TypeCategory::catTuple:
                return f(*(PyTupleInstance*)obj);
            case Type::TypeCategory::catConstDict:
                return f(*(PyConstDictInstance*)obj);
            case Type::TypeCategory::catAlternative:
                return f(*(PyAlternativeInstance*)obj);
            case Type::TypeCategory::catConcreteAlternative:
                return f(*(PyConcreteAlternativeInstance*)obj);
            case Type::TypeCategory::catPythonSubclass:
                return f(*(PyPythonSubclassInstance*)obj);
            case Type::TypeCategory::catPythonObjectOfType:
                return f(*(PyPythonObjectOfTypeInstance*)obj);
            case Type::TypeCategory::catClass:
                return f(*(PyClassInstance*)obj);
            case Type::TypeCategory::catHeldClass:
                return f(*(PyHeldClassInstance*)obj);
            case Type::TypeCategory::catFunction:
                return f(*(PyFunctionInstance*)obj);
            case Type::TypeCategory::catBoundMethod:
                return f(*(PyBoundMethodInstance*)obj);
            case Type::TypeCategory::catNone:
            case Type::TypeCategory::catInt64:
            case Type::TypeCategory::catFloat64:
            case Type::TypeCategory::catString:
            case Type::TypeCategory::catBytes:
            case Type::TypeCategory::catOneOf:
                throw std::runtime_error("No python object should ever have this type.");
            default:
                throw std::runtime_error("Invalid type category. Memory must have been corrupted.");
        }
    }

    //Calls 'f' with a reference to a non-valid instance of a PyInstance subclass. Clients
    //can use decltype to extract the actual type and dispatch using that
    template<class T>
    static auto specializeStatic(Type::TypeCategory category, const T& f) {
        switch (category) {
            case Type::TypeCategory::catBool:
                return f((PyRegisterTypeInstance<bool>*)nullptr);
            case Type::TypeCategory::catUInt8:
                return f((PyRegisterTypeInstance<uint8_t>*)nullptr);
            case Type::TypeCategory::catUInt16:
                return f((PyRegisterTypeInstance<uint16_t>*)nullptr);
            case Type::TypeCategory::catUInt32:
                return f((PyRegisterTypeInstance<uint32_t>*)nullptr);
            case Type::TypeCategory::catUInt64:
                return f((PyRegisterTypeInstance<uint64_t>*)nullptr);
            case Type::TypeCategory::catInt8:
                return f((PyRegisterTypeInstance<int8_t>*)nullptr);
            case Type::TypeCategory::catInt16:
                return f((PyRegisterTypeInstance<int16_t>*)nullptr);
            case Type::TypeCategory::catInt32:
                return f((PyRegisterTypeInstance<int32_t>*)nullptr);
            case Type::TypeCategory::catFloat32:
                return f((PyRegisterTypeInstance<float>*)nullptr);
            case Type::TypeCategory::catInt64:
                return f((PyRegisterTypeInstance<int64_t>*)nullptr);
            case Type::TypeCategory::catFloat64:
                return f((PyRegisterTypeInstance<double>*)nullptr);
            case Type::TypeCategory::catValue:
               return f((PyValueInstance*)nullptr);
            case Type::TypeCategory::catTupleOf:
                return f((PyTupleOfInstance*)nullptr);
            case Type::TypeCategory::catPointerTo:
                 return f((PyPointerToInstance*)nullptr);
            case Type::TypeCategory::catListOf:
                return f((PyListOfInstance*)nullptr);
            case Type::TypeCategory::catNamedTuple:
                return f((PyNamedTupleInstance*)nullptr);
            case Type::TypeCategory::catTuple:
                return f((PyTupleInstance*)nullptr);
            case Type::TypeCategory::catConstDict:
                return f((PyConstDictInstance*)nullptr);
            case Type::TypeCategory::catAlternative:
                return f((PyAlternativeInstance*)nullptr);
            case Type::TypeCategory::catConcreteAlternative:
                return f((PyConcreteAlternativeInstance*)nullptr);
            case Type::TypeCategory::catPythonSubclass:
                return f((PyPythonSubclassInstance*)nullptr);
            case Type::TypeCategory::catPythonObjectOfType:
                return f((PyPythonObjectOfTypeInstance*)nullptr);
            case Type::TypeCategory::catClass:
                return f((PyClassInstance*)nullptr);
            case Type::TypeCategory::catHeldClass:
                return f((PyHeldClassInstance*)nullptr);
            case Type::TypeCategory::catFunction:
                return f((PyFunctionInstance*)nullptr);
            case Type::TypeCategory::catBoundMethod:
                return f((PyBoundMethodInstance*)nullptr);
            case Type::TypeCategory::catNone:
                return f((PyNoneInstance*)nullptr);
            case Type::TypeCategory::catString:
                return f((PyStringInstance*)nullptr);
            case Type::TypeCategory::catBytes:
                return f((PyBytesInstance*)nullptr);
            case Type::TypeCategory::catOneOf:
                return f((PyOneOfInstance*)nullptr);
            case Type::TypeCategory::catForward:
                return f((PyForwardInstance*)nullptr);
            default:
                throw std::runtime_error("Invalid type category. Memory must have been corrupted.");
        }
    }

    static int reversePyOpOrdering(int op);

    template<class T>
    static int specializeForTypeReturningInt(PyObject* obj, const T& f) {
        try {
            return specialize(obj, f);
        } catch(PythonExceptionSet& e) {
            return -1;
        } catch(std::exception& e) {
            PyErr_SetString(PyExc_TypeError, e.what());
            return -1;
        }
    }

    template<class T>
    static Py_ssize_t specializeForTypeReturningSizeT(PyObject* obj, const T& f) {
        try {
            return specialize(obj, f);
        } catch(PythonExceptionSet& e) {
            return -1;
        } catch(std::exception& e) {
            PyErr_SetString(PyExc_TypeError, e.what());
            return -1;
        }
    }

    template<class T>
    static PyObject* specializeForType(PyObject* obj, const T& f) {
        try {
            return specialize(obj, f);
        } catch(PythonExceptionSet& e) {
            return NULL;
        } catch(std::exception& e) {
            PyErr_SetString(PyExc_TypeError, e.what());
            return NULL;
        }
    }

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

    //initialize a PyInstance for 'eltType'. For ints, floats, etc, with
    //actual native representations, this will produce a wrapper object (maybe not what you want)
    //rather than the standard python representation.
    template<class init_func>
    static PyObject* initialize(Type* eltType, const init_func& f) {
        if (!guaranteeForwardsResolved(eltType)) {
            return nullptr;
        }

        PyInstance* self =
            (PyInstance*)typeObj(eltType)->tp_alloc(typeObj(eltType), 0);

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

    PyInstance* duplicate() {
        return (PyInstance*)initialize(type(), [&](instance_ptr out) {
            type()->copy_constructor(out, dataPtr());
        });
    }

    instance_ptr dataPtr();

    Type* type();

    static PyMethodDef* typeMethods(Type* t);

    static PyMethodDef* typeMethodsConcrete();

    static void tp_dealloc(PyObject* self);

    static bool pyValCouldBeOfType(Type* t, PyObject* pyRepresentation);

    /**
     construct an 'eltType' from a python object at 'tgt'. If 'isExplicit' then we're invoked from an explicit
     copy constructor, so more liberal conversion is allowed than if 'isExplicit' is false, which happens
     when we're attempting to convert for purposes of method dispatch.
     */
    static void copyConstructFromPythonInstance(Type* eltType, instance_ptr tgt, PyObject* pyRepresentation, bool isExplicit=false);

    static void copyConstructFromPythonInstanceConcrete(Type* eltType, instance_ptr tgt, PyObject* pyRepresentation, bool isExplicit);

    static void constructFromPythonArguments(uint8_t* data, Type* t, PyObject* args, PyObject* kwargs);

    static void constructFromPythonArgumentsConcrete(Type* t, uint8_t* data, PyObject* args, PyObject* kwargs);

    //produce the pythonic representation of this object. for values that have a direct python representation,
    //such as integers, strings, bools, or None, we return an actual python object. Otherwise,
    //we return a pointer to a PyInstance representing the object.
    static PyObject* extractPythonObject(instance_ptr data, Type* eltType);

    //if we have a python representation that we want to use for this object, override and return not-NULL.
    //otherwise, this version takes over and returns a PyInstance wrapper for the object
    static PyObject* extractPythonObjectConcrete(Type* eltType, instance_ptr data);

    static PyObject *tp_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds);

    static PyObject* nb_rshift(PyObject* lhs, PyObject* rhs);

    static PyObject* pyUnaryOperator(PyObject* lhs, const char* op, const char* opErrRep);

    static PyObject* pyOperator(PyObject* lhs, PyObject* rhs, const char* op, const char* opErrRep);

    static PyObject* pyTernaryOperator(PyObject* lhs, PyObject* rhs, PyObject* ternary, const char* op, const char* opErrRep);

    PyObject* pyUnaryOperatorConcrete(const char* op, const char* opErrRep);

    PyObject* pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErrRep);

    PyObject* pyOperatorConcreteReverse(PyObject* lhs, const char* op, const char* opErrRep);

    PyObject* pyTernaryOperatorConcrete(PyObject* rhs, PyObject* third, const char* op, const char* opErrRep);

    static PyObject* nb_inplace_add(PyObject* lhs, PyObject* rhs);

    static PyObject* nb_inplace_subtract(PyObject* lhs, PyObject* rhs);

    static PyObject* nb_inplace_multiply(PyObject* lhs, PyObject* rhs);

    static PyObject* nb_inplace_remainder(PyObject* lhs, PyObject* rhs);

    static PyObject* nb_inplace_power(PyObject* lhs, PyObject* rhs, PyObject* modOrNone);

    static PyObject* nb_inplace_lshift(PyObject* lhs, PyObject* rhs);

    static PyObject* nb_inplace_rshift(PyObject* lhs, PyObject* rhs);

    static PyObject* nb_inplace_and(PyObject* lhs, PyObject* rhs);

    static PyObject* nb_inplace_xor(PyObject* lhs, PyObject* rhs);

    static PyObject* nb_inplace_or(PyObject* lhs, PyObject* rhs);

    static PyObject* nb_floor_divide(PyObject* lhs, PyObject* rhs);

    static PyObject* nb_true_divide(PyObject* lhs, PyObject* rhs);

    static PyObject* nb_inplace_floor_divide(PyObject* lhs, PyObject* rhs);

    static PyObject* nb_inplace_true_divide(PyObject* lhs, PyObject* rhs);

    static PyObject* nb_inplace_matrix_multiply(PyObject* lhs, PyObject* rhs);

    static PyObject* nb_power(PyObject* lhs, PyObject* rhs, PyObject* modOrNone);

    static PyObject* nb_negative(PyObject* lhs);

    static PyObject* nb_positive(PyObject* lhs);

    static PyObject* nb_absolute(PyObject* lhs);

    static PyObject* nb_invert(PyObject* lhs);

    static PyObject* nb_int(PyObject* lhs);

    static PyObject* nb_float(PyObject* lhs);

    static PyObject* nb_index(PyObject* lhs);

    static PyObject* nb_add(PyObject* lhs, PyObject* rhs);

    static PyObject* nb_subtract(PyObject* lhs, PyObject* rhs);

    static PyObject* nb_multiply(PyObject* lhs, PyObject* rhs);

    static PyObject* nb_remainder(PyObject* lhs, PyObject* rhs);

    static PyObject* nb_lshift(PyObject* lhs, PyObject* rhs);

    static PyObject* nb_and(PyObject* lhs, PyObject* rhs);

    static PyObject* nb_or(PyObject* lhs, PyObject* rhs);

    static PyObject* nb_xor(PyObject* lhs, PyObject* rhs);

    static PyObject* nb_matmul(PyObject* lhs, PyObject* rhs);

    static PyObject* nb_divmod(PyObject* lhs, PyObject* rhs);

    static PyObject* sq_item(PyObject* o, Py_ssize_t ix);

    PyObject* sq_item_concrete(Py_ssize_t ix);

    static PyTypeObject* typeObj(Type* inType);

    static PyObject* undefinedBehaviorException();

    static PySequenceMethods* sequenceMethodsFor(Type* t);

    static PyNumberMethods* numberMethods(Type* t);

    static Py_ssize_t mp_and_sq_length(PyObject* o);

    Py_ssize_t mp_and_sq_length_concrete();

    static int sq_contains(PyObject* o, PyObject* item);

    int sq_contains_concrete(PyObject* item);

    static int mp_ass_subscript(PyObject* o, PyObject* item, PyObject* value);

    int mp_ass_subscript_concrete(PyObject* item, PyObject* value);

    static PyObject* mp_subscript(PyObject* o, PyObject* item);

    PyObject* mp_subscript_concrete(PyObject* item);

    static PyMappingMethods* mappingMethods(Type* t);

    static bool isSubclassOfNativeType(PyTypeObject* typeObj);

    static bool isNativeType(PyTypeObject* typeObj);

    static Type* extractTypeFrom(PyTypeObject* typeObj, bool exact=false);

    static int tp_setattro(PyObject *o, PyObject* attrName, PyObject* attrVal);

    static PyObject* tp_call(PyObject* o, PyObject* args, PyObject* kwargs);

    PyObject* tp_call_concrete(PyObject* args, PyObject* kwargs);

    static PyObject* tp_getattro(PyObject *o, PyObject* attrName);

    PyObject* tp_getattr_concrete(PyObject* attrPyObj, const char* attrName);

    static Py_hash_t tp_hash(PyObject *o);

    static bool compare_to_python(Type* t, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp);

    static PyObject *tp_richcompare(PyObject *a, PyObject *b, int op);

    static PyObject* tp_iter(PyObject *o);

    PyObject* tp_iter_concrete();

    static PyObject* tp_iternext(PyObject *o);

    PyObject* tp_iternext_concrete();

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

    static Type* pyFunctionToForward(PyObject* arg);

    static Type* tryUnwrapPyInstanceToType(PyObject* arg);

    static PyObject* categoryToPyString(Type::TypeCategory cat);

    static Instance unwrapPyObjectToInstance(PyObject* inst);

    static Type* tryUnwrapPyInstanceToValueType(PyObject* typearg);

    static PyObject* typePtrToPyTypeRepresentation(Type* t);

    static Type* unwrapTypeArgToTypePtr(PyObject* typearg);
};
