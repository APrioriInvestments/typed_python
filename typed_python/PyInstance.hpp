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
class PythonExceptionSet {};

class PyListOfInstance;
class PyTupleOfInstance;
class PyConstDictInstance;

class PyInstance {
public:
    PyObject_HEAD

    bool mIsInitialized;
    bool mIsMatcher;
    char mIteratorFlag; //0 is keys, 1 is values, 2 is pairs
    int64_t mIteratorOffset; //-1 if we're not an iterator

    Instance mContainingInstance;

    template<class T>
    static auto check(PyObject* obj, const T& f) {
        switch (extractTypeFrom(obj->ob_type)->getTypeCategory()) {
            // case catNone:
            //     return f(*(None*)this);
            // case catBool:
            //     return f(*(Bool*)this);
            // case catUInt8:
            //     return f(*(UInt8*)this);
            // case catUInt16:
            //     return f(*(UInt16*)this);
            // case catUInt32:
            //     return f(*(UInt32*)this);
            // case catUInt64:
            //     return f(*(UInt64*)this);
            // case catInt8:
            //     return f(*(Int8*)this);
            // case catInt16:
            //     return f(*(Int16*)this);
            // case catInt32:
            //     return f(*(Int32*)this);
            // case catInt64:
            //     return f(*(Int64*)this);
            // case catString:
            //     return f(*(String*)this);
            // case catBytes:
            //     return f(*(Bytes*)this);
            // case catFloat32:
            //     return f(*(Float32*)this);
            // case catFloat64:
            //     return f(*(Float64*)this);
            // case catValue:
            //     return f(*(Value*)this);
            // case catOneOf:
            //     return f(*(OneOf*)this);
            case Type::TypeCategory::catTupleOf:
                return f(*(PyTupleOfInstance*)obj);
            // case catPointerTo:
            //     return f(*(PointerTo*)this);
            case Type::TypeCategory::catListOf:
                return f(*(PyListOfInstance*)obj);
            // case catNamedTuple:
            //     return f(*(NamedTuple*)this);
            // case catTuple:
            //     return f(*(Tuple*)this);
            case Type::TypeCategory::catConstDict:
                return f(*(PyConstDictInstance*)obj);
            // case catAlternative:
            //     return f(*(Alternative*)this);
            // case catConcreteAlternative:
            //     return f(*(ConcreteAlternative*)this);
            // case catPythonSubclass:
            //     return f(*(PythonSubclass*)this);
            // case catPythonObjectOfType:
            //     return f(*(PythonObjectOfType*)this);
            // case catClass:
            //     return f(*(Class*)this);
            // case catHeldClass:
            //     return f(*(HeldClass*)this);
            // case catFunction:
            //     return f(*(Function*)this);
            // case catBoundMethod:
            //     return f(*(BoundMethod*)this);
            // case catForward:
            //     return f(*(Forward*)this);
            default:
                throw std::runtime_error("Invalid type found");
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

    instance_ptr dataPtr();

    static PyObject* listAppend(PyObject* o, PyObject* args);

    static PyObject* listResize(PyObject* o, PyObject* args);

    static PyObject* listReserve(PyObject* o, PyObject* args);

    static PyObject* listClear(PyObject* o, PyObject* args);

    static PyObject* listReserved(PyObject* o, PyObject* args);

    static PyObject* listPop(PyObject* o, PyObject* args);

    static PyObject* listSetSizeUnsafe(PyObject* o, PyObject* args);

    static PyObject* listPointerUnsafe(PyObject* o, PyObject* args);

    static PyObject* pointerInitialize(PyObject* o, PyObject* args);

    static PyObject* pointerSet(PyObject* o, PyObject* args);

    static PyObject* pointerGet(PyObject* o, PyObject* args);

    static PyObject* pointerCast(PyObject* o, PyObject* args);

    static PyObject* constDictItems(PyObject *o);

    static PyObject* constDictKeys(PyObject *o);

    static PyObject* constDictValues(PyObject *o);

    static PyObject* constDictGet(PyObject* o, PyObject* args);

    static PyMethodDef* typeMethods(Type* t);

    static void tp_dealloc(PyObject* self);

    static bool pyValCouldBeOfType(Type* t, PyObject* pyRepresentation);

    static void copyConstructFromPythonInstance(Type* eltType, instance_ptr tgt, PyObject* pyRepresentation);

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

    //PyObject* sq_concat_concrete(PyObject* rhs);

    static PyObject* sq_item(PyObject* o, Py_ssize_t ix);

    static PyTypeObject* typeObj(Type* inType);

    static PyObject* undefinedBehaviorException();

    static PySequenceMethods* sequenceMethodsFor(Type* t);

    static PyNumberMethods* numberMethods(Type* t);

    static Py_ssize_t mp_length(PyObject* o);

    static int sq_contains(PyObject* o, PyObject* item);

    static int mp_ass_subscript(PyObject* o, PyObject* item, PyObject* value);

    static PyObject* mp_subscript(PyObject* o, PyObject* item);

    static PyMappingMethods* mappingMethods(Type* t);

    static bool isSubclassOfNativeType(PyTypeObject* typeObj);

    static bool isNativeType(PyTypeObject* typeObj);

    static Type* extractTypeFrom(PyTypeObject* typeObj, bool exact=false);

    static int classInstanceSetAttributeFromPyObject(Class* cls, uint8_t* data, PyObject* attrName, PyObject* attrVal);

    static int tp_setattro(PyObject *o, PyObject* attrName, PyObject* attrVal);

    static std::pair<bool, PyObject*> tryToCallOverload(const Function::Overload& f, PyObject* self, PyObject* args, PyObject* kwargs);

    //perform a linear scan of all specializations contained in overload and attempt to dispatch to each one.
    //returns <true, result or none> if we dispatched.
    static std::pair<bool, PyObject*> dispatchFunctionCallToNative(const Function::Overload& overload, PyObject* argTuple, PyObject *kwargs);

    //attempt to dispatch to this one exact specialization by converting each arg to the relevant type. if
    //we can't convert, then return <false, nullptr>. If we do dispatch, return <true, result or none> and set
    //the python exception if native code returns an exception.
    static std::pair<bool, PyObject*> dispatchFunctionCallToCompiledSpecialization(
                                                const Function::Overload& overload,
                                                const Function::CompiledSpecialization& specialization,
                                                PyObject* argTuple,
                                                PyObject *kwargs
                                                );

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

    static Instance unwrapPyObjectToInstance(PyObject* inst);

    static Type* tryUnwrapPyInstanceToValueType(PyObject* typearg);

    static PyObject* typePtrToPyTypeRepresentation(Type* t);

    static Type* unwrapTypeArgToTypePtr(PyObject* typearg);
};
