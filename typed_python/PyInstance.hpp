/******************************************************************************
   Copyright 2017-2019 typed_python Authors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

#pragma once

#include "util.hpp"
#include "AllTypes.hpp"
#include "ConversionLevel.hpp"

//extension of PyTypeObject that adds a Type* at the end.
struct NativeTypeWrapper {
    PyTypeObject typeObj;
    Type* mType;
};

//extension of PyTypeObject that adds a TypeCategory at the end.
struct NativeTypeCategoryWrapper {
    PyTypeObject typeObj;
    Type::TypeCategory mCategory;
};

class PyClassInstance;
class PyHeldClassInstance;
class PyListOfInstance;
class PyTupleOfInstance;
class PyDictInstance;
class PyConstDictInstance;
class PyPointerToInstance;
class PyRefToInstance;
class PyCompositeTypeInstance;
class PyTupleInstance;
class PyNamedTupleInstance;
class PyAlternativeMatcherInstance;
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
class PySubclassOfInstance;
class PyForwardInstance;
class PyEmbeddedMessageInstance;
class PySetInstance;
class PyPyCellInstance;
class PyTypedCellInstance;

template<class T>
class PyRegisterTypeInstance;

class PyInstance {
public:
    PyObject_HEAD

    char mIteratorFlag; //0 is keys, 1 is values, 2 is pairs
    int64_t mIteratorOffset; //-1 if we're not an iterator
    int64_t mContainerSize; //-1 if we're not an iterator

    Instance mContainingInstance;

    // we may be a temporary ref to a type
    instance_ptr mTemporaryRefTo;

    // initialize our fields after we have called 'tp_alloc'
    void initializeEmpty() {
        mIteratorFlag = -1;
        mIteratorOffset = -1;
        mContainerSize = -1;
        mTemporaryRefTo = nullptr;

        new (&mContainingInstance) Instance();
    }

    void resolveTemporaryReference();

    template<class T>
    static auto specialize(PyObject* obj, const T& f, Type* typeOverride = nullptr) {
        if (!typeOverride) {
            typeOverride = extractTypeFrom(obj->ob_type);
        }

        switch (typeOverride->getTypeCategory()) {
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
            case Type::TypeCategory::catRefTo:
                 return f(*(PyRefToInstance*)obj);
            case Type::TypeCategory::catListOf:
                return f(*(PyListOfInstance*)obj);
            case Type::TypeCategory::catNamedTuple:
                return f(*(PyNamedTupleInstance*)obj);
            case Type::TypeCategory::catTuple:
                return f(*(PyTupleInstance*)obj);
            case Type::TypeCategory::catConstDict:
                return f(*(PyConstDictInstance*)obj);
            case Type::TypeCategory::catDict:
                return f(*(PyDictInstance*)obj);
            case Type::TypeCategory::catSet:
                return f(*(PySetInstance*)obj);
            case Type::TypeCategory::catAlternative:
                return f(*(PyAlternativeInstance*)obj);
            case Type::TypeCategory::catConcreteAlternative:
                return f(*(PyConcreteAlternativeInstance*)obj);
            case Type::TypeCategory::catAlternativeMatcher:
                return f(*(PyAlternativeMatcherInstance*)obj);
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
            case Type::TypeCategory::catEmbeddedMessage:
                return f(*(PyEmbeddedMessageInstance*)obj);
            case Type::TypeCategory::catPyCell:
                return f(*(PyPyCellInstance*)obj);
            case Type::TypeCategory::catTypedCell:
                return f(*(PyTypedCellInstance*)obj);
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
            case Type::TypeCategory::catRefTo:
                 return f((PyRefToInstance*)nullptr);
            case Type::TypeCategory::catListOf:
                return f((PyListOfInstance*)nullptr);
            case Type::TypeCategory::catNamedTuple:
                return f((PyNamedTupleInstance*)nullptr);
            case Type::TypeCategory::catTuple:
                return f((PyTupleInstance*)nullptr);
            case Type::TypeCategory::catConstDict:
                return f((PyConstDictInstance*)nullptr);
            case Type::TypeCategory::catDict:
                return f((PyDictInstance*)nullptr);
            case Type::TypeCategory::catSet:
                return f((PySetInstance*)nullptr);
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
            case Type::TypeCategory::catAlternativeMatcher:
                return f((PyAlternativeMatcherInstance*)nullptr);
            case Type::TypeCategory::catBoundMethod:
                return f((PyBoundMethodInstance*)nullptr);
            case Type::TypeCategory::catNone:
                return f((PyNoneInstance*)nullptr);
            case Type::TypeCategory::catString:
                return f((PyStringInstance*)nullptr);
            case Type::TypeCategory::catBytes:
                return f((PyBytesInstance*)nullptr);
            case Type::TypeCategory::catEmbeddedMessage:
                return f((PyEmbeddedMessageInstance*)nullptr);
            case Type::TypeCategory::catPyCell:
                return f((PyPyCellInstance*)nullptr);
            case Type::TypeCategory::catTypedCell:
                return f((PyTypedCellInstance*)nullptr);
            case Type::TypeCategory::catOneOf:
                return f((PyOneOfInstance*)nullptr);
            case Type::TypeCategory::catForward:
                return f((PyForwardInstance*)nullptr);
            case Type::TypeCategory::catSubclassOf:
                return f((PySubclassOfInstance*)nullptr);
            default:
                throw std::runtime_error("Invalid type category. Memory must have been corrupted.");
        }
    }

    static int reversePyOpOrdering(int op);

    template<class T>
    static int specializeForTypeReturningInt(PyObject* obj, const T& f, Type* typeOverride = nullptr) {
        try {
            return specialize(obj, f, typeOverride);
        } catch(PythonExceptionSet& e) {
            return -1;
        } catch(std::exception& e) {
            PyErr_SetString(PyExc_TypeError, e.what());
            return -1;
        }
    }

    template<class T>
    static Py_ssize_t specializeForTypeReturningSizeT(PyObject* obj, const T& f, Type* typeOverride=nullptr) {
        try {
            return specialize(obj, f, typeOverride);
        } catch(PythonExceptionSet& e) {
            return -1;
        } catch(std::exception& e) {
            PyErr_SetString(PyExc_TypeError, e.what());
            return -1;
        }
    }

    template<class T>
    static PyObject* specializeForType(PyObject* obj, const T& f, Type* typeOverride=nullptr) {
        try {
            return specialize(obj, f, typeOverride);
        } catch(PythonExceptionSet& e) {
            return NULL;
        } catch(std::exception& e) {
            PyErr_SetString(PyExc_TypeError, e.what());
            return NULL;
        }
    }

    //return the standard python representation of an object of type 'eltType'
    template<class init_func>
    static PyObject* initializePythonRepresentation(Type* eltType, const init_func& f) {
        Instance instance(eltType, f);

        return extractPythonObject(instance.data(), instance.type());
    }

    //initialize a PyInstance for 'eltType'. For ints, floats, etc, with
    //actual native representations, this will produce a wrapper object (maybe not what you want)
    //rather than the standard python representation.
    template<class init_func>
    static PyObject* initialize(Type* eltType, const init_func& f) {
        eltType->assertForwardsResolvedSufficientlyToInstantiate();

        PyInstance* self =
            (PyInstance*)typeObj(eltType)->tp_alloc(typeObj(eltType), 0);

        self->initializeEmpty();

        try {
            self->initialize(f, eltType);

            return (PyObject*)self;
        } catch(...) {
            typeObj(eltType)->tp_dealloc((PyObject*)self);
            throw;
        }
    }

    static PyObject* initializeTemporaryRef(Type* eltType, instance_ptr data) {
        eltType->assertForwardsResolvedSufficientlyToInstantiate();

        PyInstance* self =
            (PyInstance*)typeObj(eltType)->tp_alloc(typeObj(eltType), 0);

        self->initializeEmpty();
        self->mTemporaryRefTo = data;

        return (PyObject*)self;
    }

    template<class init_func>
    void initialize(const init_func& i, Type* typeIfKnown = nullptr) {
        Type* type = typeIfKnown ? typeIfKnown : extractTypeFrom(((PyObject*)this)->ob_type);
        type->assertForwardsResolvedSufficientlyToInstantiate();

        mContainingInstance = Instance(type, i);
    }

    static PyObject* fromInstance(const Instance& instance) {
        return extractPythonObject(instance.data(), instance.type());
    }

    PyInstance* duplicate() {
        return (PyInstance*)initialize(type(), [&](instance_ptr out) {
            type()->copy_constructor(out, dataPtr());
        });
    }

    PyObject* createIteratorToSelf(int32_t iterTypeFlag, int64_t containerSize) {
        PyInstance* result = (PyInstance*)initialize(type(), [&](instance_ptr out) {
            type()->copy_constructor(out, dataPtr());
        });

        result->mIteratorOffset = 0;
        result->mIteratorFlag = iterTypeFlag;
        result->mContainerSize = containerSize;

        return (PyObject*)result;
    }

    instance_ptr dataPtr();

    Type* type();

    std::pair<Type*, instance_ptr> derefAnyRefTo();

    static PyMethodDef* typeMethods(Type* t);

    static PyMethodDef* typeMethodsConcrete(Type* t);

    static void tp_dealloc(PyObject* self);

    static bool pyValCouldBeOfType(Type* t, PyObject* pyRepresentation, ConversionLevel level);

    /**
     construct an 'eltType' from a python object at 'tgt'. ConversionLevel determines what
     level of type conversion we're willing to tolerate. If we can't do the conversion, we'll
     throw an exception.
     */
    static void copyConstructFromPythonInstance(Type* eltType, instance_ptr tgt, PyObject* pyRepresentation, ConversionLevel level);

    static void copyConstructFromPythonInstanceConcrete(Type* eltType, instance_ptr tgt, PyObject* pyRepresentation, ConversionLevel level);

    static void constructFromPythonArguments(uint8_t* data, Type* t, PyObject* args, PyObject* kwargs);

    static void constructFromPythonArgumentsConcrete(Type* t, uint8_t* data, PyObject* args, PyObject* kwargs);

    //produce the pythonic representation of this object. for values that have a direct python representation,
    //such as integers, strings, bools, or None, we return an actual python object. Otherwise,
    //we return a pointer to a PyInstance representing the object.
    //if 'createTemporaryRef' is 'false', then we don't create temporary references to HeldClass
    //instances.
    static PyObject* extractPythonObject(instance_ptr data, Type* eltType, bool createTemporaryRef=true);

    static PyObject* extractPythonObject(const Instance& instance);

    //if we have a python representation that we want to use for this object, override and return not-NULL.
    //otherwise, this version takes over and returns a PyInstance wrapper for the object
    static PyObject* extractPythonObjectConcrete(Type* eltType, instance_ptr data);

    //the tp_new for actual instances of typed_python Types
    static PyObject *tp_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds);

    //the tp_new_for typed_python categories. These objects represent things like 'ListOf'
    //and when called produced ListOf(T)
    static PyObject* tp_new_type(PyTypeObject *subtype, PyObject *args, PyObject *kwds);

    static PyObject* nb_rshift(PyObject* lhs, PyObject* rhs);

    static PyObject* pyUnaryOperator(PyObject* lhs, const char* op, const char* opErrRep);

    static PyObject* pyOperator(PyObject* lhs, PyObject* rhs, const char* op, const char* opErrRep);

    static PyObject* pyTernaryOperator(PyObject* lhs, PyObject* rhs, PyObject* ternary, const char* op, const char* opErrRep);

    static int pyInquiry(PyObject* lhs, const char* op, const char* opErrRep);

    PyObject* pyUnaryOperatorConcrete(const char* op, const char* opErrRep);

    PyObject* pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErrRep);

    PyObject* pyOperatorConcreteReverse(PyObject* lhs, const char* op, const char* opErrRep);

    PyObject* pyTernaryOperatorConcrete(PyObject* rhs, PyObject* third, const char* op, const char* opErrRep);

    int pyInquiryConcrete(const char* op, const char* opErrRep);

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

    static int nb_bool(PyObject* lhs);

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

    static int sq_ass_item(PyObject* o, Py_ssize_t ix, PyObject* v);

    int sq_ass_item_concrete(Py_ssize_t ix, PyObject* v);

    static PyTypeObject* typeObj(Type* inType);

    static PyObject* undefinedBehaviorException();

    static PyObject* nonTypesAcceptedAsTypes();

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

    static Type* rootNativeType(PyTypeObject* typeObj);

    static bool isNativeType(PyTypeObject* typeObj);

    static Type* extractTypeFrom(PyTypeObject* typeObj);

    // if 'obj' is a PyInstance, return its type and data ptrs, after unwrapping
    // any 'RefTo' classes, which is the standard behavior for most classes,
    // which don't specifically know anything about 'RefTo'
    static std::pair<Type*, instance_ptr> extractTypeAndPtrFrom(PyObject* obj);

    static int tp_setattro(PyObject *o, PyObject* attrName, PyObject* attrVal);

    int tp_setattr_concrete(PyObject* attrName, PyObject* attrVal);

    static PyObject* tp_call(PyObject* o, PyObject* args, PyObject* kwargs);

    PyObject* tp_call_concrete(PyObject* args, PyObject* kwargs);

    static PyObject* tp_getattro(PyObject *o, PyObject* attrName);

    PyObject* tp_getattr_concrete(PyObject* attrPyObj, const char* attrName);

    static Py_hash_t tp_hash(PyObject *o);

    /***
     compare this value to a python value using the comparison op pyComparisonOp (Py_EQ, Py_LT, etc.)
     if 'exact', then the types must be equivalent as well.
    ***/
    static bool compare_to_python(Type* t, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp);

    static bool compare_to_python_concrete(Type* t, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp);

    bool compare_as_iterator_to_python_concrete(PyObject* other, int pyComparisonOp);

    static PyObject *tp_richcompare(PyObject *a, PyObject *b, int op);

    static PyObject* tp_iter(PyObject *o);

    PyObject* tp_iter_concrete();

    static PyObject* tp_iternext(PyObject *o);

    PyObject* tp_iternext_concrete();

    static PyObject* tp_repr(PyObject *o);

    PyObject* tp_repr_concrete();

    static PyObject* tp_descr_get(PyObject* func, PyObject* obj, PyObject* type);

    static PyObject* tp_str(PyObject *o);

    PyObject* tp_str_concrete();

    static bool typeCanBeSubclassed(Type* t);

    static PyBufferProcs* bufferProcs();

    static PyObject* getInternalModuleMember(const char* name);

    static PyTypeObject* allTypesBaseType();

    static PyTypeObject* typeCategoryBaseType(Type::TypeCategory category);
    /**
         Maintains a symbol-table and returns a PyTypeObject* for the given Type*
    */
    static PyTypeObject* typeObjInternal(Type* inType);

    static void mirrorTypeInformationIntoPyType(Type* inType, PyTypeObject* pyType);

    static void mirrorTypeInformationIntoPyTypeConcrete(Type* inType, PyTypeObject* pyType);

    static Type* tryUnwrapPyInstanceToType(PyObject* arg);

    static PyObject* categoryToPyString(Type::TypeCategory cat);

    static Instance unwrapPyObjectToInstance(PyObject* inst, bool allowArbitraryPyObjects);

    // attempt to unwrap a python object as a Value(T). If allowArbitraryPyObjects, then anything
    // can be a Value. This should only be the case if we _explicitly_ convert the object to a Value
    // by calling 'Value'. Returns 'nullptr' if it's not possible.
    static Type* tryUnwrapPyInstanceToValueType(PyObject* typearg, bool allowArbitraryPyObjects);

    static PyObject* typePtrToPyTypeRepresentation(Type* t);

    /****
        Convert a python object passed as a type object to the appropriate Type*.

        This function will unwrap native python types like 'float', 'int', or 'bool'
        to their appropriate representations, convert constants to singletons, etc.
    *****/
    static Type* unwrapTypeArgToTypePtr(PyObject* typearg);
};
