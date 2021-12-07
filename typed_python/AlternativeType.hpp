/******************************************************************************
   Copyright 2017-2020 typed_python Authors

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

#include "Type.hpp"
#include "CompositeType.hpp"

PyDoc_STRVAR(Alternative_doc,
    "Alternative(\n"
    "    name,\n"
    "    Subtype1=dict(field1=type1, field2=type2, ...),\n"
    "    ..."
    "    method1=funcOrLambda,\n"
    "    ..."
    ")\n\n"
    "Produce an Alternative, a base class with a specified set of subtypes, each\n"
    "containing a specified set of members.  This is roughly equivalent to a tagged union\n"
    "in ML."
    "\n"
    "The first argument must be the name of the type given as a string.\n"
    "\n"
    "Each subsequent argument must be a keyword argument, and may either be a\n"
    "dictionary from fieldname to type, defining a concrete subclass, or a lambda\n"
    "or function, which defines a method common to all instances of the Alternative.\n"
    "\n"
    "Alternatives defined in this way can be constructed as \n"
    "\n"
    "    A.Subtype(field1=val1, ...)\n"
    "\n"
    "and support accessing members directly (e.g. x.field1) and a 'matches' member\n"
    "that lets you determine which of the specified set of subtypes you're dealing with.\n"
    "\n"
    "Alternatives are a good tool for implementing data structures like message \n"
    "protocols and syntax trees.  Every instance of the alternative is one of the \n"
    "specifically named subtypes, so consuming code can explicitly switch on the \n"
    "subtypes.\n"
    "\n"
    "Example:\n"
    "\n"
    "    # define a forward for Expression so it can be used recursively\n"
    "    # in the type definition.\n"
    "    Expression = Forward('Expression')\n"
    "\n"
    "    Expression = Alternative(\n"
    "        'Expression',\n"
    "        Constant=dict(value=object)\n"
    "        Add=dict(lhs=Expression, rhs=Expression)\n"
    "        Sub=dict(lhs=Expression, rhs=Expression)\n"
    "        Mul=dict(lhs=Expression, rhs=Expression)\n"
    "        Div=dict(lhs=Expression, rhs=Expression)\n"
    "        Variable=dict(name=str)\n"
    "        evaluate=lambda self, vars:\n"
    "            self.value if self.matches.Constant else \n"
    "            self.lhs.evaluate(vars) + self.rhs.evaluate(vars) if self.matches.Add\n"
    "            self.lhs.evaluate(vars) - self.rhs.evaluate(vars) if self.matches.Sub\n"
    "            self.lhs.evaluate(vars) * self.rhs.evaluate(vars) if self.matches.Mul\n"
    "            self.lhs.evaluate(vars) / self.rhs.evaluate(vars) if self.matches.Div\n"
    "            vars.get(self.name) if self.matches.Variable\n"
    "        )\n"
    "\n"
    "    addXAndY = Expression.Add(lhs=Expression.Variable(name='x'), rhs=Expression.Variable(name='y'))\n"
    "    assert addXAndY.evaluate({'x': 10, 'y': 20}) == 30\n\n"
);

class Alternative : public Type {
public:
    class layout {
    public:
        std::atomic<int64_t> refcount;

        int64_t which;
        uint8_t data[];
    };

    typedef layout* layout_ptr;

    Alternative(std::string name,
                std::string moduleName,
                const std::vector<std::pair<std::string, NamedTuple*> >& subtypes,
                const std::map<std::string, Function*>& methods
                ) :
            Type(TypeCategory::catAlternative),
            m_default_construction_ix(0),
            m_default_construction_type(nullptr),
            m_subtypes(subtypes),
            m_methods(methods),
            m_hasGetAttributeMagicMethod(false)
    {
        m_name = name;
        m_moduleName = moduleName;
        m_is_simple = false;
        m_hasGetAttributeMagicMethod = m_methods.find("__getattribute__") != m_methods.end();

        if (m_subtypes.size() > 255) {
            throw std::runtime_error("Can't have an alternative with more than 255 subelements");
        }

        m_doc = Alternative_doc;

        // call this _first_, so that our core properties, (which we know) are created
        // before we walk down to the ConcreteAlternatives
        _updateAfterForwardTypesChanged();

        endOfConstructorInitialization(); // finish initializing the type object.
    }

    std::string nameWithModuleConcrete() {
        if (m_moduleName.size() == 0) {
            return m_name;
        }

        return m_moduleName + "." + m_name;
    }

    bool hasGetAttributeMagicMethod() const {
        return m_hasGetAttributeMagicMethod;
    }

    bool isBinaryCompatibleWithConcrete(Type* other);

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        for (auto& subtype_pair: m_subtypes) {
            Type* t = (Type*)subtype_pair.second;
            visitor(t);
            assert(t == subtype_pair.second);
        }
        for (long k = 0; k < m_subtypes.size(); k++) {
            Type* t = concreteSubtype(k);
            visitor(t);
        }

        for (auto& method_pair: m_methods) {
            Type* t = (Type*)method_pair.second;
            visitor(t);
            assert(t == method_pair.second);
        }
    }

    bool _updateAfterForwardTypesChanged();

    ShaHash _computeIdentityHash(MutuallyRecursiveTypeGroup* groupHead = nullptr);

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions);
    static bool cmpStatic(Alternative* altT, instance_ptr left, instance_ptr right, int64_t pyComparisonOp);

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber) {
        buffer.writeBeginSingle(fieldNumber);
        m_subtypes[which(self)].second->serialize(eltPtr(self), buffer, which(self));
    }

    size_t deepBytecountConcrete(instance_ptr instance, std::unordered_set<void*>& alreadyVisited, std::set<Slab*>* outSlabs) {
        if (m_all_alternatives_empty) {
            return 0;
        }

        instance_ptr* p = *(instance_ptr**)instance;

        if (alreadyVisited.find(p) != alreadyVisited.end()) {
            return 0;
        }

        alreadyVisited.insert(p);

        if (outSlabs && Slab::slabForAlloc(p)) {
            outSlabs->insert(Slab::slabForAlloc(p));
            return 0;
        }

        return
            bytesRequiredForAllocation(m_subtypes[which(instance)].second->bytecount() + sizeof(layout)) +
            m_subtypes[which(instance)].second->deepBytecount(eltPtr(instance), alreadyVisited, outSlabs);
    }

    void deepcopyConcrete(
        instance_ptr dest,
        instance_ptr src,
        DeepcopyContext& context
    ) {
        if (m_all_alternatives_empty) {
            ((uint8_t*)dest)[0] = ((uint8_t*)src)[0];
            return;
        }

        layout_ptr& destRecordPtr = *(layout**)dest;
        layout_ptr& srcRecordPtr = *(layout**)src;

        size_t typeIx = srcRecordPtr->which;

        auto it = context.alreadyAllocated.find((instance_ptr)srcRecordPtr);

        if (it == context.alreadyAllocated.end()) {
            destRecordPtr = (layout*)context.slab->allocate(
                sizeof(layout) + m_subtypes[typeIx].second->bytecount(),
                this
            );
            destRecordPtr->refcount = 0;
            destRecordPtr->which = srcRecordPtr->which;

            m_subtypes[typeIx].second->deepcopy(
                destRecordPtr->data,
                srcRecordPtr->data,
                context
            );

            context.alreadyAllocated[(instance_ptr)srcRecordPtr] = (instance_ptr)destRecordPtr;
        } else {
            destRecordPtr = (layout_ptr)it->second;
        }

        destRecordPtr->refcount++;
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t wireType) {
        if (wireType != WireType::SINGLE) {
            throw std::runtime_error("Corrupt data (Alternative expects a SINGLE wire type)");
        }

        std::pair<size_t, size_t> fieldAndWire = buffer.readFieldNumberAndWireType();
        size_t which = fieldAndWire.first;

        if (which >= m_subtypes.size()) {
            throw std::runtime_error("Corrupt data (Alternative field number was out of bounds)");
        }

        if (m_all_alternatives_empty) {
            *(uint8_t*)self = which;
            //still need to consume whatever is in this message
            m_subtypes[which].second->deserialize(self, buffer, fieldAndWire.second);
            return;
        }

        *(layout**)self = (layout*)tp_malloc(
            sizeof(layout) +
            m_subtypes[which].second->bytecount()
            );

        layout& record = **(layout**)self;

        record.refcount = 1;
        record.which = which;

        m_subtypes[which].second->deserialize(record.data, buffer, fieldAndWire.second);
    }

    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr);

    typed_python_hash_type hash(instance_ptr left);

    instance_ptr eltPtr(instance_ptr self) const;

    int64_t which(instance_ptr self) const;

    int64_t refcount(instance_ptr self) const;

    void constructor(instance_ptr self);

    void destroy(instance_ptr self);

    void copy_constructor(instance_ptr self, instance_ptr other);

    void assign(instance_ptr self, instance_ptr other);

    static Alternative* Make(
                        std::string name,
                        std::string moduleName,
                        const std::vector<std::pair<std::string, NamedTuple*> >& types,
                        const std::map<std::string, Function*>& methods //methods preclude us from being in the memo
                        );

    Alternative* renamed(std::string newName) {
        return Make(newName, m_moduleName, m_subtypes, m_methods);
    }

    const std::vector<std::pair<std::string, NamedTuple*> >& subtypes() const {
        return m_subtypes;
    }

    bool isPODConcrete() {
        return m_all_alternatives_empty;
    }

    bool all_alternatives_empty() const {
        return m_all_alternatives_empty;
    }

    Type* pickConcreteSubclassConcrete(instance_ptr data);

    const std::map<std::string, Function*>& getMethods() const {
        return m_methods;
    }

    Type* concreteSubtype(size_t which);

    std::string moduleName() const {
        return m_moduleName;
    }

private:
    //name of the module in which this Alternative was defined.
    std::string m_moduleName;

    bool m_all_alternatives_empty;

    int m_default_construction_ix;

    Type* m_default_construction_type;

    std::vector<std::pair<std::string, NamedTuple*> > m_subtypes;

    std::vector<Type*> m_subtypes_concrete;

    std::map<std::string, Function*> m_methods;

    std::map<std::string, int> m_arg_positions;

    bool m_hasGetAttributeMagicMethod;
};
