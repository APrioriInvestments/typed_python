/**
 * Mocha Tests for Base Component Class
 */
var Component = require('../Component.js').Component;
var chai = require('chai');
var h = require('maquette').h;
var assert = chai.assert;
var AllComponents = require('../../ComponentRegistry.js').ComponentRegistry;

class SubComponent extends Component {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return (
            h('div', {id: this.props.id}, [`Child: ${this.props.id}`])
        );
    }
};

describe("Base Component Class", () => {
    describe('Base Component Construction', () => {
        it('Should error if no id is passed with props', () => {
            let fn = function(){
                return new Component({});
            };
            assert.throws(fn, Error);
        });
        it('Should construct with an empty dict of namedChildren if none are passed', () => {
            let instance = new SubComponent({id: 'subcomponent'});
            assert.exists(instance.props.namedChildren);
            assert.typeOf(instance.props.namedChildren, 'object');
        });
        it('Sets usesReplacements to true if replacement are passed', () => {
            let instance = new SubComponent({
                id: 'testComponent'
            }, ['____contents__', '____item_0__', '____item_1__']);
            assert.isTrue(instance.usesReplacements);
        });
        it('Sets usesReplacements to false if replacements are not passed', () => {
            let instance = new SubComponent({
                id: 'testcomponent'
            });
            assert.isFalse(instance.usesReplacements);
        });
    });

    describe('Base Component Relationships', () => {
        it('Plain components have no parent', () => {
            let instance = new SubComponent({id: 'hello'});
            assert.isNull(instance.parent);
        });
        it('Array Child components have parent set correctly', () => {
            let arrayChildren = [
                new SubComponent({id: 'array-child-1'}),
                new SubComponent({id: 'array-child-2'})
            ];
            let parent = new SubComponent({
                id: 'parent',
                children: arrayChildren
            });
            arrayChildren.forEach(arrayChild => {
                assert.equal(arrayChild.parent, parent);
            });
        });
        it('Named Children components have parent set correctly', () => {
            let namedChildren = {
                'namedChild1': new SubComponent({id: 'namedChild1'}),
                'namedChild2': new SubComponent({id: 'namedChild2'})
            };
            let parent = new SubComponent({
                id: 'parent',
                namedChildren: namedChildren
            });
            Object.keys(namedChildren).forEach(key => {
                let child = namedChildren[key];
                assert.equal(child.parent, parent);
            });
        });
    });

    describe('Base Component Accessors', () => {
        it('#name responds with the correct name', () => {
            let instance = new SubComponent({id: 'subcomponent'});
            assert.equal('SubComponent', instance.name);
        });
    });

    describe('Base Component Children Utilities', () => {
        it('#renderedChildren provides hyperscripts for children', () => {
            let parent = new Component({
                id:'parent1',
                children: [
                    new SubComponent({id: 'child1'}),
                    new SubComponent({id: 'child2'}),
                    new SubComponent({id: 'child3'})
                ]
            });
            let result = parent.renderedChildren;
            assert.lengthOf(result, 3);
        });
        it('#renderedChildren result objects have appropriately set keys', () => {
            let parent = new Component({
                id:'parent1',
                children: [
                    new SubComponent({id: 'child1'}),
                    new SubComponent({id: 'child2'}),
                    new SubComponent({id: 'child3'})
                ]
            });
            let result = parent.renderedChildren;
            result.forEach((childHyperscript) => {
                let id = childHyperscript.properties.id;
                assert.propertyVal(childHyperscript.properties, 'key', `parent1-child-${id}`);
            });
        });
        it('#renderChildNamed returns hyperscript from child component', () => {
            let child = new SubComponent({id: 'foo'});
            let parent = new Component({
                id: 'parent',
                namedChildren: {'mainChild': child}
            });
            let result = parent.renderChildNamed('mainChild');
            assert.exists(result.properties);
        });
    });

    /*describe('Extended Components Validation', () => {
      Object.keys(AllComponents).forEach(aComponentName => {
      let aComponent = AllComponents[aComponentName];
      describe(`${aComponent.name} Validation`, () => {
      it('Renders with the passed-in id in the top level hyperscript', () => {
      let instance = new aComponent({id:'this-component', extraData: {}});
      let rendered = instance.render();
      assert.equal(rendered.properties.id, 'this-component');
      });
      });
      });
    });*/
});
