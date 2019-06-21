var Component = require('../Component.js').Component;
var Card = require('../Card.js').Card;
var h = require('maquette').h;
var assert = require('chai').assert;

class FakeHeader extends Component {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return (
            h('p', ["CARD HEADER"])
        );
    }
}

class FakeContent extends Component {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return (
            h('p', {}, ["THIS IS THE BODY"])
        );
    }
}

describe('Card Component Tests', () => {
    describe('Children Handling', () => {
        it('Can render named child: contents', () => {
            let parent = new Card({
                id: 'parent-card',
                namedChildren: {'contents': new FakeContent({id: 'card-content'})}
            });
            let result = parent.renderChildNamed('contents');
            assert.exists(result.properties);
            assert.equal(result.text, "THIS IS THE BODY");
        });
        it('Can render named child: header', () => {
            let parent = new Card({
                id: 'parent-card',
                namedChildren: {
                    'contents': new FakeContent({id: '1'}),
                    'header': new FakeHeader({id: '2'})
                }
            });
            result = parent.renderChildNamed('header');
            assert.equal(result.text, "CARD HEADER");
        });
        it('Renders both header and contents', () => {
            let parent = new Card({
                id: 'parent-card',
                extraData: {},
                namedChildren: {
                    'contents': new FakeContent({id: 'content'}),
                    'header': new FakeHeader({id: 'header'})
                }
            });
            let result = parent.render();
            assert.exists(result.children[0].children); // header
            assert.exists(result.children[1].children[0]); // contents
        });
        it('Renders just body when no header is passed', () => {
            let parent = new Card({
                id: 'parent-card',
                extraData: {},
                namedChildren: {
                    'contents': new FakeContent({id: 'content'})
                }
            });
            let result = parent.render();
            assert.lengthOf(result.children, 1);
        });
    });
});
