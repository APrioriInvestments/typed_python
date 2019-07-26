/**
 * Tests for Message Handling in NewCellHandler
 */
require('jsdom-global')();
const maquette = require('maquette');
const h = maquette.h;
const NewCellHandler = require('../NewCellHandler.js').default;
const chai = require('chai');
const assert = chai.assert;
let projector = maquette.createProjector();
const registry = require('../ComponentRegistry').ComponentRegistry;

/* Example Messages and Structures */
let simpleRoot = {
    id: "page_root",
    cellType: "RootCell",
    parentId: null,
    nameInParent: null,
    extraData: {},
    namedChildren: {}
};

let simpleRootUpdate = Object.assign({}, simpleRoot, {
    type: "#cellUpdated",
    channel: "#main",
    shouldDisplay: true
});

let simpleRootStructure = {
    id: "page_root",
    cellType: "RootCell",
    parentId: null,
    nameInParent: null,
    extraData: {},
    namedChildren: {
        "child": {
            id: 2,
            cellType: "Sequence",
            parentId: 1,
            nameInParent: "child",
            extraData: {},
            namedChildren: {
                "elements": [
                    {
                        id: 3,
                        cellType: "Text",
                        parentId: 2,
                        nameInParent: 'elements',
                        extraData: {
                            rawText: "Hello"
                        },
                        namedChildren: {}
                    },
                    {
                        id: 4,
                        cellType: "Text",
                        parentId: 2,
                        nameInParent: "elements",
                        extraData: {
                            rawText: "World"
                        },
                        namedChildren: {}
                    }
                ]
            }
        }
    }
};

let simpleUpdate = Object.assign({}, simpleRootStructure, {
    type: "#cellUpdated",
    channel: "#main",
    shouldDisplay: true
});

describe("Basic NewCellHandler Tests", () => {
    it('Should be able to initialize', () => {
        let instance = new NewCellHandler(h, projector, registry);
        assert.exists(instance);
    });
    it('Has the passed in projector', () => {
        let handler = new NewCellHandler(h, projector, registry);
        assert.equal(projector, handler.projector);
    });
    it('Has the passed in hyperscript constructor', () => {
        let handler = new NewCellHandler(h, projector, registry);
        assert.equal(h, handler.h);
    });
});

describe("Basic Test DOM Tests", () => {
    it("Has a document and body", () => {
        assert.exists(document.body);
    });
    it("Can create and append root element", () => {
        let root = document.createElement('div');
        root.id = "page_root";
        document.body.append(root);
        let found = document.getElementById('page_root');
        assert.exists(found);
        assert.equal(found, root);
    });

    after(() => {
        let root = document.getElementById('page_root');
        if(root){
            root.remove();
        }
    });

    describe("Basic Structure Handling Tests", () => {
        var handler;
        before(() => {
            let root = document.createElement('div');
            root.id = "page_root";
            document.body.append(root);
        });

        beforeEach(() => {
            handler = new NewCellHandler(h, projector, registry);
        });

        it("Receives the update and stores new RootCell instance", () => {
            handler.receive(simpleRootUpdate);
            let stored = handler.activeComponents[simpleRoot.id];
            assert.exists(stored);
        });
        it("Updates the new Component by rendering properly to the DOM", () => {
            handler.receive(simpleRootUpdate);
            let pageRoot = document.getElementById('page_root');
            let inlineType = pageRoot.dataset.cellType;
            console.log(pageRoot.outerHTML);
            assert.exists(inlineType);
            assert.equal(inlineType, "RootCell");
        });
    });
});
