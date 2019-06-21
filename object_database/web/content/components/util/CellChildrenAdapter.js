/**
 * A Utility Class that handles creating Cell
 * Component namedChildren based on aggregate
 * replacementKey and other information provided
 * by the server.
 * NOTE: This class is a temporary hack that should
 * be deprecated once replacements are 86'd on the
 * Python side by refactoring each Cell. For now,
 * we need to translate replacement keys into concrete
 * components of the correct cell type, by name.
 */
import {ReplacementsHandler} from './ReplacementsHandler';
import {Component} from '../Component';

class CellChildrenAdapter {
    constructor(rawChildren){
        this.rawChildren = rawChildren;
        this.childrenAsDict = {};
        this.replacements = null;
        this._processRawChildren();


        // Bind methods
        this._processRawChildren = this._processRawChildren.bind(this);
        this.getRawChildByReplacement = this.getRawChildByReplacement.bind(this);
        this.getNamedChildren = this.getNamedChildren.bind(this);
    }

    getRawChildByReplacement(aString){
        let found = this.childrenAsDict[aString];
        if(found == undefined){
            return null;
        }
        return found;
    }

    /**
     * Respond with a dictionary of named children
     * that is isomorphic to the regular/enumerated
     * replacements, but with the child description object
     * as the ultimate value rather than the replacement key.
     */
    getNamedChildren(){
        let result = {};
        let regularNames = Object.keys(this.replacements.regular);
        let enumeratedNames = Object.keys(this.replacements.enumerated);
        regularNames.forEach(regularName => {
            result[regularName] = this.getRawChildByReplacement(this.replacements.regular[regularName]);
        });
        enumeratedNames.forEach(enumeratedName => {
            let enumeratedValue = this.replacements.mapReplacementsFor(enumeratedName, (replacementName) => {
                return this.getRawChildByReplacement(replacementName);
            });
            result[enumeratedName] = enumeratedValue;
        });
        return result;
    }

    _processRawChildren(){
        // rawChildren is an array of objects
        // that describe each child, and in the case
        // of the adapter, should have a property called
        // replacementKey.
        // Here we create a dict that references each child
        // description by the replacementKey
        this.rawChildren.forEach(childDescription => {
            this.childrenAsDict[childDescription.replacementKey] = childDescription;
        });
        this.allReplacementKeys = this.rawChildren.map(childDescription => {
            return childDescription.replacementKey;
        });
        this.replacements = new ReplacementsHandler(this.allReplacementKeys);
    }
}

const updateDescriptionStructure = function(cellDescription){
    if(cellDescription.children && cellDescription.children.length > 0){
        let adapter = new CellChildrenAdapter(cellDescription.children);
        cellDescription.namedChildren = adapter.getNamedChildren();
        cellDescription.children.forEach(child => {
            updateDescriptionStructure(child);
        });
        cellDescription.children = true;
    } else {
        cellDescription.namedChildren = null;
        cellDescription.children = false;
    }
};

const componentsFromData = function(sourceData, componentRegistry){
    updateDescriptionStructure(sourceData.structure);
    let _recur = function(child, childName, parentDict){
        if(Array.isArray(child)){
            parentDict[childName] = child.map(childItem => {

            })
        }
        if(!componentRegistry.hasOwnProperty(child.cellType)){
            console.log(child, childName, parentDict);
            throw new Error(`Could not find component for cell of type ${child.cellType}`);
        }
        let builder = componentRegistry[child.cellType];
        let myOwnChildren = {};
        if(child.namedChildren){
            Object.keys(child.namedChildren).forEach(myOwnChildName => {
                let myOwnChild = child.namedChildren[myOwnChildName];
                _recur(myOwnChild, myOwnChildName, myOwnChildren);
            });
        }

        let myComponent = builder({
            id: child.id,
            namedChildren: myOwnChildren
        });
        parentDict[childName] = myComponent;
        return myComponent;
    };
    let rootChildren = {};
    if(sourceData.structure.children){
        Object.keys(sourceData.structure.namedChildren).forEach(childName => {
            let child = sourceData.structure.namedChildren[childName];
            _recur(child, childName, rootChildren);
        });
    }
    return new Component({
        id: sourceData.structure.id,
        namedChildren: rootChildren
    });
}

export {
    updateDescriptionStructure,
    componentsFromData,
    CellChildrenAdapter,
    CellChildrenAdapter as default
};
