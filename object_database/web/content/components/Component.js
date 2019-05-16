/**
 * Generic base Cell Component.
 * Should be extended by other
 * Cell classes on JS side.
 */

// NOTE: For the moment we assume global
// availability of the `h` hyperscript
// constructor.
//import {h} from 'maquette';

class ReplacementsHandler {
    constructor(initialReplacements = []){
        this.replacementNames = initialReplacements;
        this.processReplacements();

        // Bind context to methods
        this.processReplacements = this.processReplacements.bind(this);
        this.addReplacement = this.addReplacement.bind(this);
        this._processEnumerated = this._processEnumerated.bind(this);
        this._sortEnumeratedValues = this._sortEnumeratedValues.bind(this);
    }

    addReplacement(replacementName){
        if(this.replacementNames.includes(replacementName)){
            return;
        }
        this.replacementNames.push(replacementName);
        this.processReplacements();
    }

    getReplacement(replacementName){
        // Checks the plain dict first and
        // if nothing is present, tries
        // the enumerated value dict.
        // Will return undefined if both fail
        let initial = this.replacementDict[replacementName];
        if(initial != undefined){
            return initial;
        }
        initial = this.enumeratedReplacementDict[replacementName];
        return initial;
    }

    processReplacements(){
        this.replacementDict = {};
        this.enumeratedReplacementDict = {};
        this.replacementNames.forEach(name => {
            let nameParts = [];
            let isEnumerated = false;
            let enumeratedVal = null;

            // First, remove all the underscores
            // to get a list of real tokens
            let tokens = name.split('_').filter(item => {
                return item != '';
            });

            // See if any of the tokens are numbers.
            // If not, they are hyphenated name parts
            // that can be used as keys. Otherwise
            // this is an enumerated kind of replacement
            // key like `____button_1__` etc/
            tokens.forEach(token => {
                let num = parseInt(token);
                if(isNaN(num)){
                    nameParts.push(token);
                } else {
                    isEnumerated = true;
                    enumeratedVal = num;
                }
            });

            if(nameParts.length == 0){
                throw Error(`Could not process replacement name for ${name}`);
            }

            let replacementKey = nameParts.join('-');
            if(isEnumerated){
                this._processEnumerated(name, replacementKey, enumeratedVal);
            } else {
                this.replacementDict[replacementKey] = name;
            }
        });

        // Sort the enumerated dict values by the index,
        // ripping those indices out. This ensures that
        // each enumerated list of replacements is in the
        // correct order
        this._sortEnumeratedValues();
    }

    _processEnumerated(name, replacementKey, index){
        if(this.enumeratedReplacementDict[replacementKey] == undefined){
            this.enumeratedReplacementDict[replacementKey] = [];
        }
        this.enumeratedReplacementDict[replacementKey].push([name, index]);
    }

    _sortEnumeratedValues(){
        Object.keys(this.enumeratedReplacementDict).forEach(key => {
            let unsorted = this.enumeratedReplacementDict[key];
            unsorted.sort((first, second) => {
                return first[1] - second[1];
            });

            // Now set the array to a version that filters out
            // the index values (we no longer need them)
            this.enumeratedReplacementDict[key] = unsorted.map(item => {
                return item[0]; // The string (item[1] is the index)
            });
        });
    }
}

class Component {
    constructor(props = {}, children = [], replacements = []){
        this.props = props;
        this.children = children;
        this.replacements = new ReplacementsHandler(replacements);

        // Bind context to methods
        this.getReplacementWithId = this.getReplacementWithId.bind(this);
        this.getReplacementElementFor = this.getReplacementElementFor.bind(this);
        this.getReplacementElementsFor = this.getReplacementElementsFor.bind(this);
    }

    render(){
        // Objects that extend from
        // me should override this
        // method in order to generate
        // some content for the vdom
        throw new Error('You must implement a `render` method on Component objects!');
    }

    /**
     * This is a hacky method that allows extended
     * components to deal cleanly with the replacement
     * strings that it will use.
     * Eventually we want to get rid of this.
     */
    getReplacementWithId(replacementName){
        let found = this.replacements.getReplacement(replacementName);
        if(found == undefined){
            return found;
        }

        if(Array.isArray(found)){
            return found.map(name => {
                return `${this.props.id}_${name}`;
            });
        }

        return `${this.props.id}_${found}`;
    }

    /**
     * Respond with a hyperscript object
     * with an ID-formatted div mapped
     * with the replacement string.
     * This is for single replacements only.
     * Replacement lists (multiple indexed
     * replacements) should use the
     * `getReplacementElementsFor()` method.
     */
    getReplacementElementFor(replacementName){
        let replacementId = this.getReplacementWithId(replacementName);
        if(replacementId == undefined){
            return null;
        } else if(Array.isArray(replacementId)){
            return null;
        }
        return h('div', {id: replacementId}, []);
    }

    /**
     * Respond with an array of hyperscript
     * objects that are divs with ids that match
     * replacement string ids for the kind of
     * replacement list that is enumerated,
     * ie `____button_1`, `____button_2__` etc.
     */
    getReplacementElementsFor(replacementName){
        let replacementIds = this.getReplacementWithId(replacementName);
        if(!Array.isArray(replacementIds)){
            return null;
        }
        return replacementIds.map(replacementId => {
            return h('div', {id: replacementId}, []);
        });
    }
}
