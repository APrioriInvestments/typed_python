class ReplacementsHandler {
    constructor(replacements){
        this.replacements = replacements;
        this.regular = {};
        this.enumerated = {};

        if(replacements){
            this.processReplacements();
        }

        // Bind context to methods
        this.processReplacements = this.processReplacements.bind(this);
        this.processRegular = this.processRegular.bind(this);
        this.hasReplacement = this.hasReplacement.bind(this);
        this.getReplacementFor = this.getReplacementFor.bind(this);
        this.getReplacementsFor = this.getReplacementsFor.bind(this);
        this.mapReplacementsFor = this.mapReplacementsFor.bind(this);
    }

    processReplacements(){
        this.replacements.forEach(replacement => {
            let replacementInfo = this.constructor.readReplacementString(replacement);
            if(replacementInfo.isEnumerated){
                this.processEnumerated(replacement, replacementInfo);
            } else {
                this.processRegular(replacement, replacementInfo);
            }
        });
        // Now we update this.enumerated to have it's top level
        // values as Arrays instead of nested dicts and we sort
        // based on the extracted indices (which are at this point
        // just keys on subdicts or multidimensional dicts)
        Object.keys(this.enumerated).forEach(key => {
            let enumeratedReplacements = this.enumerated[key];
            this.enumerated[key] = this.constructor.enumeratedValToSortedArray(enumeratedReplacements);
        });
    }

    processRegular(replacementName, replacementInfo){
        let replacementKey = this.constructor.keyFromNameParts(replacementInfo.nameParts);
        this.regular[replacementKey] = replacementName;
    }

    processEnumerated(replacementName, replacementInfo){
        let replacementKey = this.constructor.keyFromNameParts(replacementInfo.nameParts);
        let currentEntry = this.enumerated[replacementKey];

        // If it's undefined, this is the first
        // of the enumerated replacements for this
        // key, ie something like ____child_0__
        if(currentEntry == undefined){
            this.enumerated[replacementKey] = {};
            currentEntry = this.enumerated[replacementKey];
        }

        // We add the enumerated indices as keys to a dict
        // and we do this recursively across dimensions as
        // needed.
        this.constructor.processDimension(replacementInfo.enumeratedValues, currentEntry, replacementName);
    }

    // Accessing and other Convenience Methods
    hasReplacement(aReplacementName){
        if(this.regular.hasOwnProperty(aReplacementName)){
            return true;
        } else if(this.enumerated.hasOwnProperty(aReplacementName)){
            return true;
        }
        return false;
    }

    getReplacementFor(aReplacementName){
        let found = this.regular[aReplacementName];
        if(found == undefined){
            return null;
        }
        return found;
    }

    getReplacementsFor(aReplacementName){
        let found = this.enumerated[aReplacementName];
        if(found == undefined){
            return null;
        }
        return found;
    }

    mapReplacementsFor(aReplacementName, mapFunction){
        if(!this.hasReplacement(aReplacementName)){
            throw new Error(`Invalid replacement name: ${aReplacementname}`);
        }
        let root = this.getReplacementsFor(aReplacementName);
        return this._recursivelyMap(root, mapFunction);
    }

    _recursivelyMap(currentItem, mapFunction){
        if(!Array.isArray(currentItem)){
            return mapFunction(currentItem);
        }
        return currentItem.map(subItem => {
            return this._recursivelyMap(subItem, mapFunction);
        });
    }

    // Static helpers
    static processDimension(remainingVals, inDict, replacementName){
        if(remainingVals.length == 1){
            inDict[remainingVals[0]] = replacementName;
            return;
        }
        let nextKey = remainingVals[0];
        let nextDict = inDict[nextKey];
        if(nextDict == undefined){
            inDict[nextKey] = {};
            nextDict = inDict[nextKey];
        }
        this.processDimension(remainingVals.slice(1), nextDict, replacementName);
    }

    static enumeratedValToSortedArray(aDict, accumulate = []){
        if(typeof aDict !== 'object'){
            return aDict;
        }
        let sortedKeys = Object.keys(aDict).sort((first, second) => {
            return (parseInt(first) - parseInt(second));
        });
        let subEntries = sortedKeys.map(key => {
            let entry = aDict[key];
            return this.enumeratedValToSortedArray(entry);
        });
        return subEntries;
    }

    static keyFromNameParts(nameParts){
        return nameParts.join("-");
    }

    static readReplacementString(replacement){
        let nameParts = [];
        let isEnumerated = false;
        let enumeratedValues = [];
        let pieces = replacement.split('_').filter(item => {
            return item != '';
        });
        pieces.forEach(piece => {
            let num = parseInt(piece);
            if(isNaN(num)){
                nameParts.push(piece);
        } else {
            isEnumerated = true;
            enumeratedValues.push(num);
        }
        });
        return {
            nameParts,
            isEnumerated,
            enumeratedValues
        };
    }
}

export {
    ReplacementsHandler,
    ReplacementsHandler as default
};
