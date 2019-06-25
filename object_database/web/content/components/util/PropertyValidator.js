/**
 * Tool for Validating Component Properties
 */

const report = (message, errorMode, silentMode) => {
    if(errorMode == true && silentMode == false){
        console.error(message);
    } else if(silentMode == false){
        console.warn(message);
    }
};

const PropTypes = {
    errorMode: false,
    silentMode: false,
    oneOf: function(anArray){
        return function(componentName, propName, propValue, isRequired){
            for(let i = 0; i < anArray.length; i++){
                let typeCheckItem = anArray[i];
                if(typeof(typeCheckItem) == 'function'){
                    if(typeCheckItem(componentName, propName, propValue, isRequired, true)){
                        return true;
                    }
                } else if(typeCheckItem == propValue){
                    return true;
                }
            }
            let message = `${componentName} >> ${propName} must be of one of the following types: ${anArray}`;
            report(message, this.errorMode, this.silentMode);
            return false;
        }.bind(this);
    },

    getValidatorForType(typeStr){
        return function(componentName, propName, propValue, isRequired, inCompound = false){
            // We are 'in a compound validation' when the individual
            // PropType checkers (ie func, number, string, etc) are
            // being called within a compound type checker like oneOf.
            // In these cases we want to prevent the call to report,
            // which the compound check will handle on its own.
            if(inCompound == false){
                if(typeof(propValue) == typeStr){
                    return true;
                } else if(!isRequired && (propValue == undefined || propValue == null)){
                    return true;
                } else if(isRequired){
                    let message = `${componentName} >> ${propName} is a required prop, but as passed as ${propValue}!`;
                    report(message, this.errorMode, this.silentMode);
                    return false;
                } else {
                    let message = `${componentName} >> ${propName} must be of type ${typeStr}!`;
                    report(message, this.errorMode, this.silentMode);
                    return false;
                }
            // Otherwise this is a straightforward type check
            // based on the given type. We check as usual for the required
            // property and then the actual type match if needed.
            } else {
                if(isRequired && (propValue == undefined || propValue == null)){
                    let message = `${componentName} >> ${propName} is a required prop, but was passed as ${propValue}!`;
                    report(message, this.errorMode, this.silentMode);
                    return false;
                } else if(!isRequired && (propValue == undefined || propValue == null)){
                    return true;
                }
                return typeof(propValue) == typeStr;
            }
        };
    },

    get number(){
        return this.getValidatorForType('number').bind(this);
    },

    get boolean(){
        return this.getValidatorForType('boolean').bind(this);
    },

    get string(){
        return this.getValidatorForType('string').bind(this);
    },

    get object(){
        return this.getValidatorForType('object').bind(this);
    },

    get func(){
        return this.getValidatorForType('function').bind(this);
    },

    validate: function(componentName, props, propInfo){
        let propNames = new Set(Object.keys(props));
        propNames.delete('children');
        propNames.delete('namedChildren');
        propNames.delete('id');
        propNames.delete('extraData'); // For now
        let propsToValidate = Array.from(propNames);

        // Perform all the validations on each property
        // according to its description. We store whether
        // or not the given property was completely valid
        // and then evaluate the validity of all at the end.
        let validationResults = {};
        propsToValidate.forEach(propName => {
            let propVal = props[propName];
            let validationToCheck = propInfo[propName];
            if(validationToCheck){
                let hasValidDescription = this.validateDescription(componentName, propName, validationToCheck);
                let hasValidPropTypes = validationToCheck.type(componentName, propName, propVal, validationToCheck.required);
                if(hasValidDescription && hasValidPropTypes){
                    validationResults[propName] = true;
                } else {
                    validationResults[propName] = false;
                }
            } else {
                // If we get here, the consumer has passed in a prop
                // that is not present in the propTypes description.
                // We report to the console as needed and validate as false.
                let message = `${componentName} has a prop called "${propName}" that is not described in propTypes!`;
                report(message, this.errorMode, this.silentMode);
                validationResults[propName] = false;
            }
        });

        // If there were any that did not validate, return
        // false and report as much.
        let invalids = [];
        Object.keys(validationResults).forEach(key => {
            if(validationResults[key] == false){
                invalids.push(key);
            }
        });
        if(invalids.length > 0){
            return false;
        } else {
            return true;
        }
    },

    validateRequired: function(componentName, propName, propVal, isRequired){
        if(isRequired == true){
            if(propVal == null || propVal == undefined){
                let message = `${componentName} >> ${propName} requires a value, but ${propVal} was passed!`;
                report(message, this.errorMode, this.silentMode);
                return false;
            }
        }
        return true;
    },

    validateDescription: function(componentName, propName, prop){
        let desc = prop.description;
        if(desc == undefined || desc == "" || desc == null){
            let message = `${componentName} >> ${propName} has an empty description!`;
            report(message, this.errorMode, this.silentMode);
            return false;
        }
        return true;
    }
};

export {
    PropTypes
};


/***
number: function(componentName, propName, propValue, inCompound = false){
        if(inCompound == false){
            if(typeof(propValue) == 'number'){
                return true;
            } else {
                let message = `${componentName} >> ${propName} must be of type number!`;
                report(message, this.errorMode, this.silentMode);
                return false;
            }
        } else {
            return typeof(propValue) == 'number';
        }

    }.bind(this),

    string: function(componentName, propName, propValue, inCompound = false){
        if(inCompound == false){
            if(typeof(propValue) == 'string'){
                return true;
            } else {
                let message = `${componentName} >> ${propName} must be of type string!`;
                report(message, this.errorMode, this.silentMode);
                return false;
            }
        } else {
            return typeof(propValue) == 'string';
        }
    }.bind(this),

    boolean: function(componentName, propName, propValue, inCompound = false){
        if(inCompound == false){
            if(typeof(propValue) == 'boolean'){
                return true;
            } else {
                let message = `${componentName} >> ${propName} must be of type boolean!`;
                report(message, this.errorMode, this.silentMode);
                return false;
            }
        } else {
            return typeof(propValue) == 'boolean';
        }
    }.bind(this),

    object: function(componentName, propName, propValue, inCompound = false){
        if(inCompound == false){
            if(typeof(propValue) == 'object'){
                return true;
            } else {
                let message = `${componentName} >> ${propName} must be of type object!`;
                report(message, this.errorMode, this.silentMode);
                return false;
            }
        } else {
            return typeof(propValue) == 'object';
        }
    }.bind(this),

    func: function(componentName, propName, propValue, inCompound = false){
        if(inCompound == false){
            if(typeof(propValue) == 'function'){
                return true;
            } else {
                let message = `${componentName} >> ${propName} must be of type function!`;
                report(message, this.errorMode, this.silentMode);
                return false;
            }
        } else {
            return typeof(propValue) == 'function';
        }
    }.bind(this),

***/
