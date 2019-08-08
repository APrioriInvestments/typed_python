/**
 * Utility Functions for Dealing with NamedChildren structures
 * Note that for the moment, we are using these
 * functions minimally throughout the code.
 */

function _resursivelySearchArrayChildren(target, haystack, info){
    if(Array.isArray(haystack)){
        info.inArray = haystack;
        haystack.forEach(item => {
            _resursivelySearchArrayChildren(target, item, info);
        });
    } else if(target == haystack){
        info.found = true;
    }
    return info;
}

const findNamedComponent = (target, parent) => {
    let response = {
        nameInParent: null,
        found: false,
        inArray: null
    };
    let complex = [];
    let childNames = Object.keys(parent.props.namedChildren);

    // First, we check regular (ie key maps to component)
    // children while, at the same time, we sort out
    // children that are array structures and store
    // their names separately.
    // We return if we find a match here.
    for(let i = 0; i < childNames.length; i++){
        let childName = childNames[i];
        let childStruct = parent.props.namedChildren[childName];
        if(Array.isArray(childStruct)){
            complex.push(childName);
        } else if(childStruct == target){
            response.nameInParent = childName;
            response.found = true;
            return response;
        }
    }

    // If we get here, we haven't found the target
    // yet in regular children, so we need to recursively
    // go through array children and see if we can get a
    // match.
    for(let j = 0; j < childNames.length; j++){
        let childName = childNames[j];
        let struct = parent.props.namedChildren[childName];
        response.nameInParent = childName;
        let result = _resursivelySearchArrayChildren(
            target,
            struct,
            response
        );
        if(result.found){
            return response;
        }
    }
    return response;
};

function _recursiveChildrenDo(childOrArray, callback){
    if(Array.isArray(childOrArray)){
        childOrArray.forEach(item => {
            _recursiveChildrenDo(item, callback);
        });
    } else if(childOrArray && childOrArray != undefined) {
        callback(childOrArray);
    }
}

const allNamedChildrenDo = (component, callback) => {
    let namedChildren = component.props.namedChildren;
    Object.keys(namedChildren).forEach(childName => {
        let childOrArray = namedChildren[childName];
        if(Array.isArray(childOrArray)){
            _recursiveChildrenDo(childOrArray, callback);
        } else if(childOrArray && childOrArray != undefined) {
            callback(childOrArray);
        }
    });
};

export {findNamedComponent, allNamedChildrenDo};
