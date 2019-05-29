/**
 * Generic base Cell Component.
 * Should be extended by other
 * Cell classes on JS side.
 */

// NOTE: For the moment we assume global
// availability of the `h` hyperscript
// constructor and the ReplacementsHandler util class.
//import {h} from 'maquette';
//import {ReplacementsHandler} from './util/ReplacementsHandler';

class Component {
    constructor(props = {}, children = [], replacements = []){
        this.props = props;
        this.children = children;
        this.replacements = new ReplacementsHandler(replacements);

        // Bind context to methods
        this.getReplacementElementFor = this.getReplacementElementFor.bind(this);
        this.getReplacementElementsFor = this.getReplacementElementsFor.bind(this);
        this.componentDidLoad = this.componentDidLoad.bind(this);
    }

    render(){
        // Objects that extend from
        // me should override this
        // method in order to generate
        // some content for the vdom
        throw new Error('You must implement a `render` method on Component objects!');
    }

    /**
     * Object that extend from me could overwrite this method.
     * It is to be used for lifecylce management and is to be called
     * after the components loads.
    */
    componentDidLoad() {
        return null;
    }
    /**
     * Responds with a hyperscript object
     * that represents a div that is formatted
     * already for the regular replacement.
     * This only works for regular type replacements.
     * For enumerated replacements, use
     * #getReplacementElementsFor()
     */
    getReplacementElementFor(replacementName){
        let replacement = this.replacements.getReplacementFor(replacementName);
        if(replacement){
            let newId = `${this.props.id}_${replacement}`;
            return h('div', {id: newId, key: newId, 'data-maq-key': newId}, []);
        }
        return null;
    }

    /**
     * Respond with an array of hyperscript
     * objects that are divs with ids that match
     * replacement string ids for the kind of
     * replacement list that is enumerated,
     * ie `____button_1`, `____button_2__` etc.
     */
    getReplacementElementsFor(replacementName){
        if(!this.replacements.hasReplacement(replacementName)){
            return null;
        }
        return this.replacements.mapReplacementsFor(replacementName, replacement => {
            let newId = `${this.props.id}_${replacement}`;
            return (
                h('div', {id: newId, key: newId, 'data-maq-key': newId})
            );
        });
    }
}
