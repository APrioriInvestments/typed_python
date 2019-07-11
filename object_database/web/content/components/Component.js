/**
 * Generic base Cell Component.
 * Should be extended by other
 * Cell classes on JS side.
 */
import {ReplacementsHandler} from './util/ReplacementsHandler';
import {KeyListener} from './util/KeyListener';
import {PropTypes} from './util/PropertyValidator';
import {h} from 'maquette';

class Component {
    constructor(props = {}, replacements = []){
        this.isComponent = true;
        this._updateProps(props);

        // Replacements handling
        this.replacements = new ReplacementsHandler(replacements);
        this.usesReplacements = (replacements.length > 0);

        // Setup parent relationship, if
        // any. In this abstract class
        // there isn't one by default
        this.parent = null;
        this._setupChildRelationships();

        // Ensure that we have passed in an id
        // with the props. Should error otherwise.
        if(!this.props.id || this.props.id == undefined){
            throw Error('You must define an id for every component props!');
        }

        this.validateProps();

        // Bind context to methods
        this.getReplacementElementFor = this.getReplacementElementFor.bind(this);
        this.getReplacementElementsFor = this.getReplacementElementsFor.bind(this);
        this.componentDidLoad = this.componentDidLoad.bind(this);
        this.childrenDo = this.childrenDo.bind(this);
        this.namedChildrenDo = this.namedChildrenDo.bind(this);
        this.renderChildNamed = this.renderChildNamed.bind(this);
        this.renderChildrenNamed = this.renderChildrenNamed.bind(this);
        this._setupChildRelationships = this._setupChildRelationships.bind(this);
        this._updateProps = this._updateProps.bind(this);
        this._recursivelyMapNamedChildren = this._recursivelyMapNamedChildren.bind(this);
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
     * Lifecycle function that will be called just before
     * the component is removed from the global handler's
     * dict of components and also from the DOM.
     */
    componentWillUnload(){
        return null;
    }

    /**
     * Will be called on subsequent changes to the props
     * for the component. Needs to return a new props
     * object.
     * By default we simply return the passed-in
     * nextProps
     */
    componentWillReceiveProps(oldProps, nextProps){
        return nextProps;
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
            return h('div', {id: newId, key: newId}, []);
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
            return [];
        }
        return this.replacements.mapReplacementsFor(replacementName, replacement => {
            let newId = `${this.props.id}_${replacement}`;
            return (
                h('div', {id: newId, key: newId})
            );
        });
    }

    /**
     * If there is a `propTypes` object present on
     * the constructor (ie the component class),
     * then run the PropTypes validator on it.
     */
    validateProps(){
        if(this.constructor.propTypes){
            PropTypes.validate(
                this.constructor.name,
                this.props,
                this.constructor.propTypes
            );
        }
    }

    /**
     * Looks up the passed key in namedChildren and
     * if found responds with the result of calling
     * render on that child component. Returns null
     * otherwise.
     */
    renderChildNamed(key){
        let foundChild = this.props.namedChildren[key];
        if(foundChild){
            return foundChild.render();
        }
        return null;
    }

    /**
     * Looks up the passed key in namedChildren
     * and if found -- and the value is an Array
     * or Array of Arrays, responds with an
     * isomorphic structure that has the rendered
     * values of each component.
     */
    renderChildrenNamed(key){
        let foundChildren = this.props.namedChildren[key];
        if(foundChildren){
            return this._recursivelyMapNamedChildren(foundChildren, child => {
                return child.render();
            });
        }
        return [];
    }



    /**
     * Getter that will respond with the
     * constructor's (aka the 'class') name
     */
    get name(){
        return this.constructor.name;
    }

    /**
     * Getter that will respond with an
     * array of rendered (ie configured
     * hyperscript) objects that represent
     * each child. Note that we will create keys
     * for these based on the ID of this parent
     * component.
     */
    get renderedChildren(){
        if(this.props.children.length == 0){
            return [];
        }
        return this.props.children.map(childComponent => {
            let renderedChild = childComponent.render();
            renderedChild.properties.key = `${this.props.id}-child-${childComponent.props.id}`;
            return renderedChild;
        });
    }

    /** Public Util Methods **/

    /**
     * Calls the provided callback on each
     * array child for this component, with
     * the child as the sole arg to the
     * callback
     */
    childrenDo(callback){
        this.props.children.forEach(child => {
            callback(child);
        });
    }

    /**
     * Calls the provided callback on
     * each named child with key, child
     * as the two args to the callback.
     */
    namedChildrenDo(callback){
        Object.keys(this.props.namedChildren).forEach(key => {
            let child = this.props.namedChildren[key];
            callback(key, child);
        });
    }

    /** Private Util Methods **/

    /**
     * Sets the parent attribute of all incoming
     * array and/or named children to this
     * instance.
     */
    _setupChildRelationships(){
        // Named children first
        Object.keys(this.props.namedChildren).forEach(key => {
            let child = this.props.namedChildren[key];
            child.parent = this;
        });

        // Now array children
        this.props.children.forEach(child => {
            child.parent = this;
        });
    }

    /**
     * Updates this components props object
     * based on an incoming object
     */
    _updateProps(incomingProps){
        this.props = incomingProps;
        this.props.children = incomingProps.children || [];
        this.props.namedChildren = incomingProps.namedChildren || {};
        this._setupChildRelationships();
    }

    /**
     * Updates the mapped replacement keys internal
     * object for the component.
     * TODO: Remove this internal lifecycle method
     * when refactoring away from the replacement
     * structure
     */
    _updateReplacements(replacementKeys){
        this.replacements = new ReplacementsHandler(replacementKeys);
        this.usesReplacements = (replacementKeys.length > 0);
    }

    /**
     * Recursively maps a one or multidimensional
     * named children value with the given mapping
     * function.
     */
    _recursivelyMapNamedChildren(collection, callback){
        return collection.map(item => {
            if(Array.isArray(item)){
                return this._recursivelyMapNamedChildren(item, callback);
            } else {
                return callback(item);
            }
        });
    }
};

export {Component, Component as default};
