/**
 * New Primary Cells Message Handler
 * ---------------------------------
 * This class implements message handlers
 * of several varieties that come over
 * a CellSocket instance.
 */
import {h} from 'maquette';


class NewCellHandler {
    constructor(h, projector, components){
        // A constructor for
        // hyperscript objects
        this.h = h;

        // A Maquette VDOM
        // projector instance
        this.projector = projector;

        // A dictionary of available
        // Cell Components by name
        this.availableComponents = components;

        // Properties that will be updated
        // as messages come in
        this.cells = {};
        this.activeComponents = {};
        this.postscripts = [];

        // Private properties
        this._sessionId = null;

        // Bind component methods
        //this.updatePopovers = this.updatePopovers.bind(this);
        //this.appendPostscript = this.appendPostscript.bind(this);
        //this.handlePostscript = this.handlePostscript.bind(this);
        this.receive = this.receive.bind(this);
        this.cellUpdated = this.cellUpdated.bind(this);
        //this.cellDiscarded = this.cellDiscarded.bind(this);
        this.doesNotUnderstand = this.doesNotUnderstand.bind(this);
        this._createAndUpdate = this._createAndUpdate.bind(this);
        this._updateComponentProps = this._updateComponentProps.bind(this);
        this._updateNamedChildren = this._updateNamedChildren.bind(this);
        this._findOrCreateChild = this._findOrCreateChild.bind(this);
    }

    receive(message){
        switch(message.type){
        case '#cellUpdated':
            return this.cellUpdated(message, true);
        case '#cellDiscarded':
            return this.cellDiscarded(message);
        case '#appendPostscript':
            return this.appendPostscript(message);
        default:
            return this.doesNotUnderstand(message);
        }
    }

    doesNotUnderstand(message){
        let msg = `CellHandler does not understand the following message: ${message}`;
        console.error(msg);
        return;
    }

    cellUpdated(message, rootCall=false){
        if(!Object.keys(this.activeComponents).includes(message.id)){
            // In this case, we are creating a totally new Cell
            // and component combination.
            return this._createAndUpdate(message);
        }
        let component = this.activeComponents[message.id];
        this._updateComponentProps(component, message);
        this._updateNamedChildren(component, message);

        if(rootCall){
            let velement = component.render();
            let domElement = document.getElementById(component.props.id);
            this.projector.replace(domElement, () => {
                return velement;
            });
            component.componentDidUpdate();
        }
        return component;
    }


    /* Private Methods */
    _createAndUpdate(message){
        let componentClass = this.availableComponents[message.cellType];
        if(!componentClass || componentClass == undefined){
            throw new Error(`Cannot find Component for Cell Type: ${message.cellType}`);
            return;
        }
        let componentProps = Object.assign({}, message.extraData, {
            id: message.id,
            extraData: message.extraData
        });
        let newComponent = new componentClass(componentProps);
        this._updateNamedChildren(newComponent, message);
        this.activeComponents[newComponent.props.id] = newComponent;
        if(message.id == "page_root"){
            let domElement = document.getElementById('page_root');
            let velement = newComponent.render();
            this.projector.replace(domElement, () => {
                return velement;
            });
        }
        return newComponent;

    }

    _updateComponentProps(component, message){
        let nextProps = Object.assign({}, message.extraData, {
            id: message.id,
            extraData: message.extraData
        });
        let newProps = component.componentWillReceiveProps(
            component.props,
            nextProps
        );
        component.props = newProps;
    }

    _updateNamedChildren(component, message){
        let newNamedChildren = {};
        Object.keys(message.namedChildren).forEach(childName => {
            let childDescription = message.namedChildren[childName];
            newNamedChildren[childName] = this._findOrCreateChild(childDescription, component);
        });
        component.namedChildren = newNamedChildren;
    }

    _findOrCreateChild(childDescription, parentComponent){
        if(Array.isArray(childDescription)){
            return childDescription.map(item => {
                return this._findOrCreateChild(item, parentComponent);
            });
        }
        let childComponent = this.cellUpdated(childDescription);
        childComponent.parent = parentComponent;
        return childComponent;
    }
}

export {NewCellHandler, NewCellHandler as default};
