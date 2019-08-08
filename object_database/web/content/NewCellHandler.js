/**
 * New Primary Cells Message Handler
 * ---------------------------------
 * This class implements message handlers
 * of several varieties that come over
 * a CellSocket instance.
 */
import {h} from 'maquette';
import {render} from './components/Component';
import {
    findNamedComponent,
    allNamedChildrenDo
} from './components/util/NamedChildren';

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
        this._newComponents = [];
        this._updatedComponents = [];

        // Bind component methods
        this.updatePopovers = this.updatePopovers.bind(this);
        this.showConnectionClosed = this.showConnectionClosed.bind(this);
        this.connectionClosedView = this.connectionClosedView.bind(this);
        this.appendPostscript = this.appendPostscript.bind(this);
        this.handlePostscript = this.handlePostscript.bind(this);
        this.receive = this.receive.bind(this);
        this.cellUpdated = this.cellUpdated.bind(this);
        this.cellDiscarded = this.cellDiscarded.bind(this);
        this.doesNotUnderstand = this.doesNotUnderstand.bind(this);
        this._getUpdatedComponent = this._getUpdatedComponent.bind(this);
        this._createAndUpdate = this._createAndUpdate.bind(this);
        this._updateComponentProps = this._updateComponentProps.bind(this);
        this._updateNamedChildren = this._updateNamedChildren.bind(this);
        this._findOrCreateChild = this._findOrCreateChild.bind(this);
        this._removeFromParent = this._removeFromParent.bind(this);
        this._removeAllChildren = this._removeAllChildren.bind(this);
        this._callDidLoadForNew = this._callDidLoadForNew.bind(this);
        this._callDidUpdate = this._callDidUpdate.bind(this);
    }

    /**
     * Fills the page's primary div with
     * an indicator that the socket has been
     * disconnected.
     */
    showConnectionClosed(){
        this.projector.replace(
            document.getElementById("page_root"),
            this.connectionClosedView
        );
    }

    /**
     * Helper function that generates the vdom Node for
     * to be display when connection closes
     */
    connectionClosedView(){
        return this.h("main.container", {role: "main"}, [
            this.h("div", {class: "alert alert-primary center-block mt-5"},
                   ["Disconnected"])
        ]);
    }

    receive(message){
        switch(message.type){
        case '#cellUpdated':
            return this.cellUpdated(message);
        case '#cellDiscarded':
            return this.cellDiscarded(message);
        case '#appendPostscript':
            return this.appendPostscript(message);
        default:
            return this.doesNotUnderstand(message);
        }
    }

    /** Primary Message Handlers **/

    doesNotUnderstand(message){
        let msg = `CellHandler does not understand the following message: ${message}`;
        console.error(msg);
        return;
    }

    appendPostscript(message){
        this.postscripts.push(message.script);
    }

    cellUpdated(message){
        let component = this._getUpdatedComponent(message);
        let velement = render(component);
        let domElement = document.getElementById(component.props.id);
        this.projector.replace(domElement, () => {
            return velement;
        });
        this._callDidLoadForNew();
        this._callDidUpdate();
        if(message.postscript){
            this.postscripts.push(message.postscript);
        }
        return component;
    }

    cellDiscarded(message){
        let found = this.activeComponents[message.id];
        if(found){
            delete this.activeComponents[message.id];
        }
        return found;
    }

    /* Legacy Methods still required */
    /**
     * Primary method for handling
     * 'postscripts' messages, which tell
     * this object to go through it's array
     * of script strings and to evaluate them.
     * The evaluation is done on the global
     * window object explicitly.
     * NOTE: Future refactorings/restructurings
     * will remove much of the need to call eval!
     * @param {string} message - The incoming string
     * from the socket.
     */
    handlePostscript(message){
        // Elsewhere, update popovers first
        // Now we evaluate scripts coming
        // across the wire.
        this.updatePopovers();
        while(this.postscripts.length){
            let postscript = this.postscripts.pop();
            try {
                window.eval(postscript);
            } catch(e){
                console.error("ERROR RUNNING POSTSCRIPT", e);
                console.log(postscript);
            }
        }
    }

    /**
     * Convenience method that updates
     * Bootstrap-style popovers on
     * the DOM.
     * See inline comments
     */
    updatePopovers() {
        // This function requires
        // jQuery and perhaps doesn't
        // belong in this class.
        // TODO: Figure out a better way
        // ALSO NOTE:
        // -----------------
        // `getChildProp` is a const function
        // that is declared in a separate
        // script tag at the bottom of
        // page.html. That's a no-no!
        $('[data-toggle="popover"]').popover({
            html: true,
            container: 'body',
            title: function () {
                return getChildProp(this, 'title');
            },
            content: function () {
                return getChildProp(this, 'content');
            },
            placement: function (popperEl, triggeringEl) {
                let placement = triggeringEl.dataset.placement;
                if(placement == undefined){
                    return "bottom";
                }
                return placement;
            }
        });
        $('.popover-dismiss').popover({
            trigger: 'focus'
        });
    }


    /* Private Methods */
    _getUpdatedComponent(description){
        if(!Object.keys(this.activeComponents).includes(description.id.toString())){
            // In this case, we are creating a totally new Cell
            // and component combination.
            return this._createAndUpdate(description);
        }
        let component = this.activeComponents[description.id];
        this._updateComponentProps(component, description);
        this._updateNamedChildren(component, description);
        this._updatedComponents.push(component);
        return component;
    }

    _createAndUpdate(message){
        let componentClass = this.availableComponents[message.cellType];
        if(!componentClass || componentClass == undefined){
            throw new Error(`Cannot find Component for Cell Type: ${message.cellType}`);
        }
        let componentProps = Object.assign({}, message.extraData, {
            id: message.id,
            extraData: message.extraData
        });
        let newComponent = new componentClass(componentProps);
        this._updateNamedChildren(newComponent, message);
        this.activeComponents[newComponent.props.id] = newComponent;
        this._newComponents.push(newComponent);
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
        component._updateProps(newProps);
    }

    _updateNamedChildren(component, message){
        let newNamedChildren = {};
        Object.keys(message.namedChildren).forEach(childName => {
            let childDescription = message.namedChildren[childName];
            newNamedChildren[childName] = this._findOrCreateChild(childDescription, component);
        });
        component.props.namedChildren = newNamedChildren;
    }

    _findOrCreateChild(childDescription, parentComponent){
        if(Array.isArray(childDescription)){
            return childDescription.map(item => {
                return this._findOrCreateChild(item, parentComponent);
            });
        }
        let childComponent = this._getUpdatedComponent(childDescription);
        childComponent.parent = parentComponent;
        return childComponent;
    }

    _removeFromParent(component){
        let search = findNamedComponent(component, component.parent);
        if(search.found == false){
            console.warn(`Could not find ${component.name}(${component.props.id}) in parent's namedChildren`);
            return false;
        } else if(search.found && search.inArray){
            // In this case, the result was in an array.
            // So we need to find and splice that array.
            let idx = search.inArray.indexOf(component);
            if(idx < 0){
                throw new Error(`${component.name}(${component.props.id}) removal failed trying to find itself in parent children using utility method.`);
            }
            search.inArray.splice(idx, 1);
            return true;
        } else if(search.found){
            let parentNamedChildren = component.parent.props.namedChildren;
            parentNamedChildren[search.nameInParent] = null;
            return true;
        }
        return false;
    }

    _removeAllChildren(component){
        allNamedChildrenDo(component, child => {
            delete this.activeComponents[child.props.id];
        });
    }

    _callDidLoadForNew(){
        // Goes through the stored list of
        // new components created during a message handle
        // and calls componentDidLoad()
        this._newComponents.forEach(component => {
            component.componentDidLoad();
        });
        this._newComponents = [];
    }

    _callDidUpdate(){
        // Goes through the stored
        // list of existing components
        // that were updated during a
        // message handle and calls
        // componenDidUpdate()
        this._updatedComponents.forEach(component => {
            component.componentDidUpdate();
        });
        this._updatedComponents = [];
    }
}

export {NewCellHandler, NewCellHandler as default};
