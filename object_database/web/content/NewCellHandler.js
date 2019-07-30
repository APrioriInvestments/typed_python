/**
 * New Primary Cells Message Handler
 * ---------------------------------
 * This class implements message handlers
 * of several varieties that come over
 * a CellSocket instance.
 */
import {h} from 'maquette';
import {render} from './components/Component';

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
        this.updatePopovers = this.updatePopovers.bind(this);
        this.showConnectionClosed = this.showConnectionClosed.bind(this);
        this.connectionClosedView = this.connectionClosedView.bind(this);
        //this.appendPostscript = this.appendPostscript.bind(this);
        this.handlePostscript = this.handlePostscript.bind(this);
        this.receive = this.receive.bind(this);
        this.cellUpdated = this.cellUpdated.bind(this);
        //this.cellDiscarded = this.cellDiscarded.bind(this);
        this.doesNotUnderstand = this.doesNotUnderstand.bind(this);
        this._createAndUpdate = this._createAndUpdate.bind(this);
        this._updateComponentProps = this._updateComponentProps.bind(this);
        this._updateNamedChildren = this._updateNamedChildren.bind(this);
        this._findOrCreateChild = this._findOrCreateChild.bind(this);
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
        if(!Object.keys(this.activeComponents).includes(message.id.toString())){
            // In this case, we are creating a totally new Cell
            // and component combination.
            return this._createAndUpdate(message);
        }
        let component = this.activeComponents[message.id];
        this._updateComponentProps(component, message);
        this._updateNamedChildren(component, message);

        if(rootCall){
            let velement = render(component);
            let domElement = document.getElementById(component.props.id);
            this.projector.replace(domElement, () => {
                return velement;
            });
            component.componentDidUpdate();
        }
        return component;
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
        if(message.id == "page_root"){
            let domElement = document.getElementById('page_root');
            let velement = render(newComponent);
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
        let childComponent = this.cellUpdated(childDescription);
        childComponent.parent = parentComponent;
        return childComponent;
    }
}

export {NewCellHandler, NewCellHandler as default};
