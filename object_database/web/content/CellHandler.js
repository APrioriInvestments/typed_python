/**
 * CellHandler Primary Message Handler
 * ------------------------------------------
 * This class implements a service that handles
 * messages of all kinds that come in over a
 * `CellSocket`.
 * NOTE: For the moment there are only two kinds
 * of messages and therefore two handlers. We have
 * plans to change this structure to be more flexible
 * and so the API of this class will change greatly.
 */
import {h} from 'maquette';

class CellHandler {
    constructor(h, projector, components){
	// props
	this.h = h;
	this.projector = projector;
	this.components = components;
        this.activeComponents = {};

	// Instance Props
        this.postscripts = [];
        this.cells = {};
	this.DOMParser = new DOMParser();

        // Bind Instance Methods
        this.showConnectionClosed = this.showConnectionClosed.bind(this);
	this.connectionClosedView = this.connectionClosedView.bind(this);
        this.handlePostscript = this.handlePostscript.bind(this);
        this.cellUpdated = this.cellUpdated.bind(this);
        this.cellDiscarded = this.cellDiscarded.bind(this);
        this.receive = this.receive.bind(this);
        this.invalidMessageFormat = this.invalidMessageFormat.bind(this);
        this.doesNotUnderstand = this.doesNotUnderstand.bind(this);

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
     * Primary entrypoint to handling
     * normal messages.
     */
    receive(aMessage){
        if(!aMessage.type || aMessage.type == undefined){
            return this.invalidMessageFormat(aMessage);
        }
        switch(aMessage.type){
        case '#cellUpdated':
            return this.cellUpdated(aMessage);
        case '#cellDiscarded':
            return this.cellDiscarded(aMessage);
        case '#appendPostscript':
            return this.appendPostscript(aMessage);
        default:
            return this.doesNotUnderstand(aMessage);
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
     * Message that tells the handler to append
     * a specific and immediate script to
     * the postscripts list.
     */
    appendPostscript(message){
        this.postscripts.push(message.script);
    }

    /**
     * Message that tells us a Cell has been
     * discarded and is no longer needed.
     * We remove both it's DOM representation
     * and any registered components for it.
     */
    cellDiscarded(message){
        let foundCell = this.cells[message.id];
        let foundComponent = this.activeComponents[message.id];

        if(foundCell == undefined){
            console.warn(`Received discard message for non-existing Cell ${message.id}`);
        }

        if(foundComponent == undefined){
            console.warn(`Received discard message for non-existing Component ${message.id}`);
        }

        // If the cell has a domNode property,
        // it means it's a hyperscript element
        // and this is what we should deal with.
        if(foundCell && foundCell.domNode !== undefined){
            foundCell = foundCell.domNode;
        }

        if(foundComponent){
            // Unload lifecycle function
            foundComponent.componentWillUnload();
            delete this.activeComponents[message.id];
            delete this.cells[message.id];
        }

        // NOTE: The following was copied from the original
        // large handler function. Not sure we need to
        // replace rather than remove.
        // TODO: Refactor this when removing reliance
        // on replacements.
        // Instead of removing the node we replace with the a
	// `display:none` style node which effectively removes it
	// from the DOM
	if (foundCell && foundCell.parentNode !== null) {
	    this.projector.replace(foundCell, () => {
		return h("div", {style: "display:none"}, []);
	    });
	}
    }

    /**
     * Primary method for handling 'normal'
     * (ie non-postscripts) messages that have
     * been deserialized from JSON.
     * For the moment, these messages deal
     * entirely with DOM replacement operations, which
     * this method implements.
     * @param {object} message - A deserialized
     * JSON message from the server that has
     * information about elements that need to
     * be updated.
     */
    cellUpdated(message){
        let newComponents = [];
	if(this.cells["page_root"] == undefined){
            this.cells["page_root"] = document.getElementById("page_root");
            this.cells["holding_pen"] = document.getElementById("holding_pen");
        }

	// With the exception of `page_root` and `holding_pen` id nodes, all
	// elements in this.cells are virtual. Dependig on whether we are adding a
	// new node, or manipulating an existing, we neeed to work with the underlying
	// DOM node. Hence if this.cell[message.id] is a vdom element we use its
	// underlying domNode element when in operations like this.projector.replace()
	let cell = this.cells[message.id];

	if (cell !== undefined && cell.domNode !== undefined) {
	    cell = cell.domNode;
	}


	// A dictionary of ids within the object to replace.
	// Targets are real ids of other objects.
	let replacements = message.replacements;
        var component = this.activeComponents[message.id];
        if(component){
            // In this case, this component has already been
            // created and stored, so we just need to pass in
            // updated props and re-render.
            let nextProps = Object.assign({}, component.props, {
                namedChildren: message.namedChildren,
                children: message.children,
                extraData: message.extraData
            }, message.extraData);
            let nextPropsToPass = component.componentWillReceiveProps(
                component.props,
                nextProps
            );
            component._updateReplacements(message.replacementKeys);
            component._updateProps(nextPropsToPass);
            //let cClass = this.components[message.componentName];
            //var component = new cClass(nextProps, message.replacementKeys);
            var velement = component.render();
        } else {
            // In this case, we are creating this cell for the first
            // time (we have never seen its ID, and will create
            // and store a whole new Component for it)
	    let componentClass = this.components[message.componentName];
	    if (componentClass === undefined) {
                console.warn(`Could not find component for ${message.componentName}`);
		var velement = this.htmlToVDomEl(message.contents, message.id);
	    } else {
                let componentProps = Object.assign({
                    id: message.id,
                    namedChildren: message.namedChildren,
                    children: message.children,
                    extraData: message.extraData
                }, message.extraData);
		var component = new componentClass(
                    componentProps,
                    message.replacementKeys
                );
                var velement = component.render();

                // Add the new component to the comp/cell
                // stored dict.
                this.activeComponents[message.id] = component;
                newComponents.push(component);
	    }
        }

        // If the incoming message describes a Cell that
        // is not for display, then we should return from
        // the method here.
        if(!message.shouldDisplay){
            if(component){
                component.componentDidLoad();
            }
            return;
        }

        // Install the element into the dom
        if(cell === undefined){
	    // This is a totally new node.
            // For the moment, add it to the
            // holding pen.
	    this.projector.append(this.cells["holding_pen"], () => {
                return velement;
            });
	    this.cells[message.id] = velement;
        } else {
            // Replace the existing copy of
            // the node with this incoming
            // copy.
	    if(cell.parentNode === null){
		this.projector.append(this.cells["holding_pen"], () => {
		    return velement;
		});
	    } else {
		this.projector.replace(cell, () => {return velement;});
	    }
	}

        this.cells[message.id] = velement;

        // Now wire in replacements
        Object.keys(replacements).forEach((replacementKey, idx) => {
            let target = document.getElementById(replacementKey);
            let source = null;
            if(this.cells[replacements[replacementKey]] === undefined){
		// This is actually a new node.
                // We'll define it later in the
                // event stream.
		source = this.h("div", {id: replacementKey}, []);
                this.cells[replacements[replacementKey]] = source; 
		this.projector.append(this.cells["holding_pen"], () => {
		    return source;
                });
	    } else {
                // Not a new node
                source = this.cells[replacements[replacementKey]];
            }

            if(target != null){
		this.projector.replace(target, () => {
		    return source;
                });
            } else {
                let errorMsg = `In message ${message} couldn't find ${replacementKey}`;
                throw new Error(errorMsg);
            }
        });

        if(message.postscript !== undefined){
            this.postscripts.push(message.postscript);
        }

        // If we created any new components during this
        // message handling session, we finally call
        // their `componentDidLoad` lifecycle methods
        newComponents.forEach(component => {
            component.componentDidLoad();
        });

        // Remove leftover replacement divs
        // that are still in the page_root
        // after vdom insertion
        let pageRoot = document.getElementById('page_root');
        let found = pageRoot.querySelectorAll('[id*="_____"]');
        found.forEach(element => {
            element.remove();
        });
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

     /**
     * This is a (hopefully temporary) hack
     * that will intercept the first time a
     * dropdown carat is clicked and bind
     * Bootstrap Dropdown event handlers
     * to it that should be bound to the
     * identified cell. We are forced to do this
     * because the current Cells infrastructure
     * does not have flexible event binding/handling.
     * @param {string} cellId - The ID of the cell
     * to identify in the socket callback we will
     * bind to open and close events on dropdown
     */
    dropdownInitialBindFor(cellId){
        let elementId = cellId + '-dropdownMenuButton';
        let element = document.getElementById(elementId);
        if(!element){
            throw Error('Element of id ' + elementId + ' doesnt exist!');
        }
        let dropdownMenu = element.parentElement;
        let firstTimeClicked = element.dataset.firstclick == 'true';
        if(firstTimeClicked){
            $(dropdownMenu).on('show.bs.dropdown', function(){
                cellSocket.sendString(JSON.stringify({
                    event: 'dropdown',
                    target_cell: cellId,
                    isOpen: false
                }));
            });
            $(dropdownMenu).on('hide.bs.dropdown', function(){
                cellSocket.sendString(JSON.stringify({
                    event: 'dropdown',
                    target_cell: cellId,
                    isOpen: true
                }));
            });

            // Now expire the first time clicked
            element.dataset.firstclick = 'false';
        }
    }

    /**
     * A catch-all default response for messages
     * whose `type` property contains a value
     * that the system doesn't understand.
     */
    doesNotUnderstand(aMessage){
        console.error(`CellHandler does not understand the message:`, aMessage);
    }

    /**
     * An error reporting for badly-formatted
     * socket messages.
     */
    invalidMessageFormat(aMessage){
        console.error(`CellHandler received a malformatted message:`, aMessage);
    }

    /**
     * Unsafely executes any passed in string
     * as if it is valid JS against the global
     * window state.
     */
    static unsafelyExecute(aString){
        window.exec(aString);
    }

    /**
     * Helper function that takes some incoming
     * HTML string and returns a maquette hyperscript
     * VDOM element from it.
     * This uses the internal browser DOMparser() to generate the html
     * structure from the raw string and then recursively build the
     * VDOM element
     * @param {string} html - The markup to
     * transform into a real element.
     */
    htmlToVDomEl(html, id){
	let dom = this.DOMParser.parseFromString(html, "text/html");
        let element = dom.body.children[0];
        return this._domElToVdomEl(element, id);
    }

    _domElToVdomEl(domEl, id) {
	let tagName = domEl.tagName.toLocaleLowerCase();
	let attrs = {id: id};
	let index;

	for (index = 0; index < domEl.attributes.length; index++){
	    let item = domEl.attributes.item(index);
	    attrs[item.name] = item.value.trim();
	}

	if (domEl.childElementCount === 0) {
	    return h(tagName, attrs, [domEl.textContent]);
	}

	let children = [];
	for (index = 0; index < domEl.children.length; index++){
	    let child = domEl.children[index];
	    children.push(this._domElToVdomEl(child));
	}

	return h(tagName, attrs, children);
    }
}

export {CellHandler, CellHandler as default}
