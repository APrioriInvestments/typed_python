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

	// Instance Props
        this.postscripts = [];
        this.cells = {};
	this.DOMParser = new DOMParser();

        // Bind Instance Methods
        this.showConnectionClosed = this.showConnectionClosed.bind(this);
	this.connectionClosedView = this.connectionClosedView.bind(this);
        this.handlePostscript = this.handlePostscript.bind(this);
        this.handleMessage = this.handleMessage.bind(this);

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
    handleMessage(message){
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

        if(message.discard !== undefined){
            // In the case where we have received a 'discard' message,
            // but the cell requested is not available in our
            // cells collection, we simply display a warning:
            if(cell == undefined){
                console.warn(`Received discard message for non-existing cell id ${message.id}`);
                return;
            }
	    // Instead of removing the node we replace with the a
	    // `display:none` style node which effectively removes it
	    // from the DOM
	    if (cell.parentNode !== null) {
		this.projector.replace(cell, () => {
		    return h("div", {style: "display:none"}, []);
		});
	    }
	} else if(message.id !== undefined){
	    // A dictionary of ids within the object to replace.
	    // Targets are real ids of other objects.
	    let replacements = message.replacements;

	    // TODO: this is a temporary branching, to be removed with a more logical setup. As
	    // of writing if the message coming across is sending a "known" component then we use
	    // the component itself as opposed to building a vdom element from the raw html
	    let componentClass = this.components[message.component_name];
	    if (componentClass === undefined) {
                console.warn(`Could not find component for ${message.component_name}`);
		var velement = this.htmlToVDomEl(message.contents, message.id);
	    } else {
                let componentProps = Object.assign({
                    id: message.id,
                    namedChildren: message.namedChildren,
                    children: message.children,
                    extraData: message.extra_data
                }, message.extra_data);
		var component = new componentClass(
                    componentProps,
                    message.replacement_keys
                );
                var velement = component.render();
                newComponents.push(component);
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
                    //console.log("In message ", message, " couldn't find ", replacementKey);
                }
            });
        }

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
