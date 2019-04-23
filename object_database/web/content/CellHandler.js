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

class CellHandler {
    constructor(h, projector){
		// props
		this.h = h;
		this.projector = projector;

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
		this.projector.replace(document.getElementById("page_root"),
			this.connectionClosedView)
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
            try{
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
        if(this.cells["page_root"] == undefined){
            this.cells["page_root"] = document.getElementById("page_root");
            this.cells["holding_pen"] = document.getElementById("holding_pen");
        }

        if(message.discard !== undefined){
			// Instead of removing the node we replace with the a
			// `display:none` style node which effectively removes it
			// from the DOM
            this.projector.replace(this.cells[message.id], () => {
				return h("div", {style: "display:none"}, []); 
			});
        } else if(message.id !== undefined){
            // A dictionary of ids within the object to replace.
            // Targets are real ids of other objects.
            let replacements = message.replacements;
            let velement = this.htmlToVDomEl(message.contents, message.id);

            // Install the element into the dom
            if(this.cells[message.id] === undefined){
                // This is a totally new node.
                // For the moment, add it to the
                // holding pen.
				this.projector.append(this.cells["holding_pen"], () => {return velement})
                this.cells[message.id] = velement;
            } else {
                // Replace the existing copy of
                // the node with this incoming
                // copy.
                if(this.cells[message.id].parentNode === null){
					this.projector.append(this.cells["holding_pen"], () => {return velement})

                }
				this.projector.replace(this.cells[message.id], () => {return velement})
				this.projector.scheduleRender();
                this.cells[message.id] = velement;
            }

            // Now wire in replacements 
            Object.keys(replacements).forEach((replacementKey, idx) => {
                let target = document.getElementById(replacementKey);
                let source = null;
                if(this.cells[replacements[replacementKey]] === undefined){
                    // This is actually a new node.
                    // We'll define it later in the
                    // event stream.
					source = this.h("div", {id: replacementKey}, [])
                    this.cells[replacements[replacementKey]] = source; 
					this.projector.append(this.cells["holding_pen"], () => {return source})
                } else {
                    // Not a new node
                    source = this.cells[replacements[replacementKey]];
                }

                if(target != null){
					this.projector.replace(target, () => {return source});
					this.projector.scheduleRender();
                } else {
                    console.log("In message ", message, " couldn't find ", replacementKey);
                }
            });
        }

        if(message.postscript !== undefined){
            this.postscripts.push(message.postscript);
        }
    }

    /**
     * Helper function that generates the vdom Node for
	 * to be display when connection closes
     */
    connectionClosedView(){
		return this.h("main.container", {role: "main"}, [
			this.h("div", {class: "alert alert-primary center-block mt-5"}, [
				"Disconnected"
			])
		])
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
		let dom = this.DOMParser.parseFromString(html, "text/html")
        let element = dom.body.children[0];
        return this._domElToVdomEl(element, id);
    }

	_domElToVdomEl(domEl, id) {
		let tagName = domEl.tagName.toLocaleLowerCase();
		let attrs = {id: id}
		let index;
		for (index = 0; index < domEl.attributes.length; index++){
			let item = domEl.attributes.item(index)
			attrs[item.name] = item.value.trim();
		}

		if (domEl.childElementCount === 0) {
			return h(tagName, attrs, [domEl.textContent])
		}
		
		let children = [];
		for (index = 0; index < domEl.children.length; index++){
			let child = domEl.children[index]
			children.push(this._domElToVdomEl(child));
		}
		return h(tagName, attrs, children);
	}

}
