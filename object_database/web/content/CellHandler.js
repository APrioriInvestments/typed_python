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
    constructor(){
        // Instance Props
        this.postscripts = [];
        this.cells = {};

        // Bind Instance Methods
        this.showConnectionClosed = this.showConnectionClosed.bind(this);
        this.handlePostscript = this.handlePostscript.bind(this);
        this.handleMessage = this.handleMessage.bind(this);

    }

    /**
     * Fills the page's primary div with
     * an indicator that the socket has been
     * disconnected.
     */
    showConnectionClosed(){
        document.getElementById("page_root").innerHTML = `
            <main role="main" class="container">
              <div class='alert alert-primary center-block mt-5'>
                Disconnected!
              </div>
            </main>`;
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
            this.cells[message.id].remove();
        } else if(message.id !== undefined){
            // A dictionary of ids within the object to replace.
            // Targets are real ids of other objects.
            let replacements = message.replacements;
            let element = this.htmlToDomEl(message.contents);

            // Install the element into the dom
            if(this.cells[message.id] === undefined){
                // This is a totally new node.
                // For the moment, add it to the
                // holding pen.
                this.cells["holding_pen"].appendChild(element);
                this.cells[message.id] = element;
                element.id = message.id;
            } else {
                // Replace the existing copy of
                // the node with this incoming
                // copy.
                if(this.cells[message.id].parentNode === null){
                    this.cells["holding_pen"].appendChild(this.cells[message.id]);
                }
                this.cells[message.id].parentNode.replaceChild(
                    element,
                    this.cells[message.id]
                );
                this.cells[message.id] = element;
                element.id = message.id;
            }

            // Now wire in its children
            Object.keys(replacements).forEach((replacementKey, idx) => {
                let target = document.getElementById(replacementKey);
                let source = null;
                if(this.cells[replacements[replacementKey]] === undefined){
                    // This is actually a new node.
                    // We'll define it later in the
                    // event stream.
                    this.cells[replacements[replacementKey]] = document.createElement("div");
                    source = this.cells[replacements[replacementKey]];
                    source.id = replacements[replacementKey];
                    this.cells["holding_pen"].appendChild(source);
                } else {
                    // Not a new node
                    source = this.cells[replacements[replacementKey]];
                }

                if(target != null){
                    target.parentNode.replaceChild(source, target);
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
     * Helper function that takes some incoming
     * HTML string and returns a DOM-instantiated
     * element from it.
     * @param {string} html - The markup to
     * transform into a real element.
     */
    htmlToDomEl(html){
        let element = document.createElement("div");
        element.innerHTML = html;
        return element.children[0];
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
}
