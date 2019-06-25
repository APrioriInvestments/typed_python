/******/ (function(modules) { // webpackBootstrap
/******/ 	// The module cache
/******/ 	var installedModules = {};
/******/
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/
/******/ 		// Check if module is in cache
/******/ 		if(installedModules[moduleId]) {
/******/ 			return installedModules[moduleId].exports;
/******/ 		}
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = installedModules[moduleId] = {
/******/ 			i: moduleId,
/******/ 			l: false,
/******/ 			exports: {}
/******/ 		};
/******/
/******/ 		// Execute the module function
/******/ 		modules[moduleId].call(module.exports, module, module.exports, __webpack_require__);
/******/
/******/ 		// Flag the module as loaded
/******/ 		module.l = true;
/******/
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/
/******/
/******/ 	// expose the modules object (__webpack_modules__)
/******/ 	__webpack_require__.m = modules;
/******/
/******/ 	// expose the module cache
/******/ 	__webpack_require__.c = installedModules;
/******/
/******/ 	// define getter function for harmony exports
/******/ 	__webpack_require__.d = function(exports, name, getter) {
/******/ 		if(!__webpack_require__.o(exports, name)) {
/******/ 			Object.defineProperty(exports, name, { enumerable: true, get: getter });
/******/ 		}
/******/ 	};
/******/
/******/ 	// define __esModule on exports
/******/ 	__webpack_require__.r = function(exports) {
/******/ 		if(typeof Symbol !== 'undefined' && Symbol.toStringTag) {
/******/ 			Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });
/******/ 		}
/******/ 		Object.defineProperty(exports, '__esModule', { value: true });
/******/ 	};
/******/
/******/ 	// create a fake namespace object
/******/ 	// mode & 1: value is a module id, require it
/******/ 	// mode & 2: merge all properties of value into the ns
/******/ 	// mode & 4: return value when already ns object
/******/ 	// mode & 8|1: behave like require
/******/ 	__webpack_require__.t = function(value, mode) {
/******/ 		if(mode & 1) value = __webpack_require__(value);
/******/ 		if(mode & 8) return value;
/******/ 		if((mode & 4) && typeof value === 'object' && value && value.__esModule) return value;
/******/ 		var ns = Object.create(null);
/******/ 		__webpack_require__.r(ns);
/******/ 		Object.defineProperty(ns, 'default', { enumerable: true, value: value });
/******/ 		if(mode & 2 && typeof value != 'string') for(var key in value) __webpack_require__.d(ns, key, function(key) { return value[key]; }.bind(null, key));
/******/ 		return ns;
/******/ 	};
/******/
/******/ 	// getDefaultExport function for compatibility with non-harmony modules
/******/ 	__webpack_require__.n = function(module) {
/******/ 		var getter = module && module.__esModule ?
/******/ 			function getDefault() { return module['default']; } :
/******/ 			function getModuleExports() { return module; };
/******/ 		__webpack_require__.d(getter, 'a', getter);
/******/ 		return getter;
/******/ 	};
/******/
/******/ 	// Object.prototype.hasOwnProperty.call
/******/ 	__webpack_require__.o = function(object, property) { return Object.prototype.hasOwnProperty.call(object, property); };
/******/
/******/ 	// __webpack_public_path__
/******/ 	__webpack_require__.p = "";
/******/
/******/
/******/ 	// Load entry module and return exports
/******/ 	return __webpack_require__(__webpack_require__.s = "./main.js");
/******/ })
/************************************************************************/
/******/ ({

/***/ "./CellHandler.js":
/*!************************!*\
  !*** ./CellHandler.js ***!
  \************************/
/*! exports provided: CellHandler, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "CellHandler", function() { return CellHandler; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return CellHandler; });
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_0__);
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
		    return Object(maquette__WEBPACK_IMPORTED_MODULE_0__["h"])("div", {style: "display:none"}, []);
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
	    return Object(maquette__WEBPACK_IMPORTED_MODULE_0__["h"])(tagName, attrs, [domEl.textContent]);
	}

	let children = [];
	for (index = 0; index < domEl.children.length; index++){
	    let child = domEl.children[index];
	    children.push(this._domElToVdomEl(child));
	}

	return Object(maquette__WEBPACK_IMPORTED_MODULE_0__["h"])(tagName, attrs, children);
    }
}




/***/ }),

/***/ "./CellSocket.js":
/*!***********************!*\
  !*** ./CellSocket.js ***!
  \***********************/
/*! exports provided: CellSocket, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "CellSocket", function() { return CellSocket; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return CellSocket; });
/**
 * A concrete error thrown
 * if the current browser doesn't
 * support websockets, which is very
 * unlikely.
 */
class WebsocketNotSupported extends Error {
    constructor(args){
        super(args);
    }
}

/**
 * This is the global frame
 * control. We might consider
 * putting it elsewhere, but
 * `CellSocket` is its only
 * consumer.
 */
const FRAMES_PER_ACK = 10;


/**
 * CellSocket Controller
 * ---------------------
 * This class implements an instance of
 * a controller that wraps a websocket client
 * connection and knows how to handle the
 * initial routing of messages across the socket.
 * `CellSocket` instances are designed so that
 * handlers for specific types of messages can
 * register themselves with it.
 * NOTE: For the moment, most of this code
 * has been copied verbatim from the inline
 * scripts with only slight modification.
 **/
class CellSocket {
    constructor(){
        // Instance Props
        this.uri = this.getUri();
        this.socket = null;
        this.currentBuffer = {
            remaining: null,
            buffer: null,
            hasDisplay: false
        };

        /**
         * A callback for handling messages
         * that are 'postscripts'
         * @callback postscriptsHandler
         * @param {string} msg - The forwarded message
         */
        this.postscriptsHander = null;

        /**
         * A callback for handling messages
         * that are normal JSON data messages.
         * @callback messageHandler
         * @param {object} msg - The forwarded message
         */
        this.messageHandler = null;

        /**
         * A callback for handling messages
         * when the websocket connection closes.
         * @callback closeHandler
         */
        this.closeHandler = null;

        /**
         * A callback for handling messages
         * whent the socket errors
         * @callback errorHandler
         */
        this.errorHandler = null;

        // Bind Instance Methods
        this.connect = this.connect.bind(this);
        this.sendString = this.sendString.bind(this);
        this.handleRawMessage = this.handleRawMessage.bind(this);
        this.onPostscripts = this.onPostscripts.bind(this);
        this.onMessage = this.onMessage.bind(this);
        this.onClose = this.onClose.bind(this);
        this.onError = this.onError.bind(this);
    }

    /**
     * Returns a properly formatted URI
     * for the socket for any given current
     * browser location.
     * @returns {string} A URI string.
     */
    getUri(){
        let location = window.location;
        let uri = "";
        if(location.protocol === "https:"){
            uri += "wss:";
        } else {
            uri += "ws:";
        }
        uri = `${uri}//${location.host}`;
        uri = `${uri}/socket${location.pathname}${location.search}`;
        return uri;
    }

    /**
     * Tells this object's internal websocket
     * to instantiate itself and connect to
     * the provided URI. The URI will be set to
     * this instance's `uri` property first. If no
     * uri is passed, `connect()` will use the current
     * attribute's value.
     * @param {string} uri - A  URI to connect the socket
     * to.
     */
    connect(uri){
        if(uri){
            this.uri = uri;
        }
        if(window.WebSocket){
            this.socket = new WebSocket(this.uri);
        } else if(window.MozWebSocket){
            this.socket = MozWebSocket(this.uri);
        } else {
            throw new WebsocketNotSupported();
        }

        this.socket.onclose = this.closeHandler;
        this.socket.onmessage = this.handleRawMessage.bind(this);
        this.socket.onerror = this.errorHandler;
    }

    /**
     * Convenience method that sends the passed
     * string on this instance's underlying
     * websoket connection.
     * @param {string} aString - A string to send
     */
    sendString(aString){
        if(this.socket){
            this.socket.send(aString);
        }
    }

    // Ideally we move the dom operations of
    // this function out into another class or
    // context.
    /**
     * Using the internal `currentBuffer`, this
     * method checks to see if a large multi-frame
     * piece of websocket data is being sent. If so,
     * it presents and updates a specific display in
     * the DOM with the current percentage etc.
     * @param {string} msg - The message to
     * display inside the element
     */
    setLargeDownloadDisplay(msg){

        if(msg.length == 0 && !this.currentBuffer.hasDisplay){
            return;
        }

        this.currentBuffer.hasDisplay = (msg.length != 0);

        let element = document.getElementById("object_database_large_pending_download_text");
        if(element != undefined){
            element.innerHTML = msg;
        }
    }

    /**
     * Handles the `onmessage` event of the underlying
     * websocket.
     * This method knows how to fill the internal
     * buffer (to get around the frame limit) and only
     * trigger subsequent handlers for incoming messages.
     * TODO: Break out this method a bit more. It has been
     * copied nearly verbatim from the original code.
     * NOTE: For now, there are only two types of messages:
     *       'updates' (we just call these messages)
     *       'postscripts' (these are just raw non-JSON strings)
     * If a buffer is complete, this method will check to see if
     * handlers are registered for postscript/normal messages
     * and will trigger them if true in either case, passing
     * any parsed JSON data to the callbacks.
     * @param {Event} event - The `onmessage` event object
     * from the socket.
     */
    handleRawMessage(event){
        if(this.currentBuffer.remaining === null){
            this.currentBuffer.remaining = JSON.parse(event.data);
            this.currentBuffer.buffer = [];
            if(this.currentBuffer.hasDisplay && this.currentBuffer.remaining == 1){
                // SET LARGE DOWNLOAD DISPLAY
            }
            return;
        }

        this.currentBuffer.remaining -= 1;
        this.currentBuffer.buffer.push(event.data);

        if(this.currentBuffer.buffer.length % FRAMES_PER_ACK == 0){
            //ACK every tenth message. We have to do active pushback
            //because the websocket disconnects on Chrome if you jam too
            //much in at once
            this.sendString(
                JSON.stringify({
                    "ACK": this.currentBuffer.buffer.length
                }));
            let percentage = Math.round(100*this.currentBuffer.buffer.length / (this.currentBuffer.remaining + this.currentBuffer.buffer.length));
            let total = Math.round((this.currentBuffer.remaining + this.currentBuffer.buffer.length) / (1024 / 32));
            let progressStr = `(Downloaded ${percentage}% of ${total} MB)`;
            this.setLargeDownloadDisplay(progressStr);
        }

        if(this.currentBuffer.remaining > 0){
            return;
        }

        this.setLargeDownloadDisplay("");

        let joinedBuffer = this.currentBuffer.buffer.join('')

        this.currentBuffer.remaining = null;
        this.currentBuffer.buffer = null;

        let update = JSON.parse(joinedBuffer);

        if(update == 'request_ack') {
            this.sendString(JSON.stringify({'ACK': 0}))
        } else if(update == 'postscripts'){
            // updatePopovers();
            if(this.postscriptsHandler){
                this.postscriptsHandler(update);
            }
        } else {
            if(this.messageHandler){
                this.messageHandler(update);
            }
        }
    }

    /**
     * Convenience method that binds
     * the passed callback to this instance's
     * postscriptsHandler, which is some method
     * that handles messages for postscripts.
     * @param {postscriptsHandler} callback - A handler
     * callback method with the message argument.
     */
    onPostscripts(callback){
        this.postscriptsHandler = callback;
    }

    /**
     * Convenience method that binds
     * the passed callback to this instance's
     * postscriptsHandler, which is some method
     * that handles messages for postscripts.
     * @param {messageHandler} callback - A handler
     * callback method with the message argument.
     */
    onMessage(callback){
        this.messageHandler = callback;
    }

    /**
     * Convenience method that binds the
     * passed callback to the underlying
     * websocket's `onclose` handler.
     * @param {closeHandler} callback - A function
     * that handles close events on the socket.
     */
    onClose(callback){
        this.closeHandler = callback;
    }

    /**
     * Convenience method that binds the
     * passed callback to the underlying
     * websockets' `onerror` handler.
     * @param {errorHandler} callback - A function
     * that handles errors on the websocket.
     */
    onError(callback){
        this.errorHandler = callback;
    }
}





/***/ }),

/***/ "./ComponentRegistry.js":
/*!******************************!*\
  !*** ./ComponentRegistry.js ***!
  \******************************/
/*! exports provided: ComponentRegistry, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ComponentRegistry", function() { return ComponentRegistry; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return ComponentRegistry; });
/* harmony import */ var _components_AsyncDropdown__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./components/AsyncDropdown */ "./components/AsyncDropdown.js");
/* harmony import */ var _components_Badge__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./components/Badge */ "./components/Badge.js");
/* harmony import */ var _components_Button__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./components/Button */ "./components/Button.js");
/* harmony import */ var _components_ButtonGroup__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ./components/ButtonGroup */ "./components/ButtonGroup.js");
/* harmony import */ var _components_Card__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ./components/Card */ "./components/Card.js");
/* harmony import */ var _components_CardTitle__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ./components/CardTitle */ "./components/CardTitle.js");
/* harmony import */ var _components_CircleLoader__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ./components/CircleLoader */ "./components/CircleLoader.js");
/* harmony import */ var _components_Clickable__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ./components/Clickable */ "./components/Clickable.js");
/* harmony import */ var _components_Code__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ./components/Code */ "./components/Code.js");
/* harmony import */ var _components_CodeEditor__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ./components/CodeEditor */ "./components/CodeEditor.js");
/* harmony import */ var _components_CollapsiblePanel__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(/*! ./components/CollapsiblePanel */ "./components/CollapsiblePanel.js");
/* harmony import */ var _components_Columns__WEBPACK_IMPORTED_MODULE_11__ = __webpack_require__(/*! ./components/Columns */ "./components/Columns.js");
/* harmony import */ var _components_Container__WEBPACK_IMPORTED_MODULE_12__ = __webpack_require__(/*! ./components/Container */ "./components/Container.js");
/* harmony import */ var _components_ContextualDisplay__WEBPACK_IMPORTED_MODULE_13__ = __webpack_require__(/*! ./components/ContextualDisplay */ "./components/ContextualDisplay.js");
/* harmony import */ var _components_Dropdown__WEBPACK_IMPORTED_MODULE_14__ = __webpack_require__(/*! ./components/Dropdown */ "./components/Dropdown.js");
/* harmony import */ var _components_Expands__WEBPACK_IMPORTED_MODULE_15__ = __webpack_require__(/*! ./components/Expands */ "./components/Expands.js");
/* harmony import */ var _components_HeaderBar__WEBPACK_IMPORTED_MODULE_16__ = __webpack_require__(/*! ./components/HeaderBar */ "./components/HeaderBar.js");
/* harmony import */ var _components_LoadContentsFromUrl__WEBPACK_IMPORTED_MODULE_17__ = __webpack_require__(/*! ./components/LoadContentsFromUrl */ "./components/LoadContentsFromUrl.js");
/* harmony import */ var _components_LargePendingDownloadDisplay__WEBPACK_IMPORTED_MODULE_18__ = __webpack_require__(/*! ./components/LargePendingDownloadDisplay */ "./components/LargePendingDownloadDisplay.js");
/* harmony import */ var _components_Main__WEBPACK_IMPORTED_MODULE_19__ = __webpack_require__(/*! ./components/Main */ "./components/Main.js");
/* harmony import */ var _components_Modal__WEBPACK_IMPORTED_MODULE_20__ = __webpack_require__(/*! ./components/Modal */ "./components/Modal.js");
/* harmony import */ var _components_Octicon__WEBPACK_IMPORTED_MODULE_21__ = __webpack_require__(/*! ./components/Octicon */ "./components/Octicon.js");
/* harmony import */ var _components_Padding__WEBPACK_IMPORTED_MODULE_22__ = __webpack_require__(/*! ./components/Padding */ "./components/Padding.js");
/* harmony import */ var _components_Popover__WEBPACK_IMPORTED_MODULE_23__ = __webpack_require__(/*! ./components/Popover */ "./components/Popover.js");
/* harmony import */ var _components_RootCell__WEBPACK_IMPORTED_MODULE_24__ = __webpack_require__(/*! ./components/RootCell */ "./components/RootCell.js");
/* harmony import */ var _components_Sequence__WEBPACK_IMPORTED_MODULE_25__ = __webpack_require__(/*! ./components/Sequence */ "./components/Sequence.js");
/* harmony import */ var _components_Scrollable__WEBPACK_IMPORTED_MODULE_26__ = __webpack_require__(/*! ./components/Scrollable */ "./components/Scrollable.js");
/* harmony import */ var _components_SingleLineTextBox__WEBPACK_IMPORTED_MODULE_27__ = __webpack_require__(/*! ./components/SingleLineTextBox */ "./components/SingleLineTextBox.js");
/* harmony import */ var _components_Span__WEBPACK_IMPORTED_MODULE_28__ = __webpack_require__(/*! ./components/Span */ "./components/Span.js");
/* harmony import */ var _components_Subscribed__WEBPACK_IMPORTED_MODULE_29__ = __webpack_require__(/*! ./components/Subscribed */ "./components/Subscribed.js");
/* harmony import */ var _components_SubscribedSequence__WEBPACK_IMPORTED_MODULE_30__ = __webpack_require__(/*! ./components/SubscribedSequence */ "./components/SubscribedSequence.js");
/* harmony import */ var _components_Table__WEBPACK_IMPORTED_MODULE_31__ = __webpack_require__(/*! ./components/Table */ "./components/Table.js");
/* harmony import */ var _components_Tabs__WEBPACK_IMPORTED_MODULE_32__ = __webpack_require__(/*! ./components/Tabs */ "./components/Tabs.js");
/* harmony import */ var _components_Text__WEBPACK_IMPORTED_MODULE_33__ = __webpack_require__(/*! ./components/Text */ "./components/Text.js");
/* harmony import */ var _components_Traceback__WEBPACK_IMPORTED_MODULE_34__ = __webpack_require__(/*! ./components/Traceback */ "./components/Traceback.js");
/* harmony import */ var _components_NavTab__WEBPACK_IMPORTED_MODULE_35__ = __webpack_require__(/*! ./components/_NavTab */ "./components/_NavTab.js");
/* harmony import */ var _components_Grid__WEBPACK_IMPORTED_MODULE_36__ = __webpack_require__(/*! ./components/Grid */ "./components/Grid.js");
/* harmony import */ var _components_Sheet__WEBPACK_IMPORTED_MODULE_37__ = __webpack_require__(/*! ./components/Sheet */ "./components/Sheet.js");
/* harmony import */ var _components_Plot__WEBPACK_IMPORTED_MODULE_38__ = __webpack_require__(/*! ./components/Plot */ "./components/Plot.js");
/* harmony import */ var _components_PlotUpdater__WEBPACK_IMPORTED_MODULE_39__ = __webpack_require__(/*! ./components/_PlotUpdater */ "./components/_PlotUpdater.js");
/**
 * We use a singleton registry object
 * where we make available all possible
 * Components. This is useful for Webpack,
 * which only bundles explicitly used
 * Components during build time.
 */









































const ComponentRegistry = {
    AsyncDropdown: _components_AsyncDropdown__WEBPACK_IMPORTED_MODULE_0__["AsyncDropdown"],
    AsyncDropdownContent: _components_AsyncDropdown__WEBPACK_IMPORTED_MODULE_0__["AsyncDropdownContent"],
    Badge: _components_Badge__WEBPACK_IMPORTED_MODULE_1__["Badge"],
    Button: _components_Button__WEBPACK_IMPORTED_MODULE_2__["Button"],
    ButtonGroup: _components_ButtonGroup__WEBPACK_IMPORTED_MODULE_3__["ButtonGroup"],
    Card: _components_Card__WEBPACK_IMPORTED_MODULE_4__["Card"],
    CardTitle: _components_CardTitle__WEBPACK_IMPORTED_MODULE_5__["CardTitle"],
    CircleLoader: _components_CircleLoader__WEBPACK_IMPORTED_MODULE_6__["CircleLoader"],
    Clickable: _components_Clickable__WEBPACK_IMPORTED_MODULE_7__["Clickable"],
    Code: _components_Code__WEBPACK_IMPORTED_MODULE_8__["Code"],
    CodeEditor: _components_CodeEditor__WEBPACK_IMPORTED_MODULE_9__["CodeEditor"],
    CollapsiblePanel: _components_CollapsiblePanel__WEBPACK_IMPORTED_MODULE_10__["CollapsiblePanel"],
    Columns: _components_Columns__WEBPACK_IMPORTED_MODULE_11__["Columns"],
    Container: _components_Container__WEBPACK_IMPORTED_MODULE_12__["Container"],
    ContextualDisplay: _components_ContextualDisplay__WEBPACK_IMPORTED_MODULE_13__["ContextualDisplay"],
    Dropdown: _components_Dropdown__WEBPACK_IMPORTED_MODULE_14__["Dropdown"],
    Expands: _components_Expands__WEBPACK_IMPORTED_MODULE_15__["Expands"],
    HeaderBar: _components_HeaderBar__WEBPACK_IMPORTED_MODULE_16__["HeaderBar"],
    LoadContentsFromUrl: _components_LoadContentsFromUrl__WEBPACK_IMPORTED_MODULE_17__["LoadContentsFromUrl"],
    LargePendingDownloadDisplay: _components_LargePendingDownloadDisplay__WEBPACK_IMPORTED_MODULE_18__["LargePendingDownloadDisplay"],
    Main: _components_Main__WEBPACK_IMPORTED_MODULE_19__["Main"],
    Modal: _components_Modal__WEBPACK_IMPORTED_MODULE_20__["Modal"],
    Octicon: _components_Octicon__WEBPACK_IMPORTED_MODULE_21__["Octicon"],
    Padding: _components_Padding__WEBPACK_IMPORTED_MODULE_22__["Padding"],
    Popover: _components_Popover__WEBPACK_IMPORTED_MODULE_23__["Popover"],
    RootCell: _components_RootCell__WEBPACK_IMPORTED_MODULE_24__["RootCell"],
    Sequence: _components_Sequence__WEBPACK_IMPORTED_MODULE_25__["Sequence"],
    Scrollable: _components_Scrollable__WEBPACK_IMPORTED_MODULE_26__["Scrollable"],
    SingleLineTextBox: _components_SingleLineTextBox__WEBPACK_IMPORTED_MODULE_27__["SingleLineTextBox"],
    Span: _components_Span__WEBPACK_IMPORTED_MODULE_28__["Span"],
    Subscribed: _components_Subscribed__WEBPACK_IMPORTED_MODULE_29__["Subscribed"],
    SubscribedSequence: _components_SubscribedSequence__WEBPACK_IMPORTED_MODULE_30__["SubscribedSequence"],
    Table: _components_Table__WEBPACK_IMPORTED_MODULE_31__["Table"],
    Tabs: _components_Tabs__WEBPACK_IMPORTED_MODULE_32__["Tabs"],
    Text: _components_Text__WEBPACK_IMPORTED_MODULE_33__["Text"],
    Traceback: _components_Traceback__WEBPACK_IMPORTED_MODULE_34__["Traceback"],
    _NavTab: _components_NavTab__WEBPACK_IMPORTED_MODULE_35__["_NavTab"],
    Grid: _components_Grid__WEBPACK_IMPORTED_MODULE_36__["Grid"],
    Sheet: _components_Sheet__WEBPACK_IMPORTED_MODULE_37__["Sheet"],
    Plot: _components_Plot__WEBPACK_IMPORTED_MODULE_38__["Plot"],
    _PlotUpdater: _components_PlotUpdater__WEBPACK_IMPORTED_MODULE_39__["_PlotUpdater"]
};




/***/ }),

/***/ "./components/AsyncDropdown.js":
/*!*************************************!*\
  !*** ./components/AsyncDropdown.js ***!
  \*************************************/
/*! exports provided: AsyncDropdown, AsyncDropdownContent, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "AsyncDropdown", function() { return AsyncDropdown; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "AsyncDropdownContent", function() { return AsyncDropdownContent; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return AsyncDropdown; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * AsyncDropdown Cell Component
 */




/**
 * About Replacements
 * -------------------
 * This component has a single regular
 * replacement:
 * * `contents`
 *
 * NOTE: The Cells version of this child is
 * either a loading indicator, text, or a
 * AsyncDropdownContent cell.
 */

/**
 * About Named Children
 * --------------------
 * `content` (single) - Usually an AsyncDropdownContent cell
 * `loadingIndicator` (single) - A Cell that displays that the content is loading
 */
class AsyncDropdown extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
        this.addDropdownListener = this.addDropdownListener.bind(this);
        this.makeContent = this.makeContent.bind(this);
    }

    render(){
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "AsyncDropdown",
                class: "cell btn-group"
            }, [
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('a', {class: "btn btn-xs btn-outline-secondary"}, [this.props.extraData.labelText]),
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('button', {
                    class: "btn btn-xs btn-outline-secondary dropdown-toggle dropdown-toggle-split",
                    type: "button",
                    id: `${this.props.id}-dropdownMenuButton`,
                    "data-toggle": "dropdown",
                    afterCreate: this.addDropdownListener,
                    "data-firstclick": "true"
                }),
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                    id: `${this.props.id}-dropdownContentWrapper`,
                    class: "dropdown-menu"
                }, [this.makeContent()])
            ])
        );
    }

    addDropdownListener(element){
        let parentEl = element.parentElement;
        let component = this;
        let firstTimeClicked = (element.dataset.firstclick == "true");
        if(firstTimeClicked){
            $(parentEl).on('show.bs.dropdown', function(){
                cellSocket.sendString(JSON.stringify({
                    event:'dropdown',
                    target_cell: component.props.id,
                    isOpen: false
                }));
            });
            $(parentEl).on('hide.bs.dropdown', function(){
                cellSocket.sendString(JSON.stringify({
                    event: 'dropdown',
                    target_cell: component.props.id,
                    isOpen: true
                }));
            });
            element.dataset.firstclick = false;
        }
    }

    makeContent(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('contents');
        } else {
            return this.renderChildNamed('content');
        }
    }
}

/**
 * About Replacements
 * ------------------
 * This component has a single regular
 * replacement:
 * * `contents`
 */

/**
 * About Named Children
 * ---------------------
 * `content` (single) - A Cell that comprises the dropdown content
 * `loadingIndicator` (single) - A Cell that represents a visual
 *       indicating that the content is loading
 */
class AsyncDropdownContent extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeContent = this.makeContent.bind(this);
    }

    render(){
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                id: `dropdownContent-${this.props.id}`,
                "data-cell-id": this.props.id,
                "data-cell-type": "AsyncDropdownContent"
            }, [this.makeContent()])
        );
    }

    makeContent(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('contents');
        } else {
            return this.renderChildNamed('content');
        }
    }
}





/***/ }),

/***/ "./components/Badge.js":
/*!*****************************!*\
  !*** ./components/Badge.js ***!
  \*****************************/
/*! exports provided: Badge, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Badge", function() { return Badge; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return Badge; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * Badge Cell Component
 */



/**
 * About Replacements
 * ------------------
 * Badge has a single replacement:
 * * `child`
 */

/**
 * About Named Children
 * --------------------
 * `inner` - The concent cell of the Badge
 */
class Badge extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(...args);

        // Bind component methods
        this.makeInner = this.makeInner.bind(this);
    }

    render(){
        return(
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('span', {
                class: `cell badge badge-${this.props.extraData.badgeStyle}`,
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Badge"
            }, [this.makeContent()])
        );
    }

    makeInner(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('child');
        } else {
            return this.renderChildNamed('inner');
        }
    }
}




/***/ }),

/***/ "./components/Button.js":
/*!******************************!*\
  !*** ./components/Button.js ***!
  \******************************/
/*! exports provided: Button, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Button", function() { return Button; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return Button; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * Button Cell Component
 */




/**
 * About Replacements
 * ------------------
 * This component has one regular replacement:
 * `contents`
 */

/**
 * About Named Children
 * ---------------------
 * `content` (single) - The cell inside of the button (if any)
 */
class Button extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
        this.makeContent = this.makeContent.bind(this);
        this._getEvents = this._getEvent.bind(this);
        this._getHTMLClasses = this._getHTMLClasses.bind(this);
    }

    render(){
        return(
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('button', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Button",
                class: this._getHTMLClasses(),
                onclick: this._getEvent('onclick')
            }, [this.makeContent()]
             )
        );
    }

    makeContent(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('contents');
        } else {
            return this.renderChildNamed('content');
        }
    }

    _getEvent(eventName) {
        return this.props.extraData.events[eventName];
    }

    _getHTMLClasses(){
        let classString = this.props.extraData.classes.join(" ");
        // remember to trim the class string due to a maquette bug
        return classString.trim();
    }
}




/***/ }),

/***/ "./components/ButtonGroup.js":
/*!***********************************!*\
  !*** ./components/ButtonGroup.js ***!
  \***********************************/
/*! exports provided: ButtonGroup, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ButtonGroup", function() { return ButtonGroup; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return ButtonGroup; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * ButtonGroup Cell Component
 */




/**
 * About Replacements
 * ------------------
 * This component has a single enumerated
 * replacement:
 * * `button`
 */

/**
 * About Named Children
 * --------------------
 * `buttons` (array) - The constituent button cells
 */
class ButtonGroup extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeButtons = this.makeButtons.bind(this);
    }

    render(){
        return(
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "ButtonGroup",
                class: "btn-group",
                "role": "group"
            }, this.makeButtons()
             )
        );
    }

    makeButtons(){
        if(this.usesReplacements){
            return this.getReplacementElementsFor('button');
        } else {
            return this.renderChildrenNamed('buttons');
        }
    }

}




/***/ }),

/***/ "./components/Card.js":
/*!****************************!*\
  !*** ./components/Card.js ***!
  \****************************/
/*! exports provided: Card, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Card", function() { return Card; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return Card; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var _util_PropertyValidator__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./util/PropertyValidator */ "./components/util/PropertyValidator.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_2__);
/**
 * Card Cell Component
 */





/**
 * About Replacements
 * ------------------
 * This component contains two
 * regular replacements:
 * * `contents`
 * * `header`
 */

/**
 * About Named Children
 * `body` (single) - The cell to put in the body of the Card
 * `header` (single) - An optional header cell to put above
 *        body
 */
class Card extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeBody = this.makeBody.bind(this);
        this.makeHeader = this.makeHeader.bind(this);
    }

    render(){
        let bodyClass = "card-body";
        if(this.props.extraData.padding){
            bodyClass = `card-body p-${this.props.extraData.padding}`;
        }
        let bodyArea = Object(maquette__WEBPACK_IMPORTED_MODULE_2__["h"])('div', {
            class: bodyClass
        }, [this.makeBody()]);
        let header = this.makeHeader();
        let headerArea = null;
        if(header){
            headerArea = Object(maquette__WEBPACK_IMPORTED_MODULE_2__["h"])('div', {class: "card-header"}, [header]);
        }
        return Object(maquette__WEBPACK_IMPORTED_MODULE_2__["h"])('div',
            {
                class: "cell card",
                style: this.props.divStyle,
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Card"
            }, [headerArea, bodyArea]);
    }

    makeBody(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('contents');
        } else {
            return this.renderChildNamed('body');
        }
    }

    makeHeader(){
        if(this.usesReplacements){
            if(this.replacements.hasReplacement('header')){
                return this.getReplacementElementFor('header');
            }
            return null;
        } else {
            return this.renderChildNamed('header');
        }
    }
}

Card.propTypes = {
    padding: {
        description: "Padding weight as defined by Boostrap css classes.",
        type: _util_PropertyValidator__WEBPACK_IMPORTED_MODULE_1__["PropTypes"].oneOf([_util_PropertyValidator__WEBPACK_IMPORTED_MODULE_1__["PropTypes"].number, _util_PropertyValidator__WEBPACK_IMPORTED_MODULE_1__["PropTypes"].string])
    },
    divStyle: {
        description: "HTML style attribute string.",
        type: _util_PropertyValidator__WEBPACK_IMPORTED_MODULE_1__["PropTypes"].oneOf([_util_PropertyValidator__WEBPACK_IMPORTED_MODULE_1__["PropTypes"].string])
    }
};




/***/ }),

/***/ "./components/CardTitle.js":
/*!*********************************!*\
  !*** ./components/CardTitle.js ***!
  \*********************************/
/*! exports provided: CardTitle, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "CardTitle", function() { return CardTitle; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return CardTitle; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * CardTitle Cell
 */





/**
 * About Replacements
 * ------------------
 * This component has  single regular
 * replacement:
 * * `contents`
 */

/**
 * About Named Children
 * --------------------
 * `inner` (single) - The inner cell of the title component
 */
class CardTitle extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeInner = this.makeInner.bind(this);
    }

    render(){
        return(
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                id: this.props.id,
                class: "cell",
                "data-cell-id": this.props.id,
                "data-cell-type": "CardTitle"
            }, [
                this.makeInner()
            ])
        );
    }

    makeInner(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('contents');
        } else {
            return this.renderChildNamed('inner');
        }
    }
}




/***/ }),

/***/ "./components/CircleLoader.js":
/*!************************************!*\
  !*** ./components/CircleLoader.js ***!
  \************************************/
/*! exports provided: CircleLoader, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "CircleLoader", function() { return CircleLoader; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return CircleLoader; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * CircleLoader Cell Component
 */





class CircleLoader extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "CircleLoader",
                class: "spinner-grow",
                role: "status"
            })
        );
    }
}

CircleLoader.propTypes = {
};




/***/ }),

/***/ "./components/Clickable.js":
/*!*********************************!*\
  !*** ./components/Clickable.js ***!
  \*********************************/
/*! exports provided: Clickable, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Clickable", function() { return Clickable; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return Clickable; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * Clickable Cell Component
 */



/**
 * About Replacements
 * ------------------
 * This component has a single
 * regular replacement:
 * * `contents`
 */

/**
 * About Named Children
 * --------------------
 * `content` (single) - The cell that can go inside the clickable
 *        component
 */
class Clickable extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
        this.makeContent = this.makeContent.bind(this);
        this._getEvents = this._getEvent.bind(this);
    }

    render(){
        return(
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Clickable",
                onclick: this._getEvent('onclick'),
                style: this.props.extraData.divStyle
            }, [
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {}, [this.makeContent()])
            ]
            )
        );
    }

    makeContent(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('contents');
        } else {
            return this.renderChildNamed('content');
        }
    }

    _getEvent(eventName) {
        return this.props.extraData.events[eventName];
    }
}




/***/ }),

/***/ "./components/Code.js":
/*!****************************!*\
  !*** ./components/Code.js ***!
  \****************************/
/*! exports provided: Code, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Code", function() { return Code; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return Code; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * Code Cell Component
 */




/**
 * About Replacements
 * ------------------
 * This component has a single
 * regular replacement:
 * * `child`
 */

/**
 * About Named Children
 * --------------------
 * `code` (single) - Code that will be rendered inside
 */
class Code extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeCode = this.makeCode.bind(this);
    }

    render(){
        return Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('pre',
                 {
                     class: "cell code",
                     id: this.props.id,
                     "data-cell-type": "Code"
                 }, [
                     Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])("code", {}, [this.makeCode()])
                 ]
                );
    }

    makeCode(){
        if(this.usesReplacements){
            return this.getReplacementelementFor('child');
        } else {
            return this.renderChildNamed('code');
        }
    }
}




/***/ }),

/***/ "./components/CodeEditor.js":
/*!**********************************!*\
  !*** ./components/CodeEditor.js ***!
  \**********************************/
/*! exports provided: CodeEditor, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "CodeEditor", function() { return CodeEditor; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return CodeEditor; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * CodeEditor Cell Component
 */




class CodeEditor extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);
        this.editor = null;
        // used to schedule regular server updates
        this.SERVER_UPDATE_DELAY_MS = 1;
        this.editorStyle = 'width:100%;height:100%;margin:auto;border:1px solid lightgray;';

        this.setupEditor = this.setupEditor.bind(this);
        this.setupKeybindings = this.setupKeybindings.bind(this);
        this.changeHandler = this.changeHandler.bind(this);
    }

    componentDidLoad() {

        this.setupEditor();

        if (this.editor === null) {
            console.log("editor component loaded but failed to setup editor");
        } else {
            console.log("setting up editor");
            this.editor.last_edit_millis = Date.now();

            this.editor.setTheme("ace/theme/textmate");
            this.editor.session.setMode("ace/mode/python");
            this.editor.setAutoScrollEditorIntoView(true);
            this.editor.session.setUseSoftTabs(true);
            this.editor.setValue(this.props.extraData.initialText);

            if (this.props.extraData.autocomplete) {
                this.editor.setOptions({enableBasicAutocompletion: true});
                this.editor.setOptions({enableLiveAutocompletion: true});
            }

            if (this.props.extraData.noScroll) {
                this.editor.setOption("maxLines", Infinity);
            }

            if (this.props.extraData.fontSize !== undefined) {
                this.editor.setOption("fontSize", this.props.extraData.fontSize);
            }

            if (this.props.extraData.minLines !== undefined) {
                this.editor.setOption("minLines", this.props.extraData.minLines);
            }

            this.setupKeybindings();

            this.changeHandler();
        }
    }

    render(){
        return Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div',
            {
                class: "cell",
                style: this.props.extraData.divStyle,
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "CodeEditor"
            },
            [Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', { id: "editor" + this.props.id, style: this.editorStyle }, [])
        ]);
    }

    setupEditor(){
        let editorId = "editor" + this.props.id;
        // TODO These are global var defined in page.html
        // we should do something about this.

        // here we bing and inset the editor into the div rendered by
        // this.render()
        this.editor = ace.edit(editorId);
        // TODO: deal with this global editor list
        aceEditors[editorId] = this.editor;
    }

    changeHandler() {
	var editorId = this.props.id;
	var editor = this.editor;
	var SERVER_UPDATE_DELAY_MS = this.SERVER_UPDATE_DELAY_MS;
        this.editor.session.on(
            "change",
            function(delta) {
                // WS
                let responseData = {
                    event: 'editor_change',
                    'target_cell': editorId,
                    data: delta
                };
                cellSocket.sendString(JSON.stringify(responseData));
                //record that we just edited
                editor.last_edit_millis = Date.now();

		//schedule a function to run in 'SERVER_UPDATE_DELAY_MS'ms
		//that will update the server, but only if the user has stopped typing.
		// TODO unclear if this is owrking properly
		window.setTimeout(function() {
		    if (Date.now() - editor.last_edit_millis >= SERVER_UPDATE_DELAY_MS) {
			//save our current state to the remote buffer
			editor.current_iteration += 1;
			editor.last_edit_millis = Date.now();
			editor.last_edit_sent_text = editor.getValue();
			// WS
			let responseData = {
			    event: 'editing',
			    'target_cell': editorId,
			    buffer: editor.getValue(),
			    selection: editor.selection.getRange(),
			    iteration: editor.current_iteration
			};
			cellSocket.sendString(JSON.stringify(responseData));
		    }
		}, SERVER_UPDATE_DELAY_MS + 2); //note the 2ms grace period
            }
        );
    }

    setupKeybindings() {
        console.log("setting up keybindings");
        this.props.extraData.keybindings.map((kb) => {
            this.editor.commands.addCommand(
                {
                    name: 'cmd' + kb,
                    bindKey: {win: 'Ctrl-' + kb,  mac: 'Command-' + kb},
                    readOnly: true,
                    exec: () => {
                        this.editor.current_iteration += 1;
                        this.editor.last_edit_millis = Date.now();
                        this.editor.last_edit_sent_text = this.editor.getValue();

                        // WS
                        let responseData = {
                            event: 'keybinding',
                            'target_cell': this.props.id,
                            'key': kb,
                            'buffer': this.editor.getValue(),
                            'selection': this.editor.selection.getRange(),
                            'iteration': this.editor.current_iteration
                        };
                        cellSocket.sendString(JSON.stringify(responseData));
                    }
                }
            );
        });
    }
}




/***/ }),

/***/ "./components/CollapsiblePanel.js":
/*!****************************************!*\
  !*** ./components/CollapsiblePanel.js ***!
  \****************************************/
/*! exports provided: CollapsiblePanel, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "CollapsiblePanel", function() { return CollapsiblePanel; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return CollapsiblePanel; });
/* harmony import */ var _Component_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component.js */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * CollapsiblePanel Cell Component
 */



/**
 * About Replacements
 * ------------------
 * This component has two single type
 * replacements:
 * * `content`
 * * `panel`
 * Note that `panel` is only rendered
 * if the panel is expanded
 */

/**
 * About Named Children
 * --------------------
 * `content` (single) - The current content Cell of the panel
 * `panel` (single) - The current (expanded) panel view
 */
class CollapsiblePanel extends _Component_js__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makePanel = this.makePanel.bind(this);
        this.makeContent = this.makeContent.bind(this);
    }

    render(){
        if(this.props.extraData.isExpanded){
            return(
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                    class: "cell container-fluid",
                    "data-cell-id": this.props.id,
                    "data-cell-type": "CollapsiblePanel",
                    "data-expanded": true,
                    id: this.props.id,
                    style: this.props.extraData.divStyle
                }, [
                    Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {class: "row flex-nowrap no-gutters"}, [
                        Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {class: "col-md-auto"},[
                            this.makePanel()
                        ]),
                        Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {class: "col-sm"}, [
                            this.makeContent()
                        ])
                    ])
                ])
            );
        } else {
            return (
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                    class: "cell container-fluid",
                    "data-cell-id": this.props.id,
                    "data-cell-type": "CollapsiblePanel",
                    "data-expanded": false,
                    id: this.props.id,
                    style: this.props.extraData.divStyle
                }, [this.makeContent()])
            );
        }
    }

    makeContent(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('content');
        } else {
            return this.renderChildNamed('content');
        }
    }

    makePanel(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('panel');
        } else {
            return this.renderChildNamed('panel');
        }
    }
}





/***/ }),

/***/ "./components/Columns.js":
/*!*******************************!*\
  !*** ./components/Columns.js ***!
  \*******************************/
/*! exports provided: Columns, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Columns", function() { return Columns; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return Columns; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * Columns Cell Component
 */




/**
 * About Replacements
 * ------------------
 * This component has one enumerated
 * kind of replacement:
 * * `c`
 */

/**
 * About Named Children
 * --------------------
 * `elements` (array) - Cell column elements
 */
class Columns extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
        this.makeInnerChildren = this.makeInnerChildren.bind(this);
    }

    render(){
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                class: "cell container-fluid",
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Columns",
                style: this.props.extraData.divStyle
            }, [
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {class: "row flex-nowrap"}, this.makeInnerChildren())
            ])
        );
    }

    makeInnerChildren(){
        if(this.usesReplacements){
            return this.getReplacementElementsFor('c').map(replElement => {
                return (
                    Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                        class: "col-sm"
                    }, [replElement])
                );
            });
        } else {
            return this.renderChildrenNamed('elements');
        }
    }
}





/***/ }),

/***/ "./components/Component.js":
/*!*********************************!*\
  !*** ./components/Component.js ***!
  \*********************************/
/*! exports provided: Component, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Component", function() { return Component; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return Component; });
/* harmony import */ var _util_ReplacementsHandler__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./util/ReplacementsHandler */ "./components/util/ReplacementsHandler.js");
/* harmony import */ var _util_PropertyValidator__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./util/PropertyValidator */ "./components/util/PropertyValidator.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_2__);
/**
 * Generic base Cell Component.
 * Should be extended by other
 * Cell classes on JS side.
 */




class Component {
    constructor(props = {}, replacements = []){
        this.isComponent = true;
        this._updateProps(props);

        // Replacements handling
        this.replacements = new _util_ReplacementsHandler__WEBPACK_IMPORTED_MODULE_0__["ReplacementsHandler"](replacements);
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
            return Object(maquette__WEBPACK_IMPORTED_MODULE_2__["h"])('div', {id: newId, key: newId}, []);
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
                Object(maquette__WEBPACK_IMPORTED_MODULE_2__["h"])('div', {id: newId, key: newId})
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
            _util_PropertyValidator__WEBPACK_IMPORTED_MODULE_1__["PropTypes"].validate(
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




/***/ }),

/***/ "./components/Container.js":
/*!*********************************!*\
  !*** ./components/Container.js ***!
  \*********************************/
/*! exports provided: Container, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Container", function() { return Container; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return Container; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * Container Cell Component
 */




/**
 * About Replacements
 * ------------------
 * This component has a single regular
 * replacement:
 * * `child`
 */

/**
 * About Named Children
 * --------------------
 * `child` (single) - The Cell that this component contains
 */
class Container extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeChild = this.makeChild.bind(this);
    }

    render(){
        let child = this.makeChild();
        let style = "";
        if(!child){
            style = "display:none;";
        }
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Container",
                class: "cell",
                style: style
            }, [child])
        );
    }

    makeChild(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('child');
        } else {
            return this.renderChildNamed('child');
        }
    }
}




/***/ }),

/***/ "./components/ContextualDisplay.js":
/*!*****************************************!*\
  !*** ./components/ContextualDisplay.js ***!
  \*****************************************/
/*! exports provided: ContextualDisplay, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ContextualDisplay", function() { return ContextualDisplay; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return ContextualDisplay; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * ContextualDisplay Cell Component
 */




/**
 * About Replacements
 * ------------------
 * This component has a single
 * regular replacement:
 * * `child`
 */

/**
 * About Named Children
 * --------------------
 * `child` (single) - A child cell to display in a context
 */
class ContextualDisplay extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeChild = this.makeChild.bind(this);
    }

    render(){
        return Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div',
            {
                class: "cell contextualDisplay",
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "ContextualDisplay"
            }, [this.makeChild()]
        );
    }

    makeChild(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('child');
        } else {
            return this.renderChildNamed('child');
        }
    }
}




/***/ }),

/***/ "./components/Dropdown.js":
/*!********************************!*\
  !*** ./components/Dropdown.js ***!
  \********************************/
/*! exports provided: Dropdown, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Dropdown", function() { return Dropdown; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return Dropdown; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * Dropdown Cell Component
 */




/**
 * About Replacements
 * ------------------
 * This component has one regular
 * replacement:
 * * `title`
 * This component has one
 * enumerated replacement:
 * * `child`
 */

/**
 * About Named Children
 * --------------------
 * `title` (single) - A Cell that will comprise the title of
 *      the dropdown
 * `dropdownItems` (array) - An array of cells that are
 *      the items in the dropdown
 */
class Dropdown extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
        this.makeTitle = this.makeTitle.bind(this);
        this.makeItems = this.makeItems.bind(this);
    }

    render(){
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Dropdown",
                class: "btn-group"
            }, [
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('a', {class: "btn btn-xs btn-outline-secondary"}, [
                    this.makeTitle()
                ]),
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('button', {
                    class: "btn btn-xs btn-outline-secondary dropdown-toggle dropdown-toggle-split",
                    type: "button",
                    id: `${this.props.extraData.targetIdentity}-dropdownMenuButton`,
                    "data-toggle": "dropdown"
                }),
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {class: "dropdown-menu"}, this.makeItems())
            ])
        );
    }

    makeTitle(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('title');
        } else {
            return this.renderChildNamed('title');
        }
    }

    makeItems(){
        if(this.usesReplacements){
            // For some reason, due again to the Cell implementation,
            // sometimes there are not these child replacements.
            if(!this.replacements.hasReplacement('child')){
                return [];
            }
            return this.getReplacementElementsFor('child').map((element, idx) => {
                return new DropdownItem({
                    id: `${this.props.id}-item-${idx}`,
                    index: idx,
                    childSubstitute: element,
                    targetIdentity: this.props.extraData.targetIdentity,
                    dropdownItemInfo: this.props.extraData.dropdownItemInfo
                }).render();
            });
        } else {
            if(this.props.namedChildren.dropdownItems){
                return this.props.namedChildren.dropdownItems.map((itemComponent, idx) => {
                    return new DropdowItem({
                        id: `${this.propd.id}-item-${idx}`,
                        index: idx,
                        childSubstitute: itemComponent.render(),
                        targetIdentity: this.props.extraData.targetIdentity,
                        dropdownItemInfo: this.props.extraData.dropdownItemInfo
                    });
                });
            } else {
                return [];
            }
        }
    }
}


/**
 * A private subcomponent for each
 * Dropdown menu item. We need this
 * because of how callbacks are handled
 * and because the Cells version doesn't
 * already implement this kind as a separate
 * entity.
 */
class DropdownItem extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
        this.clickHandler = this.clickHandler.bind(this);
    }

    render(){
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('a', {
                class: "subcell cell-dropdown-item dropdown-item",
                key: this.props.index,
                onclick: this.clickHandler
            }, [this.props.childSubstitute])
        );
    }

    clickHandler(event){
        // This is super hacky because of the
        // current Cell implementation.
        // This whole component structure should be heavily refactored
        // once the Cells side of things starts to change.
        let whatToDo = this.props.dropdownItemInfo[this.props.index.toString()];
        if(whatToDo == 'callback'){
            let responseData = {
                event: "menu",
                ix: this.props.index,
                target_cell: this.props.targetIdentity
            };
            cellSocket.sendString(JSON.stringify(responseData));
        } else {
            window.location.href = whatToDo;
        }
    }
}




/***/ }),

/***/ "./components/Expands.js":
/*!*******************************!*\
  !*** ./components/Expands.js ***!
  \*******************************/
/*! exports provided: Expands, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Expands", function() { return Expands; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return Expands; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * Expands Cell Component
 */

/** TODO/NOTE: It appears that the open/closed
    State for this component could simply be passed
    with the Cell data, along with what to display
    in either case. This would be how it is normally
    done in large web applications.
    Consider refactoring both here and on the Cells
    side
**/





/**
 * About Replacements
 * ------------------
 * This component has two
 * regular replacements:
 * * `icon`
 * * `child`
 */

/**
 * About Named Children
 * --------------------
 * `content` (single) - The open or closed cell, depending on source
 *     open state
 * `icon` (single) - The Cell of the icon to display, also depending
 *     on closed or open state
 */
class Expands extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
        this.makeIcon = this.makeIcon.bind(this);
        this.makeContent = this.makeContent.bind(this);
        this._getEvents = this._getEvent.bind(this);
    }

    render(){
        return(
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Expands",
                style: this.props.extraData.divStyle
            },
                [
                    Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                        style: 'display:inline-block;vertical-align:top',
                        onclick: this._getEvent('onclick')
                    },
                      [this.makeIcon()]),
                    Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {style:'display:inline-block'},
                      [this.makeContent()]),
                ]
            )
        );
    }

    makeContent(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('child');
        } else {
            return this.renderChildNamed('content');
        }
    }

    makeIcon(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('icon');
        } else {
            return this.renderChildNamed('icon');
        }
    }

    _getEvent(eventName) {
        return this.props.extraData.events[eventName];
    }
}




/***/ }),

/***/ "./components/Grid.js":
/*!****************************!*\
  !*** ./components/Grid.js ***!
  \****************************/
/*! exports provided: Grid, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Grid", function() { return Grid; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return Grid; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * Grid Cell Component
 */




/**
 * About Replacements
 * ------------------
 * This component has 3 enumerable
 * replacements:
 * * `header`
 * * `rowlabel`
 * * `child`
 *
 * NOTE: Child is a 2-dimensional
 * enumerated replacement!
 */

/**
 * About Named Children
 * --------------------
 * `headers` (array) - An array of table header cells
 * `rowLabels` (array) - An array of row label cells
 * `dataCells` (array-of-array) - A 2-dimensional array
 *     of cells that serve as table data, where rows
 *     are the outer array and columns are the inner
 *     array.
 */
class Grid extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
        this.makeHeaders = this.makeHeaders.bind(this);
        this.makeRows = this.makeRows.bind(this);
        this._makeReplacementHeaderElements = this._makeReplacementHeaderElements.bind(this);
        this._makeReplacementRowElements = this._makeReplacementRowElements.bind(this);
    }

    render(){
        let topTableHeader = null;
        if(this.props.extraData.hasTopHeader){
            topTableHeader = Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('th');
        }
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('table', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Grid",
                class: "cell table-hscroll table-sm table-striped"
            }, [
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('thead', {}, [
                    Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('tr', {}, [topTableHeader, ...this.makeHeaders()])
                ]),
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('tbody', {}, this.makeRows())
            ])
        );
    }

    makeHeaders(){
        if(this.usesReplacements){
            return this._makeReplacementHeaderElements();
        } else {
            return this.renderChildrenNamed('headers').map((headerEl, colIdx) => {
                return (
                    Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('th', {key: `${this.props.id}-grid-th-${colIdx}`}, [
                        headerEl
                    ])
                );
            });
        }
    }

    makeRows(){
        if(this.usesReplacements){
            return this._makeReplacementRowElements();
        } else {
            return this.renderChildrenNamed('dataCells').map((dataRow, rowIdx) => {
                let columns = dataRow.map((column, colIdx) => {
                    return(
                        Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('td', {key: `${this.props.id}-grid-col-${rowIdx}-${colIdx}`}, [
                            column
                        ])
                    );
                });
                let rowLabelEl = null;
                if(this.props.namedChildren.rowLabels && this.props.namedChildren.rowLabels.length > 0){
                    rowLabelEl = Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('th', {key: `${this.props.id}-grid-col-${rowIdx}-${colIdx}`}, [
                        this.props.namedChildren.rowLabels[rowIdx].render()
                    ]);
                }
                return (
                    Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('tr', {key: `${this.props.id}-grid-row-${rowIdx}`}, [
                        rowLabelEl,
                        ...columns
                    ])
                );
            });
        }
    }

    _makeReplacementRowElements(){
        return this.getReplacementElementsFor('child').map((row, rowIdx) => {
            let columns = row.map((column, colIdx) => {
                return (
                    Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('td', {key: `${this.props.id}-grid-col-${rowIdx}-${colIdx}`}, [
                        column
                    ])
                );
            });
            let rowLabelEl = null;
            if(this.replacements.hasReplacement('rowlabel')){
                rowLabelEl = Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('th', {key: `${this.props.id}-grid-rowlbl-${rowIdx}`}, [
                    this.getReplacementElementsFor('rowlabel')[rowIdx]
                ]);
            }
            return (
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('tr', {key: `${this.props.id}-grid-row-${rowIdx}`}, [
                    rowLabelEl,
                    ...columns
                ])
            );
        });
    }

    _makeReplacementHeaderElements(){
        return this.getReplacementElementsFor('header').map((headerEl, colIdx) => {
            return (
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('th', {key: `${this.props.id}-grid-th-${colIdx}`}, [
                    headerEl
                ])
            );
        });
    }
}




/***/ }),

/***/ "./components/HeaderBar.js":
/*!*********************************!*\
  !*** ./components/HeaderBar.js ***!
  \*********************************/
/*! exports provided: HeaderBar, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "HeaderBar", function() { return HeaderBar; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return HeaderBar; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * HeaderBar Cell Component
 */




/**
 * About Replacements
 * ------------------
 * This component has three separate
 * enumerated replacements:
 * * `left`
 * * `right`
 * * `center`
 */

/**
 * About Named Children
 * --------------------
 * `leftItems` (array) - The items that will be on the left
 * `centerItems` (array) - The items that will be in the center
 * `rightItems` (array) - The items that will be on the right
 */
class HeaderBar extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
        this.makeElements = this.makeElements.bind(this);
        this.makeRight = this.makeRight.bind(this);
        this.makeLeft = this.makeLeft.bind(this);
        this.makeCenter = this.makeCenter.bind(this);
    }

    render(){
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                id: this.props.id,
                class: "cell p-2 bg-light flex-container",
                "data-cell-id": this.props.id,
                "data-cell-type": "HeaderBar",
                style: 'display:flex;align-items:baseline;'
            }, [
                this.makeLeft(),
                this.makeCenter(),
                this.makeRight()
            ])
        );
    }

    makeLeft(){
        let innerElements = [];
        if(this.replacements.hasReplacement('left') || this.props.namedChildren.leftItems){
            innerElements = this.makeElements('left');
        }
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {class: "flex-item", style: "flex-grow:0;"}, [
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                    class: "flex-container",
                    style: 'display:flex;justify-content:center;align-items:baseline;'
                }, innerElements)
            ])
        );
    }

    makeCenter(){
        let innerElements = [];
        if(this.replacements.hasReplacement('center') || this.props.namedChildren.centerItems){
            innerElements = this.makeElements('center');
        }
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {class: "flex-item", style: "flex-grow:1;"}, [
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                    class: "flex-container",
                    style: 'display:flex;justify-content:center;align-items:baseline;'
                }, innerElements)
            ])
        );
    }

    makeRight(){
        let innerElements = [];
        if(this.replacements.hasReplacement('right') || this.props.namedChildren.rightItems){
            innerElements = this.makeElements('right');
        }
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {class: "flex-item", style: "flex-grow:0;"}, [
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                    class: "flex-container",
                    style: 'display:flex;justify-content:center;align-items:baseline;'
                }, innerElements)
            ])
        );
    }

    makeElements(position){
        if(this.usesReplacements){
            return this.getReplacementElementsFor(position).map(element => {
                return (
                    Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('span', {class: "flex-item px-3"}, [element])
                );
            });
        } else {
            return this.renderChildrenNamed(`${position}Items`).map(element => {
                return (
                    Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('span', {class: "flex-item px-3"}, [element])
                );
            });
        }
    }
}




/***/ }),

/***/ "./components/LargePendingDownloadDisplay.js":
/*!***************************************************!*\
  !*** ./components/LargePendingDownloadDisplay.js ***!
  \***************************************************/
/*! exports provided: LargePendingDownloadDisplay, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "LargePendingDownloadDisplay", function() { return LargePendingDownloadDisplay; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return LargePendingDownloadDisplay; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * LargePendingDownloadDisplay Cell Component
 */




class LargePendingDownloadDisplay extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                id: 'object_database_large_pending_download_text',
                "data-cell-id": this.props.id,
                "data-cell-type": "LargePendingDownloadDisplay",
                class: "cell"
            })
        );
    }
}




/***/ }),

/***/ "./components/LoadContentsFromUrl.js":
/*!*******************************************!*\
  !*** ./components/LoadContentsFromUrl.js ***!
  \*******************************************/
/*! exports provided: LoadContentsFromUrl, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "LoadContentsFromUrl", function() { return LoadContentsFromUrl; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return LoadContentsFromUrl; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * LoadContentsFromUrl Cell Component
 */




class LoadContentsFromUrl extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return(
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "LoadContentsFromUrl",
            }, [Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {id: this.props.extraData['loadTargetId']}, [])]
            )
        );
    }

}




/***/ }),

/***/ "./components/Main.js":
/*!****************************!*\
  !*** ./components/Main.js ***!
  \****************************/
/*! exports provided: Main, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Main", function() { return Main; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return Main; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * Main Cell Component
 */




/**
 * About Replacements
 * --------------------
 * This component has a one
 * regular-kind replacement:
 * * `child`
 */

/**
 * About Named Children
 * --------------------
 * `child` (single) - The child cell that is wrapped
 */
class Main extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeChild = this.makeChild.bind(this);
    }

    render(){
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('main', {
                id: this.props.id,
                class: "py-md-2",
                "data-cell-id": this.props.id,
                "data-cell-type": "Main"
            }, [
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {class: "container-fluid"}, [
                    this.makeChild()
                ])
            ])
        );
    }

    makeChild(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('child');
        } else {
            return this.renderChildNamed('child');
        }
    }
}




/***/ }),

/***/ "./components/Modal.js":
/*!*****************************!*\
  !*** ./components/Modal.js ***!
  \*****************************/
/*! exports provided: Modal, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Modal", function() { return Modal; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return Modal; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * Modal Cell Component
 */




/**
 * About Replacements
 * ------------------
 * Modal has the following single replacements:
 * *`title`
 * *`message`
 * And has the following enumerated
 * replacements
 * * `button`
 */

/**
 * About Named Children
 * --------------------
 * `title` (single) - A Cell containing the title
 * `message` (single) - A Cell contianing the body of the
 *     modal message
 * `buttons` (array) - An array of button cells
 */
class Modal extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);
        this.mainStyle = 'display:block;padding-right:15px;';

        // Bind component methods
        this.makeTitle = this.makeTitle.bind(this);
        this.makeMessage = this.makeMessage.bind(this);
        this.makeButtons = this.makeButtons.bind(this);
    }

    render(){
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                class: "cell modal fade show",
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Modal",
                role: "dialog",
                style: mainStyle
            }, [
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {role: "document", class: "modal-dialog"}, [
                    Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {class: "modal-content"}, [
                        Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {class: "modal-header"}, [
                            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('h5', {class: "modal-title"}, [
                                this.makeTitle()
                            ])
                        ]),
                        Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {class: "modal-body"}, [
                            this.makeMessage()
                        ]),
                        Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {class: "modal-footer"}, this.makeButtons())
                    ])
                ])
            ])
        );
    }

    makeButtons(){
        if(this.usesReplacements){
            return this.getReplacementElementsFor('button');
        } else {
            return this.renderChildrenNamed('buttons')
        }
    }

    makeMessage(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('message');
        } else {
            return this.renderChildNamed('message');
        }
    }

    makeTitle(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('title');
        } else {
            return this.renderChildNamed('title');
        }
    }
}




/***/ }),

/***/ "./components/Octicon.js":
/*!*******************************!*\
  !*** ./components/Octicon.js ***!
  \*******************************/
/*! exports provided: Octicon, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Octicon", function() { return Octicon; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return Octicon; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * Octicon Cell Component
 */




class Octicon extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
        this._getHTMLClasses = this._getHTMLClasses.bind(this);
    }

    render(){
        return(
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('span', {
                class: this._getHTMLClasses(),
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Octicon",
                "aria-hidden": true,
                style: this.props.extraData.divStyle
            })
        );
    }

    _getHTMLClasses(){
        let classes = ["cell", "octicon"];
        this.props.extraData.octiconClasses.forEach(name => {
            classes.push(name);
        });
        return classes.join(" ");
    }
}




/***/ }),

/***/ "./components/Padding.js":
/*!*******************************!*\
  !*** ./components/Padding.js ***!
  \*******************************/
/*! exports provided: Padding, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Padding", function() { return Padding; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return Padding; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * Padding Cell Component
 */




class Padding extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('span', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Padding",
                class: "px-2"
            }, [" "])
        );
    }
}




/***/ }),

/***/ "./components/Plot.js":
/*!****************************!*\
  !*** ./components/Plot.js ***!
  \****************************/
/*! exports provided: Plot, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Plot", function() { return Plot; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return Plot; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * Plot Cell Component
 */




/**
 * About Replacements
 * ------------------
 * This component contains the following
 * regular replacements:
 * * `chart-updater`
 * * `error`
 */

/**
 * About Named Children
 * --------------------
 * `chartUpdater` (single) - The Updater cell
 * `error` (single) - An error cell, if present
 */
class Plot extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.setupPlot = this.setupPlot.bind(this);
        this.makeChartUpdater = this.makeChartUpdater.bind(this);
        this.makeError = this.makeError.bind(this);
    }

    componentDidLoad() {
        this.setupPlot();
    }

    render(){
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Plot",
                class: "cell"
            }, [
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {id: `plot${this.props.id}`, style: this.props.extraData.divStyle}),
                this.makeChartUpdater(),
                this.makeError()
            ])
        );
    }

    makeChartUpdater(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('chart-updater');
        } else {
            return this.renderChildNamed('chartUpdater');
        }
    }

    makeError(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('error');
        } else {
            return this.renderChildNamed('error');
        }
    }

    setupPlot(){
        console.log("Setting up a new plotly chart.");
        // TODO These are global var defined in page.html
        // we should do something about this.
        var plotDiv = document.getElementById('plot' + this.props.id);
        Plotly.plot(
            plotDiv,
            [],
            {
                margin: {t : 30, l: 30, r: 30, b:30 },
                xaxis: {rangeslider: {visible: false}}
            },
            { scrollZoom: true, dragmode: 'pan', displaylogo: false, displayModeBar: 'hover',
                modeBarButtons: [ ['pan2d'], ['zoom2d'], ['zoomIn2d'], ['zoomOut2d'] ] }
        );
        plotDiv.on('plotly_relayout',
            function(eventdata){
                if (plotDiv.is_server_defined_move === true) {
                    return
                }
                //if we're sending a string, then its a date object, and we want to send
                // a timestamp
                if (typeof(eventdata['xaxis.range[0]']) === 'string') {
                    eventdata = Object.assign({},eventdata);
                    eventdata["xaxis.range[0]"] = Date.parse(eventdata["xaxis.range[0]"]) / 1000.0;
                    eventdata["xaxis.range[1]"] = Date.parse(eventdata["xaxis.range[1]"]) / 1000.0;
                }

                let responseData = {
                    'event':'plot_layout',
                    'target_cell': '__identity__',
                    'data': eventdata
                };
                cellSocket.sendString(JSON.stringify(responseData));
            });
    }
}




/***/ }),

/***/ "./components/Popover.js":
/*!*******************************!*\
  !*** ./components/Popover.js ***!
  \*******************************/
/*! exports provided: Popover, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Popover", function() { return Popover; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return Popover; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * Popover Cell Component
 */




/**
 * About Replacements
 * This component contains the following
 * regular replacements:
 * * `title`
 * * `detail`
 * * `contents`
 */

/**
 * About Named Children
 * --------------------
 * `content` (single) - The content of the popover
 * `detail` (single) - Detail of the popover
 * `title` (single) - The title for the popover
 */
class Popover extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeTitle = this.makeTitle.bind(this);
        this.makeContent = this.makeContent.bind(this);
        this.makeDetail = this.makeDetail.bind(this);
    }

    render(){
        return Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div',
            {
                class: "cell popover-cell",
                style: this.props.extraData.divStyle,
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Popover"
            }, [
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('a',
                    {
                        href: "#popmain_" + this.props.id,
                        "data-toggle": "popover",
                        "data-trigger": "focus",
                        "data-bind": "#pop_" + this.props.id,
                        "data-placement": "bottom",
                        role: "button",
                        class: "btn btn-xs"
                    },
                  [this.makeContent()]
                ),
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {style: "display:none"}, [
                    Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])("div", {id: "pop_" + this.props.id}, [
                        Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])("div", {class: "data-title"}, [this.makeTitle()]),
                        Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])("div", {class: "data-content"}, [
                            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])("div", {style: "width: " + this.props.width + "px"}, [
                                this.makeDetail()]
                            )
                        ])
                    ])
                ])
            ]
        );
    }

    makeContent(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('contents');
        } else {
            return this.renderChildNamed('content');
        }
    }

    makeDetail(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('detail');
        } else {
            return this.renderChildNamed('detail');
        }
    }

    makeTitle(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('title');
        } else {
            return this.renderChildNamed('title');
        }
    }
}




/***/ }),

/***/ "./components/RootCell.js":
/*!********************************!*\
  !*** ./components/RootCell.js ***!
  \********************************/
/*! exports provided: RootCell, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "RootCell", function() { return RootCell; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return RootCell; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * RootCell Cell Component
 */




/**
 * About Replacements
 * --------------------
 * This component has a one
 * regular-kind replacement:
 * * `c`
 */

/**
 * About Named Children
 * --------------------
 * `child` (single) - The child cell this container contains
 */
class RootCell extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeChild = this.makeChild.bind(this);
    }

    render(){
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "RootCell"
            }, [this.makeChild()])
        );
    }

    makeChild(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('c');
        } else {
            return this.renderChildNamed('child');
        }
    }
}




/***/ }),

/***/ "./components/Scrollable.js":
/*!**********************************!*\
  !*** ./components/Scrollable.js ***!
  \**********************************/
/*! exports provided: Scrollable, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Scrollable", function() { return Scrollable; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return Scrollable; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * Scrollable  Component
 */




/**
 * About Replacements
 * --------------------
 * This component has a one
 * regular-kind replacement:
 * * `child`
 */

/**
 * About Named Children
 * --------------------
 * `child` (single) - The cell/component this instance contains
 */
class Scrollable extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeChild = this.makeChild.bind(this);
    }

    render(){
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Scrollable"
            }, [this.makeChild()])
        );
    }

    makeChild(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('child');
        } else {
            return this.renderChildNamed('child');
        }
    }
}




/***/ }),

/***/ "./components/Sequence.js":
/*!********************************!*\
  !*** ./components/Sequence.js ***!
  \********************************/
/*! exports provided: Sequence, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Sequence", function() { return Sequence; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return Sequence; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * Sequence Cell Component
 */




/**
 * About Replacements
 * ------------------
 * Sequence has the following enumerated
 * replacement:
 * * `c`
 */

/**
 * About Named Children
 * --------------------
 * `elements` (array) - A list of Cells that are in the
 *    sequence.
 */
class Sequence extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeElements = this.makeElements.bind(this);
    }

    render(){
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                id: this.props.id,
                class: "cell",
                "data-cell-id": this.props.id,
                "data-cell-type": "Sequence",
                style: this.props.extraData.divStyle
            }, this.makeElements())
        );
    }

    makeElements(){
        if(this.usesReplacements){
            return this.getReplacementElementsFor('c');
        } else {
            return this.renderChildrenNamed('elements');
        }
    }
}




/***/ }),

/***/ "./components/Sheet.js":
/*!*****************************!*\
  !*** ./components/Sheet.js ***!
  \*****************************/
/*! exports provided: Sheet, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Sheet", function() { return Sheet; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return Sheet; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * Sheet Cell Component
 * NOTE: This is in part a wrapper
 * for handsontables.
 */




/**
 * About Replacements
 * This component has one regular
 * replacement:
 * * `error`
 */

/**
 * About Named Children
 * --------------------
 * `error` (single) - An error cell if present
 */
class Sheet extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        this.currentTable = null;

        // Bind context to methods
        this.initializeTable = this.initializeTable.bind(this);
        this.initializeHooks = this.initializeHooks.bind(this);
        this.makeError = this.makeError.bind(this);

        /**
         * WARNING: The Cell version of Sheet is still using certain
         * postscripts because we have not yet refactored the socket
         * protocol.
         * Remove this warning about it once that happens!
         */
        console.warn(`[TODO] Sheet still uses certain postsceripts in its interaction. See component constructor comment for more information`);
    }

    componentDidLoad(){
        console.log(`#componentDidLoad called for Sheet ${this.props.id}`);
        console.log(`This sheet has the following replacements:`, this.replacements);
        this.initializeTable();
        if(this.props.extraData['handlesDoubleClick']){
            this.initializeHooks();
        }
        // Request initial data?
        cellSocket.sendString(JSON.stringify({
            event: "sheet_needs_data",
            target_cell: this.props.id,
            data: 0
        }));
    }

    render(){
        console.log(`Rendering sheet ${this.props.id}`);
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Sheet",
                class: "cell"
            }, [
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                    id: `sheet${this.props.id}`,
                    style: this.props.extraData.divStyle,
                    class: "handsontable"
                }, [this.makeError()])
            ])
        );
    }

    initializeTable(){
        console.log(`#initializeTable called for Sheet ${this.props.id}`);
        let getProperty = function(index){
            return function(row){
                return row[index];
            };
        };
        let emptyRow = [];
        let dataNeededCallback = function(eventObject){
            eventObject.target_cell = this.props.id;
            cellSocket.sendString(JSON.stringify(eventObject));
        }.bind(this);
        let data = new SyntheticIntegerArray(this.props.extraData.rowCount, emptyRow, dataNeededCallback);
        let container = document.getElementById(`sheet${this.props.id}`);
        let columnNames = this.props.extraData.columnNames;
        let columns = columnNames.map((name, idx) => {
            emptyRow.push("");
            return {data: getProperty(idx)};
        });

        this.currentTable = new Handsontable(container, {
            data,
            dataSchema: function(opts){return {};},
            colHeaders: columnNames,
            columns,
            rowHeaders:true,
            rowHeaderWidth: 100,
            viewportRowRenderingOffset: 100,
            autoColumnSize: false,
            autoRowHeight: false,
            manualColumnResize: true,
            colWidths: this.props.extraData.columnWidth,
            rowHeights: 23,
            readOnly: true,
            ManualRowMove: false
        });
        handsOnTables[this.props.id] = {
            table: this.currentTable,
            lastCellClicked: {row: -100, col: -100},
            dblClicked: true
        };
    }

    initializeHooks(){
        Handsontable.hooks.add("beforeOnCellMouseDown", (event, data) => {
            let handsOnObj = handsOnTables[this.props.id];
            let lastRow = handsOnObj.lastCellClicked.row;
            let lastCol = handsOnObj.lastCellClicked.col;

            if((lastRow == data.row) && (lastCol = data.col)){
                handsOnObj.dblClicked = true;
                setTimeout(() => {
                    if(handsOnObj.dblClicked){
                        cellSocket.sendString(JSON.stringify({
                            event: 'onCellDblClick',
                            target_cell: this.props.id,
                            row: data.row,
                            col: data.col
                        }));
                    }
                    handsOnObj.lastCellClicked = {row: -100, col: -100};
                    handsOnObj.dblClicked = false;
                }, 200);
            } else {
                handsOnObj.lastCellClicked = {row: data.row, col: data.col};
                setTimeout(() => {
                    handsOnObj.lastCellClicked = {row: -100, col: -100};
                    handsOnObj.dblClicked = false;
                }, 600);
            }
        }, this.currentTable);

        Handsontable.hooks.add("beforeOnCellContextMenu", (event, data) => {
            let handsOnObj = handsOnTables[this.props.id];
            handsOnObj.dblClicked = false;
            handsOnObj.lastCellClicked = {row: -100, col: -100};
        }, this.currentTable);

        Handsontable.hooks.add("beforeContextMenuShow", (event, data) => {
            let handsOnObj = handsOnTables[this.props.id];
            handsOnObj.dblClicked = false;
            handsOnObj.lastCellClicked = {row: -100, col: -100};
        }, this.currentTable);
    }

    makeError(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('error');
        } else {
            return this.renderChildNamed('error');
        }
    }
}

/** Copied over from Cells implementation **/
const SyntheticIntegerArray = function(size, emptyRow = [], callback){
    this.length = size;
    this.cache = {};
    this.push = function(){};
    this.splice = function(){};

    this.slice = function(low, high){
        if(high === undefined){
            high = this.length;
        }

        let res = Array(high - low);
        let initLow = low;
        while(low < high){
            let out = this.cache[low];
            if(out === undefined){
                if(callback){
                    callback({
                        event: 'sheet_needs_data',
                        data: low
                    });
                }
                out = emptyRow;
            }
            res[low - initLow] = out;
            low += 1;
        }
        return res;
    };
};




/***/ }),

/***/ "./components/SingleLineTextBox.js":
/*!*****************************************!*\
  !*** ./components/SingleLineTextBox.js ***!
  \*****************************************/
/*! exports provided: SingleLineTextBox, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "SingleLineTextBox", function() { return SingleLineTextBox; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return SingleLineTextBox; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * SingleLineTextBox Cell Component
 */




class SingleLineTextBox extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
        this.changeHandler = this.changeHandler.bind(this);
    }

    render(){
        let attrs =
            {
                class: "cell",
                id: "text_" + this.props.id,
                type: "text",
                "data-cell-id": this.props.id,
                "data-cell-type": "SingleLineTextBox",
                onchange: (event) => {this.changeHandler(event.target.value);}
            };
        if (this.props.extraData.inputValue !== undefined) {
            attrs.pattern = this.props.extraData.inputValue;
        }
        return Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('input', attrs, []);
    }

    changeHandler(val) {
        cellSocket.sendString(
            JSON.stringify(
                {
                    "event": "click",
                    "target_cell": this.props.id,
                    "text": val
                }
            )
        );
    }
}




/***/ }),

/***/ "./components/Span.js":
/*!****************************!*\
  !*** ./components/Span.js ***!
  \****************************/
/*! exports provided: Span, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Span", function() { return Span; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return Span; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * Span Cell Component
 */





class Span extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('span', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Span",
                class: "cell"
            }, [this.props.extraData.text])
        );
    }
}




/***/ }),

/***/ "./components/Subscribed.js":
/*!**********************************!*\
  !*** ./components/Subscribed.js ***!
  \**********************************/
/*! exports provided: Subscribed, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Subscribed", function() { return Subscribed; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return Subscribed; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * Subscribed Cell Component
 */




/**
 * About Replacements
 * ------------------
 * This component has a single
 * regular replacement:
 * * `contents`
 */

/**
 * About Named Children
 * --------------------
 * `content` (single) - The underlying Cell that is subscribed
 */
class Subscribed extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeContent = this.makeContent.bind(this);
    }

    render(){
        return Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div',
            {
                class: "cell subscribed",
                style: this.props.extraData.divStyle,
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Subscribed"
            }, [this.makeContent()]
        );
    }

    makeContent(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('contents');
        } else {
            return this.renderChildNamed('content');
        }
    }
}




/***/ }),

/***/ "./components/SubscribedSequence.js":
/*!******************************************!*\
  !*** ./components/SubscribedSequence.js ***!
  \******************************************/
/*! exports provided: SubscribedSequence, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "SubscribedSequence", function() { return SubscribedSequence; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return SubscribedSequence; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * SubscribedSequence Cell Component
 */




/**
 * About Replacements
 * ------------------
 * This component has a single
 * enumerated replacement:
 * * `child`
 */

/**
 * About Named Replacements
 * ------------------------
 * `children` (array) - An array of Cells that are subscribed
 */
class SubscribedSequence extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);
        //this.addReplacement('contents', '_____contents__');
        //
        // Bind context to methods
        this.makeClass = this.makeClass.bind(this);
        this.makeChildren = this.makeChildren.bind(this);
        this._makeReplacementChildren = this._makeReplacementChildren.bind(this);
    }

    render(){
        return Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div',
            {
                class: this.makeClass(),
                style: this.props.extraData.divStyle,
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "SubscribedSequence"
            }, [this.makeChildren()]
        );
    }

    makeChildren(){
        if(this.usesReplacements){
            return this._makeReplacementChildren();
        } else {
            if(this.props.extraData.asColumns){
                let formattedChildren = this.renderChildrenNamed('children').map(childEl => {
                    return(
                        Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {class: "col-sm", key: childElement.id}, [
                            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('span', {}, [childEl])
                        ])
                    );
                });
                return (
                    Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {class: "row flex-nowrap", key: `${this.props.id}-spine-wrapper`}, formattedChildren)
                );
            } else {
                return (
                    Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {key: `${this.props.id}-spine-wrapper`}, this.renderChildrenNamed('children'))
                );
            }
        }
    }

    makeClass() {
        if (this.props.extraData.asColumns) {
            return "cell subscribedSequence container-fluid";
        }
        return "cell subscribedSequence";
    }

    _makeReplacementChildren(){
        if(this.props.extraData.asColumns){
            let formattedChildren = this.getReplacementElementsFor('child').map(childElement => {
                return(
                    Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {class: "col-sm", key: childElement.id}, [
                        Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('span', {}, [childElement])
                    ])
                );
            });
            return (
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {class: "row flex-nowrap", key: `${this.props.id}-spine-wrapper`}, formattedChildren)
            );
        } else {
            return (
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {key: `${this.props.id}-spine-wrapper`}, this.getReplacementElementsFor('child'))
            );
        }
    }
}




/***/ }),

/***/ "./components/Table.js":
/*!*****************************!*\
  !*** ./components/Table.js ***!
  \*****************************/
/*! exports provided: Table, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Table", function() { return Table; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return Table; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * Table Cell Component
 */





/**
 * About Replacements
 * ------------------
 * This component has 3 regular
 * replacements:
 * * `page`
 * * `left`
 * * `right`
 * This component has 2 enumerated
 * replacements:
 * * `child`
 * * `header`
 * NOTE: `child` enumerated replacements
 * are two dimensional arrays!
 */

/**
 * About Named Children
 * --------------------
 * `headers` (array) - An array of table header cells
 * `dataCells` (array-of-array) - A 2-dimensional array
 *    structures as rows by columns that contains the
 *    table data cells
 * `page` (single) - A cell that tells which page of the
 *     table we are looking at
 * `left` (single) - A cell that shows the number on the left
 * `right` (single) - A cell that show the number on the right
 */
class Table extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
        this.makeRows = this.makeRows.bind(this);
        this.makeFirstRow = this.makeFirstRow.bind(this);
        this._makeRowElements = this._makeRowElements.bind(this);
        this._theadStyle = this._theadStyle.bind(this);
        this._getRowDisplayElements = this._getRowDisplayElements.bind(this);
    }

    render(){
        return(
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('table', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Table",
                class: "cell table-hscroll table-sm table-striped"
            }, [
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('thead', {style: this._theadStyle()},[
                    this.makeFirstRow()
                ]),
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('tbody', {}, this.makeRows())
            ])
        );
    }

    _theadStyle(){
        return "border-bottom: black;border-bottom-style:solid;border-bottom-width:thin;";
    }

    makeHeaderElements(){
        if(this.usesReplacements){
            return this.getReplacementElementsFor('header').map((replacement, idx) => {
                return Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('th', {
                    style: "vertical-align:top;",
                    key: `${this.props.id}-table-header-${idx}`
                }, [replacement]);
            });
        } else {
            return this.renderChildrenNamed('headers').map((replacement, idx) => {
                return Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('th', {
                    style: "vertical-align:top;",
                    key: `${this.props.id}-table-header-${idx}`
                }, [replacement]);
            });
        }
    }

    makeRows(){
        if(this.usesReplacements){
            return this._makeRowElements(this.getReplacementElementsFor('child'));
        } else {
            return this._makeRowElements(this.renderChildrenNamed('dataCells'));
        }
    }



    _makeRowElements(elements){
        // Note: rows are the *first* dimension
        // in the 2-dimensional array returned
        // by getting the `child` replacement elements.
        return elements.map((row, rowIdx) => {
            let columns = row.map((childElement, colIdx) => {
                return (
                    Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('td', {
                        key: `${this.props.id}-td-${rowIdx}-${colIdx}`
                    }, [childElement])
                );
            });
            let indexElement = Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('td', {}, [`${rowIdx + 1}`]);
            return (
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('tr', {key: `${this.props.id}-tr-${rowIdx}`}, [indexElement, ...columns])
            );
        });
    }

    makeFirstRow(){
        let headerElements = this.makeHeaderElements();
        return(
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('tr', {}, [
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('th', {style: "vertical-align:top;"}, [
                    Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {class: "card"}, [
                        Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {class: "card-body p-1"}, [
                            ...this._getRowDisplayElements()
                        ])
                    ])
                ]),
                ...headerElements
            ])
        );
    }

    _getRowDisplayElements(){
        if(this.usesReplacements){
            return [
                this.getReplacementElementFor('left'),
                this.getReplacementElementFor('right'),
                this.getReplacementElementFor('page'),
            ];
        } else {
            return [
                this.renderChildNamed('left'),
                this.renderChildNamed('right'),
                this.renderChildNamed('page')
            ];
        }
    }
}




/***/ }),

/***/ "./components/Tabs.js":
/*!****************************!*\
  !*** ./components/Tabs.js ***!
  \****************************/
/*! exports provided: Tabs, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Tabs", function() { return Tabs; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return Tabs; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * Tabs Cell Component
 */





/**
 * About Replacements
 * ------------------
 * This component had a single
 * regular replacement:
 * * `display`
 * This component has a single
 * enumerated replacement:
 * * `header`
 */

/**
 * About Named Children
 * --------------------
 * `display` (single) - The Cell that gets displayed when
 *      the tabs are showing
 * `headers` (array) - An array of cells that serve as
 *     the tab headers
 */
class Tabs extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeHeaders = this.makeHeaders.bind(this);
        this.makeDisplay = this.makeDisplay.bind(this);
    }

    render(){
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Tabs",
                class: "container-fluid mb-3"
            }, [
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('ul', {class: "nav nav-tabs", role: "tablist"}, this.makeHeaders()),
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {class: "tab-content"}, [
                    Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {class: "tab-pane fade show active", role: "tabpanel"}, [
                        this.makeDisplay()
                    ])
                ])
            ])
        );
    }

    makeDisplay(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('display');
        } else {
            return this.renderChildNamed('display');
        }
    }

    makeHeaders(){
        if(this.usesReplacements){
            return this.getReplacementElementsFor('header');
        } else {
            return this.renderChildrenNamed('headers');
        }
    }
}





/***/ }),

/***/ "./components/Text.js":
/*!****************************!*\
  !*** ./components/Text.js ***!
  \****************************/
/*! exports provided: Text, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Text", function() { return Text; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return Text; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * Text Cell Component
 */




class Text extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return(
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                class: "cell",
                id: this.props.id,
                style: this.props.extraData.divStyle,
                "data-cell-id": this.props.id,
                "data-cell-type": "Text"
            }, [this.props.extraData.rawText])
        );
    }
}




/***/ }),

/***/ "./components/Traceback.js":
/*!*********************************!*\
  !*** ./components/Traceback.js ***!
  \*********************************/
/*! exports provided: Traceback, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Traceback", function() { return Traceback; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return Traceback; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * Traceback Cell Component
 */




/**
 * About Replacements
 * ------------------
 * This component has a single regular
 * repalcement:
 * * `child`
 */

/**
 * About Named Children
 * `traceback` (single) - The cell containing the traceback text
 */
class  Traceback extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeTraceback = this.makeTraceback.bind(this);
    }

    render(){
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Traceback",
                class: "alert alert-primary"
            }, [this.getReplacementElementFor('child')])
        );
    }

    makeTraceback(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('child');
        } else {
            return this.renderChildNamed('traceback');
        }
    }
}





/***/ }),

/***/ "./components/_NavTab.js":
/*!*******************************!*\
  !*** ./components/_NavTab.js ***!
  \*******************************/
/*! exports provided: _NavTab, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "_NavTab", function() { return _NavTab; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return _NavTab; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * _NavTab Cell Component
 * NOTE: This should probably just be
 * rolled into the Nav component somehow,
 * or included in its module as a private
 * subcomponent.
 */




/**
 * About Replacements
 * -------------------
 * This component has one regular
 * replacement:
 * * `child`
 */

/**
 * About Named Children
 * --------------------
 * `child` (single) - The cell inside of the navigation tab
 */
class _NavTab extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
        this.makeChild = this.makeChild.bind(this);
        this.clickHandler = this.clickHandler.bind(this);
    }

    render(){
        let innerClass = "nav-link";
        if(this.props.extraData.isActive){
            innerClass += " active";
        }
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('li', {
                id: this.props.id,
                class: "nav-item",
                "data-cell-id": this.props.id,
                "data-cell-type": "_NavTab"
            }, [
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('a', {
                    class: innerClass,
                    role: "tab",
                    onclick: this.clickHandler
                }, [this.makeChild()])
            ])
        );
    }

    makeChild(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('child');
        } else {
            return this.renderChildNamed('child');
        }
    }

    clickHandler(event){
        cellSocket.sendString(
            JSON.stringify(this.props.extraData.clickData, null, 4)
        );
    }
}




/***/ }),

/***/ "./components/_PlotUpdater.js":
/*!************************************!*\
  !*** ./components/_PlotUpdater.js ***!
  \************************************/
/*! exports provided: _PlotUpdater, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "_PlotUpdater", function() { return _PlotUpdater; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return _PlotUpdater; });
/* harmony import */ var _Component__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Component */ "./components/Component.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
/**
 * _PlotUpdater Cell Component
 * NOTE: Later refactorings should result in
 * this component becoming obsolete
 */




const MAX_INTERVALS = 25;

class _PlotUpdater extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        this.runUpdate = this.runUpdate.bind(this);
        this.listenForPlot = this.listenForPlot.bind(this);
    }

    componentDidLoad() {
        // If we can find a matching Plot element
        // at this point, we simply update it.
        // Otherwise we need to 'listen' for when
        // it finally comes into the DOM.
        let initialPlotDiv = document.getElementById(`plot${this.props.extraData.plotId}`);
        if(initialPlotDiv){
            this.runUpdate(initialPlotDiv);
        } else {
            this.listenForPlot();
        }
    }

    render(){
        return Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div',
            {
                class: "cell",
                id: this.props.id,
                style: "display: none",
                "data-cell-id": this.props.id,
                "data-cell-type": "_PlotUpdater"
            }, []);
    }

    /**
     * In the event that a `_PlotUpdater` has come
     * over the wire *before* its corresponding
     * Plot has come over (which appears to be
     * common), we will set an interval of 50ms
     * and check for the matching Plot in the DOM
     * MAX_INTERVALS times, only calling `runUpdate`
     * once we've found a match.
     */
    listenForPlot(){
        let numChecks = 0;
        let plotChecker = window.setInterval(() => {
            if(numChecks > MAX_INTERVALS){
                window.clearInterval(plotChecker);
                console.error(`Could not find matching Plot ${this.props.extraData.plotId} for _PlotUpdater ${this.props.id}`);
                return;
            }
            let plotDiv = document.getElementById(`plot${this.props.extraData.plotId}`);
            if(plotDiv){
                this.runUpdate(plotDiv);
                window.clearInterval(plotChecker);
            } else {
                numChecks += 1;
            }
        }, 50);
    }

    runUpdate(aDOMElement){
        console.log("Updating plotly chart.");
        // TODO These are global var defined in page.html
        // we should do something about this.
        if (this.props.extraData.exceptionOccured) {
            console.log("plot exception occured");
            Plotly.purge(aDOMElement);
        } else {
            let data = this.props.extraData.plotData.map(mapPlotlyData);
            Plotly.react(aDOMElement, data, aDOMElement.layout);
        }
    }
}




/***/ }),

/***/ "./components/util/PropertyValidator.js":
/*!**********************************************!*\
  !*** ./components/util/PropertyValidator.js ***!
  \**********************************************/
/*! exports provided: PropTypes */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "PropTypes", function() { return PropTypes; });
/**
 * Tool for Validating Component Properties
 */

const report = (message, errorMode, silentMode) => {
    if(errorMode == true && silentMode == false){
        console.error(message);
    } else if(silentMode == false){
        console.warn(message);
    }
};

const PropTypes = {
    errorMode: false,
    silentMode: false,
    oneOf: function(anArray){
        return function(componentName, propName, propValue, isRequired){
            for(let i = 0; i < anArray.length; i++){
                let typeCheckItem = anArray[i];
                if(typeof(typeCheckItem) == 'function'){
                    if(typeCheckItem(componentName, propName, propValue, isRequired, true)){
                        return true;
                    }
                } else if(typeCheckItem == propValue){
                    return true;
                }
            }
            let message = `${componentName} >> ${propName} must be of one of the following types: ${anArray}`;
            report(message, this.errorMode, this.silentMode);
            return false;
        }.bind(this);
    },

    getValidatorForType(typeStr){
        return function(componentName, propName, propValue, isRequired, inCompound = false){
            // We are 'in a compound validation' when the individual
            // PropType checkers (ie func, number, string, etc) are
            // being called within a compound type checker like oneOf.
            // In these cases we want to prevent the call to report,
            // which the compound check will handle on its own.
            if(inCompound == false){
                if(typeof(propValue) == typeStr){
                    return true;
                } else if(!isRequired && (propValue == undefined || propValue == null)){
                    return true;
                } else if(isRequired){
                    let message = `${componentName} >> ${propName} is a required prop, but as passed as ${propValue}!`;
                    report(message, this.errorMode, this.silentMode);
                    return false;
                } else {
                    let message = `${componentName} >> ${propName} must be of type ${typeStr}!`;
                    report(message, this.errorMode, this.silentMode);
                    return false;
                }
            // Otherwise this is a straightforward type check
            // based on the given type. We check as usual for the required
            // property and then the actual type match if needed.
            } else {
                if(isRequired && (propValue == undefined || propValue == null)){
                    let message = `${componentName} >> ${propName} is a required prop, but was passed as ${propValue}!`;
                    report(message, this.errorMode, this.silentMode);
                    return false;
                } else if(!isRequired && (propValue == undefined || propValue == null)){
                    return true;
                }
                return typeof(propValue) == typeStr;
            }
        };
    },

    get number(){
        return this.getValidatorForType('number').bind(this);
    },

    get boolean(){
        return this.getValidatorForType('boolean').bind(this);
    },

    get string(){
        return this.getValidatorForType('string').bind(this);
    },

    get object(){
        return this.getValidatorForType('object').bind(this);
    },

    get func(){
        return this.getValidatorForType('function').bind(this);
    },

    validate: function(componentName, props, propInfo){
        let propNames = new Set(Object.keys(props));
        propNames.delete('children');
        propNames.delete('namedChildren');
        propNames.delete('id');
        propNames.delete('extraData'); // For now
        let propsToValidate = Array.from(propNames);

        // Perform all the validations on each property
        // according to its description. We store whether
        // or not the given property was completely valid
        // and then evaluate the validity of all at the end.
        let validationResults = {};
        propsToValidate.forEach(propName => {
            let propVal = props[propName];
            let validationToCheck = propInfo[propName];
            if(validationToCheck){
                let hasValidDescription = this.validateDescription(componentName, propName, validationToCheck);
                let hasValidPropTypes = validationToCheck.type(componentName, propName, propVal, validationToCheck.required);
                if(hasValidDescription && hasValidPropTypes){
                    validationResults[propName] = true;
                } else {
                    validationResults[propName] = false;
                }
            } else {
                // If we get here, the consumer has passed in a prop
                // that is not present in the propTypes description.
                // We report to the console as needed and validate as false.
                let message = `${componentName} has a prop called "${propName}" that is not described in propTypes!`;
                report(message, this.errorMode, this.silentMode);
                validationResults[propName] = false;
            }
        });

        // If there were any that did not validate, return
        // false and report as much.
        let invalids = [];
        Object.keys(validationResults).forEach(key => {
            if(validationResults[key] == false){
                invalids.push(key);
            }
        });
        if(invalids.length > 0){
            return false;
        } else {
            return true;
        }
    },

    validateRequired: function(componentName, propName, propVal, isRequired){
        if(isRequired == true){
            if(propVal == null || propVal == undefined){
                let message = `${componentName} >> ${propName} requires a value, but ${propVal} was passed!`;
                report(message, this.errorMode, this.silentMode);
                return false;
            }
        }
        return true;
    },

    validateDescription: function(componentName, propName, prop){
        let desc = prop.description;
        if(desc == undefined || desc == "" || desc == null){
            let message = `${componentName} >> ${propName} has an empty description!`;
            report(message, this.errorMode, this.silentMode);
            return false;
        }
        return true;
    }
};




/***
number: function(componentName, propName, propValue, inCompound = false){
        if(inCompound == false){
            if(typeof(propValue) == 'number'){
                return true;
            } else {
                let message = `${componentName} >> ${propName} must be of type number!`;
                report(message, this.errorMode, this.silentMode);
                return false;
            }
        } else {
            return typeof(propValue) == 'number';
        }

    }.bind(this),

    string: function(componentName, propName, propValue, inCompound = false){
        if(inCompound == false){
            if(typeof(propValue) == 'string'){
                return true;
            } else {
                let message = `${componentName} >> ${propName} must be of type string!`;
                report(message, this.errorMode, this.silentMode);
                return false;
            }
        } else {
            return typeof(propValue) == 'string';
        }
    }.bind(this),

    boolean: function(componentName, propName, propValue, inCompound = false){
        if(inCompound == false){
            if(typeof(propValue) == 'boolean'){
                return true;
            } else {
                let message = `${componentName} >> ${propName} must be of type boolean!`;
                report(message, this.errorMode, this.silentMode);
                return false;
            }
        } else {
            return typeof(propValue) == 'boolean';
        }
    }.bind(this),

    object: function(componentName, propName, propValue, inCompound = false){
        if(inCompound == false){
            if(typeof(propValue) == 'object'){
                return true;
            } else {
                let message = `${componentName} >> ${propName} must be of type object!`;
                report(message, this.errorMode, this.silentMode);
                return false;
            }
        } else {
            return typeof(propValue) == 'object';
        }
    }.bind(this),

    func: function(componentName, propName, propValue, inCompound = false){
        if(inCompound == false){
            if(typeof(propValue) == 'function'){
                return true;
            } else {
                let message = `${componentName} >> ${propName} must be of type function!`;
                report(message, this.errorMode, this.silentMode);
                return false;
            }
        } else {
            return typeof(propValue) == 'function';
        }
    }.bind(this),

***/


/***/ }),

/***/ "./components/util/ReplacementsHandler.js":
/*!************************************************!*\
  !*** ./components/util/ReplacementsHandler.js ***!
  \************************************************/
/*! exports provided: ReplacementsHandler, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ReplacementsHandler", function() { return ReplacementsHandler; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return ReplacementsHandler; });
class ReplacementsHandler {
    constructor(replacements){
        this.replacements = replacements;
        this.regular = {};
        this.enumerated = {};

        if(replacements){
            this.processReplacements();
        }

        // Bind context to methods
        this.processReplacements = this.processReplacements.bind(this);
        this.processRegular = this.processRegular.bind(this);
        this.hasReplacement = this.hasReplacement.bind(this);
        this.getReplacementFor = this.getReplacementFor.bind(this);
        this.getReplacementsFor = this.getReplacementsFor.bind(this);
        this.mapReplacementsFor = this.mapReplacementsFor.bind(this);
    }

    processReplacements(){
        this.replacements.forEach(replacement => {
            let replacementInfo = this.constructor.readReplacementString(replacement);
            if(replacementInfo.isEnumerated){
                this.processEnumerated(replacement, replacementInfo);
            } else {
                this.processRegular(replacement, replacementInfo);
            }
        });
        // Now we update this.enumerated to have it's top level
        // values as Arrays instead of nested dicts and we sort
        // based on the extracted indices (which are at this point
        // just keys on subdicts or multidimensional dicts)
        Object.keys(this.enumerated).forEach(key => {
            let enumeratedReplacements = this.enumerated[key];
            this.enumerated[key] = this.constructor.enumeratedValToSortedArray(enumeratedReplacements);
        });
    }

    processRegular(replacementName, replacementInfo){
        let replacementKey = this.constructor.keyFromNameParts(replacementInfo.nameParts);
        this.regular[replacementKey] = replacementName;
    }

    processEnumerated(replacementName, replacementInfo){
        let replacementKey = this.constructor.keyFromNameParts(replacementInfo.nameParts);
        let currentEntry = this.enumerated[replacementKey];

        // If it's undefined, this is the first
        // of the enumerated replacements for this
        // key, ie something like ____child_0__
        if(currentEntry == undefined){
            this.enumerated[replacementKey] = {};
            currentEntry = this.enumerated[replacementKey];
        }

        // We add the enumerated indices as keys to a dict
        // and we do this recursively across dimensions as
        // needed.
        this.constructor.processDimension(replacementInfo.enumeratedValues, currentEntry, replacementName);
    }

    // Accessing and other Convenience Methods
    hasReplacement(aReplacementName){
        if(this.regular.hasOwnProperty(aReplacementName)){
            return true;
        } else if(this.enumerated.hasOwnProperty(aReplacementName)){
            return true;
        }
        return false;
    }

    getReplacementFor(aReplacementName){
        let found = this.regular[aReplacementName];
        if(found == undefined){
            return null;
        }
        return found;
    }

    getReplacementsFor(aReplacementName){
        let found = this.enumerated[aReplacementName];
        if(found == undefined){
            return null;
        }
        return found;
    }

    mapReplacementsFor(aReplacementName, mapFunction){
        if(!this.hasReplacement(aReplacementName)){
            throw new Error(`Invalid replacement name: ${aReplacementname}`);
        }
        let root = this.getReplacementsFor(aReplacementName);
        return this._recursivelyMap(root, mapFunction);
    }

    _recursivelyMap(currentItem, mapFunction){
        if(!Array.isArray(currentItem)){
            return mapFunction(currentItem);
        }
        return currentItem.map(subItem => {
            return this._recursivelyMap(subItem, mapFunction);
        });
    }

    // Static helpers
    static processDimension(remainingVals, inDict, replacementName){
        if(remainingVals.length == 1){
            inDict[remainingVals[0]] = replacementName;
            return;
        }
        let nextKey = remainingVals[0];
        let nextDict = inDict[nextKey];
        if(nextDict == undefined){
            inDict[nextKey] = {};
            nextDict = inDict[nextKey];
        }
        this.processDimension(remainingVals.slice(1), nextDict, replacementName);
    }

    static enumeratedValToSortedArray(aDict, accumulate = []){
        if(typeof aDict !== 'object'){
            return aDict;
        }
        let sortedKeys = Object.keys(aDict).sort((first, second) => {
            return (parseInt(first) - parseInt(second));
        });
        let subEntries = sortedKeys.map(key => {
            let entry = aDict[key];
            return this.enumeratedValToSortedArray(entry);
        });
        return subEntries;
    }

    static keyFromNameParts(nameParts){
        return nameParts.join("-");
    }

    static readReplacementString(replacement){
        let nameParts = [];
        let isEnumerated = false;
        let enumeratedValues = [];
        let pieces = replacement.split('_').filter(item => {
            return item != '';
        });
        pieces.forEach(piece => {
            let num = parseInt(piece);
            if(isNaN(num)){
                nameParts.push(piece);
        } else {
            isEnumerated = true;
            enumeratedValues.push(num);
        }
        });
        return {
            nameParts,
            isEnumerated,
            enumeratedValues
        };
    }
}




/***/ }),

/***/ "./main.js":
/*!*****************!*\
  !*** ./main.js ***!
  \*****************/
/*! no exports provided */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _CellHandler__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./CellHandler */ "./CellHandler.js");
/* harmony import */ var _CellSocket__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./CellSocket */ "./CellSocket.js");
/* harmony import */ var _ComponentRegistry__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ./ComponentRegistry */ "./ComponentRegistry.js");

const h = maquette.h;
//import {langTools} from 'ace/ext/language_tools';




/**
 * Globals
 **/
window.langTools = ace.require("ace/ext/language_tools");
window.aceEditors = {};
window.handsOnTables = {};

/**
 * Initial Render
 **/
const initialRender = function(){
    return h("div", {}, [
         h("div", {id: "page_root"}, [
             h("div.container-fluid", {}, [
                 h("div.card", {class: "mt-5"}, [
                     h("div.card-body", {}, ["Loading..."])
                 ])
             ])
         ]),
         h("div", {id: "holding_pen", style: "display:none"}, [])
     ]);
};

/**
 * Cell Socket and Handler
 **/
let projector = maquette.createProjector();
const cellSocket = new _CellSocket__WEBPACK_IMPORTED_MODULE_2__["CellSocket"]();
const cellHandler = new _CellHandler__WEBPACK_IMPORTED_MODULE_1__["CellHandler"](h, projector, _ComponentRegistry__WEBPACK_IMPORTED_MODULE_3__["ComponentRegistry"]);
cellSocket.onPostscripts(cellHandler.handlePostscript);
cellSocket.onMessage(cellHandler.handleMessage);
cellSocket.onClose(cellHandler.showConnectionClosed);
cellSocket.onError(err => {
    console.error("SOCKET ERROR: ", err);
});

/** For now, we bind the current socket and handler to the global window **/
window.cellSocket = cellSocket;
window.cellHandler = cellHandler;

/** Render top level component once DOM is ready **/
document.addEventListener('DOMContentLoaded', () => {
    projector.append(document.body, initialRender);
    cellSocket.connect();
});

// TESTING; REMOVE
console.log('Main module loaded');


/***/ }),

/***/ "./node_modules/maquette/dist/maquette.umd.js":
/*!****************************************************!*\
  !*** ./node_modules/maquette/dist/maquette.umd.js ***!
  \****************************************************/
/*! no static exports found */
/***/ (function(module, exports, __webpack_require__) {

(function (global, factory) {
     true ? factory(exports) :
    undefined;
}(this, function (exports) { 'use strict';

    /* tslint:disable no-http-string */
    var NAMESPACE_W3 = 'http://www.w3.org/';
    /* tslint:enable no-http-string */
    var NAMESPACE_SVG = NAMESPACE_W3 + "2000/svg";
    var NAMESPACE_XLINK = NAMESPACE_W3 + "1999/xlink";
    var emptyArray = [];
    var extend = function (base, overrides) {
        var result = {};
        Object.keys(base).forEach(function (key) {
            result[key] = base[key];
        });
        if (overrides) {
            Object.keys(overrides).forEach(function (key) {
                result[key] = overrides[key];
            });
        }
        return result;
    };
    var same = function (vnode1, vnode2) {
        if (vnode1.vnodeSelector !== vnode2.vnodeSelector) {
            return false;
        }
        if (vnode1.properties && vnode2.properties) {
            if (vnode1.properties.key !== vnode2.properties.key) {
                return false;
            }
            return vnode1.properties.bind === vnode2.properties.bind;
        }
        return !vnode1.properties && !vnode2.properties;
    };
    var checkStyleValue = function (styleValue) {
        if (typeof styleValue !== 'string') {
            throw new Error('Style values must be strings');
        }
    };
    var findIndexOfChild = function (children, sameAs, start) {
        if (sameAs.vnodeSelector !== '') {
            // Never scan for text-nodes
            for (var i = start; i < children.length; i++) {
                if (same(children[i], sameAs)) {
                    return i;
                }
            }
        }
        return -1;
    };
    var checkDistinguishable = function (childNodes, indexToCheck, parentVNode, operation) {
        var childNode = childNodes[indexToCheck];
        if (childNode.vnodeSelector === '') {
            return; // Text nodes need not be distinguishable
        }
        var properties = childNode.properties;
        var key = properties ? (properties.key === undefined ? properties.bind : properties.key) : undefined;
        if (!key) { // A key is just assumed to be unique
            for (var i = 0; i < childNodes.length; i++) {
                if (i !== indexToCheck) {
                    var node = childNodes[i];
                    if (same(node, childNode)) {
                        throw new Error(parentVNode.vnodeSelector + " had a " + childNode.vnodeSelector + " child " + (operation === 'added' ? operation : 'removed') + ", but there is now more than one. You must add unique key properties to make them distinguishable.");
                    }
                }
            }
        }
    };
    var nodeAdded = function (vNode) {
        if (vNode.properties) {
            var enterAnimation = vNode.properties.enterAnimation;
            if (enterAnimation) {
                enterAnimation(vNode.domNode, vNode.properties);
            }
        }
    };
    var removedNodes = [];
    var requestedIdleCallback = false;
    var visitRemovedNode = function (node) {
        (node.children || []).forEach(visitRemovedNode);
        if (node.properties && node.properties.afterRemoved) {
            node.properties.afterRemoved.apply(node.properties.bind || node.properties, [node.domNode]);
        }
    };
    var processPendingNodeRemovals = function () {
        requestedIdleCallback = false;
        removedNodes.forEach(visitRemovedNode);
        removedNodes.length = 0;
    };
    var scheduleNodeRemoval = function (vNode) {
        removedNodes.push(vNode);
        if (!requestedIdleCallback) {
            requestedIdleCallback = true;
            if (typeof window !== 'undefined' && 'requestIdleCallback' in window) {
                window.requestIdleCallback(processPendingNodeRemovals, { timeout: 16 });
            }
            else {
                setTimeout(processPendingNodeRemovals, 16);
            }
        }
    };
    var nodeToRemove = function (vNode) {
        var domNode = vNode.domNode;
        if (vNode.properties) {
            var exitAnimation = vNode.properties.exitAnimation;
            if (exitAnimation) {
                domNode.style.pointerEvents = 'none';
                var removeDomNode = function () {
                    if (domNode.parentNode) {
                        domNode.parentNode.removeChild(domNode);
                        scheduleNodeRemoval(vNode);
                    }
                };
                exitAnimation(domNode, removeDomNode, vNode.properties);
                return;
            }
        }
        if (domNode.parentNode) {
            domNode.parentNode.removeChild(domNode);
            scheduleNodeRemoval(vNode);
        }
    };
    var setProperties = function (domNode, properties, projectionOptions) {
        if (!properties) {
            return;
        }
        var eventHandlerInterceptor = projectionOptions.eventHandlerInterceptor;
        var propNames = Object.keys(properties);
        var propCount = propNames.length;
        var _loop_1 = function (i) {
            var propName = propNames[i];
            var propValue = properties[propName];
            if (propName === 'className') {
                throw new Error('Property "className" is not supported, use "class".');
            }
            else if (propName === 'class') {
                toggleClasses(domNode, propValue, true);
            }
            else if (propName === 'classes') {
                // object with string keys and boolean values
                var classNames = Object.keys(propValue);
                var classNameCount = classNames.length;
                for (var j = 0; j < classNameCount; j++) {
                    var className = classNames[j];
                    if (propValue[className]) {
                        domNode.classList.add(className);
                    }
                }
            }
            else if (propName === 'styles') {
                // object with string keys and string (!) values
                var styleNames = Object.keys(propValue);
                var styleCount = styleNames.length;
                for (var j = 0; j < styleCount; j++) {
                    var styleName = styleNames[j];
                    var styleValue = propValue[styleName];
                    if (styleValue) {
                        checkStyleValue(styleValue);
                        projectionOptions.styleApplyer(domNode, styleName, styleValue);
                    }
                }
            }
            else if (propName !== 'key' && propValue !== null && propValue !== undefined) {
                var type = typeof propValue;
                if (type === 'function') {
                    if (propName.lastIndexOf('on', 0) === 0) { // lastIndexOf(,0)===0 -> startsWith
                        if (eventHandlerInterceptor) {
                            propValue = eventHandlerInterceptor(propName, propValue, domNode, properties); // intercept eventhandlers
                        }
                        if (propName === 'oninput') {
                            /* tslint:disable no-this-keyword no-invalid-this only-arrow-functions no-void-expression */
                            (function () {
                                // record the evt.target.value, because IE and Edge sometimes do a requestAnimationFrame between changing value and running oninput
                                var oldPropValue = propValue;
                                propValue = function (evt) {
                                    oldPropValue.apply(this, [evt]);
                                    evt.target['oninput-value'] = evt.target.value; // may be HTMLTextAreaElement as well
                                };
                            }());
                            /* tslint:enable */
                        }
                        domNode[propName] = propValue;
                    }
                }
                else if (projectionOptions.namespace === NAMESPACE_SVG) {
                    if (propName === 'href') {
                        domNode.setAttributeNS(NAMESPACE_XLINK, propName, propValue);
                    }
                    else {
                        // all SVG attributes are read-only in DOM, so...
                        domNode.setAttribute(propName, propValue);
                    }
                }
                else if (type === 'string' && propName !== 'value' && propName !== 'innerHTML') {
                    domNode.setAttribute(propName, propValue);
                }
                else {
                    domNode[propName] = propValue;
                }
            }
        };
        for (var i = 0; i < propCount; i++) {
            _loop_1(i);
        }
    };
    var addChildren = function (domNode, children, projectionOptions) {
        if (!children) {
            return;
        }
        for (var _i = 0, children_1 = children; _i < children_1.length; _i++) {
            var child = children_1[_i];
            createDom(child, domNode, undefined, projectionOptions);
        }
    };
    var initPropertiesAndChildren = function (domNode, vnode, projectionOptions) {
        addChildren(domNode, vnode.children, projectionOptions); // children before properties, needed for value property of <select>.
        if (vnode.text) {
            domNode.textContent = vnode.text;
        }
        setProperties(domNode, vnode.properties, projectionOptions);
        if (vnode.properties && vnode.properties.afterCreate) {
            vnode.properties.afterCreate.apply(vnode.properties.bind || vnode.properties, [domNode, projectionOptions, vnode.vnodeSelector, vnode.properties, vnode.children]);
        }
    };
    var createDom = function (vnode, parentNode, insertBefore, projectionOptions) {
        var domNode;
        var start = 0;
        var vnodeSelector = vnode.vnodeSelector;
        var doc = parentNode.ownerDocument;
        if (vnodeSelector === '') {
            domNode = vnode.domNode = doc.createTextNode(vnode.text);
            if (insertBefore !== undefined) {
                parentNode.insertBefore(domNode, insertBefore);
            }
            else {
                parentNode.appendChild(domNode);
            }
        }
        else {
            for (var i = 0; i <= vnodeSelector.length; ++i) {
                var c = vnodeSelector.charAt(i);
                if (i === vnodeSelector.length || c === '.' || c === '#') {
                    var type = vnodeSelector.charAt(start - 1);
                    var found = vnodeSelector.slice(start, i);
                    if (type === '.') {
                        domNode.classList.add(found);
                    }
                    else if (type === '#') {
                        domNode.id = found;
                    }
                    else {
                        if (found === 'svg') {
                            projectionOptions = extend(projectionOptions, { namespace: NAMESPACE_SVG });
                        }
                        if (projectionOptions.namespace !== undefined) {
                            domNode = vnode.domNode = doc.createElementNS(projectionOptions.namespace, found);
                        }
                        else {
                            domNode = vnode.domNode = (vnode.domNode || doc.createElement(found));
                            if (found === 'input' && vnode.properties && vnode.properties.type !== undefined) {
                                // IE8 and older don't support setting input type after the DOM Node has been added to the document
                                domNode.setAttribute('type', vnode.properties.type);
                            }
                        }
                        if (insertBefore !== undefined) {
                            parentNode.insertBefore(domNode, insertBefore);
                        }
                        else if (domNode.parentNode !== parentNode) {
                            parentNode.appendChild(domNode);
                        }
                    }
                    start = i + 1;
                }
            }
            initPropertiesAndChildren(domNode, vnode, projectionOptions);
        }
    };
    var updateDom;
    /**
     * Adds or removes classes from an Element
     * @param domNode the element
     * @param classes a string separated list of classes
     * @param on true means add classes, false means remove
     */
    var toggleClasses = function (domNode, classes, on) {
        if (!classes) {
            return;
        }
        classes.split(' ').forEach(function (classToToggle) {
            if (classToToggle) {
                domNode.classList.toggle(classToToggle, on);
            }
        });
    };
    var updateProperties = function (domNode, previousProperties, properties, projectionOptions) {
        if (!properties) {
            return;
        }
        var propertiesUpdated = false;
        var propNames = Object.keys(properties);
        var propCount = propNames.length;
        for (var i = 0; i < propCount; i++) {
            var propName = propNames[i];
            // assuming that properties will be nullified instead of missing is by design
            var propValue = properties[propName];
            var previousValue = previousProperties[propName];
            if (propName === 'class') {
                if (previousValue !== propValue) {
                    toggleClasses(domNode, previousValue, false);
                    toggleClasses(domNode, propValue, true);
                }
            }
            else if (propName === 'classes') {
                var classList = domNode.classList;
                var classNames = Object.keys(propValue);
                var classNameCount = classNames.length;
                for (var j = 0; j < classNameCount; j++) {
                    var className = classNames[j];
                    var on = !!propValue[className];
                    var previousOn = !!previousValue[className];
                    if (on === previousOn) {
                        continue;
                    }
                    propertiesUpdated = true;
                    if (on) {
                        classList.add(className);
                    }
                    else {
                        classList.remove(className);
                    }
                }
            }
            else if (propName === 'styles') {
                var styleNames = Object.keys(propValue);
                var styleCount = styleNames.length;
                for (var j = 0; j < styleCount; j++) {
                    var styleName = styleNames[j];
                    var newStyleValue = propValue[styleName];
                    var oldStyleValue = previousValue[styleName];
                    if (newStyleValue === oldStyleValue) {
                        continue;
                    }
                    propertiesUpdated = true;
                    if (newStyleValue) {
                        checkStyleValue(newStyleValue);
                        projectionOptions.styleApplyer(domNode, styleName, newStyleValue);
                    }
                    else {
                        projectionOptions.styleApplyer(domNode, styleName, '');
                    }
                }
            }
            else {
                if (!propValue && typeof previousValue === 'string') {
                    propValue = '';
                }
                if (propName === 'value') { // value can be manipulated by the user directly and using event.preventDefault() is not an option
                    var domValue = domNode[propName];
                    if (domValue !== propValue // The 'value' in the DOM tree !== newValue
                        && (domNode['oninput-value']
                            ? domValue === domNode['oninput-value'] // If the last reported value to 'oninput' does not match domValue, do nothing and wait for oninput
                            : propValue !== previousValue // Only update the value if the vdom changed
                        )) {
                        // The edge cases are described in the tests
                        domNode[propName] = propValue; // Reset the value, even if the virtual DOM did not change
                        domNode['oninput-value'] = undefined;
                    } // else do not update the domNode, otherwise the cursor position would be changed
                    if (propValue !== previousValue) {
                        propertiesUpdated = true;
                    }
                }
                else if (propValue !== previousValue) {
                    var type = typeof propValue;
                    if (type !== 'function' || !projectionOptions.eventHandlerInterceptor) { // Function updates are expected to be handled by the EventHandlerInterceptor
                        if (projectionOptions.namespace === NAMESPACE_SVG) {
                            if (propName === 'href') {
                                domNode.setAttributeNS(NAMESPACE_XLINK, propName, propValue);
                            }
                            else {
                                // all SVG attributes are read-only in DOM, so...
                                domNode.setAttribute(propName, propValue);
                            }
                        }
                        else if (type === 'string' && propName !== 'innerHTML') {
                            if (propName === 'role' && propValue === '') {
                                domNode.removeAttribute(propName);
                            }
                            else {
                                domNode.setAttribute(propName, propValue);
                            }
                        }
                        else if (domNode[propName] !== propValue) { // Comparison is here for side-effects in Edge with scrollLeft and scrollTop
                            domNode[propName] = propValue;
                        }
                        propertiesUpdated = true;
                    }
                }
            }
        }
        return propertiesUpdated;
    };
    var updateChildren = function (vnode, domNode, oldChildren, newChildren, projectionOptions) {
        if (oldChildren === newChildren) {
            return false;
        }
        oldChildren = oldChildren || emptyArray;
        newChildren = newChildren || emptyArray;
        var oldChildrenLength = oldChildren.length;
        var newChildrenLength = newChildren.length;
        var oldIndex = 0;
        var newIndex = 0;
        var i;
        var textUpdated = false;
        while (newIndex < newChildrenLength) {
            var oldChild = (oldIndex < oldChildrenLength) ? oldChildren[oldIndex] : undefined;
            var newChild = newChildren[newIndex];
            if (oldChild !== undefined && same(oldChild, newChild)) {
                textUpdated = updateDom(oldChild, newChild, projectionOptions) || textUpdated;
                oldIndex++;
            }
            else {
                var findOldIndex = findIndexOfChild(oldChildren, newChild, oldIndex + 1);
                if (findOldIndex >= 0) {
                    // Remove preceding missing children
                    for (i = oldIndex; i < findOldIndex; i++) {
                        nodeToRemove(oldChildren[i]);
                        checkDistinguishable(oldChildren, i, vnode, 'removed');
                    }
                    textUpdated = updateDom(oldChildren[findOldIndex], newChild, projectionOptions) || textUpdated;
                    oldIndex = findOldIndex + 1;
                }
                else {
                    // New child
                    createDom(newChild, domNode, (oldIndex < oldChildrenLength) ? oldChildren[oldIndex].domNode : undefined, projectionOptions);
                    nodeAdded(newChild);
                    checkDistinguishable(newChildren, newIndex, vnode, 'added');
                }
            }
            newIndex++;
        }
        if (oldChildrenLength > oldIndex) {
            // Remove child fragments
            for (i = oldIndex; i < oldChildrenLength; i++) {
                nodeToRemove(oldChildren[i]);
                checkDistinguishable(oldChildren, i, vnode, 'removed');
            }
        }
        return textUpdated;
    };
    updateDom = function (previous, vnode, projectionOptions) {
        var domNode = previous.domNode;
        var textUpdated = false;
        if (previous === vnode) {
            return false; // By contract, VNode objects may not be modified anymore after passing them to maquette
        }
        var updated = false;
        if (vnode.vnodeSelector === '') {
            if (vnode.text !== previous.text) {
                var newTextNode = domNode.ownerDocument.createTextNode(vnode.text);
                domNode.parentNode.replaceChild(newTextNode, domNode);
                vnode.domNode = newTextNode;
                textUpdated = true;
                return textUpdated;
            }
            vnode.domNode = domNode;
        }
        else {
            if (vnode.vnodeSelector.lastIndexOf('svg', 0) === 0) { // lastIndexOf(needle,0)===0 means StartsWith
                projectionOptions = extend(projectionOptions, { namespace: NAMESPACE_SVG });
            }
            if (previous.text !== vnode.text) {
                updated = true;
                if (vnode.text === undefined) {
                    domNode.removeChild(domNode.firstChild); // the only textnode presumably
                }
                else {
                    domNode.textContent = vnode.text;
                }
            }
            vnode.domNode = domNode;
            updated = updateChildren(vnode, domNode, previous.children, vnode.children, projectionOptions) || updated;
            updated = updateProperties(domNode, previous.properties, vnode.properties, projectionOptions) || updated;
            if (vnode.properties && vnode.properties.afterUpdate) {
                vnode.properties.afterUpdate.apply(vnode.properties.bind || vnode.properties, [domNode, projectionOptions, vnode.vnodeSelector, vnode.properties, vnode.children]);
            }
        }
        if (updated && vnode.properties && vnode.properties.updateAnimation) {
            vnode.properties.updateAnimation(domNode, vnode.properties, previous.properties);
        }
        return textUpdated;
    };
    var createProjection = function (vnode, projectionOptions) {
        return {
            getLastRender: function () { return vnode; },
            update: function (updatedVnode) {
                if (vnode.vnodeSelector !== updatedVnode.vnodeSelector) {
                    throw new Error('The selector for the root VNode may not be changed. (consider using dom.merge and add one extra level to the virtual DOM)');
                }
                var previousVNode = vnode;
                vnode = updatedVnode;
                updateDom(previousVNode, updatedVnode, projectionOptions);
            },
            domNode: vnode.domNode
        };
    };

    var DEFAULT_PROJECTION_OPTIONS = {
        namespace: undefined,
        performanceLogger: function () { return undefined; },
        eventHandlerInterceptor: undefined,
        styleApplyer: function (domNode, styleName, value) {
            // Provides a hook to add vendor prefixes for browsers that still need it.
            domNode.style[styleName] = value;
        }
    };
    var applyDefaultProjectionOptions = function (projectorOptions) {
        return extend(DEFAULT_PROJECTION_OPTIONS, projectorOptions);
    };
    var dom = {
        /**
         * Creates a real DOM tree from `vnode`. The [[Projection]] object returned will contain the resulting DOM Node in
         * its [[Projection.domNode|domNode]] property.
         * This is a low-level method. Users will typically use a [[Projector]] instead.
         * @param vnode - The root of the virtual DOM tree that was created using the [[h]] function. NOTE: [[VNode]]
         * objects may only be rendered once.
         * @param projectionOptions - Options to be used to create and update the projection.
         * @returns The [[Projection]] which also contains the DOM Node that was created.
         */
        create: function (vnode, projectionOptions) {
            projectionOptions = applyDefaultProjectionOptions(projectionOptions);
            createDom(vnode, document.createElement('div'), undefined, projectionOptions);
            return createProjection(vnode, projectionOptions);
        },
        /**
         * Appends a new child node to the DOM which is generated from a [[VNode]].
         * This is a low-level method. Users will typically use a [[Projector]] instead.
         * @param parentNode - The parent node for the new child node.
         * @param vnode - The root of the virtual DOM tree that was created using the [[h]] function. NOTE: [[VNode]]
         * objects may only be rendered once.
         * @param projectionOptions - Options to be used to create and update the [[Projection]].
         * @returns The [[Projection]] that was created.
         */
        append: function (parentNode, vnode, projectionOptions) {
            projectionOptions = applyDefaultProjectionOptions(projectionOptions);
            createDom(vnode, parentNode, undefined, projectionOptions);
            return createProjection(vnode, projectionOptions);
        },
        /**
         * Inserts a new DOM node which is generated from a [[VNode]].
         * This is a low-level method. Users wil typically use a [[Projector]] instead.
         * @param beforeNode - The node that the DOM Node is inserted before.
         * @param vnode - The root of the virtual DOM tree that was created using the [[h]] function.
         * NOTE: [[VNode]] objects may only be rendered once.
         * @param projectionOptions - Options to be used to create and update the projection, see [[createProjector]].
         * @returns The [[Projection]] that was created.
         */
        insertBefore: function (beforeNode, vnode, projectionOptions) {
            projectionOptions = applyDefaultProjectionOptions(projectionOptions);
            createDom(vnode, beforeNode.parentNode, beforeNode, projectionOptions);
            return createProjection(vnode, projectionOptions);
        },
        /**
         * Merges a new DOM node which is generated from a [[VNode]] with an existing DOM Node.
         * This means that the virtual DOM and the real DOM will have one overlapping element.
         * Therefore the selector for the root [[VNode]] will be ignored, but its properties and children will be applied to the Element provided.
         * This is a low-level method. Users wil typically use a [[Projector]] instead.
         * @param element - The existing element to adopt as the root of the new virtual DOM. Existing attributes and child nodes are preserved.
         * @param vnode - The root of the virtual DOM tree that was created using the [[h]] function. NOTE: [[VNode]] objects
         * may only be rendered once.
         * @param projectionOptions - Options to be used to create and update the projection, see [[createProjector]].
         * @returns The [[Projection]] that was created.
         */
        merge: function (element, vnode, projectionOptions) {
            projectionOptions = applyDefaultProjectionOptions(projectionOptions);
            vnode.domNode = element;
            initPropertiesAndChildren(element, vnode, projectionOptions);
            return createProjection(vnode, projectionOptions);
        },
        /**
         * Replaces an existing DOM node with a node generated from a [[VNode]].
         * This is a low-level method. Users will typically use a [[Projector]] instead.
         * @param element - The node for the [[VNode]] to replace.
         * @param vnode - The root of the virtual DOM tree that was created using the [[h]] function. NOTE: [[VNode]]
         * objects may only be rendered once.
         * @param projectionOptions - Options to be used to create and update the [[Projection]].
         * @returns The [[Projection]] that was created.
         */
        replace: function (element, vnode, projectionOptions) {
            projectionOptions = applyDefaultProjectionOptions(projectionOptions);
            createDom(vnode, element.parentNode, element, projectionOptions);
            element.parentNode.removeChild(element);
            return createProjection(vnode, projectionOptions);
        }
    };

    /* tslint:disable function-name */
    var toTextVNode = function (data) {
        return {
            vnodeSelector: '',
            properties: undefined,
            children: undefined,
            text: data.toString(),
            domNode: null
        };
    };
    var appendChildren = function (parentSelector, insertions, main) {
        for (var i = 0, length_1 = insertions.length; i < length_1; i++) {
            var item = insertions[i];
            if (Array.isArray(item)) {
                appendChildren(parentSelector, item, main);
            }
            else {
                if (item !== null && item !== undefined && item !== false) {
                    if (typeof item === 'string') {
                        item = toTextVNode(item);
                    }
                    main.push(item);
                }
            }
        }
    };
    function h(selector, properties, children) {
        if (Array.isArray(properties)) {
            children = properties;
            properties = undefined;
        }
        else if ((properties && (typeof properties === 'string' || properties.hasOwnProperty('vnodeSelector'))) ||
            (children && (typeof children === 'string' || children.hasOwnProperty('vnodeSelector')))) {
            throw new Error('h called with invalid arguments');
        }
        var text;
        var flattenedChildren;
        // Recognize a common special case where there is only a single text node
        if (children && children.length === 1 && typeof children[0] === 'string') {
            text = children[0];
        }
        else if (children) {
            flattenedChildren = [];
            appendChildren(selector, children, flattenedChildren);
            if (flattenedChildren.length === 0) {
                flattenedChildren = undefined;
            }
        }
        return {
            vnodeSelector: selector,
            properties: properties,
            children: flattenedChildren,
            text: (text === '') ? undefined : text,
            domNode: null
        };
    }

    var createParentNodePath = function (node, rootNode) {
        var parentNodePath = [];
        while (node !== rootNode) {
            parentNodePath.push(node);
            node = node.parentNode;
        }
        return parentNodePath;
    };
    var find;
    if (Array.prototype.find) {
        find = function (items, predicate) { return items.find(predicate); };
    }
    else {
        find = function (items, predicate) { return items.filter(predicate)[0]; };
    }
    var findVNodeByParentNodePath = function (vnode, parentNodePath) {
        var result = vnode;
        parentNodePath.forEach(function (node) {
            result = (result && result.children) ? find(result.children, function (child) { return child.domNode === node; }) : undefined;
        });
        return result;
    };
    var createEventHandlerInterceptor = function (projector, getProjection, performanceLogger) {
        var modifiedEventHandler = function (evt) {
            performanceLogger('domEvent', evt);
            var projection = getProjection();
            var parentNodePath = createParentNodePath(evt.currentTarget, projection.domNode);
            parentNodePath.reverse();
            var matchingVNode = findVNodeByParentNodePath(projection.getLastRender(), parentNodePath);
            projector.scheduleRender();
            var result;
            if (matchingVNode) {
                /* tslint:disable no-invalid-this */
                result = matchingVNode.properties["on" + evt.type].apply(matchingVNode.properties.bind || this, arguments);
                /* tslint:enable no-invalid-this */
            }
            performanceLogger('domEventProcessed', evt);
            return result;
        };
        return function (propertyName, eventHandler, domNode, properties) { return modifiedEventHandler; };
    };
    /**
     * Creates a [[Projector]] instance using the provided projectionOptions.
     *
     * For more information, see [[Projector]].
     *
     * @param projectorOptions   Options that influence how the DOM is rendered and updated.
     */
    var createProjector = function (projectorOptions) {
        var projector;
        var projectionOptions = applyDefaultProjectionOptions(projectorOptions);
        var performanceLogger = projectionOptions.performanceLogger;
        var renderCompleted = true;
        var scheduled;
        var stopped = false;
        var projections = [];
        var renderFunctions = []; // matches the projections array
        var addProjection = function (
        /* one of: dom.append, dom.insertBefore, dom.replace, dom.merge */
        domFunction, 
        /* the parameter of the domFunction */
        node, renderFunction) {
            var projection;
            var getProjection = function () { return projection; };
            projectionOptions.eventHandlerInterceptor = createEventHandlerInterceptor(projector, getProjection, performanceLogger);
            projection = domFunction(node, renderFunction(), projectionOptions);
            projections.push(projection);
            renderFunctions.push(renderFunction);
        };
        var doRender = function () {
            scheduled = undefined;
            if (!renderCompleted) {
                return; // The last render threw an error, it should have been logged in the browser console.
            }
            renderCompleted = false;
            performanceLogger('renderStart', undefined);
            for (var i = 0; i < projections.length; i++) {
                var updatedVnode = renderFunctions[i]();
                performanceLogger('rendered', undefined);
                projections[i].update(updatedVnode);
                performanceLogger('patched', undefined);
            }
            performanceLogger('renderDone', undefined);
            renderCompleted = true;
        };
        projector = {
            renderNow: doRender,
            scheduleRender: function () {
                if (!scheduled && !stopped) {
                    scheduled = requestAnimationFrame(doRender);
                }
            },
            stop: function () {
                if (scheduled) {
                    cancelAnimationFrame(scheduled);
                    scheduled = undefined;
                }
                stopped = true;
            },
            resume: function () {
                stopped = false;
                renderCompleted = true;
                projector.scheduleRender();
            },
            append: function (parentNode, renderFunction) {
                addProjection(dom.append, parentNode, renderFunction);
            },
            insertBefore: function (beforeNode, renderFunction) {
                addProjection(dom.insertBefore, beforeNode, renderFunction);
            },
            merge: function (domNode, renderFunction) {
                addProjection(dom.merge, domNode, renderFunction);
            },
            replace: function (domNode, renderFunction) {
                addProjection(dom.replace, domNode, renderFunction);
            },
            detach: function (renderFunction) {
                for (var i = 0; i < renderFunctions.length; i++) {
                    if (renderFunctions[i] === renderFunction) {
                        renderFunctions.splice(i, 1);
                        return projections.splice(i, 1)[0];
                    }
                }
                throw new Error('renderFunction was not found');
            }
        };
        return projector;
    };

    /**
     * Creates a [[CalculationCache]] object, useful for caching [[VNode]] trees.
     * In practice, caching of [[VNode]] trees is not needed, because achieving 60 frames per second is almost never a problem.
     * For more information, see [[CalculationCache]].
     *
     * @param <Result> The type of the value that is cached.
     */
    var createCache = function () {
        var cachedInputs;
        var cachedOutcome;
        return {
            invalidate: function () {
                cachedOutcome = undefined;
                cachedInputs = undefined;
            },
            result: function (inputs, calculation) {
                if (cachedInputs) {
                    for (var i = 0; i < inputs.length; i++) {
                        if (cachedInputs[i] !== inputs[i]) {
                            cachedOutcome = undefined;
                        }
                    }
                }
                if (!cachedOutcome) {
                    cachedOutcome = calculation();
                    cachedInputs = inputs;
                }
                return cachedOutcome;
            }
        };
    };

    /**
     * Creates a {@link Mapping} instance that keeps an array of result objects synchronized with an array of source objects.
     * See {@link http://maquettejs.org/docs/arrays.html|Working with arrays}.
     *
     * @param <Source>       The type of source items. A database-record for instance.
     * @param <Target>       The type of target items. A [[MaquetteComponent]] for instance.
     * @param getSourceKey   `function(source)` that must return a key to identify each source object. The result must either be a string or a number.
     * @param createResult   `function(source, index)` that must create a new result object from a given source. This function is identical
     *                       to the `callback` argument in `Array.map(callback)`.
     * @param updateResult   `function(source, target, index)` that updates a result to an updated source.
     */
    var createMapping = function (getSourceKey, createResult, updateResult) {
        var keys = [];
        var results = [];
        return {
            results: results,
            map: function (newSources) {
                var newKeys = newSources.map(getSourceKey);
                var oldTargets = results.slice();
                var oldIndex = 0;
                for (var i = 0; i < newSources.length; i++) {
                    var source = newSources[i];
                    var sourceKey = newKeys[i];
                    if (sourceKey === keys[oldIndex]) {
                        results[i] = oldTargets[oldIndex];
                        updateResult(source, oldTargets[oldIndex], i);
                        oldIndex++;
                    }
                    else {
                        var found = false;
                        for (var j = 1; j < keys.length + 1; j++) {
                            var searchIndex = (oldIndex + j) % keys.length;
                            if (keys[searchIndex] === sourceKey) {
                                results[i] = oldTargets[searchIndex];
                                updateResult(newSources[i], oldTargets[searchIndex], i);
                                oldIndex = searchIndex + 1;
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            results[i] = createResult(source, i);
                        }
                    }
                }
                results.length = newSources.length;
                keys = newKeys;
            }
        };
    };

    exports.createCache = createCache;
    exports.createMapping = createMapping;
    exports.createProjector = createProjector;
    exports.dom = dom;
    exports.h = h;

    Object.defineProperty(exports, '__esModule', { value: true });

}));


/***/ })

/******/ });
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly8vd2VicGFjay9ib290c3RyYXAiLCJ3ZWJwYWNrOi8vLy4vQ2VsbEhhbmRsZXIuanMiLCJ3ZWJwYWNrOi8vLy4vQ2VsbFNvY2tldC5qcyIsIndlYnBhY2s6Ly8vLi9Db21wb25lbnRSZWdpc3RyeS5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0FzeW5jRHJvcGRvd24uanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9CYWRnZS5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0J1dHRvbi5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0J1dHRvbkdyb3VwLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvQ2FyZC5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0NhcmRUaXRsZS5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0NpcmNsZUxvYWRlci5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0NsaWNrYWJsZS5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0NvZGUuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9Db2RlRWRpdG9yLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvQ29sbGFwc2libGVQYW5lbC5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0NvbHVtbnMuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9Db21wb25lbnQuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9Db250YWluZXIuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9Db250ZXh0dWFsRGlzcGxheS5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0Ryb3Bkb3duLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvRXhwYW5kcy5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0dyaWQuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9IZWFkZXJCYXIuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9MYXJnZVBlbmRpbmdEb3dubG9hZERpc3BsYXkuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9Mb2FkQ29udGVudHNGcm9tVXJsLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvTWFpbi5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL01vZGFsLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvT2N0aWNvbi5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL1BhZGRpbmcuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9QbG90LmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvUG9wb3Zlci5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL1Jvb3RDZWxsLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvU2Nyb2xsYWJsZS5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL1NlcXVlbmNlLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvU2hlZXQuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9TaW5nbGVMaW5lVGV4dEJveC5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL1NwYW4uanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9TdWJzY3JpYmVkLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvU3Vic2NyaWJlZFNlcXVlbmNlLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvVGFibGUuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9UYWJzLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvVGV4dC5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL1RyYWNlYmFjay5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL19OYXZUYWIuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9fUGxvdFVwZGF0ZXIuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy91dGlsL1Byb3BlcnR5VmFsaWRhdG9yLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvdXRpbC9SZXBsYWNlbWVudHNIYW5kbGVyLmpzIiwid2VicGFjazovLy8uL21haW4uanMiLCJ3ZWJwYWNrOi8vLy4vbm9kZV9tb2R1bGVzL21hcXVldHRlL2Rpc3QvbWFxdWV0dGUudW1kLmpzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiI7QUFBQTtBQUNBOztBQUVBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTs7QUFFQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTs7O0FBR0E7QUFDQTs7QUFFQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBLGtEQUEwQyxnQ0FBZ0M7QUFDMUU7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQSxnRUFBd0Qsa0JBQWtCO0FBQzFFO0FBQ0EseURBQWlELGNBQWM7QUFDL0Q7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGlEQUF5QyxpQ0FBaUM7QUFDMUUsd0hBQWdILG1CQUFtQixFQUFFO0FBQ3JJO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0EsbUNBQTJCLDBCQUEwQixFQUFFO0FBQ3ZELHlDQUFpQyxlQUFlO0FBQ2hEO0FBQ0E7QUFDQTs7QUFFQTtBQUNBLDhEQUFzRCwrREFBK0Q7O0FBRXJIO0FBQ0E7OztBQUdBO0FBQ0E7Ozs7Ozs7Ozs7Ozs7QUNsRkE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDMkI7O0FBRTNCO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7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QkFBNkIsY0FBYyxNQUFNLFNBQVM7QUFDMUQ7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUlFOzs7QUFHRjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsYUFBYTtBQUNiLGlDQUFpQyxjQUFjLE1BQU0sU0FBUztBQUM5RDtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTs7QUFFQSxLQUFLOztBQUVMO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsYUFBYTtBQUNiLGlDQUFpQyxjQUFjLE1BQU0sU0FBUztBQUM5RDtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBLEtBQUs7O0FBRUw7QUFDQTtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2IsaUNBQWlDLGNBQWMsTUFBTSxTQUFTO0FBQzlEO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBO0FBQ0EsS0FBSzs7QUFFTDtBQUNBO0FBQ0E7QUFDQTtBQUNBLGFBQWE7QUFDYixpQ0FBaUMsY0FBYyxNQUFNLFNBQVM7QUFDOUQ7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQSxLQUFLOztBQUVMO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsYUFBYTtBQUNiLGlDQUFpQyxjQUFjLE1BQU0sU0FBUztBQUM5RDtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBLEtBQUs7O0FBRUw7Ozs7Ozs7Ozs7Ozs7QUM5T0E7QUFBQTtBQUFBO0FBQUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsYUFBYTtBQUNiO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNB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QUFDQTtBQUNBLGFBQWE7QUFDYjtBQUNBLCtCQUErQiw0QkFBNEI7QUFDM0Q7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGFBQWE7QUFDYjtBQUNBO0FBQ0EsbUNBQW1DLG1CQUFtQjtBQUN0RDtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBLGtCQUFrQixjQUFjO0FBQ2hDLFlBQVksaUVBQWlFO0FBQzdFO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSwrQkFBK0IsdUJBQXVCO0FBQ3REO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLHVDQUF1QyxxQkFBcUI7QUFDNUQ7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQSxrREFBa0QsY0FBYzs7QUFFaEUsQ0FBQyIsImZpbGUiOiJtYWluLmJ1bmRsZS5qcyIsInNvdXJjZXNDb250ZW50IjpbIiBcdC8vIFRoZSBtb2R1bGUgY2FjaGVcbiBcdHZhciBpbnN0YWxsZWRNb2R1bGVzID0ge307XG5cbiBcdC8vIFRoZSByZXF1aXJlIGZ1bmN0aW9uXG4gXHRmdW5jdGlvbiBfX3dlYnBhY2tfcmVxdWlyZV9fKG1vZHVsZUlkKSB7XG5cbiBcdFx0Ly8gQ2hlY2sgaWYgbW9kdWxlIGlzIGluIGNhY2hlXG4gXHRcdGlmKGluc3RhbGxlZE1vZHVsZXNbbW9kdWxlSWRdKSB7XG4gXHRcdFx0cmV0dXJuIGluc3RhbGxlZE1vZHVsZXNbbW9kdWxlSWRdLmV4cG9ydHM7XG4gXHRcdH1cbiBcdFx0Ly8gQ3JlYXRlIGEgbmV3IG1vZHVsZSAoYW5kIHB1dCBpdCBpbnRvIHRoZSBjYWNoZSlcbiBcdFx0dmFyIG1vZHVsZSA9IGluc3RhbGxlZE1vZHVsZXNbbW9kdWxlSWRdID0ge1xuIFx0XHRcdGk6IG1vZHVsZUlkLFxuIFx0XHRcdGw6IGZhbHNlLFxuIFx0XHRcdGV4cG9ydHM6IHt9XG4gXHRcdH07XG5cbiBcdFx0Ly8gRXhlY3V0ZSB0aGUgbW9kdWxlIGZ1bmN0aW9uXG4gXHRcdG1vZHVsZXNbbW9kdWxlSWRdLmNhbGwobW9kdWxlLmV4cG9ydHMsIG1vZHVsZSwgbW9kdWxlLmV4cG9ydHMsIF9fd2VicGFja19yZXF1aXJlX18pO1xuXG4gXHRcdC8vIEZsYWcgdGhlIG1vZHVsZSBhcyBsb2FkZWRcbiBcdFx0bW9kdWxlLmwgPSB0cnVlO1xuXG4gXHRcdC8vIFJldHVybiB0aGUgZXhwb3J0cyBvZiB0aGUgbW9kdWxlXG4gXHRcdHJldHVybiBtb2R1bGUuZXhwb3J0cztcbiBcdH1cblxuXG4gXHQvLyBleHBvc2UgdGhlIG1vZHVsZXMgb2JqZWN0IChfX3dlYnBhY2tfbW9kdWxlc19fKVxuIFx0X193ZWJwYWNrX3JlcXVpcmVfXy5tID0gbW9kdWxlcztcblxuIFx0Ly8gZXhwb3NlIHRoZSBtb2R1bGUgY2FjaGVcbiBcdF9fd2VicGFja19yZXF1aXJlX18uYyA9IGluc3RhbGxlZE1vZHVsZXM7XG5cbiBcdC8vIGRlZmluZSBnZXR0ZXIgZnVuY3Rpb24gZm9yIGhhcm1vbnkgZXhwb3J0c1xuIFx0X193ZWJwYWNrX3JlcXVpcmVfXy5kID0gZnVuY3Rpb24oZXhwb3J0cywgbmFtZSwgZ2V0dGVyKSB7XG4gXHRcdGlmKCFfX3dlYnBhY2tfcmVxdWlyZV9fLm8oZXhwb3J0cywgbmFtZSkpIHtcbiBcdFx0XHRPYmplY3QuZGVmaW5lUHJvcGVydHkoZXhwb3J0cywgbmFtZSwgeyBlbnVtZXJhYmxlOiB0cnVlLCBnZXQ6IGdldHRlciB9KTtcbiBcdFx0fVxuIFx0fTtcblxuIFx0Ly8gZGVmaW5lIF9fZXNNb2R1bGUgb24gZXhwb3J0c1xuIFx0X193ZWJwYWNrX3JlcXVpcmVfXy5yID0gZnVuY3Rpb24oZXhwb3J0cykge1xuIFx0XHRpZih0eXBlb2YgU3ltYm9sICE9PSAndW5kZWZpbmVkJyAmJiBTeW1ib2wudG9TdHJpbmdUYWcpIHtcbiBcdFx0XHRPYmplY3QuZGVmaW5lUHJvcGVydHkoZXhwb3J0cywgU3ltYm9sLnRvU3RyaW5nVGFnLCB7IHZhbHVlOiAnTW9kdWxlJyB9KTtcbiBcdFx0fVxuIFx0XHRPYmplY3QuZGVmaW5lUHJvcGVydHkoZXhwb3J0cywgJ19fZXNNb2R1bGUnLCB7IHZhbHVlOiB0cnVlIH0pO1xuIFx0fTtcblxuIFx0Ly8gY3JlYXRlIGEgZmFrZSBuYW1lc3BhY2Ugb2JqZWN0XG4gXHQvLyBtb2RlICYgMTogdmFsdWUgaXMgYSBtb2R1bGUgaWQsIHJlcXVpcmUgaXRcbiBcdC8vIG1vZGUgJiAyOiBtZXJnZSBhbGwgcHJvcGVydGllcyBvZiB2YWx1ZSBpbnRvIHRoZSBuc1xuIFx0Ly8gbW9kZSAmIDQ6IHJldHVybiB2YWx1ZSB3aGVuIGFscmVhZHkgbnMgb2JqZWN0XG4gXHQvLyBtb2RlICYgOHwxOiBiZWhhdmUgbGlrZSByZXF1aXJlXG4gXHRfX3dlYnBhY2tfcmVxdWlyZV9fLnQgPSBmdW5jdGlvbih2YWx1ZSwgbW9kZSkge1xuIFx0XHRpZihtb2RlICYgMSkgdmFsdWUgPSBfX3dlYnBhY2tfcmVxdWlyZV9fKHZhbHVlKTtcbiBcdFx0aWYobW9kZSAmIDgpIHJldHVybiB2YWx1ZTtcbiBcdFx0aWYoKG1vZGUgJiA0KSAmJiB0eXBlb2YgdmFsdWUgPT09ICdvYmplY3QnICYmIHZhbHVlICYmIHZhbHVlLl9fZXNNb2R1bGUpIHJldHVybiB2YWx1ZTtcbiBcdFx0dmFyIG5zID0gT2JqZWN0LmNyZWF0ZShudWxsKTtcbiBcdFx0X193ZWJwYWNrX3JlcXVpcmVfXy5yKG5zKTtcbiBcdFx0T2JqZWN0LmRlZmluZVByb3BlcnR5KG5zLCAnZGVmYXVsdCcsIHsgZW51bWVyYWJsZTogdHJ1ZSwgdmFsdWU6IHZhbHVlIH0pO1xuIFx0XHRpZihtb2RlICYgMiAmJiB0eXBlb2YgdmFsdWUgIT0gJ3N0cmluZycpIGZvcih2YXIga2V5IGluIHZhbHVlKSBfX3dlYnBhY2tfcmVxdWlyZV9fLmQobnMsIGtleSwgZnVuY3Rpb24oa2V5KSB7IHJldHVybiB2YWx1ZVtrZXldOyB9LmJpbmQobnVsbCwga2V5KSk7XG4gXHRcdHJldHVybiBucztcbiBcdH07XG5cbiBcdC8vIGdldERlZmF1bHRFeHBvcnQgZnVuY3Rpb24gZm9yIGNvbXBhdGliaWxpdHkgd2l0aCBub24taGFybW9ueSBtb2R1bGVzXG4gXHRfX3dlYnBhY2tfcmVxdWlyZV9fLm4gPSBmdW5jdGlvbihtb2R1bGUpIHtcbiBcdFx0dmFyIGdldHRlciA9IG1vZHVsZSAmJiBtb2R1bGUuX19lc01vZHVsZSA/XG4gXHRcdFx0ZnVuY3Rpb24gZ2V0RGVmYXVsdCgpIHsgcmV0dXJuIG1vZHVsZVsnZGVmYXVsdCddOyB9IDpcbiBcdFx0XHRmdW5jdGlvbiBnZXRNb2R1bGVFeHBvcnRzKCkgeyByZXR1cm4gbW9kdWxlOyB9O1xuIFx0XHRfX3dlYnBhY2tfcmVxdWlyZV9fLmQoZ2V0dGVyLCAnYScsIGdldHRlcik7XG4gXHRcdHJldHVybiBnZXR0ZXI7XG4gXHR9O1xuXG4gXHQvLyBPYmplY3QucHJvdG90eXBlLmhhc093blByb3BlcnR5LmNhbGxcbiBcdF9fd2VicGFja19yZXF1aXJlX18ubyA9IGZ1bmN0aW9uKG9iamVjdCwgcHJvcGVydHkpIHsgcmV0dXJuIE9iamVjdC5wcm90b3R5cGUuaGFzT3duUHJvcGVydHkuY2FsbChvYmplY3QsIHByb3BlcnR5KTsgfTtcblxuIFx0Ly8gX193ZWJwYWNrX3B1YmxpY19wYXRoX19cbiBcdF9fd2VicGFja19yZXF1aXJlX18ucCA9IFwiXCI7XG5cblxuIFx0Ly8gTG9hZCBlbnRyeSBtb2R1bGUgYW5kIHJldHVybiBleHBvcnRzXG4gXHRyZXR1cm4gX193ZWJwYWNrX3JlcXVpcmVfXyhfX3dlYnBhY2tfcmVxdWlyZV9fLnMgPSBcIi4vbWFpbi5qc1wiKTtcbiIsIi8qKlxuICogQ2VsbEhhbmRsZXIgUHJpbWFyeSBNZXNzYWdlIEhhbmRsZXJcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjbGFzcyBpbXBsZW1lbnRzIGEgc2VydmljZSB0aGF0IGhhbmRsZXNcbiAqIG1lc3NhZ2VzIG9mIGFsbCBraW5kcyB0aGF0IGNvbWUgaW4gb3ZlciBhXG4gKiBgQ2VsbFNvY2tldGAuXG4gKiBOT1RFOiBGb3IgdGhlIG1vbWVudCB0aGVyZSBhcmUgb25seSB0d28ga2luZHNcbiAqIG9mIG1lc3NhZ2VzIGFuZCB0aGVyZWZvcmUgdHdvIGhhbmRsZXJzLiBXZSBoYXZlXG4gKiBwbGFucyB0byBjaGFuZ2UgdGhpcyBzdHJ1Y3R1cmUgdG8gYmUgbW9yZSBmbGV4aWJsZVxuICogYW5kIHNvIHRoZSBBUEkgb2YgdGhpcyBjbGFzcyB3aWxsIGNoYW5nZSBncmVhdGx5LlxuICovXG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuY2xhc3MgQ2VsbEhhbmRsZXIge1xuICAgIGNvbnN0cnVjdG9yKGgsIHByb2plY3RvciwgY29tcG9uZW50cyl7XG5cdC8vIHByb3BzXG5cdHRoaXMuaCA9IGg7XG5cdHRoaXMucHJvamVjdG9yID0gcHJvamVjdG9yO1xuXHR0aGlzLmNvbXBvbmVudHMgPSBjb21wb25lbnRzO1xuXG5cdC8vIEluc3RhbmNlIFByb3BzXG4gICAgICAgIHRoaXMucG9zdHNjcmlwdHMgPSBbXTtcbiAgICAgICAgdGhpcy5jZWxscyA9IHt9O1xuXHR0aGlzLkRPTVBhcnNlciA9IG5ldyBET01QYXJzZXIoKTtcblxuICAgICAgICAvLyBCaW5kIEluc3RhbmNlIE1ldGhvZHNcbiAgICAgICAgdGhpcy5zaG93Q29ubmVjdGlvbkNsb3NlZCA9IHRoaXMuc2hvd0Nvbm5lY3Rpb25DbG9zZWQuYmluZCh0aGlzKTtcblx0dGhpcy5jb25uZWN0aW9uQ2xvc2VkVmlldyA9IHRoaXMuY29ubmVjdGlvbkNsb3NlZFZpZXcuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5oYW5kbGVQb3N0c2NyaXB0ID0gdGhpcy5oYW5kbGVQb3N0c2NyaXB0LmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuaGFuZGxlTWVzc2FnZSA9IHRoaXMuaGFuZGxlTWVzc2FnZS5iaW5kKHRoaXMpO1xuXG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogRmlsbHMgdGhlIHBhZ2UncyBwcmltYXJ5IGRpdiB3aXRoXG4gICAgICogYW4gaW5kaWNhdG9yIHRoYXQgdGhlIHNvY2tldCBoYXMgYmVlblxuICAgICAqIGRpc2Nvbm5lY3RlZC5cbiAgICAgKi9cbiAgICBzaG93Q29ubmVjdGlvbkNsb3NlZCgpe1xuXHR0aGlzLnByb2plY3Rvci5yZXBsYWNlKFxuXHQgICAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoXCJwYWdlX3Jvb3RcIiksXG5cdCAgICB0aGlzLmNvbm5lY3Rpb25DbG9zZWRWaWV3XG5cdCk7XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogQ29udmVuaWVuY2UgbWV0aG9kIHRoYXQgdXBkYXRlc1xuICAgICAqIEJvb3RzdHJhcC1zdHlsZSBwb3BvdmVycyBvblxuICAgICAqIHRoZSBET00uXG4gICAgICogU2VlIGlubGluZSBjb21tZW50c1xuICAgICAqL1xuICAgIHVwZGF0ZVBvcG92ZXJzKCkge1xuICAgICAgICAvLyBUaGlzIGZ1bmN0aW9uIHJlcXVpcmVzXG4gICAgICAgIC8vIGpRdWVyeSBhbmQgcGVyaGFwcyBkb2Vzbid0XG4gICAgICAgIC8vIGJlbG9uZyBpbiB0aGlzIGNsYXNzLlxuICAgICAgICAvLyBUT0RPOiBGaWd1cmUgb3V0IGEgYmV0dGVyIHdheVxuICAgICAgICAvLyBBTFNPIE5PVEU6XG4gICAgICAgIC8vIC0tLS0tLS0tLS0tLS0tLS0tXG4gICAgICAgIC8vIGBnZXRDaGlsZFByb3BgIGlzIGEgY29uc3QgZnVuY3Rpb25cbiAgICAgICAgLy8gdGhhdCBpcyBkZWNsYXJlZCBpbiBhIHNlcGFyYXRlXG4gICAgICAgIC8vIHNjcmlwdCB0YWcgYXQgdGhlIGJvdHRvbSBvZlxuICAgICAgICAvLyBwYWdlLmh0bWwuIFRoYXQncyBhIG5vLW5vIVxuICAgICAgICAkKCdbZGF0YS10b2dnbGU9XCJwb3BvdmVyXCJdJykucG9wb3Zlcih7XG4gICAgICAgICAgICBodG1sOiB0cnVlLFxuICAgICAgICAgICAgY29udGFpbmVyOiAnYm9keScsXG4gICAgICAgICAgICB0aXRsZTogZnVuY3Rpb24gKCkge1xuICAgICAgICAgICAgICAgIHJldHVybiBnZXRDaGlsZFByb3AodGhpcywgJ3RpdGxlJyk7XG4gICAgICAgICAgICB9LFxuICAgICAgICAgICAgY29udGVudDogZnVuY3Rpb24gKCkge1xuICAgICAgICAgICAgICAgIHJldHVybiBnZXRDaGlsZFByb3AodGhpcywgJ2NvbnRlbnQnKTtcbiAgICAgICAgICAgIH0sXG4gICAgICAgICAgICBwbGFjZW1lbnQ6IGZ1bmN0aW9uIChwb3BwZXJFbCwgdHJpZ2dlcmluZ0VsKSB7XG4gICAgICAgICAgICAgICAgbGV0IHBsYWNlbWVudCA9IHRyaWdnZXJpbmdFbC5kYXRhc2V0LnBsYWNlbWVudDtcbiAgICAgICAgICAgICAgICBpZihwbGFjZW1lbnQgPT0gdW5kZWZpbmVkKXtcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIFwiYm90dG9tXCI7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIHJldHVybiBwbGFjZW1lbnQ7XG4gICAgICAgICAgICB9XG4gICAgICAgIH0pO1xuICAgICAgICAkKCcucG9wb3Zlci1kaXNtaXNzJykucG9wb3Zlcih7XG4gICAgICAgICAgICB0cmlnZ2VyOiAnZm9jdXMnXG4gICAgICAgIH0pO1xuICAgIH1cblxuICAgIC8qKlxuICAgICAqIFByaW1hcnkgbWV0aG9kIGZvciBoYW5kbGluZ1xuICAgICAqICdwb3N0c2NyaXB0cycgbWVzc2FnZXMsIHdoaWNoIHRlbGxcbiAgICAgKiB0aGlzIG9iamVjdCB0byBnbyB0aHJvdWdoIGl0J3MgYXJyYXlcbiAgICAgKiBvZiBzY3JpcHQgc3RyaW5ncyBhbmQgdG8gZXZhbHVhdGUgdGhlbS5cbiAgICAgKiBUaGUgZXZhbHVhdGlvbiBpcyBkb25lIG9uIHRoZSBnbG9iYWxcbiAgICAgKiB3aW5kb3cgb2JqZWN0IGV4cGxpY2l0bHkuXG4gICAgICogTk9URTogRnV0dXJlIHJlZmFjdG9yaW5ncy9yZXN0cnVjdHVyaW5nc1xuICAgICAqIHdpbGwgcmVtb3ZlIG11Y2ggb2YgdGhlIG5lZWQgdG8gY2FsbCBldmFsIVxuICAgICAqIEBwYXJhbSB7c3RyaW5nfSBtZXNzYWdlIC0gVGhlIGluY29taW5nIHN0cmluZ1xuICAgICAqIGZyb20gdGhlIHNvY2tldC5cbiAgICAgKi9cbiAgICBoYW5kbGVQb3N0c2NyaXB0KG1lc3NhZ2Upe1xuICAgICAgICAvLyBFbHNld2hlcmUsIHVwZGF0ZSBwb3BvdmVycyBmaXJzdFxuICAgICAgICAvLyBOb3cgd2UgZXZhbHVhdGUgc2NyaXB0cyBjb21pbmdcbiAgICAgICAgLy8gYWNyb3NzIHRoZSB3aXJlLlxuICAgICAgICB0aGlzLnVwZGF0ZVBvcG92ZXJzKCk7XG4gICAgICAgIHdoaWxlKHRoaXMucG9zdHNjcmlwdHMubGVuZ3RoKXtcblx0ICAgIGxldCBwb3N0c2NyaXB0ID0gdGhpcy5wb3N0c2NyaXB0cy5wb3AoKTtcblx0ICAgIHRyeSB7XG5cdFx0d2luZG93LmV2YWwocG9zdHNjcmlwdCk7XG5cdCAgICB9IGNhdGNoKGUpe1xuICAgICAgICAgICAgICAgIGNvbnNvbGUuZXJyb3IoXCJFUlJPUiBSVU5OSU5HIFBPU1RTQ1JJUFRcIiwgZSk7XG4gICAgICAgICAgICAgICAgY29uc29sZS5sb2cocG9zdHNjcmlwdCk7XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBQcmltYXJ5IG1ldGhvZCBmb3IgaGFuZGxpbmcgJ25vcm1hbCdcbiAgICAgKiAoaWUgbm9uLXBvc3RzY3JpcHRzKSBtZXNzYWdlcyB0aGF0IGhhdmVcbiAgICAgKiBiZWVuIGRlc2VyaWFsaXplZCBmcm9tIEpTT04uXG4gICAgICogRm9yIHRoZSBtb21lbnQsIHRoZXNlIG1lc3NhZ2VzIGRlYWxcbiAgICAgKiBlbnRpcmVseSB3aXRoIERPTSByZXBsYWNlbWVudCBvcGVyYXRpb25zLCB3aGljaFxuICAgICAqIHRoaXMgbWV0aG9kIGltcGxlbWVudHMuXG4gICAgICogQHBhcmFtIHtvYmplY3R9IG1lc3NhZ2UgLSBBIGRlc2VyaWFsaXplZFxuICAgICAqIEpTT04gbWVzc2FnZSBmcm9tIHRoZSBzZXJ2ZXIgdGhhdCBoYXNcbiAgICAgKiBpbmZvcm1hdGlvbiBhYm91dCBlbGVtZW50cyB0aGF0IG5lZWQgdG9cbiAgICAgKiBiZSB1cGRhdGVkLlxuICAgICAqL1xuICAgIGhhbmRsZU1lc3NhZ2UobWVzc2FnZSl7XG4gICAgICAgIGxldCBuZXdDb21wb25lbnRzID0gW107XG5cdGlmKHRoaXMuY2VsbHNbXCJwYWdlX3Jvb3RcIl0gPT0gdW5kZWZpbmVkKXtcbiAgICAgICAgICAgIHRoaXMuY2VsbHNbXCJwYWdlX3Jvb3RcIl0gPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZChcInBhZ2Vfcm9vdFwiKTtcbiAgICAgICAgICAgIHRoaXMuY2VsbHNbXCJob2xkaW5nX3BlblwiXSA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKFwiaG9sZGluZ19wZW5cIik7XG4gICAgICAgIH1cblx0Ly8gV2l0aCB0aGUgZXhjZXB0aW9uIG9mIGBwYWdlX3Jvb3RgIGFuZCBgaG9sZGluZ19wZW5gIGlkIG5vZGVzLCBhbGxcblx0Ly8gZWxlbWVudHMgaW4gdGhpcy5jZWxscyBhcmUgdmlydHVhbC4gRGVwZW5kaWcgb24gd2hldGhlciB3ZSBhcmUgYWRkaW5nIGFcblx0Ly8gbmV3IG5vZGUsIG9yIG1hbmlwdWxhdGluZyBhbiBleGlzdGluZywgd2UgbmVlZWQgdG8gd29yayB3aXRoIHRoZSB1bmRlcmx5aW5nXG5cdC8vIERPTSBub2RlLiBIZW5jZSBpZiB0aGlzLmNlbGxbbWVzc2FnZS5pZF0gaXMgYSB2ZG9tIGVsZW1lbnQgd2UgdXNlIGl0c1xuXHQvLyB1bmRlcmx5aW5nIGRvbU5vZGUgZWxlbWVudCB3aGVuIGluIG9wZXJhdGlvbnMgbGlrZSB0aGlzLnByb2plY3Rvci5yZXBsYWNlKClcblx0bGV0IGNlbGwgPSB0aGlzLmNlbGxzW21lc3NhZ2UuaWRdO1xuXG5cdGlmIChjZWxsICE9PSB1bmRlZmluZWQgJiYgY2VsbC5kb21Ob2RlICE9PSB1bmRlZmluZWQpIHtcblx0ICAgIGNlbGwgPSBjZWxsLmRvbU5vZGU7XG5cdH1cblxuICAgICAgICBpZihtZXNzYWdlLmRpc2NhcmQgIT09IHVuZGVmaW5lZCl7XG4gICAgICAgICAgICAvLyBJbiB0aGUgY2FzZSB3aGVyZSB3ZSBoYXZlIHJlY2VpdmVkIGEgJ2Rpc2NhcmQnIG1lc3NhZ2UsXG4gICAgICAgICAgICAvLyBidXQgdGhlIGNlbGwgcmVxdWVzdGVkIGlzIG5vdCBhdmFpbGFibGUgaW4gb3VyXG4gICAgICAgICAgICAvLyBjZWxscyBjb2xsZWN0aW9uLCB3ZSBzaW1wbHkgZGlzcGxheSBhIHdhcm5pbmc6XG4gICAgICAgICAgICBpZihjZWxsID09IHVuZGVmaW5lZCl7XG4gICAgICAgICAgICAgICAgY29uc29sZS53YXJuKGBSZWNlaXZlZCBkaXNjYXJkIG1lc3NhZ2UgZm9yIG5vbi1leGlzdGluZyBjZWxsIGlkICR7bWVzc2FnZS5pZH1gKTtcbiAgICAgICAgICAgICAgICByZXR1cm47XG4gICAgICAgICAgICB9XG5cdCAgICAvLyBJbnN0ZWFkIG9mIHJlbW92aW5nIHRoZSBub2RlIHdlIHJlcGxhY2Ugd2l0aCB0aGUgYVxuXHQgICAgLy8gYGRpc3BsYXk6bm9uZWAgc3R5bGUgbm9kZSB3aGljaCBlZmZlY3RpdmVseSByZW1vdmVzIGl0XG5cdCAgICAvLyBmcm9tIHRoZSBET01cblx0ICAgIGlmIChjZWxsLnBhcmVudE5vZGUgIT09IG51bGwpIHtcblx0XHR0aGlzLnByb2plY3Rvci5yZXBsYWNlKGNlbGwsICgpID0+IHtcblx0XHQgICAgcmV0dXJuIGgoXCJkaXZcIiwge3N0eWxlOiBcImRpc3BsYXk6bm9uZVwifSwgW10pO1xuXHRcdH0pO1xuXHQgICAgfVxuXHR9IGVsc2UgaWYobWVzc2FnZS5pZCAhPT0gdW5kZWZpbmVkKXtcblx0ICAgIC8vIEEgZGljdGlvbmFyeSBvZiBpZHMgd2l0aGluIHRoZSBvYmplY3QgdG8gcmVwbGFjZS5cblx0ICAgIC8vIFRhcmdldHMgYXJlIHJlYWwgaWRzIG9mIG90aGVyIG9iamVjdHMuXG5cdCAgICBsZXQgcmVwbGFjZW1lbnRzID0gbWVzc2FnZS5yZXBsYWNlbWVudHM7XG5cblx0ICAgIC8vIFRPRE86IHRoaXMgaXMgYSB0ZW1wb3JhcnkgYnJhbmNoaW5nLCB0byBiZSByZW1vdmVkIHdpdGggYSBtb3JlIGxvZ2ljYWwgc2V0dXAuIEFzXG5cdCAgICAvLyBvZiB3cml0aW5nIGlmIHRoZSBtZXNzYWdlIGNvbWluZyBhY3Jvc3MgaXMgc2VuZGluZyBhIFwia25vd25cIiBjb21wb25lbnQgdGhlbiB3ZSB1c2Vcblx0ICAgIC8vIHRoZSBjb21wb25lbnQgaXRzZWxmIGFzIG9wcG9zZWQgdG8gYnVpbGRpbmcgYSB2ZG9tIGVsZW1lbnQgZnJvbSB0aGUgcmF3IGh0bWxcblx0ICAgIGxldCBjb21wb25lbnRDbGFzcyA9IHRoaXMuY29tcG9uZW50c1ttZXNzYWdlLmNvbXBvbmVudF9uYW1lXTtcblx0ICAgIGlmIChjb21wb25lbnRDbGFzcyA9PT0gdW5kZWZpbmVkKSB7XG4gICAgICAgICAgICAgICAgY29uc29sZS53YXJuKGBDb3VsZCBub3QgZmluZCBjb21wb25lbnQgZm9yICR7bWVzc2FnZS5jb21wb25lbnRfbmFtZX1gKTtcblx0XHR2YXIgdmVsZW1lbnQgPSB0aGlzLmh0bWxUb1ZEb21FbChtZXNzYWdlLmNvbnRlbnRzLCBtZXNzYWdlLmlkKTtcblx0ICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgbGV0IGNvbXBvbmVudFByb3BzID0gT2JqZWN0LmFzc2lnbih7XG4gICAgICAgICAgICAgICAgICAgIGlkOiBtZXNzYWdlLmlkLFxuICAgICAgICAgICAgICAgICAgICBuYW1lZENoaWxkcmVuOiBtZXNzYWdlLm5hbWVkQ2hpbGRyZW4sXG4gICAgICAgICAgICAgICAgICAgIGNoaWxkcmVuOiBtZXNzYWdlLmNoaWxkcmVuLFxuICAgICAgICAgICAgICAgICAgICBleHRyYURhdGE6IG1lc3NhZ2UuZXh0cmFfZGF0YVxuICAgICAgICAgICAgICAgIH0sIG1lc3NhZ2UuZXh0cmFfZGF0YSk7XG5cdFx0dmFyIGNvbXBvbmVudCA9IG5ldyBjb21wb25lbnRDbGFzcyhcbiAgICAgICAgICAgICAgICAgICAgY29tcG9uZW50UHJvcHMsXG4gICAgICAgICAgICAgICAgICAgIG1lc3NhZ2UucmVwbGFjZW1lbnRfa2V5c1xuICAgICAgICAgICAgICAgICk7XG4gICAgICAgICAgICAgICAgdmFyIHZlbGVtZW50ID0gY29tcG9uZW50LnJlbmRlcigpO1xuICAgICAgICAgICAgICAgIG5ld0NvbXBvbmVudHMucHVzaChjb21wb25lbnQpO1xuXHQgICAgfVxuXG4gICAgICAgICAgICAvLyBJbnN0YWxsIHRoZSBlbGVtZW50IGludG8gdGhlIGRvbVxuICAgICAgICAgICAgaWYoY2VsbCA9PT0gdW5kZWZpbmVkKXtcblx0XHQvLyBUaGlzIGlzIGEgdG90YWxseSBuZXcgbm9kZS5cbiAgICAgICAgICAgICAgICAvLyBGb3IgdGhlIG1vbWVudCwgYWRkIGl0IHRvIHRoZVxuICAgICAgICAgICAgICAgIC8vIGhvbGRpbmcgcGVuLlxuXHRcdHRoaXMucHJvamVjdG9yLmFwcGVuZCh0aGlzLmNlbGxzW1wiaG9sZGluZ19wZW5cIl0sICgpID0+IHtcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIHZlbGVtZW50O1xuICAgICAgICAgICAgICAgIH0pO1xuXG5cdFx0dGhpcy5jZWxsc1ttZXNzYWdlLmlkXSA9IHZlbGVtZW50O1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICAvLyBSZXBsYWNlIHRoZSBleGlzdGluZyBjb3B5IG9mXG4gICAgICAgICAgICAgICAgLy8gdGhlIG5vZGUgd2l0aCB0aGlzIGluY29taW5nXG4gICAgICAgICAgICAgICAgLy8gY29weS5cblx0XHRpZihjZWxsLnBhcmVudE5vZGUgPT09IG51bGwpe1xuXHRcdCAgICB0aGlzLnByb2plY3Rvci5hcHBlbmQodGhpcy5jZWxsc1tcImhvbGRpbmdfcGVuXCJdLCAoKSA9PiB7XG5cdFx0XHRyZXR1cm4gdmVsZW1lbnQ7XG5cdFx0ICAgIH0pO1xuXHRcdH0gZWxzZSB7XG5cdFx0ICAgIHRoaXMucHJvamVjdG9yLnJlcGxhY2UoY2VsbCwgKCkgPT4ge3JldHVybiB2ZWxlbWVudDt9KTtcblx0XHR9XG5cdCAgICB9XG5cbiAgICAgICAgICAgIHRoaXMuY2VsbHNbbWVzc2FnZS5pZF0gPSB2ZWxlbWVudDtcblxuICAgICAgICAgICAgLy8gTm93IHdpcmUgaW4gcmVwbGFjZW1lbnRzXG4gICAgICAgICAgICBPYmplY3Qua2V5cyhyZXBsYWNlbWVudHMpLmZvckVhY2goKHJlcGxhY2VtZW50S2V5LCBpZHgpID0+IHtcbiAgICAgICAgICAgICAgICBsZXQgdGFyZ2V0ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQocmVwbGFjZW1lbnRLZXkpO1xuICAgICAgICAgICAgICAgIGxldCBzb3VyY2UgPSBudWxsO1xuICAgICAgICAgICAgICAgIGlmKHRoaXMuY2VsbHNbcmVwbGFjZW1lbnRzW3JlcGxhY2VtZW50S2V5XV0gPT09IHVuZGVmaW5lZCl7XG5cdFx0ICAgIC8vIFRoaXMgaXMgYWN0dWFsbHkgYSBuZXcgbm9kZS5cbiAgICAgICAgICAgICAgICAgICAgLy8gV2UnbGwgZGVmaW5lIGl0IGxhdGVyIGluIHRoZVxuICAgICAgICAgICAgICAgICAgICAvLyBldmVudCBzdHJlYW0uXG5cdFx0ICAgIHNvdXJjZSA9IHRoaXMuaChcImRpdlwiLCB7aWQ6IHJlcGxhY2VtZW50S2V5fSwgW10pO1xuICAgICAgICAgICAgICAgICAgICB0aGlzLmNlbGxzW3JlcGxhY2VtZW50c1tyZXBsYWNlbWVudEtleV1dID0gc291cmNlOyBcblx0XHQgICAgdGhpcy5wcm9qZWN0b3IuYXBwZW5kKHRoaXMuY2VsbHNbXCJob2xkaW5nX3BlblwiXSwgKCkgPT4ge1xuXHRcdFx0cmV0dXJuIHNvdXJjZTtcbiAgICAgICAgICAgICAgICAgICAgfSk7XG5cdFx0fSBlbHNlIHtcbiAgICAgICAgICAgICAgICAgICAgLy8gTm90IGEgbmV3IG5vZGVcbiAgICAgICAgICAgICAgICAgICAgc291cmNlID0gdGhpcy5jZWxsc1tyZXBsYWNlbWVudHNbcmVwbGFjZW1lbnRLZXldXTtcbiAgICAgICAgICAgICAgICB9XG5cbiAgICAgICAgICAgICAgICBpZih0YXJnZXQgIT0gbnVsbCl7XG5cdFx0ICAgIHRoaXMucHJvamVjdG9yLnJlcGxhY2UodGFyZ2V0LCAoKSA9PiB7XG5cdFx0XHRyZXR1cm4gc291cmNlO1xuICAgICAgICAgICAgICAgICAgICB9KTtcbiAgICAgICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgICAgICBsZXQgZXJyb3JNc2cgPSBgSW4gbWVzc2FnZSAke21lc3NhZ2V9IGNvdWxkbid0IGZpbmQgJHtyZXBsYWNlbWVudEtleX1gO1xuICAgICAgICAgICAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoZXJyb3JNc2cpO1xuICAgICAgICAgICAgICAgICAgICAvL2NvbnNvbGUubG9nKFwiSW4gbWVzc2FnZSBcIiwgbWVzc2FnZSwgXCIgY291bGRuJ3QgZmluZCBcIiwgcmVwbGFjZW1lbnRLZXkpO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH0pO1xuICAgICAgICB9XG5cbiAgICAgICAgaWYobWVzc2FnZS5wb3N0c2NyaXB0ICE9PSB1bmRlZmluZWQpe1xuICAgICAgICAgICAgdGhpcy5wb3N0c2NyaXB0cy5wdXNoKG1lc3NhZ2UucG9zdHNjcmlwdCk7XG4gICAgICAgIH1cblxuICAgICAgICAvLyBJZiB3ZSBjcmVhdGVkIGFueSBuZXcgY29tcG9uZW50cyBkdXJpbmcgdGhpc1xuICAgICAgICAvLyBtZXNzYWdlIGhhbmRsaW5nIHNlc3Npb24sIHdlIGZpbmFsbHkgY2FsbFxuICAgICAgICAvLyB0aGVpciBgY29tcG9uZW50RGlkTG9hZGAgbGlmZWN5Y2xlIG1ldGhvZHNcbiAgICAgICAgbmV3Q29tcG9uZW50cy5mb3JFYWNoKGNvbXBvbmVudCA9PiB7XG4gICAgICAgICAgICBjb21wb25lbnQuY29tcG9uZW50RGlkTG9hZCgpO1xuICAgICAgICB9KTtcblxuICAgICAgICAvLyBSZW1vdmUgbGVmdG92ZXIgcmVwbGFjZW1lbnQgZGl2c1xuICAgICAgICAvLyB0aGF0IGFyZSBzdGlsbCBpbiB0aGUgcGFnZV9yb290XG4gICAgICAgIC8vIGFmdGVyIHZkb20gaW5zZXJ0aW9uXG4gICAgICAgIGxldCBwYWdlUm9vdCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdwYWdlX3Jvb3QnKTtcbiAgICAgICAgbGV0IGZvdW5kID0gcGFnZVJvb3QucXVlcnlTZWxlY3RvckFsbCgnW2lkKj1cIl9fX19fXCJdJyk7XG4gICAgICAgIGZvdW5kLmZvckVhY2goZWxlbWVudCA9PiB7XG4gICAgICAgICAgICBlbGVtZW50LnJlbW92ZSgpO1xuICAgICAgICB9KTtcbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBIZWxwZXIgZnVuY3Rpb24gdGhhdCBnZW5lcmF0ZXMgdGhlIHZkb20gTm9kZSBmb3JcbiAgICAgKiB0byBiZSBkaXNwbGF5IHdoZW4gY29ubmVjdGlvbiBjbG9zZXNcbiAgICAgKi9cbiAgICBjb25uZWN0aW9uQ2xvc2VkVmlldygpe1xuXHRyZXR1cm4gdGhpcy5oKFwibWFpbi5jb250YWluZXJcIiwge3JvbGU6IFwibWFpblwifSwgW1xuXHQgICAgdGhpcy5oKFwiZGl2XCIsIHtjbGFzczogXCJhbGVydCBhbGVydC1wcmltYXJ5IGNlbnRlci1ibG9jayBtdC01XCJ9LFxuXHRcdFtcIkRpc2Nvbm5lY3RlZFwiXSlcblx0XSk7XG4gICAgfVxuXG4gICAgICAgIC8qKlxuICAgICAqIFRoaXMgaXMgYSAoaG9wZWZ1bGx5IHRlbXBvcmFyeSkgaGFja1xuICAgICAqIHRoYXQgd2lsbCBpbnRlcmNlcHQgdGhlIGZpcnN0IHRpbWUgYVxuICAgICAqIGRyb3Bkb3duIGNhcmF0IGlzIGNsaWNrZWQgYW5kIGJpbmRcbiAgICAgKiBCb290c3RyYXAgRHJvcGRvd24gZXZlbnQgaGFuZGxlcnNcbiAgICAgKiB0byBpdCB0aGF0IHNob3VsZCBiZSBib3VuZCB0byB0aGVcbiAgICAgKiBpZGVudGlmaWVkIGNlbGwuIFdlIGFyZSBmb3JjZWQgdG8gZG8gdGhpc1xuICAgICAqIGJlY2F1c2UgdGhlIGN1cnJlbnQgQ2VsbHMgaW5mcmFzdHJ1Y3R1cmVcbiAgICAgKiBkb2VzIG5vdCBoYXZlIGZsZXhpYmxlIGV2ZW50IGJpbmRpbmcvaGFuZGxpbmcuXG4gICAgICogQHBhcmFtIHtzdHJpbmd9IGNlbGxJZCAtIFRoZSBJRCBvZiB0aGUgY2VsbFxuICAgICAqIHRvIGlkZW50aWZ5IGluIHRoZSBzb2NrZXQgY2FsbGJhY2sgd2Ugd2lsbFxuICAgICAqIGJpbmQgdG8gb3BlbiBhbmQgY2xvc2UgZXZlbnRzIG9uIGRyb3Bkb3duXG4gICAgICovXG4gICAgZHJvcGRvd25Jbml0aWFsQmluZEZvcihjZWxsSWQpe1xuICAgICAgICBsZXQgZWxlbWVudElkID0gY2VsbElkICsgJy1kcm9wZG93bk1lbnVCdXR0b24nO1xuICAgICAgICBsZXQgZWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKGVsZW1lbnRJZCk7XG4gICAgICAgIGlmKCFlbGVtZW50KXtcbiAgICAgICAgICAgIHRocm93IEVycm9yKCdFbGVtZW50IG9mIGlkICcgKyBlbGVtZW50SWQgKyAnIGRvZXNudCBleGlzdCEnKTtcbiAgICAgICAgfVxuICAgICAgICBsZXQgZHJvcGRvd25NZW51ID0gZWxlbWVudC5wYXJlbnRFbGVtZW50O1xuICAgICAgICBsZXQgZmlyc3RUaW1lQ2xpY2tlZCA9IGVsZW1lbnQuZGF0YXNldC5maXJzdGNsaWNrID09ICd0cnVlJztcbiAgICAgICAgaWYoZmlyc3RUaW1lQ2xpY2tlZCl7XG4gICAgICAgICAgICAkKGRyb3Bkb3duTWVudSkub24oJ3Nob3cuYnMuZHJvcGRvd24nLCBmdW5jdGlvbigpe1xuICAgICAgICAgICAgICAgIGNlbGxTb2NrZXQuc2VuZFN0cmluZyhKU09OLnN0cmluZ2lmeSh7XG4gICAgICAgICAgICAgICAgICAgIGV2ZW50OiAnZHJvcGRvd24nLFxuICAgICAgICAgICAgICAgICAgICB0YXJnZXRfY2VsbDogY2VsbElkLFxuICAgICAgICAgICAgICAgICAgICBpc09wZW46IGZhbHNlXG4gICAgICAgICAgICAgICAgfSkpO1xuICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICAkKGRyb3Bkb3duTWVudSkub24oJ2hpZGUuYnMuZHJvcGRvd24nLCBmdW5jdGlvbigpe1xuICAgICAgICAgICAgICAgIGNlbGxTb2NrZXQuc2VuZFN0cmluZyhKU09OLnN0cmluZ2lmeSh7XG4gICAgICAgICAgICAgICAgICAgIGV2ZW50OiAnZHJvcGRvd24nLFxuICAgICAgICAgICAgICAgICAgICB0YXJnZXRfY2VsbDogY2VsbElkLFxuICAgICAgICAgICAgICAgICAgICBpc09wZW46IHRydWVcbiAgICAgICAgICAgICAgICB9KSk7XG4gICAgICAgICAgICB9KTtcblxuICAgICAgICAgICAgLy8gTm93IGV4cGlyZSB0aGUgZmlyc3QgdGltZSBjbGlja2VkXG4gICAgICAgICAgICBlbGVtZW50LmRhdGFzZXQuZmlyc3RjbGljayA9ICdmYWxzZSc7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBVbnNhZmVseSBleGVjdXRlcyBhbnkgcGFzc2VkIGluIHN0cmluZ1xuICAgICAqIGFzIGlmIGl0IGlzIHZhbGlkIEpTIGFnYWluc3QgdGhlIGdsb2JhbFxuICAgICAqIHdpbmRvdyBzdGF0ZS5cbiAgICAgKi9cbiAgICBzdGF0aWMgdW5zYWZlbHlFeGVjdXRlKGFTdHJpbmcpe1xuICAgICAgICB3aW5kb3cuZXhlYyhhU3RyaW5nKTtcbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBIZWxwZXIgZnVuY3Rpb24gdGhhdCB0YWtlcyBzb21lIGluY29taW5nXG4gICAgICogSFRNTCBzdHJpbmcgYW5kIHJldHVybnMgYSBtYXF1ZXR0ZSBoeXBlcnNjcmlwdFxuICAgICAqIFZET00gZWxlbWVudCBmcm9tIGl0LlxuICAgICAqIFRoaXMgdXNlcyB0aGUgaW50ZXJuYWwgYnJvd3NlciBET01wYXJzZXIoKSB0byBnZW5lcmF0ZSB0aGUgaHRtbFxuICAgICAqIHN0cnVjdHVyZSBmcm9tIHRoZSByYXcgc3RyaW5nIGFuZCB0aGVuIHJlY3Vyc2l2ZWx5IGJ1aWxkIHRoZVxuICAgICAqIFZET00gZWxlbWVudFxuICAgICAqIEBwYXJhbSB7c3RyaW5nfSBodG1sIC0gVGhlIG1hcmt1cCB0b1xuICAgICAqIHRyYW5zZm9ybSBpbnRvIGEgcmVhbCBlbGVtZW50LlxuICAgICAqL1xuICAgIGh0bWxUb1ZEb21FbChodG1sLCBpZCl7XG5cdGxldCBkb20gPSB0aGlzLkRPTVBhcnNlci5wYXJzZUZyb21TdHJpbmcoaHRtbCwgXCJ0ZXh0L2h0bWxcIik7XG4gICAgICAgIGxldCBlbGVtZW50ID0gZG9tLmJvZHkuY2hpbGRyZW5bMF07XG4gICAgICAgIHJldHVybiB0aGlzLl9kb21FbFRvVmRvbUVsKGVsZW1lbnQsIGlkKTtcbiAgICB9XG5cbiAgICBfZG9tRWxUb1Zkb21FbChkb21FbCwgaWQpIHtcblx0bGV0IHRhZ05hbWUgPSBkb21FbC50YWdOYW1lLnRvTG9jYWxlTG93ZXJDYXNlKCk7XG5cdGxldCBhdHRycyA9IHtpZDogaWR9O1xuXHRsZXQgaW5kZXg7XG5cblx0Zm9yIChpbmRleCA9IDA7IGluZGV4IDwgZG9tRWwuYXR0cmlidXRlcy5sZW5ndGg7IGluZGV4Kyspe1xuXHQgICAgbGV0IGl0ZW0gPSBkb21FbC5hdHRyaWJ1dGVzLml0ZW0oaW5kZXgpO1xuXHQgICAgYXR0cnNbaXRlbS5uYW1lXSA9IGl0ZW0udmFsdWUudHJpbSgpO1xuXHR9XG5cblx0aWYgKGRvbUVsLmNoaWxkRWxlbWVudENvdW50ID09PSAwKSB7XG5cdCAgICByZXR1cm4gaCh0YWdOYW1lLCBhdHRycywgW2RvbUVsLnRleHRDb250ZW50XSk7XG5cdH1cblxuXHRsZXQgY2hpbGRyZW4gPSBbXTtcblx0Zm9yIChpbmRleCA9IDA7IGluZGV4IDwgZG9tRWwuY2hpbGRyZW4ubGVuZ3RoOyBpbmRleCsrKXtcblx0ICAgIGxldCBjaGlsZCA9IGRvbUVsLmNoaWxkcmVuW2luZGV4XTtcblx0ICAgIGNoaWxkcmVuLnB1c2godGhpcy5fZG9tRWxUb1Zkb21FbChjaGlsZCkpO1xuXHR9XG5cblx0cmV0dXJuIGgodGFnTmFtZSwgYXR0cnMsIGNoaWxkcmVuKTtcbiAgICB9XG59XG5cbmV4cG9ydCB7Q2VsbEhhbmRsZXIsIENlbGxIYW5kbGVyIGFzIGRlZmF1bHR9XG4iLCIvKipcbiAqIEEgY29uY3JldGUgZXJyb3IgdGhyb3duXG4gKiBpZiB0aGUgY3VycmVudCBicm93c2VyIGRvZXNuJ3RcbiAqIHN1cHBvcnQgd2Vic29ja2V0cywgd2hpY2ggaXMgdmVyeVxuICogdW5saWtlbHkuXG4gKi9cbmNsYXNzIFdlYnNvY2tldE5vdFN1cHBvcnRlZCBleHRlbmRzIEVycm9yIHtcbiAgICBjb25zdHJ1Y3RvcihhcmdzKXtcbiAgICAgICAgc3VwZXIoYXJncyk7XG4gICAgfVxufVxuXG4vKipcbiAqIFRoaXMgaXMgdGhlIGdsb2JhbCBmcmFtZVxuICogY29udHJvbC4gV2UgbWlnaHQgY29uc2lkZXJcbiAqIHB1dHRpbmcgaXQgZWxzZXdoZXJlLCBidXRcbiAqIGBDZWxsU29ja2V0YCBpcyBpdHMgb25seVxuICogY29uc3VtZXIuXG4gKi9cbmNvbnN0IEZSQU1FU19QRVJfQUNLID0gMTA7XG5cblxuLyoqXG4gKiBDZWxsU29ja2V0IENvbnRyb2xsZXJcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjbGFzcyBpbXBsZW1lbnRzIGFuIGluc3RhbmNlIG9mXG4gKiBhIGNvbnRyb2xsZXIgdGhhdCB3cmFwcyBhIHdlYnNvY2tldCBjbGllbnRcbiAqIGNvbm5lY3Rpb24gYW5kIGtub3dzIGhvdyB0byBoYW5kbGUgdGhlXG4gKiBpbml0aWFsIHJvdXRpbmcgb2YgbWVzc2FnZXMgYWNyb3NzIHRoZSBzb2NrZXQuXG4gKiBgQ2VsbFNvY2tldGAgaW5zdGFuY2VzIGFyZSBkZXNpZ25lZCBzbyB0aGF0XG4gKiBoYW5kbGVycyBmb3Igc3BlY2lmaWMgdHlwZXMgb2YgbWVzc2FnZXMgY2FuXG4gKiByZWdpc3RlciB0aGVtc2VsdmVzIHdpdGggaXQuXG4gKiBOT1RFOiBGb3IgdGhlIG1vbWVudCwgbW9zdCBvZiB0aGlzIGNvZGVcbiAqIGhhcyBiZWVuIGNvcGllZCB2ZXJiYXRpbSBmcm9tIHRoZSBpbmxpbmVcbiAqIHNjcmlwdHMgd2l0aCBvbmx5IHNsaWdodCBtb2RpZmljYXRpb24uXG4gKiovXG5jbGFzcyBDZWxsU29ja2V0IHtcbiAgICBjb25zdHJ1Y3Rvcigpe1xuICAgICAgICAvLyBJbnN0YW5jZSBQcm9wc1xuICAgICAgICB0aGlzLnVyaSA9IHRoaXMuZ2V0VXJpKCk7XG4gICAgICAgIHRoaXMuc29ja2V0ID0gbnVsbDtcbiAgICAgICAgdGhpcy5jdXJyZW50QnVmZmVyID0ge1xuICAgICAgICAgICAgcmVtYWluaW5nOiBudWxsLFxuICAgICAgICAgICAgYnVmZmVyOiBudWxsLFxuICAgICAgICAgICAgaGFzRGlzcGxheTogZmFsc2VcbiAgICAgICAgfTtcblxuICAgICAgICAvKipcbiAgICAgICAgICogQSBjYWxsYmFjayBmb3IgaGFuZGxpbmcgbWVzc2FnZXNcbiAgICAgICAgICogdGhhdCBhcmUgJ3Bvc3RzY3JpcHRzJ1xuICAgICAgICAgKiBAY2FsbGJhY2sgcG9zdHNjcmlwdHNIYW5kbGVyXG4gICAgICAgICAqIEBwYXJhbSB7c3RyaW5nfSBtc2cgLSBUaGUgZm9yd2FyZGVkIG1lc3NhZ2VcbiAgICAgICAgICovXG4gICAgICAgIHRoaXMucG9zdHNjcmlwdHNIYW5kZXIgPSBudWxsO1xuXG4gICAgICAgIC8qKlxuICAgICAgICAgKiBBIGNhbGxiYWNrIGZvciBoYW5kbGluZyBtZXNzYWdlc1xuICAgICAgICAgKiB0aGF0IGFyZSBub3JtYWwgSlNPTiBkYXRhIG1lc3NhZ2VzLlxuICAgICAgICAgKiBAY2FsbGJhY2sgbWVzc2FnZUhhbmRsZXJcbiAgICAgICAgICogQHBhcmFtIHtvYmplY3R9IG1zZyAtIFRoZSBmb3J3YXJkZWQgbWVzc2FnZVxuICAgICAgICAgKi9cbiAgICAgICAgdGhpcy5tZXNzYWdlSGFuZGxlciA9IG51bGw7XG5cbiAgICAgICAgLyoqXG4gICAgICAgICAqIEEgY2FsbGJhY2sgZm9yIGhhbmRsaW5nIG1lc3NhZ2VzXG4gICAgICAgICAqIHdoZW4gdGhlIHdlYnNvY2tldCBjb25uZWN0aW9uIGNsb3Nlcy5cbiAgICAgICAgICogQGNhbGxiYWNrIGNsb3NlSGFuZGxlclxuICAgICAgICAgKi9cbiAgICAgICAgdGhpcy5jbG9zZUhhbmRsZXIgPSBudWxsO1xuXG4gICAgICAgIC8qKlxuICAgICAgICAgKiBBIGNhbGxiYWNrIGZvciBoYW5kbGluZyBtZXNzYWdlc1xuICAgICAgICAgKiB3aGVudCB0aGUgc29ja2V0IGVycm9yc1xuICAgICAgICAgKiBAY2FsbGJhY2sgZXJyb3JIYW5kbGVyXG4gICAgICAgICAqL1xuICAgICAgICB0aGlzLmVycm9ySGFuZGxlciA9IG51bGw7XG5cbiAgICAgICAgLy8gQmluZCBJbnN0YW5jZSBNZXRob2RzXG4gICAgICAgIHRoaXMuY29ubmVjdCA9IHRoaXMuY29ubmVjdC5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLnNlbmRTdHJpbmcgPSB0aGlzLnNlbmRTdHJpbmcuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5oYW5kbGVSYXdNZXNzYWdlID0gdGhpcy5oYW5kbGVSYXdNZXNzYWdlLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMub25Qb3N0c2NyaXB0cyA9IHRoaXMub25Qb3N0c2NyaXB0cy5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLm9uTWVzc2FnZSA9IHRoaXMub25NZXNzYWdlLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMub25DbG9zZSA9IHRoaXMub25DbG9zZS5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLm9uRXJyb3IgPSB0aGlzLm9uRXJyb3IuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBSZXR1cm5zIGEgcHJvcGVybHkgZm9ybWF0dGVkIFVSSVxuICAgICAqIGZvciB0aGUgc29ja2V0IGZvciBhbnkgZ2l2ZW4gY3VycmVudFxuICAgICAqIGJyb3dzZXIgbG9jYXRpb24uXG4gICAgICogQHJldHVybnMge3N0cmluZ30gQSBVUkkgc3RyaW5nLlxuICAgICAqL1xuICAgIGdldFVyaSgpe1xuICAgICAgICBsZXQgbG9jYXRpb24gPSB3aW5kb3cubG9jYXRpb247XG4gICAgICAgIGxldCB1cmkgPSBcIlwiO1xuICAgICAgICBpZihsb2NhdGlvbi5wcm90b2NvbCA9PT0gXCJodHRwczpcIil7XG4gICAgICAgICAgICB1cmkgKz0gXCJ3c3M6XCI7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICB1cmkgKz0gXCJ3czpcIjtcbiAgICAgICAgfVxuICAgICAgICB1cmkgPSBgJHt1cml9Ly8ke2xvY2F0aW9uLmhvc3R9YDtcbiAgICAgICAgdXJpID0gYCR7dXJpfS9zb2NrZXQke2xvY2F0aW9uLnBhdGhuYW1lfSR7bG9jYXRpb24uc2VhcmNofWA7XG4gICAgICAgIHJldHVybiB1cmk7XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogVGVsbHMgdGhpcyBvYmplY3QncyBpbnRlcm5hbCB3ZWJzb2NrZXRcbiAgICAgKiB0byBpbnN0YW50aWF0ZSBpdHNlbGYgYW5kIGNvbm5lY3QgdG9cbiAgICAgKiB0aGUgcHJvdmlkZWQgVVJJLiBUaGUgVVJJIHdpbGwgYmUgc2V0IHRvXG4gICAgICogdGhpcyBpbnN0YW5jZSdzIGB1cmlgIHByb3BlcnR5IGZpcnN0LiBJZiBub1xuICAgICAqIHVyaSBpcyBwYXNzZWQsIGBjb25uZWN0KClgIHdpbGwgdXNlIHRoZSBjdXJyZW50XG4gICAgICogYXR0cmlidXRlJ3MgdmFsdWUuXG4gICAgICogQHBhcmFtIHtzdHJpbmd9IHVyaSAtIEEgIFVSSSB0byBjb25uZWN0IHRoZSBzb2NrZXRcbiAgICAgKiB0by5cbiAgICAgKi9cbiAgICBjb25uZWN0KHVyaSl7XG4gICAgICAgIGlmKHVyaSl7XG4gICAgICAgICAgICB0aGlzLnVyaSA9IHVyaTtcbiAgICAgICAgfVxuICAgICAgICBpZih3aW5kb3cuV2ViU29ja2V0KXtcbiAgICAgICAgICAgIHRoaXMuc29ja2V0ID0gbmV3IFdlYlNvY2tldCh0aGlzLnVyaSk7XG4gICAgICAgIH0gZWxzZSBpZih3aW5kb3cuTW96V2ViU29ja2V0KXtcbiAgICAgICAgICAgIHRoaXMuc29ja2V0ID0gTW96V2ViU29ja2V0KHRoaXMudXJpKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHRocm93IG5ldyBXZWJzb2NrZXROb3RTdXBwb3J0ZWQoKTtcbiAgICAgICAgfVxuXG4gICAgICAgIHRoaXMuc29ja2V0Lm9uY2xvc2UgPSB0aGlzLmNsb3NlSGFuZGxlcjtcbiAgICAgICAgdGhpcy5zb2NrZXQub25tZXNzYWdlID0gdGhpcy5oYW5kbGVSYXdNZXNzYWdlLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuc29ja2V0Lm9uZXJyb3IgPSB0aGlzLmVycm9ySGFuZGxlcjtcbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBDb252ZW5pZW5jZSBtZXRob2QgdGhhdCBzZW5kcyB0aGUgcGFzc2VkXG4gICAgICogc3RyaW5nIG9uIHRoaXMgaW5zdGFuY2UncyB1bmRlcmx5aW5nXG4gICAgICogd2Vic29rZXQgY29ubmVjdGlvbi5cbiAgICAgKiBAcGFyYW0ge3N0cmluZ30gYVN0cmluZyAtIEEgc3RyaW5nIHRvIHNlbmRcbiAgICAgKi9cbiAgICBzZW5kU3RyaW5nKGFTdHJpbmcpe1xuICAgICAgICBpZih0aGlzLnNvY2tldCl7XG4gICAgICAgICAgICB0aGlzLnNvY2tldC5zZW5kKGFTdHJpbmcpO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgLy8gSWRlYWxseSB3ZSBtb3ZlIHRoZSBkb20gb3BlcmF0aW9ucyBvZlxuICAgIC8vIHRoaXMgZnVuY3Rpb24gb3V0IGludG8gYW5vdGhlciBjbGFzcyBvclxuICAgIC8vIGNvbnRleHQuXG4gICAgLyoqXG4gICAgICogVXNpbmcgdGhlIGludGVybmFsIGBjdXJyZW50QnVmZmVyYCwgdGhpc1xuICAgICAqIG1ldGhvZCBjaGVja3MgdG8gc2VlIGlmIGEgbGFyZ2UgbXVsdGktZnJhbWVcbiAgICAgKiBwaWVjZSBvZiB3ZWJzb2NrZXQgZGF0YSBpcyBiZWluZyBzZW50LiBJZiBzbyxcbiAgICAgKiBpdCBwcmVzZW50cyBhbmQgdXBkYXRlcyBhIHNwZWNpZmljIGRpc3BsYXkgaW5cbiAgICAgKiB0aGUgRE9NIHdpdGggdGhlIGN1cnJlbnQgcGVyY2VudGFnZSBldGMuXG4gICAgICogQHBhcmFtIHtzdHJpbmd9IG1zZyAtIFRoZSBtZXNzYWdlIHRvXG4gICAgICogZGlzcGxheSBpbnNpZGUgdGhlIGVsZW1lbnRcbiAgICAgKi9cbiAgICBzZXRMYXJnZURvd25sb2FkRGlzcGxheShtc2cpe1xuXG4gICAgICAgIGlmKG1zZy5sZW5ndGggPT0gMCAmJiAhdGhpcy5jdXJyZW50QnVmZmVyLmhhc0Rpc3BsYXkpe1xuICAgICAgICAgICAgcmV0dXJuO1xuICAgICAgICB9XG5cbiAgICAgICAgdGhpcy5jdXJyZW50QnVmZmVyLmhhc0Rpc3BsYXkgPSAobXNnLmxlbmd0aCAhPSAwKTtcblxuICAgICAgICBsZXQgZWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKFwib2JqZWN0X2RhdGFiYXNlX2xhcmdlX3BlbmRpbmdfZG93bmxvYWRfdGV4dFwiKTtcbiAgICAgICAgaWYoZWxlbWVudCAhPSB1bmRlZmluZWQpe1xuICAgICAgICAgICAgZWxlbWVudC5pbm5lckhUTUwgPSBtc2c7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBIYW5kbGVzIHRoZSBgb25tZXNzYWdlYCBldmVudCBvZiB0aGUgdW5kZXJseWluZ1xuICAgICAqIHdlYnNvY2tldC5cbiAgICAgKiBUaGlzIG1ldGhvZCBrbm93cyBob3cgdG8gZmlsbCB0aGUgaW50ZXJuYWxcbiAgICAgKiBidWZmZXIgKHRvIGdldCBhcm91bmQgdGhlIGZyYW1lIGxpbWl0KSBhbmQgb25seVxuICAgICAqIHRyaWdnZXIgc3Vic2VxdWVudCBoYW5kbGVycyBmb3IgaW5jb21pbmcgbWVzc2FnZXMuXG4gICAgICogVE9ETzogQnJlYWsgb3V0IHRoaXMgbWV0aG9kIGEgYml0IG1vcmUuIEl0IGhhcyBiZWVuXG4gICAgICogY29waWVkIG5lYXJseSB2ZXJiYXRpbSBmcm9tIHRoZSBvcmlnaW5hbCBjb2RlLlxuICAgICAqIE5PVEU6IEZvciBub3csIHRoZXJlIGFyZSBvbmx5IHR3byB0eXBlcyBvZiBtZXNzYWdlczpcbiAgICAgKiAgICAgICAndXBkYXRlcycgKHdlIGp1c3QgY2FsbCB0aGVzZSBtZXNzYWdlcylcbiAgICAgKiAgICAgICAncG9zdHNjcmlwdHMnICh0aGVzZSBhcmUganVzdCByYXcgbm9uLUpTT04gc3RyaW5ncylcbiAgICAgKiBJZiBhIGJ1ZmZlciBpcyBjb21wbGV0ZSwgdGhpcyBtZXRob2Qgd2lsbCBjaGVjayB0byBzZWUgaWZcbiAgICAgKiBoYW5kbGVycyBhcmUgcmVnaXN0ZXJlZCBmb3IgcG9zdHNjcmlwdC9ub3JtYWwgbWVzc2FnZXNcbiAgICAgKiBhbmQgd2lsbCB0cmlnZ2VyIHRoZW0gaWYgdHJ1ZSBpbiBlaXRoZXIgY2FzZSwgcGFzc2luZ1xuICAgICAqIGFueSBwYXJzZWQgSlNPTiBkYXRhIHRvIHRoZSBjYWxsYmFja3MuXG4gICAgICogQHBhcmFtIHtFdmVudH0gZXZlbnQgLSBUaGUgYG9ubWVzc2FnZWAgZXZlbnQgb2JqZWN0XG4gICAgICogZnJvbSB0aGUgc29ja2V0LlxuICAgICAqL1xuICAgIGhhbmRsZVJhd01lc3NhZ2UoZXZlbnQpe1xuICAgICAgICBpZih0aGlzLmN1cnJlbnRCdWZmZXIucmVtYWluaW5nID09PSBudWxsKXtcbiAgICAgICAgICAgIHRoaXMuY3VycmVudEJ1ZmZlci5yZW1haW5pbmcgPSBKU09OLnBhcnNlKGV2ZW50LmRhdGEpO1xuICAgICAgICAgICAgdGhpcy5jdXJyZW50QnVmZmVyLmJ1ZmZlciA9IFtdO1xuICAgICAgICAgICAgaWYodGhpcy5jdXJyZW50QnVmZmVyLmhhc0Rpc3BsYXkgJiYgdGhpcy5jdXJyZW50QnVmZmVyLnJlbWFpbmluZyA9PSAxKXtcbiAgICAgICAgICAgICAgICAvLyBTRVQgTEFSR0UgRE9XTkxPQUQgRElTUExBWVxuICAgICAgICAgICAgfVxuICAgICAgICAgICAgcmV0dXJuO1xuICAgICAgICB9XG5cbiAgICAgICAgdGhpcy5jdXJyZW50QnVmZmVyLnJlbWFpbmluZyAtPSAxO1xuICAgICAgICB0aGlzLmN1cnJlbnRCdWZmZXIuYnVmZmVyLnB1c2goZXZlbnQuZGF0YSk7XG5cbiAgICAgICAgaWYodGhpcy5jdXJyZW50QnVmZmVyLmJ1ZmZlci5sZW5ndGggJSBGUkFNRVNfUEVSX0FDSyA9PSAwKXtcbiAgICAgICAgICAgIC8vQUNLIGV2ZXJ5IHRlbnRoIG1lc3NhZ2UuIFdlIGhhdmUgdG8gZG8gYWN0aXZlIHB1c2hiYWNrXG4gICAgICAgICAgICAvL2JlY2F1c2UgdGhlIHdlYnNvY2tldCBkaXNjb25uZWN0cyBvbiBDaHJvbWUgaWYgeW91IGphbSB0b29cbiAgICAgICAgICAgIC8vbXVjaCBpbiBhdCBvbmNlXG4gICAgICAgICAgICB0aGlzLnNlbmRTdHJpbmcoXG4gICAgICAgICAgICAgICAgSlNPTi5zdHJpbmdpZnkoe1xuICAgICAgICAgICAgICAgICAgICBcIkFDS1wiOiB0aGlzLmN1cnJlbnRCdWZmZXIuYnVmZmVyLmxlbmd0aFxuICAgICAgICAgICAgICAgIH0pKTtcbiAgICAgICAgICAgIGxldCBwZXJjZW50YWdlID0gTWF0aC5yb3VuZCgxMDAqdGhpcy5jdXJyZW50QnVmZmVyLmJ1ZmZlci5sZW5ndGggLyAodGhpcy5jdXJyZW50QnVmZmVyLnJlbWFpbmluZyArIHRoaXMuY3VycmVudEJ1ZmZlci5idWZmZXIubGVuZ3RoKSk7XG4gICAgICAgICAgICBsZXQgdG90YWwgPSBNYXRoLnJvdW5kKCh0aGlzLmN1cnJlbnRCdWZmZXIucmVtYWluaW5nICsgdGhpcy5jdXJyZW50QnVmZmVyLmJ1ZmZlci5sZW5ndGgpIC8gKDEwMjQgLyAzMikpO1xuICAgICAgICAgICAgbGV0IHByb2dyZXNzU3RyID0gYChEb3dubG9hZGVkICR7cGVyY2VudGFnZX0lIG9mICR7dG90YWx9IE1CKWA7XG4gICAgICAgICAgICB0aGlzLnNldExhcmdlRG93bmxvYWREaXNwbGF5KHByb2dyZXNzU3RyKTtcbiAgICAgICAgfVxuXG4gICAgICAgIGlmKHRoaXMuY3VycmVudEJ1ZmZlci5yZW1haW5pbmcgPiAwKXtcbiAgICAgICAgICAgIHJldHVybjtcbiAgICAgICAgfVxuXG4gICAgICAgIHRoaXMuc2V0TGFyZ2VEb3dubG9hZERpc3BsYXkoXCJcIik7XG5cbiAgICAgICAgbGV0IGpvaW5lZEJ1ZmZlciA9IHRoaXMuY3VycmVudEJ1ZmZlci5idWZmZXIuam9pbignJylcblxuICAgICAgICB0aGlzLmN1cnJlbnRCdWZmZXIucmVtYWluaW5nID0gbnVsbDtcbiAgICAgICAgdGhpcy5jdXJyZW50QnVmZmVyLmJ1ZmZlciA9IG51bGw7XG5cbiAgICAgICAgbGV0IHVwZGF0ZSA9IEpTT04ucGFyc2Uoam9pbmVkQnVmZmVyKTtcblxuICAgICAgICBpZih1cGRhdGUgPT0gJ3JlcXVlc3RfYWNrJykge1xuICAgICAgICAgICAgdGhpcy5zZW5kU3RyaW5nKEpTT04uc3RyaW5naWZ5KHsnQUNLJzogMH0pKVxuICAgICAgICB9IGVsc2UgaWYodXBkYXRlID09ICdwb3N0c2NyaXB0cycpe1xuICAgICAgICAgICAgLy8gdXBkYXRlUG9wb3ZlcnMoKTtcbiAgICAgICAgICAgIGlmKHRoaXMucG9zdHNjcmlwdHNIYW5kbGVyKXtcbiAgICAgICAgICAgICAgICB0aGlzLnBvc3RzY3JpcHRzSGFuZGxlcih1cGRhdGUpO1xuICAgICAgICAgICAgfVxuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgaWYodGhpcy5tZXNzYWdlSGFuZGxlcil7XG4gICAgICAgICAgICAgICAgdGhpcy5tZXNzYWdlSGFuZGxlcih1cGRhdGUpO1xuICAgICAgICAgICAgfVxuICAgICAgICB9XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogQ29udmVuaWVuY2UgbWV0aG9kIHRoYXQgYmluZHNcbiAgICAgKiB0aGUgcGFzc2VkIGNhbGxiYWNrIHRvIHRoaXMgaW5zdGFuY2Unc1xuICAgICAqIHBvc3RzY3JpcHRzSGFuZGxlciwgd2hpY2ggaXMgc29tZSBtZXRob2RcbiAgICAgKiB0aGF0IGhhbmRsZXMgbWVzc2FnZXMgZm9yIHBvc3RzY3JpcHRzLlxuICAgICAqIEBwYXJhbSB7cG9zdHNjcmlwdHNIYW5kbGVyfSBjYWxsYmFjayAtIEEgaGFuZGxlclxuICAgICAqIGNhbGxiYWNrIG1ldGhvZCB3aXRoIHRoZSBtZXNzYWdlIGFyZ3VtZW50LlxuICAgICAqL1xuICAgIG9uUG9zdHNjcmlwdHMoY2FsbGJhY2spe1xuICAgICAgICB0aGlzLnBvc3RzY3JpcHRzSGFuZGxlciA9IGNhbGxiYWNrO1xuICAgIH1cblxuICAgIC8qKlxuICAgICAqIENvbnZlbmllbmNlIG1ldGhvZCB0aGF0IGJpbmRzXG4gICAgICogdGhlIHBhc3NlZCBjYWxsYmFjayB0byB0aGlzIGluc3RhbmNlJ3NcbiAgICAgKiBwb3N0c2NyaXB0c0hhbmRsZXIsIHdoaWNoIGlzIHNvbWUgbWV0aG9kXG4gICAgICogdGhhdCBoYW5kbGVzIG1lc3NhZ2VzIGZvciBwb3N0c2NyaXB0cy5cbiAgICAgKiBAcGFyYW0ge21lc3NhZ2VIYW5kbGVyfSBjYWxsYmFjayAtIEEgaGFuZGxlclxuICAgICAqIGNhbGxiYWNrIG1ldGhvZCB3aXRoIHRoZSBtZXNzYWdlIGFyZ3VtZW50LlxuICAgICAqL1xuICAgIG9uTWVzc2FnZShjYWxsYmFjayl7XG4gICAgICAgIHRoaXMubWVzc2FnZUhhbmRsZXIgPSBjYWxsYmFjaztcbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBDb252ZW5pZW5jZSBtZXRob2QgdGhhdCBiaW5kcyB0aGVcbiAgICAgKiBwYXNzZWQgY2FsbGJhY2sgdG8gdGhlIHVuZGVybHlpbmdcbiAgICAgKiB3ZWJzb2NrZXQncyBgb25jbG9zZWAgaGFuZGxlci5cbiAgICAgKiBAcGFyYW0ge2Nsb3NlSGFuZGxlcn0gY2FsbGJhY2sgLSBBIGZ1bmN0aW9uXG4gICAgICogdGhhdCBoYW5kbGVzIGNsb3NlIGV2ZW50cyBvbiB0aGUgc29ja2V0LlxuICAgICAqL1xuICAgIG9uQ2xvc2UoY2FsbGJhY2spe1xuICAgICAgICB0aGlzLmNsb3NlSGFuZGxlciA9IGNhbGxiYWNrO1xuICAgIH1cblxuICAgIC8qKlxuICAgICAqIENvbnZlbmllbmNlIG1ldGhvZCB0aGF0IGJpbmRzIHRoZVxuICAgICAqIHBhc3NlZCBjYWxsYmFjayB0byB0aGUgdW5kZXJseWluZ1xuICAgICAqIHdlYnNvY2tldHMnIGBvbmVycm9yYCBoYW5kbGVyLlxuICAgICAqIEBwYXJhbSB7ZXJyb3JIYW5kbGVyfSBjYWxsYmFjayAtIEEgZnVuY3Rpb25cbiAgICAgKiB0aGF0IGhhbmRsZXMgZXJyb3JzIG9uIHRoZSB3ZWJzb2NrZXQuXG4gICAgICovXG4gICAgb25FcnJvcihjYWxsYmFjayl7XG4gICAgICAgIHRoaXMuZXJyb3JIYW5kbGVyID0gY2FsbGJhY2s7XG4gICAgfVxufVxuXG5cbmV4cG9ydCB7Q2VsbFNvY2tldCwgQ2VsbFNvY2tldCBhcyBkZWZhdWx0fVxuIiwiLyoqXG4gKiBXZSB1c2UgYSBzaW5nbGV0b24gcmVnaXN0cnkgb2JqZWN0XG4gKiB3aGVyZSB3ZSBtYWtlIGF2YWlsYWJsZSBhbGwgcG9zc2libGVcbiAqIENvbXBvbmVudHMuIFRoaXMgaXMgdXNlZnVsIGZvciBXZWJwYWNrLFxuICogd2hpY2ggb25seSBidW5kbGVzIGV4cGxpY2l0bHkgdXNlZFxuICogQ29tcG9uZW50cyBkdXJpbmcgYnVpbGQgdGltZS5cbiAqL1xuaW1wb3J0IHtBc3luY0Ryb3Bkb3duLCBBc3luY0Ryb3Bkb3duQ29udGVudH0gZnJvbSAnLi9jb21wb25lbnRzL0FzeW5jRHJvcGRvd24nO1xuaW1wb3J0IHtCYWRnZX0gZnJvbSAnLi9jb21wb25lbnRzL0JhZGdlJztcbmltcG9ydCB7QnV0dG9ufSBmcm9tICcuL2NvbXBvbmVudHMvQnV0dG9uJztcbmltcG9ydCB7QnV0dG9uR3JvdXB9IGZyb20gJy4vY29tcG9uZW50cy9CdXR0b25Hcm91cCc7XG5pbXBvcnQge0NhcmR9IGZyb20gJy4vY29tcG9uZW50cy9DYXJkJztcbmltcG9ydCB7Q2FyZFRpdGxlfSBmcm9tICcuL2NvbXBvbmVudHMvQ2FyZFRpdGxlJztcbmltcG9ydCB7Q2lyY2xlTG9hZGVyfSBmcm9tICcuL2NvbXBvbmVudHMvQ2lyY2xlTG9hZGVyJztcbmltcG9ydCB7Q2xpY2thYmxlfSBmcm9tICcuL2NvbXBvbmVudHMvQ2xpY2thYmxlJztcbmltcG9ydCB7Q29kZX0gZnJvbSAnLi9jb21wb25lbnRzL0NvZGUnO1xuaW1wb3J0IHtDb2RlRWRpdG9yfSBmcm9tICcuL2NvbXBvbmVudHMvQ29kZUVkaXRvcic7XG5pbXBvcnQge0NvbGxhcHNpYmxlUGFuZWx9IGZyb20gJy4vY29tcG9uZW50cy9Db2xsYXBzaWJsZVBhbmVsJztcbmltcG9ydCB7Q29sdW1uc30gZnJvbSAnLi9jb21wb25lbnRzL0NvbHVtbnMnO1xuaW1wb3J0IHtDb250YWluZXJ9IGZyb20gJy4vY29tcG9uZW50cy9Db250YWluZXInO1xuaW1wb3J0IHtDb250ZXh0dWFsRGlzcGxheX0gZnJvbSAnLi9jb21wb25lbnRzL0NvbnRleHR1YWxEaXNwbGF5JztcbmltcG9ydCB7RHJvcGRvd259IGZyb20gJy4vY29tcG9uZW50cy9Ecm9wZG93bic7XG5pbXBvcnQge0V4cGFuZHN9IGZyb20gJy4vY29tcG9uZW50cy9FeHBhbmRzJztcbmltcG9ydCB7SGVhZGVyQmFyfSBmcm9tICcuL2NvbXBvbmVudHMvSGVhZGVyQmFyJztcbmltcG9ydCB7TG9hZENvbnRlbnRzRnJvbVVybH0gZnJvbSAnLi9jb21wb25lbnRzL0xvYWRDb250ZW50c0Zyb21VcmwnO1xuaW1wb3J0IHtMYXJnZVBlbmRpbmdEb3dubG9hZERpc3BsYXl9IGZyb20gJy4vY29tcG9uZW50cy9MYXJnZVBlbmRpbmdEb3dubG9hZERpc3BsYXknO1xuaW1wb3J0IHtNYWlufSBmcm9tICcuL2NvbXBvbmVudHMvTWFpbic7XG5pbXBvcnQge01vZGFsfSBmcm9tICcuL2NvbXBvbmVudHMvTW9kYWwnO1xuaW1wb3J0IHtPY3RpY29ufSBmcm9tICcuL2NvbXBvbmVudHMvT2N0aWNvbic7XG5pbXBvcnQge1BhZGRpbmd9IGZyb20gJy4vY29tcG9uZW50cy9QYWRkaW5nJztcbmltcG9ydCB7UG9wb3Zlcn0gZnJvbSAnLi9jb21wb25lbnRzL1BvcG92ZXInO1xuaW1wb3J0IHtSb290Q2VsbH0gZnJvbSAnLi9jb21wb25lbnRzL1Jvb3RDZWxsJztcbmltcG9ydCB7U2VxdWVuY2V9IGZyb20gJy4vY29tcG9uZW50cy9TZXF1ZW5jZSc7XG5pbXBvcnQge1Njcm9sbGFibGV9IGZyb20gJy4vY29tcG9uZW50cy9TY3JvbGxhYmxlJztcbmltcG9ydCB7U2luZ2xlTGluZVRleHRCb3h9IGZyb20gJy4vY29tcG9uZW50cy9TaW5nbGVMaW5lVGV4dEJveCc7XG5pbXBvcnQge1NwYW59IGZyb20gJy4vY29tcG9uZW50cy9TcGFuJztcbmltcG9ydCB7U3Vic2NyaWJlZH0gZnJvbSAnLi9jb21wb25lbnRzL1N1YnNjcmliZWQnO1xuaW1wb3J0IHtTdWJzY3JpYmVkU2VxdWVuY2V9IGZyb20gJy4vY29tcG9uZW50cy9TdWJzY3JpYmVkU2VxdWVuY2UnO1xuaW1wb3J0IHtUYWJsZX0gZnJvbSAnLi9jb21wb25lbnRzL1RhYmxlJztcbmltcG9ydCB7VGFic30gZnJvbSAnLi9jb21wb25lbnRzL1RhYnMnO1xuaW1wb3J0IHtUZXh0fSBmcm9tICcuL2NvbXBvbmVudHMvVGV4dCc7XG5pbXBvcnQge1RyYWNlYmFja30gZnJvbSAnLi9jb21wb25lbnRzL1RyYWNlYmFjayc7XG5pbXBvcnQge19OYXZUYWJ9IGZyb20gJy4vY29tcG9uZW50cy9fTmF2VGFiJztcbmltcG9ydCB7R3JpZH0gZnJvbSAnLi9jb21wb25lbnRzL0dyaWQnO1xuaW1wb3J0IHtTaGVldH0gZnJvbSAnLi9jb21wb25lbnRzL1NoZWV0JztcbmltcG9ydCB7UGxvdH0gZnJvbSAnLi9jb21wb25lbnRzL1Bsb3QnO1xuaW1wb3J0IHtfUGxvdFVwZGF0ZXJ9IGZyb20gJy4vY29tcG9uZW50cy9fUGxvdFVwZGF0ZXInO1xuXG5jb25zdCBDb21wb25lbnRSZWdpc3RyeSA9IHtcbiAgICBBc3luY0Ryb3Bkb3duLFxuICAgIEFzeW5jRHJvcGRvd25Db250ZW50LFxuICAgIEJhZGdlLFxuICAgIEJ1dHRvbixcbiAgICBCdXR0b25Hcm91cCxcbiAgICBDYXJkLFxuICAgIENhcmRUaXRsZSxcbiAgICBDaXJjbGVMb2FkZXIsXG4gICAgQ2xpY2thYmxlLFxuICAgIENvZGUsXG4gICAgQ29kZUVkaXRvcixcbiAgICBDb2xsYXBzaWJsZVBhbmVsLFxuICAgIENvbHVtbnMsXG4gICAgQ29udGFpbmVyLFxuICAgIENvbnRleHR1YWxEaXNwbGF5LFxuICAgIERyb3Bkb3duLFxuICAgIEV4cGFuZHMsXG4gICAgSGVhZGVyQmFyLFxuICAgIExvYWRDb250ZW50c0Zyb21VcmwsXG4gICAgTGFyZ2VQZW5kaW5nRG93bmxvYWREaXNwbGF5LFxuICAgIE1haW4sXG4gICAgTW9kYWwsXG4gICAgT2N0aWNvbixcbiAgICBQYWRkaW5nLFxuICAgIFBvcG92ZXIsXG4gICAgUm9vdENlbGwsXG4gICAgU2VxdWVuY2UsXG4gICAgU2Nyb2xsYWJsZSxcbiAgICBTaW5nbGVMaW5lVGV4dEJveCxcbiAgICBTcGFuLFxuICAgIFN1YnNjcmliZWQsXG4gICAgU3Vic2NyaWJlZFNlcXVlbmNlLFxuICAgIFRhYmxlLFxuICAgIFRhYnMsXG4gICAgVGV4dCxcbiAgICBUcmFjZWJhY2ssXG4gICAgX05hdlRhYixcbiAgICBHcmlkLFxuICAgIFNoZWV0LFxuICAgIFBsb3QsXG4gICAgX1Bsb3RVcGRhdGVyXG59O1xuXG5leHBvcnQge0NvbXBvbmVudFJlZ2lzdHJ5LCBDb21wb25lbnRSZWdpc3RyeSBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogQXN5bmNEcm9wZG93biBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyBhIHNpbmdsZSByZWd1bGFyXG4gKiByZXBsYWNlbWVudDpcbiAqICogYGNvbnRlbnRzYFxuICpcbiAqIE5PVEU6IFRoZSBDZWxscyB2ZXJzaW9uIG9mIHRoaXMgY2hpbGQgaXNcbiAqIGVpdGhlciBhIGxvYWRpbmcgaW5kaWNhdG9yLCB0ZXh0LCBvciBhXG4gKiBBc3luY0Ryb3Bkb3duQ29udGVudCBjZWxsLlxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgY29udGVudGAgKHNpbmdsZSkgLSBVc3VhbGx5IGFuIEFzeW5jRHJvcGRvd25Db250ZW50IGNlbGxcbiAqIGBsb2FkaW5nSW5kaWNhdG9yYCAoc2luZ2xlKSAtIEEgQ2VsbCB0aGF0IGRpc3BsYXlzIHRoYXQgdGhlIGNvbnRlbnQgaXMgbG9hZGluZ1xuICovXG5jbGFzcyBBc3luY0Ryb3Bkb3duIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICAvLyBCaW5kIGNvbnRleHQgdG8gbWV0aG9kc1xuICAgICAgICB0aGlzLmFkZERyb3Bkb3duTGlzdGVuZXIgPSB0aGlzLmFkZERyb3Bkb3duTGlzdGVuZXIuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5tYWtlQ29udGVudCA9IHRoaXMubWFrZUNvbnRlbnQuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJBc3luY0Ryb3Bkb3duXCIsXG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbCBidG4tZ3JvdXBcIlxuICAgICAgICAgICAgfSwgW1xuICAgICAgICAgICAgICAgIGgoJ2EnLCB7Y2xhc3M6IFwiYnRuIGJ0bi14cyBidG4tb3V0bGluZS1zZWNvbmRhcnlcIn0sIFt0aGlzLnByb3BzLmV4dHJhRGF0YS5sYWJlbFRleHRdKSxcbiAgICAgICAgICAgICAgICBoKCdidXR0b24nLCB7XG4gICAgICAgICAgICAgICAgICAgIGNsYXNzOiBcImJ0biBidG4teHMgYnRuLW91dGxpbmUtc2Vjb25kYXJ5IGRyb3Bkb3duLXRvZ2dsZSBkcm9wZG93bi10b2dnbGUtc3BsaXRcIixcbiAgICAgICAgICAgICAgICAgICAgdHlwZTogXCJidXR0b25cIixcbiAgICAgICAgICAgICAgICAgICAgaWQ6IGAke3RoaXMucHJvcHMuaWR9LWRyb3Bkb3duTWVudUJ1dHRvbmAsXG4gICAgICAgICAgICAgICAgICAgIFwiZGF0YS10b2dnbGVcIjogXCJkcm9wZG93blwiLFxuICAgICAgICAgICAgICAgICAgICBhZnRlckNyZWF0ZTogdGhpcy5hZGREcm9wZG93bkxpc3RlbmVyLFxuICAgICAgICAgICAgICAgICAgICBcImRhdGEtZmlyc3RjbGlja1wiOiBcInRydWVcIlxuICAgICAgICAgICAgICAgIH0pLFxuICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICAgICAgaWQ6IGAke3RoaXMucHJvcHMuaWR9LWRyb3Bkb3duQ29udGVudFdyYXBwZXJgLFxuICAgICAgICAgICAgICAgICAgICBjbGFzczogXCJkcm9wZG93bi1tZW51XCJcbiAgICAgICAgICAgICAgICB9LCBbdGhpcy5tYWtlQ29udGVudCgpXSlcbiAgICAgICAgICAgIF0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgYWRkRHJvcGRvd25MaXN0ZW5lcihlbGVtZW50KXtcbiAgICAgICAgbGV0IHBhcmVudEVsID0gZWxlbWVudC5wYXJlbnRFbGVtZW50O1xuICAgICAgICBsZXQgY29tcG9uZW50ID0gdGhpcztcbiAgICAgICAgbGV0IGZpcnN0VGltZUNsaWNrZWQgPSAoZWxlbWVudC5kYXRhc2V0LmZpcnN0Y2xpY2sgPT0gXCJ0cnVlXCIpO1xuICAgICAgICBpZihmaXJzdFRpbWVDbGlja2VkKXtcbiAgICAgICAgICAgICQocGFyZW50RWwpLm9uKCdzaG93LmJzLmRyb3Bkb3duJywgZnVuY3Rpb24oKXtcbiAgICAgICAgICAgICAgICBjZWxsU29ja2V0LnNlbmRTdHJpbmcoSlNPTi5zdHJpbmdpZnkoe1xuICAgICAgICAgICAgICAgICAgICBldmVudDonZHJvcGRvd24nLFxuICAgICAgICAgICAgICAgICAgICB0YXJnZXRfY2VsbDogY29tcG9uZW50LnByb3BzLmlkLFxuICAgICAgICAgICAgICAgICAgICBpc09wZW46IGZhbHNlXG4gICAgICAgICAgICAgICAgfSkpO1xuICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICAkKHBhcmVudEVsKS5vbignaGlkZS5icy5kcm9wZG93bicsIGZ1bmN0aW9uKCl7XG4gICAgICAgICAgICAgICAgY2VsbFNvY2tldC5zZW5kU3RyaW5nKEpTT04uc3RyaW5naWZ5KHtcbiAgICAgICAgICAgICAgICAgICAgZXZlbnQ6ICdkcm9wZG93bicsXG4gICAgICAgICAgICAgICAgICAgIHRhcmdldF9jZWxsOiBjb21wb25lbnQucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgICAgIGlzT3BlbjogdHJ1ZVxuICAgICAgICAgICAgICAgIH0pKTtcbiAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgZWxlbWVudC5kYXRhc2V0LmZpcnN0Y2xpY2sgPSBmYWxzZTtcbiAgICAgICAgfVxuICAgIH1cblxuICAgIG1ha2VDb250ZW50KCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ2NvbnRlbnRzJyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCdjb250ZW50Jyk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyBhIHNpbmdsZSByZWd1bGFyXG4gKiByZXBsYWNlbWVudDpcbiAqICogYGNvbnRlbnRzYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogYGNvbnRlbnRgIChzaW5nbGUpIC0gQSBDZWxsIHRoYXQgY29tcHJpc2VzIHRoZSBkcm9wZG93biBjb250ZW50XG4gKiBgbG9hZGluZ0luZGljYXRvcmAgKHNpbmdsZSkgLSBBIENlbGwgdGhhdCByZXByZXNlbnRzIGEgdmlzdWFsXG4gKiAgICAgICBpbmRpY2F0aW5nIHRoYXQgdGhlIGNvbnRlbnQgaXMgbG9hZGluZ1xuICovXG5jbGFzcyBBc3luY0Ryb3Bkb3duQ29udGVudCBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb21wb25lbnQgbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VDb250ZW50ID0gdGhpcy5tYWtlQ29udGVudC5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiBgZHJvcGRvd25Db250ZW50LSR7dGhpcy5wcm9wcy5pZH1gLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkFzeW5jRHJvcGRvd25Db250ZW50XCJcbiAgICAgICAgICAgIH0sIFt0aGlzLm1ha2VDb250ZW50KCldKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIG1ha2VDb250ZW50KCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ2NvbnRlbnRzJyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCdjb250ZW50Jyk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cblxuZXhwb3J0IHtcbiAgICBBc3luY0Ryb3Bkb3duLFxuICAgIEFzeW5jRHJvcGRvd25Db250ZW50LFxuICAgIEFzeW5jRHJvcGRvd24gYXMgZGVmYXVsdFxufTtcbiIsIi8qKlxuICogQmFkZ2UgQ2VsbCBDb21wb25lbnRcbiAqL1xuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBCYWRnZSBoYXMgYSBzaW5nbGUgcmVwbGFjZW1lbnQ6XG4gKiAqIGBjaGlsZGBcbiAqL1xuXG4vKipcbiAqIEFib3V0IE5hbWVkIENoaWxkcmVuXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogYGlubmVyYCAtIFRoZSBjb25jZW50IGNlbGwgb2YgdGhlIEJhZGdlXG4gKi9cbmNsYXNzIEJhZGdlIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29tcG9uZW50IG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlSW5uZXIgPSB0aGlzLm1ha2VJbm5lci5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4oXG4gICAgICAgICAgICBoKCdzcGFuJywge1xuICAgICAgICAgICAgICAgIGNsYXNzOiBgY2VsbCBiYWRnZSBiYWRnZS0ke3RoaXMucHJvcHMuZXh0cmFEYXRhLmJhZGdlU3R5bGV9YCxcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJCYWRnZVwiXG4gICAgICAgICAgICB9LCBbdGhpcy5tYWtlQ29udGVudCgpXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlSW5uZXIoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY2hpbGQnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ2lubmVyJyk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cbmV4cG9ydCB7QmFkZ2UsIEJhZGdlIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBCdXR0b24gQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyBvbmUgcmVndWxhciByZXBsYWNlbWVudDpcbiAqIGBjb250ZW50c2BcbiAqL1xuXG4vKipcbiAqIEFib3V0IE5hbWVkIENoaWxkcmVuXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIGBjb250ZW50YCAoc2luZ2xlKSAtIFRoZSBjZWxsIGluc2lkZSBvZiB0aGUgYnV0dG9uIChpZiBhbnkpXG4gKi9cbmNsYXNzIEJ1dHRvbiBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb250ZXh0IHRvIG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlQ29udGVudCA9IHRoaXMubWFrZUNvbnRlbnQuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5fZ2V0RXZlbnRzID0gdGhpcy5fZ2V0RXZlbnQuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5fZ2V0SFRNTENsYXNzZXMgPSB0aGlzLl9nZXRIVE1MQ2xhc3Nlcy5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4oXG4gICAgICAgICAgICBoKCdidXR0b24nLCB7XG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiQnV0dG9uXCIsXG4gICAgICAgICAgICAgICAgY2xhc3M6IHRoaXMuX2dldEhUTUxDbGFzc2VzKCksXG4gICAgICAgICAgICAgICAgb25jbGljazogdGhpcy5fZ2V0RXZlbnQoJ29uY2xpY2snKVxuICAgICAgICAgICAgfSwgW3RoaXMubWFrZUNvbnRlbnQoKV1cbiAgICAgICAgICAgICApXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgbWFrZUNvbnRlbnQoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY29udGVudHMnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ2NvbnRlbnQnKTtcbiAgICAgICAgfVxuICAgIH1cblxuICAgIF9nZXRFdmVudChldmVudE5hbWUpIHtcbiAgICAgICAgcmV0dXJuIHRoaXMucHJvcHMuZXh0cmFEYXRhLmV2ZW50c1tldmVudE5hbWVdO1xuICAgIH1cblxuICAgIF9nZXRIVE1MQ2xhc3Nlcygpe1xuICAgICAgICBsZXQgY2xhc3NTdHJpbmcgPSB0aGlzLnByb3BzLmV4dHJhRGF0YS5jbGFzc2VzLmpvaW4oXCIgXCIpO1xuICAgICAgICAvLyByZW1lbWJlciB0byB0cmltIHRoZSBjbGFzcyBzdHJpbmcgZHVlIHRvIGEgbWFxdWV0dGUgYnVnXG4gICAgICAgIHJldHVybiBjbGFzc1N0cmluZy50cmltKCk7XG4gICAgfVxufVxuXG5leHBvcnQge0J1dHRvbiwgQnV0dG9uIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBCdXR0b25Hcm91cCBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIGEgc2luZ2xlIGVudW1lcmF0ZWRcbiAqIHJlcGxhY2VtZW50OlxuICogKiBgYnV0dG9uYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgYnV0dG9uc2AgKGFycmF5KSAtIFRoZSBjb25zdGl0dWVudCBidXR0b24gY2VsbHNcbiAqL1xuY2xhc3MgQnV0dG9uR3JvdXAgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29tcG9uZW50IG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlQnV0dG9ucyA9IHRoaXMubWFrZUJ1dHRvbnMuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkJ1dHRvbkdyb3VwXCIsXG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiYnRuLWdyb3VwXCIsXG4gICAgICAgICAgICAgICAgXCJyb2xlXCI6IFwiZ3JvdXBcIlxuICAgICAgICAgICAgfSwgdGhpcy5tYWtlQnV0dG9ucygpXG4gICAgICAgICAgICAgKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIG1ha2VCdXR0b25zKCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRzRm9yKCdidXR0b24nKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkcmVuTmFtZWQoJ2J1dHRvbnMnKTtcbiAgICAgICAgfVxuICAgIH1cblxufVxuXG5leHBvcnQge0J1dHRvbkdyb3VwLCBCdXR0b25Hcm91cCBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogQ2FyZCBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge1Byb3BUeXBlc30gZnJvbSAnLi91dGlsL1Byb3BlcnR5VmFsaWRhdG9yJztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBUaGlzIGNvbXBvbmVudCBjb250YWlucyB0d29cbiAqIHJlZ3VsYXIgcmVwbGFjZW1lbnRzOlxuICogKiBgY29udGVudHNgXG4gKiAqIGBoZWFkZXJgXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBDaGlsZHJlblxuICogYGJvZHlgIChzaW5nbGUpIC0gVGhlIGNlbGwgdG8gcHV0IGluIHRoZSBib2R5IG9mIHRoZSBDYXJkXG4gKiBgaGVhZGVyYCAoc2luZ2xlKSAtIEFuIG9wdGlvbmFsIGhlYWRlciBjZWxsIHRvIHB1dCBhYm92ZVxuICogICAgICAgIGJvZHlcbiAqL1xuY2xhc3MgQ2FyZCBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb21wb25lbnQgbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VCb2R5ID0gdGhpcy5tYWtlQm9keS5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLm1ha2VIZWFkZXIgPSB0aGlzLm1ha2VIZWFkZXIuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgbGV0IGJvZHlDbGFzcyA9IFwiY2FyZC1ib2R5XCI7XG4gICAgICAgIGlmKHRoaXMucHJvcHMuZXh0cmFEYXRhLnBhZGRpbmcpe1xuICAgICAgICAgICAgYm9keUNsYXNzID0gYGNhcmQtYm9keSBwLSR7dGhpcy5wcm9wcy5leHRyYURhdGEucGFkZGluZ31gO1xuICAgICAgICB9XG4gICAgICAgIGxldCBib2R5QXJlYSA9IGgoJ2RpdicsIHtcbiAgICAgICAgICAgIGNsYXNzOiBib2R5Q2xhc3NcbiAgICAgICAgfSwgW3RoaXMubWFrZUJvZHkoKV0pO1xuICAgICAgICBsZXQgaGVhZGVyID0gdGhpcy5tYWtlSGVhZGVyKCk7XG4gICAgICAgIGxldCBoZWFkZXJBcmVhID0gbnVsbDtcbiAgICAgICAgaWYoaGVhZGVyKXtcbiAgICAgICAgICAgIGhlYWRlckFyZWEgPSBoKCdkaXYnLCB7Y2xhc3M6IFwiY2FyZC1oZWFkZXJcIn0sIFtoZWFkZXJdKTtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gaCgnZGl2JyxcbiAgICAgICAgICAgIHtcbiAgICAgICAgICAgICAgICBjbGFzczogXCJjZWxsIGNhcmRcIixcbiAgICAgICAgICAgICAgICBzdHlsZTogdGhpcy5wcm9wcy5kaXZTdHlsZSxcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJDYXJkXCJcbiAgICAgICAgICAgIH0sIFtoZWFkZXJBcmVhLCBib2R5QXJlYV0pO1xuICAgIH1cblxuICAgIG1ha2VCb2R5KCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ2NvbnRlbnRzJyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCdib2R5Jyk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBtYWtlSGVhZGVyKCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICBpZih0aGlzLnJlcGxhY2VtZW50cy5oYXNSZXBsYWNlbWVudCgnaGVhZGVyJykpe1xuICAgICAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignaGVhZGVyJyk7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICByZXR1cm4gbnVsbDtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ2hlYWRlcicpO1xuICAgICAgICB9XG4gICAgfVxufVxuXG5DYXJkLnByb3BUeXBlcyA9IHtcbiAgICBwYWRkaW5nOiB7XG4gICAgICAgIGRlc2NyaXB0aW9uOiBcIlBhZGRpbmcgd2VpZ2h0IGFzIGRlZmluZWQgYnkgQm9vc3RyYXAgY3NzIGNsYXNzZXMuXCIsXG4gICAgICAgIHR5cGU6IFByb3BUeXBlcy5vbmVPZihbUHJvcFR5cGVzLm51bWJlciwgUHJvcFR5cGVzLnN0cmluZ10pXG4gICAgfSxcbiAgICBkaXZTdHlsZToge1xuICAgICAgICBkZXNjcmlwdGlvbjogXCJIVE1MIHN0eWxlIGF0dHJpYnV0ZSBzdHJpbmcuXCIsXG4gICAgICAgIHR5cGU6IFByb3BUeXBlcy5vbmVPZihbUHJvcFR5cGVzLnN0cmluZ10pXG4gICAgfVxufTtcblxuZXhwb3J0IHtDYXJkLCBDYXJkIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBDYXJkVGl0bGUgQ2VsbFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBUaGlzIGNvbXBvbmVudCBoYXMgIHNpbmdsZSByZWd1bGFyXG4gKiByZXBsYWNlbWVudDpcbiAqICogYGNvbnRlbnRzYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgaW5uZXJgIChzaW5nbGUpIC0gVGhlIGlubmVyIGNlbGwgb2YgdGhlIHRpdGxlIGNvbXBvbmVudFxuICovXG5jbGFzcyBDYXJkVGl0bGUgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29tcG9uZW50IG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlSW5uZXIgPSB0aGlzLm1ha2VJbm5lci5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4oXG4gICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbFwiLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkNhcmRUaXRsZVwiXG4gICAgICAgICAgICB9LCBbXG4gICAgICAgICAgICAgICAgdGhpcy5tYWtlSW5uZXIoKVxuICAgICAgICAgICAgXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlSW5uZXIoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY29udGVudHMnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ2lubmVyJyk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cbmV4cG9ydCB7Q2FyZFRpdGxlLCBDYXJkVGl0bGUgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIENpcmNsZUxvYWRlciBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuXG5jbGFzcyBDaXJjbGVMb2FkZXIgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkNpcmNsZUxvYWRlclwiLFxuICAgICAgICAgICAgICAgIGNsYXNzOiBcInNwaW5uZXItZ3Jvd1wiLFxuICAgICAgICAgICAgICAgIHJvbGU6IFwic3RhdHVzXCJcbiAgICAgICAgICAgIH0pXG4gICAgICAgICk7XG4gICAgfVxufVxuXG5DaXJjbGVMb2FkZXIucHJvcFR5cGVzID0ge1xufTtcblxuZXhwb3J0IHtDaXJjbGVMb2FkZXIsIENpcmNsZUxvYWRlciBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogQ2xpY2thYmxlIENlbGwgQ29tcG9uZW50XG4gKi9cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIGEgc2luZ2xlXG4gKiByZWd1bGFyIHJlcGxhY2VtZW50OlxuICogKiBgY29udGVudHNgXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBDaGlsZHJlblxuICogLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIGBjb250ZW50YCAoc2luZ2xlKSAtIFRoZSBjZWxsIHRoYXQgY2FuIGdvIGluc2lkZSB0aGUgY2xpY2thYmxlXG4gKiAgICAgICAgY29tcG9uZW50XG4gKi9cbmNsYXNzIENsaWNrYWJsZSBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb250ZXh0IHRvIG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlQ29udGVudCA9IHRoaXMubWFrZUNvbnRlbnQuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5fZ2V0RXZlbnRzID0gdGhpcy5fZ2V0RXZlbnQuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkNsaWNrYWJsZVwiLFxuICAgICAgICAgICAgICAgIG9uY2xpY2s6IHRoaXMuX2dldEV2ZW50KCdvbmNsaWNrJyksXG4gICAgICAgICAgICAgICAgc3R5bGU6IHRoaXMucHJvcHMuZXh0cmFEYXRhLmRpdlN0eWxlXG4gICAgICAgICAgICB9LCBbXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge30sIFt0aGlzLm1ha2VDb250ZW50KCldKVxuICAgICAgICAgICAgXVxuICAgICAgICAgICAgKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIG1ha2VDb250ZW50KCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ2NvbnRlbnRzJyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCdjb250ZW50Jyk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBfZ2V0RXZlbnQoZXZlbnROYW1lKSB7XG4gICAgICAgIHJldHVybiB0aGlzLnByb3BzLmV4dHJhRGF0YS5ldmVudHNbZXZlbnROYW1lXTtcbiAgICB9XG59XG5cbmV4cG9ydCB7Q2xpY2thYmxlLCBDbGlja2FibGUgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIENvZGUgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyBhIHNpbmdsZVxuICogcmVndWxhciByZXBsYWNlbWVudDpcbiAqICogYGNoaWxkYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgY29kZWAgKHNpbmdsZSkgLSBDb2RlIHRoYXQgd2lsbCBiZSByZW5kZXJlZCBpbnNpZGVcbiAqL1xuY2xhc3MgQ29kZSBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb21wb25lbnQgbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VDb2RlID0gdGhpcy5tYWtlQ29kZS5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gaCgncHJlJyxcbiAgICAgICAgICAgICAgICAge1xuICAgICAgICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbCBjb2RlXCIsXG4gICAgICAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJDb2RlXCJcbiAgICAgICAgICAgICAgICAgfSwgW1xuICAgICAgICAgICAgICAgICAgICAgaChcImNvZGVcIiwge30sIFt0aGlzLm1ha2VDb2RlKCldKVxuICAgICAgICAgICAgICAgICBdXG4gICAgICAgICAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlQ29kZSgpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRlbGVtZW50Rm9yKCdjaGlsZCcpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnY29kZScpO1xuICAgICAgICB9XG4gICAgfVxufVxuXG5leHBvcnQge0NvZGUsIENvZGUgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIENvZGVFZGl0b3IgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbmNsYXNzIENvZGVFZGl0b3IgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuICAgICAgICB0aGlzLmVkaXRvciA9IG51bGw7XG4gICAgICAgIC8vIHVzZWQgdG8gc2NoZWR1bGUgcmVndWxhciBzZXJ2ZXIgdXBkYXRlc1xuICAgICAgICB0aGlzLlNFUlZFUl9VUERBVEVfREVMQVlfTVMgPSAxO1xuICAgICAgICB0aGlzLmVkaXRvclN0eWxlID0gJ3dpZHRoOjEwMCU7aGVpZ2h0OjEwMCU7bWFyZ2luOmF1dG87Ym9yZGVyOjFweCBzb2xpZCBsaWdodGdyYXk7JztcblxuICAgICAgICB0aGlzLnNldHVwRWRpdG9yID0gdGhpcy5zZXR1cEVkaXRvci5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLnNldHVwS2V5YmluZGluZ3MgPSB0aGlzLnNldHVwS2V5YmluZGluZ3MuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5jaGFuZ2VIYW5kbGVyID0gdGhpcy5jaGFuZ2VIYW5kbGVyLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgY29tcG9uZW50RGlkTG9hZCgpIHtcblxuICAgICAgICB0aGlzLnNldHVwRWRpdG9yKCk7XG5cbiAgICAgICAgaWYgKHRoaXMuZWRpdG9yID09PSBudWxsKSB7XG4gICAgICAgICAgICBjb25zb2xlLmxvZyhcImVkaXRvciBjb21wb25lbnQgbG9hZGVkIGJ1dCBmYWlsZWQgdG8gc2V0dXAgZWRpdG9yXCIpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgY29uc29sZS5sb2coXCJzZXR0aW5nIHVwIGVkaXRvclwiKTtcbiAgICAgICAgICAgIHRoaXMuZWRpdG9yLmxhc3RfZWRpdF9taWxsaXMgPSBEYXRlLm5vdygpO1xuXG4gICAgICAgICAgICB0aGlzLmVkaXRvci5zZXRUaGVtZShcImFjZS90aGVtZS90ZXh0bWF0ZVwiKTtcbiAgICAgICAgICAgIHRoaXMuZWRpdG9yLnNlc3Npb24uc2V0TW9kZShcImFjZS9tb2RlL3B5dGhvblwiKTtcbiAgICAgICAgICAgIHRoaXMuZWRpdG9yLnNldEF1dG9TY3JvbGxFZGl0b3JJbnRvVmlldyh0cnVlKTtcbiAgICAgICAgICAgIHRoaXMuZWRpdG9yLnNlc3Npb24uc2V0VXNlU29mdFRhYnModHJ1ZSk7XG4gICAgICAgICAgICB0aGlzLmVkaXRvci5zZXRWYWx1ZSh0aGlzLnByb3BzLmV4dHJhRGF0YS5pbml0aWFsVGV4dCk7XG5cbiAgICAgICAgICAgIGlmICh0aGlzLnByb3BzLmV4dHJhRGF0YS5hdXRvY29tcGxldGUpIHtcbiAgICAgICAgICAgICAgICB0aGlzLmVkaXRvci5zZXRPcHRpb25zKHtlbmFibGVCYXNpY0F1dG9jb21wbGV0aW9uOiB0cnVlfSk7XG4gICAgICAgICAgICAgICAgdGhpcy5lZGl0b3Iuc2V0T3B0aW9ucyh7ZW5hYmxlTGl2ZUF1dG9jb21wbGV0aW9uOiB0cnVlfSk7XG4gICAgICAgICAgICB9XG5cbiAgICAgICAgICAgIGlmICh0aGlzLnByb3BzLmV4dHJhRGF0YS5ub1Njcm9sbCkge1xuICAgICAgICAgICAgICAgIHRoaXMuZWRpdG9yLnNldE9wdGlvbihcIm1heExpbmVzXCIsIEluZmluaXR5KTtcbiAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgaWYgKHRoaXMucHJvcHMuZXh0cmFEYXRhLmZvbnRTaXplICE9PSB1bmRlZmluZWQpIHtcbiAgICAgICAgICAgICAgICB0aGlzLmVkaXRvci5zZXRPcHRpb24oXCJmb250U2l6ZVwiLCB0aGlzLnByb3BzLmV4dHJhRGF0YS5mb250U2l6ZSk7XG4gICAgICAgICAgICB9XG5cbiAgICAgICAgICAgIGlmICh0aGlzLnByb3BzLmV4dHJhRGF0YS5taW5MaW5lcyAhPT0gdW5kZWZpbmVkKSB7XG4gICAgICAgICAgICAgICAgdGhpcy5lZGl0b3Iuc2V0T3B0aW9uKFwibWluTGluZXNcIiwgdGhpcy5wcm9wcy5leHRyYURhdGEubWluTGluZXMpO1xuICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICB0aGlzLnNldHVwS2V5YmluZGluZ3MoKTtcblxuICAgICAgICAgICAgdGhpcy5jaGFuZ2VIYW5kbGVyKCk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIGgoJ2RpdicsXG4gICAgICAgICAgICB7XG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbFwiLFxuICAgICAgICAgICAgICAgIHN0eWxlOiB0aGlzLnByb3BzLmV4dHJhRGF0YS5kaXZTdHlsZSxcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJDb2RlRWRpdG9yXCJcbiAgICAgICAgICAgIH0sXG4gICAgICAgICAgICBbaCgnZGl2JywgeyBpZDogXCJlZGl0b3JcIiArIHRoaXMucHJvcHMuaWQsIHN0eWxlOiB0aGlzLmVkaXRvclN0eWxlIH0sIFtdKVxuICAgICAgICBdKTtcbiAgICB9XG5cbiAgICBzZXR1cEVkaXRvcigpe1xuICAgICAgICBsZXQgZWRpdG9ySWQgPSBcImVkaXRvclwiICsgdGhpcy5wcm9wcy5pZDtcbiAgICAgICAgLy8gVE9ETyBUaGVzZSBhcmUgZ2xvYmFsIHZhciBkZWZpbmVkIGluIHBhZ2UuaHRtbFxuICAgICAgICAvLyB3ZSBzaG91bGQgZG8gc29tZXRoaW5nIGFib3V0IHRoaXMuXG5cbiAgICAgICAgLy8gaGVyZSB3ZSBiaW5nIGFuZCBpbnNldCB0aGUgZWRpdG9yIGludG8gdGhlIGRpdiByZW5kZXJlZCBieVxuICAgICAgICAvLyB0aGlzLnJlbmRlcigpXG4gICAgICAgIHRoaXMuZWRpdG9yID0gYWNlLmVkaXQoZWRpdG9ySWQpO1xuICAgICAgICAvLyBUT0RPOiBkZWFsIHdpdGggdGhpcyBnbG9iYWwgZWRpdG9yIGxpc3RcbiAgICAgICAgYWNlRWRpdG9yc1tlZGl0b3JJZF0gPSB0aGlzLmVkaXRvcjtcbiAgICB9XG5cbiAgICBjaGFuZ2VIYW5kbGVyKCkge1xuXHR2YXIgZWRpdG9ySWQgPSB0aGlzLnByb3BzLmlkO1xuXHR2YXIgZWRpdG9yID0gdGhpcy5lZGl0b3I7XG5cdHZhciBTRVJWRVJfVVBEQVRFX0RFTEFZX01TID0gdGhpcy5TRVJWRVJfVVBEQVRFX0RFTEFZX01TO1xuICAgICAgICB0aGlzLmVkaXRvci5zZXNzaW9uLm9uKFxuICAgICAgICAgICAgXCJjaGFuZ2VcIixcbiAgICAgICAgICAgIGZ1bmN0aW9uKGRlbHRhKSB7XG4gICAgICAgICAgICAgICAgLy8gV1NcbiAgICAgICAgICAgICAgICBsZXQgcmVzcG9uc2VEYXRhID0ge1xuICAgICAgICAgICAgICAgICAgICBldmVudDogJ2VkaXRvcl9jaGFuZ2UnLFxuICAgICAgICAgICAgICAgICAgICAndGFyZ2V0X2NlbGwnOiBlZGl0b3JJZCxcbiAgICAgICAgICAgICAgICAgICAgZGF0YTogZGVsdGFcbiAgICAgICAgICAgICAgICB9O1xuICAgICAgICAgICAgICAgIGNlbGxTb2NrZXQuc2VuZFN0cmluZyhKU09OLnN0cmluZ2lmeShyZXNwb25zZURhdGEpKTtcbiAgICAgICAgICAgICAgICAvL3JlY29yZCB0aGF0IHdlIGp1c3QgZWRpdGVkXG4gICAgICAgICAgICAgICAgZWRpdG9yLmxhc3RfZWRpdF9taWxsaXMgPSBEYXRlLm5vdygpO1xuXG5cdFx0Ly9zY2hlZHVsZSBhIGZ1bmN0aW9uIHRvIHJ1biBpbiAnU0VSVkVSX1VQREFURV9ERUxBWV9NUydtc1xuXHRcdC8vdGhhdCB3aWxsIHVwZGF0ZSB0aGUgc2VydmVyLCBidXQgb25seSBpZiB0aGUgdXNlciBoYXMgc3RvcHBlZCB0eXBpbmcuXG5cdFx0Ly8gVE9ETyB1bmNsZWFyIGlmIHRoaXMgaXMgb3dya2luZyBwcm9wZXJseVxuXHRcdHdpbmRvdy5zZXRUaW1lb3V0KGZ1bmN0aW9uKCkge1xuXHRcdCAgICBpZiAoRGF0ZS5ub3coKSAtIGVkaXRvci5sYXN0X2VkaXRfbWlsbGlzID49IFNFUlZFUl9VUERBVEVfREVMQVlfTVMpIHtcblx0XHRcdC8vc2F2ZSBvdXIgY3VycmVudCBzdGF0ZSB0byB0aGUgcmVtb3RlIGJ1ZmZlclxuXHRcdFx0ZWRpdG9yLmN1cnJlbnRfaXRlcmF0aW9uICs9IDE7XG5cdFx0XHRlZGl0b3IubGFzdF9lZGl0X21pbGxpcyA9IERhdGUubm93KCk7XG5cdFx0XHRlZGl0b3IubGFzdF9lZGl0X3NlbnRfdGV4dCA9IGVkaXRvci5nZXRWYWx1ZSgpO1xuXHRcdFx0Ly8gV1Ncblx0XHRcdGxldCByZXNwb25zZURhdGEgPSB7XG5cdFx0XHQgICAgZXZlbnQ6ICdlZGl0aW5nJyxcblx0XHRcdCAgICAndGFyZ2V0X2NlbGwnOiBlZGl0b3JJZCxcblx0XHRcdCAgICBidWZmZXI6IGVkaXRvci5nZXRWYWx1ZSgpLFxuXHRcdFx0ICAgIHNlbGVjdGlvbjogZWRpdG9yLnNlbGVjdGlvbi5nZXRSYW5nZSgpLFxuXHRcdFx0ICAgIGl0ZXJhdGlvbjogZWRpdG9yLmN1cnJlbnRfaXRlcmF0aW9uXG5cdFx0XHR9O1xuXHRcdFx0Y2VsbFNvY2tldC5zZW5kU3RyaW5nKEpTT04uc3RyaW5naWZ5KHJlc3BvbnNlRGF0YSkpO1xuXHRcdCAgICB9XG5cdFx0fSwgU0VSVkVSX1VQREFURV9ERUxBWV9NUyArIDIpOyAvL25vdGUgdGhlIDJtcyBncmFjZSBwZXJpb2RcbiAgICAgICAgICAgIH1cbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBzZXR1cEtleWJpbmRpbmdzKCkge1xuICAgICAgICBjb25zb2xlLmxvZyhcInNldHRpbmcgdXAga2V5YmluZGluZ3NcIik7XG4gICAgICAgIHRoaXMucHJvcHMuZXh0cmFEYXRhLmtleWJpbmRpbmdzLm1hcCgoa2IpID0+IHtcbiAgICAgICAgICAgIHRoaXMuZWRpdG9yLmNvbW1hbmRzLmFkZENvbW1hbmQoXG4gICAgICAgICAgICAgICAge1xuICAgICAgICAgICAgICAgICAgICBuYW1lOiAnY21kJyArIGtiLFxuICAgICAgICAgICAgICAgICAgICBiaW5kS2V5OiB7d2luOiAnQ3RybC0nICsga2IsICBtYWM6ICdDb21tYW5kLScgKyBrYn0sXG4gICAgICAgICAgICAgICAgICAgIHJlYWRPbmx5OiB0cnVlLFxuICAgICAgICAgICAgICAgICAgICBleGVjOiAoKSA9PiB7XG4gICAgICAgICAgICAgICAgICAgICAgICB0aGlzLmVkaXRvci5jdXJyZW50X2l0ZXJhdGlvbiArPSAxO1xuICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy5lZGl0b3IubGFzdF9lZGl0X21pbGxpcyA9IERhdGUubm93KCk7XG4gICAgICAgICAgICAgICAgICAgICAgICB0aGlzLmVkaXRvci5sYXN0X2VkaXRfc2VudF90ZXh0ID0gdGhpcy5lZGl0b3IuZ2V0VmFsdWUoKTtcblxuICAgICAgICAgICAgICAgICAgICAgICAgLy8gV1NcbiAgICAgICAgICAgICAgICAgICAgICAgIGxldCByZXNwb25zZURhdGEgPSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgZXZlbnQ6ICdrZXliaW5kaW5nJyxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAndGFyZ2V0X2NlbGwnOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICdrZXknOiBrYixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAnYnVmZmVyJzogdGhpcy5lZGl0b3IuZ2V0VmFsdWUoKSxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAnc2VsZWN0aW9uJzogdGhpcy5lZGl0b3Iuc2VsZWN0aW9uLmdldFJhbmdlKCksXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgJ2l0ZXJhdGlvbic6IHRoaXMuZWRpdG9yLmN1cnJlbnRfaXRlcmF0aW9uXG4gICAgICAgICAgICAgICAgICAgICAgICB9O1xuICAgICAgICAgICAgICAgICAgICAgICAgY2VsbFNvY2tldC5zZW5kU3RyaW5nKEpTT04uc3RyaW5naWZ5KHJlc3BvbnNlRGF0YSkpO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgKTtcbiAgICAgICAgfSk7XG4gICAgfVxufVxuXG5leHBvcnQge0NvZGVFZGl0b3IsIENvZGVFZGl0b3IgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIENvbGxhcHNpYmxlUGFuZWwgQ2VsbCBDb21wb25lbnRcbiAqL1xuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50LmpzJztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBUaGlzIGNvbXBvbmVudCBoYXMgdHdvIHNpbmdsZSB0eXBlXG4gKiByZXBsYWNlbWVudHM6XG4gKiAqIGBjb250ZW50YFxuICogKiBgcGFuZWxgXG4gKiBOb3RlIHRoYXQgYHBhbmVsYCBpcyBvbmx5IHJlbmRlcmVkXG4gKiBpZiB0aGUgcGFuZWwgaXMgZXhwYW5kZWRcbiAqL1xuXG4vKipcbiAqIEFib3V0IE5hbWVkIENoaWxkcmVuXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogYGNvbnRlbnRgIChzaW5nbGUpIC0gVGhlIGN1cnJlbnQgY29udGVudCBDZWxsIG9mIHRoZSBwYW5lbFxuICogYHBhbmVsYCAoc2luZ2xlKSAtIFRoZSBjdXJyZW50IChleHBhbmRlZCkgcGFuZWwgdmlld1xuICovXG5jbGFzcyBDb2xsYXBzaWJsZVBhbmVsIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICAvLyBCaW5kIGNvbXBvbmVudCBtZXRob2RzXG4gICAgICAgIHRoaXMubWFrZVBhbmVsID0gdGhpcy5tYWtlUGFuZWwuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5tYWtlQ29udGVudCA9IHRoaXMubWFrZUNvbnRlbnQuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgaWYodGhpcy5wcm9wcy5leHRyYURhdGEuaXNFeHBhbmRlZCl7XG4gICAgICAgICAgICByZXR1cm4oXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgICAgICBjbGFzczogXCJjZWxsIGNvbnRhaW5lci1mbHVpZFwiLFxuICAgICAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiQ29sbGFwc2libGVQYW5lbFwiLFxuICAgICAgICAgICAgICAgICAgICBcImRhdGEtZXhwYW5kZWRcIjogdHJ1ZSxcbiAgICAgICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgICAgIHN0eWxlOiB0aGlzLnByb3BzLmV4dHJhRGF0YS5kaXZTdHlsZVxuICAgICAgICAgICAgICAgIH0sIFtcbiAgICAgICAgICAgICAgICAgICAgaCgnZGl2Jywge2NsYXNzOiBcInJvdyBmbGV4LW5vd3JhcCBuby1ndXR0ZXJzXCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgICAgICBoKCdkaXYnLCB7Y2xhc3M6IFwiY29sLW1kLWF1dG9cIn0sW1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRoaXMubWFrZVBhbmVsKClcbiAgICAgICAgICAgICAgICAgICAgICAgIF0pLFxuICAgICAgICAgICAgICAgICAgICAgICAgaCgnZGl2Jywge2NsYXNzOiBcImNvbC1zbVwifSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRoaXMubWFrZUNvbnRlbnQoKVxuICAgICAgICAgICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgICAgICBdKVxuICAgICAgICAgICAgKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgICAgICBjbGFzczogXCJjZWxsIGNvbnRhaW5lci1mbHVpZFwiLFxuICAgICAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiQ29sbGFwc2libGVQYW5lbFwiLFxuICAgICAgICAgICAgICAgICAgICBcImRhdGEtZXhwYW5kZWRcIjogZmFsc2UsXG4gICAgICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgICAgICBzdHlsZTogdGhpcy5wcm9wcy5leHRyYURhdGEuZGl2U3R5bGVcbiAgICAgICAgICAgICAgICB9LCBbdGhpcy5tYWtlQ29udGVudCgpXSlcbiAgICAgICAgICAgICk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBtYWtlQ29udGVudCgpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdjb250ZW50Jyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCdjb250ZW50Jyk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBtYWtlUGFuZWwoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcigncGFuZWwnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ3BhbmVsJyk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cblxuZXhwb3J0IHtDb2xsYXBzaWJsZVBhbmVsLCBDb2xsYXBzaWJsZVBhbmVsIGFzIGRlZmF1bHR9XG4iLCIvKipcbiAqIENvbHVtbnMgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyBvbmUgZW51bWVyYXRlZFxuICoga2luZCBvZiByZXBsYWNlbWVudDpcbiAqICogYGNgXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBDaGlsZHJlblxuICogLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIGBlbGVtZW50c2AgKGFycmF5KSAtIENlbGwgY29sdW1uIGVsZW1lbnRzXG4gKi9cbmNsYXNzIENvbHVtbnMgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29udGV4dCB0byBtZXRob2RzXG4gICAgICAgIHRoaXMubWFrZUlubmVyQ2hpbGRyZW4gPSB0aGlzLm1ha2VJbm5lckNoaWxkcmVuLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbCBjb250YWluZXItZmx1aWRcIixcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJDb2x1bW5zXCIsXG4gICAgICAgICAgICAgICAgc3R5bGU6IHRoaXMucHJvcHMuZXh0cmFEYXRhLmRpdlN0eWxlXG4gICAgICAgICAgICB9LCBbXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge2NsYXNzOiBcInJvdyBmbGV4LW5vd3JhcFwifSwgdGhpcy5tYWtlSW5uZXJDaGlsZHJlbigpKVxuICAgICAgICAgICAgXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlSW5uZXJDaGlsZHJlbigpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50c0ZvcignYycpLm1hcChyZXBsRWxlbWVudCA9PiB7XG4gICAgICAgICAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgICAgICAgICAgY2xhc3M6IFwiY29sLXNtXCJcbiAgICAgICAgICAgICAgICAgICAgfSwgW3JlcGxFbGVtZW50XSlcbiAgICAgICAgICAgICAgICApO1xuICAgICAgICAgICAgfSk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZHJlbk5hbWVkKCdlbGVtZW50cycpO1xuICAgICAgICB9XG4gICAgfVxufVxuXG5cbmV4cG9ydCB7Q29sdW1ucywgQ29sdW1ucyBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogR2VuZXJpYyBiYXNlIENlbGwgQ29tcG9uZW50LlxuICogU2hvdWxkIGJlIGV4dGVuZGVkIGJ5IG90aGVyXG4gKiBDZWxsIGNsYXNzZXMgb24gSlMgc2lkZS5cbiAqL1xuaW1wb3J0IHtSZXBsYWNlbWVudHNIYW5kbGVyfSBmcm9tICcuL3V0aWwvUmVwbGFjZW1lbnRzSGFuZGxlcic7XG5pbXBvcnQge1Byb3BUeXBlc30gZnJvbSAnLi91dGlsL1Byb3BlcnR5VmFsaWRhdG9yJztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG5jbGFzcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzID0ge30sIHJlcGxhY2VtZW50cyA9IFtdKXtcbiAgICAgICAgdGhpcy5pc0NvbXBvbmVudCA9IHRydWU7XG4gICAgICAgIHRoaXMuX3VwZGF0ZVByb3BzKHByb3BzKTtcblxuICAgICAgICAvLyBSZXBsYWNlbWVudHMgaGFuZGxpbmdcbiAgICAgICAgdGhpcy5yZXBsYWNlbWVudHMgPSBuZXcgUmVwbGFjZW1lbnRzSGFuZGxlcihyZXBsYWNlbWVudHMpO1xuICAgICAgICB0aGlzLnVzZXNSZXBsYWNlbWVudHMgPSAocmVwbGFjZW1lbnRzLmxlbmd0aCA+IDApO1xuXG4gICAgICAgIC8vIFNldHVwIHBhcmVudCByZWxhdGlvbnNoaXAsIGlmXG4gICAgICAgIC8vIGFueS4gSW4gdGhpcyBhYnN0cmFjdCBjbGFzc1xuICAgICAgICAvLyB0aGVyZSBpc24ndCBvbmUgYnkgZGVmYXVsdFxuICAgICAgICB0aGlzLnBhcmVudCA9IG51bGw7XG4gICAgICAgIHRoaXMuX3NldHVwQ2hpbGRSZWxhdGlvbnNoaXBzKCk7XG5cbiAgICAgICAgLy8gRW5zdXJlIHRoYXQgd2UgaGF2ZSBwYXNzZWQgaW4gYW4gaWRcbiAgICAgICAgLy8gd2l0aCB0aGUgcHJvcHMuIFNob3VsZCBlcnJvciBvdGhlcndpc2UuXG4gICAgICAgIGlmKCF0aGlzLnByb3BzLmlkIHx8IHRoaXMucHJvcHMuaWQgPT0gdW5kZWZpbmVkKXtcbiAgICAgICAgICAgIHRocm93IEVycm9yKCdZb3UgbXVzdCBkZWZpbmUgYW4gaWQgZm9yIGV2ZXJ5IGNvbXBvbmVudCBwcm9wcyEnKTtcbiAgICAgICAgfVxuXG4gICAgICAgIHRoaXMudmFsaWRhdGVQcm9wcygpO1xuXG4gICAgICAgIC8vIEJpbmQgY29udGV4dCB0byBtZXRob2RzXG4gICAgICAgIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yID0gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRzRm9yID0gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRzRm9yLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuY29tcG9uZW50RGlkTG9hZCA9IHRoaXMuY29tcG9uZW50RGlkTG9hZC5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLmNoaWxkcmVuRG8gPSB0aGlzLmNoaWxkcmVuRG8uYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5uYW1lZENoaWxkcmVuRG8gPSB0aGlzLm5hbWVkQ2hpbGRyZW5Eby5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLnJlbmRlckNoaWxkTmFtZWQgPSB0aGlzLnJlbmRlckNoaWxkTmFtZWQuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5yZW5kZXJDaGlsZHJlbk5hbWVkID0gdGhpcy5yZW5kZXJDaGlsZHJlbk5hbWVkLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuX3NldHVwQ2hpbGRSZWxhdGlvbnNoaXBzID0gdGhpcy5fc2V0dXBDaGlsZFJlbGF0aW9uc2hpcHMuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5fdXBkYXRlUHJvcHMgPSB0aGlzLl91cGRhdGVQcm9wcy5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLl9yZWN1cnNpdmVseU1hcE5hbWVkQ2hpbGRyZW4gPSB0aGlzLl9yZWN1cnNpdmVseU1hcE5hbWVkQ2hpbGRyZW4uYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgLy8gT2JqZWN0cyB0aGF0IGV4dGVuZCBmcm9tXG4gICAgICAgIC8vIG1lIHNob3VsZCBvdmVycmlkZSB0aGlzXG4gICAgICAgIC8vIG1ldGhvZCBpbiBvcmRlciB0byBnZW5lcmF0ZVxuICAgICAgICAvLyBzb21lIGNvbnRlbnQgZm9yIHRoZSB2ZG9tXG4gICAgICAgIHRocm93IG5ldyBFcnJvcignWW91IG11c3QgaW1wbGVtZW50IGEgYHJlbmRlcmAgbWV0aG9kIG9uIENvbXBvbmVudCBvYmplY3RzIScpO1xuICAgIH1cblxuICAgIC8qKlxuICAgICAqIE9iamVjdCB0aGF0IGV4dGVuZCBmcm9tIG1lIGNvdWxkIG92ZXJ3cml0ZSB0aGlzIG1ldGhvZC5cbiAgICAgKiBJdCBpcyB0byBiZSB1c2VkIGZvciBsaWZlY3lsY2UgbWFuYWdlbWVudCBhbmQgaXMgdG8gYmUgY2FsbGVkXG4gICAgICogYWZ0ZXIgdGhlIGNvbXBvbmVudHMgbG9hZHMuXG4gICAgKi9cbiAgICBjb21wb25lbnREaWRMb2FkKCkge1xuICAgICAgICByZXR1cm4gbnVsbDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogUmVzcG9uZHMgd2l0aCBhIGh5cGVyc2NyaXB0IG9iamVjdFxuICAgICAqIHRoYXQgcmVwcmVzZW50cyBhIGRpdiB0aGF0IGlzIGZvcm1hdHRlZFxuICAgICAqIGFscmVhZHkgZm9yIHRoZSByZWd1bGFyIHJlcGxhY2VtZW50LlxuICAgICAqIFRoaXMgb25seSB3b3JrcyBmb3IgcmVndWxhciB0eXBlIHJlcGxhY2VtZW50cy5cbiAgICAgKiBGb3IgZW51bWVyYXRlZCByZXBsYWNlbWVudHMsIHVzZVxuICAgICAqICNnZXRSZXBsYWNlbWVudEVsZW1lbnRzRm9yKClcbiAgICAgKi9cbiAgICBnZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IocmVwbGFjZW1lbnROYW1lKXtcbiAgICAgICAgbGV0IHJlcGxhY2VtZW50ID0gdGhpcy5yZXBsYWNlbWVudHMuZ2V0UmVwbGFjZW1lbnRGb3IocmVwbGFjZW1lbnROYW1lKTtcbiAgICAgICAgaWYocmVwbGFjZW1lbnQpe1xuICAgICAgICAgICAgbGV0IG5ld0lkID0gYCR7dGhpcy5wcm9wcy5pZH1fJHtyZXBsYWNlbWVudH1gO1xuICAgICAgICAgICAgcmV0dXJuIGgoJ2RpdicsIHtpZDogbmV3SWQsIGtleTogbmV3SWR9LCBbXSk7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIG51bGw7XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogUmVzcG9uZCB3aXRoIGFuIGFycmF5IG9mIGh5cGVyc2NyaXB0XG4gICAgICogb2JqZWN0cyB0aGF0IGFyZSBkaXZzIHdpdGggaWRzIHRoYXQgbWF0Y2hcbiAgICAgKiByZXBsYWNlbWVudCBzdHJpbmcgaWRzIGZvciB0aGUga2luZCBvZlxuICAgICAqIHJlcGxhY2VtZW50IGxpc3QgdGhhdCBpcyBlbnVtZXJhdGVkLFxuICAgICAqIGllIGBfX19fYnV0dG9uXzFgLCBgX19fX2J1dHRvbl8yX19gIGV0Yy5cbiAgICAgKi9cbiAgICBnZXRSZXBsYWNlbWVudEVsZW1lbnRzRm9yKHJlcGxhY2VtZW50TmFtZSl7XG4gICAgICAgIGlmKCF0aGlzLnJlcGxhY2VtZW50cy5oYXNSZXBsYWNlbWVudChyZXBsYWNlbWVudE5hbWUpKXtcbiAgICAgICAgICAgIHJldHVybiBudWxsO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiB0aGlzLnJlcGxhY2VtZW50cy5tYXBSZXBsYWNlbWVudHNGb3IocmVwbGFjZW1lbnROYW1lLCByZXBsYWNlbWVudCA9PiB7XG4gICAgICAgICAgICBsZXQgbmV3SWQgPSBgJHt0aGlzLnByb3BzLmlkfV8ke3JlcGxhY2VtZW50fWA7XG4gICAgICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtpZDogbmV3SWQsIGtleTogbmV3SWR9KVxuICAgICAgICAgICAgKTtcbiAgICAgICAgfSk7XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogSWYgdGhlcmUgaXMgYSBgcHJvcFR5cGVzYCBvYmplY3QgcHJlc2VudCBvblxuICAgICAqIHRoZSBjb25zdHJ1Y3RvciAoaWUgdGhlIGNvbXBvbmVudCBjbGFzcyksXG4gICAgICogdGhlbiBydW4gdGhlIFByb3BUeXBlcyB2YWxpZGF0b3Igb24gaXQuXG4gICAgICovXG4gICAgdmFsaWRhdGVQcm9wcygpe1xuICAgICAgICBpZih0aGlzLmNvbnN0cnVjdG9yLnByb3BUeXBlcyl7XG4gICAgICAgICAgICBQcm9wVHlwZXMudmFsaWRhdGUoXG4gICAgICAgICAgICAgICAgdGhpcy5jb25zdHJ1Y3Rvci5uYW1lLFxuICAgICAgICAgICAgICAgIHRoaXMucHJvcHMsXG4gICAgICAgICAgICAgICAgdGhpcy5jb25zdHJ1Y3Rvci5wcm9wVHlwZXNcbiAgICAgICAgICAgICk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBMb29rcyB1cCB0aGUgcGFzc2VkIGtleSBpbiBuYW1lZENoaWxkcmVuIGFuZFxuICAgICAqIGlmIGZvdW5kIHJlc3BvbmRzIHdpdGggdGhlIHJlc3VsdCBvZiBjYWxsaW5nXG4gICAgICogcmVuZGVyIG9uIHRoYXQgY2hpbGQgY29tcG9uZW50LiBSZXR1cm5zIG51bGxcbiAgICAgKiBvdGhlcndpc2UuXG4gICAgICovXG4gICAgcmVuZGVyQ2hpbGROYW1lZChrZXkpe1xuICAgICAgICBsZXQgZm91bmRDaGlsZCA9IHRoaXMucHJvcHMubmFtZWRDaGlsZHJlbltrZXldO1xuICAgICAgICBpZihmb3VuZENoaWxkKXtcbiAgICAgICAgICAgIHJldHVybiBmb3VuZENoaWxkLnJlbmRlcigpO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiBudWxsO1xuICAgIH1cblxuICAgIC8qKlxuICAgICAqIExvb2tzIHVwIHRoZSBwYXNzZWQga2V5IGluIG5hbWVkQ2hpbGRyZW5cbiAgICAgKiBhbmQgaWYgZm91bmQgLS0gYW5kIHRoZSB2YWx1ZSBpcyBhbiBBcnJheVxuICAgICAqIG9yIEFycmF5IG9mIEFycmF5cywgcmVzcG9uZHMgd2l0aCBhblxuICAgICAqIGlzb21vcnBoaWMgc3RydWN0dXJlIHRoYXQgaGFzIHRoZSByZW5kZXJlZFxuICAgICAqIHZhbHVlcyBvZiBlYWNoIGNvbXBvbmVudC5cbiAgICAgKi9cbiAgICByZW5kZXJDaGlsZHJlbk5hbWVkKGtleSl7XG4gICAgICAgIGxldCBmb3VuZENoaWxkcmVuID0gdGhpcy5wcm9wcy5uYW1lZENoaWxkcmVuW2tleV07XG4gICAgICAgIGlmKGZvdW5kQ2hpbGRyZW4pe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuX3JlY3Vyc2l2ZWx5TWFwTmFtZWRDaGlsZHJlbihmb3VuZENoaWxkcmVuLCBjaGlsZCA9PiB7XG4gICAgICAgICAgICAgICAgcmV0dXJuIGNoaWxkLnJlbmRlcigpO1xuICAgICAgICAgICAgfSk7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIFtdO1xuICAgIH1cblxuXG5cbiAgICAvKipcbiAgICAgKiBHZXR0ZXIgdGhhdCB3aWxsIHJlc3BvbmQgd2l0aCB0aGVcbiAgICAgKiBjb25zdHJ1Y3RvcidzIChha2EgdGhlICdjbGFzcycpIG5hbWVcbiAgICAgKi9cbiAgICBnZXQgbmFtZSgpe1xuICAgICAgICByZXR1cm4gdGhpcy5jb25zdHJ1Y3Rvci5uYW1lO1xuICAgIH1cblxuICAgIC8qKlxuICAgICAqIEdldHRlciB0aGF0IHdpbGwgcmVzcG9uZCB3aXRoIGFuXG4gICAgICogYXJyYXkgb2YgcmVuZGVyZWQgKGllIGNvbmZpZ3VyZWRcbiAgICAgKiBoeXBlcnNjcmlwdCkgb2JqZWN0cyB0aGF0IHJlcHJlc2VudFxuICAgICAqIGVhY2ggY2hpbGQuIE5vdGUgdGhhdCB3ZSB3aWxsIGNyZWF0ZSBrZXlzXG4gICAgICogZm9yIHRoZXNlIGJhc2VkIG9uIHRoZSBJRCBvZiB0aGlzIHBhcmVudFxuICAgICAqIGNvbXBvbmVudC5cbiAgICAgKi9cbiAgICBnZXQgcmVuZGVyZWRDaGlsZHJlbigpe1xuICAgICAgICBpZih0aGlzLnByb3BzLmNoaWxkcmVuLmxlbmd0aCA9PSAwKXtcbiAgICAgICAgICAgIHJldHVybiBbXTtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gdGhpcy5wcm9wcy5jaGlsZHJlbi5tYXAoY2hpbGRDb21wb25lbnQgPT4ge1xuICAgICAgICAgICAgbGV0IHJlbmRlcmVkQ2hpbGQgPSBjaGlsZENvbXBvbmVudC5yZW5kZXIoKTtcbiAgICAgICAgICAgIHJlbmRlcmVkQ2hpbGQucHJvcGVydGllcy5rZXkgPSBgJHt0aGlzLnByb3BzLmlkfS1jaGlsZC0ke2NoaWxkQ29tcG9uZW50LnByb3BzLmlkfWA7XG4gICAgICAgICAgICByZXR1cm4gcmVuZGVyZWRDaGlsZDtcbiAgICAgICAgfSk7XG4gICAgfVxuXG4gICAgLyoqIFB1YmxpYyBVdGlsIE1ldGhvZHMgKiovXG5cbiAgICAvKipcbiAgICAgKiBDYWxscyB0aGUgcHJvdmlkZWQgY2FsbGJhY2sgb24gZWFjaFxuICAgICAqIGFycmF5IGNoaWxkIGZvciB0aGlzIGNvbXBvbmVudCwgd2l0aFxuICAgICAqIHRoZSBjaGlsZCBhcyB0aGUgc29sZSBhcmcgdG8gdGhlXG4gICAgICogY2FsbGJhY2tcbiAgICAgKi9cbiAgICBjaGlsZHJlbkRvKGNhbGxiYWNrKXtcbiAgICAgICAgdGhpcy5wcm9wcy5jaGlsZHJlbi5mb3JFYWNoKGNoaWxkID0+IHtcbiAgICAgICAgICAgIGNhbGxiYWNrKGNoaWxkKTtcbiAgICAgICAgfSk7XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogQ2FsbHMgdGhlIHByb3ZpZGVkIGNhbGxiYWNrIG9uXG4gICAgICogZWFjaCBuYW1lZCBjaGlsZCB3aXRoIGtleSwgY2hpbGRcbiAgICAgKiBhcyB0aGUgdHdvIGFyZ3MgdG8gdGhlIGNhbGxiYWNrLlxuICAgICAqL1xuICAgIG5hbWVkQ2hpbGRyZW5EbyhjYWxsYmFjayl7XG4gICAgICAgIE9iamVjdC5rZXlzKHRoaXMucHJvcHMubmFtZWRDaGlsZHJlbikuZm9yRWFjaChrZXkgPT4ge1xuICAgICAgICAgICAgbGV0IGNoaWxkID0gdGhpcy5wcm9wcy5uYW1lZENoaWxkcmVuW2tleV07XG4gICAgICAgICAgICBjYWxsYmFjayhrZXksIGNoaWxkKTtcbiAgICAgICAgfSk7XG4gICAgfVxuXG4gICAgLyoqIFByaXZhdGUgVXRpbCBNZXRob2RzICoqL1xuXG4gICAgLyoqXG4gICAgICogU2V0cyB0aGUgcGFyZW50IGF0dHJpYnV0ZSBvZiBhbGwgaW5jb21pbmdcbiAgICAgKiBhcnJheSBhbmQvb3IgbmFtZWQgY2hpbGRyZW4gdG8gdGhpc1xuICAgICAqIGluc3RhbmNlLlxuICAgICAqL1xuICAgIF9zZXR1cENoaWxkUmVsYXRpb25zaGlwcygpe1xuICAgICAgICAvLyBOYW1lZCBjaGlsZHJlbiBmaXJzdFxuICAgICAgICBPYmplY3Qua2V5cyh0aGlzLnByb3BzLm5hbWVkQ2hpbGRyZW4pLmZvckVhY2goa2V5ID0+IHtcbiAgICAgICAgICAgIGxldCBjaGlsZCA9IHRoaXMucHJvcHMubmFtZWRDaGlsZHJlbltrZXldO1xuICAgICAgICAgICAgY2hpbGQucGFyZW50ID0gdGhpcztcbiAgICAgICAgfSk7XG5cbiAgICAgICAgLy8gTm93IGFycmF5IGNoaWxkcmVuXG4gICAgICAgIHRoaXMucHJvcHMuY2hpbGRyZW4uZm9yRWFjaChjaGlsZCA9PiB7XG4gICAgICAgICAgICBjaGlsZC5wYXJlbnQgPSB0aGlzO1xuICAgICAgICB9KTtcbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBVcGRhdGVzIHRoaXMgY29tcG9uZW50cyBwcm9wcyBvYmplY3RcbiAgICAgKiBiYXNlZCBvbiBhbiBpbmNvbWluZyBvYmplY3RcbiAgICAgKi9cbiAgICBfdXBkYXRlUHJvcHMoaW5jb21pbmdQcm9wcyl7XG4gICAgICAgIHRoaXMucHJvcHMgPSBpbmNvbWluZ1Byb3BzO1xuICAgICAgICB0aGlzLnByb3BzLmNoaWxkcmVuID0gaW5jb21pbmdQcm9wcy5jaGlsZHJlbiB8fCBbXTtcbiAgICAgICAgdGhpcy5wcm9wcy5uYW1lZENoaWxkcmVuID0gaW5jb21pbmdQcm9wcy5uYW1lZENoaWxkcmVuIHx8IHt9O1xuICAgICAgICB0aGlzLl9zZXR1cENoaWxkUmVsYXRpb25zaGlwcygpO1xuICAgIH1cblxuICAgIC8qKlxuICAgICAqIFJlY3Vyc2l2ZWx5IG1hcHMgYSBvbmUgb3IgbXVsdGlkaW1lbnNpb25hbFxuICAgICAqIG5hbWVkIGNoaWxkcmVuIHZhbHVlIHdpdGggdGhlIGdpdmVuIG1hcHBpbmdcbiAgICAgKiBmdW5jdGlvbi5cbiAgICAgKi9cbiAgICBfcmVjdXJzaXZlbHlNYXBOYW1lZENoaWxkcmVuKGNvbGxlY3Rpb24sIGNhbGxiYWNrKXtcbiAgICAgICAgcmV0dXJuIGNvbGxlY3Rpb24ubWFwKGl0ZW0gPT4ge1xuICAgICAgICAgICAgaWYoQXJyYXkuaXNBcnJheShpdGVtKSl7XG4gICAgICAgICAgICAgICAgcmV0dXJuIHRoaXMuX3JlY3Vyc2l2ZWx5TWFwTmFtZWRDaGlsZHJlbihpdGVtLCBjYWxsYmFjayk7XG4gICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgIHJldHVybiBjYWxsYmFjayhpdGVtKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfSk7XG4gICAgfVxufTtcblxuZXhwb3J0IHtDb21wb25lbnQsIENvbXBvbmVudCBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogQ29udGFpbmVyIENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBUaGlzIGNvbXBvbmVudCBoYXMgYSBzaW5nbGUgcmVndWxhclxuICogcmVwbGFjZW1lbnQ6XG4gKiAqIGBjaGlsZGBcbiAqL1xuXG4vKipcbiAqIEFib3V0IE5hbWVkIENoaWxkcmVuXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogYGNoaWxkYCAoc2luZ2xlKSAtIFRoZSBDZWxsIHRoYXQgdGhpcyBjb21wb25lbnQgY29udGFpbnNcbiAqL1xuY2xhc3MgQ29udGFpbmVyIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICAvLyBCaW5kIGNvbXBvbmVudCBtZXRob2RzXG4gICAgICAgIHRoaXMubWFrZUNoaWxkID0gdGhpcy5tYWtlQ2hpbGQuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgbGV0IGNoaWxkID0gdGhpcy5tYWtlQ2hpbGQoKTtcbiAgICAgICAgbGV0IHN0eWxlID0gXCJcIjtcbiAgICAgICAgaWYoIWNoaWxkKXtcbiAgICAgICAgICAgIHN0eWxlID0gXCJkaXNwbGF5Om5vbmU7XCI7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJDb250YWluZXJcIixcbiAgICAgICAgICAgICAgICBjbGFzczogXCJjZWxsXCIsXG4gICAgICAgICAgICAgICAgc3R5bGU6IHN0eWxlXG4gICAgICAgICAgICB9LCBbY2hpbGRdKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIG1ha2VDaGlsZCgpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdjaGlsZCcpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnY2hpbGQnKTtcbiAgICAgICAgfVxuICAgIH1cbn1cblxuZXhwb3J0IHtDb250YWluZXIsIENvbnRhaW5lciBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogQ29udGV4dHVhbERpc3BsYXkgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyBhIHNpbmdsZVxuICogcmVndWxhciByZXBsYWNlbWVudDpcbiAqICogYGNoaWxkYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgY2hpbGRgIChzaW5nbGUpIC0gQSBjaGlsZCBjZWxsIHRvIGRpc3BsYXkgaW4gYSBjb250ZXh0XG4gKi9cbmNsYXNzIENvbnRleHR1YWxEaXNwbGF5IGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICAvLyBCaW5kIGNvbXBvbmVudCBtZXRob2RzXG4gICAgICAgIHRoaXMubWFrZUNoaWxkID0gdGhpcy5tYWtlQ2hpbGQuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIGgoJ2RpdicsXG4gICAgICAgICAgICB7XG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbCBjb250ZXh0dWFsRGlzcGxheVwiLFxuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkNvbnRleHR1YWxEaXNwbGF5XCJcbiAgICAgICAgICAgIH0sIFt0aGlzLm1ha2VDaGlsZCgpXVxuICAgICAgICApO1xuICAgIH1cblxuICAgIG1ha2VDaGlsZCgpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdjaGlsZCcpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnY2hpbGQnKTtcbiAgICAgICAgfVxuICAgIH1cbn1cblxuZXhwb3J0IHtDb250ZXh0dWFsRGlzcGxheSwgQ29udGV4dHVhbERpc3BsYXkgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIERyb3Bkb3duIENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBUaGlzIGNvbXBvbmVudCBoYXMgb25lIHJlZ3VsYXJcbiAqIHJlcGxhY2VtZW50OlxuICogKiBgdGl0bGVgXG4gKiBUaGlzIGNvbXBvbmVudCBoYXMgb25lXG4gKiBlbnVtZXJhdGVkIHJlcGxhY2VtZW50OlxuICogKiBgY2hpbGRgXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBDaGlsZHJlblxuICogLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIGB0aXRsZWAgKHNpbmdsZSkgLSBBIENlbGwgdGhhdCB3aWxsIGNvbXByaXNlIHRoZSB0aXRsZSBvZlxuICogICAgICB0aGUgZHJvcGRvd25cbiAqIGBkcm9wZG93bkl0ZW1zYCAoYXJyYXkpIC0gQW4gYXJyYXkgb2YgY2VsbHMgdGhhdCBhcmVcbiAqICAgICAgdGhlIGl0ZW1zIGluIHRoZSBkcm9wZG93blxuICovXG5jbGFzcyBEcm9wZG93biBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb250ZXh0IHRvIG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlVGl0bGUgPSB0aGlzLm1ha2VUaXRsZS5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLm1ha2VJdGVtcyA9IHRoaXMubWFrZUl0ZW1zLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiRHJvcGRvd25cIixcbiAgICAgICAgICAgICAgICBjbGFzczogXCJidG4tZ3JvdXBcIlxuICAgICAgICAgICAgfSwgW1xuICAgICAgICAgICAgICAgIGgoJ2EnLCB7Y2xhc3M6IFwiYnRuIGJ0bi14cyBidG4tb3V0bGluZS1zZWNvbmRhcnlcIn0sIFtcbiAgICAgICAgICAgICAgICAgICAgdGhpcy5tYWtlVGl0bGUoKVxuICAgICAgICAgICAgICAgIF0pLFxuICAgICAgICAgICAgICAgIGgoJ2J1dHRvbicsIHtcbiAgICAgICAgICAgICAgICAgICAgY2xhc3M6IFwiYnRuIGJ0bi14cyBidG4tb3V0bGluZS1zZWNvbmRhcnkgZHJvcGRvd24tdG9nZ2xlIGRyb3Bkb3duLXRvZ2dsZS1zcGxpdFwiLFxuICAgICAgICAgICAgICAgICAgICB0eXBlOiBcImJ1dHRvblwiLFxuICAgICAgICAgICAgICAgICAgICBpZDogYCR7dGhpcy5wcm9wcy5leHRyYURhdGEudGFyZ2V0SWRlbnRpdHl9LWRyb3Bkb3duTWVudUJ1dHRvbmAsXG4gICAgICAgICAgICAgICAgICAgIFwiZGF0YS10b2dnbGVcIjogXCJkcm9wZG93blwiXG4gICAgICAgICAgICAgICAgfSksXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge2NsYXNzOiBcImRyb3Bkb3duLW1lbnVcIn0sIHRoaXMubWFrZUl0ZW1zKCkpXG4gICAgICAgICAgICBdKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIG1ha2VUaXRsZSgpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCd0aXRsZScpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgndGl0bGUnKTtcbiAgICAgICAgfVxuICAgIH1cblxuICAgIG1ha2VJdGVtcygpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgLy8gRm9yIHNvbWUgcmVhc29uLCBkdWUgYWdhaW4gdG8gdGhlIENlbGwgaW1wbGVtZW50YXRpb24sXG4gICAgICAgICAgICAvLyBzb21ldGltZXMgdGhlcmUgYXJlIG5vdCB0aGVzZSBjaGlsZCByZXBsYWNlbWVudHMuXG4gICAgICAgICAgICBpZighdGhpcy5yZXBsYWNlbWVudHMuaGFzUmVwbGFjZW1lbnQoJ2NoaWxkJykpe1xuICAgICAgICAgICAgICAgIHJldHVybiBbXTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudHNGb3IoJ2NoaWxkJykubWFwKChlbGVtZW50LCBpZHgpID0+IHtcbiAgICAgICAgICAgICAgICByZXR1cm4gbmV3IERyb3Bkb3duSXRlbSh7XG4gICAgICAgICAgICAgICAgICAgIGlkOiBgJHt0aGlzLnByb3BzLmlkfS1pdGVtLSR7aWR4fWAsXG4gICAgICAgICAgICAgICAgICAgIGluZGV4OiBpZHgsXG4gICAgICAgICAgICAgICAgICAgIGNoaWxkU3Vic3RpdHV0ZTogZWxlbWVudCxcbiAgICAgICAgICAgICAgICAgICAgdGFyZ2V0SWRlbnRpdHk6IHRoaXMucHJvcHMuZXh0cmFEYXRhLnRhcmdldElkZW50aXR5LFxuICAgICAgICAgICAgICAgICAgICBkcm9wZG93bkl0ZW1JbmZvOiB0aGlzLnByb3BzLmV4dHJhRGF0YS5kcm9wZG93bkl0ZW1JbmZvXG4gICAgICAgICAgICAgICAgfSkucmVuZGVyKCk7XG4gICAgICAgICAgICB9KTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIGlmKHRoaXMucHJvcHMubmFtZWRDaGlsZHJlbi5kcm9wZG93bkl0ZW1zKXtcbiAgICAgICAgICAgICAgICByZXR1cm4gdGhpcy5wcm9wcy5uYW1lZENoaWxkcmVuLmRyb3Bkb3duSXRlbXMubWFwKChpdGVtQ29tcG9uZW50LCBpZHgpID0+IHtcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIG5ldyBEcm9wZG93SXRlbSh7XG4gICAgICAgICAgICAgICAgICAgICAgICBpZDogYCR7dGhpcy5wcm9wZC5pZH0taXRlbS0ke2lkeH1gLFxuICAgICAgICAgICAgICAgICAgICAgICAgaW5kZXg6IGlkeCxcbiAgICAgICAgICAgICAgICAgICAgICAgIGNoaWxkU3Vic3RpdHV0ZTogaXRlbUNvbXBvbmVudC5yZW5kZXIoKSxcbiAgICAgICAgICAgICAgICAgICAgICAgIHRhcmdldElkZW50aXR5OiB0aGlzLnByb3BzLmV4dHJhRGF0YS50YXJnZXRJZGVudGl0eSxcbiAgICAgICAgICAgICAgICAgICAgICAgIGRyb3Bkb3duSXRlbUluZm86IHRoaXMucHJvcHMuZXh0cmFEYXRhLmRyb3Bkb3duSXRlbUluZm9cbiAgICAgICAgICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgIHJldHVybiBbXTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgIH1cbn1cblxuXG4vKipcbiAqIEEgcHJpdmF0ZSBzdWJjb21wb25lbnQgZm9yIGVhY2hcbiAqIERyb3Bkb3duIG1lbnUgaXRlbS4gV2UgbmVlZCB0aGlzXG4gKiBiZWNhdXNlIG9mIGhvdyBjYWxsYmFja3MgYXJlIGhhbmRsZWRcbiAqIGFuZCBiZWNhdXNlIHRoZSBDZWxscyB2ZXJzaW9uIGRvZXNuJ3RcbiAqIGFscmVhZHkgaW1wbGVtZW50IHRoaXMga2luZCBhcyBhIHNlcGFyYXRlXG4gKiBlbnRpdHkuXG4gKi9cbmNsYXNzIERyb3Bkb3duSXRlbSBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb250ZXh0IHRvIG1ldGhvZHNcbiAgICAgICAgdGhpcy5jbGlja0hhbmRsZXIgPSB0aGlzLmNsaWNrSGFuZGxlci5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnYScsIHtcbiAgICAgICAgICAgICAgICBjbGFzczogXCJzdWJjZWxsIGNlbGwtZHJvcGRvd24taXRlbSBkcm9wZG93bi1pdGVtXCIsXG4gICAgICAgICAgICAgICAga2V5OiB0aGlzLnByb3BzLmluZGV4LFxuICAgICAgICAgICAgICAgIG9uY2xpY2s6IHRoaXMuY2xpY2tIYW5kbGVyXG4gICAgICAgICAgICB9LCBbdGhpcy5wcm9wcy5jaGlsZFN1YnN0aXR1dGVdKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIGNsaWNrSGFuZGxlcihldmVudCl7XG4gICAgICAgIC8vIFRoaXMgaXMgc3VwZXIgaGFja3kgYmVjYXVzZSBvZiB0aGVcbiAgICAgICAgLy8gY3VycmVudCBDZWxsIGltcGxlbWVudGF0aW9uLlxuICAgICAgICAvLyBUaGlzIHdob2xlIGNvbXBvbmVudCBzdHJ1Y3R1cmUgc2hvdWxkIGJlIGhlYXZpbHkgcmVmYWN0b3JlZFxuICAgICAgICAvLyBvbmNlIHRoZSBDZWxscyBzaWRlIG9mIHRoaW5ncyBzdGFydHMgdG8gY2hhbmdlLlxuICAgICAgICBsZXQgd2hhdFRvRG8gPSB0aGlzLnByb3BzLmRyb3Bkb3duSXRlbUluZm9bdGhpcy5wcm9wcy5pbmRleC50b1N0cmluZygpXTtcbiAgICAgICAgaWYod2hhdFRvRG8gPT0gJ2NhbGxiYWNrJyl7XG4gICAgICAgICAgICBsZXQgcmVzcG9uc2VEYXRhID0ge1xuICAgICAgICAgICAgICAgIGV2ZW50OiBcIm1lbnVcIixcbiAgICAgICAgICAgICAgICBpeDogdGhpcy5wcm9wcy5pbmRleCxcbiAgICAgICAgICAgICAgICB0YXJnZXRfY2VsbDogdGhpcy5wcm9wcy50YXJnZXRJZGVudGl0eVxuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIGNlbGxTb2NrZXQuc2VuZFN0cmluZyhKU09OLnN0cmluZ2lmeShyZXNwb25zZURhdGEpKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHdpbmRvdy5sb2NhdGlvbi5ocmVmID0gd2hhdFRvRG87XG4gICAgICAgIH1cbiAgICB9XG59XG5cbmV4cG9ydCB7RHJvcGRvd24sIERyb3Bkb3duIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBFeHBhbmRzIENlbGwgQ29tcG9uZW50XG4gKi9cblxuLyoqIFRPRE8vTk9URTogSXQgYXBwZWFycyB0aGF0IHRoZSBvcGVuL2Nsb3NlZFxuICAgIFN0YXRlIGZvciB0aGlzIGNvbXBvbmVudCBjb3VsZCBzaW1wbHkgYmUgcGFzc2VkXG4gICAgd2l0aCB0aGUgQ2VsbCBkYXRhLCBhbG9uZyB3aXRoIHdoYXQgdG8gZGlzcGxheVxuICAgIGluIGVpdGhlciBjYXNlLiBUaGlzIHdvdWxkIGJlIGhvdyBpdCBpcyBub3JtYWxseVxuICAgIGRvbmUgaW4gbGFyZ2Ugd2ViIGFwcGxpY2F0aW9ucy5cbiAgICBDb25zaWRlciByZWZhY3RvcmluZyBib3RoIGhlcmUgYW5kIG9uIHRoZSBDZWxsc1xuICAgIHNpZGVcbioqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIHR3b1xuICogcmVndWxhciByZXBsYWNlbWVudHM6XG4gKiAqIGBpY29uYFxuICogKiBgY2hpbGRgXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBDaGlsZHJlblxuICogLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIGBjb250ZW50YCAoc2luZ2xlKSAtIFRoZSBvcGVuIG9yIGNsb3NlZCBjZWxsLCBkZXBlbmRpbmcgb24gc291cmNlXG4gKiAgICAgb3BlbiBzdGF0ZVxuICogYGljb25gIChzaW5nbGUpIC0gVGhlIENlbGwgb2YgdGhlIGljb24gdG8gZGlzcGxheSwgYWxzbyBkZXBlbmRpbmdcbiAqICAgICBvbiBjbG9zZWQgb3Igb3BlbiBzdGF0ZVxuICovXG5jbGFzcyBFeHBhbmRzIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICAvLyBCaW5kIGNvbnRleHQgdG8gbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VJY29uID0gdGhpcy5tYWtlSWNvbi5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLm1ha2VDb250ZW50ID0gdGhpcy5tYWtlQ29udGVudC5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLl9nZXRFdmVudHMgPSB0aGlzLl9nZXRFdmVudC5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4oXG4gICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiRXhwYW5kc1wiLFxuICAgICAgICAgICAgICAgIHN0eWxlOiB0aGlzLnByb3BzLmV4dHJhRGF0YS5kaXZTdHlsZVxuICAgICAgICAgICAgfSxcbiAgICAgICAgICAgICAgICBbXG4gICAgICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIHN0eWxlOiAnZGlzcGxheTppbmxpbmUtYmxvY2s7dmVydGljYWwtYWxpZ246dG9wJyxcbiAgICAgICAgICAgICAgICAgICAgICAgIG9uY2xpY2s6IHRoaXMuX2dldEV2ZW50KCdvbmNsaWNrJylcbiAgICAgICAgICAgICAgICAgICAgfSxcbiAgICAgICAgICAgICAgICAgICAgICBbdGhpcy5tYWtlSWNvbigpXSksXG4gICAgICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtzdHlsZTonZGlzcGxheTppbmxpbmUtYmxvY2snfSxcbiAgICAgICAgICAgICAgICAgICAgICBbdGhpcy5tYWtlQ29udGVudCgpXSksXG4gICAgICAgICAgICAgICAgXVxuICAgICAgICAgICAgKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIG1ha2VDb250ZW50KCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ2NoaWxkJyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCdjb250ZW50Jyk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBtYWtlSWNvbigpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdpY29uJyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCdpY29uJyk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBfZ2V0RXZlbnQoZXZlbnROYW1lKSB7XG4gICAgICAgIHJldHVybiB0aGlzLnByb3BzLmV4dHJhRGF0YS5ldmVudHNbZXZlbnROYW1lXTtcbiAgICB9XG59XG5cbmV4cG9ydCB7RXhwYW5kcywgRXhwYW5kcyBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogR3JpZCBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIDMgZW51bWVyYWJsZVxuICogcmVwbGFjZW1lbnRzOlxuICogKiBgaGVhZGVyYFxuICogKiBgcm93bGFiZWxgXG4gKiAqIGBjaGlsZGBcbiAqXG4gKiBOT1RFOiBDaGlsZCBpcyBhIDItZGltZW5zaW9uYWxcbiAqIGVudW1lcmF0ZWQgcmVwbGFjZW1lbnQhXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBDaGlsZHJlblxuICogLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIGBoZWFkZXJzYCAoYXJyYXkpIC0gQW4gYXJyYXkgb2YgdGFibGUgaGVhZGVyIGNlbGxzXG4gKiBgcm93TGFiZWxzYCAoYXJyYXkpIC0gQW4gYXJyYXkgb2Ygcm93IGxhYmVsIGNlbGxzXG4gKiBgZGF0YUNlbGxzYCAoYXJyYXktb2YtYXJyYXkpIC0gQSAyLWRpbWVuc2lvbmFsIGFycmF5XG4gKiAgICAgb2YgY2VsbHMgdGhhdCBzZXJ2ZSBhcyB0YWJsZSBkYXRhLCB3aGVyZSByb3dzXG4gKiAgICAgYXJlIHRoZSBvdXRlciBhcnJheSBhbmQgY29sdW1ucyBhcmUgdGhlIGlubmVyXG4gKiAgICAgYXJyYXkuXG4gKi9cbmNsYXNzIEdyaWQgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29udGV4dCB0byBtZXRob2RzXG4gICAgICAgIHRoaXMubWFrZUhlYWRlcnMgPSB0aGlzLm1ha2VIZWFkZXJzLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMubWFrZVJvd3MgPSB0aGlzLm1ha2VSb3dzLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuX21ha2VSZXBsYWNlbWVudEhlYWRlckVsZW1lbnRzID0gdGhpcy5fbWFrZVJlcGxhY2VtZW50SGVhZGVyRWxlbWVudHMuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5fbWFrZVJlcGxhY2VtZW50Um93RWxlbWVudHMgPSB0aGlzLl9tYWtlUmVwbGFjZW1lbnRSb3dFbGVtZW50cy5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICBsZXQgdG9wVGFibGVIZWFkZXIgPSBudWxsO1xuICAgICAgICBpZih0aGlzLnByb3BzLmV4dHJhRGF0YS5oYXNUb3BIZWFkZXIpe1xuICAgICAgICAgICAgdG9wVGFibGVIZWFkZXIgPSBoKCd0aCcpO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCd0YWJsZScsIHtcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJHcmlkXCIsXG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbCB0YWJsZS1oc2Nyb2xsIHRhYmxlLXNtIHRhYmxlLXN0cmlwZWRcIlxuICAgICAgICAgICAgfSwgW1xuICAgICAgICAgICAgICAgIGgoJ3RoZWFkJywge30sIFtcbiAgICAgICAgICAgICAgICAgICAgaCgndHInLCB7fSwgW3RvcFRhYmxlSGVhZGVyLCAuLi50aGlzLm1ha2VIZWFkZXJzKCldKVxuICAgICAgICAgICAgICAgIF0pLFxuICAgICAgICAgICAgICAgIGgoJ3Rib2R5Jywge30sIHRoaXMubWFrZVJvd3MoKSlcbiAgICAgICAgICAgIF0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgbWFrZUhlYWRlcnMoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLl9tYWtlUmVwbGFjZW1lbnRIZWFkZXJFbGVtZW50cygpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGRyZW5OYW1lZCgnaGVhZGVycycpLm1hcCgoaGVhZGVyRWwsIGNvbElkeCkgPT4ge1xuICAgICAgICAgICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICAgICAgICAgIGgoJ3RoJywge2tleTogYCR7dGhpcy5wcm9wcy5pZH0tZ3JpZC10aC0ke2NvbElkeH1gfSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgaGVhZGVyRWxcbiAgICAgICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgICAgICApO1xuICAgICAgICAgICAgfSk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBtYWtlUm93cygpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuX21ha2VSZXBsYWNlbWVudFJvd0VsZW1lbnRzKCk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZHJlbk5hbWVkKCdkYXRhQ2VsbHMnKS5tYXAoKGRhdGFSb3csIHJvd0lkeCkgPT4ge1xuICAgICAgICAgICAgICAgIGxldCBjb2x1bW5zID0gZGF0YVJvdy5tYXAoKGNvbHVtbiwgY29sSWR4KSA9PiB7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybihcbiAgICAgICAgICAgICAgICAgICAgICAgIGgoJ3RkJywge2tleTogYCR7dGhpcy5wcm9wcy5pZH0tZ3JpZC1jb2wtJHtyb3dJZHh9LSR7Y29sSWR4fWB9LCBbXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgY29sdW1uXG4gICAgICAgICAgICAgICAgICAgICAgICBdKVxuICAgICAgICAgICAgICAgICAgICApO1xuICAgICAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgICAgIGxldCByb3dMYWJlbEVsID0gbnVsbDtcbiAgICAgICAgICAgICAgICBpZih0aGlzLnByb3BzLm5hbWVkQ2hpbGRyZW4ucm93TGFiZWxzICYmIHRoaXMucHJvcHMubmFtZWRDaGlsZHJlbi5yb3dMYWJlbHMubGVuZ3RoID4gMCl7XG4gICAgICAgICAgICAgICAgICAgIHJvd0xhYmVsRWwgPSBoKCd0aCcsIHtrZXk6IGAke3RoaXMucHJvcHMuaWR9LWdyaWQtY29sLSR7cm93SWR4fS0ke2NvbElkeH1gfSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy5wcm9wcy5uYW1lZENoaWxkcmVuLnJvd0xhYmVsc1tyb3dJZHhdLnJlbmRlcigpXG4gICAgICAgICAgICAgICAgICAgIF0pO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgICAgICAgICBoKCd0cicsIHtrZXk6IGAke3RoaXMucHJvcHMuaWR9LWdyaWQtcm93LSR7cm93SWR4fWB9LCBbXG4gICAgICAgICAgICAgICAgICAgICAgICByb3dMYWJlbEVsLFxuICAgICAgICAgICAgICAgICAgICAgICAgLi4uY29sdW1uc1xuICAgICAgICAgICAgICAgICAgICBdKVxuICAgICAgICAgICAgICAgICk7XG4gICAgICAgICAgICB9KTtcbiAgICAgICAgfVxuICAgIH1cblxuICAgIF9tYWtlUmVwbGFjZW1lbnRSb3dFbGVtZW50cygpe1xuICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRzRm9yKCdjaGlsZCcpLm1hcCgocm93LCByb3dJZHgpID0+IHtcbiAgICAgICAgICAgIGxldCBjb2x1bW5zID0gcm93Lm1hcCgoY29sdW1uLCBjb2xJZHgpID0+IHtcbiAgICAgICAgICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgICAgICAgICBoKCd0ZCcsIHtrZXk6IGAke3RoaXMucHJvcHMuaWR9LWdyaWQtY29sLSR7cm93SWR4fS0ke2NvbElkeH1gfSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgY29sdW1uXG4gICAgICAgICAgICAgICAgICAgIF0pXG4gICAgICAgICAgICAgICAgKTtcbiAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgbGV0IHJvd0xhYmVsRWwgPSBudWxsO1xuICAgICAgICAgICAgaWYodGhpcy5yZXBsYWNlbWVudHMuaGFzUmVwbGFjZW1lbnQoJ3Jvd2xhYmVsJykpe1xuICAgICAgICAgICAgICAgIHJvd0xhYmVsRWwgPSBoKCd0aCcsIHtrZXk6IGAke3RoaXMucHJvcHMuaWR9LWdyaWQtcm93bGJsLSR7cm93SWR4fWB9LCBbXG4gICAgICAgICAgICAgICAgICAgIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50c0Zvcigncm93bGFiZWwnKVtyb3dJZHhdXG4gICAgICAgICAgICAgICAgXSk7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgICAgIGgoJ3RyJywge2tleTogYCR7dGhpcy5wcm9wcy5pZH0tZ3JpZC1yb3ctJHtyb3dJZHh9YH0sIFtcbiAgICAgICAgICAgICAgICAgICAgcm93TGFiZWxFbCxcbiAgICAgICAgICAgICAgICAgICAgLi4uY29sdW1uc1xuICAgICAgICAgICAgICAgIF0pXG4gICAgICAgICAgICApO1xuICAgICAgICB9KTtcbiAgICB9XG5cbiAgICBfbWFrZVJlcGxhY2VtZW50SGVhZGVyRWxlbWVudHMoKXtcbiAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50c0ZvcignaGVhZGVyJykubWFwKChoZWFkZXJFbCwgY29sSWR4KSA9PiB7XG4gICAgICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgICAgIGgoJ3RoJywge2tleTogYCR7dGhpcy5wcm9wcy5pZH0tZ3JpZC10aC0ke2NvbElkeH1gfSwgW1xuICAgICAgICAgICAgICAgICAgICBoZWFkZXJFbFxuICAgICAgICAgICAgICAgIF0pXG4gICAgICAgICAgICApO1xuICAgICAgICB9KTtcbiAgICB9XG59XG5cbmV4cG9ydFxue0dyaWQsIEdyaWQgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIEhlYWRlckJhciBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIHRocmVlIHNlcGFyYXRlXG4gKiBlbnVtZXJhdGVkIHJlcGxhY2VtZW50czpcbiAqICogYGxlZnRgXG4gKiAqIGByaWdodGBcbiAqICogYGNlbnRlcmBcbiAqL1xuXG4vKipcbiAqIEFib3V0IE5hbWVkIENoaWxkcmVuXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogYGxlZnRJdGVtc2AgKGFycmF5KSAtIFRoZSBpdGVtcyB0aGF0IHdpbGwgYmUgb24gdGhlIGxlZnRcbiAqIGBjZW50ZXJJdGVtc2AgKGFycmF5KSAtIFRoZSBpdGVtcyB0aGF0IHdpbGwgYmUgaW4gdGhlIGNlbnRlclxuICogYHJpZ2h0SXRlbXNgIChhcnJheSkgLSBUaGUgaXRlbXMgdGhhdCB3aWxsIGJlIG9uIHRoZSByaWdodFxuICovXG5jbGFzcyBIZWFkZXJCYXIgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29udGV4dCB0byBtZXRob2RzXG4gICAgICAgIHRoaXMubWFrZUVsZW1lbnRzID0gdGhpcy5tYWtlRWxlbWVudHMuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5tYWtlUmlnaHQgPSB0aGlzLm1ha2VSaWdodC5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLm1ha2VMZWZ0ID0gdGhpcy5tYWtlTGVmdC5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLm1ha2VDZW50ZXIgPSB0aGlzLm1ha2VDZW50ZXIuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBjbGFzczogXCJjZWxsIHAtMiBiZy1saWdodCBmbGV4LWNvbnRhaW5lclwiLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkhlYWRlckJhclwiLFxuICAgICAgICAgICAgICAgIHN0eWxlOiAnZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmJhc2VsaW5lOydcbiAgICAgICAgICAgIH0sIFtcbiAgICAgICAgICAgICAgICB0aGlzLm1ha2VMZWZ0KCksXG4gICAgICAgICAgICAgICAgdGhpcy5tYWtlQ2VudGVyKCksXG4gICAgICAgICAgICAgICAgdGhpcy5tYWtlUmlnaHQoKVxuICAgICAgICAgICAgXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlTGVmdCgpe1xuICAgICAgICBsZXQgaW5uZXJFbGVtZW50cyA9IFtdO1xuICAgICAgICBpZih0aGlzLnJlcGxhY2VtZW50cy5oYXNSZXBsYWNlbWVudCgnbGVmdCcpIHx8IHRoaXMucHJvcHMubmFtZWRDaGlsZHJlbi5sZWZ0SXRlbXMpe1xuICAgICAgICAgICAgaW5uZXJFbGVtZW50cyA9IHRoaXMubWFrZUVsZW1lbnRzKCdsZWZ0Jyk7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ2RpdicsIHtjbGFzczogXCJmbGV4LWl0ZW1cIiwgc3R5bGU6IFwiZmxleC1ncm93OjA7XCJ9LCBbXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgICAgICBjbGFzczogXCJmbGV4LWNvbnRhaW5lclwiLFxuICAgICAgICAgICAgICAgICAgICBzdHlsZTogJ2Rpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO2FsaWduLWl0ZW1zOmJhc2VsaW5lOydcbiAgICAgICAgICAgICAgICB9LCBpbm5lckVsZW1lbnRzKVxuICAgICAgICAgICAgXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlQ2VudGVyKCl7XG4gICAgICAgIGxldCBpbm5lckVsZW1lbnRzID0gW107XG4gICAgICAgIGlmKHRoaXMucmVwbGFjZW1lbnRzLmhhc1JlcGxhY2VtZW50KCdjZW50ZXInKSB8fCB0aGlzLnByb3BzLm5hbWVkQ2hpbGRyZW4uY2VudGVySXRlbXMpe1xuICAgICAgICAgICAgaW5uZXJFbGVtZW50cyA9IHRoaXMubWFrZUVsZW1lbnRzKCdjZW50ZXInKTtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnZGl2Jywge2NsYXNzOiBcImZsZXgtaXRlbVwiLCBzdHlsZTogXCJmbGV4LWdyb3c6MTtcIn0sIFtcbiAgICAgICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgICAgIGNsYXNzOiBcImZsZXgtY29udGFpbmVyXCIsXG4gICAgICAgICAgICAgICAgICAgIHN0eWxlOiAnZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpjZW50ZXI7YWxpZ24taXRlbXM6YmFzZWxpbmU7J1xuICAgICAgICAgICAgICAgIH0sIGlubmVyRWxlbWVudHMpXG4gICAgICAgICAgICBdKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIG1ha2VSaWdodCgpe1xuICAgICAgICBsZXQgaW5uZXJFbGVtZW50cyA9IFtdO1xuICAgICAgICBpZih0aGlzLnJlcGxhY2VtZW50cy5oYXNSZXBsYWNlbWVudCgncmlnaHQnKSB8fCB0aGlzLnByb3BzLm5hbWVkQ2hpbGRyZW4ucmlnaHRJdGVtcyl7XG4gICAgICAgICAgICBpbm5lckVsZW1lbnRzID0gdGhpcy5tYWtlRWxlbWVudHMoJ3JpZ2h0Jyk7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ2RpdicsIHtjbGFzczogXCJmbGV4LWl0ZW1cIiwgc3R5bGU6IFwiZmxleC1ncm93OjA7XCJ9LCBbXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgICAgICBjbGFzczogXCJmbGV4LWNvbnRhaW5lclwiLFxuICAgICAgICAgICAgICAgICAgICBzdHlsZTogJ2Rpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO2FsaWduLWl0ZW1zOmJhc2VsaW5lOydcbiAgICAgICAgICAgICAgICB9LCBpbm5lckVsZW1lbnRzKVxuICAgICAgICAgICAgXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlRWxlbWVudHMocG9zaXRpb24pe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50c0Zvcihwb3NpdGlvbikubWFwKGVsZW1lbnQgPT4ge1xuICAgICAgICAgICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICAgICAgICAgIGgoJ3NwYW4nLCB7Y2xhc3M6IFwiZmxleC1pdGVtIHB4LTNcIn0sIFtlbGVtZW50XSlcbiAgICAgICAgICAgICAgICApO1xuICAgICAgICAgICAgfSk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZHJlbk5hbWVkKGAke3Bvc2l0aW9ufUl0ZW1zYCkubWFwKGVsZW1lbnQgPT4ge1xuICAgICAgICAgICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICAgICAgICAgIGgoJ3NwYW4nLCB7Y2xhc3M6IFwiZmxleC1pdGVtIHB4LTNcIn0sIFtlbGVtZW50XSlcbiAgICAgICAgICAgICAgICApO1xuICAgICAgICAgICAgfSk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cbmV4cG9ydCB7SGVhZGVyQmFyLCBIZWFkZXJCYXIgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIExhcmdlUGVuZGluZ0Rvd25sb2FkRGlzcGxheSBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuY2xhc3MgTGFyZ2VQZW5kaW5nRG93bmxvYWREaXNwbGF5IGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICBpZDogJ29iamVjdF9kYXRhYmFzZV9sYXJnZV9wZW5kaW5nX2Rvd25sb2FkX3RleHQnLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkxhcmdlUGVuZGluZ0Rvd25sb2FkRGlzcGxheVwiLFxuICAgICAgICAgICAgICAgIGNsYXNzOiBcImNlbGxcIlxuICAgICAgICAgICAgfSlcbiAgICAgICAgKTtcbiAgICB9XG59XG5cbmV4cG9ydCB7TGFyZ2VQZW5kaW5nRG93bmxvYWREaXNwbGF5LCBMYXJnZVBlbmRpbmdEb3dubG9hZERpc3BsYXkgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIExvYWRDb250ZW50c0Zyb21VcmwgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbmNsYXNzIExvYWRDb250ZW50c0Zyb21VcmwgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4oXG4gICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiTG9hZENvbnRlbnRzRnJvbVVybFwiLFxuICAgICAgICAgICAgfSwgW2goJ2RpdicsIHtpZDogdGhpcy5wcm9wcy5leHRyYURhdGFbJ2xvYWRUYXJnZXRJZCddfSwgW10pXVxuICAgICAgICAgICAgKVxuICAgICAgICApO1xuICAgIH1cblxufVxuXG5leHBvcnQge0xvYWRDb250ZW50c0Zyb21VcmwsIExvYWRDb250ZW50c0Zyb21VcmwgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIE1haW4gQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIGEgb25lXG4gKiByZWd1bGFyLWtpbmQgcmVwbGFjZW1lbnQ6XG4gKiAqIGBjaGlsZGBcbiAqL1xuXG4vKipcbiAqIEFib3V0IE5hbWVkIENoaWxkcmVuXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogYGNoaWxkYCAoc2luZ2xlKSAtIFRoZSBjaGlsZCBjZWxsIHRoYXQgaXMgd3JhcHBlZFxuICovXG5jbGFzcyBNYWluIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICAvLyBCaW5kIGNvbXBvbmVudCBtZXRob2RzXG4gICAgICAgIHRoaXMubWFrZUNoaWxkID0gdGhpcy5tYWtlQ2hpbGQuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ21haW4nLCB7XG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgY2xhc3M6IFwicHktbWQtMlwiLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIk1haW5cIlxuICAgICAgICAgICAgfSwgW1xuICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtjbGFzczogXCJjb250YWluZXItZmx1aWRcIn0sIFtcbiAgICAgICAgICAgICAgICAgICAgdGhpcy5tYWtlQ2hpbGQoKVxuICAgICAgICAgICAgICAgIF0pXG4gICAgICAgICAgICBdKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIG1ha2VDaGlsZCgpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdjaGlsZCcpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnY2hpbGQnKTtcbiAgICAgICAgfVxuICAgIH1cbn1cblxuZXhwb3J0IHtNYWluLCBNYWluIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBNb2RhbCBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogTW9kYWwgaGFzIHRoZSBmb2xsb3dpbmcgc2luZ2xlIHJlcGxhY2VtZW50czpcbiAqICpgdGl0bGVgXG4gKiAqYG1lc3NhZ2VgXG4gKiBBbmQgaGFzIHRoZSBmb2xsb3dpbmcgZW51bWVyYXRlZFxuICogcmVwbGFjZW1lbnRzXG4gKiAqIGBidXR0b25gXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBDaGlsZHJlblxuICogLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIGB0aXRsZWAgKHNpbmdsZSkgLSBBIENlbGwgY29udGFpbmluZyB0aGUgdGl0bGVcbiAqIGBtZXNzYWdlYCAoc2luZ2xlKSAtIEEgQ2VsbCBjb250aWFuaW5nIHRoZSBib2R5IG9mIHRoZVxuICogICAgIG1vZGFsIG1lc3NhZ2VcbiAqIGBidXR0b25zYCAoYXJyYXkpIC0gQW4gYXJyYXkgb2YgYnV0dG9uIGNlbGxzXG4gKi9cbmNsYXNzIE1vZGFsIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcbiAgICAgICAgdGhpcy5tYWluU3R5bGUgPSAnZGlzcGxheTpibG9jaztwYWRkaW5nLXJpZ2h0OjE1cHg7JztcblxuICAgICAgICAvLyBCaW5kIGNvbXBvbmVudCBtZXRob2RzXG4gICAgICAgIHRoaXMubWFrZVRpdGxlID0gdGhpcy5tYWtlVGl0bGUuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5tYWtlTWVzc2FnZSA9IHRoaXMubWFrZU1lc3NhZ2UuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5tYWtlQnV0dG9ucyA9IHRoaXMubWFrZUJ1dHRvbnMuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICBjbGFzczogXCJjZWxsIG1vZGFsIGZhZGUgc2hvd1wiLFxuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIk1vZGFsXCIsXG4gICAgICAgICAgICAgICAgcm9sZTogXCJkaWFsb2dcIixcbiAgICAgICAgICAgICAgICBzdHlsZTogbWFpblN0eWxlXG4gICAgICAgICAgICB9LCBbXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge3JvbGU6IFwiZG9jdW1lbnRcIiwgY2xhc3M6IFwibW9kYWwtZGlhbG9nXCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtjbGFzczogXCJtb2RhbC1jb250ZW50XCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgICAgICBoKCdkaXYnLCB7Y2xhc3M6IFwibW9kYWwtaGVhZGVyXCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgaCgnaDUnLCB7Y2xhc3M6IFwibW9kYWwtdGl0bGVcIn0sIFtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy5tYWtlVGl0bGUoKVxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIF0pXG4gICAgICAgICAgICAgICAgICAgICAgICBdKSxcbiAgICAgICAgICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtjbGFzczogXCJtb2RhbC1ib2R5XCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy5tYWtlTWVzc2FnZSgpXG4gICAgICAgICAgICAgICAgICAgICAgICBdKSxcbiAgICAgICAgICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtjbGFzczogXCJtb2RhbC1mb290ZXJcIn0sIHRoaXMubWFrZUJ1dHRvbnMoKSlcbiAgICAgICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgICAgICBdKVxuICAgICAgICAgICAgXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlQnV0dG9ucygpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50c0ZvcignYnV0dG9uJyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZHJlbk5hbWVkKCdidXR0b25zJylcbiAgICAgICAgfVxuICAgIH1cblxuICAgIG1ha2VNZXNzYWdlKCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ21lc3NhZ2UnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ21lc3NhZ2UnKTtcbiAgICAgICAgfVxuICAgIH1cblxuICAgIG1ha2VUaXRsZSgpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCd0aXRsZScpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgndGl0bGUnKTtcbiAgICAgICAgfVxuICAgIH1cbn1cblxuZXhwb3J0IHtNb2RhbCwgTW9kYWwgYXMgZGVmYXVsdH1cbiIsIi8qKlxuICogT2N0aWNvbiBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuY2xhc3MgT2N0aWNvbiBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb250ZXh0IHRvIG1ldGhvZHNcbiAgICAgICAgdGhpcy5fZ2V0SFRNTENsYXNzZXMgPSB0aGlzLl9nZXRIVE1MQ2xhc3Nlcy5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4oXG4gICAgICAgICAgICBoKCdzcGFuJywge1xuICAgICAgICAgICAgICAgIGNsYXNzOiB0aGlzLl9nZXRIVE1MQ2xhc3NlcygpLFxuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIk9jdGljb25cIixcbiAgICAgICAgICAgICAgICBcImFyaWEtaGlkZGVuXCI6IHRydWUsXG4gICAgICAgICAgICAgICAgc3R5bGU6IHRoaXMucHJvcHMuZXh0cmFEYXRhLmRpdlN0eWxlXG4gICAgICAgICAgICB9KVxuICAgICAgICApO1xuICAgIH1cblxuICAgIF9nZXRIVE1MQ2xhc3Nlcygpe1xuICAgICAgICBsZXQgY2xhc3NlcyA9IFtcImNlbGxcIiwgXCJvY3RpY29uXCJdO1xuICAgICAgICB0aGlzLnByb3BzLmV4dHJhRGF0YS5vY3RpY29uQ2xhc3Nlcy5mb3JFYWNoKG5hbWUgPT4ge1xuICAgICAgICAgICAgY2xhc3Nlcy5wdXNoKG5hbWUpO1xuICAgICAgICB9KTtcbiAgICAgICAgcmV0dXJuIGNsYXNzZXMuam9pbihcIiBcIik7XG4gICAgfVxufVxuXG5leHBvcnQge09jdGljb24sIE9jdGljb24gYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIFBhZGRpbmcgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbmNsYXNzIFBhZGRpbmcgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnc3BhbicsIHtcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJQYWRkaW5nXCIsXG4gICAgICAgICAgICAgICAgY2xhc3M6IFwicHgtMlwiXG4gICAgICAgICAgICB9LCBbXCIgXCJdKVxuICAgICAgICApO1xuICAgIH1cbn1cblxuZXhwb3J0IHtQYWRkaW5nLCBQYWRkaW5nIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBQbG90IENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBUaGlzIGNvbXBvbmVudCBjb250YWlucyB0aGUgZm9sbG93aW5nXG4gKiByZWd1bGFyIHJlcGxhY2VtZW50czpcbiAqICogYGNoYXJ0LXVwZGF0ZXJgXG4gKiAqIGBlcnJvcmBcbiAqL1xuXG4vKipcbiAqIEFib3V0IE5hbWVkIENoaWxkcmVuXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogYGNoYXJ0VXBkYXRlcmAgKHNpbmdsZSkgLSBUaGUgVXBkYXRlciBjZWxsXG4gKiBgZXJyb3JgIChzaW5nbGUpIC0gQW4gZXJyb3IgY2VsbCwgaWYgcHJlc2VudFxuICovXG5jbGFzcyBQbG90IGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICAvLyBCaW5kIGNvbXBvbmVudCBtZXRob2RzXG4gICAgICAgIHRoaXMuc2V0dXBQbG90ID0gdGhpcy5zZXR1cFBsb3QuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5tYWtlQ2hhcnRVcGRhdGVyID0gdGhpcy5tYWtlQ2hhcnRVcGRhdGVyLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMubWFrZUVycm9yID0gdGhpcy5tYWtlRXJyb3IuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICBjb21wb25lbnREaWRMb2FkKCkge1xuICAgICAgICB0aGlzLnNldHVwUGxvdCgpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIlBsb3RcIixcbiAgICAgICAgICAgICAgICBjbGFzczogXCJjZWxsXCJcbiAgICAgICAgICAgIH0sIFtcbiAgICAgICAgICAgICAgICBoKCdkaXYnLCB7aWQ6IGBwbG90JHt0aGlzLnByb3BzLmlkfWAsIHN0eWxlOiB0aGlzLnByb3BzLmV4dHJhRGF0YS5kaXZTdHlsZX0pLFxuICAgICAgICAgICAgICAgIHRoaXMubWFrZUNoYXJ0VXBkYXRlcigpLFxuICAgICAgICAgICAgICAgIHRoaXMubWFrZUVycm9yKClcbiAgICAgICAgICAgIF0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgbWFrZUNoYXJ0VXBkYXRlcigpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdjaGFydC11cGRhdGVyJyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCdjaGFydFVwZGF0ZXInKTtcbiAgICAgICAgfVxuICAgIH1cblxuICAgIG1ha2VFcnJvcigpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdlcnJvcicpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnZXJyb3InKTtcbiAgICAgICAgfVxuICAgIH1cblxuICAgIHNldHVwUGxvdCgpe1xuICAgICAgICBjb25zb2xlLmxvZyhcIlNldHRpbmcgdXAgYSBuZXcgcGxvdGx5IGNoYXJ0LlwiKTtcbiAgICAgICAgLy8gVE9ETyBUaGVzZSBhcmUgZ2xvYmFsIHZhciBkZWZpbmVkIGluIHBhZ2UuaHRtbFxuICAgICAgICAvLyB3ZSBzaG91bGQgZG8gc29tZXRoaW5nIGFib3V0IHRoaXMuXG4gICAgICAgIHZhciBwbG90RGl2ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3Bsb3QnICsgdGhpcy5wcm9wcy5pZCk7XG4gICAgICAgIFBsb3RseS5wbG90KFxuICAgICAgICAgICAgcGxvdERpdixcbiAgICAgICAgICAgIFtdLFxuICAgICAgICAgICAge1xuICAgICAgICAgICAgICAgIG1hcmdpbjoge3QgOiAzMCwgbDogMzAsIHI6IDMwLCBiOjMwIH0sXG4gICAgICAgICAgICAgICAgeGF4aXM6IHtyYW5nZXNsaWRlcjoge3Zpc2libGU6IGZhbHNlfX1cbiAgICAgICAgICAgIH0sXG4gICAgICAgICAgICB7IHNjcm9sbFpvb206IHRydWUsIGRyYWdtb2RlOiAncGFuJywgZGlzcGxheWxvZ286IGZhbHNlLCBkaXNwbGF5TW9kZUJhcjogJ2hvdmVyJyxcbiAgICAgICAgICAgICAgICBtb2RlQmFyQnV0dG9uczogWyBbJ3BhbjJkJ10sIFsnem9vbTJkJ10sIFsnem9vbUluMmQnXSwgWyd6b29tT3V0MmQnXSBdIH1cbiAgICAgICAgKTtcbiAgICAgICAgcGxvdERpdi5vbigncGxvdGx5X3JlbGF5b3V0JyxcbiAgICAgICAgICAgIGZ1bmN0aW9uKGV2ZW50ZGF0YSl7XG4gICAgICAgICAgICAgICAgaWYgKHBsb3REaXYuaXNfc2VydmVyX2RlZmluZWRfbW92ZSA9PT0gdHJ1ZSkge1xuICAgICAgICAgICAgICAgICAgICByZXR1cm5cbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgLy9pZiB3ZSdyZSBzZW5kaW5nIGEgc3RyaW5nLCB0aGVuIGl0cyBhIGRhdGUgb2JqZWN0LCBhbmQgd2Ugd2FudCB0byBzZW5kXG4gICAgICAgICAgICAgICAgLy8gYSB0aW1lc3RhbXBcbiAgICAgICAgICAgICAgICBpZiAodHlwZW9mKGV2ZW50ZGF0YVsneGF4aXMucmFuZ2VbMF0nXSkgPT09ICdzdHJpbmcnKSB7XG4gICAgICAgICAgICAgICAgICAgIGV2ZW50ZGF0YSA9IE9iamVjdC5hc3NpZ24oe30sZXZlbnRkYXRhKTtcbiAgICAgICAgICAgICAgICAgICAgZXZlbnRkYXRhW1wieGF4aXMucmFuZ2VbMF1cIl0gPSBEYXRlLnBhcnNlKGV2ZW50ZGF0YVtcInhheGlzLnJhbmdlWzBdXCJdKSAvIDEwMDAuMDtcbiAgICAgICAgICAgICAgICAgICAgZXZlbnRkYXRhW1wieGF4aXMucmFuZ2VbMV1cIl0gPSBEYXRlLnBhcnNlKGV2ZW50ZGF0YVtcInhheGlzLnJhbmdlWzFdXCJdKSAvIDEwMDAuMDtcbiAgICAgICAgICAgICAgICB9XG5cbiAgICAgICAgICAgICAgICBsZXQgcmVzcG9uc2VEYXRhID0ge1xuICAgICAgICAgICAgICAgICAgICAnZXZlbnQnOidwbG90X2xheW91dCcsXG4gICAgICAgICAgICAgICAgICAgICd0YXJnZXRfY2VsbCc6ICdfX2lkZW50aXR5X18nLFxuICAgICAgICAgICAgICAgICAgICAnZGF0YSc6IGV2ZW50ZGF0YVxuICAgICAgICAgICAgICAgIH07XG4gICAgICAgICAgICAgICAgY2VsbFNvY2tldC5zZW5kU3RyaW5nKEpTT04uc3RyaW5naWZ5KHJlc3BvbnNlRGF0YSkpO1xuICAgICAgICAgICAgfSk7XG4gICAgfVxufVxuXG5leHBvcnQge1Bsb3QsIFBsb3QgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIFBvcG92ZXIgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiBUaGlzIGNvbXBvbmVudCBjb250YWlucyB0aGUgZm9sbG93aW5nXG4gKiByZWd1bGFyIHJlcGxhY2VtZW50czpcbiAqICogYHRpdGxlYFxuICogKiBgZGV0YWlsYFxuICogKiBgY29udGVudHNgXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBDaGlsZHJlblxuICogLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIGBjb250ZW50YCAoc2luZ2xlKSAtIFRoZSBjb250ZW50IG9mIHRoZSBwb3BvdmVyXG4gKiBgZGV0YWlsYCAoc2luZ2xlKSAtIERldGFpbCBvZiB0aGUgcG9wb3ZlclxuICogYHRpdGxlYCAoc2luZ2xlKSAtIFRoZSB0aXRsZSBmb3IgdGhlIHBvcG92ZXJcbiAqL1xuY2xhc3MgUG9wb3ZlciBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb21wb25lbnQgbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VUaXRsZSA9IHRoaXMubWFrZVRpdGxlLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMubWFrZUNvbnRlbnQgPSB0aGlzLm1ha2VDb250ZW50LmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMubWFrZURldGFpbCA9IHRoaXMubWFrZURldGFpbC5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gaCgnZGl2JyxcbiAgICAgICAgICAgIHtcbiAgICAgICAgICAgICAgICBjbGFzczogXCJjZWxsIHBvcG92ZXItY2VsbFwiLFxuICAgICAgICAgICAgICAgIHN0eWxlOiB0aGlzLnByb3BzLmV4dHJhRGF0YS5kaXZTdHlsZSxcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJQb3BvdmVyXCJcbiAgICAgICAgICAgIH0sIFtcbiAgICAgICAgICAgICAgICBoKCdhJyxcbiAgICAgICAgICAgICAgICAgICAge1xuICAgICAgICAgICAgICAgICAgICAgICAgaHJlZjogXCIjcG9wbWFpbl9cIiArIHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgICAgICAgICBcImRhdGEtdG9nZ2xlXCI6IFwicG9wb3ZlclwiLFxuICAgICAgICAgICAgICAgICAgICAgICAgXCJkYXRhLXRyaWdnZXJcIjogXCJmb2N1c1wiLFxuICAgICAgICAgICAgICAgICAgICAgICAgXCJkYXRhLWJpbmRcIjogXCIjcG9wX1wiICsgdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICAgICAgICAgIFwiZGF0YS1wbGFjZW1lbnRcIjogXCJib3R0b21cIixcbiAgICAgICAgICAgICAgICAgICAgICAgIHJvbGU6IFwiYnV0dG9uXCIsXG4gICAgICAgICAgICAgICAgICAgICAgICBjbGFzczogXCJidG4gYnRuLXhzXCJcbiAgICAgICAgICAgICAgICAgICAgfSxcbiAgICAgICAgICAgICAgICAgIFt0aGlzLm1ha2VDb250ZW50KCldXG4gICAgICAgICAgICAgICAgKSxcbiAgICAgICAgICAgICAgICBoKCdkaXYnLCB7c3R5bGU6IFwiZGlzcGxheTpub25lXCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgIGgoXCJkaXZcIiwge2lkOiBcInBvcF9cIiArIHRoaXMucHJvcHMuaWR9LCBbXG4gICAgICAgICAgICAgICAgICAgICAgICBoKFwiZGl2XCIsIHtjbGFzczogXCJkYXRhLXRpdGxlXCJ9LCBbdGhpcy5tYWtlVGl0bGUoKV0pLFxuICAgICAgICAgICAgICAgICAgICAgICAgaChcImRpdlwiLCB7Y2xhc3M6IFwiZGF0YS1jb250ZW50XCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgaChcImRpdlwiLCB7c3R5bGU6IFwid2lkdGg6IFwiICsgdGhpcy5wcm9wcy53aWR0aCArIFwicHhcIn0sIFtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy5tYWtlRGV0YWlsKCldXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgKVxuICAgICAgICAgICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgICAgICBdKVxuICAgICAgICAgICAgXVxuICAgICAgICApO1xuICAgIH1cblxuICAgIG1ha2VDb250ZW50KCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ2NvbnRlbnRzJyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCdjb250ZW50Jyk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBtYWtlRGV0YWlsKCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ2RldGFpbCcpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnZGV0YWlsJyk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBtYWtlVGl0bGUoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcigndGl0bGUnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ3RpdGxlJyk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cbmV4cG9ydCB7UG9wb3ZlciwgUG9wb3ZlciBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogUm9vdENlbGwgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIGEgb25lXG4gKiByZWd1bGFyLWtpbmQgcmVwbGFjZW1lbnQ6XG4gKiAqIGBjYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgY2hpbGRgIChzaW5nbGUpIC0gVGhlIGNoaWxkIGNlbGwgdGhpcyBjb250YWluZXIgY29udGFpbnNcbiAqL1xuY2xhc3MgUm9vdENlbGwgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29tcG9uZW50IG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlQ2hpbGQgPSB0aGlzLm1ha2VDaGlsZC5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIlJvb3RDZWxsXCJcbiAgICAgICAgICAgIH0sIFt0aGlzLm1ha2VDaGlsZCgpXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlQ2hpbGQoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignYycpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnY2hpbGQnKTtcbiAgICAgICAgfVxuICAgIH1cbn1cblxuZXhwb3J0IHtSb290Q2VsbCwgUm9vdENlbGwgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIFNjcm9sbGFibGUgIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBUaGlzIGNvbXBvbmVudCBoYXMgYSBvbmVcbiAqIHJlZ3VsYXIta2luZCByZXBsYWNlbWVudDpcbiAqICogYGNoaWxkYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgY2hpbGRgIChzaW5nbGUpIC0gVGhlIGNlbGwvY29tcG9uZW50IHRoaXMgaW5zdGFuY2UgY29udGFpbnNcbiAqL1xuY2xhc3MgU2Nyb2xsYWJsZSBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb21wb25lbnQgbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VDaGlsZCA9IHRoaXMubWFrZUNoaWxkLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiU2Nyb2xsYWJsZVwiXG4gICAgICAgICAgICB9LCBbdGhpcy5tYWtlQ2hpbGQoKV0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgbWFrZUNoaWxkKCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ2NoaWxkJyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCdjaGlsZCcpO1xuICAgICAgICB9XG4gICAgfVxufVxuXG5leHBvcnQge1Njcm9sbGFibGUsIFNjcm9sbGFibGUgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIFNlcXVlbmNlIENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBTZXF1ZW5jZSBoYXMgdGhlIGZvbGxvd2luZyBlbnVtZXJhdGVkXG4gKiByZXBsYWNlbWVudDpcbiAqICogYGNgXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBDaGlsZHJlblxuICogLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIGBlbGVtZW50c2AgKGFycmF5KSAtIEEgbGlzdCBvZiBDZWxscyB0aGF0IGFyZSBpbiB0aGVcbiAqICAgIHNlcXVlbmNlLlxuICovXG5jbGFzcyBTZXF1ZW5jZSBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb21wb25lbnQgbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VFbGVtZW50cyA9IHRoaXMubWFrZUVsZW1lbnRzLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbFwiLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIlNlcXVlbmNlXCIsXG4gICAgICAgICAgICAgICAgc3R5bGU6IHRoaXMucHJvcHMuZXh0cmFEYXRhLmRpdlN0eWxlXG4gICAgICAgICAgICB9LCB0aGlzLm1ha2VFbGVtZW50cygpKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIG1ha2VFbGVtZW50cygpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50c0ZvcignYycpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGRyZW5OYW1lZCgnZWxlbWVudHMnKTtcbiAgICAgICAgfVxuICAgIH1cbn1cblxuZXhwb3J0IHtTZXF1ZW5jZSwgU2VxdWVuY2UgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIFNoZWV0IENlbGwgQ29tcG9uZW50XG4gKiBOT1RFOiBUaGlzIGlzIGluIHBhcnQgYSB3cmFwcGVyXG4gKiBmb3IgaGFuZHNvbnRhYmxlcy5cbiAqL1xuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiBUaGlzIGNvbXBvbmVudCBoYXMgb25lIHJlZ3VsYXJcbiAqIHJlcGxhY2VtZW50OlxuICogKiBgZXJyb3JgXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBDaGlsZHJlblxuICogLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIGBlcnJvcmAgKHNpbmdsZSkgLSBBbiBlcnJvciBjZWxsIGlmIHByZXNlbnRcbiAqL1xuY2xhc3MgU2hlZXQgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIHRoaXMuY3VycmVudFRhYmxlID0gbnVsbDtcblxuICAgICAgICAvLyBCaW5kIGNvbnRleHQgdG8gbWV0aG9kc1xuICAgICAgICB0aGlzLmluaXRpYWxpemVUYWJsZSA9IHRoaXMuaW5pdGlhbGl6ZVRhYmxlLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuaW5pdGlhbGl6ZUhvb2tzID0gdGhpcy5pbml0aWFsaXplSG9va3MuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5tYWtlRXJyb3IgPSB0aGlzLm1ha2VFcnJvci5iaW5kKHRoaXMpO1xuXG4gICAgICAgIC8qKlxuICAgICAgICAgKiBXQVJOSU5HOiBUaGUgQ2VsbCB2ZXJzaW9uIG9mIFNoZWV0IGlzIHN0aWxsIHVzaW5nIGNlcnRhaW5cbiAgICAgICAgICogcG9zdHNjcmlwdHMgYmVjYXVzZSB3ZSBoYXZlIG5vdCB5ZXQgcmVmYWN0b3JlZCB0aGUgc29ja2V0XG4gICAgICAgICAqIHByb3RvY29sLlxuICAgICAgICAgKiBSZW1vdmUgdGhpcyB3YXJuaW5nIGFib3V0IGl0IG9uY2UgdGhhdCBoYXBwZW5zIVxuICAgICAgICAgKi9cbiAgICAgICAgY29uc29sZS53YXJuKGBbVE9ET10gU2hlZXQgc3RpbGwgdXNlcyBjZXJ0YWluIHBvc3RzY2VyaXB0cyBpbiBpdHMgaW50ZXJhY3Rpb24uIFNlZSBjb21wb25lbnQgY29uc3RydWN0b3IgY29tbWVudCBmb3IgbW9yZSBpbmZvcm1hdGlvbmApO1xuICAgIH1cblxuICAgIGNvbXBvbmVudERpZExvYWQoKXtcbiAgICAgICAgY29uc29sZS5sb2coYCNjb21wb25lbnREaWRMb2FkIGNhbGxlZCBmb3IgU2hlZXQgJHt0aGlzLnByb3BzLmlkfWApO1xuICAgICAgICBjb25zb2xlLmxvZyhgVGhpcyBzaGVldCBoYXMgdGhlIGZvbGxvd2luZyByZXBsYWNlbWVudHM6YCwgdGhpcy5yZXBsYWNlbWVudHMpO1xuICAgICAgICB0aGlzLmluaXRpYWxpemVUYWJsZSgpO1xuICAgICAgICBpZih0aGlzLnByb3BzLmV4dHJhRGF0YVsnaGFuZGxlc0RvdWJsZUNsaWNrJ10pe1xuICAgICAgICAgICAgdGhpcy5pbml0aWFsaXplSG9va3MoKTtcbiAgICAgICAgfVxuICAgICAgICAvLyBSZXF1ZXN0IGluaXRpYWwgZGF0YT9cbiAgICAgICAgY2VsbFNvY2tldC5zZW5kU3RyaW5nKEpTT04uc3RyaW5naWZ5KHtcbiAgICAgICAgICAgIGV2ZW50OiBcInNoZWV0X25lZWRzX2RhdGFcIixcbiAgICAgICAgICAgIHRhcmdldF9jZWxsOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgZGF0YTogMFxuICAgICAgICB9KSk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIGNvbnNvbGUubG9nKGBSZW5kZXJpbmcgc2hlZXQgJHt0aGlzLnByb3BzLmlkfWApO1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIlNoZWV0XCIsXG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbFwiXG4gICAgICAgICAgICB9LCBbXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgICAgICBpZDogYHNoZWV0JHt0aGlzLnByb3BzLmlkfWAsXG4gICAgICAgICAgICAgICAgICAgIHN0eWxlOiB0aGlzLnByb3BzLmV4dHJhRGF0YS5kaXZTdHlsZSxcbiAgICAgICAgICAgICAgICAgICAgY2xhc3M6IFwiaGFuZHNvbnRhYmxlXCJcbiAgICAgICAgICAgICAgICB9LCBbdGhpcy5tYWtlRXJyb3IoKV0pXG4gICAgICAgICAgICBdKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIGluaXRpYWxpemVUYWJsZSgpe1xuICAgICAgICBjb25zb2xlLmxvZyhgI2luaXRpYWxpemVUYWJsZSBjYWxsZWQgZm9yIFNoZWV0ICR7dGhpcy5wcm9wcy5pZH1gKTtcbiAgICAgICAgbGV0IGdldFByb3BlcnR5ID0gZnVuY3Rpb24oaW5kZXgpe1xuICAgICAgICAgICAgcmV0dXJuIGZ1bmN0aW9uKHJvdyl7XG4gICAgICAgICAgICAgICAgcmV0dXJuIHJvd1tpbmRleF07XG4gICAgICAgICAgICB9O1xuICAgICAgICB9O1xuICAgICAgICBsZXQgZW1wdHlSb3cgPSBbXTtcbiAgICAgICAgbGV0IGRhdGFOZWVkZWRDYWxsYmFjayA9IGZ1bmN0aW9uKGV2ZW50T2JqZWN0KXtcbiAgICAgICAgICAgIGV2ZW50T2JqZWN0LnRhcmdldF9jZWxsID0gdGhpcy5wcm9wcy5pZDtcbiAgICAgICAgICAgIGNlbGxTb2NrZXQuc2VuZFN0cmluZyhKU09OLnN0cmluZ2lmeShldmVudE9iamVjdCkpO1xuICAgICAgICB9LmJpbmQodGhpcyk7XG4gICAgICAgIGxldCBkYXRhID0gbmV3IFN5bnRoZXRpY0ludGVnZXJBcnJheSh0aGlzLnByb3BzLmV4dHJhRGF0YS5yb3dDb3VudCwgZW1wdHlSb3csIGRhdGFOZWVkZWRDYWxsYmFjayk7XG4gICAgICAgIGxldCBjb250YWluZXIgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZChgc2hlZXQke3RoaXMucHJvcHMuaWR9YCk7XG4gICAgICAgIGxldCBjb2x1bW5OYW1lcyA9IHRoaXMucHJvcHMuZXh0cmFEYXRhLmNvbHVtbk5hbWVzO1xuICAgICAgICBsZXQgY29sdW1ucyA9IGNvbHVtbk5hbWVzLm1hcCgobmFtZSwgaWR4KSA9PiB7XG4gICAgICAgICAgICBlbXB0eVJvdy5wdXNoKFwiXCIpO1xuICAgICAgICAgICAgcmV0dXJuIHtkYXRhOiBnZXRQcm9wZXJ0eShpZHgpfTtcbiAgICAgICAgfSk7XG5cbiAgICAgICAgdGhpcy5jdXJyZW50VGFibGUgPSBuZXcgSGFuZHNvbnRhYmxlKGNvbnRhaW5lciwge1xuICAgICAgICAgICAgZGF0YSxcbiAgICAgICAgICAgIGRhdGFTY2hlbWE6IGZ1bmN0aW9uKG9wdHMpe3JldHVybiB7fTt9LFxuICAgICAgICAgICAgY29sSGVhZGVyczogY29sdW1uTmFtZXMsXG4gICAgICAgICAgICBjb2x1bW5zLFxuICAgICAgICAgICAgcm93SGVhZGVyczp0cnVlLFxuICAgICAgICAgICAgcm93SGVhZGVyV2lkdGg6IDEwMCxcbiAgICAgICAgICAgIHZpZXdwb3J0Um93UmVuZGVyaW5nT2Zmc2V0OiAxMDAsXG4gICAgICAgICAgICBhdXRvQ29sdW1uU2l6ZTogZmFsc2UsXG4gICAgICAgICAgICBhdXRvUm93SGVpZ2h0OiBmYWxzZSxcbiAgICAgICAgICAgIG1hbnVhbENvbHVtblJlc2l6ZTogdHJ1ZSxcbiAgICAgICAgICAgIGNvbFdpZHRoczogdGhpcy5wcm9wcy5leHRyYURhdGEuY29sdW1uV2lkdGgsXG4gICAgICAgICAgICByb3dIZWlnaHRzOiAyMyxcbiAgICAgICAgICAgIHJlYWRPbmx5OiB0cnVlLFxuICAgICAgICAgICAgTWFudWFsUm93TW92ZTogZmFsc2VcbiAgICAgICAgfSk7XG4gICAgICAgIGhhbmRzT25UYWJsZXNbdGhpcy5wcm9wcy5pZF0gPSB7XG4gICAgICAgICAgICB0YWJsZTogdGhpcy5jdXJyZW50VGFibGUsXG4gICAgICAgICAgICBsYXN0Q2VsbENsaWNrZWQ6IHtyb3c6IC0xMDAsIGNvbDogLTEwMH0sXG4gICAgICAgICAgICBkYmxDbGlja2VkOiB0cnVlXG4gICAgICAgIH07XG4gICAgfVxuXG4gICAgaW5pdGlhbGl6ZUhvb2tzKCl7XG4gICAgICAgIEhhbmRzb250YWJsZS5ob29rcy5hZGQoXCJiZWZvcmVPbkNlbGxNb3VzZURvd25cIiwgKGV2ZW50LCBkYXRhKSA9PiB7XG4gICAgICAgICAgICBsZXQgaGFuZHNPbk9iaiA9IGhhbmRzT25UYWJsZXNbdGhpcy5wcm9wcy5pZF07XG4gICAgICAgICAgICBsZXQgbGFzdFJvdyA9IGhhbmRzT25PYmoubGFzdENlbGxDbGlja2VkLnJvdztcbiAgICAgICAgICAgIGxldCBsYXN0Q29sID0gaGFuZHNPbk9iai5sYXN0Q2VsbENsaWNrZWQuY29sO1xuXG4gICAgICAgICAgICBpZigobGFzdFJvdyA9PSBkYXRhLnJvdykgJiYgKGxhc3RDb2wgPSBkYXRhLmNvbCkpe1xuICAgICAgICAgICAgICAgIGhhbmRzT25PYmouZGJsQ2xpY2tlZCA9IHRydWU7XG4gICAgICAgICAgICAgICAgc2V0VGltZW91dCgoKSA9PiB7XG4gICAgICAgICAgICAgICAgICAgIGlmKGhhbmRzT25PYmouZGJsQ2xpY2tlZCl7XG4gICAgICAgICAgICAgICAgICAgICAgICBjZWxsU29ja2V0LnNlbmRTdHJpbmcoSlNPTi5zdHJpbmdpZnkoe1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGV2ZW50OiAnb25DZWxsRGJsQ2xpY2snLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRhcmdldF9jZWxsOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHJvdzogZGF0YS5yb3csXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgY29sOiBkYXRhLmNvbFxuICAgICAgICAgICAgICAgICAgICAgICAgfSkpO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIGhhbmRzT25PYmoubGFzdENlbGxDbGlja2VkID0ge3JvdzogLTEwMCwgY29sOiAtMTAwfTtcbiAgICAgICAgICAgICAgICAgICAgaGFuZHNPbk9iai5kYmxDbGlja2VkID0gZmFsc2U7XG4gICAgICAgICAgICAgICAgfSwgMjAwKTtcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgaGFuZHNPbk9iai5sYXN0Q2VsbENsaWNrZWQgPSB7cm93OiBkYXRhLnJvdywgY29sOiBkYXRhLmNvbH07XG4gICAgICAgICAgICAgICAgc2V0VGltZW91dCgoKSA9PiB7XG4gICAgICAgICAgICAgICAgICAgIGhhbmRzT25PYmoubGFzdENlbGxDbGlja2VkID0ge3JvdzogLTEwMCwgY29sOiAtMTAwfTtcbiAgICAgICAgICAgICAgICAgICAgaGFuZHNPbk9iai5kYmxDbGlja2VkID0gZmFsc2U7XG4gICAgICAgICAgICAgICAgfSwgNjAwKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfSwgdGhpcy5jdXJyZW50VGFibGUpO1xuXG4gICAgICAgIEhhbmRzb250YWJsZS5ob29rcy5hZGQoXCJiZWZvcmVPbkNlbGxDb250ZXh0TWVudVwiLCAoZXZlbnQsIGRhdGEpID0+IHtcbiAgICAgICAgICAgIGxldCBoYW5kc09uT2JqID0gaGFuZHNPblRhYmxlc1t0aGlzLnByb3BzLmlkXTtcbiAgICAgICAgICAgIGhhbmRzT25PYmouZGJsQ2xpY2tlZCA9IGZhbHNlO1xuICAgICAgICAgICAgaGFuZHNPbk9iai5sYXN0Q2VsbENsaWNrZWQgPSB7cm93OiAtMTAwLCBjb2w6IC0xMDB9O1xuICAgICAgICB9LCB0aGlzLmN1cnJlbnRUYWJsZSk7XG5cbiAgICAgICAgSGFuZHNvbnRhYmxlLmhvb2tzLmFkZChcImJlZm9yZUNvbnRleHRNZW51U2hvd1wiLCAoZXZlbnQsIGRhdGEpID0+IHtcbiAgICAgICAgICAgIGxldCBoYW5kc09uT2JqID0gaGFuZHNPblRhYmxlc1t0aGlzLnByb3BzLmlkXTtcbiAgICAgICAgICAgIGhhbmRzT25PYmouZGJsQ2xpY2tlZCA9IGZhbHNlO1xuICAgICAgICAgICAgaGFuZHNPbk9iai5sYXN0Q2VsbENsaWNrZWQgPSB7cm93OiAtMTAwLCBjb2w6IC0xMDB9O1xuICAgICAgICB9LCB0aGlzLmN1cnJlbnRUYWJsZSk7XG4gICAgfVxuXG4gICAgbWFrZUVycm9yKCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ2Vycm9yJyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCdlcnJvcicpO1xuICAgICAgICB9XG4gICAgfVxufVxuXG4vKiogQ29waWVkIG92ZXIgZnJvbSBDZWxscyBpbXBsZW1lbnRhdGlvbiAqKi9cbmNvbnN0IFN5bnRoZXRpY0ludGVnZXJBcnJheSA9IGZ1bmN0aW9uKHNpemUsIGVtcHR5Um93ID0gW10sIGNhbGxiYWNrKXtcbiAgICB0aGlzLmxlbmd0aCA9IHNpemU7XG4gICAgdGhpcy5jYWNoZSA9IHt9O1xuICAgIHRoaXMucHVzaCA9IGZ1bmN0aW9uKCl7fTtcbiAgICB0aGlzLnNwbGljZSA9IGZ1bmN0aW9uKCl7fTtcblxuICAgIHRoaXMuc2xpY2UgPSBmdW5jdGlvbihsb3csIGhpZ2gpe1xuICAgICAgICBpZihoaWdoID09PSB1bmRlZmluZWQpe1xuICAgICAgICAgICAgaGlnaCA9IHRoaXMubGVuZ3RoO1xuICAgICAgICB9XG5cbiAgICAgICAgbGV0IHJlcyA9IEFycmF5KGhpZ2ggLSBsb3cpO1xuICAgICAgICBsZXQgaW5pdExvdyA9IGxvdztcbiAgICAgICAgd2hpbGUobG93IDwgaGlnaCl7XG4gICAgICAgICAgICBsZXQgb3V0ID0gdGhpcy5jYWNoZVtsb3ddO1xuICAgICAgICAgICAgaWYob3V0ID09PSB1bmRlZmluZWQpe1xuICAgICAgICAgICAgICAgIGlmKGNhbGxiYWNrKXtcbiAgICAgICAgICAgICAgICAgICAgY2FsbGJhY2soe1xuICAgICAgICAgICAgICAgICAgICAgICAgZXZlbnQ6ICdzaGVldF9uZWVkc19kYXRhJyxcbiAgICAgICAgICAgICAgICAgICAgICAgIGRhdGE6IGxvd1xuICAgICAgICAgICAgICAgICAgICB9KTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgb3V0ID0gZW1wdHlSb3c7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICByZXNbbG93IC0gaW5pdExvd10gPSBvdXQ7XG4gICAgICAgICAgICBsb3cgKz0gMTtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gcmVzO1xuICAgIH07XG59O1xuXG5leHBvcnQge1NoZWV0LCBTaGVldCBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogU2luZ2xlTGluZVRleHRCb3ggQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbmNsYXNzIFNpbmdsZUxpbmVUZXh0Qm94IGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICAvLyBCaW5kIGNvbnRleHQgdG8gbWV0aG9kc1xuICAgICAgICB0aGlzLmNoYW5nZUhhbmRsZXIgPSB0aGlzLmNoYW5nZUhhbmRsZXIuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgbGV0IGF0dHJzID1cbiAgICAgICAgICAgIHtcbiAgICAgICAgICAgICAgICBjbGFzczogXCJjZWxsXCIsXG4gICAgICAgICAgICAgICAgaWQ6IFwidGV4dF9cIiArIHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgdHlwZTogXCJ0ZXh0XCIsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiU2luZ2xlTGluZVRleHRCb3hcIixcbiAgICAgICAgICAgICAgICBvbmNoYW5nZTogKGV2ZW50KSA9PiB7dGhpcy5jaGFuZ2VIYW5kbGVyKGV2ZW50LnRhcmdldC52YWx1ZSk7fVxuICAgICAgICAgICAgfTtcbiAgICAgICAgaWYgKHRoaXMucHJvcHMuZXh0cmFEYXRhLmlucHV0VmFsdWUgIT09IHVuZGVmaW5lZCkge1xuICAgICAgICAgICAgYXR0cnMucGF0dGVybiA9IHRoaXMucHJvcHMuZXh0cmFEYXRhLmlucHV0VmFsdWU7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIGgoJ2lucHV0JywgYXR0cnMsIFtdKTtcbiAgICB9XG5cbiAgICBjaGFuZ2VIYW5kbGVyKHZhbCkge1xuICAgICAgICBjZWxsU29ja2V0LnNlbmRTdHJpbmcoXG4gICAgICAgICAgICBKU09OLnN0cmluZ2lmeShcbiAgICAgICAgICAgICAgICB7XG4gICAgICAgICAgICAgICAgICAgIFwiZXZlbnRcIjogXCJjbGlja1wiLFxuICAgICAgICAgICAgICAgICAgICBcInRhcmdldF9jZWxsXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgICAgIFwidGV4dFwiOiB2YWxcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICApXG4gICAgICAgICk7XG4gICAgfVxufVxuXG5leHBvcnQge1NpbmdsZUxpbmVUZXh0Qm94LCBTaW5nbGVMaW5lVGV4dEJveCBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogU3BhbiBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuXG5jbGFzcyBTcGFuIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ3NwYW4nLCB7XG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiU3BhblwiLFxuICAgICAgICAgICAgICAgIGNsYXNzOiBcImNlbGxcIlxuICAgICAgICAgICAgfSwgW3RoaXMucHJvcHMuZXh0cmFEYXRhLnRleHRdKVxuICAgICAgICApO1xuICAgIH1cbn1cblxuZXhwb3J0IHtTcGFuLCBTcGFuIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBTdWJzY3JpYmVkIENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBUaGlzIGNvbXBvbmVudCBoYXMgYSBzaW5nbGVcbiAqIHJlZ3VsYXIgcmVwbGFjZW1lbnQ6XG4gKiAqIGBjb250ZW50c2BcbiAqL1xuXG4vKipcbiAqIEFib3V0IE5hbWVkIENoaWxkcmVuXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogYGNvbnRlbnRgIChzaW5nbGUpIC0gVGhlIHVuZGVybHlpbmcgQ2VsbCB0aGF0IGlzIHN1YnNjcmliZWRcbiAqL1xuY2xhc3MgU3Vic2NyaWJlZCBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb21wb25lbnQgbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VDb250ZW50ID0gdGhpcy5tYWtlQ29udGVudC5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gaCgnZGl2JyxcbiAgICAgICAgICAgIHtcbiAgICAgICAgICAgICAgICBjbGFzczogXCJjZWxsIHN1YnNjcmliZWRcIixcbiAgICAgICAgICAgICAgICBzdHlsZTogdGhpcy5wcm9wcy5leHRyYURhdGEuZGl2U3R5bGUsXG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiU3Vic2NyaWJlZFwiXG4gICAgICAgICAgICB9LCBbdGhpcy5tYWtlQ29udGVudCgpXVxuICAgICAgICApO1xuICAgIH1cblxuICAgIG1ha2VDb250ZW50KCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ2NvbnRlbnRzJyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCdjb250ZW50Jyk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cbmV4cG9ydCB7U3Vic2NyaWJlZCwgU3Vic2NyaWJlZCBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogU3Vic2NyaWJlZFNlcXVlbmNlIENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBUaGlzIGNvbXBvbmVudCBoYXMgYSBzaW5nbGVcbiAqIGVudW1lcmF0ZWQgcmVwbGFjZW1lbnQ6XG4gKiAqIGBjaGlsZGBcbiAqL1xuXG4vKipcbiAqIEFib3V0IE5hbWVkIFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgY2hpbGRyZW5gIChhcnJheSkgLSBBbiBhcnJheSBvZiBDZWxscyB0aGF0IGFyZSBzdWJzY3JpYmVkXG4gKi9cbmNsYXNzIFN1YnNjcmliZWRTZXF1ZW5jZSBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG4gICAgICAgIC8vdGhpcy5hZGRSZXBsYWNlbWVudCgnY29udGVudHMnLCAnX19fX19jb250ZW50c19fJyk7XG4gICAgICAgIC8vXG4gICAgICAgIC8vIEJpbmQgY29udGV4dCB0byBtZXRob2RzXG4gICAgICAgIHRoaXMubWFrZUNsYXNzID0gdGhpcy5tYWtlQ2xhc3MuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5tYWtlQ2hpbGRyZW4gPSB0aGlzLm1ha2VDaGlsZHJlbi5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLl9tYWtlUmVwbGFjZW1lbnRDaGlsZHJlbiA9IHRoaXMuX21ha2VSZXBsYWNlbWVudENoaWxkcmVuLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybiBoKCdkaXYnLFxuICAgICAgICAgICAge1xuICAgICAgICAgICAgICAgIGNsYXNzOiB0aGlzLm1ha2VDbGFzcygpLFxuICAgICAgICAgICAgICAgIHN0eWxlOiB0aGlzLnByb3BzLmV4dHJhRGF0YS5kaXZTdHlsZSxcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJTdWJzY3JpYmVkU2VxdWVuY2VcIlxuICAgICAgICAgICAgfSwgW3RoaXMubWFrZUNoaWxkcmVuKCldXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgbWFrZUNoaWxkcmVuKCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5fbWFrZVJlcGxhY2VtZW50Q2hpbGRyZW4oKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIGlmKHRoaXMucHJvcHMuZXh0cmFEYXRhLmFzQ29sdW1ucyl7XG4gICAgICAgICAgICAgICAgbGV0IGZvcm1hdHRlZENoaWxkcmVuID0gdGhpcy5yZW5kZXJDaGlsZHJlbk5hbWVkKCdjaGlsZHJlbicpLm1hcChjaGlsZEVsID0+IHtcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuKFxuICAgICAgICAgICAgICAgICAgICAgICAgaCgnZGl2Jywge2NsYXNzOiBcImNvbC1zbVwiLCBrZXk6IGNoaWxkRWxlbWVudC5pZH0sIFtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBoKCdzcGFuJywge30sIFtjaGlsZEVsXSlcbiAgICAgICAgICAgICAgICAgICAgICAgIF0pXG4gICAgICAgICAgICAgICAgICAgICk7XG4gICAgICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgICAgICAgICAgaCgnZGl2Jywge2NsYXNzOiBcInJvdyBmbGV4LW5vd3JhcFwiLCBrZXk6IGAke3RoaXMucHJvcHMuaWR9LXNwaW5lLXdyYXBwZXJgfSwgZm9ybWF0dGVkQ2hpbGRyZW4pXG4gICAgICAgICAgICAgICAgKTtcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgICAgICAgICAgaCgnZGl2Jywge2tleTogYCR7dGhpcy5wcm9wcy5pZH0tc3BpbmUtd3JhcHBlcmB9LCB0aGlzLnJlbmRlckNoaWxkcmVuTmFtZWQoJ2NoaWxkcmVuJykpXG4gICAgICAgICAgICAgICAgKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgIH1cblxuICAgIG1ha2VDbGFzcygpIHtcbiAgICAgICAgaWYgKHRoaXMucHJvcHMuZXh0cmFEYXRhLmFzQ29sdW1ucykge1xuICAgICAgICAgICAgcmV0dXJuIFwiY2VsbCBzdWJzY3JpYmVkU2VxdWVuY2UgY29udGFpbmVyLWZsdWlkXCI7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIFwiY2VsbCBzdWJzY3JpYmVkU2VxdWVuY2VcIjtcbiAgICB9XG5cbiAgICBfbWFrZVJlcGxhY2VtZW50Q2hpbGRyZW4oKXtcbiAgICAgICAgaWYodGhpcy5wcm9wcy5leHRyYURhdGEuYXNDb2x1bW5zKXtcbiAgICAgICAgICAgIGxldCBmb3JtYXR0ZWRDaGlsZHJlbiA9IHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50c0ZvcignY2hpbGQnKS5tYXAoY2hpbGRFbGVtZW50ID0+IHtcbiAgICAgICAgICAgICAgICByZXR1cm4oXG4gICAgICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtjbGFzczogXCJjb2wtc21cIiwga2V5OiBjaGlsZEVsZW1lbnQuaWR9LCBbXG4gICAgICAgICAgICAgICAgICAgICAgICBoKCdzcGFuJywge30sIFtjaGlsZEVsZW1lbnRdKVxuICAgICAgICAgICAgICAgICAgICBdKVxuICAgICAgICAgICAgICAgICk7XG4gICAgICAgICAgICB9KTtcbiAgICAgICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge2NsYXNzOiBcInJvdyBmbGV4LW5vd3JhcFwiLCBrZXk6IGAke3RoaXMucHJvcHMuaWR9LXNwaW5lLXdyYXBwZXJgfSwgZm9ybWF0dGVkQ2hpbGRyZW4pXG4gICAgICAgICAgICApO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgICAgICBoKCdkaXYnLCB7a2V5OiBgJHt0aGlzLnByb3BzLmlkfS1zcGluZS13cmFwcGVyYH0sIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50c0ZvcignY2hpbGQnKSlcbiAgICAgICAgICAgICk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cbmV4cG9ydCB7U3Vic2NyaWJlZFNlcXVlbmNlLCBTdWJzY3JpYmVkU2VxdWVuY2UgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIFRhYmxlIENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyAzIHJlZ3VsYXJcbiAqIHJlcGxhY2VtZW50czpcbiAqICogYHBhZ2VgXG4gKiAqIGBsZWZ0YFxuICogKiBgcmlnaHRgXG4gKiBUaGlzIGNvbXBvbmVudCBoYXMgMiBlbnVtZXJhdGVkXG4gKiByZXBsYWNlbWVudHM6XG4gKiAqIGBjaGlsZGBcbiAqICogYGhlYWRlcmBcbiAqIE5PVEU6IGBjaGlsZGAgZW51bWVyYXRlZCByZXBsYWNlbWVudHNcbiAqIGFyZSB0d28gZGltZW5zaW9uYWwgYXJyYXlzIVxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgaGVhZGVyc2AgKGFycmF5KSAtIEFuIGFycmF5IG9mIHRhYmxlIGhlYWRlciBjZWxsc1xuICogYGRhdGFDZWxsc2AgKGFycmF5LW9mLWFycmF5KSAtIEEgMi1kaW1lbnNpb25hbCBhcnJheVxuICogICAgc3RydWN0dXJlcyBhcyByb3dzIGJ5IGNvbHVtbnMgdGhhdCBjb250YWlucyB0aGVcbiAqICAgIHRhYmxlIGRhdGEgY2VsbHNcbiAqIGBwYWdlYCAoc2luZ2xlKSAtIEEgY2VsbCB0aGF0IHRlbGxzIHdoaWNoIHBhZ2Ugb2YgdGhlXG4gKiAgICAgdGFibGUgd2UgYXJlIGxvb2tpbmcgYXRcbiAqIGBsZWZ0YCAoc2luZ2xlKSAtIEEgY2VsbCB0aGF0IHNob3dzIHRoZSBudW1iZXIgb24gdGhlIGxlZnRcbiAqIGByaWdodGAgKHNpbmdsZSkgLSBBIGNlbGwgdGhhdCBzaG93IHRoZSBudW1iZXIgb24gdGhlIHJpZ2h0XG4gKi9cbmNsYXNzIFRhYmxlIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICAvLyBCaW5kIGNvbnRleHQgdG8gbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VSb3dzID0gdGhpcy5tYWtlUm93cy5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLm1ha2VGaXJzdFJvdyA9IHRoaXMubWFrZUZpcnN0Um93LmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuX21ha2VSb3dFbGVtZW50cyA9IHRoaXMuX21ha2VSb3dFbGVtZW50cy5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLl90aGVhZFN0eWxlID0gdGhpcy5fdGhlYWRTdHlsZS5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLl9nZXRSb3dEaXNwbGF5RWxlbWVudHMgPSB0aGlzLl9nZXRSb3dEaXNwbGF5RWxlbWVudHMuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuKFxuICAgICAgICAgICAgaCgndGFibGUnLCB7XG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiVGFibGVcIixcbiAgICAgICAgICAgICAgICBjbGFzczogXCJjZWxsIHRhYmxlLWhzY3JvbGwgdGFibGUtc20gdGFibGUtc3RyaXBlZFwiXG4gICAgICAgICAgICB9LCBbXG4gICAgICAgICAgICAgICAgaCgndGhlYWQnLCB7c3R5bGU6IHRoaXMuX3RoZWFkU3R5bGUoKX0sW1xuICAgICAgICAgICAgICAgICAgICB0aGlzLm1ha2VGaXJzdFJvdygpXG4gICAgICAgICAgICAgICAgXSksXG4gICAgICAgICAgICAgICAgaCgndGJvZHknLCB7fSwgdGhpcy5tYWtlUm93cygpKVxuICAgICAgICAgICAgXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBfdGhlYWRTdHlsZSgpe1xuICAgICAgICByZXR1cm4gXCJib3JkZXItYm90dG9tOiBibGFjaztib3JkZXItYm90dG9tLXN0eWxlOnNvbGlkO2JvcmRlci1ib3R0b20td2lkdGg6dGhpbjtcIjtcbiAgICB9XG5cbiAgICBtYWtlSGVhZGVyRWxlbWVudHMoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudHNGb3IoJ2hlYWRlcicpLm1hcCgocmVwbGFjZW1lbnQsIGlkeCkgPT4ge1xuICAgICAgICAgICAgICAgIHJldHVybiBoKCd0aCcsIHtcbiAgICAgICAgICAgICAgICAgICAgc3R5bGU6IFwidmVydGljYWwtYWxpZ246dG9wO1wiLFxuICAgICAgICAgICAgICAgICAgICBrZXk6IGAke3RoaXMucHJvcHMuaWR9LXRhYmxlLWhlYWRlci0ke2lkeH1gXG4gICAgICAgICAgICAgICAgfSwgW3JlcGxhY2VtZW50XSk7XG4gICAgICAgICAgICB9KTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkcmVuTmFtZWQoJ2hlYWRlcnMnKS5tYXAoKHJlcGxhY2VtZW50LCBpZHgpID0+IHtcbiAgICAgICAgICAgICAgICByZXR1cm4gaCgndGgnLCB7XG4gICAgICAgICAgICAgICAgICAgIHN0eWxlOiBcInZlcnRpY2FsLWFsaWduOnRvcDtcIixcbiAgICAgICAgICAgICAgICAgICAga2V5OiBgJHt0aGlzLnByb3BzLmlkfS10YWJsZS1oZWFkZXItJHtpZHh9YFxuICAgICAgICAgICAgICAgIH0sIFtyZXBsYWNlbWVudF0pO1xuICAgICAgICAgICAgfSk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBtYWtlUm93cygpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuX21ha2VSb3dFbGVtZW50cyh0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudHNGb3IoJ2NoaWxkJykpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuX21ha2VSb3dFbGVtZW50cyh0aGlzLnJlbmRlckNoaWxkcmVuTmFtZWQoJ2RhdGFDZWxscycpKTtcbiAgICAgICAgfVxuICAgIH1cblxuXG5cbiAgICBfbWFrZVJvd0VsZW1lbnRzKGVsZW1lbnRzKXtcbiAgICAgICAgLy8gTm90ZTogcm93cyBhcmUgdGhlICpmaXJzdCogZGltZW5zaW9uXG4gICAgICAgIC8vIGluIHRoZSAyLWRpbWVuc2lvbmFsIGFycmF5IHJldHVybmVkXG4gICAgICAgIC8vIGJ5IGdldHRpbmcgdGhlIGBjaGlsZGAgcmVwbGFjZW1lbnQgZWxlbWVudHMuXG4gICAgICAgIHJldHVybiBlbGVtZW50cy5tYXAoKHJvdywgcm93SWR4KSA9PiB7XG4gICAgICAgICAgICBsZXQgY29sdW1ucyA9IHJvdy5tYXAoKGNoaWxkRWxlbWVudCwgY29sSWR4KSA9PiB7XG4gICAgICAgICAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgICAgICAgICAgaCgndGQnLCB7XG4gICAgICAgICAgICAgICAgICAgICAgICBrZXk6IGAke3RoaXMucHJvcHMuaWR9LXRkLSR7cm93SWR4fS0ke2NvbElkeH1gXG4gICAgICAgICAgICAgICAgICAgIH0sIFtjaGlsZEVsZW1lbnRdKVxuICAgICAgICAgICAgICAgICk7XG4gICAgICAgICAgICB9KTtcbiAgICAgICAgICAgIGxldCBpbmRleEVsZW1lbnQgPSBoKCd0ZCcsIHt9LCBbYCR7cm93SWR4ICsgMX1gXSk7XG4gICAgICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgICAgIGgoJ3RyJywge2tleTogYCR7dGhpcy5wcm9wcy5pZH0tdHItJHtyb3dJZHh9YH0sIFtpbmRleEVsZW1lbnQsIC4uLmNvbHVtbnNdKVxuICAgICAgICAgICAgKTtcbiAgICAgICAgfSk7XG4gICAgfVxuXG4gICAgbWFrZUZpcnN0Um93KCl7XG4gICAgICAgIGxldCBoZWFkZXJFbGVtZW50cyA9IHRoaXMubWFrZUhlYWRlckVsZW1lbnRzKCk7XG4gICAgICAgIHJldHVybihcbiAgICAgICAgICAgIGgoJ3RyJywge30sIFtcbiAgICAgICAgICAgICAgICBoKCd0aCcsIHtzdHlsZTogXCJ2ZXJ0aWNhbC1hbGlnbjp0b3A7XCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtjbGFzczogXCJjYXJkXCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgICAgICBoKCdkaXYnLCB7Y2xhc3M6IFwiY2FyZC1ib2R5IHAtMVwifSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIC4uLnRoaXMuX2dldFJvd0Rpc3BsYXlFbGVtZW50cygpXG4gICAgICAgICAgICAgICAgICAgICAgICBdKVxuICAgICAgICAgICAgICAgICAgICBdKVxuICAgICAgICAgICAgICAgIF0pLFxuICAgICAgICAgICAgICAgIC4uLmhlYWRlckVsZW1lbnRzXG4gICAgICAgICAgICBdKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIF9nZXRSb3dEaXNwbGF5RWxlbWVudHMoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiBbXG4gICAgICAgICAgICAgICAgdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ2xlZnQnKSxcbiAgICAgICAgICAgICAgICB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcigncmlnaHQnKSxcbiAgICAgICAgICAgICAgICB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcigncGFnZScpLFxuICAgICAgICAgICAgXTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiBbXG4gICAgICAgICAgICAgICAgdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCdsZWZ0JyksXG4gICAgICAgICAgICAgICAgdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCdyaWdodCcpLFxuICAgICAgICAgICAgICAgIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgncGFnZScpXG4gICAgICAgICAgICBdO1xuICAgICAgICB9XG4gICAgfVxufVxuXG5leHBvcnQge1RhYmxlLCBUYWJsZSBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogVGFicyBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBUaGlzIGNvbXBvbmVudCBoYWQgYSBzaW5nbGVcbiAqIHJlZ3VsYXIgcmVwbGFjZW1lbnQ6XG4gKiAqIGBkaXNwbGF5YFxuICogVGhpcyBjb21wb25lbnQgaGFzIGEgc2luZ2xlXG4gKiBlbnVtZXJhdGVkIHJlcGxhY2VtZW50OlxuICogKiBgaGVhZGVyYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgZGlzcGxheWAgKHNpbmdsZSkgLSBUaGUgQ2VsbCB0aGF0IGdldHMgZGlzcGxheWVkIHdoZW5cbiAqICAgICAgdGhlIHRhYnMgYXJlIHNob3dpbmdcbiAqIGBoZWFkZXJzYCAoYXJyYXkpIC0gQW4gYXJyYXkgb2YgY2VsbHMgdGhhdCBzZXJ2ZSBhc1xuICogICAgIHRoZSB0YWIgaGVhZGVyc1xuICovXG5jbGFzcyBUYWJzIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICAvLyBCaW5kIGNvbXBvbmVudCBtZXRob2RzXG4gICAgICAgIHRoaXMubWFrZUhlYWRlcnMgPSB0aGlzLm1ha2VIZWFkZXJzLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMubWFrZURpc3BsYXkgPSB0aGlzLm1ha2VEaXNwbGF5LmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiVGFic1wiLFxuICAgICAgICAgICAgICAgIGNsYXNzOiBcImNvbnRhaW5lci1mbHVpZCBtYi0zXCJcbiAgICAgICAgICAgIH0sIFtcbiAgICAgICAgICAgICAgICBoKCd1bCcsIHtjbGFzczogXCJuYXYgbmF2LXRhYnNcIiwgcm9sZTogXCJ0YWJsaXN0XCJ9LCB0aGlzLm1ha2VIZWFkZXJzKCkpLFxuICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtjbGFzczogXCJ0YWItY29udGVudFwifSwgW1xuICAgICAgICAgICAgICAgICAgICBoKCdkaXYnLCB7Y2xhc3M6IFwidGFiLXBhbmUgZmFkZSBzaG93IGFjdGl2ZVwiLCByb2xlOiBcInRhYnBhbmVsXCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgICAgICB0aGlzLm1ha2VEaXNwbGF5KClcbiAgICAgICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgICAgICBdKVxuICAgICAgICAgICAgXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlRGlzcGxheSgpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdkaXNwbGF5Jyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCdkaXNwbGF5Jyk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBtYWtlSGVhZGVycygpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50c0ZvcignaGVhZGVyJyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZHJlbk5hbWVkKCdoZWFkZXJzJyk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cblxuZXhwb3J0IHtUYWJzLCBUYWJzIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBUZXh0IENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG5jbGFzcyBUZXh0IGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGNsYXNzOiBcImNlbGxcIixcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBzdHlsZTogdGhpcy5wcm9wcy5leHRyYURhdGEuZGl2U3R5bGUsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiVGV4dFwiXG4gICAgICAgICAgICB9LCBbdGhpcy5wcm9wcy5leHRyYURhdGEucmF3VGV4dF0pXG4gICAgICAgICk7XG4gICAgfVxufVxuXG5leHBvcnQge1RleHQsIFRleHQgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIFRyYWNlYmFjayBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIGEgc2luZ2xlIHJlZ3VsYXJcbiAqIHJlcGFsY2VtZW50OlxuICogKiBgY2hpbGRgXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBDaGlsZHJlblxuICogYHRyYWNlYmFja2AgKHNpbmdsZSkgLSBUaGUgY2VsbCBjb250YWluaW5nIHRoZSB0cmFjZWJhY2sgdGV4dFxuICovXG5jbGFzcyAgVHJhY2ViYWNrIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICAvLyBCaW5kIGNvbXBvbmVudCBtZXRob2RzXG4gICAgICAgIHRoaXMubWFrZVRyYWNlYmFjayA9IHRoaXMubWFrZVRyYWNlYmFjay5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIlRyYWNlYmFja1wiLFxuICAgICAgICAgICAgICAgIGNsYXNzOiBcImFsZXJ0IGFsZXJ0LXByaW1hcnlcIlxuICAgICAgICAgICAgfSwgW3RoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdjaGlsZCcpXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlVHJhY2ViYWNrKCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ2NoaWxkJyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCd0cmFjZWJhY2snKTtcbiAgICAgICAgfVxuICAgIH1cbn1cblxuXG5leHBvcnQge1RyYWNlYmFjaywgVHJhY2ViYWNrIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBfTmF2VGFiIENlbGwgQ29tcG9uZW50XG4gKiBOT1RFOiBUaGlzIHNob3VsZCBwcm9iYWJseSBqdXN0IGJlXG4gKiByb2xsZWQgaW50byB0aGUgTmF2IGNvbXBvbmVudCBzb21laG93LFxuICogb3IgaW5jbHVkZWQgaW4gaXRzIG1vZHVsZSBhcyBhIHByaXZhdGVcbiAqIHN1YmNvbXBvbmVudC5cbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBUaGlzIGNvbXBvbmVudCBoYXMgb25lIHJlZ3VsYXJcbiAqIHJlcGxhY2VtZW50OlxuICogKiBgY2hpbGRgXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBDaGlsZHJlblxuICogLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIGBjaGlsZGAgKHNpbmdsZSkgLSBUaGUgY2VsbCBpbnNpZGUgb2YgdGhlIG5hdmlnYXRpb24gdGFiXG4gKi9cbmNsYXNzIF9OYXZUYWIgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29udGV4dCB0byBtZXRob2RzXG4gICAgICAgIHRoaXMubWFrZUNoaWxkID0gdGhpcy5tYWtlQ2hpbGQuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5jbGlja0hhbmRsZXIgPSB0aGlzLmNsaWNrSGFuZGxlci5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICBsZXQgaW5uZXJDbGFzcyA9IFwibmF2LWxpbmtcIjtcbiAgICAgICAgaWYodGhpcy5wcm9wcy5leHRyYURhdGEuaXNBY3RpdmUpe1xuICAgICAgICAgICAgaW5uZXJDbGFzcyArPSBcIiBhY3RpdmVcIjtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnbGknLCB7XG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgY2xhc3M6IFwibmF2LWl0ZW1cIixcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJfTmF2VGFiXCJcbiAgICAgICAgICAgIH0sIFtcbiAgICAgICAgICAgICAgICBoKCdhJywge1xuICAgICAgICAgICAgICAgICAgICBjbGFzczogaW5uZXJDbGFzcyxcbiAgICAgICAgICAgICAgICAgICAgcm9sZTogXCJ0YWJcIixcbiAgICAgICAgICAgICAgICAgICAgb25jbGljazogdGhpcy5jbGlja0hhbmRsZXJcbiAgICAgICAgICAgICAgICB9LCBbdGhpcy5tYWtlQ2hpbGQoKV0pXG4gICAgICAgICAgICBdKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIG1ha2VDaGlsZCgpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdjaGlsZCcpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnY2hpbGQnKTtcbiAgICAgICAgfVxuICAgIH1cblxuICAgIGNsaWNrSGFuZGxlcihldmVudCl7XG4gICAgICAgIGNlbGxTb2NrZXQuc2VuZFN0cmluZyhcbiAgICAgICAgICAgIEpTT04uc3RyaW5naWZ5KHRoaXMucHJvcHMuZXh0cmFEYXRhLmNsaWNrRGF0YSwgbnVsbCwgNClcbiAgICAgICAgKTtcbiAgICB9XG59XG5cbmV4cG9ydCB7X05hdlRhYiwgX05hdlRhYiBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogX1Bsb3RVcGRhdGVyIENlbGwgQ29tcG9uZW50XG4gKiBOT1RFOiBMYXRlciByZWZhY3RvcmluZ3Mgc2hvdWxkIHJlc3VsdCBpblxuICogdGhpcyBjb21wb25lbnQgYmVjb21pbmcgb2Jzb2xldGVcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbmNvbnN0IE1BWF9JTlRFUlZBTFMgPSAyNTtcblxuY2xhc3MgX1Bsb3RVcGRhdGVyIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICB0aGlzLnJ1blVwZGF0ZSA9IHRoaXMucnVuVXBkYXRlLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMubGlzdGVuRm9yUGxvdCA9IHRoaXMubGlzdGVuRm9yUGxvdC5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIGNvbXBvbmVudERpZExvYWQoKSB7XG4gICAgICAgIC8vIElmIHdlIGNhbiBmaW5kIGEgbWF0Y2hpbmcgUGxvdCBlbGVtZW50XG4gICAgICAgIC8vIGF0IHRoaXMgcG9pbnQsIHdlIHNpbXBseSB1cGRhdGUgaXQuXG4gICAgICAgIC8vIE90aGVyd2lzZSB3ZSBuZWVkIHRvICdsaXN0ZW4nIGZvciB3aGVuXG4gICAgICAgIC8vIGl0IGZpbmFsbHkgY29tZXMgaW50byB0aGUgRE9NLlxuICAgICAgICBsZXQgaW5pdGlhbFBsb3REaXYgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZChgcGxvdCR7dGhpcy5wcm9wcy5leHRyYURhdGEucGxvdElkfWApO1xuICAgICAgICBpZihpbml0aWFsUGxvdERpdil7XG4gICAgICAgICAgICB0aGlzLnJ1blVwZGF0ZShpbml0aWFsUGxvdERpdik7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICB0aGlzLmxpc3RlbkZvclBsb3QoKTtcbiAgICAgICAgfVxuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gaCgnZGl2JyxcbiAgICAgICAgICAgIHtcbiAgICAgICAgICAgICAgICBjbGFzczogXCJjZWxsXCIsXG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgc3R5bGU6IFwiZGlzcGxheTogbm9uZVwiLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIl9QbG90VXBkYXRlclwiXG4gICAgICAgICAgICB9LCBbXSk7XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogSW4gdGhlIGV2ZW50IHRoYXQgYSBgX1Bsb3RVcGRhdGVyYCBoYXMgY29tZVxuICAgICAqIG92ZXIgdGhlIHdpcmUgKmJlZm9yZSogaXRzIGNvcnJlc3BvbmRpbmdcbiAgICAgKiBQbG90IGhhcyBjb21lIG92ZXIgKHdoaWNoIGFwcGVhcnMgdG8gYmVcbiAgICAgKiBjb21tb24pLCB3ZSB3aWxsIHNldCBhbiBpbnRlcnZhbCBvZiA1MG1zXG4gICAgICogYW5kIGNoZWNrIGZvciB0aGUgbWF0Y2hpbmcgUGxvdCBpbiB0aGUgRE9NXG4gICAgICogTUFYX0lOVEVSVkFMUyB0aW1lcywgb25seSBjYWxsaW5nIGBydW5VcGRhdGVgXG4gICAgICogb25jZSB3ZSd2ZSBmb3VuZCBhIG1hdGNoLlxuICAgICAqL1xuICAgIGxpc3RlbkZvclBsb3QoKXtcbiAgICAgICAgbGV0IG51bUNoZWNrcyA9IDA7XG4gICAgICAgIGxldCBwbG90Q2hlY2tlciA9IHdpbmRvdy5zZXRJbnRlcnZhbCgoKSA9PiB7XG4gICAgICAgICAgICBpZihudW1DaGVja3MgPiBNQVhfSU5URVJWQUxTKXtcbiAgICAgICAgICAgICAgICB3aW5kb3cuY2xlYXJJbnRlcnZhbChwbG90Q2hlY2tlcik7XG4gICAgICAgICAgICAgICAgY29uc29sZS5lcnJvcihgQ291bGQgbm90IGZpbmQgbWF0Y2hpbmcgUGxvdCAke3RoaXMucHJvcHMuZXh0cmFEYXRhLnBsb3RJZH0gZm9yIF9QbG90VXBkYXRlciAke3RoaXMucHJvcHMuaWR9YCk7XG4gICAgICAgICAgICAgICAgcmV0dXJuO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgbGV0IHBsb3REaXYgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZChgcGxvdCR7dGhpcy5wcm9wcy5leHRyYURhdGEucGxvdElkfWApO1xuICAgICAgICAgICAgaWYocGxvdERpdil7XG4gICAgICAgICAgICAgICAgdGhpcy5ydW5VcGRhdGUocGxvdERpdik7XG4gICAgICAgICAgICAgICAgd2luZG93LmNsZWFySW50ZXJ2YWwocGxvdENoZWNrZXIpO1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICBudW1DaGVja3MgKz0gMTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfSwgNTApO1xuICAgIH1cblxuICAgIHJ1blVwZGF0ZShhRE9NRWxlbWVudCl7XG4gICAgICAgIGNvbnNvbGUubG9nKFwiVXBkYXRpbmcgcGxvdGx5IGNoYXJ0LlwiKTtcbiAgICAgICAgLy8gVE9ETyBUaGVzZSBhcmUgZ2xvYmFsIHZhciBkZWZpbmVkIGluIHBhZ2UuaHRtbFxuICAgICAgICAvLyB3ZSBzaG91bGQgZG8gc29tZXRoaW5nIGFib3V0IHRoaXMuXG4gICAgICAgIGlmICh0aGlzLnByb3BzLmV4dHJhRGF0YS5leGNlcHRpb25PY2N1cmVkKSB7XG4gICAgICAgICAgICBjb25zb2xlLmxvZyhcInBsb3QgZXhjZXB0aW9uIG9jY3VyZWRcIik7XG4gICAgICAgICAgICBQbG90bHkucHVyZ2UoYURPTUVsZW1lbnQpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgbGV0IGRhdGEgPSB0aGlzLnByb3BzLmV4dHJhRGF0YS5wbG90RGF0YS5tYXAobWFwUGxvdGx5RGF0YSk7XG4gICAgICAgICAgICBQbG90bHkucmVhY3QoYURPTUVsZW1lbnQsIGRhdGEsIGFET01FbGVtZW50LmxheW91dCk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cbmV4cG9ydCB7X1Bsb3RVcGRhdGVyLCBfUGxvdFVwZGF0ZXIgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIFRvb2wgZm9yIFZhbGlkYXRpbmcgQ29tcG9uZW50IFByb3BlcnRpZXNcbiAqL1xuXG5jb25zdCByZXBvcnQgPSAobWVzc2FnZSwgZXJyb3JNb2RlLCBzaWxlbnRNb2RlKSA9PiB7XG4gICAgaWYoZXJyb3JNb2RlID09IHRydWUgJiYgc2lsZW50TW9kZSA9PSBmYWxzZSl7XG4gICAgICAgIGNvbnNvbGUuZXJyb3IobWVzc2FnZSk7XG4gICAgfSBlbHNlIGlmKHNpbGVudE1vZGUgPT0gZmFsc2Upe1xuICAgICAgICBjb25zb2xlLndhcm4obWVzc2FnZSk7XG4gICAgfVxufTtcblxuY29uc3QgUHJvcFR5cGVzID0ge1xuICAgIGVycm9yTW9kZTogZmFsc2UsXG4gICAgc2lsZW50TW9kZTogZmFsc2UsXG4gICAgb25lT2Y6IGZ1bmN0aW9uKGFuQXJyYXkpe1xuICAgICAgICByZXR1cm4gZnVuY3Rpb24oY29tcG9uZW50TmFtZSwgcHJvcE5hbWUsIHByb3BWYWx1ZSwgaXNSZXF1aXJlZCl7XG4gICAgICAgICAgICBmb3IobGV0IGkgPSAwOyBpIDwgYW5BcnJheS5sZW5ndGg7IGkrKyl7XG4gICAgICAgICAgICAgICAgbGV0IHR5cGVDaGVja0l0ZW0gPSBhbkFycmF5W2ldO1xuICAgICAgICAgICAgICAgIGlmKHR5cGVvZih0eXBlQ2hlY2tJdGVtKSA9PSAnZnVuY3Rpb24nKXtcbiAgICAgICAgICAgICAgICAgICAgaWYodHlwZUNoZWNrSXRlbShjb21wb25lbnROYW1lLCBwcm9wTmFtZSwgcHJvcFZhbHVlLCBpc1JlcXVpcmVkLCB0cnVlKSl7XG4gICAgICAgICAgICAgICAgICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIH0gZWxzZSBpZih0eXBlQ2hlY2tJdGVtID09IHByb3BWYWx1ZSl7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIGxldCBtZXNzYWdlID0gYCR7Y29tcG9uZW50TmFtZX0gPj4gJHtwcm9wTmFtZX0gbXVzdCBiZSBvZiBvbmUgb2YgdGhlIGZvbGxvd2luZyB0eXBlczogJHthbkFycmF5fWA7XG4gICAgICAgICAgICByZXBvcnQobWVzc2FnZSwgdGhpcy5lcnJvck1vZGUsIHRoaXMuc2lsZW50TW9kZSk7XG4gICAgICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICAgIH0uYmluZCh0aGlzKTtcbiAgICB9LFxuXG4gICAgZ2V0VmFsaWRhdG9yRm9yVHlwZSh0eXBlU3RyKXtcbiAgICAgICAgcmV0dXJuIGZ1bmN0aW9uKGNvbXBvbmVudE5hbWUsIHByb3BOYW1lLCBwcm9wVmFsdWUsIGlzUmVxdWlyZWQsIGluQ29tcG91bmQgPSBmYWxzZSl7XG4gICAgICAgICAgICAvLyBXZSBhcmUgJ2luIGEgY29tcG91bmQgdmFsaWRhdGlvbicgd2hlbiB0aGUgaW5kaXZpZHVhbFxuICAgICAgICAgICAgLy8gUHJvcFR5cGUgY2hlY2tlcnMgKGllIGZ1bmMsIG51bWJlciwgc3RyaW5nLCBldGMpIGFyZVxuICAgICAgICAgICAgLy8gYmVpbmcgY2FsbGVkIHdpdGhpbiBhIGNvbXBvdW5kIHR5cGUgY2hlY2tlciBsaWtlIG9uZU9mLlxuICAgICAgICAgICAgLy8gSW4gdGhlc2UgY2FzZXMgd2Ugd2FudCB0byBwcmV2ZW50IHRoZSBjYWxsIHRvIHJlcG9ydCxcbiAgICAgICAgICAgIC8vIHdoaWNoIHRoZSBjb21wb3VuZCBjaGVjayB3aWxsIGhhbmRsZSBvbiBpdHMgb3duLlxuICAgICAgICAgICAgaWYoaW5Db21wb3VuZCA9PSBmYWxzZSl7XG4gICAgICAgICAgICAgICAgaWYodHlwZW9mKHByb3BWYWx1ZSkgPT0gdHlwZVN0cil7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgICAgICAgICAgIH0gZWxzZSBpZighaXNSZXF1aXJlZCAmJiAocHJvcFZhbHVlID09IHVuZGVmaW5lZCB8fCBwcm9wVmFsdWUgPT0gbnVsbCkpe1xuICAgICAgICAgICAgICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgICAgICAgICAgICB9IGVsc2UgaWYoaXNSZXF1aXJlZCl7XG4gICAgICAgICAgICAgICAgICAgIGxldCBtZXNzYWdlID0gYCR7Y29tcG9uZW50TmFtZX0gPj4gJHtwcm9wTmFtZX0gaXMgYSByZXF1aXJlZCBwcm9wLCBidXQgYXMgcGFzc2VkIGFzICR7cHJvcFZhbHVlfSFgO1xuICAgICAgICAgICAgICAgICAgICByZXBvcnQobWVzc2FnZSwgdGhpcy5lcnJvck1vZGUsIHRoaXMuc2lsZW50TW9kZSk7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgICAgICBsZXQgbWVzc2FnZSA9IGAke2NvbXBvbmVudE5hbWV9ID4+ICR7cHJvcE5hbWV9IG11c3QgYmUgb2YgdHlwZSAke3R5cGVTdHJ9IWA7XG4gICAgICAgICAgICAgICAgICAgIHJlcG9ydChtZXNzYWdlLCB0aGlzLmVycm9yTW9kZSwgdGhpcy5zaWxlbnRNb2RlKTtcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIC8vIE90aGVyd2lzZSB0aGlzIGlzIGEgc3RyYWlnaHRmb3J3YXJkIHR5cGUgY2hlY2tcbiAgICAgICAgICAgIC8vIGJhc2VkIG9uIHRoZSBnaXZlbiB0eXBlLiBXZSBjaGVjayBhcyB1c3VhbCBmb3IgdGhlIHJlcXVpcmVkXG4gICAgICAgICAgICAvLyBwcm9wZXJ0eSBhbmQgdGhlbiB0aGUgYWN0dWFsIHR5cGUgbWF0Y2ggaWYgbmVlZGVkLlxuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICBpZihpc1JlcXVpcmVkICYmIChwcm9wVmFsdWUgPT0gdW5kZWZpbmVkIHx8IHByb3BWYWx1ZSA9PSBudWxsKSl7XG4gICAgICAgICAgICAgICAgICAgIGxldCBtZXNzYWdlID0gYCR7Y29tcG9uZW50TmFtZX0gPj4gJHtwcm9wTmFtZX0gaXMgYSByZXF1aXJlZCBwcm9wLCBidXQgd2FzIHBhc3NlZCBhcyAke3Byb3BWYWx1ZX0hYDtcbiAgICAgICAgICAgICAgICAgICAgcmVwb3J0KG1lc3NhZ2UsIHRoaXMuZXJyb3JNb2RlLCB0aGlzLnNpbGVudE1vZGUpO1xuICAgICAgICAgICAgICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICAgICAgICAgICAgfSBlbHNlIGlmKCFpc1JlcXVpcmVkICYmIChwcm9wVmFsdWUgPT0gdW5kZWZpbmVkIHx8IHByb3BWYWx1ZSA9PSBudWxsKSl7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICByZXR1cm4gdHlwZW9mKHByb3BWYWx1ZSkgPT0gdHlwZVN0cjtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfTtcbiAgICB9LFxuXG4gICAgZ2V0IG51bWJlcigpe1xuICAgICAgICByZXR1cm4gdGhpcy5nZXRWYWxpZGF0b3JGb3JUeXBlKCdudW1iZXInKS5iaW5kKHRoaXMpO1xuICAgIH0sXG5cbiAgICBnZXQgYm9vbGVhbigpe1xuICAgICAgICByZXR1cm4gdGhpcy5nZXRWYWxpZGF0b3JGb3JUeXBlKCdib29sZWFuJykuYmluZCh0aGlzKTtcbiAgICB9LFxuXG4gICAgZ2V0IHN0cmluZygpe1xuICAgICAgICByZXR1cm4gdGhpcy5nZXRWYWxpZGF0b3JGb3JUeXBlKCdzdHJpbmcnKS5iaW5kKHRoaXMpO1xuICAgIH0sXG5cbiAgICBnZXQgb2JqZWN0KCl7XG4gICAgICAgIHJldHVybiB0aGlzLmdldFZhbGlkYXRvckZvclR5cGUoJ29iamVjdCcpLmJpbmQodGhpcyk7XG4gICAgfSxcblxuICAgIGdldCBmdW5jKCl7XG4gICAgICAgIHJldHVybiB0aGlzLmdldFZhbGlkYXRvckZvclR5cGUoJ2Z1bmN0aW9uJykuYmluZCh0aGlzKTtcbiAgICB9LFxuXG4gICAgdmFsaWRhdGU6IGZ1bmN0aW9uKGNvbXBvbmVudE5hbWUsIHByb3BzLCBwcm9wSW5mbyl7XG4gICAgICAgIGxldCBwcm9wTmFtZXMgPSBuZXcgU2V0KE9iamVjdC5rZXlzKHByb3BzKSk7XG4gICAgICAgIHByb3BOYW1lcy5kZWxldGUoJ2NoaWxkcmVuJyk7XG4gICAgICAgIHByb3BOYW1lcy5kZWxldGUoJ25hbWVkQ2hpbGRyZW4nKTtcbiAgICAgICAgcHJvcE5hbWVzLmRlbGV0ZSgnaWQnKTtcbiAgICAgICAgcHJvcE5hbWVzLmRlbGV0ZSgnZXh0cmFEYXRhJyk7IC8vIEZvciBub3dcbiAgICAgICAgbGV0IHByb3BzVG9WYWxpZGF0ZSA9IEFycmF5LmZyb20ocHJvcE5hbWVzKTtcblxuICAgICAgICAvLyBQZXJmb3JtIGFsbCB0aGUgdmFsaWRhdGlvbnMgb24gZWFjaCBwcm9wZXJ0eVxuICAgICAgICAvLyBhY2NvcmRpbmcgdG8gaXRzIGRlc2NyaXB0aW9uLiBXZSBzdG9yZSB3aGV0aGVyXG4gICAgICAgIC8vIG9yIG5vdCB0aGUgZ2l2ZW4gcHJvcGVydHkgd2FzIGNvbXBsZXRlbHkgdmFsaWRcbiAgICAgICAgLy8gYW5kIHRoZW4gZXZhbHVhdGUgdGhlIHZhbGlkaXR5IG9mIGFsbCBhdCB0aGUgZW5kLlxuICAgICAgICBsZXQgdmFsaWRhdGlvblJlc3VsdHMgPSB7fTtcbiAgICAgICAgcHJvcHNUb1ZhbGlkYXRlLmZvckVhY2gocHJvcE5hbWUgPT4ge1xuICAgICAgICAgICAgbGV0IHByb3BWYWwgPSBwcm9wc1twcm9wTmFtZV07XG4gICAgICAgICAgICBsZXQgdmFsaWRhdGlvblRvQ2hlY2sgPSBwcm9wSW5mb1twcm9wTmFtZV07XG4gICAgICAgICAgICBpZih2YWxpZGF0aW9uVG9DaGVjayl7XG4gICAgICAgICAgICAgICAgbGV0IGhhc1ZhbGlkRGVzY3JpcHRpb24gPSB0aGlzLnZhbGlkYXRlRGVzY3JpcHRpb24oY29tcG9uZW50TmFtZSwgcHJvcE5hbWUsIHZhbGlkYXRpb25Ub0NoZWNrKTtcbiAgICAgICAgICAgICAgICBsZXQgaGFzVmFsaWRQcm9wVHlwZXMgPSB2YWxpZGF0aW9uVG9DaGVjay50eXBlKGNvbXBvbmVudE5hbWUsIHByb3BOYW1lLCBwcm9wVmFsLCB2YWxpZGF0aW9uVG9DaGVjay5yZXF1aXJlZCk7XG4gICAgICAgICAgICAgICAgaWYoaGFzVmFsaWREZXNjcmlwdGlvbiAmJiBoYXNWYWxpZFByb3BUeXBlcyl7XG4gICAgICAgICAgICAgICAgICAgIHZhbGlkYXRpb25SZXN1bHRzW3Byb3BOYW1lXSA9IHRydWU7XG4gICAgICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICAgICAgdmFsaWRhdGlvblJlc3VsdHNbcHJvcE5hbWVdID0gZmFsc2U7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICAvLyBJZiB3ZSBnZXQgaGVyZSwgdGhlIGNvbnN1bWVyIGhhcyBwYXNzZWQgaW4gYSBwcm9wXG4gICAgICAgICAgICAgICAgLy8gdGhhdCBpcyBub3QgcHJlc2VudCBpbiB0aGUgcHJvcFR5cGVzIGRlc2NyaXB0aW9uLlxuICAgICAgICAgICAgICAgIC8vIFdlIHJlcG9ydCB0byB0aGUgY29uc29sZSBhcyBuZWVkZWQgYW5kIHZhbGlkYXRlIGFzIGZhbHNlLlxuICAgICAgICAgICAgICAgIGxldCBtZXNzYWdlID0gYCR7Y29tcG9uZW50TmFtZX0gaGFzIGEgcHJvcCBjYWxsZWQgXCIke3Byb3BOYW1lfVwiIHRoYXQgaXMgbm90IGRlc2NyaWJlZCBpbiBwcm9wVHlwZXMhYDtcbiAgICAgICAgICAgICAgICByZXBvcnQobWVzc2FnZSwgdGhpcy5lcnJvck1vZGUsIHRoaXMuc2lsZW50TW9kZSk7XG4gICAgICAgICAgICAgICAgdmFsaWRhdGlvblJlc3VsdHNbcHJvcE5hbWVdID0gZmFsc2U7XG4gICAgICAgICAgICB9XG4gICAgICAgIH0pO1xuXG4gICAgICAgIC8vIElmIHRoZXJlIHdlcmUgYW55IHRoYXQgZGlkIG5vdCB2YWxpZGF0ZSwgcmV0dXJuXG4gICAgICAgIC8vIGZhbHNlIGFuZCByZXBvcnQgYXMgbXVjaC5cbiAgICAgICAgbGV0IGludmFsaWRzID0gW107XG4gICAgICAgIE9iamVjdC5rZXlzKHZhbGlkYXRpb25SZXN1bHRzKS5mb3JFYWNoKGtleSA9PiB7XG4gICAgICAgICAgICBpZih2YWxpZGF0aW9uUmVzdWx0c1trZXldID09IGZhbHNlKXtcbiAgICAgICAgICAgICAgICBpbnZhbGlkcy5wdXNoKGtleSk7XG4gICAgICAgICAgICB9XG4gICAgICAgIH0pO1xuICAgICAgICBpZihpbnZhbGlkcy5sZW5ndGggPiAwKXtcbiAgICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgICB9XG4gICAgfSxcblxuICAgIHZhbGlkYXRlUmVxdWlyZWQ6IGZ1bmN0aW9uKGNvbXBvbmVudE5hbWUsIHByb3BOYW1lLCBwcm9wVmFsLCBpc1JlcXVpcmVkKXtcbiAgICAgICAgaWYoaXNSZXF1aXJlZCA9PSB0cnVlKXtcbiAgICAgICAgICAgIGlmKHByb3BWYWwgPT0gbnVsbCB8fCBwcm9wVmFsID09IHVuZGVmaW5lZCl7XG4gICAgICAgICAgICAgICAgbGV0IG1lc3NhZ2UgPSBgJHtjb21wb25lbnROYW1lfSA+PiAke3Byb3BOYW1lfSByZXF1aXJlcyBhIHZhbHVlLCBidXQgJHtwcm9wVmFsfSB3YXMgcGFzc2VkIWA7XG4gICAgICAgICAgICAgICAgcmVwb3J0KG1lc3NhZ2UsIHRoaXMuZXJyb3JNb2RlLCB0aGlzLnNpbGVudE1vZGUpO1xuICAgICAgICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICB9LFxuXG4gICAgdmFsaWRhdGVEZXNjcmlwdGlvbjogZnVuY3Rpb24oY29tcG9uZW50TmFtZSwgcHJvcE5hbWUsIHByb3Ape1xuICAgICAgICBsZXQgZGVzYyA9IHByb3AuZGVzY3JpcHRpb247XG4gICAgICAgIGlmKGRlc2MgPT0gdW5kZWZpbmVkIHx8IGRlc2MgPT0gXCJcIiB8fCBkZXNjID09IG51bGwpe1xuICAgICAgICAgICAgbGV0IG1lc3NhZ2UgPSBgJHtjb21wb25lbnROYW1lfSA+PiAke3Byb3BOYW1lfSBoYXMgYW4gZW1wdHkgZGVzY3JpcHRpb24hYDtcbiAgICAgICAgICAgIHJlcG9ydChtZXNzYWdlLCB0aGlzLmVycm9yTW9kZSwgdGhpcy5zaWxlbnRNb2RlKTtcbiAgICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICB9XG59O1xuXG5leHBvcnQge1xuICAgIFByb3BUeXBlc1xufTtcblxuXG4vKioqXG5udW1iZXI6IGZ1bmN0aW9uKGNvbXBvbmVudE5hbWUsIHByb3BOYW1lLCBwcm9wVmFsdWUsIGluQ29tcG91bmQgPSBmYWxzZSl7XG4gICAgICAgIGlmKGluQ29tcG91bmQgPT0gZmFsc2Upe1xuICAgICAgICAgICAgaWYodHlwZW9mKHByb3BWYWx1ZSkgPT0gJ251bWJlcicpe1xuICAgICAgICAgICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICBsZXQgbWVzc2FnZSA9IGAke2NvbXBvbmVudE5hbWV9ID4+ICR7cHJvcE5hbWV9IG11c3QgYmUgb2YgdHlwZSBudW1iZXIhYDtcbiAgICAgICAgICAgICAgICByZXBvcnQobWVzc2FnZSwgdGhpcy5lcnJvck1vZGUsIHRoaXMuc2lsZW50TW9kZSk7XG4gICAgICAgICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICAgICAgfVxuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHR5cGVvZihwcm9wVmFsdWUpID09ICdudW1iZXInO1xuICAgICAgICB9XG5cbiAgICB9LmJpbmQodGhpcyksXG5cbiAgICBzdHJpbmc6IGZ1bmN0aW9uKGNvbXBvbmVudE5hbWUsIHByb3BOYW1lLCBwcm9wVmFsdWUsIGluQ29tcG91bmQgPSBmYWxzZSl7XG4gICAgICAgIGlmKGluQ29tcG91bmQgPT0gZmFsc2Upe1xuICAgICAgICAgICAgaWYodHlwZW9mKHByb3BWYWx1ZSkgPT0gJ3N0cmluZycpe1xuICAgICAgICAgICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICBsZXQgbWVzc2FnZSA9IGAke2NvbXBvbmVudE5hbWV9ID4+ICR7cHJvcE5hbWV9IG11c3QgYmUgb2YgdHlwZSBzdHJpbmchYDtcbiAgICAgICAgICAgICAgICByZXBvcnQobWVzc2FnZSwgdGhpcy5lcnJvck1vZGUsIHRoaXMuc2lsZW50TW9kZSk7XG4gICAgICAgICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICAgICAgfVxuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHR5cGVvZihwcm9wVmFsdWUpID09ICdzdHJpbmcnO1xuICAgICAgICB9XG4gICAgfS5iaW5kKHRoaXMpLFxuXG4gICAgYm9vbGVhbjogZnVuY3Rpb24oY29tcG9uZW50TmFtZSwgcHJvcE5hbWUsIHByb3BWYWx1ZSwgaW5Db21wb3VuZCA9IGZhbHNlKXtcbiAgICAgICAgaWYoaW5Db21wb3VuZCA9PSBmYWxzZSl7XG4gICAgICAgICAgICBpZih0eXBlb2YocHJvcFZhbHVlKSA9PSAnYm9vbGVhbicpe1xuICAgICAgICAgICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICBsZXQgbWVzc2FnZSA9IGAke2NvbXBvbmVudE5hbWV9ID4+ICR7cHJvcE5hbWV9IG11c3QgYmUgb2YgdHlwZSBib29sZWFuIWA7XG4gICAgICAgICAgICAgICAgcmVwb3J0KG1lc3NhZ2UsIHRoaXMuZXJyb3JNb2RlLCB0aGlzLnNpbGVudE1vZGUpO1xuICAgICAgICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0eXBlb2YocHJvcFZhbHVlKSA9PSAnYm9vbGVhbic7XG4gICAgICAgIH1cbiAgICB9LmJpbmQodGhpcyksXG5cbiAgICBvYmplY3Q6IGZ1bmN0aW9uKGNvbXBvbmVudE5hbWUsIHByb3BOYW1lLCBwcm9wVmFsdWUsIGluQ29tcG91bmQgPSBmYWxzZSl7XG4gICAgICAgIGlmKGluQ29tcG91bmQgPT0gZmFsc2Upe1xuICAgICAgICAgICAgaWYodHlwZW9mKHByb3BWYWx1ZSkgPT0gJ29iamVjdCcpe1xuICAgICAgICAgICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICBsZXQgbWVzc2FnZSA9IGAke2NvbXBvbmVudE5hbWV9ID4+ICR7cHJvcE5hbWV9IG11c3QgYmUgb2YgdHlwZSBvYmplY3QhYDtcbiAgICAgICAgICAgICAgICByZXBvcnQobWVzc2FnZSwgdGhpcy5lcnJvck1vZGUsIHRoaXMuc2lsZW50TW9kZSk7XG4gICAgICAgICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICAgICAgfVxuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHR5cGVvZihwcm9wVmFsdWUpID09ICdvYmplY3QnO1xuICAgICAgICB9XG4gICAgfS5iaW5kKHRoaXMpLFxuXG4gICAgZnVuYzogZnVuY3Rpb24oY29tcG9uZW50TmFtZSwgcHJvcE5hbWUsIHByb3BWYWx1ZSwgaW5Db21wb3VuZCA9IGZhbHNlKXtcbiAgICAgICAgaWYoaW5Db21wb3VuZCA9PSBmYWxzZSl7XG4gICAgICAgICAgICBpZih0eXBlb2YocHJvcFZhbHVlKSA9PSAnZnVuY3Rpb24nKXtcbiAgICAgICAgICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgbGV0IG1lc3NhZ2UgPSBgJHtjb21wb25lbnROYW1lfSA+PiAke3Byb3BOYW1lfSBtdXN0IGJlIG9mIHR5cGUgZnVuY3Rpb24hYDtcbiAgICAgICAgICAgICAgICByZXBvcnQobWVzc2FnZSwgdGhpcy5lcnJvck1vZGUsIHRoaXMuc2lsZW50TW9kZSk7XG4gICAgICAgICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICAgICAgfVxuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHR5cGVvZihwcm9wVmFsdWUpID09ICdmdW5jdGlvbic7XG4gICAgICAgIH1cbiAgICB9LmJpbmQodGhpcyksXG5cbioqKi9cbiIsImNsYXNzIFJlcGxhY2VtZW50c0hhbmRsZXIge1xuICAgIGNvbnN0cnVjdG9yKHJlcGxhY2VtZW50cyl7XG4gICAgICAgIHRoaXMucmVwbGFjZW1lbnRzID0gcmVwbGFjZW1lbnRzO1xuICAgICAgICB0aGlzLnJlZ3VsYXIgPSB7fTtcbiAgICAgICAgdGhpcy5lbnVtZXJhdGVkID0ge307XG5cbiAgICAgICAgaWYocmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHRoaXMucHJvY2Vzc1JlcGxhY2VtZW50cygpO1xuICAgICAgICB9XG5cbiAgICAgICAgLy8gQmluZCBjb250ZXh0IHRvIG1ldGhvZHNcbiAgICAgICAgdGhpcy5wcm9jZXNzUmVwbGFjZW1lbnRzID0gdGhpcy5wcm9jZXNzUmVwbGFjZW1lbnRzLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMucHJvY2Vzc1JlZ3VsYXIgPSB0aGlzLnByb2Nlc3NSZWd1bGFyLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuaGFzUmVwbGFjZW1lbnQgPSB0aGlzLmhhc1JlcGxhY2VtZW50LmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuZ2V0UmVwbGFjZW1lbnRGb3IgPSB0aGlzLmdldFJlcGxhY2VtZW50Rm9yLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuZ2V0UmVwbGFjZW1lbnRzRm9yID0gdGhpcy5nZXRSZXBsYWNlbWVudHNGb3IuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5tYXBSZXBsYWNlbWVudHNGb3IgPSB0aGlzLm1hcFJlcGxhY2VtZW50c0Zvci5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHByb2Nlc3NSZXBsYWNlbWVudHMoKXtcbiAgICAgICAgdGhpcy5yZXBsYWNlbWVudHMuZm9yRWFjaChyZXBsYWNlbWVudCA9PiB7XG4gICAgICAgICAgICBsZXQgcmVwbGFjZW1lbnRJbmZvID0gdGhpcy5jb25zdHJ1Y3Rvci5yZWFkUmVwbGFjZW1lbnRTdHJpbmcocmVwbGFjZW1lbnQpO1xuICAgICAgICAgICAgaWYocmVwbGFjZW1lbnRJbmZvLmlzRW51bWVyYXRlZCl7XG4gICAgICAgICAgICAgICAgdGhpcy5wcm9jZXNzRW51bWVyYXRlZChyZXBsYWNlbWVudCwgcmVwbGFjZW1lbnRJbmZvKTtcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgdGhpcy5wcm9jZXNzUmVndWxhcihyZXBsYWNlbWVudCwgcmVwbGFjZW1lbnRJbmZvKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfSk7XG4gICAgICAgIC8vIE5vdyB3ZSB1cGRhdGUgdGhpcy5lbnVtZXJhdGVkIHRvIGhhdmUgaXQncyB0b3AgbGV2ZWxcbiAgICAgICAgLy8gdmFsdWVzIGFzIEFycmF5cyBpbnN0ZWFkIG9mIG5lc3RlZCBkaWN0cyBhbmQgd2Ugc29ydFxuICAgICAgICAvLyBiYXNlZCBvbiB0aGUgZXh0cmFjdGVkIGluZGljZXMgKHdoaWNoIGFyZSBhdCB0aGlzIHBvaW50XG4gICAgICAgIC8vIGp1c3Qga2V5cyBvbiBzdWJkaWN0cyBvciBtdWx0aWRpbWVuc2lvbmFsIGRpY3RzKVxuICAgICAgICBPYmplY3Qua2V5cyh0aGlzLmVudW1lcmF0ZWQpLmZvckVhY2goa2V5ID0+IHtcbiAgICAgICAgICAgIGxldCBlbnVtZXJhdGVkUmVwbGFjZW1lbnRzID0gdGhpcy5lbnVtZXJhdGVkW2tleV07XG4gICAgICAgICAgICB0aGlzLmVudW1lcmF0ZWRba2V5XSA9IHRoaXMuY29uc3RydWN0b3IuZW51bWVyYXRlZFZhbFRvU29ydGVkQXJyYXkoZW51bWVyYXRlZFJlcGxhY2VtZW50cyk7XG4gICAgICAgIH0pO1xuICAgIH1cblxuICAgIHByb2Nlc3NSZWd1bGFyKHJlcGxhY2VtZW50TmFtZSwgcmVwbGFjZW1lbnRJbmZvKXtcbiAgICAgICAgbGV0IHJlcGxhY2VtZW50S2V5ID0gdGhpcy5jb25zdHJ1Y3Rvci5rZXlGcm9tTmFtZVBhcnRzKHJlcGxhY2VtZW50SW5mby5uYW1lUGFydHMpO1xuICAgICAgICB0aGlzLnJlZ3VsYXJbcmVwbGFjZW1lbnRLZXldID0gcmVwbGFjZW1lbnROYW1lO1xuICAgIH1cblxuICAgIHByb2Nlc3NFbnVtZXJhdGVkKHJlcGxhY2VtZW50TmFtZSwgcmVwbGFjZW1lbnRJbmZvKXtcbiAgICAgICAgbGV0IHJlcGxhY2VtZW50S2V5ID0gdGhpcy5jb25zdHJ1Y3Rvci5rZXlGcm9tTmFtZVBhcnRzKHJlcGxhY2VtZW50SW5mby5uYW1lUGFydHMpO1xuICAgICAgICBsZXQgY3VycmVudEVudHJ5ID0gdGhpcy5lbnVtZXJhdGVkW3JlcGxhY2VtZW50S2V5XTtcblxuICAgICAgICAvLyBJZiBpdCdzIHVuZGVmaW5lZCwgdGhpcyBpcyB0aGUgZmlyc3RcbiAgICAgICAgLy8gb2YgdGhlIGVudW1lcmF0ZWQgcmVwbGFjZW1lbnRzIGZvciB0aGlzXG4gICAgICAgIC8vIGtleSwgaWUgc29tZXRoaW5nIGxpa2UgX19fX2NoaWxkXzBfX1xuICAgICAgICBpZihjdXJyZW50RW50cnkgPT0gdW5kZWZpbmVkKXtcbiAgICAgICAgICAgIHRoaXMuZW51bWVyYXRlZFtyZXBsYWNlbWVudEtleV0gPSB7fTtcbiAgICAgICAgICAgIGN1cnJlbnRFbnRyeSA9IHRoaXMuZW51bWVyYXRlZFtyZXBsYWNlbWVudEtleV07XG4gICAgICAgIH1cblxuICAgICAgICAvLyBXZSBhZGQgdGhlIGVudW1lcmF0ZWQgaW5kaWNlcyBhcyBrZXlzIHRvIGEgZGljdFxuICAgICAgICAvLyBhbmQgd2UgZG8gdGhpcyByZWN1cnNpdmVseSBhY3Jvc3MgZGltZW5zaW9ucyBhc1xuICAgICAgICAvLyBuZWVkZWQuXG4gICAgICAgIHRoaXMuY29uc3RydWN0b3IucHJvY2Vzc0RpbWVuc2lvbihyZXBsYWNlbWVudEluZm8uZW51bWVyYXRlZFZhbHVlcywgY3VycmVudEVudHJ5LCByZXBsYWNlbWVudE5hbWUpO1xuICAgIH1cblxuICAgIC8vIEFjY2Vzc2luZyBhbmQgb3RoZXIgQ29udmVuaWVuY2UgTWV0aG9kc1xuICAgIGhhc1JlcGxhY2VtZW50KGFSZXBsYWNlbWVudE5hbWUpe1xuICAgICAgICBpZih0aGlzLnJlZ3VsYXIuaGFzT3duUHJvcGVydHkoYVJlcGxhY2VtZW50TmFtZSkpe1xuICAgICAgICAgICAgcmV0dXJuIHRydWU7XG4gICAgICAgIH0gZWxzZSBpZih0aGlzLmVudW1lcmF0ZWQuaGFzT3duUHJvcGVydHkoYVJlcGxhY2VtZW50TmFtZSkpe1xuICAgICAgICAgICAgcmV0dXJuIHRydWU7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgIH1cblxuICAgIGdldFJlcGxhY2VtZW50Rm9yKGFSZXBsYWNlbWVudE5hbWUpe1xuICAgICAgICBsZXQgZm91bmQgPSB0aGlzLnJlZ3VsYXJbYVJlcGxhY2VtZW50TmFtZV07XG4gICAgICAgIGlmKGZvdW5kID09IHVuZGVmaW5lZCl7XG4gICAgICAgICAgICByZXR1cm4gbnVsbDtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gZm91bmQ7XG4gICAgfVxuXG4gICAgZ2V0UmVwbGFjZW1lbnRzRm9yKGFSZXBsYWNlbWVudE5hbWUpe1xuICAgICAgICBsZXQgZm91bmQgPSB0aGlzLmVudW1lcmF0ZWRbYVJlcGxhY2VtZW50TmFtZV07XG4gICAgICAgIGlmKGZvdW5kID09IHVuZGVmaW5lZCl7XG4gICAgICAgICAgICByZXR1cm4gbnVsbDtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gZm91bmQ7XG4gICAgfVxuXG4gICAgbWFwUmVwbGFjZW1lbnRzRm9yKGFSZXBsYWNlbWVudE5hbWUsIG1hcEZ1bmN0aW9uKXtcbiAgICAgICAgaWYoIXRoaXMuaGFzUmVwbGFjZW1lbnQoYVJlcGxhY2VtZW50TmFtZSkpe1xuICAgICAgICAgICAgdGhyb3cgbmV3IEVycm9yKGBJbnZhbGlkIHJlcGxhY2VtZW50IG5hbWU6ICR7YVJlcGxhY2VtZW50bmFtZX1gKTtcbiAgICAgICAgfVxuICAgICAgICBsZXQgcm9vdCA9IHRoaXMuZ2V0UmVwbGFjZW1lbnRzRm9yKGFSZXBsYWNlbWVudE5hbWUpO1xuICAgICAgICByZXR1cm4gdGhpcy5fcmVjdXJzaXZlbHlNYXAocm9vdCwgbWFwRnVuY3Rpb24pO1xuICAgIH1cblxuICAgIF9yZWN1cnNpdmVseU1hcChjdXJyZW50SXRlbSwgbWFwRnVuY3Rpb24pe1xuICAgICAgICBpZighQXJyYXkuaXNBcnJheShjdXJyZW50SXRlbSkpe1xuICAgICAgICAgICAgcmV0dXJuIG1hcEZ1bmN0aW9uKGN1cnJlbnRJdGVtKTtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gY3VycmVudEl0ZW0ubWFwKHN1Ykl0ZW0gPT4ge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuX3JlY3Vyc2l2ZWx5TWFwKHN1Ykl0ZW0sIG1hcEZ1bmN0aW9uKTtcbiAgICAgICAgfSk7XG4gICAgfVxuXG4gICAgLy8gU3RhdGljIGhlbHBlcnNcbiAgICBzdGF0aWMgcHJvY2Vzc0RpbWVuc2lvbihyZW1haW5pbmdWYWxzLCBpbkRpY3QsIHJlcGxhY2VtZW50TmFtZSl7XG4gICAgICAgIGlmKHJlbWFpbmluZ1ZhbHMubGVuZ3RoID09IDEpe1xuICAgICAgICAgICAgaW5EaWN0W3JlbWFpbmluZ1ZhbHNbMF1dID0gcmVwbGFjZW1lbnROYW1lO1xuICAgICAgICAgICAgcmV0dXJuO1xuICAgICAgICB9XG4gICAgICAgIGxldCBuZXh0S2V5ID0gcmVtYWluaW5nVmFsc1swXTtcbiAgICAgICAgbGV0IG5leHREaWN0ID0gaW5EaWN0W25leHRLZXldO1xuICAgICAgICBpZihuZXh0RGljdCA9PSB1bmRlZmluZWQpe1xuICAgICAgICAgICAgaW5EaWN0W25leHRLZXldID0ge307XG4gICAgICAgICAgICBuZXh0RGljdCA9IGluRGljdFtuZXh0S2V5XTtcbiAgICAgICAgfVxuICAgICAgICB0aGlzLnByb2Nlc3NEaW1lbnNpb24ocmVtYWluaW5nVmFscy5zbGljZSgxKSwgbmV4dERpY3QsIHJlcGxhY2VtZW50TmFtZSk7XG4gICAgfVxuXG4gICAgc3RhdGljIGVudW1lcmF0ZWRWYWxUb1NvcnRlZEFycmF5KGFEaWN0LCBhY2N1bXVsYXRlID0gW10pe1xuICAgICAgICBpZih0eXBlb2YgYURpY3QgIT09ICdvYmplY3QnKXtcbiAgICAgICAgICAgIHJldHVybiBhRGljdDtcbiAgICAgICAgfVxuICAgICAgICBsZXQgc29ydGVkS2V5cyA9IE9iamVjdC5rZXlzKGFEaWN0KS5zb3J0KChmaXJzdCwgc2Vjb25kKSA9PiB7XG4gICAgICAgICAgICByZXR1cm4gKHBhcnNlSW50KGZpcnN0KSAtIHBhcnNlSW50KHNlY29uZCkpO1xuICAgICAgICB9KTtcbiAgICAgICAgbGV0IHN1YkVudHJpZXMgPSBzb3J0ZWRLZXlzLm1hcChrZXkgPT4ge1xuICAgICAgICAgICAgbGV0IGVudHJ5ID0gYURpY3Rba2V5XTtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmVudW1lcmF0ZWRWYWxUb1NvcnRlZEFycmF5KGVudHJ5KTtcbiAgICAgICAgfSk7XG4gICAgICAgIHJldHVybiBzdWJFbnRyaWVzO1xuICAgIH1cblxuICAgIHN0YXRpYyBrZXlGcm9tTmFtZVBhcnRzKG5hbWVQYXJ0cyl7XG4gICAgICAgIHJldHVybiBuYW1lUGFydHMuam9pbihcIi1cIik7XG4gICAgfVxuXG4gICAgc3RhdGljIHJlYWRSZXBsYWNlbWVudFN0cmluZyhyZXBsYWNlbWVudCl7XG4gICAgICAgIGxldCBuYW1lUGFydHMgPSBbXTtcbiAgICAgICAgbGV0IGlzRW51bWVyYXRlZCA9IGZhbHNlO1xuICAgICAgICBsZXQgZW51bWVyYXRlZFZhbHVlcyA9IFtdO1xuICAgICAgICBsZXQgcGllY2VzID0gcmVwbGFjZW1lbnQuc3BsaXQoJ18nKS5maWx0ZXIoaXRlbSA9PiB7XG4gICAgICAgICAgICByZXR1cm4gaXRlbSAhPSAnJztcbiAgICAgICAgfSk7XG4gICAgICAgIHBpZWNlcy5mb3JFYWNoKHBpZWNlID0+IHtcbiAgICAgICAgICAgIGxldCBudW0gPSBwYXJzZUludChwaWVjZSk7XG4gICAgICAgICAgICBpZihpc05hTihudW0pKXtcbiAgICAgICAgICAgICAgICBuYW1lUGFydHMucHVzaChwaWVjZSk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICBpc0VudW1lcmF0ZWQgPSB0cnVlO1xuICAgICAgICAgICAgZW51bWVyYXRlZFZhbHVlcy5wdXNoKG51bSk7XG4gICAgICAgIH1cbiAgICAgICAgfSk7XG4gICAgICAgIHJldHVybiB7XG4gICAgICAgICAgICBuYW1lUGFydHMsXG4gICAgICAgICAgICBpc0VudW1lcmF0ZWQsXG4gICAgICAgICAgICBlbnVtZXJhdGVkVmFsdWVzXG4gICAgICAgIH07XG4gICAgfVxufVxuXG5leHBvcnQge1xuICAgIFJlcGxhY2VtZW50c0hhbmRsZXIsXG4gICAgUmVwbGFjZW1lbnRzSGFuZGxlciBhcyBkZWZhdWx0XG59O1xuIiwiaW1wb3J0ICdtYXF1ZXR0ZSc7XG5jb25zdCBoID0gbWFxdWV0dGUuaDtcbi8vaW1wb3J0IHtsYW5nVG9vbHN9IGZyb20gJ2FjZS9leHQvbGFuZ3VhZ2VfdG9vbHMnO1xuaW1wb3J0IHtDZWxsSGFuZGxlcn0gZnJvbSAnLi9DZWxsSGFuZGxlcic7XG5pbXBvcnQge0NlbGxTb2NrZXR9IGZyb20gJy4vQ2VsbFNvY2tldCc7XG5pbXBvcnQge0NvbXBvbmVudFJlZ2lzdHJ5fSBmcm9tICcuL0NvbXBvbmVudFJlZ2lzdHJ5JztcblxuLyoqXG4gKiBHbG9iYWxzXG4gKiovXG53aW5kb3cubGFuZ1Rvb2xzID0gYWNlLnJlcXVpcmUoXCJhY2UvZXh0L2xhbmd1YWdlX3Rvb2xzXCIpO1xud2luZG93LmFjZUVkaXRvcnMgPSB7fTtcbndpbmRvdy5oYW5kc09uVGFibGVzID0ge307XG5cbi8qKlxuICogSW5pdGlhbCBSZW5kZXJcbiAqKi9cbmNvbnN0IGluaXRpYWxSZW5kZXIgPSBmdW5jdGlvbigpe1xuICAgIHJldHVybiBoKFwiZGl2XCIsIHt9LCBbXG4gICAgICAgICBoKFwiZGl2XCIsIHtpZDogXCJwYWdlX3Jvb3RcIn0sIFtcbiAgICAgICAgICAgICBoKFwiZGl2LmNvbnRhaW5lci1mbHVpZFwiLCB7fSwgW1xuICAgICAgICAgICAgICAgICBoKFwiZGl2LmNhcmRcIiwge2NsYXNzOiBcIm10LTVcIn0sIFtcbiAgICAgICAgICAgICAgICAgICAgIGgoXCJkaXYuY2FyZC1ib2R5XCIsIHt9LCBbXCJMb2FkaW5nLi4uXCJdKVxuICAgICAgICAgICAgICAgICBdKVxuICAgICAgICAgICAgIF0pXG4gICAgICAgICBdKSxcbiAgICAgICAgIGgoXCJkaXZcIiwge2lkOiBcImhvbGRpbmdfcGVuXCIsIHN0eWxlOiBcImRpc3BsYXk6bm9uZVwifSwgW10pXG4gICAgIF0pO1xufTtcblxuLyoqXG4gKiBDZWxsIFNvY2tldCBhbmQgSGFuZGxlclxuICoqL1xubGV0IHByb2plY3RvciA9IG1hcXVldHRlLmNyZWF0ZVByb2plY3RvcigpO1xuY29uc3QgY2VsbFNvY2tldCA9IG5ldyBDZWxsU29ja2V0KCk7XG5jb25zdCBjZWxsSGFuZGxlciA9IG5ldyBDZWxsSGFuZGxlcihoLCBwcm9qZWN0b3IsIENvbXBvbmVudFJlZ2lzdHJ5KTtcbmNlbGxTb2NrZXQub25Qb3N0c2NyaXB0cyhjZWxsSGFuZGxlci5oYW5kbGVQb3N0c2NyaXB0KTtcbmNlbGxTb2NrZXQub25NZXNzYWdlKGNlbGxIYW5kbGVyLmhhbmRsZU1lc3NhZ2UpO1xuY2VsbFNvY2tldC5vbkNsb3NlKGNlbGxIYW5kbGVyLnNob3dDb25uZWN0aW9uQ2xvc2VkKTtcbmNlbGxTb2NrZXQub25FcnJvcihlcnIgPT4ge1xuICAgIGNvbnNvbGUuZXJyb3IoXCJTT0NLRVQgRVJST1I6IFwiLCBlcnIpO1xufSk7XG5cbi8qKiBGb3Igbm93LCB3ZSBiaW5kIHRoZSBjdXJyZW50IHNvY2tldCBhbmQgaGFuZGxlciB0byB0aGUgZ2xvYmFsIHdpbmRvdyAqKi9cbndpbmRvdy5jZWxsU29ja2V0ID0gY2VsbFNvY2tldDtcbndpbmRvdy5jZWxsSGFuZGxlciA9IGNlbGxIYW5kbGVyO1xuXG4vKiogUmVuZGVyIHRvcCBsZXZlbCBjb21wb25lbnQgb25jZSBET00gaXMgcmVhZHkgKiovXG5kb2N1bWVudC5hZGRFdmVudExpc3RlbmVyKCdET01Db250ZW50TG9hZGVkJywgKCkgPT4ge1xuICAgIHByb2plY3Rvci5hcHBlbmQoZG9jdW1lbnQuYm9keSwgaW5pdGlhbFJlbmRlcik7XG4gICAgY2VsbFNvY2tldC5jb25uZWN0KCk7XG59KTtcblxuLy8gVEVTVElORzsgUkVNT1ZFXG5jb25zb2xlLmxvZygnTWFpbiBtb2R1bGUgbG9hZGVkJyk7XG4iLCIoZnVuY3Rpb24gKGdsb2JhbCwgZmFjdG9yeSkge1xuICAgIHR5cGVvZiBleHBvcnRzID09PSAnb2JqZWN0JyAmJiB0eXBlb2YgbW9kdWxlICE9PSAndW5kZWZpbmVkJyA/IGZhY3RvcnkoZXhwb3J0cykgOlxuICAgIHR5cGVvZiBkZWZpbmUgPT09ICdmdW5jdGlvbicgJiYgZGVmaW5lLmFtZCA/IGRlZmluZShbJ2V4cG9ydHMnXSwgZmFjdG9yeSkgOlxuICAgIChnbG9iYWwgPSBnbG9iYWwgfHwgc2VsZiwgZmFjdG9yeShnbG9iYWwubWFxdWV0dGUgPSB7fSkpO1xufSh0aGlzLCBmdW5jdGlvbiAoZXhwb3J0cykgeyAndXNlIHN0cmljdCc7XG5cbiAgICAvKiB0c2xpbnQ6ZGlzYWJsZSBuby1odHRwLXN0cmluZyAqL1xyXG4gICAgdmFyIE5BTUVTUEFDRV9XMyA9ICdodHRwOi8vd3d3LnczLm9yZy8nO1xyXG4gICAgLyogdHNsaW50OmVuYWJsZSBuby1odHRwLXN0cmluZyAqL1xyXG4gICAgdmFyIE5BTUVTUEFDRV9TVkcgPSBOQU1FU1BBQ0VfVzMgKyBcIjIwMDAvc3ZnXCI7XHJcbiAgICB2YXIgTkFNRVNQQUNFX1hMSU5LID0gTkFNRVNQQUNFX1czICsgXCIxOTk5L3hsaW5rXCI7XHJcbiAgICB2YXIgZW1wdHlBcnJheSA9IFtdO1xyXG4gICAgdmFyIGV4dGVuZCA9IGZ1bmN0aW9uIChiYXNlLCBvdmVycmlkZXMpIHtcclxuICAgICAgICB2YXIgcmVzdWx0ID0ge307XHJcbiAgICAgICAgT2JqZWN0LmtleXMoYmFzZSkuZm9yRWFjaChmdW5jdGlvbiAoa2V5KSB7XHJcbiAgICAgICAgICAgIHJlc3VsdFtrZXldID0gYmFzZVtrZXldO1xyXG4gICAgICAgIH0pO1xyXG4gICAgICAgIGlmIChvdmVycmlkZXMpIHtcclxuICAgICAgICAgICAgT2JqZWN0LmtleXMob3ZlcnJpZGVzKS5mb3JFYWNoKGZ1bmN0aW9uIChrZXkpIHtcclxuICAgICAgICAgICAgICAgIHJlc3VsdFtrZXldID0gb3ZlcnJpZGVzW2tleV07XHJcbiAgICAgICAgICAgIH0pO1xyXG4gICAgICAgIH1cclxuICAgICAgICByZXR1cm4gcmVzdWx0O1xyXG4gICAgfTtcclxuICAgIHZhciBzYW1lID0gZnVuY3Rpb24gKHZub2RlMSwgdm5vZGUyKSB7XHJcbiAgICAgICAgaWYgKHZub2RlMS52bm9kZVNlbGVjdG9yICE9PSB2bm9kZTIudm5vZGVTZWxlY3Rvcikge1xyXG4gICAgICAgICAgICByZXR1cm4gZmFsc2U7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGlmICh2bm9kZTEucHJvcGVydGllcyAmJiB2bm9kZTIucHJvcGVydGllcykge1xyXG4gICAgICAgICAgICBpZiAodm5vZGUxLnByb3BlcnRpZXMua2V5ICE9PSB2bm9kZTIucHJvcGVydGllcy5rZXkpIHtcclxuICAgICAgICAgICAgICAgIHJldHVybiBmYWxzZTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICByZXR1cm4gdm5vZGUxLnByb3BlcnRpZXMuYmluZCA9PT0gdm5vZGUyLnByb3BlcnRpZXMuYmluZDtcclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuICF2bm9kZTEucHJvcGVydGllcyAmJiAhdm5vZGUyLnByb3BlcnRpZXM7XHJcbiAgICB9O1xyXG4gICAgdmFyIGNoZWNrU3R5bGVWYWx1ZSA9IGZ1bmN0aW9uIChzdHlsZVZhbHVlKSB7XHJcbiAgICAgICAgaWYgKHR5cGVvZiBzdHlsZVZhbHVlICE9PSAnc3RyaW5nJykge1xyXG4gICAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ1N0eWxlIHZhbHVlcyBtdXN0IGJlIHN0cmluZ3MnKTtcclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgdmFyIGZpbmRJbmRleE9mQ2hpbGQgPSBmdW5jdGlvbiAoY2hpbGRyZW4sIHNhbWVBcywgc3RhcnQpIHtcclxuICAgICAgICBpZiAoc2FtZUFzLnZub2RlU2VsZWN0b3IgIT09ICcnKSB7XHJcbiAgICAgICAgICAgIC8vIE5ldmVyIHNjYW4gZm9yIHRleHQtbm9kZXNcclxuICAgICAgICAgICAgZm9yICh2YXIgaSA9IHN0YXJ0OyBpIDwgY2hpbGRyZW4ubGVuZ3RoOyBpKyspIHtcclxuICAgICAgICAgICAgICAgIGlmIChzYW1lKGNoaWxkcmVuW2ldLCBzYW1lQXMpKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIGk7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuIC0xO1xyXG4gICAgfTtcclxuICAgIHZhciBjaGVja0Rpc3Rpbmd1aXNoYWJsZSA9IGZ1bmN0aW9uIChjaGlsZE5vZGVzLCBpbmRleFRvQ2hlY2ssIHBhcmVudFZOb2RlLCBvcGVyYXRpb24pIHtcclxuICAgICAgICB2YXIgY2hpbGROb2RlID0gY2hpbGROb2Rlc1tpbmRleFRvQ2hlY2tdO1xyXG4gICAgICAgIGlmIChjaGlsZE5vZGUudm5vZGVTZWxlY3RvciA9PT0gJycpIHtcclxuICAgICAgICAgICAgcmV0dXJuOyAvLyBUZXh0IG5vZGVzIG5lZWQgbm90IGJlIGRpc3Rpbmd1aXNoYWJsZVxyXG4gICAgICAgIH1cclxuICAgICAgICB2YXIgcHJvcGVydGllcyA9IGNoaWxkTm9kZS5wcm9wZXJ0aWVzO1xyXG4gICAgICAgIHZhciBrZXkgPSBwcm9wZXJ0aWVzID8gKHByb3BlcnRpZXMua2V5ID09PSB1bmRlZmluZWQgPyBwcm9wZXJ0aWVzLmJpbmQgOiBwcm9wZXJ0aWVzLmtleSkgOiB1bmRlZmluZWQ7XHJcbiAgICAgICAgaWYgKCFrZXkpIHsgLy8gQSBrZXkgaXMganVzdCBhc3N1bWVkIHRvIGJlIHVuaXF1ZVxyXG4gICAgICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IGNoaWxkTm9kZXMubGVuZ3RoOyBpKyspIHtcclxuICAgICAgICAgICAgICAgIGlmIChpICE9PSBpbmRleFRvQ2hlY2spIHtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgbm9kZSA9IGNoaWxkTm9kZXNbaV07XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKHNhbWUobm9kZSwgY2hpbGROb2RlKSkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IocGFyZW50Vk5vZGUudm5vZGVTZWxlY3RvciArIFwiIGhhZCBhIFwiICsgY2hpbGROb2RlLnZub2RlU2VsZWN0b3IgKyBcIiBjaGlsZCBcIiArIChvcGVyYXRpb24gPT09ICdhZGRlZCcgPyBvcGVyYXRpb24gOiAncmVtb3ZlZCcpICsgXCIsIGJ1dCB0aGVyZSBpcyBub3cgbW9yZSB0aGFuIG9uZS4gWW91IG11c3QgYWRkIHVuaXF1ZSBrZXkgcHJvcGVydGllcyB0byBtYWtlIHRoZW0gZGlzdGluZ3Vpc2hhYmxlLlwiKTtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgdmFyIG5vZGVBZGRlZCA9IGZ1bmN0aW9uICh2Tm9kZSkge1xyXG4gICAgICAgIGlmICh2Tm9kZS5wcm9wZXJ0aWVzKSB7XHJcbiAgICAgICAgICAgIHZhciBlbnRlckFuaW1hdGlvbiA9IHZOb2RlLnByb3BlcnRpZXMuZW50ZXJBbmltYXRpb247XHJcbiAgICAgICAgICAgIGlmIChlbnRlckFuaW1hdGlvbikge1xyXG4gICAgICAgICAgICAgICAgZW50ZXJBbmltYXRpb24odk5vZGUuZG9tTm9kZSwgdk5vZGUucHJvcGVydGllcyk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgdmFyIHJlbW92ZWROb2RlcyA9IFtdO1xyXG4gICAgdmFyIHJlcXVlc3RlZElkbGVDYWxsYmFjayA9IGZhbHNlO1xyXG4gICAgdmFyIHZpc2l0UmVtb3ZlZE5vZGUgPSBmdW5jdGlvbiAobm9kZSkge1xyXG4gICAgICAgIChub2RlLmNoaWxkcmVuIHx8IFtdKS5mb3JFYWNoKHZpc2l0UmVtb3ZlZE5vZGUpO1xyXG4gICAgICAgIGlmIChub2RlLnByb3BlcnRpZXMgJiYgbm9kZS5wcm9wZXJ0aWVzLmFmdGVyUmVtb3ZlZCkge1xyXG4gICAgICAgICAgICBub2RlLnByb3BlcnRpZXMuYWZ0ZXJSZW1vdmVkLmFwcGx5KG5vZGUucHJvcGVydGllcy5iaW5kIHx8IG5vZGUucHJvcGVydGllcywgW25vZGUuZG9tTm9kZV0pO1xyXG4gICAgICAgIH1cclxuICAgIH07XHJcbiAgICB2YXIgcHJvY2Vzc1BlbmRpbmdOb2RlUmVtb3ZhbHMgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgcmVxdWVzdGVkSWRsZUNhbGxiYWNrID0gZmFsc2U7XHJcbiAgICAgICAgcmVtb3ZlZE5vZGVzLmZvckVhY2godmlzaXRSZW1vdmVkTm9kZSk7XHJcbiAgICAgICAgcmVtb3ZlZE5vZGVzLmxlbmd0aCA9IDA7XHJcbiAgICB9O1xyXG4gICAgdmFyIHNjaGVkdWxlTm9kZVJlbW92YWwgPSBmdW5jdGlvbiAodk5vZGUpIHtcclxuICAgICAgICByZW1vdmVkTm9kZXMucHVzaCh2Tm9kZSk7XHJcbiAgICAgICAgaWYgKCFyZXF1ZXN0ZWRJZGxlQ2FsbGJhY2spIHtcclxuICAgICAgICAgICAgcmVxdWVzdGVkSWRsZUNhbGxiYWNrID0gdHJ1ZTtcclxuICAgICAgICAgICAgaWYgKHR5cGVvZiB3aW5kb3cgIT09ICd1bmRlZmluZWQnICYmICdyZXF1ZXN0SWRsZUNhbGxiYWNrJyBpbiB3aW5kb3cpIHtcclxuICAgICAgICAgICAgICAgIHdpbmRvdy5yZXF1ZXN0SWRsZUNhbGxiYWNrKHByb2Nlc3NQZW5kaW5nTm9kZVJlbW92YWxzLCB7IHRpbWVvdXQ6IDE2IH0pO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgc2V0VGltZW91dChwcm9jZXNzUGVuZGluZ05vZGVSZW1vdmFscywgMTYpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIHZhciBub2RlVG9SZW1vdmUgPSBmdW5jdGlvbiAodk5vZGUpIHtcclxuICAgICAgICB2YXIgZG9tTm9kZSA9IHZOb2RlLmRvbU5vZGU7XHJcbiAgICAgICAgaWYgKHZOb2RlLnByb3BlcnRpZXMpIHtcclxuICAgICAgICAgICAgdmFyIGV4aXRBbmltYXRpb24gPSB2Tm9kZS5wcm9wZXJ0aWVzLmV4aXRBbmltYXRpb247XHJcbiAgICAgICAgICAgIGlmIChleGl0QW5pbWF0aW9uKSB7XHJcbiAgICAgICAgICAgICAgICBkb21Ob2RlLnN0eWxlLnBvaW50ZXJFdmVudHMgPSAnbm9uZSc7XHJcbiAgICAgICAgICAgICAgICB2YXIgcmVtb3ZlRG9tTm9kZSA9IGZ1bmN0aW9uICgpIHtcclxuICAgICAgICAgICAgICAgICAgICBpZiAoZG9tTm9kZS5wYXJlbnROb2RlKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGRvbU5vZGUucGFyZW50Tm9kZS5yZW1vdmVDaGlsZChkb21Ob2RlKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgc2NoZWR1bGVOb2RlUmVtb3ZhbCh2Tm9kZSk7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgfTtcclxuICAgICAgICAgICAgICAgIGV4aXRBbmltYXRpb24oZG9tTm9kZSwgcmVtb3ZlRG9tTm9kZSwgdk5vZGUucHJvcGVydGllcyk7XHJcbiAgICAgICAgICAgICAgICByZXR1cm47XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgaWYgKGRvbU5vZGUucGFyZW50Tm9kZSkge1xyXG4gICAgICAgICAgICBkb21Ob2RlLnBhcmVudE5vZGUucmVtb3ZlQ2hpbGQoZG9tTm9kZSk7XHJcbiAgICAgICAgICAgIHNjaGVkdWxlTm9kZVJlbW92YWwodk5vZGUpO1xyXG4gICAgICAgIH1cclxuICAgIH07XHJcbiAgICB2YXIgc2V0UHJvcGVydGllcyA9IGZ1bmN0aW9uIChkb21Ob2RlLCBwcm9wZXJ0aWVzLCBwcm9qZWN0aW9uT3B0aW9ucykge1xyXG4gICAgICAgIGlmICghcHJvcGVydGllcykge1xyXG4gICAgICAgICAgICByZXR1cm47XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHZhciBldmVudEhhbmRsZXJJbnRlcmNlcHRvciA9IHByb2plY3Rpb25PcHRpb25zLmV2ZW50SGFuZGxlckludGVyY2VwdG9yO1xyXG4gICAgICAgIHZhciBwcm9wTmFtZXMgPSBPYmplY3Qua2V5cyhwcm9wZXJ0aWVzKTtcclxuICAgICAgICB2YXIgcHJvcENvdW50ID0gcHJvcE5hbWVzLmxlbmd0aDtcclxuICAgICAgICB2YXIgX2xvb3BfMSA9IGZ1bmN0aW9uIChpKSB7XHJcbiAgICAgICAgICAgIHZhciBwcm9wTmFtZSA9IHByb3BOYW1lc1tpXTtcclxuICAgICAgICAgICAgdmFyIHByb3BWYWx1ZSA9IHByb3BlcnRpZXNbcHJvcE5hbWVdO1xyXG4gICAgICAgICAgICBpZiAocHJvcE5hbWUgPT09ICdjbGFzc05hbWUnKSB7XHJcbiAgICAgICAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ1Byb3BlcnR5IFwiY2xhc3NOYW1lXCIgaXMgbm90IHN1cHBvcnRlZCwgdXNlIFwiY2xhc3NcIi4nKTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBlbHNlIGlmIChwcm9wTmFtZSA9PT0gJ2NsYXNzJykge1xyXG4gICAgICAgICAgICAgICAgdG9nZ2xlQ2xhc3Nlcyhkb21Ob2RlLCBwcm9wVmFsdWUsIHRydWUpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGVsc2UgaWYgKHByb3BOYW1lID09PSAnY2xhc3NlcycpIHtcclxuICAgICAgICAgICAgICAgIC8vIG9iamVjdCB3aXRoIHN0cmluZyBrZXlzIGFuZCBib29sZWFuIHZhbHVlc1xyXG4gICAgICAgICAgICAgICAgdmFyIGNsYXNzTmFtZXMgPSBPYmplY3Qua2V5cyhwcm9wVmFsdWUpO1xyXG4gICAgICAgICAgICAgICAgdmFyIGNsYXNzTmFtZUNvdW50ID0gY2xhc3NOYW1lcy5sZW5ndGg7XHJcbiAgICAgICAgICAgICAgICBmb3IgKHZhciBqID0gMDsgaiA8IGNsYXNzTmFtZUNvdW50OyBqKyspIHtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgY2xhc3NOYW1lID0gY2xhc3NOYW1lc1tqXTtcclxuICAgICAgICAgICAgICAgICAgICBpZiAocHJvcFZhbHVlW2NsYXNzTmFtZV0pIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgZG9tTm9kZS5jbGFzc0xpc3QuYWRkKGNsYXNzTmFtZSk7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGVsc2UgaWYgKHByb3BOYW1lID09PSAnc3R5bGVzJykge1xyXG4gICAgICAgICAgICAgICAgLy8gb2JqZWN0IHdpdGggc3RyaW5nIGtleXMgYW5kIHN0cmluZyAoISkgdmFsdWVzXHJcbiAgICAgICAgICAgICAgICB2YXIgc3R5bGVOYW1lcyA9IE9iamVjdC5rZXlzKHByb3BWYWx1ZSk7XHJcbiAgICAgICAgICAgICAgICB2YXIgc3R5bGVDb3VudCA9IHN0eWxlTmFtZXMubGVuZ3RoO1xyXG4gICAgICAgICAgICAgICAgZm9yICh2YXIgaiA9IDA7IGogPCBzdHlsZUNvdW50OyBqKyspIHtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgc3R5bGVOYW1lID0gc3R5bGVOYW1lc1tqXTtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgc3R5bGVWYWx1ZSA9IHByb3BWYWx1ZVtzdHlsZU5hbWVdO1xyXG4gICAgICAgICAgICAgICAgICAgIGlmIChzdHlsZVZhbHVlKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGNoZWNrU3R5bGVWYWx1ZShzdHlsZVZhbHVlKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgcHJvamVjdGlvbk9wdGlvbnMuc3R5bGVBcHBseWVyKGRvbU5vZGUsIHN0eWxlTmFtZSwgc3R5bGVWYWx1ZSk7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGVsc2UgaWYgKHByb3BOYW1lICE9PSAna2V5JyAmJiBwcm9wVmFsdWUgIT09IG51bGwgJiYgcHJvcFZhbHVlICE9PSB1bmRlZmluZWQpIHtcclxuICAgICAgICAgICAgICAgIHZhciB0eXBlID0gdHlwZW9mIHByb3BWYWx1ZTtcclxuICAgICAgICAgICAgICAgIGlmICh0eXBlID09PSAnZnVuY3Rpb24nKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKHByb3BOYW1lLmxhc3RJbmRleE9mKCdvbicsIDApID09PSAwKSB7IC8vIGxhc3RJbmRleE9mKCwwKT09PTAgLT4gc3RhcnRzV2l0aFxyXG4gICAgICAgICAgICAgICAgICAgICAgICBpZiAoZXZlbnRIYW5kbGVySW50ZXJjZXB0b3IpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHByb3BWYWx1ZSA9IGV2ZW50SGFuZGxlckludGVyY2VwdG9yKHByb3BOYW1lLCBwcm9wVmFsdWUsIGRvbU5vZGUsIHByb3BlcnRpZXMpOyAvLyBpbnRlcmNlcHQgZXZlbnRoYW5kbGVyc1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGlmIChwcm9wTmFtZSA9PT0gJ29uaW5wdXQnKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAvKiB0c2xpbnQ6ZGlzYWJsZSBuby10aGlzLWtleXdvcmQgbm8taW52YWxpZC10aGlzIG9ubHktYXJyb3ctZnVuY3Rpb25zIG5vLXZvaWQtZXhwcmVzc2lvbiAqL1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgKGZ1bmN0aW9uICgpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAvLyByZWNvcmQgdGhlIGV2dC50YXJnZXQudmFsdWUsIGJlY2F1c2UgSUUgYW5kIEVkZ2Ugc29tZXRpbWVzIGRvIGEgcmVxdWVzdEFuaW1hdGlvbkZyYW1lIGJldHdlZW4gY2hhbmdpbmcgdmFsdWUgYW5kIHJ1bm5pbmcgb25pbnB1dFxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHZhciBvbGRQcm9wVmFsdWUgPSBwcm9wVmFsdWU7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcHJvcFZhbHVlID0gZnVuY3Rpb24gKGV2dCkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBvbGRQcm9wVmFsdWUuYXBwbHkodGhpcywgW2V2dF0pO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBldnQudGFyZ2V0WydvbmlucHV0LXZhbHVlJ10gPSBldnQudGFyZ2V0LnZhbHVlOyAvLyBtYXkgYmUgSFRNTFRleHRBcmVhRWxlbWVudCBhcyB3ZWxsXHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0oKSk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAvKiB0c2xpbnQ6ZW5hYmxlICovXHJcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICAgICAgZG9tTm9kZVtwcm9wTmFtZV0gPSBwcm9wVmFsdWU7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgZWxzZSBpZiAocHJvamVjdGlvbk9wdGlvbnMubmFtZXNwYWNlID09PSBOQU1FU1BBQ0VfU1ZHKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKHByb3BOYW1lID09PSAnaHJlZicpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgZG9tTm9kZS5zZXRBdHRyaWJ1dGVOUyhOQU1FU1BBQ0VfWExJTkssIHByb3BOYW1lLCBwcm9wVmFsdWUpO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgLy8gYWxsIFNWRyBhdHRyaWJ1dGVzIGFyZSByZWFkLW9ubHkgaW4gRE9NLCBzby4uLlxyXG4gICAgICAgICAgICAgICAgICAgICAgICBkb21Ob2RlLnNldEF0dHJpYnV0ZShwcm9wTmFtZSwgcHJvcFZhbHVlKTtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICBlbHNlIGlmICh0eXBlID09PSAnc3RyaW5nJyAmJiBwcm9wTmFtZSAhPT0gJ3ZhbHVlJyAmJiBwcm9wTmFtZSAhPT0gJ2lubmVySFRNTCcpIHtcclxuICAgICAgICAgICAgICAgICAgICBkb21Ob2RlLnNldEF0dHJpYnV0ZShwcm9wTmFtZSwgcHJvcFZhbHVlKTtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgICAgIGRvbU5vZGVbcHJvcE5hbWVdID0gcHJvcFZhbHVlO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfTtcclxuICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IHByb3BDb3VudDsgaSsrKSB7XHJcbiAgICAgICAgICAgIF9sb29wXzEoaSk7XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIHZhciBhZGRDaGlsZHJlbiA9IGZ1bmN0aW9uIChkb21Ob2RlLCBjaGlsZHJlbiwgcHJvamVjdGlvbk9wdGlvbnMpIHtcclxuICAgICAgICBpZiAoIWNoaWxkcmVuKSB7XHJcbiAgICAgICAgICAgIHJldHVybjtcclxuICAgICAgICB9XHJcbiAgICAgICAgZm9yICh2YXIgX2kgPSAwLCBjaGlsZHJlbl8xID0gY2hpbGRyZW47IF9pIDwgY2hpbGRyZW5fMS5sZW5ndGg7IF9pKyspIHtcclxuICAgICAgICAgICAgdmFyIGNoaWxkID0gY2hpbGRyZW5fMVtfaV07XHJcbiAgICAgICAgICAgIGNyZWF0ZURvbShjaGlsZCwgZG9tTm9kZSwgdW5kZWZpbmVkLCBwcm9qZWN0aW9uT3B0aW9ucyk7XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIHZhciBpbml0UHJvcGVydGllc0FuZENoaWxkcmVuID0gZnVuY3Rpb24gKGRvbU5vZGUsIHZub2RlLCBwcm9qZWN0aW9uT3B0aW9ucykge1xyXG4gICAgICAgIGFkZENoaWxkcmVuKGRvbU5vZGUsIHZub2RlLmNoaWxkcmVuLCBwcm9qZWN0aW9uT3B0aW9ucyk7IC8vIGNoaWxkcmVuIGJlZm9yZSBwcm9wZXJ0aWVzLCBuZWVkZWQgZm9yIHZhbHVlIHByb3BlcnR5IG9mIDxzZWxlY3Q+LlxyXG4gICAgICAgIGlmICh2bm9kZS50ZXh0KSB7XHJcbiAgICAgICAgICAgIGRvbU5vZGUudGV4dENvbnRlbnQgPSB2bm9kZS50ZXh0O1xyXG4gICAgICAgIH1cclxuICAgICAgICBzZXRQcm9wZXJ0aWVzKGRvbU5vZGUsIHZub2RlLnByb3BlcnRpZXMsIHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICBpZiAodm5vZGUucHJvcGVydGllcyAmJiB2bm9kZS5wcm9wZXJ0aWVzLmFmdGVyQ3JlYXRlKSB7XHJcbiAgICAgICAgICAgIHZub2RlLnByb3BlcnRpZXMuYWZ0ZXJDcmVhdGUuYXBwbHkodm5vZGUucHJvcGVydGllcy5iaW5kIHx8IHZub2RlLnByb3BlcnRpZXMsIFtkb21Ob2RlLCBwcm9qZWN0aW9uT3B0aW9ucywgdm5vZGUudm5vZGVTZWxlY3Rvciwgdm5vZGUucHJvcGVydGllcywgdm5vZGUuY2hpbGRyZW5dKTtcclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgdmFyIGNyZWF0ZURvbSA9IGZ1bmN0aW9uICh2bm9kZSwgcGFyZW50Tm9kZSwgaW5zZXJ0QmVmb3JlLCBwcm9qZWN0aW9uT3B0aW9ucykge1xyXG4gICAgICAgIHZhciBkb21Ob2RlO1xyXG4gICAgICAgIHZhciBzdGFydCA9IDA7XHJcbiAgICAgICAgdmFyIHZub2RlU2VsZWN0b3IgPSB2bm9kZS52bm9kZVNlbGVjdG9yO1xyXG4gICAgICAgIHZhciBkb2MgPSBwYXJlbnROb2RlLm93bmVyRG9jdW1lbnQ7XHJcbiAgICAgICAgaWYgKHZub2RlU2VsZWN0b3IgPT09ICcnKSB7XHJcbiAgICAgICAgICAgIGRvbU5vZGUgPSB2bm9kZS5kb21Ob2RlID0gZG9jLmNyZWF0ZVRleHROb2RlKHZub2RlLnRleHQpO1xyXG4gICAgICAgICAgICBpZiAoaW5zZXJ0QmVmb3JlICE9PSB1bmRlZmluZWQpIHtcclxuICAgICAgICAgICAgICAgIHBhcmVudE5vZGUuaW5zZXJ0QmVmb3JlKGRvbU5vZGUsIGluc2VydEJlZm9yZSk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICBwYXJlbnROb2RlLmFwcGVuZENoaWxkKGRvbU5vZGUpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8PSB2bm9kZVNlbGVjdG9yLmxlbmd0aDsgKytpKSB7XHJcbiAgICAgICAgICAgICAgICB2YXIgYyA9IHZub2RlU2VsZWN0b3IuY2hhckF0KGkpO1xyXG4gICAgICAgICAgICAgICAgaWYgKGkgPT09IHZub2RlU2VsZWN0b3IubGVuZ3RoIHx8IGMgPT09ICcuJyB8fCBjID09PSAnIycpIHtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgdHlwZSA9IHZub2RlU2VsZWN0b3IuY2hhckF0KHN0YXJ0IC0gMSk7XHJcbiAgICAgICAgICAgICAgICAgICAgdmFyIGZvdW5kID0gdm5vZGVTZWxlY3Rvci5zbGljZShzdGFydCwgaSk7XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKHR5cGUgPT09ICcuJykge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBkb21Ob2RlLmNsYXNzTGlzdC5hZGQoZm91bmQpO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICBlbHNlIGlmICh0eXBlID09PSAnIycpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgZG9tTm9kZS5pZCA9IGZvdW5kO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgaWYgKGZvdW5kID09PSAnc3ZnJykge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgcHJvamVjdGlvbk9wdGlvbnMgPSBleHRlbmQocHJvamVjdGlvbk9wdGlvbnMsIHsgbmFtZXNwYWNlOiBOQU1FU1BBQ0VfU1ZHIH0pO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGlmIChwcm9qZWN0aW9uT3B0aW9ucy5uYW1lc3BhY2UgIT09IHVuZGVmaW5lZCkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgZG9tTm9kZSA9IHZub2RlLmRvbU5vZGUgPSBkb2MuY3JlYXRlRWxlbWVudE5TKHByb2plY3Rpb25PcHRpb25zLm5hbWVzcGFjZSwgZm91bmQpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgZG9tTm9kZSA9IHZub2RlLmRvbU5vZGUgPSAodm5vZGUuZG9tTm9kZSB8fCBkb2MuY3JlYXRlRWxlbWVudChmb3VuZCkpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgaWYgKGZvdW5kID09PSAnaW5wdXQnICYmIHZub2RlLnByb3BlcnRpZXMgJiYgdm5vZGUucHJvcGVydGllcy50eXBlICE9PSB1bmRlZmluZWQpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAvLyBJRTggYW5kIG9sZGVyIGRvbid0IHN1cHBvcnQgc2V0dGluZyBpbnB1dCB0eXBlIGFmdGVyIHRoZSBET00gTm9kZSBoYXMgYmVlbiBhZGRlZCB0byB0aGUgZG9jdW1lbnRcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBkb21Ob2RlLnNldEF0dHJpYnV0ZSgndHlwZScsIHZub2RlLnByb3BlcnRpZXMudHlwZSk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICAgICAgaWYgKGluc2VydEJlZm9yZSAhPT0gdW5kZWZpbmVkKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBwYXJlbnROb2RlLmluc2VydEJlZm9yZShkb21Ob2RlLCBpbnNlcnRCZWZvcmUpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGVsc2UgaWYgKGRvbU5vZGUucGFyZW50Tm9kZSAhPT0gcGFyZW50Tm9kZSkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgcGFyZW50Tm9kZS5hcHBlbmRDaGlsZChkb21Ob2RlKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICBzdGFydCA9IGkgKyAxO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGluaXRQcm9wZXJ0aWVzQW5kQ2hpbGRyZW4oZG9tTm9kZSwgdm5vZGUsIHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgdmFyIHVwZGF0ZURvbTtcclxuICAgIC8qKlxyXG4gICAgICogQWRkcyBvciByZW1vdmVzIGNsYXNzZXMgZnJvbSBhbiBFbGVtZW50XHJcbiAgICAgKiBAcGFyYW0gZG9tTm9kZSB0aGUgZWxlbWVudFxyXG4gICAgICogQHBhcmFtIGNsYXNzZXMgYSBzdHJpbmcgc2VwYXJhdGVkIGxpc3Qgb2YgY2xhc3Nlc1xyXG4gICAgICogQHBhcmFtIG9uIHRydWUgbWVhbnMgYWRkIGNsYXNzZXMsIGZhbHNlIG1lYW5zIHJlbW92ZVxyXG4gICAgICovXHJcbiAgICB2YXIgdG9nZ2xlQ2xhc3NlcyA9IGZ1bmN0aW9uIChkb21Ob2RlLCBjbGFzc2VzLCBvbikge1xyXG4gICAgICAgIGlmICghY2xhc3Nlcykge1xyXG4gICAgICAgICAgICByZXR1cm47XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGNsYXNzZXMuc3BsaXQoJyAnKS5mb3JFYWNoKGZ1bmN0aW9uIChjbGFzc1RvVG9nZ2xlKSB7XHJcbiAgICAgICAgICAgIGlmIChjbGFzc1RvVG9nZ2xlKSB7XHJcbiAgICAgICAgICAgICAgICBkb21Ob2RlLmNsYXNzTGlzdC50b2dnbGUoY2xhc3NUb1RvZ2dsZSwgb24pO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfSk7XHJcbiAgICB9O1xyXG4gICAgdmFyIHVwZGF0ZVByb3BlcnRpZXMgPSBmdW5jdGlvbiAoZG9tTm9kZSwgcHJldmlvdXNQcm9wZXJ0aWVzLCBwcm9wZXJ0aWVzLCBwcm9qZWN0aW9uT3B0aW9ucykge1xyXG4gICAgICAgIGlmICghcHJvcGVydGllcykge1xyXG4gICAgICAgICAgICByZXR1cm47XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHZhciBwcm9wZXJ0aWVzVXBkYXRlZCA9IGZhbHNlO1xyXG4gICAgICAgIHZhciBwcm9wTmFtZXMgPSBPYmplY3Qua2V5cyhwcm9wZXJ0aWVzKTtcclxuICAgICAgICB2YXIgcHJvcENvdW50ID0gcHJvcE5hbWVzLmxlbmd0aDtcclxuICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IHByb3BDb3VudDsgaSsrKSB7XHJcbiAgICAgICAgICAgIHZhciBwcm9wTmFtZSA9IHByb3BOYW1lc1tpXTtcclxuICAgICAgICAgICAgLy8gYXNzdW1pbmcgdGhhdCBwcm9wZXJ0aWVzIHdpbGwgYmUgbnVsbGlmaWVkIGluc3RlYWQgb2YgbWlzc2luZyBpcyBieSBkZXNpZ25cclxuICAgICAgICAgICAgdmFyIHByb3BWYWx1ZSA9IHByb3BlcnRpZXNbcHJvcE5hbWVdO1xyXG4gICAgICAgICAgICB2YXIgcHJldmlvdXNWYWx1ZSA9IHByZXZpb3VzUHJvcGVydGllc1twcm9wTmFtZV07XHJcbiAgICAgICAgICAgIGlmIChwcm9wTmFtZSA9PT0gJ2NsYXNzJykge1xyXG4gICAgICAgICAgICAgICAgaWYgKHByZXZpb3VzVmFsdWUgIT09IHByb3BWYWx1ZSkge1xyXG4gICAgICAgICAgICAgICAgICAgIHRvZ2dsZUNsYXNzZXMoZG9tTm9kZSwgcHJldmlvdXNWYWx1ZSwgZmFsc2UpO1xyXG4gICAgICAgICAgICAgICAgICAgIHRvZ2dsZUNsYXNzZXMoZG9tTm9kZSwgcHJvcFZhbHVlLCB0cnVlKTtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBlbHNlIGlmIChwcm9wTmFtZSA9PT0gJ2NsYXNzZXMnKSB7XHJcbiAgICAgICAgICAgICAgICB2YXIgY2xhc3NMaXN0ID0gZG9tTm9kZS5jbGFzc0xpc3Q7XHJcbiAgICAgICAgICAgICAgICB2YXIgY2xhc3NOYW1lcyA9IE9iamVjdC5rZXlzKHByb3BWYWx1ZSk7XHJcbiAgICAgICAgICAgICAgICB2YXIgY2xhc3NOYW1lQ291bnQgPSBjbGFzc05hbWVzLmxlbmd0aDtcclxuICAgICAgICAgICAgICAgIGZvciAodmFyIGogPSAwOyBqIDwgY2xhc3NOYW1lQ291bnQ7IGorKykge1xyXG4gICAgICAgICAgICAgICAgICAgIHZhciBjbGFzc05hbWUgPSBjbGFzc05hbWVzW2pdO1xyXG4gICAgICAgICAgICAgICAgICAgIHZhciBvbiA9ICEhcHJvcFZhbHVlW2NsYXNzTmFtZV07XHJcbiAgICAgICAgICAgICAgICAgICAgdmFyIHByZXZpb3VzT24gPSAhIXByZXZpb3VzVmFsdWVbY2xhc3NOYW1lXTtcclxuICAgICAgICAgICAgICAgICAgICBpZiAob24gPT09IHByZXZpb3VzT24pIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgY29udGludWU7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgIHByb3BlcnRpZXNVcGRhdGVkID0gdHJ1ZTtcclxuICAgICAgICAgICAgICAgICAgICBpZiAob24pIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgY2xhc3NMaXN0LmFkZChjbGFzc05hbWUpO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgY2xhc3NMaXN0LnJlbW92ZShjbGFzc05hbWUpO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBlbHNlIGlmIChwcm9wTmFtZSA9PT0gJ3N0eWxlcycpIHtcclxuICAgICAgICAgICAgICAgIHZhciBzdHlsZU5hbWVzID0gT2JqZWN0LmtleXMocHJvcFZhbHVlKTtcclxuICAgICAgICAgICAgICAgIHZhciBzdHlsZUNvdW50ID0gc3R5bGVOYW1lcy5sZW5ndGg7XHJcbiAgICAgICAgICAgICAgICBmb3IgKHZhciBqID0gMDsgaiA8IHN0eWxlQ291bnQ7IGorKykge1xyXG4gICAgICAgICAgICAgICAgICAgIHZhciBzdHlsZU5hbWUgPSBzdHlsZU5hbWVzW2pdO1xyXG4gICAgICAgICAgICAgICAgICAgIHZhciBuZXdTdHlsZVZhbHVlID0gcHJvcFZhbHVlW3N0eWxlTmFtZV07XHJcbiAgICAgICAgICAgICAgICAgICAgdmFyIG9sZFN0eWxlVmFsdWUgPSBwcmV2aW91c1ZhbHVlW3N0eWxlTmFtZV07XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKG5ld1N0eWxlVmFsdWUgPT09IG9sZFN0eWxlVmFsdWUpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgY29udGludWU7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgIHByb3BlcnRpZXNVcGRhdGVkID0gdHJ1ZTtcclxuICAgICAgICAgICAgICAgICAgICBpZiAobmV3U3R5bGVWYWx1ZSkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBjaGVja1N0eWxlVmFsdWUobmV3U3R5bGVWYWx1ZSk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHByb2plY3Rpb25PcHRpb25zLnN0eWxlQXBwbHllcihkb21Ob2RlLCBzdHlsZU5hbWUsIG5ld1N0eWxlVmFsdWUpO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgcHJvamVjdGlvbk9wdGlvbnMuc3R5bGVBcHBseWVyKGRvbU5vZGUsIHN0eWxlTmFtZSwgJycpO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgIGlmICghcHJvcFZhbHVlICYmIHR5cGVvZiBwcmV2aW91c1ZhbHVlID09PSAnc3RyaW5nJykge1xyXG4gICAgICAgICAgICAgICAgICAgIHByb3BWYWx1ZSA9ICcnO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgaWYgKHByb3BOYW1lID09PSAndmFsdWUnKSB7IC8vIHZhbHVlIGNhbiBiZSBtYW5pcHVsYXRlZCBieSB0aGUgdXNlciBkaXJlY3RseSBhbmQgdXNpbmcgZXZlbnQucHJldmVudERlZmF1bHQoKSBpcyBub3QgYW4gb3B0aW9uXHJcbiAgICAgICAgICAgICAgICAgICAgdmFyIGRvbVZhbHVlID0gZG9tTm9kZVtwcm9wTmFtZV07XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKGRvbVZhbHVlICE9PSBwcm9wVmFsdWUgLy8gVGhlICd2YWx1ZScgaW4gdGhlIERPTSB0cmVlICE9PSBuZXdWYWx1ZVxyXG4gICAgICAgICAgICAgICAgICAgICAgICAmJiAoZG9tTm9kZVsnb25pbnB1dC12YWx1ZSddXHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICA/IGRvbVZhbHVlID09PSBkb21Ob2RlWydvbmlucHV0LXZhbHVlJ10gLy8gSWYgdGhlIGxhc3QgcmVwb3J0ZWQgdmFsdWUgdG8gJ29uaW5wdXQnIGRvZXMgbm90IG1hdGNoIGRvbVZhbHVlLCBkbyBub3RoaW5nIGFuZCB3YWl0IGZvciBvbmlucHV0XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICA6IHByb3BWYWx1ZSAhPT0gcHJldmlvdXNWYWx1ZSAvLyBPbmx5IHVwZGF0ZSB0aGUgdmFsdWUgaWYgdGhlIHZkb20gY2hhbmdlZFxyXG4gICAgICAgICAgICAgICAgICAgICAgICApKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIC8vIFRoZSBlZGdlIGNhc2VzIGFyZSBkZXNjcmliZWQgaW4gdGhlIHRlc3RzXHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGRvbU5vZGVbcHJvcE5hbWVdID0gcHJvcFZhbHVlOyAvLyBSZXNldCB0aGUgdmFsdWUsIGV2ZW4gaWYgdGhlIHZpcnR1YWwgRE9NIGRpZCBub3QgY2hhbmdlXHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGRvbU5vZGVbJ29uaW5wdXQtdmFsdWUnXSA9IHVuZGVmaW5lZDtcclxuICAgICAgICAgICAgICAgICAgICB9IC8vIGVsc2UgZG8gbm90IHVwZGF0ZSB0aGUgZG9tTm9kZSwgb3RoZXJ3aXNlIHRoZSBjdXJzb3IgcG9zaXRpb24gd291bGQgYmUgY2hhbmdlZFxyXG4gICAgICAgICAgICAgICAgICAgIGlmIChwcm9wVmFsdWUgIT09IHByZXZpb3VzVmFsdWUpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgcHJvcGVydGllc1VwZGF0ZWQgPSB0cnVlO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIGVsc2UgaWYgKHByb3BWYWx1ZSAhPT0gcHJldmlvdXNWYWx1ZSkge1xyXG4gICAgICAgICAgICAgICAgICAgIHZhciB0eXBlID0gdHlwZW9mIHByb3BWYWx1ZTtcclxuICAgICAgICAgICAgICAgICAgICBpZiAodHlwZSAhPT0gJ2Z1bmN0aW9uJyB8fCAhcHJvamVjdGlvbk9wdGlvbnMuZXZlbnRIYW5kbGVySW50ZXJjZXB0b3IpIHsgLy8gRnVuY3Rpb24gdXBkYXRlcyBhcmUgZXhwZWN0ZWQgdG8gYmUgaGFuZGxlZCBieSB0aGUgRXZlbnRIYW5kbGVySW50ZXJjZXB0b3JcclxuICAgICAgICAgICAgICAgICAgICAgICAgaWYgKHByb2plY3Rpb25PcHRpb25zLm5hbWVzcGFjZSA9PT0gTkFNRVNQQUNFX1NWRykge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgaWYgKHByb3BOYW1lID09PSAnaHJlZicpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBkb21Ob2RlLnNldEF0dHJpYnV0ZU5TKE5BTUVTUEFDRV9YTElOSywgcHJvcE5hbWUsIHByb3BWYWx1ZSk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAvLyBhbGwgU1ZHIGF0dHJpYnV0ZXMgYXJlIHJlYWQtb25seSBpbiBET00sIHNvLi4uXHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZG9tTm9kZS5zZXRBdHRyaWJ1dGUocHJvcE5hbWUsIHByb3BWYWx1ZSk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICAgICAgZWxzZSBpZiAodHlwZSA9PT0gJ3N0cmluZycgJiYgcHJvcE5hbWUgIT09ICdpbm5lckhUTUwnKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAocHJvcE5hbWUgPT09ICdyb2xlJyAmJiBwcm9wVmFsdWUgPT09ICcnKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZG9tTm9kZS5yZW1vdmVBdHRyaWJ1dGUocHJvcE5hbWUpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZG9tTm9kZS5zZXRBdHRyaWJ1dGUocHJvcE5hbWUsIHByb3BWYWx1ZSk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICAgICAgZWxzZSBpZiAoZG9tTm9kZVtwcm9wTmFtZV0gIT09IHByb3BWYWx1ZSkgeyAvLyBDb21wYXJpc29uIGlzIGhlcmUgZm9yIHNpZGUtZWZmZWN0cyBpbiBFZGdlIHdpdGggc2Nyb2xsTGVmdCBhbmQgc2Nyb2xsVG9wXHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBkb21Ob2RlW3Byb3BOYW1lXSA9IHByb3BWYWx1ZTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgICAgICBwcm9wZXJ0aWVzVXBkYXRlZCA9IHRydWU7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHJldHVybiBwcm9wZXJ0aWVzVXBkYXRlZDtcclxuICAgIH07XHJcbiAgICB2YXIgdXBkYXRlQ2hpbGRyZW4gPSBmdW5jdGlvbiAodm5vZGUsIGRvbU5vZGUsIG9sZENoaWxkcmVuLCBuZXdDaGlsZHJlbiwgcHJvamVjdGlvbk9wdGlvbnMpIHtcclxuICAgICAgICBpZiAob2xkQ2hpbGRyZW4gPT09IG5ld0NoaWxkcmVuKSB7XHJcbiAgICAgICAgICAgIHJldHVybiBmYWxzZTtcclxuICAgICAgICB9XHJcbiAgICAgICAgb2xkQ2hpbGRyZW4gPSBvbGRDaGlsZHJlbiB8fCBlbXB0eUFycmF5O1xyXG4gICAgICAgIG5ld0NoaWxkcmVuID0gbmV3Q2hpbGRyZW4gfHwgZW1wdHlBcnJheTtcclxuICAgICAgICB2YXIgb2xkQ2hpbGRyZW5MZW5ndGggPSBvbGRDaGlsZHJlbi5sZW5ndGg7XHJcbiAgICAgICAgdmFyIG5ld0NoaWxkcmVuTGVuZ3RoID0gbmV3Q2hpbGRyZW4ubGVuZ3RoO1xyXG4gICAgICAgIHZhciBvbGRJbmRleCA9IDA7XHJcbiAgICAgICAgdmFyIG5ld0luZGV4ID0gMDtcclxuICAgICAgICB2YXIgaTtcclxuICAgICAgICB2YXIgdGV4dFVwZGF0ZWQgPSBmYWxzZTtcclxuICAgICAgICB3aGlsZSAobmV3SW5kZXggPCBuZXdDaGlsZHJlbkxlbmd0aCkge1xyXG4gICAgICAgICAgICB2YXIgb2xkQ2hpbGQgPSAob2xkSW5kZXggPCBvbGRDaGlsZHJlbkxlbmd0aCkgPyBvbGRDaGlsZHJlbltvbGRJbmRleF0gOiB1bmRlZmluZWQ7XHJcbiAgICAgICAgICAgIHZhciBuZXdDaGlsZCA9IG5ld0NoaWxkcmVuW25ld0luZGV4XTtcclxuICAgICAgICAgICAgaWYgKG9sZENoaWxkICE9PSB1bmRlZmluZWQgJiYgc2FtZShvbGRDaGlsZCwgbmV3Q2hpbGQpKSB7XHJcbiAgICAgICAgICAgICAgICB0ZXh0VXBkYXRlZCA9IHVwZGF0ZURvbShvbGRDaGlsZCwgbmV3Q2hpbGQsIHByb2plY3Rpb25PcHRpb25zKSB8fCB0ZXh0VXBkYXRlZDtcclxuICAgICAgICAgICAgICAgIG9sZEluZGV4Kys7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICB2YXIgZmluZE9sZEluZGV4ID0gZmluZEluZGV4T2ZDaGlsZChvbGRDaGlsZHJlbiwgbmV3Q2hpbGQsIG9sZEluZGV4ICsgMSk7XHJcbiAgICAgICAgICAgICAgICBpZiAoZmluZE9sZEluZGV4ID49IDApIHtcclxuICAgICAgICAgICAgICAgICAgICAvLyBSZW1vdmUgcHJlY2VkaW5nIG1pc3NpbmcgY2hpbGRyZW5cclxuICAgICAgICAgICAgICAgICAgICBmb3IgKGkgPSBvbGRJbmRleDsgaSA8IGZpbmRPbGRJbmRleDsgaSsrKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIG5vZGVUb1JlbW92ZShvbGRDaGlsZHJlbltpXSk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGNoZWNrRGlzdGluZ3Vpc2hhYmxlKG9sZENoaWxkcmVuLCBpLCB2bm9kZSwgJ3JlbW92ZWQnKTtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgdGV4dFVwZGF0ZWQgPSB1cGRhdGVEb20ob2xkQ2hpbGRyZW5bZmluZE9sZEluZGV4XSwgbmV3Q2hpbGQsIHByb2plY3Rpb25PcHRpb25zKSB8fCB0ZXh0VXBkYXRlZDtcclxuICAgICAgICAgICAgICAgICAgICBvbGRJbmRleCA9IGZpbmRPbGRJbmRleCArIDE7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgICAvLyBOZXcgY2hpbGRcclxuICAgICAgICAgICAgICAgICAgICBjcmVhdGVEb20obmV3Q2hpbGQsIGRvbU5vZGUsIChvbGRJbmRleCA8IG9sZENoaWxkcmVuTGVuZ3RoKSA/IG9sZENoaWxkcmVuW29sZEluZGV4XS5kb21Ob2RlIDogdW5kZWZpbmVkLCBwcm9qZWN0aW9uT3B0aW9ucyk7XHJcbiAgICAgICAgICAgICAgICAgICAgbm9kZUFkZGVkKG5ld0NoaWxkKTtcclxuICAgICAgICAgICAgICAgICAgICBjaGVja0Rpc3Rpbmd1aXNoYWJsZShuZXdDaGlsZHJlbiwgbmV3SW5kZXgsIHZub2RlLCAnYWRkZWQnKTtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBuZXdJbmRleCsrO1xyXG4gICAgICAgIH1cclxuICAgICAgICBpZiAob2xkQ2hpbGRyZW5MZW5ndGggPiBvbGRJbmRleCkge1xyXG4gICAgICAgICAgICAvLyBSZW1vdmUgY2hpbGQgZnJhZ21lbnRzXHJcbiAgICAgICAgICAgIGZvciAoaSA9IG9sZEluZGV4OyBpIDwgb2xkQ2hpbGRyZW5MZW5ndGg7IGkrKykge1xyXG4gICAgICAgICAgICAgICAgbm9kZVRvUmVtb3ZlKG9sZENoaWxkcmVuW2ldKTtcclxuICAgICAgICAgICAgICAgIGNoZWNrRGlzdGluZ3Vpc2hhYmxlKG9sZENoaWxkcmVuLCBpLCB2bm9kZSwgJ3JlbW92ZWQnKTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgICAgICByZXR1cm4gdGV4dFVwZGF0ZWQ7XHJcbiAgICB9O1xyXG4gICAgdXBkYXRlRG9tID0gZnVuY3Rpb24gKHByZXZpb3VzLCB2bm9kZSwgcHJvamVjdGlvbk9wdGlvbnMpIHtcclxuICAgICAgICB2YXIgZG9tTm9kZSA9IHByZXZpb3VzLmRvbU5vZGU7XHJcbiAgICAgICAgdmFyIHRleHRVcGRhdGVkID0gZmFsc2U7XHJcbiAgICAgICAgaWYgKHByZXZpb3VzID09PSB2bm9kZSkge1xyXG4gICAgICAgICAgICByZXR1cm4gZmFsc2U7IC8vIEJ5IGNvbnRyYWN0LCBWTm9kZSBvYmplY3RzIG1heSBub3QgYmUgbW9kaWZpZWQgYW55bW9yZSBhZnRlciBwYXNzaW5nIHRoZW0gdG8gbWFxdWV0dGVcclxuICAgICAgICB9XHJcbiAgICAgICAgdmFyIHVwZGF0ZWQgPSBmYWxzZTtcclxuICAgICAgICBpZiAodm5vZGUudm5vZGVTZWxlY3RvciA9PT0gJycpIHtcclxuICAgICAgICAgICAgaWYgKHZub2RlLnRleHQgIT09IHByZXZpb3VzLnRleHQpIHtcclxuICAgICAgICAgICAgICAgIHZhciBuZXdUZXh0Tm9kZSA9IGRvbU5vZGUub3duZXJEb2N1bWVudC5jcmVhdGVUZXh0Tm9kZSh2bm9kZS50ZXh0KTtcclxuICAgICAgICAgICAgICAgIGRvbU5vZGUucGFyZW50Tm9kZS5yZXBsYWNlQ2hpbGQobmV3VGV4dE5vZGUsIGRvbU5vZGUpO1xyXG4gICAgICAgICAgICAgICAgdm5vZGUuZG9tTm9kZSA9IG5ld1RleHROb2RlO1xyXG4gICAgICAgICAgICAgICAgdGV4dFVwZGF0ZWQgPSB0cnVlO1xyXG4gICAgICAgICAgICAgICAgcmV0dXJuIHRleHRVcGRhdGVkO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIHZub2RlLmRvbU5vZGUgPSBkb21Ob2RlO1xyXG4gICAgICAgIH1cclxuICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgaWYgKHZub2RlLnZub2RlU2VsZWN0b3IubGFzdEluZGV4T2YoJ3N2ZycsIDApID09PSAwKSB7IC8vIGxhc3RJbmRleE9mKG5lZWRsZSwwKT09PTAgbWVhbnMgU3RhcnRzV2l0aFxyXG4gICAgICAgICAgICAgICAgcHJvamVjdGlvbk9wdGlvbnMgPSBleHRlbmQocHJvamVjdGlvbk9wdGlvbnMsIHsgbmFtZXNwYWNlOiBOQU1FU1BBQ0VfU1ZHIH0pO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGlmIChwcmV2aW91cy50ZXh0ICE9PSB2bm9kZS50ZXh0KSB7XHJcbiAgICAgICAgICAgICAgICB1cGRhdGVkID0gdHJ1ZTtcclxuICAgICAgICAgICAgICAgIGlmICh2bm9kZS50ZXh0ID09PSB1bmRlZmluZWQpIHtcclxuICAgICAgICAgICAgICAgICAgICBkb21Ob2RlLnJlbW92ZUNoaWxkKGRvbU5vZGUuZmlyc3RDaGlsZCk7IC8vIHRoZSBvbmx5IHRleHRub2RlIHByZXN1bWFibHlcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgICAgIGRvbU5vZGUudGV4dENvbnRlbnQgPSB2bm9kZS50ZXh0O1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIHZub2RlLmRvbU5vZGUgPSBkb21Ob2RlO1xyXG4gICAgICAgICAgICB1cGRhdGVkID0gdXBkYXRlQ2hpbGRyZW4odm5vZGUsIGRvbU5vZGUsIHByZXZpb3VzLmNoaWxkcmVuLCB2bm9kZS5jaGlsZHJlbiwgcHJvamVjdGlvbk9wdGlvbnMpIHx8IHVwZGF0ZWQ7XHJcbiAgICAgICAgICAgIHVwZGF0ZWQgPSB1cGRhdGVQcm9wZXJ0aWVzKGRvbU5vZGUsIHByZXZpb3VzLnByb3BlcnRpZXMsIHZub2RlLnByb3BlcnRpZXMsIHByb2plY3Rpb25PcHRpb25zKSB8fCB1cGRhdGVkO1xyXG4gICAgICAgICAgICBpZiAodm5vZGUucHJvcGVydGllcyAmJiB2bm9kZS5wcm9wZXJ0aWVzLmFmdGVyVXBkYXRlKSB7XHJcbiAgICAgICAgICAgICAgICB2bm9kZS5wcm9wZXJ0aWVzLmFmdGVyVXBkYXRlLmFwcGx5KHZub2RlLnByb3BlcnRpZXMuYmluZCB8fCB2bm9kZS5wcm9wZXJ0aWVzLCBbZG9tTm9kZSwgcHJvamVjdGlvbk9wdGlvbnMsIHZub2RlLnZub2RlU2VsZWN0b3IsIHZub2RlLnByb3BlcnRpZXMsIHZub2RlLmNoaWxkcmVuXSk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgaWYgKHVwZGF0ZWQgJiYgdm5vZGUucHJvcGVydGllcyAmJiB2bm9kZS5wcm9wZXJ0aWVzLnVwZGF0ZUFuaW1hdGlvbikge1xyXG4gICAgICAgICAgICB2bm9kZS5wcm9wZXJ0aWVzLnVwZGF0ZUFuaW1hdGlvbihkb21Ob2RlLCB2bm9kZS5wcm9wZXJ0aWVzLCBwcmV2aW91cy5wcm9wZXJ0aWVzKTtcclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuIHRleHRVcGRhdGVkO1xyXG4gICAgfTtcclxuICAgIHZhciBjcmVhdGVQcm9qZWN0aW9uID0gZnVuY3Rpb24gKHZub2RlLCBwcm9qZWN0aW9uT3B0aW9ucykge1xyXG4gICAgICAgIHJldHVybiB7XHJcbiAgICAgICAgICAgIGdldExhc3RSZW5kZXI6IGZ1bmN0aW9uICgpIHsgcmV0dXJuIHZub2RlOyB9LFxyXG4gICAgICAgICAgICB1cGRhdGU6IGZ1bmN0aW9uICh1cGRhdGVkVm5vZGUpIHtcclxuICAgICAgICAgICAgICAgIGlmICh2bm9kZS52bm9kZVNlbGVjdG9yICE9PSB1cGRhdGVkVm5vZGUudm5vZGVTZWxlY3Rvcikge1xyXG4gICAgICAgICAgICAgICAgICAgIHRocm93IG5ldyBFcnJvcignVGhlIHNlbGVjdG9yIGZvciB0aGUgcm9vdCBWTm9kZSBtYXkgbm90IGJlIGNoYW5nZWQuIChjb25zaWRlciB1c2luZyBkb20ubWVyZ2UgYW5kIGFkZCBvbmUgZXh0cmEgbGV2ZWwgdG8gdGhlIHZpcnR1YWwgRE9NKScpO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgdmFyIHByZXZpb3VzVk5vZGUgPSB2bm9kZTtcclxuICAgICAgICAgICAgICAgIHZub2RlID0gdXBkYXRlZFZub2RlO1xyXG4gICAgICAgICAgICAgICAgdXBkYXRlRG9tKHByZXZpb3VzVk5vZGUsIHVwZGF0ZWRWbm9kZSwgcHJvamVjdGlvbk9wdGlvbnMpO1xyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICBkb21Ob2RlOiB2bm9kZS5kb21Ob2RlXHJcbiAgICAgICAgfTtcclxuICAgIH07XG5cbiAgICB2YXIgREVGQVVMVF9QUk9KRUNUSU9OX09QVElPTlMgPSB7XHJcbiAgICAgICAgbmFtZXNwYWNlOiB1bmRlZmluZWQsXHJcbiAgICAgICAgcGVyZm9ybWFuY2VMb2dnZXI6IGZ1bmN0aW9uICgpIHsgcmV0dXJuIHVuZGVmaW5lZDsgfSxcclxuICAgICAgICBldmVudEhhbmRsZXJJbnRlcmNlcHRvcjogdW5kZWZpbmVkLFxyXG4gICAgICAgIHN0eWxlQXBwbHllcjogZnVuY3Rpb24gKGRvbU5vZGUsIHN0eWxlTmFtZSwgdmFsdWUpIHtcclxuICAgICAgICAgICAgLy8gUHJvdmlkZXMgYSBob29rIHRvIGFkZCB2ZW5kb3IgcHJlZml4ZXMgZm9yIGJyb3dzZXJzIHRoYXQgc3RpbGwgbmVlZCBpdC5cclxuICAgICAgICAgICAgZG9tTm9kZS5zdHlsZVtzdHlsZU5hbWVdID0gdmFsdWU7XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIHZhciBhcHBseURlZmF1bHRQcm9qZWN0aW9uT3B0aW9ucyA9IGZ1bmN0aW9uIChwcm9qZWN0b3JPcHRpb25zKSB7XHJcbiAgICAgICAgcmV0dXJuIGV4dGVuZChERUZBVUxUX1BST0pFQ1RJT05fT1BUSU9OUywgcHJvamVjdG9yT3B0aW9ucyk7XHJcbiAgICB9O1xyXG4gICAgdmFyIGRvbSA9IHtcclxuICAgICAgICAvKipcclxuICAgICAgICAgKiBDcmVhdGVzIGEgcmVhbCBET00gdHJlZSBmcm9tIGB2bm9kZWAuIFRoZSBbW1Byb2plY3Rpb25dXSBvYmplY3QgcmV0dXJuZWQgd2lsbCBjb250YWluIHRoZSByZXN1bHRpbmcgRE9NIE5vZGUgaW5cclxuICAgICAgICAgKiBpdHMgW1tQcm9qZWN0aW9uLmRvbU5vZGV8ZG9tTm9kZV1dIHByb3BlcnR5LlxyXG4gICAgICAgICAqIFRoaXMgaXMgYSBsb3ctbGV2ZWwgbWV0aG9kLiBVc2VycyB3aWxsIHR5cGljYWxseSB1c2UgYSBbW1Byb2plY3Rvcl1dIGluc3RlYWQuXHJcbiAgICAgICAgICogQHBhcmFtIHZub2RlIC0gVGhlIHJvb3Qgb2YgdGhlIHZpcnR1YWwgRE9NIHRyZWUgdGhhdCB3YXMgY3JlYXRlZCB1c2luZyB0aGUgW1toXV0gZnVuY3Rpb24uIE5PVEU6IFtbVk5vZGVdXVxyXG4gICAgICAgICAqIG9iamVjdHMgbWF5IG9ubHkgYmUgcmVuZGVyZWQgb25jZS5cclxuICAgICAgICAgKiBAcGFyYW0gcHJvamVjdGlvbk9wdGlvbnMgLSBPcHRpb25zIHRvIGJlIHVzZWQgdG8gY3JlYXRlIGFuZCB1cGRhdGUgdGhlIHByb2plY3Rpb24uXHJcbiAgICAgICAgICogQHJldHVybnMgVGhlIFtbUHJvamVjdGlvbl1dIHdoaWNoIGFsc28gY29udGFpbnMgdGhlIERPTSBOb2RlIHRoYXQgd2FzIGNyZWF0ZWQuXHJcbiAgICAgICAgICovXHJcbiAgICAgICAgY3JlYXRlOiBmdW5jdGlvbiAodm5vZGUsIHByb2plY3Rpb25PcHRpb25zKSB7XHJcbiAgICAgICAgICAgIHByb2plY3Rpb25PcHRpb25zID0gYXBwbHlEZWZhdWx0UHJvamVjdGlvbk9wdGlvbnMocHJvamVjdGlvbk9wdGlvbnMpO1xyXG4gICAgICAgICAgICBjcmVhdGVEb20odm5vZGUsIGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2RpdicpLCB1bmRlZmluZWQsIHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICAgICAgcmV0dXJuIGNyZWF0ZVByb2plY3Rpb24odm5vZGUsIHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICB9LFxyXG4gICAgICAgIC8qKlxyXG4gICAgICAgICAqIEFwcGVuZHMgYSBuZXcgY2hpbGQgbm9kZSB0byB0aGUgRE9NIHdoaWNoIGlzIGdlbmVyYXRlZCBmcm9tIGEgW1tWTm9kZV1dLlxyXG4gICAgICAgICAqIFRoaXMgaXMgYSBsb3ctbGV2ZWwgbWV0aG9kLiBVc2VycyB3aWxsIHR5cGljYWxseSB1c2UgYSBbW1Byb2plY3Rvcl1dIGluc3RlYWQuXHJcbiAgICAgICAgICogQHBhcmFtIHBhcmVudE5vZGUgLSBUaGUgcGFyZW50IG5vZGUgZm9yIHRoZSBuZXcgY2hpbGQgbm9kZS5cclxuICAgICAgICAgKiBAcGFyYW0gdm5vZGUgLSBUaGUgcm9vdCBvZiB0aGUgdmlydHVhbCBET00gdHJlZSB0aGF0IHdhcyBjcmVhdGVkIHVzaW5nIHRoZSBbW2hdXSBmdW5jdGlvbi4gTk9URTogW1tWTm9kZV1dXHJcbiAgICAgICAgICogb2JqZWN0cyBtYXkgb25seSBiZSByZW5kZXJlZCBvbmNlLlxyXG4gICAgICAgICAqIEBwYXJhbSBwcm9qZWN0aW9uT3B0aW9ucyAtIE9wdGlvbnMgdG8gYmUgdXNlZCB0byBjcmVhdGUgYW5kIHVwZGF0ZSB0aGUgW1tQcm9qZWN0aW9uXV0uXHJcbiAgICAgICAgICogQHJldHVybnMgVGhlIFtbUHJvamVjdGlvbl1dIHRoYXQgd2FzIGNyZWF0ZWQuXHJcbiAgICAgICAgICovXHJcbiAgICAgICAgYXBwZW5kOiBmdW5jdGlvbiAocGFyZW50Tm9kZSwgdm5vZGUsIHByb2plY3Rpb25PcHRpb25zKSB7XHJcbiAgICAgICAgICAgIHByb2plY3Rpb25PcHRpb25zID0gYXBwbHlEZWZhdWx0UHJvamVjdGlvbk9wdGlvbnMocHJvamVjdGlvbk9wdGlvbnMpO1xyXG4gICAgICAgICAgICBjcmVhdGVEb20odm5vZGUsIHBhcmVudE5vZGUsIHVuZGVmaW5lZCwgcHJvamVjdGlvbk9wdGlvbnMpO1xyXG4gICAgICAgICAgICByZXR1cm4gY3JlYXRlUHJvamVjdGlvbih2bm9kZSwgcHJvamVjdGlvbk9wdGlvbnMpO1xyXG4gICAgICAgIH0sXHJcbiAgICAgICAgLyoqXHJcbiAgICAgICAgICogSW5zZXJ0cyBhIG5ldyBET00gbm9kZSB3aGljaCBpcyBnZW5lcmF0ZWQgZnJvbSBhIFtbVk5vZGVdXS5cclxuICAgICAgICAgKiBUaGlzIGlzIGEgbG93LWxldmVsIG1ldGhvZC4gVXNlcnMgd2lsIHR5cGljYWxseSB1c2UgYSBbW1Byb2plY3Rvcl1dIGluc3RlYWQuXHJcbiAgICAgICAgICogQHBhcmFtIGJlZm9yZU5vZGUgLSBUaGUgbm9kZSB0aGF0IHRoZSBET00gTm9kZSBpcyBpbnNlcnRlZCBiZWZvcmUuXHJcbiAgICAgICAgICogQHBhcmFtIHZub2RlIC0gVGhlIHJvb3Qgb2YgdGhlIHZpcnR1YWwgRE9NIHRyZWUgdGhhdCB3YXMgY3JlYXRlZCB1c2luZyB0aGUgW1toXV0gZnVuY3Rpb24uXHJcbiAgICAgICAgICogTk9URTogW1tWTm9kZV1dIG9iamVjdHMgbWF5IG9ubHkgYmUgcmVuZGVyZWQgb25jZS5cclxuICAgICAgICAgKiBAcGFyYW0gcHJvamVjdGlvbk9wdGlvbnMgLSBPcHRpb25zIHRvIGJlIHVzZWQgdG8gY3JlYXRlIGFuZCB1cGRhdGUgdGhlIHByb2plY3Rpb24sIHNlZSBbW2NyZWF0ZVByb2plY3Rvcl1dLlxyXG4gICAgICAgICAqIEByZXR1cm5zIFRoZSBbW1Byb2plY3Rpb25dXSB0aGF0IHdhcyBjcmVhdGVkLlxyXG4gICAgICAgICAqL1xyXG4gICAgICAgIGluc2VydEJlZm9yZTogZnVuY3Rpb24gKGJlZm9yZU5vZGUsIHZub2RlLCBwcm9qZWN0aW9uT3B0aW9ucykge1xyXG4gICAgICAgICAgICBwcm9qZWN0aW9uT3B0aW9ucyA9IGFwcGx5RGVmYXVsdFByb2plY3Rpb25PcHRpb25zKHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICAgICAgY3JlYXRlRG9tKHZub2RlLCBiZWZvcmVOb2RlLnBhcmVudE5vZGUsIGJlZm9yZU5vZGUsIHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICAgICAgcmV0dXJuIGNyZWF0ZVByb2plY3Rpb24odm5vZGUsIHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICB9LFxyXG4gICAgICAgIC8qKlxyXG4gICAgICAgICAqIE1lcmdlcyBhIG5ldyBET00gbm9kZSB3aGljaCBpcyBnZW5lcmF0ZWQgZnJvbSBhIFtbVk5vZGVdXSB3aXRoIGFuIGV4aXN0aW5nIERPTSBOb2RlLlxyXG4gICAgICAgICAqIFRoaXMgbWVhbnMgdGhhdCB0aGUgdmlydHVhbCBET00gYW5kIHRoZSByZWFsIERPTSB3aWxsIGhhdmUgb25lIG92ZXJsYXBwaW5nIGVsZW1lbnQuXHJcbiAgICAgICAgICogVGhlcmVmb3JlIHRoZSBzZWxlY3RvciBmb3IgdGhlIHJvb3QgW1tWTm9kZV1dIHdpbGwgYmUgaWdub3JlZCwgYnV0IGl0cyBwcm9wZXJ0aWVzIGFuZCBjaGlsZHJlbiB3aWxsIGJlIGFwcGxpZWQgdG8gdGhlIEVsZW1lbnQgcHJvdmlkZWQuXHJcbiAgICAgICAgICogVGhpcyBpcyBhIGxvdy1sZXZlbCBtZXRob2QuIFVzZXJzIHdpbCB0eXBpY2FsbHkgdXNlIGEgW1tQcm9qZWN0b3JdXSBpbnN0ZWFkLlxyXG4gICAgICAgICAqIEBwYXJhbSBlbGVtZW50IC0gVGhlIGV4aXN0aW5nIGVsZW1lbnQgdG8gYWRvcHQgYXMgdGhlIHJvb3Qgb2YgdGhlIG5ldyB2aXJ0dWFsIERPTS4gRXhpc3RpbmcgYXR0cmlidXRlcyBhbmQgY2hpbGQgbm9kZXMgYXJlIHByZXNlcnZlZC5cclxuICAgICAgICAgKiBAcGFyYW0gdm5vZGUgLSBUaGUgcm9vdCBvZiB0aGUgdmlydHVhbCBET00gdHJlZSB0aGF0IHdhcyBjcmVhdGVkIHVzaW5nIHRoZSBbW2hdXSBmdW5jdGlvbi4gTk9URTogW1tWTm9kZV1dIG9iamVjdHNcclxuICAgICAgICAgKiBtYXkgb25seSBiZSByZW5kZXJlZCBvbmNlLlxyXG4gICAgICAgICAqIEBwYXJhbSBwcm9qZWN0aW9uT3B0aW9ucyAtIE9wdGlvbnMgdG8gYmUgdXNlZCB0byBjcmVhdGUgYW5kIHVwZGF0ZSB0aGUgcHJvamVjdGlvbiwgc2VlIFtbY3JlYXRlUHJvamVjdG9yXV0uXHJcbiAgICAgICAgICogQHJldHVybnMgVGhlIFtbUHJvamVjdGlvbl1dIHRoYXQgd2FzIGNyZWF0ZWQuXHJcbiAgICAgICAgICovXHJcbiAgICAgICAgbWVyZ2U6IGZ1bmN0aW9uIChlbGVtZW50LCB2bm9kZSwgcHJvamVjdGlvbk9wdGlvbnMpIHtcclxuICAgICAgICAgICAgcHJvamVjdGlvbk9wdGlvbnMgPSBhcHBseURlZmF1bHRQcm9qZWN0aW9uT3B0aW9ucyhwcm9qZWN0aW9uT3B0aW9ucyk7XHJcbiAgICAgICAgICAgIHZub2RlLmRvbU5vZGUgPSBlbGVtZW50O1xyXG4gICAgICAgICAgICBpbml0UHJvcGVydGllc0FuZENoaWxkcmVuKGVsZW1lbnQsIHZub2RlLCBwcm9qZWN0aW9uT3B0aW9ucyk7XHJcbiAgICAgICAgICAgIHJldHVybiBjcmVhdGVQcm9qZWN0aW9uKHZub2RlLCBwcm9qZWN0aW9uT3B0aW9ucyk7XHJcbiAgICAgICAgfSxcclxuICAgICAgICAvKipcclxuICAgICAgICAgKiBSZXBsYWNlcyBhbiBleGlzdGluZyBET00gbm9kZSB3aXRoIGEgbm9kZSBnZW5lcmF0ZWQgZnJvbSBhIFtbVk5vZGVdXS5cclxuICAgICAgICAgKiBUaGlzIGlzIGEgbG93LWxldmVsIG1ldGhvZC4gVXNlcnMgd2lsbCB0eXBpY2FsbHkgdXNlIGEgW1tQcm9qZWN0b3JdXSBpbnN0ZWFkLlxyXG4gICAgICAgICAqIEBwYXJhbSBlbGVtZW50IC0gVGhlIG5vZGUgZm9yIHRoZSBbW1ZOb2RlXV0gdG8gcmVwbGFjZS5cclxuICAgICAgICAgKiBAcGFyYW0gdm5vZGUgLSBUaGUgcm9vdCBvZiB0aGUgdmlydHVhbCBET00gdHJlZSB0aGF0IHdhcyBjcmVhdGVkIHVzaW5nIHRoZSBbW2hdXSBmdW5jdGlvbi4gTk9URTogW1tWTm9kZV1dXHJcbiAgICAgICAgICogb2JqZWN0cyBtYXkgb25seSBiZSByZW5kZXJlZCBvbmNlLlxyXG4gICAgICAgICAqIEBwYXJhbSBwcm9qZWN0aW9uT3B0aW9ucyAtIE9wdGlvbnMgdG8gYmUgdXNlZCB0byBjcmVhdGUgYW5kIHVwZGF0ZSB0aGUgW1tQcm9qZWN0aW9uXV0uXHJcbiAgICAgICAgICogQHJldHVybnMgVGhlIFtbUHJvamVjdGlvbl1dIHRoYXQgd2FzIGNyZWF0ZWQuXHJcbiAgICAgICAgICovXHJcbiAgICAgICAgcmVwbGFjZTogZnVuY3Rpb24gKGVsZW1lbnQsIHZub2RlLCBwcm9qZWN0aW9uT3B0aW9ucykge1xyXG4gICAgICAgICAgICBwcm9qZWN0aW9uT3B0aW9ucyA9IGFwcGx5RGVmYXVsdFByb2plY3Rpb25PcHRpb25zKHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICAgICAgY3JlYXRlRG9tKHZub2RlLCBlbGVtZW50LnBhcmVudE5vZGUsIGVsZW1lbnQsIHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICAgICAgZWxlbWVudC5wYXJlbnROb2RlLnJlbW92ZUNoaWxkKGVsZW1lbnQpO1xyXG4gICAgICAgICAgICByZXR1cm4gY3JlYXRlUHJvamVjdGlvbih2bm9kZSwgcHJvamVjdGlvbk9wdGlvbnMpO1xyXG4gICAgICAgIH1cclxuICAgIH07XG5cbiAgICAvKiB0c2xpbnQ6ZGlzYWJsZSBmdW5jdGlvbi1uYW1lICovXHJcbiAgICB2YXIgdG9UZXh0Vk5vZGUgPSBmdW5jdGlvbiAoZGF0YSkge1xyXG4gICAgICAgIHJldHVybiB7XHJcbiAgICAgICAgICAgIHZub2RlU2VsZWN0b3I6ICcnLFxyXG4gICAgICAgICAgICBwcm9wZXJ0aWVzOiB1bmRlZmluZWQsXHJcbiAgICAgICAgICAgIGNoaWxkcmVuOiB1bmRlZmluZWQsXHJcbiAgICAgICAgICAgIHRleHQ6IGRhdGEudG9TdHJpbmcoKSxcclxuICAgICAgICAgICAgZG9tTm9kZTogbnVsbFxyXG4gICAgICAgIH07XHJcbiAgICB9O1xyXG4gICAgdmFyIGFwcGVuZENoaWxkcmVuID0gZnVuY3Rpb24gKHBhcmVudFNlbGVjdG9yLCBpbnNlcnRpb25zLCBtYWluKSB7XHJcbiAgICAgICAgZm9yICh2YXIgaSA9IDAsIGxlbmd0aF8xID0gaW5zZXJ0aW9ucy5sZW5ndGg7IGkgPCBsZW5ndGhfMTsgaSsrKSB7XHJcbiAgICAgICAgICAgIHZhciBpdGVtID0gaW5zZXJ0aW9uc1tpXTtcclxuICAgICAgICAgICAgaWYgKEFycmF5LmlzQXJyYXkoaXRlbSkpIHtcclxuICAgICAgICAgICAgICAgIGFwcGVuZENoaWxkcmVuKHBhcmVudFNlbGVjdG9yLCBpdGVtLCBtYWluKTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgIGlmIChpdGVtICE9PSBudWxsICYmIGl0ZW0gIT09IHVuZGVmaW5lZCAmJiBpdGVtICE9PSBmYWxzZSkge1xyXG4gICAgICAgICAgICAgICAgICAgIGlmICh0eXBlb2YgaXRlbSA9PT0gJ3N0cmluZycpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgaXRlbSA9IHRvVGV4dFZOb2RlKGl0ZW0pO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICBtYWluLnB1c2goaXRlbSk7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgZnVuY3Rpb24gaChzZWxlY3RvciwgcHJvcGVydGllcywgY2hpbGRyZW4pIHtcclxuICAgICAgICBpZiAoQXJyYXkuaXNBcnJheShwcm9wZXJ0aWVzKSkge1xyXG4gICAgICAgICAgICBjaGlsZHJlbiA9IHByb3BlcnRpZXM7XHJcbiAgICAgICAgICAgIHByb3BlcnRpZXMgPSB1bmRlZmluZWQ7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGVsc2UgaWYgKChwcm9wZXJ0aWVzICYmICh0eXBlb2YgcHJvcGVydGllcyA9PT0gJ3N0cmluZycgfHwgcHJvcGVydGllcy5oYXNPd25Qcm9wZXJ0eSgndm5vZGVTZWxlY3RvcicpKSkgfHxcclxuICAgICAgICAgICAgKGNoaWxkcmVuICYmICh0eXBlb2YgY2hpbGRyZW4gPT09ICdzdHJpbmcnIHx8IGNoaWxkcmVuLmhhc093blByb3BlcnR5KCd2bm9kZVNlbGVjdG9yJykpKSkge1xyXG4gICAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ2ggY2FsbGVkIHdpdGggaW52YWxpZCBhcmd1bWVudHMnKTtcclxuICAgICAgICB9XHJcbiAgICAgICAgdmFyIHRleHQ7XHJcbiAgICAgICAgdmFyIGZsYXR0ZW5lZENoaWxkcmVuO1xyXG4gICAgICAgIC8vIFJlY29nbml6ZSBhIGNvbW1vbiBzcGVjaWFsIGNhc2Ugd2hlcmUgdGhlcmUgaXMgb25seSBhIHNpbmdsZSB0ZXh0IG5vZGVcclxuICAgICAgICBpZiAoY2hpbGRyZW4gJiYgY2hpbGRyZW4ubGVuZ3RoID09PSAxICYmIHR5cGVvZiBjaGlsZHJlblswXSA9PT0gJ3N0cmluZycpIHtcclxuICAgICAgICAgICAgdGV4dCA9IGNoaWxkcmVuWzBdO1xyXG4gICAgICAgIH1cclxuICAgICAgICBlbHNlIGlmIChjaGlsZHJlbikge1xyXG4gICAgICAgICAgICBmbGF0dGVuZWRDaGlsZHJlbiA9IFtdO1xyXG4gICAgICAgICAgICBhcHBlbmRDaGlsZHJlbihzZWxlY3RvciwgY2hpbGRyZW4sIGZsYXR0ZW5lZENoaWxkcmVuKTtcclxuICAgICAgICAgICAgaWYgKGZsYXR0ZW5lZENoaWxkcmVuLmxlbmd0aCA9PT0gMCkge1xyXG4gICAgICAgICAgICAgICAgZmxhdHRlbmVkQ2hpbGRyZW4gPSB1bmRlZmluZWQ7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuIHtcclxuICAgICAgICAgICAgdm5vZGVTZWxlY3Rvcjogc2VsZWN0b3IsXHJcbiAgICAgICAgICAgIHByb3BlcnRpZXM6IHByb3BlcnRpZXMsXHJcbiAgICAgICAgICAgIGNoaWxkcmVuOiBmbGF0dGVuZWRDaGlsZHJlbixcclxuICAgICAgICAgICAgdGV4dDogKHRleHQgPT09ICcnKSA/IHVuZGVmaW5lZCA6IHRleHQsXHJcbiAgICAgICAgICAgIGRvbU5vZGU6IG51bGxcclxuICAgICAgICB9O1xyXG4gICAgfVxuXG4gICAgdmFyIGNyZWF0ZVBhcmVudE5vZGVQYXRoID0gZnVuY3Rpb24gKG5vZGUsIHJvb3ROb2RlKSB7XHJcbiAgICAgICAgdmFyIHBhcmVudE5vZGVQYXRoID0gW107XHJcbiAgICAgICAgd2hpbGUgKG5vZGUgIT09IHJvb3ROb2RlKSB7XHJcbiAgICAgICAgICAgIHBhcmVudE5vZGVQYXRoLnB1c2gobm9kZSk7XHJcbiAgICAgICAgICAgIG5vZGUgPSBub2RlLnBhcmVudE5vZGU7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHJldHVybiBwYXJlbnROb2RlUGF0aDtcclxuICAgIH07XHJcbiAgICB2YXIgZmluZDtcclxuICAgIGlmIChBcnJheS5wcm90b3R5cGUuZmluZCkge1xyXG4gICAgICAgIGZpbmQgPSBmdW5jdGlvbiAoaXRlbXMsIHByZWRpY2F0ZSkgeyByZXR1cm4gaXRlbXMuZmluZChwcmVkaWNhdGUpOyB9O1xyXG4gICAgfVxyXG4gICAgZWxzZSB7XHJcbiAgICAgICAgZmluZCA9IGZ1bmN0aW9uIChpdGVtcywgcHJlZGljYXRlKSB7IHJldHVybiBpdGVtcy5maWx0ZXIocHJlZGljYXRlKVswXTsgfTtcclxuICAgIH1cclxuICAgIHZhciBmaW5kVk5vZGVCeVBhcmVudE5vZGVQYXRoID0gZnVuY3Rpb24gKHZub2RlLCBwYXJlbnROb2RlUGF0aCkge1xyXG4gICAgICAgIHZhciByZXN1bHQgPSB2bm9kZTtcclxuICAgICAgICBwYXJlbnROb2RlUGF0aC5mb3JFYWNoKGZ1bmN0aW9uIChub2RlKSB7XHJcbiAgICAgICAgICAgIHJlc3VsdCA9IChyZXN1bHQgJiYgcmVzdWx0LmNoaWxkcmVuKSA/IGZpbmQocmVzdWx0LmNoaWxkcmVuLCBmdW5jdGlvbiAoY2hpbGQpIHsgcmV0dXJuIGNoaWxkLmRvbU5vZGUgPT09IG5vZGU7IH0pIDogdW5kZWZpbmVkO1xyXG4gICAgICAgIH0pO1xyXG4gICAgICAgIHJldHVybiByZXN1bHQ7XHJcbiAgICB9O1xyXG4gICAgdmFyIGNyZWF0ZUV2ZW50SGFuZGxlckludGVyY2VwdG9yID0gZnVuY3Rpb24gKHByb2plY3RvciwgZ2V0UHJvamVjdGlvbiwgcGVyZm9ybWFuY2VMb2dnZXIpIHtcclxuICAgICAgICB2YXIgbW9kaWZpZWRFdmVudEhhbmRsZXIgPSBmdW5jdGlvbiAoZXZ0KSB7XHJcbiAgICAgICAgICAgIHBlcmZvcm1hbmNlTG9nZ2VyKCdkb21FdmVudCcsIGV2dCk7XHJcbiAgICAgICAgICAgIHZhciBwcm9qZWN0aW9uID0gZ2V0UHJvamVjdGlvbigpO1xyXG4gICAgICAgICAgICB2YXIgcGFyZW50Tm9kZVBhdGggPSBjcmVhdGVQYXJlbnROb2RlUGF0aChldnQuY3VycmVudFRhcmdldCwgcHJvamVjdGlvbi5kb21Ob2RlKTtcclxuICAgICAgICAgICAgcGFyZW50Tm9kZVBhdGgucmV2ZXJzZSgpO1xyXG4gICAgICAgICAgICB2YXIgbWF0Y2hpbmdWTm9kZSA9IGZpbmRWTm9kZUJ5UGFyZW50Tm9kZVBhdGgocHJvamVjdGlvbi5nZXRMYXN0UmVuZGVyKCksIHBhcmVudE5vZGVQYXRoKTtcclxuICAgICAgICAgICAgcHJvamVjdG9yLnNjaGVkdWxlUmVuZGVyKCk7XHJcbiAgICAgICAgICAgIHZhciByZXN1bHQ7XHJcbiAgICAgICAgICAgIGlmIChtYXRjaGluZ1ZOb2RlKSB7XHJcbiAgICAgICAgICAgICAgICAvKiB0c2xpbnQ6ZGlzYWJsZSBuby1pbnZhbGlkLXRoaXMgKi9cclxuICAgICAgICAgICAgICAgIHJlc3VsdCA9IG1hdGNoaW5nVk5vZGUucHJvcGVydGllc1tcIm9uXCIgKyBldnQudHlwZV0uYXBwbHkobWF0Y2hpbmdWTm9kZS5wcm9wZXJ0aWVzLmJpbmQgfHwgdGhpcywgYXJndW1lbnRzKTtcclxuICAgICAgICAgICAgICAgIC8qIHRzbGludDplbmFibGUgbm8taW52YWxpZC10aGlzICovXHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgcGVyZm9ybWFuY2VMb2dnZXIoJ2RvbUV2ZW50UHJvY2Vzc2VkJywgZXZ0KTtcclxuICAgICAgICAgICAgcmV0dXJuIHJlc3VsdDtcclxuICAgICAgICB9O1xyXG4gICAgICAgIHJldHVybiBmdW5jdGlvbiAocHJvcGVydHlOYW1lLCBldmVudEhhbmRsZXIsIGRvbU5vZGUsIHByb3BlcnRpZXMpIHsgcmV0dXJuIG1vZGlmaWVkRXZlbnRIYW5kbGVyOyB9O1xyXG4gICAgfTtcclxuICAgIC8qKlxyXG4gICAgICogQ3JlYXRlcyBhIFtbUHJvamVjdG9yXV0gaW5zdGFuY2UgdXNpbmcgdGhlIHByb3ZpZGVkIHByb2plY3Rpb25PcHRpb25zLlxyXG4gICAgICpcclxuICAgICAqIEZvciBtb3JlIGluZm9ybWF0aW9uLCBzZWUgW1tQcm9qZWN0b3JdXS5cclxuICAgICAqXHJcbiAgICAgKiBAcGFyYW0gcHJvamVjdG9yT3B0aW9ucyAgIE9wdGlvbnMgdGhhdCBpbmZsdWVuY2UgaG93IHRoZSBET00gaXMgcmVuZGVyZWQgYW5kIHVwZGF0ZWQuXHJcbiAgICAgKi9cclxuICAgIHZhciBjcmVhdGVQcm9qZWN0b3IgPSBmdW5jdGlvbiAocHJvamVjdG9yT3B0aW9ucykge1xyXG4gICAgICAgIHZhciBwcm9qZWN0b3I7XHJcbiAgICAgICAgdmFyIHByb2plY3Rpb25PcHRpb25zID0gYXBwbHlEZWZhdWx0UHJvamVjdGlvbk9wdGlvbnMocHJvamVjdG9yT3B0aW9ucyk7XHJcbiAgICAgICAgdmFyIHBlcmZvcm1hbmNlTG9nZ2VyID0gcHJvamVjdGlvbk9wdGlvbnMucGVyZm9ybWFuY2VMb2dnZXI7XHJcbiAgICAgICAgdmFyIHJlbmRlckNvbXBsZXRlZCA9IHRydWU7XHJcbiAgICAgICAgdmFyIHNjaGVkdWxlZDtcclxuICAgICAgICB2YXIgc3RvcHBlZCA9IGZhbHNlO1xyXG4gICAgICAgIHZhciBwcm9qZWN0aW9ucyA9IFtdO1xyXG4gICAgICAgIHZhciByZW5kZXJGdW5jdGlvbnMgPSBbXTsgLy8gbWF0Y2hlcyB0aGUgcHJvamVjdGlvbnMgYXJyYXlcclxuICAgICAgICB2YXIgYWRkUHJvamVjdGlvbiA9IGZ1bmN0aW9uIChcclxuICAgICAgICAvKiBvbmUgb2Y6IGRvbS5hcHBlbmQsIGRvbS5pbnNlcnRCZWZvcmUsIGRvbS5yZXBsYWNlLCBkb20ubWVyZ2UgKi9cclxuICAgICAgICBkb21GdW5jdGlvbiwgXHJcbiAgICAgICAgLyogdGhlIHBhcmFtZXRlciBvZiB0aGUgZG9tRnVuY3Rpb24gKi9cclxuICAgICAgICBub2RlLCByZW5kZXJGdW5jdGlvbikge1xyXG4gICAgICAgICAgICB2YXIgcHJvamVjdGlvbjtcclxuICAgICAgICAgICAgdmFyIGdldFByb2plY3Rpb24gPSBmdW5jdGlvbiAoKSB7IHJldHVybiBwcm9qZWN0aW9uOyB9O1xyXG4gICAgICAgICAgICBwcm9qZWN0aW9uT3B0aW9ucy5ldmVudEhhbmRsZXJJbnRlcmNlcHRvciA9IGNyZWF0ZUV2ZW50SGFuZGxlckludGVyY2VwdG9yKHByb2plY3RvciwgZ2V0UHJvamVjdGlvbiwgcGVyZm9ybWFuY2VMb2dnZXIpO1xyXG4gICAgICAgICAgICBwcm9qZWN0aW9uID0gZG9tRnVuY3Rpb24obm9kZSwgcmVuZGVyRnVuY3Rpb24oKSwgcHJvamVjdGlvbk9wdGlvbnMpO1xyXG4gICAgICAgICAgICBwcm9qZWN0aW9ucy5wdXNoKHByb2plY3Rpb24pO1xyXG4gICAgICAgICAgICByZW5kZXJGdW5jdGlvbnMucHVzaChyZW5kZXJGdW5jdGlvbik7XHJcbiAgICAgICAgfTtcclxuICAgICAgICB2YXIgZG9SZW5kZXIgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgICAgIHNjaGVkdWxlZCA9IHVuZGVmaW5lZDtcclxuICAgICAgICAgICAgaWYgKCFyZW5kZXJDb21wbGV0ZWQpIHtcclxuICAgICAgICAgICAgICAgIHJldHVybjsgLy8gVGhlIGxhc3QgcmVuZGVyIHRocmV3IGFuIGVycm9yLCBpdCBzaG91bGQgaGF2ZSBiZWVuIGxvZ2dlZCBpbiB0aGUgYnJvd3NlciBjb25zb2xlLlxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIHJlbmRlckNvbXBsZXRlZCA9IGZhbHNlO1xyXG4gICAgICAgICAgICBwZXJmb3JtYW5jZUxvZ2dlcigncmVuZGVyU3RhcnQnLCB1bmRlZmluZWQpO1xyXG4gICAgICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IHByb2plY3Rpb25zLmxlbmd0aDsgaSsrKSB7XHJcbiAgICAgICAgICAgICAgICB2YXIgdXBkYXRlZFZub2RlID0gcmVuZGVyRnVuY3Rpb25zW2ldKCk7XHJcbiAgICAgICAgICAgICAgICBwZXJmb3JtYW5jZUxvZ2dlcigncmVuZGVyZWQnLCB1bmRlZmluZWQpO1xyXG4gICAgICAgICAgICAgICAgcHJvamVjdGlvbnNbaV0udXBkYXRlKHVwZGF0ZWRWbm9kZSk7XHJcbiAgICAgICAgICAgICAgICBwZXJmb3JtYW5jZUxvZ2dlcigncGF0Y2hlZCcsIHVuZGVmaW5lZCk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgcGVyZm9ybWFuY2VMb2dnZXIoJ3JlbmRlckRvbmUnLCB1bmRlZmluZWQpO1xyXG4gICAgICAgICAgICByZW5kZXJDb21wbGV0ZWQgPSB0cnVlO1xyXG4gICAgICAgIH07XHJcbiAgICAgICAgcHJvamVjdG9yID0ge1xyXG4gICAgICAgICAgICByZW5kZXJOb3c6IGRvUmVuZGVyLFxyXG4gICAgICAgICAgICBzY2hlZHVsZVJlbmRlcjogZnVuY3Rpb24gKCkge1xyXG4gICAgICAgICAgICAgICAgaWYgKCFzY2hlZHVsZWQgJiYgIXN0b3BwZWQpIHtcclxuICAgICAgICAgICAgICAgICAgICBzY2hlZHVsZWQgPSByZXF1ZXN0QW5pbWF0aW9uRnJhbWUoZG9SZW5kZXIpO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICBzdG9wOiBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgICAgICAgICBpZiAoc2NoZWR1bGVkKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgY2FuY2VsQW5pbWF0aW9uRnJhbWUoc2NoZWR1bGVkKTtcclxuICAgICAgICAgICAgICAgICAgICBzY2hlZHVsZWQgPSB1bmRlZmluZWQ7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICBzdG9wcGVkID0gdHJ1ZTtcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAgcmVzdW1lOiBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgICAgICAgICBzdG9wcGVkID0gZmFsc2U7XHJcbiAgICAgICAgICAgICAgICByZW5kZXJDb21wbGV0ZWQgPSB0cnVlO1xyXG4gICAgICAgICAgICAgICAgcHJvamVjdG9yLnNjaGVkdWxlUmVuZGVyKCk7XHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIGFwcGVuZDogZnVuY3Rpb24gKHBhcmVudE5vZGUsIHJlbmRlckZ1bmN0aW9uKSB7XHJcbiAgICAgICAgICAgICAgICBhZGRQcm9qZWN0aW9uKGRvbS5hcHBlbmQsIHBhcmVudE5vZGUsIHJlbmRlckZ1bmN0aW9uKTtcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAgaW5zZXJ0QmVmb3JlOiBmdW5jdGlvbiAoYmVmb3JlTm9kZSwgcmVuZGVyRnVuY3Rpb24pIHtcclxuICAgICAgICAgICAgICAgIGFkZFByb2plY3Rpb24oZG9tLmluc2VydEJlZm9yZSwgYmVmb3JlTm9kZSwgcmVuZGVyRnVuY3Rpb24pO1xyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICBtZXJnZTogZnVuY3Rpb24gKGRvbU5vZGUsIHJlbmRlckZ1bmN0aW9uKSB7XHJcbiAgICAgICAgICAgICAgICBhZGRQcm9qZWN0aW9uKGRvbS5tZXJnZSwgZG9tTm9kZSwgcmVuZGVyRnVuY3Rpb24pO1xyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICByZXBsYWNlOiBmdW5jdGlvbiAoZG9tTm9kZSwgcmVuZGVyRnVuY3Rpb24pIHtcclxuICAgICAgICAgICAgICAgIGFkZFByb2plY3Rpb24oZG9tLnJlcGxhY2UsIGRvbU5vZGUsIHJlbmRlckZ1bmN0aW9uKTtcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAgZGV0YWNoOiBmdW5jdGlvbiAocmVuZGVyRnVuY3Rpb24pIHtcclxuICAgICAgICAgICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgcmVuZGVyRnVuY3Rpb25zLmxlbmd0aDsgaSsrKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKHJlbmRlckZ1bmN0aW9uc1tpXSA9PT0gcmVuZGVyRnVuY3Rpb24pIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgcmVuZGVyRnVuY3Rpb25zLnNwbGljZShpLCAxKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgcmV0dXJuIHByb2plY3Rpb25zLnNwbGljZShpLCAxKVswXTtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ3JlbmRlckZ1bmN0aW9uIHdhcyBub3QgZm91bmQnKTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH07XHJcbiAgICAgICAgcmV0dXJuIHByb2plY3RvcjtcclxuICAgIH07XG5cbiAgICAvKipcclxuICAgICAqIENyZWF0ZXMgYSBbW0NhbGN1bGF0aW9uQ2FjaGVdXSBvYmplY3QsIHVzZWZ1bCBmb3IgY2FjaGluZyBbW1ZOb2RlXV0gdHJlZXMuXHJcbiAgICAgKiBJbiBwcmFjdGljZSwgY2FjaGluZyBvZiBbW1ZOb2RlXV0gdHJlZXMgaXMgbm90IG5lZWRlZCwgYmVjYXVzZSBhY2hpZXZpbmcgNjAgZnJhbWVzIHBlciBzZWNvbmQgaXMgYWxtb3N0IG5ldmVyIGEgcHJvYmxlbS5cclxuICAgICAqIEZvciBtb3JlIGluZm9ybWF0aW9uLCBzZWUgW1tDYWxjdWxhdGlvbkNhY2hlXV0uXHJcbiAgICAgKlxyXG4gICAgICogQHBhcmFtIDxSZXN1bHQ+IFRoZSB0eXBlIG9mIHRoZSB2YWx1ZSB0aGF0IGlzIGNhY2hlZC5cclxuICAgICAqL1xyXG4gICAgdmFyIGNyZWF0ZUNhY2hlID0gZnVuY3Rpb24gKCkge1xyXG4gICAgICAgIHZhciBjYWNoZWRJbnB1dHM7XHJcbiAgICAgICAgdmFyIGNhY2hlZE91dGNvbWU7XHJcbiAgICAgICAgcmV0dXJuIHtcclxuICAgICAgICAgICAgaW52YWxpZGF0ZTogZnVuY3Rpb24gKCkge1xyXG4gICAgICAgICAgICAgICAgY2FjaGVkT3V0Y29tZSA9IHVuZGVmaW5lZDtcclxuICAgICAgICAgICAgICAgIGNhY2hlZElucHV0cyA9IHVuZGVmaW5lZDtcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAgcmVzdWx0OiBmdW5jdGlvbiAoaW5wdXRzLCBjYWxjdWxhdGlvbikge1xyXG4gICAgICAgICAgICAgICAgaWYgKGNhY2hlZElucHV0cykge1xyXG4gICAgICAgICAgICAgICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgaW5wdXRzLmxlbmd0aDsgaSsrKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGlmIChjYWNoZWRJbnB1dHNbaV0gIT09IGlucHV0c1tpXSkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgY2FjaGVkT3V0Y29tZSA9IHVuZGVmaW5lZDtcclxuICAgICAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIGlmICghY2FjaGVkT3V0Y29tZSkge1xyXG4gICAgICAgICAgICAgICAgICAgIGNhY2hlZE91dGNvbWUgPSBjYWxjdWxhdGlvbigpO1xyXG4gICAgICAgICAgICAgICAgICAgIGNhY2hlZElucHV0cyA9IGlucHV0cztcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIHJldHVybiBjYWNoZWRPdXRjb21lO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfTtcclxuICAgIH07XG5cbiAgICAvKipcclxuICAgICAqIENyZWF0ZXMgYSB7QGxpbmsgTWFwcGluZ30gaW5zdGFuY2UgdGhhdCBrZWVwcyBhbiBhcnJheSBvZiByZXN1bHQgb2JqZWN0cyBzeW5jaHJvbml6ZWQgd2l0aCBhbiBhcnJheSBvZiBzb3VyY2Ugb2JqZWN0cy5cclxuICAgICAqIFNlZSB7QGxpbmsgaHR0cDovL21hcXVldHRlanMub3JnL2RvY3MvYXJyYXlzLmh0bWx8V29ya2luZyB3aXRoIGFycmF5c30uXHJcbiAgICAgKlxyXG4gICAgICogQHBhcmFtIDxTb3VyY2U+ICAgICAgIFRoZSB0eXBlIG9mIHNvdXJjZSBpdGVtcy4gQSBkYXRhYmFzZS1yZWNvcmQgZm9yIGluc3RhbmNlLlxyXG4gICAgICogQHBhcmFtIDxUYXJnZXQ+ICAgICAgIFRoZSB0eXBlIG9mIHRhcmdldCBpdGVtcy4gQSBbW01hcXVldHRlQ29tcG9uZW50XV0gZm9yIGluc3RhbmNlLlxyXG4gICAgICogQHBhcmFtIGdldFNvdXJjZUtleSAgIGBmdW5jdGlvbihzb3VyY2UpYCB0aGF0IG11c3QgcmV0dXJuIGEga2V5IHRvIGlkZW50aWZ5IGVhY2ggc291cmNlIG9iamVjdC4gVGhlIHJlc3VsdCBtdXN0IGVpdGhlciBiZSBhIHN0cmluZyBvciBhIG51bWJlci5cclxuICAgICAqIEBwYXJhbSBjcmVhdGVSZXN1bHQgICBgZnVuY3Rpb24oc291cmNlLCBpbmRleClgIHRoYXQgbXVzdCBjcmVhdGUgYSBuZXcgcmVzdWx0IG9iamVjdCBmcm9tIGEgZ2l2ZW4gc291cmNlLiBUaGlzIGZ1bmN0aW9uIGlzIGlkZW50aWNhbFxyXG4gICAgICogICAgICAgICAgICAgICAgICAgICAgIHRvIHRoZSBgY2FsbGJhY2tgIGFyZ3VtZW50IGluIGBBcnJheS5tYXAoY2FsbGJhY2spYC5cclxuICAgICAqIEBwYXJhbSB1cGRhdGVSZXN1bHQgICBgZnVuY3Rpb24oc291cmNlLCB0YXJnZXQsIGluZGV4KWAgdGhhdCB1cGRhdGVzIGEgcmVzdWx0IHRvIGFuIHVwZGF0ZWQgc291cmNlLlxyXG4gICAgICovXHJcbiAgICB2YXIgY3JlYXRlTWFwcGluZyA9IGZ1bmN0aW9uIChnZXRTb3VyY2VLZXksIGNyZWF0ZVJlc3VsdCwgdXBkYXRlUmVzdWx0KSB7XHJcbiAgICAgICAgdmFyIGtleXMgPSBbXTtcclxuICAgICAgICB2YXIgcmVzdWx0cyA9IFtdO1xyXG4gICAgICAgIHJldHVybiB7XHJcbiAgICAgICAgICAgIHJlc3VsdHM6IHJlc3VsdHMsXHJcbiAgICAgICAgICAgIG1hcDogZnVuY3Rpb24gKG5ld1NvdXJjZXMpIHtcclxuICAgICAgICAgICAgICAgIHZhciBuZXdLZXlzID0gbmV3U291cmNlcy5tYXAoZ2V0U291cmNlS2V5KTtcclxuICAgICAgICAgICAgICAgIHZhciBvbGRUYXJnZXRzID0gcmVzdWx0cy5zbGljZSgpO1xyXG4gICAgICAgICAgICAgICAgdmFyIG9sZEluZGV4ID0gMDtcclxuICAgICAgICAgICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgbmV3U291cmNlcy5sZW5ndGg7IGkrKykge1xyXG4gICAgICAgICAgICAgICAgICAgIHZhciBzb3VyY2UgPSBuZXdTb3VyY2VzW2ldO1xyXG4gICAgICAgICAgICAgICAgICAgIHZhciBzb3VyY2VLZXkgPSBuZXdLZXlzW2ldO1xyXG4gICAgICAgICAgICAgICAgICAgIGlmIChzb3VyY2VLZXkgPT09IGtleXNbb2xkSW5kZXhdKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHJlc3VsdHNbaV0gPSBvbGRUYXJnZXRzW29sZEluZGV4XTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgdXBkYXRlUmVzdWx0KHNvdXJjZSwgb2xkVGFyZ2V0c1tvbGRJbmRleF0sIGkpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBvbGRJbmRleCsrO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgdmFyIGZvdW5kID0gZmFsc2U7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGZvciAodmFyIGogPSAxOyBqIDwga2V5cy5sZW5ndGggKyAxOyBqKyspIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHZhciBzZWFyY2hJbmRleCA9IChvbGRJbmRleCArIGopICUga2V5cy5sZW5ndGg7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAoa2V5c1tzZWFyY2hJbmRleF0gPT09IHNvdXJjZUtleSkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHJlc3VsdHNbaV0gPSBvbGRUYXJnZXRzW3NlYXJjaEluZGV4XTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB1cGRhdGVSZXN1bHQobmV3U291cmNlc1tpXSwgb2xkVGFyZ2V0c1tzZWFyY2hJbmRleF0sIGkpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG9sZEluZGV4ID0gc2VhcmNoSW5kZXggKyAxO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGZvdW5kID0gdHJ1ZTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBicmVhaztcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgICAgICBpZiAoIWZvdW5kKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICByZXN1bHRzW2ldID0gY3JlYXRlUmVzdWx0KHNvdXJjZSwgaSk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICByZXN1bHRzLmxlbmd0aCA9IG5ld1NvdXJjZXMubGVuZ3RoO1xyXG4gICAgICAgICAgICAgICAga2V5cyA9IG5ld0tleXM7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9O1xyXG4gICAgfTtcblxuICAgIGV4cG9ydHMuY3JlYXRlQ2FjaGUgPSBjcmVhdGVDYWNoZTtcbiAgICBleHBvcnRzLmNyZWF0ZU1hcHBpbmcgPSBjcmVhdGVNYXBwaW5nO1xuICAgIGV4cG9ydHMuY3JlYXRlUHJvamVjdG9yID0gY3JlYXRlUHJvamVjdG9yO1xuICAgIGV4cG9ydHMuZG9tID0gZG9tO1xuICAgIGV4cG9ydHMuaCA9IGg7XG5cbiAgICBPYmplY3QuZGVmaW5lUHJvcGVydHkoZXhwb3J0cywgJ19fZXNNb2R1bGUnLCB7IHZhbHVlOiB0cnVlIH0pO1xuXG59KSk7XG4iXSwic291cmNlUm9vdCI6IiJ9