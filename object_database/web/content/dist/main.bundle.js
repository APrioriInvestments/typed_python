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
            return this.getReplacementElementFor('child');
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
            return [];
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly8vd2VicGFjay9ib290c3RyYXAiLCJ3ZWJwYWNrOi8vLy4vQ2VsbEhhbmRsZXIuanMiLCJ3ZWJwYWNrOi8vLy4vQ2VsbFNvY2tldC5qcyIsIndlYnBhY2s6Ly8vLi9Db21wb25lbnRSZWdpc3RyeS5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0FzeW5jRHJvcGRvd24uanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9CYWRnZS5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0J1dHRvbi5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0J1dHRvbkdyb3VwLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvQ2FyZC5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0NhcmRUaXRsZS5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0NpcmNsZUxvYWRlci5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0NsaWNrYWJsZS5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0NvZGUuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9Db2RlRWRpdG9yLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvQ29sbGFwc2libGVQYW5lbC5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0NvbHVtbnMuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9Db21wb25lbnQuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9Db250YWluZXIuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9Db250ZXh0dWFsRGlzcGxheS5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0Ryb3Bkb3duLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvRXhwYW5kcy5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0dyaWQuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9IZWFkZXJCYXIuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9MYXJnZVBlbmRpbmdEb3dubG9hZERpc3BsYXkuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9Mb2FkQ29udGVudHNGcm9tVXJsLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvTWFpbi5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL01vZGFsLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvT2N0aWNvbi5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL1BhZGRpbmcuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9QbG90LmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvUG9wb3Zlci5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL1Jvb3RDZWxsLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvU2Nyb2xsYWJsZS5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL1NlcXVlbmNlLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvU2hlZXQuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9TaW5nbGVMaW5lVGV4dEJveC5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL1NwYW4uanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9TdWJzY3JpYmVkLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvU3Vic2NyaWJlZFNlcXVlbmNlLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvVGFibGUuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9UYWJzLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvVGV4dC5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL1RyYWNlYmFjay5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL19OYXZUYWIuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9fUGxvdFVwZGF0ZXIuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy91dGlsL1Byb3BlcnR5VmFsaWRhdG9yLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvdXRpbC9SZXBsYWNlbWVudHNIYW5kbGVyLmpzIiwid2VicGFjazovLy8uL21haW4uanMiLCJ3ZWJwYWNrOi8vLy4vbm9kZV9tb2R1bGVzL21hcXVldHRlL2Rpc3QvbWFxdWV0dGUudW1kLmpzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiI7QUFBQTtBQUNBOztBQUVBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTs7QUFFQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTs7O0FBR0E7QUFDQTs7QUFFQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBLGtEQUEwQyxnQ0FBZ0M7QUFDMUU7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQSxnRUFBd0Qsa0JBQWtCO0FBQzFFO0FBQ0EseURBQWlELGNBQWM7QUFDL0Q7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGlEQUF5QyxpQ0FBaUM7QUFDMUUsd0hBQWdILG1CQUFtQixFQUFFO0FBQ3JJO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0EsbUNBQTJCLDBCQUEwQixFQUFFO0FBQ3ZELHlDQUFpQyxlQUFlO0FBQ2hEO0FBQ0E7QUFDQTs7QUFFQTtBQUNBLDhEQUFzRCwrREFBK0Q7O0FBRXJIO0FBQ0E7OztBQUdBO0FBQ0E7Ozs7Ozs7Ozs7Ozs7QUNsRkE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDMkI7O0FBRTNCO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7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QkFBNkIsY0FBYyxNQUFNLFNBQVM7QUFDMUQ7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUlFOzs7QUFHRjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsYUFBYTtBQUNiLGlDQUFpQyxjQUFjLE1BQU0sU0FBUztBQUM5RDtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTs7QUFFQSxLQUFLOztBQUVMO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsYUFBYTtBQUNiLGlDQUFpQyxjQUFjLE1BQU0sU0FBUztBQUM5RDtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBLEtBQUs7O0FBRUw7QUFDQTtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2IsaUNBQWlDLGNBQWMsTUFBTSxTQUFTO0FBQzlEO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBO0FBQ0EsS0FBSzs7QUFFTDtBQUNBO0FBQ0E7QUFDQTtBQUNBLGFBQWE7QUFDYixpQ0FBaUMsY0FBYyxNQUFNLFNBQVM7QUFDOUQ7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQSxLQUFLOztBQUVMO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsYUFBYTtBQUNiLGlDQUFpQyxjQUFjLE1BQU0sU0FBUztBQUM5RDtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBLEtBQUs7O0FBRUw7Ozs7Ozs7Ozs7Ozs7QUM5T0E7QUFBQTtBQUFBO0FBQUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsYUFBYTtBQUNiO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBLHlEQUF5RCxpQkFBaUI7QUFDMUU7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBOztBQUVBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUtFOzs7Ozs7Ozs7Ozs7O0FDcEtGO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFrQjtBQUNsQjtBQUNBLFVBQVUsVUFBVTtBQUNzQjtBQUNGO0FBQ2M7O0FBRXREO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLHNCQUFzQjtBQUN0QixtQkFBbUIsZ0JBQWdCO0FBQ25DLHdDQUF3QztBQUN4QyxnQ0FBZ0MsY0FBYztBQUM5QywwQ0FBMEM7QUFDMUM7QUFDQTtBQUNBO0FBQ0EsbUJBQW1CLHlDQUF5QztBQUM1RDtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsdUJBQXVCLHNEQUFVO0FBQ2pDLHdCQUF3Qix3REFBVyxlQUFlLG9FQUFpQjtBQUNuRTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsQ0FBQzs7QUFFRDtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQSxDQUFDOztBQUVELFdBQVc7QUFDWDs7Ozs7Ozs7Ozs7O0FDdERBO0FBQ0EsSUFBSSxLQUE0RDtBQUNoRSxJQUFJLFNBQ3dEO0FBQzVELENBQUMsMkJBQTJCOztBQUU1QjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2I7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLCtCQUErQixxQkFBcUI7QUFDcEQ7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxtQkFBbUI7QUFDbkI7QUFDQTtBQUNBO0FBQ0EsbUJBQW1CO0FBQ25CLDJCQUEyQix1QkFBdUI7QUFDbEQ7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLHdFQUF3RSxjQUFjO0FBQ3RGO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSwrQkFBK0Isb0JBQW9CO0FBQ25EO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsK0JBQStCLGdCQUFnQjtBQUMvQztBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsOERBQThEO0FBQzlEO0FBQ0EsMEdBQTBHO0FBQzFHO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxtRkFBbUY7QUFDbkY7QUFDQSw2QkFBNkI7QUFDN0I7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSx1QkFBdUIsZUFBZTtBQUN0QztBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLCtDQUErQyx3QkFBd0I7QUFDdkU7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGdFQUFnRTtBQUNoRTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsMkJBQTJCLDJCQUEyQjtBQUN0RDtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSwyRUFBMkUsMkJBQTJCO0FBQ3RHO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsdUJBQXVCLGVBQWU7QUFDdEM7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLCtCQUErQixvQkFBb0I7QUFDbkQ7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsK0JBQStCLGdCQUFnQjtBQUMvQztBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsMkNBQTJDO0FBQzNDO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0Esc0RBQXNEO0FBQ3REO0FBQ0EscUJBQXFCO0FBQ3JCO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLDRGQUE0RjtBQUM1RjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsbUVBQW1FO0FBQ25FO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxzQ0FBc0Msa0JBQWtCO0FBQ3hEO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSw4QkFBOEIsdUJBQXVCO0FBQ3JEO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EseUJBQXlCO0FBQ3pCO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0Esa0VBQWtFO0FBQ2xFLCtEQUErRCwyQkFBMkI7QUFDMUY7QUFDQTtBQUNBO0FBQ0E7QUFDQSw0REFBNEQ7QUFDNUQ7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSx3Q0FBd0MsY0FBYyxFQUFFO0FBQ3hEO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsYUFBYTtBQUNiO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0Esd0NBQXdDLGtCQUFrQixFQUFFO0FBQzVEO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLHFEQUFxRCxjQUFjO0FBQ25FO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSw0Q0FBNEMsOEJBQThCO0FBQzFFO0FBQ0E7QUFDQSw0Q0FBNEMsbUNBQW1DO0FBQy9FO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsMkZBQTJGLCtCQUErQixFQUFFO0FBQzVILFNBQVM7QUFDVDtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLDJFQUEyRSw2QkFBNkI7QUFDeEc7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxpQ0FBaUM7QUFDakM7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsNkNBQTZDLG1CQUFtQjtBQUNoRTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsdUJBQXVCO0FBQ3ZCO0FBQ0E7QUFDQTtBQUNBLDJCQUEyQix3QkFBd0I7QUFDbkQ7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGFBQWE7QUFDYjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2I7QUFDQTtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2I7QUFDQTtBQUNBLGFBQWE7QUFDYjtBQUNBO0FBQ0EsYUFBYTtBQUNiO0FBQ0E7QUFDQSxhQUFhO0FBQ2I7QUFDQTtBQUNBLGFBQWE7QUFDYjtBQUNBLCtCQUErQiw0QkFBNEI7QUFDM0Q7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGFBQWE7QUFDYjtBQUNBO0FBQ0EsbUNBQW1DLG1CQUFtQjtBQUN0RDtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBLGtCQUFrQixjQUFjO0FBQ2hDLFlBQVksaUVBQWlFO0FBQzdFO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSwrQkFBK0IsdUJBQXVCO0FBQ3REO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLHVDQUF1QyxxQkFBcUI7QUFDNUQ7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQSxrREFBa0QsY0FBYzs7QUFFaEUsQ0FBQyIsImZpbGUiOiJtYWluLmJ1bmRsZS5qcyIsInNvdXJjZXNDb250ZW50IjpbIiBcdC8vIFRoZSBtb2R1bGUgY2FjaGVcbiBcdHZhciBpbnN0YWxsZWRNb2R1bGVzID0ge307XG5cbiBcdC8vIFRoZSByZXF1aXJlIGZ1bmN0aW9uXG4gXHRmdW5jdGlvbiBfX3dlYnBhY2tfcmVxdWlyZV9fKG1vZHVsZUlkKSB7XG5cbiBcdFx0Ly8gQ2hlY2sgaWYgbW9kdWxlIGlzIGluIGNhY2hlXG4gXHRcdGlmKGluc3RhbGxlZE1vZHVsZXNbbW9kdWxlSWRdKSB7XG4gXHRcdFx0cmV0dXJuIGluc3RhbGxlZE1vZHVsZXNbbW9kdWxlSWRdLmV4cG9ydHM7XG4gXHRcdH1cbiBcdFx0Ly8gQ3JlYXRlIGEgbmV3IG1vZHVsZSAoYW5kIHB1dCBpdCBpbnRvIHRoZSBjYWNoZSlcbiBcdFx0dmFyIG1vZHVsZSA9IGluc3RhbGxlZE1vZHVsZXNbbW9kdWxlSWRdID0ge1xuIFx0XHRcdGk6IG1vZHVsZUlkLFxuIFx0XHRcdGw6IGZhbHNlLFxuIFx0XHRcdGV4cG9ydHM6IHt9XG4gXHRcdH07XG5cbiBcdFx0Ly8gRXhlY3V0ZSB0aGUgbW9kdWxlIGZ1bmN0aW9uXG4gXHRcdG1vZHVsZXNbbW9kdWxlSWRdLmNhbGwobW9kdWxlLmV4cG9ydHMsIG1vZHVsZSwgbW9kdWxlLmV4cG9ydHMsIF9fd2VicGFja19yZXF1aXJlX18pO1xuXG4gXHRcdC8vIEZsYWcgdGhlIG1vZHVsZSBhcyBsb2FkZWRcbiBcdFx0bW9kdWxlLmwgPSB0cnVlO1xuXG4gXHRcdC8vIFJldHVybiB0aGUgZXhwb3J0cyBvZiB0aGUgbW9kdWxlXG4gXHRcdHJldHVybiBtb2R1bGUuZXhwb3J0cztcbiBcdH1cblxuXG4gXHQvLyBleHBvc2UgdGhlIG1vZHVsZXMgb2JqZWN0IChfX3dlYnBhY2tfbW9kdWxlc19fKVxuIFx0X193ZWJwYWNrX3JlcXVpcmVfXy5tID0gbW9kdWxlcztcblxuIFx0Ly8gZXhwb3NlIHRoZSBtb2R1bGUgY2FjaGVcbiBcdF9fd2VicGFja19yZXF1aXJlX18uYyA9IGluc3RhbGxlZE1vZHVsZXM7XG5cbiBcdC8vIGRlZmluZSBnZXR0ZXIgZnVuY3Rpb24gZm9yIGhhcm1vbnkgZXhwb3J0c1xuIFx0X193ZWJwYWNrX3JlcXVpcmVfXy5kID0gZnVuY3Rpb24oZXhwb3J0cywgbmFtZSwgZ2V0dGVyKSB7XG4gXHRcdGlmKCFfX3dlYnBhY2tfcmVxdWlyZV9fLm8oZXhwb3J0cywgbmFtZSkpIHtcbiBcdFx0XHRPYmplY3QuZGVmaW5lUHJvcGVydHkoZXhwb3J0cywgbmFtZSwgeyBlbnVtZXJhYmxlOiB0cnVlLCBnZXQ6IGdldHRlciB9KTtcbiBcdFx0fVxuIFx0fTtcblxuIFx0Ly8gZGVmaW5lIF9fZXNNb2R1bGUgb24gZXhwb3J0c1xuIFx0X193ZWJwYWNrX3JlcXVpcmVfXy5yID0gZnVuY3Rpb24oZXhwb3J0cykge1xuIFx0XHRpZih0eXBlb2YgU3ltYm9sICE9PSAndW5kZWZpbmVkJyAmJiBTeW1ib2wudG9TdHJpbmdUYWcpIHtcbiBcdFx0XHRPYmplY3QuZGVmaW5lUHJvcGVydHkoZXhwb3J0cywgU3ltYm9sLnRvU3RyaW5nVGFnLCB7IHZhbHVlOiAnTW9kdWxlJyB9KTtcbiBcdFx0fVxuIFx0XHRPYmplY3QuZGVmaW5lUHJvcGVydHkoZXhwb3J0cywgJ19fZXNNb2R1bGUnLCB7IHZhbHVlOiB0cnVlIH0pO1xuIFx0fTtcblxuIFx0Ly8gY3JlYXRlIGEgZmFrZSBuYW1lc3BhY2Ugb2JqZWN0XG4gXHQvLyBtb2RlICYgMTogdmFsdWUgaXMgYSBtb2R1bGUgaWQsIHJlcXVpcmUgaXRcbiBcdC8vIG1vZGUgJiAyOiBtZXJnZSBhbGwgcHJvcGVydGllcyBvZiB2YWx1ZSBpbnRvIHRoZSBuc1xuIFx0Ly8gbW9kZSAmIDQ6IHJldHVybiB2YWx1ZSB3aGVuIGFscmVhZHkgbnMgb2JqZWN0XG4gXHQvLyBtb2RlICYgOHwxOiBiZWhhdmUgbGlrZSByZXF1aXJlXG4gXHRfX3dlYnBhY2tfcmVxdWlyZV9fLnQgPSBmdW5jdGlvbih2YWx1ZSwgbW9kZSkge1xuIFx0XHRpZihtb2RlICYgMSkgdmFsdWUgPSBfX3dlYnBhY2tfcmVxdWlyZV9fKHZhbHVlKTtcbiBcdFx0aWYobW9kZSAmIDgpIHJldHVybiB2YWx1ZTtcbiBcdFx0aWYoKG1vZGUgJiA0KSAmJiB0eXBlb2YgdmFsdWUgPT09ICdvYmplY3QnICYmIHZhbHVlICYmIHZhbHVlLl9fZXNNb2R1bGUpIHJldHVybiB2YWx1ZTtcbiBcdFx0dmFyIG5zID0gT2JqZWN0LmNyZWF0ZShudWxsKTtcbiBcdFx0X193ZWJwYWNrX3JlcXVpcmVfXy5yKG5zKTtcbiBcdFx0T2JqZWN0LmRlZmluZVByb3BlcnR5KG5zLCAnZGVmYXVsdCcsIHsgZW51bWVyYWJsZTogdHJ1ZSwgdmFsdWU6IHZhbHVlIH0pO1xuIFx0XHRpZihtb2RlICYgMiAmJiB0eXBlb2YgdmFsdWUgIT0gJ3N0cmluZycpIGZvcih2YXIga2V5IGluIHZhbHVlKSBfX3dlYnBhY2tfcmVxdWlyZV9fLmQobnMsIGtleSwgZnVuY3Rpb24oa2V5KSB7IHJldHVybiB2YWx1ZVtrZXldOyB9LmJpbmQobnVsbCwga2V5KSk7XG4gXHRcdHJldHVybiBucztcbiBcdH07XG5cbiBcdC8vIGdldERlZmF1bHRFeHBvcnQgZnVuY3Rpb24gZm9yIGNvbXBhdGliaWxpdHkgd2l0aCBub24taGFybW9ueSBtb2R1bGVzXG4gXHRfX3dlYnBhY2tfcmVxdWlyZV9fLm4gPSBmdW5jdGlvbihtb2R1bGUpIHtcbiBcdFx0dmFyIGdldHRlciA9IG1vZHVsZSAmJiBtb2R1bGUuX19lc01vZHVsZSA/XG4gXHRcdFx0ZnVuY3Rpb24gZ2V0RGVmYXVsdCgpIHsgcmV0dXJuIG1vZHVsZVsnZGVmYXVsdCddOyB9IDpcbiBcdFx0XHRmdW5jdGlvbiBnZXRNb2R1bGVFeHBvcnRzKCkgeyByZXR1cm4gbW9kdWxlOyB9O1xuIFx0XHRfX3dlYnBhY2tfcmVxdWlyZV9fLmQoZ2V0dGVyLCAnYScsIGdldHRlcik7XG4gXHRcdHJldHVybiBnZXR0ZXI7XG4gXHR9O1xuXG4gXHQvLyBPYmplY3QucHJvdG90eXBlLmhhc093blByb3BlcnR5LmNhbGxcbiBcdF9fd2VicGFja19yZXF1aXJlX18ubyA9IGZ1bmN0aW9uKG9iamVjdCwgcHJvcGVydHkpIHsgcmV0dXJuIE9iamVjdC5wcm90b3R5cGUuaGFzT3duUHJvcGVydHkuY2FsbChvYmplY3QsIHByb3BlcnR5KTsgfTtcblxuIFx0Ly8gX193ZWJwYWNrX3B1YmxpY19wYXRoX19cbiBcdF9fd2VicGFja19yZXF1aXJlX18ucCA9IFwiXCI7XG5cblxuIFx0Ly8gTG9hZCBlbnRyeSBtb2R1bGUgYW5kIHJldHVybiBleHBvcnRzXG4gXHRyZXR1cm4gX193ZWJwYWNrX3JlcXVpcmVfXyhfX3dlYnBhY2tfcmVxdWlyZV9fLnMgPSBcIi4vbWFpbi5qc1wiKTtcbiIsIi8qKlxuICogQ2VsbEhhbmRsZXIgUHJpbWFyeSBNZXNzYWdlIEhhbmRsZXJcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjbGFzcyBpbXBsZW1lbnRzIGEgc2VydmljZSB0aGF0IGhhbmRsZXNcbiAqIG1lc3NhZ2VzIG9mIGFsbCBraW5kcyB0aGF0IGNvbWUgaW4gb3ZlciBhXG4gKiBgQ2VsbFNvY2tldGAuXG4gKiBOT1RFOiBGb3IgdGhlIG1vbWVudCB0aGVyZSBhcmUgb25seSB0d28ga2luZHNcbiAqIG9mIG1lc3NhZ2VzIGFuZCB0aGVyZWZvcmUgdHdvIGhhbmRsZXJzLiBXZSBoYXZlXG4gKiBwbGFucyB0byBjaGFuZ2UgdGhpcyBzdHJ1Y3R1cmUgdG8gYmUgbW9yZSBmbGV4aWJsZVxuICogYW5kIHNvIHRoZSBBUEkgb2YgdGhpcyBjbGFzcyB3aWxsIGNoYW5nZSBncmVhdGx5LlxuICovXG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuY2xhc3MgQ2VsbEhhbmRsZXIge1xuICAgIGNvbnN0cnVjdG9yKGgsIHByb2plY3RvciwgY29tcG9uZW50cyl7XG5cdC8vIHByb3BzXG5cdHRoaXMuaCA9IGg7XG5cdHRoaXMucHJvamVjdG9yID0gcHJvamVjdG9yO1xuXHR0aGlzLmNvbXBvbmVudHMgPSBjb21wb25lbnRzO1xuXG5cdC8vIEluc3RhbmNlIFByb3BzXG4gICAgICAgIHRoaXMucG9zdHNjcmlwdHMgPSBbXTtcbiAgICAgICAgdGhpcy5jZWxscyA9IHt9O1xuXHR0aGlzLkRPTVBhcnNlciA9IG5ldyBET01QYXJzZXIoKTtcblxuICAgICAgICAvLyBCaW5kIEluc3RhbmNlIE1ldGhvZHNcbiAgICAgICAgdGhpcy5zaG93Q29ubmVjdGlvbkNsb3NlZCA9IHRoaXMuc2hvd0Nvbm5lY3Rpb25DbG9zZWQuYmluZCh0aGlzKTtcblx0dGhpcy5jb25uZWN0aW9uQ2xvc2VkVmlldyA9IHRoaXMuY29ubmVjdGlvbkNsb3NlZFZpZXcuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5oYW5kbGVQb3N0c2NyaXB0ID0gdGhpcy5oYW5kbGVQb3N0c2NyaXB0LmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuaGFuZGxlTWVzc2FnZSA9IHRoaXMuaGFuZGxlTWVzc2FnZS5iaW5kKHRoaXMpO1xuXG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogRmlsbHMgdGhlIHBhZ2UncyBwcmltYXJ5IGRpdiB3aXRoXG4gICAgICogYW4gaW5kaWNhdG9yIHRoYXQgdGhlIHNvY2tldCBoYXMgYmVlblxuICAgICAqIGRpc2Nvbm5lY3RlZC5cbiAgICAgKi9cbiAgICBzaG93Q29ubmVjdGlvbkNsb3NlZCgpe1xuXHR0aGlzLnByb2plY3Rvci5yZXBsYWNlKFxuXHQgICAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoXCJwYWdlX3Jvb3RcIiksXG5cdCAgICB0aGlzLmNvbm5lY3Rpb25DbG9zZWRWaWV3XG5cdCk7XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogQ29udmVuaWVuY2UgbWV0aG9kIHRoYXQgdXBkYXRlc1xuICAgICAqIEJvb3RzdHJhcC1zdHlsZSBwb3BvdmVycyBvblxuICAgICAqIHRoZSBET00uXG4gICAgICogU2VlIGlubGluZSBjb21tZW50c1xuICAgICAqL1xuICAgIHVwZGF0ZVBvcG92ZXJzKCkge1xuICAgICAgICAvLyBUaGlzIGZ1bmN0aW9uIHJlcXVpcmVzXG4gICAgICAgIC8vIGpRdWVyeSBhbmQgcGVyaGFwcyBkb2Vzbid0XG4gICAgICAgIC8vIGJlbG9uZyBpbiB0aGlzIGNsYXNzLlxuICAgICAgICAvLyBUT0RPOiBGaWd1cmUgb3V0IGEgYmV0dGVyIHdheVxuICAgICAgICAvLyBBTFNPIE5PVEU6XG4gICAgICAgIC8vIC0tLS0tLS0tLS0tLS0tLS0tXG4gICAgICAgIC8vIGBnZXRDaGlsZFByb3BgIGlzIGEgY29uc3QgZnVuY3Rpb25cbiAgICAgICAgLy8gdGhhdCBpcyBkZWNsYXJlZCBpbiBhIHNlcGFyYXRlXG4gICAgICAgIC8vIHNjcmlwdCB0YWcgYXQgdGhlIGJvdHRvbSBvZlxuICAgICAgICAvLyBwYWdlLmh0bWwuIFRoYXQncyBhIG5vLW5vIVxuICAgICAgICAkKCdbZGF0YS10b2dnbGU9XCJwb3BvdmVyXCJdJykucG9wb3Zlcih7XG4gICAgICAgICAgICBodG1sOiB0cnVlLFxuICAgICAgICAgICAgY29udGFpbmVyOiAnYm9keScsXG4gICAgICAgICAgICB0aXRsZTogZnVuY3Rpb24gKCkge1xuICAgICAgICAgICAgICAgIHJldHVybiBnZXRDaGlsZFByb3AodGhpcywgJ3RpdGxlJyk7XG4gICAgICAgICAgICB9LFxuICAgICAgICAgICAgY29udGVudDogZnVuY3Rpb24gKCkge1xuICAgICAgICAgICAgICAgIHJldHVybiBnZXRDaGlsZFByb3AodGhpcywgJ2NvbnRlbnQnKTtcbiAgICAgICAgICAgIH0sXG4gICAgICAgICAgICBwbGFjZW1lbnQ6IGZ1bmN0aW9uIChwb3BwZXJFbCwgdHJpZ2dlcmluZ0VsKSB7XG4gICAgICAgICAgICAgICAgbGV0IHBsYWNlbWVudCA9IHRyaWdnZXJpbmdFbC5kYXRhc2V0LnBsYWNlbWVudDtcbiAgICAgICAgICAgICAgICBpZihwbGFjZW1lbnQgPT0gdW5kZWZpbmVkKXtcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIFwiYm90dG9tXCI7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIHJldHVybiBwbGFjZW1lbnQ7XG4gICAgICAgICAgICB9XG4gICAgICAgIH0pO1xuICAgICAgICAkKCcucG9wb3Zlci1kaXNtaXNzJykucG9wb3Zlcih7XG4gICAgICAgICAgICB0cmlnZ2VyOiAnZm9jdXMnXG4gICAgICAgIH0pO1xuICAgIH1cblxuICAgIC8qKlxuICAgICAqIFByaW1hcnkgbWV0aG9kIGZvciBoYW5kbGluZ1xuICAgICAqICdwb3N0c2NyaXB0cycgbWVzc2FnZXMsIHdoaWNoIHRlbGxcbiAgICAgKiB0aGlzIG9iamVjdCB0byBnbyB0aHJvdWdoIGl0J3MgYXJyYXlcbiAgICAgKiBvZiBzY3JpcHQgc3RyaW5ncyBhbmQgdG8gZXZhbHVhdGUgdGhlbS5cbiAgICAgKiBUaGUgZXZhbHVhdGlvbiBpcyBkb25lIG9uIHRoZSBnbG9iYWxcbiAgICAgKiB3aW5kb3cgb2JqZWN0IGV4cGxpY2l0bHkuXG4gICAgICogTk9URTogRnV0dXJlIHJlZmFjdG9yaW5ncy9yZXN0cnVjdHVyaW5nc1xuICAgICAqIHdpbGwgcmVtb3ZlIG11Y2ggb2YgdGhlIG5lZWQgdG8gY2FsbCBldmFsIVxuICAgICAqIEBwYXJhbSB7c3RyaW5nfSBtZXNzYWdlIC0gVGhlIGluY29taW5nIHN0cmluZ1xuICAgICAqIGZyb20gdGhlIHNvY2tldC5cbiAgICAgKi9cbiAgICBoYW5kbGVQb3N0c2NyaXB0KG1lc3NhZ2Upe1xuICAgICAgICAvLyBFbHNld2hlcmUsIHVwZGF0ZSBwb3BvdmVycyBmaXJzdFxuICAgICAgICAvLyBOb3cgd2UgZXZhbHVhdGUgc2NyaXB0cyBjb21pbmdcbiAgICAgICAgLy8gYWNyb3NzIHRoZSB3aXJlLlxuICAgICAgICB0aGlzLnVwZGF0ZVBvcG92ZXJzKCk7XG4gICAgICAgIHdoaWxlKHRoaXMucG9zdHNjcmlwdHMubGVuZ3RoKXtcblx0ICAgIGxldCBwb3N0c2NyaXB0ID0gdGhpcy5wb3N0c2NyaXB0cy5wb3AoKTtcblx0ICAgIHRyeSB7XG5cdFx0d2luZG93LmV2YWwocG9zdHNjcmlwdCk7XG5cdCAgICB9IGNhdGNoKGUpe1xuICAgICAgICAgICAgICAgIGNvbnNvbGUuZXJyb3IoXCJFUlJPUiBSVU5OSU5HIFBPU1RTQ1JJUFRcIiwgZSk7XG4gICAgICAgICAgICAgICAgY29uc29sZS5sb2cocG9zdHNjcmlwdCk7XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBQcmltYXJ5IG1ldGhvZCBmb3IgaGFuZGxpbmcgJ25vcm1hbCdcbiAgICAgKiAoaWUgbm9uLXBvc3RzY3JpcHRzKSBtZXNzYWdlcyB0aGF0IGhhdmVcbiAgICAgKiBiZWVuIGRlc2VyaWFsaXplZCBmcm9tIEpTT04uXG4gICAgICogRm9yIHRoZSBtb21lbnQsIHRoZXNlIG1lc3NhZ2VzIGRlYWxcbiAgICAgKiBlbnRpcmVseSB3aXRoIERPTSByZXBsYWNlbWVudCBvcGVyYXRpb25zLCB3aGljaFxuICAgICAqIHRoaXMgbWV0aG9kIGltcGxlbWVudHMuXG4gICAgICogQHBhcmFtIHtvYmplY3R9IG1lc3NhZ2UgLSBBIGRlc2VyaWFsaXplZFxuICAgICAqIEpTT04gbWVzc2FnZSBmcm9tIHRoZSBzZXJ2ZXIgdGhhdCBoYXNcbiAgICAgKiBpbmZvcm1hdGlvbiBhYm91dCBlbGVtZW50cyB0aGF0IG5lZWQgdG9cbiAgICAgKiBiZSB1cGRhdGVkLlxuICAgICAqL1xuICAgIGhhbmRsZU1lc3NhZ2UobWVzc2FnZSl7XG4gICAgICAgIGxldCBuZXdDb21wb25lbnRzID0gW107XG5cdGlmKHRoaXMuY2VsbHNbXCJwYWdlX3Jvb3RcIl0gPT0gdW5kZWZpbmVkKXtcbiAgICAgICAgICAgIHRoaXMuY2VsbHNbXCJwYWdlX3Jvb3RcIl0gPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZChcInBhZ2Vfcm9vdFwiKTtcbiAgICAgICAgICAgIHRoaXMuY2VsbHNbXCJob2xkaW5nX3BlblwiXSA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKFwiaG9sZGluZ19wZW5cIik7XG4gICAgICAgIH1cblx0Ly8gV2l0aCB0aGUgZXhjZXB0aW9uIG9mIGBwYWdlX3Jvb3RgIGFuZCBgaG9sZGluZ19wZW5gIGlkIG5vZGVzLCBhbGxcblx0Ly8gZWxlbWVudHMgaW4gdGhpcy5jZWxscyBhcmUgdmlydHVhbC4gRGVwZW5kaWcgb24gd2hldGhlciB3ZSBhcmUgYWRkaW5nIGFcblx0Ly8gbmV3IG5vZGUsIG9yIG1hbmlwdWxhdGluZyBhbiBleGlzdGluZywgd2UgbmVlZWQgdG8gd29yayB3aXRoIHRoZSB1bmRlcmx5aW5nXG5cdC8vIERPTSBub2RlLiBIZW5jZSBpZiB0aGlzLmNlbGxbbWVzc2FnZS5pZF0gaXMgYSB2ZG9tIGVsZW1lbnQgd2UgdXNlIGl0c1xuXHQvLyB1bmRlcmx5aW5nIGRvbU5vZGUgZWxlbWVudCB3aGVuIGluIG9wZXJhdGlvbnMgbGlrZSB0aGlzLnByb2plY3Rvci5yZXBsYWNlKClcblx0bGV0IGNlbGwgPSB0aGlzLmNlbGxzW21lc3NhZ2UuaWRdO1xuXG5cdGlmIChjZWxsICE9PSB1bmRlZmluZWQgJiYgY2VsbC5kb21Ob2RlICE9PSB1bmRlZmluZWQpIHtcblx0ICAgIGNlbGwgPSBjZWxsLmRvbU5vZGU7XG5cdH1cblxuICAgICAgICBpZihtZXNzYWdlLmRpc2NhcmQgIT09IHVuZGVmaW5lZCl7XG4gICAgICAgICAgICAvLyBJbiB0aGUgY2FzZSB3aGVyZSB3ZSBoYXZlIHJlY2VpdmVkIGEgJ2Rpc2NhcmQnIG1lc3NhZ2UsXG4gICAgICAgICAgICAvLyBidXQgdGhlIGNlbGwgcmVxdWVzdGVkIGlzIG5vdCBhdmFpbGFibGUgaW4gb3VyXG4gICAgICAgICAgICAvLyBjZWxscyBjb2xsZWN0aW9uLCB3ZSBzaW1wbHkgZGlzcGxheSBhIHdhcm5pbmc6XG4gICAgICAgICAgICBpZihjZWxsID09IHVuZGVmaW5lZCl7XG4gICAgICAgICAgICAgICAgY29uc29sZS53YXJuKGBSZWNlaXZlZCBkaXNjYXJkIG1lc3NhZ2UgZm9yIG5vbi1leGlzdGluZyBjZWxsIGlkICR7bWVzc2FnZS5pZH1gKTtcbiAgICAgICAgICAgICAgICByZXR1cm47XG4gICAgICAgICAgICB9XG5cdCAgICAvLyBJbnN0ZWFkIG9mIHJlbW92aW5nIHRoZSBub2RlIHdlIHJlcGxhY2Ugd2l0aCB0aGUgYVxuXHQgICAgLy8gYGRpc3BsYXk6bm9uZWAgc3R5bGUgbm9kZSB3aGljaCBlZmZlY3RpdmVseSByZW1vdmVzIGl0XG5cdCAgICAvLyBmcm9tIHRoZSBET01cblx0ICAgIGlmIChjZWxsLnBhcmVudE5vZGUgIT09IG51bGwpIHtcblx0XHR0aGlzLnByb2plY3Rvci5yZXBsYWNlKGNlbGwsICgpID0+IHtcblx0XHQgICAgcmV0dXJuIGgoXCJkaXZcIiwge3N0eWxlOiBcImRpc3BsYXk6bm9uZVwifSwgW10pO1xuXHRcdH0pO1xuXHQgICAgfVxuXHR9IGVsc2UgaWYobWVzc2FnZS5pZCAhPT0gdW5kZWZpbmVkKXtcblx0ICAgIC8vIEEgZGljdGlvbmFyeSBvZiBpZHMgd2l0aGluIHRoZSBvYmplY3QgdG8gcmVwbGFjZS5cblx0ICAgIC8vIFRhcmdldHMgYXJlIHJlYWwgaWRzIG9mIG90aGVyIG9iamVjdHMuXG5cdCAgICBsZXQgcmVwbGFjZW1lbnRzID0gbWVzc2FnZS5yZXBsYWNlbWVudHM7XG5cblx0ICAgIC8vIFRPRE86IHRoaXMgaXMgYSB0ZW1wb3JhcnkgYnJhbmNoaW5nLCB0byBiZSByZW1vdmVkIHdpdGggYSBtb3JlIGxvZ2ljYWwgc2V0dXAuIEFzXG5cdCAgICAvLyBvZiB3cml0aW5nIGlmIHRoZSBtZXNzYWdlIGNvbWluZyBhY3Jvc3MgaXMgc2VuZGluZyBhIFwia25vd25cIiBjb21wb25lbnQgdGhlbiB3ZSB1c2Vcblx0ICAgIC8vIHRoZSBjb21wb25lbnQgaXRzZWxmIGFzIG9wcG9zZWQgdG8gYnVpbGRpbmcgYSB2ZG9tIGVsZW1lbnQgZnJvbSB0aGUgcmF3IGh0bWxcblx0ICAgIGxldCBjb21wb25lbnRDbGFzcyA9IHRoaXMuY29tcG9uZW50c1ttZXNzYWdlLmNvbXBvbmVudF9uYW1lXTtcblx0ICAgIGlmIChjb21wb25lbnRDbGFzcyA9PT0gdW5kZWZpbmVkKSB7XG4gICAgICAgICAgICAgICAgY29uc29sZS53YXJuKGBDb3VsZCBub3QgZmluZCBjb21wb25lbnQgZm9yICR7bWVzc2FnZS5jb21wb25lbnRfbmFtZX1gKTtcblx0XHR2YXIgdmVsZW1lbnQgPSB0aGlzLmh0bWxUb1ZEb21FbChtZXNzYWdlLmNvbnRlbnRzLCBtZXNzYWdlLmlkKTtcblx0ICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgbGV0IGNvbXBvbmVudFByb3BzID0gT2JqZWN0LmFzc2lnbih7XG4gICAgICAgICAgICAgICAgICAgIGlkOiBtZXNzYWdlLmlkLFxuICAgICAgICAgICAgICAgICAgICBuYW1lZENoaWxkcmVuOiBtZXNzYWdlLm5hbWVkQ2hpbGRyZW4sXG4gICAgICAgICAgICAgICAgICAgIGNoaWxkcmVuOiBtZXNzYWdlLmNoaWxkcmVuLFxuICAgICAgICAgICAgICAgICAgICBleHRyYURhdGE6IG1lc3NhZ2UuZXh0cmFfZGF0YVxuICAgICAgICAgICAgICAgIH0sIG1lc3NhZ2UuZXh0cmFfZGF0YSk7XG5cdFx0dmFyIGNvbXBvbmVudCA9IG5ldyBjb21wb25lbnRDbGFzcyhcbiAgICAgICAgICAgICAgICAgICAgY29tcG9uZW50UHJvcHMsXG4gICAgICAgICAgICAgICAgICAgIG1lc3NhZ2UucmVwbGFjZW1lbnRfa2V5c1xuICAgICAgICAgICAgICAgICk7XG4gICAgICAgICAgICAgICAgdmFyIHZlbGVtZW50ID0gY29tcG9uZW50LnJlbmRlcigpO1xuICAgICAgICAgICAgICAgIG5ld0NvbXBvbmVudHMucHVzaChjb21wb25lbnQpO1xuXHQgICAgfVxuXG4gICAgICAgICAgICAvLyBJbnN0YWxsIHRoZSBlbGVtZW50IGludG8gdGhlIGRvbVxuICAgICAgICAgICAgaWYoY2VsbCA9PT0gdW5kZWZpbmVkKXtcblx0XHQvLyBUaGlzIGlzIGEgdG90YWxseSBuZXcgbm9kZS5cbiAgICAgICAgICAgICAgICAvLyBGb3IgdGhlIG1vbWVudCwgYWRkIGl0IHRvIHRoZVxuICAgICAgICAgICAgICAgIC8vIGhvbGRpbmcgcGVuLlxuXHRcdHRoaXMucHJvamVjdG9yLmFwcGVuZCh0aGlzLmNlbGxzW1wiaG9sZGluZ19wZW5cIl0sICgpID0+IHtcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIHZlbGVtZW50O1xuICAgICAgICAgICAgICAgIH0pO1xuXG5cdFx0dGhpcy5jZWxsc1ttZXNzYWdlLmlkXSA9IHZlbGVtZW50O1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICAvLyBSZXBsYWNlIHRoZSBleGlzdGluZyBjb3B5IG9mXG4gICAgICAgICAgICAgICAgLy8gdGhlIG5vZGUgd2l0aCB0aGlzIGluY29taW5nXG4gICAgICAgICAgICAgICAgLy8gY29weS5cblx0XHRpZihjZWxsLnBhcmVudE5vZGUgPT09IG51bGwpe1xuXHRcdCAgICB0aGlzLnByb2plY3Rvci5hcHBlbmQodGhpcy5jZWxsc1tcImhvbGRpbmdfcGVuXCJdLCAoKSA9PiB7XG5cdFx0XHRyZXR1cm4gdmVsZW1lbnQ7XG5cdFx0ICAgIH0pO1xuXHRcdH0gZWxzZSB7XG5cdFx0ICAgIHRoaXMucHJvamVjdG9yLnJlcGxhY2UoY2VsbCwgKCkgPT4ge3JldHVybiB2ZWxlbWVudDt9KTtcblx0XHR9XG5cdCAgICB9XG5cbiAgICAgICAgICAgIHRoaXMuY2VsbHNbbWVzc2FnZS5pZF0gPSB2ZWxlbWVudDtcblxuICAgICAgICAgICAgLy8gTm93IHdpcmUgaW4gcmVwbGFjZW1lbnRzXG4gICAgICAgICAgICBPYmplY3Qua2V5cyhyZXBsYWNlbWVudHMpLmZvckVhY2goKHJlcGxhY2VtZW50S2V5LCBpZHgpID0+IHtcbiAgICAgICAgICAgICAgICBsZXQgdGFyZ2V0ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQocmVwbGFjZW1lbnRLZXkpO1xuICAgICAgICAgICAgICAgIGxldCBzb3VyY2UgPSBudWxsO1xuICAgICAgICAgICAgICAgIGlmKHRoaXMuY2VsbHNbcmVwbGFjZW1lbnRzW3JlcGxhY2VtZW50S2V5XV0gPT09IHVuZGVmaW5lZCl7XG5cdFx0ICAgIC8vIFRoaXMgaXMgYWN0dWFsbHkgYSBuZXcgbm9kZS5cbiAgICAgICAgICAgICAgICAgICAgLy8gV2UnbGwgZGVmaW5lIGl0IGxhdGVyIGluIHRoZVxuICAgICAgICAgICAgICAgICAgICAvLyBldmVudCBzdHJlYW0uXG5cdFx0ICAgIHNvdXJjZSA9IHRoaXMuaChcImRpdlwiLCB7aWQ6IHJlcGxhY2VtZW50S2V5fSwgW10pO1xuICAgICAgICAgICAgICAgICAgICB0aGlzLmNlbGxzW3JlcGxhY2VtZW50c1tyZXBsYWNlbWVudEtleV1dID0gc291cmNlOyBcblx0XHQgICAgdGhpcy5wcm9qZWN0b3IuYXBwZW5kKHRoaXMuY2VsbHNbXCJob2xkaW5nX3BlblwiXSwgKCkgPT4ge1xuXHRcdFx0cmV0dXJuIHNvdXJjZTtcbiAgICAgICAgICAgICAgICAgICAgfSk7XG5cdFx0fSBlbHNlIHtcbiAgICAgICAgICAgICAgICAgICAgLy8gTm90IGEgbmV3IG5vZGVcbiAgICAgICAgICAgICAgICAgICAgc291cmNlID0gdGhpcy5jZWxsc1tyZXBsYWNlbWVudHNbcmVwbGFjZW1lbnRLZXldXTtcbiAgICAgICAgICAgICAgICB9XG5cbiAgICAgICAgICAgICAgICBpZih0YXJnZXQgIT0gbnVsbCl7XG5cdFx0ICAgIHRoaXMucHJvamVjdG9yLnJlcGxhY2UodGFyZ2V0LCAoKSA9PiB7XG5cdFx0XHRyZXR1cm4gc291cmNlO1xuICAgICAgICAgICAgICAgICAgICB9KTtcbiAgICAgICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgICAgICBsZXQgZXJyb3JNc2cgPSBgSW4gbWVzc2FnZSAke21lc3NhZ2V9IGNvdWxkbid0IGZpbmQgJHtyZXBsYWNlbWVudEtleX1gO1xuICAgICAgICAgICAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoZXJyb3JNc2cpO1xuICAgICAgICAgICAgICAgICAgICAvL2NvbnNvbGUubG9nKFwiSW4gbWVzc2FnZSBcIiwgbWVzc2FnZSwgXCIgY291bGRuJ3QgZmluZCBcIiwgcmVwbGFjZW1lbnRLZXkpO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH0pO1xuICAgICAgICB9XG5cbiAgICAgICAgaWYobWVzc2FnZS5wb3N0c2NyaXB0ICE9PSB1bmRlZmluZWQpe1xuICAgICAgICAgICAgdGhpcy5wb3N0c2NyaXB0cy5wdXNoKG1lc3NhZ2UucG9zdHNjcmlwdCk7XG4gICAgICAgIH1cblxuICAgICAgICAvLyBJZiB3ZSBjcmVhdGVkIGFueSBuZXcgY29tcG9uZW50cyBkdXJpbmcgdGhpc1xuICAgICAgICAvLyBtZXNzYWdlIGhhbmRsaW5nIHNlc3Npb24sIHdlIGZpbmFsbHkgY2FsbFxuICAgICAgICAvLyB0aGVpciBgY29tcG9uZW50RGlkTG9hZGAgbGlmZWN5Y2xlIG1ldGhvZHNcbiAgICAgICAgbmV3Q29tcG9uZW50cy5mb3JFYWNoKGNvbXBvbmVudCA9PiB7XG4gICAgICAgICAgICBjb21wb25lbnQuY29tcG9uZW50RGlkTG9hZCgpO1xuICAgICAgICB9KTtcblxuICAgICAgICAvLyBSZW1vdmUgbGVmdG92ZXIgcmVwbGFjZW1lbnQgZGl2c1xuICAgICAgICAvLyB0aGF0IGFyZSBzdGlsbCBpbiB0aGUgcGFnZV9yb290XG4gICAgICAgIC8vIGFmdGVyIHZkb20gaW5zZXJ0aW9uXG4gICAgICAgIGxldCBwYWdlUm9vdCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdwYWdlX3Jvb3QnKTtcbiAgICAgICAgbGV0IGZvdW5kID0gcGFnZVJvb3QucXVlcnlTZWxlY3RvckFsbCgnW2lkKj1cIl9fX19fXCJdJyk7XG4gICAgICAgIGZvdW5kLmZvckVhY2goZWxlbWVudCA9PiB7XG4gICAgICAgICAgICBlbGVtZW50LnJlbW92ZSgpO1xuICAgICAgICB9KTtcbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBIZWxwZXIgZnVuY3Rpb24gdGhhdCBnZW5lcmF0ZXMgdGhlIHZkb20gTm9kZSBmb3JcbiAgICAgKiB0byBiZSBkaXNwbGF5IHdoZW4gY29ubmVjdGlvbiBjbG9zZXNcbiAgICAgKi9cbiAgICBjb25uZWN0aW9uQ2xvc2VkVmlldygpe1xuXHRyZXR1cm4gdGhpcy5oKFwibWFpbi5jb250YWluZXJcIiwge3JvbGU6IFwibWFpblwifSwgW1xuXHQgICAgdGhpcy5oKFwiZGl2XCIsIHtjbGFzczogXCJhbGVydCBhbGVydC1wcmltYXJ5IGNlbnRlci1ibG9jayBtdC01XCJ9LFxuXHRcdFtcIkRpc2Nvbm5lY3RlZFwiXSlcblx0XSk7XG4gICAgfVxuXG4gICAgICAgIC8qKlxuICAgICAqIFRoaXMgaXMgYSAoaG9wZWZ1bGx5IHRlbXBvcmFyeSkgaGFja1xuICAgICAqIHRoYXQgd2lsbCBpbnRlcmNlcHQgdGhlIGZpcnN0IHRpbWUgYVxuICAgICAqIGRyb3Bkb3duIGNhcmF0IGlzIGNsaWNrZWQgYW5kIGJpbmRcbiAgICAgKiBCb290c3RyYXAgRHJvcGRvd24gZXZlbnQgaGFuZGxlcnNcbiAgICAgKiB0byBpdCB0aGF0IHNob3VsZCBiZSBib3VuZCB0byB0aGVcbiAgICAgKiBpZGVudGlmaWVkIGNlbGwuIFdlIGFyZSBmb3JjZWQgdG8gZG8gdGhpc1xuICAgICAqIGJlY2F1c2UgdGhlIGN1cnJlbnQgQ2VsbHMgaW5mcmFzdHJ1Y3R1cmVcbiAgICAgKiBkb2VzIG5vdCBoYXZlIGZsZXhpYmxlIGV2ZW50IGJpbmRpbmcvaGFuZGxpbmcuXG4gICAgICogQHBhcmFtIHtzdHJpbmd9IGNlbGxJZCAtIFRoZSBJRCBvZiB0aGUgY2VsbFxuICAgICAqIHRvIGlkZW50aWZ5IGluIHRoZSBzb2NrZXQgY2FsbGJhY2sgd2Ugd2lsbFxuICAgICAqIGJpbmQgdG8gb3BlbiBhbmQgY2xvc2UgZXZlbnRzIG9uIGRyb3Bkb3duXG4gICAgICovXG4gICAgZHJvcGRvd25Jbml0aWFsQmluZEZvcihjZWxsSWQpe1xuICAgICAgICBsZXQgZWxlbWVudElkID0gY2VsbElkICsgJy1kcm9wZG93bk1lbnVCdXR0b24nO1xuICAgICAgICBsZXQgZWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKGVsZW1lbnRJZCk7XG4gICAgICAgIGlmKCFlbGVtZW50KXtcbiAgICAgICAgICAgIHRocm93IEVycm9yKCdFbGVtZW50IG9mIGlkICcgKyBlbGVtZW50SWQgKyAnIGRvZXNudCBleGlzdCEnKTtcbiAgICAgICAgfVxuICAgICAgICBsZXQgZHJvcGRvd25NZW51ID0gZWxlbWVudC5wYXJlbnRFbGVtZW50O1xuICAgICAgICBsZXQgZmlyc3RUaW1lQ2xpY2tlZCA9IGVsZW1lbnQuZGF0YXNldC5maXJzdGNsaWNrID09ICd0cnVlJztcbiAgICAgICAgaWYoZmlyc3RUaW1lQ2xpY2tlZCl7XG4gICAgICAgICAgICAkKGRyb3Bkb3duTWVudSkub24oJ3Nob3cuYnMuZHJvcGRvd24nLCBmdW5jdGlvbigpe1xuICAgICAgICAgICAgICAgIGNlbGxTb2NrZXQuc2VuZFN0cmluZyhKU09OLnN0cmluZ2lmeSh7XG4gICAgICAgICAgICAgICAgICAgIGV2ZW50OiAnZHJvcGRvd24nLFxuICAgICAgICAgICAgICAgICAgICB0YXJnZXRfY2VsbDogY2VsbElkLFxuICAgICAgICAgICAgICAgICAgICBpc09wZW46IGZhbHNlXG4gICAgICAgICAgICAgICAgfSkpO1xuICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICAkKGRyb3Bkb3duTWVudSkub24oJ2hpZGUuYnMuZHJvcGRvd24nLCBmdW5jdGlvbigpe1xuICAgICAgICAgICAgICAgIGNlbGxTb2NrZXQuc2VuZFN0cmluZyhKU09OLnN0cmluZ2lmeSh7XG4gICAgICAgICAgICAgICAgICAgIGV2ZW50OiAnZHJvcGRvd24nLFxuICAgICAgICAgICAgICAgICAgICB0YXJnZXRfY2VsbDogY2VsbElkLFxuICAgICAgICAgICAgICAgICAgICBpc09wZW46IHRydWVcbiAgICAgICAgICAgICAgICB9KSk7XG4gICAgICAgICAgICB9KTtcblxuICAgICAgICAgICAgLy8gTm93IGV4cGlyZSB0aGUgZmlyc3QgdGltZSBjbGlja2VkXG4gICAgICAgICAgICBlbGVtZW50LmRhdGFzZXQuZmlyc3RjbGljayA9ICdmYWxzZSc7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBVbnNhZmVseSBleGVjdXRlcyBhbnkgcGFzc2VkIGluIHN0cmluZ1xuICAgICAqIGFzIGlmIGl0IGlzIHZhbGlkIEpTIGFnYWluc3QgdGhlIGdsb2JhbFxuICAgICAqIHdpbmRvdyBzdGF0ZS5cbiAgICAgKi9cbiAgICBzdGF0aWMgdW5zYWZlbHlFeGVjdXRlKGFTdHJpbmcpe1xuICAgICAgICB3aW5kb3cuZXhlYyhhU3RyaW5nKTtcbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBIZWxwZXIgZnVuY3Rpb24gdGhhdCB0YWtlcyBzb21lIGluY29taW5nXG4gICAgICogSFRNTCBzdHJpbmcgYW5kIHJldHVybnMgYSBtYXF1ZXR0ZSBoeXBlcnNjcmlwdFxuICAgICAqIFZET00gZWxlbWVudCBmcm9tIGl0LlxuICAgICAqIFRoaXMgdXNlcyB0aGUgaW50ZXJuYWwgYnJvd3NlciBET01wYXJzZXIoKSB0byBnZW5lcmF0ZSB0aGUgaHRtbFxuICAgICAqIHN0cnVjdHVyZSBmcm9tIHRoZSByYXcgc3RyaW5nIGFuZCB0aGVuIHJlY3Vyc2l2ZWx5IGJ1aWxkIHRoZVxuICAgICAqIFZET00gZWxlbWVudFxuICAgICAqIEBwYXJhbSB7c3RyaW5nfSBodG1sIC0gVGhlIG1hcmt1cCB0b1xuICAgICAqIHRyYW5zZm9ybSBpbnRvIGEgcmVhbCBlbGVtZW50LlxuICAgICAqL1xuICAgIGh0bWxUb1ZEb21FbChodG1sLCBpZCl7XG5cdGxldCBkb20gPSB0aGlzLkRPTVBhcnNlci5wYXJzZUZyb21TdHJpbmcoaHRtbCwgXCJ0ZXh0L2h0bWxcIik7XG4gICAgICAgIGxldCBlbGVtZW50ID0gZG9tLmJvZHkuY2hpbGRyZW5bMF07XG4gICAgICAgIHJldHVybiB0aGlzLl9kb21FbFRvVmRvbUVsKGVsZW1lbnQsIGlkKTtcbiAgICB9XG5cbiAgICBfZG9tRWxUb1Zkb21FbChkb21FbCwgaWQpIHtcblx0bGV0IHRhZ05hbWUgPSBkb21FbC50YWdOYW1lLnRvTG9jYWxlTG93ZXJDYXNlKCk7XG5cdGxldCBhdHRycyA9IHtpZDogaWR9O1xuXHRsZXQgaW5kZXg7XG5cblx0Zm9yIChpbmRleCA9IDA7IGluZGV4IDwgZG9tRWwuYXR0cmlidXRlcy5sZW5ndGg7IGluZGV4Kyspe1xuXHQgICAgbGV0IGl0ZW0gPSBkb21FbC5hdHRyaWJ1dGVzLml0ZW0oaW5kZXgpO1xuXHQgICAgYXR0cnNbaXRlbS5uYW1lXSA9IGl0ZW0udmFsdWUudHJpbSgpO1xuXHR9XG5cblx0aWYgKGRvbUVsLmNoaWxkRWxlbWVudENvdW50ID09PSAwKSB7XG5cdCAgICByZXR1cm4gaCh0YWdOYW1lLCBhdHRycywgW2RvbUVsLnRleHRDb250ZW50XSk7XG5cdH1cblxuXHRsZXQgY2hpbGRyZW4gPSBbXTtcblx0Zm9yIChpbmRleCA9IDA7IGluZGV4IDwgZG9tRWwuY2hpbGRyZW4ubGVuZ3RoOyBpbmRleCsrKXtcblx0ICAgIGxldCBjaGlsZCA9IGRvbUVsLmNoaWxkcmVuW2luZGV4XTtcblx0ICAgIGNoaWxkcmVuLnB1c2godGhpcy5fZG9tRWxUb1Zkb21FbChjaGlsZCkpO1xuXHR9XG5cblx0cmV0dXJuIGgodGFnTmFtZSwgYXR0cnMsIGNoaWxkcmVuKTtcbiAgICB9XG59XG5cbmV4cG9ydCB7Q2VsbEhhbmRsZXIsIENlbGxIYW5kbGVyIGFzIGRlZmF1bHR9XG4iLCIvKipcbiAqIEEgY29uY3JldGUgZXJyb3IgdGhyb3duXG4gKiBpZiB0aGUgY3VycmVudCBicm93c2VyIGRvZXNuJ3RcbiAqIHN1cHBvcnQgd2Vic29ja2V0cywgd2hpY2ggaXMgdmVyeVxuICogdW5saWtlbHkuXG4gKi9cbmNsYXNzIFdlYnNvY2tldE5vdFN1cHBvcnRlZCBleHRlbmRzIEVycm9yIHtcbiAgICBjb25zdHJ1Y3RvcihhcmdzKXtcbiAgICAgICAgc3VwZXIoYXJncyk7XG4gICAgfVxufVxuXG4vKipcbiAqIFRoaXMgaXMgdGhlIGdsb2JhbCBmcmFtZVxuICogY29udHJvbC4gV2UgbWlnaHQgY29uc2lkZXJcbiAqIHB1dHRpbmcgaXQgZWxzZXdoZXJlLCBidXRcbiAqIGBDZWxsU29ja2V0YCBpcyBpdHMgb25seVxuICogY29uc3VtZXIuXG4gKi9cbmNvbnN0IEZSQU1FU19QRVJfQUNLID0gMTA7XG5cblxuLyoqXG4gKiBDZWxsU29ja2V0IENvbnRyb2xsZXJcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjbGFzcyBpbXBsZW1lbnRzIGFuIGluc3RhbmNlIG9mXG4gKiBhIGNvbnRyb2xsZXIgdGhhdCB3cmFwcyBhIHdlYnNvY2tldCBjbGllbnRcbiAqIGNvbm5lY3Rpb24gYW5kIGtub3dzIGhvdyB0byBoYW5kbGUgdGhlXG4gKiBpbml0aWFsIHJvdXRpbmcgb2YgbWVzc2FnZXMgYWNyb3NzIHRoZSBzb2NrZXQuXG4gKiBgQ2VsbFNvY2tldGAgaW5zdGFuY2VzIGFyZSBkZXNpZ25lZCBzbyB0aGF0XG4gKiBoYW5kbGVycyBmb3Igc3BlY2lmaWMgdHlwZXMgb2YgbWVzc2FnZXMgY2FuXG4gKiByZWdpc3RlciB0aGVtc2VsdmVzIHdpdGggaXQuXG4gKiBOT1RFOiBGb3IgdGhlIG1vbWVudCwgbW9zdCBvZiB0aGlzIGNvZGVcbiAqIGhhcyBiZWVuIGNvcGllZCB2ZXJiYXRpbSBmcm9tIHRoZSBpbmxpbmVcbiAqIHNjcmlwdHMgd2l0aCBvbmx5IHNsaWdodCBtb2RpZmljYXRpb24uXG4gKiovXG5jbGFzcyBDZWxsU29ja2V0IHtcbiAgICBjb25zdHJ1Y3Rvcigpe1xuICAgICAgICAvLyBJbnN0YW5jZSBQcm9wc1xuICAgICAgICB0aGlzLnVyaSA9IHRoaXMuZ2V0VXJpKCk7XG4gICAgICAgIHRoaXMuc29ja2V0ID0gbnVsbDtcbiAgICAgICAgdGhpcy5jdXJyZW50QnVmZmVyID0ge1xuICAgICAgICAgICAgcmVtYWluaW5nOiBudWxsLFxuICAgICAgICAgICAgYnVmZmVyOiBudWxsLFxuICAgICAgICAgICAgaGFzRGlzcGxheTogZmFsc2VcbiAgICAgICAgfTtcblxuICAgICAgICAvKipcbiAgICAgICAgICogQSBjYWxsYmFjayBmb3IgaGFuZGxpbmcgbWVzc2FnZXNcbiAgICAgICAgICogdGhhdCBhcmUgJ3Bvc3RzY3JpcHRzJ1xuICAgICAgICAgKiBAY2FsbGJhY2sgcG9zdHNjcmlwdHNIYW5kbGVyXG4gICAgICAgICAqIEBwYXJhbSB7c3RyaW5nfSBtc2cgLSBUaGUgZm9yd2FyZGVkIG1lc3NhZ2VcbiAgICAgICAgICovXG4gICAgICAgIHRoaXMucG9zdHNjcmlwdHNIYW5kZXIgPSBudWxsO1xuXG4gICAgICAgIC8qKlxuICAgICAgICAgKiBBIGNhbGxiYWNrIGZvciBoYW5kbGluZyBtZXNzYWdlc1xuICAgICAgICAgKiB0aGF0IGFyZSBub3JtYWwgSlNPTiBkYXRhIG1lc3NhZ2VzLlxuICAgICAgICAgKiBAY2FsbGJhY2sgbWVzc2FnZUhhbmRsZXJcbiAgICAgICAgICogQHBhcmFtIHtvYmplY3R9IG1zZyAtIFRoZSBmb3J3YXJkZWQgbWVzc2FnZVxuICAgICAgICAgKi9cbiAgICAgICAgdGhpcy5tZXNzYWdlSGFuZGxlciA9IG51bGw7XG5cbiAgICAgICAgLyoqXG4gICAgICAgICAqIEEgY2FsbGJhY2sgZm9yIGhhbmRsaW5nIG1lc3NhZ2VzXG4gICAgICAgICAqIHdoZW4gdGhlIHdlYnNvY2tldCBjb25uZWN0aW9uIGNsb3Nlcy5cbiAgICAgICAgICogQGNhbGxiYWNrIGNsb3NlSGFuZGxlclxuICAgICAgICAgKi9cbiAgICAgICAgdGhpcy5jbG9zZUhhbmRsZXIgPSBudWxsO1xuXG4gICAgICAgIC8qKlxuICAgICAgICAgKiBBIGNhbGxiYWNrIGZvciBoYW5kbGluZyBtZXNzYWdlc1xuICAgICAgICAgKiB3aGVudCB0aGUgc29ja2V0IGVycm9yc1xuICAgICAgICAgKiBAY2FsbGJhY2sgZXJyb3JIYW5kbGVyXG4gICAgICAgICAqL1xuICAgICAgICB0aGlzLmVycm9ySGFuZGxlciA9IG51bGw7XG5cbiAgICAgICAgLy8gQmluZCBJbnN0YW5jZSBNZXRob2RzXG4gICAgICAgIHRoaXMuY29ubmVjdCA9IHRoaXMuY29ubmVjdC5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLnNlbmRTdHJpbmcgPSB0aGlzLnNlbmRTdHJpbmcuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5oYW5kbGVSYXdNZXNzYWdlID0gdGhpcy5oYW5kbGVSYXdNZXNzYWdlLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMub25Qb3N0c2NyaXB0cyA9IHRoaXMub25Qb3N0c2NyaXB0cy5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLm9uTWVzc2FnZSA9IHRoaXMub25NZXNzYWdlLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMub25DbG9zZSA9IHRoaXMub25DbG9zZS5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLm9uRXJyb3IgPSB0aGlzLm9uRXJyb3IuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBSZXR1cm5zIGEgcHJvcGVybHkgZm9ybWF0dGVkIFVSSVxuICAgICAqIGZvciB0aGUgc29ja2V0IGZvciBhbnkgZ2l2ZW4gY3VycmVudFxuICAgICAqIGJyb3dzZXIgbG9jYXRpb24uXG4gICAgICogQHJldHVybnMge3N0cmluZ30gQSBVUkkgc3RyaW5nLlxuICAgICAqL1xuICAgIGdldFVyaSgpe1xuICAgICAgICBsZXQgbG9jYXRpb24gPSB3aW5kb3cubG9jYXRpb247XG4gICAgICAgIGxldCB1cmkgPSBcIlwiO1xuICAgICAgICBpZihsb2NhdGlvbi5wcm90b2NvbCA9PT0gXCJodHRwczpcIil7XG4gICAgICAgICAgICB1cmkgKz0gXCJ3c3M6XCI7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICB1cmkgKz0gXCJ3czpcIjtcbiAgICAgICAgfVxuICAgICAgICB1cmkgPSBgJHt1cml9Ly8ke2xvY2F0aW9uLmhvc3R9YDtcbiAgICAgICAgdXJpID0gYCR7dXJpfS9zb2NrZXQke2xvY2F0aW9uLnBhdGhuYW1lfSR7bG9jYXRpb24uc2VhcmNofWA7XG4gICAgICAgIHJldHVybiB1cmk7XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogVGVsbHMgdGhpcyBvYmplY3QncyBpbnRlcm5hbCB3ZWJzb2NrZXRcbiAgICAgKiB0byBpbnN0YW50aWF0ZSBpdHNlbGYgYW5kIGNvbm5lY3QgdG9cbiAgICAgKiB0aGUgcHJvdmlkZWQgVVJJLiBUaGUgVVJJIHdpbGwgYmUgc2V0IHRvXG4gICAgICogdGhpcyBpbnN0YW5jZSdzIGB1cmlgIHByb3BlcnR5IGZpcnN0LiBJZiBub1xuICAgICAqIHVyaSBpcyBwYXNzZWQsIGBjb25uZWN0KClgIHdpbGwgdXNlIHRoZSBjdXJyZW50XG4gICAgICogYXR0cmlidXRlJ3MgdmFsdWUuXG4gICAgICogQHBhcmFtIHtzdHJpbmd9IHVyaSAtIEEgIFVSSSB0byBjb25uZWN0IHRoZSBzb2NrZXRcbiAgICAgKiB0by5cbiAgICAgKi9cbiAgICBjb25uZWN0KHVyaSl7XG4gICAgICAgIGlmKHVyaSl7XG4gICAgICAgICAgICB0aGlzLnVyaSA9IHVyaTtcbiAgICAgICAgfVxuICAgICAgICBpZih3aW5kb3cuV2ViU29ja2V0KXtcbiAgICAgICAgICAgIHRoaXMuc29ja2V0ID0gbmV3IFdlYlNvY2tldCh0aGlzLnVyaSk7XG4gICAgICAgIH0gZWxzZSBpZih3aW5kb3cuTW96V2ViU29ja2V0KXtcbiAgICAgICAgICAgIHRoaXMuc29ja2V0ID0gTW96V2ViU29ja2V0KHRoaXMudXJpKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHRocm93IG5ldyBXZWJzb2NrZXROb3RTdXBwb3J0ZWQoKTtcbiAgICAgICAgfVxuXG4gICAgICAgIHRoaXMuc29ja2V0Lm9uY2xvc2UgPSB0aGlzLmNsb3NlSGFuZGxlcjtcbiAgICAgICAgdGhpcy5zb2NrZXQub25tZXNzYWdlID0gdGhpcy5oYW5kbGVSYXdNZXNzYWdlLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuc29ja2V0Lm9uZXJyb3IgPSB0aGlzLmVycm9ySGFuZGxlcjtcbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBDb252ZW5pZW5jZSBtZXRob2QgdGhhdCBzZW5kcyB0aGUgcGFzc2VkXG4gICAgICogc3RyaW5nIG9uIHRoaXMgaW5zdGFuY2UncyB1bmRlcmx5aW5nXG4gICAgICogd2Vic29rZXQgY29ubmVjdGlvbi5cbiAgICAgKiBAcGFyYW0ge3N0cmluZ30gYVN0cmluZyAtIEEgc3RyaW5nIHRvIHNlbmRcbiAgICAgKi9cbiAgICBzZW5kU3RyaW5nKGFTdHJpbmcpe1xuICAgICAgICBpZih0aGlzLnNvY2tldCl7XG4gICAgICAgICAgICB0aGlzLnNvY2tldC5zZW5kKGFTdHJpbmcpO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgLy8gSWRlYWxseSB3ZSBtb3ZlIHRoZSBkb20gb3BlcmF0aW9ucyBvZlxuICAgIC8vIHRoaXMgZnVuY3Rpb24gb3V0IGludG8gYW5vdGhlciBjbGFzcyBvclxuICAgIC8vIGNvbnRleHQuXG4gICAgLyoqXG4gICAgICogVXNpbmcgdGhlIGludGVybmFsIGBjdXJyZW50QnVmZmVyYCwgdGhpc1xuICAgICAqIG1ldGhvZCBjaGVja3MgdG8gc2VlIGlmIGEgbGFyZ2UgbXVsdGktZnJhbWVcbiAgICAgKiBwaWVjZSBvZiB3ZWJzb2NrZXQgZGF0YSBpcyBiZWluZyBzZW50LiBJZiBzbyxcbiAgICAgKiBpdCBwcmVzZW50cyBhbmQgdXBkYXRlcyBhIHNwZWNpZmljIGRpc3BsYXkgaW5cbiAgICAgKiB0aGUgRE9NIHdpdGggdGhlIGN1cnJlbnQgcGVyY2VudGFnZSBldGMuXG4gICAgICogQHBhcmFtIHtzdHJpbmd9IG1zZyAtIFRoZSBtZXNzYWdlIHRvXG4gICAgICogZGlzcGxheSBpbnNpZGUgdGhlIGVsZW1lbnRcbiAgICAgKi9cbiAgICBzZXRMYXJnZURvd25sb2FkRGlzcGxheShtc2cpe1xuXG4gICAgICAgIGlmKG1zZy5sZW5ndGggPT0gMCAmJiAhdGhpcy5jdXJyZW50QnVmZmVyLmhhc0Rpc3BsYXkpe1xuICAgICAgICAgICAgcmV0dXJuO1xuICAgICAgICB9XG5cbiAgICAgICAgdGhpcy5jdXJyZW50QnVmZmVyLmhhc0Rpc3BsYXkgPSAobXNnLmxlbmd0aCAhPSAwKTtcblxuICAgICAgICBsZXQgZWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKFwib2JqZWN0X2RhdGFiYXNlX2xhcmdlX3BlbmRpbmdfZG93bmxvYWRfdGV4dFwiKTtcbiAgICAgICAgaWYoZWxlbWVudCAhPSB1bmRlZmluZWQpe1xuICAgICAgICAgICAgZWxlbWVudC5pbm5lckhUTUwgPSBtc2c7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBIYW5kbGVzIHRoZSBgb25tZXNzYWdlYCBldmVudCBvZiB0aGUgdW5kZXJseWluZ1xuICAgICAqIHdlYnNvY2tldC5cbiAgICAgKiBUaGlzIG1ldGhvZCBrbm93cyBob3cgdG8gZmlsbCB0aGUgaW50ZXJuYWxcbiAgICAgKiBidWZmZXIgKHRvIGdldCBhcm91bmQgdGhlIGZyYW1lIGxpbWl0KSBhbmQgb25seVxuICAgICAqIHRyaWdnZXIgc3Vic2VxdWVudCBoYW5kbGVycyBmb3IgaW5jb21pbmcgbWVzc2FnZXMuXG4gICAgICogVE9ETzogQnJlYWsgb3V0IHRoaXMgbWV0aG9kIGEgYml0IG1vcmUuIEl0IGhhcyBiZWVuXG4gICAgICogY29waWVkIG5lYXJseSB2ZXJiYXRpbSBmcm9tIHRoZSBvcmlnaW5hbCBjb2RlLlxuICAgICAqIE5PVEU6IEZvciBub3csIHRoZXJlIGFyZSBvbmx5IHR3byB0eXBlcyBvZiBtZXNzYWdlczpcbiAgICAgKiAgICAgICAndXBkYXRlcycgKHdlIGp1c3QgY2FsbCB0aGVzZSBtZXNzYWdlcylcbiAgICAgKiAgICAgICAncG9zdHNjcmlwdHMnICh0aGVzZSBhcmUganVzdCByYXcgbm9uLUpTT04gc3RyaW5ncylcbiAgICAgKiBJZiBhIGJ1ZmZlciBpcyBjb21wbGV0ZSwgdGhpcyBtZXRob2Qgd2lsbCBjaGVjayB0byBzZWUgaWZcbiAgICAgKiBoYW5kbGVycyBhcmUgcmVnaXN0ZXJlZCBmb3IgcG9zdHNjcmlwdC9ub3JtYWwgbWVzc2FnZXNcbiAgICAgKiBhbmQgd2lsbCB0cmlnZ2VyIHRoZW0gaWYgdHJ1ZSBpbiBlaXRoZXIgY2FzZSwgcGFzc2luZ1xuICAgICAqIGFueSBwYXJzZWQgSlNPTiBkYXRhIHRvIHRoZSBjYWxsYmFja3MuXG4gICAgICogQHBhcmFtIHtFdmVudH0gZXZlbnQgLSBUaGUgYG9ubWVzc2FnZWAgZXZlbnQgb2JqZWN0XG4gICAgICogZnJvbSB0aGUgc29ja2V0LlxuICAgICAqL1xuICAgIGhhbmRsZVJhd01lc3NhZ2UoZXZlbnQpe1xuICAgICAgICBpZih0aGlzLmN1cnJlbnRCdWZmZXIucmVtYWluaW5nID09PSBudWxsKXtcbiAgICAgICAgICAgIHRoaXMuY3VycmVudEJ1ZmZlci5yZW1haW5pbmcgPSBKU09OLnBhcnNlKGV2ZW50LmRhdGEpO1xuICAgICAgICAgICAgdGhpcy5jdXJyZW50QnVmZmVyLmJ1ZmZlciA9IFtdO1xuICAgICAgICAgICAgaWYodGhpcy5jdXJyZW50QnVmZmVyLmhhc0Rpc3BsYXkgJiYgdGhpcy5jdXJyZW50QnVmZmVyLnJlbWFpbmluZyA9PSAxKXtcbiAgICAgICAgICAgICAgICAvLyBTRVQgTEFSR0UgRE9XTkxPQUQgRElTUExBWVxuICAgICAgICAgICAgfVxuICAgICAgICAgICAgcmV0dXJuO1xuICAgICAgICB9XG5cbiAgICAgICAgdGhpcy5jdXJyZW50QnVmZmVyLnJlbWFpbmluZyAtPSAxO1xuICAgICAgICB0aGlzLmN1cnJlbnRCdWZmZXIuYnVmZmVyLnB1c2goZXZlbnQuZGF0YSk7XG5cbiAgICAgICAgaWYodGhpcy5jdXJyZW50QnVmZmVyLmJ1ZmZlci5sZW5ndGggJSBGUkFNRVNfUEVSX0FDSyA9PSAwKXtcbiAgICAgICAgICAgIC8vQUNLIGV2ZXJ5IHRlbnRoIG1lc3NhZ2UuIFdlIGhhdmUgdG8gZG8gYWN0aXZlIHB1c2hiYWNrXG4gICAgICAgICAgICAvL2JlY2F1c2UgdGhlIHdlYnNvY2tldCBkaXNjb25uZWN0cyBvbiBDaHJvbWUgaWYgeW91IGphbSB0b29cbiAgICAgICAgICAgIC8vbXVjaCBpbiBhdCBvbmNlXG4gICAgICAgICAgICB0aGlzLnNlbmRTdHJpbmcoXG4gICAgICAgICAgICAgICAgSlNPTi5zdHJpbmdpZnkoe1xuICAgICAgICAgICAgICAgICAgICBcIkFDS1wiOiB0aGlzLmN1cnJlbnRCdWZmZXIuYnVmZmVyLmxlbmd0aFxuICAgICAgICAgICAgICAgIH0pKTtcbiAgICAgICAgICAgIGxldCBwZXJjZW50YWdlID0gTWF0aC5yb3VuZCgxMDAqdGhpcy5jdXJyZW50QnVmZmVyLmJ1ZmZlci5sZW5ndGggLyAodGhpcy5jdXJyZW50QnVmZmVyLnJlbWFpbmluZyArIHRoaXMuY3VycmVudEJ1ZmZlci5idWZmZXIubGVuZ3RoKSk7XG4gICAgICAgICAgICBsZXQgdG90YWwgPSBNYXRoLnJvdW5kKCh0aGlzLmN1cnJlbnRCdWZmZXIucmVtYWluaW5nICsgdGhpcy5jdXJyZW50QnVmZmVyLmJ1ZmZlci5sZW5ndGgpIC8gKDEwMjQgLyAzMikpO1xuICAgICAgICAgICAgbGV0IHByb2dyZXNzU3RyID0gYChEb3dubG9hZGVkICR7cGVyY2VudGFnZX0lIG9mICR7dG90YWx9IE1CKWA7XG4gICAgICAgICAgICB0aGlzLnNldExhcmdlRG93bmxvYWREaXNwbGF5KHByb2dyZXNzU3RyKTtcbiAgICAgICAgfVxuXG4gICAgICAgIGlmKHRoaXMuY3VycmVudEJ1ZmZlci5yZW1haW5pbmcgPiAwKXtcbiAgICAgICAgICAgIHJldHVybjtcbiAgICAgICAgfVxuXG4gICAgICAgIHRoaXMuc2V0TGFyZ2VEb3dubG9hZERpc3BsYXkoXCJcIik7XG5cbiAgICAgICAgbGV0IGpvaW5lZEJ1ZmZlciA9IHRoaXMuY3VycmVudEJ1ZmZlci5idWZmZXIuam9pbignJylcblxuICAgICAgICB0aGlzLmN1cnJlbnRCdWZmZXIucmVtYWluaW5nID0gbnVsbDtcbiAgICAgICAgdGhpcy5jdXJyZW50QnVmZmVyLmJ1ZmZlciA9IG51bGw7XG5cbiAgICAgICAgbGV0IHVwZGF0ZSA9IEpTT04ucGFyc2Uoam9pbmVkQnVmZmVyKTtcblxuICAgICAgICBpZih1cGRhdGUgPT0gJ3JlcXVlc3RfYWNrJykge1xuICAgICAgICAgICAgdGhpcy5zZW5kU3RyaW5nKEpTT04uc3RyaW5naWZ5KHsnQUNLJzogMH0pKVxuICAgICAgICB9IGVsc2UgaWYodXBkYXRlID09ICdwb3N0c2NyaXB0cycpe1xuICAgICAgICAgICAgLy8gdXBkYXRlUG9wb3ZlcnMoKTtcbiAgICAgICAgICAgIGlmKHRoaXMucG9zdHNjcmlwdHNIYW5kbGVyKXtcbiAgICAgICAgICAgICAgICB0aGlzLnBvc3RzY3JpcHRzSGFuZGxlcih1cGRhdGUpO1xuICAgICAgICAgICAgfVxuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgaWYodGhpcy5tZXNzYWdlSGFuZGxlcil7XG4gICAgICAgICAgICAgICAgdGhpcy5tZXNzYWdlSGFuZGxlcih1cGRhdGUpO1xuICAgICAgICAgICAgfVxuICAgICAgICB9XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogQ29udmVuaWVuY2UgbWV0aG9kIHRoYXQgYmluZHNcbiAgICAgKiB0aGUgcGFzc2VkIGNhbGxiYWNrIHRvIHRoaXMgaW5zdGFuY2Unc1xuICAgICAqIHBvc3RzY3JpcHRzSGFuZGxlciwgd2hpY2ggaXMgc29tZSBtZXRob2RcbiAgICAgKiB0aGF0IGhhbmRsZXMgbWVzc2FnZXMgZm9yIHBvc3RzY3JpcHRzLlxuICAgICAqIEBwYXJhbSB7cG9zdHNjcmlwdHNIYW5kbGVyfSBjYWxsYmFjayAtIEEgaGFuZGxlclxuICAgICAqIGNhbGxiYWNrIG1ldGhvZCB3aXRoIHRoZSBtZXNzYWdlIGFyZ3VtZW50LlxuICAgICAqL1xuICAgIG9uUG9zdHNjcmlwdHMoY2FsbGJhY2spe1xuICAgICAgICB0aGlzLnBvc3RzY3JpcHRzSGFuZGxlciA9IGNhbGxiYWNrO1xuICAgIH1cblxuICAgIC8qKlxuICAgICAqIENvbnZlbmllbmNlIG1ldGhvZCB0aGF0IGJpbmRzXG4gICAgICogdGhlIHBhc3NlZCBjYWxsYmFjayB0byB0aGlzIGluc3RhbmNlJ3NcbiAgICAgKiBwb3N0c2NyaXB0c0hhbmRsZXIsIHdoaWNoIGlzIHNvbWUgbWV0aG9kXG4gICAgICogdGhhdCBoYW5kbGVzIG1lc3NhZ2VzIGZvciBwb3N0c2NyaXB0cy5cbiAgICAgKiBAcGFyYW0ge21lc3NhZ2VIYW5kbGVyfSBjYWxsYmFjayAtIEEgaGFuZGxlclxuICAgICAqIGNhbGxiYWNrIG1ldGhvZCB3aXRoIHRoZSBtZXNzYWdlIGFyZ3VtZW50LlxuICAgICAqL1xuICAgIG9uTWVzc2FnZShjYWxsYmFjayl7XG4gICAgICAgIHRoaXMubWVzc2FnZUhhbmRsZXIgPSBjYWxsYmFjaztcbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBDb252ZW5pZW5jZSBtZXRob2QgdGhhdCBiaW5kcyB0aGVcbiAgICAgKiBwYXNzZWQgY2FsbGJhY2sgdG8gdGhlIHVuZGVybHlpbmdcbiAgICAgKiB3ZWJzb2NrZXQncyBgb25jbG9zZWAgaGFuZGxlci5cbiAgICAgKiBAcGFyYW0ge2Nsb3NlSGFuZGxlcn0gY2FsbGJhY2sgLSBBIGZ1bmN0aW9uXG4gICAgICogdGhhdCBoYW5kbGVzIGNsb3NlIGV2ZW50cyBvbiB0aGUgc29ja2V0LlxuICAgICAqL1xuICAgIG9uQ2xvc2UoY2FsbGJhY2spe1xuICAgICAgICB0aGlzLmNsb3NlSGFuZGxlciA9IGNhbGxiYWNrO1xuICAgIH1cblxuICAgIC8qKlxuICAgICAqIENvbnZlbmllbmNlIG1ldGhvZCB0aGF0IGJpbmRzIHRoZVxuICAgICAqIHBhc3NlZCBjYWxsYmFjayB0byB0aGUgdW5kZXJseWluZ1xuICAgICAqIHdlYnNvY2tldHMnIGBvbmVycm9yYCBoYW5kbGVyLlxuICAgICAqIEBwYXJhbSB7ZXJyb3JIYW5kbGVyfSBjYWxsYmFjayAtIEEgZnVuY3Rpb25cbiAgICAgKiB0aGF0IGhhbmRsZXMgZXJyb3JzIG9uIHRoZSB3ZWJzb2NrZXQuXG4gICAgICovXG4gICAgb25FcnJvcihjYWxsYmFjayl7XG4gICAgICAgIHRoaXMuZXJyb3JIYW5kbGVyID0gY2FsbGJhY2s7XG4gICAgfVxufVxuXG5cbmV4cG9ydCB7Q2VsbFNvY2tldCwgQ2VsbFNvY2tldCBhcyBkZWZhdWx0fVxuIiwiLyoqXG4gKiBXZSB1c2UgYSBzaW5nbGV0b24gcmVnaXN0cnkgb2JqZWN0XG4gKiB3aGVyZSB3ZSBtYWtlIGF2YWlsYWJsZSBhbGwgcG9zc2libGVcbiAqIENvbXBvbmVudHMuIFRoaXMgaXMgdXNlZnVsIGZvciBXZWJwYWNrLFxuICogd2hpY2ggb25seSBidW5kbGVzIGV4cGxpY2l0bHkgdXNlZFxuICogQ29tcG9uZW50cyBkdXJpbmcgYnVpbGQgdGltZS5cbiAqL1xuaW1wb3J0IHtBc3luY0Ryb3Bkb3duLCBBc3luY0Ryb3Bkb3duQ29udGVudH0gZnJvbSAnLi9jb21wb25lbnRzL0FzeW5jRHJvcGRvd24nO1xuaW1wb3J0IHtCYWRnZX0gZnJvbSAnLi9jb21wb25lbnRzL0JhZGdlJztcbmltcG9ydCB7QnV0dG9ufSBmcm9tICcuL2NvbXBvbmVudHMvQnV0dG9uJztcbmltcG9ydCB7QnV0dG9uR3JvdXB9IGZyb20gJy4vY29tcG9uZW50cy9CdXR0b25Hcm91cCc7XG5pbXBvcnQge0NhcmR9IGZyb20gJy4vY29tcG9uZW50cy9DYXJkJztcbmltcG9ydCB7Q2FyZFRpdGxlfSBmcm9tICcuL2NvbXBvbmVudHMvQ2FyZFRpdGxlJztcbmltcG9ydCB7Q2lyY2xlTG9hZGVyfSBmcm9tICcuL2NvbXBvbmVudHMvQ2lyY2xlTG9hZGVyJztcbmltcG9ydCB7Q2xpY2thYmxlfSBmcm9tICcuL2NvbXBvbmVudHMvQ2xpY2thYmxlJztcbmltcG9ydCB7Q29kZX0gZnJvbSAnLi9jb21wb25lbnRzL0NvZGUnO1xuaW1wb3J0IHtDb2RlRWRpdG9yfSBmcm9tICcuL2NvbXBvbmVudHMvQ29kZUVkaXRvcic7XG5pbXBvcnQge0NvbGxhcHNpYmxlUGFuZWx9IGZyb20gJy4vY29tcG9uZW50cy9Db2xsYXBzaWJsZVBhbmVsJztcbmltcG9ydCB7Q29sdW1uc30gZnJvbSAnLi9jb21wb25lbnRzL0NvbHVtbnMnO1xuaW1wb3J0IHtDb250YWluZXJ9IGZyb20gJy4vY29tcG9uZW50cy9Db250YWluZXInO1xuaW1wb3J0IHtDb250ZXh0dWFsRGlzcGxheX0gZnJvbSAnLi9jb21wb25lbnRzL0NvbnRleHR1YWxEaXNwbGF5JztcbmltcG9ydCB7RHJvcGRvd259IGZyb20gJy4vY29tcG9uZW50cy9Ecm9wZG93bic7XG5pbXBvcnQge0V4cGFuZHN9IGZyb20gJy4vY29tcG9uZW50cy9FeHBhbmRzJztcbmltcG9ydCB7SGVhZGVyQmFyfSBmcm9tICcuL2NvbXBvbmVudHMvSGVhZGVyQmFyJztcbmltcG9ydCB7TG9hZENvbnRlbnRzRnJvbVVybH0gZnJvbSAnLi9jb21wb25lbnRzL0xvYWRDb250ZW50c0Zyb21VcmwnO1xuaW1wb3J0IHtMYXJnZVBlbmRpbmdEb3dubG9hZERpc3BsYXl9IGZyb20gJy4vY29tcG9uZW50cy9MYXJnZVBlbmRpbmdEb3dubG9hZERpc3BsYXknO1xuaW1wb3J0IHtNYWlufSBmcm9tICcuL2NvbXBvbmVudHMvTWFpbic7XG5pbXBvcnQge01vZGFsfSBmcm9tICcuL2NvbXBvbmVudHMvTW9kYWwnO1xuaW1wb3J0IHtPY3RpY29ufSBmcm9tICcuL2NvbXBvbmVudHMvT2N0aWNvbic7XG5pbXBvcnQge1BhZGRpbmd9IGZyb20gJy4vY29tcG9uZW50cy9QYWRkaW5nJztcbmltcG9ydCB7UG9wb3Zlcn0gZnJvbSAnLi9jb21wb25lbnRzL1BvcG92ZXInO1xuaW1wb3J0IHtSb290Q2VsbH0gZnJvbSAnLi9jb21wb25lbnRzL1Jvb3RDZWxsJztcbmltcG9ydCB7U2VxdWVuY2V9IGZyb20gJy4vY29tcG9uZW50cy9TZXF1ZW5jZSc7XG5pbXBvcnQge1Njcm9sbGFibGV9IGZyb20gJy4vY29tcG9uZW50cy9TY3JvbGxhYmxlJztcbmltcG9ydCB7U2luZ2xlTGluZVRleHRCb3h9IGZyb20gJy4vY29tcG9uZW50cy9TaW5nbGVMaW5lVGV4dEJveCc7XG5pbXBvcnQge1NwYW59IGZyb20gJy4vY29tcG9uZW50cy9TcGFuJztcbmltcG9ydCB7U3Vic2NyaWJlZH0gZnJvbSAnLi9jb21wb25lbnRzL1N1YnNjcmliZWQnO1xuaW1wb3J0IHtTdWJzY3JpYmVkU2VxdWVuY2V9IGZyb20gJy4vY29tcG9uZW50cy9TdWJzY3JpYmVkU2VxdWVuY2UnO1xuaW1wb3J0IHtUYWJsZX0gZnJvbSAnLi9jb21wb25lbnRzL1RhYmxlJztcbmltcG9ydCB7VGFic30gZnJvbSAnLi9jb21wb25lbnRzL1RhYnMnO1xuaW1wb3J0IHtUZXh0fSBmcm9tICcuL2NvbXBvbmVudHMvVGV4dCc7XG5pbXBvcnQge1RyYWNlYmFja30gZnJvbSAnLi9jb21wb25lbnRzL1RyYWNlYmFjayc7XG5pbXBvcnQge19OYXZUYWJ9IGZyb20gJy4vY29tcG9uZW50cy9fTmF2VGFiJztcbmltcG9ydCB7R3JpZH0gZnJvbSAnLi9jb21wb25lbnRzL0dyaWQnO1xuaW1wb3J0IHtTaGVldH0gZnJvbSAnLi9jb21wb25lbnRzL1NoZWV0JztcbmltcG9ydCB7UGxvdH0gZnJvbSAnLi9jb21wb25lbnRzL1Bsb3QnO1xuaW1wb3J0IHtfUGxvdFVwZGF0ZXJ9IGZyb20gJy4vY29tcG9uZW50cy9fUGxvdFVwZGF0ZXInO1xuXG5jb25zdCBDb21wb25lbnRSZWdpc3RyeSA9IHtcbiAgICBBc3luY0Ryb3Bkb3duLFxuICAgIEFzeW5jRHJvcGRvd25Db250ZW50LFxuICAgIEJhZGdlLFxuICAgIEJ1dHRvbixcbiAgICBCdXR0b25Hcm91cCxcbiAgICBDYXJkLFxuICAgIENhcmRUaXRsZSxcbiAgICBDaXJjbGVMb2FkZXIsXG4gICAgQ2xpY2thYmxlLFxuICAgIENvZGUsXG4gICAgQ29kZUVkaXRvcixcbiAgICBDb2xsYXBzaWJsZVBhbmVsLFxuICAgIENvbHVtbnMsXG4gICAgQ29udGFpbmVyLFxuICAgIENvbnRleHR1YWxEaXNwbGF5LFxuICAgIERyb3Bkb3duLFxuICAgIEV4cGFuZHMsXG4gICAgSGVhZGVyQmFyLFxuICAgIExvYWRDb250ZW50c0Zyb21VcmwsXG4gICAgTGFyZ2VQZW5kaW5nRG93bmxvYWREaXNwbGF5LFxuICAgIE1haW4sXG4gICAgTW9kYWwsXG4gICAgT2N0aWNvbixcbiAgICBQYWRkaW5nLFxuICAgIFBvcG92ZXIsXG4gICAgUm9vdENlbGwsXG4gICAgU2VxdWVuY2UsXG4gICAgU2Nyb2xsYWJsZSxcbiAgICBTaW5nbGVMaW5lVGV4dEJveCxcbiAgICBTcGFuLFxuICAgIFN1YnNjcmliZWQsXG4gICAgU3Vic2NyaWJlZFNlcXVlbmNlLFxuICAgIFRhYmxlLFxuICAgIFRhYnMsXG4gICAgVGV4dCxcbiAgICBUcmFjZWJhY2ssXG4gICAgX05hdlRhYixcbiAgICBHcmlkLFxuICAgIFNoZWV0LFxuICAgIFBsb3QsXG4gICAgX1Bsb3RVcGRhdGVyXG59O1xuXG5leHBvcnQge0NvbXBvbmVudFJlZ2lzdHJ5LCBDb21wb25lbnRSZWdpc3RyeSBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogQXN5bmNEcm9wZG93biBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyBhIHNpbmdsZSByZWd1bGFyXG4gKiByZXBsYWNlbWVudDpcbiAqICogYGNvbnRlbnRzYFxuICpcbiAqIE5PVEU6IFRoZSBDZWxscyB2ZXJzaW9uIG9mIHRoaXMgY2hpbGQgaXNcbiAqIGVpdGhlciBhIGxvYWRpbmcgaW5kaWNhdG9yLCB0ZXh0LCBvciBhXG4gKiBBc3luY0Ryb3Bkb3duQ29udGVudCBjZWxsLlxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgY29udGVudGAgKHNpbmdsZSkgLSBVc3VhbGx5IGFuIEFzeW5jRHJvcGRvd25Db250ZW50IGNlbGxcbiAqIGBsb2FkaW5nSW5kaWNhdG9yYCAoc2luZ2xlKSAtIEEgQ2VsbCB0aGF0IGRpc3BsYXlzIHRoYXQgdGhlIGNvbnRlbnQgaXMgbG9hZGluZ1xuICovXG5jbGFzcyBBc3luY0Ryb3Bkb3duIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICAvLyBCaW5kIGNvbnRleHQgdG8gbWV0aG9kc1xuICAgICAgICB0aGlzLmFkZERyb3Bkb3duTGlzdGVuZXIgPSB0aGlzLmFkZERyb3Bkb3duTGlzdGVuZXIuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5tYWtlQ29udGVudCA9IHRoaXMubWFrZUNvbnRlbnQuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJBc3luY0Ryb3Bkb3duXCIsXG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbCBidG4tZ3JvdXBcIlxuICAgICAgICAgICAgfSwgW1xuICAgICAgICAgICAgICAgIGgoJ2EnLCB7Y2xhc3M6IFwiYnRuIGJ0bi14cyBidG4tb3V0bGluZS1zZWNvbmRhcnlcIn0sIFt0aGlzLnByb3BzLmV4dHJhRGF0YS5sYWJlbFRleHRdKSxcbiAgICAgICAgICAgICAgICBoKCdidXR0b24nLCB7XG4gICAgICAgICAgICAgICAgICAgIGNsYXNzOiBcImJ0biBidG4teHMgYnRuLW91dGxpbmUtc2Vjb25kYXJ5IGRyb3Bkb3duLXRvZ2dsZSBkcm9wZG93bi10b2dnbGUtc3BsaXRcIixcbiAgICAgICAgICAgICAgICAgICAgdHlwZTogXCJidXR0b25cIixcbiAgICAgICAgICAgICAgICAgICAgaWQ6IGAke3RoaXMucHJvcHMuaWR9LWRyb3Bkb3duTWVudUJ1dHRvbmAsXG4gICAgICAgICAgICAgICAgICAgIFwiZGF0YS10b2dnbGVcIjogXCJkcm9wZG93blwiLFxuICAgICAgICAgICAgICAgICAgICBhZnRlckNyZWF0ZTogdGhpcy5hZGREcm9wZG93bkxpc3RlbmVyLFxuICAgICAgICAgICAgICAgICAgICBcImRhdGEtZmlyc3RjbGlja1wiOiBcInRydWVcIlxuICAgICAgICAgICAgICAgIH0pLFxuICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICAgICAgaWQ6IGAke3RoaXMucHJvcHMuaWR9LWRyb3Bkb3duQ29udGVudFdyYXBwZXJgLFxuICAgICAgICAgICAgICAgICAgICBjbGFzczogXCJkcm9wZG93bi1tZW51XCJcbiAgICAgICAgICAgICAgICB9LCBbdGhpcy5tYWtlQ29udGVudCgpXSlcbiAgICAgICAgICAgIF0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgYWRkRHJvcGRvd25MaXN0ZW5lcihlbGVtZW50KXtcbiAgICAgICAgbGV0IHBhcmVudEVsID0gZWxlbWVudC5wYXJlbnRFbGVtZW50O1xuICAgICAgICBsZXQgY29tcG9uZW50ID0gdGhpcztcbiAgICAgICAgbGV0IGZpcnN0VGltZUNsaWNrZWQgPSAoZWxlbWVudC5kYXRhc2V0LmZpcnN0Y2xpY2sgPT0gXCJ0cnVlXCIpO1xuICAgICAgICBpZihmaXJzdFRpbWVDbGlja2VkKXtcbiAgICAgICAgICAgICQocGFyZW50RWwpLm9uKCdzaG93LmJzLmRyb3Bkb3duJywgZnVuY3Rpb24oKXtcbiAgICAgICAgICAgICAgICBjZWxsU29ja2V0LnNlbmRTdHJpbmcoSlNPTi5zdHJpbmdpZnkoe1xuICAgICAgICAgICAgICAgICAgICBldmVudDonZHJvcGRvd24nLFxuICAgICAgICAgICAgICAgICAgICB0YXJnZXRfY2VsbDogY29tcG9uZW50LnByb3BzLmlkLFxuICAgICAgICAgICAgICAgICAgICBpc09wZW46IGZhbHNlXG4gICAgICAgICAgICAgICAgfSkpO1xuICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICAkKHBhcmVudEVsKS5vbignaGlkZS5icy5kcm9wZG93bicsIGZ1bmN0aW9uKCl7XG4gICAgICAgICAgICAgICAgY2VsbFNvY2tldC5zZW5kU3RyaW5nKEpTT04uc3RyaW5naWZ5KHtcbiAgICAgICAgICAgICAgICAgICAgZXZlbnQ6ICdkcm9wZG93bicsXG4gICAgICAgICAgICAgICAgICAgIHRhcmdldF9jZWxsOiBjb21wb25lbnQucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgICAgIGlzT3BlbjogdHJ1ZVxuICAgICAgICAgICAgICAgIH0pKTtcbiAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgZWxlbWVudC5kYXRhc2V0LmZpcnN0Y2xpY2sgPSBmYWxzZTtcbiAgICAgICAgfVxuICAgIH1cblxuICAgIG1ha2VDb250ZW50KCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ2NvbnRlbnRzJyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCdjb250ZW50Jyk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyBhIHNpbmdsZSByZWd1bGFyXG4gKiByZXBsYWNlbWVudDpcbiAqICogYGNvbnRlbnRzYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogYGNvbnRlbnRgIChzaW5nbGUpIC0gQSBDZWxsIHRoYXQgY29tcHJpc2VzIHRoZSBkcm9wZG93biBjb250ZW50XG4gKiBgbG9hZGluZ0luZGljYXRvcmAgKHNpbmdsZSkgLSBBIENlbGwgdGhhdCByZXByZXNlbnRzIGEgdmlzdWFsXG4gKiAgICAgICBpbmRpY2F0aW5nIHRoYXQgdGhlIGNvbnRlbnQgaXMgbG9hZGluZ1xuICovXG5jbGFzcyBBc3luY0Ryb3Bkb3duQ29udGVudCBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb21wb25lbnQgbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VDb250ZW50ID0gdGhpcy5tYWtlQ29udGVudC5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiBgZHJvcGRvd25Db250ZW50LSR7dGhpcy5wcm9wcy5pZH1gLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkFzeW5jRHJvcGRvd25Db250ZW50XCJcbiAgICAgICAgICAgIH0sIFt0aGlzLm1ha2VDb250ZW50KCldKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIG1ha2VDb250ZW50KCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ2NvbnRlbnRzJyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCdjb250ZW50Jyk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cblxuZXhwb3J0IHtcbiAgICBBc3luY0Ryb3Bkb3duLFxuICAgIEFzeW5jRHJvcGRvd25Db250ZW50LFxuICAgIEFzeW5jRHJvcGRvd24gYXMgZGVmYXVsdFxufTtcbiIsIi8qKlxuICogQmFkZ2UgQ2VsbCBDb21wb25lbnRcbiAqL1xuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBCYWRnZSBoYXMgYSBzaW5nbGUgcmVwbGFjZW1lbnQ6XG4gKiAqIGBjaGlsZGBcbiAqL1xuXG4vKipcbiAqIEFib3V0IE5hbWVkIENoaWxkcmVuXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogYGlubmVyYCAtIFRoZSBjb25jZW50IGNlbGwgb2YgdGhlIEJhZGdlXG4gKi9cbmNsYXNzIEJhZGdlIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29tcG9uZW50IG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlSW5uZXIgPSB0aGlzLm1ha2VJbm5lci5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4oXG4gICAgICAgICAgICBoKCdzcGFuJywge1xuICAgICAgICAgICAgICAgIGNsYXNzOiBgY2VsbCBiYWRnZSBiYWRnZS0ke3RoaXMucHJvcHMuZXh0cmFEYXRhLmJhZGdlU3R5bGV9YCxcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJCYWRnZVwiXG4gICAgICAgICAgICB9LCBbdGhpcy5tYWtlQ29udGVudCgpXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlSW5uZXIoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY2hpbGQnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ2lubmVyJyk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cbmV4cG9ydCB7QmFkZ2UsIEJhZGdlIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBCdXR0b24gQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyBvbmUgcmVndWxhciByZXBsYWNlbWVudDpcbiAqIGBjb250ZW50c2BcbiAqL1xuXG4vKipcbiAqIEFib3V0IE5hbWVkIENoaWxkcmVuXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIGBjb250ZW50YCAoc2luZ2xlKSAtIFRoZSBjZWxsIGluc2lkZSBvZiB0aGUgYnV0dG9uIChpZiBhbnkpXG4gKi9cbmNsYXNzIEJ1dHRvbiBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb250ZXh0IHRvIG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlQ29udGVudCA9IHRoaXMubWFrZUNvbnRlbnQuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5fZ2V0RXZlbnRzID0gdGhpcy5fZ2V0RXZlbnQuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5fZ2V0SFRNTENsYXNzZXMgPSB0aGlzLl9nZXRIVE1MQ2xhc3Nlcy5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4oXG4gICAgICAgICAgICBoKCdidXR0b24nLCB7XG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiQnV0dG9uXCIsXG4gICAgICAgICAgICAgICAgY2xhc3M6IHRoaXMuX2dldEhUTUxDbGFzc2VzKCksXG4gICAgICAgICAgICAgICAgb25jbGljazogdGhpcy5fZ2V0RXZlbnQoJ29uY2xpY2snKVxuICAgICAgICAgICAgfSwgW3RoaXMubWFrZUNvbnRlbnQoKV1cbiAgICAgICAgICAgICApXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgbWFrZUNvbnRlbnQoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY29udGVudHMnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ2NvbnRlbnQnKTtcbiAgICAgICAgfVxuICAgIH1cblxuICAgIF9nZXRFdmVudChldmVudE5hbWUpIHtcbiAgICAgICAgcmV0dXJuIHRoaXMucHJvcHMuZXh0cmFEYXRhLmV2ZW50c1tldmVudE5hbWVdO1xuICAgIH1cblxuICAgIF9nZXRIVE1MQ2xhc3Nlcygpe1xuICAgICAgICBsZXQgY2xhc3NTdHJpbmcgPSB0aGlzLnByb3BzLmV4dHJhRGF0YS5jbGFzc2VzLmpvaW4oXCIgXCIpO1xuICAgICAgICAvLyByZW1lbWJlciB0byB0cmltIHRoZSBjbGFzcyBzdHJpbmcgZHVlIHRvIGEgbWFxdWV0dGUgYnVnXG4gICAgICAgIHJldHVybiBjbGFzc1N0cmluZy50cmltKCk7XG4gICAgfVxufVxuXG5leHBvcnQge0J1dHRvbiwgQnV0dG9uIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBCdXR0b25Hcm91cCBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIGEgc2luZ2xlIGVudW1lcmF0ZWRcbiAqIHJlcGxhY2VtZW50OlxuICogKiBgYnV0dG9uYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgYnV0dG9uc2AgKGFycmF5KSAtIFRoZSBjb25zdGl0dWVudCBidXR0b24gY2VsbHNcbiAqL1xuY2xhc3MgQnV0dG9uR3JvdXAgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29tcG9uZW50IG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlQnV0dG9ucyA9IHRoaXMubWFrZUJ1dHRvbnMuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkJ1dHRvbkdyb3VwXCIsXG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiYnRuLWdyb3VwXCIsXG4gICAgICAgICAgICAgICAgXCJyb2xlXCI6IFwiZ3JvdXBcIlxuICAgICAgICAgICAgfSwgdGhpcy5tYWtlQnV0dG9ucygpXG4gICAgICAgICAgICAgKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIG1ha2VCdXR0b25zKCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRzRm9yKCdidXR0b24nKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkcmVuTmFtZWQoJ2J1dHRvbnMnKTtcbiAgICAgICAgfVxuICAgIH1cblxufVxuXG5leHBvcnQge0J1dHRvbkdyb3VwLCBCdXR0b25Hcm91cCBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogQ2FyZCBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge1Byb3BUeXBlc30gZnJvbSAnLi91dGlsL1Byb3BlcnR5VmFsaWRhdG9yJztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBUaGlzIGNvbXBvbmVudCBjb250YWlucyB0d29cbiAqIHJlZ3VsYXIgcmVwbGFjZW1lbnRzOlxuICogKiBgY29udGVudHNgXG4gKiAqIGBoZWFkZXJgXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBDaGlsZHJlblxuICogYGJvZHlgIChzaW5nbGUpIC0gVGhlIGNlbGwgdG8gcHV0IGluIHRoZSBib2R5IG9mIHRoZSBDYXJkXG4gKiBgaGVhZGVyYCAoc2luZ2xlKSAtIEFuIG9wdGlvbmFsIGhlYWRlciBjZWxsIHRvIHB1dCBhYm92ZVxuICogICAgICAgIGJvZHlcbiAqL1xuY2xhc3MgQ2FyZCBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb21wb25lbnQgbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VCb2R5ID0gdGhpcy5tYWtlQm9keS5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLm1ha2VIZWFkZXIgPSB0aGlzLm1ha2VIZWFkZXIuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgbGV0IGJvZHlDbGFzcyA9IFwiY2FyZC1ib2R5XCI7XG4gICAgICAgIGlmKHRoaXMucHJvcHMuZXh0cmFEYXRhLnBhZGRpbmcpe1xuICAgICAgICAgICAgYm9keUNsYXNzID0gYGNhcmQtYm9keSBwLSR7dGhpcy5wcm9wcy5leHRyYURhdGEucGFkZGluZ31gO1xuICAgICAgICB9XG4gICAgICAgIGxldCBib2R5QXJlYSA9IGgoJ2RpdicsIHtcbiAgICAgICAgICAgIGNsYXNzOiBib2R5Q2xhc3NcbiAgICAgICAgfSwgW3RoaXMubWFrZUJvZHkoKV0pO1xuICAgICAgICBsZXQgaGVhZGVyID0gdGhpcy5tYWtlSGVhZGVyKCk7XG4gICAgICAgIGxldCBoZWFkZXJBcmVhID0gbnVsbDtcbiAgICAgICAgaWYoaGVhZGVyKXtcbiAgICAgICAgICAgIGhlYWRlckFyZWEgPSBoKCdkaXYnLCB7Y2xhc3M6IFwiY2FyZC1oZWFkZXJcIn0sIFtoZWFkZXJdKTtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gaCgnZGl2JyxcbiAgICAgICAgICAgIHtcbiAgICAgICAgICAgICAgICBjbGFzczogXCJjZWxsIGNhcmRcIixcbiAgICAgICAgICAgICAgICBzdHlsZTogdGhpcy5wcm9wcy5kaXZTdHlsZSxcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJDYXJkXCJcbiAgICAgICAgICAgIH0sIFtoZWFkZXJBcmVhLCBib2R5QXJlYV0pO1xuICAgIH1cblxuICAgIG1ha2VCb2R5KCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ2NvbnRlbnRzJyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCdib2R5Jyk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBtYWtlSGVhZGVyKCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICBpZih0aGlzLnJlcGxhY2VtZW50cy5oYXNSZXBsYWNlbWVudCgnaGVhZGVyJykpe1xuICAgICAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignaGVhZGVyJyk7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICByZXR1cm4gbnVsbDtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ2hlYWRlcicpO1xuICAgICAgICB9XG4gICAgfVxufVxuXG5DYXJkLnByb3BUeXBlcyA9IHtcbiAgICBwYWRkaW5nOiB7XG4gICAgICAgIGRlc2NyaXB0aW9uOiBcIlBhZGRpbmcgd2VpZ2h0IGFzIGRlZmluZWQgYnkgQm9vc3RyYXAgY3NzIGNsYXNzZXMuXCIsXG4gICAgICAgIHR5cGU6IFByb3BUeXBlcy5vbmVPZihbUHJvcFR5cGVzLm51bWJlciwgUHJvcFR5cGVzLnN0cmluZ10pXG4gICAgfSxcbiAgICBkaXZTdHlsZToge1xuICAgICAgICBkZXNjcmlwdGlvbjogXCJIVE1MIHN0eWxlIGF0dHJpYnV0ZSBzdHJpbmcuXCIsXG4gICAgICAgIHR5cGU6IFByb3BUeXBlcy5vbmVPZihbUHJvcFR5cGVzLnN0cmluZ10pXG4gICAgfVxufTtcblxuZXhwb3J0IHtDYXJkLCBDYXJkIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBDYXJkVGl0bGUgQ2VsbFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBUaGlzIGNvbXBvbmVudCBoYXMgIHNpbmdsZSByZWd1bGFyXG4gKiByZXBsYWNlbWVudDpcbiAqICogYGNvbnRlbnRzYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgaW5uZXJgIChzaW5nbGUpIC0gVGhlIGlubmVyIGNlbGwgb2YgdGhlIHRpdGxlIGNvbXBvbmVudFxuICovXG5jbGFzcyBDYXJkVGl0bGUgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29tcG9uZW50IG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlSW5uZXIgPSB0aGlzLm1ha2VJbm5lci5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4oXG4gICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbFwiLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkNhcmRUaXRsZVwiXG4gICAgICAgICAgICB9LCBbXG4gICAgICAgICAgICAgICAgdGhpcy5tYWtlSW5uZXIoKVxuICAgICAgICAgICAgXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlSW5uZXIoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY29udGVudHMnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ2lubmVyJyk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cbmV4cG9ydCB7Q2FyZFRpdGxlLCBDYXJkVGl0bGUgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIENpcmNsZUxvYWRlciBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuXG5jbGFzcyBDaXJjbGVMb2FkZXIgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkNpcmNsZUxvYWRlclwiLFxuICAgICAgICAgICAgICAgIGNsYXNzOiBcInNwaW5uZXItZ3Jvd1wiLFxuICAgICAgICAgICAgICAgIHJvbGU6IFwic3RhdHVzXCJcbiAgICAgICAgICAgIH0pXG4gICAgICAgICk7XG4gICAgfVxufVxuXG5DaXJjbGVMb2FkZXIucHJvcFR5cGVzID0ge1xufTtcblxuZXhwb3J0IHtDaXJjbGVMb2FkZXIsIENpcmNsZUxvYWRlciBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogQ2xpY2thYmxlIENlbGwgQ29tcG9uZW50XG4gKi9cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIGEgc2luZ2xlXG4gKiByZWd1bGFyIHJlcGxhY2VtZW50OlxuICogKiBgY29udGVudHNgXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBDaGlsZHJlblxuICogLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIGBjb250ZW50YCAoc2luZ2xlKSAtIFRoZSBjZWxsIHRoYXQgY2FuIGdvIGluc2lkZSB0aGUgY2xpY2thYmxlXG4gKiAgICAgICAgY29tcG9uZW50XG4gKi9cbmNsYXNzIENsaWNrYWJsZSBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb250ZXh0IHRvIG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlQ29udGVudCA9IHRoaXMubWFrZUNvbnRlbnQuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5fZ2V0RXZlbnRzID0gdGhpcy5fZ2V0RXZlbnQuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkNsaWNrYWJsZVwiLFxuICAgICAgICAgICAgICAgIG9uY2xpY2s6IHRoaXMuX2dldEV2ZW50KCdvbmNsaWNrJyksXG4gICAgICAgICAgICAgICAgc3R5bGU6IHRoaXMucHJvcHMuZXh0cmFEYXRhLmRpdlN0eWxlXG4gICAgICAgICAgICB9LCBbXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge30sIFt0aGlzLm1ha2VDb250ZW50KCldKVxuICAgICAgICAgICAgXVxuICAgICAgICAgICAgKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIG1ha2VDb250ZW50KCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ2NvbnRlbnRzJyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCdjb250ZW50Jyk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBfZ2V0RXZlbnQoZXZlbnROYW1lKSB7XG4gICAgICAgIHJldHVybiB0aGlzLnByb3BzLmV4dHJhRGF0YS5ldmVudHNbZXZlbnROYW1lXTtcbiAgICB9XG59XG5cbmV4cG9ydCB7Q2xpY2thYmxlLCBDbGlja2FibGUgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIENvZGUgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyBhIHNpbmdsZVxuICogcmVndWxhciByZXBsYWNlbWVudDpcbiAqICogYGNoaWxkYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgY29kZWAgKHNpbmdsZSkgLSBDb2RlIHRoYXQgd2lsbCBiZSByZW5kZXJlZCBpbnNpZGVcbiAqL1xuY2xhc3MgQ29kZSBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb21wb25lbnQgbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VDb2RlID0gdGhpcy5tYWtlQ29kZS5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gaCgncHJlJyxcbiAgICAgICAgICAgICAgICAge1xuICAgICAgICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbCBjb2RlXCIsXG4gICAgICAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJDb2RlXCJcbiAgICAgICAgICAgICAgICAgfSwgW1xuICAgICAgICAgICAgICAgICAgICAgaChcImNvZGVcIiwge30sIFt0aGlzLm1ha2VDb2RlKCldKVxuICAgICAgICAgICAgICAgICBdXG4gICAgICAgICAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlQ29kZSgpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdjaGlsZCcpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnY29kZScpO1xuICAgICAgICB9XG4gICAgfVxufVxuXG5leHBvcnQge0NvZGUsIENvZGUgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIENvZGVFZGl0b3IgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbmNsYXNzIENvZGVFZGl0b3IgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuICAgICAgICB0aGlzLmVkaXRvciA9IG51bGw7XG4gICAgICAgIC8vIHVzZWQgdG8gc2NoZWR1bGUgcmVndWxhciBzZXJ2ZXIgdXBkYXRlc1xuICAgICAgICB0aGlzLlNFUlZFUl9VUERBVEVfREVMQVlfTVMgPSAxO1xuICAgICAgICB0aGlzLmVkaXRvclN0eWxlID0gJ3dpZHRoOjEwMCU7aGVpZ2h0OjEwMCU7bWFyZ2luOmF1dG87Ym9yZGVyOjFweCBzb2xpZCBsaWdodGdyYXk7JztcblxuICAgICAgICB0aGlzLnNldHVwRWRpdG9yID0gdGhpcy5zZXR1cEVkaXRvci5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLnNldHVwS2V5YmluZGluZ3MgPSB0aGlzLnNldHVwS2V5YmluZGluZ3MuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5jaGFuZ2VIYW5kbGVyID0gdGhpcy5jaGFuZ2VIYW5kbGVyLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgY29tcG9uZW50RGlkTG9hZCgpIHtcblxuICAgICAgICB0aGlzLnNldHVwRWRpdG9yKCk7XG5cbiAgICAgICAgaWYgKHRoaXMuZWRpdG9yID09PSBudWxsKSB7XG4gICAgICAgICAgICBjb25zb2xlLmxvZyhcImVkaXRvciBjb21wb25lbnQgbG9hZGVkIGJ1dCBmYWlsZWQgdG8gc2V0dXAgZWRpdG9yXCIpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgY29uc29sZS5sb2coXCJzZXR0aW5nIHVwIGVkaXRvclwiKTtcbiAgICAgICAgICAgIHRoaXMuZWRpdG9yLmxhc3RfZWRpdF9taWxsaXMgPSBEYXRlLm5vdygpO1xuXG4gICAgICAgICAgICB0aGlzLmVkaXRvci5zZXRUaGVtZShcImFjZS90aGVtZS90ZXh0bWF0ZVwiKTtcbiAgICAgICAgICAgIHRoaXMuZWRpdG9yLnNlc3Npb24uc2V0TW9kZShcImFjZS9tb2RlL3B5dGhvblwiKTtcbiAgICAgICAgICAgIHRoaXMuZWRpdG9yLnNldEF1dG9TY3JvbGxFZGl0b3JJbnRvVmlldyh0cnVlKTtcbiAgICAgICAgICAgIHRoaXMuZWRpdG9yLnNlc3Npb24uc2V0VXNlU29mdFRhYnModHJ1ZSk7XG4gICAgICAgICAgICB0aGlzLmVkaXRvci5zZXRWYWx1ZSh0aGlzLnByb3BzLmV4dHJhRGF0YS5pbml0aWFsVGV4dCk7XG5cbiAgICAgICAgICAgIGlmICh0aGlzLnByb3BzLmV4dHJhRGF0YS5hdXRvY29tcGxldGUpIHtcbiAgICAgICAgICAgICAgICB0aGlzLmVkaXRvci5zZXRPcHRpb25zKHtlbmFibGVCYXNpY0F1dG9jb21wbGV0aW9uOiB0cnVlfSk7XG4gICAgICAgICAgICAgICAgdGhpcy5lZGl0b3Iuc2V0T3B0aW9ucyh7ZW5hYmxlTGl2ZUF1dG9jb21wbGV0aW9uOiB0cnVlfSk7XG4gICAgICAgICAgICB9XG5cbiAgICAgICAgICAgIGlmICh0aGlzLnByb3BzLmV4dHJhRGF0YS5ub1Njcm9sbCkge1xuICAgICAgICAgICAgICAgIHRoaXMuZWRpdG9yLnNldE9wdGlvbihcIm1heExpbmVzXCIsIEluZmluaXR5KTtcbiAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgaWYgKHRoaXMucHJvcHMuZXh0cmFEYXRhLmZvbnRTaXplICE9PSB1bmRlZmluZWQpIHtcbiAgICAgICAgICAgICAgICB0aGlzLmVkaXRvci5zZXRPcHRpb24oXCJmb250U2l6ZVwiLCB0aGlzLnByb3BzLmV4dHJhRGF0YS5mb250U2l6ZSk7XG4gICAgICAgICAgICB9XG5cbiAgICAgICAgICAgIGlmICh0aGlzLnByb3BzLmV4dHJhRGF0YS5taW5MaW5lcyAhPT0gdW5kZWZpbmVkKSB7XG4gICAgICAgICAgICAgICAgdGhpcy5lZGl0b3Iuc2V0T3B0aW9uKFwibWluTGluZXNcIiwgdGhpcy5wcm9wcy5leHRyYURhdGEubWluTGluZXMpO1xuICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICB0aGlzLnNldHVwS2V5YmluZGluZ3MoKTtcblxuICAgICAgICAgICAgdGhpcy5jaGFuZ2VIYW5kbGVyKCk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIGgoJ2RpdicsXG4gICAgICAgICAgICB7XG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbFwiLFxuICAgICAgICAgICAgICAgIHN0eWxlOiB0aGlzLnByb3BzLmV4dHJhRGF0YS5kaXZTdHlsZSxcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJDb2RlRWRpdG9yXCJcbiAgICAgICAgICAgIH0sXG4gICAgICAgICAgICBbaCgnZGl2JywgeyBpZDogXCJlZGl0b3JcIiArIHRoaXMucHJvcHMuaWQsIHN0eWxlOiB0aGlzLmVkaXRvclN0eWxlIH0sIFtdKVxuICAgICAgICBdKTtcbiAgICB9XG5cbiAgICBzZXR1cEVkaXRvcigpe1xuICAgICAgICBsZXQgZWRpdG9ySWQgPSBcImVkaXRvclwiICsgdGhpcy5wcm9wcy5pZDtcbiAgICAgICAgLy8gVE9ETyBUaGVzZSBhcmUgZ2xvYmFsIHZhciBkZWZpbmVkIGluIHBhZ2UuaHRtbFxuICAgICAgICAvLyB3ZSBzaG91bGQgZG8gc29tZXRoaW5nIGFib3V0IHRoaXMuXG5cbiAgICAgICAgLy8gaGVyZSB3ZSBiaW5nIGFuZCBpbnNldCB0aGUgZWRpdG9yIGludG8gdGhlIGRpdiByZW5kZXJlZCBieVxuICAgICAgICAvLyB0aGlzLnJlbmRlcigpXG4gICAgICAgIHRoaXMuZWRpdG9yID0gYWNlLmVkaXQoZWRpdG9ySWQpO1xuICAgICAgICAvLyBUT0RPOiBkZWFsIHdpdGggdGhpcyBnbG9iYWwgZWRpdG9yIGxpc3RcbiAgICAgICAgYWNlRWRpdG9yc1tlZGl0b3JJZF0gPSB0aGlzLmVkaXRvcjtcbiAgICB9XG5cbiAgICBjaGFuZ2VIYW5kbGVyKCkge1xuXHR2YXIgZWRpdG9ySWQgPSB0aGlzLnByb3BzLmlkO1xuXHR2YXIgZWRpdG9yID0gdGhpcy5lZGl0b3I7XG5cdHZhciBTRVJWRVJfVVBEQVRFX0RFTEFZX01TID0gdGhpcy5TRVJWRVJfVVBEQVRFX0RFTEFZX01TO1xuICAgICAgICB0aGlzLmVkaXRvci5zZXNzaW9uLm9uKFxuICAgICAgICAgICAgXCJjaGFuZ2VcIixcbiAgICAgICAgICAgIGZ1bmN0aW9uKGRlbHRhKSB7XG4gICAgICAgICAgICAgICAgLy8gV1NcbiAgICAgICAgICAgICAgICBsZXQgcmVzcG9uc2VEYXRhID0ge1xuICAgICAgICAgICAgICAgICAgICBldmVudDogJ2VkaXRvcl9jaGFuZ2UnLFxuICAgICAgICAgICAgICAgICAgICAndGFyZ2V0X2NlbGwnOiBlZGl0b3JJZCxcbiAgICAgICAgICAgICAgICAgICAgZGF0YTogZGVsdGFcbiAgICAgICAgICAgICAgICB9O1xuICAgICAgICAgICAgICAgIGNlbGxTb2NrZXQuc2VuZFN0cmluZyhKU09OLnN0cmluZ2lmeShyZXNwb25zZURhdGEpKTtcbiAgICAgICAgICAgICAgICAvL3JlY29yZCB0aGF0IHdlIGp1c3QgZWRpdGVkXG4gICAgICAgICAgICAgICAgZWRpdG9yLmxhc3RfZWRpdF9taWxsaXMgPSBEYXRlLm5vdygpO1xuXG5cdFx0Ly9zY2hlZHVsZSBhIGZ1bmN0aW9uIHRvIHJ1biBpbiAnU0VSVkVSX1VQREFURV9ERUxBWV9NUydtc1xuXHRcdC8vdGhhdCB3aWxsIHVwZGF0ZSB0aGUgc2VydmVyLCBidXQgb25seSBpZiB0aGUgdXNlciBoYXMgc3RvcHBlZCB0eXBpbmcuXG5cdFx0Ly8gVE9ETyB1bmNsZWFyIGlmIHRoaXMgaXMgb3dya2luZyBwcm9wZXJseVxuXHRcdHdpbmRvdy5zZXRUaW1lb3V0KGZ1bmN0aW9uKCkge1xuXHRcdCAgICBpZiAoRGF0ZS5ub3coKSAtIGVkaXRvci5sYXN0X2VkaXRfbWlsbGlzID49IFNFUlZFUl9VUERBVEVfREVMQVlfTVMpIHtcblx0XHRcdC8vc2F2ZSBvdXIgY3VycmVudCBzdGF0ZSB0byB0aGUgcmVtb3RlIGJ1ZmZlclxuXHRcdFx0ZWRpdG9yLmN1cnJlbnRfaXRlcmF0aW9uICs9IDE7XG5cdFx0XHRlZGl0b3IubGFzdF9lZGl0X21pbGxpcyA9IERhdGUubm93KCk7XG5cdFx0XHRlZGl0b3IubGFzdF9lZGl0X3NlbnRfdGV4dCA9IGVkaXRvci5nZXRWYWx1ZSgpO1xuXHRcdFx0Ly8gV1Ncblx0XHRcdGxldCByZXNwb25zZURhdGEgPSB7XG5cdFx0XHQgICAgZXZlbnQ6ICdlZGl0aW5nJyxcblx0XHRcdCAgICAndGFyZ2V0X2NlbGwnOiBlZGl0b3JJZCxcblx0XHRcdCAgICBidWZmZXI6IGVkaXRvci5nZXRWYWx1ZSgpLFxuXHRcdFx0ICAgIHNlbGVjdGlvbjogZWRpdG9yLnNlbGVjdGlvbi5nZXRSYW5nZSgpLFxuXHRcdFx0ICAgIGl0ZXJhdGlvbjogZWRpdG9yLmN1cnJlbnRfaXRlcmF0aW9uXG5cdFx0XHR9O1xuXHRcdFx0Y2VsbFNvY2tldC5zZW5kU3RyaW5nKEpTT04uc3RyaW5naWZ5KHJlc3BvbnNlRGF0YSkpO1xuXHRcdCAgICB9XG5cdFx0fSwgU0VSVkVSX1VQREFURV9ERUxBWV9NUyArIDIpOyAvL25vdGUgdGhlIDJtcyBncmFjZSBwZXJpb2RcbiAgICAgICAgICAgIH1cbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBzZXR1cEtleWJpbmRpbmdzKCkge1xuICAgICAgICBjb25zb2xlLmxvZyhcInNldHRpbmcgdXAga2V5YmluZGluZ3NcIik7XG4gICAgICAgIHRoaXMucHJvcHMuZXh0cmFEYXRhLmtleWJpbmRpbmdzLm1hcCgoa2IpID0+IHtcbiAgICAgICAgICAgIHRoaXMuZWRpdG9yLmNvbW1hbmRzLmFkZENvbW1hbmQoXG4gICAgICAgICAgICAgICAge1xuICAgICAgICAgICAgICAgICAgICBuYW1lOiAnY21kJyArIGtiLFxuICAgICAgICAgICAgICAgICAgICBiaW5kS2V5OiB7d2luOiAnQ3RybC0nICsga2IsICBtYWM6ICdDb21tYW5kLScgKyBrYn0sXG4gICAgICAgICAgICAgICAgICAgIHJlYWRPbmx5OiB0cnVlLFxuICAgICAgICAgICAgICAgICAgICBleGVjOiAoKSA9PiB7XG4gICAgICAgICAgICAgICAgICAgICAgICB0aGlzLmVkaXRvci5jdXJyZW50X2l0ZXJhdGlvbiArPSAxO1xuICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy5lZGl0b3IubGFzdF9lZGl0X21pbGxpcyA9IERhdGUubm93KCk7XG4gICAgICAgICAgICAgICAgICAgICAgICB0aGlzLmVkaXRvci5sYXN0X2VkaXRfc2VudF90ZXh0ID0gdGhpcy5lZGl0b3IuZ2V0VmFsdWUoKTtcblxuICAgICAgICAgICAgICAgICAgICAgICAgLy8gV1NcbiAgICAgICAgICAgICAgICAgICAgICAgIGxldCByZXNwb25zZURhdGEgPSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgZXZlbnQ6ICdrZXliaW5kaW5nJyxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAndGFyZ2V0X2NlbGwnOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICdrZXknOiBrYixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAnYnVmZmVyJzogdGhpcy5lZGl0b3IuZ2V0VmFsdWUoKSxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAnc2VsZWN0aW9uJzogdGhpcy5lZGl0b3Iuc2VsZWN0aW9uLmdldFJhbmdlKCksXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgJ2l0ZXJhdGlvbic6IHRoaXMuZWRpdG9yLmN1cnJlbnRfaXRlcmF0aW9uXG4gICAgICAgICAgICAgICAgICAgICAgICB9O1xuICAgICAgICAgICAgICAgICAgICAgICAgY2VsbFNvY2tldC5zZW5kU3RyaW5nKEpTT04uc3RyaW5naWZ5KHJlc3BvbnNlRGF0YSkpO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgKTtcbiAgICAgICAgfSk7XG4gICAgfVxufVxuXG5leHBvcnQge0NvZGVFZGl0b3IsIENvZGVFZGl0b3IgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIENvbGxhcHNpYmxlUGFuZWwgQ2VsbCBDb21wb25lbnRcbiAqL1xuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50LmpzJztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBUaGlzIGNvbXBvbmVudCBoYXMgdHdvIHNpbmdsZSB0eXBlXG4gKiByZXBsYWNlbWVudHM6XG4gKiAqIGBjb250ZW50YFxuICogKiBgcGFuZWxgXG4gKiBOb3RlIHRoYXQgYHBhbmVsYCBpcyBvbmx5IHJlbmRlcmVkXG4gKiBpZiB0aGUgcGFuZWwgaXMgZXhwYW5kZWRcbiAqL1xuXG4vKipcbiAqIEFib3V0IE5hbWVkIENoaWxkcmVuXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogYGNvbnRlbnRgIChzaW5nbGUpIC0gVGhlIGN1cnJlbnQgY29udGVudCBDZWxsIG9mIHRoZSBwYW5lbFxuICogYHBhbmVsYCAoc2luZ2xlKSAtIFRoZSBjdXJyZW50IChleHBhbmRlZCkgcGFuZWwgdmlld1xuICovXG5jbGFzcyBDb2xsYXBzaWJsZVBhbmVsIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICAvLyBCaW5kIGNvbXBvbmVudCBtZXRob2RzXG4gICAgICAgIHRoaXMubWFrZVBhbmVsID0gdGhpcy5tYWtlUGFuZWwuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5tYWtlQ29udGVudCA9IHRoaXMubWFrZUNvbnRlbnQuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgaWYodGhpcy5wcm9wcy5leHRyYURhdGEuaXNFeHBhbmRlZCl7XG4gICAgICAgICAgICByZXR1cm4oXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgICAgICBjbGFzczogXCJjZWxsIGNvbnRhaW5lci1mbHVpZFwiLFxuICAgICAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiQ29sbGFwc2libGVQYW5lbFwiLFxuICAgICAgICAgICAgICAgICAgICBcImRhdGEtZXhwYW5kZWRcIjogdHJ1ZSxcbiAgICAgICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgICAgIHN0eWxlOiB0aGlzLnByb3BzLmV4dHJhRGF0YS5kaXZTdHlsZVxuICAgICAgICAgICAgICAgIH0sIFtcbiAgICAgICAgICAgICAgICAgICAgaCgnZGl2Jywge2NsYXNzOiBcInJvdyBmbGV4LW5vd3JhcCBuby1ndXR0ZXJzXCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgICAgICBoKCdkaXYnLCB7Y2xhc3M6IFwiY29sLW1kLWF1dG9cIn0sW1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRoaXMubWFrZVBhbmVsKClcbiAgICAgICAgICAgICAgICAgICAgICAgIF0pLFxuICAgICAgICAgICAgICAgICAgICAgICAgaCgnZGl2Jywge2NsYXNzOiBcImNvbC1zbVwifSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRoaXMubWFrZUNvbnRlbnQoKVxuICAgICAgICAgICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgICAgICBdKVxuICAgICAgICAgICAgKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgICAgICBjbGFzczogXCJjZWxsIGNvbnRhaW5lci1mbHVpZFwiLFxuICAgICAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiQ29sbGFwc2libGVQYW5lbFwiLFxuICAgICAgICAgICAgICAgICAgICBcImRhdGEtZXhwYW5kZWRcIjogZmFsc2UsXG4gICAgICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgICAgICBzdHlsZTogdGhpcy5wcm9wcy5leHRyYURhdGEuZGl2U3R5bGVcbiAgICAgICAgICAgICAgICB9LCBbdGhpcy5tYWtlQ29udGVudCgpXSlcbiAgICAgICAgICAgICk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBtYWtlQ29udGVudCgpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdjb250ZW50Jyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCdjb250ZW50Jyk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBtYWtlUGFuZWwoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcigncGFuZWwnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ3BhbmVsJyk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cblxuZXhwb3J0IHtDb2xsYXBzaWJsZVBhbmVsLCBDb2xsYXBzaWJsZVBhbmVsIGFzIGRlZmF1bHR9XG4iLCIvKipcbiAqIENvbHVtbnMgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyBvbmUgZW51bWVyYXRlZFxuICoga2luZCBvZiByZXBsYWNlbWVudDpcbiAqICogYGNgXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBDaGlsZHJlblxuICogLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIGBlbGVtZW50c2AgKGFycmF5KSAtIENlbGwgY29sdW1uIGVsZW1lbnRzXG4gKi9cbmNsYXNzIENvbHVtbnMgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29udGV4dCB0byBtZXRob2RzXG4gICAgICAgIHRoaXMubWFrZUlubmVyQ2hpbGRyZW4gPSB0aGlzLm1ha2VJbm5lckNoaWxkcmVuLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbCBjb250YWluZXItZmx1aWRcIixcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJDb2x1bW5zXCIsXG4gICAgICAgICAgICAgICAgc3R5bGU6IHRoaXMucHJvcHMuZXh0cmFEYXRhLmRpdlN0eWxlXG4gICAgICAgICAgICB9LCBbXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge2NsYXNzOiBcInJvdyBmbGV4LW5vd3JhcFwifSwgdGhpcy5tYWtlSW5uZXJDaGlsZHJlbigpKVxuICAgICAgICAgICAgXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlSW5uZXJDaGlsZHJlbigpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50c0ZvcignYycpLm1hcChyZXBsRWxlbWVudCA9PiB7XG4gICAgICAgICAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgICAgICAgICAgY2xhc3M6IFwiY29sLXNtXCJcbiAgICAgICAgICAgICAgICAgICAgfSwgW3JlcGxFbGVtZW50XSlcbiAgICAgICAgICAgICAgICApO1xuICAgICAgICAgICAgfSk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZHJlbk5hbWVkKCdlbGVtZW50cycpO1xuICAgICAgICB9XG4gICAgfVxufVxuXG5cbmV4cG9ydCB7Q29sdW1ucywgQ29sdW1ucyBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogR2VuZXJpYyBiYXNlIENlbGwgQ29tcG9uZW50LlxuICogU2hvdWxkIGJlIGV4dGVuZGVkIGJ5IG90aGVyXG4gKiBDZWxsIGNsYXNzZXMgb24gSlMgc2lkZS5cbiAqL1xuaW1wb3J0IHtSZXBsYWNlbWVudHNIYW5kbGVyfSBmcm9tICcuL3V0aWwvUmVwbGFjZW1lbnRzSGFuZGxlcic7XG5pbXBvcnQge1Byb3BUeXBlc30gZnJvbSAnLi91dGlsL1Byb3BlcnR5VmFsaWRhdG9yJztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG5jbGFzcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzID0ge30sIHJlcGxhY2VtZW50cyA9IFtdKXtcbiAgICAgICAgdGhpcy5pc0NvbXBvbmVudCA9IHRydWU7XG4gICAgICAgIHRoaXMuX3VwZGF0ZVByb3BzKHByb3BzKTtcblxuICAgICAgICAvLyBSZXBsYWNlbWVudHMgaGFuZGxpbmdcbiAgICAgICAgdGhpcy5yZXBsYWNlbWVudHMgPSBuZXcgUmVwbGFjZW1lbnRzSGFuZGxlcihyZXBsYWNlbWVudHMpO1xuICAgICAgICB0aGlzLnVzZXNSZXBsYWNlbWVudHMgPSAocmVwbGFjZW1lbnRzLmxlbmd0aCA+IDApO1xuXG4gICAgICAgIC8vIFNldHVwIHBhcmVudCByZWxhdGlvbnNoaXAsIGlmXG4gICAgICAgIC8vIGFueS4gSW4gdGhpcyBhYnN0cmFjdCBjbGFzc1xuICAgICAgICAvLyB0aGVyZSBpc24ndCBvbmUgYnkgZGVmYXVsdFxuICAgICAgICB0aGlzLnBhcmVudCA9IG51bGw7XG4gICAgICAgIHRoaXMuX3NldHVwQ2hpbGRSZWxhdGlvbnNoaXBzKCk7XG5cbiAgICAgICAgLy8gRW5zdXJlIHRoYXQgd2UgaGF2ZSBwYXNzZWQgaW4gYW4gaWRcbiAgICAgICAgLy8gd2l0aCB0aGUgcHJvcHMuIFNob3VsZCBlcnJvciBvdGhlcndpc2UuXG4gICAgICAgIGlmKCF0aGlzLnByb3BzLmlkIHx8IHRoaXMucHJvcHMuaWQgPT0gdW5kZWZpbmVkKXtcbiAgICAgICAgICAgIHRocm93IEVycm9yKCdZb3UgbXVzdCBkZWZpbmUgYW4gaWQgZm9yIGV2ZXJ5IGNvbXBvbmVudCBwcm9wcyEnKTtcbiAgICAgICAgfVxuXG4gICAgICAgIHRoaXMudmFsaWRhdGVQcm9wcygpO1xuXG4gICAgICAgIC8vIEJpbmQgY29udGV4dCB0byBtZXRob2RzXG4gICAgICAgIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yID0gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRzRm9yID0gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRzRm9yLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuY29tcG9uZW50RGlkTG9hZCA9IHRoaXMuY29tcG9uZW50RGlkTG9hZC5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLmNoaWxkcmVuRG8gPSB0aGlzLmNoaWxkcmVuRG8uYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5uYW1lZENoaWxkcmVuRG8gPSB0aGlzLm5hbWVkQ2hpbGRyZW5Eby5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLnJlbmRlckNoaWxkTmFtZWQgPSB0aGlzLnJlbmRlckNoaWxkTmFtZWQuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5yZW5kZXJDaGlsZHJlbk5hbWVkID0gdGhpcy5yZW5kZXJDaGlsZHJlbk5hbWVkLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuX3NldHVwQ2hpbGRSZWxhdGlvbnNoaXBzID0gdGhpcy5fc2V0dXBDaGlsZFJlbGF0aW9uc2hpcHMuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5fdXBkYXRlUHJvcHMgPSB0aGlzLl91cGRhdGVQcm9wcy5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLl9yZWN1cnNpdmVseU1hcE5hbWVkQ2hpbGRyZW4gPSB0aGlzLl9yZWN1cnNpdmVseU1hcE5hbWVkQ2hpbGRyZW4uYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgLy8gT2JqZWN0cyB0aGF0IGV4dGVuZCBmcm9tXG4gICAgICAgIC8vIG1lIHNob3VsZCBvdmVycmlkZSB0aGlzXG4gICAgICAgIC8vIG1ldGhvZCBpbiBvcmRlciB0byBnZW5lcmF0ZVxuICAgICAgICAvLyBzb21lIGNvbnRlbnQgZm9yIHRoZSB2ZG9tXG4gICAgICAgIHRocm93IG5ldyBFcnJvcignWW91IG11c3QgaW1wbGVtZW50IGEgYHJlbmRlcmAgbWV0aG9kIG9uIENvbXBvbmVudCBvYmplY3RzIScpO1xuICAgIH1cblxuICAgIC8qKlxuICAgICAqIE9iamVjdCB0aGF0IGV4dGVuZCBmcm9tIG1lIGNvdWxkIG92ZXJ3cml0ZSB0aGlzIG1ldGhvZC5cbiAgICAgKiBJdCBpcyB0byBiZSB1c2VkIGZvciBsaWZlY3lsY2UgbWFuYWdlbWVudCBhbmQgaXMgdG8gYmUgY2FsbGVkXG4gICAgICogYWZ0ZXIgdGhlIGNvbXBvbmVudHMgbG9hZHMuXG4gICAgKi9cbiAgICBjb21wb25lbnREaWRMb2FkKCkge1xuICAgICAgICByZXR1cm4gbnVsbDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogUmVzcG9uZHMgd2l0aCBhIGh5cGVyc2NyaXB0IG9iamVjdFxuICAgICAqIHRoYXQgcmVwcmVzZW50cyBhIGRpdiB0aGF0IGlzIGZvcm1hdHRlZFxuICAgICAqIGFscmVhZHkgZm9yIHRoZSByZWd1bGFyIHJlcGxhY2VtZW50LlxuICAgICAqIFRoaXMgb25seSB3b3JrcyBmb3IgcmVndWxhciB0eXBlIHJlcGxhY2VtZW50cy5cbiAgICAgKiBGb3IgZW51bWVyYXRlZCByZXBsYWNlbWVudHMsIHVzZVxuICAgICAqICNnZXRSZXBsYWNlbWVudEVsZW1lbnRzRm9yKClcbiAgICAgKi9cbiAgICBnZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IocmVwbGFjZW1lbnROYW1lKXtcbiAgICAgICAgbGV0IHJlcGxhY2VtZW50ID0gdGhpcy5yZXBsYWNlbWVudHMuZ2V0UmVwbGFjZW1lbnRGb3IocmVwbGFjZW1lbnROYW1lKTtcbiAgICAgICAgaWYocmVwbGFjZW1lbnQpe1xuICAgICAgICAgICAgbGV0IG5ld0lkID0gYCR7dGhpcy5wcm9wcy5pZH1fJHtyZXBsYWNlbWVudH1gO1xuICAgICAgICAgICAgcmV0dXJuIGgoJ2RpdicsIHtpZDogbmV3SWQsIGtleTogbmV3SWR9LCBbXSk7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIG51bGw7XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogUmVzcG9uZCB3aXRoIGFuIGFycmF5IG9mIGh5cGVyc2NyaXB0XG4gICAgICogb2JqZWN0cyB0aGF0IGFyZSBkaXZzIHdpdGggaWRzIHRoYXQgbWF0Y2hcbiAgICAgKiByZXBsYWNlbWVudCBzdHJpbmcgaWRzIGZvciB0aGUga2luZCBvZlxuICAgICAqIHJlcGxhY2VtZW50IGxpc3QgdGhhdCBpcyBlbnVtZXJhdGVkLFxuICAgICAqIGllIGBfX19fYnV0dG9uXzFgLCBgX19fX2J1dHRvbl8yX19gIGV0Yy5cbiAgICAgKi9cbiAgICBnZXRSZXBsYWNlbWVudEVsZW1lbnRzRm9yKHJlcGxhY2VtZW50TmFtZSl7XG4gICAgICAgIGlmKCF0aGlzLnJlcGxhY2VtZW50cy5oYXNSZXBsYWNlbWVudChyZXBsYWNlbWVudE5hbWUpKXtcbiAgICAgICAgICAgIHJldHVybiBbXTtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gdGhpcy5yZXBsYWNlbWVudHMubWFwUmVwbGFjZW1lbnRzRm9yKHJlcGxhY2VtZW50TmFtZSwgcmVwbGFjZW1lbnQgPT4ge1xuICAgICAgICAgICAgbGV0IG5ld0lkID0gYCR7dGhpcy5wcm9wcy5pZH1fJHtyZXBsYWNlbWVudH1gO1xuICAgICAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgICAgICBoKCdkaXYnLCB7aWQ6IG5ld0lkLCBrZXk6IG5ld0lkfSlcbiAgICAgICAgICAgICk7XG4gICAgICAgIH0pO1xuICAgIH1cblxuICAgIC8qKlxuICAgICAqIElmIHRoZXJlIGlzIGEgYHByb3BUeXBlc2Agb2JqZWN0IHByZXNlbnQgb25cbiAgICAgKiB0aGUgY29uc3RydWN0b3IgKGllIHRoZSBjb21wb25lbnQgY2xhc3MpLFxuICAgICAqIHRoZW4gcnVuIHRoZSBQcm9wVHlwZXMgdmFsaWRhdG9yIG9uIGl0LlxuICAgICAqL1xuICAgIHZhbGlkYXRlUHJvcHMoKXtcbiAgICAgICAgaWYodGhpcy5jb25zdHJ1Y3Rvci5wcm9wVHlwZXMpe1xuICAgICAgICAgICAgUHJvcFR5cGVzLnZhbGlkYXRlKFxuICAgICAgICAgICAgICAgIHRoaXMuY29uc3RydWN0b3IubmFtZSxcbiAgICAgICAgICAgICAgICB0aGlzLnByb3BzLFxuICAgICAgICAgICAgICAgIHRoaXMuY29uc3RydWN0b3IucHJvcFR5cGVzXG4gICAgICAgICAgICApO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogTG9va3MgdXAgdGhlIHBhc3NlZCBrZXkgaW4gbmFtZWRDaGlsZHJlbiBhbmRcbiAgICAgKiBpZiBmb3VuZCByZXNwb25kcyB3aXRoIHRoZSByZXN1bHQgb2YgY2FsbGluZ1xuICAgICAqIHJlbmRlciBvbiB0aGF0IGNoaWxkIGNvbXBvbmVudC4gUmV0dXJucyBudWxsXG4gICAgICogb3RoZXJ3aXNlLlxuICAgICAqL1xuICAgIHJlbmRlckNoaWxkTmFtZWQoa2V5KXtcbiAgICAgICAgbGV0IGZvdW5kQ2hpbGQgPSB0aGlzLnByb3BzLm5hbWVkQ2hpbGRyZW5ba2V5XTtcbiAgICAgICAgaWYoZm91bmRDaGlsZCl7XG4gICAgICAgICAgICByZXR1cm4gZm91bmRDaGlsZC5yZW5kZXIoKTtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gbnVsbDtcbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBMb29rcyB1cCB0aGUgcGFzc2VkIGtleSBpbiBuYW1lZENoaWxkcmVuXG4gICAgICogYW5kIGlmIGZvdW5kIC0tIGFuZCB0aGUgdmFsdWUgaXMgYW4gQXJyYXlcbiAgICAgKiBvciBBcnJheSBvZiBBcnJheXMsIHJlc3BvbmRzIHdpdGggYW5cbiAgICAgKiBpc29tb3JwaGljIHN0cnVjdHVyZSB0aGF0IGhhcyB0aGUgcmVuZGVyZWRcbiAgICAgKiB2YWx1ZXMgb2YgZWFjaCBjb21wb25lbnQuXG4gICAgICovXG4gICAgcmVuZGVyQ2hpbGRyZW5OYW1lZChrZXkpe1xuICAgICAgICBsZXQgZm91bmRDaGlsZHJlbiA9IHRoaXMucHJvcHMubmFtZWRDaGlsZHJlbltrZXldO1xuICAgICAgICBpZihmb3VuZENoaWxkcmVuKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLl9yZWN1cnNpdmVseU1hcE5hbWVkQ2hpbGRyZW4oZm91bmRDaGlsZHJlbiwgY2hpbGQgPT4ge1xuICAgICAgICAgICAgICAgIHJldHVybiBjaGlsZC5yZW5kZXIoKTtcbiAgICAgICAgICAgIH0pO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiBbXTtcbiAgICB9XG5cblxuXG4gICAgLyoqXG4gICAgICogR2V0dGVyIHRoYXQgd2lsbCByZXNwb25kIHdpdGggdGhlXG4gICAgICogY29uc3RydWN0b3IncyAoYWthIHRoZSAnY2xhc3MnKSBuYW1lXG4gICAgICovXG4gICAgZ2V0IG5hbWUoKXtcbiAgICAgICAgcmV0dXJuIHRoaXMuY29uc3RydWN0b3IubmFtZTtcbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBHZXR0ZXIgdGhhdCB3aWxsIHJlc3BvbmQgd2l0aCBhblxuICAgICAqIGFycmF5IG9mIHJlbmRlcmVkIChpZSBjb25maWd1cmVkXG4gICAgICogaHlwZXJzY3JpcHQpIG9iamVjdHMgdGhhdCByZXByZXNlbnRcbiAgICAgKiBlYWNoIGNoaWxkLiBOb3RlIHRoYXQgd2Ugd2lsbCBjcmVhdGUga2V5c1xuICAgICAqIGZvciB0aGVzZSBiYXNlZCBvbiB0aGUgSUQgb2YgdGhpcyBwYXJlbnRcbiAgICAgKiBjb21wb25lbnQuXG4gICAgICovXG4gICAgZ2V0IHJlbmRlcmVkQ2hpbGRyZW4oKXtcbiAgICAgICAgaWYodGhpcy5wcm9wcy5jaGlsZHJlbi5sZW5ndGggPT0gMCl7XG4gICAgICAgICAgICByZXR1cm4gW107XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIHRoaXMucHJvcHMuY2hpbGRyZW4ubWFwKGNoaWxkQ29tcG9uZW50ID0+IHtcbiAgICAgICAgICAgIGxldCByZW5kZXJlZENoaWxkID0gY2hpbGRDb21wb25lbnQucmVuZGVyKCk7XG4gICAgICAgICAgICByZW5kZXJlZENoaWxkLnByb3BlcnRpZXMua2V5ID0gYCR7dGhpcy5wcm9wcy5pZH0tY2hpbGQtJHtjaGlsZENvbXBvbmVudC5wcm9wcy5pZH1gO1xuICAgICAgICAgICAgcmV0dXJuIHJlbmRlcmVkQ2hpbGQ7XG4gICAgICAgIH0pO1xuICAgIH1cblxuICAgIC8qKiBQdWJsaWMgVXRpbCBNZXRob2RzICoqL1xuXG4gICAgLyoqXG4gICAgICogQ2FsbHMgdGhlIHByb3ZpZGVkIGNhbGxiYWNrIG9uIGVhY2hcbiAgICAgKiBhcnJheSBjaGlsZCBmb3IgdGhpcyBjb21wb25lbnQsIHdpdGhcbiAgICAgKiB0aGUgY2hpbGQgYXMgdGhlIHNvbGUgYXJnIHRvIHRoZVxuICAgICAqIGNhbGxiYWNrXG4gICAgICovXG4gICAgY2hpbGRyZW5EbyhjYWxsYmFjayl7XG4gICAgICAgIHRoaXMucHJvcHMuY2hpbGRyZW4uZm9yRWFjaChjaGlsZCA9PiB7XG4gICAgICAgICAgICBjYWxsYmFjayhjaGlsZCk7XG4gICAgICAgIH0pO1xuICAgIH1cblxuICAgIC8qKlxuICAgICAqIENhbGxzIHRoZSBwcm92aWRlZCBjYWxsYmFjayBvblxuICAgICAqIGVhY2ggbmFtZWQgY2hpbGQgd2l0aCBrZXksIGNoaWxkXG4gICAgICogYXMgdGhlIHR3byBhcmdzIHRvIHRoZSBjYWxsYmFjay5cbiAgICAgKi9cbiAgICBuYW1lZENoaWxkcmVuRG8oY2FsbGJhY2spe1xuICAgICAgICBPYmplY3Qua2V5cyh0aGlzLnByb3BzLm5hbWVkQ2hpbGRyZW4pLmZvckVhY2goa2V5ID0+IHtcbiAgICAgICAgICAgIGxldCBjaGlsZCA9IHRoaXMucHJvcHMubmFtZWRDaGlsZHJlbltrZXldO1xuICAgICAgICAgICAgY2FsbGJhY2soa2V5LCBjaGlsZCk7XG4gICAgICAgIH0pO1xuICAgIH1cblxuICAgIC8qKiBQcml2YXRlIFV0aWwgTWV0aG9kcyAqKi9cblxuICAgIC8qKlxuICAgICAqIFNldHMgdGhlIHBhcmVudCBhdHRyaWJ1dGUgb2YgYWxsIGluY29taW5nXG4gICAgICogYXJyYXkgYW5kL29yIG5hbWVkIGNoaWxkcmVuIHRvIHRoaXNcbiAgICAgKiBpbnN0YW5jZS5cbiAgICAgKi9cbiAgICBfc2V0dXBDaGlsZFJlbGF0aW9uc2hpcHMoKXtcbiAgICAgICAgLy8gTmFtZWQgY2hpbGRyZW4gZmlyc3RcbiAgICAgICAgT2JqZWN0LmtleXModGhpcy5wcm9wcy5uYW1lZENoaWxkcmVuKS5mb3JFYWNoKGtleSA9PiB7XG4gICAgICAgICAgICBsZXQgY2hpbGQgPSB0aGlzLnByb3BzLm5hbWVkQ2hpbGRyZW5ba2V5XTtcbiAgICAgICAgICAgIGNoaWxkLnBhcmVudCA9IHRoaXM7XG4gICAgICAgIH0pO1xuXG4gICAgICAgIC8vIE5vdyBhcnJheSBjaGlsZHJlblxuICAgICAgICB0aGlzLnByb3BzLmNoaWxkcmVuLmZvckVhY2goY2hpbGQgPT4ge1xuICAgICAgICAgICAgY2hpbGQucGFyZW50ID0gdGhpcztcbiAgICAgICAgfSk7XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogVXBkYXRlcyB0aGlzIGNvbXBvbmVudHMgcHJvcHMgb2JqZWN0XG4gICAgICogYmFzZWQgb24gYW4gaW5jb21pbmcgb2JqZWN0XG4gICAgICovXG4gICAgX3VwZGF0ZVByb3BzKGluY29taW5nUHJvcHMpe1xuICAgICAgICB0aGlzLnByb3BzID0gaW5jb21pbmdQcm9wcztcbiAgICAgICAgdGhpcy5wcm9wcy5jaGlsZHJlbiA9IGluY29taW5nUHJvcHMuY2hpbGRyZW4gfHwgW107XG4gICAgICAgIHRoaXMucHJvcHMubmFtZWRDaGlsZHJlbiA9IGluY29taW5nUHJvcHMubmFtZWRDaGlsZHJlbiB8fCB7fTtcbiAgICAgICAgdGhpcy5fc2V0dXBDaGlsZFJlbGF0aW9uc2hpcHMoKTtcbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBSZWN1cnNpdmVseSBtYXBzIGEgb25lIG9yIG11bHRpZGltZW5zaW9uYWxcbiAgICAgKiBuYW1lZCBjaGlsZHJlbiB2YWx1ZSB3aXRoIHRoZSBnaXZlbiBtYXBwaW5nXG4gICAgICogZnVuY3Rpb24uXG4gICAgICovXG4gICAgX3JlY3Vyc2l2ZWx5TWFwTmFtZWRDaGlsZHJlbihjb2xsZWN0aW9uLCBjYWxsYmFjayl7XG4gICAgICAgIHJldHVybiBjb2xsZWN0aW9uLm1hcChpdGVtID0+IHtcbiAgICAgICAgICAgIGlmKEFycmF5LmlzQXJyYXkoaXRlbSkpe1xuICAgICAgICAgICAgICAgIHJldHVybiB0aGlzLl9yZWN1cnNpdmVseU1hcE5hbWVkQ2hpbGRyZW4oaXRlbSwgY2FsbGJhY2spO1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICByZXR1cm4gY2FsbGJhY2soaXRlbSk7XG4gICAgICAgICAgICB9XG4gICAgICAgIH0pO1xuICAgIH1cbn07XG5cbmV4cG9ydCB7Q29tcG9uZW50LCBDb21wb25lbnQgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIENvbnRhaW5lciBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIGEgc2luZ2xlIHJlZ3VsYXJcbiAqIHJlcGxhY2VtZW50OlxuICogKiBgY2hpbGRgXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBDaGlsZHJlblxuICogLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIGBjaGlsZGAgKHNpbmdsZSkgLSBUaGUgQ2VsbCB0aGF0IHRoaXMgY29tcG9uZW50IGNvbnRhaW5zXG4gKi9cbmNsYXNzIENvbnRhaW5lciBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb21wb25lbnQgbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VDaGlsZCA9IHRoaXMubWFrZUNoaWxkLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIGxldCBjaGlsZCA9IHRoaXMubWFrZUNoaWxkKCk7XG4gICAgICAgIGxldCBzdHlsZSA9IFwiXCI7XG4gICAgICAgIGlmKCFjaGlsZCl7XG4gICAgICAgICAgICBzdHlsZSA9IFwiZGlzcGxheTpub25lO1wiO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiQ29udGFpbmVyXCIsXG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbFwiLFxuICAgICAgICAgICAgICAgIHN0eWxlOiBzdHlsZVxuICAgICAgICAgICAgfSwgW2NoaWxkXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlQ2hpbGQoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY2hpbGQnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ2NoaWxkJyk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cbmV4cG9ydCB7Q29udGFpbmVyLCBDb250YWluZXIgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIENvbnRleHR1YWxEaXNwbGF5IENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBUaGlzIGNvbXBvbmVudCBoYXMgYSBzaW5nbGVcbiAqIHJlZ3VsYXIgcmVwbGFjZW1lbnQ6XG4gKiAqIGBjaGlsZGBcbiAqL1xuXG4vKipcbiAqIEFib3V0IE5hbWVkIENoaWxkcmVuXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogYGNoaWxkYCAoc2luZ2xlKSAtIEEgY2hpbGQgY2VsbCB0byBkaXNwbGF5IGluIGEgY29udGV4dFxuICovXG5jbGFzcyBDb250ZXh0dWFsRGlzcGxheSBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb21wb25lbnQgbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VDaGlsZCA9IHRoaXMubWFrZUNoaWxkLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybiBoKCdkaXYnLFxuICAgICAgICAgICAge1xuICAgICAgICAgICAgICAgIGNsYXNzOiBcImNlbGwgY29udGV4dHVhbERpc3BsYXlcIixcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJDb250ZXh0dWFsRGlzcGxheVwiXG4gICAgICAgICAgICB9LCBbdGhpcy5tYWtlQ2hpbGQoKV1cbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlQ2hpbGQoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY2hpbGQnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ2NoaWxkJyk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cbmV4cG9ydCB7Q29udGV4dHVhbERpc3BsYXksIENvbnRleHR1YWxEaXNwbGF5IGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBEcm9wZG93biBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIG9uZSByZWd1bGFyXG4gKiByZXBsYWNlbWVudDpcbiAqICogYHRpdGxlYFxuICogVGhpcyBjb21wb25lbnQgaGFzIG9uZVxuICogZW51bWVyYXRlZCByZXBsYWNlbWVudDpcbiAqICogYGNoaWxkYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgdGl0bGVgIChzaW5nbGUpIC0gQSBDZWxsIHRoYXQgd2lsbCBjb21wcmlzZSB0aGUgdGl0bGUgb2ZcbiAqICAgICAgdGhlIGRyb3Bkb3duXG4gKiBgZHJvcGRvd25JdGVtc2AgKGFycmF5KSAtIEFuIGFycmF5IG9mIGNlbGxzIHRoYXQgYXJlXG4gKiAgICAgIHRoZSBpdGVtcyBpbiB0aGUgZHJvcGRvd25cbiAqL1xuY2xhc3MgRHJvcGRvd24gZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29udGV4dCB0byBtZXRob2RzXG4gICAgICAgIHRoaXMubWFrZVRpdGxlID0gdGhpcy5tYWtlVGl0bGUuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5tYWtlSXRlbXMgPSB0aGlzLm1ha2VJdGVtcy5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkRyb3Bkb3duXCIsXG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiYnRuLWdyb3VwXCJcbiAgICAgICAgICAgIH0sIFtcbiAgICAgICAgICAgICAgICBoKCdhJywge2NsYXNzOiBcImJ0biBidG4teHMgYnRuLW91dGxpbmUtc2Vjb25kYXJ5XCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgIHRoaXMubWFrZVRpdGxlKClcbiAgICAgICAgICAgICAgICBdKSxcbiAgICAgICAgICAgICAgICBoKCdidXR0b24nLCB7XG4gICAgICAgICAgICAgICAgICAgIGNsYXNzOiBcImJ0biBidG4teHMgYnRuLW91dGxpbmUtc2Vjb25kYXJ5IGRyb3Bkb3duLXRvZ2dsZSBkcm9wZG93bi10b2dnbGUtc3BsaXRcIixcbiAgICAgICAgICAgICAgICAgICAgdHlwZTogXCJidXR0b25cIixcbiAgICAgICAgICAgICAgICAgICAgaWQ6IGAke3RoaXMucHJvcHMuZXh0cmFEYXRhLnRhcmdldElkZW50aXR5fS1kcm9wZG93bk1lbnVCdXR0b25gLFxuICAgICAgICAgICAgICAgICAgICBcImRhdGEtdG9nZ2xlXCI6IFwiZHJvcGRvd25cIlxuICAgICAgICAgICAgICAgIH0pLFxuICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtjbGFzczogXCJkcm9wZG93bi1tZW51XCJ9LCB0aGlzLm1ha2VJdGVtcygpKVxuICAgICAgICAgICAgXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlVGl0bGUoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcigndGl0bGUnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ3RpdGxlJyk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBtYWtlSXRlbXMoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIC8vIEZvciBzb21lIHJlYXNvbiwgZHVlIGFnYWluIHRvIHRoZSBDZWxsIGltcGxlbWVudGF0aW9uLFxuICAgICAgICAgICAgLy8gc29tZXRpbWVzIHRoZXJlIGFyZSBub3QgdGhlc2UgY2hpbGQgcmVwbGFjZW1lbnRzLlxuICAgICAgICAgICAgaWYoIXRoaXMucmVwbGFjZW1lbnRzLmhhc1JlcGxhY2VtZW50KCdjaGlsZCcpKXtcbiAgICAgICAgICAgICAgICByZXR1cm4gW107XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRzRm9yKCdjaGlsZCcpLm1hcCgoZWxlbWVudCwgaWR4KSA9PiB7XG4gICAgICAgICAgICAgICAgcmV0dXJuIG5ldyBEcm9wZG93bkl0ZW0oe1xuICAgICAgICAgICAgICAgICAgICBpZDogYCR7dGhpcy5wcm9wcy5pZH0taXRlbS0ke2lkeH1gLFxuICAgICAgICAgICAgICAgICAgICBpbmRleDogaWR4LFxuICAgICAgICAgICAgICAgICAgICBjaGlsZFN1YnN0aXR1dGU6IGVsZW1lbnQsXG4gICAgICAgICAgICAgICAgICAgIHRhcmdldElkZW50aXR5OiB0aGlzLnByb3BzLmV4dHJhRGF0YS50YXJnZXRJZGVudGl0eSxcbiAgICAgICAgICAgICAgICAgICAgZHJvcGRvd25JdGVtSW5mbzogdGhpcy5wcm9wcy5leHRyYURhdGEuZHJvcGRvd25JdGVtSW5mb1xuICAgICAgICAgICAgICAgIH0pLnJlbmRlcigpO1xuICAgICAgICAgICAgfSk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICBpZih0aGlzLnByb3BzLm5hbWVkQ2hpbGRyZW4uZHJvcGRvd25JdGVtcyl7XG4gICAgICAgICAgICAgICAgcmV0dXJuIHRoaXMucHJvcHMubmFtZWRDaGlsZHJlbi5kcm9wZG93bkl0ZW1zLm1hcCgoaXRlbUNvbXBvbmVudCwgaWR4KSA9PiB7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiBuZXcgRHJvcGRvd0l0ZW0oe1xuICAgICAgICAgICAgICAgICAgICAgICAgaWQ6IGAke3RoaXMucHJvcGQuaWR9LWl0ZW0tJHtpZHh9YCxcbiAgICAgICAgICAgICAgICAgICAgICAgIGluZGV4OiBpZHgsXG4gICAgICAgICAgICAgICAgICAgICAgICBjaGlsZFN1YnN0aXR1dGU6IGl0ZW1Db21wb25lbnQucmVuZGVyKCksXG4gICAgICAgICAgICAgICAgICAgICAgICB0YXJnZXRJZGVudGl0eTogdGhpcy5wcm9wcy5leHRyYURhdGEudGFyZ2V0SWRlbnRpdHksXG4gICAgICAgICAgICAgICAgICAgICAgICBkcm9wZG93bkl0ZW1JbmZvOiB0aGlzLnByb3BzLmV4dHJhRGF0YS5kcm9wZG93bkl0ZW1JbmZvXG4gICAgICAgICAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICByZXR1cm4gW107XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICB9XG59XG5cblxuLyoqXG4gKiBBIHByaXZhdGUgc3ViY29tcG9uZW50IGZvciBlYWNoXG4gKiBEcm9wZG93biBtZW51IGl0ZW0uIFdlIG5lZWQgdGhpc1xuICogYmVjYXVzZSBvZiBob3cgY2FsbGJhY2tzIGFyZSBoYW5kbGVkXG4gKiBhbmQgYmVjYXVzZSB0aGUgQ2VsbHMgdmVyc2lvbiBkb2Vzbid0XG4gKiBhbHJlYWR5IGltcGxlbWVudCB0aGlzIGtpbmQgYXMgYSBzZXBhcmF0ZVxuICogZW50aXR5LlxuICovXG5jbGFzcyBEcm9wZG93bkl0ZW0gZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29udGV4dCB0byBtZXRob2RzXG4gICAgICAgIHRoaXMuY2xpY2tIYW5kbGVyID0gdGhpcy5jbGlja0hhbmRsZXIuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ2EnLCB7XG4gICAgICAgICAgICAgICAgY2xhc3M6IFwic3ViY2VsbCBjZWxsLWRyb3Bkb3duLWl0ZW0gZHJvcGRvd24taXRlbVwiLFxuICAgICAgICAgICAgICAgIGtleTogdGhpcy5wcm9wcy5pbmRleCxcbiAgICAgICAgICAgICAgICBvbmNsaWNrOiB0aGlzLmNsaWNrSGFuZGxlclxuICAgICAgICAgICAgfSwgW3RoaXMucHJvcHMuY2hpbGRTdWJzdGl0dXRlXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBjbGlja0hhbmRsZXIoZXZlbnQpe1xuICAgICAgICAvLyBUaGlzIGlzIHN1cGVyIGhhY2t5IGJlY2F1c2Ugb2YgdGhlXG4gICAgICAgIC8vIGN1cnJlbnQgQ2VsbCBpbXBsZW1lbnRhdGlvbi5cbiAgICAgICAgLy8gVGhpcyB3aG9sZSBjb21wb25lbnQgc3RydWN0dXJlIHNob3VsZCBiZSBoZWF2aWx5IHJlZmFjdG9yZWRcbiAgICAgICAgLy8gb25jZSB0aGUgQ2VsbHMgc2lkZSBvZiB0aGluZ3Mgc3RhcnRzIHRvIGNoYW5nZS5cbiAgICAgICAgbGV0IHdoYXRUb0RvID0gdGhpcy5wcm9wcy5kcm9wZG93bkl0ZW1JbmZvW3RoaXMucHJvcHMuaW5kZXgudG9TdHJpbmcoKV07XG4gICAgICAgIGlmKHdoYXRUb0RvID09ICdjYWxsYmFjaycpe1xuICAgICAgICAgICAgbGV0IHJlc3BvbnNlRGF0YSA9IHtcbiAgICAgICAgICAgICAgICBldmVudDogXCJtZW51XCIsXG4gICAgICAgICAgICAgICAgaXg6IHRoaXMucHJvcHMuaW5kZXgsXG4gICAgICAgICAgICAgICAgdGFyZ2V0X2NlbGw6IHRoaXMucHJvcHMudGFyZ2V0SWRlbnRpdHlcbiAgICAgICAgICAgIH07XG4gICAgICAgICAgICBjZWxsU29ja2V0LnNlbmRTdHJpbmcoSlNPTi5zdHJpbmdpZnkocmVzcG9uc2VEYXRhKSk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICB3aW5kb3cubG9jYXRpb24uaHJlZiA9IHdoYXRUb0RvO1xuICAgICAgICB9XG4gICAgfVxufVxuXG5leHBvcnQge0Ryb3Bkb3duLCBEcm9wZG93biBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogRXhwYW5kcyBDZWxsIENvbXBvbmVudFxuICovXG5cbi8qKiBUT0RPL05PVEU6IEl0IGFwcGVhcnMgdGhhdCB0aGUgb3Blbi9jbG9zZWRcbiAgICBTdGF0ZSBmb3IgdGhpcyBjb21wb25lbnQgY291bGQgc2ltcGx5IGJlIHBhc3NlZFxuICAgIHdpdGggdGhlIENlbGwgZGF0YSwgYWxvbmcgd2l0aCB3aGF0IHRvIGRpc3BsYXlcbiAgICBpbiBlaXRoZXIgY2FzZS4gVGhpcyB3b3VsZCBiZSBob3cgaXQgaXMgbm9ybWFsbHlcbiAgICBkb25lIGluIGxhcmdlIHdlYiBhcHBsaWNhdGlvbnMuXG4gICAgQ29uc2lkZXIgcmVmYWN0b3JpbmcgYm90aCBoZXJlIGFuZCBvbiB0aGUgQ2VsbHNcbiAgICBzaWRlXG4qKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyB0d29cbiAqIHJlZ3VsYXIgcmVwbGFjZW1lbnRzOlxuICogKiBgaWNvbmBcbiAqICogYGNoaWxkYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgY29udGVudGAgKHNpbmdsZSkgLSBUaGUgb3BlbiBvciBjbG9zZWQgY2VsbCwgZGVwZW5kaW5nIG9uIHNvdXJjZVxuICogICAgIG9wZW4gc3RhdGVcbiAqIGBpY29uYCAoc2luZ2xlKSAtIFRoZSBDZWxsIG9mIHRoZSBpY29uIHRvIGRpc3BsYXksIGFsc28gZGVwZW5kaW5nXG4gKiAgICAgb24gY2xvc2VkIG9yIG9wZW4gc3RhdGVcbiAqL1xuY2xhc3MgRXhwYW5kcyBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb250ZXh0IHRvIG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlSWNvbiA9IHRoaXMubWFrZUljb24uYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5tYWtlQ29udGVudCA9IHRoaXMubWFrZUNvbnRlbnQuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5fZ2V0RXZlbnRzID0gdGhpcy5fZ2V0RXZlbnQuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkV4cGFuZHNcIixcbiAgICAgICAgICAgICAgICBzdHlsZTogdGhpcy5wcm9wcy5leHRyYURhdGEuZGl2U3R5bGVcbiAgICAgICAgICAgIH0sXG4gICAgICAgICAgICAgICAgW1xuICAgICAgICAgICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgICAgICAgICBzdHlsZTogJ2Rpc3BsYXk6aW5saW5lLWJsb2NrO3ZlcnRpY2FsLWFsaWduOnRvcCcsXG4gICAgICAgICAgICAgICAgICAgICAgICBvbmNsaWNrOiB0aGlzLl9nZXRFdmVudCgnb25jbGljaycpXG4gICAgICAgICAgICAgICAgICAgIH0sXG4gICAgICAgICAgICAgICAgICAgICAgW3RoaXMubWFrZUljb24oKV0pLFxuICAgICAgICAgICAgICAgICAgICBoKCdkaXYnLCB7c3R5bGU6J2Rpc3BsYXk6aW5saW5lLWJsb2NrJ30sXG4gICAgICAgICAgICAgICAgICAgICAgW3RoaXMubWFrZUNvbnRlbnQoKV0pLFxuICAgICAgICAgICAgICAgIF1cbiAgICAgICAgICAgIClcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlQ29udGVudCgpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdjaGlsZCcpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnY29udGVudCcpO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgbWFrZUljb24oKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignaWNvbicpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnaWNvbicpO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgX2dldEV2ZW50KGV2ZW50TmFtZSkge1xuICAgICAgICByZXR1cm4gdGhpcy5wcm9wcy5leHRyYURhdGEuZXZlbnRzW2V2ZW50TmFtZV07XG4gICAgfVxufVxuXG5leHBvcnQge0V4cGFuZHMsIEV4cGFuZHMgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIEdyaWQgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyAzIGVudW1lcmFibGVcbiAqIHJlcGxhY2VtZW50czpcbiAqICogYGhlYWRlcmBcbiAqICogYHJvd2xhYmVsYFxuICogKiBgY2hpbGRgXG4gKlxuICogTk9URTogQ2hpbGQgaXMgYSAyLWRpbWVuc2lvbmFsXG4gKiBlbnVtZXJhdGVkIHJlcGxhY2VtZW50IVxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgaGVhZGVyc2AgKGFycmF5KSAtIEFuIGFycmF5IG9mIHRhYmxlIGhlYWRlciBjZWxsc1xuICogYHJvd0xhYmVsc2AgKGFycmF5KSAtIEFuIGFycmF5IG9mIHJvdyBsYWJlbCBjZWxsc1xuICogYGRhdGFDZWxsc2AgKGFycmF5LW9mLWFycmF5KSAtIEEgMi1kaW1lbnNpb25hbCBhcnJheVxuICogICAgIG9mIGNlbGxzIHRoYXQgc2VydmUgYXMgdGFibGUgZGF0YSwgd2hlcmUgcm93c1xuICogICAgIGFyZSB0aGUgb3V0ZXIgYXJyYXkgYW5kIGNvbHVtbnMgYXJlIHRoZSBpbm5lclxuICogICAgIGFycmF5LlxuICovXG5jbGFzcyBHcmlkIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICAvLyBCaW5kIGNvbnRleHQgdG8gbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VIZWFkZXJzID0gdGhpcy5tYWtlSGVhZGVycy5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLm1ha2VSb3dzID0gdGhpcy5tYWtlUm93cy5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLl9tYWtlUmVwbGFjZW1lbnRIZWFkZXJFbGVtZW50cyA9IHRoaXMuX21ha2VSZXBsYWNlbWVudEhlYWRlckVsZW1lbnRzLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuX21ha2VSZXBsYWNlbWVudFJvd0VsZW1lbnRzID0gdGhpcy5fbWFrZVJlcGxhY2VtZW50Um93RWxlbWVudHMuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgbGV0IHRvcFRhYmxlSGVhZGVyID0gbnVsbDtcbiAgICAgICAgaWYodGhpcy5wcm9wcy5leHRyYURhdGEuaGFzVG9wSGVhZGVyKXtcbiAgICAgICAgICAgIHRvcFRhYmxlSGVhZGVyID0gaCgndGgnKTtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgndGFibGUnLCB7XG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiR3JpZFwiLFxuICAgICAgICAgICAgICAgIGNsYXNzOiBcImNlbGwgdGFibGUtaHNjcm9sbCB0YWJsZS1zbSB0YWJsZS1zdHJpcGVkXCJcbiAgICAgICAgICAgIH0sIFtcbiAgICAgICAgICAgICAgICBoKCd0aGVhZCcsIHt9LCBbXG4gICAgICAgICAgICAgICAgICAgIGgoJ3RyJywge30sIFt0b3BUYWJsZUhlYWRlciwgLi4udGhpcy5tYWtlSGVhZGVycygpXSlcbiAgICAgICAgICAgICAgICBdKSxcbiAgICAgICAgICAgICAgICBoKCd0Ym9keScsIHt9LCB0aGlzLm1ha2VSb3dzKCkpXG4gICAgICAgICAgICBdKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIG1ha2VIZWFkZXJzKCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5fbWFrZVJlcGxhY2VtZW50SGVhZGVyRWxlbWVudHMoKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkcmVuTmFtZWQoJ2hlYWRlcnMnKS5tYXAoKGhlYWRlckVsLCBjb2xJZHgpID0+IHtcbiAgICAgICAgICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgICAgICAgICBoKCd0aCcsIHtrZXk6IGAke3RoaXMucHJvcHMuaWR9LWdyaWQtdGgtJHtjb2xJZHh9YH0sIFtcbiAgICAgICAgICAgICAgICAgICAgICAgIGhlYWRlckVsXG4gICAgICAgICAgICAgICAgICAgIF0pXG4gICAgICAgICAgICAgICAgKTtcbiAgICAgICAgICAgIH0pO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgbWFrZVJvd3MoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLl9tYWtlUmVwbGFjZW1lbnRSb3dFbGVtZW50cygpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGRyZW5OYW1lZCgnZGF0YUNlbGxzJykubWFwKChkYXRhUm93LCByb3dJZHgpID0+IHtcbiAgICAgICAgICAgICAgICBsZXQgY29sdW1ucyA9IGRhdGFSb3cubWFwKChjb2x1bW4sIGNvbElkeCkgPT4ge1xuICAgICAgICAgICAgICAgICAgICByZXR1cm4oXG4gICAgICAgICAgICAgICAgICAgICAgICBoKCd0ZCcsIHtrZXk6IGAke3RoaXMucHJvcHMuaWR9LWdyaWQtY29sLSR7cm93SWR4fS0ke2NvbElkeH1gfSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNvbHVtblxuICAgICAgICAgICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgICAgICAgICAgKTtcbiAgICAgICAgICAgICAgICB9KTtcbiAgICAgICAgICAgICAgICBsZXQgcm93TGFiZWxFbCA9IG51bGw7XG4gICAgICAgICAgICAgICAgaWYodGhpcy5wcm9wcy5uYW1lZENoaWxkcmVuLnJvd0xhYmVscyAmJiB0aGlzLnByb3BzLm5hbWVkQ2hpbGRyZW4ucm93TGFiZWxzLmxlbmd0aCA+IDApe1xuICAgICAgICAgICAgICAgICAgICByb3dMYWJlbEVsID0gaCgndGgnLCB7a2V5OiBgJHt0aGlzLnByb3BzLmlkfS1ncmlkLWNvbC0ke3Jvd0lkeH0tJHtjb2xJZHh9YH0sIFtcbiAgICAgICAgICAgICAgICAgICAgICAgIHRoaXMucHJvcHMubmFtZWRDaGlsZHJlbi5yb3dMYWJlbHNbcm93SWR4XS5yZW5kZXIoKVxuICAgICAgICAgICAgICAgICAgICBdKTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgICAgICAgICAgaCgndHInLCB7a2V5OiBgJHt0aGlzLnByb3BzLmlkfS1ncmlkLXJvdy0ke3Jvd0lkeH1gfSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgcm93TGFiZWxFbCxcbiAgICAgICAgICAgICAgICAgICAgICAgIC4uLmNvbHVtbnNcbiAgICAgICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgICAgICApO1xuICAgICAgICAgICAgfSk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBfbWFrZVJlcGxhY2VtZW50Um93RWxlbWVudHMoKXtcbiAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50c0ZvcignY2hpbGQnKS5tYXAoKHJvdywgcm93SWR4KSA9PiB7XG4gICAgICAgICAgICBsZXQgY29sdW1ucyA9IHJvdy5tYXAoKGNvbHVtbiwgY29sSWR4KSA9PiB7XG4gICAgICAgICAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgICAgICAgICAgaCgndGQnLCB7a2V5OiBgJHt0aGlzLnByb3BzLmlkfS1ncmlkLWNvbC0ke3Jvd0lkeH0tJHtjb2xJZHh9YH0sIFtcbiAgICAgICAgICAgICAgICAgICAgICAgIGNvbHVtblxuICAgICAgICAgICAgICAgICAgICBdKVxuICAgICAgICAgICAgICAgICk7XG4gICAgICAgICAgICB9KTtcbiAgICAgICAgICAgIGxldCByb3dMYWJlbEVsID0gbnVsbDtcbiAgICAgICAgICAgIGlmKHRoaXMucmVwbGFjZW1lbnRzLmhhc1JlcGxhY2VtZW50KCdyb3dsYWJlbCcpKXtcbiAgICAgICAgICAgICAgICByb3dMYWJlbEVsID0gaCgndGgnLCB7a2V5OiBgJHt0aGlzLnByb3BzLmlkfS1ncmlkLXJvd2xibC0ke3Jvd0lkeH1gfSwgW1xuICAgICAgICAgICAgICAgICAgICB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudHNGb3IoJ3Jvd2xhYmVsJylbcm93SWR4XVxuICAgICAgICAgICAgICAgIF0pO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgICAgICBoKCd0cicsIHtrZXk6IGAke3RoaXMucHJvcHMuaWR9LWdyaWQtcm93LSR7cm93SWR4fWB9LCBbXG4gICAgICAgICAgICAgICAgICAgIHJvd0xhYmVsRWwsXG4gICAgICAgICAgICAgICAgICAgIC4uLmNvbHVtbnNcbiAgICAgICAgICAgICAgICBdKVxuICAgICAgICAgICAgKTtcbiAgICAgICAgfSk7XG4gICAgfVxuXG4gICAgX21ha2VSZXBsYWNlbWVudEhlYWRlckVsZW1lbnRzKCl7XG4gICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudHNGb3IoJ2hlYWRlcicpLm1hcCgoaGVhZGVyRWwsIGNvbElkeCkgPT4ge1xuICAgICAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgICAgICBoKCd0aCcsIHtrZXk6IGAke3RoaXMucHJvcHMuaWR9LWdyaWQtdGgtJHtjb2xJZHh9YH0sIFtcbiAgICAgICAgICAgICAgICAgICAgaGVhZGVyRWxcbiAgICAgICAgICAgICAgICBdKVxuICAgICAgICAgICAgKTtcbiAgICAgICAgfSk7XG4gICAgfVxufVxuXG5leHBvcnRcbntHcmlkLCBHcmlkIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBIZWFkZXJCYXIgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyB0aHJlZSBzZXBhcmF0ZVxuICogZW51bWVyYXRlZCByZXBsYWNlbWVudHM6XG4gKiAqIGBsZWZ0YFxuICogKiBgcmlnaHRgXG4gKiAqIGBjZW50ZXJgXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBDaGlsZHJlblxuICogLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIGBsZWZ0SXRlbXNgIChhcnJheSkgLSBUaGUgaXRlbXMgdGhhdCB3aWxsIGJlIG9uIHRoZSBsZWZ0XG4gKiBgY2VudGVySXRlbXNgIChhcnJheSkgLSBUaGUgaXRlbXMgdGhhdCB3aWxsIGJlIGluIHRoZSBjZW50ZXJcbiAqIGByaWdodEl0ZW1zYCAoYXJyYXkpIC0gVGhlIGl0ZW1zIHRoYXQgd2lsbCBiZSBvbiB0aGUgcmlnaHRcbiAqL1xuY2xhc3MgSGVhZGVyQmFyIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICAvLyBCaW5kIGNvbnRleHQgdG8gbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VFbGVtZW50cyA9IHRoaXMubWFrZUVsZW1lbnRzLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMubWFrZVJpZ2h0ID0gdGhpcy5tYWtlUmlnaHQuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5tYWtlTGVmdCA9IHRoaXMubWFrZUxlZnQuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5tYWtlQ2VudGVyID0gdGhpcy5tYWtlQ2VudGVyLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbCBwLTIgYmctbGlnaHQgZmxleC1jb250YWluZXJcIixcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJIZWFkZXJCYXJcIixcbiAgICAgICAgICAgICAgICBzdHlsZTogJ2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpiYXNlbGluZTsnXG4gICAgICAgICAgICB9LCBbXG4gICAgICAgICAgICAgICAgdGhpcy5tYWtlTGVmdCgpLFxuICAgICAgICAgICAgICAgIHRoaXMubWFrZUNlbnRlcigpLFxuICAgICAgICAgICAgICAgIHRoaXMubWFrZVJpZ2h0KClcbiAgICAgICAgICAgIF0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgbWFrZUxlZnQoKXtcbiAgICAgICAgbGV0IGlubmVyRWxlbWVudHMgPSBbXTtcbiAgICAgICAgaWYodGhpcy5yZXBsYWNlbWVudHMuaGFzUmVwbGFjZW1lbnQoJ2xlZnQnKSB8fCB0aGlzLnByb3BzLm5hbWVkQ2hpbGRyZW4ubGVmdEl0ZW1zKXtcbiAgICAgICAgICAgIGlubmVyRWxlbWVudHMgPSB0aGlzLm1ha2VFbGVtZW50cygnbGVmdCcpO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdkaXYnLCB7Y2xhc3M6IFwiZmxleC1pdGVtXCIsIHN0eWxlOiBcImZsZXgtZ3JvdzowO1wifSwgW1xuICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICAgICAgY2xhc3M6IFwiZmxleC1jb250YWluZXJcIixcbiAgICAgICAgICAgICAgICAgICAgc3R5bGU6ICdkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OmNlbnRlcjthbGlnbi1pdGVtczpiYXNlbGluZTsnXG4gICAgICAgICAgICAgICAgfSwgaW5uZXJFbGVtZW50cylcbiAgICAgICAgICAgIF0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgbWFrZUNlbnRlcigpe1xuICAgICAgICBsZXQgaW5uZXJFbGVtZW50cyA9IFtdO1xuICAgICAgICBpZih0aGlzLnJlcGxhY2VtZW50cy5oYXNSZXBsYWNlbWVudCgnY2VudGVyJykgfHwgdGhpcy5wcm9wcy5uYW1lZENoaWxkcmVuLmNlbnRlckl0ZW1zKXtcbiAgICAgICAgICAgIGlubmVyRWxlbWVudHMgPSB0aGlzLm1ha2VFbGVtZW50cygnY2VudGVyJyk7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ2RpdicsIHtjbGFzczogXCJmbGV4LWl0ZW1cIiwgc3R5bGU6IFwiZmxleC1ncm93OjE7XCJ9LCBbXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgICAgICBjbGFzczogXCJmbGV4LWNvbnRhaW5lclwiLFxuICAgICAgICAgICAgICAgICAgICBzdHlsZTogJ2Rpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO2FsaWduLWl0ZW1zOmJhc2VsaW5lOydcbiAgICAgICAgICAgICAgICB9LCBpbm5lckVsZW1lbnRzKVxuICAgICAgICAgICAgXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlUmlnaHQoKXtcbiAgICAgICAgbGV0IGlubmVyRWxlbWVudHMgPSBbXTtcbiAgICAgICAgaWYodGhpcy5yZXBsYWNlbWVudHMuaGFzUmVwbGFjZW1lbnQoJ3JpZ2h0JykgfHwgdGhpcy5wcm9wcy5uYW1lZENoaWxkcmVuLnJpZ2h0SXRlbXMpe1xuICAgICAgICAgICAgaW5uZXJFbGVtZW50cyA9IHRoaXMubWFrZUVsZW1lbnRzKCdyaWdodCcpO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdkaXYnLCB7Y2xhc3M6IFwiZmxleC1pdGVtXCIsIHN0eWxlOiBcImZsZXgtZ3JvdzowO1wifSwgW1xuICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICAgICAgY2xhc3M6IFwiZmxleC1jb250YWluZXJcIixcbiAgICAgICAgICAgICAgICAgICAgc3R5bGU6ICdkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OmNlbnRlcjthbGlnbi1pdGVtczpiYXNlbGluZTsnXG4gICAgICAgICAgICAgICAgfSwgaW5uZXJFbGVtZW50cylcbiAgICAgICAgICAgIF0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgbWFrZUVsZW1lbnRzKHBvc2l0aW9uKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudHNGb3IocG9zaXRpb24pLm1hcChlbGVtZW50ID0+IHtcbiAgICAgICAgICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgICAgICAgICBoKCdzcGFuJywge2NsYXNzOiBcImZsZXgtaXRlbSBweC0zXCJ9LCBbZWxlbWVudF0pXG4gICAgICAgICAgICAgICAgKTtcbiAgICAgICAgICAgIH0pO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGRyZW5OYW1lZChgJHtwb3NpdGlvbn1JdGVtc2ApLm1hcChlbGVtZW50ID0+IHtcbiAgICAgICAgICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgICAgICAgICBoKCdzcGFuJywge2NsYXNzOiBcImZsZXgtaXRlbSBweC0zXCJ9LCBbZWxlbWVudF0pXG4gICAgICAgICAgICAgICAgKTtcbiAgICAgICAgICAgIH0pO1xuICAgICAgICB9XG4gICAgfVxufVxuXG5leHBvcnQge0hlYWRlckJhciwgSGVhZGVyQmFyIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBMYXJnZVBlbmRpbmdEb3dubG9hZERpc3BsYXkgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbmNsYXNzIExhcmdlUGVuZGluZ0Rvd25sb2FkRGlzcGxheSBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgaWQ6ICdvYmplY3RfZGF0YWJhc2VfbGFyZ2VfcGVuZGluZ19kb3dubG9hZF90ZXh0JyxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJMYXJnZVBlbmRpbmdEb3dubG9hZERpc3BsYXlcIixcbiAgICAgICAgICAgICAgICBjbGFzczogXCJjZWxsXCJcbiAgICAgICAgICAgIH0pXG4gICAgICAgICk7XG4gICAgfVxufVxuXG5leHBvcnQge0xhcmdlUGVuZGluZ0Rvd25sb2FkRGlzcGxheSwgTGFyZ2VQZW5kaW5nRG93bmxvYWREaXNwbGF5IGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBMb2FkQ29udGVudHNGcm9tVXJsIENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG5jbGFzcyBMb2FkQ29udGVudHNGcm9tVXJsIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkxvYWRDb250ZW50c0Zyb21VcmxcIixcbiAgICAgICAgICAgIH0sIFtoKCdkaXYnLCB7aWQ6IHRoaXMucHJvcHMuZXh0cmFEYXRhWydsb2FkVGFyZ2V0SWQnXX0sIFtdKV1cbiAgICAgICAgICAgIClcbiAgICAgICAgKTtcbiAgICB9XG5cbn1cblxuZXhwb3J0IHtMb2FkQ29udGVudHNGcm9tVXJsLCBMb2FkQ29udGVudHNGcm9tVXJsIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBNYWluIENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyBhIG9uZVxuICogcmVndWxhci1raW5kIHJlcGxhY2VtZW50OlxuICogKiBgY2hpbGRgXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBDaGlsZHJlblxuICogLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIGBjaGlsZGAgKHNpbmdsZSkgLSBUaGUgY2hpbGQgY2VsbCB0aGF0IGlzIHdyYXBwZWRcbiAqL1xuY2xhc3MgTWFpbiBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb21wb25lbnQgbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VDaGlsZCA9IHRoaXMubWFrZUNoaWxkLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdtYWluJywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIGNsYXNzOiBcInB5LW1kLTJcIixcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJNYWluXCJcbiAgICAgICAgICAgIH0sIFtcbiAgICAgICAgICAgICAgICBoKCdkaXYnLCB7Y2xhc3M6IFwiY29udGFpbmVyLWZsdWlkXCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgIHRoaXMubWFrZUNoaWxkKClcbiAgICAgICAgICAgICAgICBdKVxuICAgICAgICAgICAgXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlQ2hpbGQoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY2hpbGQnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ2NoaWxkJyk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cbmV4cG9ydCB7TWFpbiwgTWFpbiBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogTW9kYWwgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIE1vZGFsIGhhcyB0aGUgZm9sbG93aW5nIHNpbmdsZSByZXBsYWNlbWVudHM6XG4gKiAqYHRpdGxlYFxuICogKmBtZXNzYWdlYFxuICogQW5kIGhhcyB0aGUgZm9sbG93aW5nIGVudW1lcmF0ZWRcbiAqIHJlcGxhY2VtZW50c1xuICogKiBgYnV0dG9uYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgdGl0bGVgIChzaW5nbGUpIC0gQSBDZWxsIGNvbnRhaW5pbmcgdGhlIHRpdGxlXG4gKiBgbWVzc2FnZWAgKHNpbmdsZSkgLSBBIENlbGwgY29udGlhbmluZyB0aGUgYm9keSBvZiB0aGVcbiAqICAgICBtb2RhbCBtZXNzYWdlXG4gKiBgYnV0dG9uc2AgKGFycmF5KSAtIEFuIGFycmF5IG9mIGJ1dHRvbiBjZWxsc1xuICovXG5jbGFzcyBNb2RhbCBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG4gICAgICAgIHRoaXMubWFpblN0eWxlID0gJ2Rpc3BsYXk6YmxvY2s7cGFkZGluZy1yaWdodDoxNXB4Oyc7XG5cbiAgICAgICAgLy8gQmluZCBjb21wb25lbnQgbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VUaXRsZSA9IHRoaXMubWFrZVRpdGxlLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMubWFrZU1lc3NhZ2UgPSB0aGlzLm1ha2VNZXNzYWdlLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMubWFrZUJ1dHRvbnMgPSB0aGlzLm1ha2VCdXR0b25zLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbCBtb2RhbCBmYWRlIHNob3dcIixcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJNb2RhbFwiLFxuICAgICAgICAgICAgICAgIHJvbGU6IFwiZGlhbG9nXCIsXG4gICAgICAgICAgICAgICAgc3R5bGU6IG1haW5TdHlsZVxuICAgICAgICAgICAgfSwgW1xuICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtyb2xlOiBcImRvY3VtZW50XCIsIGNsYXNzOiBcIm1vZGFsLWRpYWxvZ1wifSwgW1xuICAgICAgICAgICAgICAgICAgICBoKCdkaXYnLCB7Y2xhc3M6IFwibW9kYWwtY29udGVudFwifSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgaCgnZGl2Jywge2NsYXNzOiBcIm1vZGFsLWhlYWRlclwifSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGgoJ2g1Jywge2NsYXNzOiBcIm1vZGFsLXRpdGxlXCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRoaXMubWFrZVRpdGxlKClcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBdKVxuICAgICAgICAgICAgICAgICAgICAgICAgXSksXG4gICAgICAgICAgICAgICAgICAgICAgICBoKCdkaXYnLCB7Y2xhc3M6IFwibW9kYWwtYm9keVwifSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRoaXMubWFrZU1lc3NhZ2UoKVxuICAgICAgICAgICAgICAgICAgICAgICAgXSksXG4gICAgICAgICAgICAgICAgICAgICAgICBoKCdkaXYnLCB7Y2xhc3M6IFwibW9kYWwtZm9vdGVyXCJ9LCB0aGlzLm1ha2VCdXR0b25zKCkpXG4gICAgICAgICAgICAgICAgICAgIF0pXG4gICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgIF0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgbWFrZUJ1dHRvbnMoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudHNGb3IoJ2J1dHRvbicpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGRyZW5OYW1lZCgnYnV0dG9ucycpXG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBtYWtlTWVzc2FnZSgpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdtZXNzYWdlJyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCdtZXNzYWdlJyk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBtYWtlVGl0bGUoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcigndGl0bGUnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ3RpdGxlJyk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cbmV4cG9ydCB7TW9kYWwsIE1vZGFsIGFzIGRlZmF1bHR9XG4iLCIvKipcbiAqIE9jdGljb24gQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbmNsYXNzIE9jdGljb24gZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29udGV4dCB0byBtZXRob2RzXG4gICAgICAgIHRoaXMuX2dldEhUTUxDbGFzc2VzID0gdGhpcy5fZ2V0SFRNTENsYXNzZXMuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuKFxuICAgICAgICAgICAgaCgnc3BhbicsIHtcbiAgICAgICAgICAgICAgICBjbGFzczogdGhpcy5fZ2V0SFRNTENsYXNzZXMoKSxcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJPY3RpY29uXCIsXG4gICAgICAgICAgICAgICAgXCJhcmlhLWhpZGRlblwiOiB0cnVlLFxuICAgICAgICAgICAgICAgIHN0eWxlOiB0aGlzLnByb3BzLmV4dHJhRGF0YS5kaXZTdHlsZVxuICAgICAgICAgICAgfSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBfZ2V0SFRNTENsYXNzZXMoKXtcbiAgICAgICAgbGV0IGNsYXNzZXMgPSBbXCJjZWxsXCIsIFwib2N0aWNvblwiXTtcbiAgICAgICAgdGhpcy5wcm9wcy5leHRyYURhdGEub2N0aWNvbkNsYXNzZXMuZm9yRWFjaChuYW1lID0+IHtcbiAgICAgICAgICAgIGNsYXNzZXMucHVzaChuYW1lKTtcbiAgICAgICAgfSk7XG4gICAgICAgIHJldHVybiBjbGFzc2VzLmpvaW4oXCIgXCIpO1xuICAgIH1cbn1cblxuZXhwb3J0IHtPY3RpY29uLCBPY3RpY29uIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBQYWRkaW5nIENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG5jbGFzcyBQYWRkaW5nIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ3NwYW4nLCB7XG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiUGFkZGluZ1wiLFxuICAgICAgICAgICAgICAgIGNsYXNzOiBcInB4LTJcIlxuICAgICAgICAgICAgfSwgW1wiIFwiXSlcbiAgICAgICAgKTtcbiAgICB9XG59XG5cbmV4cG9ydCB7UGFkZGluZywgUGFkZGluZyBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogUGxvdCBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgY29udGFpbnMgdGhlIGZvbGxvd2luZ1xuICogcmVndWxhciByZXBsYWNlbWVudHM6XG4gKiAqIGBjaGFydC11cGRhdGVyYFxuICogKiBgZXJyb3JgXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBDaGlsZHJlblxuICogLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIGBjaGFydFVwZGF0ZXJgIChzaW5nbGUpIC0gVGhlIFVwZGF0ZXIgY2VsbFxuICogYGVycm9yYCAoc2luZ2xlKSAtIEFuIGVycm9yIGNlbGwsIGlmIHByZXNlbnRcbiAqL1xuY2xhc3MgUGxvdCBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb21wb25lbnQgbWV0aG9kc1xuICAgICAgICB0aGlzLnNldHVwUGxvdCA9IHRoaXMuc2V0dXBQbG90LmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMubWFrZUNoYXJ0VXBkYXRlciA9IHRoaXMubWFrZUNoYXJ0VXBkYXRlci5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLm1ha2VFcnJvciA9IHRoaXMubWFrZUVycm9yLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgY29tcG9uZW50RGlkTG9hZCgpIHtcbiAgICAgICAgdGhpcy5zZXR1cFBsb3QoKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJQbG90XCIsXG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbFwiXG4gICAgICAgICAgICB9LCBbXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge2lkOiBgcGxvdCR7dGhpcy5wcm9wcy5pZH1gLCBzdHlsZTogdGhpcy5wcm9wcy5leHRyYURhdGEuZGl2U3R5bGV9KSxcbiAgICAgICAgICAgICAgICB0aGlzLm1ha2VDaGFydFVwZGF0ZXIoKSxcbiAgICAgICAgICAgICAgICB0aGlzLm1ha2VFcnJvcigpXG4gICAgICAgICAgICBdKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIG1ha2VDaGFydFVwZGF0ZXIoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY2hhcnQtdXBkYXRlcicpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnY2hhcnRVcGRhdGVyJyk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBtYWtlRXJyb3IoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignZXJyb3InKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ2Vycm9yJyk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBzZXR1cFBsb3QoKXtcbiAgICAgICAgY29uc29sZS5sb2coXCJTZXR0aW5nIHVwIGEgbmV3IHBsb3RseSBjaGFydC5cIik7XG4gICAgICAgIC8vIFRPRE8gVGhlc2UgYXJlIGdsb2JhbCB2YXIgZGVmaW5lZCBpbiBwYWdlLmh0bWxcbiAgICAgICAgLy8gd2Ugc2hvdWxkIGRvIHNvbWV0aGluZyBhYm91dCB0aGlzLlxuICAgICAgICB2YXIgcGxvdERpdiA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdwbG90JyArIHRoaXMucHJvcHMuaWQpO1xuICAgICAgICBQbG90bHkucGxvdChcbiAgICAgICAgICAgIHBsb3REaXYsXG4gICAgICAgICAgICBbXSxcbiAgICAgICAgICAgIHtcbiAgICAgICAgICAgICAgICBtYXJnaW46IHt0IDogMzAsIGw6IDMwLCByOiAzMCwgYjozMCB9LFxuICAgICAgICAgICAgICAgIHhheGlzOiB7cmFuZ2VzbGlkZXI6IHt2aXNpYmxlOiBmYWxzZX19XG4gICAgICAgICAgICB9LFxuICAgICAgICAgICAgeyBzY3JvbGxab29tOiB0cnVlLCBkcmFnbW9kZTogJ3BhbicsIGRpc3BsYXlsb2dvOiBmYWxzZSwgZGlzcGxheU1vZGVCYXI6ICdob3ZlcicsXG4gICAgICAgICAgICAgICAgbW9kZUJhckJ1dHRvbnM6IFsgWydwYW4yZCddLCBbJ3pvb20yZCddLCBbJ3pvb21JbjJkJ10sIFsnem9vbU91dDJkJ10gXSB9XG4gICAgICAgICk7XG4gICAgICAgIHBsb3REaXYub24oJ3Bsb3RseV9yZWxheW91dCcsXG4gICAgICAgICAgICBmdW5jdGlvbihldmVudGRhdGEpe1xuICAgICAgICAgICAgICAgIGlmIChwbG90RGl2LmlzX3NlcnZlcl9kZWZpbmVkX21vdmUgPT09IHRydWUpIHtcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuXG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIC8vaWYgd2UncmUgc2VuZGluZyBhIHN0cmluZywgdGhlbiBpdHMgYSBkYXRlIG9iamVjdCwgYW5kIHdlIHdhbnQgdG8gc2VuZFxuICAgICAgICAgICAgICAgIC8vIGEgdGltZXN0YW1wXG4gICAgICAgICAgICAgICAgaWYgKHR5cGVvZihldmVudGRhdGFbJ3hheGlzLnJhbmdlWzBdJ10pID09PSAnc3RyaW5nJykge1xuICAgICAgICAgICAgICAgICAgICBldmVudGRhdGEgPSBPYmplY3QuYXNzaWduKHt9LGV2ZW50ZGF0YSk7XG4gICAgICAgICAgICAgICAgICAgIGV2ZW50ZGF0YVtcInhheGlzLnJhbmdlWzBdXCJdID0gRGF0ZS5wYXJzZShldmVudGRhdGFbXCJ4YXhpcy5yYW5nZVswXVwiXSkgLyAxMDAwLjA7XG4gICAgICAgICAgICAgICAgICAgIGV2ZW50ZGF0YVtcInhheGlzLnJhbmdlWzFdXCJdID0gRGF0ZS5wYXJzZShldmVudGRhdGFbXCJ4YXhpcy5yYW5nZVsxXVwiXSkgLyAxMDAwLjA7XG4gICAgICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICAgICAgbGV0IHJlc3BvbnNlRGF0YSA9IHtcbiAgICAgICAgICAgICAgICAgICAgJ2V2ZW50JzoncGxvdF9sYXlvdXQnLFxuICAgICAgICAgICAgICAgICAgICAndGFyZ2V0X2NlbGwnOiAnX19pZGVudGl0eV9fJyxcbiAgICAgICAgICAgICAgICAgICAgJ2RhdGEnOiBldmVudGRhdGFcbiAgICAgICAgICAgICAgICB9O1xuICAgICAgICAgICAgICAgIGNlbGxTb2NrZXQuc2VuZFN0cmluZyhKU09OLnN0cmluZ2lmeShyZXNwb25zZURhdGEpKTtcbiAgICAgICAgICAgIH0pO1xuICAgIH1cbn1cblxuZXhwb3J0IHtQbG90LCBQbG90IGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBQb3BvdmVyIENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogVGhpcyBjb21wb25lbnQgY29udGFpbnMgdGhlIGZvbGxvd2luZ1xuICogcmVndWxhciByZXBsYWNlbWVudHM6XG4gKiAqIGB0aXRsZWBcbiAqICogYGRldGFpbGBcbiAqICogYGNvbnRlbnRzYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgY29udGVudGAgKHNpbmdsZSkgLSBUaGUgY29udGVudCBvZiB0aGUgcG9wb3ZlclxuICogYGRldGFpbGAgKHNpbmdsZSkgLSBEZXRhaWwgb2YgdGhlIHBvcG92ZXJcbiAqIGB0aXRsZWAgKHNpbmdsZSkgLSBUaGUgdGl0bGUgZm9yIHRoZSBwb3BvdmVyXG4gKi9cbmNsYXNzIFBvcG92ZXIgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29tcG9uZW50IG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlVGl0bGUgPSB0aGlzLm1ha2VUaXRsZS5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLm1ha2VDb250ZW50ID0gdGhpcy5tYWtlQ29udGVudC5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLm1ha2VEZXRhaWwgPSB0aGlzLm1ha2VEZXRhaWwuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIGgoJ2RpdicsXG4gICAgICAgICAgICB7XG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbCBwb3BvdmVyLWNlbGxcIixcbiAgICAgICAgICAgICAgICBzdHlsZTogdGhpcy5wcm9wcy5leHRyYURhdGEuZGl2U3R5bGUsXG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiUG9wb3ZlclwiXG4gICAgICAgICAgICB9LCBbXG4gICAgICAgICAgICAgICAgaCgnYScsXG4gICAgICAgICAgICAgICAgICAgIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIGhyZWY6IFwiI3BvcG1haW5fXCIgKyB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgICAgICAgICAgXCJkYXRhLXRvZ2dsZVwiOiBcInBvcG92ZXJcIixcbiAgICAgICAgICAgICAgICAgICAgICAgIFwiZGF0YS10cmlnZ2VyXCI6IFwiZm9jdXNcIixcbiAgICAgICAgICAgICAgICAgICAgICAgIFwiZGF0YS1iaW5kXCI6IFwiI3BvcF9cIiArIHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgICAgICAgICBcImRhdGEtcGxhY2VtZW50XCI6IFwiYm90dG9tXCIsXG4gICAgICAgICAgICAgICAgICAgICAgICByb2xlOiBcImJ1dHRvblwiLFxuICAgICAgICAgICAgICAgICAgICAgICAgY2xhc3M6IFwiYnRuIGJ0bi14c1wiXG4gICAgICAgICAgICAgICAgICAgIH0sXG4gICAgICAgICAgICAgICAgICBbdGhpcy5tYWtlQ29udGVudCgpXVxuICAgICAgICAgICAgICAgICksXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge3N0eWxlOiBcImRpc3BsYXk6bm9uZVwifSwgW1xuICAgICAgICAgICAgICAgICAgICBoKFwiZGl2XCIsIHtpZDogXCJwb3BfXCIgKyB0aGlzLnByb3BzLmlkfSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgaChcImRpdlwiLCB7Y2xhc3M6IFwiZGF0YS10aXRsZVwifSwgW3RoaXMubWFrZVRpdGxlKCldKSxcbiAgICAgICAgICAgICAgICAgICAgICAgIGgoXCJkaXZcIiwge2NsYXNzOiBcImRhdGEtY29udGVudFwifSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGgoXCJkaXZcIiwge3N0eWxlOiBcIndpZHRoOiBcIiArIHRoaXMucHJvcHMud2lkdGggKyBcInB4XCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRoaXMubWFrZURldGFpbCgpXVxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIClcbiAgICAgICAgICAgICAgICAgICAgICAgIF0pXG4gICAgICAgICAgICAgICAgICAgIF0pXG4gICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgIF1cbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlQ29udGVudCgpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdjb250ZW50cycpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnY29udGVudCcpO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgbWFrZURldGFpbCgpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdkZXRhaWwnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ2RldGFpbCcpO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgbWFrZVRpdGxlKCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ3RpdGxlJyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCd0aXRsZScpO1xuICAgICAgICB9XG4gICAgfVxufVxuXG5leHBvcnQge1BvcG92ZXIsIFBvcG92ZXIgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIFJvb3RDZWxsIENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyBhIG9uZVxuICogcmVndWxhci1raW5kIHJlcGxhY2VtZW50OlxuICogKiBgY2BcbiAqL1xuXG4vKipcbiAqIEFib3V0IE5hbWVkIENoaWxkcmVuXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogYGNoaWxkYCAoc2luZ2xlKSAtIFRoZSBjaGlsZCBjZWxsIHRoaXMgY29udGFpbmVyIGNvbnRhaW5zXG4gKi9cbmNsYXNzIFJvb3RDZWxsIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICAvLyBCaW5kIGNvbXBvbmVudCBtZXRob2RzXG4gICAgICAgIHRoaXMubWFrZUNoaWxkID0gdGhpcy5tYWtlQ2hpbGQuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJSb290Q2VsbFwiXG4gICAgICAgICAgICB9LCBbdGhpcy5tYWtlQ2hpbGQoKV0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgbWFrZUNoaWxkKCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ2MnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ2NoaWxkJyk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cbmV4cG9ydCB7Um9vdENlbGwsIFJvb3RDZWxsIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBTY3JvbGxhYmxlICBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIGEgb25lXG4gKiByZWd1bGFyLWtpbmQgcmVwbGFjZW1lbnQ6XG4gKiAqIGBjaGlsZGBcbiAqL1xuXG4vKipcbiAqIEFib3V0IE5hbWVkIENoaWxkcmVuXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogYGNoaWxkYCAoc2luZ2xlKSAtIFRoZSBjZWxsL2NvbXBvbmVudCB0aGlzIGluc3RhbmNlIGNvbnRhaW5zXG4gKi9cbmNsYXNzIFNjcm9sbGFibGUgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29tcG9uZW50IG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlQ2hpbGQgPSB0aGlzLm1ha2VDaGlsZC5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIlNjcm9sbGFibGVcIlxuICAgICAgICAgICAgfSwgW3RoaXMubWFrZUNoaWxkKCldKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIG1ha2VDaGlsZCgpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdjaGlsZCcpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnY2hpbGQnKTtcbiAgICAgICAgfVxuICAgIH1cbn1cblxuZXhwb3J0IHtTY3JvbGxhYmxlLCBTY3JvbGxhYmxlIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBTZXF1ZW5jZSBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogU2VxdWVuY2UgaGFzIHRoZSBmb2xsb3dpbmcgZW51bWVyYXRlZFxuICogcmVwbGFjZW1lbnQ6XG4gKiAqIGBjYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgZWxlbWVudHNgIChhcnJheSkgLSBBIGxpc3Qgb2YgQ2VsbHMgdGhhdCBhcmUgaW4gdGhlXG4gKiAgICBzZXF1ZW5jZS5cbiAqL1xuY2xhc3MgU2VxdWVuY2UgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29tcG9uZW50IG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlRWxlbWVudHMgPSB0aGlzLm1ha2VFbGVtZW50cy5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIGNsYXNzOiBcImNlbGxcIixcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJTZXF1ZW5jZVwiLFxuICAgICAgICAgICAgICAgIHN0eWxlOiB0aGlzLnByb3BzLmV4dHJhRGF0YS5kaXZTdHlsZVxuICAgICAgICAgICAgfSwgdGhpcy5tYWtlRWxlbWVudHMoKSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlRWxlbWVudHMoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudHNGb3IoJ2MnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkcmVuTmFtZWQoJ2VsZW1lbnRzJyk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cbmV4cG9ydCB7U2VxdWVuY2UsIFNlcXVlbmNlIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBTaGVldCBDZWxsIENvbXBvbmVudFxuICogTk9URTogVGhpcyBpcyBpbiBwYXJ0IGEgd3JhcHBlclxuICogZm9yIGhhbmRzb250YWJsZXMuXG4gKi9cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogVGhpcyBjb21wb25lbnQgaGFzIG9uZSByZWd1bGFyXG4gKiByZXBsYWNlbWVudDpcbiAqICogYGVycm9yYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgZXJyb3JgIChzaW5nbGUpIC0gQW4gZXJyb3IgY2VsbCBpZiBwcmVzZW50XG4gKi9cbmNsYXNzIFNoZWV0IGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICB0aGlzLmN1cnJlbnRUYWJsZSA9IG51bGw7XG5cbiAgICAgICAgLy8gQmluZCBjb250ZXh0IHRvIG1ldGhvZHNcbiAgICAgICAgdGhpcy5pbml0aWFsaXplVGFibGUgPSB0aGlzLmluaXRpYWxpemVUYWJsZS5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLmluaXRpYWxpemVIb29rcyA9IHRoaXMuaW5pdGlhbGl6ZUhvb2tzLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMubWFrZUVycm9yID0gdGhpcy5tYWtlRXJyb3IuYmluZCh0aGlzKTtcblxuICAgICAgICAvKipcbiAgICAgICAgICogV0FSTklORzogVGhlIENlbGwgdmVyc2lvbiBvZiBTaGVldCBpcyBzdGlsbCB1c2luZyBjZXJ0YWluXG4gICAgICAgICAqIHBvc3RzY3JpcHRzIGJlY2F1c2Ugd2UgaGF2ZSBub3QgeWV0IHJlZmFjdG9yZWQgdGhlIHNvY2tldFxuICAgICAgICAgKiBwcm90b2NvbC5cbiAgICAgICAgICogUmVtb3ZlIHRoaXMgd2FybmluZyBhYm91dCBpdCBvbmNlIHRoYXQgaGFwcGVucyFcbiAgICAgICAgICovXG4gICAgICAgIGNvbnNvbGUud2FybihgW1RPRE9dIFNoZWV0IHN0aWxsIHVzZXMgY2VydGFpbiBwb3N0c2NlcmlwdHMgaW4gaXRzIGludGVyYWN0aW9uLiBTZWUgY29tcG9uZW50IGNvbnN0cnVjdG9yIGNvbW1lbnQgZm9yIG1vcmUgaW5mb3JtYXRpb25gKTtcbiAgICB9XG5cbiAgICBjb21wb25lbnREaWRMb2FkKCl7XG4gICAgICAgIGNvbnNvbGUubG9nKGAjY29tcG9uZW50RGlkTG9hZCBjYWxsZWQgZm9yIFNoZWV0ICR7dGhpcy5wcm9wcy5pZH1gKTtcbiAgICAgICAgY29uc29sZS5sb2coYFRoaXMgc2hlZXQgaGFzIHRoZSBmb2xsb3dpbmcgcmVwbGFjZW1lbnRzOmAsIHRoaXMucmVwbGFjZW1lbnRzKTtcbiAgICAgICAgdGhpcy5pbml0aWFsaXplVGFibGUoKTtcbiAgICAgICAgaWYodGhpcy5wcm9wcy5leHRyYURhdGFbJ2hhbmRsZXNEb3VibGVDbGljayddKXtcbiAgICAgICAgICAgIHRoaXMuaW5pdGlhbGl6ZUhvb2tzKCk7XG4gICAgICAgIH1cbiAgICAgICAgLy8gUmVxdWVzdCBpbml0aWFsIGRhdGE/XG4gICAgICAgIGNlbGxTb2NrZXQuc2VuZFN0cmluZyhKU09OLnN0cmluZ2lmeSh7XG4gICAgICAgICAgICBldmVudDogXCJzaGVldF9uZWVkc19kYXRhXCIsXG4gICAgICAgICAgICB0YXJnZXRfY2VsbDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgIGRhdGE6IDBcbiAgICAgICAgfSkpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICBjb25zb2xlLmxvZyhgUmVuZGVyaW5nIHNoZWV0ICR7dGhpcy5wcm9wcy5pZH1gKTtcbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJTaGVldFwiLFxuICAgICAgICAgICAgICAgIGNsYXNzOiBcImNlbGxcIlxuICAgICAgICAgICAgfSwgW1xuICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICAgICAgaWQ6IGBzaGVldCR7dGhpcy5wcm9wcy5pZH1gLFxuICAgICAgICAgICAgICAgICAgICBzdHlsZTogdGhpcy5wcm9wcy5leHRyYURhdGEuZGl2U3R5bGUsXG4gICAgICAgICAgICAgICAgICAgIGNsYXNzOiBcImhhbmRzb250YWJsZVwiXG4gICAgICAgICAgICAgICAgfSwgW3RoaXMubWFrZUVycm9yKCldKVxuICAgICAgICAgICAgXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBpbml0aWFsaXplVGFibGUoKXtcbiAgICAgICAgY29uc29sZS5sb2coYCNpbml0aWFsaXplVGFibGUgY2FsbGVkIGZvciBTaGVldCAke3RoaXMucHJvcHMuaWR9YCk7XG4gICAgICAgIGxldCBnZXRQcm9wZXJ0eSA9IGZ1bmN0aW9uKGluZGV4KXtcbiAgICAgICAgICAgIHJldHVybiBmdW5jdGlvbihyb3cpe1xuICAgICAgICAgICAgICAgIHJldHVybiByb3dbaW5kZXhdO1xuICAgICAgICAgICAgfTtcbiAgICAgICAgfTtcbiAgICAgICAgbGV0IGVtcHR5Um93ID0gW107XG4gICAgICAgIGxldCBkYXRhTmVlZGVkQ2FsbGJhY2sgPSBmdW5jdGlvbihldmVudE9iamVjdCl7XG4gICAgICAgICAgICBldmVudE9iamVjdC50YXJnZXRfY2VsbCA9IHRoaXMucHJvcHMuaWQ7XG4gICAgICAgICAgICBjZWxsU29ja2V0LnNlbmRTdHJpbmcoSlNPTi5zdHJpbmdpZnkoZXZlbnRPYmplY3QpKTtcbiAgICAgICAgfS5iaW5kKHRoaXMpO1xuICAgICAgICBsZXQgZGF0YSA9IG5ldyBTeW50aGV0aWNJbnRlZ2VyQXJyYXkodGhpcy5wcm9wcy5leHRyYURhdGEucm93Q291bnQsIGVtcHR5Um93LCBkYXRhTmVlZGVkQ2FsbGJhY2spO1xuICAgICAgICBsZXQgY29udGFpbmVyID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoYHNoZWV0JHt0aGlzLnByb3BzLmlkfWApO1xuICAgICAgICBsZXQgY29sdW1uTmFtZXMgPSB0aGlzLnByb3BzLmV4dHJhRGF0YS5jb2x1bW5OYW1lcztcbiAgICAgICAgbGV0IGNvbHVtbnMgPSBjb2x1bW5OYW1lcy5tYXAoKG5hbWUsIGlkeCkgPT4ge1xuICAgICAgICAgICAgZW1wdHlSb3cucHVzaChcIlwiKTtcbiAgICAgICAgICAgIHJldHVybiB7ZGF0YTogZ2V0UHJvcGVydHkoaWR4KX07XG4gICAgICAgIH0pO1xuXG4gICAgICAgIHRoaXMuY3VycmVudFRhYmxlID0gbmV3IEhhbmRzb250YWJsZShjb250YWluZXIsIHtcbiAgICAgICAgICAgIGRhdGEsXG4gICAgICAgICAgICBkYXRhU2NoZW1hOiBmdW5jdGlvbihvcHRzKXtyZXR1cm4ge307fSxcbiAgICAgICAgICAgIGNvbEhlYWRlcnM6IGNvbHVtbk5hbWVzLFxuICAgICAgICAgICAgY29sdW1ucyxcbiAgICAgICAgICAgIHJvd0hlYWRlcnM6dHJ1ZSxcbiAgICAgICAgICAgIHJvd0hlYWRlcldpZHRoOiAxMDAsXG4gICAgICAgICAgICB2aWV3cG9ydFJvd1JlbmRlcmluZ09mZnNldDogMTAwLFxuICAgICAgICAgICAgYXV0b0NvbHVtblNpemU6IGZhbHNlLFxuICAgICAgICAgICAgYXV0b1Jvd0hlaWdodDogZmFsc2UsXG4gICAgICAgICAgICBtYW51YWxDb2x1bW5SZXNpemU6IHRydWUsXG4gICAgICAgICAgICBjb2xXaWR0aHM6IHRoaXMucHJvcHMuZXh0cmFEYXRhLmNvbHVtbldpZHRoLFxuICAgICAgICAgICAgcm93SGVpZ2h0czogMjMsXG4gICAgICAgICAgICByZWFkT25seTogdHJ1ZSxcbiAgICAgICAgICAgIE1hbnVhbFJvd01vdmU6IGZhbHNlXG4gICAgICAgIH0pO1xuICAgICAgICBoYW5kc09uVGFibGVzW3RoaXMucHJvcHMuaWRdID0ge1xuICAgICAgICAgICAgdGFibGU6IHRoaXMuY3VycmVudFRhYmxlLFxuICAgICAgICAgICAgbGFzdENlbGxDbGlja2VkOiB7cm93OiAtMTAwLCBjb2w6IC0xMDB9LFxuICAgICAgICAgICAgZGJsQ2xpY2tlZDogdHJ1ZVxuICAgICAgICB9O1xuICAgIH1cblxuICAgIGluaXRpYWxpemVIb29rcygpe1xuICAgICAgICBIYW5kc29udGFibGUuaG9va3MuYWRkKFwiYmVmb3JlT25DZWxsTW91c2VEb3duXCIsIChldmVudCwgZGF0YSkgPT4ge1xuICAgICAgICAgICAgbGV0IGhhbmRzT25PYmogPSBoYW5kc09uVGFibGVzW3RoaXMucHJvcHMuaWRdO1xuICAgICAgICAgICAgbGV0IGxhc3RSb3cgPSBoYW5kc09uT2JqLmxhc3RDZWxsQ2xpY2tlZC5yb3c7XG4gICAgICAgICAgICBsZXQgbGFzdENvbCA9IGhhbmRzT25PYmoubGFzdENlbGxDbGlja2VkLmNvbDtcblxuICAgICAgICAgICAgaWYoKGxhc3RSb3cgPT0gZGF0YS5yb3cpICYmIChsYXN0Q29sID0gZGF0YS5jb2wpKXtcbiAgICAgICAgICAgICAgICBoYW5kc09uT2JqLmRibENsaWNrZWQgPSB0cnVlO1xuICAgICAgICAgICAgICAgIHNldFRpbWVvdXQoKCkgPT4ge1xuICAgICAgICAgICAgICAgICAgICBpZihoYW5kc09uT2JqLmRibENsaWNrZWQpe1xuICAgICAgICAgICAgICAgICAgICAgICAgY2VsbFNvY2tldC5zZW5kU3RyaW5nKEpTT04uc3RyaW5naWZ5KHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBldmVudDogJ29uQ2VsbERibENsaWNrJyxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB0YXJnZXRfY2VsbDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICByb3c6IGRhdGEucm93LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNvbDogZGF0YS5jb2xcbiAgICAgICAgICAgICAgICAgICAgICAgIH0pKTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICBoYW5kc09uT2JqLmxhc3RDZWxsQ2xpY2tlZCA9IHtyb3c6IC0xMDAsIGNvbDogLTEwMH07XG4gICAgICAgICAgICAgICAgICAgIGhhbmRzT25PYmouZGJsQ2xpY2tlZCA9IGZhbHNlO1xuICAgICAgICAgICAgICAgIH0sIDIwMCk7XG4gICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgIGhhbmRzT25PYmoubGFzdENlbGxDbGlja2VkID0ge3JvdzogZGF0YS5yb3csIGNvbDogZGF0YS5jb2x9O1xuICAgICAgICAgICAgICAgIHNldFRpbWVvdXQoKCkgPT4ge1xuICAgICAgICAgICAgICAgICAgICBoYW5kc09uT2JqLmxhc3RDZWxsQ2xpY2tlZCA9IHtyb3c6IC0xMDAsIGNvbDogLTEwMH07XG4gICAgICAgICAgICAgICAgICAgIGhhbmRzT25PYmouZGJsQ2xpY2tlZCA9IGZhbHNlO1xuICAgICAgICAgICAgICAgIH0sIDYwMCk7XG4gICAgICAgICAgICB9XG4gICAgICAgIH0sIHRoaXMuY3VycmVudFRhYmxlKTtcblxuICAgICAgICBIYW5kc29udGFibGUuaG9va3MuYWRkKFwiYmVmb3JlT25DZWxsQ29udGV4dE1lbnVcIiwgKGV2ZW50LCBkYXRhKSA9PiB7XG4gICAgICAgICAgICBsZXQgaGFuZHNPbk9iaiA9IGhhbmRzT25UYWJsZXNbdGhpcy5wcm9wcy5pZF07XG4gICAgICAgICAgICBoYW5kc09uT2JqLmRibENsaWNrZWQgPSBmYWxzZTtcbiAgICAgICAgICAgIGhhbmRzT25PYmoubGFzdENlbGxDbGlja2VkID0ge3JvdzogLTEwMCwgY29sOiAtMTAwfTtcbiAgICAgICAgfSwgdGhpcy5jdXJyZW50VGFibGUpO1xuXG4gICAgICAgIEhhbmRzb250YWJsZS5ob29rcy5hZGQoXCJiZWZvcmVDb250ZXh0TWVudVNob3dcIiwgKGV2ZW50LCBkYXRhKSA9PiB7XG4gICAgICAgICAgICBsZXQgaGFuZHNPbk9iaiA9IGhhbmRzT25UYWJsZXNbdGhpcy5wcm9wcy5pZF07XG4gICAgICAgICAgICBoYW5kc09uT2JqLmRibENsaWNrZWQgPSBmYWxzZTtcbiAgICAgICAgICAgIGhhbmRzT25PYmoubGFzdENlbGxDbGlja2VkID0ge3JvdzogLTEwMCwgY29sOiAtMTAwfTtcbiAgICAgICAgfSwgdGhpcy5jdXJyZW50VGFibGUpO1xuICAgIH1cblxuICAgIG1ha2VFcnJvcigpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdlcnJvcicpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnZXJyb3InKTtcbiAgICAgICAgfVxuICAgIH1cbn1cblxuLyoqIENvcGllZCBvdmVyIGZyb20gQ2VsbHMgaW1wbGVtZW50YXRpb24gKiovXG5jb25zdCBTeW50aGV0aWNJbnRlZ2VyQXJyYXkgPSBmdW5jdGlvbihzaXplLCBlbXB0eVJvdyA9IFtdLCBjYWxsYmFjayl7XG4gICAgdGhpcy5sZW5ndGggPSBzaXplO1xuICAgIHRoaXMuY2FjaGUgPSB7fTtcbiAgICB0aGlzLnB1c2ggPSBmdW5jdGlvbigpe307XG4gICAgdGhpcy5zcGxpY2UgPSBmdW5jdGlvbigpe307XG5cbiAgICB0aGlzLnNsaWNlID0gZnVuY3Rpb24obG93LCBoaWdoKXtcbiAgICAgICAgaWYoaGlnaCA9PT0gdW5kZWZpbmVkKXtcbiAgICAgICAgICAgIGhpZ2ggPSB0aGlzLmxlbmd0aDtcbiAgICAgICAgfVxuXG4gICAgICAgIGxldCByZXMgPSBBcnJheShoaWdoIC0gbG93KTtcbiAgICAgICAgbGV0IGluaXRMb3cgPSBsb3c7XG4gICAgICAgIHdoaWxlKGxvdyA8IGhpZ2gpe1xuICAgICAgICAgICAgbGV0IG91dCA9IHRoaXMuY2FjaGVbbG93XTtcbiAgICAgICAgICAgIGlmKG91dCA9PT0gdW5kZWZpbmVkKXtcbiAgICAgICAgICAgICAgICBpZihjYWxsYmFjayl7XG4gICAgICAgICAgICAgICAgICAgIGNhbGxiYWNrKHtcbiAgICAgICAgICAgICAgICAgICAgICAgIGV2ZW50OiAnc2hlZXRfbmVlZHNfZGF0YScsXG4gICAgICAgICAgICAgICAgICAgICAgICBkYXRhOiBsb3dcbiAgICAgICAgICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIG91dCA9IGVtcHR5Um93O1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgcmVzW2xvdyAtIGluaXRMb3ddID0gb3V0O1xuICAgICAgICAgICAgbG93ICs9IDE7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIHJlcztcbiAgICB9O1xufTtcblxuZXhwb3J0IHtTaGVldCwgU2hlZXQgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIFNpbmdsZUxpbmVUZXh0Qm94IENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG5jbGFzcyBTaW5nbGVMaW5lVGV4dEJveCBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb250ZXh0IHRvIG1ldGhvZHNcbiAgICAgICAgdGhpcy5jaGFuZ2VIYW5kbGVyID0gdGhpcy5jaGFuZ2VIYW5kbGVyLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIGxldCBhdHRycyA9XG4gICAgICAgICAgICB7XG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbFwiLFxuICAgICAgICAgICAgICAgIGlkOiBcInRleHRfXCIgKyB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIHR5cGU6IFwidGV4dFwiLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIlNpbmdsZUxpbmVUZXh0Qm94XCIsXG4gICAgICAgICAgICAgICAgb25jaGFuZ2U6IChldmVudCkgPT4ge3RoaXMuY2hhbmdlSGFuZGxlcihldmVudC50YXJnZXQudmFsdWUpO31cbiAgICAgICAgICAgIH07XG4gICAgICAgIGlmICh0aGlzLnByb3BzLmV4dHJhRGF0YS5pbnB1dFZhbHVlICE9PSB1bmRlZmluZWQpIHtcbiAgICAgICAgICAgIGF0dHJzLnBhdHRlcm4gPSB0aGlzLnByb3BzLmV4dHJhRGF0YS5pbnB1dFZhbHVlO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiBoKCdpbnB1dCcsIGF0dHJzLCBbXSk7XG4gICAgfVxuXG4gICAgY2hhbmdlSGFuZGxlcih2YWwpIHtcbiAgICAgICAgY2VsbFNvY2tldC5zZW5kU3RyaW5nKFxuICAgICAgICAgICAgSlNPTi5zdHJpbmdpZnkoXG4gICAgICAgICAgICAgICAge1xuICAgICAgICAgICAgICAgICAgICBcImV2ZW50XCI6IFwiY2xpY2tcIixcbiAgICAgICAgICAgICAgICAgICAgXCJ0YXJnZXRfY2VsbFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgICAgICBcInRleHRcIjogdmFsXG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgKVxuICAgICAgICApO1xuICAgIH1cbn1cblxuZXhwb3J0IHtTaW5nbGVMaW5lVGV4dEJveCwgU2luZ2xlTGluZVRleHRCb3ggYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIFNwYW4gQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cblxuY2xhc3MgU3BhbiBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdzcGFuJywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIlNwYW5cIixcbiAgICAgICAgICAgICAgICBjbGFzczogXCJjZWxsXCJcbiAgICAgICAgICAgIH0sIFt0aGlzLnByb3BzLmV4dHJhRGF0YS50ZXh0XSlcbiAgICAgICAgKTtcbiAgICB9XG59XG5cbmV4cG9ydCB7U3BhbiwgU3BhbiBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogU3Vic2NyaWJlZCBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIGEgc2luZ2xlXG4gKiByZWd1bGFyIHJlcGxhY2VtZW50OlxuICogKiBgY29udGVudHNgXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBDaGlsZHJlblxuICogLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIGBjb250ZW50YCAoc2luZ2xlKSAtIFRoZSB1bmRlcmx5aW5nIENlbGwgdGhhdCBpcyBzdWJzY3JpYmVkXG4gKi9cbmNsYXNzIFN1YnNjcmliZWQgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29tcG9uZW50IG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlQ29udGVudCA9IHRoaXMubWFrZUNvbnRlbnQuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIGgoJ2RpdicsXG4gICAgICAgICAgICB7XG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbCBzdWJzY3JpYmVkXCIsXG4gICAgICAgICAgICAgICAgc3R5bGU6IHRoaXMucHJvcHMuZXh0cmFEYXRhLmRpdlN0eWxlLFxuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIlN1YnNjcmliZWRcIlxuICAgICAgICAgICAgfSwgW3RoaXMubWFrZUNvbnRlbnQoKV1cbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlQ29udGVudCgpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdjb250ZW50cycpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnY29udGVudCcpO1xuICAgICAgICB9XG4gICAgfVxufVxuXG5leHBvcnQge1N1YnNjcmliZWQsIFN1YnNjcmliZWQgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIFN1YnNjcmliZWRTZXF1ZW5jZSBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIGEgc2luZ2xlXG4gKiBlbnVtZXJhdGVkIHJlcGxhY2VtZW50OlxuICogKiBgY2hpbGRgXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogYGNoaWxkcmVuYCAoYXJyYXkpIC0gQW4gYXJyYXkgb2YgQ2VsbHMgdGhhdCBhcmUgc3Vic2NyaWJlZFxuICovXG5jbGFzcyBTdWJzY3JpYmVkU2VxdWVuY2UgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuICAgICAgICAvL3RoaXMuYWRkUmVwbGFjZW1lbnQoJ2NvbnRlbnRzJywgJ19fX19fY29udGVudHNfXycpO1xuICAgICAgICAvL1xuICAgICAgICAvLyBCaW5kIGNvbnRleHQgdG8gbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VDbGFzcyA9IHRoaXMubWFrZUNsYXNzLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMubWFrZUNoaWxkcmVuID0gdGhpcy5tYWtlQ2hpbGRyZW4uYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5fbWFrZVJlcGxhY2VtZW50Q2hpbGRyZW4gPSB0aGlzLl9tYWtlUmVwbGFjZW1lbnRDaGlsZHJlbi5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gaCgnZGl2JyxcbiAgICAgICAgICAgIHtcbiAgICAgICAgICAgICAgICBjbGFzczogdGhpcy5tYWtlQ2xhc3MoKSxcbiAgICAgICAgICAgICAgICBzdHlsZTogdGhpcy5wcm9wcy5leHRyYURhdGEuZGl2U3R5bGUsXG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiU3Vic2NyaWJlZFNlcXVlbmNlXCJcbiAgICAgICAgICAgIH0sIFt0aGlzLm1ha2VDaGlsZHJlbigpXVxuICAgICAgICApO1xuICAgIH1cblxuICAgIG1ha2VDaGlsZHJlbigpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuX21ha2VSZXBsYWNlbWVudENoaWxkcmVuKCk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICBpZih0aGlzLnByb3BzLmV4dHJhRGF0YS5hc0NvbHVtbnMpe1xuICAgICAgICAgICAgICAgIGxldCBmb3JtYXR0ZWRDaGlsZHJlbiA9IHRoaXMucmVuZGVyQ2hpbGRyZW5OYW1lZCgnY2hpbGRyZW4nKS5tYXAoY2hpbGRFbCA9PiB7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybihcbiAgICAgICAgICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtjbGFzczogXCJjb2wtc21cIiwga2V5OiBjaGlsZEVsZW1lbnQuaWR9LCBbXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgaCgnc3BhbicsIHt9LCBbY2hpbGRFbF0pXG4gICAgICAgICAgICAgICAgICAgICAgICBdKVxuICAgICAgICAgICAgICAgICAgICApO1xuICAgICAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtjbGFzczogXCJyb3cgZmxleC1ub3dyYXBcIiwga2V5OiBgJHt0aGlzLnByb3BzLmlkfS1zcGluZS13cmFwcGVyYH0sIGZvcm1hdHRlZENoaWxkcmVuKVxuICAgICAgICAgICAgICAgICk7XG4gICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtrZXk6IGAke3RoaXMucHJvcHMuaWR9LXNwaW5lLXdyYXBwZXJgfSwgdGhpcy5yZW5kZXJDaGlsZHJlbk5hbWVkKCdjaGlsZHJlbicpKVxuICAgICAgICAgICAgICAgICk7XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBtYWtlQ2xhc3MoKSB7XG4gICAgICAgIGlmICh0aGlzLnByb3BzLmV4dHJhRGF0YS5hc0NvbHVtbnMpIHtcbiAgICAgICAgICAgIHJldHVybiBcImNlbGwgc3Vic2NyaWJlZFNlcXVlbmNlIGNvbnRhaW5lci1mbHVpZFwiO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiBcImNlbGwgc3Vic2NyaWJlZFNlcXVlbmNlXCI7XG4gICAgfVxuXG4gICAgX21ha2VSZXBsYWNlbWVudENoaWxkcmVuKCl7XG4gICAgICAgIGlmKHRoaXMucHJvcHMuZXh0cmFEYXRhLmFzQ29sdW1ucyl7XG4gICAgICAgICAgICBsZXQgZm9ybWF0dGVkQ2hpbGRyZW4gPSB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudHNGb3IoJ2NoaWxkJykubWFwKGNoaWxkRWxlbWVudCA9PiB7XG4gICAgICAgICAgICAgICAgcmV0dXJuKFxuICAgICAgICAgICAgICAgICAgICBoKCdkaXYnLCB7Y2xhc3M6IFwiY29sLXNtXCIsIGtleTogY2hpbGRFbGVtZW50LmlkfSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgaCgnc3BhbicsIHt9LCBbY2hpbGRFbGVtZW50XSlcbiAgICAgICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgICAgICApO1xuICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtjbGFzczogXCJyb3cgZmxleC1ub3dyYXBcIiwga2V5OiBgJHt0aGlzLnByb3BzLmlkfS1zcGluZS13cmFwcGVyYH0sIGZvcm1hdHRlZENoaWxkcmVuKVxuICAgICAgICAgICAgKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge2tleTogYCR7dGhpcy5wcm9wcy5pZH0tc3BpbmUtd3JhcHBlcmB9LCB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudHNGb3IoJ2NoaWxkJykpXG4gICAgICAgICAgICApO1xuICAgICAgICB9XG4gICAgfVxufVxuXG5leHBvcnQge1N1YnNjcmliZWRTZXF1ZW5jZSwgU3Vic2NyaWJlZFNlcXVlbmNlIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBUYWJsZSBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBUaGlzIGNvbXBvbmVudCBoYXMgMyByZWd1bGFyXG4gKiByZXBsYWNlbWVudHM6XG4gKiAqIGBwYWdlYFxuICogKiBgbGVmdGBcbiAqICogYHJpZ2h0YFxuICogVGhpcyBjb21wb25lbnQgaGFzIDIgZW51bWVyYXRlZFxuICogcmVwbGFjZW1lbnRzOlxuICogKiBgY2hpbGRgXG4gKiAqIGBoZWFkZXJgXG4gKiBOT1RFOiBgY2hpbGRgIGVudW1lcmF0ZWQgcmVwbGFjZW1lbnRzXG4gKiBhcmUgdHdvIGRpbWVuc2lvbmFsIGFycmF5cyFcbiAqL1xuXG4vKipcbiAqIEFib3V0IE5hbWVkIENoaWxkcmVuXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogYGhlYWRlcnNgIChhcnJheSkgLSBBbiBhcnJheSBvZiB0YWJsZSBoZWFkZXIgY2VsbHNcbiAqIGBkYXRhQ2VsbHNgIChhcnJheS1vZi1hcnJheSkgLSBBIDItZGltZW5zaW9uYWwgYXJyYXlcbiAqICAgIHN0cnVjdHVyZXMgYXMgcm93cyBieSBjb2x1bW5zIHRoYXQgY29udGFpbnMgdGhlXG4gKiAgICB0YWJsZSBkYXRhIGNlbGxzXG4gKiBgcGFnZWAgKHNpbmdsZSkgLSBBIGNlbGwgdGhhdCB0ZWxscyB3aGljaCBwYWdlIG9mIHRoZVxuICogICAgIHRhYmxlIHdlIGFyZSBsb29raW5nIGF0XG4gKiBgbGVmdGAgKHNpbmdsZSkgLSBBIGNlbGwgdGhhdCBzaG93cyB0aGUgbnVtYmVyIG9uIHRoZSBsZWZ0XG4gKiBgcmlnaHRgIChzaW5nbGUpIC0gQSBjZWxsIHRoYXQgc2hvdyB0aGUgbnVtYmVyIG9uIHRoZSByaWdodFxuICovXG5jbGFzcyBUYWJsZSBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb250ZXh0IHRvIG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlUm93cyA9IHRoaXMubWFrZVJvd3MuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5tYWtlRmlyc3RSb3cgPSB0aGlzLm1ha2VGaXJzdFJvdy5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLl9tYWtlUm93RWxlbWVudHMgPSB0aGlzLl9tYWtlUm93RWxlbWVudHMuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5fdGhlYWRTdHlsZSA9IHRoaXMuX3RoZWFkU3R5bGUuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5fZ2V0Um93RGlzcGxheUVsZW1lbnRzID0gdGhpcy5fZ2V0Um93RGlzcGxheUVsZW1lbnRzLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybihcbiAgICAgICAgICAgIGgoJ3RhYmxlJywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIlRhYmxlXCIsXG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbCB0YWJsZS1oc2Nyb2xsIHRhYmxlLXNtIHRhYmxlLXN0cmlwZWRcIlxuICAgICAgICAgICAgfSwgW1xuICAgICAgICAgICAgICAgIGgoJ3RoZWFkJywge3N0eWxlOiB0aGlzLl90aGVhZFN0eWxlKCl9LFtcbiAgICAgICAgICAgICAgICAgICAgdGhpcy5tYWtlRmlyc3RSb3coKVxuICAgICAgICAgICAgICAgIF0pLFxuICAgICAgICAgICAgICAgIGgoJ3Rib2R5Jywge30sIHRoaXMubWFrZVJvd3MoKSlcbiAgICAgICAgICAgIF0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgX3RoZWFkU3R5bGUoKXtcbiAgICAgICAgcmV0dXJuIFwiYm9yZGVyLWJvdHRvbTogYmxhY2s7Ym9yZGVyLWJvdHRvbS1zdHlsZTpzb2xpZDtib3JkZXItYm90dG9tLXdpZHRoOnRoaW47XCI7XG4gICAgfVxuXG4gICAgbWFrZUhlYWRlckVsZW1lbnRzKCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRzRm9yKCdoZWFkZXInKS5tYXAoKHJlcGxhY2VtZW50LCBpZHgpID0+IHtcbiAgICAgICAgICAgICAgICByZXR1cm4gaCgndGgnLCB7XG4gICAgICAgICAgICAgICAgICAgIHN0eWxlOiBcInZlcnRpY2FsLWFsaWduOnRvcDtcIixcbiAgICAgICAgICAgICAgICAgICAga2V5OiBgJHt0aGlzLnByb3BzLmlkfS10YWJsZS1oZWFkZXItJHtpZHh9YFxuICAgICAgICAgICAgICAgIH0sIFtyZXBsYWNlbWVudF0pO1xuICAgICAgICAgICAgfSk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZHJlbk5hbWVkKCdoZWFkZXJzJykubWFwKChyZXBsYWNlbWVudCwgaWR4KSA9PiB7XG4gICAgICAgICAgICAgICAgcmV0dXJuIGgoJ3RoJywge1xuICAgICAgICAgICAgICAgICAgICBzdHlsZTogXCJ2ZXJ0aWNhbC1hbGlnbjp0b3A7XCIsXG4gICAgICAgICAgICAgICAgICAgIGtleTogYCR7dGhpcy5wcm9wcy5pZH0tdGFibGUtaGVhZGVyLSR7aWR4fWBcbiAgICAgICAgICAgICAgICB9LCBbcmVwbGFjZW1lbnRdKTtcbiAgICAgICAgICAgIH0pO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgbWFrZVJvd3MoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLl9tYWtlUm93RWxlbWVudHModGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRzRm9yKCdjaGlsZCcpKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLl9tYWtlUm93RWxlbWVudHModGhpcy5yZW5kZXJDaGlsZHJlbk5hbWVkKCdkYXRhQ2VsbHMnKSk7XG4gICAgICAgIH1cbiAgICB9XG5cblxuXG4gICAgX21ha2VSb3dFbGVtZW50cyhlbGVtZW50cyl7XG4gICAgICAgIC8vIE5vdGU6IHJvd3MgYXJlIHRoZSAqZmlyc3QqIGRpbWVuc2lvblxuICAgICAgICAvLyBpbiB0aGUgMi1kaW1lbnNpb25hbCBhcnJheSByZXR1cm5lZFxuICAgICAgICAvLyBieSBnZXR0aW5nIHRoZSBgY2hpbGRgIHJlcGxhY2VtZW50IGVsZW1lbnRzLlxuICAgICAgICByZXR1cm4gZWxlbWVudHMubWFwKChyb3csIHJvd0lkeCkgPT4ge1xuICAgICAgICAgICAgbGV0IGNvbHVtbnMgPSByb3cubWFwKChjaGlsZEVsZW1lbnQsIGNvbElkeCkgPT4ge1xuICAgICAgICAgICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICAgICAgICAgIGgoJ3RkJywge1xuICAgICAgICAgICAgICAgICAgICAgICAga2V5OiBgJHt0aGlzLnByb3BzLmlkfS10ZC0ke3Jvd0lkeH0tJHtjb2xJZHh9YFxuICAgICAgICAgICAgICAgICAgICB9LCBbY2hpbGRFbGVtZW50XSlcbiAgICAgICAgICAgICAgICApO1xuICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICBsZXQgaW5kZXhFbGVtZW50ID0gaCgndGQnLCB7fSwgW2Ake3Jvd0lkeCArIDF9YF0pO1xuICAgICAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgICAgICBoKCd0cicsIHtrZXk6IGAke3RoaXMucHJvcHMuaWR9LXRyLSR7cm93SWR4fWB9LCBbaW5kZXhFbGVtZW50LCAuLi5jb2x1bW5zXSlcbiAgICAgICAgICAgICk7XG4gICAgICAgIH0pO1xuICAgIH1cblxuICAgIG1ha2VGaXJzdFJvdygpe1xuICAgICAgICBsZXQgaGVhZGVyRWxlbWVudHMgPSB0aGlzLm1ha2VIZWFkZXJFbGVtZW50cygpO1xuICAgICAgICByZXR1cm4oXG4gICAgICAgICAgICBoKCd0cicsIHt9LCBbXG4gICAgICAgICAgICAgICAgaCgndGgnLCB7c3R5bGU6IFwidmVydGljYWwtYWxpZ246dG9wO1wifSwgW1xuICAgICAgICAgICAgICAgICAgICBoKCdkaXYnLCB7Y2xhc3M6IFwiY2FyZFwifSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgaCgnZGl2Jywge2NsYXNzOiBcImNhcmQtYm9keSBwLTFcIn0sIFtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAuLi50aGlzLl9nZXRSb3dEaXNwbGF5RWxlbWVudHMoKVxuICAgICAgICAgICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgICAgICBdKSxcbiAgICAgICAgICAgICAgICAuLi5oZWFkZXJFbGVtZW50c1xuICAgICAgICAgICAgXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBfZ2V0Um93RGlzcGxheUVsZW1lbnRzKCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gW1xuICAgICAgICAgICAgICAgIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdsZWZ0JyksXG4gICAgICAgICAgICAgICAgdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ3JpZ2h0JyksXG4gICAgICAgICAgICAgICAgdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ3BhZ2UnKSxcbiAgICAgICAgICAgIF07XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gW1xuICAgICAgICAgICAgICAgIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnbGVmdCcpLFxuICAgICAgICAgICAgICAgIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgncmlnaHQnKSxcbiAgICAgICAgICAgICAgICB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ3BhZ2UnKVxuICAgICAgICAgICAgXTtcbiAgICAgICAgfVxuICAgIH1cbn1cblxuZXhwb3J0IHtUYWJsZSwgVGFibGUgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIFRhYnMgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFkIGEgc2luZ2xlXG4gKiByZWd1bGFyIHJlcGxhY2VtZW50OlxuICogKiBgZGlzcGxheWBcbiAqIFRoaXMgY29tcG9uZW50IGhhcyBhIHNpbmdsZVxuICogZW51bWVyYXRlZCByZXBsYWNlbWVudDpcbiAqICogYGhlYWRlcmBcbiAqL1xuXG4vKipcbiAqIEFib3V0IE5hbWVkIENoaWxkcmVuXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogYGRpc3BsYXlgIChzaW5nbGUpIC0gVGhlIENlbGwgdGhhdCBnZXRzIGRpc3BsYXllZCB3aGVuXG4gKiAgICAgIHRoZSB0YWJzIGFyZSBzaG93aW5nXG4gKiBgaGVhZGVyc2AgKGFycmF5KSAtIEFuIGFycmF5IG9mIGNlbGxzIHRoYXQgc2VydmUgYXNcbiAqICAgICB0aGUgdGFiIGhlYWRlcnNcbiAqL1xuY2xhc3MgVGFicyBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb21wb25lbnQgbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VIZWFkZXJzID0gdGhpcy5tYWtlSGVhZGVycy5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLm1ha2VEaXNwbGF5ID0gdGhpcy5tYWtlRGlzcGxheS5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIlRhYnNcIixcbiAgICAgICAgICAgICAgICBjbGFzczogXCJjb250YWluZXItZmx1aWQgbWItM1wiXG4gICAgICAgICAgICB9LCBbXG4gICAgICAgICAgICAgICAgaCgndWwnLCB7Y2xhc3M6IFwibmF2IG5hdi10YWJzXCIsIHJvbGU6IFwidGFibGlzdFwifSwgdGhpcy5tYWtlSGVhZGVycygpKSxcbiAgICAgICAgICAgICAgICBoKCdkaXYnLCB7Y2xhc3M6IFwidGFiLWNvbnRlbnRcIn0sIFtcbiAgICAgICAgICAgICAgICAgICAgaCgnZGl2Jywge2NsYXNzOiBcInRhYi1wYW5lIGZhZGUgc2hvdyBhY3RpdmVcIiwgcm9sZTogXCJ0YWJwYW5lbFwifSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy5tYWtlRGlzcGxheSgpXG4gICAgICAgICAgICAgICAgICAgIF0pXG4gICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgIF0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgbWFrZURpc3BsYXkoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignZGlzcGxheScpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnZGlzcGxheScpO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgbWFrZUhlYWRlcnMoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudHNGb3IoJ2hlYWRlcicpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGRyZW5OYW1lZCgnaGVhZGVycycpO1xuICAgICAgICB9XG4gICAgfVxufVxuXG5cbmV4cG9ydCB7VGFicywgVGFicyBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogVGV4dCBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuY2xhc3MgVGV4dCBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybihcbiAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICBjbGFzczogXCJjZWxsXCIsXG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgc3R5bGU6IHRoaXMucHJvcHMuZXh0cmFEYXRhLmRpdlN0eWxlLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIlRleHRcIlxuICAgICAgICAgICAgfSwgW3RoaXMucHJvcHMuZXh0cmFEYXRhLnJhd1RleHRdKVxuICAgICAgICApO1xuICAgIH1cbn1cblxuZXhwb3J0IHtUZXh0LCBUZXh0IGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBUcmFjZWJhY2sgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyBhIHNpbmdsZSByZWd1bGFyXG4gKiByZXBhbGNlbWVudDpcbiAqICogYGNoaWxkYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIGB0cmFjZWJhY2tgIChzaW5nbGUpIC0gVGhlIGNlbGwgY29udGFpbmluZyB0aGUgdHJhY2ViYWNrIHRleHRcbiAqL1xuY2xhc3MgIFRyYWNlYmFjayBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb21wb25lbnQgbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VUcmFjZWJhY2sgPSB0aGlzLm1ha2VUcmFjZWJhY2suYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJUcmFjZWJhY2tcIixcbiAgICAgICAgICAgICAgICBjbGFzczogXCJhbGVydCBhbGVydC1wcmltYXJ5XCJcbiAgICAgICAgICAgIH0sIFt0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY2hpbGQnKV0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgbWFrZVRyYWNlYmFjaygpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdjaGlsZCcpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgndHJhY2ViYWNrJyk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cblxuZXhwb3J0IHtUcmFjZWJhY2ssIFRyYWNlYmFjayBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogX05hdlRhYiBDZWxsIENvbXBvbmVudFxuICogTk9URTogVGhpcyBzaG91bGQgcHJvYmFibHkganVzdCBiZVxuICogcm9sbGVkIGludG8gdGhlIE5hdiBjb21wb25lbnQgc29tZWhvdyxcbiAqIG9yIGluY2x1ZGVkIGluIGl0cyBtb2R1bGUgYXMgYSBwcml2YXRlXG4gKiBzdWJjb21wb25lbnQuXG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIG9uZSByZWd1bGFyXG4gKiByZXBsYWNlbWVudDpcbiAqICogYGNoaWxkYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgY2hpbGRgIChzaW5nbGUpIC0gVGhlIGNlbGwgaW5zaWRlIG9mIHRoZSBuYXZpZ2F0aW9uIHRhYlxuICovXG5jbGFzcyBfTmF2VGFiIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICAvLyBCaW5kIGNvbnRleHQgdG8gbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VDaGlsZCA9IHRoaXMubWFrZUNoaWxkLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuY2xpY2tIYW5kbGVyID0gdGhpcy5jbGlja0hhbmRsZXIuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgbGV0IGlubmVyQ2xhc3MgPSBcIm5hdi1saW5rXCI7XG4gICAgICAgIGlmKHRoaXMucHJvcHMuZXh0cmFEYXRhLmlzQWN0aXZlKXtcbiAgICAgICAgICAgIGlubmVyQ2xhc3MgKz0gXCIgYWN0aXZlXCI7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ2xpJywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIGNsYXNzOiBcIm5hdi1pdGVtXCIsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiX05hdlRhYlwiXG4gICAgICAgICAgICB9LCBbXG4gICAgICAgICAgICAgICAgaCgnYScsIHtcbiAgICAgICAgICAgICAgICAgICAgY2xhc3M6IGlubmVyQ2xhc3MsXG4gICAgICAgICAgICAgICAgICAgIHJvbGU6IFwidGFiXCIsXG4gICAgICAgICAgICAgICAgICAgIG9uY2xpY2s6IHRoaXMuY2xpY2tIYW5kbGVyXG4gICAgICAgICAgICAgICAgfSwgW3RoaXMubWFrZUNoaWxkKCldKVxuICAgICAgICAgICAgXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlQ2hpbGQoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY2hpbGQnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ2NoaWxkJyk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBjbGlja0hhbmRsZXIoZXZlbnQpe1xuICAgICAgICBjZWxsU29ja2V0LnNlbmRTdHJpbmcoXG4gICAgICAgICAgICBKU09OLnN0cmluZ2lmeSh0aGlzLnByb3BzLmV4dHJhRGF0YS5jbGlja0RhdGEsIG51bGwsIDQpXG4gICAgICAgICk7XG4gICAgfVxufVxuXG5leHBvcnQge19OYXZUYWIsIF9OYXZUYWIgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIF9QbG90VXBkYXRlciBDZWxsIENvbXBvbmVudFxuICogTk9URTogTGF0ZXIgcmVmYWN0b3JpbmdzIHNob3VsZCByZXN1bHQgaW5cbiAqIHRoaXMgY29tcG9uZW50IGJlY29taW5nIG9ic29sZXRlXG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG5jb25zdCBNQVhfSU5URVJWQUxTID0gMjU7XG5cbmNsYXNzIF9QbG90VXBkYXRlciBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgdGhpcy5ydW5VcGRhdGUgPSB0aGlzLnJ1blVwZGF0ZS5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLmxpc3RlbkZvclBsb3QgPSB0aGlzLmxpc3RlbkZvclBsb3QuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICBjb21wb25lbnREaWRMb2FkKCkge1xuICAgICAgICAvLyBJZiB3ZSBjYW4gZmluZCBhIG1hdGNoaW5nIFBsb3QgZWxlbWVudFxuICAgICAgICAvLyBhdCB0aGlzIHBvaW50LCB3ZSBzaW1wbHkgdXBkYXRlIGl0LlxuICAgICAgICAvLyBPdGhlcndpc2Ugd2UgbmVlZCB0byAnbGlzdGVuJyBmb3Igd2hlblxuICAgICAgICAvLyBpdCBmaW5hbGx5IGNvbWVzIGludG8gdGhlIERPTS5cbiAgICAgICAgbGV0IGluaXRpYWxQbG90RGl2ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoYHBsb3Qke3RoaXMucHJvcHMuZXh0cmFEYXRhLnBsb3RJZH1gKTtcbiAgICAgICAgaWYoaW5pdGlhbFBsb3REaXYpe1xuICAgICAgICAgICAgdGhpcy5ydW5VcGRhdGUoaW5pdGlhbFBsb3REaXYpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgdGhpcy5saXN0ZW5Gb3JQbG90KCk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIGgoJ2RpdicsXG4gICAgICAgICAgICB7XG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbFwiLFxuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIHN0eWxlOiBcImRpc3BsYXk6IG5vbmVcIixcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJfUGxvdFVwZGF0ZXJcIlxuICAgICAgICAgICAgfSwgW10pO1xuICAgIH1cblxuICAgIC8qKlxuICAgICAqIEluIHRoZSBldmVudCB0aGF0IGEgYF9QbG90VXBkYXRlcmAgaGFzIGNvbWVcbiAgICAgKiBvdmVyIHRoZSB3aXJlICpiZWZvcmUqIGl0cyBjb3JyZXNwb25kaW5nXG4gICAgICogUGxvdCBoYXMgY29tZSBvdmVyICh3aGljaCBhcHBlYXJzIHRvIGJlXG4gICAgICogY29tbW9uKSwgd2Ugd2lsbCBzZXQgYW4gaW50ZXJ2YWwgb2YgNTBtc1xuICAgICAqIGFuZCBjaGVjayBmb3IgdGhlIG1hdGNoaW5nIFBsb3QgaW4gdGhlIERPTVxuICAgICAqIE1BWF9JTlRFUlZBTFMgdGltZXMsIG9ubHkgY2FsbGluZyBgcnVuVXBkYXRlYFxuICAgICAqIG9uY2Ugd2UndmUgZm91bmQgYSBtYXRjaC5cbiAgICAgKi9cbiAgICBsaXN0ZW5Gb3JQbG90KCl7XG4gICAgICAgIGxldCBudW1DaGVja3MgPSAwO1xuICAgICAgICBsZXQgcGxvdENoZWNrZXIgPSB3aW5kb3cuc2V0SW50ZXJ2YWwoKCkgPT4ge1xuICAgICAgICAgICAgaWYobnVtQ2hlY2tzID4gTUFYX0lOVEVSVkFMUyl7XG4gICAgICAgICAgICAgICAgd2luZG93LmNsZWFySW50ZXJ2YWwocGxvdENoZWNrZXIpO1xuICAgICAgICAgICAgICAgIGNvbnNvbGUuZXJyb3IoYENvdWxkIG5vdCBmaW5kIG1hdGNoaW5nIFBsb3QgJHt0aGlzLnByb3BzLmV4dHJhRGF0YS5wbG90SWR9IGZvciBfUGxvdFVwZGF0ZXIgJHt0aGlzLnByb3BzLmlkfWApO1xuICAgICAgICAgICAgICAgIHJldHVybjtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIGxldCBwbG90RGl2ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoYHBsb3Qke3RoaXMucHJvcHMuZXh0cmFEYXRhLnBsb3RJZH1gKTtcbiAgICAgICAgICAgIGlmKHBsb3REaXYpe1xuICAgICAgICAgICAgICAgIHRoaXMucnVuVXBkYXRlKHBsb3REaXYpO1xuICAgICAgICAgICAgICAgIHdpbmRvdy5jbGVhckludGVydmFsKHBsb3RDaGVja2VyKTtcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgbnVtQ2hlY2tzICs9IDE7XG4gICAgICAgICAgICB9XG4gICAgICAgIH0sIDUwKTtcbiAgICB9XG5cbiAgICBydW5VcGRhdGUoYURPTUVsZW1lbnQpe1xuICAgICAgICBjb25zb2xlLmxvZyhcIlVwZGF0aW5nIHBsb3RseSBjaGFydC5cIik7XG4gICAgICAgIC8vIFRPRE8gVGhlc2UgYXJlIGdsb2JhbCB2YXIgZGVmaW5lZCBpbiBwYWdlLmh0bWxcbiAgICAgICAgLy8gd2Ugc2hvdWxkIGRvIHNvbWV0aGluZyBhYm91dCB0aGlzLlxuICAgICAgICBpZiAodGhpcy5wcm9wcy5leHRyYURhdGEuZXhjZXB0aW9uT2NjdXJlZCkge1xuICAgICAgICAgICAgY29uc29sZS5sb2coXCJwbG90IGV4Y2VwdGlvbiBvY2N1cmVkXCIpO1xuICAgICAgICAgICAgUGxvdGx5LnB1cmdlKGFET01FbGVtZW50KTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIGxldCBkYXRhID0gdGhpcy5wcm9wcy5leHRyYURhdGEucGxvdERhdGEubWFwKG1hcFBsb3RseURhdGEpO1xuICAgICAgICAgICAgUGxvdGx5LnJlYWN0KGFET01FbGVtZW50LCBkYXRhLCBhRE9NRWxlbWVudC5sYXlvdXQpO1xuICAgICAgICB9XG4gICAgfVxufVxuXG5leHBvcnQge19QbG90VXBkYXRlciwgX1Bsb3RVcGRhdGVyIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBUb29sIGZvciBWYWxpZGF0aW5nIENvbXBvbmVudCBQcm9wZXJ0aWVzXG4gKi9cblxuY29uc3QgcmVwb3J0ID0gKG1lc3NhZ2UsIGVycm9yTW9kZSwgc2lsZW50TW9kZSkgPT4ge1xuICAgIGlmKGVycm9yTW9kZSA9PSB0cnVlICYmIHNpbGVudE1vZGUgPT0gZmFsc2Upe1xuICAgICAgICBjb25zb2xlLmVycm9yKG1lc3NhZ2UpO1xuICAgIH0gZWxzZSBpZihzaWxlbnRNb2RlID09IGZhbHNlKXtcbiAgICAgICAgY29uc29sZS53YXJuKG1lc3NhZ2UpO1xuICAgIH1cbn07XG5cbmNvbnN0IFByb3BUeXBlcyA9IHtcbiAgICBlcnJvck1vZGU6IGZhbHNlLFxuICAgIHNpbGVudE1vZGU6IGZhbHNlLFxuICAgIG9uZU9mOiBmdW5jdGlvbihhbkFycmF5KXtcbiAgICAgICAgcmV0dXJuIGZ1bmN0aW9uKGNvbXBvbmVudE5hbWUsIHByb3BOYW1lLCBwcm9wVmFsdWUsIGlzUmVxdWlyZWQpe1xuICAgICAgICAgICAgZm9yKGxldCBpID0gMDsgaSA8IGFuQXJyYXkubGVuZ3RoOyBpKyspe1xuICAgICAgICAgICAgICAgIGxldCB0eXBlQ2hlY2tJdGVtID0gYW5BcnJheVtpXTtcbiAgICAgICAgICAgICAgICBpZih0eXBlb2YodHlwZUNoZWNrSXRlbSkgPT0gJ2Z1bmN0aW9uJyl7XG4gICAgICAgICAgICAgICAgICAgIGlmKHR5cGVDaGVja0l0ZW0oY29tcG9uZW50TmFtZSwgcHJvcE5hbWUsIHByb3BWYWx1ZSwgaXNSZXF1aXJlZCwgdHJ1ZSkpe1xuICAgICAgICAgICAgICAgICAgICAgICAgcmV0dXJuIHRydWU7XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB9IGVsc2UgaWYodHlwZUNoZWNrSXRlbSA9PSBwcm9wVmFsdWUpe1xuICAgICAgICAgICAgICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBsZXQgbWVzc2FnZSA9IGAke2NvbXBvbmVudE5hbWV9ID4+ICR7cHJvcE5hbWV9IG11c3QgYmUgb2Ygb25lIG9mIHRoZSBmb2xsb3dpbmcgdHlwZXM6ICR7YW5BcnJheX1gO1xuICAgICAgICAgICAgcmVwb3J0KG1lc3NhZ2UsIHRoaXMuZXJyb3JNb2RlLCB0aGlzLnNpbGVudE1vZGUpO1xuICAgICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICB9LmJpbmQodGhpcyk7XG4gICAgfSxcblxuICAgIGdldFZhbGlkYXRvckZvclR5cGUodHlwZVN0cil7XG4gICAgICAgIHJldHVybiBmdW5jdGlvbihjb21wb25lbnROYW1lLCBwcm9wTmFtZSwgcHJvcFZhbHVlLCBpc1JlcXVpcmVkLCBpbkNvbXBvdW5kID0gZmFsc2Upe1xuICAgICAgICAgICAgLy8gV2UgYXJlICdpbiBhIGNvbXBvdW5kIHZhbGlkYXRpb24nIHdoZW4gdGhlIGluZGl2aWR1YWxcbiAgICAgICAgICAgIC8vIFByb3BUeXBlIGNoZWNrZXJzIChpZSBmdW5jLCBudW1iZXIsIHN0cmluZywgZXRjKSBhcmVcbiAgICAgICAgICAgIC8vIGJlaW5nIGNhbGxlZCB3aXRoaW4gYSBjb21wb3VuZCB0eXBlIGNoZWNrZXIgbGlrZSBvbmVPZi5cbiAgICAgICAgICAgIC8vIEluIHRoZXNlIGNhc2VzIHdlIHdhbnQgdG8gcHJldmVudCB0aGUgY2FsbCB0byByZXBvcnQsXG4gICAgICAgICAgICAvLyB3aGljaCB0aGUgY29tcG91bmQgY2hlY2sgd2lsbCBoYW5kbGUgb24gaXRzIG93bi5cbiAgICAgICAgICAgIGlmKGluQ29tcG91bmQgPT0gZmFsc2Upe1xuICAgICAgICAgICAgICAgIGlmKHR5cGVvZihwcm9wVmFsdWUpID09IHR5cGVTdHIpe1xuICAgICAgICAgICAgICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgICAgICAgICAgICB9IGVsc2UgaWYoIWlzUmVxdWlyZWQgJiYgKHByb3BWYWx1ZSA9PSB1bmRlZmluZWQgfHwgcHJvcFZhbHVlID09IG51bGwpKXtcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIHRydWU7XG4gICAgICAgICAgICAgICAgfSBlbHNlIGlmKGlzUmVxdWlyZWQpe1xuICAgICAgICAgICAgICAgICAgICBsZXQgbWVzc2FnZSA9IGAke2NvbXBvbmVudE5hbWV9ID4+ICR7cHJvcE5hbWV9IGlzIGEgcmVxdWlyZWQgcHJvcCwgYnV0IGFzIHBhc3NlZCBhcyAke3Byb3BWYWx1ZX0hYDtcbiAgICAgICAgICAgICAgICAgICAgcmVwb3J0KG1lc3NhZ2UsIHRoaXMuZXJyb3JNb2RlLCB0aGlzLnNpbGVudE1vZGUpO1xuICAgICAgICAgICAgICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICAgICAgbGV0IG1lc3NhZ2UgPSBgJHtjb21wb25lbnROYW1lfSA+PiAke3Byb3BOYW1lfSBtdXN0IGJlIG9mIHR5cGUgJHt0eXBlU3RyfSFgO1xuICAgICAgICAgICAgICAgICAgICByZXBvcnQobWVzc2FnZSwgdGhpcy5lcnJvck1vZGUsIHRoaXMuc2lsZW50TW9kZSk7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAvLyBPdGhlcndpc2UgdGhpcyBpcyBhIHN0cmFpZ2h0Zm9yd2FyZCB0eXBlIGNoZWNrXG4gICAgICAgICAgICAvLyBiYXNlZCBvbiB0aGUgZ2l2ZW4gdHlwZS4gV2UgY2hlY2sgYXMgdXN1YWwgZm9yIHRoZSByZXF1aXJlZFxuICAgICAgICAgICAgLy8gcHJvcGVydHkgYW5kIHRoZW4gdGhlIGFjdHVhbCB0eXBlIG1hdGNoIGlmIG5lZWRlZC5cbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgaWYoaXNSZXF1aXJlZCAmJiAocHJvcFZhbHVlID09IHVuZGVmaW5lZCB8fCBwcm9wVmFsdWUgPT0gbnVsbCkpe1xuICAgICAgICAgICAgICAgICAgICBsZXQgbWVzc2FnZSA9IGAke2NvbXBvbmVudE5hbWV9ID4+ICR7cHJvcE5hbWV9IGlzIGEgcmVxdWlyZWQgcHJvcCwgYnV0IHdhcyBwYXNzZWQgYXMgJHtwcm9wVmFsdWV9IWA7XG4gICAgICAgICAgICAgICAgICAgIHJlcG9ydChtZXNzYWdlLCB0aGlzLmVycm9yTW9kZSwgdGhpcy5zaWxlbnRNb2RlKTtcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICAgICAgICAgIH0gZWxzZSBpZighaXNSZXF1aXJlZCAmJiAocHJvcFZhbHVlID09IHVuZGVmaW5lZCB8fCBwcm9wVmFsdWUgPT0gbnVsbCkpe1xuICAgICAgICAgICAgICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgcmV0dXJuIHR5cGVvZihwcm9wVmFsdWUpID09IHR5cGVTdHI7XG4gICAgICAgICAgICB9XG4gICAgICAgIH07XG4gICAgfSxcblxuICAgIGdldCBudW1iZXIoKXtcbiAgICAgICAgcmV0dXJuIHRoaXMuZ2V0VmFsaWRhdG9yRm9yVHlwZSgnbnVtYmVyJykuYmluZCh0aGlzKTtcbiAgICB9LFxuXG4gICAgZ2V0IGJvb2xlYW4oKXtcbiAgICAgICAgcmV0dXJuIHRoaXMuZ2V0VmFsaWRhdG9yRm9yVHlwZSgnYm9vbGVhbicpLmJpbmQodGhpcyk7XG4gICAgfSxcblxuICAgIGdldCBzdHJpbmcoKXtcbiAgICAgICAgcmV0dXJuIHRoaXMuZ2V0VmFsaWRhdG9yRm9yVHlwZSgnc3RyaW5nJykuYmluZCh0aGlzKTtcbiAgICB9LFxuXG4gICAgZ2V0IG9iamVjdCgpe1xuICAgICAgICByZXR1cm4gdGhpcy5nZXRWYWxpZGF0b3JGb3JUeXBlKCdvYmplY3QnKS5iaW5kKHRoaXMpO1xuICAgIH0sXG5cbiAgICBnZXQgZnVuYygpe1xuICAgICAgICByZXR1cm4gdGhpcy5nZXRWYWxpZGF0b3JGb3JUeXBlKCdmdW5jdGlvbicpLmJpbmQodGhpcyk7XG4gICAgfSxcblxuICAgIHZhbGlkYXRlOiBmdW5jdGlvbihjb21wb25lbnROYW1lLCBwcm9wcywgcHJvcEluZm8pe1xuICAgICAgICBsZXQgcHJvcE5hbWVzID0gbmV3IFNldChPYmplY3Qua2V5cyhwcm9wcykpO1xuICAgICAgICBwcm9wTmFtZXMuZGVsZXRlKCdjaGlsZHJlbicpO1xuICAgICAgICBwcm9wTmFtZXMuZGVsZXRlKCduYW1lZENoaWxkcmVuJyk7XG4gICAgICAgIHByb3BOYW1lcy5kZWxldGUoJ2lkJyk7XG4gICAgICAgIHByb3BOYW1lcy5kZWxldGUoJ2V4dHJhRGF0YScpOyAvLyBGb3Igbm93XG4gICAgICAgIGxldCBwcm9wc1RvVmFsaWRhdGUgPSBBcnJheS5mcm9tKHByb3BOYW1lcyk7XG5cbiAgICAgICAgLy8gUGVyZm9ybSBhbGwgdGhlIHZhbGlkYXRpb25zIG9uIGVhY2ggcHJvcGVydHlcbiAgICAgICAgLy8gYWNjb3JkaW5nIHRvIGl0cyBkZXNjcmlwdGlvbi4gV2Ugc3RvcmUgd2hldGhlclxuICAgICAgICAvLyBvciBub3QgdGhlIGdpdmVuIHByb3BlcnR5IHdhcyBjb21wbGV0ZWx5IHZhbGlkXG4gICAgICAgIC8vIGFuZCB0aGVuIGV2YWx1YXRlIHRoZSB2YWxpZGl0eSBvZiBhbGwgYXQgdGhlIGVuZC5cbiAgICAgICAgbGV0IHZhbGlkYXRpb25SZXN1bHRzID0ge307XG4gICAgICAgIHByb3BzVG9WYWxpZGF0ZS5mb3JFYWNoKHByb3BOYW1lID0+IHtcbiAgICAgICAgICAgIGxldCBwcm9wVmFsID0gcHJvcHNbcHJvcE5hbWVdO1xuICAgICAgICAgICAgbGV0IHZhbGlkYXRpb25Ub0NoZWNrID0gcHJvcEluZm9bcHJvcE5hbWVdO1xuICAgICAgICAgICAgaWYodmFsaWRhdGlvblRvQ2hlY2spe1xuICAgICAgICAgICAgICAgIGxldCBoYXNWYWxpZERlc2NyaXB0aW9uID0gdGhpcy52YWxpZGF0ZURlc2NyaXB0aW9uKGNvbXBvbmVudE5hbWUsIHByb3BOYW1lLCB2YWxpZGF0aW9uVG9DaGVjayk7XG4gICAgICAgICAgICAgICAgbGV0IGhhc1ZhbGlkUHJvcFR5cGVzID0gdmFsaWRhdGlvblRvQ2hlY2sudHlwZShjb21wb25lbnROYW1lLCBwcm9wTmFtZSwgcHJvcFZhbCwgdmFsaWRhdGlvblRvQ2hlY2sucmVxdWlyZWQpO1xuICAgICAgICAgICAgICAgIGlmKGhhc1ZhbGlkRGVzY3JpcHRpb24gJiYgaGFzVmFsaWRQcm9wVHlwZXMpe1xuICAgICAgICAgICAgICAgICAgICB2YWxpZGF0aW9uUmVzdWx0c1twcm9wTmFtZV0gPSB0cnVlO1xuICAgICAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgICAgIHZhbGlkYXRpb25SZXN1bHRzW3Byb3BOYW1lXSA9IGZhbHNlO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgLy8gSWYgd2UgZ2V0IGhlcmUsIHRoZSBjb25zdW1lciBoYXMgcGFzc2VkIGluIGEgcHJvcFxuICAgICAgICAgICAgICAgIC8vIHRoYXQgaXMgbm90IHByZXNlbnQgaW4gdGhlIHByb3BUeXBlcyBkZXNjcmlwdGlvbi5cbiAgICAgICAgICAgICAgICAvLyBXZSByZXBvcnQgdG8gdGhlIGNvbnNvbGUgYXMgbmVlZGVkIGFuZCB2YWxpZGF0ZSBhcyBmYWxzZS5cbiAgICAgICAgICAgICAgICBsZXQgbWVzc2FnZSA9IGAke2NvbXBvbmVudE5hbWV9IGhhcyBhIHByb3AgY2FsbGVkIFwiJHtwcm9wTmFtZX1cIiB0aGF0IGlzIG5vdCBkZXNjcmliZWQgaW4gcHJvcFR5cGVzIWA7XG4gICAgICAgICAgICAgICAgcmVwb3J0KG1lc3NhZ2UsIHRoaXMuZXJyb3JNb2RlLCB0aGlzLnNpbGVudE1vZGUpO1xuICAgICAgICAgICAgICAgIHZhbGlkYXRpb25SZXN1bHRzW3Byb3BOYW1lXSA9IGZhbHNlO1xuICAgICAgICAgICAgfVxuICAgICAgICB9KTtcblxuICAgICAgICAvLyBJZiB0aGVyZSB3ZXJlIGFueSB0aGF0IGRpZCBub3QgdmFsaWRhdGUsIHJldHVyblxuICAgICAgICAvLyBmYWxzZSBhbmQgcmVwb3J0IGFzIG11Y2guXG4gICAgICAgIGxldCBpbnZhbGlkcyA9IFtdO1xuICAgICAgICBPYmplY3Qua2V5cyh2YWxpZGF0aW9uUmVzdWx0cykuZm9yRWFjaChrZXkgPT4ge1xuICAgICAgICAgICAgaWYodmFsaWRhdGlvblJlc3VsdHNba2V5XSA9PSBmYWxzZSl7XG4gICAgICAgICAgICAgICAgaW52YWxpZHMucHVzaChrZXkpO1xuICAgICAgICAgICAgfVxuICAgICAgICB9KTtcbiAgICAgICAgaWYoaW52YWxpZHMubGVuZ3RoID4gMCl7XG4gICAgICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgICAgfVxuICAgIH0sXG5cbiAgICB2YWxpZGF0ZVJlcXVpcmVkOiBmdW5jdGlvbihjb21wb25lbnROYW1lLCBwcm9wTmFtZSwgcHJvcFZhbCwgaXNSZXF1aXJlZCl7XG4gICAgICAgIGlmKGlzUmVxdWlyZWQgPT0gdHJ1ZSl7XG4gICAgICAgICAgICBpZihwcm9wVmFsID09IG51bGwgfHwgcHJvcFZhbCA9PSB1bmRlZmluZWQpe1xuICAgICAgICAgICAgICAgIGxldCBtZXNzYWdlID0gYCR7Y29tcG9uZW50TmFtZX0gPj4gJHtwcm9wTmFtZX0gcmVxdWlyZXMgYSB2YWx1ZSwgYnV0ICR7cHJvcFZhbH0gd2FzIHBhc3NlZCFgO1xuICAgICAgICAgICAgICAgIHJlcG9ydChtZXNzYWdlLCB0aGlzLmVycm9yTW9kZSwgdGhpcy5zaWxlbnRNb2RlKTtcbiAgICAgICAgICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIHRydWU7XG4gICAgfSxcblxuICAgIHZhbGlkYXRlRGVzY3JpcHRpb246IGZ1bmN0aW9uKGNvbXBvbmVudE5hbWUsIHByb3BOYW1lLCBwcm9wKXtcbiAgICAgICAgbGV0IGRlc2MgPSBwcm9wLmRlc2NyaXB0aW9uO1xuICAgICAgICBpZihkZXNjID09IHVuZGVmaW5lZCB8fCBkZXNjID09IFwiXCIgfHwgZGVzYyA9PSBudWxsKXtcbiAgICAgICAgICAgIGxldCBtZXNzYWdlID0gYCR7Y29tcG9uZW50TmFtZX0gPj4gJHtwcm9wTmFtZX0gaGFzIGFuIGVtcHR5IGRlc2NyaXB0aW9uIWA7XG4gICAgICAgICAgICByZXBvcnQobWVzc2FnZSwgdGhpcy5lcnJvck1vZGUsIHRoaXMuc2lsZW50TW9kZSk7XG4gICAgICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIHRydWU7XG4gICAgfVxufTtcblxuZXhwb3J0IHtcbiAgICBQcm9wVHlwZXNcbn07XG5cblxuLyoqKlxubnVtYmVyOiBmdW5jdGlvbihjb21wb25lbnROYW1lLCBwcm9wTmFtZSwgcHJvcFZhbHVlLCBpbkNvbXBvdW5kID0gZmFsc2Upe1xuICAgICAgICBpZihpbkNvbXBvdW5kID09IGZhbHNlKXtcbiAgICAgICAgICAgIGlmKHR5cGVvZihwcm9wVmFsdWUpID09ICdudW1iZXInKXtcbiAgICAgICAgICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgbGV0IG1lc3NhZ2UgPSBgJHtjb21wb25lbnROYW1lfSA+PiAke3Byb3BOYW1lfSBtdXN0IGJlIG9mIHR5cGUgbnVtYmVyIWA7XG4gICAgICAgICAgICAgICAgcmVwb3J0KG1lc3NhZ2UsIHRoaXMuZXJyb3JNb2RlLCB0aGlzLnNpbGVudE1vZGUpO1xuICAgICAgICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0eXBlb2YocHJvcFZhbHVlKSA9PSAnbnVtYmVyJztcbiAgICAgICAgfVxuXG4gICAgfS5iaW5kKHRoaXMpLFxuXG4gICAgc3RyaW5nOiBmdW5jdGlvbihjb21wb25lbnROYW1lLCBwcm9wTmFtZSwgcHJvcFZhbHVlLCBpbkNvbXBvdW5kID0gZmFsc2Upe1xuICAgICAgICBpZihpbkNvbXBvdW5kID09IGZhbHNlKXtcbiAgICAgICAgICAgIGlmKHR5cGVvZihwcm9wVmFsdWUpID09ICdzdHJpbmcnKXtcbiAgICAgICAgICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgbGV0IG1lc3NhZ2UgPSBgJHtjb21wb25lbnROYW1lfSA+PiAke3Byb3BOYW1lfSBtdXN0IGJlIG9mIHR5cGUgc3RyaW5nIWA7XG4gICAgICAgICAgICAgICAgcmVwb3J0KG1lc3NhZ2UsIHRoaXMuZXJyb3JNb2RlLCB0aGlzLnNpbGVudE1vZGUpO1xuICAgICAgICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0eXBlb2YocHJvcFZhbHVlKSA9PSAnc3RyaW5nJztcbiAgICAgICAgfVxuICAgIH0uYmluZCh0aGlzKSxcblxuICAgIGJvb2xlYW46IGZ1bmN0aW9uKGNvbXBvbmVudE5hbWUsIHByb3BOYW1lLCBwcm9wVmFsdWUsIGluQ29tcG91bmQgPSBmYWxzZSl7XG4gICAgICAgIGlmKGluQ29tcG91bmQgPT0gZmFsc2Upe1xuICAgICAgICAgICAgaWYodHlwZW9mKHByb3BWYWx1ZSkgPT0gJ2Jvb2xlYW4nKXtcbiAgICAgICAgICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgbGV0IG1lc3NhZ2UgPSBgJHtjb21wb25lbnROYW1lfSA+PiAke3Byb3BOYW1lfSBtdXN0IGJlIG9mIHR5cGUgYm9vbGVhbiFgO1xuICAgICAgICAgICAgICAgIHJlcG9ydChtZXNzYWdlLCB0aGlzLmVycm9yTW9kZSwgdGhpcy5zaWxlbnRNb2RlKTtcbiAgICAgICAgICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICAgICAgICB9XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdHlwZW9mKHByb3BWYWx1ZSkgPT0gJ2Jvb2xlYW4nO1xuICAgICAgICB9XG4gICAgfS5iaW5kKHRoaXMpLFxuXG4gICAgb2JqZWN0OiBmdW5jdGlvbihjb21wb25lbnROYW1lLCBwcm9wTmFtZSwgcHJvcFZhbHVlLCBpbkNvbXBvdW5kID0gZmFsc2Upe1xuICAgICAgICBpZihpbkNvbXBvdW5kID09IGZhbHNlKXtcbiAgICAgICAgICAgIGlmKHR5cGVvZihwcm9wVmFsdWUpID09ICdvYmplY3QnKXtcbiAgICAgICAgICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgbGV0IG1lc3NhZ2UgPSBgJHtjb21wb25lbnROYW1lfSA+PiAke3Byb3BOYW1lfSBtdXN0IGJlIG9mIHR5cGUgb2JqZWN0IWA7XG4gICAgICAgICAgICAgICAgcmVwb3J0KG1lc3NhZ2UsIHRoaXMuZXJyb3JNb2RlLCB0aGlzLnNpbGVudE1vZGUpO1xuICAgICAgICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0eXBlb2YocHJvcFZhbHVlKSA9PSAnb2JqZWN0JztcbiAgICAgICAgfVxuICAgIH0uYmluZCh0aGlzKSxcblxuICAgIGZ1bmM6IGZ1bmN0aW9uKGNvbXBvbmVudE5hbWUsIHByb3BOYW1lLCBwcm9wVmFsdWUsIGluQ29tcG91bmQgPSBmYWxzZSl7XG4gICAgICAgIGlmKGluQ29tcG91bmQgPT0gZmFsc2Upe1xuICAgICAgICAgICAgaWYodHlwZW9mKHByb3BWYWx1ZSkgPT0gJ2Z1bmN0aW9uJyl7XG4gICAgICAgICAgICAgICAgcmV0dXJuIHRydWU7XG4gICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgIGxldCBtZXNzYWdlID0gYCR7Y29tcG9uZW50TmFtZX0gPj4gJHtwcm9wTmFtZX0gbXVzdCBiZSBvZiB0eXBlIGZ1bmN0aW9uIWA7XG4gICAgICAgICAgICAgICAgcmVwb3J0KG1lc3NhZ2UsIHRoaXMuZXJyb3JNb2RlLCB0aGlzLnNpbGVudE1vZGUpO1xuICAgICAgICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0eXBlb2YocHJvcFZhbHVlKSA9PSAnZnVuY3Rpb24nO1xuICAgICAgICB9XG4gICAgfS5iaW5kKHRoaXMpLFxuXG4qKiovXG4iLCJjbGFzcyBSZXBsYWNlbWVudHNIYW5kbGVyIHtcbiAgICBjb25zdHJ1Y3RvcihyZXBsYWNlbWVudHMpe1xuICAgICAgICB0aGlzLnJlcGxhY2VtZW50cyA9IHJlcGxhY2VtZW50cztcbiAgICAgICAgdGhpcy5yZWd1bGFyID0ge307XG4gICAgICAgIHRoaXMuZW51bWVyYXRlZCA9IHt9O1xuXG4gICAgICAgIGlmKHJlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICB0aGlzLnByb2Nlc3NSZXBsYWNlbWVudHMoKTtcbiAgICAgICAgfVxuXG4gICAgICAgIC8vIEJpbmQgY29udGV4dCB0byBtZXRob2RzXG4gICAgICAgIHRoaXMucHJvY2Vzc1JlcGxhY2VtZW50cyA9IHRoaXMucHJvY2Vzc1JlcGxhY2VtZW50cy5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLnByb2Nlc3NSZWd1bGFyID0gdGhpcy5wcm9jZXNzUmVndWxhci5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLmhhc1JlcGxhY2VtZW50ID0gdGhpcy5oYXNSZXBsYWNlbWVudC5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLmdldFJlcGxhY2VtZW50Rm9yID0gdGhpcy5nZXRSZXBsYWNlbWVudEZvci5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLmdldFJlcGxhY2VtZW50c0ZvciA9IHRoaXMuZ2V0UmVwbGFjZW1lbnRzRm9yLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMubWFwUmVwbGFjZW1lbnRzRm9yID0gdGhpcy5tYXBSZXBsYWNlbWVudHNGb3IuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICBwcm9jZXNzUmVwbGFjZW1lbnRzKCl7XG4gICAgICAgIHRoaXMucmVwbGFjZW1lbnRzLmZvckVhY2gocmVwbGFjZW1lbnQgPT4ge1xuICAgICAgICAgICAgbGV0IHJlcGxhY2VtZW50SW5mbyA9IHRoaXMuY29uc3RydWN0b3IucmVhZFJlcGxhY2VtZW50U3RyaW5nKHJlcGxhY2VtZW50KTtcbiAgICAgICAgICAgIGlmKHJlcGxhY2VtZW50SW5mby5pc0VudW1lcmF0ZWQpe1xuICAgICAgICAgICAgICAgIHRoaXMucHJvY2Vzc0VudW1lcmF0ZWQocmVwbGFjZW1lbnQsIHJlcGxhY2VtZW50SW5mbyk7XG4gICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgIHRoaXMucHJvY2Vzc1JlZ3VsYXIocmVwbGFjZW1lbnQsIHJlcGxhY2VtZW50SW5mbyk7XG4gICAgICAgICAgICB9XG4gICAgICAgIH0pO1xuICAgICAgICAvLyBOb3cgd2UgdXBkYXRlIHRoaXMuZW51bWVyYXRlZCB0byBoYXZlIGl0J3MgdG9wIGxldmVsXG4gICAgICAgIC8vIHZhbHVlcyBhcyBBcnJheXMgaW5zdGVhZCBvZiBuZXN0ZWQgZGljdHMgYW5kIHdlIHNvcnRcbiAgICAgICAgLy8gYmFzZWQgb24gdGhlIGV4dHJhY3RlZCBpbmRpY2VzICh3aGljaCBhcmUgYXQgdGhpcyBwb2ludFxuICAgICAgICAvLyBqdXN0IGtleXMgb24gc3ViZGljdHMgb3IgbXVsdGlkaW1lbnNpb25hbCBkaWN0cylcbiAgICAgICAgT2JqZWN0LmtleXModGhpcy5lbnVtZXJhdGVkKS5mb3JFYWNoKGtleSA9PiB7XG4gICAgICAgICAgICBsZXQgZW51bWVyYXRlZFJlcGxhY2VtZW50cyA9IHRoaXMuZW51bWVyYXRlZFtrZXldO1xuICAgICAgICAgICAgdGhpcy5lbnVtZXJhdGVkW2tleV0gPSB0aGlzLmNvbnN0cnVjdG9yLmVudW1lcmF0ZWRWYWxUb1NvcnRlZEFycmF5KGVudW1lcmF0ZWRSZXBsYWNlbWVudHMpO1xuICAgICAgICB9KTtcbiAgICB9XG5cbiAgICBwcm9jZXNzUmVndWxhcihyZXBsYWNlbWVudE5hbWUsIHJlcGxhY2VtZW50SW5mbyl7XG4gICAgICAgIGxldCByZXBsYWNlbWVudEtleSA9IHRoaXMuY29uc3RydWN0b3Iua2V5RnJvbU5hbWVQYXJ0cyhyZXBsYWNlbWVudEluZm8ubmFtZVBhcnRzKTtcbiAgICAgICAgdGhpcy5yZWd1bGFyW3JlcGxhY2VtZW50S2V5XSA9IHJlcGxhY2VtZW50TmFtZTtcbiAgICB9XG5cbiAgICBwcm9jZXNzRW51bWVyYXRlZChyZXBsYWNlbWVudE5hbWUsIHJlcGxhY2VtZW50SW5mbyl7XG4gICAgICAgIGxldCByZXBsYWNlbWVudEtleSA9IHRoaXMuY29uc3RydWN0b3Iua2V5RnJvbU5hbWVQYXJ0cyhyZXBsYWNlbWVudEluZm8ubmFtZVBhcnRzKTtcbiAgICAgICAgbGV0IGN1cnJlbnRFbnRyeSA9IHRoaXMuZW51bWVyYXRlZFtyZXBsYWNlbWVudEtleV07XG5cbiAgICAgICAgLy8gSWYgaXQncyB1bmRlZmluZWQsIHRoaXMgaXMgdGhlIGZpcnN0XG4gICAgICAgIC8vIG9mIHRoZSBlbnVtZXJhdGVkIHJlcGxhY2VtZW50cyBmb3IgdGhpc1xuICAgICAgICAvLyBrZXksIGllIHNvbWV0aGluZyBsaWtlIF9fX19jaGlsZF8wX19cbiAgICAgICAgaWYoY3VycmVudEVudHJ5ID09IHVuZGVmaW5lZCl7XG4gICAgICAgICAgICB0aGlzLmVudW1lcmF0ZWRbcmVwbGFjZW1lbnRLZXldID0ge307XG4gICAgICAgICAgICBjdXJyZW50RW50cnkgPSB0aGlzLmVudW1lcmF0ZWRbcmVwbGFjZW1lbnRLZXldO1xuICAgICAgICB9XG5cbiAgICAgICAgLy8gV2UgYWRkIHRoZSBlbnVtZXJhdGVkIGluZGljZXMgYXMga2V5cyB0byBhIGRpY3RcbiAgICAgICAgLy8gYW5kIHdlIGRvIHRoaXMgcmVjdXJzaXZlbHkgYWNyb3NzIGRpbWVuc2lvbnMgYXNcbiAgICAgICAgLy8gbmVlZGVkLlxuICAgICAgICB0aGlzLmNvbnN0cnVjdG9yLnByb2Nlc3NEaW1lbnNpb24ocmVwbGFjZW1lbnRJbmZvLmVudW1lcmF0ZWRWYWx1ZXMsIGN1cnJlbnRFbnRyeSwgcmVwbGFjZW1lbnROYW1lKTtcbiAgICB9XG5cbiAgICAvLyBBY2Nlc3NpbmcgYW5kIG90aGVyIENvbnZlbmllbmNlIE1ldGhvZHNcbiAgICBoYXNSZXBsYWNlbWVudChhUmVwbGFjZW1lbnROYW1lKXtcbiAgICAgICAgaWYodGhpcy5yZWd1bGFyLmhhc093blByb3BlcnR5KGFSZXBsYWNlbWVudE5hbWUpKXtcbiAgICAgICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgICB9IGVsc2UgaWYodGhpcy5lbnVtZXJhdGVkLmhhc093blByb3BlcnR5KGFSZXBsYWNlbWVudE5hbWUpKXtcbiAgICAgICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiBmYWxzZTtcbiAgICB9XG5cbiAgICBnZXRSZXBsYWNlbWVudEZvcihhUmVwbGFjZW1lbnROYW1lKXtcbiAgICAgICAgbGV0IGZvdW5kID0gdGhpcy5yZWd1bGFyW2FSZXBsYWNlbWVudE5hbWVdO1xuICAgICAgICBpZihmb3VuZCA9PSB1bmRlZmluZWQpe1xuICAgICAgICAgICAgcmV0dXJuIG51bGw7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIGZvdW5kO1xuICAgIH1cblxuICAgIGdldFJlcGxhY2VtZW50c0ZvcihhUmVwbGFjZW1lbnROYW1lKXtcbiAgICAgICAgbGV0IGZvdW5kID0gdGhpcy5lbnVtZXJhdGVkW2FSZXBsYWNlbWVudE5hbWVdO1xuICAgICAgICBpZihmb3VuZCA9PSB1bmRlZmluZWQpe1xuICAgICAgICAgICAgcmV0dXJuIG51bGw7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIGZvdW5kO1xuICAgIH1cblxuICAgIG1hcFJlcGxhY2VtZW50c0ZvcihhUmVwbGFjZW1lbnROYW1lLCBtYXBGdW5jdGlvbil7XG4gICAgICAgIGlmKCF0aGlzLmhhc1JlcGxhY2VtZW50KGFSZXBsYWNlbWVudE5hbWUpKXtcbiAgICAgICAgICAgIHRocm93IG5ldyBFcnJvcihgSW52YWxpZCByZXBsYWNlbWVudCBuYW1lOiAke2FSZXBsYWNlbWVudG5hbWV9YCk7XG4gICAgICAgIH1cbiAgICAgICAgbGV0IHJvb3QgPSB0aGlzLmdldFJlcGxhY2VtZW50c0ZvcihhUmVwbGFjZW1lbnROYW1lKTtcbiAgICAgICAgcmV0dXJuIHRoaXMuX3JlY3Vyc2l2ZWx5TWFwKHJvb3QsIG1hcEZ1bmN0aW9uKTtcbiAgICB9XG5cbiAgICBfcmVjdXJzaXZlbHlNYXAoY3VycmVudEl0ZW0sIG1hcEZ1bmN0aW9uKXtcbiAgICAgICAgaWYoIUFycmF5LmlzQXJyYXkoY3VycmVudEl0ZW0pKXtcbiAgICAgICAgICAgIHJldHVybiBtYXBGdW5jdGlvbihjdXJyZW50SXRlbSk7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIGN1cnJlbnRJdGVtLm1hcChzdWJJdGVtID0+IHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLl9yZWN1cnNpdmVseU1hcChzdWJJdGVtLCBtYXBGdW5jdGlvbik7XG4gICAgICAgIH0pO1xuICAgIH1cblxuICAgIC8vIFN0YXRpYyBoZWxwZXJzXG4gICAgc3RhdGljIHByb2Nlc3NEaW1lbnNpb24ocmVtYWluaW5nVmFscywgaW5EaWN0LCByZXBsYWNlbWVudE5hbWUpe1xuICAgICAgICBpZihyZW1haW5pbmdWYWxzLmxlbmd0aCA9PSAxKXtcbiAgICAgICAgICAgIGluRGljdFtyZW1haW5pbmdWYWxzWzBdXSA9IHJlcGxhY2VtZW50TmFtZTtcbiAgICAgICAgICAgIHJldHVybjtcbiAgICAgICAgfVxuICAgICAgICBsZXQgbmV4dEtleSA9IHJlbWFpbmluZ1ZhbHNbMF07XG4gICAgICAgIGxldCBuZXh0RGljdCA9IGluRGljdFtuZXh0S2V5XTtcbiAgICAgICAgaWYobmV4dERpY3QgPT0gdW5kZWZpbmVkKXtcbiAgICAgICAgICAgIGluRGljdFtuZXh0S2V5XSA9IHt9O1xuICAgICAgICAgICAgbmV4dERpY3QgPSBpbkRpY3RbbmV4dEtleV07XG4gICAgICAgIH1cbiAgICAgICAgdGhpcy5wcm9jZXNzRGltZW5zaW9uKHJlbWFpbmluZ1ZhbHMuc2xpY2UoMSksIG5leHREaWN0LCByZXBsYWNlbWVudE5hbWUpO1xuICAgIH1cblxuICAgIHN0YXRpYyBlbnVtZXJhdGVkVmFsVG9Tb3J0ZWRBcnJheShhRGljdCwgYWNjdW11bGF0ZSA9IFtdKXtcbiAgICAgICAgaWYodHlwZW9mIGFEaWN0ICE9PSAnb2JqZWN0Jyl7XG4gICAgICAgICAgICByZXR1cm4gYURpY3Q7XG4gICAgICAgIH1cbiAgICAgICAgbGV0IHNvcnRlZEtleXMgPSBPYmplY3Qua2V5cyhhRGljdCkuc29ydCgoZmlyc3QsIHNlY29uZCkgPT4ge1xuICAgICAgICAgICAgcmV0dXJuIChwYXJzZUludChmaXJzdCkgLSBwYXJzZUludChzZWNvbmQpKTtcbiAgICAgICAgfSk7XG4gICAgICAgIGxldCBzdWJFbnRyaWVzID0gc29ydGVkS2V5cy5tYXAoa2V5ID0+IHtcbiAgICAgICAgICAgIGxldCBlbnRyeSA9IGFEaWN0W2tleV07XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5lbnVtZXJhdGVkVmFsVG9Tb3J0ZWRBcnJheShlbnRyeSk7XG4gICAgICAgIH0pO1xuICAgICAgICByZXR1cm4gc3ViRW50cmllcztcbiAgICB9XG5cbiAgICBzdGF0aWMga2V5RnJvbU5hbWVQYXJ0cyhuYW1lUGFydHMpe1xuICAgICAgICByZXR1cm4gbmFtZVBhcnRzLmpvaW4oXCItXCIpO1xuICAgIH1cblxuICAgIHN0YXRpYyByZWFkUmVwbGFjZW1lbnRTdHJpbmcocmVwbGFjZW1lbnQpe1xuICAgICAgICBsZXQgbmFtZVBhcnRzID0gW107XG4gICAgICAgIGxldCBpc0VudW1lcmF0ZWQgPSBmYWxzZTtcbiAgICAgICAgbGV0IGVudW1lcmF0ZWRWYWx1ZXMgPSBbXTtcbiAgICAgICAgbGV0IHBpZWNlcyA9IHJlcGxhY2VtZW50LnNwbGl0KCdfJykuZmlsdGVyKGl0ZW0gPT4ge1xuICAgICAgICAgICAgcmV0dXJuIGl0ZW0gIT0gJyc7XG4gICAgICAgIH0pO1xuICAgICAgICBwaWVjZXMuZm9yRWFjaChwaWVjZSA9PiB7XG4gICAgICAgICAgICBsZXQgbnVtID0gcGFyc2VJbnQocGllY2UpO1xuICAgICAgICAgICAgaWYoaXNOYU4obnVtKSl7XG4gICAgICAgICAgICAgICAgbmFtZVBhcnRzLnB1c2gocGllY2UpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgaXNFbnVtZXJhdGVkID0gdHJ1ZTtcbiAgICAgICAgICAgIGVudW1lcmF0ZWRWYWx1ZXMucHVzaChudW0pO1xuICAgICAgICB9XG4gICAgICAgIH0pO1xuICAgICAgICByZXR1cm4ge1xuICAgICAgICAgICAgbmFtZVBhcnRzLFxuICAgICAgICAgICAgaXNFbnVtZXJhdGVkLFxuICAgICAgICAgICAgZW51bWVyYXRlZFZhbHVlc1xuICAgICAgICB9O1xuICAgIH1cbn1cblxuZXhwb3J0IHtcbiAgICBSZXBsYWNlbWVudHNIYW5kbGVyLFxuICAgIFJlcGxhY2VtZW50c0hhbmRsZXIgYXMgZGVmYXVsdFxufTtcbiIsImltcG9ydCAnbWFxdWV0dGUnO1xuY29uc3QgaCA9IG1hcXVldHRlLmg7XG4vL2ltcG9ydCB7bGFuZ1Rvb2xzfSBmcm9tICdhY2UvZXh0L2xhbmd1YWdlX3Rvb2xzJztcbmltcG9ydCB7Q2VsbEhhbmRsZXJ9IGZyb20gJy4vQ2VsbEhhbmRsZXInO1xuaW1wb3J0IHtDZWxsU29ja2V0fSBmcm9tICcuL0NlbGxTb2NrZXQnO1xuaW1wb3J0IHtDb21wb25lbnRSZWdpc3RyeX0gZnJvbSAnLi9Db21wb25lbnRSZWdpc3RyeSc7XG5cbi8qKlxuICogR2xvYmFsc1xuICoqL1xud2luZG93LmxhbmdUb29scyA9IGFjZS5yZXF1aXJlKFwiYWNlL2V4dC9sYW5ndWFnZV90b29sc1wiKTtcbndpbmRvdy5hY2VFZGl0b3JzID0ge307XG53aW5kb3cuaGFuZHNPblRhYmxlcyA9IHt9O1xuXG4vKipcbiAqIEluaXRpYWwgUmVuZGVyXG4gKiovXG5jb25zdCBpbml0aWFsUmVuZGVyID0gZnVuY3Rpb24oKXtcbiAgICByZXR1cm4gaChcImRpdlwiLCB7fSwgW1xuICAgICAgICAgaChcImRpdlwiLCB7aWQ6IFwicGFnZV9yb290XCJ9LCBbXG4gICAgICAgICAgICAgaChcImRpdi5jb250YWluZXItZmx1aWRcIiwge30sIFtcbiAgICAgICAgICAgICAgICAgaChcImRpdi5jYXJkXCIsIHtjbGFzczogXCJtdC01XCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgICBoKFwiZGl2LmNhcmQtYm9keVwiLCB7fSwgW1wiTG9hZGluZy4uLlwiXSlcbiAgICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgICBdKVxuICAgICAgICAgXSksXG4gICAgICAgICBoKFwiZGl2XCIsIHtpZDogXCJob2xkaW5nX3BlblwiLCBzdHlsZTogXCJkaXNwbGF5Om5vbmVcIn0sIFtdKVxuICAgICBdKTtcbn07XG5cbi8qKlxuICogQ2VsbCBTb2NrZXQgYW5kIEhhbmRsZXJcbiAqKi9cbmxldCBwcm9qZWN0b3IgPSBtYXF1ZXR0ZS5jcmVhdGVQcm9qZWN0b3IoKTtcbmNvbnN0IGNlbGxTb2NrZXQgPSBuZXcgQ2VsbFNvY2tldCgpO1xuY29uc3QgY2VsbEhhbmRsZXIgPSBuZXcgQ2VsbEhhbmRsZXIoaCwgcHJvamVjdG9yLCBDb21wb25lbnRSZWdpc3RyeSk7XG5jZWxsU29ja2V0Lm9uUG9zdHNjcmlwdHMoY2VsbEhhbmRsZXIuaGFuZGxlUG9zdHNjcmlwdCk7XG5jZWxsU29ja2V0Lm9uTWVzc2FnZShjZWxsSGFuZGxlci5oYW5kbGVNZXNzYWdlKTtcbmNlbGxTb2NrZXQub25DbG9zZShjZWxsSGFuZGxlci5zaG93Q29ubmVjdGlvbkNsb3NlZCk7XG5jZWxsU29ja2V0Lm9uRXJyb3IoZXJyID0+IHtcbiAgICBjb25zb2xlLmVycm9yKFwiU09DS0VUIEVSUk9SOiBcIiwgZXJyKTtcbn0pO1xuXG4vKiogRm9yIG5vdywgd2UgYmluZCB0aGUgY3VycmVudCBzb2NrZXQgYW5kIGhhbmRsZXIgdG8gdGhlIGdsb2JhbCB3aW5kb3cgKiovXG53aW5kb3cuY2VsbFNvY2tldCA9IGNlbGxTb2NrZXQ7XG53aW5kb3cuY2VsbEhhbmRsZXIgPSBjZWxsSGFuZGxlcjtcblxuLyoqIFJlbmRlciB0b3AgbGV2ZWwgY29tcG9uZW50IG9uY2UgRE9NIGlzIHJlYWR5ICoqL1xuZG9jdW1lbnQuYWRkRXZlbnRMaXN0ZW5lcignRE9NQ29udGVudExvYWRlZCcsICgpID0+IHtcbiAgICBwcm9qZWN0b3IuYXBwZW5kKGRvY3VtZW50LmJvZHksIGluaXRpYWxSZW5kZXIpO1xuICAgIGNlbGxTb2NrZXQuY29ubmVjdCgpO1xufSk7XG5cbi8vIFRFU1RJTkc7IFJFTU9WRVxuY29uc29sZS5sb2coJ01haW4gbW9kdWxlIGxvYWRlZCcpO1xuIiwiKGZ1bmN0aW9uIChnbG9iYWwsIGZhY3RvcnkpIHtcbiAgICB0eXBlb2YgZXhwb3J0cyA9PT0gJ29iamVjdCcgJiYgdHlwZW9mIG1vZHVsZSAhPT0gJ3VuZGVmaW5lZCcgPyBmYWN0b3J5KGV4cG9ydHMpIDpcbiAgICB0eXBlb2YgZGVmaW5lID09PSAnZnVuY3Rpb24nICYmIGRlZmluZS5hbWQgPyBkZWZpbmUoWydleHBvcnRzJ10sIGZhY3RvcnkpIDpcbiAgICAoZ2xvYmFsID0gZ2xvYmFsIHx8IHNlbGYsIGZhY3RvcnkoZ2xvYmFsLm1hcXVldHRlID0ge30pKTtcbn0odGhpcywgZnVuY3Rpb24gKGV4cG9ydHMpIHsgJ3VzZSBzdHJpY3QnO1xuXG4gICAgLyogdHNsaW50OmRpc2FibGUgbm8taHR0cC1zdHJpbmcgKi9cclxuICAgIHZhciBOQU1FU1BBQ0VfVzMgPSAnaHR0cDovL3d3dy53My5vcmcvJztcclxuICAgIC8qIHRzbGludDplbmFibGUgbm8taHR0cC1zdHJpbmcgKi9cclxuICAgIHZhciBOQU1FU1BBQ0VfU1ZHID0gTkFNRVNQQUNFX1czICsgXCIyMDAwL3N2Z1wiO1xyXG4gICAgdmFyIE5BTUVTUEFDRV9YTElOSyA9IE5BTUVTUEFDRV9XMyArIFwiMTk5OS94bGlua1wiO1xyXG4gICAgdmFyIGVtcHR5QXJyYXkgPSBbXTtcclxuICAgIHZhciBleHRlbmQgPSBmdW5jdGlvbiAoYmFzZSwgb3ZlcnJpZGVzKSB7XHJcbiAgICAgICAgdmFyIHJlc3VsdCA9IHt9O1xyXG4gICAgICAgIE9iamVjdC5rZXlzKGJhc2UpLmZvckVhY2goZnVuY3Rpb24gKGtleSkge1xyXG4gICAgICAgICAgICByZXN1bHRba2V5XSA9IGJhc2Vba2V5XTtcclxuICAgICAgICB9KTtcclxuICAgICAgICBpZiAob3ZlcnJpZGVzKSB7XHJcbiAgICAgICAgICAgIE9iamVjdC5rZXlzKG92ZXJyaWRlcykuZm9yRWFjaChmdW5jdGlvbiAoa2V5KSB7XHJcbiAgICAgICAgICAgICAgICByZXN1bHRba2V5XSA9IG92ZXJyaWRlc1trZXldO1xyXG4gICAgICAgICAgICB9KTtcclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuIHJlc3VsdDtcclxuICAgIH07XHJcbiAgICB2YXIgc2FtZSA9IGZ1bmN0aW9uICh2bm9kZTEsIHZub2RlMikge1xyXG4gICAgICAgIGlmICh2bm9kZTEudm5vZGVTZWxlY3RvciAhPT0gdm5vZGUyLnZub2RlU2VsZWN0b3IpIHtcclxuICAgICAgICAgICAgcmV0dXJuIGZhbHNlO1xyXG4gICAgICAgIH1cclxuICAgICAgICBpZiAodm5vZGUxLnByb3BlcnRpZXMgJiYgdm5vZGUyLnByb3BlcnRpZXMpIHtcclxuICAgICAgICAgICAgaWYgKHZub2RlMS5wcm9wZXJ0aWVzLmtleSAhPT0gdm5vZGUyLnByb3BlcnRpZXMua2V5KSB7XHJcbiAgICAgICAgICAgICAgICByZXR1cm4gZmFsc2U7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgcmV0dXJuIHZub2RlMS5wcm9wZXJ0aWVzLmJpbmQgPT09IHZub2RlMi5wcm9wZXJ0aWVzLmJpbmQ7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHJldHVybiAhdm5vZGUxLnByb3BlcnRpZXMgJiYgIXZub2RlMi5wcm9wZXJ0aWVzO1xyXG4gICAgfTtcclxuICAgIHZhciBjaGVja1N0eWxlVmFsdWUgPSBmdW5jdGlvbiAoc3R5bGVWYWx1ZSkge1xyXG4gICAgICAgIGlmICh0eXBlb2Ygc3R5bGVWYWx1ZSAhPT0gJ3N0cmluZycpIHtcclxuICAgICAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdTdHlsZSB2YWx1ZXMgbXVzdCBiZSBzdHJpbmdzJyk7XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIHZhciBmaW5kSW5kZXhPZkNoaWxkID0gZnVuY3Rpb24gKGNoaWxkcmVuLCBzYW1lQXMsIHN0YXJ0KSB7XHJcbiAgICAgICAgaWYgKHNhbWVBcy52bm9kZVNlbGVjdG9yICE9PSAnJykge1xyXG4gICAgICAgICAgICAvLyBOZXZlciBzY2FuIGZvciB0ZXh0LW5vZGVzXHJcbiAgICAgICAgICAgIGZvciAodmFyIGkgPSBzdGFydDsgaSA8IGNoaWxkcmVuLmxlbmd0aDsgaSsrKSB7XHJcbiAgICAgICAgICAgICAgICBpZiAoc2FtZShjaGlsZHJlbltpXSwgc2FtZUFzKSkge1xyXG4gICAgICAgICAgICAgICAgICAgIHJldHVybiBpO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHJldHVybiAtMTtcclxuICAgIH07XHJcbiAgICB2YXIgY2hlY2tEaXN0aW5ndWlzaGFibGUgPSBmdW5jdGlvbiAoY2hpbGROb2RlcywgaW5kZXhUb0NoZWNrLCBwYXJlbnRWTm9kZSwgb3BlcmF0aW9uKSB7XHJcbiAgICAgICAgdmFyIGNoaWxkTm9kZSA9IGNoaWxkTm9kZXNbaW5kZXhUb0NoZWNrXTtcclxuICAgICAgICBpZiAoY2hpbGROb2RlLnZub2RlU2VsZWN0b3IgPT09ICcnKSB7XHJcbiAgICAgICAgICAgIHJldHVybjsgLy8gVGV4dCBub2RlcyBuZWVkIG5vdCBiZSBkaXN0aW5ndWlzaGFibGVcclxuICAgICAgICB9XHJcbiAgICAgICAgdmFyIHByb3BlcnRpZXMgPSBjaGlsZE5vZGUucHJvcGVydGllcztcclxuICAgICAgICB2YXIga2V5ID0gcHJvcGVydGllcyA/IChwcm9wZXJ0aWVzLmtleSA9PT0gdW5kZWZpbmVkID8gcHJvcGVydGllcy5iaW5kIDogcHJvcGVydGllcy5rZXkpIDogdW5kZWZpbmVkO1xyXG4gICAgICAgIGlmICgha2V5KSB7IC8vIEEga2V5IGlzIGp1c3QgYXNzdW1lZCB0byBiZSB1bmlxdWVcclxuICAgICAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCBjaGlsZE5vZGVzLmxlbmd0aDsgaSsrKSB7XHJcbiAgICAgICAgICAgICAgICBpZiAoaSAhPT0gaW5kZXhUb0NoZWNrKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgdmFyIG5vZGUgPSBjaGlsZE5vZGVzW2ldO1xyXG4gICAgICAgICAgICAgICAgICAgIGlmIChzYW1lKG5vZGUsIGNoaWxkTm9kZSkpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgdGhyb3cgbmV3IEVycm9yKHBhcmVudFZOb2RlLnZub2RlU2VsZWN0b3IgKyBcIiBoYWQgYSBcIiArIGNoaWxkTm9kZS52bm9kZVNlbGVjdG9yICsgXCIgY2hpbGQgXCIgKyAob3BlcmF0aW9uID09PSAnYWRkZWQnID8gb3BlcmF0aW9uIDogJ3JlbW92ZWQnKSArIFwiLCBidXQgdGhlcmUgaXMgbm93IG1vcmUgdGhhbiBvbmUuIFlvdSBtdXN0IGFkZCB1bmlxdWUga2V5IHByb3BlcnRpZXMgdG8gbWFrZSB0aGVtIGRpc3Rpbmd1aXNoYWJsZS5cIik7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIHZhciBub2RlQWRkZWQgPSBmdW5jdGlvbiAodk5vZGUpIHtcclxuICAgICAgICBpZiAodk5vZGUucHJvcGVydGllcykge1xyXG4gICAgICAgICAgICB2YXIgZW50ZXJBbmltYXRpb24gPSB2Tm9kZS5wcm9wZXJ0aWVzLmVudGVyQW5pbWF0aW9uO1xyXG4gICAgICAgICAgICBpZiAoZW50ZXJBbmltYXRpb24pIHtcclxuICAgICAgICAgICAgICAgIGVudGVyQW5pbWF0aW9uKHZOb2RlLmRvbU5vZGUsIHZOb2RlLnByb3BlcnRpZXMpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIHZhciByZW1vdmVkTm9kZXMgPSBbXTtcclxuICAgIHZhciByZXF1ZXN0ZWRJZGxlQ2FsbGJhY2sgPSBmYWxzZTtcclxuICAgIHZhciB2aXNpdFJlbW92ZWROb2RlID0gZnVuY3Rpb24gKG5vZGUpIHtcclxuICAgICAgICAobm9kZS5jaGlsZHJlbiB8fCBbXSkuZm9yRWFjaCh2aXNpdFJlbW92ZWROb2RlKTtcclxuICAgICAgICBpZiAobm9kZS5wcm9wZXJ0aWVzICYmIG5vZGUucHJvcGVydGllcy5hZnRlclJlbW92ZWQpIHtcclxuICAgICAgICAgICAgbm9kZS5wcm9wZXJ0aWVzLmFmdGVyUmVtb3ZlZC5hcHBseShub2RlLnByb3BlcnRpZXMuYmluZCB8fCBub2RlLnByb3BlcnRpZXMsIFtub2RlLmRvbU5vZGVdKTtcclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgdmFyIHByb2Nlc3NQZW5kaW5nTm9kZVJlbW92YWxzID0gZnVuY3Rpb24gKCkge1xyXG4gICAgICAgIHJlcXVlc3RlZElkbGVDYWxsYmFjayA9IGZhbHNlO1xyXG4gICAgICAgIHJlbW92ZWROb2Rlcy5mb3JFYWNoKHZpc2l0UmVtb3ZlZE5vZGUpO1xyXG4gICAgICAgIHJlbW92ZWROb2Rlcy5sZW5ndGggPSAwO1xyXG4gICAgfTtcclxuICAgIHZhciBzY2hlZHVsZU5vZGVSZW1vdmFsID0gZnVuY3Rpb24gKHZOb2RlKSB7XHJcbiAgICAgICAgcmVtb3ZlZE5vZGVzLnB1c2godk5vZGUpO1xyXG4gICAgICAgIGlmICghcmVxdWVzdGVkSWRsZUNhbGxiYWNrKSB7XHJcbiAgICAgICAgICAgIHJlcXVlc3RlZElkbGVDYWxsYmFjayA9IHRydWU7XHJcbiAgICAgICAgICAgIGlmICh0eXBlb2Ygd2luZG93ICE9PSAndW5kZWZpbmVkJyAmJiAncmVxdWVzdElkbGVDYWxsYmFjaycgaW4gd2luZG93KSB7XHJcbiAgICAgICAgICAgICAgICB3aW5kb3cucmVxdWVzdElkbGVDYWxsYmFjayhwcm9jZXNzUGVuZGluZ05vZGVSZW1vdmFscywgeyB0aW1lb3V0OiAxNiB9KTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgIHNldFRpbWVvdXQocHJvY2Vzc1BlbmRpbmdOb2RlUmVtb3ZhbHMsIDE2KTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgIH07XHJcbiAgICB2YXIgbm9kZVRvUmVtb3ZlID0gZnVuY3Rpb24gKHZOb2RlKSB7XHJcbiAgICAgICAgdmFyIGRvbU5vZGUgPSB2Tm9kZS5kb21Ob2RlO1xyXG4gICAgICAgIGlmICh2Tm9kZS5wcm9wZXJ0aWVzKSB7XHJcbiAgICAgICAgICAgIHZhciBleGl0QW5pbWF0aW9uID0gdk5vZGUucHJvcGVydGllcy5leGl0QW5pbWF0aW9uO1xyXG4gICAgICAgICAgICBpZiAoZXhpdEFuaW1hdGlvbikge1xyXG4gICAgICAgICAgICAgICAgZG9tTm9kZS5zdHlsZS5wb2ludGVyRXZlbnRzID0gJ25vbmUnO1xyXG4gICAgICAgICAgICAgICAgdmFyIHJlbW92ZURvbU5vZGUgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKGRvbU5vZGUucGFyZW50Tm9kZSkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBkb21Ob2RlLnBhcmVudE5vZGUucmVtb3ZlQ2hpbGQoZG9tTm9kZSk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHNjaGVkdWxlTm9kZVJlbW92YWwodk5vZGUpO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIH07XHJcbiAgICAgICAgICAgICAgICBleGl0QW5pbWF0aW9uKGRvbU5vZGUsIHJlbW92ZURvbU5vZGUsIHZOb2RlLnByb3BlcnRpZXMpO1xyXG4gICAgICAgICAgICAgICAgcmV0dXJuO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGlmIChkb21Ob2RlLnBhcmVudE5vZGUpIHtcclxuICAgICAgICAgICAgZG9tTm9kZS5wYXJlbnROb2RlLnJlbW92ZUNoaWxkKGRvbU5vZGUpO1xyXG4gICAgICAgICAgICBzY2hlZHVsZU5vZGVSZW1vdmFsKHZOb2RlKTtcclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgdmFyIHNldFByb3BlcnRpZXMgPSBmdW5jdGlvbiAoZG9tTm9kZSwgcHJvcGVydGllcywgcHJvamVjdGlvbk9wdGlvbnMpIHtcclxuICAgICAgICBpZiAoIXByb3BlcnRpZXMpIHtcclxuICAgICAgICAgICAgcmV0dXJuO1xyXG4gICAgICAgIH1cclxuICAgICAgICB2YXIgZXZlbnRIYW5kbGVySW50ZXJjZXB0b3IgPSBwcm9qZWN0aW9uT3B0aW9ucy5ldmVudEhhbmRsZXJJbnRlcmNlcHRvcjtcclxuICAgICAgICB2YXIgcHJvcE5hbWVzID0gT2JqZWN0LmtleXMocHJvcGVydGllcyk7XHJcbiAgICAgICAgdmFyIHByb3BDb3VudCA9IHByb3BOYW1lcy5sZW5ndGg7XHJcbiAgICAgICAgdmFyIF9sb29wXzEgPSBmdW5jdGlvbiAoaSkge1xyXG4gICAgICAgICAgICB2YXIgcHJvcE5hbWUgPSBwcm9wTmFtZXNbaV07XHJcbiAgICAgICAgICAgIHZhciBwcm9wVmFsdWUgPSBwcm9wZXJ0aWVzW3Byb3BOYW1lXTtcclxuICAgICAgICAgICAgaWYgKHByb3BOYW1lID09PSAnY2xhc3NOYW1lJykge1xyXG4gICAgICAgICAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdQcm9wZXJ0eSBcImNsYXNzTmFtZVwiIGlzIG5vdCBzdXBwb3J0ZWQsIHVzZSBcImNsYXNzXCIuJyk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgZWxzZSBpZiAocHJvcE5hbWUgPT09ICdjbGFzcycpIHtcclxuICAgICAgICAgICAgICAgIHRvZ2dsZUNsYXNzZXMoZG9tTm9kZSwgcHJvcFZhbHVlLCB0cnVlKTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBlbHNlIGlmIChwcm9wTmFtZSA9PT0gJ2NsYXNzZXMnKSB7XHJcbiAgICAgICAgICAgICAgICAvLyBvYmplY3Qgd2l0aCBzdHJpbmcga2V5cyBhbmQgYm9vbGVhbiB2YWx1ZXNcclxuICAgICAgICAgICAgICAgIHZhciBjbGFzc05hbWVzID0gT2JqZWN0LmtleXMocHJvcFZhbHVlKTtcclxuICAgICAgICAgICAgICAgIHZhciBjbGFzc05hbWVDb3VudCA9IGNsYXNzTmFtZXMubGVuZ3RoO1xyXG4gICAgICAgICAgICAgICAgZm9yICh2YXIgaiA9IDA7IGogPCBjbGFzc05hbWVDb3VudDsgaisrKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgdmFyIGNsYXNzTmFtZSA9IGNsYXNzTmFtZXNbal07XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKHByb3BWYWx1ZVtjbGFzc05hbWVdKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGRvbU5vZGUuY2xhc3NMaXN0LmFkZChjbGFzc05hbWUpO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBlbHNlIGlmIChwcm9wTmFtZSA9PT0gJ3N0eWxlcycpIHtcclxuICAgICAgICAgICAgICAgIC8vIG9iamVjdCB3aXRoIHN0cmluZyBrZXlzIGFuZCBzdHJpbmcgKCEpIHZhbHVlc1xyXG4gICAgICAgICAgICAgICAgdmFyIHN0eWxlTmFtZXMgPSBPYmplY3Qua2V5cyhwcm9wVmFsdWUpO1xyXG4gICAgICAgICAgICAgICAgdmFyIHN0eWxlQ291bnQgPSBzdHlsZU5hbWVzLmxlbmd0aDtcclxuICAgICAgICAgICAgICAgIGZvciAodmFyIGogPSAwOyBqIDwgc3R5bGVDb3VudDsgaisrKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgdmFyIHN0eWxlTmFtZSA9IHN0eWxlTmFtZXNbal07XHJcbiAgICAgICAgICAgICAgICAgICAgdmFyIHN0eWxlVmFsdWUgPSBwcm9wVmFsdWVbc3R5bGVOYW1lXTtcclxuICAgICAgICAgICAgICAgICAgICBpZiAoc3R5bGVWYWx1ZSkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBjaGVja1N0eWxlVmFsdWUoc3R5bGVWYWx1ZSk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHByb2plY3Rpb25PcHRpb25zLnN0eWxlQXBwbHllcihkb21Ob2RlLCBzdHlsZU5hbWUsIHN0eWxlVmFsdWUpO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBlbHNlIGlmIChwcm9wTmFtZSAhPT0gJ2tleScgJiYgcHJvcFZhbHVlICE9PSBudWxsICYmIHByb3BWYWx1ZSAhPT0gdW5kZWZpbmVkKSB7XHJcbiAgICAgICAgICAgICAgICB2YXIgdHlwZSA9IHR5cGVvZiBwcm9wVmFsdWU7XHJcbiAgICAgICAgICAgICAgICBpZiAodHlwZSA9PT0gJ2Z1bmN0aW9uJykge1xyXG4gICAgICAgICAgICAgICAgICAgIGlmIChwcm9wTmFtZS5sYXN0SW5kZXhPZignb24nLCAwKSA9PT0gMCkgeyAvLyBsYXN0SW5kZXhPZigsMCk9PT0wIC0+IHN0YXJ0c1dpdGhcclxuICAgICAgICAgICAgICAgICAgICAgICAgaWYgKGV2ZW50SGFuZGxlckludGVyY2VwdG9yKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBwcm9wVmFsdWUgPSBldmVudEhhbmRsZXJJbnRlcmNlcHRvcihwcm9wTmFtZSwgcHJvcFZhbHVlLCBkb21Ob2RlLCBwcm9wZXJ0aWVzKTsgLy8gaW50ZXJjZXB0IGV2ZW50aGFuZGxlcnNcclxuICAgICAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgICAgICBpZiAocHJvcE5hbWUgPT09ICdvbmlucHV0Jykge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgLyogdHNsaW50OmRpc2FibGUgbm8tdGhpcy1rZXl3b3JkIG5vLWludmFsaWQtdGhpcyBvbmx5LWFycm93LWZ1bmN0aW9ucyBuby12b2lkLWV4cHJlc3Npb24gKi9cclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIChmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgLy8gcmVjb3JkIHRoZSBldnQudGFyZ2V0LnZhbHVlLCBiZWNhdXNlIElFIGFuZCBFZGdlIHNvbWV0aW1lcyBkbyBhIHJlcXVlc3RBbmltYXRpb25GcmFtZSBiZXR3ZWVuIGNoYW5naW5nIHZhbHVlIGFuZCBydW5uaW5nIG9uaW5wdXRcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB2YXIgb2xkUHJvcFZhbHVlID0gcHJvcFZhbHVlO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHByb3BWYWx1ZSA9IGZ1bmN0aW9uIChldnQpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgb2xkUHJvcFZhbHVlLmFwcGx5KHRoaXMsIFtldnRdKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZXZ0LnRhcmdldFsnb25pbnB1dC12YWx1ZSddID0gZXZ0LnRhcmdldC52YWx1ZTsgLy8gbWF5IGJlIEhUTUxUZXh0QXJlYUVsZW1lbnQgYXMgd2VsbFxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH07XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9KCkpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgLyogdHNsaW50OmVuYWJsZSAqL1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGRvbU5vZGVbcHJvcE5hbWVdID0gcHJvcFZhbHVlO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIGVsc2UgaWYgKHByb2plY3Rpb25PcHRpb25zLm5hbWVzcGFjZSA9PT0gTkFNRVNQQUNFX1NWRykge1xyXG4gICAgICAgICAgICAgICAgICAgIGlmIChwcm9wTmFtZSA9PT0gJ2hyZWYnKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGRvbU5vZGUuc2V0QXR0cmlidXRlTlMoTkFNRVNQQUNFX1hMSU5LLCBwcm9wTmFtZSwgcHJvcFZhbHVlKTtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIC8vIGFsbCBTVkcgYXR0cmlidXRlcyBhcmUgcmVhZC1vbmx5IGluIERPTSwgc28uLi5cclxuICAgICAgICAgICAgICAgICAgICAgICAgZG9tTm9kZS5zZXRBdHRyaWJ1dGUocHJvcE5hbWUsIHByb3BWYWx1ZSk7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgZWxzZSBpZiAodHlwZSA9PT0gJ3N0cmluZycgJiYgcHJvcE5hbWUgIT09ICd2YWx1ZScgJiYgcHJvcE5hbWUgIT09ICdpbm5lckhUTUwnKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgZG9tTm9kZS5zZXRBdHRyaWJ1dGUocHJvcE5hbWUsIHByb3BWYWx1ZSk7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgICBkb21Ob2RlW3Byb3BOYW1lXSA9IHByb3BWYWx1ZTtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH07XHJcbiAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCBwcm9wQ291bnQ7IGkrKykge1xyXG4gICAgICAgICAgICBfbG9vcF8xKGkpO1xyXG4gICAgICAgIH1cclxuICAgIH07XHJcbiAgICB2YXIgYWRkQ2hpbGRyZW4gPSBmdW5jdGlvbiAoZG9tTm9kZSwgY2hpbGRyZW4sIHByb2plY3Rpb25PcHRpb25zKSB7XHJcbiAgICAgICAgaWYgKCFjaGlsZHJlbikge1xyXG4gICAgICAgICAgICByZXR1cm47XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGZvciAodmFyIF9pID0gMCwgY2hpbGRyZW5fMSA9IGNoaWxkcmVuOyBfaSA8IGNoaWxkcmVuXzEubGVuZ3RoOyBfaSsrKSB7XHJcbiAgICAgICAgICAgIHZhciBjaGlsZCA9IGNoaWxkcmVuXzFbX2ldO1xyXG4gICAgICAgICAgICBjcmVhdGVEb20oY2hpbGQsIGRvbU5vZGUsIHVuZGVmaW5lZCwgcHJvamVjdGlvbk9wdGlvbnMpO1xyXG4gICAgICAgIH1cclxuICAgIH07XHJcbiAgICB2YXIgaW5pdFByb3BlcnRpZXNBbmRDaGlsZHJlbiA9IGZ1bmN0aW9uIChkb21Ob2RlLCB2bm9kZSwgcHJvamVjdGlvbk9wdGlvbnMpIHtcclxuICAgICAgICBhZGRDaGlsZHJlbihkb21Ob2RlLCB2bm9kZS5jaGlsZHJlbiwgcHJvamVjdGlvbk9wdGlvbnMpOyAvLyBjaGlsZHJlbiBiZWZvcmUgcHJvcGVydGllcywgbmVlZGVkIGZvciB2YWx1ZSBwcm9wZXJ0eSBvZiA8c2VsZWN0Pi5cclxuICAgICAgICBpZiAodm5vZGUudGV4dCkge1xyXG4gICAgICAgICAgICBkb21Ob2RlLnRleHRDb250ZW50ID0gdm5vZGUudGV4dDtcclxuICAgICAgICB9XHJcbiAgICAgICAgc2V0UHJvcGVydGllcyhkb21Ob2RlLCB2bm9kZS5wcm9wZXJ0aWVzLCBwcm9qZWN0aW9uT3B0aW9ucyk7XHJcbiAgICAgICAgaWYgKHZub2RlLnByb3BlcnRpZXMgJiYgdm5vZGUucHJvcGVydGllcy5hZnRlckNyZWF0ZSkge1xyXG4gICAgICAgICAgICB2bm9kZS5wcm9wZXJ0aWVzLmFmdGVyQ3JlYXRlLmFwcGx5KHZub2RlLnByb3BlcnRpZXMuYmluZCB8fCB2bm9kZS5wcm9wZXJ0aWVzLCBbZG9tTm9kZSwgcHJvamVjdGlvbk9wdGlvbnMsIHZub2RlLnZub2RlU2VsZWN0b3IsIHZub2RlLnByb3BlcnRpZXMsIHZub2RlLmNoaWxkcmVuXSk7XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIHZhciBjcmVhdGVEb20gPSBmdW5jdGlvbiAodm5vZGUsIHBhcmVudE5vZGUsIGluc2VydEJlZm9yZSwgcHJvamVjdGlvbk9wdGlvbnMpIHtcclxuICAgICAgICB2YXIgZG9tTm9kZTtcclxuICAgICAgICB2YXIgc3RhcnQgPSAwO1xyXG4gICAgICAgIHZhciB2bm9kZVNlbGVjdG9yID0gdm5vZGUudm5vZGVTZWxlY3RvcjtcclxuICAgICAgICB2YXIgZG9jID0gcGFyZW50Tm9kZS5vd25lckRvY3VtZW50O1xyXG4gICAgICAgIGlmICh2bm9kZVNlbGVjdG9yID09PSAnJykge1xyXG4gICAgICAgICAgICBkb21Ob2RlID0gdm5vZGUuZG9tTm9kZSA9IGRvYy5jcmVhdGVUZXh0Tm9kZSh2bm9kZS50ZXh0KTtcclxuICAgICAgICAgICAgaWYgKGluc2VydEJlZm9yZSAhPT0gdW5kZWZpbmVkKSB7XHJcbiAgICAgICAgICAgICAgICBwYXJlbnROb2RlLmluc2VydEJlZm9yZShkb21Ob2RlLCBpbnNlcnRCZWZvcmUpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgcGFyZW50Tm9kZS5hcHBlbmRDaGlsZChkb21Ob2RlKTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPD0gdm5vZGVTZWxlY3Rvci5sZW5ndGg7ICsraSkge1xyXG4gICAgICAgICAgICAgICAgdmFyIGMgPSB2bm9kZVNlbGVjdG9yLmNoYXJBdChpKTtcclxuICAgICAgICAgICAgICAgIGlmIChpID09PSB2bm9kZVNlbGVjdG9yLmxlbmd0aCB8fCBjID09PSAnLicgfHwgYyA9PT0gJyMnKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgdmFyIHR5cGUgPSB2bm9kZVNlbGVjdG9yLmNoYXJBdChzdGFydCAtIDEpO1xyXG4gICAgICAgICAgICAgICAgICAgIHZhciBmb3VuZCA9IHZub2RlU2VsZWN0b3Iuc2xpY2Uoc3RhcnQsIGkpO1xyXG4gICAgICAgICAgICAgICAgICAgIGlmICh0eXBlID09PSAnLicpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgZG9tTm9kZS5jbGFzc0xpc3QuYWRkKGZvdW5kKTtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgZWxzZSBpZiAodHlwZSA9PT0gJyMnKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGRvbU5vZGUuaWQgPSBmb3VuZDtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGlmIChmb3VuZCA9PT0gJ3N2ZycpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHByb2plY3Rpb25PcHRpb25zID0gZXh0ZW5kKHByb2plY3Rpb25PcHRpb25zLCB7IG5hbWVzcGFjZTogTkFNRVNQQUNFX1NWRyB9KTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgICAgICBpZiAocHJvamVjdGlvbk9wdGlvbnMubmFtZXNwYWNlICE9PSB1bmRlZmluZWQpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGRvbU5vZGUgPSB2bm9kZS5kb21Ob2RlID0gZG9jLmNyZWF0ZUVsZW1lbnROUyhwcm9qZWN0aW9uT3B0aW9ucy5uYW1lc3BhY2UsIGZvdW5kKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGRvbU5vZGUgPSB2bm9kZS5kb21Ob2RlID0gKHZub2RlLmRvbU5vZGUgfHwgZG9jLmNyZWF0ZUVsZW1lbnQoZm91bmQpKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmIChmb3VuZCA9PT0gJ2lucHV0JyAmJiB2bm9kZS5wcm9wZXJ0aWVzICYmIHZub2RlLnByb3BlcnRpZXMudHlwZSAhPT0gdW5kZWZpbmVkKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgLy8gSUU4IGFuZCBvbGRlciBkb24ndCBzdXBwb3J0IHNldHRpbmcgaW5wdXQgdHlwZSBhZnRlciB0aGUgRE9NIE5vZGUgaGFzIGJlZW4gYWRkZWQgdG8gdGhlIGRvY3VtZW50XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZG9tTm9kZS5zZXRBdHRyaWJ1dGUoJ3R5cGUnLCB2bm9kZS5wcm9wZXJ0aWVzLnR5cGUpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGlmIChpbnNlcnRCZWZvcmUgIT09IHVuZGVmaW5lZCkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgcGFyZW50Tm9kZS5pbnNlcnRCZWZvcmUoZG9tTm9kZSwgaW5zZXJ0QmVmb3JlKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgICAgICBlbHNlIGlmIChkb21Ob2RlLnBhcmVudE5vZGUgIT09IHBhcmVudE5vZGUpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHBhcmVudE5vZGUuYXBwZW5kQ2hpbGQoZG9tTm9kZSk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgc3RhcnQgPSBpICsgMTtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBpbml0UHJvcGVydGllc0FuZENoaWxkcmVuKGRvbU5vZGUsIHZub2RlLCBwcm9qZWN0aW9uT3B0aW9ucyk7XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIHZhciB1cGRhdGVEb207XHJcbiAgICAvKipcclxuICAgICAqIEFkZHMgb3IgcmVtb3ZlcyBjbGFzc2VzIGZyb20gYW4gRWxlbWVudFxyXG4gICAgICogQHBhcmFtIGRvbU5vZGUgdGhlIGVsZW1lbnRcclxuICAgICAqIEBwYXJhbSBjbGFzc2VzIGEgc3RyaW5nIHNlcGFyYXRlZCBsaXN0IG9mIGNsYXNzZXNcclxuICAgICAqIEBwYXJhbSBvbiB0cnVlIG1lYW5zIGFkZCBjbGFzc2VzLCBmYWxzZSBtZWFucyByZW1vdmVcclxuICAgICAqL1xyXG4gICAgdmFyIHRvZ2dsZUNsYXNzZXMgPSBmdW5jdGlvbiAoZG9tTm9kZSwgY2xhc3Nlcywgb24pIHtcclxuICAgICAgICBpZiAoIWNsYXNzZXMpIHtcclxuICAgICAgICAgICAgcmV0dXJuO1xyXG4gICAgICAgIH1cclxuICAgICAgICBjbGFzc2VzLnNwbGl0KCcgJykuZm9yRWFjaChmdW5jdGlvbiAoY2xhc3NUb1RvZ2dsZSkge1xyXG4gICAgICAgICAgICBpZiAoY2xhc3NUb1RvZ2dsZSkge1xyXG4gICAgICAgICAgICAgICAgZG9tTm9kZS5jbGFzc0xpc3QudG9nZ2xlKGNsYXNzVG9Ub2dnbGUsIG9uKTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH0pO1xyXG4gICAgfTtcclxuICAgIHZhciB1cGRhdGVQcm9wZXJ0aWVzID0gZnVuY3Rpb24gKGRvbU5vZGUsIHByZXZpb3VzUHJvcGVydGllcywgcHJvcGVydGllcywgcHJvamVjdGlvbk9wdGlvbnMpIHtcclxuICAgICAgICBpZiAoIXByb3BlcnRpZXMpIHtcclxuICAgICAgICAgICAgcmV0dXJuO1xyXG4gICAgICAgIH1cclxuICAgICAgICB2YXIgcHJvcGVydGllc1VwZGF0ZWQgPSBmYWxzZTtcclxuICAgICAgICB2YXIgcHJvcE5hbWVzID0gT2JqZWN0LmtleXMocHJvcGVydGllcyk7XHJcbiAgICAgICAgdmFyIHByb3BDb3VudCA9IHByb3BOYW1lcy5sZW5ndGg7XHJcbiAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCBwcm9wQ291bnQ7IGkrKykge1xyXG4gICAgICAgICAgICB2YXIgcHJvcE5hbWUgPSBwcm9wTmFtZXNbaV07XHJcbiAgICAgICAgICAgIC8vIGFzc3VtaW5nIHRoYXQgcHJvcGVydGllcyB3aWxsIGJlIG51bGxpZmllZCBpbnN0ZWFkIG9mIG1pc3NpbmcgaXMgYnkgZGVzaWduXHJcbiAgICAgICAgICAgIHZhciBwcm9wVmFsdWUgPSBwcm9wZXJ0aWVzW3Byb3BOYW1lXTtcclxuICAgICAgICAgICAgdmFyIHByZXZpb3VzVmFsdWUgPSBwcmV2aW91c1Byb3BlcnRpZXNbcHJvcE5hbWVdO1xyXG4gICAgICAgICAgICBpZiAocHJvcE5hbWUgPT09ICdjbGFzcycpIHtcclxuICAgICAgICAgICAgICAgIGlmIChwcmV2aW91c1ZhbHVlICE9PSBwcm9wVmFsdWUpIHtcclxuICAgICAgICAgICAgICAgICAgICB0b2dnbGVDbGFzc2VzKGRvbU5vZGUsIHByZXZpb3VzVmFsdWUsIGZhbHNlKTtcclxuICAgICAgICAgICAgICAgICAgICB0b2dnbGVDbGFzc2VzKGRvbU5vZGUsIHByb3BWYWx1ZSwgdHJ1ZSk7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgZWxzZSBpZiAocHJvcE5hbWUgPT09ICdjbGFzc2VzJykge1xyXG4gICAgICAgICAgICAgICAgdmFyIGNsYXNzTGlzdCA9IGRvbU5vZGUuY2xhc3NMaXN0O1xyXG4gICAgICAgICAgICAgICAgdmFyIGNsYXNzTmFtZXMgPSBPYmplY3Qua2V5cyhwcm9wVmFsdWUpO1xyXG4gICAgICAgICAgICAgICAgdmFyIGNsYXNzTmFtZUNvdW50ID0gY2xhc3NOYW1lcy5sZW5ndGg7XHJcbiAgICAgICAgICAgICAgICBmb3IgKHZhciBqID0gMDsgaiA8IGNsYXNzTmFtZUNvdW50OyBqKyspIHtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgY2xhc3NOYW1lID0gY2xhc3NOYW1lc1tqXTtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgb24gPSAhIXByb3BWYWx1ZVtjbGFzc05hbWVdO1xyXG4gICAgICAgICAgICAgICAgICAgIHZhciBwcmV2aW91c09uID0gISFwcmV2aW91c1ZhbHVlW2NsYXNzTmFtZV07XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKG9uID09PSBwcmV2aW91c09uKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGNvbnRpbnVlO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICBwcm9wZXJ0aWVzVXBkYXRlZCA9IHRydWU7XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKG9uKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGNsYXNzTGlzdC5hZGQoY2xhc3NOYW1lKTtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGNsYXNzTGlzdC5yZW1vdmUoY2xhc3NOYW1lKTtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgZWxzZSBpZiAocHJvcE5hbWUgPT09ICdzdHlsZXMnKSB7XHJcbiAgICAgICAgICAgICAgICB2YXIgc3R5bGVOYW1lcyA9IE9iamVjdC5rZXlzKHByb3BWYWx1ZSk7XHJcbiAgICAgICAgICAgICAgICB2YXIgc3R5bGVDb3VudCA9IHN0eWxlTmFtZXMubGVuZ3RoO1xyXG4gICAgICAgICAgICAgICAgZm9yICh2YXIgaiA9IDA7IGogPCBzdHlsZUNvdW50OyBqKyspIHtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgc3R5bGVOYW1lID0gc3R5bGVOYW1lc1tqXTtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgbmV3U3R5bGVWYWx1ZSA9IHByb3BWYWx1ZVtzdHlsZU5hbWVdO1xyXG4gICAgICAgICAgICAgICAgICAgIHZhciBvbGRTdHlsZVZhbHVlID0gcHJldmlvdXNWYWx1ZVtzdHlsZU5hbWVdO1xyXG4gICAgICAgICAgICAgICAgICAgIGlmIChuZXdTdHlsZVZhbHVlID09PSBvbGRTdHlsZVZhbHVlKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGNvbnRpbnVlO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICBwcm9wZXJ0aWVzVXBkYXRlZCA9IHRydWU7XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKG5ld1N0eWxlVmFsdWUpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgY2hlY2tTdHlsZVZhbHVlKG5ld1N0eWxlVmFsdWUpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBwcm9qZWN0aW9uT3B0aW9ucy5zdHlsZUFwcGx5ZXIoZG9tTm9kZSwgc3R5bGVOYW1lLCBuZXdTdHlsZVZhbHVlKTtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHByb2plY3Rpb25PcHRpb25zLnN0eWxlQXBwbHllcihkb21Ob2RlLCBzdHlsZU5hbWUsICcnKTtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICBpZiAoIXByb3BWYWx1ZSAmJiB0eXBlb2YgcHJldmlvdXNWYWx1ZSA9PT0gJ3N0cmluZycpIHtcclxuICAgICAgICAgICAgICAgICAgICBwcm9wVmFsdWUgPSAnJztcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIGlmIChwcm9wTmFtZSA9PT0gJ3ZhbHVlJykgeyAvLyB2YWx1ZSBjYW4gYmUgbWFuaXB1bGF0ZWQgYnkgdGhlIHVzZXIgZGlyZWN0bHkgYW5kIHVzaW5nIGV2ZW50LnByZXZlbnREZWZhdWx0KCkgaXMgbm90IGFuIG9wdGlvblxyXG4gICAgICAgICAgICAgICAgICAgIHZhciBkb21WYWx1ZSA9IGRvbU5vZGVbcHJvcE5hbWVdO1xyXG4gICAgICAgICAgICAgICAgICAgIGlmIChkb21WYWx1ZSAhPT0gcHJvcFZhbHVlIC8vIFRoZSAndmFsdWUnIGluIHRoZSBET00gdHJlZSAhPT0gbmV3VmFsdWVcclxuICAgICAgICAgICAgICAgICAgICAgICAgJiYgKGRvbU5vZGVbJ29uaW5wdXQtdmFsdWUnXVxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgPyBkb21WYWx1ZSA9PT0gZG9tTm9kZVsnb25pbnB1dC12YWx1ZSddIC8vIElmIHRoZSBsYXN0IHJlcG9ydGVkIHZhbHVlIHRvICdvbmlucHV0JyBkb2VzIG5vdCBtYXRjaCBkb21WYWx1ZSwgZG8gbm90aGluZyBhbmQgd2FpdCBmb3Igb25pbnB1dFxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgOiBwcm9wVmFsdWUgIT09IHByZXZpb3VzVmFsdWUgLy8gT25seSB1cGRhdGUgdGhlIHZhbHVlIGlmIHRoZSB2ZG9tIGNoYW5nZWRcclxuICAgICAgICAgICAgICAgICAgICAgICAgKSkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAvLyBUaGUgZWRnZSBjYXNlcyBhcmUgZGVzY3JpYmVkIGluIHRoZSB0ZXN0c1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBkb21Ob2RlW3Byb3BOYW1lXSA9IHByb3BWYWx1ZTsgLy8gUmVzZXQgdGhlIHZhbHVlLCBldmVuIGlmIHRoZSB2aXJ0dWFsIERPTSBkaWQgbm90IGNoYW5nZVxyXG4gICAgICAgICAgICAgICAgICAgICAgICBkb21Ob2RlWydvbmlucHV0LXZhbHVlJ10gPSB1bmRlZmluZWQ7XHJcbiAgICAgICAgICAgICAgICAgICAgfSAvLyBlbHNlIGRvIG5vdCB1cGRhdGUgdGhlIGRvbU5vZGUsIG90aGVyd2lzZSB0aGUgY3Vyc29yIHBvc2l0aW9uIHdvdWxkIGJlIGNoYW5nZWRcclxuICAgICAgICAgICAgICAgICAgICBpZiAocHJvcFZhbHVlICE9PSBwcmV2aW91c1ZhbHVlKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHByb3BlcnRpZXNVcGRhdGVkID0gdHJ1ZTtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICBlbHNlIGlmIChwcm9wVmFsdWUgIT09IHByZXZpb3VzVmFsdWUpIHtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgdHlwZSA9IHR5cGVvZiBwcm9wVmFsdWU7XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKHR5cGUgIT09ICdmdW5jdGlvbicgfHwgIXByb2plY3Rpb25PcHRpb25zLmV2ZW50SGFuZGxlckludGVyY2VwdG9yKSB7IC8vIEZ1bmN0aW9uIHVwZGF0ZXMgYXJlIGV4cGVjdGVkIHRvIGJlIGhhbmRsZWQgYnkgdGhlIEV2ZW50SGFuZGxlckludGVyY2VwdG9yXHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGlmIChwcm9qZWN0aW9uT3B0aW9ucy5uYW1lc3BhY2UgPT09IE5BTUVTUEFDRV9TVkcpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmIChwcm9wTmFtZSA9PT0gJ2hyZWYnKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZG9tTm9kZS5zZXRBdHRyaWJ1dGVOUyhOQU1FU1BBQ0VfWExJTkssIHByb3BOYW1lLCBwcm9wVmFsdWUpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgLy8gYWxsIFNWRyBhdHRyaWJ1dGVzIGFyZSByZWFkLW9ubHkgaW4gRE9NLCBzby4uLlxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGRvbU5vZGUuc2V0QXR0cmlidXRlKHByb3BOYW1lLCBwcm9wVmFsdWUpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGVsc2UgaWYgKHR5cGUgPT09ICdzdHJpbmcnICYmIHByb3BOYW1lICE9PSAnaW5uZXJIVE1MJykge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgaWYgKHByb3BOYW1lID09PSAncm9sZScgJiYgcHJvcFZhbHVlID09PSAnJykge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGRvbU5vZGUucmVtb3ZlQXR0cmlidXRlKHByb3BOYW1lKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGRvbU5vZGUuc2V0QXR0cmlidXRlKHByb3BOYW1lLCBwcm9wVmFsdWUpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGVsc2UgaWYgKGRvbU5vZGVbcHJvcE5hbWVdICE9PSBwcm9wVmFsdWUpIHsgLy8gQ29tcGFyaXNvbiBpcyBoZXJlIGZvciBzaWRlLWVmZmVjdHMgaW4gRWRnZSB3aXRoIHNjcm9sbExlZnQgYW5kIHNjcm9sbFRvcFxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgZG9tTm9kZVtwcm9wTmFtZV0gPSBwcm9wVmFsdWU7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICAgICAgcHJvcGVydGllc1VwZGF0ZWQgPSB0cnVlO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgICAgICByZXR1cm4gcHJvcGVydGllc1VwZGF0ZWQ7XHJcbiAgICB9O1xyXG4gICAgdmFyIHVwZGF0ZUNoaWxkcmVuID0gZnVuY3Rpb24gKHZub2RlLCBkb21Ob2RlLCBvbGRDaGlsZHJlbiwgbmV3Q2hpbGRyZW4sIHByb2plY3Rpb25PcHRpb25zKSB7XHJcbiAgICAgICAgaWYgKG9sZENoaWxkcmVuID09PSBuZXdDaGlsZHJlbikge1xyXG4gICAgICAgICAgICByZXR1cm4gZmFsc2U7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIG9sZENoaWxkcmVuID0gb2xkQ2hpbGRyZW4gfHwgZW1wdHlBcnJheTtcclxuICAgICAgICBuZXdDaGlsZHJlbiA9IG5ld0NoaWxkcmVuIHx8IGVtcHR5QXJyYXk7XHJcbiAgICAgICAgdmFyIG9sZENoaWxkcmVuTGVuZ3RoID0gb2xkQ2hpbGRyZW4ubGVuZ3RoO1xyXG4gICAgICAgIHZhciBuZXdDaGlsZHJlbkxlbmd0aCA9IG5ld0NoaWxkcmVuLmxlbmd0aDtcclxuICAgICAgICB2YXIgb2xkSW5kZXggPSAwO1xyXG4gICAgICAgIHZhciBuZXdJbmRleCA9IDA7XHJcbiAgICAgICAgdmFyIGk7XHJcbiAgICAgICAgdmFyIHRleHRVcGRhdGVkID0gZmFsc2U7XHJcbiAgICAgICAgd2hpbGUgKG5ld0luZGV4IDwgbmV3Q2hpbGRyZW5MZW5ndGgpIHtcclxuICAgICAgICAgICAgdmFyIG9sZENoaWxkID0gKG9sZEluZGV4IDwgb2xkQ2hpbGRyZW5MZW5ndGgpID8gb2xkQ2hpbGRyZW5bb2xkSW5kZXhdIDogdW5kZWZpbmVkO1xyXG4gICAgICAgICAgICB2YXIgbmV3Q2hpbGQgPSBuZXdDaGlsZHJlbltuZXdJbmRleF07XHJcbiAgICAgICAgICAgIGlmIChvbGRDaGlsZCAhPT0gdW5kZWZpbmVkICYmIHNhbWUob2xkQ2hpbGQsIG5ld0NoaWxkKSkge1xyXG4gICAgICAgICAgICAgICAgdGV4dFVwZGF0ZWQgPSB1cGRhdGVEb20ob2xkQ2hpbGQsIG5ld0NoaWxkLCBwcm9qZWN0aW9uT3B0aW9ucykgfHwgdGV4dFVwZGF0ZWQ7XHJcbiAgICAgICAgICAgICAgICBvbGRJbmRleCsrO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgdmFyIGZpbmRPbGRJbmRleCA9IGZpbmRJbmRleE9mQ2hpbGQob2xkQ2hpbGRyZW4sIG5ld0NoaWxkLCBvbGRJbmRleCArIDEpO1xyXG4gICAgICAgICAgICAgICAgaWYgKGZpbmRPbGRJbmRleCA+PSAwKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgLy8gUmVtb3ZlIHByZWNlZGluZyBtaXNzaW5nIGNoaWxkcmVuXHJcbiAgICAgICAgICAgICAgICAgICAgZm9yIChpID0gb2xkSW5kZXg7IGkgPCBmaW5kT2xkSW5kZXg7IGkrKykge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBub2RlVG9SZW1vdmUob2xkQ2hpbGRyZW5baV0pO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBjaGVja0Rpc3Rpbmd1aXNoYWJsZShvbGRDaGlsZHJlbiwgaSwgdm5vZGUsICdyZW1vdmVkJyk7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgIHRleHRVcGRhdGVkID0gdXBkYXRlRG9tKG9sZENoaWxkcmVuW2ZpbmRPbGRJbmRleF0sIG5ld0NoaWxkLCBwcm9qZWN0aW9uT3B0aW9ucykgfHwgdGV4dFVwZGF0ZWQ7XHJcbiAgICAgICAgICAgICAgICAgICAgb2xkSW5kZXggPSBmaW5kT2xkSW5kZXggKyAxO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICAgICAgLy8gTmV3IGNoaWxkXHJcbiAgICAgICAgICAgICAgICAgICAgY3JlYXRlRG9tKG5ld0NoaWxkLCBkb21Ob2RlLCAob2xkSW5kZXggPCBvbGRDaGlsZHJlbkxlbmd0aCkgPyBvbGRDaGlsZHJlbltvbGRJbmRleF0uZG9tTm9kZSA6IHVuZGVmaW5lZCwgcHJvamVjdGlvbk9wdGlvbnMpO1xyXG4gICAgICAgICAgICAgICAgICAgIG5vZGVBZGRlZChuZXdDaGlsZCk7XHJcbiAgICAgICAgICAgICAgICAgICAgY2hlY2tEaXN0aW5ndWlzaGFibGUobmV3Q2hpbGRyZW4sIG5ld0luZGV4LCB2bm9kZSwgJ2FkZGVkJyk7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgbmV3SW5kZXgrKztcclxuICAgICAgICB9XHJcbiAgICAgICAgaWYgKG9sZENoaWxkcmVuTGVuZ3RoID4gb2xkSW5kZXgpIHtcclxuICAgICAgICAgICAgLy8gUmVtb3ZlIGNoaWxkIGZyYWdtZW50c1xyXG4gICAgICAgICAgICBmb3IgKGkgPSBvbGRJbmRleDsgaSA8IG9sZENoaWxkcmVuTGVuZ3RoOyBpKyspIHtcclxuICAgICAgICAgICAgICAgIG5vZGVUb1JlbW92ZShvbGRDaGlsZHJlbltpXSk7XHJcbiAgICAgICAgICAgICAgICBjaGVja0Rpc3Rpbmd1aXNoYWJsZShvbGRDaGlsZHJlbiwgaSwgdm5vZGUsICdyZW1vdmVkJyk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuIHRleHRVcGRhdGVkO1xyXG4gICAgfTtcclxuICAgIHVwZGF0ZURvbSA9IGZ1bmN0aW9uIChwcmV2aW91cywgdm5vZGUsIHByb2plY3Rpb25PcHRpb25zKSB7XHJcbiAgICAgICAgdmFyIGRvbU5vZGUgPSBwcmV2aW91cy5kb21Ob2RlO1xyXG4gICAgICAgIHZhciB0ZXh0VXBkYXRlZCA9IGZhbHNlO1xyXG4gICAgICAgIGlmIChwcmV2aW91cyA9PT0gdm5vZGUpIHtcclxuICAgICAgICAgICAgcmV0dXJuIGZhbHNlOyAvLyBCeSBjb250cmFjdCwgVk5vZGUgb2JqZWN0cyBtYXkgbm90IGJlIG1vZGlmaWVkIGFueW1vcmUgYWZ0ZXIgcGFzc2luZyB0aGVtIHRvIG1hcXVldHRlXHJcbiAgICAgICAgfVxyXG4gICAgICAgIHZhciB1cGRhdGVkID0gZmFsc2U7XHJcbiAgICAgICAgaWYgKHZub2RlLnZub2RlU2VsZWN0b3IgPT09ICcnKSB7XHJcbiAgICAgICAgICAgIGlmICh2bm9kZS50ZXh0ICE9PSBwcmV2aW91cy50ZXh0KSB7XHJcbiAgICAgICAgICAgICAgICB2YXIgbmV3VGV4dE5vZGUgPSBkb21Ob2RlLm93bmVyRG9jdW1lbnQuY3JlYXRlVGV4dE5vZGUodm5vZGUudGV4dCk7XHJcbiAgICAgICAgICAgICAgICBkb21Ob2RlLnBhcmVudE5vZGUucmVwbGFjZUNoaWxkKG5ld1RleHROb2RlLCBkb21Ob2RlKTtcclxuICAgICAgICAgICAgICAgIHZub2RlLmRvbU5vZGUgPSBuZXdUZXh0Tm9kZTtcclxuICAgICAgICAgICAgICAgIHRleHRVcGRhdGVkID0gdHJ1ZTtcclxuICAgICAgICAgICAgICAgIHJldHVybiB0ZXh0VXBkYXRlZDtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB2bm9kZS5kb21Ob2RlID0gZG9tTm9kZTtcclxuICAgICAgICB9XHJcbiAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgIGlmICh2bm9kZS52bm9kZVNlbGVjdG9yLmxhc3RJbmRleE9mKCdzdmcnLCAwKSA9PT0gMCkgeyAvLyBsYXN0SW5kZXhPZihuZWVkbGUsMCk9PT0wIG1lYW5zIFN0YXJ0c1dpdGhcclxuICAgICAgICAgICAgICAgIHByb2plY3Rpb25PcHRpb25zID0gZXh0ZW5kKHByb2plY3Rpb25PcHRpb25zLCB7IG5hbWVzcGFjZTogTkFNRVNQQUNFX1NWRyB9KTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBpZiAocHJldmlvdXMudGV4dCAhPT0gdm5vZGUudGV4dCkge1xyXG4gICAgICAgICAgICAgICAgdXBkYXRlZCA9IHRydWU7XHJcbiAgICAgICAgICAgICAgICBpZiAodm5vZGUudGV4dCA9PT0gdW5kZWZpbmVkKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgZG9tTm9kZS5yZW1vdmVDaGlsZChkb21Ob2RlLmZpcnN0Q2hpbGQpOyAvLyB0aGUgb25seSB0ZXh0bm9kZSBwcmVzdW1hYmx5XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgICBkb21Ob2RlLnRleHRDb250ZW50ID0gdm5vZGUudGV4dDtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB2bm9kZS5kb21Ob2RlID0gZG9tTm9kZTtcclxuICAgICAgICAgICAgdXBkYXRlZCA9IHVwZGF0ZUNoaWxkcmVuKHZub2RlLCBkb21Ob2RlLCBwcmV2aW91cy5jaGlsZHJlbiwgdm5vZGUuY2hpbGRyZW4sIHByb2plY3Rpb25PcHRpb25zKSB8fCB1cGRhdGVkO1xyXG4gICAgICAgICAgICB1cGRhdGVkID0gdXBkYXRlUHJvcGVydGllcyhkb21Ob2RlLCBwcmV2aW91cy5wcm9wZXJ0aWVzLCB2bm9kZS5wcm9wZXJ0aWVzLCBwcm9qZWN0aW9uT3B0aW9ucykgfHwgdXBkYXRlZDtcclxuICAgICAgICAgICAgaWYgKHZub2RlLnByb3BlcnRpZXMgJiYgdm5vZGUucHJvcGVydGllcy5hZnRlclVwZGF0ZSkge1xyXG4gICAgICAgICAgICAgICAgdm5vZGUucHJvcGVydGllcy5hZnRlclVwZGF0ZS5hcHBseSh2bm9kZS5wcm9wZXJ0aWVzLmJpbmQgfHwgdm5vZGUucHJvcGVydGllcywgW2RvbU5vZGUsIHByb2plY3Rpb25PcHRpb25zLCB2bm9kZS52bm9kZVNlbGVjdG9yLCB2bm9kZS5wcm9wZXJ0aWVzLCB2bm9kZS5jaGlsZHJlbl0pO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGlmICh1cGRhdGVkICYmIHZub2RlLnByb3BlcnRpZXMgJiYgdm5vZGUucHJvcGVydGllcy51cGRhdGVBbmltYXRpb24pIHtcclxuICAgICAgICAgICAgdm5vZGUucHJvcGVydGllcy51cGRhdGVBbmltYXRpb24oZG9tTm9kZSwgdm5vZGUucHJvcGVydGllcywgcHJldmlvdXMucHJvcGVydGllcyk7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHJldHVybiB0ZXh0VXBkYXRlZDtcclxuICAgIH07XHJcbiAgICB2YXIgY3JlYXRlUHJvamVjdGlvbiA9IGZ1bmN0aW9uICh2bm9kZSwgcHJvamVjdGlvbk9wdGlvbnMpIHtcclxuICAgICAgICByZXR1cm4ge1xyXG4gICAgICAgICAgICBnZXRMYXN0UmVuZGVyOiBmdW5jdGlvbiAoKSB7IHJldHVybiB2bm9kZTsgfSxcclxuICAgICAgICAgICAgdXBkYXRlOiBmdW5jdGlvbiAodXBkYXRlZFZub2RlKSB7XHJcbiAgICAgICAgICAgICAgICBpZiAodm5vZGUudm5vZGVTZWxlY3RvciAhPT0gdXBkYXRlZFZub2RlLnZub2RlU2VsZWN0b3IpIHtcclxuICAgICAgICAgICAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ1RoZSBzZWxlY3RvciBmb3IgdGhlIHJvb3QgVk5vZGUgbWF5IG5vdCBiZSBjaGFuZ2VkLiAoY29uc2lkZXIgdXNpbmcgZG9tLm1lcmdlIGFuZCBhZGQgb25lIGV4dHJhIGxldmVsIHRvIHRoZSB2aXJ0dWFsIERPTSknKTtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIHZhciBwcmV2aW91c1ZOb2RlID0gdm5vZGU7XHJcbiAgICAgICAgICAgICAgICB2bm9kZSA9IHVwZGF0ZWRWbm9kZTtcclxuICAgICAgICAgICAgICAgIHVwZGF0ZURvbShwcmV2aW91c1ZOb2RlLCB1cGRhdGVkVm5vZGUsIHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAgZG9tTm9kZTogdm5vZGUuZG9tTm9kZVxyXG4gICAgICAgIH07XHJcbiAgICB9O1xuXG4gICAgdmFyIERFRkFVTFRfUFJPSkVDVElPTl9PUFRJT05TID0ge1xyXG4gICAgICAgIG5hbWVzcGFjZTogdW5kZWZpbmVkLFxyXG4gICAgICAgIHBlcmZvcm1hbmNlTG9nZ2VyOiBmdW5jdGlvbiAoKSB7IHJldHVybiB1bmRlZmluZWQ7IH0sXHJcbiAgICAgICAgZXZlbnRIYW5kbGVySW50ZXJjZXB0b3I6IHVuZGVmaW5lZCxcclxuICAgICAgICBzdHlsZUFwcGx5ZXI6IGZ1bmN0aW9uIChkb21Ob2RlLCBzdHlsZU5hbWUsIHZhbHVlKSB7XHJcbiAgICAgICAgICAgIC8vIFByb3ZpZGVzIGEgaG9vayB0byBhZGQgdmVuZG9yIHByZWZpeGVzIGZvciBicm93c2VycyB0aGF0IHN0aWxsIG5lZWQgaXQuXHJcbiAgICAgICAgICAgIGRvbU5vZGUuc3R5bGVbc3R5bGVOYW1lXSA9IHZhbHVlO1xyXG4gICAgICAgIH1cclxuICAgIH07XHJcbiAgICB2YXIgYXBwbHlEZWZhdWx0UHJvamVjdGlvbk9wdGlvbnMgPSBmdW5jdGlvbiAocHJvamVjdG9yT3B0aW9ucykge1xyXG4gICAgICAgIHJldHVybiBleHRlbmQoREVGQVVMVF9QUk9KRUNUSU9OX09QVElPTlMsIHByb2plY3Rvck9wdGlvbnMpO1xyXG4gICAgfTtcclxuICAgIHZhciBkb20gPSB7XHJcbiAgICAgICAgLyoqXHJcbiAgICAgICAgICogQ3JlYXRlcyBhIHJlYWwgRE9NIHRyZWUgZnJvbSBgdm5vZGVgLiBUaGUgW1tQcm9qZWN0aW9uXV0gb2JqZWN0IHJldHVybmVkIHdpbGwgY29udGFpbiB0aGUgcmVzdWx0aW5nIERPTSBOb2RlIGluXHJcbiAgICAgICAgICogaXRzIFtbUHJvamVjdGlvbi5kb21Ob2RlfGRvbU5vZGVdXSBwcm9wZXJ0eS5cclxuICAgICAgICAgKiBUaGlzIGlzIGEgbG93LWxldmVsIG1ldGhvZC4gVXNlcnMgd2lsbCB0eXBpY2FsbHkgdXNlIGEgW1tQcm9qZWN0b3JdXSBpbnN0ZWFkLlxyXG4gICAgICAgICAqIEBwYXJhbSB2bm9kZSAtIFRoZSByb290IG9mIHRoZSB2aXJ0dWFsIERPTSB0cmVlIHRoYXQgd2FzIGNyZWF0ZWQgdXNpbmcgdGhlIFtbaF1dIGZ1bmN0aW9uLiBOT1RFOiBbW1ZOb2RlXV1cclxuICAgICAgICAgKiBvYmplY3RzIG1heSBvbmx5IGJlIHJlbmRlcmVkIG9uY2UuXHJcbiAgICAgICAgICogQHBhcmFtIHByb2plY3Rpb25PcHRpb25zIC0gT3B0aW9ucyB0byBiZSB1c2VkIHRvIGNyZWF0ZSBhbmQgdXBkYXRlIHRoZSBwcm9qZWN0aW9uLlxyXG4gICAgICAgICAqIEByZXR1cm5zIFRoZSBbW1Byb2plY3Rpb25dXSB3aGljaCBhbHNvIGNvbnRhaW5zIHRoZSBET00gTm9kZSB0aGF0IHdhcyBjcmVhdGVkLlxyXG4gICAgICAgICAqL1xyXG4gICAgICAgIGNyZWF0ZTogZnVuY3Rpb24gKHZub2RlLCBwcm9qZWN0aW9uT3B0aW9ucykge1xyXG4gICAgICAgICAgICBwcm9qZWN0aW9uT3B0aW9ucyA9IGFwcGx5RGVmYXVsdFByb2plY3Rpb25PcHRpb25zKHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICAgICAgY3JlYXRlRG9tKHZub2RlLCBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdkaXYnKSwgdW5kZWZpbmVkLCBwcm9qZWN0aW9uT3B0aW9ucyk7XHJcbiAgICAgICAgICAgIHJldHVybiBjcmVhdGVQcm9qZWN0aW9uKHZub2RlLCBwcm9qZWN0aW9uT3B0aW9ucyk7XHJcbiAgICAgICAgfSxcclxuICAgICAgICAvKipcclxuICAgICAgICAgKiBBcHBlbmRzIGEgbmV3IGNoaWxkIG5vZGUgdG8gdGhlIERPTSB3aGljaCBpcyBnZW5lcmF0ZWQgZnJvbSBhIFtbVk5vZGVdXS5cclxuICAgICAgICAgKiBUaGlzIGlzIGEgbG93LWxldmVsIG1ldGhvZC4gVXNlcnMgd2lsbCB0eXBpY2FsbHkgdXNlIGEgW1tQcm9qZWN0b3JdXSBpbnN0ZWFkLlxyXG4gICAgICAgICAqIEBwYXJhbSBwYXJlbnROb2RlIC0gVGhlIHBhcmVudCBub2RlIGZvciB0aGUgbmV3IGNoaWxkIG5vZGUuXHJcbiAgICAgICAgICogQHBhcmFtIHZub2RlIC0gVGhlIHJvb3Qgb2YgdGhlIHZpcnR1YWwgRE9NIHRyZWUgdGhhdCB3YXMgY3JlYXRlZCB1c2luZyB0aGUgW1toXV0gZnVuY3Rpb24uIE5PVEU6IFtbVk5vZGVdXVxyXG4gICAgICAgICAqIG9iamVjdHMgbWF5IG9ubHkgYmUgcmVuZGVyZWQgb25jZS5cclxuICAgICAgICAgKiBAcGFyYW0gcHJvamVjdGlvbk9wdGlvbnMgLSBPcHRpb25zIHRvIGJlIHVzZWQgdG8gY3JlYXRlIGFuZCB1cGRhdGUgdGhlIFtbUHJvamVjdGlvbl1dLlxyXG4gICAgICAgICAqIEByZXR1cm5zIFRoZSBbW1Byb2plY3Rpb25dXSB0aGF0IHdhcyBjcmVhdGVkLlxyXG4gICAgICAgICAqL1xyXG4gICAgICAgIGFwcGVuZDogZnVuY3Rpb24gKHBhcmVudE5vZGUsIHZub2RlLCBwcm9qZWN0aW9uT3B0aW9ucykge1xyXG4gICAgICAgICAgICBwcm9qZWN0aW9uT3B0aW9ucyA9IGFwcGx5RGVmYXVsdFByb2plY3Rpb25PcHRpb25zKHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICAgICAgY3JlYXRlRG9tKHZub2RlLCBwYXJlbnROb2RlLCB1bmRlZmluZWQsIHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICAgICAgcmV0dXJuIGNyZWF0ZVByb2plY3Rpb24odm5vZGUsIHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICB9LFxyXG4gICAgICAgIC8qKlxyXG4gICAgICAgICAqIEluc2VydHMgYSBuZXcgRE9NIG5vZGUgd2hpY2ggaXMgZ2VuZXJhdGVkIGZyb20gYSBbW1ZOb2RlXV0uXHJcbiAgICAgICAgICogVGhpcyBpcyBhIGxvdy1sZXZlbCBtZXRob2QuIFVzZXJzIHdpbCB0eXBpY2FsbHkgdXNlIGEgW1tQcm9qZWN0b3JdXSBpbnN0ZWFkLlxyXG4gICAgICAgICAqIEBwYXJhbSBiZWZvcmVOb2RlIC0gVGhlIG5vZGUgdGhhdCB0aGUgRE9NIE5vZGUgaXMgaW5zZXJ0ZWQgYmVmb3JlLlxyXG4gICAgICAgICAqIEBwYXJhbSB2bm9kZSAtIFRoZSByb290IG9mIHRoZSB2aXJ0dWFsIERPTSB0cmVlIHRoYXQgd2FzIGNyZWF0ZWQgdXNpbmcgdGhlIFtbaF1dIGZ1bmN0aW9uLlxyXG4gICAgICAgICAqIE5PVEU6IFtbVk5vZGVdXSBvYmplY3RzIG1heSBvbmx5IGJlIHJlbmRlcmVkIG9uY2UuXHJcbiAgICAgICAgICogQHBhcmFtIHByb2plY3Rpb25PcHRpb25zIC0gT3B0aW9ucyB0byBiZSB1c2VkIHRvIGNyZWF0ZSBhbmQgdXBkYXRlIHRoZSBwcm9qZWN0aW9uLCBzZWUgW1tjcmVhdGVQcm9qZWN0b3JdXS5cclxuICAgICAgICAgKiBAcmV0dXJucyBUaGUgW1tQcm9qZWN0aW9uXV0gdGhhdCB3YXMgY3JlYXRlZC5cclxuICAgICAgICAgKi9cclxuICAgICAgICBpbnNlcnRCZWZvcmU6IGZ1bmN0aW9uIChiZWZvcmVOb2RlLCB2bm9kZSwgcHJvamVjdGlvbk9wdGlvbnMpIHtcclxuICAgICAgICAgICAgcHJvamVjdGlvbk9wdGlvbnMgPSBhcHBseURlZmF1bHRQcm9qZWN0aW9uT3B0aW9ucyhwcm9qZWN0aW9uT3B0aW9ucyk7XHJcbiAgICAgICAgICAgIGNyZWF0ZURvbSh2bm9kZSwgYmVmb3JlTm9kZS5wYXJlbnROb2RlLCBiZWZvcmVOb2RlLCBwcm9qZWN0aW9uT3B0aW9ucyk7XHJcbiAgICAgICAgICAgIHJldHVybiBjcmVhdGVQcm9qZWN0aW9uKHZub2RlLCBwcm9qZWN0aW9uT3B0aW9ucyk7XHJcbiAgICAgICAgfSxcclxuICAgICAgICAvKipcclxuICAgICAgICAgKiBNZXJnZXMgYSBuZXcgRE9NIG5vZGUgd2hpY2ggaXMgZ2VuZXJhdGVkIGZyb20gYSBbW1ZOb2RlXV0gd2l0aCBhbiBleGlzdGluZyBET00gTm9kZS5cclxuICAgICAgICAgKiBUaGlzIG1lYW5zIHRoYXQgdGhlIHZpcnR1YWwgRE9NIGFuZCB0aGUgcmVhbCBET00gd2lsbCBoYXZlIG9uZSBvdmVybGFwcGluZyBlbGVtZW50LlxyXG4gICAgICAgICAqIFRoZXJlZm9yZSB0aGUgc2VsZWN0b3IgZm9yIHRoZSByb290IFtbVk5vZGVdXSB3aWxsIGJlIGlnbm9yZWQsIGJ1dCBpdHMgcHJvcGVydGllcyBhbmQgY2hpbGRyZW4gd2lsbCBiZSBhcHBsaWVkIHRvIHRoZSBFbGVtZW50IHByb3ZpZGVkLlxyXG4gICAgICAgICAqIFRoaXMgaXMgYSBsb3ctbGV2ZWwgbWV0aG9kLiBVc2VycyB3aWwgdHlwaWNhbGx5IHVzZSBhIFtbUHJvamVjdG9yXV0gaW5zdGVhZC5cclxuICAgICAgICAgKiBAcGFyYW0gZWxlbWVudCAtIFRoZSBleGlzdGluZyBlbGVtZW50IHRvIGFkb3B0IGFzIHRoZSByb290IG9mIHRoZSBuZXcgdmlydHVhbCBET00uIEV4aXN0aW5nIGF0dHJpYnV0ZXMgYW5kIGNoaWxkIG5vZGVzIGFyZSBwcmVzZXJ2ZWQuXHJcbiAgICAgICAgICogQHBhcmFtIHZub2RlIC0gVGhlIHJvb3Qgb2YgdGhlIHZpcnR1YWwgRE9NIHRyZWUgdGhhdCB3YXMgY3JlYXRlZCB1c2luZyB0aGUgW1toXV0gZnVuY3Rpb24uIE5PVEU6IFtbVk5vZGVdXSBvYmplY3RzXHJcbiAgICAgICAgICogbWF5IG9ubHkgYmUgcmVuZGVyZWQgb25jZS5cclxuICAgICAgICAgKiBAcGFyYW0gcHJvamVjdGlvbk9wdGlvbnMgLSBPcHRpb25zIHRvIGJlIHVzZWQgdG8gY3JlYXRlIGFuZCB1cGRhdGUgdGhlIHByb2plY3Rpb24sIHNlZSBbW2NyZWF0ZVByb2plY3Rvcl1dLlxyXG4gICAgICAgICAqIEByZXR1cm5zIFRoZSBbW1Byb2plY3Rpb25dXSB0aGF0IHdhcyBjcmVhdGVkLlxyXG4gICAgICAgICAqL1xyXG4gICAgICAgIG1lcmdlOiBmdW5jdGlvbiAoZWxlbWVudCwgdm5vZGUsIHByb2plY3Rpb25PcHRpb25zKSB7XHJcbiAgICAgICAgICAgIHByb2plY3Rpb25PcHRpb25zID0gYXBwbHlEZWZhdWx0UHJvamVjdGlvbk9wdGlvbnMocHJvamVjdGlvbk9wdGlvbnMpO1xyXG4gICAgICAgICAgICB2bm9kZS5kb21Ob2RlID0gZWxlbWVudDtcclxuICAgICAgICAgICAgaW5pdFByb3BlcnRpZXNBbmRDaGlsZHJlbihlbGVtZW50LCB2bm9kZSwgcHJvamVjdGlvbk9wdGlvbnMpO1xyXG4gICAgICAgICAgICByZXR1cm4gY3JlYXRlUHJvamVjdGlvbih2bm9kZSwgcHJvamVjdGlvbk9wdGlvbnMpO1xyXG4gICAgICAgIH0sXHJcbiAgICAgICAgLyoqXHJcbiAgICAgICAgICogUmVwbGFjZXMgYW4gZXhpc3RpbmcgRE9NIG5vZGUgd2l0aCBhIG5vZGUgZ2VuZXJhdGVkIGZyb20gYSBbW1ZOb2RlXV0uXHJcbiAgICAgICAgICogVGhpcyBpcyBhIGxvdy1sZXZlbCBtZXRob2QuIFVzZXJzIHdpbGwgdHlwaWNhbGx5IHVzZSBhIFtbUHJvamVjdG9yXV0gaW5zdGVhZC5cclxuICAgICAgICAgKiBAcGFyYW0gZWxlbWVudCAtIFRoZSBub2RlIGZvciB0aGUgW1tWTm9kZV1dIHRvIHJlcGxhY2UuXHJcbiAgICAgICAgICogQHBhcmFtIHZub2RlIC0gVGhlIHJvb3Qgb2YgdGhlIHZpcnR1YWwgRE9NIHRyZWUgdGhhdCB3YXMgY3JlYXRlZCB1c2luZyB0aGUgW1toXV0gZnVuY3Rpb24uIE5PVEU6IFtbVk5vZGVdXVxyXG4gICAgICAgICAqIG9iamVjdHMgbWF5IG9ubHkgYmUgcmVuZGVyZWQgb25jZS5cclxuICAgICAgICAgKiBAcGFyYW0gcHJvamVjdGlvbk9wdGlvbnMgLSBPcHRpb25zIHRvIGJlIHVzZWQgdG8gY3JlYXRlIGFuZCB1cGRhdGUgdGhlIFtbUHJvamVjdGlvbl1dLlxyXG4gICAgICAgICAqIEByZXR1cm5zIFRoZSBbW1Byb2plY3Rpb25dXSB0aGF0IHdhcyBjcmVhdGVkLlxyXG4gICAgICAgICAqL1xyXG4gICAgICAgIHJlcGxhY2U6IGZ1bmN0aW9uIChlbGVtZW50LCB2bm9kZSwgcHJvamVjdGlvbk9wdGlvbnMpIHtcclxuICAgICAgICAgICAgcHJvamVjdGlvbk9wdGlvbnMgPSBhcHBseURlZmF1bHRQcm9qZWN0aW9uT3B0aW9ucyhwcm9qZWN0aW9uT3B0aW9ucyk7XHJcbiAgICAgICAgICAgIGNyZWF0ZURvbSh2bm9kZSwgZWxlbWVudC5wYXJlbnROb2RlLCBlbGVtZW50LCBwcm9qZWN0aW9uT3B0aW9ucyk7XHJcbiAgICAgICAgICAgIGVsZW1lbnQucGFyZW50Tm9kZS5yZW1vdmVDaGlsZChlbGVtZW50KTtcclxuICAgICAgICAgICAgcmV0dXJuIGNyZWF0ZVByb2plY3Rpb24odm5vZGUsIHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICB9XHJcbiAgICB9O1xuXG4gICAgLyogdHNsaW50OmRpc2FibGUgZnVuY3Rpb24tbmFtZSAqL1xyXG4gICAgdmFyIHRvVGV4dFZOb2RlID0gZnVuY3Rpb24gKGRhdGEpIHtcclxuICAgICAgICByZXR1cm4ge1xyXG4gICAgICAgICAgICB2bm9kZVNlbGVjdG9yOiAnJyxcclxuICAgICAgICAgICAgcHJvcGVydGllczogdW5kZWZpbmVkLFxyXG4gICAgICAgICAgICBjaGlsZHJlbjogdW5kZWZpbmVkLFxyXG4gICAgICAgICAgICB0ZXh0OiBkYXRhLnRvU3RyaW5nKCksXHJcbiAgICAgICAgICAgIGRvbU5vZGU6IG51bGxcclxuICAgICAgICB9O1xyXG4gICAgfTtcclxuICAgIHZhciBhcHBlbmRDaGlsZHJlbiA9IGZ1bmN0aW9uIChwYXJlbnRTZWxlY3RvciwgaW5zZXJ0aW9ucywgbWFpbikge1xyXG4gICAgICAgIGZvciAodmFyIGkgPSAwLCBsZW5ndGhfMSA9IGluc2VydGlvbnMubGVuZ3RoOyBpIDwgbGVuZ3RoXzE7IGkrKykge1xyXG4gICAgICAgICAgICB2YXIgaXRlbSA9IGluc2VydGlvbnNbaV07XHJcbiAgICAgICAgICAgIGlmIChBcnJheS5pc0FycmF5KGl0ZW0pKSB7XHJcbiAgICAgICAgICAgICAgICBhcHBlbmRDaGlsZHJlbihwYXJlbnRTZWxlY3RvciwgaXRlbSwgbWFpbik7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICBpZiAoaXRlbSAhPT0gbnVsbCAmJiBpdGVtICE9PSB1bmRlZmluZWQgJiYgaXRlbSAhPT0gZmFsc2UpIHtcclxuICAgICAgICAgICAgICAgICAgICBpZiAodHlwZW9mIGl0ZW0gPT09ICdzdHJpbmcnKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGl0ZW0gPSB0b1RleHRWTm9kZShpdGVtKTtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgbWFpbi5wdXNoKGl0ZW0pO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIGZ1bmN0aW9uIGgoc2VsZWN0b3IsIHByb3BlcnRpZXMsIGNoaWxkcmVuKSB7XHJcbiAgICAgICAgaWYgKEFycmF5LmlzQXJyYXkocHJvcGVydGllcykpIHtcclxuICAgICAgICAgICAgY2hpbGRyZW4gPSBwcm9wZXJ0aWVzO1xyXG4gICAgICAgICAgICBwcm9wZXJ0aWVzID0gdW5kZWZpbmVkO1xyXG4gICAgICAgIH1cclxuICAgICAgICBlbHNlIGlmICgocHJvcGVydGllcyAmJiAodHlwZW9mIHByb3BlcnRpZXMgPT09ICdzdHJpbmcnIHx8IHByb3BlcnRpZXMuaGFzT3duUHJvcGVydHkoJ3Zub2RlU2VsZWN0b3InKSkpIHx8XHJcbiAgICAgICAgICAgIChjaGlsZHJlbiAmJiAodHlwZW9mIGNoaWxkcmVuID09PSAnc3RyaW5nJyB8fCBjaGlsZHJlbi5oYXNPd25Qcm9wZXJ0eSgndm5vZGVTZWxlY3RvcicpKSkpIHtcclxuICAgICAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdoIGNhbGxlZCB3aXRoIGludmFsaWQgYXJndW1lbnRzJyk7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHZhciB0ZXh0O1xyXG4gICAgICAgIHZhciBmbGF0dGVuZWRDaGlsZHJlbjtcclxuICAgICAgICAvLyBSZWNvZ25pemUgYSBjb21tb24gc3BlY2lhbCBjYXNlIHdoZXJlIHRoZXJlIGlzIG9ubHkgYSBzaW5nbGUgdGV4dCBub2RlXHJcbiAgICAgICAgaWYgKGNoaWxkcmVuICYmIGNoaWxkcmVuLmxlbmd0aCA9PT0gMSAmJiB0eXBlb2YgY2hpbGRyZW5bMF0gPT09ICdzdHJpbmcnKSB7XHJcbiAgICAgICAgICAgIHRleHQgPSBjaGlsZHJlblswXTtcclxuICAgICAgICB9XHJcbiAgICAgICAgZWxzZSBpZiAoY2hpbGRyZW4pIHtcclxuICAgICAgICAgICAgZmxhdHRlbmVkQ2hpbGRyZW4gPSBbXTtcclxuICAgICAgICAgICAgYXBwZW5kQ2hpbGRyZW4oc2VsZWN0b3IsIGNoaWxkcmVuLCBmbGF0dGVuZWRDaGlsZHJlbik7XHJcbiAgICAgICAgICAgIGlmIChmbGF0dGVuZWRDaGlsZHJlbi5sZW5ndGggPT09IDApIHtcclxuICAgICAgICAgICAgICAgIGZsYXR0ZW5lZENoaWxkcmVuID0gdW5kZWZpbmVkO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHJldHVybiB7XHJcbiAgICAgICAgICAgIHZub2RlU2VsZWN0b3I6IHNlbGVjdG9yLFxyXG4gICAgICAgICAgICBwcm9wZXJ0aWVzOiBwcm9wZXJ0aWVzLFxyXG4gICAgICAgICAgICBjaGlsZHJlbjogZmxhdHRlbmVkQ2hpbGRyZW4sXHJcbiAgICAgICAgICAgIHRleHQ6ICh0ZXh0ID09PSAnJykgPyB1bmRlZmluZWQgOiB0ZXh0LFxyXG4gICAgICAgICAgICBkb21Ob2RlOiBudWxsXHJcbiAgICAgICAgfTtcclxuICAgIH1cblxuICAgIHZhciBjcmVhdGVQYXJlbnROb2RlUGF0aCA9IGZ1bmN0aW9uIChub2RlLCByb290Tm9kZSkge1xyXG4gICAgICAgIHZhciBwYXJlbnROb2RlUGF0aCA9IFtdO1xyXG4gICAgICAgIHdoaWxlIChub2RlICE9PSByb290Tm9kZSkge1xyXG4gICAgICAgICAgICBwYXJlbnROb2RlUGF0aC5wdXNoKG5vZGUpO1xyXG4gICAgICAgICAgICBub2RlID0gbm9kZS5wYXJlbnROb2RlO1xyXG4gICAgICAgIH1cclxuICAgICAgICByZXR1cm4gcGFyZW50Tm9kZVBhdGg7XHJcbiAgICB9O1xyXG4gICAgdmFyIGZpbmQ7XHJcbiAgICBpZiAoQXJyYXkucHJvdG90eXBlLmZpbmQpIHtcclxuICAgICAgICBmaW5kID0gZnVuY3Rpb24gKGl0ZW1zLCBwcmVkaWNhdGUpIHsgcmV0dXJuIGl0ZW1zLmZpbmQocHJlZGljYXRlKTsgfTtcclxuICAgIH1cclxuICAgIGVsc2Uge1xyXG4gICAgICAgIGZpbmQgPSBmdW5jdGlvbiAoaXRlbXMsIHByZWRpY2F0ZSkgeyByZXR1cm4gaXRlbXMuZmlsdGVyKHByZWRpY2F0ZSlbMF07IH07XHJcbiAgICB9XHJcbiAgICB2YXIgZmluZFZOb2RlQnlQYXJlbnROb2RlUGF0aCA9IGZ1bmN0aW9uICh2bm9kZSwgcGFyZW50Tm9kZVBhdGgpIHtcclxuICAgICAgICB2YXIgcmVzdWx0ID0gdm5vZGU7XHJcbiAgICAgICAgcGFyZW50Tm9kZVBhdGguZm9yRWFjaChmdW5jdGlvbiAobm9kZSkge1xyXG4gICAgICAgICAgICByZXN1bHQgPSAocmVzdWx0ICYmIHJlc3VsdC5jaGlsZHJlbikgPyBmaW5kKHJlc3VsdC5jaGlsZHJlbiwgZnVuY3Rpb24gKGNoaWxkKSB7IHJldHVybiBjaGlsZC5kb21Ob2RlID09PSBub2RlOyB9KSA6IHVuZGVmaW5lZDtcclxuICAgICAgICB9KTtcclxuICAgICAgICByZXR1cm4gcmVzdWx0O1xyXG4gICAgfTtcclxuICAgIHZhciBjcmVhdGVFdmVudEhhbmRsZXJJbnRlcmNlcHRvciA9IGZ1bmN0aW9uIChwcm9qZWN0b3IsIGdldFByb2plY3Rpb24sIHBlcmZvcm1hbmNlTG9nZ2VyKSB7XHJcbiAgICAgICAgdmFyIG1vZGlmaWVkRXZlbnRIYW5kbGVyID0gZnVuY3Rpb24gKGV2dCkge1xyXG4gICAgICAgICAgICBwZXJmb3JtYW5jZUxvZ2dlcignZG9tRXZlbnQnLCBldnQpO1xyXG4gICAgICAgICAgICB2YXIgcHJvamVjdGlvbiA9IGdldFByb2plY3Rpb24oKTtcclxuICAgICAgICAgICAgdmFyIHBhcmVudE5vZGVQYXRoID0gY3JlYXRlUGFyZW50Tm9kZVBhdGgoZXZ0LmN1cnJlbnRUYXJnZXQsIHByb2plY3Rpb24uZG9tTm9kZSk7XHJcbiAgICAgICAgICAgIHBhcmVudE5vZGVQYXRoLnJldmVyc2UoKTtcclxuICAgICAgICAgICAgdmFyIG1hdGNoaW5nVk5vZGUgPSBmaW5kVk5vZGVCeVBhcmVudE5vZGVQYXRoKHByb2plY3Rpb24uZ2V0TGFzdFJlbmRlcigpLCBwYXJlbnROb2RlUGF0aCk7XHJcbiAgICAgICAgICAgIHByb2plY3Rvci5zY2hlZHVsZVJlbmRlcigpO1xyXG4gICAgICAgICAgICB2YXIgcmVzdWx0O1xyXG4gICAgICAgICAgICBpZiAobWF0Y2hpbmdWTm9kZSkge1xyXG4gICAgICAgICAgICAgICAgLyogdHNsaW50OmRpc2FibGUgbm8taW52YWxpZC10aGlzICovXHJcbiAgICAgICAgICAgICAgICByZXN1bHQgPSBtYXRjaGluZ1ZOb2RlLnByb3BlcnRpZXNbXCJvblwiICsgZXZ0LnR5cGVdLmFwcGx5KG1hdGNoaW5nVk5vZGUucHJvcGVydGllcy5iaW5kIHx8IHRoaXMsIGFyZ3VtZW50cyk7XHJcbiAgICAgICAgICAgICAgICAvKiB0c2xpbnQ6ZW5hYmxlIG5vLWludmFsaWQtdGhpcyAqL1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIHBlcmZvcm1hbmNlTG9nZ2VyKCdkb21FdmVudFByb2Nlc3NlZCcsIGV2dCk7XHJcbiAgICAgICAgICAgIHJldHVybiByZXN1bHQ7XHJcbiAgICAgICAgfTtcclxuICAgICAgICByZXR1cm4gZnVuY3Rpb24gKHByb3BlcnR5TmFtZSwgZXZlbnRIYW5kbGVyLCBkb21Ob2RlLCBwcm9wZXJ0aWVzKSB7IHJldHVybiBtb2RpZmllZEV2ZW50SGFuZGxlcjsgfTtcclxuICAgIH07XHJcbiAgICAvKipcclxuICAgICAqIENyZWF0ZXMgYSBbW1Byb2plY3Rvcl1dIGluc3RhbmNlIHVzaW5nIHRoZSBwcm92aWRlZCBwcm9qZWN0aW9uT3B0aW9ucy5cclxuICAgICAqXHJcbiAgICAgKiBGb3IgbW9yZSBpbmZvcm1hdGlvbiwgc2VlIFtbUHJvamVjdG9yXV0uXHJcbiAgICAgKlxyXG4gICAgICogQHBhcmFtIHByb2plY3Rvck9wdGlvbnMgICBPcHRpb25zIHRoYXQgaW5mbHVlbmNlIGhvdyB0aGUgRE9NIGlzIHJlbmRlcmVkIGFuZCB1cGRhdGVkLlxyXG4gICAgICovXHJcbiAgICB2YXIgY3JlYXRlUHJvamVjdG9yID0gZnVuY3Rpb24gKHByb2plY3Rvck9wdGlvbnMpIHtcclxuICAgICAgICB2YXIgcHJvamVjdG9yO1xyXG4gICAgICAgIHZhciBwcm9qZWN0aW9uT3B0aW9ucyA9IGFwcGx5RGVmYXVsdFByb2plY3Rpb25PcHRpb25zKHByb2plY3Rvck9wdGlvbnMpO1xyXG4gICAgICAgIHZhciBwZXJmb3JtYW5jZUxvZ2dlciA9IHByb2plY3Rpb25PcHRpb25zLnBlcmZvcm1hbmNlTG9nZ2VyO1xyXG4gICAgICAgIHZhciByZW5kZXJDb21wbGV0ZWQgPSB0cnVlO1xyXG4gICAgICAgIHZhciBzY2hlZHVsZWQ7XHJcbiAgICAgICAgdmFyIHN0b3BwZWQgPSBmYWxzZTtcclxuICAgICAgICB2YXIgcHJvamVjdGlvbnMgPSBbXTtcclxuICAgICAgICB2YXIgcmVuZGVyRnVuY3Rpb25zID0gW107IC8vIG1hdGNoZXMgdGhlIHByb2plY3Rpb25zIGFycmF5XHJcbiAgICAgICAgdmFyIGFkZFByb2plY3Rpb24gPSBmdW5jdGlvbiAoXHJcbiAgICAgICAgLyogb25lIG9mOiBkb20uYXBwZW5kLCBkb20uaW5zZXJ0QmVmb3JlLCBkb20ucmVwbGFjZSwgZG9tLm1lcmdlICovXHJcbiAgICAgICAgZG9tRnVuY3Rpb24sIFxyXG4gICAgICAgIC8qIHRoZSBwYXJhbWV0ZXIgb2YgdGhlIGRvbUZ1bmN0aW9uICovXHJcbiAgICAgICAgbm9kZSwgcmVuZGVyRnVuY3Rpb24pIHtcclxuICAgICAgICAgICAgdmFyIHByb2plY3Rpb247XHJcbiAgICAgICAgICAgIHZhciBnZXRQcm9qZWN0aW9uID0gZnVuY3Rpb24gKCkgeyByZXR1cm4gcHJvamVjdGlvbjsgfTtcclxuICAgICAgICAgICAgcHJvamVjdGlvbk9wdGlvbnMuZXZlbnRIYW5kbGVySW50ZXJjZXB0b3IgPSBjcmVhdGVFdmVudEhhbmRsZXJJbnRlcmNlcHRvcihwcm9qZWN0b3IsIGdldFByb2plY3Rpb24sIHBlcmZvcm1hbmNlTG9nZ2VyKTtcclxuICAgICAgICAgICAgcHJvamVjdGlvbiA9IGRvbUZ1bmN0aW9uKG5vZGUsIHJlbmRlckZ1bmN0aW9uKCksIHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICAgICAgcHJvamVjdGlvbnMucHVzaChwcm9qZWN0aW9uKTtcclxuICAgICAgICAgICAgcmVuZGVyRnVuY3Rpb25zLnB1c2gocmVuZGVyRnVuY3Rpb24pO1xyXG4gICAgICAgIH07XHJcbiAgICAgICAgdmFyIGRvUmVuZGVyID0gZnVuY3Rpb24gKCkge1xyXG4gICAgICAgICAgICBzY2hlZHVsZWQgPSB1bmRlZmluZWQ7XHJcbiAgICAgICAgICAgIGlmICghcmVuZGVyQ29tcGxldGVkKSB7XHJcbiAgICAgICAgICAgICAgICByZXR1cm47IC8vIFRoZSBsYXN0IHJlbmRlciB0aHJldyBhbiBlcnJvciwgaXQgc2hvdWxkIGhhdmUgYmVlbiBsb2dnZWQgaW4gdGhlIGJyb3dzZXIgY29uc29sZS5cclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICByZW5kZXJDb21wbGV0ZWQgPSBmYWxzZTtcclxuICAgICAgICAgICAgcGVyZm9ybWFuY2VMb2dnZXIoJ3JlbmRlclN0YXJ0JywgdW5kZWZpbmVkKTtcclxuICAgICAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCBwcm9qZWN0aW9ucy5sZW5ndGg7IGkrKykge1xyXG4gICAgICAgICAgICAgICAgdmFyIHVwZGF0ZWRWbm9kZSA9IHJlbmRlckZ1bmN0aW9uc1tpXSgpO1xyXG4gICAgICAgICAgICAgICAgcGVyZm9ybWFuY2VMb2dnZXIoJ3JlbmRlcmVkJywgdW5kZWZpbmVkKTtcclxuICAgICAgICAgICAgICAgIHByb2plY3Rpb25zW2ldLnVwZGF0ZSh1cGRhdGVkVm5vZGUpO1xyXG4gICAgICAgICAgICAgICAgcGVyZm9ybWFuY2VMb2dnZXIoJ3BhdGNoZWQnLCB1bmRlZmluZWQpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIHBlcmZvcm1hbmNlTG9nZ2VyKCdyZW5kZXJEb25lJywgdW5kZWZpbmVkKTtcclxuICAgICAgICAgICAgcmVuZGVyQ29tcGxldGVkID0gdHJ1ZTtcclxuICAgICAgICB9O1xyXG4gICAgICAgIHByb2plY3RvciA9IHtcclxuICAgICAgICAgICAgcmVuZGVyTm93OiBkb1JlbmRlcixcclxuICAgICAgICAgICAgc2NoZWR1bGVSZW5kZXI6IGZ1bmN0aW9uICgpIHtcclxuICAgICAgICAgICAgICAgIGlmICghc2NoZWR1bGVkICYmICFzdG9wcGVkKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgc2NoZWR1bGVkID0gcmVxdWVzdEFuaW1hdGlvbkZyYW1lKGRvUmVuZGVyKTtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAgc3RvcDogZnVuY3Rpb24gKCkge1xyXG4gICAgICAgICAgICAgICAgaWYgKHNjaGVkdWxlZCkge1xyXG4gICAgICAgICAgICAgICAgICAgIGNhbmNlbEFuaW1hdGlvbkZyYW1lKHNjaGVkdWxlZCk7XHJcbiAgICAgICAgICAgICAgICAgICAgc2NoZWR1bGVkID0gdW5kZWZpbmVkO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgc3RvcHBlZCA9IHRydWU7XHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIHJlc3VtZTogZnVuY3Rpb24gKCkge1xyXG4gICAgICAgICAgICAgICAgc3RvcHBlZCA9IGZhbHNlO1xyXG4gICAgICAgICAgICAgICAgcmVuZGVyQ29tcGxldGVkID0gdHJ1ZTtcclxuICAgICAgICAgICAgICAgIHByb2plY3Rvci5zY2hlZHVsZVJlbmRlcigpO1xyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICBhcHBlbmQ6IGZ1bmN0aW9uIChwYXJlbnROb2RlLCByZW5kZXJGdW5jdGlvbikge1xyXG4gICAgICAgICAgICAgICAgYWRkUHJvamVjdGlvbihkb20uYXBwZW5kLCBwYXJlbnROb2RlLCByZW5kZXJGdW5jdGlvbik7XHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIGluc2VydEJlZm9yZTogZnVuY3Rpb24gKGJlZm9yZU5vZGUsIHJlbmRlckZ1bmN0aW9uKSB7XHJcbiAgICAgICAgICAgICAgICBhZGRQcm9qZWN0aW9uKGRvbS5pbnNlcnRCZWZvcmUsIGJlZm9yZU5vZGUsIHJlbmRlckZ1bmN0aW9uKTtcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAgbWVyZ2U6IGZ1bmN0aW9uIChkb21Ob2RlLCByZW5kZXJGdW5jdGlvbikge1xyXG4gICAgICAgICAgICAgICAgYWRkUHJvamVjdGlvbihkb20ubWVyZ2UsIGRvbU5vZGUsIHJlbmRlckZ1bmN0aW9uKTtcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAgcmVwbGFjZTogZnVuY3Rpb24gKGRvbU5vZGUsIHJlbmRlckZ1bmN0aW9uKSB7XHJcbiAgICAgICAgICAgICAgICBhZGRQcm9qZWN0aW9uKGRvbS5yZXBsYWNlLCBkb21Ob2RlLCByZW5kZXJGdW5jdGlvbik7XHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIGRldGFjaDogZnVuY3Rpb24gKHJlbmRlckZ1bmN0aW9uKSB7XHJcbiAgICAgICAgICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IHJlbmRlckZ1bmN0aW9ucy5sZW5ndGg7IGkrKykge1xyXG4gICAgICAgICAgICAgICAgICAgIGlmIChyZW5kZXJGdW5jdGlvbnNbaV0gPT09IHJlbmRlckZ1bmN0aW9uKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHJlbmRlckZ1bmN0aW9ucy5zcGxpY2UoaSwgMSk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHJldHVybiBwcm9qZWN0aW9ucy5zcGxpY2UoaSwgMSlbMF07XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdyZW5kZXJGdW5jdGlvbiB3YXMgbm90IGZvdW5kJyk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9O1xyXG4gICAgICAgIHJldHVybiBwcm9qZWN0b3I7XHJcbiAgICB9O1xuXG4gICAgLyoqXHJcbiAgICAgKiBDcmVhdGVzIGEgW1tDYWxjdWxhdGlvbkNhY2hlXV0gb2JqZWN0LCB1c2VmdWwgZm9yIGNhY2hpbmcgW1tWTm9kZV1dIHRyZWVzLlxyXG4gICAgICogSW4gcHJhY3RpY2UsIGNhY2hpbmcgb2YgW1tWTm9kZV1dIHRyZWVzIGlzIG5vdCBuZWVkZWQsIGJlY2F1c2UgYWNoaWV2aW5nIDYwIGZyYW1lcyBwZXIgc2Vjb25kIGlzIGFsbW9zdCBuZXZlciBhIHByb2JsZW0uXHJcbiAgICAgKiBGb3IgbW9yZSBpbmZvcm1hdGlvbiwgc2VlIFtbQ2FsY3VsYXRpb25DYWNoZV1dLlxyXG4gICAgICpcclxuICAgICAqIEBwYXJhbSA8UmVzdWx0PiBUaGUgdHlwZSBvZiB0aGUgdmFsdWUgdGhhdCBpcyBjYWNoZWQuXHJcbiAgICAgKi9cclxuICAgIHZhciBjcmVhdGVDYWNoZSA9IGZ1bmN0aW9uICgpIHtcclxuICAgICAgICB2YXIgY2FjaGVkSW5wdXRzO1xyXG4gICAgICAgIHZhciBjYWNoZWRPdXRjb21lO1xyXG4gICAgICAgIHJldHVybiB7XHJcbiAgICAgICAgICAgIGludmFsaWRhdGU6IGZ1bmN0aW9uICgpIHtcclxuICAgICAgICAgICAgICAgIGNhY2hlZE91dGNvbWUgPSB1bmRlZmluZWQ7XHJcbiAgICAgICAgICAgICAgICBjYWNoZWRJbnB1dHMgPSB1bmRlZmluZWQ7XHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIHJlc3VsdDogZnVuY3Rpb24gKGlucHV0cywgY2FsY3VsYXRpb24pIHtcclxuICAgICAgICAgICAgICAgIGlmIChjYWNoZWRJbnB1dHMpIHtcclxuICAgICAgICAgICAgICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IGlucHV0cy5sZW5ndGg7IGkrKykge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBpZiAoY2FjaGVkSW5wdXRzW2ldICE9PSBpbnB1dHNbaV0pIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNhY2hlZE91dGNvbWUgPSB1bmRlZmluZWQ7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICBpZiAoIWNhY2hlZE91dGNvbWUpIHtcclxuICAgICAgICAgICAgICAgICAgICBjYWNoZWRPdXRjb21lID0gY2FsY3VsYXRpb24oKTtcclxuICAgICAgICAgICAgICAgICAgICBjYWNoZWRJbnB1dHMgPSBpbnB1dHM7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICByZXR1cm4gY2FjaGVkT3V0Y29tZTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH07XHJcbiAgICB9O1xuXG4gICAgLyoqXHJcbiAgICAgKiBDcmVhdGVzIGEge0BsaW5rIE1hcHBpbmd9IGluc3RhbmNlIHRoYXQga2VlcHMgYW4gYXJyYXkgb2YgcmVzdWx0IG9iamVjdHMgc3luY2hyb25pemVkIHdpdGggYW4gYXJyYXkgb2Ygc291cmNlIG9iamVjdHMuXHJcbiAgICAgKiBTZWUge0BsaW5rIGh0dHA6Ly9tYXF1ZXR0ZWpzLm9yZy9kb2NzL2FycmF5cy5odG1sfFdvcmtpbmcgd2l0aCBhcnJheXN9LlxyXG4gICAgICpcclxuICAgICAqIEBwYXJhbSA8U291cmNlPiAgICAgICBUaGUgdHlwZSBvZiBzb3VyY2UgaXRlbXMuIEEgZGF0YWJhc2UtcmVjb3JkIGZvciBpbnN0YW5jZS5cclxuICAgICAqIEBwYXJhbSA8VGFyZ2V0PiAgICAgICBUaGUgdHlwZSBvZiB0YXJnZXQgaXRlbXMuIEEgW1tNYXF1ZXR0ZUNvbXBvbmVudF1dIGZvciBpbnN0YW5jZS5cclxuICAgICAqIEBwYXJhbSBnZXRTb3VyY2VLZXkgICBgZnVuY3Rpb24oc291cmNlKWAgdGhhdCBtdXN0IHJldHVybiBhIGtleSB0byBpZGVudGlmeSBlYWNoIHNvdXJjZSBvYmplY3QuIFRoZSByZXN1bHQgbXVzdCBlaXRoZXIgYmUgYSBzdHJpbmcgb3IgYSBudW1iZXIuXHJcbiAgICAgKiBAcGFyYW0gY3JlYXRlUmVzdWx0ICAgYGZ1bmN0aW9uKHNvdXJjZSwgaW5kZXgpYCB0aGF0IG11c3QgY3JlYXRlIGEgbmV3IHJlc3VsdCBvYmplY3QgZnJvbSBhIGdpdmVuIHNvdXJjZS4gVGhpcyBmdW5jdGlvbiBpcyBpZGVudGljYWxcclxuICAgICAqICAgICAgICAgICAgICAgICAgICAgICB0byB0aGUgYGNhbGxiYWNrYCBhcmd1bWVudCBpbiBgQXJyYXkubWFwKGNhbGxiYWNrKWAuXHJcbiAgICAgKiBAcGFyYW0gdXBkYXRlUmVzdWx0ICAgYGZ1bmN0aW9uKHNvdXJjZSwgdGFyZ2V0LCBpbmRleClgIHRoYXQgdXBkYXRlcyBhIHJlc3VsdCB0byBhbiB1cGRhdGVkIHNvdXJjZS5cclxuICAgICAqL1xyXG4gICAgdmFyIGNyZWF0ZU1hcHBpbmcgPSBmdW5jdGlvbiAoZ2V0U291cmNlS2V5LCBjcmVhdGVSZXN1bHQsIHVwZGF0ZVJlc3VsdCkge1xyXG4gICAgICAgIHZhciBrZXlzID0gW107XHJcbiAgICAgICAgdmFyIHJlc3VsdHMgPSBbXTtcclxuICAgICAgICByZXR1cm4ge1xyXG4gICAgICAgICAgICByZXN1bHRzOiByZXN1bHRzLFxyXG4gICAgICAgICAgICBtYXA6IGZ1bmN0aW9uIChuZXdTb3VyY2VzKSB7XHJcbiAgICAgICAgICAgICAgICB2YXIgbmV3S2V5cyA9IG5ld1NvdXJjZXMubWFwKGdldFNvdXJjZUtleSk7XHJcbiAgICAgICAgICAgICAgICB2YXIgb2xkVGFyZ2V0cyA9IHJlc3VsdHMuc2xpY2UoKTtcclxuICAgICAgICAgICAgICAgIHZhciBvbGRJbmRleCA9IDA7XHJcbiAgICAgICAgICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IG5ld1NvdXJjZXMubGVuZ3RoOyBpKyspIHtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgc291cmNlID0gbmV3U291cmNlc1tpXTtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgc291cmNlS2V5ID0gbmV3S2V5c1tpXTtcclxuICAgICAgICAgICAgICAgICAgICBpZiAoc291cmNlS2V5ID09PSBrZXlzW29sZEluZGV4XSkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICByZXN1bHRzW2ldID0gb2xkVGFyZ2V0c1tvbGRJbmRleF07XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHVwZGF0ZVJlc3VsdChzb3VyY2UsIG9sZFRhcmdldHNbb2xkSW5kZXhdLCBpKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgb2xkSW5kZXgrKztcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHZhciBmb3VuZCA9IGZhbHNlO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBmb3IgKHZhciBqID0gMTsgaiA8IGtleXMubGVuZ3RoICsgMTsgaisrKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB2YXIgc2VhcmNoSW5kZXggPSAob2xkSW5kZXggKyBqKSAlIGtleXMubGVuZ3RoO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgaWYgKGtleXNbc2VhcmNoSW5kZXhdID09PSBzb3VyY2VLZXkpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICByZXN1bHRzW2ldID0gb2xkVGFyZ2V0c1tzZWFyY2hJbmRleF07XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdXBkYXRlUmVzdWx0KG5ld1NvdXJjZXNbaV0sIG9sZFRhcmdldHNbc2VhcmNoSW5kZXhdLCBpKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBvbGRJbmRleCA9IHNlYXJjaEluZGV4ICsgMTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBmb3VuZCA9IHRydWU7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgYnJlYWs7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICAgICAgaWYgKCFmb3VuZCkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgcmVzdWx0c1tpXSA9IGNyZWF0ZVJlc3VsdChzb3VyY2UsIGkpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgcmVzdWx0cy5sZW5ndGggPSBuZXdTb3VyY2VzLmxlbmd0aDtcclxuICAgICAgICAgICAgICAgIGtleXMgPSBuZXdLZXlzO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfTtcclxuICAgIH07XG5cbiAgICBleHBvcnRzLmNyZWF0ZUNhY2hlID0gY3JlYXRlQ2FjaGU7XG4gICAgZXhwb3J0cy5jcmVhdGVNYXBwaW5nID0gY3JlYXRlTWFwcGluZztcbiAgICBleHBvcnRzLmNyZWF0ZVByb2plY3RvciA9IGNyZWF0ZVByb2plY3RvcjtcbiAgICBleHBvcnRzLmRvbSA9IGRvbTtcbiAgICBleHBvcnRzLmggPSBoO1xuXG4gICAgT2JqZWN0LmRlZmluZVByb3BlcnR5KGV4cG9ydHMsICdfX2VzTW9kdWxlJywgeyB2YWx1ZTogdHJ1ZSB9KTtcblxufSkpO1xuIl0sInNvdXJjZVJvb3QiOiIifQ==