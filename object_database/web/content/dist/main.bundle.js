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
                console.log('Show dropdown triggered');
                cellSocket.sendString(JSON.stringify({
                    event:'dropdown',
                    target_cell: component.props.id,
                    isOpen: false
                }));
            });
            $(parentEl).on('hide.bs.dropdown', function(){
                console.log('hide dropdown triggered');
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly8vd2VicGFjay9ib290c3RyYXAiLCJ3ZWJwYWNrOi8vLy4vQ2VsbEhhbmRsZXIuanMiLCJ3ZWJwYWNrOi8vLy4vQ2VsbFNvY2tldC5qcyIsIndlYnBhY2s6Ly8vLi9Db21wb25lbnRSZWdpc3RyeS5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0FzeW5jRHJvcGRvd24uanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9CYWRnZS5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0J1dHRvbi5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0J1dHRvbkdyb3VwLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvQ2FyZC5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0NhcmRUaXRsZS5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0NpcmNsZUxvYWRlci5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0NsaWNrYWJsZS5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0NvZGUuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9Db2RlRWRpdG9yLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvQ29sbGFwc2libGVQYW5lbC5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0NvbHVtbnMuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9Db21wb25lbnQuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9Db250YWluZXIuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9Db250ZXh0dWFsRGlzcGxheS5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0Ryb3Bkb3duLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvRXhwYW5kcy5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0dyaWQuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9IZWFkZXJCYXIuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9MYXJnZVBlbmRpbmdEb3dubG9hZERpc3BsYXkuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9Mb2FkQ29udGVudHNGcm9tVXJsLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvTWFpbi5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL01vZGFsLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvT2N0aWNvbi5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL1BhZGRpbmcuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9QbG90LmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvUG9wb3Zlci5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL1Jvb3RDZWxsLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvU2Nyb2xsYWJsZS5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL1NlcXVlbmNlLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvU2hlZXQuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9TaW5nbGVMaW5lVGV4dEJveC5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL1NwYW4uanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9TdWJzY3JpYmVkLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvU3Vic2NyaWJlZFNlcXVlbmNlLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvVGFibGUuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9UYWJzLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvVGV4dC5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL1RyYWNlYmFjay5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL19OYXZUYWIuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9fUGxvdFVwZGF0ZXIuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy91dGlsL1Byb3BlcnR5VmFsaWRhdG9yLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvdXRpbC9SZXBsYWNlbWVudHNIYW5kbGVyLmpzIiwid2VicGFjazovLy8uL21haW4uanMiLCJ3ZWJwYWNrOi8vLy4vbm9kZV9tb2R1bGVzL21hcXVldHRlL2Rpc3QvbWFxdWV0dGUudW1kLmpzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiI7QUFBQTtBQUNBOztBQUVBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTs7QUFFQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTs7O0FBR0E7QUFDQTs7QUFFQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBLGtEQUEwQyxnQ0FBZ0M7QUFDMUU7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQSxnRUFBd0Qsa0JBQWtCO0FBQzFFO0FBQ0EseURBQWlELGNBQWM7QUFDL0Q7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGlEQUF5QyxpQ0FBaUM7QUFDMUUsd0hBQWdILG1CQUFtQixFQUFFO0FBQ3JJO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0EsbUNBQTJCLDBCQUEwQixFQUFFO0FBQ3ZELHlDQUFpQyxlQUFlO0FBQ2hEO0FBQ0E7QUFDQTs7QUFFQTtBQUNBLDhEQUFzRCwrREFBK0Q7O0FBRXJIO0FBQ0E7OztBQUdBO0FBQ0E7Ozs7Ozs7Ozs7Ozs7QUNsRkE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDMkI7O0FBRTNCO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2I7QUFDQTtBQUNBLGFBQWE7QUFDYjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBO0FBQ0EsU0FBUztBQUNUOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGVBQWUsT0FBTztBQUN0QjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsTUFBTTtBQUNOO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxlQUFlLE9BQU87QUFDdEI7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxrRkFBa0YsV0FBVztBQUM3RjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGFBQWEsa0RBQUMsU0FBUyxzQkFBc0I7QUFDN0MsR0FBRztBQUNIO0FBQ0EsRUFBRTtBQUNGO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsNkRBQTZELHVCQUF1QjtBQUNwRjtBQUNBLE1BQU07QUFDTjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsaUJBQWlCO0FBQ2pCO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsaUJBQWlCOztBQUVqQjtBQUNBLGFBQWE7QUFDYjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxPQUFPO0FBQ1AsR0FBRztBQUNILDBDQUEwQyxpQkFBaUI7QUFDM0Q7QUFDQTs7QUFFQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsOEJBQThCLG1CQUFtQjtBQUNqRCxzRTtBQUNBO0FBQ0E7QUFDQSxxQkFBcUI7QUFDckIsR0FBRztBQUNIO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQSxxQkFBcUI7QUFDckIsaUJBQWlCO0FBQ2pCLGlEQUFpRCxRQUFRLGlCQUFpQixlQUFlO0FBQ3pGO0FBQ0E7QUFDQTtBQUNBLGFBQWE7QUFDYjs7QUFFQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLFNBQVM7O0FBRVQ7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGtDQUFrQyxhQUFhO0FBQy9DLG9CQUFvQiwrQ0FBK0M7QUFDbkU7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGVBQWUsT0FBTztBQUN0QjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsaUJBQWlCO0FBQ2pCLGFBQWE7QUFDYjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsaUJBQWlCO0FBQ2pCLGFBQWE7O0FBRWI7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGVBQWUsT0FBTztBQUN0QjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0EsY0FBYztBQUNkOztBQUVBLGdCQUFnQixpQ0FBaUM7QUFDakQ7QUFDQTtBQUNBOztBQUVBO0FBQ0EsWUFBWSxrREFBQztBQUNiOztBQUVBO0FBQ0EsZ0JBQWdCLCtCQUErQjtBQUMvQztBQUNBO0FBQ0E7O0FBRUEsUUFBUSxrREFBQztBQUNUO0FBQ0E7O0FBRTRDOzs7Ozs7Ozs7Ozs7O0FDMVc1QztBQUFBO0FBQUE7QUFBQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7OztBQUdBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLG1CQUFtQixPQUFPO0FBQzFCO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQSxtQkFBbUIsT0FBTztBQUMxQjtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsaUJBQWlCLE9BQU87QUFDeEI7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQSxpQkFBaUIsSUFBSSxJQUFJLGNBQWM7QUFDdkMsaUJBQWlCLElBQUksU0FBUyxrQkFBa0IsRUFBRSxnQkFBZ0I7QUFDbEU7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGVBQWUsT0FBTztBQUN0QjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0EsU0FBUztBQUNUO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQSxlQUFlLE9BQU87QUFDdEI7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGVBQWUsT0FBTztBQUN0QjtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBOztBQUVBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsZUFBZSxNQUFNO0FBQ3JCO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGlCQUFpQjtBQUNqQjtBQUNBO0FBQ0EsNkNBQTZDLFdBQVcsT0FBTyxNQUFNO0FBQ3JFO0FBQ0E7O0FBRUE7QUFDQTtBQUNBOztBQUVBOztBQUVBOztBQUVBO0FBQ0E7O0FBRUE7O0FBRUE7QUFDQSw0Q0FBNEMsU0FBUztBQUNyRCxTQUFTO0FBQ1Q7QUFDQTtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsZUFBZSxtQkFBbUI7QUFDbEM7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsZUFBZSxlQUFlO0FBQzlCO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQSxlQUFlLGFBQWE7QUFDNUI7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGVBQWUsYUFBYTtBQUM1QjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7OztBQUcwQzs7Ozs7Ozs7Ozs7OztBQ25TMUM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUMrRTtBQUN0QztBQUNFO0FBQ1U7QUFDZDtBQUNVO0FBQ007QUFDTjtBQUNWO0FBQ1k7QUFDWTtBQUNsQjtBQUNJO0FBQ2dCO0FBQ2xCO0FBQ0Y7QUFDSTtBQUNvQjtBQUNnQjtBQUM5QztBQUNFO0FBQ0k7QUFDQTtBQUNBO0FBQ0U7QUFDQTtBQUNJO0FBQ2M7QUFDMUI7QUFDWTtBQUNnQjtBQUMxQjtBQUNGO0FBQ0E7QUFDVTtBQUNKO0FBQ047QUFDRTtBQUNGO0FBQ2dCOztBQUV2RDtBQUNBLElBQUksc0ZBQWE7QUFDakIsSUFBSSxvR0FBb0I7QUFDeEIsSUFBSSw4REFBSztBQUNULElBQUksaUVBQU07QUFDVixJQUFJLGdGQUFXO0FBQ2YsSUFBSSwyREFBSTtBQUNSLElBQUksMEVBQVM7QUFDYixJQUFJLG1GQUFZO0FBQ2hCLElBQUksMEVBQVM7QUFDYixJQUFJLDJEQUFJO0FBQ1IsSUFBSSw2RUFBVTtBQUNkLElBQUksZ0dBQWdCO0FBQ3BCLElBQUkscUVBQU87QUFDWCxJQUFJLDJFQUFTO0FBQ2IsSUFBSSxtR0FBaUI7QUFDckIsSUFBSSx3RUFBUTtBQUNaLElBQUkscUVBQU87QUFDWCxJQUFJLDJFQUFTO0FBQ2IsSUFBSSx5R0FBbUI7QUFDdkIsSUFBSSxpSUFBMkI7QUFDL0IsSUFBSSw0REFBSTtBQUNSLElBQUksK0RBQUs7QUFDVCxJQUFJLHFFQUFPO0FBQ1gsSUFBSSxxRUFBTztBQUNYLElBQUkscUVBQU87QUFDWCxJQUFJLHdFQUFRO0FBQ1osSUFBSSx3RUFBUTtBQUNaLElBQUksOEVBQVU7QUFDZCxJQUFJLG1HQUFpQjtBQUNyQixJQUFJLDREQUFJO0FBQ1IsSUFBSSw4RUFBVTtBQUNkLElBQUksc0dBQWtCO0FBQ3RCLElBQUksK0RBQUs7QUFDVCxJQUFJLDREQUFJO0FBQ1IsSUFBSSw0REFBSTtBQUNSLElBQUksMkVBQVM7QUFDYixJQUFJLG9FQUFPO0FBQ1gsSUFBSSw0REFBSTtBQUNSLElBQUksK0RBQUs7QUFDVCxJQUFJLDREQUFJO0FBQ1IsSUFBSSxtRkFBWTtBQUNoQjs7QUFFeUQ7Ozs7Ozs7Ozs7Ozs7QUM1RnpEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFDQTtBQUNBOztBQUVzQztBQUNYOztBQUUzQjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLDRCQUE0QixvREFBUztBQUNyQztBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQSxZQUFZLGtEQUFDO0FBQ2I7QUFDQTtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2IsZ0JBQWdCLGtEQUFDLE9BQU8sMENBQTBDO0FBQ2xFLGdCQUFnQixrREFBQztBQUNqQjtBQUNBO0FBQ0EsMkJBQTJCLGNBQWM7QUFDekM7QUFDQTtBQUNBO0FBQ0EsaUJBQWlCO0FBQ2pCLGdCQUFnQixrREFBQztBQUNqQiwyQkFBMkIsY0FBYztBQUN6QztBQUNBLGlCQUFpQjtBQUNqQjtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGlCQUFpQjtBQUNqQixhQUFhO0FBQ2I7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsaUJBQWlCO0FBQ2pCLGFBQWE7QUFDYjtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsbUNBQW1DLG9EQUFTO0FBQzVDO0FBQ0E7O0FBRUE7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQSxZQUFZLGtEQUFDO0FBQ2IsdUNBQXVDLGNBQWM7QUFDckQ7QUFDQTtBQUNBLGFBQWE7QUFDYjtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBO0FBQ0E7QUFDQTs7O0FBT0U7Ozs7Ozs7Ozs7Ozs7QUM1SUY7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFDQTtBQUNBO0FBQ3NDO0FBQ1g7O0FBRTNCO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0Esb0JBQW9CLG9EQUFTO0FBQzdCO0FBQ0E7O0FBRUE7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQSxZQUFZLGtEQUFDO0FBQ2IsMkNBQTJDLGdDQUFnQztBQUMzRTtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2I7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBO0FBQ0E7O0FBRWlDOzs7Ozs7Ozs7Ozs7O0FDOUNqQztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUNBO0FBQ0E7O0FBRXNDO0FBQ1g7O0FBRTNCO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EscUJBQXFCLG9EQUFTO0FBQzlCO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0EsWUFBWSxrREFBQztBQUNiO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2I7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFbUM7Ozs7Ozs7Ozs7Ozs7QUM3RG5DO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQ0E7QUFDQTs7QUFFc0M7QUFDWDs7QUFFM0I7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLDBCQUEwQixvREFBUztBQUNuQztBQUNBOztBQUVBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0EsWUFBWSxrREFBQztBQUNiO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2I7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBO0FBQ0E7O0FBRUE7O0FBRTZDOzs7Ozs7Ozs7Ozs7O0FDbkQ3QztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQ0E7QUFDQTs7QUFFc0M7QUFDYTtBQUN4Qjs7QUFFM0I7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxtQkFBbUIsb0RBQVM7QUFDNUI7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQSx1Q0FBdUMsNkJBQTZCO0FBQ3BFO0FBQ0EsdUJBQXVCLGtEQUFDO0FBQ3hCO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTtBQUNBLHlCQUF5QixrREFBQyxTQUFTLHFCQUFxQjtBQUN4RDtBQUNBLGVBQWUsa0RBQUM7QUFDaEI7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsYUFBYTtBQUNiOztBQUVBO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBLGNBQWMsaUVBQVMsUUFBUSxpRUFBUyxTQUFTLGlFQUFTO0FBQzFELEtBQUs7QUFDTDtBQUNBO0FBQ0EsY0FBYyxpRUFBUyxRQUFRLGlFQUFTO0FBQ3hDO0FBQ0E7O0FBRStCOzs7Ozs7Ozs7Ozs7O0FDdEYvQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUNBO0FBQ0E7O0FBRXNDO0FBQ1g7OztBQUczQjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0Esd0JBQXdCLG9EQUFTO0FBQ2pDO0FBQ0E7O0FBRUE7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQSxZQUFZLGtEQUFDO0FBQ2I7QUFDQTtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2I7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTtBQUNBOztBQUV5Qzs7Ozs7Ozs7Ozs7OztBQ25EekM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFDQTtBQUNBOztBQUVzQztBQUNYOzs7QUFHM0IsMkJBQTJCLG9EQUFTO0FBQ3BDO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0EsWUFBWSxrREFBQztBQUNiO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2I7QUFDQTtBQUNBOztBQUVBO0FBQ0E7O0FBRStDOzs7Ozs7Ozs7Ozs7O0FDN0IvQztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUNBO0FBQ0E7QUFDc0M7QUFDWDs7QUFFM0I7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0Esd0JBQXdCLG9EQUFTO0FBQ2pDO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBLFlBQVksa0RBQUM7QUFDYjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsYUFBYTtBQUNiLGdCQUFnQixrREFBQyxVQUFVO0FBQzNCO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7O0FBRXlDOzs7Ozs7Ozs7Ozs7O0FDekR6QztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUNBO0FBQ0E7O0FBRXNDO0FBQ1g7O0FBRTNCO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxtQkFBbUIsb0RBQVM7QUFDNUI7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQSxlQUFlLGtEQUFDO0FBQ2hCO0FBQ0E7QUFDQTtBQUNBO0FBQ0Esa0JBQWtCO0FBQ2xCLHFCQUFxQixrREFBQyxXQUFXO0FBQ2pDO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBO0FBQ0E7O0FBRStCOzs7Ozs7Ozs7Ozs7O0FDakQvQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUNBO0FBQ0E7O0FBRXNDO0FBQ1g7O0FBRTNCLHlCQUF5QixvREFBUztBQUNsQztBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsdUNBQXVDLFlBQVksWUFBWSwyQkFBMkI7O0FBRTFGO0FBQ0E7QUFDQTtBQUNBOztBQUVBOztBQUVBOztBQUVBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0Esd0NBQXdDLGdDQUFnQztBQUN4RSx3Q0FBd0MsK0JBQStCO0FBQ3ZFOztBQUVBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBOztBQUVBOztBQUVBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBLGVBQWUsa0RBQUM7QUFDaEI7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsYUFBYTtBQUNiLGFBQWEsa0RBQUMsU0FBUyx3REFBd0Q7QUFDL0U7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxHQUFHLDhCQUE4QjtBQUNqQztBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsOEJBQThCLHlDQUF5QztBQUN2RTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7O0FBRTJDOzs7Ozs7Ozs7Ozs7O0FDM0ozQztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUNBO0FBQ0E7QUFDeUM7QUFDZDs7QUFFM0I7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsK0JBQStCLHVEQUFTO0FBQ3hDO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0EsZ0JBQWdCLGtEQUFDO0FBQ2pCO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGlCQUFpQjtBQUNqQixvQkFBb0Isa0RBQUMsU0FBUyxvQ0FBb0M7QUFDbEUsd0JBQXdCLGtEQUFDLFNBQVMscUJBQXFCO0FBQ3ZEO0FBQ0E7QUFDQSx3QkFBd0Isa0RBQUMsU0FBUyxnQkFBZ0I7QUFDbEQ7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBLGdCQUFnQixrREFBQztBQUNqQjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxpQkFBaUI7QUFDakI7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTtBQUNBOzs7QUFHc0Q7Ozs7Ozs7Ozs7Ozs7QUNyRnREO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQ0E7QUFDQTs7QUFFc0M7QUFDWDs7QUFFM0I7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLHNCQUFzQixvREFBUztBQUMvQjtBQUNBOztBQUVBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0EsWUFBWSxrREFBQztBQUNiO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2IsZ0JBQWdCLGtEQUFDLFNBQVMseUJBQXlCO0FBQ25EO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLG9CQUFvQixrREFBQztBQUNyQjtBQUNBLHFCQUFxQjtBQUNyQjtBQUNBLGFBQWE7QUFDYixTQUFTO0FBQ1Q7QUFDQTtBQUNBO0FBQ0E7OztBQUdxQzs7Ozs7Ozs7Ozs7OztBQzFEckM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQytEO0FBQ1o7QUFDeEI7O0FBRTNCO0FBQ0EsMEJBQTBCO0FBQzFCO0FBQ0E7O0FBRUE7QUFDQSxnQ0FBZ0MsNkVBQW1CO0FBQ25EOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSwyQkFBMkIsY0FBYyxHQUFHLFlBQVk7QUFDeEQsbUJBQW1CLGtEQUFDLFNBQVMsc0JBQXNCO0FBQ25EO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSwyQkFBMkIsY0FBYyxHQUFHLFlBQVk7QUFDeEQ7QUFDQSxnQkFBZ0Isa0RBQUMsU0FBUyxzQkFBc0I7QUFDaEQ7QUFDQSxTQUFTO0FBQ1Q7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxZQUFZLGlFQUFTO0FBQ3JCO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2I7QUFDQTtBQUNBOzs7O0FBSUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLDhDQUE4QyxjQUFjLFNBQVMsd0JBQXdCO0FBQzdGO0FBQ0EsU0FBUztBQUNUOztBQUVBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7O0FBRUE7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxTQUFTOztBQUVUO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2I7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBOztBQUV5Qzs7Ozs7Ozs7Ozs7OztBQ3JQekM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFDQTtBQUNBOztBQUVzQztBQUNYOztBQUUzQjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0Esd0JBQXdCLG9EQUFTO0FBQ2pDO0FBQ0E7O0FBRUE7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0Esa0NBQWtDO0FBQ2xDO0FBQ0E7QUFDQSxZQUFZLGtEQUFDO0FBQ2I7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGFBQWE7QUFDYjtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBO0FBQ0E7QUFDQTs7QUFFeUM7Ozs7Ozs7Ozs7Ozs7QUN0RHpDO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQ0E7QUFDQTs7QUFFc0M7QUFDWDs7QUFFM0I7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGdDQUFnQyxvREFBUztBQUN6QztBQUNBOztBQUVBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBLGVBQWUsa0RBQUM7QUFDaEI7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGFBQWE7QUFDYjtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBO0FBQ0E7QUFDQTs7QUFFeUQ7Ozs7Ozs7Ozs7Ozs7QUNoRHpEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQ0E7QUFDQTs7QUFFc0M7QUFDWDs7QUFFM0I7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLHVCQUF1QixvREFBUztBQUNoQztBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQSxZQUFZLGtEQUFDO0FBQ2I7QUFDQTtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2IsZ0JBQWdCLGtEQUFDLE9BQU8sMENBQTBDO0FBQ2xFO0FBQ0E7QUFDQSxnQkFBZ0Isa0RBQUM7QUFDakI7QUFDQTtBQUNBLDJCQUEyQixvQ0FBb0M7QUFDL0Q7QUFDQSxpQkFBaUI7QUFDakIsZ0JBQWdCLGtEQUFDLFNBQVMsdUJBQXVCO0FBQ2pEO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLDJCQUEyQixjQUFjLFFBQVEsSUFBSTtBQUNyRDtBQUNBO0FBQ0E7QUFDQTtBQUNBLGlCQUFpQjtBQUNqQixhQUFhO0FBQ2IsU0FBUztBQUNUO0FBQ0E7QUFDQTtBQUNBLCtCQUErQixjQUFjLFFBQVEsSUFBSTtBQUN6RDtBQUNBO0FBQ0E7QUFDQTtBQUNBLHFCQUFxQjtBQUNyQixpQkFBaUI7QUFDakIsYUFBYTtBQUNiO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7OztBQUdBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSwyQkFBMkIsb0RBQVM7QUFDcEM7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBLFlBQVksa0RBQUM7QUFDYjtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2I7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBO0FBQ0E7QUFDQTs7QUFFdUM7Ozs7Ozs7Ozs7Ozs7QUNqSnZDO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVzQztBQUNYOzs7QUFHM0I7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0Esc0JBQXNCLG9EQUFTO0FBQy9CO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0EsWUFBWSxrREFBQztBQUNiO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsYUFBYTtBQUNiO0FBQ0Esb0JBQW9CLGtEQUFDO0FBQ3JCLHFEQUFxRDtBQUNyRDtBQUNBLHFCQUFxQjtBQUNyQjtBQUNBLG9CQUFvQixrREFBQyxTQUFTLDZCQUE2QjtBQUMzRDtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTs7QUFFcUM7Ozs7Ozs7Ozs7Ozs7QUN0RnJDO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQ0E7QUFDQTs7QUFFc0M7QUFDWDs7QUFFM0I7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsbUJBQW1CLG9EQUFTO0FBQzVCO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBLDZCQUE2QixrREFBQztBQUM5QjtBQUNBO0FBQ0EsWUFBWSxrREFBQztBQUNiO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsYUFBYTtBQUNiLGdCQUFnQixrREFBQyxZQUFZO0FBQzdCLG9CQUFvQixrREFBQyxTQUFTO0FBQzlCO0FBQ0EsZ0JBQWdCLGtEQUFDLFlBQVk7QUFDN0I7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBO0FBQ0Esb0JBQW9CLGtEQUFDLFFBQVEsUUFBUSxjQUFjLFdBQVcsT0FBTyxFQUFFO0FBQ3ZFO0FBQ0E7QUFDQTtBQUNBLGFBQWE7QUFDYjtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBO0FBQ0E7QUFDQSx3QkFBd0Isa0RBQUMsUUFBUSxRQUFRLGNBQWMsWUFBWSxPQUFPLEdBQUcsT0FBTyxFQUFFO0FBQ3RGO0FBQ0E7QUFDQTtBQUNBLGlCQUFpQjtBQUNqQjtBQUNBO0FBQ0EsaUNBQWlDLGtEQUFDLFFBQVEsUUFBUSxjQUFjLFlBQVksT0FBTyxHQUFHLE9BQU8sRUFBRTtBQUMvRjtBQUNBO0FBQ0E7QUFDQTtBQUNBLG9CQUFvQixrREFBQyxRQUFRLFFBQVEsY0FBYyxZQUFZLE9BQU8sRUFBRTtBQUN4RTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGFBQWE7QUFDYjtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0Esb0JBQW9CLGtEQUFDLFFBQVEsUUFBUSxjQUFjLFlBQVksT0FBTyxHQUFHLE9BQU8sRUFBRTtBQUNsRjtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2I7QUFDQTtBQUNBLDZCQUE2QixrREFBQyxRQUFRLFFBQVEsY0FBYyxlQUFlLE9BQU8sRUFBRTtBQUNwRjtBQUNBO0FBQ0E7QUFDQTtBQUNBLGdCQUFnQixrREFBQyxRQUFRLFFBQVEsY0FBYyxZQUFZLE9BQU8sRUFBRTtBQUNwRTtBQUNBO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDs7QUFFQTtBQUNBO0FBQ0E7QUFDQSxnQkFBZ0Isa0RBQUMsUUFBUSxRQUFRLGNBQWMsV0FBVyxPQUFPLEVBQUU7QUFDbkU7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7O0FBR3dCOzs7Ozs7Ozs7Ozs7O0FDM0l4QjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUNBO0FBQ0E7O0FBRXNDO0FBQ1g7O0FBRTNCO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLHdCQUF3QixvREFBUztBQUNqQztBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0EsWUFBWSxrREFBQztBQUNiO0FBQ0E7QUFDQTtBQUNBO0FBQ0EscUNBQXFDLHFCQUFxQjtBQUMxRCxhQUFhO0FBQ2I7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLFlBQVksa0RBQUMsU0FBUyx3Q0FBd0MsRUFBRTtBQUNoRSxnQkFBZ0Isa0RBQUM7QUFDakI7QUFDQSx5Q0FBeUMsdUJBQXVCLHFCQUFxQjtBQUNyRixpQkFBaUI7QUFDakI7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLFlBQVksa0RBQUMsU0FBUyx3Q0FBd0MsRUFBRTtBQUNoRSxnQkFBZ0Isa0RBQUM7QUFDakI7QUFDQSx5Q0FBeUMsdUJBQXVCLHFCQUFxQjtBQUNyRixpQkFBaUI7QUFDakI7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLFlBQVksa0RBQUMsU0FBUyx3Q0FBd0MsRUFBRTtBQUNoRSxnQkFBZ0Isa0RBQUM7QUFDakI7QUFDQSx5Q0FBeUMsdUJBQXVCLHFCQUFxQjtBQUNyRixpQkFBaUI7QUFDakI7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0Esb0JBQW9CLGtEQUFDLFVBQVUsd0JBQXdCO0FBQ3ZEO0FBQ0EsYUFBYTtBQUNiLFNBQVM7QUFDVCwrQ0FBK0MsU0FBUztBQUN4RDtBQUNBLG9CQUFvQixrREFBQyxVQUFVLHdCQUF3QjtBQUN2RDtBQUNBLGFBQWE7QUFDYjtBQUNBO0FBQ0E7O0FBRXlDOzs7Ozs7Ozs7Ozs7O0FDakh6QztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUNBO0FBQ0E7O0FBRXNDO0FBQ1g7O0FBRTNCLDBDQUEwQyxvREFBUztBQUNuRDtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBLFlBQVksa0RBQUM7QUFDYjtBQUNBO0FBQ0E7QUFDQTtBQUNBLGFBQWE7QUFDYjtBQUNBO0FBQ0E7O0FBRTZFOzs7Ozs7Ozs7Ozs7O0FDeEI3RTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUNBO0FBQ0E7O0FBRXNDO0FBQ1g7O0FBRTNCLGtDQUFrQyxvREFBUztBQUMzQztBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBLFlBQVksa0RBQUM7QUFDYjtBQUNBO0FBQ0E7QUFDQSxhQUFhLEdBQUcsa0RBQUMsU0FBUyx5Q0FBeUM7QUFDbkU7QUFDQTtBQUNBOztBQUVBOztBQUU2RDs7Ozs7Ozs7Ozs7OztBQ3pCN0Q7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFDQTtBQUNBOztBQUVzQztBQUNYOztBQUUzQjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsbUJBQW1CLG9EQUFTO0FBQzVCO0FBQ0E7O0FBRUE7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQSxZQUFZLGtEQUFDO0FBQ2I7QUFDQTtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2IsZ0JBQWdCLGtEQUFDLFNBQVMseUJBQXlCO0FBQ25EO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTtBQUNBOztBQUUrQjs7Ozs7Ozs7Ozs7OztBQ3BEL0I7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFDQTtBQUNBOztBQUVzQztBQUNYOztBQUUzQjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0Esb0JBQW9CLG9EQUFTO0FBQzdCO0FBQ0E7QUFDQSx3Q0FBd0MsbUJBQW1COztBQUUzRDtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQSxZQUFZLGtEQUFDO0FBQ2I7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsYUFBYTtBQUNiLGdCQUFnQixrREFBQyxTQUFTLHdDQUF3QztBQUNsRSxvQkFBb0Isa0RBQUMsU0FBUyx1QkFBdUI7QUFDckQsd0JBQXdCLGtEQUFDLFNBQVMsc0JBQXNCO0FBQ3hELDRCQUE0QixrREFBQyxRQUFRLHFCQUFxQjtBQUMxRDtBQUNBO0FBQ0E7QUFDQSx3QkFBd0Isa0RBQUMsU0FBUyxvQkFBb0I7QUFDdEQ7QUFDQTtBQUNBLHdCQUF3QixrREFBQyxTQUFTLHNCQUFzQjtBQUN4RDtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBO0FBQ0E7O0FBRWdDOzs7Ozs7Ozs7Ozs7O0FDekZoQztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUNBO0FBQ0E7O0FBRXNDO0FBQ1g7O0FBRTNCLHNCQUFzQixvREFBUztBQUMvQjtBQUNBOztBQUVBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0EsWUFBWSxrREFBQztBQUNiO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGFBQWE7QUFDYjtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTs7QUFFcUM7Ozs7Ozs7Ozs7Ozs7QUNyQ3JDO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQ0E7QUFDQTs7QUFFc0M7QUFDWDs7QUFFM0Isc0JBQXNCLG9EQUFTO0FBQy9CO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0EsWUFBWSxrREFBQztBQUNiO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsYUFBYTtBQUNiO0FBQ0E7QUFDQTs7QUFFcUM7Ozs7Ozs7Ozs7Ozs7QUN4QnJDO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQ0E7QUFDQTs7QUFFc0M7QUFDWDs7QUFFM0I7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxtQkFBbUIsb0RBQVM7QUFDNUI7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0EsWUFBWSxrREFBQztBQUNiO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsYUFBYTtBQUNiLGdCQUFnQixrREFBQyxTQUFTLFdBQVcsY0FBYyx3Q0FBd0M7QUFDM0Y7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EseUJBQXlCLDRCQUE0QjtBQUNyRCx3QkFBd0IsY0FBYztBQUN0QyxhQUFhO0FBQ2IsYUFBYTtBQUNiO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsZ0RBQWdEO0FBQ2hEO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2I7QUFDQTs7QUFFK0I7Ozs7Ozs7Ozs7Ozs7QUN6Ry9CO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQ0E7QUFDQTs7QUFFc0M7QUFDWDs7QUFFM0I7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLHNCQUFzQixvREFBUztBQUMvQjtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQSxlQUFlLGtEQUFDO0FBQ2hCO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGFBQWE7QUFDYixnQkFBZ0Isa0RBQUM7QUFDakI7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLHFCQUFxQjtBQUNyQjtBQUNBO0FBQ0EsZ0JBQWdCLGtEQUFDLFNBQVMsc0JBQXNCO0FBQ2hELG9CQUFvQixrREFBQyxTQUFTLDJCQUEyQjtBQUN6RCx3QkFBd0Isa0RBQUMsU0FBUyxvQkFBb0I7QUFDdEQsd0JBQXdCLGtEQUFDLFNBQVMsc0JBQXNCO0FBQ3hELDRCQUE0QixrREFBQyxTQUFTLDJDQUEyQztBQUNqRjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBO0FBQ0E7O0FBRXFDOzs7Ozs7Ozs7Ozs7O0FDN0ZyQztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUNBO0FBQ0E7O0FBRXNDO0FBQ1g7O0FBRTNCO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSx1QkFBdUIsb0RBQVM7QUFDaEM7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBLFlBQVksa0RBQUM7QUFDYjtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2I7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBO0FBQ0E7O0FBRXVDOzs7Ozs7Ozs7Ozs7O0FDL0N2QztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUNBO0FBQ0E7O0FBRXNDO0FBQ1g7O0FBRTNCO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSx5QkFBeUIsb0RBQVM7QUFDbEM7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBLFlBQVksa0RBQUM7QUFDYjtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2I7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBO0FBQ0E7O0FBRTJDOzs7Ozs7Ozs7Ozs7O0FDL0MzQztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUNBO0FBQ0E7O0FBRXNDO0FBQ1g7O0FBRTNCO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLHVCQUF1QixvREFBUztBQUNoQztBQUNBOztBQUVBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0EsWUFBWSxrREFBQztBQUNiO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2I7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBO0FBQ0E7O0FBRXVDOzs7Ozs7Ozs7Ozs7O0FDbER2QztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ3NDO0FBQ1g7OztBQUczQjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLG9CQUFvQixvREFBUztBQUM3QjtBQUNBOztBQUVBOztBQUVBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQSwwREFBMEQsY0FBYztBQUN4RTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDs7QUFFQTtBQUNBLHVDQUF1QyxjQUFjO0FBQ3JEO0FBQ0EsWUFBWSxrREFBQztBQUNiO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsYUFBYTtBQUNiLGdCQUFnQixrREFBQztBQUNqQixnQ0FBZ0MsY0FBYztBQUM5QztBQUNBO0FBQ0EsaUJBQWlCO0FBQ2pCO0FBQ0E7QUFDQTs7QUFFQTtBQUNBLHlEQUF5RCxjQUFjO0FBQ3ZFO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBLHdEQUF3RCxjQUFjO0FBQ3RFO0FBQ0E7QUFDQTtBQUNBLG9CQUFvQjtBQUNwQixTQUFTOztBQUVUO0FBQ0E7QUFDQSx1Q0FBdUMsV0FBVztBQUNsRDtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBLDhCQUE4QixxQkFBcUI7QUFDbkQ7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EseUJBQXlCO0FBQ3pCO0FBQ0Esa0RBQWtEO0FBQ2xEO0FBQ0EsaUJBQWlCO0FBQ2pCLGFBQWE7QUFDYiw4Q0FBOEM7QUFDOUM7QUFDQSxrREFBa0Q7QUFDbEQ7QUFDQSxpQkFBaUI7QUFDakI7QUFDQSxTQUFTOztBQUVUO0FBQ0E7QUFDQTtBQUNBLDBDQUEwQztBQUMxQyxTQUFTOztBQUVUO0FBQ0E7QUFDQTtBQUNBLDBDQUEwQztBQUMxQyxTQUFTO0FBQ1Q7O0FBRUE7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxxQkFBcUI7QUFDckI7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVpQzs7Ozs7Ozs7Ozs7OztBQ3hNakM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFDQTtBQUNBOztBQUVzQztBQUNYOztBQUUzQixnQ0FBZ0Msb0RBQVM7QUFDekM7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLHNDQUFzQztBQUN0QztBQUNBO0FBQ0E7QUFDQTtBQUNBLGVBQWUsa0RBQUM7QUFDaEI7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUV5RDs7Ozs7Ozs7Ozs7OztBQzVDekQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFDQTtBQUNBOztBQUVzQztBQUNYOzs7QUFHM0IsbUJBQW1CLG9EQUFTO0FBQzVCO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0EsWUFBWSxrREFBQztBQUNiO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsYUFBYTtBQUNiO0FBQ0E7QUFDQTs7QUFFK0I7Ozs7Ozs7Ozs7Ozs7QUN6Qi9CO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQ0E7QUFDQTs7QUFFc0M7QUFDWDs7QUFFM0I7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLHlCQUF5QixvREFBUztBQUNsQztBQUNBOztBQUVBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBLGVBQWUsa0RBQUM7QUFDaEI7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsYUFBYTtBQUNiO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTtBQUNBOztBQUUyQzs7Ozs7Ozs7Ozs7OztBQ2pEM0M7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFDQTtBQUNBOztBQUVzQztBQUNYOztBQUUzQjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsaUNBQWlDLG9EQUFTO0FBQzFDO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBLGVBQWUsa0RBQUM7QUFDaEI7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsYUFBYTtBQUNiO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTtBQUNBLHdCQUF3QixrREFBQyxTQUFTLHNDQUFzQztBQUN4RSw0QkFBNEIsa0RBQUMsV0FBVztBQUN4QztBQUNBO0FBQ0EsaUJBQWlCO0FBQ2pCO0FBQ0Esb0JBQW9CLGtEQUFDLFNBQVMsa0NBQWtDLGNBQWMsZ0JBQWdCO0FBQzlGO0FBQ0EsYUFBYTtBQUNiO0FBQ0Esb0JBQW9CLGtEQUFDLFNBQVMsUUFBUSxjQUFjLGdCQUFnQjtBQUNwRTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQSxvQkFBb0Isa0RBQUMsU0FBUyxzQ0FBc0M7QUFDcEUsd0JBQXdCLGtEQUFDLFdBQVc7QUFDcEM7QUFDQTtBQUNBLGFBQWE7QUFDYjtBQUNBLGdCQUFnQixrREFBQyxTQUFTLGtDQUFrQyxjQUFjLGdCQUFnQjtBQUMxRjtBQUNBLFNBQVM7QUFDVDtBQUNBLGdCQUFnQixrREFBQyxTQUFTLFFBQVEsY0FBYyxnQkFBZ0I7QUFDaEU7QUFDQTtBQUNBO0FBQ0E7O0FBRTJEOzs7Ozs7Ozs7Ozs7O0FDN0YzRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUNBO0FBQ0E7O0FBRXNDO0FBQ1g7OztBQUczQjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0Esb0JBQW9CLG9EQUFTO0FBQzdCO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBLFlBQVksa0RBQUM7QUFDYjtBQUNBO0FBQ0E7QUFDQTtBQUNBLGFBQWE7QUFDYixnQkFBZ0Isa0RBQUMsV0FBVywwQkFBMEI7QUFDdEQ7QUFDQTtBQUNBLGdCQUFnQixrREFBQyxZQUFZO0FBQzdCO0FBQ0E7QUFDQTs7QUFFQTtBQUNBLHFDQUFxQywwQkFBMEIseUJBQXlCO0FBQ3hGOztBQUVBO0FBQ0E7QUFDQTtBQUNBLHVCQUF1QixrREFBQztBQUN4QiwrQ0FBK0M7QUFDL0MsNEJBQTRCLGNBQWMsZ0JBQWdCLElBQUk7QUFDOUQsaUJBQWlCO0FBQ2pCLGFBQWE7QUFDYixTQUFTO0FBQ1Q7QUFDQSx1QkFBdUIsa0RBQUM7QUFDeEIsK0NBQStDO0FBQy9DLDRCQUE0QixjQUFjLGdCQUFnQixJQUFJO0FBQzlELGlCQUFpQjtBQUNqQixhQUFhO0FBQ2I7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBOzs7O0FBSUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxvQkFBb0Isa0RBQUM7QUFDckIsZ0NBQWdDLGNBQWMsTUFBTSxPQUFPLEdBQUcsT0FBTztBQUNyRSxxQkFBcUI7QUFDckI7QUFDQSxhQUFhO0FBQ2IsK0JBQStCLGtEQUFDLFNBQVMsTUFBTSxXQUFXO0FBQzFEO0FBQ0EsZ0JBQWdCLGtEQUFDLFFBQVEsUUFBUSxjQUFjLE1BQU0sT0FBTyxFQUFFO0FBQzlEO0FBQ0EsU0FBUztBQUNUOztBQUVBO0FBQ0E7QUFDQTtBQUNBLFlBQVksa0RBQUMsU0FBUztBQUN0QixnQkFBZ0Isa0RBQUMsUUFBUSwyQkFBMkIsRUFBRTtBQUN0RCxvQkFBb0Isa0RBQUMsU0FBUyxjQUFjO0FBQzVDLHdCQUF3QixrREFBQyxTQUFTLHVCQUF1QjtBQUN6RDtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRWlDOzs7Ozs7Ozs7Ozs7O0FDcEpqQztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUNBO0FBQ0E7O0FBRXNDO0FBQ1g7OztBQUczQjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsbUJBQW1CLG9EQUFTO0FBQzVCO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBLFlBQVksa0RBQUM7QUFDYjtBQUNBO0FBQ0E7QUFDQTtBQUNBLGFBQWE7QUFDYixnQkFBZ0Isa0RBQUMsUUFBUSx1Q0FBdUM7QUFDaEUsZ0JBQWdCLGtEQUFDLFNBQVMscUJBQXFCO0FBQy9DLG9CQUFvQixrREFBQyxTQUFTLHFEQUFxRDtBQUNuRjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBO0FBQ0E7OztBQUcrQjs7Ozs7Ozs7Ozs7OztBQ3hFL0I7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFDQTtBQUNBOztBQUVzQztBQUNYOztBQUUzQixtQkFBbUIsb0RBQVM7QUFDNUI7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQSxZQUFZLGtEQUFDO0FBQ2I7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGFBQWE7QUFDYjtBQUNBO0FBQ0E7O0FBRStCOzs7Ozs7Ozs7Ozs7O0FDekIvQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUNBO0FBQ0E7O0FBRXNDO0FBQ1g7O0FBRTNCO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EseUJBQXlCLG9EQUFTO0FBQ2xDO0FBQ0E7O0FBRUE7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQSxZQUFZLGtEQUFDO0FBQ2I7QUFDQTtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2I7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBO0FBQ0E7OztBQUd5Qzs7Ozs7Ozs7Ozs7OztBQ2hEekM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRXNDO0FBQ1g7O0FBRTNCO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxzQkFBc0Isb0RBQVM7QUFDL0I7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxZQUFZLGtEQUFDO0FBQ2I7QUFDQTtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2IsZ0JBQWdCLGtEQUFDO0FBQ2pCO0FBQ0E7QUFDQTtBQUNBLGlCQUFpQjtBQUNqQjtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRXFDOzs7Ozs7Ozs7Ozs7O0FDckVyQztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVzQztBQUNYOztBQUUzQjs7QUFFQSwyQkFBMkIsb0RBQVM7QUFDcEM7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLDREQUE0RCw0QkFBNEI7QUFDeEY7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBO0FBQ0E7O0FBRUE7QUFDQSxlQUFlLGtEQUFDO0FBQ2hCO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGFBQWE7QUFDYjs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsOERBQThELDRCQUE0QixvQkFBb0IsY0FBYztBQUM1SDtBQUNBO0FBQ0EseURBQXlELDRCQUE0QjtBQUNyRjtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2I7QUFDQTtBQUNBLFNBQVM7QUFDVDs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUUrQzs7Ozs7Ozs7Ozs7OztBQ3BGL0M7QUFBQTtBQUFBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQSxLQUFLO0FBQ0w7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSwwQkFBMEIsb0JBQW9CO0FBQzlDO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxpQkFBaUI7QUFDakI7QUFDQTtBQUNBO0FBQ0EsNkJBQTZCLGNBQWMsTUFBTSxTQUFTLDBDQUEwQyxRQUFRO0FBQzVHO0FBQ0E7QUFDQSxTQUFTO0FBQ1QsS0FBSzs7QUFFTDtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGlCQUFpQjtBQUNqQjtBQUNBLGlCQUFpQjtBQUNqQixxQ0FBcUMsY0FBYyxNQUFNLFNBQVMsd0NBQXdDLFVBQVU7QUFDcEg7QUFDQTtBQUNBLGlCQUFpQjtBQUNqQixxQ0FBcUMsY0FBYyxNQUFNLFNBQVMsbUJBQW1CLFFBQVE7QUFDN0Y7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsYUFBYTtBQUNiO0FBQ0EscUNBQXFDLGNBQWMsTUFBTSxTQUFTLHlDQUF5QyxVQUFVO0FBQ3JIO0FBQ0E7QUFDQSxpQkFBaUI7QUFDakI7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLEtBQUs7O0FBRUw7QUFDQTtBQUNBLEtBQUs7O0FBRUw7QUFDQTtBQUNBLEtBQUs7O0FBRUw7QUFDQTtBQUNBLEtBQUs7O0FBRUw7QUFDQTtBQUNBLEtBQUs7O0FBRUw7QUFDQTtBQUNBLEtBQUs7O0FBRUw7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLHNDQUFzQztBQUN0Qzs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGlCQUFpQjtBQUNqQjtBQUNBO0FBQ0EsYUFBYTtBQUNiO0FBQ0E7QUFDQTtBQUNBLGlDQUFpQyxjQUFjLHNCQUFzQixTQUFTO0FBQzlFO0FBQ0E7QUFDQTtBQUNBLFNBQVM7O0FBRVQ7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBO0FBQ0EsS0FBSzs7QUFFTDtBQUNBO0FBQ0E7QUFDQSxpQ0FBaUMsY0FBYyxNQUFNLFNBQVMseUJBQXlCLFFBQVE7QUFDL0Y7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLEtBQUs7O0FBRUw7QUFDQTtBQUNBO0FBQ0EsNkJBQTZCLGNBQWMsTUFBTSxTQUFTO0FBQzFEO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFJRTs7O0FBR0Y7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGFBQWE7QUFDYixpQ0FBaUMsY0FBYyxNQUFNLFNBQVM7QUFDOUQ7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7O0FBRUEsS0FBSzs7QUFFTDtBQUNBO0FBQ0E7QUFDQTtBQUNBLGFBQWE7QUFDYixpQ0FBaUMsY0FBYyxNQUFNLFNBQVM7QUFDOUQ7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQSxLQUFLOztBQUVMO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsYUFBYTtBQUNiLGlDQUFpQyxjQUFjLE1BQU0sU0FBUztBQUM5RDtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBLEtBQUs7O0FBRUw7QUFDQTtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2IsaUNBQWlDLGNBQWMsTUFBTSxTQUFTO0FBQzlEO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBO0FBQ0EsS0FBSzs7QUFFTDtBQUNBO0FBQ0E7QUFDQTtBQUNBLGFBQWE7QUFDYixpQ0FBaUMsY0FBYyxNQUFNLFNBQVM7QUFDOUQ7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQSxLQUFLOztBQUVMOzs7Ozs7Ozs7Ozs7O0FDOU9BO0FBQUE7QUFBQTtBQUFBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGFBQWE7QUFDYjtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUOztBQUVBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQSx5REFBeUQsaUJBQWlCO0FBQzFFO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFLRTs7Ozs7Ozs7Ozs7OztBQ3BLRjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBa0I7QUFDbEI7QUFDQSxVQUFVLFVBQVU7QUFDc0I7QUFDRjtBQUNjOztBQUV0RDtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQSxzQkFBc0I7QUFDdEIsbUJBQW1CLGdCQUFnQjtBQUNuQyx3Q0FBd0M7QUFDeEMsZ0NBQWdDLGNBQWM7QUFDOUMsMENBQTBDO0FBQzFDO0FBQ0E7QUFDQTtBQUNBLG1CQUFtQix5Q0FBeUM7QUFDNUQ7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLHVCQUF1QixzREFBVTtBQUNqQyx3QkFBd0Isd0RBQVcsZUFBZSxvRUFBaUI7QUFDbkU7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLENBQUM7O0FBRUQ7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsQ0FBQzs7QUFFRCxXQUFXO0FBQ1g7Ozs7Ozs7Ozs7OztBQ3REQTtBQUNBLElBQUksS0FBNEQ7QUFDaEUsSUFBSSxTQUN3RDtBQUM1RCxDQUFDLDJCQUEyQjs7QUFFNUI7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBO0FBQ0EsYUFBYTtBQUNiO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSwrQkFBK0IscUJBQXFCO0FBQ3BEO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsbUJBQW1CO0FBQ25CO0FBQ0E7QUFDQTtBQUNBLG1CQUFtQjtBQUNuQiwyQkFBMkIsdUJBQXVCO0FBQ2xEO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSx3RUFBd0UsY0FBYztBQUN0RjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsK0JBQStCLG9CQUFvQjtBQUNuRDtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLCtCQUErQixnQkFBZ0I7QUFDL0M7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLDhEQUE4RDtBQUM5RDtBQUNBLDBHQUEwRztBQUMxRztBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsbUZBQW1GO0FBQ25GO0FBQ0EsNkJBQTZCO0FBQzdCO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsdUJBQXVCLGVBQWU7QUFDdEM7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSwrQ0FBK0Msd0JBQXdCO0FBQ3ZFO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxnRUFBZ0U7QUFDaEU7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLDJCQUEyQiwyQkFBMkI7QUFDdEQ7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsMkVBQTJFLDJCQUEyQjtBQUN0RztBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLHVCQUF1QixlQUFlO0FBQ3RDO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSwrQkFBK0Isb0JBQW9CO0FBQ25EO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLCtCQUErQixnQkFBZ0I7QUFDL0M7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLDJDQUEyQztBQUMzQztBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLHNEQUFzRDtBQUN0RDtBQUNBLHFCQUFxQjtBQUNyQjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSw0RkFBNEY7QUFDNUY7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLG1FQUFtRTtBQUNuRTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0Esc0NBQXNDLGtCQUFrQjtBQUN4RDtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsOEJBQThCLHVCQUF1QjtBQUNyRDtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLHlCQUF5QjtBQUN6QjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGtFQUFrRTtBQUNsRSwrREFBK0QsMkJBQTJCO0FBQzFGO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsNERBQTREO0FBQzVEO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0Esd0NBQXdDLGNBQWMsRUFBRTtBQUN4RDtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGFBQWE7QUFDYjtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBLHdDQUF3QyxrQkFBa0IsRUFBRTtBQUM1RDtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxxREFBcUQsY0FBYztBQUNuRTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsNENBQTRDLDhCQUE4QjtBQUMxRTtBQUNBO0FBQ0EsNENBQTRDLG1DQUFtQztBQUMvRTtBQUNBO0FBQ0E7QUFDQTtBQUNBLDJGQUEyRiwrQkFBK0IsRUFBRTtBQUM1SCxTQUFTO0FBQ1Q7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSwyRUFBMkUsNkJBQTZCO0FBQ3hHO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsaUNBQWlDO0FBQ2pDO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLDZDQUE2QyxtQkFBbUI7QUFDaEU7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLHVCQUF1QjtBQUN2QjtBQUNBO0FBQ0E7QUFDQSwyQkFBMkIsd0JBQXdCO0FBQ25EO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2I7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsYUFBYTtBQUNiO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsYUFBYTtBQUNiO0FBQ0E7QUFDQSxhQUFhO0FBQ2I7QUFDQTtBQUNBLGFBQWE7QUFDYjtBQUNBO0FBQ0EsYUFBYTtBQUNiO0FBQ0E7QUFDQSxhQUFhO0FBQ2I7QUFDQSwrQkFBK0IsNEJBQTRCO0FBQzNEO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2I7QUFDQTtBQUNBLG1DQUFtQyxtQkFBbUI7QUFDdEQ7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQSxrQkFBa0IsY0FBYztBQUNoQyxZQUFZLGlFQUFpRTtBQUM3RTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsK0JBQStCLHVCQUF1QjtBQUN0RDtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSx1Q0FBdUMscUJBQXFCO0FBQzVEO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUEsa0RBQWtELGNBQWM7O0FBRWhFLENBQUMiLCJmaWxlIjoibWFpbi5idW5kbGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyIgXHQvLyBUaGUgbW9kdWxlIGNhY2hlXG4gXHR2YXIgaW5zdGFsbGVkTW9kdWxlcyA9IHt9O1xuXG4gXHQvLyBUaGUgcmVxdWlyZSBmdW5jdGlvblxuIFx0ZnVuY3Rpb24gX193ZWJwYWNrX3JlcXVpcmVfXyhtb2R1bGVJZCkge1xuXG4gXHRcdC8vIENoZWNrIGlmIG1vZHVsZSBpcyBpbiBjYWNoZVxuIFx0XHRpZihpbnN0YWxsZWRNb2R1bGVzW21vZHVsZUlkXSkge1xuIFx0XHRcdHJldHVybiBpbnN0YWxsZWRNb2R1bGVzW21vZHVsZUlkXS5leHBvcnRzO1xuIFx0XHR9XG4gXHRcdC8vIENyZWF0ZSBhIG5ldyBtb2R1bGUgKGFuZCBwdXQgaXQgaW50byB0aGUgY2FjaGUpXG4gXHRcdHZhciBtb2R1bGUgPSBpbnN0YWxsZWRNb2R1bGVzW21vZHVsZUlkXSA9IHtcbiBcdFx0XHRpOiBtb2R1bGVJZCxcbiBcdFx0XHRsOiBmYWxzZSxcbiBcdFx0XHRleHBvcnRzOiB7fVxuIFx0XHR9O1xuXG4gXHRcdC8vIEV4ZWN1dGUgdGhlIG1vZHVsZSBmdW5jdGlvblxuIFx0XHRtb2R1bGVzW21vZHVsZUlkXS5jYWxsKG1vZHVsZS5leHBvcnRzLCBtb2R1bGUsIG1vZHVsZS5leHBvcnRzLCBfX3dlYnBhY2tfcmVxdWlyZV9fKTtcblxuIFx0XHQvLyBGbGFnIHRoZSBtb2R1bGUgYXMgbG9hZGVkXG4gXHRcdG1vZHVsZS5sID0gdHJ1ZTtcblxuIFx0XHQvLyBSZXR1cm4gdGhlIGV4cG9ydHMgb2YgdGhlIG1vZHVsZVxuIFx0XHRyZXR1cm4gbW9kdWxlLmV4cG9ydHM7XG4gXHR9XG5cblxuIFx0Ly8gZXhwb3NlIHRoZSBtb2R1bGVzIG9iamVjdCAoX193ZWJwYWNrX21vZHVsZXNfXylcbiBcdF9fd2VicGFja19yZXF1aXJlX18ubSA9IG1vZHVsZXM7XG5cbiBcdC8vIGV4cG9zZSB0aGUgbW9kdWxlIGNhY2hlXG4gXHRfX3dlYnBhY2tfcmVxdWlyZV9fLmMgPSBpbnN0YWxsZWRNb2R1bGVzO1xuXG4gXHQvLyBkZWZpbmUgZ2V0dGVyIGZ1bmN0aW9uIGZvciBoYXJtb255IGV4cG9ydHNcbiBcdF9fd2VicGFja19yZXF1aXJlX18uZCA9IGZ1bmN0aW9uKGV4cG9ydHMsIG5hbWUsIGdldHRlcikge1xuIFx0XHRpZighX193ZWJwYWNrX3JlcXVpcmVfXy5vKGV4cG9ydHMsIG5hbWUpKSB7XG4gXHRcdFx0T2JqZWN0LmRlZmluZVByb3BlcnR5KGV4cG9ydHMsIG5hbWUsIHsgZW51bWVyYWJsZTogdHJ1ZSwgZ2V0OiBnZXR0ZXIgfSk7XG4gXHRcdH1cbiBcdH07XG5cbiBcdC8vIGRlZmluZSBfX2VzTW9kdWxlIG9uIGV4cG9ydHNcbiBcdF9fd2VicGFja19yZXF1aXJlX18uciA9IGZ1bmN0aW9uKGV4cG9ydHMpIHtcbiBcdFx0aWYodHlwZW9mIFN5bWJvbCAhPT0gJ3VuZGVmaW5lZCcgJiYgU3ltYm9sLnRvU3RyaW5nVGFnKSB7XG4gXHRcdFx0T2JqZWN0LmRlZmluZVByb3BlcnR5KGV4cG9ydHMsIFN5bWJvbC50b1N0cmluZ1RhZywgeyB2YWx1ZTogJ01vZHVsZScgfSk7XG4gXHRcdH1cbiBcdFx0T2JqZWN0LmRlZmluZVByb3BlcnR5KGV4cG9ydHMsICdfX2VzTW9kdWxlJywgeyB2YWx1ZTogdHJ1ZSB9KTtcbiBcdH07XG5cbiBcdC8vIGNyZWF0ZSBhIGZha2UgbmFtZXNwYWNlIG9iamVjdFxuIFx0Ly8gbW9kZSAmIDE6IHZhbHVlIGlzIGEgbW9kdWxlIGlkLCByZXF1aXJlIGl0XG4gXHQvLyBtb2RlICYgMjogbWVyZ2UgYWxsIHByb3BlcnRpZXMgb2YgdmFsdWUgaW50byB0aGUgbnNcbiBcdC8vIG1vZGUgJiA0OiByZXR1cm4gdmFsdWUgd2hlbiBhbHJlYWR5IG5zIG9iamVjdFxuIFx0Ly8gbW9kZSAmIDh8MTogYmVoYXZlIGxpa2UgcmVxdWlyZVxuIFx0X193ZWJwYWNrX3JlcXVpcmVfXy50ID0gZnVuY3Rpb24odmFsdWUsIG1vZGUpIHtcbiBcdFx0aWYobW9kZSAmIDEpIHZhbHVlID0gX193ZWJwYWNrX3JlcXVpcmVfXyh2YWx1ZSk7XG4gXHRcdGlmKG1vZGUgJiA4KSByZXR1cm4gdmFsdWU7XG4gXHRcdGlmKChtb2RlICYgNCkgJiYgdHlwZW9mIHZhbHVlID09PSAnb2JqZWN0JyAmJiB2YWx1ZSAmJiB2YWx1ZS5fX2VzTW9kdWxlKSByZXR1cm4gdmFsdWU7XG4gXHRcdHZhciBucyA9IE9iamVjdC5jcmVhdGUobnVsbCk7XG4gXHRcdF9fd2VicGFja19yZXF1aXJlX18ucihucyk7XG4gXHRcdE9iamVjdC5kZWZpbmVQcm9wZXJ0eShucywgJ2RlZmF1bHQnLCB7IGVudW1lcmFibGU6IHRydWUsIHZhbHVlOiB2YWx1ZSB9KTtcbiBcdFx0aWYobW9kZSAmIDIgJiYgdHlwZW9mIHZhbHVlICE9ICdzdHJpbmcnKSBmb3IodmFyIGtleSBpbiB2YWx1ZSkgX193ZWJwYWNrX3JlcXVpcmVfXy5kKG5zLCBrZXksIGZ1bmN0aW9uKGtleSkgeyByZXR1cm4gdmFsdWVba2V5XTsgfS5iaW5kKG51bGwsIGtleSkpO1xuIFx0XHRyZXR1cm4gbnM7XG4gXHR9O1xuXG4gXHQvLyBnZXREZWZhdWx0RXhwb3J0IGZ1bmN0aW9uIGZvciBjb21wYXRpYmlsaXR5IHdpdGggbm9uLWhhcm1vbnkgbW9kdWxlc1xuIFx0X193ZWJwYWNrX3JlcXVpcmVfXy5uID0gZnVuY3Rpb24obW9kdWxlKSB7XG4gXHRcdHZhciBnZXR0ZXIgPSBtb2R1bGUgJiYgbW9kdWxlLl9fZXNNb2R1bGUgP1xuIFx0XHRcdGZ1bmN0aW9uIGdldERlZmF1bHQoKSB7IHJldHVybiBtb2R1bGVbJ2RlZmF1bHQnXTsgfSA6XG4gXHRcdFx0ZnVuY3Rpb24gZ2V0TW9kdWxlRXhwb3J0cygpIHsgcmV0dXJuIG1vZHVsZTsgfTtcbiBcdFx0X193ZWJwYWNrX3JlcXVpcmVfXy5kKGdldHRlciwgJ2EnLCBnZXR0ZXIpO1xuIFx0XHRyZXR1cm4gZ2V0dGVyO1xuIFx0fTtcblxuIFx0Ly8gT2JqZWN0LnByb3RvdHlwZS5oYXNPd25Qcm9wZXJ0eS5jYWxsXG4gXHRfX3dlYnBhY2tfcmVxdWlyZV9fLm8gPSBmdW5jdGlvbihvYmplY3QsIHByb3BlcnR5KSB7IHJldHVybiBPYmplY3QucHJvdG90eXBlLmhhc093blByb3BlcnR5LmNhbGwob2JqZWN0LCBwcm9wZXJ0eSk7IH07XG5cbiBcdC8vIF9fd2VicGFja19wdWJsaWNfcGF0aF9fXG4gXHRfX3dlYnBhY2tfcmVxdWlyZV9fLnAgPSBcIlwiO1xuXG5cbiBcdC8vIExvYWQgZW50cnkgbW9kdWxlIGFuZCByZXR1cm4gZXhwb3J0c1xuIFx0cmV0dXJuIF9fd2VicGFja19yZXF1aXJlX18oX193ZWJwYWNrX3JlcXVpcmVfXy5zID0gXCIuL21haW4uanNcIik7XG4iLCIvKipcbiAqIENlbGxIYW5kbGVyIFByaW1hcnkgTWVzc2FnZSBIYW5kbGVyXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY2xhc3MgaW1wbGVtZW50cyBhIHNlcnZpY2UgdGhhdCBoYW5kbGVzXG4gKiBtZXNzYWdlcyBvZiBhbGwga2luZHMgdGhhdCBjb21lIGluIG92ZXIgYVxuICogYENlbGxTb2NrZXRgLlxuICogTk9URTogRm9yIHRoZSBtb21lbnQgdGhlcmUgYXJlIG9ubHkgdHdvIGtpbmRzXG4gKiBvZiBtZXNzYWdlcyBhbmQgdGhlcmVmb3JlIHR3byBoYW5kbGVycy4gV2UgaGF2ZVxuICogcGxhbnMgdG8gY2hhbmdlIHRoaXMgc3RydWN0dXJlIHRvIGJlIG1vcmUgZmxleGlibGVcbiAqIGFuZCBzbyB0aGUgQVBJIG9mIHRoaXMgY2xhc3Mgd2lsbCBjaGFuZ2UgZ3JlYXRseS5cbiAqL1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbmNsYXNzIENlbGxIYW5kbGVyIHtcbiAgICBjb25zdHJ1Y3RvcihoLCBwcm9qZWN0b3IsIGNvbXBvbmVudHMpe1xuXHQvLyBwcm9wc1xuXHR0aGlzLmggPSBoO1xuXHR0aGlzLnByb2plY3RvciA9IHByb2plY3Rvcjtcblx0dGhpcy5jb21wb25lbnRzID0gY29tcG9uZW50cztcblxuXHQvLyBJbnN0YW5jZSBQcm9wc1xuICAgICAgICB0aGlzLnBvc3RzY3JpcHRzID0gW107XG4gICAgICAgIHRoaXMuY2VsbHMgPSB7fTtcblx0dGhpcy5ET01QYXJzZXIgPSBuZXcgRE9NUGFyc2VyKCk7XG5cbiAgICAgICAgLy8gQmluZCBJbnN0YW5jZSBNZXRob2RzXG4gICAgICAgIHRoaXMuc2hvd0Nvbm5lY3Rpb25DbG9zZWQgPSB0aGlzLnNob3dDb25uZWN0aW9uQ2xvc2VkLmJpbmQodGhpcyk7XG5cdHRoaXMuY29ubmVjdGlvbkNsb3NlZFZpZXcgPSB0aGlzLmNvbm5lY3Rpb25DbG9zZWRWaWV3LmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuaGFuZGxlUG9zdHNjcmlwdCA9IHRoaXMuaGFuZGxlUG9zdHNjcmlwdC5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLmhhbmRsZU1lc3NhZ2UgPSB0aGlzLmhhbmRsZU1lc3NhZ2UuYmluZCh0aGlzKTtcblxuICAgIH1cblxuICAgIC8qKlxuICAgICAqIEZpbGxzIHRoZSBwYWdlJ3MgcHJpbWFyeSBkaXYgd2l0aFxuICAgICAqIGFuIGluZGljYXRvciB0aGF0IHRoZSBzb2NrZXQgaGFzIGJlZW5cbiAgICAgKiBkaXNjb25uZWN0ZWQuXG4gICAgICovXG4gICAgc2hvd0Nvbm5lY3Rpb25DbG9zZWQoKXtcblx0dGhpcy5wcm9qZWN0b3IucmVwbGFjZShcblx0ICAgIGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKFwicGFnZV9yb290XCIpLFxuXHQgICAgdGhpcy5jb25uZWN0aW9uQ2xvc2VkVmlld1xuXHQpO1xuICAgIH1cblxuICAgIC8qKlxuICAgICAqIENvbnZlbmllbmNlIG1ldGhvZCB0aGF0IHVwZGF0ZXNcbiAgICAgKiBCb290c3RyYXAtc3R5bGUgcG9wb3ZlcnMgb25cbiAgICAgKiB0aGUgRE9NLlxuICAgICAqIFNlZSBpbmxpbmUgY29tbWVudHNcbiAgICAgKi9cbiAgICB1cGRhdGVQb3BvdmVycygpIHtcbiAgICAgICAgLy8gVGhpcyBmdW5jdGlvbiByZXF1aXJlc1xuICAgICAgICAvLyBqUXVlcnkgYW5kIHBlcmhhcHMgZG9lc24ndFxuICAgICAgICAvLyBiZWxvbmcgaW4gdGhpcyBjbGFzcy5cbiAgICAgICAgLy8gVE9ETzogRmlndXJlIG91dCBhIGJldHRlciB3YXlcbiAgICAgICAgLy8gQUxTTyBOT1RFOlxuICAgICAgICAvLyAtLS0tLS0tLS0tLS0tLS0tLVxuICAgICAgICAvLyBgZ2V0Q2hpbGRQcm9wYCBpcyBhIGNvbnN0IGZ1bmN0aW9uXG4gICAgICAgIC8vIHRoYXQgaXMgZGVjbGFyZWQgaW4gYSBzZXBhcmF0ZVxuICAgICAgICAvLyBzY3JpcHQgdGFnIGF0IHRoZSBib3R0b20gb2ZcbiAgICAgICAgLy8gcGFnZS5odG1sLiBUaGF0J3MgYSBuby1ubyFcbiAgICAgICAgJCgnW2RhdGEtdG9nZ2xlPVwicG9wb3ZlclwiXScpLnBvcG92ZXIoe1xuICAgICAgICAgICAgaHRtbDogdHJ1ZSxcbiAgICAgICAgICAgIGNvbnRhaW5lcjogJ2JvZHknLFxuICAgICAgICAgICAgdGl0bGU6IGZ1bmN0aW9uICgpIHtcbiAgICAgICAgICAgICAgICByZXR1cm4gZ2V0Q2hpbGRQcm9wKHRoaXMsICd0aXRsZScpO1xuICAgICAgICAgICAgfSxcbiAgICAgICAgICAgIGNvbnRlbnQ6IGZ1bmN0aW9uICgpIHtcbiAgICAgICAgICAgICAgICByZXR1cm4gZ2V0Q2hpbGRQcm9wKHRoaXMsICdjb250ZW50Jyk7XG4gICAgICAgICAgICB9LFxuICAgICAgICAgICAgcGxhY2VtZW50OiBmdW5jdGlvbiAocG9wcGVyRWwsIHRyaWdnZXJpbmdFbCkge1xuICAgICAgICAgICAgICAgIGxldCBwbGFjZW1lbnQgPSB0cmlnZ2VyaW5nRWwuZGF0YXNldC5wbGFjZW1lbnQ7XG4gICAgICAgICAgICAgICAgaWYocGxhY2VtZW50ID09IHVuZGVmaW5lZCl7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiBcImJvdHRvbVwiO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICByZXR1cm4gcGxhY2VtZW50O1xuICAgICAgICAgICAgfVxuICAgICAgICB9KTtcbiAgICAgICAgJCgnLnBvcG92ZXItZGlzbWlzcycpLnBvcG92ZXIoe1xuICAgICAgICAgICAgdHJpZ2dlcjogJ2ZvY3VzJ1xuICAgICAgICB9KTtcbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBQcmltYXJ5IG1ldGhvZCBmb3IgaGFuZGxpbmdcbiAgICAgKiAncG9zdHNjcmlwdHMnIG1lc3NhZ2VzLCB3aGljaCB0ZWxsXG4gICAgICogdGhpcyBvYmplY3QgdG8gZ28gdGhyb3VnaCBpdCdzIGFycmF5XG4gICAgICogb2Ygc2NyaXB0IHN0cmluZ3MgYW5kIHRvIGV2YWx1YXRlIHRoZW0uXG4gICAgICogVGhlIGV2YWx1YXRpb24gaXMgZG9uZSBvbiB0aGUgZ2xvYmFsXG4gICAgICogd2luZG93IG9iamVjdCBleHBsaWNpdGx5LlxuICAgICAqIE5PVEU6IEZ1dHVyZSByZWZhY3RvcmluZ3MvcmVzdHJ1Y3R1cmluZ3NcbiAgICAgKiB3aWxsIHJlbW92ZSBtdWNoIG9mIHRoZSBuZWVkIHRvIGNhbGwgZXZhbCFcbiAgICAgKiBAcGFyYW0ge3N0cmluZ30gbWVzc2FnZSAtIFRoZSBpbmNvbWluZyBzdHJpbmdcbiAgICAgKiBmcm9tIHRoZSBzb2NrZXQuXG4gICAgICovXG4gICAgaGFuZGxlUG9zdHNjcmlwdChtZXNzYWdlKXtcbiAgICAgICAgLy8gRWxzZXdoZXJlLCB1cGRhdGUgcG9wb3ZlcnMgZmlyc3RcbiAgICAgICAgLy8gTm93IHdlIGV2YWx1YXRlIHNjcmlwdHMgY29taW5nXG4gICAgICAgIC8vIGFjcm9zcyB0aGUgd2lyZS5cbiAgICAgICAgdGhpcy51cGRhdGVQb3BvdmVycygpO1xuICAgICAgICB3aGlsZSh0aGlzLnBvc3RzY3JpcHRzLmxlbmd0aCl7XG5cdCAgICBsZXQgcG9zdHNjcmlwdCA9IHRoaXMucG9zdHNjcmlwdHMucG9wKCk7XG5cdCAgICB0cnkge1xuXHRcdHdpbmRvdy5ldmFsKHBvc3RzY3JpcHQpO1xuXHQgICAgfSBjYXRjaChlKXtcbiAgICAgICAgICAgICAgICBjb25zb2xlLmVycm9yKFwiRVJST1IgUlVOTklORyBQT1NUU0NSSVBUXCIsIGUpO1xuICAgICAgICAgICAgICAgIGNvbnNvbGUubG9nKHBvc3RzY3JpcHQpO1xuICAgICAgICAgICAgfVxuICAgICAgICB9XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogUHJpbWFyeSBtZXRob2QgZm9yIGhhbmRsaW5nICdub3JtYWwnXG4gICAgICogKGllIG5vbi1wb3N0c2NyaXB0cykgbWVzc2FnZXMgdGhhdCBoYXZlXG4gICAgICogYmVlbiBkZXNlcmlhbGl6ZWQgZnJvbSBKU09OLlxuICAgICAqIEZvciB0aGUgbW9tZW50LCB0aGVzZSBtZXNzYWdlcyBkZWFsXG4gICAgICogZW50aXJlbHkgd2l0aCBET00gcmVwbGFjZW1lbnQgb3BlcmF0aW9ucywgd2hpY2hcbiAgICAgKiB0aGlzIG1ldGhvZCBpbXBsZW1lbnRzLlxuICAgICAqIEBwYXJhbSB7b2JqZWN0fSBtZXNzYWdlIC0gQSBkZXNlcmlhbGl6ZWRcbiAgICAgKiBKU09OIG1lc3NhZ2UgZnJvbSB0aGUgc2VydmVyIHRoYXQgaGFzXG4gICAgICogaW5mb3JtYXRpb24gYWJvdXQgZWxlbWVudHMgdGhhdCBuZWVkIHRvXG4gICAgICogYmUgdXBkYXRlZC5cbiAgICAgKi9cbiAgICBoYW5kbGVNZXNzYWdlKG1lc3NhZ2Upe1xuICAgICAgICBsZXQgbmV3Q29tcG9uZW50cyA9IFtdO1xuXHRpZih0aGlzLmNlbGxzW1wicGFnZV9yb290XCJdID09IHVuZGVmaW5lZCl7XG4gICAgICAgICAgICB0aGlzLmNlbGxzW1wicGFnZV9yb290XCJdID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoXCJwYWdlX3Jvb3RcIik7XG4gICAgICAgICAgICB0aGlzLmNlbGxzW1wiaG9sZGluZ19wZW5cIl0gPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZChcImhvbGRpbmdfcGVuXCIpO1xuICAgICAgICB9XG5cdC8vIFdpdGggdGhlIGV4Y2VwdGlvbiBvZiBgcGFnZV9yb290YCBhbmQgYGhvbGRpbmdfcGVuYCBpZCBub2RlcywgYWxsXG5cdC8vIGVsZW1lbnRzIGluIHRoaXMuY2VsbHMgYXJlIHZpcnR1YWwuIERlcGVuZGlnIG9uIHdoZXRoZXIgd2UgYXJlIGFkZGluZyBhXG5cdC8vIG5ldyBub2RlLCBvciBtYW5pcHVsYXRpbmcgYW4gZXhpc3RpbmcsIHdlIG5lZWVkIHRvIHdvcmsgd2l0aCB0aGUgdW5kZXJseWluZ1xuXHQvLyBET00gbm9kZS4gSGVuY2UgaWYgdGhpcy5jZWxsW21lc3NhZ2UuaWRdIGlzIGEgdmRvbSBlbGVtZW50IHdlIHVzZSBpdHNcblx0Ly8gdW5kZXJseWluZyBkb21Ob2RlIGVsZW1lbnQgd2hlbiBpbiBvcGVyYXRpb25zIGxpa2UgdGhpcy5wcm9qZWN0b3IucmVwbGFjZSgpXG5cdGxldCBjZWxsID0gdGhpcy5jZWxsc1ttZXNzYWdlLmlkXTtcblxuXHRpZiAoY2VsbCAhPT0gdW5kZWZpbmVkICYmIGNlbGwuZG9tTm9kZSAhPT0gdW5kZWZpbmVkKSB7XG5cdCAgICBjZWxsID0gY2VsbC5kb21Ob2RlO1xuXHR9XG5cbiAgICAgICAgaWYobWVzc2FnZS5kaXNjYXJkICE9PSB1bmRlZmluZWQpe1xuICAgICAgICAgICAgLy8gSW4gdGhlIGNhc2Ugd2hlcmUgd2UgaGF2ZSByZWNlaXZlZCBhICdkaXNjYXJkJyBtZXNzYWdlLFxuICAgICAgICAgICAgLy8gYnV0IHRoZSBjZWxsIHJlcXVlc3RlZCBpcyBub3QgYXZhaWxhYmxlIGluIG91clxuICAgICAgICAgICAgLy8gY2VsbHMgY29sbGVjdGlvbiwgd2Ugc2ltcGx5IGRpc3BsYXkgYSB3YXJuaW5nOlxuICAgICAgICAgICAgaWYoY2VsbCA9PSB1bmRlZmluZWQpe1xuICAgICAgICAgICAgICAgIGNvbnNvbGUud2FybihgUmVjZWl2ZWQgZGlzY2FyZCBtZXNzYWdlIGZvciBub24tZXhpc3RpbmcgY2VsbCBpZCAke21lc3NhZ2UuaWR9YCk7XG4gICAgICAgICAgICAgICAgcmV0dXJuO1xuICAgICAgICAgICAgfVxuXHQgICAgLy8gSW5zdGVhZCBvZiByZW1vdmluZyB0aGUgbm9kZSB3ZSByZXBsYWNlIHdpdGggdGhlIGFcblx0ICAgIC8vIGBkaXNwbGF5Om5vbmVgIHN0eWxlIG5vZGUgd2hpY2ggZWZmZWN0aXZlbHkgcmVtb3ZlcyBpdFxuXHQgICAgLy8gZnJvbSB0aGUgRE9NXG5cdCAgICBpZiAoY2VsbC5wYXJlbnROb2RlICE9PSBudWxsKSB7XG5cdFx0dGhpcy5wcm9qZWN0b3IucmVwbGFjZShjZWxsLCAoKSA9PiB7XG5cdFx0ICAgIHJldHVybiBoKFwiZGl2XCIsIHtzdHlsZTogXCJkaXNwbGF5Om5vbmVcIn0sIFtdKTtcblx0XHR9KTtcblx0ICAgIH1cblx0fSBlbHNlIGlmKG1lc3NhZ2UuaWQgIT09IHVuZGVmaW5lZCl7XG5cdCAgICAvLyBBIGRpY3Rpb25hcnkgb2YgaWRzIHdpdGhpbiB0aGUgb2JqZWN0IHRvIHJlcGxhY2UuXG5cdCAgICAvLyBUYXJnZXRzIGFyZSByZWFsIGlkcyBvZiBvdGhlciBvYmplY3RzLlxuXHQgICAgbGV0IHJlcGxhY2VtZW50cyA9IG1lc3NhZ2UucmVwbGFjZW1lbnRzO1xuXG5cdCAgICAvLyBUT0RPOiB0aGlzIGlzIGEgdGVtcG9yYXJ5IGJyYW5jaGluZywgdG8gYmUgcmVtb3ZlZCB3aXRoIGEgbW9yZSBsb2dpY2FsIHNldHVwLiBBc1xuXHQgICAgLy8gb2Ygd3JpdGluZyBpZiB0aGUgbWVzc2FnZSBjb21pbmcgYWNyb3NzIGlzIHNlbmRpbmcgYSBcImtub3duXCIgY29tcG9uZW50IHRoZW4gd2UgdXNlXG5cdCAgICAvLyB0aGUgY29tcG9uZW50IGl0c2VsZiBhcyBvcHBvc2VkIHRvIGJ1aWxkaW5nIGEgdmRvbSBlbGVtZW50IGZyb20gdGhlIHJhdyBodG1sXG5cdCAgICBsZXQgY29tcG9uZW50Q2xhc3MgPSB0aGlzLmNvbXBvbmVudHNbbWVzc2FnZS5jb21wb25lbnRfbmFtZV07XG5cdCAgICBpZiAoY29tcG9uZW50Q2xhc3MgPT09IHVuZGVmaW5lZCkge1xuICAgICAgICAgICAgICAgIGNvbnNvbGUud2FybihgQ291bGQgbm90IGZpbmQgY29tcG9uZW50IGZvciAke21lc3NhZ2UuY29tcG9uZW50X25hbWV9YCk7XG5cdFx0dmFyIHZlbGVtZW50ID0gdGhpcy5odG1sVG9WRG9tRWwobWVzc2FnZS5jb250ZW50cywgbWVzc2FnZS5pZCk7XG5cdCAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgIGxldCBjb21wb25lbnRQcm9wcyA9IE9iamVjdC5hc3NpZ24oe1xuICAgICAgICAgICAgICAgICAgICBpZDogbWVzc2FnZS5pZCxcbiAgICAgICAgICAgICAgICAgICAgbmFtZWRDaGlsZHJlbjogbWVzc2FnZS5uYW1lZENoaWxkcmVuLFxuICAgICAgICAgICAgICAgICAgICBjaGlsZHJlbjogbWVzc2FnZS5jaGlsZHJlbixcbiAgICAgICAgICAgICAgICAgICAgZXh0cmFEYXRhOiBtZXNzYWdlLmV4dHJhX2RhdGFcbiAgICAgICAgICAgICAgICB9LCBtZXNzYWdlLmV4dHJhX2RhdGEpO1xuXHRcdHZhciBjb21wb25lbnQgPSBuZXcgY29tcG9uZW50Q2xhc3MoXG4gICAgICAgICAgICAgICAgICAgIGNvbXBvbmVudFByb3BzLFxuICAgICAgICAgICAgICAgICAgICBtZXNzYWdlLnJlcGxhY2VtZW50X2tleXNcbiAgICAgICAgICAgICAgICApO1xuICAgICAgICAgICAgICAgIHZhciB2ZWxlbWVudCA9IGNvbXBvbmVudC5yZW5kZXIoKTtcbiAgICAgICAgICAgICAgICBuZXdDb21wb25lbnRzLnB1c2goY29tcG9uZW50KTtcblx0ICAgIH1cblxuICAgICAgICAgICAgLy8gSW5zdGFsbCB0aGUgZWxlbWVudCBpbnRvIHRoZSBkb21cbiAgICAgICAgICAgIGlmKGNlbGwgPT09IHVuZGVmaW5lZCl7XG5cdFx0Ly8gVGhpcyBpcyBhIHRvdGFsbHkgbmV3IG5vZGUuXG4gICAgICAgICAgICAgICAgLy8gRm9yIHRoZSBtb21lbnQsIGFkZCBpdCB0byB0aGVcbiAgICAgICAgICAgICAgICAvLyBob2xkaW5nIHBlbi5cblx0XHR0aGlzLnByb2plY3Rvci5hcHBlbmQodGhpcy5jZWxsc1tcImhvbGRpbmdfcGVuXCJdLCAoKSA9PiB7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiB2ZWxlbWVudDtcbiAgICAgICAgICAgICAgICB9KTtcblxuXHRcdHRoaXMuY2VsbHNbbWVzc2FnZS5pZF0gPSB2ZWxlbWVudDtcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgLy8gUmVwbGFjZSB0aGUgZXhpc3RpbmcgY29weSBvZlxuICAgICAgICAgICAgICAgIC8vIHRoZSBub2RlIHdpdGggdGhpcyBpbmNvbWluZ1xuICAgICAgICAgICAgICAgIC8vIGNvcHkuXG5cdFx0aWYoY2VsbC5wYXJlbnROb2RlID09PSBudWxsKXtcblx0XHQgICAgdGhpcy5wcm9qZWN0b3IuYXBwZW5kKHRoaXMuY2VsbHNbXCJob2xkaW5nX3BlblwiXSwgKCkgPT4ge1xuXHRcdFx0cmV0dXJuIHZlbGVtZW50O1xuXHRcdCAgICB9KTtcblx0XHR9IGVsc2Uge1xuXHRcdCAgICB0aGlzLnByb2plY3Rvci5yZXBsYWNlKGNlbGwsICgpID0+IHtyZXR1cm4gdmVsZW1lbnQ7fSk7XG5cdFx0fVxuXHQgICAgfVxuXG4gICAgICAgICAgICB0aGlzLmNlbGxzW21lc3NhZ2UuaWRdID0gdmVsZW1lbnQ7XG5cbiAgICAgICAgICAgIC8vIE5vdyB3aXJlIGluIHJlcGxhY2VtZW50c1xuICAgICAgICAgICAgT2JqZWN0LmtleXMocmVwbGFjZW1lbnRzKS5mb3JFYWNoKChyZXBsYWNlbWVudEtleSwgaWR4KSA9PiB7XG4gICAgICAgICAgICAgICAgbGV0IHRhcmdldCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKHJlcGxhY2VtZW50S2V5KTtcbiAgICAgICAgICAgICAgICBsZXQgc291cmNlID0gbnVsbDtcbiAgICAgICAgICAgICAgICBpZih0aGlzLmNlbGxzW3JlcGxhY2VtZW50c1tyZXBsYWNlbWVudEtleV1dID09PSB1bmRlZmluZWQpe1xuXHRcdCAgICAvLyBUaGlzIGlzIGFjdHVhbGx5IGEgbmV3IG5vZGUuXG4gICAgICAgICAgICAgICAgICAgIC8vIFdlJ2xsIGRlZmluZSBpdCBsYXRlciBpbiB0aGVcbiAgICAgICAgICAgICAgICAgICAgLy8gZXZlbnQgc3RyZWFtLlxuXHRcdCAgICBzb3VyY2UgPSB0aGlzLmgoXCJkaXZcIiwge2lkOiByZXBsYWNlbWVudEtleX0sIFtdKTtcbiAgICAgICAgICAgICAgICAgICAgdGhpcy5jZWxsc1tyZXBsYWNlbWVudHNbcmVwbGFjZW1lbnRLZXldXSA9IHNvdXJjZTsgXG5cdFx0ICAgIHRoaXMucHJvamVjdG9yLmFwcGVuZCh0aGlzLmNlbGxzW1wiaG9sZGluZ19wZW5cIl0sICgpID0+IHtcblx0XHRcdHJldHVybiBzb3VyY2U7XG4gICAgICAgICAgICAgICAgICAgIH0pO1xuXHRcdH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgICAgIC8vIE5vdCBhIG5ldyBub2RlXG4gICAgICAgICAgICAgICAgICAgIHNvdXJjZSA9IHRoaXMuY2VsbHNbcmVwbGFjZW1lbnRzW3JlcGxhY2VtZW50S2V5XV07XG4gICAgICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICAgICAgaWYodGFyZ2V0ICE9IG51bGwpe1xuXHRcdCAgICB0aGlzLnByb2plY3Rvci5yZXBsYWNlKHRhcmdldCwgKCkgPT4ge1xuXHRcdFx0cmV0dXJuIHNvdXJjZTtcbiAgICAgICAgICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICAgICAgbGV0IGVycm9yTXNnID0gYEluIG1lc3NhZ2UgJHttZXNzYWdlfSBjb3VsZG4ndCBmaW5kICR7cmVwbGFjZW1lbnRLZXl9YDtcbiAgICAgICAgICAgICAgICAgICAgdGhyb3cgbmV3IEVycm9yKGVycm9yTXNnKTtcbiAgICAgICAgICAgICAgICAgICAgLy9jb25zb2xlLmxvZyhcIkluIG1lc3NhZ2UgXCIsIG1lc3NhZ2UsIFwiIGNvdWxkbid0IGZpbmQgXCIsIHJlcGxhY2VtZW50S2V5KTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9KTtcbiAgICAgICAgfVxuXG4gICAgICAgIGlmKG1lc3NhZ2UucG9zdHNjcmlwdCAhPT0gdW5kZWZpbmVkKXtcbiAgICAgICAgICAgIHRoaXMucG9zdHNjcmlwdHMucHVzaChtZXNzYWdlLnBvc3RzY3JpcHQpO1xuICAgICAgICB9XG5cbiAgICAgICAgLy8gSWYgd2UgY3JlYXRlZCBhbnkgbmV3IGNvbXBvbmVudHMgZHVyaW5nIHRoaXNcbiAgICAgICAgLy8gbWVzc2FnZSBoYW5kbGluZyBzZXNzaW9uLCB3ZSBmaW5hbGx5IGNhbGxcbiAgICAgICAgLy8gdGhlaXIgYGNvbXBvbmVudERpZExvYWRgIGxpZmVjeWNsZSBtZXRob2RzXG4gICAgICAgIG5ld0NvbXBvbmVudHMuZm9yRWFjaChjb21wb25lbnQgPT4ge1xuICAgICAgICAgICAgY29tcG9uZW50LmNvbXBvbmVudERpZExvYWQoKTtcbiAgICAgICAgfSk7XG5cbiAgICAgICAgLy8gUmVtb3ZlIGxlZnRvdmVyIHJlcGxhY2VtZW50IGRpdnNcbiAgICAgICAgLy8gdGhhdCBhcmUgc3RpbGwgaW4gdGhlIHBhZ2Vfcm9vdFxuICAgICAgICAvLyBhZnRlciB2ZG9tIGluc2VydGlvblxuICAgICAgICBsZXQgcGFnZVJvb3QgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncGFnZV9yb290Jyk7XG4gICAgICAgIGxldCBmb3VuZCA9IHBhZ2VSb290LnF1ZXJ5U2VsZWN0b3JBbGwoJ1tpZCo9XCJfX19fX1wiXScpO1xuICAgICAgICBmb3VuZC5mb3JFYWNoKGVsZW1lbnQgPT4ge1xuICAgICAgICAgICAgZWxlbWVudC5yZW1vdmUoKTtcbiAgICAgICAgfSk7XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogSGVscGVyIGZ1bmN0aW9uIHRoYXQgZ2VuZXJhdGVzIHRoZSB2ZG9tIE5vZGUgZm9yXG4gICAgICogdG8gYmUgZGlzcGxheSB3aGVuIGNvbm5lY3Rpb24gY2xvc2VzXG4gICAgICovXG4gICAgY29ubmVjdGlvbkNsb3NlZFZpZXcoKXtcblx0cmV0dXJuIHRoaXMuaChcIm1haW4uY29udGFpbmVyXCIsIHtyb2xlOiBcIm1haW5cIn0sIFtcblx0ICAgIHRoaXMuaChcImRpdlwiLCB7Y2xhc3M6IFwiYWxlcnQgYWxlcnQtcHJpbWFyeSBjZW50ZXItYmxvY2sgbXQtNVwifSxcblx0XHRbXCJEaXNjb25uZWN0ZWRcIl0pXG5cdF0pO1xuICAgIH1cblxuICAgICAgICAvKipcbiAgICAgKiBUaGlzIGlzIGEgKGhvcGVmdWxseSB0ZW1wb3JhcnkpIGhhY2tcbiAgICAgKiB0aGF0IHdpbGwgaW50ZXJjZXB0IHRoZSBmaXJzdCB0aW1lIGFcbiAgICAgKiBkcm9wZG93biBjYXJhdCBpcyBjbGlja2VkIGFuZCBiaW5kXG4gICAgICogQm9vdHN0cmFwIERyb3Bkb3duIGV2ZW50IGhhbmRsZXJzXG4gICAgICogdG8gaXQgdGhhdCBzaG91bGQgYmUgYm91bmQgdG8gdGhlXG4gICAgICogaWRlbnRpZmllZCBjZWxsLiBXZSBhcmUgZm9yY2VkIHRvIGRvIHRoaXNcbiAgICAgKiBiZWNhdXNlIHRoZSBjdXJyZW50IENlbGxzIGluZnJhc3RydWN0dXJlXG4gICAgICogZG9lcyBub3QgaGF2ZSBmbGV4aWJsZSBldmVudCBiaW5kaW5nL2hhbmRsaW5nLlxuICAgICAqIEBwYXJhbSB7c3RyaW5nfSBjZWxsSWQgLSBUaGUgSUQgb2YgdGhlIGNlbGxcbiAgICAgKiB0byBpZGVudGlmeSBpbiB0aGUgc29ja2V0IGNhbGxiYWNrIHdlIHdpbGxcbiAgICAgKiBiaW5kIHRvIG9wZW4gYW5kIGNsb3NlIGV2ZW50cyBvbiBkcm9wZG93blxuICAgICAqL1xuICAgIGRyb3Bkb3duSW5pdGlhbEJpbmRGb3IoY2VsbElkKXtcbiAgICAgICAgbGV0IGVsZW1lbnRJZCA9IGNlbGxJZCArICctZHJvcGRvd25NZW51QnV0dG9uJztcbiAgICAgICAgbGV0IGVsZW1lbnQgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZChlbGVtZW50SWQpO1xuICAgICAgICBpZighZWxlbWVudCl7XG4gICAgICAgICAgICB0aHJvdyBFcnJvcignRWxlbWVudCBvZiBpZCAnICsgZWxlbWVudElkICsgJyBkb2VzbnQgZXhpc3QhJyk7XG4gICAgICAgIH1cbiAgICAgICAgbGV0IGRyb3Bkb3duTWVudSA9IGVsZW1lbnQucGFyZW50RWxlbWVudDtcbiAgICAgICAgbGV0IGZpcnN0VGltZUNsaWNrZWQgPSBlbGVtZW50LmRhdGFzZXQuZmlyc3RjbGljayA9PSAndHJ1ZSc7XG4gICAgICAgIGlmKGZpcnN0VGltZUNsaWNrZWQpe1xuICAgICAgICAgICAgJChkcm9wZG93bk1lbnUpLm9uKCdzaG93LmJzLmRyb3Bkb3duJywgZnVuY3Rpb24oKXtcbiAgICAgICAgICAgICAgICBjZWxsU29ja2V0LnNlbmRTdHJpbmcoSlNPTi5zdHJpbmdpZnkoe1xuICAgICAgICAgICAgICAgICAgICBldmVudDogJ2Ryb3Bkb3duJyxcbiAgICAgICAgICAgICAgICAgICAgdGFyZ2V0X2NlbGw6IGNlbGxJZCxcbiAgICAgICAgICAgICAgICAgICAgaXNPcGVuOiBmYWxzZVxuICAgICAgICAgICAgICAgIH0pKTtcbiAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgJChkcm9wZG93bk1lbnUpLm9uKCdoaWRlLmJzLmRyb3Bkb3duJywgZnVuY3Rpb24oKXtcbiAgICAgICAgICAgICAgICBjZWxsU29ja2V0LnNlbmRTdHJpbmcoSlNPTi5zdHJpbmdpZnkoe1xuICAgICAgICAgICAgICAgICAgICBldmVudDogJ2Ryb3Bkb3duJyxcbiAgICAgICAgICAgICAgICAgICAgdGFyZ2V0X2NlbGw6IGNlbGxJZCxcbiAgICAgICAgICAgICAgICAgICAgaXNPcGVuOiB0cnVlXG4gICAgICAgICAgICAgICAgfSkpO1xuICAgICAgICAgICAgfSk7XG5cbiAgICAgICAgICAgIC8vIE5vdyBleHBpcmUgdGhlIGZpcnN0IHRpbWUgY2xpY2tlZFxuICAgICAgICAgICAgZWxlbWVudC5kYXRhc2V0LmZpcnN0Y2xpY2sgPSAnZmFsc2UnO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogVW5zYWZlbHkgZXhlY3V0ZXMgYW55IHBhc3NlZCBpbiBzdHJpbmdcbiAgICAgKiBhcyBpZiBpdCBpcyB2YWxpZCBKUyBhZ2FpbnN0IHRoZSBnbG9iYWxcbiAgICAgKiB3aW5kb3cgc3RhdGUuXG4gICAgICovXG4gICAgc3RhdGljIHVuc2FmZWx5RXhlY3V0ZShhU3RyaW5nKXtcbiAgICAgICAgd2luZG93LmV4ZWMoYVN0cmluZyk7XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogSGVscGVyIGZ1bmN0aW9uIHRoYXQgdGFrZXMgc29tZSBpbmNvbWluZ1xuICAgICAqIEhUTUwgc3RyaW5nIGFuZCByZXR1cm5zIGEgbWFxdWV0dGUgaHlwZXJzY3JpcHRcbiAgICAgKiBWRE9NIGVsZW1lbnQgZnJvbSBpdC5cbiAgICAgKiBUaGlzIHVzZXMgdGhlIGludGVybmFsIGJyb3dzZXIgRE9NcGFyc2VyKCkgdG8gZ2VuZXJhdGUgdGhlIGh0bWxcbiAgICAgKiBzdHJ1Y3R1cmUgZnJvbSB0aGUgcmF3IHN0cmluZyBhbmQgdGhlbiByZWN1cnNpdmVseSBidWlsZCB0aGVcbiAgICAgKiBWRE9NIGVsZW1lbnRcbiAgICAgKiBAcGFyYW0ge3N0cmluZ30gaHRtbCAtIFRoZSBtYXJrdXAgdG9cbiAgICAgKiB0cmFuc2Zvcm0gaW50byBhIHJlYWwgZWxlbWVudC5cbiAgICAgKi9cbiAgICBodG1sVG9WRG9tRWwoaHRtbCwgaWQpe1xuXHRsZXQgZG9tID0gdGhpcy5ET01QYXJzZXIucGFyc2VGcm9tU3RyaW5nKGh0bWwsIFwidGV4dC9odG1sXCIpO1xuICAgICAgICBsZXQgZWxlbWVudCA9IGRvbS5ib2R5LmNoaWxkcmVuWzBdO1xuICAgICAgICByZXR1cm4gdGhpcy5fZG9tRWxUb1Zkb21FbChlbGVtZW50LCBpZCk7XG4gICAgfVxuXG4gICAgX2RvbUVsVG9WZG9tRWwoZG9tRWwsIGlkKSB7XG5cdGxldCB0YWdOYW1lID0gZG9tRWwudGFnTmFtZS50b0xvY2FsZUxvd2VyQ2FzZSgpO1xuXHRsZXQgYXR0cnMgPSB7aWQ6IGlkfTtcblx0bGV0IGluZGV4O1xuXG5cdGZvciAoaW5kZXggPSAwOyBpbmRleCA8IGRvbUVsLmF0dHJpYnV0ZXMubGVuZ3RoOyBpbmRleCsrKXtcblx0ICAgIGxldCBpdGVtID0gZG9tRWwuYXR0cmlidXRlcy5pdGVtKGluZGV4KTtcblx0ICAgIGF0dHJzW2l0ZW0ubmFtZV0gPSBpdGVtLnZhbHVlLnRyaW0oKTtcblx0fVxuXG5cdGlmIChkb21FbC5jaGlsZEVsZW1lbnRDb3VudCA9PT0gMCkge1xuXHQgICAgcmV0dXJuIGgodGFnTmFtZSwgYXR0cnMsIFtkb21FbC50ZXh0Q29udGVudF0pO1xuXHR9XG5cblx0bGV0IGNoaWxkcmVuID0gW107XG5cdGZvciAoaW5kZXggPSAwOyBpbmRleCA8IGRvbUVsLmNoaWxkcmVuLmxlbmd0aDsgaW5kZXgrKyl7XG5cdCAgICBsZXQgY2hpbGQgPSBkb21FbC5jaGlsZHJlbltpbmRleF07XG5cdCAgICBjaGlsZHJlbi5wdXNoKHRoaXMuX2RvbUVsVG9WZG9tRWwoY2hpbGQpKTtcblx0fVxuXG5cdHJldHVybiBoKHRhZ05hbWUsIGF0dHJzLCBjaGlsZHJlbik7XG4gICAgfVxufVxuXG5leHBvcnQge0NlbGxIYW5kbGVyLCBDZWxsSGFuZGxlciBhcyBkZWZhdWx0fVxuIiwiLyoqXG4gKiBBIGNvbmNyZXRlIGVycm9yIHRocm93blxuICogaWYgdGhlIGN1cnJlbnQgYnJvd3NlciBkb2Vzbid0XG4gKiBzdXBwb3J0IHdlYnNvY2tldHMsIHdoaWNoIGlzIHZlcnlcbiAqIHVubGlrZWx5LlxuICovXG5jbGFzcyBXZWJzb2NrZXROb3RTdXBwb3J0ZWQgZXh0ZW5kcyBFcnJvciB7XG4gICAgY29uc3RydWN0b3IoYXJncyl7XG4gICAgICAgIHN1cGVyKGFyZ3MpO1xuICAgIH1cbn1cblxuLyoqXG4gKiBUaGlzIGlzIHRoZSBnbG9iYWwgZnJhbWVcbiAqIGNvbnRyb2wuIFdlIG1pZ2h0IGNvbnNpZGVyXG4gKiBwdXR0aW5nIGl0IGVsc2V3aGVyZSwgYnV0XG4gKiBgQ2VsbFNvY2tldGAgaXMgaXRzIG9ubHlcbiAqIGNvbnN1bWVyLlxuICovXG5jb25zdCBGUkFNRVNfUEVSX0FDSyA9IDEwO1xuXG5cbi8qKlxuICogQ2VsbFNvY2tldCBDb250cm9sbGVyXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY2xhc3MgaW1wbGVtZW50cyBhbiBpbnN0YW5jZSBvZlxuICogYSBjb250cm9sbGVyIHRoYXQgd3JhcHMgYSB3ZWJzb2NrZXQgY2xpZW50XG4gKiBjb25uZWN0aW9uIGFuZCBrbm93cyBob3cgdG8gaGFuZGxlIHRoZVxuICogaW5pdGlhbCByb3V0aW5nIG9mIG1lc3NhZ2VzIGFjcm9zcyB0aGUgc29ja2V0LlxuICogYENlbGxTb2NrZXRgIGluc3RhbmNlcyBhcmUgZGVzaWduZWQgc28gdGhhdFxuICogaGFuZGxlcnMgZm9yIHNwZWNpZmljIHR5cGVzIG9mIG1lc3NhZ2VzIGNhblxuICogcmVnaXN0ZXIgdGhlbXNlbHZlcyB3aXRoIGl0LlxuICogTk9URTogRm9yIHRoZSBtb21lbnQsIG1vc3Qgb2YgdGhpcyBjb2RlXG4gKiBoYXMgYmVlbiBjb3BpZWQgdmVyYmF0aW0gZnJvbSB0aGUgaW5saW5lXG4gKiBzY3JpcHRzIHdpdGggb25seSBzbGlnaHQgbW9kaWZpY2F0aW9uLlxuICoqL1xuY2xhc3MgQ2VsbFNvY2tldCB7XG4gICAgY29uc3RydWN0b3IoKXtcbiAgICAgICAgLy8gSW5zdGFuY2UgUHJvcHNcbiAgICAgICAgdGhpcy51cmkgPSB0aGlzLmdldFVyaSgpO1xuICAgICAgICB0aGlzLnNvY2tldCA9IG51bGw7XG4gICAgICAgIHRoaXMuY3VycmVudEJ1ZmZlciA9IHtcbiAgICAgICAgICAgIHJlbWFpbmluZzogbnVsbCxcbiAgICAgICAgICAgIGJ1ZmZlcjogbnVsbCxcbiAgICAgICAgICAgIGhhc0Rpc3BsYXk6IGZhbHNlXG4gICAgICAgIH07XG5cbiAgICAgICAgLyoqXG4gICAgICAgICAqIEEgY2FsbGJhY2sgZm9yIGhhbmRsaW5nIG1lc3NhZ2VzXG4gICAgICAgICAqIHRoYXQgYXJlICdwb3N0c2NyaXB0cydcbiAgICAgICAgICogQGNhbGxiYWNrIHBvc3RzY3JpcHRzSGFuZGxlclxuICAgICAgICAgKiBAcGFyYW0ge3N0cmluZ30gbXNnIC0gVGhlIGZvcndhcmRlZCBtZXNzYWdlXG4gICAgICAgICAqL1xuICAgICAgICB0aGlzLnBvc3RzY3JpcHRzSGFuZGVyID0gbnVsbDtcblxuICAgICAgICAvKipcbiAgICAgICAgICogQSBjYWxsYmFjayBmb3IgaGFuZGxpbmcgbWVzc2FnZXNcbiAgICAgICAgICogdGhhdCBhcmUgbm9ybWFsIEpTT04gZGF0YSBtZXNzYWdlcy5cbiAgICAgICAgICogQGNhbGxiYWNrIG1lc3NhZ2VIYW5kbGVyXG4gICAgICAgICAqIEBwYXJhbSB7b2JqZWN0fSBtc2cgLSBUaGUgZm9yd2FyZGVkIG1lc3NhZ2VcbiAgICAgICAgICovXG4gICAgICAgIHRoaXMubWVzc2FnZUhhbmRsZXIgPSBudWxsO1xuXG4gICAgICAgIC8qKlxuICAgICAgICAgKiBBIGNhbGxiYWNrIGZvciBoYW5kbGluZyBtZXNzYWdlc1xuICAgICAgICAgKiB3aGVuIHRoZSB3ZWJzb2NrZXQgY29ubmVjdGlvbiBjbG9zZXMuXG4gICAgICAgICAqIEBjYWxsYmFjayBjbG9zZUhhbmRsZXJcbiAgICAgICAgICovXG4gICAgICAgIHRoaXMuY2xvc2VIYW5kbGVyID0gbnVsbDtcblxuICAgICAgICAvKipcbiAgICAgICAgICogQSBjYWxsYmFjayBmb3IgaGFuZGxpbmcgbWVzc2FnZXNcbiAgICAgICAgICogd2hlbnQgdGhlIHNvY2tldCBlcnJvcnNcbiAgICAgICAgICogQGNhbGxiYWNrIGVycm9ySGFuZGxlclxuICAgICAgICAgKi9cbiAgICAgICAgdGhpcy5lcnJvckhhbmRsZXIgPSBudWxsO1xuXG4gICAgICAgIC8vIEJpbmQgSW5zdGFuY2UgTWV0aG9kc1xuICAgICAgICB0aGlzLmNvbm5lY3QgPSB0aGlzLmNvbm5lY3QuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5zZW5kU3RyaW5nID0gdGhpcy5zZW5kU3RyaW5nLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuaGFuZGxlUmF3TWVzc2FnZSA9IHRoaXMuaGFuZGxlUmF3TWVzc2FnZS5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLm9uUG9zdHNjcmlwdHMgPSB0aGlzLm9uUG9zdHNjcmlwdHMuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5vbk1lc3NhZ2UgPSB0aGlzLm9uTWVzc2FnZS5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLm9uQ2xvc2UgPSB0aGlzLm9uQ2xvc2UuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5vbkVycm9yID0gdGhpcy5vbkVycm9yLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogUmV0dXJucyBhIHByb3Blcmx5IGZvcm1hdHRlZCBVUklcbiAgICAgKiBmb3IgdGhlIHNvY2tldCBmb3IgYW55IGdpdmVuIGN1cnJlbnRcbiAgICAgKiBicm93c2VyIGxvY2F0aW9uLlxuICAgICAqIEByZXR1cm5zIHtzdHJpbmd9IEEgVVJJIHN0cmluZy5cbiAgICAgKi9cbiAgICBnZXRVcmkoKXtcbiAgICAgICAgbGV0IGxvY2F0aW9uID0gd2luZG93LmxvY2F0aW9uO1xuICAgICAgICBsZXQgdXJpID0gXCJcIjtcbiAgICAgICAgaWYobG9jYXRpb24ucHJvdG9jb2wgPT09IFwiaHR0cHM6XCIpe1xuICAgICAgICAgICAgdXJpICs9IFwid3NzOlwiO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgdXJpICs9IFwid3M6XCI7XG4gICAgICAgIH1cbiAgICAgICAgdXJpID0gYCR7dXJpfS8vJHtsb2NhdGlvbi5ob3N0fWA7XG4gICAgICAgIHVyaSA9IGAke3VyaX0vc29ja2V0JHtsb2NhdGlvbi5wYXRobmFtZX0ke2xvY2F0aW9uLnNlYXJjaH1gO1xuICAgICAgICByZXR1cm4gdXJpO1xuICAgIH1cblxuICAgIC8qKlxuICAgICAqIFRlbGxzIHRoaXMgb2JqZWN0J3MgaW50ZXJuYWwgd2Vic29ja2V0XG4gICAgICogdG8gaW5zdGFudGlhdGUgaXRzZWxmIGFuZCBjb25uZWN0IHRvXG4gICAgICogdGhlIHByb3ZpZGVkIFVSSS4gVGhlIFVSSSB3aWxsIGJlIHNldCB0b1xuICAgICAqIHRoaXMgaW5zdGFuY2UncyBgdXJpYCBwcm9wZXJ0eSBmaXJzdC4gSWYgbm9cbiAgICAgKiB1cmkgaXMgcGFzc2VkLCBgY29ubmVjdCgpYCB3aWxsIHVzZSB0aGUgY3VycmVudFxuICAgICAqIGF0dHJpYnV0ZSdzIHZhbHVlLlxuICAgICAqIEBwYXJhbSB7c3RyaW5nfSB1cmkgLSBBICBVUkkgdG8gY29ubmVjdCB0aGUgc29ja2V0XG4gICAgICogdG8uXG4gICAgICovXG4gICAgY29ubmVjdCh1cmkpe1xuICAgICAgICBpZih1cmkpe1xuICAgICAgICAgICAgdGhpcy51cmkgPSB1cmk7XG4gICAgICAgIH1cbiAgICAgICAgaWYod2luZG93LldlYlNvY2tldCl7XG4gICAgICAgICAgICB0aGlzLnNvY2tldCA9IG5ldyBXZWJTb2NrZXQodGhpcy51cmkpO1xuICAgICAgICB9IGVsc2UgaWYod2luZG93Lk1veldlYlNvY2tldCl7XG4gICAgICAgICAgICB0aGlzLnNvY2tldCA9IE1veldlYlNvY2tldCh0aGlzLnVyaSk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICB0aHJvdyBuZXcgV2Vic29ja2V0Tm90U3VwcG9ydGVkKCk7XG4gICAgICAgIH1cblxuICAgICAgICB0aGlzLnNvY2tldC5vbmNsb3NlID0gdGhpcy5jbG9zZUhhbmRsZXI7XG4gICAgICAgIHRoaXMuc29ja2V0Lm9ubWVzc2FnZSA9IHRoaXMuaGFuZGxlUmF3TWVzc2FnZS5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLnNvY2tldC5vbmVycm9yID0gdGhpcy5lcnJvckhhbmRsZXI7XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogQ29udmVuaWVuY2UgbWV0aG9kIHRoYXQgc2VuZHMgdGhlIHBhc3NlZFxuICAgICAqIHN0cmluZyBvbiB0aGlzIGluc3RhbmNlJ3MgdW5kZXJseWluZ1xuICAgICAqIHdlYnNva2V0IGNvbm5lY3Rpb24uXG4gICAgICogQHBhcmFtIHtzdHJpbmd9IGFTdHJpbmcgLSBBIHN0cmluZyB0byBzZW5kXG4gICAgICovXG4gICAgc2VuZFN0cmluZyhhU3RyaW5nKXtcbiAgICAgICAgaWYodGhpcy5zb2NrZXQpe1xuICAgICAgICAgICAgdGhpcy5zb2NrZXQuc2VuZChhU3RyaW5nKTtcbiAgICAgICAgfVxuICAgIH1cblxuICAgIC8vIElkZWFsbHkgd2UgbW92ZSB0aGUgZG9tIG9wZXJhdGlvbnMgb2ZcbiAgICAvLyB0aGlzIGZ1bmN0aW9uIG91dCBpbnRvIGFub3RoZXIgY2xhc3Mgb3JcbiAgICAvLyBjb250ZXh0LlxuICAgIC8qKlxuICAgICAqIFVzaW5nIHRoZSBpbnRlcm5hbCBgY3VycmVudEJ1ZmZlcmAsIHRoaXNcbiAgICAgKiBtZXRob2QgY2hlY2tzIHRvIHNlZSBpZiBhIGxhcmdlIG11bHRpLWZyYW1lXG4gICAgICogcGllY2Ugb2Ygd2Vic29ja2V0IGRhdGEgaXMgYmVpbmcgc2VudC4gSWYgc28sXG4gICAgICogaXQgcHJlc2VudHMgYW5kIHVwZGF0ZXMgYSBzcGVjaWZpYyBkaXNwbGF5IGluXG4gICAgICogdGhlIERPTSB3aXRoIHRoZSBjdXJyZW50IHBlcmNlbnRhZ2UgZXRjLlxuICAgICAqIEBwYXJhbSB7c3RyaW5nfSBtc2cgLSBUaGUgbWVzc2FnZSB0b1xuICAgICAqIGRpc3BsYXkgaW5zaWRlIHRoZSBlbGVtZW50XG4gICAgICovXG4gICAgc2V0TGFyZ2VEb3dubG9hZERpc3BsYXkobXNnKXtcblxuICAgICAgICBpZihtc2cubGVuZ3RoID09IDAgJiYgIXRoaXMuY3VycmVudEJ1ZmZlci5oYXNEaXNwbGF5KXtcbiAgICAgICAgICAgIHJldHVybjtcbiAgICAgICAgfVxuXG4gICAgICAgIHRoaXMuY3VycmVudEJ1ZmZlci5oYXNEaXNwbGF5ID0gKG1zZy5sZW5ndGggIT0gMCk7XG5cbiAgICAgICAgbGV0IGVsZW1lbnQgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZChcIm9iamVjdF9kYXRhYmFzZV9sYXJnZV9wZW5kaW5nX2Rvd25sb2FkX3RleHRcIik7XG4gICAgICAgIGlmKGVsZW1lbnQgIT0gdW5kZWZpbmVkKXtcbiAgICAgICAgICAgIGVsZW1lbnQuaW5uZXJIVE1MID0gbXNnO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogSGFuZGxlcyB0aGUgYG9ubWVzc2FnZWAgZXZlbnQgb2YgdGhlIHVuZGVybHlpbmdcbiAgICAgKiB3ZWJzb2NrZXQuXG4gICAgICogVGhpcyBtZXRob2Qga25vd3MgaG93IHRvIGZpbGwgdGhlIGludGVybmFsXG4gICAgICogYnVmZmVyICh0byBnZXQgYXJvdW5kIHRoZSBmcmFtZSBsaW1pdCkgYW5kIG9ubHlcbiAgICAgKiB0cmlnZ2VyIHN1YnNlcXVlbnQgaGFuZGxlcnMgZm9yIGluY29taW5nIG1lc3NhZ2VzLlxuICAgICAqIFRPRE86IEJyZWFrIG91dCB0aGlzIG1ldGhvZCBhIGJpdCBtb3JlLiBJdCBoYXMgYmVlblxuICAgICAqIGNvcGllZCBuZWFybHkgdmVyYmF0aW0gZnJvbSB0aGUgb3JpZ2luYWwgY29kZS5cbiAgICAgKiBOT1RFOiBGb3Igbm93LCB0aGVyZSBhcmUgb25seSB0d28gdHlwZXMgb2YgbWVzc2FnZXM6XG4gICAgICogICAgICAgJ3VwZGF0ZXMnICh3ZSBqdXN0IGNhbGwgdGhlc2UgbWVzc2FnZXMpXG4gICAgICogICAgICAgJ3Bvc3RzY3JpcHRzJyAodGhlc2UgYXJlIGp1c3QgcmF3IG5vbi1KU09OIHN0cmluZ3MpXG4gICAgICogSWYgYSBidWZmZXIgaXMgY29tcGxldGUsIHRoaXMgbWV0aG9kIHdpbGwgY2hlY2sgdG8gc2VlIGlmXG4gICAgICogaGFuZGxlcnMgYXJlIHJlZ2lzdGVyZWQgZm9yIHBvc3RzY3JpcHQvbm9ybWFsIG1lc3NhZ2VzXG4gICAgICogYW5kIHdpbGwgdHJpZ2dlciB0aGVtIGlmIHRydWUgaW4gZWl0aGVyIGNhc2UsIHBhc3NpbmdcbiAgICAgKiBhbnkgcGFyc2VkIEpTT04gZGF0YSB0byB0aGUgY2FsbGJhY2tzLlxuICAgICAqIEBwYXJhbSB7RXZlbnR9IGV2ZW50IC0gVGhlIGBvbm1lc3NhZ2VgIGV2ZW50IG9iamVjdFxuICAgICAqIGZyb20gdGhlIHNvY2tldC5cbiAgICAgKi9cbiAgICBoYW5kbGVSYXdNZXNzYWdlKGV2ZW50KXtcbiAgICAgICAgaWYodGhpcy5jdXJyZW50QnVmZmVyLnJlbWFpbmluZyA9PT0gbnVsbCl7XG4gICAgICAgICAgICB0aGlzLmN1cnJlbnRCdWZmZXIucmVtYWluaW5nID0gSlNPTi5wYXJzZShldmVudC5kYXRhKTtcbiAgICAgICAgICAgIHRoaXMuY3VycmVudEJ1ZmZlci5idWZmZXIgPSBbXTtcbiAgICAgICAgICAgIGlmKHRoaXMuY3VycmVudEJ1ZmZlci5oYXNEaXNwbGF5ICYmIHRoaXMuY3VycmVudEJ1ZmZlci5yZW1haW5pbmcgPT0gMSl7XG4gICAgICAgICAgICAgICAgLy8gU0VUIExBUkdFIERPV05MT0FEIERJU1BMQVlcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIHJldHVybjtcbiAgICAgICAgfVxuXG4gICAgICAgIHRoaXMuY3VycmVudEJ1ZmZlci5yZW1haW5pbmcgLT0gMTtcbiAgICAgICAgdGhpcy5jdXJyZW50QnVmZmVyLmJ1ZmZlci5wdXNoKGV2ZW50LmRhdGEpO1xuXG4gICAgICAgIGlmKHRoaXMuY3VycmVudEJ1ZmZlci5idWZmZXIubGVuZ3RoICUgRlJBTUVTX1BFUl9BQ0sgPT0gMCl7XG4gICAgICAgICAgICAvL0FDSyBldmVyeSB0ZW50aCBtZXNzYWdlLiBXZSBoYXZlIHRvIGRvIGFjdGl2ZSBwdXNoYmFja1xuICAgICAgICAgICAgLy9iZWNhdXNlIHRoZSB3ZWJzb2NrZXQgZGlzY29ubmVjdHMgb24gQ2hyb21lIGlmIHlvdSBqYW0gdG9vXG4gICAgICAgICAgICAvL211Y2ggaW4gYXQgb25jZVxuICAgICAgICAgICAgdGhpcy5zZW5kU3RyaW5nKFxuICAgICAgICAgICAgICAgIEpTT04uc3RyaW5naWZ5KHtcbiAgICAgICAgICAgICAgICAgICAgXCJBQ0tcIjogdGhpcy5jdXJyZW50QnVmZmVyLmJ1ZmZlci5sZW5ndGhcbiAgICAgICAgICAgICAgICB9KSk7XG4gICAgICAgICAgICBsZXQgcGVyY2VudGFnZSA9IE1hdGgucm91bmQoMTAwKnRoaXMuY3VycmVudEJ1ZmZlci5idWZmZXIubGVuZ3RoIC8gKHRoaXMuY3VycmVudEJ1ZmZlci5yZW1haW5pbmcgKyB0aGlzLmN1cnJlbnRCdWZmZXIuYnVmZmVyLmxlbmd0aCkpO1xuICAgICAgICAgICAgbGV0IHRvdGFsID0gTWF0aC5yb3VuZCgodGhpcy5jdXJyZW50QnVmZmVyLnJlbWFpbmluZyArIHRoaXMuY3VycmVudEJ1ZmZlci5idWZmZXIubGVuZ3RoKSAvICgxMDI0IC8gMzIpKTtcbiAgICAgICAgICAgIGxldCBwcm9ncmVzc1N0ciA9IGAoRG93bmxvYWRlZCAke3BlcmNlbnRhZ2V9JSBvZiAke3RvdGFsfSBNQilgO1xuICAgICAgICAgICAgdGhpcy5zZXRMYXJnZURvd25sb2FkRGlzcGxheShwcm9ncmVzc1N0cik7XG4gICAgICAgIH1cblxuICAgICAgICBpZih0aGlzLmN1cnJlbnRCdWZmZXIucmVtYWluaW5nID4gMCl7XG4gICAgICAgICAgICByZXR1cm47XG4gICAgICAgIH1cblxuICAgICAgICB0aGlzLnNldExhcmdlRG93bmxvYWREaXNwbGF5KFwiXCIpO1xuXG4gICAgICAgIGxldCBqb2luZWRCdWZmZXIgPSB0aGlzLmN1cnJlbnRCdWZmZXIuYnVmZmVyLmpvaW4oJycpXG5cbiAgICAgICAgdGhpcy5jdXJyZW50QnVmZmVyLnJlbWFpbmluZyA9IG51bGw7XG4gICAgICAgIHRoaXMuY3VycmVudEJ1ZmZlci5idWZmZXIgPSBudWxsO1xuXG4gICAgICAgIGxldCB1cGRhdGUgPSBKU09OLnBhcnNlKGpvaW5lZEJ1ZmZlcik7XG5cbiAgICAgICAgaWYodXBkYXRlID09ICdyZXF1ZXN0X2FjaycpIHtcbiAgICAgICAgICAgIHRoaXMuc2VuZFN0cmluZyhKU09OLnN0cmluZ2lmeSh7J0FDSyc6IDB9KSlcbiAgICAgICAgfSBlbHNlIGlmKHVwZGF0ZSA9PSAncG9zdHNjcmlwdHMnKXtcbiAgICAgICAgICAgIC8vIHVwZGF0ZVBvcG92ZXJzKCk7XG4gICAgICAgICAgICBpZih0aGlzLnBvc3RzY3JpcHRzSGFuZGxlcil7XG4gICAgICAgICAgICAgICAgdGhpcy5wb3N0c2NyaXB0c0hhbmRsZXIodXBkYXRlKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIGlmKHRoaXMubWVzc2FnZUhhbmRsZXIpe1xuICAgICAgICAgICAgICAgIHRoaXMubWVzc2FnZUhhbmRsZXIodXBkYXRlKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgIH1cblxuICAgIC8qKlxuICAgICAqIENvbnZlbmllbmNlIG1ldGhvZCB0aGF0IGJpbmRzXG4gICAgICogdGhlIHBhc3NlZCBjYWxsYmFjayB0byB0aGlzIGluc3RhbmNlJ3NcbiAgICAgKiBwb3N0c2NyaXB0c0hhbmRsZXIsIHdoaWNoIGlzIHNvbWUgbWV0aG9kXG4gICAgICogdGhhdCBoYW5kbGVzIG1lc3NhZ2VzIGZvciBwb3N0c2NyaXB0cy5cbiAgICAgKiBAcGFyYW0ge3Bvc3RzY3JpcHRzSGFuZGxlcn0gY2FsbGJhY2sgLSBBIGhhbmRsZXJcbiAgICAgKiBjYWxsYmFjayBtZXRob2Qgd2l0aCB0aGUgbWVzc2FnZSBhcmd1bWVudC5cbiAgICAgKi9cbiAgICBvblBvc3RzY3JpcHRzKGNhbGxiYWNrKXtcbiAgICAgICAgdGhpcy5wb3N0c2NyaXB0c0hhbmRsZXIgPSBjYWxsYmFjaztcbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBDb252ZW5pZW5jZSBtZXRob2QgdGhhdCBiaW5kc1xuICAgICAqIHRoZSBwYXNzZWQgY2FsbGJhY2sgdG8gdGhpcyBpbnN0YW5jZSdzXG4gICAgICogcG9zdHNjcmlwdHNIYW5kbGVyLCB3aGljaCBpcyBzb21lIG1ldGhvZFxuICAgICAqIHRoYXQgaGFuZGxlcyBtZXNzYWdlcyBmb3IgcG9zdHNjcmlwdHMuXG4gICAgICogQHBhcmFtIHttZXNzYWdlSGFuZGxlcn0gY2FsbGJhY2sgLSBBIGhhbmRsZXJcbiAgICAgKiBjYWxsYmFjayBtZXRob2Qgd2l0aCB0aGUgbWVzc2FnZSBhcmd1bWVudC5cbiAgICAgKi9cbiAgICBvbk1lc3NhZ2UoY2FsbGJhY2spe1xuICAgICAgICB0aGlzLm1lc3NhZ2VIYW5kbGVyID0gY2FsbGJhY2s7XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogQ29udmVuaWVuY2UgbWV0aG9kIHRoYXQgYmluZHMgdGhlXG4gICAgICogcGFzc2VkIGNhbGxiYWNrIHRvIHRoZSB1bmRlcmx5aW5nXG4gICAgICogd2Vic29ja2V0J3MgYG9uY2xvc2VgIGhhbmRsZXIuXG4gICAgICogQHBhcmFtIHtjbG9zZUhhbmRsZXJ9IGNhbGxiYWNrIC0gQSBmdW5jdGlvblxuICAgICAqIHRoYXQgaGFuZGxlcyBjbG9zZSBldmVudHMgb24gdGhlIHNvY2tldC5cbiAgICAgKi9cbiAgICBvbkNsb3NlKGNhbGxiYWNrKXtcbiAgICAgICAgdGhpcy5jbG9zZUhhbmRsZXIgPSBjYWxsYmFjaztcbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBDb252ZW5pZW5jZSBtZXRob2QgdGhhdCBiaW5kcyB0aGVcbiAgICAgKiBwYXNzZWQgY2FsbGJhY2sgdG8gdGhlIHVuZGVybHlpbmdcbiAgICAgKiB3ZWJzb2NrZXRzJyBgb25lcnJvcmAgaGFuZGxlci5cbiAgICAgKiBAcGFyYW0ge2Vycm9ySGFuZGxlcn0gY2FsbGJhY2sgLSBBIGZ1bmN0aW9uXG4gICAgICogdGhhdCBoYW5kbGVzIGVycm9ycyBvbiB0aGUgd2Vic29ja2V0LlxuICAgICAqL1xuICAgIG9uRXJyb3IoY2FsbGJhY2spe1xuICAgICAgICB0aGlzLmVycm9ySGFuZGxlciA9IGNhbGxiYWNrO1xuICAgIH1cbn1cblxuXG5leHBvcnQge0NlbGxTb2NrZXQsIENlbGxTb2NrZXQgYXMgZGVmYXVsdH1cbiIsIi8qKlxuICogV2UgdXNlIGEgc2luZ2xldG9uIHJlZ2lzdHJ5IG9iamVjdFxuICogd2hlcmUgd2UgbWFrZSBhdmFpbGFibGUgYWxsIHBvc3NpYmxlXG4gKiBDb21wb25lbnRzLiBUaGlzIGlzIHVzZWZ1bCBmb3IgV2VicGFjayxcbiAqIHdoaWNoIG9ubHkgYnVuZGxlcyBleHBsaWNpdGx5IHVzZWRcbiAqIENvbXBvbmVudHMgZHVyaW5nIGJ1aWxkIHRpbWUuXG4gKi9cbmltcG9ydCB7QXN5bmNEcm9wZG93biwgQXN5bmNEcm9wZG93bkNvbnRlbnR9IGZyb20gJy4vY29tcG9uZW50cy9Bc3luY0Ryb3Bkb3duJztcbmltcG9ydCB7QmFkZ2V9IGZyb20gJy4vY29tcG9uZW50cy9CYWRnZSc7XG5pbXBvcnQge0J1dHRvbn0gZnJvbSAnLi9jb21wb25lbnRzL0J1dHRvbic7XG5pbXBvcnQge0J1dHRvbkdyb3VwfSBmcm9tICcuL2NvbXBvbmVudHMvQnV0dG9uR3JvdXAnO1xuaW1wb3J0IHtDYXJkfSBmcm9tICcuL2NvbXBvbmVudHMvQ2FyZCc7XG5pbXBvcnQge0NhcmRUaXRsZX0gZnJvbSAnLi9jb21wb25lbnRzL0NhcmRUaXRsZSc7XG5pbXBvcnQge0NpcmNsZUxvYWRlcn0gZnJvbSAnLi9jb21wb25lbnRzL0NpcmNsZUxvYWRlcic7XG5pbXBvcnQge0NsaWNrYWJsZX0gZnJvbSAnLi9jb21wb25lbnRzL0NsaWNrYWJsZSc7XG5pbXBvcnQge0NvZGV9IGZyb20gJy4vY29tcG9uZW50cy9Db2RlJztcbmltcG9ydCB7Q29kZUVkaXRvcn0gZnJvbSAnLi9jb21wb25lbnRzL0NvZGVFZGl0b3InO1xuaW1wb3J0IHtDb2xsYXBzaWJsZVBhbmVsfSBmcm9tICcuL2NvbXBvbmVudHMvQ29sbGFwc2libGVQYW5lbCc7XG5pbXBvcnQge0NvbHVtbnN9IGZyb20gJy4vY29tcG9uZW50cy9Db2x1bW5zJztcbmltcG9ydCB7Q29udGFpbmVyfSBmcm9tICcuL2NvbXBvbmVudHMvQ29udGFpbmVyJztcbmltcG9ydCB7Q29udGV4dHVhbERpc3BsYXl9IGZyb20gJy4vY29tcG9uZW50cy9Db250ZXh0dWFsRGlzcGxheSc7XG5pbXBvcnQge0Ryb3Bkb3dufSBmcm9tICcuL2NvbXBvbmVudHMvRHJvcGRvd24nO1xuaW1wb3J0IHtFeHBhbmRzfSBmcm9tICcuL2NvbXBvbmVudHMvRXhwYW5kcyc7XG5pbXBvcnQge0hlYWRlckJhcn0gZnJvbSAnLi9jb21wb25lbnRzL0hlYWRlckJhcic7XG5pbXBvcnQge0xvYWRDb250ZW50c0Zyb21Vcmx9IGZyb20gJy4vY29tcG9uZW50cy9Mb2FkQ29udGVudHNGcm9tVXJsJztcbmltcG9ydCB7TGFyZ2VQZW5kaW5nRG93bmxvYWREaXNwbGF5fSBmcm9tICcuL2NvbXBvbmVudHMvTGFyZ2VQZW5kaW5nRG93bmxvYWREaXNwbGF5JztcbmltcG9ydCB7TWFpbn0gZnJvbSAnLi9jb21wb25lbnRzL01haW4nO1xuaW1wb3J0IHtNb2RhbH0gZnJvbSAnLi9jb21wb25lbnRzL01vZGFsJztcbmltcG9ydCB7T2N0aWNvbn0gZnJvbSAnLi9jb21wb25lbnRzL09jdGljb24nO1xuaW1wb3J0IHtQYWRkaW5nfSBmcm9tICcuL2NvbXBvbmVudHMvUGFkZGluZyc7XG5pbXBvcnQge1BvcG92ZXJ9IGZyb20gJy4vY29tcG9uZW50cy9Qb3BvdmVyJztcbmltcG9ydCB7Um9vdENlbGx9IGZyb20gJy4vY29tcG9uZW50cy9Sb290Q2VsbCc7XG5pbXBvcnQge1NlcXVlbmNlfSBmcm9tICcuL2NvbXBvbmVudHMvU2VxdWVuY2UnO1xuaW1wb3J0IHtTY3JvbGxhYmxlfSBmcm9tICcuL2NvbXBvbmVudHMvU2Nyb2xsYWJsZSc7XG5pbXBvcnQge1NpbmdsZUxpbmVUZXh0Qm94fSBmcm9tICcuL2NvbXBvbmVudHMvU2luZ2xlTGluZVRleHRCb3gnO1xuaW1wb3J0IHtTcGFufSBmcm9tICcuL2NvbXBvbmVudHMvU3Bhbic7XG5pbXBvcnQge1N1YnNjcmliZWR9IGZyb20gJy4vY29tcG9uZW50cy9TdWJzY3JpYmVkJztcbmltcG9ydCB7U3Vic2NyaWJlZFNlcXVlbmNlfSBmcm9tICcuL2NvbXBvbmVudHMvU3Vic2NyaWJlZFNlcXVlbmNlJztcbmltcG9ydCB7VGFibGV9IGZyb20gJy4vY29tcG9uZW50cy9UYWJsZSc7XG5pbXBvcnQge1RhYnN9IGZyb20gJy4vY29tcG9uZW50cy9UYWJzJztcbmltcG9ydCB7VGV4dH0gZnJvbSAnLi9jb21wb25lbnRzL1RleHQnO1xuaW1wb3J0IHtUcmFjZWJhY2t9IGZyb20gJy4vY29tcG9uZW50cy9UcmFjZWJhY2snO1xuaW1wb3J0IHtfTmF2VGFifSBmcm9tICcuL2NvbXBvbmVudHMvX05hdlRhYic7XG5pbXBvcnQge0dyaWR9IGZyb20gJy4vY29tcG9uZW50cy9HcmlkJztcbmltcG9ydCB7U2hlZXR9IGZyb20gJy4vY29tcG9uZW50cy9TaGVldCc7XG5pbXBvcnQge1Bsb3R9IGZyb20gJy4vY29tcG9uZW50cy9QbG90JztcbmltcG9ydCB7X1Bsb3RVcGRhdGVyfSBmcm9tICcuL2NvbXBvbmVudHMvX1Bsb3RVcGRhdGVyJztcblxuY29uc3QgQ29tcG9uZW50UmVnaXN0cnkgPSB7XG4gICAgQXN5bmNEcm9wZG93bixcbiAgICBBc3luY0Ryb3Bkb3duQ29udGVudCxcbiAgICBCYWRnZSxcbiAgICBCdXR0b24sXG4gICAgQnV0dG9uR3JvdXAsXG4gICAgQ2FyZCxcbiAgICBDYXJkVGl0bGUsXG4gICAgQ2lyY2xlTG9hZGVyLFxuICAgIENsaWNrYWJsZSxcbiAgICBDb2RlLFxuICAgIENvZGVFZGl0b3IsXG4gICAgQ29sbGFwc2libGVQYW5lbCxcbiAgICBDb2x1bW5zLFxuICAgIENvbnRhaW5lcixcbiAgICBDb250ZXh0dWFsRGlzcGxheSxcbiAgICBEcm9wZG93bixcbiAgICBFeHBhbmRzLFxuICAgIEhlYWRlckJhcixcbiAgICBMb2FkQ29udGVudHNGcm9tVXJsLFxuICAgIExhcmdlUGVuZGluZ0Rvd25sb2FkRGlzcGxheSxcbiAgICBNYWluLFxuICAgIE1vZGFsLFxuICAgIE9jdGljb24sXG4gICAgUGFkZGluZyxcbiAgICBQb3BvdmVyLFxuICAgIFJvb3RDZWxsLFxuICAgIFNlcXVlbmNlLFxuICAgIFNjcm9sbGFibGUsXG4gICAgU2luZ2xlTGluZVRleHRCb3gsXG4gICAgU3BhbixcbiAgICBTdWJzY3JpYmVkLFxuICAgIFN1YnNjcmliZWRTZXF1ZW5jZSxcbiAgICBUYWJsZSxcbiAgICBUYWJzLFxuICAgIFRleHQsXG4gICAgVHJhY2ViYWNrLFxuICAgIF9OYXZUYWIsXG4gICAgR3JpZCxcbiAgICBTaGVldCxcbiAgICBQbG90LFxuICAgIF9QbG90VXBkYXRlclxufTtcblxuZXhwb3J0IHtDb21wb25lbnRSZWdpc3RyeSwgQ29tcG9uZW50UmVnaXN0cnkgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIEFzeW5jRHJvcGRvd24gQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBUaGlzIGNvbXBvbmVudCBoYXMgYSBzaW5nbGUgcmVndWxhclxuICogcmVwbGFjZW1lbnQ6XG4gKiAqIGBjb250ZW50c2BcbiAqXG4gKiBOT1RFOiBUaGUgQ2VsbHMgdmVyc2lvbiBvZiB0aGlzIGNoaWxkIGlzXG4gKiBlaXRoZXIgYSBsb2FkaW5nIGluZGljYXRvciwgdGV4dCwgb3IgYVxuICogQXN5bmNEcm9wZG93bkNvbnRlbnQgY2VsbC5cbiAqL1xuXG4vKipcbiAqIEFib3V0IE5hbWVkIENoaWxkcmVuXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogYGNvbnRlbnRgIChzaW5nbGUpIC0gVXN1YWxseSBhbiBBc3luY0Ryb3Bkb3duQ29udGVudCBjZWxsXG4gKiBgbG9hZGluZ0luZGljYXRvcmAgKHNpbmdsZSkgLSBBIENlbGwgdGhhdCBkaXNwbGF5cyB0aGF0IHRoZSBjb250ZW50IGlzIGxvYWRpbmdcbiAqL1xuY2xhc3MgQXN5bmNEcm9wZG93biBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb250ZXh0IHRvIG1ldGhvZHNcbiAgICAgICAgdGhpcy5hZGREcm9wZG93bkxpc3RlbmVyID0gdGhpcy5hZGREcm9wZG93bkxpc3RlbmVyLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMubWFrZUNvbnRlbnQgPSB0aGlzLm1ha2VDb250ZW50LmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiQXN5bmNEcm9wZG93blwiLFxuICAgICAgICAgICAgICAgIGNsYXNzOiBcImNlbGwgYnRuLWdyb3VwXCJcbiAgICAgICAgICAgIH0sIFtcbiAgICAgICAgICAgICAgICBoKCdhJywge2NsYXNzOiBcImJ0biBidG4teHMgYnRuLW91dGxpbmUtc2Vjb25kYXJ5XCJ9LCBbdGhpcy5wcm9wcy5leHRyYURhdGEubGFiZWxUZXh0XSksXG4gICAgICAgICAgICAgICAgaCgnYnV0dG9uJywge1xuICAgICAgICAgICAgICAgICAgICBjbGFzczogXCJidG4gYnRuLXhzIGJ0bi1vdXRsaW5lLXNlY29uZGFyeSBkcm9wZG93bi10b2dnbGUgZHJvcGRvd24tdG9nZ2xlLXNwbGl0XCIsXG4gICAgICAgICAgICAgICAgICAgIHR5cGU6IFwiYnV0dG9uXCIsXG4gICAgICAgICAgICAgICAgICAgIGlkOiBgJHt0aGlzLnByb3BzLmlkfS1kcm9wZG93bk1lbnVCdXR0b25gLFxuICAgICAgICAgICAgICAgICAgICBcImRhdGEtdG9nZ2xlXCI6IFwiZHJvcGRvd25cIixcbiAgICAgICAgICAgICAgICAgICAgYWZ0ZXJDcmVhdGU6IHRoaXMuYWRkRHJvcGRvd25MaXN0ZW5lcixcbiAgICAgICAgICAgICAgICAgICAgXCJkYXRhLWZpcnN0Y2xpY2tcIjogXCJ0cnVlXCJcbiAgICAgICAgICAgICAgICB9KSxcbiAgICAgICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgICAgIGlkOiBgJHt0aGlzLnByb3BzLmlkfS1kcm9wZG93bkNvbnRlbnRXcmFwcGVyYCxcbiAgICAgICAgICAgICAgICAgICAgY2xhc3M6IFwiZHJvcGRvd24tbWVudVwiXG4gICAgICAgICAgICAgICAgfSwgW3RoaXMubWFrZUNvbnRlbnQoKV0pXG4gICAgICAgICAgICBdKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIGFkZERyb3Bkb3duTGlzdGVuZXIoZWxlbWVudCl7XG4gICAgICAgIGxldCBwYXJlbnRFbCA9IGVsZW1lbnQucGFyZW50RWxlbWVudDtcbiAgICAgICAgbGV0IGNvbXBvbmVudCA9IHRoaXM7XG4gICAgICAgIGxldCBmaXJzdFRpbWVDbGlja2VkID0gKGVsZW1lbnQuZGF0YXNldC5maXJzdGNsaWNrID09IFwidHJ1ZVwiKTtcbiAgICAgICAgaWYoZmlyc3RUaW1lQ2xpY2tlZCl7XG4gICAgICAgICAgICAkKHBhcmVudEVsKS5vbignc2hvdy5icy5kcm9wZG93bicsIGZ1bmN0aW9uKCl7XG4gICAgICAgICAgICAgICAgY29uc29sZS5sb2coJ1Nob3cgZHJvcGRvd24gdHJpZ2dlcmVkJyk7XG4gICAgICAgICAgICAgICAgY2VsbFNvY2tldC5zZW5kU3RyaW5nKEpTT04uc3RyaW5naWZ5KHtcbiAgICAgICAgICAgICAgICAgICAgZXZlbnQ6J2Ryb3Bkb3duJyxcbiAgICAgICAgICAgICAgICAgICAgdGFyZ2V0X2NlbGw6IGNvbXBvbmVudC5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICAgICAgaXNPcGVuOiBmYWxzZVxuICAgICAgICAgICAgICAgIH0pKTtcbiAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgJChwYXJlbnRFbCkub24oJ2hpZGUuYnMuZHJvcGRvd24nLCBmdW5jdGlvbigpe1xuICAgICAgICAgICAgICAgIGNvbnNvbGUubG9nKCdoaWRlIGRyb3Bkb3duIHRyaWdnZXJlZCcpO1xuICAgICAgICAgICAgICAgIGNlbGxTb2NrZXQuc2VuZFN0cmluZyhKU09OLnN0cmluZ2lmeSh7XG4gICAgICAgICAgICAgICAgICAgIGV2ZW50OiAnZHJvcGRvd24nLFxuICAgICAgICAgICAgICAgICAgICB0YXJnZXRfY2VsbDogY29tcG9uZW50LnByb3BzLmlkLFxuICAgICAgICAgICAgICAgICAgICBpc09wZW46IHRydWVcbiAgICAgICAgICAgICAgICB9KSk7XG4gICAgICAgICAgICB9KTtcbiAgICAgICAgICAgIGVsZW1lbnQuZGF0YXNldC5maXJzdGNsaWNrID0gZmFsc2U7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBtYWtlQ29udGVudCgpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdjb250ZW50cycpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnY29udGVudCcpO1xuICAgICAgICB9XG4gICAgfVxufVxuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBUaGlzIGNvbXBvbmVudCBoYXMgYSBzaW5nbGUgcmVndWxhclxuICogcmVwbGFjZW1lbnQ6XG4gKiAqIGBjb250ZW50c2BcbiAqL1xuXG4vKipcbiAqIEFib3V0IE5hbWVkIENoaWxkcmVuXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIGBjb250ZW50YCAoc2luZ2xlKSAtIEEgQ2VsbCB0aGF0IGNvbXByaXNlcyB0aGUgZHJvcGRvd24gY29udGVudFxuICogYGxvYWRpbmdJbmRpY2F0b3JgIChzaW5nbGUpIC0gQSBDZWxsIHRoYXQgcmVwcmVzZW50cyBhIHZpc3VhbFxuICogICAgICAgaW5kaWNhdGluZyB0aGF0IHRoZSBjb250ZW50IGlzIGxvYWRpbmdcbiAqL1xuY2xhc3MgQXN5bmNEcm9wZG93bkNvbnRlbnQgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29tcG9uZW50IG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlQ29udGVudCA9IHRoaXMubWFrZUNvbnRlbnQuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICBpZDogYGRyb3Bkb3duQ29udGVudC0ke3RoaXMucHJvcHMuaWR9YCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJBc3luY0Ryb3Bkb3duQ29udGVudFwiXG4gICAgICAgICAgICB9LCBbdGhpcy5tYWtlQ29udGVudCgpXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlQ29udGVudCgpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdjb250ZW50cycpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnY29udGVudCcpO1xuICAgICAgICB9XG4gICAgfVxufVxuXG5cbmV4cG9ydCB7XG4gICAgQXN5bmNEcm9wZG93bixcbiAgICBBc3luY0Ryb3Bkb3duQ29udGVudCxcbiAgICBBc3luY0Ryb3Bkb3duIGFzIGRlZmF1bHRcbn07XG4iLCIvKipcbiAqIEJhZGdlIENlbGwgQ29tcG9uZW50XG4gKi9cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogQmFkZ2UgaGFzIGEgc2luZ2xlIHJlcGxhY2VtZW50OlxuICogKiBgY2hpbGRgXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBDaGlsZHJlblxuICogLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIGBpbm5lcmAgLSBUaGUgY29uY2VudCBjZWxsIG9mIHRoZSBCYWRnZVxuICovXG5jbGFzcyBCYWRnZSBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlciguLi5hcmdzKTtcblxuICAgICAgICAvLyBCaW5kIGNvbXBvbmVudCBtZXRob2RzXG4gICAgICAgIHRoaXMubWFrZUlubmVyID0gdGhpcy5tYWtlSW5uZXIuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuKFxuICAgICAgICAgICAgaCgnc3BhbicsIHtcbiAgICAgICAgICAgICAgICBjbGFzczogYGNlbGwgYmFkZ2UgYmFkZ2UtJHt0aGlzLnByb3BzLmV4dHJhRGF0YS5iYWRnZVN0eWxlfWAsXG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiQmFkZ2VcIlxuICAgICAgICAgICAgfSwgW3RoaXMubWFrZUNvbnRlbnQoKV0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgbWFrZUlubmVyKCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ2NoaWxkJyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCdpbm5lcicpO1xuICAgICAgICB9XG4gICAgfVxufVxuXG5leHBvcnQge0JhZGdlLCBCYWRnZSBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogQnV0dG9uIENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBUaGlzIGNvbXBvbmVudCBoYXMgb25lIHJlZ3VsYXIgcmVwbGFjZW1lbnQ6XG4gKiBgY29udGVudHNgXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBDaGlsZHJlblxuICogLS0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgY29udGVudGAgKHNpbmdsZSkgLSBUaGUgY2VsbCBpbnNpZGUgb2YgdGhlIGJ1dHRvbiAoaWYgYW55KVxuICovXG5jbGFzcyBCdXR0b24gZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29udGV4dCB0byBtZXRob2RzXG4gICAgICAgIHRoaXMubWFrZUNvbnRlbnQgPSB0aGlzLm1ha2VDb250ZW50LmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuX2dldEV2ZW50cyA9IHRoaXMuX2dldEV2ZW50LmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuX2dldEhUTUxDbGFzc2VzID0gdGhpcy5fZ2V0SFRNTENsYXNzZXMuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuKFxuICAgICAgICAgICAgaCgnYnV0dG9uJywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkJ1dHRvblwiLFxuICAgICAgICAgICAgICAgIGNsYXNzOiB0aGlzLl9nZXRIVE1MQ2xhc3NlcygpLFxuICAgICAgICAgICAgICAgIG9uY2xpY2s6IHRoaXMuX2dldEV2ZW50KCdvbmNsaWNrJylcbiAgICAgICAgICAgIH0sIFt0aGlzLm1ha2VDb250ZW50KCldXG4gICAgICAgICAgICAgKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIG1ha2VDb250ZW50KCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ2NvbnRlbnRzJyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCdjb250ZW50Jyk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBfZ2V0RXZlbnQoZXZlbnROYW1lKSB7XG4gICAgICAgIHJldHVybiB0aGlzLnByb3BzLmV4dHJhRGF0YS5ldmVudHNbZXZlbnROYW1lXTtcbiAgICB9XG5cbiAgICBfZ2V0SFRNTENsYXNzZXMoKXtcbiAgICAgICAgbGV0IGNsYXNzU3RyaW5nID0gdGhpcy5wcm9wcy5leHRyYURhdGEuY2xhc3Nlcy5qb2luKFwiIFwiKTtcbiAgICAgICAgLy8gcmVtZW1iZXIgdG8gdHJpbSB0aGUgY2xhc3Mgc3RyaW5nIGR1ZSB0byBhIG1hcXVldHRlIGJ1Z1xuICAgICAgICByZXR1cm4gY2xhc3NTdHJpbmcudHJpbSgpO1xuICAgIH1cbn1cblxuZXhwb3J0IHtCdXR0b24sIEJ1dHRvbiBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogQnV0dG9uR3JvdXAgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyBhIHNpbmdsZSBlbnVtZXJhdGVkXG4gKiByZXBsYWNlbWVudDpcbiAqICogYGJ1dHRvbmBcbiAqL1xuXG4vKipcbiAqIEFib3V0IE5hbWVkIENoaWxkcmVuXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogYGJ1dHRvbnNgIChhcnJheSkgLSBUaGUgY29uc3RpdHVlbnQgYnV0dG9uIGNlbGxzXG4gKi9cbmNsYXNzIEJ1dHRvbkdyb3VwIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICAvLyBCaW5kIGNvbXBvbmVudCBtZXRob2RzXG4gICAgICAgIHRoaXMubWFrZUJ1dHRvbnMgPSB0aGlzLm1ha2VCdXR0b25zLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybihcbiAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJCdXR0b25Hcm91cFwiLFxuICAgICAgICAgICAgICAgIGNsYXNzOiBcImJ0bi1ncm91cFwiLFxuICAgICAgICAgICAgICAgIFwicm9sZVwiOiBcImdyb3VwXCJcbiAgICAgICAgICAgIH0sIHRoaXMubWFrZUJ1dHRvbnMoKVxuICAgICAgICAgICAgIClcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlQnV0dG9ucygpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50c0ZvcignYnV0dG9uJyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZHJlbk5hbWVkKCdidXR0b25zJyk7XG4gICAgICAgIH1cbiAgICB9XG5cbn1cblxuZXhwb3J0IHtCdXR0b25Hcm91cCwgQnV0dG9uR3JvdXAgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIENhcmQgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtQcm9wVHlwZXN9IGZyb20gJy4vdXRpbC9Qcm9wZXJ0eVZhbGlkYXRvcic7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgY29udGFpbnMgdHdvXG4gKiByZWd1bGFyIHJlcGxhY2VtZW50czpcbiAqICogYGNvbnRlbnRzYFxuICogKiBgaGVhZGVyYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIGBib2R5YCAoc2luZ2xlKSAtIFRoZSBjZWxsIHRvIHB1dCBpbiB0aGUgYm9keSBvZiB0aGUgQ2FyZFxuICogYGhlYWRlcmAgKHNpbmdsZSkgLSBBbiBvcHRpb25hbCBoZWFkZXIgY2VsbCB0byBwdXQgYWJvdmVcbiAqICAgICAgICBib2R5XG4gKi9cbmNsYXNzIENhcmQgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29tcG9uZW50IG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlQm9keSA9IHRoaXMubWFrZUJvZHkuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5tYWtlSGVhZGVyID0gdGhpcy5tYWtlSGVhZGVyLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIGxldCBib2R5Q2xhc3MgPSBcImNhcmQtYm9keVwiO1xuICAgICAgICBpZih0aGlzLnByb3BzLmV4dHJhRGF0YS5wYWRkaW5nKXtcbiAgICAgICAgICAgIGJvZHlDbGFzcyA9IGBjYXJkLWJvZHkgcC0ke3RoaXMucHJvcHMuZXh0cmFEYXRhLnBhZGRpbmd9YDtcbiAgICAgICAgfVxuICAgICAgICBsZXQgYm9keUFyZWEgPSBoKCdkaXYnLCB7XG4gICAgICAgICAgICBjbGFzczogYm9keUNsYXNzXG4gICAgICAgIH0sIFt0aGlzLm1ha2VCb2R5KCldKTtcbiAgICAgICAgbGV0IGhlYWRlciA9IHRoaXMubWFrZUhlYWRlcigpO1xuICAgICAgICBsZXQgaGVhZGVyQXJlYSA9IG51bGw7XG4gICAgICAgIGlmKGhlYWRlcil7XG4gICAgICAgICAgICBoZWFkZXJBcmVhID0gaCgnZGl2Jywge2NsYXNzOiBcImNhcmQtaGVhZGVyXCJ9LCBbaGVhZGVyXSk7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIGgoJ2RpdicsXG4gICAgICAgICAgICB7XG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbCBjYXJkXCIsXG4gICAgICAgICAgICAgICAgc3R5bGU6IHRoaXMucHJvcHMuZGl2U3R5bGUsXG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiQ2FyZFwiXG4gICAgICAgICAgICB9LCBbaGVhZGVyQXJlYSwgYm9keUFyZWFdKTtcbiAgICB9XG5cbiAgICBtYWtlQm9keSgpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdjb250ZW50cycpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnYm9keScpO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgbWFrZUhlYWRlcigpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgaWYodGhpcy5yZXBsYWNlbWVudHMuaGFzUmVwbGFjZW1lbnQoJ2hlYWRlcicpKXtcbiAgICAgICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ2hlYWRlcicpO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgcmV0dXJuIG51bGw7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCdoZWFkZXInKTtcbiAgICAgICAgfVxuICAgIH1cbn1cblxuQ2FyZC5wcm9wVHlwZXMgPSB7XG4gICAgcGFkZGluZzoge1xuICAgICAgICBkZXNjcmlwdGlvbjogXCJQYWRkaW5nIHdlaWdodCBhcyBkZWZpbmVkIGJ5IEJvb3N0cmFwIGNzcyBjbGFzc2VzLlwiLFxuICAgICAgICB0eXBlOiBQcm9wVHlwZXMub25lT2YoW1Byb3BUeXBlcy5udW1iZXIsIFByb3BUeXBlcy5zdHJpbmddKVxuICAgIH0sXG4gICAgZGl2U3R5bGU6IHtcbiAgICAgICAgZGVzY3JpcHRpb246IFwiSFRNTCBzdHlsZSBhdHRyaWJ1dGUgc3RyaW5nLlwiLFxuICAgICAgICB0eXBlOiBQcm9wVHlwZXMub25lT2YoW1Byb3BUeXBlcy5zdHJpbmddKVxuICAgIH1cbn07XG5cbmV4cG9ydCB7Q2FyZCwgQ2FyZCBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogQ2FyZFRpdGxlIENlbGxcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzICBzaW5nbGUgcmVndWxhclxuICogcmVwbGFjZW1lbnQ6XG4gKiAqIGBjb250ZW50c2BcbiAqL1xuXG4vKipcbiAqIEFib3V0IE5hbWVkIENoaWxkcmVuXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogYGlubmVyYCAoc2luZ2xlKSAtIFRoZSBpbm5lciBjZWxsIG9mIHRoZSB0aXRsZSBjb21wb25lbnRcbiAqL1xuY2xhc3MgQ2FyZFRpdGxlIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICAvLyBCaW5kIGNvbXBvbmVudCBtZXRob2RzXG4gICAgICAgIHRoaXMubWFrZUlubmVyID0gdGhpcy5tYWtlSW5uZXIuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIGNsYXNzOiBcImNlbGxcIixcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJDYXJkVGl0bGVcIlxuICAgICAgICAgICAgfSwgW1xuICAgICAgICAgICAgICAgIHRoaXMubWFrZUlubmVyKClcbiAgICAgICAgICAgIF0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgbWFrZUlubmVyKCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ2NvbnRlbnRzJyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCdpbm5lcicpO1xuICAgICAgICB9XG4gICAgfVxufVxuXG5leHBvcnQge0NhcmRUaXRsZSwgQ2FyZFRpdGxlIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBDaXJjbGVMb2FkZXIgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cblxuY2xhc3MgQ2lyY2xlTG9hZGVyIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJDaXJjbGVMb2FkZXJcIixcbiAgICAgICAgICAgICAgICBjbGFzczogXCJzcGlubmVyLWdyb3dcIixcbiAgICAgICAgICAgICAgICByb2xlOiBcInN0YXR1c1wiXG4gICAgICAgICAgICB9KVxuICAgICAgICApO1xuICAgIH1cbn1cblxuQ2lyY2xlTG9hZGVyLnByb3BUeXBlcyA9IHtcbn07XG5cbmV4cG9ydCB7Q2lyY2xlTG9hZGVyLCBDaXJjbGVMb2FkZXIgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIENsaWNrYWJsZSBDZWxsIENvbXBvbmVudFxuICovXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyBhIHNpbmdsZVxuICogcmVndWxhciByZXBsYWNlbWVudDpcbiAqICogYGNvbnRlbnRzYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgY29udGVudGAgKHNpbmdsZSkgLSBUaGUgY2VsbCB0aGF0IGNhbiBnbyBpbnNpZGUgdGhlIGNsaWNrYWJsZVxuICogICAgICAgIGNvbXBvbmVudFxuICovXG5jbGFzcyBDbGlja2FibGUgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29udGV4dCB0byBtZXRob2RzXG4gICAgICAgIHRoaXMubWFrZUNvbnRlbnQgPSB0aGlzLm1ha2VDb250ZW50LmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuX2dldEV2ZW50cyA9IHRoaXMuX2dldEV2ZW50LmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybihcbiAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJDbGlja2FibGVcIixcbiAgICAgICAgICAgICAgICBvbmNsaWNrOiB0aGlzLl9nZXRFdmVudCgnb25jbGljaycpLFxuICAgICAgICAgICAgICAgIHN0eWxlOiB0aGlzLnByb3BzLmV4dHJhRGF0YS5kaXZTdHlsZVxuICAgICAgICAgICAgfSwgW1xuICAgICAgICAgICAgICAgIGgoJ2RpdicsIHt9LCBbdGhpcy5tYWtlQ29udGVudCgpXSlcbiAgICAgICAgICAgIF1cbiAgICAgICAgICAgIClcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlQ29udGVudCgpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdjb250ZW50cycpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnY29udGVudCcpO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgX2dldEV2ZW50KGV2ZW50TmFtZSkge1xuICAgICAgICByZXR1cm4gdGhpcy5wcm9wcy5leHRyYURhdGEuZXZlbnRzW2V2ZW50TmFtZV07XG4gICAgfVxufVxuXG5leHBvcnQge0NsaWNrYWJsZSwgQ2xpY2thYmxlIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBDb2RlIENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBUaGlzIGNvbXBvbmVudCBoYXMgYSBzaW5nbGVcbiAqIHJlZ3VsYXIgcmVwbGFjZW1lbnQ6XG4gKiAqIGBjaGlsZGBcbiAqL1xuXG4vKipcbiAqIEFib3V0IE5hbWVkIENoaWxkcmVuXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogYGNvZGVgIChzaW5nbGUpIC0gQ29kZSB0aGF0IHdpbGwgYmUgcmVuZGVyZWQgaW5zaWRlXG4gKi9cbmNsYXNzIENvZGUgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29tcG9uZW50IG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlQ29kZSA9IHRoaXMubWFrZUNvZGUuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIGgoJ3ByZScsXG4gICAgICAgICAgICAgICAgIHtcbiAgICAgICAgICAgICAgICAgICAgIGNsYXNzOiBcImNlbGwgY29kZVwiLFxuICAgICAgICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiQ29kZVwiXG4gICAgICAgICAgICAgICAgIH0sIFtcbiAgICAgICAgICAgICAgICAgICAgIGgoXCJjb2RlXCIsIHt9LCBbdGhpcy5tYWtlQ29kZSgpXSlcbiAgICAgICAgICAgICAgICAgXVxuICAgICAgICAgICAgICAgICk7XG4gICAgfVxuXG4gICAgbWFrZUNvZGUoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50ZWxlbWVudEZvcignY2hpbGQnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ2NvZGUnKTtcbiAgICAgICAgfVxuICAgIH1cbn1cblxuZXhwb3J0IHtDb2RlLCBDb2RlIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBDb2RlRWRpdG9yIENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG5jbGFzcyBDb2RlRWRpdG9yIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcbiAgICAgICAgdGhpcy5lZGl0b3IgPSBudWxsO1xuICAgICAgICAvLyB1c2VkIHRvIHNjaGVkdWxlIHJlZ3VsYXIgc2VydmVyIHVwZGF0ZXNcbiAgICAgICAgdGhpcy5TRVJWRVJfVVBEQVRFX0RFTEFZX01TID0gMTtcbiAgICAgICAgdGhpcy5lZGl0b3JTdHlsZSA9ICd3aWR0aDoxMDAlO2hlaWdodDoxMDAlO21hcmdpbjphdXRvO2JvcmRlcjoxcHggc29saWQgbGlnaHRncmF5Oyc7XG5cbiAgICAgICAgdGhpcy5zZXR1cEVkaXRvciA9IHRoaXMuc2V0dXBFZGl0b3IuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5zZXR1cEtleWJpbmRpbmdzID0gdGhpcy5zZXR1cEtleWJpbmRpbmdzLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuY2hhbmdlSGFuZGxlciA9IHRoaXMuY2hhbmdlSGFuZGxlci5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIGNvbXBvbmVudERpZExvYWQoKSB7XG5cbiAgICAgICAgdGhpcy5zZXR1cEVkaXRvcigpO1xuXG4gICAgICAgIGlmICh0aGlzLmVkaXRvciA9PT0gbnVsbCkge1xuICAgICAgICAgICAgY29uc29sZS5sb2coXCJlZGl0b3IgY29tcG9uZW50IGxvYWRlZCBidXQgZmFpbGVkIHRvIHNldHVwIGVkaXRvclwiKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIGNvbnNvbGUubG9nKFwic2V0dGluZyB1cCBlZGl0b3JcIik7XG4gICAgICAgICAgICB0aGlzLmVkaXRvci5sYXN0X2VkaXRfbWlsbGlzID0gRGF0ZS5ub3coKTtcblxuICAgICAgICAgICAgdGhpcy5lZGl0b3Iuc2V0VGhlbWUoXCJhY2UvdGhlbWUvdGV4dG1hdGVcIik7XG4gICAgICAgICAgICB0aGlzLmVkaXRvci5zZXNzaW9uLnNldE1vZGUoXCJhY2UvbW9kZS9weXRob25cIik7XG4gICAgICAgICAgICB0aGlzLmVkaXRvci5zZXRBdXRvU2Nyb2xsRWRpdG9ySW50b1ZpZXcodHJ1ZSk7XG4gICAgICAgICAgICB0aGlzLmVkaXRvci5zZXNzaW9uLnNldFVzZVNvZnRUYWJzKHRydWUpO1xuICAgICAgICAgICAgdGhpcy5lZGl0b3Iuc2V0VmFsdWUodGhpcy5wcm9wcy5leHRyYURhdGEuaW5pdGlhbFRleHQpO1xuXG4gICAgICAgICAgICBpZiAodGhpcy5wcm9wcy5leHRyYURhdGEuYXV0b2NvbXBsZXRlKSB7XG4gICAgICAgICAgICAgICAgdGhpcy5lZGl0b3Iuc2V0T3B0aW9ucyh7ZW5hYmxlQmFzaWNBdXRvY29tcGxldGlvbjogdHJ1ZX0pO1xuICAgICAgICAgICAgICAgIHRoaXMuZWRpdG9yLnNldE9wdGlvbnMoe2VuYWJsZUxpdmVBdXRvY29tcGxldGlvbjogdHJ1ZX0pO1xuICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICBpZiAodGhpcy5wcm9wcy5leHRyYURhdGEubm9TY3JvbGwpIHtcbiAgICAgICAgICAgICAgICB0aGlzLmVkaXRvci5zZXRPcHRpb24oXCJtYXhMaW5lc1wiLCBJbmZpbml0eSk7XG4gICAgICAgICAgICB9XG5cbiAgICAgICAgICAgIGlmICh0aGlzLnByb3BzLmV4dHJhRGF0YS5mb250U2l6ZSAhPT0gdW5kZWZpbmVkKSB7XG4gICAgICAgICAgICAgICAgdGhpcy5lZGl0b3Iuc2V0T3B0aW9uKFwiZm9udFNpemVcIiwgdGhpcy5wcm9wcy5leHRyYURhdGEuZm9udFNpemUpO1xuICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICBpZiAodGhpcy5wcm9wcy5leHRyYURhdGEubWluTGluZXMgIT09IHVuZGVmaW5lZCkge1xuICAgICAgICAgICAgICAgIHRoaXMuZWRpdG9yLnNldE9wdGlvbihcIm1pbkxpbmVzXCIsIHRoaXMucHJvcHMuZXh0cmFEYXRhLm1pbkxpbmVzKTtcbiAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgdGhpcy5zZXR1cEtleWJpbmRpbmdzKCk7XG5cbiAgICAgICAgICAgIHRoaXMuY2hhbmdlSGFuZGxlcigpO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybiBoKCdkaXYnLFxuICAgICAgICAgICAge1xuICAgICAgICAgICAgICAgIGNsYXNzOiBcImNlbGxcIixcbiAgICAgICAgICAgICAgICBzdHlsZTogdGhpcy5wcm9wcy5leHRyYURhdGEuZGl2U3R5bGUsXG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiQ29kZUVkaXRvclwiXG4gICAgICAgICAgICB9LFxuICAgICAgICAgICAgW2goJ2RpdicsIHsgaWQ6IFwiZWRpdG9yXCIgKyB0aGlzLnByb3BzLmlkLCBzdHlsZTogdGhpcy5lZGl0b3JTdHlsZSB9LCBbXSlcbiAgICAgICAgXSk7XG4gICAgfVxuXG4gICAgc2V0dXBFZGl0b3IoKXtcbiAgICAgICAgbGV0IGVkaXRvcklkID0gXCJlZGl0b3JcIiArIHRoaXMucHJvcHMuaWQ7XG4gICAgICAgIC8vIFRPRE8gVGhlc2UgYXJlIGdsb2JhbCB2YXIgZGVmaW5lZCBpbiBwYWdlLmh0bWxcbiAgICAgICAgLy8gd2Ugc2hvdWxkIGRvIHNvbWV0aGluZyBhYm91dCB0aGlzLlxuXG4gICAgICAgIC8vIGhlcmUgd2UgYmluZyBhbmQgaW5zZXQgdGhlIGVkaXRvciBpbnRvIHRoZSBkaXYgcmVuZGVyZWQgYnlcbiAgICAgICAgLy8gdGhpcy5yZW5kZXIoKVxuICAgICAgICB0aGlzLmVkaXRvciA9IGFjZS5lZGl0KGVkaXRvcklkKTtcbiAgICAgICAgLy8gVE9ETzogZGVhbCB3aXRoIHRoaXMgZ2xvYmFsIGVkaXRvciBsaXN0XG4gICAgICAgIGFjZUVkaXRvcnNbZWRpdG9ySWRdID0gdGhpcy5lZGl0b3I7XG4gICAgfVxuXG4gICAgY2hhbmdlSGFuZGxlcigpIHtcblx0dmFyIGVkaXRvcklkID0gdGhpcy5wcm9wcy5pZDtcblx0dmFyIGVkaXRvciA9IHRoaXMuZWRpdG9yO1xuXHR2YXIgU0VSVkVSX1VQREFURV9ERUxBWV9NUyA9IHRoaXMuU0VSVkVSX1VQREFURV9ERUxBWV9NUztcbiAgICAgICAgdGhpcy5lZGl0b3Iuc2Vzc2lvbi5vbihcbiAgICAgICAgICAgIFwiY2hhbmdlXCIsXG4gICAgICAgICAgICBmdW5jdGlvbihkZWx0YSkge1xuICAgICAgICAgICAgICAgIC8vIFdTXG4gICAgICAgICAgICAgICAgbGV0IHJlc3BvbnNlRGF0YSA9IHtcbiAgICAgICAgICAgICAgICAgICAgZXZlbnQ6ICdlZGl0b3JfY2hhbmdlJyxcbiAgICAgICAgICAgICAgICAgICAgJ3RhcmdldF9jZWxsJzogZWRpdG9ySWQsXG4gICAgICAgICAgICAgICAgICAgIGRhdGE6IGRlbHRhXG4gICAgICAgICAgICAgICAgfTtcbiAgICAgICAgICAgICAgICBjZWxsU29ja2V0LnNlbmRTdHJpbmcoSlNPTi5zdHJpbmdpZnkocmVzcG9uc2VEYXRhKSk7XG4gICAgICAgICAgICAgICAgLy9yZWNvcmQgdGhhdCB3ZSBqdXN0IGVkaXRlZFxuICAgICAgICAgICAgICAgIGVkaXRvci5sYXN0X2VkaXRfbWlsbGlzID0gRGF0ZS5ub3coKTtcblxuXHRcdC8vc2NoZWR1bGUgYSBmdW5jdGlvbiB0byBydW4gaW4gJ1NFUlZFUl9VUERBVEVfREVMQVlfTVMnbXNcblx0XHQvL3RoYXQgd2lsbCB1cGRhdGUgdGhlIHNlcnZlciwgYnV0IG9ubHkgaWYgdGhlIHVzZXIgaGFzIHN0b3BwZWQgdHlwaW5nLlxuXHRcdC8vIFRPRE8gdW5jbGVhciBpZiB0aGlzIGlzIG93cmtpbmcgcHJvcGVybHlcblx0XHR3aW5kb3cuc2V0VGltZW91dChmdW5jdGlvbigpIHtcblx0XHQgICAgaWYgKERhdGUubm93KCkgLSBlZGl0b3IubGFzdF9lZGl0X21pbGxpcyA+PSBTRVJWRVJfVVBEQVRFX0RFTEFZX01TKSB7XG5cdFx0XHQvL3NhdmUgb3VyIGN1cnJlbnQgc3RhdGUgdG8gdGhlIHJlbW90ZSBidWZmZXJcblx0XHRcdGVkaXRvci5jdXJyZW50X2l0ZXJhdGlvbiArPSAxO1xuXHRcdFx0ZWRpdG9yLmxhc3RfZWRpdF9taWxsaXMgPSBEYXRlLm5vdygpO1xuXHRcdFx0ZWRpdG9yLmxhc3RfZWRpdF9zZW50X3RleHQgPSBlZGl0b3IuZ2V0VmFsdWUoKTtcblx0XHRcdC8vIFdTXG5cdFx0XHRsZXQgcmVzcG9uc2VEYXRhID0ge1xuXHRcdFx0ICAgIGV2ZW50OiAnZWRpdGluZycsXG5cdFx0XHQgICAgJ3RhcmdldF9jZWxsJzogZWRpdG9ySWQsXG5cdFx0XHQgICAgYnVmZmVyOiBlZGl0b3IuZ2V0VmFsdWUoKSxcblx0XHRcdCAgICBzZWxlY3Rpb246IGVkaXRvci5zZWxlY3Rpb24uZ2V0UmFuZ2UoKSxcblx0XHRcdCAgICBpdGVyYXRpb246IGVkaXRvci5jdXJyZW50X2l0ZXJhdGlvblxuXHRcdFx0fTtcblx0XHRcdGNlbGxTb2NrZXQuc2VuZFN0cmluZyhKU09OLnN0cmluZ2lmeShyZXNwb25zZURhdGEpKTtcblx0XHQgICAgfVxuXHRcdH0sIFNFUlZFUl9VUERBVEVfREVMQVlfTVMgKyAyKTsgLy9ub3RlIHRoZSAybXMgZ3JhY2UgcGVyaW9kXG4gICAgICAgICAgICB9XG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgc2V0dXBLZXliaW5kaW5ncygpIHtcbiAgICAgICAgY29uc29sZS5sb2coXCJzZXR0aW5nIHVwIGtleWJpbmRpbmdzXCIpO1xuICAgICAgICB0aGlzLnByb3BzLmV4dHJhRGF0YS5rZXliaW5kaW5ncy5tYXAoKGtiKSA9PiB7XG4gICAgICAgICAgICB0aGlzLmVkaXRvci5jb21tYW5kcy5hZGRDb21tYW5kKFxuICAgICAgICAgICAgICAgIHtcbiAgICAgICAgICAgICAgICAgICAgbmFtZTogJ2NtZCcgKyBrYixcbiAgICAgICAgICAgICAgICAgICAgYmluZEtleToge3dpbjogJ0N0cmwtJyArIGtiLCAgbWFjOiAnQ29tbWFuZC0nICsga2J9LFxuICAgICAgICAgICAgICAgICAgICByZWFkT25seTogdHJ1ZSxcbiAgICAgICAgICAgICAgICAgICAgZXhlYzogKCkgPT4ge1xuICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy5lZGl0b3IuY3VycmVudF9pdGVyYXRpb24gKz0gMTtcbiAgICAgICAgICAgICAgICAgICAgICAgIHRoaXMuZWRpdG9yLmxhc3RfZWRpdF9taWxsaXMgPSBEYXRlLm5vdygpO1xuICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy5lZGl0b3IubGFzdF9lZGl0X3NlbnRfdGV4dCA9IHRoaXMuZWRpdG9yLmdldFZhbHVlKCk7XG5cbiAgICAgICAgICAgICAgICAgICAgICAgIC8vIFdTXG4gICAgICAgICAgICAgICAgICAgICAgICBsZXQgcmVzcG9uc2VEYXRhID0ge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGV2ZW50OiAna2V5YmluZGluZycsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgJ3RhcmdldF9jZWxsJzogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAna2V5Jzoga2IsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgJ2J1ZmZlcic6IHRoaXMuZWRpdG9yLmdldFZhbHVlKCksXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgJ3NlbGVjdGlvbic6IHRoaXMuZWRpdG9yLnNlbGVjdGlvbi5nZXRSYW5nZSgpLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICdpdGVyYXRpb24nOiB0aGlzLmVkaXRvci5jdXJyZW50X2l0ZXJhdGlvblxuICAgICAgICAgICAgICAgICAgICAgICAgfTtcbiAgICAgICAgICAgICAgICAgICAgICAgIGNlbGxTb2NrZXQuc2VuZFN0cmluZyhKU09OLnN0cmluZ2lmeShyZXNwb25zZURhdGEpKTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICk7XG4gICAgICAgIH0pO1xuICAgIH1cbn1cblxuZXhwb3J0IHtDb2RlRWRpdG9yLCBDb2RlRWRpdG9yIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBDb2xsYXBzaWJsZVBhbmVsIENlbGwgQ29tcG9uZW50XG4gKi9cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudC5qcyc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIHR3byBzaW5nbGUgdHlwZVxuICogcmVwbGFjZW1lbnRzOlxuICogKiBgY29udGVudGBcbiAqICogYHBhbmVsYFxuICogTm90ZSB0aGF0IGBwYW5lbGAgaXMgb25seSByZW5kZXJlZFxuICogaWYgdGhlIHBhbmVsIGlzIGV4cGFuZGVkXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBDaGlsZHJlblxuICogLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIGBjb250ZW50YCAoc2luZ2xlKSAtIFRoZSBjdXJyZW50IGNvbnRlbnQgQ2VsbCBvZiB0aGUgcGFuZWxcbiAqIGBwYW5lbGAgKHNpbmdsZSkgLSBUaGUgY3VycmVudCAoZXhwYW5kZWQpIHBhbmVsIHZpZXdcbiAqL1xuY2xhc3MgQ29sbGFwc2libGVQYW5lbCBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb21wb25lbnQgbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VQYW5lbCA9IHRoaXMubWFrZVBhbmVsLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMubWFrZUNvbnRlbnQgPSB0aGlzLm1ha2VDb250ZW50LmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIGlmKHRoaXMucHJvcHMuZXh0cmFEYXRhLmlzRXhwYW5kZWQpe1xuICAgICAgICAgICAgcmV0dXJuKFxuICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbCBjb250YWluZXItZmx1aWRcIixcbiAgICAgICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkNvbGxhcHNpYmxlUGFuZWxcIixcbiAgICAgICAgICAgICAgICAgICAgXCJkYXRhLWV4cGFuZGVkXCI6IHRydWUsXG4gICAgICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgICAgICBzdHlsZTogdGhpcy5wcm9wcy5leHRyYURhdGEuZGl2U3R5bGVcbiAgICAgICAgICAgICAgICB9LCBbXG4gICAgICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtjbGFzczogXCJyb3cgZmxleC1ub3dyYXAgbm8tZ3V0dGVyc1wifSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgaCgnZGl2Jywge2NsYXNzOiBcImNvbC1tZC1hdXRvXCJ9LFtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aGlzLm1ha2VQYW5lbCgpXG4gICAgICAgICAgICAgICAgICAgICAgICBdKSxcbiAgICAgICAgICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtjbGFzczogXCJjb2wtc21cIn0sIFtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aGlzLm1ha2VDb250ZW50KClcbiAgICAgICAgICAgICAgICAgICAgICAgIF0pXG4gICAgICAgICAgICAgICAgICAgIF0pXG4gICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgICk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbCBjb250YWluZXItZmx1aWRcIixcbiAgICAgICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkNvbGxhcHNpYmxlUGFuZWxcIixcbiAgICAgICAgICAgICAgICAgICAgXCJkYXRhLWV4cGFuZGVkXCI6IGZhbHNlLFxuICAgICAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICAgICAgc3R5bGU6IHRoaXMucHJvcHMuZXh0cmFEYXRhLmRpdlN0eWxlXG4gICAgICAgICAgICAgICAgfSwgW3RoaXMubWFrZUNvbnRlbnQoKV0pXG4gICAgICAgICAgICApO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgbWFrZUNvbnRlbnQoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY29udGVudCcpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnY29udGVudCcpO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgbWFrZVBhbmVsKCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ3BhbmVsJyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCdwYW5lbCcpO1xuICAgICAgICB9XG4gICAgfVxufVxuXG5cbmV4cG9ydCB7Q29sbGFwc2libGVQYW5lbCwgQ29sbGFwc2libGVQYW5lbCBhcyBkZWZhdWx0fVxuIiwiLyoqXG4gKiBDb2x1bW5zIENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBUaGlzIGNvbXBvbmVudCBoYXMgb25lIGVudW1lcmF0ZWRcbiAqIGtpbmQgb2YgcmVwbGFjZW1lbnQ6XG4gKiAqIGBjYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgZWxlbWVudHNgIChhcnJheSkgLSBDZWxsIGNvbHVtbiBlbGVtZW50c1xuICovXG5jbGFzcyBDb2x1bW5zIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICAvLyBCaW5kIGNvbnRleHQgdG8gbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VJbm5lckNoaWxkcmVuID0gdGhpcy5tYWtlSW5uZXJDaGlsZHJlbi5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGNsYXNzOiBcImNlbGwgY29udGFpbmVyLWZsdWlkXCIsXG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiQ29sdW1uc1wiLFxuICAgICAgICAgICAgICAgIHN0eWxlOiB0aGlzLnByb3BzLmV4dHJhRGF0YS5kaXZTdHlsZVxuICAgICAgICAgICAgfSwgW1xuICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtjbGFzczogXCJyb3cgZmxleC1ub3dyYXBcIn0sIHRoaXMubWFrZUlubmVyQ2hpbGRyZW4oKSlcbiAgICAgICAgICAgIF0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgbWFrZUlubmVyQ2hpbGRyZW4oKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudHNGb3IoJ2MnKS5tYXAocmVwbEVsZW1lbnQgPT4ge1xuICAgICAgICAgICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIGNsYXNzOiBcImNvbC1zbVwiXG4gICAgICAgICAgICAgICAgICAgIH0sIFtyZXBsRWxlbWVudF0pXG4gICAgICAgICAgICAgICAgKTtcbiAgICAgICAgICAgIH0pO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGRyZW5OYW1lZCgnZWxlbWVudHMnKTtcbiAgICAgICAgfVxuICAgIH1cbn1cblxuXG5leHBvcnQge0NvbHVtbnMsIENvbHVtbnMgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIEdlbmVyaWMgYmFzZSBDZWxsIENvbXBvbmVudC5cbiAqIFNob3VsZCBiZSBleHRlbmRlZCBieSBvdGhlclxuICogQ2VsbCBjbGFzc2VzIG9uIEpTIHNpZGUuXG4gKi9cbmltcG9ydCB7UmVwbGFjZW1lbnRzSGFuZGxlcn0gZnJvbSAnLi91dGlsL1JlcGxhY2VtZW50c0hhbmRsZXInO1xuaW1wb3J0IHtQcm9wVHlwZXN9IGZyb20gJy4vdXRpbC9Qcm9wZXJ0eVZhbGlkYXRvcic7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuY2xhc3MgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcyA9IHt9LCByZXBsYWNlbWVudHMgPSBbXSl7XG4gICAgICAgIHRoaXMuaXNDb21wb25lbnQgPSB0cnVlO1xuICAgICAgICB0aGlzLl91cGRhdGVQcm9wcyhwcm9wcyk7XG5cbiAgICAgICAgLy8gUmVwbGFjZW1lbnRzIGhhbmRsaW5nXG4gICAgICAgIHRoaXMucmVwbGFjZW1lbnRzID0gbmV3IFJlcGxhY2VtZW50c0hhbmRsZXIocmVwbGFjZW1lbnRzKTtcbiAgICAgICAgdGhpcy51c2VzUmVwbGFjZW1lbnRzID0gKHJlcGxhY2VtZW50cy5sZW5ndGggPiAwKTtcblxuICAgICAgICAvLyBTZXR1cCBwYXJlbnQgcmVsYXRpb25zaGlwLCBpZlxuICAgICAgICAvLyBhbnkuIEluIHRoaXMgYWJzdHJhY3QgY2xhc3NcbiAgICAgICAgLy8gdGhlcmUgaXNuJ3Qgb25lIGJ5IGRlZmF1bHRcbiAgICAgICAgdGhpcy5wYXJlbnQgPSBudWxsO1xuICAgICAgICB0aGlzLl9zZXR1cENoaWxkUmVsYXRpb25zaGlwcygpO1xuXG4gICAgICAgIC8vIEVuc3VyZSB0aGF0IHdlIGhhdmUgcGFzc2VkIGluIGFuIGlkXG4gICAgICAgIC8vIHdpdGggdGhlIHByb3BzLiBTaG91bGQgZXJyb3Igb3RoZXJ3aXNlLlxuICAgICAgICBpZighdGhpcy5wcm9wcy5pZCB8fCB0aGlzLnByb3BzLmlkID09IHVuZGVmaW5lZCl7XG4gICAgICAgICAgICB0aHJvdyBFcnJvcignWW91IG11c3QgZGVmaW5lIGFuIGlkIGZvciBldmVyeSBjb21wb25lbnQgcHJvcHMhJyk7XG4gICAgICAgIH1cblxuICAgICAgICB0aGlzLnZhbGlkYXRlUHJvcHMoKTtcblxuICAgICAgICAvLyBCaW5kIGNvbnRleHQgdG8gbWV0aG9kc1xuICAgICAgICB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvciA9IHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50c0ZvciA9IHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50c0Zvci5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLmNvbXBvbmVudERpZExvYWQgPSB0aGlzLmNvbXBvbmVudERpZExvYWQuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5jaGlsZHJlbkRvID0gdGhpcy5jaGlsZHJlbkRvLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMubmFtZWRDaGlsZHJlbkRvID0gdGhpcy5uYW1lZENoaWxkcmVuRG8uYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5yZW5kZXJDaGlsZE5hbWVkID0gdGhpcy5yZW5kZXJDaGlsZE5hbWVkLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMucmVuZGVyQ2hpbGRyZW5OYW1lZCA9IHRoaXMucmVuZGVyQ2hpbGRyZW5OYW1lZC5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLl9zZXR1cENoaWxkUmVsYXRpb25zaGlwcyA9IHRoaXMuX3NldHVwQ2hpbGRSZWxhdGlvbnNoaXBzLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuX3VwZGF0ZVByb3BzID0gdGhpcy5fdXBkYXRlUHJvcHMuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5fcmVjdXJzaXZlbHlNYXBOYW1lZENoaWxkcmVuID0gdGhpcy5fcmVjdXJzaXZlbHlNYXBOYW1lZENoaWxkcmVuLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIC8vIE9iamVjdHMgdGhhdCBleHRlbmQgZnJvbVxuICAgICAgICAvLyBtZSBzaG91bGQgb3ZlcnJpZGUgdGhpc1xuICAgICAgICAvLyBtZXRob2QgaW4gb3JkZXIgdG8gZ2VuZXJhdGVcbiAgICAgICAgLy8gc29tZSBjb250ZW50IGZvciB0aGUgdmRvbVxuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ1lvdSBtdXN0IGltcGxlbWVudCBhIGByZW5kZXJgIG1ldGhvZCBvbiBDb21wb25lbnQgb2JqZWN0cyEnKTtcbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBPYmplY3QgdGhhdCBleHRlbmQgZnJvbSBtZSBjb3VsZCBvdmVyd3JpdGUgdGhpcyBtZXRob2QuXG4gICAgICogSXQgaXMgdG8gYmUgdXNlZCBmb3IgbGlmZWN5bGNlIG1hbmFnZW1lbnQgYW5kIGlzIHRvIGJlIGNhbGxlZFxuICAgICAqIGFmdGVyIHRoZSBjb21wb25lbnRzIGxvYWRzLlxuICAgICovXG4gICAgY29tcG9uZW50RGlkTG9hZCgpIHtcbiAgICAgICAgcmV0dXJuIG51bGw7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIFJlc3BvbmRzIHdpdGggYSBoeXBlcnNjcmlwdCBvYmplY3RcbiAgICAgKiB0aGF0IHJlcHJlc2VudHMgYSBkaXYgdGhhdCBpcyBmb3JtYXR0ZWRcbiAgICAgKiBhbHJlYWR5IGZvciB0aGUgcmVndWxhciByZXBsYWNlbWVudC5cbiAgICAgKiBUaGlzIG9ubHkgd29ya3MgZm9yIHJlZ3VsYXIgdHlwZSByZXBsYWNlbWVudHMuXG4gICAgICogRm9yIGVudW1lcmF0ZWQgcmVwbGFjZW1lbnRzLCB1c2VcbiAgICAgKiAjZ2V0UmVwbGFjZW1lbnRFbGVtZW50c0ZvcigpXG4gICAgICovXG4gICAgZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKHJlcGxhY2VtZW50TmFtZSl7XG4gICAgICAgIGxldCByZXBsYWNlbWVudCA9IHRoaXMucmVwbGFjZW1lbnRzLmdldFJlcGxhY2VtZW50Rm9yKHJlcGxhY2VtZW50TmFtZSk7XG4gICAgICAgIGlmKHJlcGxhY2VtZW50KXtcbiAgICAgICAgICAgIGxldCBuZXdJZCA9IGAke3RoaXMucHJvcHMuaWR9XyR7cmVwbGFjZW1lbnR9YDtcbiAgICAgICAgICAgIHJldHVybiBoKCdkaXYnLCB7aWQ6IG5ld0lkLCBrZXk6IG5ld0lkfSwgW10pO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiBudWxsO1xuICAgIH1cblxuICAgIC8qKlxuICAgICAqIFJlc3BvbmQgd2l0aCBhbiBhcnJheSBvZiBoeXBlcnNjcmlwdFxuICAgICAqIG9iamVjdHMgdGhhdCBhcmUgZGl2cyB3aXRoIGlkcyB0aGF0IG1hdGNoXG4gICAgICogcmVwbGFjZW1lbnQgc3RyaW5nIGlkcyBmb3IgdGhlIGtpbmQgb2ZcbiAgICAgKiByZXBsYWNlbWVudCBsaXN0IHRoYXQgaXMgZW51bWVyYXRlZCxcbiAgICAgKiBpZSBgX19fX2J1dHRvbl8xYCwgYF9fX19idXR0b25fMl9fYCBldGMuXG4gICAgICovXG4gICAgZ2V0UmVwbGFjZW1lbnRFbGVtZW50c0ZvcihyZXBsYWNlbWVudE5hbWUpe1xuICAgICAgICBpZighdGhpcy5yZXBsYWNlbWVudHMuaGFzUmVwbGFjZW1lbnQocmVwbGFjZW1lbnROYW1lKSl7XG4gICAgICAgICAgICByZXR1cm4gbnVsbDtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gdGhpcy5yZXBsYWNlbWVudHMubWFwUmVwbGFjZW1lbnRzRm9yKHJlcGxhY2VtZW50TmFtZSwgcmVwbGFjZW1lbnQgPT4ge1xuICAgICAgICAgICAgbGV0IG5ld0lkID0gYCR7dGhpcy5wcm9wcy5pZH1fJHtyZXBsYWNlbWVudH1gO1xuICAgICAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgICAgICBoKCdkaXYnLCB7aWQ6IG5ld0lkLCBrZXk6IG5ld0lkfSlcbiAgICAgICAgICAgICk7XG4gICAgICAgIH0pO1xuICAgIH1cblxuICAgIC8qKlxuICAgICAqIElmIHRoZXJlIGlzIGEgYHByb3BUeXBlc2Agb2JqZWN0IHByZXNlbnQgb25cbiAgICAgKiB0aGUgY29uc3RydWN0b3IgKGllIHRoZSBjb21wb25lbnQgY2xhc3MpLFxuICAgICAqIHRoZW4gcnVuIHRoZSBQcm9wVHlwZXMgdmFsaWRhdG9yIG9uIGl0LlxuICAgICAqL1xuICAgIHZhbGlkYXRlUHJvcHMoKXtcbiAgICAgICAgaWYodGhpcy5jb25zdHJ1Y3Rvci5wcm9wVHlwZXMpe1xuICAgICAgICAgICAgUHJvcFR5cGVzLnZhbGlkYXRlKFxuICAgICAgICAgICAgICAgIHRoaXMuY29uc3RydWN0b3IubmFtZSxcbiAgICAgICAgICAgICAgICB0aGlzLnByb3BzLFxuICAgICAgICAgICAgICAgIHRoaXMuY29uc3RydWN0b3IucHJvcFR5cGVzXG4gICAgICAgICAgICApO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogTG9va3MgdXAgdGhlIHBhc3NlZCBrZXkgaW4gbmFtZWRDaGlsZHJlbiBhbmRcbiAgICAgKiBpZiBmb3VuZCByZXNwb25kcyB3aXRoIHRoZSByZXN1bHQgb2YgY2FsbGluZ1xuICAgICAqIHJlbmRlciBvbiB0aGF0IGNoaWxkIGNvbXBvbmVudC4gUmV0dXJucyBudWxsXG4gICAgICogb3RoZXJ3aXNlLlxuICAgICAqL1xuICAgIHJlbmRlckNoaWxkTmFtZWQoa2V5KXtcbiAgICAgICAgbGV0IGZvdW5kQ2hpbGQgPSB0aGlzLnByb3BzLm5hbWVkQ2hpbGRyZW5ba2V5XTtcbiAgICAgICAgaWYoZm91bmRDaGlsZCl7XG4gICAgICAgICAgICByZXR1cm4gZm91bmRDaGlsZC5yZW5kZXIoKTtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gbnVsbDtcbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBMb29rcyB1cCB0aGUgcGFzc2VkIGtleSBpbiBuYW1lZENoaWxkcmVuXG4gICAgICogYW5kIGlmIGZvdW5kIC0tIGFuZCB0aGUgdmFsdWUgaXMgYW4gQXJyYXlcbiAgICAgKiBvciBBcnJheSBvZiBBcnJheXMsIHJlc3BvbmRzIHdpdGggYW5cbiAgICAgKiBpc29tb3JwaGljIHN0cnVjdHVyZSB0aGF0IGhhcyB0aGUgcmVuZGVyZWRcbiAgICAgKiB2YWx1ZXMgb2YgZWFjaCBjb21wb25lbnQuXG4gICAgICovXG4gICAgcmVuZGVyQ2hpbGRyZW5OYW1lZChrZXkpe1xuICAgICAgICBsZXQgZm91bmRDaGlsZHJlbiA9IHRoaXMucHJvcHMubmFtZWRDaGlsZHJlbltrZXldO1xuICAgICAgICBpZihmb3VuZENoaWxkcmVuKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLl9yZWN1cnNpdmVseU1hcE5hbWVkQ2hpbGRyZW4oZm91bmRDaGlsZHJlbiwgY2hpbGQgPT4ge1xuICAgICAgICAgICAgICAgIHJldHVybiBjaGlsZC5yZW5kZXIoKTtcbiAgICAgICAgICAgIH0pO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiBbXTtcbiAgICB9XG5cblxuXG4gICAgLyoqXG4gICAgICogR2V0dGVyIHRoYXQgd2lsbCByZXNwb25kIHdpdGggdGhlXG4gICAgICogY29uc3RydWN0b3IncyAoYWthIHRoZSAnY2xhc3MnKSBuYW1lXG4gICAgICovXG4gICAgZ2V0IG5hbWUoKXtcbiAgICAgICAgcmV0dXJuIHRoaXMuY29uc3RydWN0b3IubmFtZTtcbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBHZXR0ZXIgdGhhdCB3aWxsIHJlc3BvbmQgd2l0aCBhblxuICAgICAqIGFycmF5IG9mIHJlbmRlcmVkIChpZSBjb25maWd1cmVkXG4gICAgICogaHlwZXJzY3JpcHQpIG9iamVjdHMgdGhhdCByZXByZXNlbnRcbiAgICAgKiBlYWNoIGNoaWxkLiBOb3RlIHRoYXQgd2Ugd2lsbCBjcmVhdGUga2V5c1xuICAgICAqIGZvciB0aGVzZSBiYXNlZCBvbiB0aGUgSUQgb2YgdGhpcyBwYXJlbnRcbiAgICAgKiBjb21wb25lbnQuXG4gICAgICovXG4gICAgZ2V0IHJlbmRlcmVkQ2hpbGRyZW4oKXtcbiAgICAgICAgaWYodGhpcy5wcm9wcy5jaGlsZHJlbi5sZW5ndGggPT0gMCl7XG4gICAgICAgICAgICByZXR1cm4gW107XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIHRoaXMucHJvcHMuY2hpbGRyZW4ubWFwKGNoaWxkQ29tcG9uZW50ID0+IHtcbiAgICAgICAgICAgIGxldCByZW5kZXJlZENoaWxkID0gY2hpbGRDb21wb25lbnQucmVuZGVyKCk7XG4gICAgICAgICAgICByZW5kZXJlZENoaWxkLnByb3BlcnRpZXMua2V5ID0gYCR7dGhpcy5wcm9wcy5pZH0tY2hpbGQtJHtjaGlsZENvbXBvbmVudC5wcm9wcy5pZH1gO1xuICAgICAgICAgICAgcmV0dXJuIHJlbmRlcmVkQ2hpbGQ7XG4gICAgICAgIH0pO1xuICAgIH1cblxuICAgIC8qKiBQdWJsaWMgVXRpbCBNZXRob2RzICoqL1xuXG4gICAgLyoqXG4gICAgICogQ2FsbHMgdGhlIHByb3ZpZGVkIGNhbGxiYWNrIG9uIGVhY2hcbiAgICAgKiBhcnJheSBjaGlsZCBmb3IgdGhpcyBjb21wb25lbnQsIHdpdGhcbiAgICAgKiB0aGUgY2hpbGQgYXMgdGhlIHNvbGUgYXJnIHRvIHRoZVxuICAgICAqIGNhbGxiYWNrXG4gICAgICovXG4gICAgY2hpbGRyZW5EbyhjYWxsYmFjayl7XG4gICAgICAgIHRoaXMucHJvcHMuY2hpbGRyZW4uZm9yRWFjaChjaGlsZCA9PiB7XG4gICAgICAgICAgICBjYWxsYmFjayhjaGlsZCk7XG4gICAgICAgIH0pO1xuICAgIH1cblxuICAgIC8qKlxuICAgICAqIENhbGxzIHRoZSBwcm92aWRlZCBjYWxsYmFjayBvblxuICAgICAqIGVhY2ggbmFtZWQgY2hpbGQgd2l0aCBrZXksIGNoaWxkXG4gICAgICogYXMgdGhlIHR3byBhcmdzIHRvIHRoZSBjYWxsYmFjay5cbiAgICAgKi9cbiAgICBuYW1lZENoaWxkcmVuRG8oY2FsbGJhY2spe1xuICAgICAgICBPYmplY3Qua2V5cyh0aGlzLnByb3BzLm5hbWVkQ2hpbGRyZW4pLmZvckVhY2goa2V5ID0+IHtcbiAgICAgICAgICAgIGxldCBjaGlsZCA9IHRoaXMucHJvcHMubmFtZWRDaGlsZHJlbltrZXldO1xuICAgICAgICAgICAgY2FsbGJhY2soa2V5LCBjaGlsZCk7XG4gICAgICAgIH0pO1xuICAgIH1cblxuICAgIC8qKiBQcml2YXRlIFV0aWwgTWV0aG9kcyAqKi9cblxuICAgIC8qKlxuICAgICAqIFNldHMgdGhlIHBhcmVudCBhdHRyaWJ1dGUgb2YgYWxsIGluY29taW5nXG4gICAgICogYXJyYXkgYW5kL29yIG5hbWVkIGNoaWxkcmVuIHRvIHRoaXNcbiAgICAgKiBpbnN0YW5jZS5cbiAgICAgKi9cbiAgICBfc2V0dXBDaGlsZFJlbGF0aW9uc2hpcHMoKXtcbiAgICAgICAgLy8gTmFtZWQgY2hpbGRyZW4gZmlyc3RcbiAgICAgICAgT2JqZWN0LmtleXModGhpcy5wcm9wcy5uYW1lZENoaWxkcmVuKS5mb3JFYWNoKGtleSA9PiB7XG4gICAgICAgICAgICBsZXQgY2hpbGQgPSB0aGlzLnByb3BzLm5hbWVkQ2hpbGRyZW5ba2V5XTtcbiAgICAgICAgICAgIGNoaWxkLnBhcmVudCA9IHRoaXM7XG4gICAgICAgIH0pO1xuXG4gICAgICAgIC8vIE5vdyBhcnJheSBjaGlsZHJlblxuICAgICAgICB0aGlzLnByb3BzLmNoaWxkcmVuLmZvckVhY2goY2hpbGQgPT4ge1xuICAgICAgICAgICAgY2hpbGQucGFyZW50ID0gdGhpcztcbiAgICAgICAgfSk7XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogVXBkYXRlcyB0aGlzIGNvbXBvbmVudHMgcHJvcHMgb2JqZWN0XG4gICAgICogYmFzZWQgb24gYW4gaW5jb21pbmcgb2JqZWN0XG4gICAgICovXG4gICAgX3VwZGF0ZVByb3BzKGluY29taW5nUHJvcHMpe1xuICAgICAgICB0aGlzLnByb3BzID0gaW5jb21pbmdQcm9wcztcbiAgICAgICAgdGhpcy5wcm9wcy5jaGlsZHJlbiA9IGluY29taW5nUHJvcHMuY2hpbGRyZW4gfHwgW107XG4gICAgICAgIHRoaXMucHJvcHMubmFtZWRDaGlsZHJlbiA9IGluY29taW5nUHJvcHMubmFtZWRDaGlsZHJlbiB8fCB7fTtcbiAgICAgICAgdGhpcy5fc2V0dXBDaGlsZFJlbGF0aW9uc2hpcHMoKTtcbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBSZWN1cnNpdmVseSBtYXBzIGEgb25lIG9yIG11bHRpZGltZW5zaW9uYWxcbiAgICAgKiBuYW1lZCBjaGlsZHJlbiB2YWx1ZSB3aXRoIHRoZSBnaXZlbiBtYXBwaW5nXG4gICAgICogZnVuY3Rpb24uXG4gICAgICovXG4gICAgX3JlY3Vyc2l2ZWx5TWFwTmFtZWRDaGlsZHJlbihjb2xsZWN0aW9uLCBjYWxsYmFjayl7XG4gICAgICAgIHJldHVybiBjb2xsZWN0aW9uLm1hcChpdGVtID0+IHtcbiAgICAgICAgICAgIGlmKEFycmF5LmlzQXJyYXkoaXRlbSkpe1xuICAgICAgICAgICAgICAgIHJldHVybiB0aGlzLl9yZWN1cnNpdmVseU1hcE5hbWVkQ2hpbGRyZW4oaXRlbSwgY2FsbGJhY2spO1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICByZXR1cm4gY2FsbGJhY2soaXRlbSk7XG4gICAgICAgICAgICB9XG4gICAgICAgIH0pO1xuICAgIH1cbn07XG5cbmV4cG9ydCB7Q29tcG9uZW50LCBDb21wb25lbnQgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIENvbnRhaW5lciBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIGEgc2luZ2xlIHJlZ3VsYXJcbiAqIHJlcGxhY2VtZW50OlxuICogKiBgY2hpbGRgXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBDaGlsZHJlblxuICogLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIGBjaGlsZGAgKHNpbmdsZSkgLSBUaGUgQ2VsbCB0aGF0IHRoaXMgY29tcG9uZW50IGNvbnRhaW5zXG4gKi9cbmNsYXNzIENvbnRhaW5lciBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb21wb25lbnQgbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VDaGlsZCA9IHRoaXMubWFrZUNoaWxkLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIGxldCBjaGlsZCA9IHRoaXMubWFrZUNoaWxkKCk7XG4gICAgICAgIGxldCBzdHlsZSA9IFwiXCI7XG4gICAgICAgIGlmKCFjaGlsZCl7XG4gICAgICAgICAgICBzdHlsZSA9IFwiZGlzcGxheTpub25lO1wiO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiQ29udGFpbmVyXCIsXG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbFwiLFxuICAgICAgICAgICAgICAgIHN0eWxlOiBzdHlsZVxuICAgICAgICAgICAgfSwgW2NoaWxkXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlQ2hpbGQoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY2hpbGQnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ2NoaWxkJyk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cbmV4cG9ydCB7Q29udGFpbmVyLCBDb250YWluZXIgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIENvbnRleHR1YWxEaXNwbGF5IENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBUaGlzIGNvbXBvbmVudCBoYXMgYSBzaW5nbGVcbiAqIHJlZ3VsYXIgcmVwbGFjZW1lbnQ6XG4gKiAqIGBjaGlsZGBcbiAqL1xuXG4vKipcbiAqIEFib3V0IE5hbWVkIENoaWxkcmVuXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogYGNoaWxkYCAoc2luZ2xlKSAtIEEgY2hpbGQgY2VsbCB0byBkaXNwbGF5IGluIGEgY29udGV4dFxuICovXG5jbGFzcyBDb250ZXh0dWFsRGlzcGxheSBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb21wb25lbnQgbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VDaGlsZCA9IHRoaXMubWFrZUNoaWxkLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybiBoKCdkaXYnLFxuICAgICAgICAgICAge1xuICAgICAgICAgICAgICAgIGNsYXNzOiBcImNlbGwgY29udGV4dHVhbERpc3BsYXlcIixcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJDb250ZXh0dWFsRGlzcGxheVwiXG4gICAgICAgICAgICB9LCBbdGhpcy5tYWtlQ2hpbGQoKV1cbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlQ2hpbGQoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY2hpbGQnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ2NoaWxkJyk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cbmV4cG9ydCB7Q29udGV4dHVhbERpc3BsYXksIENvbnRleHR1YWxEaXNwbGF5IGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBEcm9wZG93biBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIG9uZSByZWd1bGFyXG4gKiByZXBsYWNlbWVudDpcbiAqICogYHRpdGxlYFxuICogVGhpcyBjb21wb25lbnQgaGFzIG9uZVxuICogZW51bWVyYXRlZCByZXBsYWNlbWVudDpcbiAqICogYGNoaWxkYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgdGl0bGVgIChzaW5nbGUpIC0gQSBDZWxsIHRoYXQgd2lsbCBjb21wcmlzZSB0aGUgdGl0bGUgb2ZcbiAqICAgICAgdGhlIGRyb3Bkb3duXG4gKiBgZHJvcGRvd25JdGVtc2AgKGFycmF5KSAtIEFuIGFycmF5IG9mIGNlbGxzIHRoYXQgYXJlXG4gKiAgICAgIHRoZSBpdGVtcyBpbiB0aGUgZHJvcGRvd25cbiAqL1xuY2xhc3MgRHJvcGRvd24gZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29udGV4dCB0byBtZXRob2RzXG4gICAgICAgIHRoaXMubWFrZVRpdGxlID0gdGhpcy5tYWtlVGl0bGUuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5tYWtlSXRlbXMgPSB0aGlzLm1ha2VJdGVtcy5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkRyb3Bkb3duXCIsXG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiYnRuLWdyb3VwXCJcbiAgICAgICAgICAgIH0sIFtcbiAgICAgICAgICAgICAgICBoKCdhJywge2NsYXNzOiBcImJ0biBidG4teHMgYnRuLW91dGxpbmUtc2Vjb25kYXJ5XCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgIHRoaXMubWFrZVRpdGxlKClcbiAgICAgICAgICAgICAgICBdKSxcbiAgICAgICAgICAgICAgICBoKCdidXR0b24nLCB7XG4gICAgICAgICAgICAgICAgICAgIGNsYXNzOiBcImJ0biBidG4teHMgYnRuLW91dGxpbmUtc2Vjb25kYXJ5IGRyb3Bkb3duLXRvZ2dsZSBkcm9wZG93bi10b2dnbGUtc3BsaXRcIixcbiAgICAgICAgICAgICAgICAgICAgdHlwZTogXCJidXR0b25cIixcbiAgICAgICAgICAgICAgICAgICAgaWQ6IGAke3RoaXMucHJvcHMuZXh0cmFEYXRhLnRhcmdldElkZW50aXR5fS1kcm9wZG93bk1lbnVCdXR0b25gLFxuICAgICAgICAgICAgICAgICAgICBcImRhdGEtdG9nZ2xlXCI6IFwiZHJvcGRvd25cIlxuICAgICAgICAgICAgICAgIH0pLFxuICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtjbGFzczogXCJkcm9wZG93bi1tZW51XCJ9LCB0aGlzLm1ha2VJdGVtcygpKVxuICAgICAgICAgICAgXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlVGl0bGUoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcigndGl0bGUnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ3RpdGxlJyk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBtYWtlSXRlbXMoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIC8vIEZvciBzb21lIHJlYXNvbiwgZHVlIGFnYWluIHRvIHRoZSBDZWxsIGltcGxlbWVudGF0aW9uLFxuICAgICAgICAgICAgLy8gc29tZXRpbWVzIHRoZXJlIGFyZSBub3QgdGhlc2UgY2hpbGQgcmVwbGFjZW1lbnRzLlxuICAgICAgICAgICAgaWYoIXRoaXMucmVwbGFjZW1lbnRzLmhhc1JlcGxhY2VtZW50KCdjaGlsZCcpKXtcbiAgICAgICAgICAgICAgICByZXR1cm4gW107XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRzRm9yKCdjaGlsZCcpLm1hcCgoZWxlbWVudCwgaWR4KSA9PiB7XG4gICAgICAgICAgICAgICAgcmV0dXJuIG5ldyBEcm9wZG93bkl0ZW0oe1xuICAgICAgICAgICAgICAgICAgICBpZDogYCR7dGhpcy5wcm9wcy5pZH0taXRlbS0ke2lkeH1gLFxuICAgICAgICAgICAgICAgICAgICBpbmRleDogaWR4LFxuICAgICAgICAgICAgICAgICAgICBjaGlsZFN1YnN0aXR1dGU6IGVsZW1lbnQsXG4gICAgICAgICAgICAgICAgICAgIHRhcmdldElkZW50aXR5OiB0aGlzLnByb3BzLmV4dHJhRGF0YS50YXJnZXRJZGVudGl0eSxcbiAgICAgICAgICAgICAgICAgICAgZHJvcGRvd25JdGVtSW5mbzogdGhpcy5wcm9wcy5leHRyYURhdGEuZHJvcGRvd25JdGVtSW5mb1xuICAgICAgICAgICAgICAgIH0pLnJlbmRlcigpO1xuICAgICAgICAgICAgfSk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICBpZih0aGlzLnByb3BzLm5hbWVkQ2hpbGRyZW4uZHJvcGRvd25JdGVtcyl7XG4gICAgICAgICAgICAgICAgcmV0dXJuIHRoaXMucHJvcHMubmFtZWRDaGlsZHJlbi5kcm9wZG93bkl0ZW1zLm1hcCgoaXRlbUNvbXBvbmVudCwgaWR4KSA9PiB7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiBuZXcgRHJvcGRvd0l0ZW0oe1xuICAgICAgICAgICAgICAgICAgICAgICAgaWQ6IGAke3RoaXMucHJvcGQuaWR9LWl0ZW0tJHtpZHh9YCxcbiAgICAgICAgICAgICAgICAgICAgICAgIGluZGV4OiBpZHgsXG4gICAgICAgICAgICAgICAgICAgICAgICBjaGlsZFN1YnN0aXR1dGU6IGl0ZW1Db21wb25lbnQucmVuZGVyKCksXG4gICAgICAgICAgICAgICAgICAgICAgICB0YXJnZXRJZGVudGl0eTogdGhpcy5wcm9wcy5leHRyYURhdGEudGFyZ2V0SWRlbnRpdHksXG4gICAgICAgICAgICAgICAgICAgICAgICBkcm9wZG93bkl0ZW1JbmZvOiB0aGlzLnByb3BzLmV4dHJhRGF0YS5kcm9wZG93bkl0ZW1JbmZvXG4gICAgICAgICAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICByZXR1cm4gW107XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICB9XG59XG5cblxuLyoqXG4gKiBBIHByaXZhdGUgc3ViY29tcG9uZW50IGZvciBlYWNoXG4gKiBEcm9wZG93biBtZW51IGl0ZW0uIFdlIG5lZWQgdGhpc1xuICogYmVjYXVzZSBvZiBob3cgY2FsbGJhY2tzIGFyZSBoYW5kbGVkXG4gKiBhbmQgYmVjYXVzZSB0aGUgQ2VsbHMgdmVyc2lvbiBkb2Vzbid0XG4gKiBhbHJlYWR5IGltcGxlbWVudCB0aGlzIGtpbmQgYXMgYSBzZXBhcmF0ZVxuICogZW50aXR5LlxuICovXG5jbGFzcyBEcm9wZG93bkl0ZW0gZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29udGV4dCB0byBtZXRob2RzXG4gICAgICAgIHRoaXMuY2xpY2tIYW5kbGVyID0gdGhpcy5jbGlja0hhbmRsZXIuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ2EnLCB7XG4gICAgICAgICAgICAgICAgY2xhc3M6IFwic3ViY2VsbCBjZWxsLWRyb3Bkb3duLWl0ZW0gZHJvcGRvd24taXRlbVwiLFxuICAgICAgICAgICAgICAgIGtleTogdGhpcy5wcm9wcy5pbmRleCxcbiAgICAgICAgICAgICAgICBvbmNsaWNrOiB0aGlzLmNsaWNrSGFuZGxlclxuICAgICAgICAgICAgfSwgW3RoaXMucHJvcHMuY2hpbGRTdWJzdGl0dXRlXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBjbGlja0hhbmRsZXIoZXZlbnQpe1xuICAgICAgICAvLyBUaGlzIGlzIHN1cGVyIGhhY2t5IGJlY2F1c2Ugb2YgdGhlXG4gICAgICAgIC8vIGN1cnJlbnQgQ2VsbCBpbXBsZW1lbnRhdGlvbi5cbiAgICAgICAgLy8gVGhpcyB3aG9sZSBjb21wb25lbnQgc3RydWN0dXJlIHNob3VsZCBiZSBoZWF2aWx5IHJlZmFjdG9yZWRcbiAgICAgICAgLy8gb25jZSB0aGUgQ2VsbHMgc2lkZSBvZiB0aGluZ3Mgc3RhcnRzIHRvIGNoYW5nZS5cbiAgICAgICAgbGV0IHdoYXRUb0RvID0gdGhpcy5wcm9wcy5kcm9wZG93bkl0ZW1JbmZvW3RoaXMucHJvcHMuaW5kZXgudG9TdHJpbmcoKV07XG4gICAgICAgIGlmKHdoYXRUb0RvID09ICdjYWxsYmFjaycpe1xuICAgICAgICAgICAgbGV0IHJlc3BvbnNlRGF0YSA9IHtcbiAgICAgICAgICAgICAgICBldmVudDogXCJtZW51XCIsXG4gICAgICAgICAgICAgICAgaXg6IHRoaXMucHJvcHMuaW5kZXgsXG4gICAgICAgICAgICAgICAgdGFyZ2V0X2NlbGw6IHRoaXMucHJvcHMudGFyZ2V0SWRlbnRpdHlcbiAgICAgICAgICAgIH07XG4gICAgICAgICAgICBjZWxsU29ja2V0LnNlbmRTdHJpbmcoSlNPTi5zdHJpbmdpZnkocmVzcG9uc2VEYXRhKSk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICB3aW5kb3cubG9jYXRpb24uaHJlZiA9IHdoYXRUb0RvO1xuICAgICAgICB9XG4gICAgfVxufVxuXG5leHBvcnQge0Ryb3Bkb3duLCBEcm9wZG93biBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogRXhwYW5kcyBDZWxsIENvbXBvbmVudFxuICovXG5cbi8qKiBUT0RPL05PVEU6IEl0IGFwcGVhcnMgdGhhdCB0aGUgb3Blbi9jbG9zZWRcbiAgICBTdGF0ZSBmb3IgdGhpcyBjb21wb25lbnQgY291bGQgc2ltcGx5IGJlIHBhc3NlZFxuICAgIHdpdGggdGhlIENlbGwgZGF0YSwgYWxvbmcgd2l0aCB3aGF0IHRvIGRpc3BsYXlcbiAgICBpbiBlaXRoZXIgY2FzZS4gVGhpcyB3b3VsZCBiZSBob3cgaXQgaXMgbm9ybWFsbHlcbiAgICBkb25lIGluIGxhcmdlIHdlYiBhcHBsaWNhdGlvbnMuXG4gICAgQ29uc2lkZXIgcmVmYWN0b3JpbmcgYm90aCBoZXJlIGFuZCBvbiB0aGUgQ2VsbHNcbiAgICBzaWRlXG4qKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyB0d29cbiAqIHJlZ3VsYXIgcmVwbGFjZW1lbnRzOlxuICogKiBgaWNvbmBcbiAqICogYGNoaWxkYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgY29udGVudGAgKHNpbmdsZSkgLSBUaGUgb3BlbiBvciBjbG9zZWQgY2VsbCwgZGVwZW5kaW5nIG9uIHNvdXJjZVxuICogICAgIG9wZW4gc3RhdGVcbiAqIGBpY29uYCAoc2luZ2xlKSAtIFRoZSBDZWxsIG9mIHRoZSBpY29uIHRvIGRpc3BsYXksIGFsc28gZGVwZW5kaW5nXG4gKiAgICAgb24gY2xvc2VkIG9yIG9wZW4gc3RhdGVcbiAqL1xuY2xhc3MgRXhwYW5kcyBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb250ZXh0IHRvIG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlSWNvbiA9IHRoaXMubWFrZUljb24uYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5tYWtlQ29udGVudCA9IHRoaXMubWFrZUNvbnRlbnQuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5fZ2V0RXZlbnRzID0gdGhpcy5fZ2V0RXZlbnQuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkV4cGFuZHNcIixcbiAgICAgICAgICAgICAgICBzdHlsZTogdGhpcy5wcm9wcy5leHRyYURhdGEuZGl2U3R5bGVcbiAgICAgICAgICAgIH0sXG4gICAgICAgICAgICAgICAgW1xuICAgICAgICAgICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgICAgICAgICBzdHlsZTogJ2Rpc3BsYXk6aW5saW5lLWJsb2NrO3ZlcnRpY2FsLWFsaWduOnRvcCcsXG4gICAgICAgICAgICAgICAgICAgICAgICBvbmNsaWNrOiB0aGlzLl9nZXRFdmVudCgnb25jbGljaycpXG4gICAgICAgICAgICAgICAgICAgIH0sXG4gICAgICAgICAgICAgICAgICAgICAgW3RoaXMubWFrZUljb24oKV0pLFxuICAgICAgICAgICAgICAgICAgICBoKCdkaXYnLCB7c3R5bGU6J2Rpc3BsYXk6aW5saW5lLWJsb2NrJ30sXG4gICAgICAgICAgICAgICAgICAgICAgW3RoaXMubWFrZUNvbnRlbnQoKV0pLFxuICAgICAgICAgICAgICAgIF1cbiAgICAgICAgICAgIClcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlQ29udGVudCgpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdjaGlsZCcpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnY29udGVudCcpO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgbWFrZUljb24oKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignaWNvbicpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnaWNvbicpO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgX2dldEV2ZW50KGV2ZW50TmFtZSkge1xuICAgICAgICByZXR1cm4gdGhpcy5wcm9wcy5leHRyYURhdGEuZXZlbnRzW2V2ZW50TmFtZV07XG4gICAgfVxufVxuXG5leHBvcnQge0V4cGFuZHMsIEV4cGFuZHMgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIEdyaWQgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyAzIGVudW1lcmFibGVcbiAqIHJlcGxhY2VtZW50czpcbiAqICogYGhlYWRlcmBcbiAqICogYHJvd2xhYmVsYFxuICogKiBgY2hpbGRgXG4gKlxuICogTk9URTogQ2hpbGQgaXMgYSAyLWRpbWVuc2lvbmFsXG4gKiBlbnVtZXJhdGVkIHJlcGxhY2VtZW50IVxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgaGVhZGVyc2AgKGFycmF5KSAtIEFuIGFycmF5IG9mIHRhYmxlIGhlYWRlciBjZWxsc1xuICogYHJvd0xhYmVsc2AgKGFycmF5KSAtIEFuIGFycmF5IG9mIHJvdyBsYWJlbCBjZWxsc1xuICogYGRhdGFDZWxsc2AgKGFycmF5LW9mLWFycmF5KSAtIEEgMi1kaW1lbnNpb25hbCBhcnJheVxuICogICAgIG9mIGNlbGxzIHRoYXQgc2VydmUgYXMgdGFibGUgZGF0YSwgd2hlcmUgcm93c1xuICogICAgIGFyZSB0aGUgb3V0ZXIgYXJyYXkgYW5kIGNvbHVtbnMgYXJlIHRoZSBpbm5lclxuICogICAgIGFycmF5LlxuICovXG5jbGFzcyBHcmlkIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICAvLyBCaW5kIGNvbnRleHQgdG8gbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VIZWFkZXJzID0gdGhpcy5tYWtlSGVhZGVycy5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLm1ha2VSb3dzID0gdGhpcy5tYWtlUm93cy5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLl9tYWtlUmVwbGFjZW1lbnRIZWFkZXJFbGVtZW50cyA9IHRoaXMuX21ha2VSZXBsYWNlbWVudEhlYWRlckVsZW1lbnRzLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuX21ha2VSZXBsYWNlbWVudFJvd0VsZW1lbnRzID0gdGhpcy5fbWFrZVJlcGxhY2VtZW50Um93RWxlbWVudHMuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgbGV0IHRvcFRhYmxlSGVhZGVyID0gbnVsbDtcbiAgICAgICAgaWYodGhpcy5wcm9wcy5leHRyYURhdGEuaGFzVG9wSGVhZGVyKXtcbiAgICAgICAgICAgIHRvcFRhYmxlSGVhZGVyID0gaCgndGgnKTtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgndGFibGUnLCB7XG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiR3JpZFwiLFxuICAgICAgICAgICAgICAgIGNsYXNzOiBcImNlbGwgdGFibGUtaHNjcm9sbCB0YWJsZS1zbSB0YWJsZS1zdHJpcGVkXCJcbiAgICAgICAgICAgIH0sIFtcbiAgICAgICAgICAgICAgICBoKCd0aGVhZCcsIHt9LCBbXG4gICAgICAgICAgICAgICAgICAgIGgoJ3RyJywge30sIFt0b3BUYWJsZUhlYWRlciwgLi4udGhpcy5tYWtlSGVhZGVycygpXSlcbiAgICAgICAgICAgICAgICBdKSxcbiAgICAgICAgICAgICAgICBoKCd0Ym9keScsIHt9LCB0aGlzLm1ha2VSb3dzKCkpXG4gICAgICAgICAgICBdKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIG1ha2VIZWFkZXJzKCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5fbWFrZVJlcGxhY2VtZW50SGVhZGVyRWxlbWVudHMoKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkcmVuTmFtZWQoJ2hlYWRlcnMnKS5tYXAoKGhlYWRlckVsLCBjb2xJZHgpID0+IHtcbiAgICAgICAgICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgICAgICAgICBoKCd0aCcsIHtrZXk6IGAke3RoaXMucHJvcHMuaWR9LWdyaWQtdGgtJHtjb2xJZHh9YH0sIFtcbiAgICAgICAgICAgICAgICAgICAgICAgIGhlYWRlckVsXG4gICAgICAgICAgICAgICAgICAgIF0pXG4gICAgICAgICAgICAgICAgKTtcbiAgICAgICAgICAgIH0pO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgbWFrZVJvd3MoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLl9tYWtlUmVwbGFjZW1lbnRSb3dFbGVtZW50cygpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGRyZW5OYW1lZCgnZGF0YUNlbGxzJykubWFwKChkYXRhUm93LCByb3dJZHgpID0+IHtcbiAgICAgICAgICAgICAgICBsZXQgY29sdW1ucyA9IGRhdGFSb3cubWFwKChjb2x1bW4sIGNvbElkeCkgPT4ge1xuICAgICAgICAgICAgICAgICAgICByZXR1cm4oXG4gICAgICAgICAgICAgICAgICAgICAgICBoKCd0ZCcsIHtrZXk6IGAke3RoaXMucHJvcHMuaWR9LWdyaWQtY29sLSR7cm93SWR4fS0ke2NvbElkeH1gfSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNvbHVtblxuICAgICAgICAgICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgICAgICAgICAgKTtcbiAgICAgICAgICAgICAgICB9KTtcbiAgICAgICAgICAgICAgICBsZXQgcm93TGFiZWxFbCA9IG51bGw7XG4gICAgICAgICAgICAgICAgaWYodGhpcy5wcm9wcy5uYW1lZENoaWxkcmVuLnJvd0xhYmVscyAmJiB0aGlzLnByb3BzLm5hbWVkQ2hpbGRyZW4ucm93TGFiZWxzLmxlbmd0aCA+IDApe1xuICAgICAgICAgICAgICAgICAgICByb3dMYWJlbEVsID0gaCgndGgnLCB7a2V5OiBgJHt0aGlzLnByb3BzLmlkfS1ncmlkLWNvbC0ke3Jvd0lkeH0tJHtjb2xJZHh9YH0sIFtcbiAgICAgICAgICAgICAgICAgICAgICAgIHRoaXMucHJvcHMubmFtZWRDaGlsZHJlbi5yb3dMYWJlbHNbcm93SWR4XS5yZW5kZXIoKVxuICAgICAgICAgICAgICAgICAgICBdKTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgICAgICAgICAgaCgndHInLCB7a2V5OiBgJHt0aGlzLnByb3BzLmlkfS1ncmlkLXJvdy0ke3Jvd0lkeH1gfSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgcm93TGFiZWxFbCxcbiAgICAgICAgICAgICAgICAgICAgICAgIC4uLmNvbHVtbnNcbiAgICAgICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgICAgICApO1xuICAgICAgICAgICAgfSk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBfbWFrZVJlcGxhY2VtZW50Um93RWxlbWVudHMoKXtcbiAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50c0ZvcignY2hpbGQnKS5tYXAoKHJvdywgcm93SWR4KSA9PiB7XG4gICAgICAgICAgICBsZXQgY29sdW1ucyA9IHJvdy5tYXAoKGNvbHVtbiwgY29sSWR4KSA9PiB7XG4gICAgICAgICAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgICAgICAgICAgaCgndGQnLCB7a2V5OiBgJHt0aGlzLnByb3BzLmlkfS1ncmlkLWNvbC0ke3Jvd0lkeH0tJHtjb2xJZHh9YH0sIFtcbiAgICAgICAgICAgICAgICAgICAgICAgIGNvbHVtblxuICAgICAgICAgICAgICAgICAgICBdKVxuICAgICAgICAgICAgICAgICk7XG4gICAgICAgICAgICB9KTtcbiAgICAgICAgICAgIGxldCByb3dMYWJlbEVsID0gbnVsbDtcbiAgICAgICAgICAgIGlmKHRoaXMucmVwbGFjZW1lbnRzLmhhc1JlcGxhY2VtZW50KCdyb3dsYWJlbCcpKXtcbiAgICAgICAgICAgICAgICByb3dMYWJlbEVsID0gaCgndGgnLCB7a2V5OiBgJHt0aGlzLnByb3BzLmlkfS1ncmlkLXJvd2xibC0ke3Jvd0lkeH1gfSwgW1xuICAgICAgICAgICAgICAgICAgICB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudHNGb3IoJ3Jvd2xhYmVsJylbcm93SWR4XVxuICAgICAgICAgICAgICAgIF0pO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgICAgICBoKCd0cicsIHtrZXk6IGAke3RoaXMucHJvcHMuaWR9LWdyaWQtcm93LSR7cm93SWR4fWB9LCBbXG4gICAgICAgICAgICAgICAgICAgIHJvd0xhYmVsRWwsXG4gICAgICAgICAgICAgICAgICAgIC4uLmNvbHVtbnNcbiAgICAgICAgICAgICAgICBdKVxuICAgICAgICAgICAgKTtcbiAgICAgICAgfSk7XG4gICAgfVxuXG4gICAgX21ha2VSZXBsYWNlbWVudEhlYWRlckVsZW1lbnRzKCl7XG4gICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudHNGb3IoJ2hlYWRlcicpLm1hcCgoaGVhZGVyRWwsIGNvbElkeCkgPT4ge1xuICAgICAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgICAgICBoKCd0aCcsIHtrZXk6IGAke3RoaXMucHJvcHMuaWR9LWdyaWQtdGgtJHtjb2xJZHh9YH0sIFtcbiAgICAgICAgICAgICAgICAgICAgaGVhZGVyRWxcbiAgICAgICAgICAgICAgICBdKVxuICAgICAgICAgICAgKTtcbiAgICAgICAgfSk7XG4gICAgfVxufVxuXG5leHBvcnRcbntHcmlkLCBHcmlkIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBIZWFkZXJCYXIgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyB0aHJlZSBzZXBhcmF0ZVxuICogZW51bWVyYXRlZCByZXBsYWNlbWVudHM6XG4gKiAqIGBsZWZ0YFxuICogKiBgcmlnaHRgXG4gKiAqIGBjZW50ZXJgXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBDaGlsZHJlblxuICogLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIGBsZWZ0SXRlbXNgIChhcnJheSkgLSBUaGUgaXRlbXMgdGhhdCB3aWxsIGJlIG9uIHRoZSBsZWZ0XG4gKiBgY2VudGVySXRlbXNgIChhcnJheSkgLSBUaGUgaXRlbXMgdGhhdCB3aWxsIGJlIGluIHRoZSBjZW50ZXJcbiAqIGByaWdodEl0ZW1zYCAoYXJyYXkpIC0gVGhlIGl0ZW1zIHRoYXQgd2lsbCBiZSBvbiB0aGUgcmlnaHRcbiAqL1xuY2xhc3MgSGVhZGVyQmFyIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICAvLyBCaW5kIGNvbnRleHQgdG8gbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VFbGVtZW50cyA9IHRoaXMubWFrZUVsZW1lbnRzLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMubWFrZVJpZ2h0ID0gdGhpcy5tYWtlUmlnaHQuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5tYWtlTGVmdCA9IHRoaXMubWFrZUxlZnQuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5tYWtlQ2VudGVyID0gdGhpcy5tYWtlQ2VudGVyLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbCBwLTIgYmctbGlnaHQgZmxleC1jb250YWluZXJcIixcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJIZWFkZXJCYXJcIixcbiAgICAgICAgICAgICAgICBzdHlsZTogJ2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpiYXNlbGluZTsnXG4gICAgICAgICAgICB9LCBbXG4gICAgICAgICAgICAgICAgdGhpcy5tYWtlTGVmdCgpLFxuICAgICAgICAgICAgICAgIHRoaXMubWFrZUNlbnRlcigpLFxuICAgICAgICAgICAgICAgIHRoaXMubWFrZVJpZ2h0KClcbiAgICAgICAgICAgIF0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgbWFrZUxlZnQoKXtcbiAgICAgICAgbGV0IGlubmVyRWxlbWVudHMgPSBbXTtcbiAgICAgICAgaWYodGhpcy5yZXBsYWNlbWVudHMuaGFzUmVwbGFjZW1lbnQoJ2xlZnQnKSB8fCB0aGlzLnByb3BzLm5hbWVkQ2hpbGRyZW4ubGVmdEl0ZW1zKXtcbiAgICAgICAgICAgIGlubmVyRWxlbWVudHMgPSB0aGlzLm1ha2VFbGVtZW50cygnbGVmdCcpO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdkaXYnLCB7Y2xhc3M6IFwiZmxleC1pdGVtXCIsIHN0eWxlOiBcImZsZXgtZ3JvdzowO1wifSwgW1xuICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICAgICAgY2xhc3M6IFwiZmxleC1jb250YWluZXJcIixcbiAgICAgICAgICAgICAgICAgICAgc3R5bGU6ICdkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OmNlbnRlcjthbGlnbi1pdGVtczpiYXNlbGluZTsnXG4gICAgICAgICAgICAgICAgfSwgaW5uZXJFbGVtZW50cylcbiAgICAgICAgICAgIF0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgbWFrZUNlbnRlcigpe1xuICAgICAgICBsZXQgaW5uZXJFbGVtZW50cyA9IFtdO1xuICAgICAgICBpZih0aGlzLnJlcGxhY2VtZW50cy5oYXNSZXBsYWNlbWVudCgnY2VudGVyJykgfHwgdGhpcy5wcm9wcy5uYW1lZENoaWxkcmVuLmNlbnRlckl0ZW1zKXtcbiAgICAgICAgICAgIGlubmVyRWxlbWVudHMgPSB0aGlzLm1ha2VFbGVtZW50cygnY2VudGVyJyk7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ2RpdicsIHtjbGFzczogXCJmbGV4LWl0ZW1cIiwgc3R5bGU6IFwiZmxleC1ncm93OjE7XCJ9LCBbXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgICAgICBjbGFzczogXCJmbGV4LWNvbnRhaW5lclwiLFxuICAgICAgICAgICAgICAgICAgICBzdHlsZTogJ2Rpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO2FsaWduLWl0ZW1zOmJhc2VsaW5lOydcbiAgICAgICAgICAgICAgICB9LCBpbm5lckVsZW1lbnRzKVxuICAgICAgICAgICAgXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlUmlnaHQoKXtcbiAgICAgICAgbGV0IGlubmVyRWxlbWVudHMgPSBbXTtcbiAgICAgICAgaWYodGhpcy5yZXBsYWNlbWVudHMuaGFzUmVwbGFjZW1lbnQoJ3JpZ2h0JykgfHwgdGhpcy5wcm9wcy5uYW1lZENoaWxkcmVuLnJpZ2h0SXRlbXMpe1xuICAgICAgICAgICAgaW5uZXJFbGVtZW50cyA9IHRoaXMubWFrZUVsZW1lbnRzKCdyaWdodCcpO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdkaXYnLCB7Y2xhc3M6IFwiZmxleC1pdGVtXCIsIHN0eWxlOiBcImZsZXgtZ3JvdzowO1wifSwgW1xuICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICAgICAgY2xhc3M6IFwiZmxleC1jb250YWluZXJcIixcbiAgICAgICAgICAgICAgICAgICAgc3R5bGU6ICdkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OmNlbnRlcjthbGlnbi1pdGVtczpiYXNlbGluZTsnXG4gICAgICAgICAgICAgICAgfSwgaW5uZXJFbGVtZW50cylcbiAgICAgICAgICAgIF0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgbWFrZUVsZW1lbnRzKHBvc2l0aW9uKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudHNGb3IocG9zaXRpb24pLm1hcChlbGVtZW50ID0+IHtcbiAgICAgICAgICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgICAgICAgICBoKCdzcGFuJywge2NsYXNzOiBcImZsZXgtaXRlbSBweC0zXCJ9LCBbZWxlbWVudF0pXG4gICAgICAgICAgICAgICAgKTtcbiAgICAgICAgICAgIH0pO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGRyZW5OYW1lZChgJHtwb3NpdGlvbn1JdGVtc2ApLm1hcChlbGVtZW50ID0+IHtcbiAgICAgICAgICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgICAgICAgICBoKCdzcGFuJywge2NsYXNzOiBcImZsZXgtaXRlbSBweC0zXCJ9LCBbZWxlbWVudF0pXG4gICAgICAgICAgICAgICAgKTtcbiAgICAgICAgICAgIH0pO1xuICAgICAgICB9XG4gICAgfVxufVxuXG5leHBvcnQge0hlYWRlckJhciwgSGVhZGVyQmFyIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBMYXJnZVBlbmRpbmdEb3dubG9hZERpc3BsYXkgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbmNsYXNzIExhcmdlUGVuZGluZ0Rvd25sb2FkRGlzcGxheSBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgaWQ6ICdvYmplY3RfZGF0YWJhc2VfbGFyZ2VfcGVuZGluZ19kb3dubG9hZF90ZXh0JyxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJMYXJnZVBlbmRpbmdEb3dubG9hZERpc3BsYXlcIixcbiAgICAgICAgICAgICAgICBjbGFzczogXCJjZWxsXCJcbiAgICAgICAgICAgIH0pXG4gICAgICAgICk7XG4gICAgfVxufVxuXG5leHBvcnQge0xhcmdlUGVuZGluZ0Rvd25sb2FkRGlzcGxheSwgTGFyZ2VQZW5kaW5nRG93bmxvYWREaXNwbGF5IGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBMb2FkQ29udGVudHNGcm9tVXJsIENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG5jbGFzcyBMb2FkQ29udGVudHNGcm9tVXJsIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkxvYWRDb250ZW50c0Zyb21VcmxcIixcbiAgICAgICAgICAgIH0sIFtoKCdkaXYnLCB7aWQ6IHRoaXMucHJvcHMuZXh0cmFEYXRhWydsb2FkVGFyZ2V0SWQnXX0sIFtdKV1cbiAgICAgICAgICAgIClcbiAgICAgICAgKTtcbiAgICB9XG5cbn1cblxuZXhwb3J0IHtMb2FkQ29udGVudHNGcm9tVXJsLCBMb2FkQ29udGVudHNGcm9tVXJsIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBNYWluIENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyBhIG9uZVxuICogcmVndWxhci1raW5kIHJlcGxhY2VtZW50OlxuICogKiBgY2hpbGRgXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBDaGlsZHJlblxuICogLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIGBjaGlsZGAgKHNpbmdsZSkgLSBUaGUgY2hpbGQgY2VsbCB0aGF0IGlzIHdyYXBwZWRcbiAqL1xuY2xhc3MgTWFpbiBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb21wb25lbnQgbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VDaGlsZCA9IHRoaXMubWFrZUNoaWxkLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdtYWluJywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIGNsYXNzOiBcInB5LW1kLTJcIixcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJNYWluXCJcbiAgICAgICAgICAgIH0sIFtcbiAgICAgICAgICAgICAgICBoKCdkaXYnLCB7Y2xhc3M6IFwiY29udGFpbmVyLWZsdWlkXCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgIHRoaXMubWFrZUNoaWxkKClcbiAgICAgICAgICAgICAgICBdKVxuICAgICAgICAgICAgXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlQ2hpbGQoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY2hpbGQnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ2NoaWxkJyk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cbmV4cG9ydCB7TWFpbiwgTWFpbiBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogTW9kYWwgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIE1vZGFsIGhhcyB0aGUgZm9sbG93aW5nIHNpbmdsZSByZXBsYWNlbWVudHM6XG4gKiAqYHRpdGxlYFxuICogKmBtZXNzYWdlYFxuICogQW5kIGhhcyB0aGUgZm9sbG93aW5nIGVudW1lcmF0ZWRcbiAqIHJlcGxhY2VtZW50c1xuICogKiBgYnV0dG9uYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgdGl0bGVgIChzaW5nbGUpIC0gQSBDZWxsIGNvbnRhaW5pbmcgdGhlIHRpdGxlXG4gKiBgbWVzc2FnZWAgKHNpbmdsZSkgLSBBIENlbGwgY29udGlhbmluZyB0aGUgYm9keSBvZiB0aGVcbiAqICAgICBtb2RhbCBtZXNzYWdlXG4gKiBgYnV0dG9uc2AgKGFycmF5KSAtIEFuIGFycmF5IG9mIGJ1dHRvbiBjZWxsc1xuICovXG5jbGFzcyBNb2RhbCBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG4gICAgICAgIHRoaXMubWFpblN0eWxlID0gJ2Rpc3BsYXk6YmxvY2s7cGFkZGluZy1yaWdodDoxNXB4Oyc7XG5cbiAgICAgICAgLy8gQmluZCBjb21wb25lbnQgbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VUaXRsZSA9IHRoaXMubWFrZVRpdGxlLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMubWFrZU1lc3NhZ2UgPSB0aGlzLm1ha2VNZXNzYWdlLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMubWFrZUJ1dHRvbnMgPSB0aGlzLm1ha2VCdXR0b25zLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbCBtb2RhbCBmYWRlIHNob3dcIixcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJNb2RhbFwiLFxuICAgICAgICAgICAgICAgIHJvbGU6IFwiZGlhbG9nXCIsXG4gICAgICAgICAgICAgICAgc3R5bGU6IG1haW5TdHlsZVxuICAgICAgICAgICAgfSwgW1xuICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtyb2xlOiBcImRvY3VtZW50XCIsIGNsYXNzOiBcIm1vZGFsLWRpYWxvZ1wifSwgW1xuICAgICAgICAgICAgICAgICAgICBoKCdkaXYnLCB7Y2xhc3M6IFwibW9kYWwtY29udGVudFwifSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgaCgnZGl2Jywge2NsYXNzOiBcIm1vZGFsLWhlYWRlclwifSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGgoJ2g1Jywge2NsYXNzOiBcIm1vZGFsLXRpdGxlXCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRoaXMubWFrZVRpdGxlKClcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBdKVxuICAgICAgICAgICAgICAgICAgICAgICAgXSksXG4gICAgICAgICAgICAgICAgICAgICAgICBoKCdkaXYnLCB7Y2xhc3M6IFwibW9kYWwtYm9keVwifSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRoaXMubWFrZU1lc3NhZ2UoKVxuICAgICAgICAgICAgICAgICAgICAgICAgXSksXG4gICAgICAgICAgICAgICAgICAgICAgICBoKCdkaXYnLCB7Y2xhc3M6IFwibW9kYWwtZm9vdGVyXCJ9LCB0aGlzLm1ha2VCdXR0b25zKCkpXG4gICAgICAgICAgICAgICAgICAgIF0pXG4gICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgIF0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgbWFrZUJ1dHRvbnMoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudHNGb3IoJ2J1dHRvbicpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGRyZW5OYW1lZCgnYnV0dG9ucycpXG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBtYWtlTWVzc2FnZSgpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdtZXNzYWdlJyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCdtZXNzYWdlJyk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBtYWtlVGl0bGUoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcigndGl0bGUnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ3RpdGxlJyk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cbmV4cG9ydCB7TW9kYWwsIE1vZGFsIGFzIGRlZmF1bHR9XG4iLCIvKipcbiAqIE9jdGljb24gQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbmNsYXNzIE9jdGljb24gZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29udGV4dCB0byBtZXRob2RzXG4gICAgICAgIHRoaXMuX2dldEhUTUxDbGFzc2VzID0gdGhpcy5fZ2V0SFRNTENsYXNzZXMuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuKFxuICAgICAgICAgICAgaCgnc3BhbicsIHtcbiAgICAgICAgICAgICAgICBjbGFzczogdGhpcy5fZ2V0SFRNTENsYXNzZXMoKSxcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJPY3RpY29uXCIsXG4gICAgICAgICAgICAgICAgXCJhcmlhLWhpZGRlblwiOiB0cnVlLFxuICAgICAgICAgICAgICAgIHN0eWxlOiB0aGlzLnByb3BzLmV4dHJhRGF0YS5kaXZTdHlsZVxuICAgICAgICAgICAgfSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBfZ2V0SFRNTENsYXNzZXMoKXtcbiAgICAgICAgbGV0IGNsYXNzZXMgPSBbXCJjZWxsXCIsIFwib2N0aWNvblwiXTtcbiAgICAgICAgdGhpcy5wcm9wcy5leHRyYURhdGEub2N0aWNvbkNsYXNzZXMuZm9yRWFjaChuYW1lID0+IHtcbiAgICAgICAgICAgIGNsYXNzZXMucHVzaChuYW1lKTtcbiAgICAgICAgfSk7XG4gICAgICAgIHJldHVybiBjbGFzc2VzLmpvaW4oXCIgXCIpO1xuICAgIH1cbn1cblxuZXhwb3J0IHtPY3RpY29uLCBPY3RpY29uIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBQYWRkaW5nIENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG5jbGFzcyBQYWRkaW5nIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ3NwYW4nLCB7XG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiUGFkZGluZ1wiLFxuICAgICAgICAgICAgICAgIGNsYXNzOiBcInB4LTJcIlxuICAgICAgICAgICAgfSwgW1wiIFwiXSlcbiAgICAgICAgKTtcbiAgICB9XG59XG5cbmV4cG9ydCB7UGFkZGluZywgUGFkZGluZyBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogUGxvdCBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgY29udGFpbnMgdGhlIGZvbGxvd2luZ1xuICogcmVndWxhciByZXBsYWNlbWVudHM6XG4gKiAqIGBjaGFydC11cGRhdGVyYFxuICogKiBgZXJyb3JgXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBDaGlsZHJlblxuICogLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIGBjaGFydFVwZGF0ZXJgIChzaW5nbGUpIC0gVGhlIFVwZGF0ZXIgY2VsbFxuICogYGVycm9yYCAoc2luZ2xlKSAtIEFuIGVycm9yIGNlbGwsIGlmIHByZXNlbnRcbiAqL1xuY2xhc3MgUGxvdCBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb21wb25lbnQgbWV0aG9kc1xuICAgICAgICB0aGlzLnNldHVwUGxvdCA9IHRoaXMuc2V0dXBQbG90LmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMubWFrZUNoYXJ0VXBkYXRlciA9IHRoaXMubWFrZUNoYXJ0VXBkYXRlci5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLm1ha2VFcnJvciA9IHRoaXMubWFrZUVycm9yLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgY29tcG9uZW50RGlkTG9hZCgpIHtcbiAgICAgICAgdGhpcy5zZXR1cFBsb3QoKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJQbG90XCIsXG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbFwiXG4gICAgICAgICAgICB9LCBbXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge2lkOiBgcGxvdCR7dGhpcy5wcm9wcy5pZH1gLCBzdHlsZTogdGhpcy5wcm9wcy5leHRyYURhdGEuZGl2U3R5bGV9KSxcbiAgICAgICAgICAgICAgICB0aGlzLm1ha2VDaGFydFVwZGF0ZXIoKSxcbiAgICAgICAgICAgICAgICB0aGlzLm1ha2VFcnJvcigpXG4gICAgICAgICAgICBdKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIG1ha2VDaGFydFVwZGF0ZXIoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY2hhcnQtdXBkYXRlcicpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnY2hhcnRVcGRhdGVyJyk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBtYWtlRXJyb3IoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignZXJyb3InKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ2Vycm9yJyk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBzZXR1cFBsb3QoKXtcbiAgICAgICAgY29uc29sZS5sb2coXCJTZXR0aW5nIHVwIGEgbmV3IHBsb3RseSBjaGFydC5cIik7XG4gICAgICAgIC8vIFRPRE8gVGhlc2UgYXJlIGdsb2JhbCB2YXIgZGVmaW5lZCBpbiBwYWdlLmh0bWxcbiAgICAgICAgLy8gd2Ugc2hvdWxkIGRvIHNvbWV0aGluZyBhYm91dCB0aGlzLlxuICAgICAgICB2YXIgcGxvdERpdiA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdwbG90JyArIHRoaXMucHJvcHMuaWQpO1xuICAgICAgICBQbG90bHkucGxvdChcbiAgICAgICAgICAgIHBsb3REaXYsXG4gICAgICAgICAgICBbXSxcbiAgICAgICAgICAgIHtcbiAgICAgICAgICAgICAgICBtYXJnaW46IHt0IDogMzAsIGw6IDMwLCByOiAzMCwgYjozMCB9LFxuICAgICAgICAgICAgICAgIHhheGlzOiB7cmFuZ2VzbGlkZXI6IHt2aXNpYmxlOiBmYWxzZX19XG4gICAgICAgICAgICB9LFxuICAgICAgICAgICAgeyBzY3JvbGxab29tOiB0cnVlLCBkcmFnbW9kZTogJ3BhbicsIGRpc3BsYXlsb2dvOiBmYWxzZSwgZGlzcGxheU1vZGVCYXI6ICdob3ZlcicsXG4gICAgICAgICAgICAgICAgbW9kZUJhckJ1dHRvbnM6IFsgWydwYW4yZCddLCBbJ3pvb20yZCddLCBbJ3pvb21JbjJkJ10sIFsnem9vbU91dDJkJ10gXSB9XG4gICAgICAgICk7XG4gICAgICAgIHBsb3REaXYub24oJ3Bsb3RseV9yZWxheW91dCcsXG4gICAgICAgICAgICBmdW5jdGlvbihldmVudGRhdGEpe1xuICAgICAgICAgICAgICAgIGlmIChwbG90RGl2LmlzX3NlcnZlcl9kZWZpbmVkX21vdmUgPT09IHRydWUpIHtcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuXG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIC8vaWYgd2UncmUgc2VuZGluZyBhIHN0cmluZywgdGhlbiBpdHMgYSBkYXRlIG9iamVjdCwgYW5kIHdlIHdhbnQgdG8gc2VuZFxuICAgICAgICAgICAgICAgIC8vIGEgdGltZXN0YW1wXG4gICAgICAgICAgICAgICAgaWYgKHR5cGVvZihldmVudGRhdGFbJ3hheGlzLnJhbmdlWzBdJ10pID09PSAnc3RyaW5nJykge1xuICAgICAgICAgICAgICAgICAgICBldmVudGRhdGEgPSBPYmplY3QuYXNzaWduKHt9LGV2ZW50ZGF0YSk7XG4gICAgICAgICAgICAgICAgICAgIGV2ZW50ZGF0YVtcInhheGlzLnJhbmdlWzBdXCJdID0gRGF0ZS5wYXJzZShldmVudGRhdGFbXCJ4YXhpcy5yYW5nZVswXVwiXSkgLyAxMDAwLjA7XG4gICAgICAgICAgICAgICAgICAgIGV2ZW50ZGF0YVtcInhheGlzLnJhbmdlWzFdXCJdID0gRGF0ZS5wYXJzZShldmVudGRhdGFbXCJ4YXhpcy5yYW5nZVsxXVwiXSkgLyAxMDAwLjA7XG4gICAgICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICAgICAgbGV0IHJlc3BvbnNlRGF0YSA9IHtcbiAgICAgICAgICAgICAgICAgICAgJ2V2ZW50JzoncGxvdF9sYXlvdXQnLFxuICAgICAgICAgICAgICAgICAgICAndGFyZ2V0X2NlbGwnOiAnX19pZGVudGl0eV9fJyxcbiAgICAgICAgICAgICAgICAgICAgJ2RhdGEnOiBldmVudGRhdGFcbiAgICAgICAgICAgICAgICB9O1xuICAgICAgICAgICAgICAgIGNlbGxTb2NrZXQuc2VuZFN0cmluZyhKU09OLnN0cmluZ2lmeShyZXNwb25zZURhdGEpKTtcbiAgICAgICAgICAgIH0pO1xuICAgIH1cbn1cblxuZXhwb3J0IHtQbG90LCBQbG90IGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBQb3BvdmVyIENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogVGhpcyBjb21wb25lbnQgY29udGFpbnMgdGhlIGZvbGxvd2luZ1xuICogcmVndWxhciByZXBsYWNlbWVudHM6XG4gKiAqIGB0aXRsZWBcbiAqICogYGRldGFpbGBcbiAqICogYGNvbnRlbnRzYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgY29udGVudGAgKHNpbmdsZSkgLSBUaGUgY29udGVudCBvZiB0aGUgcG9wb3ZlclxuICogYGRldGFpbGAgKHNpbmdsZSkgLSBEZXRhaWwgb2YgdGhlIHBvcG92ZXJcbiAqIGB0aXRsZWAgKHNpbmdsZSkgLSBUaGUgdGl0bGUgZm9yIHRoZSBwb3BvdmVyXG4gKi9cbmNsYXNzIFBvcG92ZXIgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29tcG9uZW50IG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlVGl0bGUgPSB0aGlzLm1ha2VUaXRsZS5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLm1ha2VDb250ZW50ID0gdGhpcy5tYWtlQ29udGVudC5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLm1ha2VEZXRhaWwgPSB0aGlzLm1ha2VEZXRhaWwuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIGgoJ2RpdicsXG4gICAgICAgICAgICB7XG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbCBwb3BvdmVyLWNlbGxcIixcbiAgICAgICAgICAgICAgICBzdHlsZTogdGhpcy5wcm9wcy5leHRyYURhdGEuZGl2U3R5bGUsXG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiUG9wb3ZlclwiXG4gICAgICAgICAgICB9LCBbXG4gICAgICAgICAgICAgICAgaCgnYScsXG4gICAgICAgICAgICAgICAgICAgIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIGhyZWY6IFwiI3BvcG1haW5fXCIgKyB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgICAgICAgICAgXCJkYXRhLXRvZ2dsZVwiOiBcInBvcG92ZXJcIixcbiAgICAgICAgICAgICAgICAgICAgICAgIFwiZGF0YS10cmlnZ2VyXCI6IFwiZm9jdXNcIixcbiAgICAgICAgICAgICAgICAgICAgICAgIFwiZGF0YS1iaW5kXCI6IFwiI3BvcF9cIiArIHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgICAgICAgICBcImRhdGEtcGxhY2VtZW50XCI6IFwiYm90dG9tXCIsXG4gICAgICAgICAgICAgICAgICAgICAgICByb2xlOiBcImJ1dHRvblwiLFxuICAgICAgICAgICAgICAgICAgICAgICAgY2xhc3M6IFwiYnRuIGJ0bi14c1wiXG4gICAgICAgICAgICAgICAgICAgIH0sXG4gICAgICAgICAgICAgICAgICBbdGhpcy5tYWtlQ29udGVudCgpXVxuICAgICAgICAgICAgICAgICksXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge3N0eWxlOiBcImRpc3BsYXk6bm9uZVwifSwgW1xuICAgICAgICAgICAgICAgICAgICBoKFwiZGl2XCIsIHtpZDogXCJwb3BfXCIgKyB0aGlzLnByb3BzLmlkfSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgaChcImRpdlwiLCB7Y2xhc3M6IFwiZGF0YS10aXRsZVwifSwgW3RoaXMubWFrZVRpdGxlKCldKSxcbiAgICAgICAgICAgICAgICAgICAgICAgIGgoXCJkaXZcIiwge2NsYXNzOiBcImRhdGEtY29udGVudFwifSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGgoXCJkaXZcIiwge3N0eWxlOiBcIndpZHRoOiBcIiArIHRoaXMucHJvcHMud2lkdGggKyBcInB4XCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRoaXMubWFrZURldGFpbCgpXVxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIClcbiAgICAgICAgICAgICAgICAgICAgICAgIF0pXG4gICAgICAgICAgICAgICAgICAgIF0pXG4gICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgIF1cbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlQ29udGVudCgpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdjb250ZW50cycpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnY29udGVudCcpO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgbWFrZURldGFpbCgpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdkZXRhaWwnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ2RldGFpbCcpO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgbWFrZVRpdGxlKCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ3RpdGxlJyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCd0aXRsZScpO1xuICAgICAgICB9XG4gICAgfVxufVxuXG5leHBvcnQge1BvcG92ZXIsIFBvcG92ZXIgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIFJvb3RDZWxsIENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyBhIG9uZVxuICogcmVndWxhci1raW5kIHJlcGxhY2VtZW50OlxuICogKiBgY2BcbiAqL1xuXG4vKipcbiAqIEFib3V0IE5hbWVkIENoaWxkcmVuXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogYGNoaWxkYCAoc2luZ2xlKSAtIFRoZSBjaGlsZCBjZWxsIHRoaXMgY29udGFpbmVyIGNvbnRhaW5zXG4gKi9cbmNsYXNzIFJvb3RDZWxsIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICAvLyBCaW5kIGNvbXBvbmVudCBtZXRob2RzXG4gICAgICAgIHRoaXMubWFrZUNoaWxkID0gdGhpcy5tYWtlQ2hpbGQuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJSb290Q2VsbFwiXG4gICAgICAgICAgICB9LCBbdGhpcy5tYWtlQ2hpbGQoKV0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgbWFrZUNoaWxkKCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ2MnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ2NoaWxkJyk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cbmV4cG9ydCB7Um9vdENlbGwsIFJvb3RDZWxsIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBTY3JvbGxhYmxlICBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIGEgb25lXG4gKiByZWd1bGFyLWtpbmQgcmVwbGFjZW1lbnQ6XG4gKiAqIGBjaGlsZGBcbiAqL1xuXG4vKipcbiAqIEFib3V0IE5hbWVkIENoaWxkcmVuXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogYGNoaWxkYCAoc2luZ2xlKSAtIFRoZSBjZWxsL2NvbXBvbmVudCB0aGlzIGluc3RhbmNlIGNvbnRhaW5zXG4gKi9cbmNsYXNzIFNjcm9sbGFibGUgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29tcG9uZW50IG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlQ2hpbGQgPSB0aGlzLm1ha2VDaGlsZC5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIlNjcm9sbGFibGVcIlxuICAgICAgICAgICAgfSwgW3RoaXMubWFrZUNoaWxkKCldKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIG1ha2VDaGlsZCgpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdjaGlsZCcpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnY2hpbGQnKTtcbiAgICAgICAgfVxuICAgIH1cbn1cblxuZXhwb3J0IHtTY3JvbGxhYmxlLCBTY3JvbGxhYmxlIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBTZXF1ZW5jZSBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogU2VxdWVuY2UgaGFzIHRoZSBmb2xsb3dpbmcgZW51bWVyYXRlZFxuICogcmVwbGFjZW1lbnQ6XG4gKiAqIGBjYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgZWxlbWVudHNgIChhcnJheSkgLSBBIGxpc3Qgb2YgQ2VsbHMgdGhhdCBhcmUgaW4gdGhlXG4gKiAgICBzZXF1ZW5jZS5cbiAqL1xuY2xhc3MgU2VxdWVuY2UgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29tcG9uZW50IG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlRWxlbWVudHMgPSB0aGlzLm1ha2VFbGVtZW50cy5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIGNsYXNzOiBcImNlbGxcIixcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJTZXF1ZW5jZVwiLFxuICAgICAgICAgICAgICAgIHN0eWxlOiB0aGlzLnByb3BzLmV4dHJhRGF0YS5kaXZTdHlsZVxuICAgICAgICAgICAgfSwgdGhpcy5tYWtlRWxlbWVudHMoKSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlRWxlbWVudHMoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudHNGb3IoJ2MnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkcmVuTmFtZWQoJ2VsZW1lbnRzJyk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cbmV4cG9ydCB7U2VxdWVuY2UsIFNlcXVlbmNlIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBTaGVldCBDZWxsIENvbXBvbmVudFxuICogTk9URTogVGhpcyBpcyBpbiBwYXJ0IGEgd3JhcHBlclxuICogZm9yIGhhbmRzb250YWJsZXMuXG4gKi9cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogVGhpcyBjb21wb25lbnQgaGFzIG9uZSByZWd1bGFyXG4gKiByZXBsYWNlbWVudDpcbiAqICogYGVycm9yYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgZXJyb3JgIChzaW5nbGUpIC0gQW4gZXJyb3IgY2VsbCBpZiBwcmVzZW50XG4gKi9cbmNsYXNzIFNoZWV0IGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICB0aGlzLmN1cnJlbnRUYWJsZSA9IG51bGw7XG5cbiAgICAgICAgLy8gQmluZCBjb250ZXh0IHRvIG1ldGhvZHNcbiAgICAgICAgdGhpcy5pbml0aWFsaXplVGFibGUgPSB0aGlzLmluaXRpYWxpemVUYWJsZS5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLmluaXRpYWxpemVIb29rcyA9IHRoaXMuaW5pdGlhbGl6ZUhvb2tzLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMubWFrZUVycm9yID0gdGhpcy5tYWtlRXJyb3IuYmluZCh0aGlzKTtcblxuICAgICAgICAvKipcbiAgICAgICAgICogV0FSTklORzogVGhlIENlbGwgdmVyc2lvbiBvZiBTaGVldCBpcyBzdGlsbCB1c2luZyBjZXJ0YWluXG4gICAgICAgICAqIHBvc3RzY3JpcHRzIGJlY2F1c2Ugd2UgaGF2ZSBub3QgeWV0IHJlZmFjdG9yZWQgdGhlIHNvY2tldFxuICAgICAgICAgKiBwcm90b2NvbC5cbiAgICAgICAgICogUmVtb3ZlIHRoaXMgd2FybmluZyBhYm91dCBpdCBvbmNlIHRoYXQgaGFwcGVucyFcbiAgICAgICAgICovXG4gICAgICAgIGNvbnNvbGUud2FybihgW1RPRE9dIFNoZWV0IHN0aWxsIHVzZXMgY2VydGFpbiBwb3N0c2NlcmlwdHMgaW4gaXRzIGludGVyYWN0aW9uLiBTZWUgY29tcG9uZW50IGNvbnN0cnVjdG9yIGNvbW1lbnQgZm9yIG1vcmUgaW5mb3JtYXRpb25gKTtcbiAgICB9XG5cbiAgICBjb21wb25lbnREaWRMb2FkKCl7XG4gICAgICAgIGNvbnNvbGUubG9nKGAjY29tcG9uZW50RGlkTG9hZCBjYWxsZWQgZm9yIFNoZWV0ICR7dGhpcy5wcm9wcy5pZH1gKTtcbiAgICAgICAgY29uc29sZS5sb2coYFRoaXMgc2hlZXQgaGFzIHRoZSBmb2xsb3dpbmcgcmVwbGFjZW1lbnRzOmAsIHRoaXMucmVwbGFjZW1lbnRzKTtcbiAgICAgICAgdGhpcy5pbml0aWFsaXplVGFibGUoKTtcbiAgICAgICAgaWYodGhpcy5wcm9wcy5leHRyYURhdGFbJ2hhbmRsZXNEb3VibGVDbGljayddKXtcbiAgICAgICAgICAgIHRoaXMuaW5pdGlhbGl6ZUhvb2tzKCk7XG4gICAgICAgIH1cbiAgICAgICAgLy8gUmVxdWVzdCBpbml0aWFsIGRhdGE/XG4gICAgICAgIGNlbGxTb2NrZXQuc2VuZFN0cmluZyhKU09OLnN0cmluZ2lmeSh7XG4gICAgICAgICAgICBldmVudDogXCJzaGVldF9uZWVkc19kYXRhXCIsXG4gICAgICAgICAgICB0YXJnZXRfY2VsbDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgIGRhdGE6IDBcbiAgICAgICAgfSkpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICBjb25zb2xlLmxvZyhgUmVuZGVyaW5nIHNoZWV0ICR7dGhpcy5wcm9wcy5pZH1gKTtcbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJTaGVldFwiLFxuICAgICAgICAgICAgICAgIGNsYXNzOiBcImNlbGxcIlxuICAgICAgICAgICAgfSwgW1xuICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICAgICAgaWQ6IGBzaGVldCR7dGhpcy5wcm9wcy5pZH1gLFxuICAgICAgICAgICAgICAgICAgICBzdHlsZTogdGhpcy5wcm9wcy5leHRyYURhdGEuZGl2U3R5bGUsXG4gICAgICAgICAgICAgICAgICAgIGNsYXNzOiBcImhhbmRzb250YWJsZVwiXG4gICAgICAgICAgICAgICAgfSwgW3RoaXMubWFrZUVycm9yKCldKVxuICAgICAgICAgICAgXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBpbml0aWFsaXplVGFibGUoKXtcbiAgICAgICAgY29uc29sZS5sb2coYCNpbml0aWFsaXplVGFibGUgY2FsbGVkIGZvciBTaGVldCAke3RoaXMucHJvcHMuaWR9YCk7XG4gICAgICAgIGxldCBnZXRQcm9wZXJ0eSA9IGZ1bmN0aW9uKGluZGV4KXtcbiAgICAgICAgICAgIHJldHVybiBmdW5jdGlvbihyb3cpe1xuICAgICAgICAgICAgICAgIHJldHVybiByb3dbaW5kZXhdO1xuICAgICAgICAgICAgfTtcbiAgICAgICAgfTtcbiAgICAgICAgbGV0IGVtcHR5Um93ID0gW107XG4gICAgICAgIGxldCBkYXRhTmVlZGVkQ2FsbGJhY2sgPSBmdW5jdGlvbihldmVudE9iamVjdCl7XG4gICAgICAgICAgICBldmVudE9iamVjdC50YXJnZXRfY2VsbCA9IHRoaXMucHJvcHMuaWQ7XG4gICAgICAgICAgICBjZWxsU29ja2V0LnNlbmRTdHJpbmcoSlNPTi5zdHJpbmdpZnkoZXZlbnRPYmplY3QpKTtcbiAgICAgICAgfS5iaW5kKHRoaXMpO1xuICAgICAgICBsZXQgZGF0YSA9IG5ldyBTeW50aGV0aWNJbnRlZ2VyQXJyYXkodGhpcy5wcm9wcy5leHRyYURhdGEucm93Q291bnQsIGVtcHR5Um93LCBkYXRhTmVlZGVkQ2FsbGJhY2spO1xuICAgICAgICBsZXQgY29udGFpbmVyID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoYHNoZWV0JHt0aGlzLnByb3BzLmlkfWApO1xuICAgICAgICBsZXQgY29sdW1uTmFtZXMgPSB0aGlzLnByb3BzLmV4dHJhRGF0YS5jb2x1bW5OYW1lcztcbiAgICAgICAgbGV0IGNvbHVtbnMgPSBjb2x1bW5OYW1lcy5tYXAoKG5hbWUsIGlkeCkgPT4ge1xuICAgICAgICAgICAgZW1wdHlSb3cucHVzaChcIlwiKTtcbiAgICAgICAgICAgIHJldHVybiB7ZGF0YTogZ2V0UHJvcGVydHkoaWR4KX07XG4gICAgICAgIH0pO1xuXG4gICAgICAgIHRoaXMuY3VycmVudFRhYmxlID0gbmV3IEhhbmRzb250YWJsZShjb250YWluZXIsIHtcbiAgICAgICAgICAgIGRhdGEsXG4gICAgICAgICAgICBkYXRhU2NoZW1hOiBmdW5jdGlvbihvcHRzKXtyZXR1cm4ge307fSxcbiAgICAgICAgICAgIGNvbEhlYWRlcnM6IGNvbHVtbk5hbWVzLFxuICAgICAgICAgICAgY29sdW1ucyxcbiAgICAgICAgICAgIHJvd0hlYWRlcnM6dHJ1ZSxcbiAgICAgICAgICAgIHJvd0hlYWRlcldpZHRoOiAxMDAsXG4gICAgICAgICAgICB2aWV3cG9ydFJvd1JlbmRlcmluZ09mZnNldDogMTAwLFxuICAgICAgICAgICAgYXV0b0NvbHVtblNpemU6IGZhbHNlLFxuICAgICAgICAgICAgYXV0b1Jvd0hlaWdodDogZmFsc2UsXG4gICAgICAgICAgICBtYW51YWxDb2x1bW5SZXNpemU6IHRydWUsXG4gICAgICAgICAgICBjb2xXaWR0aHM6IHRoaXMucHJvcHMuZXh0cmFEYXRhLmNvbHVtbldpZHRoLFxuICAgICAgICAgICAgcm93SGVpZ2h0czogMjMsXG4gICAgICAgICAgICByZWFkT25seTogdHJ1ZSxcbiAgICAgICAgICAgIE1hbnVhbFJvd01vdmU6IGZhbHNlXG4gICAgICAgIH0pO1xuICAgICAgICBoYW5kc09uVGFibGVzW3RoaXMucHJvcHMuaWRdID0ge1xuICAgICAgICAgICAgdGFibGU6IHRoaXMuY3VycmVudFRhYmxlLFxuICAgICAgICAgICAgbGFzdENlbGxDbGlja2VkOiB7cm93OiAtMTAwLCBjb2w6IC0xMDB9LFxuICAgICAgICAgICAgZGJsQ2xpY2tlZDogdHJ1ZVxuICAgICAgICB9O1xuICAgIH1cblxuICAgIGluaXRpYWxpemVIb29rcygpe1xuICAgICAgICBIYW5kc29udGFibGUuaG9va3MuYWRkKFwiYmVmb3JlT25DZWxsTW91c2VEb3duXCIsIChldmVudCwgZGF0YSkgPT4ge1xuICAgICAgICAgICAgbGV0IGhhbmRzT25PYmogPSBoYW5kc09uVGFibGVzW3RoaXMucHJvcHMuaWRdO1xuICAgICAgICAgICAgbGV0IGxhc3RSb3cgPSBoYW5kc09uT2JqLmxhc3RDZWxsQ2xpY2tlZC5yb3c7XG4gICAgICAgICAgICBsZXQgbGFzdENvbCA9IGhhbmRzT25PYmoubGFzdENlbGxDbGlja2VkLmNvbDtcblxuICAgICAgICAgICAgaWYoKGxhc3RSb3cgPT0gZGF0YS5yb3cpICYmIChsYXN0Q29sID0gZGF0YS5jb2wpKXtcbiAgICAgICAgICAgICAgICBoYW5kc09uT2JqLmRibENsaWNrZWQgPSB0cnVlO1xuICAgICAgICAgICAgICAgIHNldFRpbWVvdXQoKCkgPT4ge1xuICAgICAgICAgICAgICAgICAgICBpZihoYW5kc09uT2JqLmRibENsaWNrZWQpe1xuICAgICAgICAgICAgICAgICAgICAgICAgY2VsbFNvY2tldC5zZW5kU3RyaW5nKEpTT04uc3RyaW5naWZ5KHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBldmVudDogJ29uQ2VsbERibENsaWNrJyxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB0YXJnZXRfY2VsbDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICByb3c6IGRhdGEucm93LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNvbDogZGF0YS5jb2xcbiAgICAgICAgICAgICAgICAgICAgICAgIH0pKTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICBoYW5kc09uT2JqLmxhc3RDZWxsQ2xpY2tlZCA9IHtyb3c6IC0xMDAsIGNvbDogLTEwMH07XG4gICAgICAgICAgICAgICAgICAgIGhhbmRzT25PYmouZGJsQ2xpY2tlZCA9IGZhbHNlO1xuICAgICAgICAgICAgICAgIH0sIDIwMCk7XG4gICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgIGhhbmRzT25PYmoubGFzdENlbGxDbGlja2VkID0ge3JvdzogZGF0YS5yb3csIGNvbDogZGF0YS5jb2x9O1xuICAgICAgICAgICAgICAgIHNldFRpbWVvdXQoKCkgPT4ge1xuICAgICAgICAgICAgICAgICAgICBoYW5kc09uT2JqLmxhc3RDZWxsQ2xpY2tlZCA9IHtyb3c6IC0xMDAsIGNvbDogLTEwMH07XG4gICAgICAgICAgICAgICAgICAgIGhhbmRzT25PYmouZGJsQ2xpY2tlZCA9IGZhbHNlO1xuICAgICAgICAgICAgICAgIH0sIDYwMCk7XG4gICAgICAgICAgICB9XG4gICAgICAgIH0sIHRoaXMuY3VycmVudFRhYmxlKTtcblxuICAgICAgICBIYW5kc29udGFibGUuaG9va3MuYWRkKFwiYmVmb3JlT25DZWxsQ29udGV4dE1lbnVcIiwgKGV2ZW50LCBkYXRhKSA9PiB7XG4gICAgICAgICAgICBsZXQgaGFuZHNPbk9iaiA9IGhhbmRzT25UYWJsZXNbdGhpcy5wcm9wcy5pZF07XG4gICAgICAgICAgICBoYW5kc09uT2JqLmRibENsaWNrZWQgPSBmYWxzZTtcbiAgICAgICAgICAgIGhhbmRzT25PYmoubGFzdENlbGxDbGlja2VkID0ge3JvdzogLTEwMCwgY29sOiAtMTAwfTtcbiAgICAgICAgfSwgdGhpcy5jdXJyZW50VGFibGUpO1xuXG4gICAgICAgIEhhbmRzb250YWJsZS5ob29rcy5hZGQoXCJiZWZvcmVDb250ZXh0TWVudVNob3dcIiwgKGV2ZW50LCBkYXRhKSA9PiB7XG4gICAgICAgICAgICBsZXQgaGFuZHNPbk9iaiA9IGhhbmRzT25UYWJsZXNbdGhpcy5wcm9wcy5pZF07XG4gICAgICAgICAgICBoYW5kc09uT2JqLmRibENsaWNrZWQgPSBmYWxzZTtcbiAgICAgICAgICAgIGhhbmRzT25PYmoubGFzdENlbGxDbGlja2VkID0ge3JvdzogLTEwMCwgY29sOiAtMTAwfTtcbiAgICAgICAgfSwgdGhpcy5jdXJyZW50VGFibGUpO1xuICAgIH1cblxuICAgIG1ha2VFcnJvcigpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdlcnJvcicpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnZXJyb3InKTtcbiAgICAgICAgfVxuICAgIH1cbn1cblxuLyoqIENvcGllZCBvdmVyIGZyb20gQ2VsbHMgaW1wbGVtZW50YXRpb24gKiovXG5jb25zdCBTeW50aGV0aWNJbnRlZ2VyQXJyYXkgPSBmdW5jdGlvbihzaXplLCBlbXB0eVJvdyA9IFtdLCBjYWxsYmFjayl7XG4gICAgdGhpcy5sZW5ndGggPSBzaXplO1xuICAgIHRoaXMuY2FjaGUgPSB7fTtcbiAgICB0aGlzLnB1c2ggPSBmdW5jdGlvbigpe307XG4gICAgdGhpcy5zcGxpY2UgPSBmdW5jdGlvbigpe307XG5cbiAgICB0aGlzLnNsaWNlID0gZnVuY3Rpb24obG93LCBoaWdoKXtcbiAgICAgICAgaWYoaGlnaCA9PT0gdW5kZWZpbmVkKXtcbiAgICAgICAgICAgIGhpZ2ggPSB0aGlzLmxlbmd0aDtcbiAgICAgICAgfVxuXG4gICAgICAgIGxldCByZXMgPSBBcnJheShoaWdoIC0gbG93KTtcbiAgICAgICAgbGV0IGluaXRMb3cgPSBsb3c7XG4gICAgICAgIHdoaWxlKGxvdyA8IGhpZ2gpe1xuICAgICAgICAgICAgbGV0IG91dCA9IHRoaXMuY2FjaGVbbG93XTtcbiAgICAgICAgICAgIGlmKG91dCA9PT0gdW5kZWZpbmVkKXtcbiAgICAgICAgICAgICAgICBpZihjYWxsYmFjayl7XG4gICAgICAgICAgICAgICAgICAgIGNhbGxiYWNrKHtcbiAgICAgICAgICAgICAgICAgICAgICAgIGV2ZW50OiAnc2hlZXRfbmVlZHNfZGF0YScsXG4gICAgICAgICAgICAgICAgICAgICAgICBkYXRhOiBsb3dcbiAgICAgICAgICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIG91dCA9IGVtcHR5Um93O1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgcmVzW2xvdyAtIGluaXRMb3ddID0gb3V0O1xuICAgICAgICAgICAgbG93ICs9IDE7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIHJlcztcbiAgICB9O1xufTtcblxuZXhwb3J0IHtTaGVldCwgU2hlZXQgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIFNpbmdsZUxpbmVUZXh0Qm94IENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG5jbGFzcyBTaW5nbGVMaW5lVGV4dEJveCBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb250ZXh0IHRvIG1ldGhvZHNcbiAgICAgICAgdGhpcy5jaGFuZ2VIYW5kbGVyID0gdGhpcy5jaGFuZ2VIYW5kbGVyLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIGxldCBhdHRycyA9XG4gICAgICAgICAgICB7XG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbFwiLFxuICAgICAgICAgICAgICAgIGlkOiBcInRleHRfXCIgKyB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIHR5cGU6IFwidGV4dFwiLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIlNpbmdsZUxpbmVUZXh0Qm94XCIsXG4gICAgICAgICAgICAgICAgb25jaGFuZ2U6IChldmVudCkgPT4ge3RoaXMuY2hhbmdlSGFuZGxlcihldmVudC50YXJnZXQudmFsdWUpO31cbiAgICAgICAgICAgIH07XG4gICAgICAgIGlmICh0aGlzLnByb3BzLmV4dHJhRGF0YS5pbnB1dFZhbHVlICE9PSB1bmRlZmluZWQpIHtcbiAgICAgICAgICAgIGF0dHJzLnBhdHRlcm4gPSB0aGlzLnByb3BzLmV4dHJhRGF0YS5pbnB1dFZhbHVlO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiBoKCdpbnB1dCcsIGF0dHJzLCBbXSk7XG4gICAgfVxuXG4gICAgY2hhbmdlSGFuZGxlcih2YWwpIHtcbiAgICAgICAgY2VsbFNvY2tldC5zZW5kU3RyaW5nKFxuICAgICAgICAgICAgSlNPTi5zdHJpbmdpZnkoXG4gICAgICAgICAgICAgICAge1xuICAgICAgICAgICAgICAgICAgICBcImV2ZW50XCI6IFwiY2xpY2tcIixcbiAgICAgICAgICAgICAgICAgICAgXCJ0YXJnZXRfY2VsbFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgICAgICBcInRleHRcIjogdmFsXG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgKVxuICAgICAgICApO1xuICAgIH1cbn1cblxuZXhwb3J0IHtTaW5nbGVMaW5lVGV4dEJveCwgU2luZ2xlTGluZVRleHRCb3ggYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIFNwYW4gQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cblxuY2xhc3MgU3BhbiBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdzcGFuJywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIlNwYW5cIixcbiAgICAgICAgICAgICAgICBjbGFzczogXCJjZWxsXCJcbiAgICAgICAgICAgIH0sIFt0aGlzLnByb3BzLmV4dHJhRGF0YS50ZXh0XSlcbiAgICAgICAgKTtcbiAgICB9XG59XG5cbmV4cG9ydCB7U3BhbiwgU3BhbiBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogU3Vic2NyaWJlZCBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIGEgc2luZ2xlXG4gKiByZWd1bGFyIHJlcGxhY2VtZW50OlxuICogKiBgY29udGVudHNgXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBDaGlsZHJlblxuICogLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIGBjb250ZW50YCAoc2luZ2xlKSAtIFRoZSB1bmRlcmx5aW5nIENlbGwgdGhhdCBpcyBzdWJzY3JpYmVkXG4gKi9cbmNsYXNzIFN1YnNjcmliZWQgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29tcG9uZW50IG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlQ29udGVudCA9IHRoaXMubWFrZUNvbnRlbnQuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIGgoJ2RpdicsXG4gICAgICAgICAgICB7XG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbCBzdWJzY3JpYmVkXCIsXG4gICAgICAgICAgICAgICAgc3R5bGU6IHRoaXMucHJvcHMuZXh0cmFEYXRhLmRpdlN0eWxlLFxuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIlN1YnNjcmliZWRcIlxuICAgICAgICAgICAgfSwgW3RoaXMubWFrZUNvbnRlbnQoKV1cbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlQ29udGVudCgpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdjb250ZW50cycpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnY29udGVudCcpO1xuICAgICAgICB9XG4gICAgfVxufVxuXG5leHBvcnQge1N1YnNjcmliZWQsIFN1YnNjcmliZWQgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIFN1YnNjcmliZWRTZXF1ZW5jZSBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIGEgc2luZ2xlXG4gKiBlbnVtZXJhdGVkIHJlcGxhY2VtZW50OlxuICogKiBgY2hpbGRgXG4gKi9cblxuLyoqXG4gKiBBYm91dCBOYW1lZCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogYGNoaWxkcmVuYCAoYXJyYXkpIC0gQW4gYXJyYXkgb2YgQ2VsbHMgdGhhdCBhcmUgc3Vic2NyaWJlZFxuICovXG5jbGFzcyBTdWJzY3JpYmVkU2VxdWVuY2UgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuICAgICAgICAvL3RoaXMuYWRkUmVwbGFjZW1lbnQoJ2NvbnRlbnRzJywgJ19fX19fY29udGVudHNfXycpO1xuICAgICAgICAvL1xuICAgICAgICAvLyBCaW5kIGNvbnRleHQgdG8gbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VDbGFzcyA9IHRoaXMubWFrZUNsYXNzLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMubWFrZUNoaWxkcmVuID0gdGhpcy5tYWtlQ2hpbGRyZW4uYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5fbWFrZVJlcGxhY2VtZW50Q2hpbGRyZW4gPSB0aGlzLl9tYWtlUmVwbGFjZW1lbnRDaGlsZHJlbi5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gaCgnZGl2JyxcbiAgICAgICAgICAgIHtcbiAgICAgICAgICAgICAgICBjbGFzczogdGhpcy5tYWtlQ2xhc3MoKSxcbiAgICAgICAgICAgICAgICBzdHlsZTogdGhpcy5wcm9wcy5leHRyYURhdGEuZGl2U3R5bGUsXG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiU3Vic2NyaWJlZFNlcXVlbmNlXCJcbiAgICAgICAgICAgIH0sIFt0aGlzLm1ha2VDaGlsZHJlbigpXVxuICAgICAgICApO1xuICAgIH1cblxuICAgIG1ha2VDaGlsZHJlbigpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuX21ha2VSZXBsYWNlbWVudENoaWxkcmVuKCk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICBpZih0aGlzLnByb3BzLmV4dHJhRGF0YS5hc0NvbHVtbnMpe1xuICAgICAgICAgICAgICAgIGxldCBmb3JtYXR0ZWRDaGlsZHJlbiA9IHRoaXMucmVuZGVyQ2hpbGRyZW5OYW1lZCgnY2hpbGRyZW4nKS5tYXAoY2hpbGRFbCA9PiB7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybihcbiAgICAgICAgICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtjbGFzczogXCJjb2wtc21cIiwga2V5OiBjaGlsZEVsZW1lbnQuaWR9LCBbXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgaCgnc3BhbicsIHt9LCBbY2hpbGRFbF0pXG4gICAgICAgICAgICAgICAgICAgICAgICBdKVxuICAgICAgICAgICAgICAgICAgICApO1xuICAgICAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtjbGFzczogXCJyb3cgZmxleC1ub3dyYXBcIiwga2V5OiBgJHt0aGlzLnByb3BzLmlkfS1zcGluZS13cmFwcGVyYH0sIGZvcm1hdHRlZENoaWxkcmVuKVxuICAgICAgICAgICAgICAgICk7XG4gICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtrZXk6IGAke3RoaXMucHJvcHMuaWR9LXNwaW5lLXdyYXBwZXJgfSwgdGhpcy5yZW5kZXJDaGlsZHJlbk5hbWVkKCdjaGlsZHJlbicpKVxuICAgICAgICAgICAgICAgICk7XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBtYWtlQ2xhc3MoKSB7XG4gICAgICAgIGlmICh0aGlzLnByb3BzLmV4dHJhRGF0YS5hc0NvbHVtbnMpIHtcbiAgICAgICAgICAgIHJldHVybiBcImNlbGwgc3Vic2NyaWJlZFNlcXVlbmNlIGNvbnRhaW5lci1mbHVpZFwiO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiBcImNlbGwgc3Vic2NyaWJlZFNlcXVlbmNlXCI7XG4gICAgfVxuXG4gICAgX21ha2VSZXBsYWNlbWVudENoaWxkcmVuKCl7XG4gICAgICAgIGlmKHRoaXMucHJvcHMuZXh0cmFEYXRhLmFzQ29sdW1ucyl7XG4gICAgICAgICAgICBsZXQgZm9ybWF0dGVkQ2hpbGRyZW4gPSB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudHNGb3IoJ2NoaWxkJykubWFwKGNoaWxkRWxlbWVudCA9PiB7XG4gICAgICAgICAgICAgICAgcmV0dXJuKFxuICAgICAgICAgICAgICAgICAgICBoKCdkaXYnLCB7Y2xhc3M6IFwiY29sLXNtXCIsIGtleTogY2hpbGRFbGVtZW50LmlkfSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgaCgnc3BhbicsIHt9LCBbY2hpbGRFbGVtZW50XSlcbiAgICAgICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgICAgICApO1xuICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtjbGFzczogXCJyb3cgZmxleC1ub3dyYXBcIiwga2V5OiBgJHt0aGlzLnByb3BzLmlkfS1zcGluZS13cmFwcGVyYH0sIGZvcm1hdHRlZENoaWxkcmVuKVxuICAgICAgICAgICAgKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge2tleTogYCR7dGhpcy5wcm9wcy5pZH0tc3BpbmUtd3JhcHBlcmB9LCB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudHNGb3IoJ2NoaWxkJykpXG4gICAgICAgICAgICApO1xuICAgICAgICB9XG4gICAgfVxufVxuXG5leHBvcnQge1N1YnNjcmliZWRTZXF1ZW5jZSwgU3Vic2NyaWJlZFNlcXVlbmNlIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBUYWJsZSBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBUaGlzIGNvbXBvbmVudCBoYXMgMyByZWd1bGFyXG4gKiByZXBsYWNlbWVudHM6XG4gKiAqIGBwYWdlYFxuICogKiBgbGVmdGBcbiAqICogYHJpZ2h0YFxuICogVGhpcyBjb21wb25lbnQgaGFzIDIgZW51bWVyYXRlZFxuICogcmVwbGFjZW1lbnRzOlxuICogKiBgY2hpbGRgXG4gKiAqIGBoZWFkZXJgXG4gKiBOT1RFOiBgY2hpbGRgIGVudW1lcmF0ZWQgcmVwbGFjZW1lbnRzXG4gKiBhcmUgdHdvIGRpbWVuc2lvbmFsIGFycmF5cyFcbiAqL1xuXG4vKipcbiAqIEFib3V0IE5hbWVkIENoaWxkcmVuXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogYGhlYWRlcnNgIChhcnJheSkgLSBBbiBhcnJheSBvZiB0YWJsZSBoZWFkZXIgY2VsbHNcbiAqIGBkYXRhQ2VsbHNgIChhcnJheS1vZi1hcnJheSkgLSBBIDItZGltZW5zaW9uYWwgYXJyYXlcbiAqICAgIHN0cnVjdHVyZXMgYXMgcm93cyBieSBjb2x1bW5zIHRoYXQgY29udGFpbnMgdGhlXG4gKiAgICB0YWJsZSBkYXRhIGNlbGxzXG4gKiBgcGFnZWAgKHNpbmdsZSkgLSBBIGNlbGwgdGhhdCB0ZWxscyB3aGljaCBwYWdlIG9mIHRoZVxuICogICAgIHRhYmxlIHdlIGFyZSBsb29raW5nIGF0XG4gKiBgbGVmdGAgKHNpbmdsZSkgLSBBIGNlbGwgdGhhdCBzaG93cyB0aGUgbnVtYmVyIG9uIHRoZSBsZWZ0XG4gKiBgcmlnaHRgIChzaW5nbGUpIC0gQSBjZWxsIHRoYXQgc2hvdyB0aGUgbnVtYmVyIG9uIHRoZSByaWdodFxuICovXG5jbGFzcyBUYWJsZSBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb250ZXh0IHRvIG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlUm93cyA9IHRoaXMubWFrZVJvd3MuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5tYWtlRmlyc3RSb3cgPSB0aGlzLm1ha2VGaXJzdFJvdy5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLl9tYWtlUm93RWxlbWVudHMgPSB0aGlzLl9tYWtlUm93RWxlbWVudHMuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5fdGhlYWRTdHlsZSA9IHRoaXMuX3RoZWFkU3R5bGUuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5fZ2V0Um93RGlzcGxheUVsZW1lbnRzID0gdGhpcy5fZ2V0Um93RGlzcGxheUVsZW1lbnRzLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybihcbiAgICAgICAgICAgIGgoJ3RhYmxlJywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIlRhYmxlXCIsXG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbCB0YWJsZS1oc2Nyb2xsIHRhYmxlLXNtIHRhYmxlLXN0cmlwZWRcIlxuICAgICAgICAgICAgfSwgW1xuICAgICAgICAgICAgICAgIGgoJ3RoZWFkJywge3N0eWxlOiB0aGlzLl90aGVhZFN0eWxlKCl9LFtcbiAgICAgICAgICAgICAgICAgICAgdGhpcy5tYWtlRmlyc3RSb3coKVxuICAgICAgICAgICAgICAgIF0pLFxuICAgICAgICAgICAgICAgIGgoJ3Rib2R5Jywge30sIHRoaXMubWFrZVJvd3MoKSlcbiAgICAgICAgICAgIF0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgX3RoZWFkU3R5bGUoKXtcbiAgICAgICAgcmV0dXJuIFwiYm9yZGVyLWJvdHRvbTogYmxhY2s7Ym9yZGVyLWJvdHRvbS1zdHlsZTpzb2xpZDtib3JkZXItYm90dG9tLXdpZHRoOnRoaW47XCI7XG4gICAgfVxuXG4gICAgbWFrZUhlYWRlckVsZW1lbnRzKCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRzRm9yKCdoZWFkZXInKS5tYXAoKHJlcGxhY2VtZW50LCBpZHgpID0+IHtcbiAgICAgICAgICAgICAgICByZXR1cm4gaCgndGgnLCB7XG4gICAgICAgICAgICAgICAgICAgIHN0eWxlOiBcInZlcnRpY2FsLWFsaWduOnRvcDtcIixcbiAgICAgICAgICAgICAgICAgICAga2V5OiBgJHt0aGlzLnByb3BzLmlkfS10YWJsZS1oZWFkZXItJHtpZHh9YFxuICAgICAgICAgICAgICAgIH0sIFtyZXBsYWNlbWVudF0pO1xuICAgICAgICAgICAgfSk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZHJlbk5hbWVkKCdoZWFkZXJzJykubWFwKChyZXBsYWNlbWVudCwgaWR4KSA9PiB7XG4gICAgICAgICAgICAgICAgcmV0dXJuIGgoJ3RoJywge1xuICAgICAgICAgICAgICAgICAgICBzdHlsZTogXCJ2ZXJ0aWNhbC1hbGlnbjp0b3A7XCIsXG4gICAgICAgICAgICAgICAgICAgIGtleTogYCR7dGhpcy5wcm9wcy5pZH0tdGFibGUtaGVhZGVyLSR7aWR4fWBcbiAgICAgICAgICAgICAgICB9LCBbcmVwbGFjZW1lbnRdKTtcbiAgICAgICAgICAgIH0pO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgbWFrZVJvd3MoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLl9tYWtlUm93RWxlbWVudHModGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRzRm9yKCdjaGlsZCcpKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLl9tYWtlUm93RWxlbWVudHModGhpcy5yZW5kZXJDaGlsZHJlbk5hbWVkKCdkYXRhQ2VsbHMnKSk7XG4gICAgICAgIH1cbiAgICB9XG5cblxuXG4gICAgX21ha2VSb3dFbGVtZW50cyhlbGVtZW50cyl7XG4gICAgICAgIC8vIE5vdGU6IHJvd3MgYXJlIHRoZSAqZmlyc3QqIGRpbWVuc2lvblxuICAgICAgICAvLyBpbiB0aGUgMi1kaW1lbnNpb25hbCBhcnJheSByZXR1cm5lZFxuICAgICAgICAvLyBieSBnZXR0aW5nIHRoZSBgY2hpbGRgIHJlcGxhY2VtZW50IGVsZW1lbnRzLlxuICAgICAgICByZXR1cm4gZWxlbWVudHMubWFwKChyb3csIHJvd0lkeCkgPT4ge1xuICAgICAgICAgICAgbGV0IGNvbHVtbnMgPSByb3cubWFwKChjaGlsZEVsZW1lbnQsIGNvbElkeCkgPT4ge1xuICAgICAgICAgICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICAgICAgICAgIGgoJ3RkJywge1xuICAgICAgICAgICAgICAgICAgICAgICAga2V5OiBgJHt0aGlzLnByb3BzLmlkfS10ZC0ke3Jvd0lkeH0tJHtjb2xJZHh9YFxuICAgICAgICAgICAgICAgICAgICB9LCBbY2hpbGRFbGVtZW50XSlcbiAgICAgICAgICAgICAgICApO1xuICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICBsZXQgaW5kZXhFbGVtZW50ID0gaCgndGQnLCB7fSwgW2Ake3Jvd0lkeCArIDF9YF0pO1xuICAgICAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgICAgICBoKCd0cicsIHtrZXk6IGAke3RoaXMucHJvcHMuaWR9LXRyLSR7cm93SWR4fWB9LCBbaW5kZXhFbGVtZW50LCAuLi5jb2x1bW5zXSlcbiAgICAgICAgICAgICk7XG4gICAgICAgIH0pO1xuICAgIH1cblxuICAgIG1ha2VGaXJzdFJvdygpe1xuICAgICAgICBsZXQgaGVhZGVyRWxlbWVudHMgPSB0aGlzLm1ha2VIZWFkZXJFbGVtZW50cygpO1xuICAgICAgICByZXR1cm4oXG4gICAgICAgICAgICBoKCd0cicsIHt9LCBbXG4gICAgICAgICAgICAgICAgaCgndGgnLCB7c3R5bGU6IFwidmVydGljYWwtYWxpZ246dG9wO1wifSwgW1xuICAgICAgICAgICAgICAgICAgICBoKCdkaXYnLCB7Y2xhc3M6IFwiY2FyZFwifSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgaCgnZGl2Jywge2NsYXNzOiBcImNhcmQtYm9keSBwLTFcIn0sIFtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAuLi50aGlzLl9nZXRSb3dEaXNwbGF5RWxlbWVudHMoKVxuICAgICAgICAgICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgICAgICBdKSxcbiAgICAgICAgICAgICAgICAuLi5oZWFkZXJFbGVtZW50c1xuICAgICAgICAgICAgXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBfZ2V0Um93RGlzcGxheUVsZW1lbnRzKCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICByZXR1cm4gW1xuICAgICAgICAgICAgICAgIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdsZWZ0JyksXG4gICAgICAgICAgICAgICAgdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ3JpZ2h0JyksXG4gICAgICAgICAgICAgICAgdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ3BhZ2UnKSxcbiAgICAgICAgICAgIF07XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gW1xuICAgICAgICAgICAgICAgIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnbGVmdCcpLFxuICAgICAgICAgICAgICAgIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgncmlnaHQnKSxcbiAgICAgICAgICAgICAgICB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ3BhZ2UnKVxuICAgICAgICAgICAgXTtcbiAgICAgICAgfVxuICAgIH1cbn1cblxuZXhwb3J0IHtUYWJsZSwgVGFibGUgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIFRhYnMgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFkIGEgc2luZ2xlXG4gKiByZWd1bGFyIHJlcGxhY2VtZW50OlxuICogKiBgZGlzcGxheWBcbiAqIFRoaXMgY29tcG9uZW50IGhhcyBhIHNpbmdsZVxuICogZW51bWVyYXRlZCByZXBsYWNlbWVudDpcbiAqICogYGhlYWRlcmBcbiAqL1xuXG4vKipcbiAqIEFib3V0IE5hbWVkIENoaWxkcmVuXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogYGRpc3BsYXlgIChzaW5nbGUpIC0gVGhlIENlbGwgdGhhdCBnZXRzIGRpc3BsYXllZCB3aGVuXG4gKiAgICAgIHRoZSB0YWJzIGFyZSBzaG93aW5nXG4gKiBgaGVhZGVyc2AgKGFycmF5KSAtIEFuIGFycmF5IG9mIGNlbGxzIHRoYXQgc2VydmUgYXNcbiAqICAgICB0aGUgdGFiIGhlYWRlcnNcbiAqL1xuY2xhc3MgVGFicyBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb21wb25lbnQgbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VIZWFkZXJzID0gdGhpcy5tYWtlSGVhZGVycy5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLm1ha2VEaXNwbGF5ID0gdGhpcy5tYWtlRGlzcGxheS5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIlRhYnNcIixcbiAgICAgICAgICAgICAgICBjbGFzczogXCJjb250YWluZXItZmx1aWQgbWItM1wiXG4gICAgICAgICAgICB9LCBbXG4gICAgICAgICAgICAgICAgaCgndWwnLCB7Y2xhc3M6IFwibmF2IG5hdi10YWJzXCIsIHJvbGU6IFwidGFibGlzdFwifSwgdGhpcy5tYWtlSGVhZGVycygpKSxcbiAgICAgICAgICAgICAgICBoKCdkaXYnLCB7Y2xhc3M6IFwidGFiLWNvbnRlbnRcIn0sIFtcbiAgICAgICAgICAgICAgICAgICAgaCgnZGl2Jywge2NsYXNzOiBcInRhYi1wYW5lIGZhZGUgc2hvdyBhY3RpdmVcIiwgcm9sZTogXCJ0YWJwYW5lbFwifSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy5tYWtlRGlzcGxheSgpXG4gICAgICAgICAgICAgICAgICAgIF0pXG4gICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgIF0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgbWFrZURpc3BsYXkoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignZGlzcGxheScpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnZGlzcGxheScpO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgbWFrZUhlYWRlcnMoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudHNGb3IoJ2hlYWRlcicpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGRyZW5OYW1lZCgnaGVhZGVycycpO1xuICAgICAgICB9XG4gICAgfVxufVxuXG5cbmV4cG9ydCB7VGFicywgVGFicyBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogVGV4dCBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuY2xhc3MgVGV4dCBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybihcbiAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICBjbGFzczogXCJjZWxsXCIsXG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgc3R5bGU6IHRoaXMucHJvcHMuZXh0cmFEYXRhLmRpdlN0eWxlLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIlRleHRcIlxuICAgICAgICAgICAgfSwgW3RoaXMucHJvcHMuZXh0cmFEYXRhLnJhd1RleHRdKVxuICAgICAgICApO1xuICAgIH1cbn1cblxuZXhwb3J0IHtUZXh0LCBUZXh0IGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBUcmFjZWJhY2sgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyBhIHNpbmdsZSByZWd1bGFyXG4gKiByZXBhbGNlbWVudDpcbiAqICogYGNoaWxkYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIGB0cmFjZWJhY2tgIChzaW5nbGUpIC0gVGhlIGNlbGwgY29udGFpbmluZyB0aGUgdHJhY2ViYWNrIHRleHRcbiAqL1xuY2xhc3MgIFRyYWNlYmFjayBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb21wb25lbnQgbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VUcmFjZWJhY2sgPSB0aGlzLm1ha2VUcmFjZWJhY2suYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJUcmFjZWJhY2tcIixcbiAgICAgICAgICAgICAgICBjbGFzczogXCJhbGVydCBhbGVydC1wcmltYXJ5XCJcbiAgICAgICAgICAgIH0sIFt0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY2hpbGQnKV0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgbWFrZVRyYWNlYmFjaygpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdjaGlsZCcpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgndHJhY2ViYWNrJyk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cblxuZXhwb3J0IHtUcmFjZWJhY2ssIFRyYWNlYmFjayBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogX05hdlRhYiBDZWxsIENvbXBvbmVudFxuICogTk9URTogVGhpcyBzaG91bGQgcHJvYmFibHkganVzdCBiZVxuICogcm9sbGVkIGludG8gdGhlIE5hdiBjb21wb25lbnQgc29tZWhvdyxcbiAqIG9yIGluY2x1ZGVkIGluIGl0cyBtb2R1bGUgYXMgYSBwcml2YXRlXG4gKiBzdWJjb21wb25lbnQuXG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIG9uZSByZWd1bGFyXG4gKiByZXBsYWNlbWVudDpcbiAqICogYGNoaWxkYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgY2hpbGRgIChzaW5nbGUpIC0gVGhlIGNlbGwgaW5zaWRlIG9mIHRoZSBuYXZpZ2F0aW9uIHRhYlxuICovXG5jbGFzcyBfTmF2VGFiIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICAvLyBCaW5kIGNvbnRleHQgdG8gbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VDaGlsZCA9IHRoaXMubWFrZUNoaWxkLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuY2xpY2tIYW5kbGVyID0gdGhpcy5jbGlja0hhbmRsZXIuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgbGV0IGlubmVyQ2xhc3MgPSBcIm5hdi1saW5rXCI7XG4gICAgICAgIGlmKHRoaXMucHJvcHMuZXh0cmFEYXRhLmlzQWN0aXZlKXtcbiAgICAgICAgICAgIGlubmVyQ2xhc3MgKz0gXCIgYWN0aXZlXCI7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ2xpJywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIGNsYXNzOiBcIm5hdi1pdGVtXCIsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiX05hdlRhYlwiXG4gICAgICAgICAgICB9LCBbXG4gICAgICAgICAgICAgICAgaCgnYScsIHtcbiAgICAgICAgICAgICAgICAgICAgY2xhc3M6IGlubmVyQ2xhc3MsXG4gICAgICAgICAgICAgICAgICAgIHJvbGU6IFwidGFiXCIsXG4gICAgICAgICAgICAgICAgICAgIG9uY2xpY2s6IHRoaXMuY2xpY2tIYW5kbGVyXG4gICAgICAgICAgICAgICAgfSwgW3RoaXMubWFrZUNoaWxkKCldKVxuICAgICAgICAgICAgXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlQ2hpbGQoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY2hpbGQnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ2NoaWxkJyk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBjbGlja0hhbmRsZXIoZXZlbnQpe1xuICAgICAgICBjZWxsU29ja2V0LnNlbmRTdHJpbmcoXG4gICAgICAgICAgICBKU09OLnN0cmluZ2lmeSh0aGlzLnByb3BzLmV4dHJhRGF0YS5jbGlja0RhdGEsIG51bGwsIDQpXG4gICAgICAgICk7XG4gICAgfVxufVxuXG5leHBvcnQge19OYXZUYWIsIF9OYXZUYWIgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIF9QbG90VXBkYXRlciBDZWxsIENvbXBvbmVudFxuICogTk9URTogTGF0ZXIgcmVmYWN0b3JpbmdzIHNob3VsZCByZXN1bHQgaW5cbiAqIHRoaXMgY29tcG9uZW50IGJlY29taW5nIG9ic29sZXRlXG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG5jb25zdCBNQVhfSU5URVJWQUxTID0gMjU7XG5cbmNsYXNzIF9QbG90VXBkYXRlciBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgdGhpcy5ydW5VcGRhdGUgPSB0aGlzLnJ1blVwZGF0ZS5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLmxpc3RlbkZvclBsb3QgPSB0aGlzLmxpc3RlbkZvclBsb3QuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICBjb21wb25lbnREaWRMb2FkKCkge1xuICAgICAgICAvLyBJZiB3ZSBjYW4gZmluZCBhIG1hdGNoaW5nIFBsb3QgZWxlbWVudFxuICAgICAgICAvLyBhdCB0aGlzIHBvaW50LCB3ZSBzaW1wbHkgdXBkYXRlIGl0LlxuICAgICAgICAvLyBPdGhlcndpc2Ugd2UgbmVlZCB0byAnbGlzdGVuJyBmb3Igd2hlblxuICAgICAgICAvLyBpdCBmaW5hbGx5IGNvbWVzIGludG8gdGhlIERPTS5cbiAgICAgICAgbGV0IGluaXRpYWxQbG90RGl2ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoYHBsb3Qke3RoaXMucHJvcHMuZXh0cmFEYXRhLnBsb3RJZH1gKTtcbiAgICAgICAgaWYoaW5pdGlhbFBsb3REaXYpe1xuICAgICAgICAgICAgdGhpcy5ydW5VcGRhdGUoaW5pdGlhbFBsb3REaXYpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgdGhpcy5saXN0ZW5Gb3JQbG90KCk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIGgoJ2RpdicsXG4gICAgICAgICAgICB7XG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbFwiLFxuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIHN0eWxlOiBcImRpc3BsYXk6IG5vbmVcIixcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJfUGxvdFVwZGF0ZXJcIlxuICAgICAgICAgICAgfSwgW10pO1xuICAgIH1cblxuICAgIC8qKlxuICAgICAqIEluIHRoZSBldmVudCB0aGF0IGEgYF9QbG90VXBkYXRlcmAgaGFzIGNvbWVcbiAgICAgKiBvdmVyIHRoZSB3aXJlICpiZWZvcmUqIGl0cyBjb3JyZXNwb25kaW5nXG4gICAgICogUGxvdCBoYXMgY29tZSBvdmVyICh3aGljaCBhcHBlYXJzIHRvIGJlXG4gICAgICogY29tbW9uKSwgd2Ugd2lsbCBzZXQgYW4gaW50ZXJ2YWwgb2YgNTBtc1xuICAgICAqIGFuZCBjaGVjayBmb3IgdGhlIG1hdGNoaW5nIFBsb3QgaW4gdGhlIERPTVxuICAgICAqIE1BWF9JTlRFUlZBTFMgdGltZXMsIG9ubHkgY2FsbGluZyBgcnVuVXBkYXRlYFxuICAgICAqIG9uY2Ugd2UndmUgZm91bmQgYSBtYXRjaC5cbiAgICAgKi9cbiAgICBsaXN0ZW5Gb3JQbG90KCl7XG4gICAgICAgIGxldCBudW1DaGVja3MgPSAwO1xuICAgICAgICBsZXQgcGxvdENoZWNrZXIgPSB3aW5kb3cuc2V0SW50ZXJ2YWwoKCkgPT4ge1xuICAgICAgICAgICAgaWYobnVtQ2hlY2tzID4gTUFYX0lOVEVSVkFMUyl7XG4gICAgICAgICAgICAgICAgd2luZG93LmNsZWFySW50ZXJ2YWwocGxvdENoZWNrZXIpO1xuICAgICAgICAgICAgICAgIGNvbnNvbGUuZXJyb3IoYENvdWxkIG5vdCBmaW5kIG1hdGNoaW5nIFBsb3QgJHt0aGlzLnByb3BzLmV4dHJhRGF0YS5wbG90SWR9IGZvciBfUGxvdFVwZGF0ZXIgJHt0aGlzLnByb3BzLmlkfWApO1xuICAgICAgICAgICAgICAgIHJldHVybjtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIGxldCBwbG90RGl2ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoYHBsb3Qke3RoaXMucHJvcHMuZXh0cmFEYXRhLnBsb3RJZH1gKTtcbiAgICAgICAgICAgIGlmKHBsb3REaXYpe1xuICAgICAgICAgICAgICAgIHRoaXMucnVuVXBkYXRlKHBsb3REaXYpO1xuICAgICAgICAgICAgICAgIHdpbmRvdy5jbGVhckludGVydmFsKHBsb3RDaGVja2VyKTtcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgbnVtQ2hlY2tzICs9IDE7XG4gICAgICAgICAgICB9XG4gICAgICAgIH0sIDUwKTtcbiAgICB9XG5cbiAgICBydW5VcGRhdGUoYURPTUVsZW1lbnQpe1xuICAgICAgICBjb25zb2xlLmxvZyhcIlVwZGF0aW5nIHBsb3RseSBjaGFydC5cIik7XG4gICAgICAgIC8vIFRPRE8gVGhlc2UgYXJlIGdsb2JhbCB2YXIgZGVmaW5lZCBpbiBwYWdlLmh0bWxcbiAgICAgICAgLy8gd2Ugc2hvdWxkIGRvIHNvbWV0aGluZyBhYm91dCB0aGlzLlxuICAgICAgICBpZiAodGhpcy5wcm9wcy5leHRyYURhdGEuZXhjZXB0aW9uT2NjdXJlZCkge1xuICAgICAgICAgICAgY29uc29sZS5sb2coXCJwbG90IGV4Y2VwdGlvbiBvY2N1cmVkXCIpO1xuICAgICAgICAgICAgUGxvdGx5LnB1cmdlKGFET01FbGVtZW50KTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIGxldCBkYXRhID0gdGhpcy5wcm9wcy5leHRyYURhdGEucGxvdERhdGEubWFwKG1hcFBsb3RseURhdGEpO1xuICAgICAgICAgICAgUGxvdGx5LnJlYWN0KGFET01FbGVtZW50LCBkYXRhLCBhRE9NRWxlbWVudC5sYXlvdXQpO1xuICAgICAgICB9XG4gICAgfVxufVxuXG5leHBvcnQge19QbG90VXBkYXRlciwgX1Bsb3RVcGRhdGVyIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBUb29sIGZvciBWYWxpZGF0aW5nIENvbXBvbmVudCBQcm9wZXJ0aWVzXG4gKi9cblxuY29uc3QgcmVwb3J0ID0gKG1lc3NhZ2UsIGVycm9yTW9kZSwgc2lsZW50TW9kZSkgPT4ge1xuICAgIGlmKGVycm9yTW9kZSA9PSB0cnVlICYmIHNpbGVudE1vZGUgPT0gZmFsc2Upe1xuICAgICAgICBjb25zb2xlLmVycm9yKG1lc3NhZ2UpO1xuICAgIH0gZWxzZSBpZihzaWxlbnRNb2RlID09IGZhbHNlKXtcbiAgICAgICAgY29uc29sZS53YXJuKG1lc3NhZ2UpO1xuICAgIH1cbn07XG5cbmNvbnN0IFByb3BUeXBlcyA9IHtcbiAgICBlcnJvck1vZGU6IGZhbHNlLFxuICAgIHNpbGVudE1vZGU6IGZhbHNlLFxuICAgIG9uZU9mOiBmdW5jdGlvbihhbkFycmF5KXtcbiAgICAgICAgcmV0dXJuIGZ1bmN0aW9uKGNvbXBvbmVudE5hbWUsIHByb3BOYW1lLCBwcm9wVmFsdWUsIGlzUmVxdWlyZWQpe1xuICAgICAgICAgICAgZm9yKGxldCBpID0gMDsgaSA8IGFuQXJyYXkubGVuZ3RoOyBpKyspe1xuICAgICAgICAgICAgICAgIGxldCB0eXBlQ2hlY2tJdGVtID0gYW5BcnJheVtpXTtcbiAgICAgICAgICAgICAgICBpZih0eXBlb2YodHlwZUNoZWNrSXRlbSkgPT0gJ2Z1bmN0aW9uJyl7XG4gICAgICAgICAgICAgICAgICAgIGlmKHR5cGVDaGVja0l0ZW0oY29tcG9uZW50TmFtZSwgcHJvcE5hbWUsIHByb3BWYWx1ZSwgaXNSZXF1aXJlZCwgdHJ1ZSkpe1xuICAgICAgICAgICAgICAgICAgICAgICAgcmV0dXJuIHRydWU7XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB9IGVsc2UgaWYodHlwZUNoZWNrSXRlbSA9PSBwcm9wVmFsdWUpe1xuICAgICAgICAgICAgICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBsZXQgbWVzc2FnZSA9IGAke2NvbXBvbmVudE5hbWV9ID4+ICR7cHJvcE5hbWV9IG11c3QgYmUgb2Ygb25lIG9mIHRoZSBmb2xsb3dpbmcgdHlwZXM6ICR7YW5BcnJheX1gO1xuICAgICAgICAgICAgcmVwb3J0KG1lc3NhZ2UsIHRoaXMuZXJyb3JNb2RlLCB0aGlzLnNpbGVudE1vZGUpO1xuICAgICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICB9LmJpbmQodGhpcyk7XG4gICAgfSxcblxuICAgIGdldFZhbGlkYXRvckZvclR5cGUodHlwZVN0cil7XG4gICAgICAgIHJldHVybiBmdW5jdGlvbihjb21wb25lbnROYW1lLCBwcm9wTmFtZSwgcHJvcFZhbHVlLCBpc1JlcXVpcmVkLCBpbkNvbXBvdW5kID0gZmFsc2Upe1xuICAgICAgICAgICAgLy8gV2UgYXJlICdpbiBhIGNvbXBvdW5kIHZhbGlkYXRpb24nIHdoZW4gdGhlIGluZGl2aWR1YWxcbiAgICAgICAgICAgIC8vIFByb3BUeXBlIGNoZWNrZXJzIChpZSBmdW5jLCBudW1iZXIsIHN0cmluZywgZXRjKSBhcmVcbiAgICAgICAgICAgIC8vIGJlaW5nIGNhbGxlZCB3aXRoaW4gYSBjb21wb3VuZCB0eXBlIGNoZWNrZXIgbGlrZSBvbmVPZi5cbiAgICAgICAgICAgIC8vIEluIHRoZXNlIGNhc2VzIHdlIHdhbnQgdG8gcHJldmVudCB0aGUgY2FsbCB0byByZXBvcnQsXG4gICAgICAgICAgICAvLyB3aGljaCB0aGUgY29tcG91bmQgY2hlY2sgd2lsbCBoYW5kbGUgb24gaXRzIG93bi5cbiAgICAgICAgICAgIGlmKGluQ29tcG91bmQgPT0gZmFsc2Upe1xuICAgICAgICAgICAgICAgIGlmKHR5cGVvZihwcm9wVmFsdWUpID09IHR5cGVTdHIpe1xuICAgICAgICAgICAgICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgICAgICAgICAgICB9IGVsc2UgaWYoIWlzUmVxdWlyZWQgJiYgKHByb3BWYWx1ZSA9PSB1bmRlZmluZWQgfHwgcHJvcFZhbHVlID09IG51bGwpKXtcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIHRydWU7XG4gICAgICAgICAgICAgICAgfSBlbHNlIGlmKGlzUmVxdWlyZWQpe1xuICAgICAgICAgICAgICAgICAgICBsZXQgbWVzc2FnZSA9IGAke2NvbXBvbmVudE5hbWV9ID4+ICR7cHJvcE5hbWV9IGlzIGEgcmVxdWlyZWQgcHJvcCwgYnV0IGFzIHBhc3NlZCBhcyAke3Byb3BWYWx1ZX0hYDtcbiAgICAgICAgICAgICAgICAgICAgcmVwb3J0KG1lc3NhZ2UsIHRoaXMuZXJyb3JNb2RlLCB0aGlzLnNpbGVudE1vZGUpO1xuICAgICAgICAgICAgICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICAgICAgbGV0IG1lc3NhZ2UgPSBgJHtjb21wb25lbnROYW1lfSA+PiAke3Byb3BOYW1lfSBtdXN0IGJlIG9mIHR5cGUgJHt0eXBlU3RyfSFgO1xuICAgICAgICAgICAgICAgICAgICByZXBvcnQobWVzc2FnZSwgdGhpcy5lcnJvck1vZGUsIHRoaXMuc2lsZW50TW9kZSk7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAvLyBPdGhlcndpc2UgdGhpcyBpcyBhIHN0cmFpZ2h0Zm9yd2FyZCB0eXBlIGNoZWNrXG4gICAgICAgICAgICAvLyBiYXNlZCBvbiB0aGUgZ2l2ZW4gdHlwZS4gV2UgY2hlY2sgYXMgdXN1YWwgZm9yIHRoZSByZXF1aXJlZFxuICAgICAgICAgICAgLy8gcHJvcGVydHkgYW5kIHRoZW4gdGhlIGFjdHVhbCB0eXBlIG1hdGNoIGlmIG5lZWRlZC5cbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgaWYoaXNSZXF1aXJlZCAmJiAocHJvcFZhbHVlID09IHVuZGVmaW5lZCB8fCBwcm9wVmFsdWUgPT0gbnVsbCkpe1xuICAgICAgICAgICAgICAgICAgICBsZXQgbWVzc2FnZSA9IGAke2NvbXBvbmVudE5hbWV9ID4+ICR7cHJvcE5hbWV9IGlzIGEgcmVxdWlyZWQgcHJvcCwgYnV0IHdhcyBwYXNzZWQgYXMgJHtwcm9wVmFsdWV9IWA7XG4gICAgICAgICAgICAgICAgICAgIHJlcG9ydChtZXNzYWdlLCB0aGlzLmVycm9yTW9kZSwgdGhpcy5zaWxlbnRNb2RlKTtcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICAgICAgICAgIH0gZWxzZSBpZighaXNSZXF1aXJlZCAmJiAocHJvcFZhbHVlID09IHVuZGVmaW5lZCB8fCBwcm9wVmFsdWUgPT0gbnVsbCkpe1xuICAgICAgICAgICAgICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgcmV0dXJuIHR5cGVvZihwcm9wVmFsdWUpID09IHR5cGVTdHI7XG4gICAgICAgICAgICB9XG4gICAgICAgIH07XG4gICAgfSxcblxuICAgIGdldCBudW1iZXIoKXtcbiAgICAgICAgcmV0dXJuIHRoaXMuZ2V0VmFsaWRhdG9yRm9yVHlwZSgnbnVtYmVyJykuYmluZCh0aGlzKTtcbiAgICB9LFxuXG4gICAgZ2V0IGJvb2xlYW4oKXtcbiAgICAgICAgcmV0dXJuIHRoaXMuZ2V0VmFsaWRhdG9yRm9yVHlwZSgnYm9vbGVhbicpLmJpbmQodGhpcyk7XG4gICAgfSxcblxuICAgIGdldCBzdHJpbmcoKXtcbiAgICAgICAgcmV0dXJuIHRoaXMuZ2V0VmFsaWRhdG9yRm9yVHlwZSgnc3RyaW5nJykuYmluZCh0aGlzKTtcbiAgICB9LFxuXG4gICAgZ2V0IG9iamVjdCgpe1xuICAgICAgICByZXR1cm4gdGhpcy5nZXRWYWxpZGF0b3JGb3JUeXBlKCdvYmplY3QnKS5iaW5kKHRoaXMpO1xuICAgIH0sXG5cbiAgICBnZXQgZnVuYygpe1xuICAgICAgICByZXR1cm4gdGhpcy5nZXRWYWxpZGF0b3JGb3JUeXBlKCdmdW5jdGlvbicpLmJpbmQodGhpcyk7XG4gICAgfSxcblxuICAgIHZhbGlkYXRlOiBmdW5jdGlvbihjb21wb25lbnROYW1lLCBwcm9wcywgcHJvcEluZm8pe1xuICAgICAgICBsZXQgcHJvcE5hbWVzID0gbmV3IFNldChPYmplY3Qua2V5cyhwcm9wcykpO1xuICAgICAgICBwcm9wTmFtZXMuZGVsZXRlKCdjaGlsZHJlbicpO1xuICAgICAgICBwcm9wTmFtZXMuZGVsZXRlKCduYW1lZENoaWxkcmVuJyk7XG4gICAgICAgIHByb3BOYW1lcy5kZWxldGUoJ2lkJyk7XG4gICAgICAgIHByb3BOYW1lcy5kZWxldGUoJ2V4dHJhRGF0YScpOyAvLyBGb3Igbm93XG4gICAgICAgIGxldCBwcm9wc1RvVmFsaWRhdGUgPSBBcnJheS5mcm9tKHByb3BOYW1lcyk7XG5cbiAgICAgICAgLy8gUGVyZm9ybSBhbGwgdGhlIHZhbGlkYXRpb25zIG9uIGVhY2ggcHJvcGVydHlcbiAgICAgICAgLy8gYWNjb3JkaW5nIHRvIGl0cyBkZXNjcmlwdGlvbi4gV2Ugc3RvcmUgd2hldGhlclxuICAgICAgICAvLyBvciBub3QgdGhlIGdpdmVuIHByb3BlcnR5IHdhcyBjb21wbGV0ZWx5IHZhbGlkXG4gICAgICAgIC8vIGFuZCB0aGVuIGV2YWx1YXRlIHRoZSB2YWxpZGl0eSBvZiBhbGwgYXQgdGhlIGVuZC5cbiAgICAgICAgbGV0IHZhbGlkYXRpb25SZXN1bHRzID0ge307XG4gICAgICAgIHByb3BzVG9WYWxpZGF0ZS5mb3JFYWNoKHByb3BOYW1lID0+IHtcbiAgICAgICAgICAgIGxldCBwcm9wVmFsID0gcHJvcHNbcHJvcE5hbWVdO1xuICAgICAgICAgICAgbGV0IHZhbGlkYXRpb25Ub0NoZWNrID0gcHJvcEluZm9bcHJvcE5hbWVdO1xuICAgICAgICAgICAgaWYodmFsaWRhdGlvblRvQ2hlY2spe1xuICAgICAgICAgICAgICAgIGxldCBoYXNWYWxpZERlc2NyaXB0aW9uID0gdGhpcy52YWxpZGF0ZURlc2NyaXB0aW9uKGNvbXBvbmVudE5hbWUsIHByb3BOYW1lLCB2YWxpZGF0aW9uVG9DaGVjayk7XG4gICAgICAgICAgICAgICAgbGV0IGhhc1ZhbGlkUHJvcFR5cGVzID0gdmFsaWRhdGlvblRvQ2hlY2sudHlwZShjb21wb25lbnROYW1lLCBwcm9wTmFtZSwgcHJvcFZhbCwgdmFsaWRhdGlvblRvQ2hlY2sucmVxdWlyZWQpO1xuICAgICAgICAgICAgICAgIGlmKGhhc1ZhbGlkRGVzY3JpcHRpb24gJiYgaGFzVmFsaWRQcm9wVHlwZXMpe1xuICAgICAgICAgICAgICAgICAgICB2YWxpZGF0aW9uUmVzdWx0c1twcm9wTmFtZV0gPSB0cnVlO1xuICAgICAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgICAgIHZhbGlkYXRpb25SZXN1bHRzW3Byb3BOYW1lXSA9IGZhbHNlO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgLy8gSWYgd2UgZ2V0IGhlcmUsIHRoZSBjb25zdW1lciBoYXMgcGFzc2VkIGluIGEgcHJvcFxuICAgICAgICAgICAgICAgIC8vIHRoYXQgaXMgbm90IHByZXNlbnQgaW4gdGhlIHByb3BUeXBlcyBkZXNjcmlwdGlvbi5cbiAgICAgICAgICAgICAgICAvLyBXZSByZXBvcnQgdG8gdGhlIGNvbnNvbGUgYXMgbmVlZGVkIGFuZCB2YWxpZGF0ZSBhcyBmYWxzZS5cbiAgICAgICAgICAgICAgICBsZXQgbWVzc2FnZSA9IGAke2NvbXBvbmVudE5hbWV9IGhhcyBhIHByb3AgY2FsbGVkIFwiJHtwcm9wTmFtZX1cIiB0aGF0IGlzIG5vdCBkZXNjcmliZWQgaW4gcHJvcFR5cGVzIWA7XG4gICAgICAgICAgICAgICAgcmVwb3J0KG1lc3NhZ2UsIHRoaXMuZXJyb3JNb2RlLCB0aGlzLnNpbGVudE1vZGUpO1xuICAgICAgICAgICAgICAgIHZhbGlkYXRpb25SZXN1bHRzW3Byb3BOYW1lXSA9IGZhbHNlO1xuICAgICAgICAgICAgfVxuICAgICAgICB9KTtcblxuICAgICAgICAvLyBJZiB0aGVyZSB3ZXJlIGFueSB0aGF0IGRpZCBub3QgdmFsaWRhdGUsIHJldHVyblxuICAgICAgICAvLyBmYWxzZSBhbmQgcmVwb3J0IGFzIG11Y2guXG4gICAgICAgIGxldCBpbnZhbGlkcyA9IFtdO1xuICAgICAgICBPYmplY3Qua2V5cyh2YWxpZGF0aW9uUmVzdWx0cykuZm9yRWFjaChrZXkgPT4ge1xuICAgICAgICAgICAgaWYodmFsaWRhdGlvblJlc3VsdHNba2V5XSA9PSBmYWxzZSl7XG4gICAgICAgICAgICAgICAgaW52YWxpZHMucHVzaChrZXkpO1xuICAgICAgICAgICAgfVxuICAgICAgICB9KTtcbiAgICAgICAgaWYoaW52YWxpZHMubGVuZ3RoID4gMCl7XG4gICAgICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgICAgfVxuICAgIH0sXG5cbiAgICB2YWxpZGF0ZVJlcXVpcmVkOiBmdW5jdGlvbihjb21wb25lbnROYW1lLCBwcm9wTmFtZSwgcHJvcFZhbCwgaXNSZXF1aXJlZCl7XG4gICAgICAgIGlmKGlzUmVxdWlyZWQgPT0gdHJ1ZSl7XG4gICAgICAgICAgICBpZihwcm9wVmFsID09IG51bGwgfHwgcHJvcFZhbCA9PSB1bmRlZmluZWQpe1xuICAgICAgICAgICAgICAgIGxldCBtZXNzYWdlID0gYCR7Y29tcG9uZW50TmFtZX0gPj4gJHtwcm9wTmFtZX0gcmVxdWlyZXMgYSB2YWx1ZSwgYnV0ICR7cHJvcFZhbH0gd2FzIHBhc3NlZCFgO1xuICAgICAgICAgICAgICAgIHJlcG9ydChtZXNzYWdlLCB0aGlzLmVycm9yTW9kZSwgdGhpcy5zaWxlbnRNb2RlKTtcbiAgICAgICAgICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIHRydWU7XG4gICAgfSxcblxuICAgIHZhbGlkYXRlRGVzY3JpcHRpb246IGZ1bmN0aW9uKGNvbXBvbmVudE5hbWUsIHByb3BOYW1lLCBwcm9wKXtcbiAgICAgICAgbGV0IGRlc2MgPSBwcm9wLmRlc2NyaXB0aW9uO1xuICAgICAgICBpZihkZXNjID09IHVuZGVmaW5lZCB8fCBkZXNjID09IFwiXCIgfHwgZGVzYyA9PSBudWxsKXtcbiAgICAgICAgICAgIGxldCBtZXNzYWdlID0gYCR7Y29tcG9uZW50TmFtZX0gPj4gJHtwcm9wTmFtZX0gaGFzIGFuIGVtcHR5IGRlc2NyaXB0aW9uIWA7XG4gICAgICAgICAgICByZXBvcnQobWVzc2FnZSwgdGhpcy5lcnJvck1vZGUsIHRoaXMuc2lsZW50TW9kZSk7XG4gICAgICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIHRydWU7XG4gICAgfVxufTtcblxuZXhwb3J0IHtcbiAgICBQcm9wVHlwZXNcbn07XG5cblxuLyoqKlxubnVtYmVyOiBmdW5jdGlvbihjb21wb25lbnROYW1lLCBwcm9wTmFtZSwgcHJvcFZhbHVlLCBpbkNvbXBvdW5kID0gZmFsc2Upe1xuICAgICAgICBpZihpbkNvbXBvdW5kID09IGZhbHNlKXtcbiAgICAgICAgICAgIGlmKHR5cGVvZihwcm9wVmFsdWUpID09ICdudW1iZXInKXtcbiAgICAgICAgICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgbGV0IG1lc3NhZ2UgPSBgJHtjb21wb25lbnROYW1lfSA+PiAke3Byb3BOYW1lfSBtdXN0IGJlIG9mIHR5cGUgbnVtYmVyIWA7XG4gICAgICAgICAgICAgICAgcmVwb3J0KG1lc3NhZ2UsIHRoaXMuZXJyb3JNb2RlLCB0aGlzLnNpbGVudE1vZGUpO1xuICAgICAgICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0eXBlb2YocHJvcFZhbHVlKSA9PSAnbnVtYmVyJztcbiAgICAgICAgfVxuXG4gICAgfS5iaW5kKHRoaXMpLFxuXG4gICAgc3RyaW5nOiBmdW5jdGlvbihjb21wb25lbnROYW1lLCBwcm9wTmFtZSwgcHJvcFZhbHVlLCBpbkNvbXBvdW5kID0gZmFsc2Upe1xuICAgICAgICBpZihpbkNvbXBvdW5kID09IGZhbHNlKXtcbiAgICAgICAgICAgIGlmKHR5cGVvZihwcm9wVmFsdWUpID09ICdzdHJpbmcnKXtcbiAgICAgICAgICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgbGV0IG1lc3NhZ2UgPSBgJHtjb21wb25lbnROYW1lfSA+PiAke3Byb3BOYW1lfSBtdXN0IGJlIG9mIHR5cGUgc3RyaW5nIWA7XG4gICAgICAgICAgICAgICAgcmVwb3J0KG1lc3NhZ2UsIHRoaXMuZXJyb3JNb2RlLCB0aGlzLnNpbGVudE1vZGUpO1xuICAgICAgICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0eXBlb2YocHJvcFZhbHVlKSA9PSAnc3RyaW5nJztcbiAgICAgICAgfVxuICAgIH0uYmluZCh0aGlzKSxcblxuICAgIGJvb2xlYW46IGZ1bmN0aW9uKGNvbXBvbmVudE5hbWUsIHByb3BOYW1lLCBwcm9wVmFsdWUsIGluQ29tcG91bmQgPSBmYWxzZSl7XG4gICAgICAgIGlmKGluQ29tcG91bmQgPT0gZmFsc2Upe1xuICAgICAgICAgICAgaWYodHlwZW9mKHByb3BWYWx1ZSkgPT0gJ2Jvb2xlYW4nKXtcbiAgICAgICAgICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgbGV0IG1lc3NhZ2UgPSBgJHtjb21wb25lbnROYW1lfSA+PiAke3Byb3BOYW1lfSBtdXN0IGJlIG9mIHR5cGUgYm9vbGVhbiFgO1xuICAgICAgICAgICAgICAgIHJlcG9ydChtZXNzYWdlLCB0aGlzLmVycm9yTW9kZSwgdGhpcy5zaWxlbnRNb2RlKTtcbiAgICAgICAgICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICAgICAgICB9XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdHlwZW9mKHByb3BWYWx1ZSkgPT0gJ2Jvb2xlYW4nO1xuICAgICAgICB9XG4gICAgfS5iaW5kKHRoaXMpLFxuXG4gICAgb2JqZWN0OiBmdW5jdGlvbihjb21wb25lbnROYW1lLCBwcm9wTmFtZSwgcHJvcFZhbHVlLCBpbkNvbXBvdW5kID0gZmFsc2Upe1xuICAgICAgICBpZihpbkNvbXBvdW5kID09IGZhbHNlKXtcbiAgICAgICAgICAgIGlmKHR5cGVvZihwcm9wVmFsdWUpID09ICdvYmplY3QnKXtcbiAgICAgICAgICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgbGV0IG1lc3NhZ2UgPSBgJHtjb21wb25lbnROYW1lfSA+PiAke3Byb3BOYW1lfSBtdXN0IGJlIG9mIHR5cGUgb2JqZWN0IWA7XG4gICAgICAgICAgICAgICAgcmVwb3J0KG1lc3NhZ2UsIHRoaXMuZXJyb3JNb2RlLCB0aGlzLnNpbGVudE1vZGUpO1xuICAgICAgICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0eXBlb2YocHJvcFZhbHVlKSA9PSAnb2JqZWN0JztcbiAgICAgICAgfVxuICAgIH0uYmluZCh0aGlzKSxcblxuICAgIGZ1bmM6IGZ1bmN0aW9uKGNvbXBvbmVudE5hbWUsIHByb3BOYW1lLCBwcm9wVmFsdWUsIGluQ29tcG91bmQgPSBmYWxzZSl7XG4gICAgICAgIGlmKGluQ29tcG91bmQgPT0gZmFsc2Upe1xuICAgICAgICAgICAgaWYodHlwZW9mKHByb3BWYWx1ZSkgPT0gJ2Z1bmN0aW9uJyl7XG4gICAgICAgICAgICAgICAgcmV0dXJuIHRydWU7XG4gICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgIGxldCBtZXNzYWdlID0gYCR7Y29tcG9uZW50TmFtZX0gPj4gJHtwcm9wTmFtZX0gbXVzdCBiZSBvZiB0eXBlIGZ1bmN0aW9uIWA7XG4gICAgICAgICAgICAgICAgcmVwb3J0KG1lc3NhZ2UsIHRoaXMuZXJyb3JNb2RlLCB0aGlzLnNpbGVudE1vZGUpO1xuICAgICAgICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0eXBlb2YocHJvcFZhbHVlKSA9PSAnZnVuY3Rpb24nO1xuICAgICAgICB9XG4gICAgfS5iaW5kKHRoaXMpLFxuXG4qKiovXG4iLCJjbGFzcyBSZXBsYWNlbWVudHNIYW5kbGVyIHtcbiAgICBjb25zdHJ1Y3RvcihyZXBsYWNlbWVudHMpe1xuICAgICAgICB0aGlzLnJlcGxhY2VtZW50cyA9IHJlcGxhY2VtZW50cztcbiAgICAgICAgdGhpcy5yZWd1bGFyID0ge307XG4gICAgICAgIHRoaXMuZW51bWVyYXRlZCA9IHt9O1xuXG4gICAgICAgIGlmKHJlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICB0aGlzLnByb2Nlc3NSZXBsYWNlbWVudHMoKTtcbiAgICAgICAgfVxuXG4gICAgICAgIC8vIEJpbmQgY29udGV4dCB0byBtZXRob2RzXG4gICAgICAgIHRoaXMucHJvY2Vzc1JlcGxhY2VtZW50cyA9IHRoaXMucHJvY2Vzc1JlcGxhY2VtZW50cy5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLnByb2Nlc3NSZWd1bGFyID0gdGhpcy5wcm9jZXNzUmVndWxhci5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLmhhc1JlcGxhY2VtZW50ID0gdGhpcy5oYXNSZXBsYWNlbWVudC5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLmdldFJlcGxhY2VtZW50Rm9yID0gdGhpcy5nZXRSZXBsYWNlbWVudEZvci5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLmdldFJlcGxhY2VtZW50c0ZvciA9IHRoaXMuZ2V0UmVwbGFjZW1lbnRzRm9yLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMubWFwUmVwbGFjZW1lbnRzRm9yID0gdGhpcy5tYXBSZXBsYWNlbWVudHNGb3IuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICBwcm9jZXNzUmVwbGFjZW1lbnRzKCl7XG4gICAgICAgIHRoaXMucmVwbGFjZW1lbnRzLmZvckVhY2gocmVwbGFjZW1lbnQgPT4ge1xuICAgICAgICAgICAgbGV0IHJlcGxhY2VtZW50SW5mbyA9IHRoaXMuY29uc3RydWN0b3IucmVhZFJlcGxhY2VtZW50U3RyaW5nKHJlcGxhY2VtZW50KTtcbiAgICAgICAgICAgIGlmKHJlcGxhY2VtZW50SW5mby5pc0VudW1lcmF0ZWQpe1xuICAgICAgICAgICAgICAgIHRoaXMucHJvY2Vzc0VudW1lcmF0ZWQocmVwbGFjZW1lbnQsIHJlcGxhY2VtZW50SW5mbyk7XG4gICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgIHRoaXMucHJvY2Vzc1JlZ3VsYXIocmVwbGFjZW1lbnQsIHJlcGxhY2VtZW50SW5mbyk7XG4gICAgICAgICAgICB9XG4gICAgICAgIH0pO1xuICAgICAgICAvLyBOb3cgd2UgdXBkYXRlIHRoaXMuZW51bWVyYXRlZCB0byBoYXZlIGl0J3MgdG9wIGxldmVsXG4gICAgICAgIC8vIHZhbHVlcyBhcyBBcnJheXMgaW5zdGVhZCBvZiBuZXN0ZWQgZGljdHMgYW5kIHdlIHNvcnRcbiAgICAgICAgLy8gYmFzZWQgb24gdGhlIGV4dHJhY3RlZCBpbmRpY2VzICh3aGljaCBhcmUgYXQgdGhpcyBwb2ludFxuICAgICAgICAvLyBqdXN0IGtleXMgb24gc3ViZGljdHMgb3IgbXVsdGlkaW1lbnNpb25hbCBkaWN0cylcbiAgICAgICAgT2JqZWN0LmtleXModGhpcy5lbnVtZXJhdGVkKS5mb3JFYWNoKGtleSA9PiB7XG4gICAgICAgICAgICBsZXQgZW51bWVyYXRlZFJlcGxhY2VtZW50cyA9IHRoaXMuZW51bWVyYXRlZFtrZXldO1xuICAgICAgICAgICAgdGhpcy5lbnVtZXJhdGVkW2tleV0gPSB0aGlzLmNvbnN0cnVjdG9yLmVudW1lcmF0ZWRWYWxUb1NvcnRlZEFycmF5KGVudW1lcmF0ZWRSZXBsYWNlbWVudHMpO1xuICAgICAgICB9KTtcbiAgICB9XG5cbiAgICBwcm9jZXNzUmVndWxhcihyZXBsYWNlbWVudE5hbWUsIHJlcGxhY2VtZW50SW5mbyl7XG4gICAgICAgIGxldCByZXBsYWNlbWVudEtleSA9IHRoaXMuY29uc3RydWN0b3Iua2V5RnJvbU5hbWVQYXJ0cyhyZXBsYWNlbWVudEluZm8ubmFtZVBhcnRzKTtcbiAgICAgICAgdGhpcy5yZWd1bGFyW3JlcGxhY2VtZW50S2V5XSA9IHJlcGxhY2VtZW50TmFtZTtcbiAgICB9XG5cbiAgICBwcm9jZXNzRW51bWVyYXRlZChyZXBsYWNlbWVudE5hbWUsIHJlcGxhY2VtZW50SW5mbyl7XG4gICAgICAgIGxldCByZXBsYWNlbWVudEtleSA9IHRoaXMuY29uc3RydWN0b3Iua2V5RnJvbU5hbWVQYXJ0cyhyZXBsYWNlbWVudEluZm8ubmFtZVBhcnRzKTtcbiAgICAgICAgbGV0IGN1cnJlbnRFbnRyeSA9IHRoaXMuZW51bWVyYXRlZFtyZXBsYWNlbWVudEtleV07XG5cbiAgICAgICAgLy8gSWYgaXQncyB1bmRlZmluZWQsIHRoaXMgaXMgdGhlIGZpcnN0XG4gICAgICAgIC8vIG9mIHRoZSBlbnVtZXJhdGVkIHJlcGxhY2VtZW50cyBmb3IgdGhpc1xuICAgICAgICAvLyBrZXksIGllIHNvbWV0aGluZyBsaWtlIF9fX19jaGlsZF8wX19cbiAgICAgICAgaWYoY3VycmVudEVudHJ5ID09IHVuZGVmaW5lZCl7XG4gICAgICAgICAgICB0aGlzLmVudW1lcmF0ZWRbcmVwbGFjZW1lbnRLZXldID0ge307XG4gICAgICAgICAgICBjdXJyZW50RW50cnkgPSB0aGlzLmVudW1lcmF0ZWRbcmVwbGFjZW1lbnRLZXldO1xuICAgICAgICB9XG5cbiAgICAgICAgLy8gV2UgYWRkIHRoZSBlbnVtZXJhdGVkIGluZGljZXMgYXMga2V5cyB0byBhIGRpY3RcbiAgICAgICAgLy8gYW5kIHdlIGRvIHRoaXMgcmVjdXJzaXZlbHkgYWNyb3NzIGRpbWVuc2lvbnMgYXNcbiAgICAgICAgLy8gbmVlZGVkLlxuICAgICAgICB0aGlzLmNvbnN0cnVjdG9yLnByb2Nlc3NEaW1lbnNpb24ocmVwbGFjZW1lbnRJbmZvLmVudW1lcmF0ZWRWYWx1ZXMsIGN1cnJlbnRFbnRyeSwgcmVwbGFjZW1lbnROYW1lKTtcbiAgICB9XG5cbiAgICAvLyBBY2Nlc3NpbmcgYW5kIG90aGVyIENvbnZlbmllbmNlIE1ldGhvZHNcbiAgICBoYXNSZXBsYWNlbWVudChhUmVwbGFjZW1lbnROYW1lKXtcbiAgICAgICAgaWYodGhpcy5yZWd1bGFyLmhhc093blByb3BlcnR5KGFSZXBsYWNlbWVudE5hbWUpKXtcbiAgICAgICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgICB9IGVsc2UgaWYodGhpcy5lbnVtZXJhdGVkLmhhc093blByb3BlcnR5KGFSZXBsYWNlbWVudE5hbWUpKXtcbiAgICAgICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiBmYWxzZTtcbiAgICB9XG5cbiAgICBnZXRSZXBsYWNlbWVudEZvcihhUmVwbGFjZW1lbnROYW1lKXtcbiAgICAgICAgbGV0IGZvdW5kID0gdGhpcy5yZWd1bGFyW2FSZXBsYWNlbWVudE5hbWVdO1xuICAgICAgICBpZihmb3VuZCA9PSB1bmRlZmluZWQpe1xuICAgICAgICAgICAgcmV0dXJuIG51bGw7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIGZvdW5kO1xuICAgIH1cblxuICAgIGdldFJlcGxhY2VtZW50c0ZvcihhUmVwbGFjZW1lbnROYW1lKXtcbiAgICAgICAgbGV0IGZvdW5kID0gdGhpcy5lbnVtZXJhdGVkW2FSZXBsYWNlbWVudE5hbWVdO1xuICAgICAgICBpZihmb3VuZCA9PSB1bmRlZmluZWQpe1xuICAgICAgICAgICAgcmV0dXJuIG51bGw7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIGZvdW5kO1xuICAgIH1cblxuICAgIG1hcFJlcGxhY2VtZW50c0ZvcihhUmVwbGFjZW1lbnROYW1lLCBtYXBGdW5jdGlvbil7XG4gICAgICAgIGlmKCF0aGlzLmhhc1JlcGxhY2VtZW50KGFSZXBsYWNlbWVudE5hbWUpKXtcbiAgICAgICAgICAgIHRocm93IG5ldyBFcnJvcihgSW52YWxpZCByZXBsYWNlbWVudCBuYW1lOiAke2FSZXBsYWNlbWVudG5hbWV9YCk7XG4gICAgICAgIH1cbiAgICAgICAgbGV0IHJvb3QgPSB0aGlzLmdldFJlcGxhY2VtZW50c0ZvcihhUmVwbGFjZW1lbnROYW1lKTtcbiAgICAgICAgcmV0dXJuIHRoaXMuX3JlY3Vyc2l2ZWx5TWFwKHJvb3QsIG1hcEZ1bmN0aW9uKTtcbiAgICB9XG5cbiAgICBfcmVjdXJzaXZlbHlNYXAoY3VycmVudEl0ZW0sIG1hcEZ1bmN0aW9uKXtcbiAgICAgICAgaWYoIUFycmF5LmlzQXJyYXkoY3VycmVudEl0ZW0pKXtcbiAgICAgICAgICAgIHJldHVybiBtYXBGdW5jdGlvbihjdXJyZW50SXRlbSk7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIGN1cnJlbnRJdGVtLm1hcChzdWJJdGVtID0+IHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLl9yZWN1cnNpdmVseU1hcChzdWJJdGVtLCBtYXBGdW5jdGlvbik7XG4gICAgICAgIH0pO1xuICAgIH1cblxuICAgIC8vIFN0YXRpYyBoZWxwZXJzXG4gICAgc3RhdGljIHByb2Nlc3NEaW1lbnNpb24ocmVtYWluaW5nVmFscywgaW5EaWN0LCByZXBsYWNlbWVudE5hbWUpe1xuICAgICAgICBpZihyZW1haW5pbmdWYWxzLmxlbmd0aCA9PSAxKXtcbiAgICAgICAgICAgIGluRGljdFtyZW1haW5pbmdWYWxzWzBdXSA9IHJlcGxhY2VtZW50TmFtZTtcbiAgICAgICAgICAgIHJldHVybjtcbiAgICAgICAgfVxuICAgICAgICBsZXQgbmV4dEtleSA9IHJlbWFpbmluZ1ZhbHNbMF07XG4gICAgICAgIGxldCBuZXh0RGljdCA9IGluRGljdFtuZXh0S2V5XTtcbiAgICAgICAgaWYobmV4dERpY3QgPT0gdW5kZWZpbmVkKXtcbiAgICAgICAgICAgIGluRGljdFtuZXh0S2V5XSA9IHt9O1xuICAgICAgICAgICAgbmV4dERpY3QgPSBpbkRpY3RbbmV4dEtleV07XG4gICAgICAgIH1cbiAgICAgICAgdGhpcy5wcm9jZXNzRGltZW5zaW9uKHJlbWFpbmluZ1ZhbHMuc2xpY2UoMSksIG5leHREaWN0LCByZXBsYWNlbWVudE5hbWUpO1xuICAgIH1cblxuICAgIHN0YXRpYyBlbnVtZXJhdGVkVmFsVG9Tb3J0ZWRBcnJheShhRGljdCwgYWNjdW11bGF0ZSA9IFtdKXtcbiAgICAgICAgaWYodHlwZW9mIGFEaWN0ICE9PSAnb2JqZWN0Jyl7XG4gICAgICAgICAgICByZXR1cm4gYURpY3Q7XG4gICAgICAgIH1cbiAgICAgICAgbGV0IHNvcnRlZEtleXMgPSBPYmplY3Qua2V5cyhhRGljdCkuc29ydCgoZmlyc3QsIHNlY29uZCkgPT4ge1xuICAgICAgICAgICAgcmV0dXJuIChwYXJzZUludChmaXJzdCkgLSBwYXJzZUludChzZWNvbmQpKTtcbiAgICAgICAgfSk7XG4gICAgICAgIGxldCBzdWJFbnRyaWVzID0gc29ydGVkS2V5cy5tYXAoa2V5ID0+IHtcbiAgICAgICAgICAgIGxldCBlbnRyeSA9IGFEaWN0W2tleV07XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5lbnVtZXJhdGVkVmFsVG9Tb3J0ZWRBcnJheShlbnRyeSk7XG4gICAgICAgIH0pO1xuICAgICAgICByZXR1cm4gc3ViRW50cmllcztcbiAgICB9XG5cbiAgICBzdGF0aWMga2V5RnJvbU5hbWVQYXJ0cyhuYW1lUGFydHMpe1xuICAgICAgICByZXR1cm4gbmFtZVBhcnRzLmpvaW4oXCItXCIpO1xuICAgIH1cblxuICAgIHN0YXRpYyByZWFkUmVwbGFjZW1lbnRTdHJpbmcocmVwbGFjZW1lbnQpe1xuICAgICAgICBsZXQgbmFtZVBhcnRzID0gW107XG4gICAgICAgIGxldCBpc0VudW1lcmF0ZWQgPSBmYWxzZTtcbiAgICAgICAgbGV0IGVudW1lcmF0ZWRWYWx1ZXMgPSBbXTtcbiAgICAgICAgbGV0IHBpZWNlcyA9IHJlcGxhY2VtZW50LnNwbGl0KCdfJykuZmlsdGVyKGl0ZW0gPT4ge1xuICAgICAgICAgICAgcmV0dXJuIGl0ZW0gIT0gJyc7XG4gICAgICAgIH0pO1xuICAgICAgICBwaWVjZXMuZm9yRWFjaChwaWVjZSA9PiB7XG4gICAgICAgICAgICBsZXQgbnVtID0gcGFyc2VJbnQocGllY2UpO1xuICAgICAgICAgICAgaWYoaXNOYU4obnVtKSl7XG4gICAgICAgICAgICAgICAgbmFtZVBhcnRzLnB1c2gocGllY2UpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgaXNFbnVtZXJhdGVkID0gdHJ1ZTtcbiAgICAgICAgICAgIGVudW1lcmF0ZWRWYWx1ZXMucHVzaChudW0pO1xuICAgICAgICB9XG4gICAgICAgIH0pO1xuICAgICAgICByZXR1cm4ge1xuICAgICAgICAgICAgbmFtZVBhcnRzLFxuICAgICAgICAgICAgaXNFbnVtZXJhdGVkLFxuICAgICAgICAgICAgZW51bWVyYXRlZFZhbHVlc1xuICAgICAgICB9O1xuICAgIH1cbn1cblxuZXhwb3J0IHtcbiAgICBSZXBsYWNlbWVudHNIYW5kbGVyLFxuICAgIFJlcGxhY2VtZW50c0hhbmRsZXIgYXMgZGVmYXVsdFxufTtcbiIsImltcG9ydCAnbWFxdWV0dGUnO1xuY29uc3QgaCA9IG1hcXVldHRlLmg7XG4vL2ltcG9ydCB7bGFuZ1Rvb2xzfSBmcm9tICdhY2UvZXh0L2xhbmd1YWdlX3Rvb2xzJztcbmltcG9ydCB7Q2VsbEhhbmRsZXJ9IGZyb20gJy4vQ2VsbEhhbmRsZXInO1xuaW1wb3J0IHtDZWxsU29ja2V0fSBmcm9tICcuL0NlbGxTb2NrZXQnO1xuaW1wb3J0IHtDb21wb25lbnRSZWdpc3RyeX0gZnJvbSAnLi9Db21wb25lbnRSZWdpc3RyeSc7XG5cbi8qKlxuICogR2xvYmFsc1xuICoqL1xud2luZG93LmxhbmdUb29scyA9IGFjZS5yZXF1aXJlKFwiYWNlL2V4dC9sYW5ndWFnZV90b29sc1wiKTtcbndpbmRvdy5hY2VFZGl0b3JzID0ge307XG53aW5kb3cuaGFuZHNPblRhYmxlcyA9IHt9O1xuXG4vKipcbiAqIEluaXRpYWwgUmVuZGVyXG4gKiovXG5jb25zdCBpbml0aWFsUmVuZGVyID0gZnVuY3Rpb24oKXtcbiAgICByZXR1cm4gaChcImRpdlwiLCB7fSwgW1xuICAgICAgICAgaChcImRpdlwiLCB7aWQ6IFwicGFnZV9yb290XCJ9LCBbXG4gICAgICAgICAgICAgaChcImRpdi5jb250YWluZXItZmx1aWRcIiwge30sIFtcbiAgICAgICAgICAgICAgICAgaChcImRpdi5jYXJkXCIsIHtjbGFzczogXCJtdC01XCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgICBoKFwiZGl2LmNhcmQtYm9keVwiLCB7fSwgW1wiTG9hZGluZy4uLlwiXSlcbiAgICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgICBdKVxuICAgICAgICAgXSksXG4gICAgICAgICBoKFwiZGl2XCIsIHtpZDogXCJob2xkaW5nX3BlblwiLCBzdHlsZTogXCJkaXNwbGF5Om5vbmVcIn0sIFtdKVxuICAgICBdKTtcbn07XG5cbi8qKlxuICogQ2VsbCBTb2NrZXQgYW5kIEhhbmRsZXJcbiAqKi9cbmxldCBwcm9qZWN0b3IgPSBtYXF1ZXR0ZS5jcmVhdGVQcm9qZWN0b3IoKTtcbmNvbnN0IGNlbGxTb2NrZXQgPSBuZXcgQ2VsbFNvY2tldCgpO1xuY29uc3QgY2VsbEhhbmRsZXIgPSBuZXcgQ2VsbEhhbmRsZXIoaCwgcHJvamVjdG9yLCBDb21wb25lbnRSZWdpc3RyeSk7XG5jZWxsU29ja2V0Lm9uUG9zdHNjcmlwdHMoY2VsbEhhbmRsZXIuaGFuZGxlUG9zdHNjcmlwdCk7XG5jZWxsU29ja2V0Lm9uTWVzc2FnZShjZWxsSGFuZGxlci5oYW5kbGVNZXNzYWdlKTtcbmNlbGxTb2NrZXQub25DbG9zZShjZWxsSGFuZGxlci5zaG93Q29ubmVjdGlvbkNsb3NlZCk7XG5jZWxsU29ja2V0Lm9uRXJyb3IoZXJyID0+IHtcbiAgICBjb25zb2xlLmVycm9yKFwiU09DS0VUIEVSUk9SOiBcIiwgZXJyKTtcbn0pO1xuXG4vKiogRm9yIG5vdywgd2UgYmluZCB0aGUgY3VycmVudCBzb2NrZXQgYW5kIGhhbmRsZXIgdG8gdGhlIGdsb2JhbCB3aW5kb3cgKiovXG53aW5kb3cuY2VsbFNvY2tldCA9IGNlbGxTb2NrZXQ7XG53aW5kb3cuY2VsbEhhbmRsZXIgPSBjZWxsSGFuZGxlcjtcblxuLyoqIFJlbmRlciB0b3AgbGV2ZWwgY29tcG9uZW50IG9uY2UgRE9NIGlzIHJlYWR5ICoqL1xuZG9jdW1lbnQuYWRkRXZlbnRMaXN0ZW5lcignRE9NQ29udGVudExvYWRlZCcsICgpID0+IHtcbiAgICBwcm9qZWN0b3IuYXBwZW5kKGRvY3VtZW50LmJvZHksIGluaXRpYWxSZW5kZXIpO1xuICAgIGNlbGxTb2NrZXQuY29ubmVjdCgpO1xufSk7XG5cbi8vIFRFU1RJTkc7IFJFTU9WRVxuY29uc29sZS5sb2coJ01haW4gbW9kdWxlIGxvYWRlZCcpO1xuIiwiKGZ1bmN0aW9uIChnbG9iYWwsIGZhY3RvcnkpIHtcbiAgICB0eXBlb2YgZXhwb3J0cyA9PT0gJ29iamVjdCcgJiYgdHlwZW9mIG1vZHVsZSAhPT0gJ3VuZGVmaW5lZCcgPyBmYWN0b3J5KGV4cG9ydHMpIDpcbiAgICB0eXBlb2YgZGVmaW5lID09PSAnZnVuY3Rpb24nICYmIGRlZmluZS5hbWQgPyBkZWZpbmUoWydleHBvcnRzJ10sIGZhY3RvcnkpIDpcbiAgICAoZ2xvYmFsID0gZ2xvYmFsIHx8IHNlbGYsIGZhY3RvcnkoZ2xvYmFsLm1hcXVldHRlID0ge30pKTtcbn0odGhpcywgZnVuY3Rpb24gKGV4cG9ydHMpIHsgJ3VzZSBzdHJpY3QnO1xuXG4gICAgLyogdHNsaW50OmRpc2FibGUgbm8taHR0cC1zdHJpbmcgKi9cclxuICAgIHZhciBOQU1FU1BBQ0VfVzMgPSAnaHR0cDovL3d3dy53My5vcmcvJztcclxuICAgIC8qIHRzbGludDplbmFibGUgbm8taHR0cC1zdHJpbmcgKi9cclxuICAgIHZhciBOQU1FU1BBQ0VfU1ZHID0gTkFNRVNQQUNFX1czICsgXCIyMDAwL3N2Z1wiO1xyXG4gICAgdmFyIE5BTUVTUEFDRV9YTElOSyA9IE5BTUVTUEFDRV9XMyArIFwiMTk5OS94bGlua1wiO1xyXG4gICAgdmFyIGVtcHR5QXJyYXkgPSBbXTtcclxuICAgIHZhciBleHRlbmQgPSBmdW5jdGlvbiAoYmFzZSwgb3ZlcnJpZGVzKSB7XHJcbiAgICAgICAgdmFyIHJlc3VsdCA9IHt9O1xyXG4gICAgICAgIE9iamVjdC5rZXlzKGJhc2UpLmZvckVhY2goZnVuY3Rpb24gKGtleSkge1xyXG4gICAgICAgICAgICByZXN1bHRba2V5XSA9IGJhc2Vba2V5XTtcclxuICAgICAgICB9KTtcclxuICAgICAgICBpZiAob3ZlcnJpZGVzKSB7XHJcbiAgICAgICAgICAgIE9iamVjdC5rZXlzKG92ZXJyaWRlcykuZm9yRWFjaChmdW5jdGlvbiAoa2V5KSB7XHJcbiAgICAgICAgICAgICAgICByZXN1bHRba2V5XSA9IG92ZXJyaWRlc1trZXldO1xyXG4gICAgICAgICAgICB9KTtcclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuIHJlc3VsdDtcclxuICAgIH07XHJcbiAgICB2YXIgc2FtZSA9IGZ1bmN0aW9uICh2bm9kZTEsIHZub2RlMikge1xyXG4gICAgICAgIGlmICh2bm9kZTEudm5vZGVTZWxlY3RvciAhPT0gdm5vZGUyLnZub2RlU2VsZWN0b3IpIHtcclxuICAgICAgICAgICAgcmV0dXJuIGZhbHNlO1xyXG4gICAgICAgIH1cclxuICAgICAgICBpZiAodm5vZGUxLnByb3BlcnRpZXMgJiYgdm5vZGUyLnByb3BlcnRpZXMpIHtcclxuICAgICAgICAgICAgaWYgKHZub2RlMS5wcm9wZXJ0aWVzLmtleSAhPT0gdm5vZGUyLnByb3BlcnRpZXMua2V5KSB7XHJcbiAgICAgICAgICAgICAgICByZXR1cm4gZmFsc2U7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgcmV0dXJuIHZub2RlMS5wcm9wZXJ0aWVzLmJpbmQgPT09IHZub2RlMi5wcm9wZXJ0aWVzLmJpbmQ7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHJldHVybiAhdm5vZGUxLnByb3BlcnRpZXMgJiYgIXZub2RlMi5wcm9wZXJ0aWVzO1xyXG4gICAgfTtcclxuICAgIHZhciBjaGVja1N0eWxlVmFsdWUgPSBmdW5jdGlvbiAoc3R5bGVWYWx1ZSkge1xyXG4gICAgICAgIGlmICh0eXBlb2Ygc3R5bGVWYWx1ZSAhPT0gJ3N0cmluZycpIHtcclxuICAgICAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdTdHlsZSB2YWx1ZXMgbXVzdCBiZSBzdHJpbmdzJyk7XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIHZhciBmaW5kSW5kZXhPZkNoaWxkID0gZnVuY3Rpb24gKGNoaWxkcmVuLCBzYW1lQXMsIHN0YXJ0KSB7XHJcbiAgICAgICAgaWYgKHNhbWVBcy52bm9kZVNlbGVjdG9yICE9PSAnJykge1xyXG4gICAgICAgICAgICAvLyBOZXZlciBzY2FuIGZvciB0ZXh0LW5vZGVzXHJcbiAgICAgICAgICAgIGZvciAodmFyIGkgPSBzdGFydDsgaSA8IGNoaWxkcmVuLmxlbmd0aDsgaSsrKSB7XHJcbiAgICAgICAgICAgICAgICBpZiAoc2FtZShjaGlsZHJlbltpXSwgc2FtZUFzKSkge1xyXG4gICAgICAgICAgICAgICAgICAgIHJldHVybiBpO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHJldHVybiAtMTtcclxuICAgIH07XHJcbiAgICB2YXIgY2hlY2tEaXN0aW5ndWlzaGFibGUgPSBmdW5jdGlvbiAoY2hpbGROb2RlcywgaW5kZXhUb0NoZWNrLCBwYXJlbnRWTm9kZSwgb3BlcmF0aW9uKSB7XHJcbiAgICAgICAgdmFyIGNoaWxkTm9kZSA9IGNoaWxkTm9kZXNbaW5kZXhUb0NoZWNrXTtcclxuICAgICAgICBpZiAoY2hpbGROb2RlLnZub2RlU2VsZWN0b3IgPT09ICcnKSB7XHJcbiAgICAgICAgICAgIHJldHVybjsgLy8gVGV4dCBub2RlcyBuZWVkIG5vdCBiZSBkaXN0aW5ndWlzaGFibGVcclxuICAgICAgICB9XHJcbiAgICAgICAgdmFyIHByb3BlcnRpZXMgPSBjaGlsZE5vZGUucHJvcGVydGllcztcclxuICAgICAgICB2YXIga2V5ID0gcHJvcGVydGllcyA/IChwcm9wZXJ0aWVzLmtleSA9PT0gdW5kZWZpbmVkID8gcHJvcGVydGllcy5iaW5kIDogcHJvcGVydGllcy5rZXkpIDogdW5kZWZpbmVkO1xyXG4gICAgICAgIGlmICgha2V5KSB7IC8vIEEga2V5IGlzIGp1c3QgYXNzdW1lZCB0byBiZSB1bmlxdWVcclxuICAgICAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCBjaGlsZE5vZGVzLmxlbmd0aDsgaSsrKSB7XHJcbiAgICAgICAgICAgICAgICBpZiAoaSAhPT0gaW5kZXhUb0NoZWNrKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgdmFyIG5vZGUgPSBjaGlsZE5vZGVzW2ldO1xyXG4gICAgICAgICAgICAgICAgICAgIGlmIChzYW1lKG5vZGUsIGNoaWxkTm9kZSkpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgdGhyb3cgbmV3IEVycm9yKHBhcmVudFZOb2RlLnZub2RlU2VsZWN0b3IgKyBcIiBoYWQgYSBcIiArIGNoaWxkTm9kZS52bm9kZVNlbGVjdG9yICsgXCIgY2hpbGQgXCIgKyAob3BlcmF0aW9uID09PSAnYWRkZWQnID8gb3BlcmF0aW9uIDogJ3JlbW92ZWQnKSArIFwiLCBidXQgdGhlcmUgaXMgbm93IG1vcmUgdGhhbiBvbmUuIFlvdSBtdXN0IGFkZCB1bmlxdWUga2V5IHByb3BlcnRpZXMgdG8gbWFrZSB0aGVtIGRpc3Rpbmd1aXNoYWJsZS5cIik7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIHZhciBub2RlQWRkZWQgPSBmdW5jdGlvbiAodk5vZGUpIHtcclxuICAgICAgICBpZiAodk5vZGUucHJvcGVydGllcykge1xyXG4gICAgICAgICAgICB2YXIgZW50ZXJBbmltYXRpb24gPSB2Tm9kZS5wcm9wZXJ0aWVzLmVudGVyQW5pbWF0aW9uO1xyXG4gICAgICAgICAgICBpZiAoZW50ZXJBbmltYXRpb24pIHtcclxuICAgICAgICAgICAgICAgIGVudGVyQW5pbWF0aW9uKHZOb2RlLmRvbU5vZGUsIHZOb2RlLnByb3BlcnRpZXMpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIHZhciByZW1vdmVkTm9kZXMgPSBbXTtcclxuICAgIHZhciByZXF1ZXN0ZWRJZGxlQ2FsbGJhY2sgPSBmYWxzZTtcclxuICAgIHZhciB2aXNpdFJlbW92ZWROb2RlID0gZnVuY3Rpb24gKG5vZGUpIHtcclxuICAgICAgICAobm9kZS5jaGlsZHJlbiB8fCBbXSkuZm9yRWFjaCh2aXNpdFJlbW92ZWROb2RlKTtcclxuICAgICAgICBpZiAobm9kZS5wcm9wZXJ0aWVzICYmIG5vZGUucHJvcGVydGllcy5hZnRlclJlbW92ZWQpIHtcclxuICAgICAgICAgICAgbm9kZS5wcm9wZXJ0aWVzLmFmdGVyUmVtb3ZlZC5hcHBseShub2RlLnByb3BlcnRpZXMuYmluZCB8fCBub2RlLnByb3BlcnRpZXMsIFtub2RlLmRvbU5vZGVdKTtcclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgdmFyIHByb2Nlc3NQZW5kaW5nTm9kZVJlbW92YWxzID0gZnVuY3Rpb24gKCkge1xyXG4gICAgICAgIHJlcXVlc3RlZElkbGVDYWxsYmFjayA9IGZhbHNlO1xyXG4gICAgICAgIHJlbW92ZWROb2Rlcy5mb3JFYWNoKHZpc2l0UmVtb3ZlZE5vZGUpO1xyXG4gICAgICAgIHJlbW92ZWROb2Rlcy5sZW5ndGggPSAwO1xyXG4gICAgfTtcclxuICAgIHZhciBzY2hlZHVsZU5vZGVSZW1vdmFsID0gZnVuY3Rpb24gKHZOb2RlKSB7XHJcbiAgICAgICAgcmVtb3ZlZE5vZGVzLnB1c2godk5vZGUpO1xyXG4gICAgICAgIGlmICghcmVxdWVzdGVkSWRsZUNhbGxiYWNrKSB7XHJcbiAgICAgICAgICAgIHJlcXVlc3RlZElkbGVDYWxsYmFjayA9IHRydWU7XHJcbiAgICAgICAgICAgIGlmICh0eXBlb2Ygd2luZG93ICE9PSAndW5kZWZpbmVkJyAmJiAncmVxdWVzdElkbGVDYWxsYmFjaycgaW4gd2luZG93KSB7XHJcbiAgICAgICAgICAgICAgICB3aW5kb3cucmVxdWVzdElkbGVDYWxsYmFjayhwcm9jZXNzUGVuZGluZ05vZGVSZW1vdmFscywgeyB0aW1lb3V0OiAxNiB9KTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgIHNldFRpbWVvdXQocHJvY2Vzc1BlbmRpbmdOb2RlUmVtb3ZhbHMsIDE2KTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgIH07XHJcbiAgICB2YXIgbm9kZVRvUmVtb3ZlID0gZnVuY3Rpb24gKHZOb2RlKSB7XHJcbiAgICAgICAgdmFyIGRvbU5vZGUgPSB2Tm9kZS5kb21Ob2RlO1xyXG4gICAgICAgIGlmICh2Tm9kZS5wcm9wZXJ0aWVzKSB7XHJcbiAgICAgICAgICAgIHZhciBleGl0QW5pbWF0aW9uID0gdk5vZGUucHJvcGVydGllcy5leGl0QW5pbWF0aW9uO1xyXG4gICAgICAgICAgICBpZiAoZXhpdEFuaW1hdGlvbikge1xyXG4gICAgICAgICAgICAgICAgZG9tTm9kZS5zdHlsZS5wb2ludGVyRXZlbnRzID0gJ25vbmUnO1xyXG4gICAgICAgICAgICAgICAgdmFyIHJlbW92ZURvbU5vZGUgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKGRvbU5vZGUucGFyZW50Tm9kZSkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBkb21Ob2RlLnBhcmVudE5vZGUucmVtb3ZlQ2hpbGQoZG9tTm9kZSk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHNjaGVkdWxlTm9kZVJlbW92YWwodk5vZGUpO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIH07XHJcbiAgICAgICAgICAgICAgICBleGl0QW5pbWF0aW9uKGRvbU5vZGUsIHJlbW92ZURvbU5vZGUsIHZOb2RlLnByb3BlcnRpZXMpO1xyXG4gICAgICAgICAgICAgICAgcmV0dXJuO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGlmIChkb21Ob2RlLnBhcmVudE5vZGUpIHtcclxuICAgICAgICAgICAgZG9tTm9kZS5wYXJlbnROb2RlLnJlbW92ZUNoaWxkKGRvbU5vZGUpO1xyXG4gICAgICAgICAgICBzY2hlZHVsZU5vZGVSZW1vdmFsKHZOb2RlKTtcclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgdmFyIHNldFByb3BlcnRpZXMgPSBmdW5jdGlvbiAoZG9tTm9kZSwgcHJvcGVydGllcywgcHJvamVjdGlvbk9wdGlvbnMpIHtcclxuICAgICAgICBpZiAoIXByb3BlcnRpZXMpIHtcclxuICAgICAgICAgICAgcmV0dXJuO1xyXG4gICAgICAgIH1cclxuICAgICAgICB2YXIgZXZlbnRIYW5kbGVySW50ZXJjZXB0b3IgPSBwcm9qZWN0aW9uT3B0aW9ucy5ldmVudEhhbmRsZXJJbnRlcmNlcHRvcjtcclxuICAgICAgICB2YXIgcHJvcE5hbWVzID0gT2JqZWN0LmtleXMocHJvcGVydGllcyk7XHJcbiAgICAgICAgdmFyIHByb3BDb3VudCA9IHByb3BOYW1lcy5sZW5ndGg7XHJcbiAgICAgICAgdmFyIF9sb29wXzEgPSBmdW5jdGlvbiAoaSkge1xyXG4gICAgICAgICAgICB2YXIgcHJvcE5hbWUgPSBwcm9wTmFtZXNbaV07XHJcbiAgICAgICAgICAgIHZhciBwcm9wVmFsdWUgPSBwcm9wZXJ0aWVzW3Byb3BOYW1lXTtcclxuICAgICAgICAgICAgaWYgKHByb3BOYW1lID09PSAnY2xhc3NOYW1lJykge1xyXG4gICAgICAgICAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdQcm9wZXJ0eSBcImNsYXNzTmFtZVwiIGlzIG5vdCBzdXBwb3J0ZWQsIHVzZSBcImNsYXNzXCIuJyk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgZWxzZSBpZiAocHJvcE5hbWUgPT09ICdjbGFzcycpIHtcclxuICAgICAgICAgICAgICAgIHRvZ2dsZUNsYXNzZXMoZG9tTm9kZSwgcHJvcFZhbHVlLCB0cnVlKTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBlbHNlIGlmIChwcm9wTmFtZSA9PT0gJ2NsYXNzZXMnKSB7XHJcbiAgICAgICAgICAgICAgICAvLyBvYmplY3Qgd2l0aCBzdHJpbmcga2V5cyBhbmQgYm9vbGVhbiB2YWx1ZXNcclxuICAgICAgICAgICAgICAgIHZhciBjbGFzc05hbWVzID0gT2JqZWN0LmtleXMocHJvcFZhbHVlKTtcclxuICAgICAgICAgICAgICAgIHZhciBjbGFzc05hbWVDb3VudCA9IGNsYXNzTmFtZXMubGVuZ3RoO1xyXG4gICAgICAgICAgICAgICAgZm9yICh2YXIgaiA9IDA7IGogPCBjbGFzc05hbWVDb3VudDsgaisrKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgdmFyIGNsYXNzTmFtZSA9IGNsYXNzTmFtZXNbal07XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKHByb3BWYWx1ZVtjbGFzc05hbWVdKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGRvbU5vZGUuY2xhc3NMaXN0LmFkZChjbGFzc05hbWUpO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBlbHNlIGlmIChwcm9wTmFtZSA9PT0gJ3N0eWxlcycpIHtcclxuICAgICAgICAgICAgICAgIC8vIG9iamVjdCB3aXRoIHN0cmluZyBrZXlzIGFuZCBzdHJpbmcgKCEpIHZhbHVlc1xyXG4gICAgICAgICAgICAgICAgdmFyIHN0eWxlTmFtZXMgPSBPYmplY3Qua2V5cyhwcm9wVmFsdWUpO1xyXG4gICAgICAgICAgICAgICAgdmFyIHN0eWxlQ291bnQgPSBzdHlsZU5hbWVzLmxlbmd0aDtcclxuICAgICAgICAgICAgICAgIGZvciAodmFyIGogPSAwOyBqIDwgc3R5bGVDb3VudDsgaisrKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgdmFyIHN0eWxlTmFtZSA9IHN0eWxlTmFtZXNbal07XHJcbiAgICAgICAgICAgICAgICAgICAgdmFyIHN0eWxlVmFsdWUgPSBwcm9wVmFsdWVbc3R5bGVOYW1lXTtcclxuICAgICAgICAgICAgICAgICAgICBpZiAoc3R5bGVWYWx1ZSkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBjaGVja1N0eWxlVmFsdWUoc3R5bGVWYWx1ZSk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHByb2plY3Rpb25PcHRpb25zLnN0eWxlQXBwbHllcihkb21Ob2RlLCBzdHlsZU5hbWUsIHN0eWxlVmFsdWUpO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBlbHNlIGlmIChwcm9wTmFtZSAhPT0gJ2tleScgJiYgcHJvcFZhbHVlICE9PSBudWxsICYmIHByb3BWYWx1ZSAhPT0gdW5kZWZpbmVkKSB7XHJcbiAgICAgICAgICAgICAgICB2YXIgdHlwZSA9IHR5cGVvZiBwcm9wVmFsdWU7XHJcbiAgICAgICAgICAgICAgICBpZiAodHlwZSA9PT0gJ2Z1bmN0aW9uJykge1xyXG4gICAgICAgICAgICAgICAgICAgIGlmIChwcm9wTmFtZS5sYXN0SW5kZXhPZignb24nLCAwKSA9PT0gMCkgeyAvLyBsYXN0SW5kZXhPZigsMCk9PT0wIC0+IHN0YXJ0c1dpdGhcclxuICAgICAgICAgICAgICAgICAgICAgICAgaWYgKGV2ZW50SGFuZGxlckludGVyY2VwdG9yKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBwcm9wVmFsdWUgPSBldmVudEhhbmRsZXJJbnRlcmNlcHRvcihwcm9wTmFtZSwgcHJvcFZhbHVlLCBkb21Ob2RlLCBwcm9wZXJ0aWVzKTsgLy8gaW50ZXJjZXB0IGV2ZW50aGFuZGxlcnNcclxuICAgICAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgICAgICBpZiAocHJvcE5hbWUgPT09ICdvbmlucHV0Jykge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgLyogdHNsaW50OmRpc2FibGUgbm8tdGhpcy1rZXl3b3JkIG5vLWludmFsaWQtdGhpcyBvbmx5LWFycm93LWZ1bmN0aW9ucyBuby12b2lkLWV4cHJlc3Npb24gKi9cclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIChmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgLy8gcmVjb3JkIHRoZSBldnQudGFyZ2V0LnZhbHVlLCBiZWNhdXNlIElFIGFuZCBFZGdlIHNvbWV0aW1lcyBkbyBhIHJlcXVlc3RBbmltYXRpb25GcmFtZSBiZXR3ZWVuIGNoYW5naW5nIHZhbHVlIGFuZCBydW5uaW5nIG9uaW5wdXRcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB2YXIgb2xkUHJvcFZhbHVlID0gcHJvcFZhbHVlO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHByb3BWYWx1ZSA9IGZ1bmN0aW9uIChldnQpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgb2xkUHJvcFZhbHVlLmFwcGx5KHRoaXMsIFtldnRdKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZXZ0LnRhcmdldFsnb25pbnB1dC12YWx1ZSddID0gZXZ0LnRhcmdldC52YWx1ZTsgLy8gbWF5IGJlIEhUTUxUZXh0QXJlYUVsZW1lbnQgYXMgd2VsbFxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH07XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9KCkpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgLyogdHNsaW50OmVuYWJsZSAqL1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGRvbU5vZGVbcHJvcE5hbWVdID0gcHJvcFZhbHVlO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIGVsc2UgaWYgKHByb2plY3Rpb25PcHRpb25zLm5hbWVzcGFjZSA9PT0gTkFNRVNQQUNFX1NWRykge1xyXG4gICAgICAgICAgICAgICAgICAgIGlmIChwcm9wTmFtZSA9PT0gJ2hyZWYnKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGRvbU5vZGUuc2V0QXR0cmlidXRlTlMoTkFNRVNQQUNFX1hMSU5LLCBwcm9wTmFtZSwgcHJvcFZhbHVlKTtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIC8vIGFsbCBTVkcgYXR0cmlidXRlcyBhcmUgcmVhZC1vbmx5IGluIERPTSwgc28uLi5cclxuICAgICAgICAgICAgICAgICAgICAgICAgZG9tTm9kZS5zZXRBdHRyaWJ1dGUocHJvcE5hbWUsIHByb3BWYWx1ZSk7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgZWxzZSBpZiAodHlwZSA9PT0gJ3N0cmluZycgJiYgcHJvcE5hbWUgIT09ICd2YWx1ZScgJiYgcHJvcE5hbWUgIT09ICdpbm5lckhUTUwnKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgZG9tTm9kZS5zZXRBdHRyaWJ1dGUocHJvcE5hbWUsIHByb3BWYWx1ZSk7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgICBkb21Ob2RlW3Byb3BOYW1lXSA9IHByb3BWYWx1ZTtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH07XHJcbiAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCBwcm9wQ291bnQ7IGkrKykge1xyXG4gICAgICAgICAgICBfbG9vcF8xKGkpO1xyXG4gICAgICAgIH1cclxuICAgIH07XHJcbiAgICB2YXIgYWRkQ2hpbGRyZW4gPSBmdW5jdGlvbiAoZG9tTm9kZSwgY2hpbGRyZW4sIHByb2plY3Rpb25PcHRpb25zKSB7XHJcbiAgICAgICAgaWYgKCFjaGlsZHJlbikge1xyXG4gICAgICAgICAgICByZXR1cm47XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGZvciAodmFyIF9pID0gMCwgY2hpbGRyZW5fMSA9IGNoaWxkcmVuOyBfaSA8IGNoaWxkcmVuXzEubGVuZ3RoOyBfaSsrKSB7XHJcbiAgICAgICAgICAgIHZhciBjaGlsZCA9IGNoaWxkcmVuXzFbX2ldO1xyXG4gICAgICAgICAgICBjcmVhdGVEb20oY2hpbGQsIGRvbU5vZGUsIHVuZGVmaW5lZCwgcHJvamVjdGlvbk9wdGlvbnMpO1xyXG4gICAgICAgIH1cclxuICAgIH07XHJcbiAgICB2YXIgaW5pdFByb3BlcnRpZXNBbmRDaGlsZHJlbiA9IGZ1bmN0aW9uIChkb21Ob2RlLCB2bm9kZSwgcHJvamVjdGlvbk9wdGlvbnMpIHtcclxuICAgICAgICBhZGRDaGlsZHJlbihkb21Ob2RlLCB2bm9kZS5jaGlsZHJlbiwgcHJvamVjdGlvbk9wdGlvbnMpOyAvLyBjaGlsZHJlbiBiZWZvcmUgcHJvcGVydGllcywgbmVlZGVkIGZvciB2YWx1ZSBwcm9wZXJ0eSBvZiA8c2VsZWN0Pi5cclxuICAgICAgICBpZiAodm5vZGUudGV4dCkge1xyXG4gICAgICAgICAgICBkb21Ob2RlLnRleHRDb250ZW50ID0gdm5vZGUudGV4dDtcclxuICAgICAgICB9XHJcbiAgICAgICAgc2V0UHJvcGVydGllcyhkb21Ob2RlLCB2bm9kZS5wcm9wZXJ0aWVzLCBwcm9qZWN0aW9uT3B0aW9ucyk7XHJcbiAgICAgICAgaWYgKHZub2RlLnByb3BlcnRpZXMgJiYgdm5vZGUucHJvcGVydGllcy5hZnRlckNyZWF0ZSkge1xyXG4gICAgICAgICAgICB2bm9kZS5wcm9wZXJ0aWVzLmFmdGVyQ3JlYXRlLmFwcGx5KHZub2RlLnByb3BlcnRpZXMuYmluZCB8fCB2bm9kZS5wcm9wZXJ0aWVzLCBbZG9tTm9kZSwgcHJvamVjdGlvbk9wdGlvbnMsIHZub2RlLnZub2RlU2VsZWN0b3IsIHZub2RlLnByb3BlcnRpZXMsIHZub2RlLmNoaWxkcmVuXSk7XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIHZhciBjcmVhdGVEb20gPSBmdW5jdGlvbiAodm5vZGUsIHBhcmVudE5vZGUsIGluc2VydEJlZm9yZSwgcHJvamVjdGlvbk9wdGlvbnMpIHtcclxuICAgICAgICB2YXIgZG9tTm9kZTtcclxuICAgICAgICB2YXIgc3RhcnQgPSAwO1xyXG4gICAgICAgIHZhciB2bm9kZVNlbGVjdG9yID0gdm5vZGUudm5vZGVTZWxlY3RvcjtcclxuICAgICAgICB2YXIgZG9jID0gcGFyZW50Tm9kZS5vd25lckRvY3VtZW50O1xyXG4gICAgICAgIGlmICh2bm9kZVNlbGVjdG9yID09PSAnJykge1xyXG4gICAgICAgICAgICBkb21Ob2RlID0gdm5vZGUuZG9tTm9kZSA9IGRvYy5jcmVhdGVUZXh0Tm9kZSh2bm9kZS50ZXh0KTtcclxuICAgICAgICAgICAgaWYgKGluc2VydEJlZm9yZSAhPT0gdW5kZWZpbmVkKSB7XHJcbiAgICAgICAgICAgICAgICBwYXJlbnROb2RlLmluc2VydEJlZm9yZShkb21Ob2RlLCBpbnNlcnRCZWZvcmUpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgcGFyZW50Tm9kZS5hcHBlbmRDaGlsZChkb21Ob2RlKTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPD0gdm5vZGVTZWxlY3Rvci5sZW5ndGg7ICsraSkge1xyXG4gICAgICAgICAgICAgICAgdmFyIGMgPSB2bm9kZVNlbGVjdG9yLmNoYXJBdChpKTtcclxuICAgICAgICAgICAgICAgIGlmIChpID09PSB2bm9kZVNlbGVjdG9yLmxlbmd0aCB8fCBjID09PSAnLicgfHwgYyA9PT0gJyMnKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgdmFyIHR5cGUgPSB2bm9kZVNlbGVjdG9yLmNoYXJBdChzdGFydCAtIDEpO1xyXG4gICAgICAgICAgICAgICAgICAgIHZhciBmb3VuZCA9IHZub2RlU2VsZWN0b3Iuc2xpY2Uoc3RhcnQsIGkpO1xyXG4gICAgICAgICAgICAgICAgICAgIGlmICh0eXBlID09PSAnLicpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgZG9tTm9kZS5jbGFzc0xpc3QuYWRkKGZvdW5kKTtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgZWxzZSBpZiAodHlwZSA9PT0gJyMnKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGRvbU5vZGUuaWQgPSBmb3VuZDtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGlmIChmb3VuZCA9PT0gJ3N2ZycpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHByb2plY3Rpb25PcHRpb25zID0gZXh0ZW5kKHByb2plY3Rpb25PcHRpb25zLCB7IG5hbWVzcGFjZTogTkFNRVNQQUNFX1NWRyB9KTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgICAgICBpZiAocHJvamVjdGlvbk9wdGlvbnMubmFtZXNwYWNlICE9PSB1bmRlZmluZWQpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGRvbU5vZGUgPSB2bm9kZS5kb21Ob2RlID0gZG9jLmNyZWF0ZUVsZW1lbnROUyhwcm9qZWN0aW9uT3B0aW9ucy5uYW1lc3BhY2UsIGZvdW5kKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGRvbU5vZGUgPSB2bm9kZS5kb21Ob2RlID0gKHZub2RlLmRvbU5vZGUgfHwgZG9jLmNyZWF0ZUVsZW1lbnQoZm91bmQpKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmIChmb3VuZCA9PT0gJ2lucHV0JyAmJiB2bm9kZS5wcm9wZXJ0aWVzICYmIHZub2RlLnByb3BlcnRpZXMudHlwZSAhPT0gdW5kZWZpbmVkKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgLy8gSUU4IGFuZCBvbGRlciBkb24ndCBzdXBwb3J0IHNldHRpbmcgaW5wdXQgdHlwZSBhZnRlciB0aGUgRE9NIE5vZGUgaGFzIGJlZW4gYWRkZWQgdG8gdGhlIGRvY3VtZW50XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZG9tTm9kZS5zZXRBdHRyaWJ1dGUoJ3R5cGUnLCB2bm9kZS5wcm9wZXJ0aWVzLnR5cGUpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGlmIChpbnNlcnRCZWZvcmUgIT09IHVuZGVmaW5lZCkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgcGFyZW50Tm9kZS5pbnNlcnRCZWZvcmUoZG9tTm9kZSwgaW5zZXJ0QmVmb3JlKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgICAgICBlbHNlIGlmIChkb21Ob2RlLnBhcmVudE5vZGUgIT09IHBhcmVudE5vZGUpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHBhcmVudE5vZGUuYXBwZW5kQ2hpbGQoZG9tTm9kZSk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgc3RhcnQgPSBpICsgMTtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBpbml0UHJvcGVydGllc0FuZENoaWxkcmVuKGRvbU5vZGUsIHZub2RlLCBwcm9qZWN0aW9uT3B0aW9ucyk7XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIHZhciB1cGRhdGVEb207XHJcbiAgICAvKipcclxuICAgICAqIEFkZHMgb3IgcmVtb3ZlcyBjbGFzc2VzIGZyb20gYW4gRWxlbWVudFxyXG4gICAgICogQHBhcmFtIGRvbU5vZGUgdGhlIGVsZW1lbnRcclxuICAgICAqIEBwYXJhbSBjbGFzc2VzIGEgc3RyaW5nIHNlcGFyYXRlZCBsaXN0IG9mIGNsYXNzZXNcclxuICAgICAqIEBwYXJhbSBvbiB0cnVlIG1lYW5zIGFkZCBjbGFzc2VzLCBmYWxzZSBtZWFucyByZW1vdmVcclxuICAgICAqL1xyXG4gICAgdmFyIHRvZ2dsZUNsYXNzZXMgPSBmdW5jdGlvbiAoZG9tTm9kZSwgY2xhc3Nlcywgb24pIHtcclxuICAgICAgICBpZiAoIWNsYXNzZXMpIHtcclxuICAgICAgICAgICAgcmV0dXJuO1xyXG4gICAgICAgIH1cclxuICAgICAgICBjbGFzc2VzLnNwbGl0KCcgJykuZm9yRWFjaChmdW5jdGlvbiAoY2xhc3NUb1RvZ2dsZSkge1xyXG4gICAgICAgICAgICBpZiAoY2xhc3NUb1RvZ2dsZSkge1xyXG4gICAgICAgICAgICAgICAgZG9tTm9kZS5jbGFzc0xpc3QudG9nZ2xlKGNsYXNzVG9Ub2dnbGUsIG9uKTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH0pO1xyXG4gICAgfTtcclxuICAgIHZhciB1cGRhdGVQcm9wZXJ0aWVzID0gZnVuY3Rpb24gKGRvbU5vZGUsIHByZXZpb3VzUHJvcGVydGllcywgcHJvcGVydGllcywgcHJvamVjdGlvbk9wdGlvbnMpIHtcclxuICAgICAgICBpZiAoIXByb3BlcnRpZXMpIHtcclxuICAgICAgICAgICAgcmV0dXJuO1xyXG4gICAgICAgIH1cclxuICAgICAgICB2YXIgcHJvcGVydGllc1VwZGF0ZWQgPSBmYWxzZTtcclxuICAgICAgICB2YXIgcHJvcE5hbWVzID0gT2JqZWN0LmtleXMocHJvcGVydGllcyk7XHJcbiAgICAgICAgdmFyIHByb3BDb3VudCA9IHByb3BOYW1lcy5sZW5ndGg7XHJcbiAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCBwcm9wQ291bnQ7IGkrKykge1xyXG4gICAgICAgICAgICB2YXIgcHJvcE5hbWUgPSBwcm9wTmFtZXNbaV07XHJcbiAgICAgICAgICAgIC8vIGFzc3VtaW5nIHRoYXQgcHJvcGVydGllcyB3aWxsIGJlIG51bGxpZmllZCBpbnN0ZWFkIG9mIG1pc3NpbmcgaXMgYnkgZGVzaWduXHJcbiAgICAgICAgICAgIHZhciBwcm9wVmFsdWUgPSBwcm9wZXJ0aWVzW3Byb3BOYW1lXTtcclxuICAgICAgICAgICAgdmFyIHByZXZpb3VzVmFsdWUgPSBwcmV2aW91c1Byb3BlcnRpZXNbcHJvcE5hbWVdO1xyXG4gICAgICAgICAgICBpZiAocHJvcE5hbWUgPT09ICdjbGFzcycpIHtcclxuICAgICAgICAgICAgICAgIGlmIChwcmV2aW91c1ZhbHVlICE9PSBwcm9wVmFsdWUpIHtcclxuICAgICAgICAgICAgICAgICAgICB0b2dnbGVDbGFzc2VzKGRvbU5vZGUsIHByZXZpb3VzVmFsdWUsIGZhbHNlKTtcclxuICAgICAgICAgICAgICAgICAgICB0b2dnbGVDbGFzc2VzKGRvbU5vZGUsIHByb3BWYWx1ZSwgdHJ1ZSk7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgZWxzZSBpZiAocHJvcE5hbWUgPT09ICdjbGFzc2VzJykge1xyXG4gICAgICAgICAgICAgICAgdmFyIGNsYXNzTGlzdCA9IGRvbU5vZGUuY2xhc3NMaXN0O1xyXG4gICAgICAgICAgICAgICAgdmFyIGNsYXNzTmFtZXMgPSBPYmplY3Qua2V5cyhwcm9wVmFsdWUpO1xyXG4gICAgICAgICAgICAgICAgdmFyIGNsYXNzTmFtZUNvdW50ID0gY2xhc3NOYW1lcy5sZW5ndGg7XHJcbiAgICAgICAgICAgICAgICBmb3IgKHZhciBqID0gMDsgaiA8IGNsYXNzTmFtZUNvdW50OyBqKyspIHtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgY2xhc3NOYW1lID0gY2xhc3NOYW1lc1tqXTtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgb24gPSAhIXByb3BWYWx1ZVtjbGFzc05hbWVdO1xyXG4gICAgICAgICAgICAgICAgICAgIHZhciBwcmV2aW91c09uID0gISFwcmV2aW91c1ZhbHVlW2NsYXNzTmFtZV07XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKG9uID09PSBwcmV2aW91c09uKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGNvbnRpbnVlO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICBwcm9wZXJ0aWVzVXBkYXRlZCA9IHRydWU7XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKG9uKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGNsYXNzTGlzdC5hZGQoY2xhc3NOYW1lKTtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGNsYXNzTGlzdC5yZW1vdmUoY2xhc3NOYW1lKTtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgZWxzZSBpZiAocHJvcE5hbWUgPT09ICdzdHlsZXMnKSB7XHJcbiAgICAgICAgICAgICAgICB2YXIgc3R5bGVOYW1lcyA9IE9iamVjdC5rZXlzKHByb3BWYWx1ZSk7XHJcbiAgICAgICAgICAgICAgICB2YXIgc3R5bGVDb3VudCA9IHN0eWxlTmFtZXMubGVuZ3RoO1xyXG4gICAgICAgICAgICAgICAgZm9yICh2YXIgaiA9IDA7IGogPCBzdHlsZUNvdW50OyBqKyspIHtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgc3R5bGVOYW1lID0gc3R5bGVOYW1lc1tqXTtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgbmV3U3R5bGVWYWx1ZSA9IHByb3BWYWx1ZVtzdHlsZU5hbWVdO1xyXG4gICAgICAgICAgICAgICAgICAgIHZhciBvbGRTdHlsZVZhbHVlID0gcHJldmlvdXNWYWx1ZVtzdHlsZU5hbWVdO1xyXG4gICAgICAgICAgICAgICAgICAgIGlmIChuZXdTdHlsZVZhbHVlID09PSBvbGRTdHlsZVZhbHVlKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGNvbnRpbnVlO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICBwcm9wZXJ0aWVzVXBkYXRlZCA9IHRydWU7XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKG5ld1N0eWxlVmFsdWUpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgY2hlY2tTdHlsZVZhbHVlKG5ld1N0eWxlVmFsdWUpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBwcm9qZWN0aW9uT3B0aW9ucy5zdHlsZUFwcGx5ZXIoZG9tTm9kZSwgc3R5bGVOYW1lLCBuZXdTdHlsZVZhbHVlKTtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHByb2plY3Rpb25PcHRpb25zLnN0eWxlQXBwbHllcihkb21Ob2RlLCBzdHlsZU5hbWUsICcnKTtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICBpZiAoIXByb3BWYWx1ZSAmJiB0eXBlb2YgcHJldmlvdXNWYWx1ZSA9PT0gJ3N0cmluZycpIHtcclxuICAgICAgICAgICAgICAgICAgICBwcm9wVmFsdWUgPSAnJztcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIGlmIChwcm9wTmFtZSA9PT0gJ3ZhbHVlJykgeyAvLyB2YWx1ZSBjYW4gYmUgbWFuaXB1bGF0ZWQgYnkgdGhlIHVzZXIgZGlyZWN0bHkgYW5kIHVzaW5nIGV2ZW50LnByZXZlbnREZWZhdWx0KCkgaXMgbm90IGFuIG9wdGlvblxyXG4gICAgICAgICAgICAgICAgICAgIHZhciBkb21WYWx1ZSA9IGRvbU5vZGVbcHJvcE5hbWVdO1xyXG4gICAgICAgICAgICAgICAgICAgIGlmIChkb21WYWx1ZSAhPT0gcHJvcFZhbHVlIC8vIFRoZSAndmFsdWUnIGluIHRoZSBET00gdHJlZSAhPT0gbmV3VmFsdWVcclxuICAgICAgICAgICAgICAgICAgICAgICAgJiYgKGRvbU5vZGVbJ29uaW5wdXQtdmFsdWUnXVxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgPyBkb21WYWx1ZSA9PT0gZG9tTm9kZVsnb25pbnB1dC12YWx1ZSddIC8vIElmIHRoZSBsYXN0IHJlcG9ydGVkIHZhbHVlIHRvICdvbmlucHV0JyBkb2VzIG5vdCBtYXRjaCBkb21WYWx1ZSwgZG8gbm90aGluZyBhbmQgd2FpdCBmb3Igb25pbnB1dFxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgOiBwcm9wVmFsdWUgIT09IHByZXZpb3VzVmFsdWUgLy8gT25seSB1cGRhdGUgdGhlIHZhbHVlIGlmIHRoZSB2ZG9tIGNoYW5nZWRcclxuICAgICAgICAgICAgICAgICAgICAgICAgKSkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAvLyBUaGUgZWRnZSBjYXNlcyBhcmUgZGVzY3JpYmVkIGluIHRoZSB0ZXN0c1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBkb21Ob2RlW3Byb3BOYW1lXSA9IHByb3BWYWx1ZTsgLy8gUmVzZXQgdGhlIHZhbHVlLCBldmVuIGlmIHRoZSB2aXJ0dWFsIERPTSBkaWQgbm90IGNoYW5nZVxyXG4gICAgICAgICAgICAgICAgICAgICAgICBkb21Ob2RlWydvbmlucHV0LXZhbHVlJ10gPSB1bmRlZmluZWQ7XHJcbiAgICAgICAgICAgICAgICAgICAgfSAvLyBlbHNlIGRvIG5vdCB1cGRhdGUgdGhlIGRvbU5vZGUsIG90aGVyd2lzZSB0aGUgY3Vyc29yIHBvc2l0aW9uIHdvdWxkIGJlIGNoYW5nZWRcclxuICAgICAgICAgICAgICAgICAgICBpZiAocHJvcFZhbHVlICE9PSBwcmV2aW91c1ZhbHVlKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHByb3BlcnRpZXNVcGRhdGVkID0gdHJ1ZTtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICBlbHNlIGlmIChwcm9wVmFsdWUgIT09IHByZXZpb3VzVmFsdWUpIHtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgdHlwZSA9IHR5cGVvZiBwcm9wVmFsdWU7XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKHR5cGUgIT09ICdmdW5jdGlvbicgfHwgIXByb2plY3Rpb25PcHRpb25zLmV2ZW50SGFuZGxlckludGVyY2VwdG9yKSB7IC8vIEZ1bmN0aW9uIHVwZGF0ZXMgYXJlIGV4cGVjdGVkIHRvIGJlIGhhbmRsZWQgYnkgdGhlIEV2ZW50SGFuZGxlckludGVyY2VwdG9yXHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGlmIChwcm9qZWN0aW9uT3B0aW9ucy5uYW1lc3BhY2UgPT09IE5BTUVTUEFDRV9TVkcpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmIChwcm9wTmFtZSA9PT0gJ2hyZWYnKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZG9tTm9kZS5zZXRBdHRyaWJ1dGVOUyhOQU1FU1BBQ0VfWExJTkssIHByb3BOYW1lLCBwcm9wVmFsdWUpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgLy8gYWxsIFNWRyBhdHRyaWJ1dGVzIGFyZSByZWFkLW9ubHkgaW4gRE9NLCBzby4uLlxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGRvbU5vZGUuc2V0QXR0cmlidXRlKHByb3BOYW1lLCBwcm9wVmFsdWUpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGVsc2UgaWYgKHR5cGUgPT09ICdzdHJpbmcnICYmIHByb3BOYW1lICE9PSAnaW5uZXJIVE1MJykge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgaWYgKHByb3BOYW1lID09PSAncm9sZScgJiYgcHJvcFZhbHVlID09PSAnJykge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGRvbU5vZGUucmVtb3ZlQXR0cmlidXRlKHByb3BOYW1lKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGRvbU5vZGUuc2V0QXR0cmlidXRlKHByb3BOYW1lLCBwcm9wVmFsdWUpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGVsc2UgaWYgKGRvbU5vZGVbcHJvcE5hbWVdICE9PSBwcm9wVmFsdWUpIHsgLy8gQ29tcGFyaXNvbiBpcyBoZXJlIGZvciBzaWRlLWVmZmVjdHMgaW4gRWRnZSB3aXRoIHNjcm9sbExlZnQgYW5kIHNjcm9sbFRvcFxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgZG9tTm9kZVtwcm9wTmFtZV0gPSBwcm9wVmFsdWU7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICAgICAgcHJvcGVydGllc1VwZGF0ZWQgPSB0cnVlO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgICAgICByZXR1cm4gcHJvcGVydGllc1VwZGF0ZWQ7XHJcbiAgICB9O1xyXG4gICAgdmFyIHVwZGF0ZUNoaWxkcmVuID0gZnVuY3Rpb24gKHZub2RlLCBkb21Ob2RlLCBvbGRDaGlsZHJlbiwgbmV3Q2hpbGRyZW4sIHByb2plY3Rpb25PcHRpb25zKSB7XHJcbiAgICAgICAgaWYgKG9sZENoaWxkcmVuID09PSBuZXdDaGlsZHJlbikge1xyXG4gICAgICAgICAgICByZXR1cm4gZmFsc2U7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIG9sZENoaWxkcmVuID0gb2xkQ2hpbGRyZW4gfHwgZW1wdHlBcnJheTtcclxuICAgICAgICBuZXdDaGlsZHJlbiA9IG5ld0NoaWxkcmVuIHx8IGVtcHR5QXJyYXk7XHJcbiAgICAgICAgdmFyIG9sZENoaWxkcmVuTGVuZ3RoID0gb2xkQ2hpbGRyZW4ubGVuZ3RoO1xyXG4gICAgICAgIHZhciBuZXdDaGlsZHJlbkxlbmd0aCA9IG5ld0NoaWxkcmVuLmxlbmd0aDtcclxuICAgICAgICB2YXIgb2xkSW5kZXggPSAwO1xyXG4gICAgICAgIHZhciBuZXdJbmRleCA9IDA7XHJcbiAgICAgICAgdmFyIGk7XHJcbiAgICAgICAgdmFyIHRleHRVcGRhdGVkID0gZmFsc2U7XHJcbiAgICAgICAgd2hpbGUgKG5ld0luZGV4IDwgbmV3Q2hpbGRyZW5MZW5ndGgpIHtcclxuICAgICAgICAgICAgdmFyIG9sZENoaWxkID0gKG9sZEluZGV4IDwgb2xkQ2hpbGRyZW5MZW5ndGgpID8gb2xkQ2hpbGRyZW5bb2xkSW5kZXhdIDogdW5kZWZpbmVkO1xyXG4gICAgICAgICAgICB2YXIgbmV3Q2hpbGQgPSBuZXdDaGlsZHJlbltuZXdJbmRleF07XHJcbiAgICAgICAgICAgIGlmIChvbGRDaGlsZCAhPT0gdW5kZWZpbmVkICYmIHNhbWUob2xkQ2hpbGQsIG5ld0NoaWxkKSkge1xyXG4gICAgICAgICAgICAgICAgdGV4dFVwZGF0ZWQgPSB1cGRhdGVEb20ob2xkQ2hpbGQsIG5ld0NoaWxkLCBwcm9qZWN0aW9uT3B0aW9ucykgfHwgdGV4dFVwZGF0ZWQ7XHJcbiAgICAgICAgICAgICAgICBvbGRJbmRleCsrO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgdmFyIGZpbmRPbGRJbmRleCA9IGZpbmRJbmRleE9mQ2hpbGQob2xkQ2hpbGRyZW4sIG5ld0NoaWxkLCBvbGRJbmRleCArIDEpO1xyXG4gICAgICAgICAgICAgICAgaWYgKGZpbmRPbGRJbmRleCA+PSAwKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgLy8gUmVtb3ZlIHByZWNlZGluZyBtaXNzaW5nIGNoaWxkcmVuXHJcbiAgICAgICAgICAgICAgICAgICAgZm9yIChpID0gb2xkSW5kZXg7IGkgPCBmaW5kT2xkSW5kZXg7IGkrKykge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBub2RlVG9SZW1vdmUob2xkQ2hpbGRyZW5baV0pO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBjaGVja0Rpc3Rpbmd1aXNoYWJsZShvbGRDaGlsZHJlbiwgaSwgdm5vZGUsICdyZW1vdmVkJyk7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgIHRleHRVcGRhdGVkID0gdXBkYXRlRG9tKG9sZENoaWxkcmVuW2ZpbmRPbGRJbmRleF0sIG5ld0NoaWxkLCBwcm9qZWN0aW9uT3B0aW9ucykgfHwgdGV4dFVwZGF0ZWQ7XHJcbiAgICAgICAgICAgICAgICAgICAgb2xkSW5kZXggPSBmaW5kT2xkSW5kZXggKyAxO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICAgICAgLy8gTmV3IGNoaWxkXHJcbiAgICAgICAgICAgICAgICAgICAgY3JlYXRlRG9tKG5ld0NoaWxkLCBkb21Ob2RlLCAob2xkSW5kZXggPCBvbGRDaGlsZHJlbkxlbmd0aCkgPyBvbGRDaGlsZHJlbltvbGRJbmRleF0uZG9tTm9kZSA6IHVuZGVmaW5lZCwgcHJvamVjdGlvbk9wdGlvbnMpO1xyXG4gICAgICAgICAgICAgICAgICAgIG5vZGVBZGRlZChuZXdDaGlsZCk7XHJcbiAgICAgICAgICAgICAgICAgICAgY2hlY2tEaXN0aW5ndWlzaGFibGUobmV3Q2hpbGRyZW4sIG5ld0luZGV4LCB2bm9kZSwgJ2FkZGVkJyk7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgbmV3SW5kZXgrKztcclxuICAgICAgICB9XHJcbiAgICAgICAgaWYgKG9sZENoaWxkcmVuTGVuZ3RoID4gb2xkSW5kZXgpIHtcclxuICAgICAgICAgICAgLy8gUmVtb3ZlIGNoaWxkIGZyYWdtZW50c1xyXG4gICAgICAgICAgICBmb3IgKGkgPSBvbGRJbmRleDsgaSA8IG9sZENoaWxkcmVuTGVuZ3RoOyBpKyspIHtcclxuICAgICAgICAgICAgICAgIG5vZGVUb1JlbW92ZShvbGRDaGlsZHJlbltpXSk7XHJcbiAgICAgICAgICAgICAgICBjaGVja0Rpc3Rpbmd1aXNoYWJsZShvbGRDaGlsZHJlbiwgaSwgdm5vZGUsICdyZW1vdmVkJyk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuIHRleHRVcGRhdGVkO1xyXG4gICAgfTtcclxuICAgIHVwZGF0ZURvbSA9IGZ1bmN0aW9uIChwcmV2aW91cywgdm5vZGUsIHByb2plY3Rpb25PcHRpb25zKSB7XHJcbiAgICAgICAgdmFyIGRvbU5vZGUgPSBwcmV2aW91cy5kb21Ob2RlO1xyXG4gICAgICAgIHZhciB0ZXh0VXBkYXRlZCA9IGZhbHNlO1xyXG4gICAgICAgIGlmIChwcmV2aW91cyA9PT0gdm5vZGUpIHtcclxuICAgICAgICAgICAgcmV0dXJuIGZhbHNlOyAvLyBCeSBjb250cmFjdCwgVk5vZGUgb2JqZWN0cyBtYXkgbm90IGJlIG1vZGlmaWVkIGFueW1vcmUgYWZ0ZXIgcGFzc2luZyB0aGVtIHRvIG1hcXVldHRlXHJcbiAgICAgICAgfVxyXG4gICAgICAgIHZhciB1cGRhdGVkID0gZmFsc2U7XHJcbiAgICAgICAgaWYgKHZub2RlLnZub2RlU2VsZWN0b3IgPT09ICcnKSB7XHJcbiAgICAgICAgICAgIGlmICh2bm9kZS50ZXh0ICE9PSBwcmV2aW91cy50ZXh0KSB7XHJcbiAgICAgICAgICAgICAgICB2YXIgbmV3VGV4dE5vZGUgPSBkb21Ob2RlLm93bmVyRG9jdW1lbnQuY3JlYXRlVGV4dE5vZGUodm5vZGUudGV4dCk7XHJcbiAgICAgICAgICAgICAgICBkb21Ob2RlLnBhcmVudE5vZGUucmVwbGFjZUNoaWxkKG5ld1RleHROb2RlLCBkb21Ob2RlKTtcclxuICAgICAgICAgICAgICAgIHZub2RlLmRvbU5vZGUgPSBuZXdUZXh0Tm9kZTtcclxuICAgICAgICAgICAgICAgIHRleHRVcGRhdGVkID0gdHJ1ZTtcclxuICAgICAgICAgICAgICAgIHJldHVybiB0ZXh0VXBkYXRlZDtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB2bm9kZS5kb21Ob2RlID0gZG9tTm9kZTtcclxuICAgICAgICB9XHJcbiAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgIGlmICh2bm9kZS52bm9kZVNlbGVjdG9yLmxhc3RJbmRleE9mKCdzdmcnLCAwKSA9PT0gMCkgeyAvLyBsYXN0SW5kZXhPZihuZWVkbGUsMCk9PT0wIG1lYW5zIFN0YXJ0c1dpdGhcclxuICAgICAgICAgICAgICAgIHByb2plY3Rpb25PcHRpb25zID0gZXh0ZW5kKHByb2plY3Rpb25PcHRpb25zLCB7IG5hbWVzcGFjZTogTkFNRVNQQUNFX1NWRyB9KTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBpZiAocHJldmlvdXMudGV4dCAhPT0gdm5vZGUudGV4dCkge1xyXG4gICAgICAgICAgICAgICAgdXBkYXRlZCA9IHRydWU7XHJcbiAgICAgICAgICAgICAgICBpZiAodm5vZGUudGV4dCA9PT0gdW5kZWZpbmVkKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgZG9tTm9kZS5yZW1vdmVDaGlsZChkb21Ob2RlLmZpcnN0Q2hpbGQpOyAvLyB0aGUgb25seSB0ZXh0bm9kZSBwcmVzdW1hYmx5XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgICBkb21Ob2RlLnRleHRDb250ZW50ID0gdm5vZGUudGV4dDtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB2bm9kZS5kb21Ob2RlID0gZG9tTm9kZTtcclxuICAgICAgICAgICAgdXBkYXRlZCA9IHVwZGF0ZUNoaWxkcmVuKHZub2RlLCBkb21Ob2RlLCBwcmV2aW91cy5jaGlsZHJlbiwgdm5vZGUuY2hpbGRyZW4sIHByb2plY3Rpb25PcHRpb25zKSB8fCB1cGRhdGVkO1xyXG4gICAgICAgICAgICB1cGRhdGVkID0gdXBkYXRlUHJvcGVydGllcyhkb21Ob2RlLCBwcmV2aW91cy5wcm9wZXJ0aWVzLCB2bm9kZS5wcm9wZXJ0aWVzLCBwcm9qZWN0aW9uT3B0aW9ucykgfHwgdXBkYXRlZDtcclxuICAgICAgICAgICAgaWYgKHZub2RlLnByb3BlcnRpZXMgJiYgdm5vZGUucHJvcGVydGllcy5hZnRlclVwZGF0ZSkge1xyXG4gICAgICAgICAgICAgICAgdm5vZGUucHJvcGVydGllcy5hZnRlclVwZGF0ZS5hcHBseSh2bm9kZS5wcm9wZXJ0aWVzLmJpbmQgfHwgdm5vZGUucHJvcGVydGllcywgW2RvbU5vZGUsIHByb2plY3Rpb25PcHRpb25zLCB2bm9kZS52bm9kZVNlbGVjdG9yLCB2bm9kZS5wcm9wZXJ0aWVzLCB2bm9kZS5jaGlsZHJlbl0pO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGlmICh1cGRhdGVkICYmIHZub2RlLnByb3BlcnRpZXMgJiYgdm5vZGUucHJvcGVydGllcy51cGRhdGVBbmltYXRpb24pIHtcclxuICAgICAgICAgICAgdm5vZGUucHJvcGVydGllcy51cGRhdGVBbmltYXRpb24oZG9tTm9kZSwgdm5vZGUucHJvcGVydGllcywgcHJldmlvdXMucHJvcGVydGllcyk7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHJldHVybiB0ZXh0VXBkYXRlZDtcclxuICAgIH07XHJcbiAgICB2YXIgY3JlYXRlUHJvamVjdGlvbiA9IGZ1bmN0aW9uICh2bm9kZSwgcHJvamVjdGlvbk9wdGlvbnMpIHtcclxuICAgICAgICByZXR1cm4ge1xyXG4gICAgICAgICAgICBnZXRMYXN0UmVuZGVyOiBmdW5jdGlvbiAoKSB7IHJldHVybiB2bm9kZTsgfSxcclxuICAgICAgICAgICAgdXBkYXRlOiBmdW5jdGlvbiAodXBkYXRlZFZub2RlKSB7XHJcbiAgICAgICAgICAgICAgICBpZiAodm5vZGUudm5vZGVTZWxlY3RvciAhPT0gdXBkYXRlZFZub2RlLnZub2RlU2VsZWN0b3IpIHtcclxuICAgICAgICAgICAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ1RoZSBzZWxlY3RvciBmb3IgdGhlIHJvb3QgVk5vZGUgbWF5IG5vdCBiZSBjaGFuZ2VkLiAoY29uc2lkZXIgdXNpbmcgZG9tLm1lcmdlIGFuZCBhZGQgb25lIGV4dHJhIGxldmVsIHRvIHRoZSB2aXJ0dWFsIERPTSknKTtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIHZhciBwcmV2aW91c1ZOb2RlID0gdm5vZGU7XHJcbiAgICAgICAgICAgICAgICB2bm9kZSA9IHVwZGF0ZWRWbm9kZTtcclxuICAgICAgICAgICAgICAgIHVwZGF0ZURvbShwcmV2aW91c1ZOb2RlLCB1cGRhdGVkVm5vZGUsIHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAgZG9tTm9kZTogdm5vZGUuZG9tTm9kZVxyXG4gICAgICAgIH07XHJcbiAgICB9O1xuXG4gICAgdmFyIERFRkFVTFRfUFJPSkVDVElPTl9PUFRJT05TID0ge1xyXG4gICAgICAgIG5hbWVzcGFjZTogdW5kZWZpbmVkLFxyXG4gICAgICAgIHBlcmZvcm1hbmNlTG9nZ2VyOiBmdW5jdGlvbiAoKSB7IHJldHVybiB1bmRlZmluZWQ7IH0sXHJcbiAgICAgICAgZXZlbnRIYW5kbGVySW50ZXJjZXB0b3I6IHVuZGVmaW5lZCxcclxuICAgICAgICBzdHlsZUFwcGx5ZXI6IGZ1bmN0aW9uIChkb21Ob2RlLCBzdHlsZU5hbWUsIHZhbHVlKSB7XHJcbiAgICAgICAgICAgIC8vIFByb3ZpZGVzIGEgaG9vayB0byBhZGQgdmVuZG9yIHByZWZpeGVzIGZvciBicm93c2VycyB0aGF0IHN0aWxsIG5lZWQgaXQuXHJcbiAgICAgICAgICAgIGRvbU5vZGUuc3R5bGVbc3R5bGVOYW1lXSA9IHZhbHVlO1xyXG4gICAgICAgIH1cclxuICAgIH07XHJcbiAgICB2YXIgYXBwbHlEZWZhdWx0UHJvamVjdGlvbk9wdGlvbnMgPSBmdW5jdGlvbiAocHJvamVjdG9yT3B0aW9ucykge1xyXG4gICAgICAgIHJldHVybiBleHRlbmQoREVGQVVMVF9QUk9KRUNUSU9OX09QVElPTlMsIHByb2plY3Rvck9wdGlvbnMpO1xyXG4gICAgfTtcclxuICAgIHZhciBkb20gPSB7XHJcbiAgICAgICAgLyoqXHJcbiAgICAgICAgICogQ3JlYXRlcyBhIHJlYWwgRE9NIHRyZWUgZnJvbSBgdm5vZGVgLiBUaGUgW1tQcm9qZWN0aW9uXV0gb2JqZWN0IHJldHVybmVkIHdpbGwgY29udGFpbiB0aGUgcmVzdWx0aW5nIERPTSBOb2RlIGluXHJcbiAgICAgICAgICogaXRzIFtbUHJvamVjdGlvbi5kb21Ob2RlfGRvbU5vZGVdXSBwcm9wZXJ0eS5cclxuICAgICAgICAgKiBUaGlzIGlzIGEgbG93LWxldmVsIG1ldGhvZC4gVXNlcnMgd2lsbCB0eXBpY2FsbHkgdXNlIGEgW1tQcm9qZWN0b3JdXSBpbnN0ZWFkLlxyXG4gICAgICAgICAqIEBwYXJhbSB2bm9kZSAtIFRoZSByb290IG9mIHRoZSB2aXJ0dWFsIERPTSB0cmVlIHRoYXQgd2FzIGNyZWF0ZWQgdXNpbmcgdGhlIFtbaF1dIGZ1bmN0aW9uLiBOT1RFOiBbW1ZOb2RlXV1cclxuICAgICAgICAgKiBvYmplY3RzIG1heSBvbmx5IGJlIHJlbmRlcmVkIG9uY2UuXHJcbiAgICAgICAgICogQHBhcmFtIHByb2plY3Rpb25PcHRpb25zIC0gT3B0aW9ucyB0byBiZSB1c2VkIHRvIGNyZWF0ZSBhbmQgdXBkYXRlIHRoZSBwcm9qZWN0aW9uLlxyXG4gICAgICAgICAqIEByZXR1cm5zIFRoZSBbW1Byb2plY3Rpb25dXSB3aGljaCBhbHNvIGNvbnRhaW5zIHRoZSBET00gTm9kZSB0aGF0IHdhcyBjcmVhdGVkLlxyXG4gICAgICAgICAqL1xyXG4gICAgICAgIGNyZWF0ZTogZnVuY3Rpb24gKHZub2RlLCBwcm9qZWN0aW9uT3B0aW9ucykge1xyXG4gICAgICAgICAgICBwcm9qZWN0aW9uT3B0aW9ucyA9IGFwcGx5RGVmYXVsdFByb2plY3Rpb25PcHRpb25zKHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICAgICAgY3JlYXRlRG9tKHZub2RlLCBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdkaXYnKSwgdW5kZWZpbmVkLCBwcm9qZWN0aW9uT3B0aW9ucyk7XHJcbiAgICAgICAgICAgIHJldHVybiBjcmVhdGVQcm9qZWN0aW9uKHZub2RlLCBwcm9qZWN0aW9uT3B0aW9ucyk7XHJcbiAgICAgICAgfSxcclxuICAgICAgICAvKipcclxuICAgICAgICAgKiBBcHBlbmRzIGEgbmV3IGNoaWxkIG5vZGUgdG8gdGhlIERPTSB3aGljaCBpcyBnZW5lcmF0ZWQgZnJvbSBhIFtbVk5vZGVdXS5cclxuICAgICAgICAgKiBUaGlzIGlzIGEgbG93LWxldmVsIG1ldGhvZC4gVXNlcnMgd2lsbCB0eXBpY2FsbHkgdXNlIGEgW1tQcm9qZWN0b3JdXSBpbnN0ZWFkLlxyXG4gICAgICAgICAqIEBwYXJhbSBwYXJlbnROb2RlIC0gVGhlIHBhcmVudCBub2RlIGZvciB0aGUgbmV3IGNoaWxkIG5vZGUuXHJcbiAgICAgICAgICogQHBhcmFtIHZub2RlIC0gVGhlIHJvb3Qgb2YgdGhlIHZpcnR1YWwgRE9NIHRyZWUgdGhhdCB3YXMgY3JlYXRlZCB1c2luZyB0aGUgW1toXV0gZnVuY3Rpb24uIE5PVEU6IFtbVk5vZGVdXVxyXG4gICAgICAgICAqIG9iamVjdHMgbWF5IG9ubHkgYmUgcmVuZGVyZWQgb25jZS5cclxuICAgICAgICAgKiBAcGFyYW0gcHJvamVjdGlvbk9wdGlvbnMgLSBPcHRpb25zIHRvIGJlIHVzZWQgdG8gY3JlYXRlIGFuZCB1cGRhdGUgdGhlIFtbUHJvamVjdGlvbl1dLlxyXG4gICAgICAgICAqIEByZXR1cm5zIFRoZSBbW1Byb2plY3Rpb25dXSB0aGF0IHdhcyBjcmVhdGVkLlxyXG4gICAgICAgICAqL1xyXG4gICAgICAgIGFwcGVuZDogZnVuY3Rpb24gKHBhcmVudE5vZGUsIHZub2RlLCBwcm9qZWN0aW9uT3B0aW9ucykge1xyXG4gICAgICAgICAgICBwcm9qZWN0aW9uT3B0aW9ucyA9IGFwcGx5RGVmYXVsdFByb2plY3Rpb25PcHRpb25zKHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICAgICAgY3JlYXRlRG9tKHZub2RlLCBwYXJlbnROb2RlLCB1bmRlZmluZWQsIHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICAgICAgcmV0dXJuIGNyZWF0ZVByb2plY3Rpb24odm5vZGUsIHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICB9LFxyXG4gICAgICAgIC8qKlxyXG4gICAgICAgICAqIEluc2VydHMgYSBuZXcgRE9NIG5vZGUgd2hpY2ggaXMgZ2VuZXJhdGVkIGZyb20gYSBbW1ZOb2RlXV0uXHJcbiAgICAgICAgICogVGhpcyBpcyBhIGxvdy1sZXZlbCBtZXRob2QuIFVzZXJzIHdpbCB0eXBpY2FsbHkgdXNlIGEgW1tQcm9qZWN0b3JdXSBpbnN0ZWFkLlxyXG4gICAgICAgICAqIEBwYXJhbSBiZWZvcmVOb2RlIC0gVGhlIG5vZGUgdGhhdCB0aGUgRE9NIE5vZGUgaXMgaW5zZXJ0ZWQgYmVmb3JlLlxyXG4gICAgICAgICAqIEBwYXJhbSB2bm9kZSAtIFRoZSByb290IG9mIHRoZSB2aXJ0dWFsIERPTSB0cmVlIHRoYXQgd2FzIGNyZWF0ZWQgdXNpbmcgdGhlIFtbaF1dIGZ1bmN0aW9uLlxyXG4gICAgICAgICAqIE5PVEU6IFtbVk5vZGVdXSBvYmplY3RzIG1heSBvbmx5IGJlIHJlbmRlcmVkIG9uY2UuXHJcbiAgICAgICAgICogQHBhcmFtIHByb2plY3Rpb25PcHRpb25zIC0gT3B0aW9ucyB0byBiZSB1c2VkIHRvIGNyZWF0ZSBhbmQgdXBkYXRlIHRoZSBwcm9qZWN0aW9uLCBzZWUgW1tjcmVhdGVQcm9qZWN0b3JdXS5cclxuICAgICAgICAgKiBAcmV0dXJucyBUaGUgW1tQcm9qZWN0aW9uXV0gdGhhdCB3YXMgY3JlYXRlZC5cclxuICAgICAgICAgKi9cclxuICAgICAgICBpbnNlcnRCZWZvcmU6IGZ1bmN0aW9uIChiZWZvcmVOb2RlLCB2bm9kZSwgcHJvamVjdGlvbk9wdGlvbnMpIHtcclxuICAgICAgICAgICAgcHJvamVjdGlvbk9wdGlvbnMgPSBhcHBseURlZmF1bHRQcm9qZWN0aW9uT3B0aW9ucyhwcm9qZWN0aW9uT3B0aW9ucyk7XHJcbiAgICAgICAgICAgIGNyZWF0ZURvbSh2bm9kZSwgYmVmb3JlTm9kZS5wYXJlbnROb2RlLCBiZWZvcmVOb2RlLCBwcm9qZWN0aW9uT3B0aW9ucyk7XHJcbiAgICAgICAgICAgIHJldHVybiBjcmVhdGVQcm9qZWN0aW9uKHZub2RlLCBwcm9qZWN0aW9uT3B0aW9ucyk7XHJcbiAgICAgICAgfSxcclxuICAgICAgICAvKipcclxuICAgICAgICAgKiBNZXJnZXMgYSBuZXcgRE9NIG5vZGUgd2hpY2ggaXMgZ2VuZXJhdGVkIGZyb20gYSBbW1ZOb2RlXV0gd2l0aCBhbiBleGlzdGluZyBET00gTm9kZS5cclxuICAgICAgICAgKiBUaGlzIG1lYW5zIHRoYXQgdGhlIHZpcnR1YWwgRE9NIGFuZCB0aGUgcmVhbCBET00gd2lsbCBoYXZlIG9uZSBvdmVybGFwcGluZyBlbGVtZW50LlxyXG4gICAgICAgICAqIFRoZXJlZm9yZSB0aGUgc2VsZWN0b3IgZm9yIHRoZSByb290IFtbVk5vZGVdXSB3aWxsIGJlIGlnbm9yZWQsIGJ1dCBpdHMgcHJvcGVydGllcyBhbmQgY2hpbGRyZW4gd2lsbCBiZSBhcHBsaWVkIHRvIHRoZSBFbGVtZW50IHByb3ZpZGVkLlxyXG4gICAgICAgICAqIFRoaXMgaXMgYSBsb3ctbGV2ZWwgbWV0aG9kLiBVc2VycyB3aWwgdHlwaWNhbGx5IHVzZSBhIFtbUHJvamVjdG9yXV0gaW5zdGVhZC5cclxuICAgICAgICAgKiBAcGFyYW0gZWxlbWVudCAtIFRoZSBleGlzdGluZyBlbGVtZW50IHRvIGFkb3B0IGFzIHRoZSByb290IG9mIHRoZSBuZXcgdmlydHVhbCBET00uIEV4aXN0aW5nIGF0dHJpYnV0ZXMgYW5kIGNoaWxkIG5vZGVzIGFyZSBwcmVzZXJ2ZWQuXHJcbiAgICAgICAgICogQHBhcmFtIHZub2RlIC0gVGhlIHJvb3Qgb2YgdGhlIHZpcnR1YWwgRE9NIHRyZWUgdGhhdCB3YXMgY3JlYXRlZCB1c2luZyB0aGUgW1toXV0gZnVuY3Rpb24uIE5PVEU6IFtbVk5vZGVdXSBvYmplY3RzXHJcbiAgICAgICAgICogbWF5IG9ubHkgYmUgcmVuZGVyZWQgb25jZS5cclxuICAgICAgICAgKiBAcGFyYW0gcHJvamVjdGlvbk9wdGlvbnMgLSBPcHRpb25zIHRvIGJlIHVzZWQgdG8gY3JlYXRlIGFuZCB1cGRhdGUgdGhlIHByb2plY3Rpb24sIHNlZSBbW2NyZWF0ZVByb2plY3Rvcl1dLlxyXG4gICAgICAgICAqIEByZXR1cm5zIFRoZSBbW1Byb2plY3Rpb25dXSB0aGF0IHdhcyBjcmVhdGVkLlxyXG4gICAgICAgICAqL1xyXG4gICAgICAgIG1lcmdlOiBmdW5jdGlvbiAoZWxlbWVudCwgdm5vZGUsIHByb2plY3Rpb25PcHRpb25zKSB7XHJcbiAgICAgICAgICAgIHByb2plY3Rpb25PcHRpb25zID0gYXBwbHlEZWZhdWx0UHJvamVjdGlvbk9wdGlvbnMocHJvamVjdGlvbk9wdGlvbnMpO1xyXG4gICAgICAgICAgICB2bm9kZS5kb21Ob2RlID0gZWxlbWVudDtcclxuICAgICAgICAgICAgaW5pdFByb3BlcnRpZXNBbmRDaGlsZHJlbihlbGVtZW50LCB2bm9kZSwgcHJvamVjdGlvbk9wdGlvbnMpO1xyXG4gICAgICAgICAgICByZXR1cm4gY3JlYXRlUHJvamVjdGlvbih2bm9kZSwgcHJvamVjdGlvbk9wdGlvbnMpO1xyXG4gICAgICAgIH0sXHJcbiAgICAgICAgLyoqXHJcbiAgICAgICAgICogUmVwbGFjZXMgYW4gZXhpc3RpbmcgRE9NIG5vZGUgd2l0aCBhIG5vZGUgZ2VuZXJhdGVkIGZyb20gYSBbW1ZOb2RlXV0uXHJcbiAgICAgICAgICogVGhpcyBpcyBhIGxvdy1sZXZlbCBtZXRob2QuIFVzZXJzIHdpbGwgdHlwaWNhbGx5IHVzZSBhIFtbUHJvamVjdG9yXV0gaW5zdGVhZC5cclxuICAgICAgICAgKiBAcGFyYW0gZWxlbWVudCAtIFRoZSBub2RlIGZvciB0aGUgW1tWTm9kZV1dIHRvIHJlcGxhY2UuXHJcbiAgICAgICAgICogQHBhcmFtIHZub2RlIC0gVGhlIHJvb3Qgb2YgdGhlIHZpcnR1YWwgRE9NIHRyZWUgdGhhdCB3YXMgY3JlYXRlZCB1c2luZyB0aGUgW1toXV0gZnVuY3Rpb24uIE5PVEU6IFtbVk5vZGVdXVxyXG4gICAgICAgICAqIG9iamVjdHMgbWF5IG9ubHkgYmUgcmVuZGVyZWQgb25jZS5cclxuICAgICAgICAgKiBAcGFyYW0gcHJvamVjdGlvbk9wdGlvbnMgLSBPcHRpb25zIHRvIGJlIHVzZWQgdG8gY3JlYXRlIGFuZCB1cGRhdGUgdGhlIFtbUHJvamVjdGlvbl1dLlxyXG4gICAgICAgICAqIEByZXR1cm5zIFRoZSBbW1Byb2plY3Rpb25dXSB0aGF0IHdhcyBjcmVhdGVkLlxyXG4gICAgICAgICAqL1xyXG4gICAgICAgIHJlcGxhY2U6IGZ1bmN0aW9uIChlbGVtZW50LCB2bm9kZSwgcHJvamVjdGlvbk9wdGlvbnMpIHtcclxuICAgICAgICAgICAgcHJvamVjdGlvbk9wdGlvbnMgPSBhcHBseURlZmF1bHRQcm9qZWN0aW9uT3B0aW9ucyhwcm9qZWN0aW9uT3B0aW9ucyk7XHJcbiAgICAgICAgICAgIGNyZWF0ZURvbSh2bm9kZSwgZWxlbWVudC5wYXJlbnROb2RlLCBlbGVtZW50LCBwcm9qZWN0aW9uT3B0aW9ucyk7XHJcbiAgICAgICAgICAgIGVsZW1lbnQucGFyZW50Tm9kZS5yZW1vdmVDaGlsZChlbGVtZW50KTtcclxuICAgICAgICAgICAgcmV0dXJuIGNyZWF0ZVByb2plY3Rpb24odm5vZGUsIHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICB9XHJcbiAgICB9O1xuXG4gICAgLyogdHNsaW50OmRpc2FibGUgZnVuY3Rpb24tbmFtZSAqL1xyXG4gICAgdmFyIHRvVGV4dFZOb2RlID0gZnVuY3Rpb24gKGRhdGEpIHtcclxuICAgICAgICByZXR1cm4ge1xyXG4gICAgICAgICAgICB2bm9kZVNlbGVjdG9yOiAnJyxcclxuICAgICAgICAgICAgcHJvcGVydGllczogdW5kZWZpbmVkLFxyXG4gICAgICAgICAgICBjaGlsZHJlbjogdW5kZWZpbmVkLFxyXG4gICAgICAgICAgICB0ZXh0OiBkYXRhLnRvU3RyaW5nKCksXHJcbiAgICAgICAgICAgIGRvbU5vZGU6IG51bGxcclxuICAgICAgICB9O1xyXG4gICAgfTtcclxuICAgIHZhciBhcHBlbmRDaGlsZHJlbiA9IGZ1bmN0aW9uIChwYXJlbnRTZWxlY3RvciwgaW5zZXJ0aW9ucywgbWFpbikge1xyXG4gICAgICAgIGZvciAodmFyIGkgPSAwLCBsZW5ndGhfMSA9IGluc2VydGlvbnMubGVuZ3RoOyBpIDwgbGVuZ3RoXzE7IGkrKykge1xyXG4gICAgICAgICAgICB2YXIgaXRlbSA9IGluc2VydGlvbnNbaV07XHJcbiAgICAgICAgICAgIGlmIChBcnJheS5pc0FycmF5KGl0ZW0pKSB7XHJcbiAgICAgICAgICAgICAgICBhcHBlbmRDaGlsZHJlbihwYXJlbnRTZWxlY3RvciwgaXRlbSwgbWFpbik7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICBpZiAoaXRlbSAhPT0gbnVsbCAmJiBpdGVtICE9PSB1bmRlZmluZWQgJiYgaXRlbSAhPT0gZmFsc2UpIHtcclxuICAgICAgICAgICAgICAgICAgICBpZiAodHlwZW9mIGl0ZW0gPT09ICdzdHJpbmcnKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGl0ZW0gPSB0b1RleHRWTm9kZShpdGVtKTtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgbWFpbi5wdXNoKGl0ZW0pO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIGZ1bmN0aW9uIGgoc2VsZWN0b3IsIHByb3BlcnRpZXMsIGNoaWxkcmVuKSB7XHJcbiAgICAgICAgaWYgKEFycmF5LmlzQXJyYXkocHJvcGVydGllcykpIHtcclxuICAgICAgICAgICAgY2hpbGRyZW4gPSBwcm9wZXJ0aWVzO1xyXG4gICAgICAgICAgICBwcm9wZXJ0aWVzID0gdW5kZWZpbmVkO1xyXG4gICAgICAgIH1cclxuICAgICAgICBlbHNlIGlmICgocHJvcGVydGllcyAmJiAodHlwZW9mIHByb3BlcnRpZXMgPT09ICdzdHJpbmcnIHx8IHByb3BlcnRpZXMuaGFzT3duUHJvcGVydHkoJ3Zub2RlU2VsZWN0b3InKSkpIHx8XHJcbiAgICAgICAgICAgIChjaGlsZHJlbiAmJiAodHlwZW9mIGNoaWxkcmVuID09PSAnc3RyaW5nJyB8fCBjaGlsZHJlbi5oYXNPd25Qcm9wZXJ0eSgndm5vZGVTZWxlY3RvcicpKSkpIHtcclxuICAgICAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdoIGNhbGxlZCB3aXRoIGludmFsaWQgYXJndW1lbnRzJyk7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHZhciB0ZXh0O1xyXG4gICAgICAgIHZhciBmbGF0dGVuZWRDaGlsZHJlbjtcclxuICAgICAgICAvLyBSZWNvZ25pemUgYSBjb21tb24gc3BlY2lhbCBjYXNlIHdoZXJlIHRoZXJlIGlzIG9ubHkgYSBzaW5nbGUgdGV4dCBub2RlXHJcbiAgICAgICAgaWYgKGNoaWxkcmVuICYmIGNoaWxkcmVuLmxlbmd0aCA9PT0gMSAmJiB0eXBlb2YgY2hpbGRyZW5bMF0gPT09ICdzdHJpbmcnKSB7XHJcbiAgICAgICAgICAgIHRleHQgPSBjaGlsZHJlblswXTtcclxuICAgICAgICB9XHJcbiAgICAgICAgZWxzZSBpZiAoY2hpbGRyZW4pIHtcclxuICAgICAgICAgICAgZmxhdHRlbmVkQ2hpbGRyZW4gPSBbXTtcclxuICAgICAgICAgICAgYXBwZW5kQ2hpbGRyZW4oc2VsZWN0b3IsIGNoaWxkcmVuLCBmbGF0dGVuZWRDaGlsZHJlbik7XHJcbiAgICAgICAgICAgIGlmIChmbGF0dGVuZWRDaGlsZHJlbi5sZW5ndGggPT09IDApIHtcclxuICAgICAgICAgICAgICAgIGZsYXR0ZW5lZENoaWxkcmVuID0gdW5kZWZpbmVkO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHJldHVybiB7XHJcbiAgICAgICAgICAgIHZub2RlU2VsZWN0b3I6IHNlbGVjdG9yLFxyXG4gICAgICAgICAgICBwcm9wZXJ0aWVzOiBwcm9wZXJ0aWVzLFxyXG4gICAgICAgICAgICBjaGlsZHJlbjogZmxhdHRlbmVkQ2hpbGRyZW4sXHJcbiAgICAgICAgICAgIHRleHQ6ICh0ZXh0ID09PSAnJykgPyB1bmRlZmluZWQgOiB0ZXh0LFxyXG4gICAgICAgICAgICBkb21Ob2RlOiBudWxsXHJcbiAgICAgICAgfTtcclxuICAgIH1cblxuICAgIHZhciBjcmVhdGVQYXJlbnROb2RlUGF0aCA9IGZ1bmN0aW9uIChub2RlLCByb290Tm9kZSkge1xyXG4gICAgICAgIHZhciBwYXJlbnROb2RlUGF0aCA9IFtdO1xyXG4gICAgICAgIHdoaWxlIChub2RlICE9PSByb290Tm9kZSkge1xyXG4gICAgICAgICAgICBwYXJlbnROb2RlUGF0aC5wdXNoKG5vZGUpO1xyXG4gICAgICAgICAgICBub2RlID0gbm9kZS5wYXJlbnROb2RlO1xyXG4gICAgICAgIH1cclxuICAgICAgICByZXR1cm4gcGFyZW50Tm9kZVBhdGg7XHJcbiAgICB9O1xyXG4gICAgdmFyIGZpbmQ7XHJcbiAgICBpZiAoQXJyYXkucHJvdG90eXBlLmZpbmQpIHtcclxuICAgICAgICBmaW5kID0gZnVuY3Rpb24gKGl0ZW1zLCBwcmVkaWNhdGUpIHsgcmV0dXJuIGl0ZW1zLmZpbmQocHJlZGljYXRlKTsgfTtcclxuICAgIH1cclxuICAgIGVsc2Uge1xyXG4gICAgICAgIGZpbmQgPSBmdW5jdGlvbiAoaXRlbXMsIHByZWRpY2F0ZSkgeyByZXR1cm4gaXRlbXMuZmlsdGVyKHByZWRpY2F0ZSlbMF07IH07XHJcbiAgICB9XHJcbiAgICB2YXIgZmluZFZOb2RlQnlQYXJlbnROb2RlUGF0aCA9IGZ1bmN0aW9uICh2bm9kZSwgcGFyZW50Tm9kZVBhdGgpIHtcclxuICAgICAgICB2YXIgcmVzdWx0ID0gdm5vZGU7XHJcbiAgICAgICAgcGFyZW50Tm9kZVBhdGguZm9yRWFjaChmdW5jdGlvbiAobm9kZSkge1xyXG4gICAgICAgICAgICByZXN1bHQgPSAocmVzdWx0ICYmIHJlc3VsdC5jaGlsZHJlbikgPyBmaW5kKHJlc3VsdC5jaGlsZHJlbiwgZnVuY3Rpb24gKGNoaWxkKSB7IHJldHVybiBjaGlsZC5kb21Ob2RlID09PSBub2RlOyB9KSA6IHVuZGVmaW5lZDtcclxuICAgICAgICB9KTtcclxuICAgICAgICByZXR1cm4gcmVzdWx0O1xyXG4gICAgfTtcclxuICAgIHZhciBjcmVhdGVFdmVudEhhbmRsZXJJbnRlcmNlcHRvciA9IGZ1bmN0aW9uIChwcm9qZWN0b3IsIGdldFByb2plY3Rpb24sIHBlcmZvcm1hbmNlTG9nZ2VyKSB7XHJcbiAgICAgICAgdmFyIG1vZGlmaWVkRXZlbnRIYW5kbGVyID0gZnVuY3Rpb24gKGV2dCkge1xyXG4gICAgICAgICAgICBwZXJmb3JtYW5jZUxvZ2dlcignZG9tRXZlbnQnLCBldnQpO1xyXG4gICAgICAgICAgICB2YXIgcHJvamVjdGlvbiA9IGdldFByb2plY3Rpb24oKTtcclxuICAgICAgICAgICAgdmFyIHBhcmVudE5vZGVQYXRoID0gY3JlYXRlUGFyZW50Tm9kZVBhdGgoZXZ0LmN1cnJlbnRUYXJnZXQsIHByb2plY3Rpb24uZG9tTm9kZSk7XHJcbiAgICAgICAgICAgIHBhcmVudE5vZGVQYXRoLnJldmVyc2UoKTtcclxuICAgICAgICAgICAgdmFyIG1hdGNoaW5nVk5vZGUgPSBmaW5kVk5vZGVCeVBhcmVudE5vZGVQYXRoKHByb2plY3Rpb24uZ2V0TGFzdFJlbmRlcigpLCBwYXJlbnROb2RlUGF0aCk7XHJcbiAgICAgICAgICAgIHByb2plY3Rvci5zY2hlZHVsZVJlbmRlcigpO1xyXG4gICAgICAgICAgICB2YXIgcmVzdWx0O1xyXG4gICAgICAgICAgICBpZiAobWF0Y2hpbmdWTm9kZSkge1xyXG4gICAgICAgICAgICAgICAgLyogdHNsaW50OmRpc2FibGUgbm8taW52YWxpZC10aGlzICovXHJcbiAgICAgICAgICAgICAgICByZXN1bHQgPSBtYXRjaGluZ1ZOb2RlLnByb3BlcnRpZXNbXCJvblwiICsgZXZ0LnR5cGVdLmFwcGx5KG1hdGNoaW5nVk5vZGUucHJvcGVydGllcy5iaW5kIHx8IHRoaXMsIGFyZ3VtZW50cyk7XHJcbiAgICAgICAgICAgICAgICAvKiB0c2xpbnQ6ZW5hYmxlIG5vLWludmFsaWQtdGhpcyAqL1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIHBlcmZvcm1hbmNlTG9nZ2VyKCdkb21FdmVudFByb2Nlc3NlZCcsIGV2dCk7XHJcbiAgICAgICAgICAgIHJldHVybiByZXN1bHQ7XHJcbiAgICAgICAgfTtcclxuICAgICAgICByZXR1cm4gZnVuY3Rpb24gKHByb3BlcnR5TmFtZSwgZXZlbnRIYW5kbGVyLCBkb21Ob2RlLCBwcm9wZXJ0aWVzKSB7IHJldHVybiBtb2RpZmllZEV2ZW50SGFuZGxlcjsgfTtcclxuICAgIH07XHJcbiAgICAvKipcclxuICAgICAqIENyZWF0ZXMgYSBbW1Byb2plY3Rvcl1dIGluc3RhbmNlIHVzaW5nIHRoZSBwcm92aWRlZCBwcm9qZWN0aW9uT3B0aW9ucy5cclxuICAgICAqXHJcbiAgICAgKiBGb3IgbW9yZSBpbmZvcm1hdGlvbiwgc2VlIFtbUHJvamVjdG9yXV0uXHJcbiAgICAgKlxyXG4gICAgICogQHBhcmFtIHByb2plY3Rvck9wdGlvbnMgICBPcHRpb25zIHRoYXQgaW5mbHVlbmNlIGhvdyB0aGUgRE9NIGlzIHJlbmRlcmVkIGFuZCB1cGRhdGVkLlxyXG4gICAgICovXHJcbiAgICB2YXIgY3JlYXRlUHJvamVjdG9yID0gZnVuY3Rpb24gKHByb2plY3Rvck9wdGlvbnMpIHtcclxuICAgICAgICB2YXIgcHJvamVjdG9yO1xyXG4gICAgICAgIHZhciBwcm9qZWN0aW9uT3B0aW9ucyA9IGFwcGx5RGVmYXVsdFByb2plY3Rpb25PcHRpb25zKHByb2plY3Rvck9wdGlvbnMpO1xyXG4gICAgICAgIHZhciBwZXJmb3JtYW5jZUxvZ2dlciA9IHByb2plY3Rpb25PcHRpb25zLnBlcmZvcm1hbmNlTG9nZ2VyO1xyXG4gICAgICAgIHZhciByZW5kZXJDb21wbGV0ZWQgPSB0cnVlO1xyXG4gICAgICAgIHZhciBzY2hlZHVsZWQ7XHJcbiAgICAgICAgdmFyIHN0b3BwZWQgPSBmYWxzZTtcclxuICAgICAgICB2YXIgcHJvamVjdGlvbnMgPSBbXTtcclxuICAgICAgICB2YXIgcmVuZGVyRnVuY3Rpb25zID0gW107IC8vIG1hdGNoZXMgdGhlIHByb2plY3Rpb25zIGFycmF5XHJcbiAgICAgICAgdmFyIGFkZFByb2plY3Rpb24gPSBmdW5jdGlvbiAoXHJcbiAgICAgICAgLyogb25lIG9mOiBkb20uYXBwZW5kLCBkb20uaW5zZXJ0QmVmb3JlLCBkb20ucmVwbGFjZSwgZG9tLm1lcmdlICovXHJcbiAgICAgICAgZG9tRnVuY3Rpb24sIFxyXG4gICAgICAgIC8qIHRoZSBwYXJhbWV0ZXIgb2YgdGhlIGRvbUZ1bmN0aW9uICovXHJcbiAgICAgICAgbm9kZSwgcmVuZGVyRnVuY3Rpb24pIHtcclxuICAgICAgICAgICAgdmFyIHByb2plY3Rpb247XHJcbiAgICAgICAgICAgIHZhciBnZXRQcm9qZWN0aW9uID0gZnVuY3Rpb24gKCkgeyByZXR1cm4gcHJvamVjdGlvbjsgfTtcclxuICAgICAgICAgICAgcHJvamVjdGlvbk9wdGlvbnMuZXZlbnRIYW5kbGVySW50ZXJjZXB0b3IgPSBjcmVhdGVFdmVudEhhbmRsZXJJbnRlcmNlcHRvcihwcm9qZWN0b3IsIGdldFByb2plY3Rpb24sIHBlcmZvcm1hbmNlTG9nZ2VyKTtcclxuICAgICAgICAgICAgcHJvamVjdGlvbiA9IGRvbUZ1bmN0aW9uKG5vZGUsIHJlbmRlckZ1bmN0aW9uKCksIHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICAgICAgcHJvamVjdGlvbnMucHVzaChwcm9qZWN0aW9uKTtcclxuICAgICAgICAgICAgcmVuZGVyRnVuY3Rpb25zLnB1c2gocmVuZGVyRnVuY3Rpb24pO1xyXG4gICAgICAgIH07XHJcbiAgICAgICAgdmFyIGRvUmVuZGVyID0gZnVuY3Rpb24gKCkge1xyXG4gICAgICAgICAgICBzY2hlZHVsZWQgPSB1bmRlZmluZWQ7XHJcbiAgICAgICAgICAgIGlmICghcmVuZGVyQ29tcGxldGVkKSB7XHJcbiAgICAgICAgICAgICAgICByZXR1cm47IC8vIFRoZSBsYXN0IHJlbmRlciB0aHJldyBhbiBlcnJvciwgaXQgc2hvdWxkIGhhdmUgYmVlbiBsb2dnZWQgaW4gdGhlIGJyb3dzZXIgY29uc29sZS5cclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICByZW5kZXJDb21wbGV0ZWQgPSBmYWxzZTtcclxuICAgICAgICAgICAgcGVyZm9ybWFuY2VMb2dnZXIoJ3JlbmRlclN0YXJ0JywgdW5kZWZpbmVkKTtcclxuICAgICAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCBwcm9qZWN0aW9ucy5sZW5ndGg7IGkrKykge1xyXG4gICAgICAgICAgICAgICAgdmFyIHVwZGF0ZWRWbm9kZSA9IHJlbmRlckZ1bmN0aW9uc1tpXSgpO1xyXG4gICAgICAgICAgICAgICAgcGVyZm9ybWFuY2VMb2dnZXIoJ3JlbmRlcmVkJywgdW5kZWZpbmVkKTtcclxuICAgICAgICAgICAgICAgIHByb2plY3Rpb25zW2ldLnVwZGF0ZSh1cGRhdGVkVm5vZGUpO1xyXG4gICAgICAgICAgICAgICAgcGVyZm9ybWFuY2VMb2dnZXIoJ3BhdGNoZWQnLCB1bmRlZmluZWQpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIHBlcmZvcm1hbmNlTG9nZ2VyKCdyZW5kZXJEb25lJywgdW5kZWZpbmVkKTtcclxuICAgICAgICAgICAgcmVuZGVyQ29tcGxldGVkID0gdHJ1ZTtcclxuICAgICAgICB9O1xyXG4gICAgICAgIHByb2plY3RvciA9IHtcclxuICAgICAgICAgICAgcmVuZGVyTm93OiBkb1JlbmRlcixcclxuICAgICAgICAgICAgc2NoZWR1bGVSZW5kZXI6IGZ1bmN0aW9uICgpIHtcclxuICAgICAgICAgICAgICAgIGlmICghc2NoZWR1bGVkICYmICFzdG9wcGVkKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgc2NoZWR1bGVkID0gcmVxdWVzdEFuaW1hdGlvbkZyYW1lKGRvUmVuZGVyKTtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAgc3RvcDogZnVuY3Rpb24gKCkge1xyXG4gICAgICAgICAgICAgICAgaWYgKHNjaGVkdWxlZCkge1xyXG4gICAgICAgICAgICAgICAgICAgIGNhbmNlbEFuaW1hdGlvbkZyYW1lKHNjaGVkdWxlZCk7XHJcbiAgICAgICAgICAgICAgICAgICAgc2NoZWR1bGVkID0gdW5kZWZpbmVkO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgc3RvcHBlZCA9IHRydWU7XHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIHJlc3VtZTogZnVuY3Rpb24gKCkge1xyXG4gICAgICAgICAgICAgICAgc3RvcHBlZCA9IGZhbHNlO1xyXG4gICAgICAgICAgICAgICAgcmVuZGVyQ29tcGxldGVkID0gdHJ1ZTtcclxuICAgICAgICAgICAgICAgIHByb2plY3Rvci5zY2hlZHVsZVJlbmRlcigpO1xyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICBhcHBlbmQ6IGZ1bmN0aW9uIChwYXJlbnROb2RlLCByZW5kZXJGdW5jdGlvbikge1xyXG4gICAgICAgICAgICAgICAgYWRkUHJvamVjdGlvbihkb20uYXBwZW5kLCBwYXJlbnROb2RlLCByZW5kZXJGdW5jdGlvbik7XHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIGluc2VydEJlZm9yZTogZnVuY3Rpb24gKGJlZm9yZU5vZGUsIHJlbmRlckZ1bmN0aW9uKSB7XHJcbiAgICAgICAgICAgICAgICBhZGRQcm9qZWN0aW9uKGRvbS5pbnNlcnRCZWZvcmUsIGJlZm9yZU5vZGUsIHJlbmRlckZ1bmN0aW9uKTtcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAgbWVyZ2U6IGZ1bmN0aW9uIChkb21Ob2RlLCByZW5kZXJGdW5jdGlvbikge1xyXG4gICAgICAgICAgICAgICAgYWRkUHJvamVjdGlvbihkb20ubWVyZ2UsIGRvbU5vZGUsIHJlbmRlckZ1bmN0aW9uKTtcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAgcmVwbGFjZTogZnVuY3Rpb24gKGRvbU5vZGUsIHJlbmRlckZ1bmN0aW9uKSB7XHJcbiAgICAgICAgICAgICAgICBhZGRQcm9qZWN0aW9uKGRvbS5yZXBsYWNlLCBkb21Ob2RlLCByZW5kZXJGdW5jdGlvbik7XHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIGRldGFjaDogZnVuY3Rpb24gKHJlbmRlckZ1bmN0aW9uKSB7XHJcbiAgICAgICAgICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IHJlbmRlckZ1bmN0aW9ucy5sZW5ndGg7IGkrKykge1xyXG4gICAgICAgICAgICAgICAgICAgIGlmIChyZW5kZXJGdW5jdGlvbnNbaV0gPT09IHJlbmRlckZ1bmN0aW9uKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHJlbmRlckZ1bmN0aW9ucy5zcGxpY2UoaSwgMSk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHJldHVybiBwcm9qZWN0aW9ucy5zcGxpY2UoaSwgMSlbMF07XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdyZW5kZXJGdW5jdGlvbiB3YXMgbm90IGZvdW5kJyk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9O1xyXG4gICAgICAgIHJldHVybiBwcm9qZWN0b3I7XHJcbiAgICB9O1xuXG4gICAgLyoqXHJcbiAgICAgKiBDcmVhdGVzIGEgW1tDYWxjdWxhdGlvbkNhY2hlXV0gb2JqZWN0LCB1c2VmdWwgZm9yIGNhY2hpbmcgW1tWTm9kZV1dIHRyZWVzLlxyXG4gICAgICogSW4gcHJhY3RpY2UsIGNhY2hpbmcgb2YgW1tWTm9kZV1dIHRyZWVzIGlzIG5vdCBuZWVkZWQsIGJlY2F1c2UgYWNoaWV2aW5nIDYwIGZyYW1lcyBwZXIgc2Vjb25kIGlzIGFsbW9zdCBuZXZlciBhIHByb2JsZW0uXHJcbiAgICAgKiBGb3IgbW9yZSBpbmZvcm1hdGlvbiwgc2VlIFtbQ2FsY3VsYXRpb25DYWNoZV1dLlxyXG4gICAgICpcclxuICAgICAqIEBwYXJhbSA8UmVzdWx0PiBUaGUgdHlwZSBvZiB0aGUgdmFsdWUgdGhhdCBpcyBjYWNoZWQuXHJcbiAgICAgKi9cclxuICAgIHZhciBjcmVhdGVDYWNoZSA9IGZ1bmN0aW9uICgpIHtcclxuICAgICAgICB2YXIgY2FjaGVkSW5wdXRzO1xyXG4gICAgICAgIHZhciBjYWNoZWRPdXRjb21lO1xyXG4gICAgICAgIHJldHVybiB7XHJcbiAgICAgICAgICAgIGludmFsaWRhdGU6IGZ1bmN0aW9uICgpIHtcclxuICAgICAgICAgICAgICAgIGNhY2hlZE91dGNvbWUgPSB1bmRlZmluZWQ7XHJcbiAgICAgICAgICAgICAgICBjYWNoZWRJbnB1dHMgPSB1bmRlZmluZWQ7XHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIHJlc3VsdDogZnVuY3Rpb24gKGlucHV0cywgY2FsY3VsYXRpb24pIHtcclxuICAgICAgICAgICAgICAgIGlmIChjYWNoZWRJbnB1dHMpIHtcclxuICAgICAgICAgICAgICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IGlucHV0cy5sZW5ndGg7IGkrKykge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBpZiAoY2FjaGVkSW5wdXRzW2ldICE9PSBpbnB1dHNbaV0pIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNhY2hlZE91dGNvbWUgPSB1bmRlZmluZWQ7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICBpZiAoIWNhY2hlZE91dGNvbWUpIHtcclxuICAgICAgICAgICAgICAgICAgICBjYWNoZWRPdXRjb21lID0gY2FsY3VsYXRpb24oKTtcclxuICAgICAgICAgICAgICAgICAgICBjYWNoZWRJbnB1dHMgPSBpbnB1dHM7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICByZXR1cm4gY2FjaGVkT3V0Y29tZTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH07XHJcbiAgICB9O1xuXG4gICAgLyoqXHJcbiAgICAgKiBDcmVhdGVzIGEge0BsaW5rIE1hcHBpbmd9IGluc3RhbmNlIHRoYXQga2VlcHMgYW4gYXJyYXkgb2YgcmVzdWx0IG9iamVjdHMgc3luY2hyb25pemVkIHdpdGggYW4gYXJyYXkgb2Ygc291cmNlIG9iamVjdHMuXHJcbiAgICAgKiBTZWUge0BsaW5rIGh0dHA6Ly9tYXF1ZXR0ZWpzLm9yZy9kb2NzL2FycmF5cy5odG1sfFdvcmtpbmcgd2l0aCBhcnJheXN9LlxyXG4gICAgICpcclxuICAgICAqIEBwYXJhbSA8U291cmNlPiAgICAgICBUaGUgdHlwZSBvZiBzb3VyY2UgaXRlbXMuIEEgZGF0YWJhc2UtcmVjb3JkIGZvciBpbnN0YW5jZS5cclxuICAgICAqIEBwYXJhbSA8VGFyZ2V0PiAgICAgICBUaGUgdHlwZSBvZiB0YXJnZXQgaXRlbXMuIEEgW1tNYXF1ZXR0ZUNvbXBvbmVudF1dIGZvciBpbnN0YW5jZS5cclxuICAgICAqIEBwYXJhbSBnZXRTb3VyY2VLZXkgICBgZnVuY3Rpb24oc291cmNlKWAgdGhhdCBtdXN0IHJldHVybiBhIGtleSB0byBpZGVudGlmeSBlYWNoIHNvdXJjZSBvYmplY3QuIFRoZSByZXN1bHQgbXVzdCBlaXRoZXIgYmUgYSBzdHJpbmcgb3IgYSBudW1iZXIuXHJcbiAgICAgKiBAcGFyYW0gY3JlYXRlUmVzdWx0ICAgYGZ1bmN0aW9uKHNvdXJjZSwgaW5kZXgpYCB0aGF0IG11c3QgY3JlYXRlIGEgbmV3IHJlc3VsdCBvYmplY3QgZnJvbSBhIGdpdmVuIHNvdXJjZS4gVGhpcyBmdW5jdGlvbiBpcyBpZGVudGljYWxcclxuICAgICAqICAgICAgICAgICAgICAgICAgICAgICB0byB0aGUgYGNhbGxiYWNrYCBhcmd1bWVudCBpbiBgQXJyYXkubWFwKGNhbGxiYWNrKWAuXHJcbiAgICAgKiBAcGFyYW0gdXBkYXRlUmVzdWx0ICAgYGZ1bmN0aW9uKHNvdXJjZSwgdGFyZ2V0LCBpbmRleClgIHRoYXQgdXBkYXRlcyBhIHJlc3VsdCB0byBhbiB1cGRhdGVkIHNvdXJjZS5cclxuICAgICAqL1xyXG4gICAgdmFyIGNyZWF0ZU1hcHBpbmcgPSBmdW5jdGlvbiAoZ2V0U291cmNlS2V5LCBjcmVhdGVSZXN1bHQsIHVwZGF0ZVJlc3VsdCkge1xyXG4gICAgICAgIHZhciBrZXlzID0gW107XHJcbiAgICAgICAgdmFyIHJlc3VsdHMgPSBbXTtcclxuICAgICAgICByZXR1cm4ge1xyXG4gICAgICAgICAgICByZXN1bHRzOiByZXN1bHRzLFxyXG4gICAgICAgICAgICBtYXA6IGZ1bmN0aW9uIChuZXdTb3VyY2VzKSB7XHJcbiAgICAgICAgICAgICAgICB2YXIgbmV3S2V5cyA9IG5ld1NvdXJjZXMubWFwKGdldFNvdXJjZUtleSk7XHJcbiAgICAgICAgICAgICAgICB2YXIgb2xkVGFyZ2V0cyA9IHJlc3VsdHMuc2xpY2UoKTtcclxuICAgICAgICAgICAgICAgIHZhciBvbGRJbmRleCA9IDA7XHJcbiAgICAgICAgICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IG5ld1NvdXJjZXMubGVuZ3RoOyBpKyspIHtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgc291cmNlID0gbmV3U291cmNlc1tpXTtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgc291cmNlS2V5ID0gbmV3S2V5c1tpXTtcclxuICAgICAgICAgICAgICAgICAgICBpZiAoc291cmNlS2V5ID09PSBrZXlzW29sZEluZGV4XSkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICByZXN1bHRzW2ldID0gb2xkVGFyZ2V0c1tvbGRJbmRleF07XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHVwZGF0ZVJlc3VsdChzb3VyY2UsIG9sZFRhcmdldHNbb2xkSW5kZXhdLCBpKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgb2xkSW5kZXgrKztcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHZhciBmb3VuZCA9IGZhbHNlO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBmb3IgKHZhciBqID0gMTsgaiA8IGtleXMubGVuZ3RoICsgMTsgaisrKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB2YXIgc2VhcmNoSW5kZXggPSAob2xkSW5kZXggKyBqKSAlIGtleXMubGVuZ3RoO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgaWYgKGtleXNbc2VhcmNoSW5kZXhdID09PSBzb3VyY2VLZXkpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICByZXN1bHRzW2ldID0gb2xkVGFyZ2V0c1tzZWFyY2hJbmRleF07XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdXBkYXRlUmVzdWx0KG5ld1NvdXJjZXNbaV0sIG9sZFRhcmdldHNbc2VhcmNoSW5kZXhdLCBpKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBvbGRJbmRleCA9IHNlYXJjaEluZGV4ICsgMTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBmb3VuZCA9IHRydWU7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgYnJlYWs7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICAgICAgaWYgKCFmb3VuZCkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgcmVzdWx0c1tpXSA9IGNyZWF0ZVJlc3VsdChzb3VyY2UsIGkpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgcmVzdWx0cy5sZW5ndGggPSBuZXdTb3VyY2VzLmxlbmd0aDtcclxuICAgICAgICAgICAgICAgIGtleXMgPSBuZXdLZXlzO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfTtcclxuICAgIH07XG5cbiAgICBleHBvcnRzLmNyZWF0ZUNhY2hlID0gY3JlYXRlQ2FjaGU7XG4gICAgZXhwb3J0cy5jcmVhdGVNYXBwaW5nID0gY3JlYXRlTWFwcGluZztcbiAgICBleHBvcnRzLmNyZWF0ZVByb2plY3RvciA9IGNyZWF0ZVByb2plY3RvcjtcbiAgICBleHBvcnRzLmRvbSA9IGRvbTtcbiAgICBleHBvcnRzLmggPSBoO1xuXG4gICAgT2JqZWN0LmRlZmluZVByb3BlcnR5KGV4cG9ydHMsICdfX2VzTW9kdWxlJywgeyB2YWx1ZTogdHJ1ZSB9KTtcblxufSkpO1xuIl0sInNvdXJjZVJvb3QiOiIifQ==