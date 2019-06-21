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
        if(message.component_name == 'Plot' || message.component_name == '_PlotUpdater'){
            console.log('Attempting to mount ' + message.component_name);
        }
        //console.dir(this.cells["holding_pen"]);
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
                console.info(`Could not find component for ${message.component_name}`);
		var velement = this.htmlToVDomEl(message.contents, message.id);
	    } else {
		var component = new componentClass(
                    {
                        id: message.id,
                        extraData: message.extra_data
                    },
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
		    source = this.h("div", {id: replacementKey, class: 'shit'}, []);
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
        // after vdom insertion.
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
class AsyncDropdown extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
        this.addDropdownListener = this.addDropdownListener.bind(this);
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
                }, [this.getReplacementElementFor('contents')])
            ])
        );
    }

    addDropdownListener(element){
        let parentEl = element.parentElement;
        let firstTimeClicked = (element.dataset.firstclick == "true");
        let component = this;
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
}

/**
 * About Replacements
 * ------------------
 * This component has a single regular
 * replacement:
 * * `contents`
 */
class AsyncDropdownContent extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                id: `dropdownContent-${this.props.id}`,
                "data-cell-id": this.props.id,
                "data-cell-type": "AsyncDropdownContent"
            }, [this.getReplacementElementFor('contents')])
        );
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
class Badge extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(...args);
    }

    render(){
        return(
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('span', {
                class: `cell badge badge-${this.props.extraData.badgeStyle}`,
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Badge"
            }, [this.getReplacementElementFor('child')])
        );
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




class Button extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
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
            }, [this.getReplacementElementFor('contents')]
            ) 
        );
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
class ButtonGroup extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return(
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "ButtonGroup",
                class: "btn-group",
                "role": "group"
            }, this.getReplacementElementsFor('button')
             )
        );
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
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! maquette */ "./node_modules/maquette/dist/maquette.umd.js");
/* harmony import */ var maquette__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(maquette__WEBPACK_IMPORTED_MODULE_1__);
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
class Card extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeBody = this.makeBody.bind(this);
        this.makeHeader = this.makeHeader.bind(this);
    }

    render(){
        let bodyClass = 'card-body';
        if(this.props.extraData.padding){
            bodyClass = `card-body p-${this.props.extraData.padding}`;
        }
        let bodyArea = Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
            class: bodyClass
        }, [this.makeBody()]);
        let header = this.makeHeader();
        let headerArea = null;
        if(header){
            headerArea = Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {class: "card-header"}, [header]);
        }
        return Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div',
            {
                class: "cell card",
                style: this.props.extraData.divStyle,
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Card"
            }, [headerArea, bodyArea]);
    }

    makeBody(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('contents');
        } else {
            return this.renderChildNamed('contents');
        }
    }

    makeHeader(){
        if(this.usesReplacements){
            if(this.replacements.hasReplacement('header')){
                return this.getReplacementElementFor('header');
            }
        } else {
            return this.renderChildNamed('header');
        }
        return null;
    }
};

console.log('Card module loaded');



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
class CardTitle extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return(
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                id: this.props.id,
                class: "cell",
                "data-cell-id": this.props.id,
                "data-cell-type": "CardTitle"
            }, [
                this.getReplacementElementFor('contents')
            ])
        );
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
class Clickable extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
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
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {}, [this.getReplacementElementFor('contents')])
            ]
            )
        );
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

class CollapsiblePanel extends _Component_js__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);
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
                            this.getReplacementElementFor('panel')
                        ]),
                        Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {class: "col-sm"}, [
                            this.getReplacementElementFor('content')
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
                }, [this.getReplacementElementFor('content')])
            );
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
        return this.getReplacementElementsFor('c').map(replElement => {
            return (
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                    class: "col-sm"
                }, [replElement])
            );
        });
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
        this._updateProps(props);
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
class Container extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        let child = this.getReplacementElementFor('child');
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
class ContextualDisplay extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div',
            {
                class: "cell contextualDisplay",
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "ContextualDisplay"
            }, [this.getReplacementElementFor('child')]
        );
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
class Dropdown extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
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
                    this.getReplacementElementFor('title')
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

    makeItems(){
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





/**
 * About Replacements
 * ------------------
 * This component has two
 * regular replacements:
 * * `icon`
 * * `child`
 */
class Expands extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
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
                        [this.getReplacementElementFor('icon')]),
                    Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {style:'display:inline-block'},
                        [this.getReplacementElementFor('child')]),
                ]
            )
        );
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
 *
 * NOTE: Child is a 2-dimensional
 * enumerated replacement!
 */
class Grid extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
        this._makeHeaderElements = this._makeHeaderElements.bind(this);
        this._makeRowElements = this._makeRowElements.bind(this);
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
                    Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('tr', {}, [topTableHeader, ...this._makeHeaderElements()])
                ]),
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('tbody', {}, this._makeRowElements())
            ])
        );
    }

    _makeRowElements(){
        if (this.replacements.hasReplacement('child')) {
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
        } else {
            return []
        }
    }

    _makeHeaderElements(){
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
        if(this.replacements.hasReplacement('left')){
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
        if(this.replacements.hasReplacement('center')){
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
        if(this.replacements.hasReplacement('right')){
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
        return this.getReplacementElementsFor(position).map(element => {
            return (
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('span', {class: "flex-item px-3"}, [element])
            );
        });
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
class Main extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);
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
                    this.getReplacementElementFor('child')
                ])
            ])
        );
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
class Modal extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);
        this.mainStyle = 'display:block;padding-right:15px;';
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
                                this.getReplacementElementFor('title')
                            ])
                        ]),
                        Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {class: "modal-body"}, [
                            this.getReplacementElementFor('message')
                        ]),
                        Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {class: "modal-footer"}, [
                            this.getReplacementElementsFor('button')
                        ])
                    ])
                ])
            ])
        );
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
class Plot extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        this.setupPlot = this.setupPlot.bind(this);
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
                this.getReplacementElementFor('chart-updater'),
                this.getReplacementElementFor('error')
            ])
        );
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
class Popover extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div',
            {
                class: "cell",
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
                    [this.getReplacementElementFor('contents')]
                ),
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {style: "display:none"}, [
                    Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])("div", {id: "pop_" + this.props.id}, [
                        Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])("div", {class: "data-title"}, [this.getReplacementElementFor("title")]),
                        Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])("div", {class: "data-content"}, [
                            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])("div", {style: "width: " + this.props.width + "px"}, [
                                this.getReplacementElementFor('detail')]
                            )
                        ])
                    ])
                ])
            ]
        );
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
 * * `child`
 */
class RootCell extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "RootCell"
            }, [this.getReplacementElementFor('c')])
        );
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
class Scrollable extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Scrollable"
            }, [this.getReplacementElementFor('child')])
        );
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
class Sheet extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        this.currentTable = null;

        // Bind context to methods
        this.initializeTable = this.initializeTable.bind(this);
        this.initializeHooks = this.initializeHooks.bind(this);

        /**
         * WARINING: The Cell version of Sheet is still using
         * certian postscripts because we have not yet refactored
         * the socket protocol.
         * Remove this warning about it once that happens!
         **/
        console.warn(`[TODO] Sheet still uses certain postscripts in its interaction. See component constructor comment`);
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
                }, [this.getReplacementElementFor('error')])
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
class Subscribed extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);
        //this.addReplacement('contents', '_____contents__');
    }

    render(){
        return Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div',
            {
                class: "cell subscribed",
                style: this.props.extraData.divStyle,
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Subscribed"
            }, [this.getReplacementElementFor('contents')]
        );
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
class SubscribedSequence extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);
        //this.addReplacement('contents', '_____contents__');
        //
        // Bind context to methods
        this.makeClass = this.makeClass.bind(this);
        this.makeChildren = this.makeChildren.bind(this);
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

    makeClass() {
        if (this.props.extraData.asColumns) {
            return "cell subscribedSequence container-fluid";
        }
        return "cell subscribedSequence";
    }

    makeChildren(){
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
class Table extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
        this._makeHeaderElements = this._makeHeaderElements.bind(this);
        this._makeRowElements = this._makeRowElements.bind(this);
        this._makeFirstRowElement = this._makeFirstRowElement.bind(this);
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
                    this._makeFirstRowElement()
                ]),
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('tbody', {}, this._makeRowElements())
            ])
        );
    }

    _theadStyle(){
        return "border-bottom: black;border-bottom-style:solid;border-bottom-width:thin;";
    }

    _makeHeaderElements(){
        return this.getReplacementElementsFor('header').map((replacement, idx) => {
            return Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('th', {
                style: "vertical-align:top;",
                key: `${this.props.id}-table-header-${idx}`
            }, [replacement]);
        });
    }

    _makeRowElements(){
        // Note: rows are the *first* dimension
        // in the 2-dimensional array returned
        // by getting the `child` replacement elements.
        return this.getReplacementElementsFor('child').map((row, rowIdx) => {
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

    _makeFirstRowElement(){
        let headerElements = this._makeHeaderElements();
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
        return [
            this.getReplacementElementFor('left'),
            this.getReplacementElementFor('right'),
            this.getReplacementElementFor('page'),
        ];
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
class Tabs extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Tabs",
                class: "container-fluid mb-3"
            }, [
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('ul', {class: "nav nav-tabs", role: "tablist"}, [
                    this.getReplacementElementsFor('header')
                ]),
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {class: "tab-content"}, [
                    Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {class: "tab-pane fade show active", role: "tabpanel"}, [
                        this.getReplacementElementFor('display')
                    ])
                ])
            ])
        );
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
class  Traceback extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return (
            Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Traceback",
                class: "alert alert-primary"
            }, [
                Object(maquette__WEBPACK_IMPORTED_MODULE_1__["h"])('pre', {}, [this.getReplacementElementFor('child')])
            ])
        );
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
class _NavTab extends _Component__WEBPACK_IMPORTED_MODULE_0__["Component"] {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
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
                }, [this.getReplacementElementFor('child')])
            ])
        );
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly8vd2VicGFjay9ib290c3RyYXAiLCJ3ZWJwYWNrOi8vLy4vQ2VsbEhhbmRsZXIuanMiLCJ3ZWJwYWNrOi8vLy4vQ2VsbFNvY2tldC5qcyIsIndlYnBhY2s6Ly8vLi9Db21wb25lbnRSZWdpc3RyeS5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0FzeW5jRHJvcGRvd24uanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9CYWRnZS5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0J1dHRvbi5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0J1dHRvbkdyb3VwLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvQ2FyZC5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0NhcmRUaXRsZS5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0NpcmNsZUxvYWRlci5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0NsaWNrYWJsZS5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0NvZGUuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9Db2RlRWRpdG9yLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvQ29sbGFwc2libGVQYW5lbC5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0NvbHVtbnMuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9Db21wb25lbnQuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9Db250YWluZXIuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9Db250ZXh0dWFsRGlzcGxheS5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0Ryb3Bkb3duLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvRXhwYW5kcy5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL0dyaWQuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9IZWFkZXJCYXIuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9MYXJnZVBlbmRpbmdEb3dubG9hZERpc3BsYXkuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9Mb2FkQ29udGVudHNGcm9tVXJsLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvTWFpbi5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL01vZGFsLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvT2N0aWNvbi5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL1BhZGRpbmcuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9QbG90LmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvUG9wb3Zlci5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL1Jvb3RDZWxsLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvU2Nyb2xsYWJsZS5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL1NlcXVlbmNlLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvU2hlZXQuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9TaW5nbGVMaW5lVGV4dEJveC5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL1NwYW4uanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9TdWJzY3JpYmVkLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvU3Vic2NyaWJlZFNlcXVlbmNlLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvVGFibGUuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9UYWJzLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvVGV4dC5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL1RyYWNlYmFjay5qcyIsIndlYnBhY2s6Ly8vLi9jb21wb25lbnRzL19OYXZUYWIuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy9fUGxvdFVwZGF0ZXIuanMiLCJ3ZWJwYWNrOi8vLy4vY29tcG9uZW50cy91dGlsL1Byb3BlcnR5VmFsaWRhdG9yLmpzIiwid2VicGFjazovLy8uL2NvbXBvbmVudHMvdXRpbC9SZXBsYWNlbWVudHNIYW5kbGVyLmpzIiwid2VicGFjazovLy8uL21haW4uanMiLCJ3ZWJwYWNrOi8vLy4vbm9kZV9tb2R1bGVzL21hcXVldHRlL2Rpc3QvbWFxdWV0dGUudW1kLmpzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiI7QUFBQTtBQUNBOztBQUVBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTs7QUFFQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTs7O0FBR0E7QUFDQTs7QUFFQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBLGtEQUEwQyxnQ0FBZ0M7QUFDMUU7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQSxnRUFBd0Qsa0JBQWtCO0FBQzFFO0FBQ0EseURBQWlELGNBQWM7QUFDL0Q7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGlEQUF5QyxpQ0FBaUM7QUFDMUUsd0hBQWdILG1CQUFtQixFQUFFO0FBQ3JJO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0EsbUNBQTJCLDBCQUEwQixFQUFFO0FBQ3ZELHlDQUFpQyxlQUFlO0FBQ2hEO0FBQ0E7QUFDQTs7QUFFQTtBQUNBLDhEQUFzRCwrREFBK0Q7O0FBRXJIO0FBQ0E7OztBQUdBO0FBQ0E7Ozs7Ozs7Ozs7Ozs7QUNsRkE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRTJCOztBQUUzQjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNB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LFNBQVM7QUFDOUQ7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7O0FBRUEsS0FBSzs7QUFFTDtBQUNBO0FBQ0E7QUFDQTtBQUNBLGFBQWE7QUFDYixpQ0FBaUMsY0FBYyxNQUFNLFNBQVM7QUFDOUQ7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQSxLQUFLOztBQUVMO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsYUFBYTtBQUNiLGlDQUFpQyxjQUFjLE1BQU0sU0FBUztBQUM5RDtBQUNBO0FBQ0E7QUFDQSxTQUFTO0FBQ1Q7QUFDQTtBQUNBLEtBQUs7O0FBRUw7QUFDQTtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2IsaUNBQWlDLGNBQWMsTUFBTSxTQUFTO0FBQzlEO0FBQ0E7QUFDQTtBQUNBLFNBQVM7QUFDVDtBQUNBO0FBQ0EsS0FBSzs7QUFFTDtBQUNBO0FBQ0E7QUFDQTtBQUNBLGFBQWE7QUFDYixpQ0FBaUMsY0FBYyxNQUFNLFNBQVM7QUFDOUQ7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQSxLQUFLOztBQUVMOzs7Ozs7Ozs7Ozs7O0FDN09BO0FBQUE7QUFBQTtBQUFBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGFBQWE7QUFDYjtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUOztBQUVBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FB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O0FBQ0EsNERBQTREO0FBQzVEO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0Esd0NBQXdDLGNBQWMsRUFBRTtBQUN4RDtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGFBQWE7QUFDYjtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBLHdDQUF3QyxrQkFBa0IsRUFBRTtBQUM1RDtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxxREFBcUQsY0FBYztBQUNuRTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsNENBQTRDLDhCQUE4QjtBQUMxRTtBQUNBO0FBQ0EsNENBQTRDLG1DQUFtQztBQUMvRTtBQUNBO0FBQ0E7QUFDQTtBQUNBLDJGQUEyRiwrQkFBK0IsRUFBRTtBQUM1SCxTQUFTO0FBQ1Q7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSwyRUFBMkUsNkJBQTZCO0FBQ3hHO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsaUNBQWlDO0FBQ2pDO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLDZDQUE2QyxtQkFBbUI7QUFDaEU7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLHVCQUF1QjtBQUN2QjtBQUNBO0FBQ0E7QUFDQSwyQkFBMkIsd0JBQXdCO0FBQ25EO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2I7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsYUFBYTtBQUNiO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsYUFBYTtBQUNiO0FBQ0E7QUFDQSxhQUFhO0FBQ2I7QUFDQTtBQUNBLGFBQWE7QUFDYjtBQUNBO0FBQ0EsYUFBYTtBQUNiO0FBQ0E7QUFDQSxhQUFhO0FBQ2I7QUFDQSwrQkFBK0IsNEJBQTRCO0FBQzNEO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxhQUFhO0FBQ2I7QUFDQTtBQUNBLG1DQUFtQyxtQkFBbUI7QUFDdEQ7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQSxrQkFBa0IsY0FBYztBQUNoQyxZQUFZLGlFQUFpRTtBQUM3RTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsK0JBQStCLHVCQUF1QjtBQUN0RDtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSx1Q0FBdUMscUJBQXFCO0FBQzVEO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUEsa0RBQWtELGNBQWM7O0FBRWhFLENBQUMiLCJmaWxlIjoibWFpbi5idW5kbGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyIgXHQvLyBUaGUgbW9kdWxlIGNhY2hlXG4gXHR2YXIgaW5zdGFsbGVkTW9kdWxlcyA9IHt9O1xuXG4gXHQvLyBUaGUgcmVxdWlyZSBmdW5jdGlvblxuIFx0ZnVuY3Rpb24gX193ZWJwYWNrX3JlcXVpcmVfXyhtb2R1bGVJZCkge1xuXG4gXHRcdC8vIENoZWNrIGlmIG1vZHVsZSBpcyBpbiBjYWNoZVxuIFx0XHRpZihpbnN0YWxsZWRNb2R1bGVzW21vZHVsZUlkXSkge1xuIFx0XHRcdHJldHVybiBpbnN0YWxsZWRNb2R1bGVzW21vZHVsZUlkXS5leHBvcnRzO1xuIFx0XHR9XG4gXHRcdC8vIENyZWF0ZSBhIG5ldyBtb2R1bGUgKGFuZCBwdXQgaXQgaW50byB0aGUgY2FjaGUpXG4gXHRcdHZhciBtb2R1bGUgPSBpbnN0YWxsZWRNb2R1bGVzW21vZHVsZUlkXSA9IHtcbiBcdFx0XHRpOiBtb2R1bGVJZCxcbiBcdFx0XHRsOiBmYWxzZSxcbiBcdFx0XHRleHBvcnRzOiB7fVxuIFx0XHR9O1xuXG4gXHRcdC8vIEV4ZWN1dGUgdGhlIG1vZHVsZSBmdW5jdGlvblxuIFx0XHRtb2R1bGVzW21vZHVsZUlkXS5jYWxsKG1vZHVsZS5leHBvcnRzLCBtb2R1bGUsIG1vZHVsZS5leHBvcnRzLCBfX3dlYnBhY2tfcmVxdWlyZV9fKTtcblxuIFx0XHQvLyBGbGFnIHRoZSBtb2R1bGUgYXMgbG9hZGVkXG4gXHRcdG1vZHVsZS5sID0gdHJ1ZTtcblxuIFx0XHQvLyBSZXR1cm4gdGhlIGV4cG9ydHMgb2YgdGhlIG1vZHVsZVxuIFx0XHRyZXR1cm4gbW9kdWxlLmV4cG9ydHM7XG4gXHR9XG5cblxuIFx0Ly8gZXhwb3NlIHRoZSBtb2R1bGVzIG9iamVjdCAoX193ZWJwYWNrX21vZHVsZXNfXylcbiBcdF9fd2VicGFja19yZXF1aXJlX18ubSA9IG1vZHVsZXM7XG5cbiBcdC8vIGV4cG9zZSB0aGUgbW9kdWxlIGNhY2hlXG4gXHRfX3dlYnBhY2tfcmVxdWlyZV9fLmMgPSBpbnN0YWxsZWRNb2R1bGVzO1xuXG4gXHQvLyBkZWZpbmUgZ2V0dGVyIGZ1bmN0aW9uIGZvciBoYXJtb255IGV4cG9ydHNcbiBcdF9fd2VicGFja19yZXF1aXJlX18uZCA9IGZ1bmN0aW9uKGV4cG9ydHMsIG5hbWUsIGdldHRlcikge1xuIFx0XHRpZighX193ZWJwYWNrX3JlcXVpcmVfXy5vKGV4cG9ydHMsIG5hbWUpKSB7XG4gXHRcdFx0T2JqZWN0LmRlZmluZVByb3BlcnR5KGV4cG9ydHMsIG5hbWUsIHsgZW51bWVyYWJsZTogdHJ1ZSwgZ2V0OiBnZXR0ZXIgfSk7XG4gXHRcdH1cbiBcdH07XG5cbiBcdC8vIGRlZmluZSBfX2VzTW9kdWxlIG9uIGV4cG9ydHNcbiBcdF9fd2VicGFja19yZXF1aXJlX18uciA9IGZ1bmN0aW9uKGV4cG9ydHMpIHtcbiBcdFx0aWYodHlwZW9mIFN5bWJvbCAhPT0gJ3VuZGVmaW5lZCcgJiYgU3ltYm9sLnRvU3RyaW5nVGFnKSB7XG4gXHRcdFx0T2JqZWN0LmRlZmluZVByb3BlcnR5KGV4cG9ydHMsIFN5bWJvbC50b1N0cmluZ1RhZywgeyB2YWx1ZTogJ01vZHVsZScgfSk7XG4gXHRcdH1cbiBcdFx0T2JqZWN0LmRlZmluZVByb3BlcnR5KGV4cG9ydHMsICdfX2VzTW9kdWxlJywgeyB2YWx1ZTogdHJ1ZSB9KTtcbiBcdH07XG5cbiBcdC8vIGNyZWF0ZSBhIGZha2UgbmFtZXNwYWNlIG9iamVjdFxuIFx0Ly8gbW9kZSAmIDE6IHZhbHVlIGlzIGEgbW9kdWxlIGlkLCByZXF1aXJlIGl0XG4gXHQvLyBtb2RlICYgMjogbWVyZ2UgYWxsIHByb3BlcnRpZXMgb2YgdmFsdWUgaW50byB0aGUgbnNcbiBcdC8vIG1vZGUgJiA0OiByZXR1cm4gdmFsdWUgd2hlbiBhbHJlYWR5IG5zIG9iamVjdFxuIFx0Ly8gbW9kZSAmIDh8MTogYmVoYXZlIGxpa2UgcmVxdWlyZVxuIFx0X193ZWJwYWNrX3JlcXVpcmVfXy50ID0gZnVuY3Rpb24odmFsdWUsIG1vZGUpIHtcbiBcdFx0aWYobW9kZSAmIDEpIHZhbHVlID0gX193ZWJwYWNrX3JlcXVpcmVfXyh2YWx1ZSk7XG4gXHRcdGlmKG1vZGUgJiA4KSByZXR1cm4gdmFsdWU7XG4gXHRcdGlmKChtb2RlICYgNCkgJiYgdHlwZW9mIHZhbHVlID09PSAnb2JqZWN0JyAmJiB2YWx1ZSAmJiB2YWx1ZS5fX2VzTW9kdWxlKSByZXR1cm4gdmFsdWU7XG4gXHRcdHZhciBucyA9IE9iamVjdC5jcmVhdGUobnVsbCk7XG4gXHRcdF9fd2VicGFja19yZXF1aXJlX18ucihucyk7XG4gXHRcdE9iamVjdC5kZWZpbmVQcm9wZXJ0eShucywgJ2RlZmF1bHQnLCB7IGVudW1lcmFibGU6IHRydWUsIHZhbHVlOiB2YWx1ZSB9KTtcbiBcdFx0aWYobW9kZSAmIDIgJiYgdHlwZW9mIHZhbHVlICE9ICdzdHJpbmcnKSBmb3IodmFyIGtleSBpbiB2YWx1ZSkgX193ZWJwYWNrX3JlcXVpcmVfXy5kKG5zLCBrZXksIGZ1bmN0aW9uKGtleSkgeyByZXR1cm4gdmFsdWVba2V5XTsgfS5iaW5kKG51bGwsIGtleSkpO1xuIFx0XHRyZXR1cm4gbnM7XG4gXHR9O1xuXG4gXHQvLyBnZXREZWZhdWx0RXhwb3J0IGZ1bmN0aW9uIGZvciBjb21wYXRpYmlsaXR5IHdpdGggbm9uLWhhcm1vbnkgbW9kdWxlc1xuIFx0X193ZWJwYWNrX3JlcXVpcmVfXy5uID0gZnVuY3Rpb24obW9kdWxlKSB7XG4gXHRcdHZhciBnZXR0ZXIgPSBtb2R1bGUgJiYgbW9kdWxlLl9fZXNNb2R1bGUgP1xuIFx0XHRcdGZ1bmN0aW9uIGdldERlZmF1bHQoKSB7IHJldHVybiBtb2R1bGVbJ2RlZmF1bHQnXTsgfSA6XG4gXHRcdFx0ZnVuY3Rpb24gZ2V0TW9kdWxlRXhwb3J0cygpIHsgcmV0dXJuIG1vZHVsZTsgfTtcbiBcdFx0X193ZWJwYWNrX3JlcXVpcmVfXy5kKGdldHRlciwgJ2EnLCBnZXR0ZXIpO1xuIFx0XHRyZXR1cm4gZ2V0dGVyO1xuIFx0fTtcblxuIFx0Ly8gT2JqZWN0LnByb3RvdHlwZS5oYXNPd25Qcm9wZXJ0eS5jYWxsXG4gXHRfX3dlYnBhY2tfcmVxdWlyZV9fLm8gPSBmdW5jdGlvbihvYmplY3QsIHByb3BlcnR5KSB7IHJldHVybiBPYmplY3QucHJvdG90eXBlLmhhc093blByb3BlcnR5LmNhbGwob2JqZWN0LCBwcm9wZXJ0eSk7IH07XG5cbiBcdC8vIF9fd2VicGFja19wdWJsaWNfcGF0aF9fXG4gXHRfX3dlYnBhY2tfcmVxdWlyZV9fLnAgPSBcIlwiO1xuXG5cbiBcdC8vIExvYWQgZW50cnkgbW9kdWxlIGFuZCByZXR1cm4gZXhwb3J0c1xuIFx0cmV0dXJuIF9fd2VicGFja19yZXF1aXJlX18oX193ZWJwYWNrX3JlcXVpcmVfXy5zID0gXCIuL21haW4uanNcIik7XG4iLCIvKipcbiAqIENlbGxIYW5kbGVyIFByaW1hcnkgTWVzc2FnZSBIYW5kbGVyXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY2xhc3MgaW1wbGVtZW50cyBhIHNlcnZpY2UgdGhhdCBoYW5kbGVzXG4gKiBtZXNzYWdlcyBvZiBhbGwga2luZHMgdGhhdCBjb21lIGluIG92ZXIgYVxuICogYENlbGxTb2NrZXRgLlxuICogTk9URTogRm9yIHRoZSBtb21lbnQgdGhlcmUgYXJlIG9ubHkgdHdvIGtpbmRzXG4gKiBvZiBtZXNzYWdlcyBhbmQgdGhlcmVmb3JlIHR3byBoYW5kbGVycy4gV2UgaGF2ZVxuICogcGxhbnMgdG8gY2hhbmdlIHRoaXMgc3RydWN0dXJlIHRvIGJlIG1vcmUgZmxleGlibGVcbiAqIGFuZCBzbyB0aGUgQVBJIG9mIHRoaXMgY2xhc3Mgd2lsbCBjaGFuZ2UgZ3JlYXRseS5cbiAqL1xuXG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuY2xhc3MgQ2VsbEhhbmRsZXIge1xuICAgIGNvbnN0cnVjdG9yKGgsIHByb2plY3RvciwgY29tcG9uZW50cyl7XG5cdC8vIHByb3BzXG5cdHRoaXMuaCA9IGg7XG5cdHRoaXMucHJvamVjdG9yID0gcHJvamVjdG9yO1xuXHR0aGlzLmNvbXBvbmVudHMgPSBjb21wb25lbnRzO1xuXG5cdC8vIEluc3RhbmNlIFByb3BzXG4gICAgICAgIHRoaXMucG9zdHNjcmlwdHMgPSBbXTtcbiAgICAgICAgdGhpcy5jZWxscyA9IHt9O1xuXHR0aGlzLkRPTVBhcnNlciA9IG5ldyBET01QYXJzZXIoKTtcblxuICAgICAgICAvLyBCaW5kIEluc3RhbmNlIE1ldGhvZHNcbiAgICAgICAgdGhpcy5zaG93Q29ubmVjdGlvbkNsb3NlZCA9IHRoaXMuc2hvd0Nvbm5lY3Rpb25DbG9zZWQuYmluZCh0aGlzKTtcblx0dGhpcy5jb25uZWN0aW9uQ2xvc2VkVmlldyA9IHRoaXMuY29ubmVjdGlvbkNsb3NlZFZpZXcuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5oYW5kbGVQb3N0c2NyaXB0ID0gdGhpcy5oYW5kbGVQb3N0c2NyaXB0LmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuaGFuZGxlTWVzc2FnZSA9IHRoaXMuaGFuZGxlTWVzc2FnZS5iaW5kKHRoaXMpO1xuXG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogRmlsbHMgdGhlIHBhZ2UncyBwcmltYXJ5IGRpdiB3aXRoXG4gICAgICogYW4gaW5kaWNhdG9yIHRoYXQgdGhlIHNvY2tldCBoYXMgYmVlblxuICAgICAqIGRpc2Nvbm5lY3RlZC5cbiAgICAgKi9cbiAgICBzaG93Q29ubmVjdGlvbkNsb3NlZCgpe1xuXHR0aGlzLnByb2plY3Rvci5yZXBsYWNlKFxuXHQgICAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoXCJwYWdlX3Jvb3RcIiksXG5cdCAgICB0aGlzLmNvbm5lY3Rpb25DbG9zZWRWaWV3XG5cdCk7XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogQ29udmVuaWVuY2UgbWV0aG9kIHRoYXQgdXBkYXRlc1xuICAgICAqIEJvb3RzdHJhcC1zdHlsZSBwb3BvdmVycyBvblxuICAgICAqIHRoZSBET00uXG4gICAgICogU2VlIGlubGluZSBjb21tZW50c1xuICAgICAqL1xuICAgIHVwZGF0ZVBvcG92ZXJzKCkge1xuICAgICAgICAvLyBUaGlzIGZ1bmN0aW9uIHJlcXVpcmVzXG4gICAgICAgIC8vIGpRdWVyeSBhbmQgcGVyaGFwcyBkb2Vzbid0XG4gICAgICAgIC8vIGJlbG9uZyBpbiB0aGlzIGNsYXNzLlxuICAgICAgICAvLyBUT0RPOiBGaWd1cmUgb3V0IGEgYmV0dGVyIHdheVxuICAgICAgICAvLyBBTFNPIE5PVEU6XG4gICAgICAgIC8vIC0tLS0tLS0tLS0tLS0tLS0tXG4gICAgICAgIC8vIGBnZXRDaGlsZFByb3BgIGlzIGEgY29uc3QgZnVuY3Rpb25cbiAgICAgICAgLy8gdGhhdCBpcyBkZWNsYXJlZCBpbiBhIHNlcGFyYXRlXG4gICAgICAgIC8vIHNjcmlwdCB0YWcgYXQgdGhlIGJvdHRvbSBvZlxuICAgICAgICAvLyBwYWdlLmh0bWwuIFRoYXQncyBhIG5vLW5vIVxuICAgICAgICAkKCdbZGF0YS10b2dnbGU9XCJwb3BvdmVyXCJdJykucG9wb3Zlcih7XG4gICAgICAgICAgICBodG1sOiB0cnVlLFxuICAgICAgICAgICAgY29udGFpbmVyOiAnYm9keScsXG4gICAgICAgICAgICB0aXRsZTogZnVuY3Rpb24gKCkge1xuICAgICAgICAgICAgICAgIHJldHVybiBnZXRDaGlsZFByb3AodGhpcywgJ3RpdGxlJyk7XG4gICAgICAgICAgICB9LFxuICAgICAgICAgICAgY29udGVudDogZnVuY3Rpb24gKCkge1xuICAgICAgICAgICAgICAgIHJldHVybiBnZXRDaGlsZFByb3AodGhpcywgJ2NvbnRlbnQnKTtcbiAgICAgICAgICAgIH0sXG4gICAgICAgICAgICBwbGFjZW1lbnQ6IGZ1bmN0aW9uIChwb3BwZXJFbCwgdHJpZ2dlcmluZ0VsKSB7XG4gICAgICAgICAgICAgICAgbGV0IHBsYWNlbWVudCA9IHRyaWdnZXJpbmdFbC5kYXRhc2V0LnBsYWNlbWVudDtcbiAgICAgICAgICAgICAgICBpZihwbGFjZW1lbnQgPT0gdW5kZWZpbmVkKXtcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIFwiYm90dG9tXCI7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIHJldHVybiBwbGFjZW1lbnQ7XG4gICAgICAgICAgICB9XG4gICAgICAgIH0pO1xuICAgICAgICAkKCcucG9wb3Zlci1kaXNtaXNzJykucG9wb3Zlcih7XG4gICAgICAgICAgICB0cmlnZ2VyOiAnZm9jdXMnXG4gICAgICAgIH0pO1xuICAgIH1cblxuICAgIC8qKlxuICAgICAqIFByaW1hcnkgbWV0aG9kIGZvciBoYW5kbGluZ1xuICAgICAqICdwb3N0c2NyaXB0cycgbWVzc2FnZXMsIHdoaWNoIHRlbGxcbiAgICAgKiB0aGlzIG9iamVjdCB0byBnbyB0aHJvdWdoIGl0J3MgYXJyYXlcbiAgICAgKiBvZiBzY3JpcHQgc3RyaW5ncyBhbmQgdG8gZXZhbHVhdGUgdGhlbS5cbiAgICAgKiBUaGUgZXZhbHVhdGlvbiBpcyBkb25lIG9uIHRoZSBnbG9iYWxcbiAgICAgKiB3aW5kb3cgb2JqZWN0IGV4cGxpY2l0bHkuXG4gICAgICogTk9URTogRnV0dXJlIHJlZmFjdG9yaW5ncy9yZXN0cnVjdHVyaW5nc1xuICAgICAqIHdpbGwgcmVtb3ZlIG11Y2ggb2YgdGhlIG5lZWQgdG8gY2FsbCBldmFsIVxuICAgICAqIEBwYXJhbSB7c3RyaW5nfSBtZXNzYWdlIC0gVGhlIGluY29taW5nIHN0cmluZ1xuICAgICAqIGZyb20gdGhlIHNvY2tldC5cbiAgICAgKi9cbiAgICBoYW5kbGVQb3N0c2NyaXB0KG1lc3NhZ2Upe1xuICAgICAgICAvLyBFbHNld2hlcmUsIHVwZGF0ZSBwb3BvdmVycyBmaXJzdFxuICAgICAgICAvLyBOb3cgd2UgZXZhbHVhdGUgc2NyaXB0cyBjb21pbmdcbiAgICAgICAgLy8gYWNyb3NzIHRoZSB3aXJlLlxuICAgICAgICB0aGlzLnVwZGF0ZVBvcG92ZXJzKCk7XG4gICAgICAgIHdoaWxlKHRoaXMucG9zdHNjcmlwdHMubGVuZ3RoKXtcblx0ICAgIGxldCBwb3N0c2NyaXB0ID0gdGhpcy5wb3N0c2NyaXB0cy5wb3AoKTtcblx0ICAgIHRyeSB7XG5cdFx0d2luZG93LmV2YWwocG9zdHNjcmlwdCk7XG5cdCAgICB9IGNhdGNoKGUpe1xuICAgICAgICAgICAgICAgIGNvbnNvbGUuZXJyb3IoXCJFUlJPUiBSVU5OSU5HIFBPU1RTQ1JJUFRcIiwgZSk7XG4gICAgICAgICAgICAgICAgY29uc29sZS5sb2cocG9zdHNjcmlwdCk7XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBQcmltYXJ5IG1ldGhvZCBmb3IgaGFuZGxpbmcgJ25vcm1hbCdcbiAgICAgKiAoaWUgbm9uLXBvc3RzY3JpcHRzKSBtZXNzYWdlcyB0aGF0IGhhdmVcbiAgICAgKiBiZWVuIGRlc2VyaWFsaXplZCBmcm9tIEpTT04uXG4gICAgICogRm9yIHRoZSBtb21lbnQsIHRoZXNlIG1lc3NhZ2VzIGRlYWxcbiAgICAgKiBlbnRpcmVseSB3aXRoIERPTSByZXBsYWNlbWVudCBvcGVyYXRpb25zLCB3aGljaFxuICAgICAqIHRoaXMgbWV0aG9kIGltcGxlbWVudHMuXG4gICAgICogQHBhcmFtIHtvYmplY3R9IG1lc3NhZ2UgLSBBIGRlc2VyaWFsaXplZFxuICAgICAqIEpTT04gbWVzc2FnZSBmcm9tIHRoZSBzZXJ2ZXIgdGhhdCBoYXNcbiAgICAgKiBpbmZvcm1hdGlvbiBhYm91dCBlbGVtZW50cyB0aGF0IG5lZWQgdG9cbiAgICAgKiBiZSB1cGRhdGVkLlxuICAgICAqL1xuICAgIGhhbmRsZU1lc3NhZ2UobWVzc2FnZSl7XG4gICAgICAgIGxldCBuZXdDb21wb25lbnRzID0gW107XG4gICAgICAgIGlmKG1lc3NhZ2UuY29tcG9uZW50X25hbWUgPT0gJ1Bsb3QnIHx8IG1lc3NhZ2UuY29tcG9uZW50X25hbWUgPT0gJ19QbG90VXBkYXRlcicpe1xuICAgICAgICAgICAgY29uc29sZS5sb2coJ0F0dGVtcHRpbmcgdG8gbW91bnQgJyArIG1lc3NhZ2UuY29tcG9uZW50X25hbWUpO1xuICAgICAgICB9XG4gICAgICAgIC8vY29uc29sZS5kaXIodGhpcy5jZWxsc1tcImhvbGRpbmdfcGVuXCJdKTtcblx0aWYodGhpcy5jZWxsc1tcInBhZ2Vfcm9vdFwiXSA9PSB1bmRlZmluZWQpe1xuICAgICAgICAgICAgdGhpcy5jZWxsc1tcInBhZ2Vfcm9vdFwiXSA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKFwicGFnZV9yb290XCIpO1xuICAgICAgICAgICAgdGhpcy5jZWxsc1tcImhvbGRpbmdfcGVuXCJdID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoXCJob2xkaW5nX3BlblwiKTtcbiAgICAgICAgfVxuXHQvLyBXaXRoIHRoZSBleGNlcHRpb24gb2YgYHBhZ2Vfcm9vdGAgYW5kIGBob2xkaW5nX3BlbmAgaWQgbm9kZXMsIGFsbFxuXHQvLyBlbGVtZW50cyBpbiB0aGlzLmNlbGxzIGFyZSB2aXJ0dWFsLiBEZXBlbmRpZyBvbiB3aGV0aGVyIHdlIGFyZSBhZGRpbmcgYVxuXHQvLyBuZXcgbm9kZSwgb3IgbWFuaXB1bGF0aW5nIGFuIGV4aXN0aW5nLCB3ZSBuZWVlZCB0byB3b3JrIHdpdGggdGhlIHVuZGVybHlpbmdcblx0Ly8gRE9NIG5vZGUuIEhlbmNlIGlmIHRoaXMuY2VsbFttZXNzYWdlLmlkXSBpcyBhIHZkb20gZWxlbWVudCB3ZSB1c2UgaXRzXG5cdC8vIHVuZGVybHlpbmcgZG9tTm9kZSBlbGVtZW50IHdoZW4gaW4gb3BlcmF0aW9ucyBsaWtlIHRoaXMucHJvamVjdG9yLnJlcGxhY2UoKVxuXHRsZXQgY2VsbCA9IHRoaXMuY2VsbHNbbWVzc2FnZS5pZF07XG5cblx0aWYgKGNlbGwgIT09IHVuZGVmaW5lZCAmJiBjZWxsLmRvbU5vZGUgIT09IHVuZGVmaW5lZCkge1xuXHQgICAgY2VsbCA9IGNlbGwuZG9tTm9kZTtcblx0fVxuXG4gICAgICAgIGlmKG1lc3NhZ2UuZGlzY2FyZCAhPT0gdW5kZWZpbmVkKXtcbiAgICAgICAgICAgIC8vIEluIHRoZSBjYXNlIHdoZXJlIHdlIGhhdmUgcmVjZWl2ZWQgYSAnZGlzY2FyZCcgbWVzc2FnZSxcbiAgICAgICAgICAgIC8vIGJ1dCB0aGUgY2VsbCByZXF1ZXN0ZWQgaXMgbm90IGF2YWlsYWJsZSBpbiBvdXJcbiAgICAgICAgICAgIC8vIGNlbGxzIGNvbGxlY3Rpb24sIHdlIHNpbXBseSBkaXNwbGF5IGEgd2FybmluZzpcbiAgICAgICAgICAgIGlmKGNlbGwgPT0gdW5kZWZpbmVkKXtcbiAgICAgICAgICAgICAgICBjb25zb2xlLndhcm4oYFJlY2VpdmVkIGRpc2NhcmQgbWVzc2FnZSBmb3Igbm9uLWV4aXN0aW5nIGNlbGwgaWQgJHttZXNzYWdlLmlkfWApO1xuICAgICAgICAgICAgICAgIHJldHVybjtcbiAgICAgICAgICAgIH1cblx0ICAgIC8vIEluc3RlYWQgb2YgcmVtb3ZpbmcgdGhlIG5vZGUgd2UgcmVwbGFjZSB3aXRoIHRoZSBhXG5cdCAgICAvLyBgZGlzcGxheTpub25lYCBzdHlsZSBub2RlIHdoaWNoIGVmZmVjdGl2ZWx5IHJlbW92ZXMgaXRcblx0ICAgIC8vIGZyb20gdGhlIERPTVxuXHQgICAgaWYgKGNlbGwucGFyZW50Tm9kZSAhPT0gbnVsbCkge1xuXHRcdHRoaXMucHJvamVjdG9yLnJlcGxhY2UoY2VsbCwgKCkgPT4ge1xuXHRcdCAgICByZXR1cm4gaChcImRpdlwiLCB7c3R5bGU6IFwiZGlzcGxheTpub25lXCJ9LCBbXSk7XG5cdFx0fSk7XG5cdCAgICB9XG5cdH0gZWxzZSBpZihtZXNzYWdlLmlkICE9PSB1bmRlZmluZWQpe1xuXHQgICAgLy8gQSBkaWN0aW9uYXJ5IG9mIGlkcyB3aXRoaW4gdGhlIG9iamVjdCB0byByZXBsYWNlLlxuXHQgICAgLy8gVGFyZ2V0cyBhcmUgcmVhbCBpZHMgb2Ygb3RoZXIgb2JqZWN0cy5cblx0ICAgIGxldCByZXBsYWNlbWVudHMgPSBtZXNzYWdlLnJlcGxhY2VtZW50cztcblxuXHQgICAgLy8gVE9ETzogdGhpcyBpcyBhIHRlbXBvcmFyeSBicmFuY2hpbmcsIHRvIGJlIHJlbW92ZWQgd2l0aCBhIG1vcmUgbG9naWNhbCBzZXR1cC4gQXNcblx0ICAgIC8vIG9mIHdyaXRpbmcgaWYgdGhlIG1lc3NhZ2UgY29taW5nIGFjcm9zcyBpcyBzZW5kaW5nIGEgXCJrbm93blwiIGNvbXBvbmVudCB0aGVuIHdlIHVzZVxuXHQgICAgLy8gdGhlIGNvbXBvbmVudCBpdHNlbGYgYXMgb3Bwb3NlZCB0byBidWlsZGluZyBhIHZkb20gZWxlbWVudCBmcm9tIHRoZSByYXcgaHRtbFxuXHQgICAgbGV0IGNvbXBvbmVudENsYXNzID0gdGhpcy5jb21wb25lbnRzW21lc3NhZ2UuY29tcG9uZW50X25hbWVdO1xuXHQgICAgaWYgKGNvbXBvbmVudENsYXNzID09PSB1bmRlZmluZWQpIHtcbiAgICAgICAgICAgICAgICBjb25zb2xlLmluZm8oYENvdWxkIG5vdCBmaW5kIGNvbXBvbmVudCBmb3IgJHttZXNzYWdlLmNvbXBvbmVudF9uYW1lfWApO1xuXHRcdHZhciB2ZWxlbWVudCA9IHRoaXMuaHRtbFRvVkRvbUVsKG1lc3NhZ2UuY29udGVudHMsIG1lc3NhZ2UuaWQpO1xuXHQgICAgfSBlbHNlIHtcblx0XHR2YXIgY29tcG9uZW50ID0gbmV3IGNvbXBvbmVudENsYXNzKFxuICAgICAgICAgICAgICAgICAgICB7XG4gICAgICAgICAgICAgICAgICAgICAgICBpZDogbWVzc2FnZS5pZCxcbiAgICAgICAgICAgICAgICAgICAgICAgIGV4dHJhRGF0YTogbWVzc2FnZS5leHRyYV9kYXRhXG4gICAgICAgICAgICAgICAgICAgIH0sXG4gICAgICAgICAgICAgICAgICAgIG1lc3NhZ2UucmVwbGFjZW1lbnRfa2V5c1xuICAgICAgICAgICAgICAgICk7XG4gICAgICAgICAgICAgICAgdmFyIHZlbGVtZW50ID0gY29tcG9uZW50LnJlbmRlcigpO1xuICAgICAgICAgICAgICAgIG5ld0NvbXBvbmVudHMucHVzaChjb21wb25lbnQpO1xuXHQgICAgfVxuXG4gICAgICAgICAgICAvLyBJbnN0YWxsIHRoZSBlbGVtZW50IGludG8gdGhlIGRvbVxuICAgICAgICAgICAgaWYoY2VsbCA9PT0gdW5kZWZpbmVkKXtcblx0XHQvLyBUaGlzIGlzIGEgdG90YWxseSBuZXcgbm9kZS5cbiAgICAgICAgICAgICAgICAvLyBGb3IgdGhlIG1vbWVudCwgYWRkIGl0IHRvIHRoZVxuICAgICAgICAgICAgICAgIC8vIGhvbGRpbmcgcGVuLlxuXHRcdHRoaXMucHJvamVjdG9yLmFwcGVuZCh0aGlzLmNlbGxzW1wiaG9sZGluZ19wZW5cIl0sICgpID0+IHtcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIHZlbGVtZW50O1xuICAgICAgICAgICAgICAgIH0pO1xuXG5cdFx0dGhpcy5jZWxsc1ttZXNzYWdlLmlkXSA9IHZlbGVtZW50O1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICAvLyBSZXBsYWNlIHRoZSBleGlzdGluZyBjb3B5IG9mXG4gICAgICAgICAgICAgICAgLy8gdGhlIG5vZGUgd2l0aCB0aGlzIGluY29taW5nXG4gICAgICAgICAgICAgICAgLy8gY29weS5cblx0XHRpZihjZWxsLnBhcmVudE5vZGUgPT09IG51bGwpe1xuXHRcdCAgICB0aGlzLnByb2plY3Rvci5hcHBlbmQodGhpcy5jZWxsc1tcImhvbGRpbmdfcGVuXCJdLCAoKSA9PiB7XG5cdFx0XHRyZXR1cm4gdmVsZW1lbnQ7XG5cdFx0ICAgIH0pO1xuXHRcdH0gZWxzZSB7XG5cdFx0ICAgIHRoaXMucHJvamVjdG9yLnJlcGxhY2UoY2VsbCwgKCkgPT4ge3JldHVybiB2ZWxlbWVudDt9KTtcblx0XHR9XG5cdCAgICB9XG5cbiAgICAgICAgICAgIHRoaXMuY2VsbHNbbWVzc2FnZS5pZF0gPSB2ZWxlbWVudDtcblxuICAgICAgICAgICAgLy8gTm93IHdpcmUgaW4gcmVwbGFjZW1lbnRzXG4gICAgICAgICAgICBPYmplY3Qua2V5cyhyZXBsYWNlbWVudHMpLmZvckVhY2goKHJlcGxhY2VtZW50S2V5LCBpZHgpID0+IHtcbiAgICAgICAgICAgICAgICBsZXQgdGFyZ2V0ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQocmVwbGFjZW1lbnRLZXkpO1xuICAgICAgICAgICAgICAgIGxldCBzb3VyY2UgPSBudWxsO1xuICAgICAgICAgICAgICAgIGlmKHRoaXMuY2VsbHNbcmVwbGFjZW1lbnRzW3JlcGxhY2VtZW50S2V5XV0gPT09IHVuZGVmaW5lZCl7XG5cdFx0ICAgIC8vIFRoaXMgaXMgYWN0dWFsbHkgYSBuZXcgbm9kZS5cbiAgICAgICAgICAgICAgICAgICAgLy8gV2UnbGwgZGVmaW5lIGl0IGxhdGVyIGluIHRoZVxuICAgICAgICAgICAgICAgICAgICAvLyBldmVudCBzdHJlYW0uXG5cdFx0ICAgIHNvdXJjZSA9IHRoaXMuaChcImRpdlwiLCB7aWQ6IHJlcGxhY2VtZW50S2V5LCBjbGFzczogJ3NoaXQnfSwgW10pO1xuICAgICAgICAgICAgICAgICAgICB0aGlzLmNlbGxzW3JlcGxhY2VtZW50c1tyZXBsYWNlbWVudEtleV1dID0gc291cmNlOyBcblx0XHQgICAgdGhpcy5wcm9qZWN0b3IuYXBwZW5kKHRoaXMuY2VsbHNbXCJob2xkaW5nX3BlblwiXSwgKCkgPT4ge1xuXHRcdFx0cmV0dXJuIHNvdXJjZTtcbiAgICAgICAgICAgICAgICAgICAgfSk7XG5cdFx0fSBlbHNlIHtcbiAgICAgICAgICAgICAgICAgICAgLy8gTm90IGEgbmV3IG5vZGVcbiAgICAgICAgICAgICAgICAgICAgc291cmNlID0gdGhpcy5jZWxsc1tyZXBsYWNlbWVudHNbcmVwbGFjZW1lbnRLZXldXTtcbiAgICAgICAgICAgICAgICB9XG5cbiAgICAgICAgICAgICAgICBpZih0YXJnZXQgIT0gbnVsbCl7XG5cdFx0ICAgIHRoaXMucHJvamVjdG9yLnJlcGxhY2UodGFyZ2V0LCAoKSA9PiB7XG5cdFx0XHRyZXR1cm4gc291cmNlO1xuICAgICAgICAgICAgICAgICAgICB9KTtcbiAgICAgICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgICAgICBsZXQgZXJyb3JNc2cgPSBgSW4gbWVzc2FnZSAke21lc3NhZ2V9IGNvdWxkbid0IGZpbmQgJHtyZXBsYWNlbWVudEtleX1gO1xuICAgICAgICAgICAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoZXJyb3JNc2cpO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH0pO1xuICAgICAgICB9XG5cbiAgICAgICAgaWYobWVzc2FnZS5wb3N0c2NyaXB0ICE9PSB1bmRlZmluZWQpe1xuICAgICAgICAgICAgdGhpcy5wb3N0c2NyaXB0cy5wdXNoKG1lc3NhZ2UucG9zdHNjcmlwdCk7XG4gICAgICAgIH1cblxuICAgICAgICAvLyBJZiB3ZSBjcmVhdGVkIGFueSBuZXcgY29tcG9uZW50cyBkdXJpbmcgdGhpc1xuICAgICAgICAvLyBtZXNzYWdlIGhhbmRsaW5nIHNlc3Npb24sIHdlIGZpbmFsbHkgY2FsbFxuICAgICAgICAvLyB0aGVpciBgY29tcG9uZW50RGlkTG9hZGAgbGlmZWN5Y2xlIG1ldGhvZHNcbiAgICAgICAgbmV3Q29tcG9uZW50cy5mb3JFYWNoKGNvbXBvbmVudCA9PiB7XG4gICAgICAgICAgICBjb21wb25lbnQuY29tcG9uZW50RGlkTG9hZCgpO1xuICAgICAgICB9KTtcblxuICAgICAgICAvLyBSZW1vdmUgbGVmdG92ZXIgcmVwbGFjZW1lbnQgZGl2c1xuICAgICAgICAvLyB0aGF0IGFyZSBzdGlsbCBpbiB0aGUgcGFnZV9yb290XG4gICAgICAgIC8vIGFmdGVyIHZkb20gaW5zZXJ0aW9uLlxuICAgICAgICBsZXQgcGFnZVJvb3QgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncGFnZV9yb290Jyk7XG4gICAgICAgIGxldCBmb3VuZCA9IHBhZ2VSb290LnF1ZXJ5U2VsZWN0b3JBbGwoJ1tpZCo9XCJfX19fX1wiXScpO1xuICAgICAgICBmb3VuZC5mb3JFYWNoKGVsZW1lbnQgPT4ge1xuICAgICAgICAgICAgZWxlbWVudC5yZW1vdmUoKTtcbiAgICAgICAgfSk7XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogSGVscGVyIGZ1bmN0aW9uIHRoYXQgZ2VuZXJhdGVzIHRoZSB2ZG9tIE5vZGUgZm9yXG4gICAgICogdG8gYmUgZGlzcGxheSB3aGVuIGNvbm5lY3Rpb24gY2xvc2VzXG4gICAgICovXG4gICAgY29ubmVjdGlvbkNsb3NlZFZpZXcoKXtcblx0cmV0dXJuIHRoaXMuaChcIm1haW4uY29udGFpbmVyXCIsIHtyb2xlOiBcIm1haW5cIn0sIFtcblx0ICAgIHRoaXMuaChcImRpdlwiLCB7Y2xhc3M6IFwiYWxlcnQgYWxlcnQtcHJpbWFyeSBjZW50ZXItYmxvY2sgbXQtNVwifSxcblx0XHRbXCJEaXNjb25uZWN0ZWRcIl0pXG5cdF0pO1xuICAgIH1cblxuICAgICAgICAvKipcbiAgICAgKiBUaGlzIGlzIGEgKGhvcGVmdWxseSB0ZW1wb3JhcnkpIGhhY2tcbiAgICAgKiB0aGF0IHdpbGwgaW50ZXJjZXB0IHRoZSBmaXJzdCB0aW1lIGFcbiAgICAgKiBkcm9wZG93biBjYXJhdCBpcyBjbGlja2VkIGFuZCBiaW5kXG4gICAgICogQm9vdHN0cmFwIERyb3Bkb3duIGV2ZW50IGhhbmRsZXJzXG4gICAgICogdG8gaXQgdGhhdCBzaG91bGQgYmUgYm91bmQgdG8gdGhlXG4gICAgICogaWRlbnRpZmllZCBjZWxsLiBXZSBhcmUgZm9yY2VkIHRvIGRvIHRoaXNcbiAgICAgKiBiZWNhdXNlIHRoZSBjdXJyZW50IENlbGxzIGluZnJhc3RydWN0dXJlXG4gICAgICogZG9lcyBub3QgaGF2ZSBmbGV4aWJsZSBldmVudCBiaW5kaW5nL2hhbmRsaW5nLlxuICAgICAqIEBwYXJhbSB7c3RyaW5nfSBjZWxsSWQgLSBUaGUgSUQgb2YgdGhlIGNlbGxcbiAgICAgKiB0byBpZGVudGlmeSBpbiB0aGUgc29ja2V0IGNhbGxiYWNrIHdlIHdpbGxcbiAgICAgKiBiaW5kIHRvIG9wZW4gYW5kIGNsb3NlIGV2ZW50cyBvbiBkcm9wZG93blxuICAgICAqL1xuICAgIGRyb3Bkb3duSW5pdGlhbEJpbmRGb3IoY2VsbElkKXtcbiAgICAgICAgbGV0IGVsZW1lbnRJZCA9IGNlbGxJZCArICctZHJvcGRvd25NZW51QnV0dG9uJztcbiAgICAgICAgbGV0IGVsZW1lbnQgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZChlbGVtZW50SWQpO1xuICAgICAgICBpZighZWxlbWVudCl7XG4gICAgICAgICAgICB0aHJvdyBFcnJvcignRWxlbWVudCBvZiBpZCAnICsgZWxlbWVudElkICsgJyBkb2VzbnQgZXhpc3QhJyk7XG4gICAgICAgIH1cbiAgICAgICAgbGV0IGRyb3Bkb3duTWVudSA9IGVsZW1lbnQucGFyZW50RWxlbWVudDtcbiAgICAgICAgbGV0IGZpcnN0VGltZUNsaWNrZWQgPSBlbGVtZW50LmRhdGFzZXQuZmlyc3RjbGljayA9PSAndHJ1ZSc7XG4gICAgICAgIGlmKGZpcnN0VGltZUNsaWNrZWQpe1xuICAgICAgICAgICAgJChkcm9wZG93bk1lbnUpLm9uKCdzaG93LmJzLmRyb3Bkb3duJywgZnVuY3Rpb24oKXtcbiAgICAgICAgICAgICAgICBjZWxsU29ja2V0LnNlbmRTdHJpbmcoSlNPTi5zdHJpbmdpZnkoe1xuICAgICAgICAgICAgICAgICAgICBldmVudDogJ2Ryb3Bkb3duJyxcbiAgICAgICAgICAgICAgICAgICAgdGFyZ2V0X2NlbGw6IGNlbGxJZCxcbiAgICAgICAgICAgICAgICAgICAgaXNPcGVuOiBmYWxzZVxuICAgICAgICAgICAgICAgIH0pKTtcbiAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgJChkcm9wZG93bk1lbnUpLm9uKCdoaWRlLmJzLmRyb3Bkb3duJywgZnVuY3Rpb24oKXtcbiAgICAgICAgICAgICAgICBjZWxsU29ja2V0LnNlbmRTdHJpbmcoSlNPTi5zdHJpbmdpZnkoe1xuICAgICAgICAgICAgICAgICAgICBldmVudDogJ2Ryb3Bkb3duJyxcbiAgICAgICAgICAgICAgICAgICAgdGFyZ2V0X2NlbGw6IGNlbGxJZCxcbiAgICAgICAgICAgICAgICAgICAgaXNPcGVuOiB0cnVlXG4gICAgICAgICAgICAgICAgfSkpO1xuICAgICAgICAgICAgfSk7XG5cbiAgICAgICAgICAgIC8vIE5vdyBleHBpcmUgdGhlIGZpcnN0IHRpbWUgY2xpY2tlZFxuICAgICAgICAgICAgZWxlbWVudC5kYXRhc2V0LmZpcnN0Y2xpY2sgPSAnZmFsc2UnO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogVW5zYWZlbHkgZXhlY3V0ZXMgYW55IHBhc3NlZCBpbiBzdHJpbmdcbiAgICAgKiBhcyBpZiBpdCBpcyB2YWxpZCBKUyBhZ2FpbnN0IHRoZSBnbG9iYWxcbiAgICAgKiB3aW5kb3cgc3RhdGUuXG4gICAgICovXG4gICAgc3RhdGljIHVuc2FmZWx5RXhlY3V0ZShhU3RyaW5nKXtcbiAgICAgICAgd2luZG93LmV4ZWMoYVN0cmluZyk7XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogSGVscGVyIGZ1bmN0aW9uIHRoYXQgdGFrZXMgc29tZSBpbmNvbWluZ1xuICAgICAqIEhUTUwgc3RyaW5nIGFuZCByZXR1cm5zIGEgbWFxdWV0dGUgaHlwZXJzY3JpcHRcbiAgICAgKiBWRE9NIGVsZW1lbnQgZnJvbSBpdC5cbiAgICAgKiBUaGlzIHVzZXMgdGhlIGludGVybmFsIGJyb3dzZXIgRE9NcGFyc2VyKCkgdG8gZ2VuZXJhdGUgdGhlIGh0bWxcbiAgICAgKiBzdHJ1Y3R1cmUgZnJvbSB0aGUgcmF3IHN0cmluZyBhbmQgdGhlbiByZWN1cnNpdmVseSBidWlsZCB0aGUgXG4gICAgICogVkRPTSBlbGVtZW50XG4gICAgICogQHBhcmFtIHtzdHJpbmd9IGh0bWwgLSBUaGUgbWFya3VwIHRvXG4gICAgICogdHJhbnNmb3JtIGludG8gYSByZWFsIGVsZW1lbnQuXG4gICAgICovXG4gICAgaHRtbFRvVkRvbUVsKGh0bWwsIGlkKXtcblx0bGV0IGRvbSA9IHRoaXMuRE9NUGFyc2VyLnBhcnNlRnJvbVN0cmluZyhodG1sLCBcInRleHQvaHRtbFwiKTtcbiAgICAgICAgbGV0IGVsZW1lbnQgPSBkb20uYm9keS5jaGlsZHJlblswXTtcbiAgICAgICAgcmV0dXJuIHRoaXMuX2RvbUVsVG9WZG9tRWwoZWxlbWVudCwgaWQpO1xuICAgIH1cblxuICAgIF9kb21FbFRvVmRvbUVsKGRvbUVsLCBpZCkge1xuXHRsZXQgdGFnTmFtZSA9IGRvbUVsLnRhZ05hbWUudG9Mb2NhbGVMb3dlckNhc2UoKTtcblx0bGV0IGF0dHJzID0ge2lkOiBpZH07XG5cdGxldCBpbmRleDtcblxuXHRmb3IgKGluZGV4ID0gMDsgaW5kZXggPCBkb21FbC5hdHRyaWJ1dGVzLmxlbmd0aDsgaW5kZXgrKyl7XG5cdCAgICBsZXQgaXRlbSA9IGRvbUVsLmF0dHJpYnV0ZXMuaXRlbShpbmRleCk7XG5cdCAgICBhdHRyc1tpdGVtLm5hbWVdID0gaXRlbS52YWx1ZS50cmltKCk7XG5cdH1cblx0XG5cdGlmIChkb21FbC5jaGlsZEVsZW1lbnRDb3VudCA9PT0gMCkge1xuXHQgICAgcmV0dXJuIGgodGFnTmFtZSwgYXR0cnMsIFtkb21FbC50ZXh0Q29udGVudF0pO1xuXHR9XG5cdFxuXHRsZXQgY2hpbGRyZW4gPSBbXTtcblx0Zm9yIChpbmRleCA9IDA7IGluZGV4IDwgZG9tRWwuY2hpbGRyZW4ubGVuZ3RoOyBpbmRleCsrKXtcblx0ICAgIGxldCBjaGlsZCA9IGRvbUVsLmNoaWxkcmVuW2luZGV4XTtcblx0ICAgIGNoaWxkcmVuLnB1c2godGhpcy5fZG9tRWxUb1Zkb21FbChjaGlsZCkpO1xuXHR9XG5cdFxuXHRyZXR1cm4gaCh0YWdOYW1lLCBhdHRycywgY2hpbGRyZW4pO1xuICAgIH1cbn1cblxuXG5leHBvcnQge0NlbGxIYW5kbGVyLCBDZWxsSGFuZGxlciBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogQSBjb25jcmV0ZSBlcnJvciB0aHJvd25cbiAqIGlmIHRoZSBjdXJyZW50IGJyb3dzZXIgZG9lc24ndFxuICogc3VwcG9ydCB3ZWJzb2NrZXRzLCB3aGljaCBpcyB2ZXJ5XG4gKiB1bmxpa2VseS5cbiAqL1xuY2xhc3MgV2Vic29ja2V0Tm90U3VwcG9ydGVkIGV4dGVuZHMgRXJyb3Ige1xuICAgIGNvbnN0cnVjdG9yKGFyZ3Mpe1xuICAgICAgICBzdXBlcihhcmdzKTtcbiAgICB9XG59XG5cbi8qKlxuICogVGhpcyBpcyB0aGUgZ2xvYmFsIGZyYW1lXG4gKiBjb250cm9sLiBXZSBtaWdodCBjb25zaWRlclxuICogcHV0dGluZyBpdCBlbHNld2hlcmUsIGJ1dFxuICogYENlbGxTb2NrZXRgIGlzIGl0cyBvbmx5XG4gKiBjb25zdW1lci5cbiAqL1xuY29uc3QgRlJBTUVTX1BFUl9BQ0sgPSAxMDtcblxuXG4vKipcbiAqIENlbGxTb2NrZXQgQ29udHJvbGxlclxuICogLS0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBUaGlzIGNsYXNzIGltcGxlbWVudHMgYW4gaW5zdGFuY2Ugb2ZcbiAqIGEgY29udHJvbGxlciB0aGF0IHdyYXBzIGEgd2Vic29ja2V0IGNsaWVudFxuICogY29ubmVjdGlvbiBhbmQga25vd3MgaG93IHRvIGhhbmRsZSB0aGVcbiAqIGluaXRpYWwgcm91dGluZyBvZiBtZXNzYWdlcyBhY3Jvc3MgdGhlIHNvY2tldC5cbiAqIGBDZWxsU29ja2V0YCBpbnN0YW5jZXMgYXJlIGRlc2lnbmVkIHNvIHRoYXRcbiAqIGhhbmRsZXJzIGZvciBzcGVjaWZpYyB0eXBlcyBvZiBtZXNzYWdlcyBjYW5cbiAqIHJlZ2lzdGVyIHRoZW1zZWx2ZXMgd2l0aCBpdC5cbiAqIE5PVEU6IEZvciB0aGUgbW9tZW50LCBtb3N0IG9mIHRoaXMgY29kZVxuICogaGFzIGJlZW4gY29waWVkIHZlcmJhdGltIGZyb20gdGhlIGlubGluZVxuICogc2NyaXB0cyB3aXRoIG9ubHkgc2xpZ2h0IG1vZGlmaWNhdGlvbi5cbiAqKi9cbmNsYXNzIENlbGxTb2NrZXQge1xuICAgIGNvbnN0cnVjdG9yKCl7XG4gICAgICAgIC8vIEluc3RhbmNlIFByb3BzXG4gICAgICAgIHRoaXMudXJpID0gdGhpcy5nZXRVcmkoKTtcbiAgICAgICAgdGhpcy5zb2NrZXQgPSBudWxsO1xuICAgICAgICB0aGlzLmN1cnJlbnRCdWZmZXIgPSB7XG4gICAgICAgICAgICByZW1haW5pbmc6IG51bGwsXG4gICAgICAgICAgICBidWZmZXI6IG51bGwsXG4gICAgICAgICAgICBoYXNEaXNwbGF5OiBmYWxzZVxuICAgICAgICB9O1xuXG4gICAgICAgIC8qKlxuICAgICAgICAgKiBBIGNhbGxiYWNrIGZvciBoYW5kbGluZyBtZXNzYWdlc1xuICAgICAgICAgKiB0aGF0IGFyZSAncG9zdHNjcmlwdHMnXG4gICAgICAgICAqIEBjYWxsYmFjayBwb3N0c2NyaXB0c0hhbmRsZXJcbiAgICAgICAgICogQHBhcmFtIHtzdHJpbmd9IG1zZyAtIFRoZSBmb3J3YXJkZWQgbWVzc2FnZVxuICAgICAgICAgKi9cbiAgICAgICAgdGhpcy5wb3N0c2NyaXB0c0hhbmRlciA9IG51bGw7XG5cbiAgICAgICAgLyoqXG4gICAgICAgICAqIEEgY2FsbGJhY2sgZm9yIGhhbmRsaW5nIG1lc3NhZ2VzXG4gICAgICAgICAqIHRoYXQgYXJlIG5vcm1hbCBKU09OIGRhdGEgbWVzc2FnZXMuXG4gICAgICAgICAqIEBjYWxsYmFjayBtZXNzYWdlSGFuZGxlclxuICAgICAgICAgKiBAcGFyYW0ge29iamVjdH0gbXNnIC0gVGhlIGZvcndhcmRlZCBtZXNzYWdlXG4gICAgICAgICAqL1xuICAgICAgICB0aGlzLm1lc3NhZ2VIYW5kbGVyID0gbnVsbDtcblxuICAgICAgICAvKipcbiAgICAgICAgICogQSBjYWxsYmFjayBmb3IgaGFuZGxpbmcgbWVzc2FnZXNcbiAgICAgICAgICogd2hlbiB0aGUgd2Vic29ja2V0IGNvbm5lY3Rpb24gY2xvc2VzLlxuICAgICAgICAgKiBAY2FsbGJhY2sgY2xvc2VIYW5kbGVyXG4gICAgICAgICAqL1xuICAgICAgICB0aGlzLmNsb3NlSGFuZGxlciA9IG51bGw7XG5cbiAgICAgICAgLyoqXG4gICAgICAgICAqIEEgY2FsbGJhY2sgZm9yIGhhbmRsaW5nIG1lc3NhZ2VzXG4gICAgICAgICAqIHdoZW50IHRoZSBzb2NrZXQgZXJyb3JzXG4gICAgICAgICAqIEBjYWxsYmFjayBlcnJvckhhbmRsZXJcbiAgICAgICAgICovXG4gICAgICAgIHRoaXMuZXJyb3JIYW5kbGVyID0gbnVsbDtcblxuICAgICAgICAvLyBCaW5kIEluc3RhbmNlIE1ldGhvZHNcbiAgICAgICAgdGhpcy5jb25uZWN0ID0gdGhpcy5jb25uZWN0LmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuc2VuZFN0cmluZyA9IHRoaXMuc2VuZFN0cmluZy5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLmhhbmRsZVJhd01lc3NhZ2UgPSB0aGlzLmhhbmRsZVJhd01lc3NhZ2UuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5vblBvc3RzY3JpcHRzID0gdGhpcy5vblBvc3RzY3JpcHRzLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMub25NZXNzYWdlID0gdGhpcy5vbk1lc3NhZ2UuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5vbkNsb3NlID0gdGhpcy5vbkNsb3NlLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMub25FcnJvciA9IHRoaXMub25FcnJvci5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIC8qKlxuICAgICAqIFJldHVybnMgYSBwcm9wZXJseSBmb3JtYXR0ZWQgVVJJXG4gICAgICogZm9yIHRoZSBzb2NrZXQgZm9yIGFueSBnaXZlbiBjdXJyZW50XG4gICAgICogYnJvd3NlciBsb2NhdGlvbi5cbiAgICAgKiBAcmV0dXJucyB7c3RyaW5nfSBBIFVSSSBzdHJpbmcuXG4gICAgICovXG4gICAgZ2V0VXJpKCl7XG4gICAgICAgIGxldCBsb2NhdGlvbiA9IHdpbmRvdy5sb2NhdGlvbjtcbiAgICAgICAgbGV0IHVyaSA9IFwiXCI7XG4gICAgICAgIGlmKGxvY2F0aW9uLnByb3RvY29sID09PSBcImh0dHBzOlwiKXtcbiAgICAgICAgICAgIHVyaSArPSBcIndzczpcIjtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHVyaSArPSBcIndzOlwiO1xuICAgICAgICB9XG4gICAgICAgIHVyaSA9IGAke3VyaX0vLyR7bG9jYXRpb24uaG9zdH1gO1xuICAgICAgICB1cmkgPSBgJHt1cml9L3NvY2tldCR7bG9jYXRpb24ucGF0aG5hbWV9JHtsb2NhdGlvbi5zZWFyY2h9YDtcbiAgICAgICAgcmV0dXJuIHVyaTtcbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBUZWxscyB0aGlzIG9iamVjdCdzIGludGVybmFsIHdlYnNvY2tldFxuICAgICAqIHRvIGluc3RhbnRpYXRlIGl0c2VsZiBhbmQgY29ubmVjdCB0b1xuICAgICAqIHRoZSBwcm92aWRlZCBVUkkuIFRoZSBVUkkgd2lsbCBiZSBzZXQgdG9cbiAgICAgKiB0aGlzIGluc3RhbmNlJ3MgYHVyaWAgcHJvcGVydHkgZmlyc3QuIElmIG5vXG4gICAgICogdXJpIGlzIHBhc3NlZCwgYGNvbm5lY3QoKWAgd2lsbCB1c2UgdGhlIGN1cnJlbnRcbiAgICAgKiBhdHRyaWJ1dGUncyB2YWx1ZS5cbiAgICAgKiBAcGFyYW0ge3N0cmluZ30gdXJpIC0gQSAgVVJJIHRvIGNvbm5lY3QgdGhlIHNvY2tldFxuICAgICAqIHRvLlxuICAgICAqL1xuICAgIGNvbm5lY3QodXJpKXtcbiAgICAgICAgaWYodXJpKXtcbiAgICAgICAgICAgIHRoaXMudXJpID0gdXJpO1xuICAgICAgICB9XG4gICAgICAgIGlmKHdpbmRvdy5XZWJTb2NrZXQpe1xuICAgICAgICAgICAgdGhpcy5zb2NrZXQgPSBuZXcgV2ViU29ja2V0KHRoaXMudXJpKTtcbiAgICAgICAgfSBlbHNlIGlmKHdpbmRvdy5Nb3pXZWJTb2NrZXQpe1xuICAgICAgICAgICAgdGhpcy5zb2NrZXQgPSBNb3pXZWJTb2NrZXQodGhpcy51cmkpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgdGhyb3cgbmV3IFdlYnNvY2tldE5vdFN1cHBvcnRlZCgpO1xuICAgICAgICB9XG5cbiAgICAgICAgdGhpcy5zb2NrZXQub25jbG9zZSA9IHRoaXMuY2xvc2VIYW5kbGVyO1xuICAgICAgICB0aGlzLnNvY2tldC5vbm1lc3NhZ2UgPSB0aGlzLmhhbmRsZVJhd01lc3NhZ2UuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5zb2NrZXQub25lcnJvciA9IHRoaXMuZXJyb3JIYW5kbGVyO1xuICAgIH1cblxuICAgIC8qKlxuICAgICAqIENvbnZlbmllbmNlIG1ldGhvZCB0aGF0IHNlbmRzIHRoZSBwYXNzZWRcbiAgICAgKiBzdHJpbmcgb24gdGhpcyBpbnN0YW5jZSdzIHVuZGVybHlpbmdcbiAgICAgKiB3ZWJzb2tldCBjb25uZWN0aW9uLlxuICAgICAqIEBwYXJhbSB7c3RyaW5nfSBhU3RyaW5nIC0gQSBzdHJpbmcgdG8gc2VuZFxuICAgICAqL1xuICAgIHNlbmRTdHJpbmcoYVN0cmluZyl7XG4gICAgICAgIGlmKHRoaXMuc29ja2V0KXtcbiAgICAgICAgICAgIHRoaXMuc29ja2V0LnNlbmQoYVN0cmluZyk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICAvLyBJZGVhbGx5IHdlIG1vdmUgdGhlIGRvbSBvcGVyYXRpb25zIG9mXG4gICAgLy8gdGhpcyBmdW5jdGlvbiBvdXQgaW50byBhbm90aGVyIGNsYXNzIG9yXG4gICAgLy8gY29udGV4dC5cbiAgICAvKipcbiAgICAgKiBVc2luZyB0aGUgaW50ZXJuYWwgYGN1cnJlbnRCdWZmZXJgLCB0aGlzXG4gICAgICogbWV0aG9kIGNoZWNrcyB0byBzZWUgaWYgYSBsYXJnZSBtdWx0aS1mcmFtZVxuICAgICAqIHBpZWNlIG9mIHdlYnNvY2tldCBkYXRhIGlzIGJlaW5nIHNlbnQuIElmIHNvLFxuICAgICAqIGl0IHByZXNlbnRzIGFuZCB1cGRhdGVzIGEgc3BlY2lmaWMgZGlzcGxheSBpblxuICAgICAqIHRoZSBET00gd2l0aCB0aGUgY3VycmVudCBwZXJjZW50YWdlIGV0Yy5cbiAgICAgKiBAcGFyYW0ge3N0cmluZ30gbXNnIC0gVGhlIG1lc3NhZ2UgdG9cbiAgICAgKiBkaXNwbGF5IGluc2lkZSB0aGUgZWxlbWVudFxuICAgICAqL1xuICAgIHNldExhcmdlRG93bmxvYWREaXNwbGF5KG1zZyl7XG5cbiAgICAgICAgaWYobXNnLmxlbmd0aCA9PSAwICYmICF0aGlzLmN1cnJlbnRCdWZmZXIuaGFzRGlzcGxheSl7XG4gICAgICAgICAgICByZXR1cm47XG4gICAgICAgIH1cblxuICAgICAgICB0aGlzLmN1cnJlbnRCdWZmZXIuaGFzRGlzcGxheSA9IChtc2cubGVuZ3RoICE9IDApO1xuXG4gICAgICAgIGxldCBlbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoXCJvYmplY3RfZGF0YWJhc2VfbGFyZ2VfcGVuZGluZ19kb3dubG9hZF90ZXh0XCIpO1xuICAgICAgICBpZihlbGVtZW50ICE9IHVuZGVmaW5lZCl7XG4gICAgICAgICAgICBlbGVtZW50LmlubmVySFRNTCA9IG1zZztcbiAgICAgICAgfVxuICAgIH1cblxuICAgIC8qKlxuICAgICAqIEhhbmRsZXMgdGhlIGBvbm1lc3NhZ2VgIGV2ZW50IG9mIHRoZSB1bmRlcmx5aW5nXG4gICAgICogd2Vic29ja2V0LlxuICAgICAqIFRoaXMgbWV0aG9kIGtub3dzIGhvdyB0byBmaWxsIHRoZSBpbnRlcm5hbFxuICAgICAqIGJ1ZmZlciAodG8gZ2V0IGFyb3VuZCB0aGUgZnJhbWUgbGltaXQpIGFuZCBvbmx5XG4gICAgICogdHJpZ2dlciBzdWJzZXF1ZW50IGhhbmRsZXJzIGZvciBpbmNvbWluZyBtZXNzYWdlcy5cbiAgICAgKiBUT0RPOiBCcmVhayBvdXQgdGhpcyBtZXRob2QgYSBiaXQgbW9yZS4gSXQgaGFzIGJlZW5cbiAgICAgKiBjb3BpZWQgbmVhcmx5IHZlcmJhdGltIGZyb20gdGhlIG9yaWdpbmFsIGNvZGUuXG4gICAgICogTk9URTogRm9yIG5vdywgdGhlcmUgYXJlIG9ubHkgdHdvIHR5cGVzIG9mIG1lc3NhZ2VzOlxuICAgICAqICAgICAgICd1cGRhdGVzJyAod2UganVzdCBjYWxsIHRoZXNlIG1lc3NhZ2VzKVxuICAgICAqICAgICAgICdwb3N0c2NyaXB0cycgKHRoZXNlIGFyZSBqdXN0IHJhdyBub24tSlNPTiBzdHJpbmdzKVxuICAgICAqIElmIGEgYnVmZmVyIGlzIGNvbXBsZXRlLCB0aGlzIG1ldGhvZCB3aWxsIGNoZWNrIHRvIHNlZSBpZlxuICAgICAqIGhhbmRsZXJzIGFyZSByZWdpc3RlcmVkIGZvciBwb3N0c2NyaXB0L25vcm1hbCBtZXNzYWdlc1xuICAgICAqIGFuZCB3aWxsIHRyaWdnZXIgdGhlbSBpZiB0cnVlIGluIGVpdGhlciBjYXNlLCBwYXNzaW5nXG4gICAgICogYW55IHBhcnNlZCBKU09OIGRhdGEgdG8gdGhlIGNhbGxiYWNrcy5cbiAgICAgKiBAcGFyYW0ge0V2ZW50fSBldmVudCAtIFRoZSBgb25tZXNzYWdlYCBldmVudCBvYmplY3RcbiAgICAgKiBmcm9tIHRoZSBzb2NrZXQuXG4gICAgICovXG4gICAgaGFuZGxlUmF3TWVzc2FnZShldmVudCl7XG4gICAgICAgIGlmKHRoaXMuY3VycmVudEJ1ZmZlci5yZW1haW5pbmcgPT09IG51bGwpe1xuICAgICAgICAgICAgdGhpcy5jdXJyZW50QnVmZmVyLnJlbWFpbmluZyA9IEpTT04ucGFyc2UoZXZlbnQuZGF0YSk7XG4gICAgICAgICAgICB0aGlzLmN1cnJlbnRCdWZmZXIuYnVmZmVyID0gW107XG4gICAgICAgICAgICBpZih0aGlzLmN1cnJlbnRCdWZmZXIuaGFzRGlzcGxheSAmJiB0aGlzLmN1cnJlbnRCdWZmZXIucmVtYWluaW5nID09IDEpe1xuICAgICAgICAgICAgICAgIC8vIFNFVCBMQVJHRSBET1dOTE9BRCBESVNQTEFZXG4gICAgICAgICAgICB9XG4gICAgICAgICAgICByZXR1cm47XG4gICAgICAgIH1cblxuICAgICAgICB0aGlzLmN1cnJlbnRCdWZmZXIucmVtYWluaW5nIC09IDE7XG4gICAgICAgIHRoaXMuY3VycmVudEJ1ZmZlci5idWZmZXIucHVzaChldmVudC5kYXRhKTtcblxuICAgICAgICBpZih0aGlzLmN1cnJlbnRCdWZmZXIuYnVmZmVyLmxlbmd0aCAlIEZSQU1FU19QRVJfQUNLID09IDApe1xuICAgICAgICAgICAgLy9BQ0sgZXZlcnkgdGVudGggbWVzc2FnZS4gV2UgaGF2ZSB0byBkbyBhY3RpdmUgcHVzaGJhY2tcbiAgICAgICAgICAgIC8vYmVjYXVzZSB0aGUgd2Vic29ja2V0IGRpc2Nvbm5lY3RzIG9uIENocm9tZSBpZiB5b3UgamFtIHRvb1xuICAgICAgICAgICAgLy9tdWNoIGluIGF0IG9uY2VcbiAgICAgICAgICAgIHRoaXMuc2VuZFN0cmluZyhcbiAgICAgICAgICAgICAgICBKU09OLnN0cmluZ2lmeSh7XG4gICAgICAgICAgICAgICAgICAgIFwiQUNLXCI6IHRoaXMuY3VycmVudEJ1ZmZlci5idWZmZXIubGVuZ3RoXG4gICAgICAgICAgICAgICAgfSkpO1xuICAgICAgICAgICAgbGV0IHBlcmNlbnRhZ2UgPSBNYXRoLnJvdW5kKDEwMCp0aGlzLmN1cnJlbnRCdWZmZXIuYnVmZmVyLmxlbmd0aCAvICh0aGlzLmN1cnJlbnRCdWZmZXIucmVtYWluaW5nICsgdGhpcy5jdXJyZW50QnVmZmVyLmJ1ZmZlci5sZW5ndGgpKTtcbiAgICAgICAgICAgIGxldCB0b3RhbCA9IE1hdGgucm91bmQoKHRoaXMuY3VycmVudEJ1ZmZlci5yZW1haW5pbmcgKyB0aGlzLmN1cnJlbnRCdWZmZXIuYnVmZmVyLmxlbmd0aCkgLyAoMTAyNCAvIDMyKSk7XG4gICAgICAgICAgICBsZXQgcHJvZ3Jlc3NTdHIgPSBgKERvd25sb2FkZWQgJHtwZXJjZW50YWdlfSUgb2YgJHt0b3RhbH0gTUIpYDtcbiAgICAgICAgICAgIHRoaXMuc2V0TGFyZ2VEb3dubG9hZERpc3BsYXkocHJvZ3Jlc3NTdHIpO1xuICAgICAgICB9XG5cbiAgICAgICAgaWYodGhpcy5jdXJyZW50QnVmZmVyLnJlbWFpbmluZyA+IDApe1xuICAgICAgICAgICAgcmV0dXJuO1xuICAgICAgICB9XG5cbiAgICAgICAgdGhpcy5zZXRMYXJnZURvd25sb2FkRGlzcGxheShcIlwiKTtcblxuICAgICAgICBsZXQgam9pbmVkQnVmZmVyID0gdGhpcy5jdXJyZW50QnVmZmVyLmJ1ZmZlci5qb2luKCcnKVxuXG4gICAgICAgIHRoaXMuY3VycmVudEJ1ZmZlci5yZW1haW5pbmcgPSBudWxsO1xuICAgICAgICB0aGlzLmN1cnJlbnRCdWZmZXIuYnVmZmVyID0gbnVsbDtcblxuICAgICAgICBsZXQgdXBkYXRlID0gSlNPTi5wYXJzZShqb2luZWRCdWZmZXIpO1xuXG4gICAgICAgIGlmKHVwZGF0ZSA9PSAncmVxdWVzdF9hY2snKSB7XG4gICAgICAgICAgICB0aGlzLnNlbmRTdHJpbmcoSlNPTi5zdHJpbmdpZnkoeydBQ0snOiAwfSkpXG4gICAgICAgIH0gZWxzZSBpZih1cGRhdGUgPT0gJ3Bvc3RzY3JpcHRzJyl7XG4gICAgICAgICAgICAvLyB1cGRhdGVQb3BvdmVycygpO1xuICAgICAgICAgICAgaWYodGhpcy5wb3N0c2NyaXB0c0hhbmRsZXIpe1xuICAgICAgICAgICAgICAgIHRoaXMucG9zdHNjcmlwdHNIYW5kbGVyKHVwZGF0ZSk7XG4gICAgICAgICAgICB9XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICBpZih0aGlzLm1lc3NhZ2VIYW5kbGVyKXtcbiAgICAgICAgICAgICAgICB0aGlzLm1lc3NhZ2VIYW5kbGVyKHVwZGF0ZSk7XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBDb252ZW5pZW5jZSBtZXRob2QgdGhhdCBiaW5kc1xuICAgICAqIHRoZSBwYXNzZWQgY2FsbGJhY2sgdG8gdGhpcyBpbnN0YW5jZSdzXG4gICAgICogcG9zdHNjcmlwdHNIYW5kbGVyLCB3aGljaCBpcyBzb21lIG1ldGhvZFxuICAgICAqIHRoYXQgaGFuZGxlcyBtZXNzYWdlcyBmb3IgcG9zdHNjcmlwdHMuXG4gICAgICogQHBhcmFtIHtwb3N0c2NyaXB0c0hhbmRsZXJ9IGNhbGxiYWNrIC0gQSBoYW5kbGVyXG4gICAgICogY2FsbGJhY2sgbWV0aG9kIHdpdGggdGhlIG1lc3NhZ2UgYXJndW1lbnQuXG4gICAgICovXG4gICAgb25Qb3N0c2NyaXB0cyhjYWxsYmFjayl7XG4gICAgICAgIHRoaXMucG9zdHNjcmlwdHNIYW5kbGVyID0gY2FsbGJhY2s7XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogQ29udmVuaWVuY2UgbWV0aG9kIHRoYXQgYmluZHNcbiAgICAgKiB0aGUgcGFzc2VkIGNhbGxiYWNrIHRvIHRoaXMgaW5zdGFuY2Unc1xuICAgICAqIHBvc3RzY3JpcHRzSGFuZGxlciwgd2hpY2ggaXMgc29tZSBtZXRob2RcbiAgICAgKiB0aGF0IGhhbmRsZXMgbWVzc2FnZXMgZm9yIHBvc3RzY3JpcHRzLlxuICAgICAqIEBwYXJhbSB7bWVzc2FnZUhhbmRsZXJ9IGNhbGxiYWNrIC0gQSBoYW5kbGVyXG4gICAgICogY2FsbGJhY2sgbWV0aG9kIHdpdGggdGhlIG1lc3NhZ2UgYXJndW1lbnQuXG4gICAgICovXG4gICAgb25NZXNzYWdlKGNhbGxiYWNrKXtcbiAgICAgICAgdGhpcy5tZXNzYWdlSGFuZGxlciA9IGNhbGxiYWNrO1xuICAgIH1cblxuICAgIC8qKlxuICAgICAqIENvbnZlbmllbmNlIG1ldGhvZCB0aGF0IGJpbmRzIHRoZVxuICAgICAqIHBhc3NlZCBjYWxsYmFjayB0byB0aGUgdW5kZXJseWluZ1xuICAgICAqIHdlYnNvY2tldCdzIGBvbmNsb3NlYCBoYW5kbGVyLlxuICAgICAqIEBwYXJhbSB7Y2xvc2VIYW5kbGVyfSBjYWxsYmFjayAtIEEgZnVuY3Rpb25cbiAgICAgKiB0aGF0IGhhbmRsZXMgY2xvc2UgZXZlbnRzIG9uIHRoZSBzb2NrZXQuXG4gICAgICovXG4gICAgb25DbG9zZShjYWxsYmFjayl7XG4gICAgICAgIHRoaXMuY2xvc2VIYW5kbGVyID0gY2FsbGJhY2s7XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogQ29udmVuaWVuY2UgbWV0aG9kIHRoYXQgYmluZHMgdGhlXG4gICAgICogcGFzc2VkIGNhbGxiYWNrIHRvIHRoZSB1bmRlcmx5aW5nXG4gICAgICogd2Vic29ja2V0cycgYG9uZXJyb3JgIGhhbmRsZXIuXG4gICAgICogQHBhcmFtIHtlcnJvckhhbmRsZXJ9IGNhbGxiYWNrIC0gQSBmdW5jdGlvblxuICAgICAqIHRoYXQgaGFuZGxlcyBlcnJvcnMgb24gdGhlIHdlYnNvY2tldC5cbiAgICAgKi9cbiAgICBvbkVycm9yKGNhbGxiYWNrKXtcbiAgICAgICAgdGhpcy5lcnJvckhhbmRsZXIgPSBjYWxsYmFjaztcbiAgICB9XG59XG5cblxuZXhwb3J0IHtDZWxsU29ja2V0LCBDZWxsU29ja2V0IGFzIGRlZmF1bHR9XG4iLCIvKipcbiAqIFdlIHVzZSBhIHNpbmdsZXRvbiByZWdpc3RyeSBvYmplY3RcbiAqIHdoZXJlIHdlIG1ha2UgYXZhaWxhYmxlIGFsbCBwb3NzaWJsZVxuICogQ29tcG9uZW50cy4gVGhpcyBpcyB1c2VmdWwgZm9yIFdlYnBhY2ssXG4gKiB3aGljaCBvbmx5IGJ1bmRsZXMgZXhwbGljaXRseSB1c2VkXG4gKiBDb21wb25lbnRzIGR1cmluZyBidWlsZCB0aW1lLlxuICovXG5pbXBvcnQge0FzeW5jRHJvcGRvd24sIEFzeW5jRHJvcGRvd25Db250ZW50fSBmcm9tICcuL2NvbXBvbmVudHMvQXN5bmNEcm9wZG93bic7XG5pbXBvcnQge0JhZGdlfSBmcm9tICcuL2NvbXBvbmVudHMvQmFkZ2UnO1xuaW1wb3J0IHtCdXR0b259IGZyb20gJy4vY29tcG9uZW50cy9CdXR0b24nO1xuaW1wb3J0IHtCdXR0b25Hcm91cH0gZnJvbSAnLi9jb21wb25lbnRzL0J1dHRvbkdyb3VwJztcbmltcG9ydCB7Q2FyZH0gZnJvbSAnLi9jb21wb25lbnRzL0NhcmQnO1xuaW1wb3J0IHtDYXJkVGl0bGV9IGZyb20gJy4vY29tcG9uZW50cy9DYXJkVGl0bGUnO1xuaW1wb3J0IHtDaXJjbGVMb2FkZXJ9IGZyb20gJy4vY29tcG9uZW50cy9DaXJjbGVMb2FkZXInO1xuaW1wb3J0IHtDbGlja2FibGV9IGZyb20gJy4vY29tcG9uZW50cy9DbGlja2FibGUnO1xuaW1wb3J0IHtDb2RlfSBmcm9tICcuL2NvbXBvbmVudHMvQ29kZSc7XG5pbXBvcnQge0NvZGVFZGl0b3J9IGZyb20gJy4vY29tcG9uZW50cy9Db2RlRWRpdG9yJztcbmltcG9ydCB7Q29sbGFwc2libGVQYW5lbH0gZnJvbSAnLi9jb21wb25lbnRzL0NvbGxhcHNpYmxlUGFuZWwnO1xuaW1wb3J0IHtDb2x1bW5zfSBmcm9tICcuL2NvbXBvbmVudHMvQ29sdW1ucyc7XG5pbXBvcnQge0NvbnRhaW5lcn0gZnJvbSAnLi9jb21wb25lbnRzL0NvbnRhaW5lcic7XG5pbXBvcnQge0NvbnRleHR1YWxEaXNwbGF5fSBmcm9tICcuL2NvbXBvbmVudHMvQ29udGV4dHVhbERpc3BsYXknO1xuaW1wb3J0IHtEcm9wZG93bn0gZnJvbSAnLi9jb21wb25lbnRzL0Ryb3Bkb3duJztcbmltcG9ydCB7RXhwYW5kc30gZnJvbSAnLi9jb21wb25lbnRzL0V4cGFuZHMnO1xuaW1wb3J0IHtIZWFkZXJCYXJ9IGZyb20gJy4vY29tcG9uZW50cy9IZWFkZXJCYXInO1xuaW1wb3J0IHtMb2FkQ29udGVudHNGcm9tVXJsfSBmcm9tICcuL2NvbXBvbmVudHMvTG9hZENvbnRlbnRzRnJvbVVybCc7XG5pbXBvcnQge0xhcmdlUGVuZGluZ0Rvd25sb2FkRGlzcGxheX0gZnJvbSAnLi9jb21wb25lbnRzL0xhcmdlUGVuZGluZ0Rvd25sb2FkRGlzcGxheSc7XG5pbXBvcnQge01haW59IGZyb20gJy4vY29tcG9uZW50cy9NYWluJztcbmltcG9ydCB7TW9kYWx9IGZyb20gJy4vY29tcG9uZW50cy9Nb2RhbCc7XG5pbXBvcnQge09jdGljb259IGZyb20gJy4vY29tcG9uZW50cy9PY3RpY29uJztcbmltcG9ydCB7UGFkZGluZ30gZnJvbSAnLi9jb21wb25lbnRzL1BhZGRpbmcnO1xuaW1wb3J0IHtQb3BvdmVyfSBmcm9tICcuL2NvbXBvbmVudHMvUG9wb3Zlcic7XG5pbXBvcnQge1Jvb3RDZWxsfSBmcm9tICcuL2NvbXBvbmVudHMvUm9vdENlbGwnO1xuaW1wb3J0IHtTZXF1ZW5jZX0gZnJvbSAnLi9jb21wb25lbnRzL1NlcXVlbmNlJztcbmltcG9ydCB7U2Nyb2xsYWJsZX0gZnJvbSAnLi9jb21wb25lbnRzL1Njcm9sbGFibGUnO1xuaW1wb3J0IHtTaW5nbGVMaW5lVGV4dEJveH0gZnJvbSAnLi9jb21wb25lbnRzL1NpbmdsZUxpbmVUZXh0Qm94JztcbmltcG9ydCB7U3Bhbn0gZnJvbSAnLi9jb21wb25lbnRzL1NwYW4nO1xuaW1wb3J0IHtTdWJzY3JpYmVkfSBmcm9tICcuL2NvbXBvbmVudHMvU3Vic2NyaWJlZCc7XG5pbXBvcnQge1N1YnNjcmliZWRTZXF1ZW5jZX0gZnJvbSAnLi9jb21wb25lbnRzL1N1YnNjcmliZWRTZXF1ZW5jZSc7XG5pbXBvcnQge1RhYmxlfSBmcm9tICcuL2NvbXBvbmVudHMvVGFibGUnO1xuaW1wb3J0IHtUYWJzfSBmcm9tICcuL2NvbXBvbmVudHMvVGFicyc7XG5pbXBvcnQge1RleHR9IGZyb20gJy4vY29tcG9uZW50cy9UZXh0JztcbmltcG9ydCB7VHJhY2ViYWNrfSBmcm9tICcuL2NvbXBvbmVudHMvVHJhY2ViYWNrJztcbmltcG9ydCB7X05hdlRhYn0gZnJvbSAnLi9jb21wb25lbnRzL19OYXZUYWInO1xuaW1wb3J0IHtHcmlkfSBmcm9tICcuL2NvbXBvbmVudHMvR3JpZCc7XG5pbXBvcnQge1NoZWV0fSBmcm9tICcuL2NvbXBvbmVudHMvU2hlZXQnO1xuaW1wb3J0IHtQbG90fSBmcm9tICcuL2NvbXBvbmVudHMvUGxvdCc7XG5pbXBvcnQge19QbG90VXBkYXRlcn0gZnJvbSAnLi9jb21wb25lbnRzL19QbG90VXBkYXRlcic7XG5cbmNvbnN0IENvbXBvbmVudFJlZ2lzdHJ5ID0ge1xuICAgIEFzeW5jRHJvcGRvd24sXG4gICAgQXN5bmNEcm9wZG93bkNvbnRlbnQsXG4gICAgQmFkZ2UsXG4gICAgQnV0dG9uLFxuICAgIEJ1dHRvbkdyb3VwLFxuICAgIENhcmQsXG4gICAgQ2FyZFRpdGxlLFxuICAgIENpcmNsZUxvYWRlcixcbiAgICBDbGlja2FibGUsXG4gICAgQ29kZSxcbiAgICBDb2RlRWRpdG9yLFxuICAgIENvbGxhcHNpYmxlUGFuZWwsXG4gICAgQ29sdW1ucyxcbiAgICBDb250YWluZXIsXG4gICAgQ29udGV4dHVhbERpc3BsYXksXG4gICAgRHJvcGRvd24sXG4gICAgRXhwYW5kcyxcbiAgICBIZWFkZXJCYXIsXG4gICAgTG9hZENvbnRlbnRzRnJvbVVybCxcbiAgICBMYXJnZVBlbmRpbmdEb3dubG9hZERpc3BsYXksXG4gICAgTWFpbixcbiAgICBNb2RhbCxcbiAgICBPY3RpY29uLFxuICAgIFBhZGRpbmcsXG4gICAgUG9wb3ZlcixcbiAgICBSb290Q2VsbCxcbiAgICBTZXF1ZW5jZSxcbiAgICBTY3JvbGxhYmxlLFxuICAgIFNpbmdsZUxpbmVUZXh0Qm94LFxuICAgIFNwYW4sXG4gICAgU3Vic2NyaWJlZCxcbiAgICBTdWJzY3JpYmVkU2VxdWVuY2UsXG4gICAgVGFibGUsXG4gICAgVGFicyxcbiAgICBUZXh0LFxuICAgIFRyYWNlYmFjayxcbiAgICBfTmF2VGFiLFxuICAgIEdyaWQsXG4gICAgU2hlZXQsXG4gICAgUGxvdCxcbiAgICBfUGxvdFVwZGF0ZXJcbn07XG5cbmV4cG9ydCB7Q29tcG9uZW50UmVnaXN0cnksIENvbXBvbmVudFJlZ2lzdHJ5IGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBBc3luY0Ryb3Bkb3duIENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIGEgc2luZ2xlIHJlZ3VsYXJcbiAqIHJlcGxhY2VtZW50OlxuICogKiBgY29udGVudHNgXG4gKlxuICogTk9URTogVGhlIENlbGxzIHZlcnNpb24gb2YgdGhpcyBjaGlsZCBpc1xuICogZWl0aGVyIGEgbG9hZGluZyBpbmRpY2F0b3IsIHRleHQsIG9yIGFcbiAqIEFzeW5jRHJvcGRvd25Db250ZW50IGNlbGwuXG4gKi9cbmNsYXNzIEFzeW5jRHJvcGRvd24gZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29udGV4dCB0byBtZXRob2RzXG4gICAgICAgIHRoaXMuYWRkRHJvcGRvd25MaXN0ZW5lciA9IHRoaXMuYWRkRHJvcGRvd25MaXN0ZW5lci5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkFzeW5jRHJvcGRvd25cIixcbiAgICAgICAgICAgICAgICBjbGFzczogXCJjZWxsIGJ0bi1ncm91cFwiXG4gICAgICAgICAgICB9LCBbXG4gICAgICAgICAgICAgICAgaCgnYScsIHtjbGFzczogXCJidG4gYnRuLXhzIGJ0bi1vdXRsaW5lLXNlY29uZGFyeVwifSwgW3RoaXMucHJvcHMuZXh0cmFEYXRhLmxhYmVsVGV4dF0pLFxuICAgICAgICAgICAgICAgIGgoJ2J1dHRvbicsIHtcbiAgICAgICAgICAgICAgICAgICAgY2xhc3M6IFwiYnRuIGJ0bi14cyBidG4tb3V0bGluZS1zZWNvbmRhcnkgZHJvcGRvd24tdG9nZ2xlIGRyb3Bkb3duLXRvZ2dsZS1zcGxpdFwiLFxuICAgICAgICAgICAgICAgICAgICB0eXBlOiBcImJ1dHRvblwiLFxuICAgICAgICAgICAgICAgICAgICBpZDogYCR7dGhpcy5wcm9wcy5pZH0tZHJvcGRvd25NZW51QnV0dG9uYCxcbiAgICAgICAgICAgICAgICAgICAgXCJkYXRhLXRvZ2dsZVwiOiBcImRyb3Bkb3duXCIsXG4gICAgICAgICAgICAgICAgICAgIGFmdGVyQ3JlYXRlOiB0aGlzLmFkZERyb3Bkb3duTGlzdGVuZXIsXG4gICAgICAgICAgICAgICAgICAgIFwiZGF0YS1maXJzdGNsaWNrXCI6IFwidHJ1ZVwiXG4gICAgICAgICAgICAgICAgfSksXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgICAgICBpZDogYCR7dGhpcy5wcm9wcy5pZH0tZHJvcGRvd25Db250ZW50V3JhcHBlcmAsXG4gICAgICAgICAgICAgICAgICAgIGNsYXNzOiBcImRyb3Bkb3duLW1lbnVcIlxuICAgICAgICAgICAgICAgIH0sIFt0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY29udGVudHMnKV0pXG4gICAgICAgICAgICBdKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIGFkZERyb3Bkb3duTGlzdGVuZXIoZWxlbWVudCl7XG4gICAgICAgIGxldCBwYXJlbnRFbCA9IGVsZW1lbnQucGFyZW50RWxlbWVudDtcbiAgICAgICAgbGV0IGZpcnN0VGltZUNsaWNrZWQgPSAoZWxlbWVudC5kYXRhc2V0LmZpcnN0Y2xpY2sgPT0gXCJ0cnVlXCIpO1xuICAgICAgICBsZXQgY29tcG9uZW50ID0gdGhpcztcbiAgICAgICAgaWYoZmlyc3RUaW1lQ2xpY2tlZCl7XG4gICAgICAgICAgICAkKHBhcmVudEVsKS5vbignc2hvdy5icy5kcm9wZG93bicsIGZ1bmN0aW9uKCl7XG4gICAgICAgICAgICAgICAgY2VsbFNvY2tldC5zZW5kU3RyaW5nKEpTT04uc3RyaW5naWZ5KHtcbiAgICAgICAgICAgICAgICAgICAgZXZlbnQ6J2Ryb3Bkb3duJyxcbiAgICAgICAgICAgICAgICAgICAgdGFyZ2V0X2NlbGw6IGNvbXBvbmVudC5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICAgICAgaXNPcGVuOiBmYWxzZVxuICAgICAgICAgICAgICAgIH0pKTtcbiAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgJChwYXJlbnRFbCkub24oJ2hpZGUuYnMuZHJvcGRvd24nLCBmdW5jdGlvbigpe1xuICAgICAgICAgICAgICAgIGNlbGxTb2NrZXQuc2VuZFN0cmluZyhKU09OLnN0cmluZ2lmeSh7XG4gICAgICAgICAgICAgICAgICAgIGV2ZW50OiAnZHJvcGRvd24nLFxuICAgICAgICAgICAgICAgICAgICB0YXJnZXRfY2VsbDogY29tcG9uZW50LnByb3BzLmlkLFxuICAgICAgICAgICAgICAgICAgICBpc09wZW46IHRydWVcbiAgICAgICAgICAgICAgICB9KSk7XG4gICAgICAgICAgICB9KTtcbiAgICAgICAgICAgIGVsZW1lbnQuZGF0YXNldC5maXJzdGNsaWNrID0gZmFsc2U7XG4gICAgICAgIH1cbiAgICB9XG59XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyBhIHNpbmdsZSByZWd1bGFyXG4gKiByZXBsYWNlbWVudDpcbiAqICogYGNvbnRlbnRzYFxuICovXG5jbGFzcyBBc3luY0Ryb3Bkb3duQ29udGVudCBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgaWQ6IGBkcm9wZG93bkNvbnRlbnQtJHt0aGlzLnByb3BzLmlkfWAsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiQXN5bmNEcm9wZG93bkNvbnRlbnRcIlxuICAgICAgICAgICAgfSwgW3RoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdjb250ZW50cycpXSlcbiAgICAgICAgKTtcbiAgICB9XG59XG5cblxuZXhwb3J0IHtcbiAgICBBc3luY0Ryb3Bkb3duLFxuICAgIEFzeW5jRHJvcGRvd25Db250ZW50LFxuICAgIEFzeW5jRHJvcGRvd24gYXMgZGVmYXVsdFxufTtcbiIsIi8qKlxuICogQmFkZ2UgQ2VsbCBDb21wb25lbnRcbiAqL1xuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBCYWRnZSBoYXMgYSBzaW5nbGUgcmVwbGFjZW1lbnQ6XG4gKiAqIGBjaGlsZGBcbiAqL1xuY2xhc3MgQmFkZ2UgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIoLi4uYXJncyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybihcbiAgICAgICAgICAgIGgoJ3NwYW4nLCB7XG4gICAgICAgICAgICAgICAgY2xhc3M6IGBjZWxsIGJhZGdlIGJhZGdlLSR7dGhpcy5wcm9wcy5leHRyYURhdGEuYmFkZ2VTdHlsZX1gLFxuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkJhZGdlXCJcbiAgICAgICAgICAgIH0sIFt0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY2hpbGQnKV0pXG4gICAgICAgICk7XG4gICAgfVxufVxuXG5leHBvcnQge0JhZGdlLCBCYWRnZSBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogQnV0dG9uIENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG5jbGFzcyBCdXR0b24gZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29udGV4dCB0byBtZXRob2RzXG4gICAgICAgIHRoaXMuX2dldEV2ZW50cyA9IHRoaXMuX2dldEV2ZW50LmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuX2dldEhUTUxDbGFzc2VzID0gdGhpcy5fZ2V0SFRNTENsYXNzZXMuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuKFxuICAgICAgICAgICAgaCgnYnV0dG9uJywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkJ1dHRvblwiLFxuICAgICAgICAgICAgICAgIGNsYXNzOiB0aGlzLl9nZXRIVE1MQ2xhc3NlcygpLFxuICAgICAgICAgICAgICAgIG9uY2xpY2s6IHRoaXMuX2dldEV2ZW50KCdvbmNsaWNrJylcbiAgICAgICAgICAgIH0sIFt0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY29udGVudHMnKV1cbiAgICAgICAgICAgICkgXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgX2dldEV2ZW50KGV2ZW50TmFtZSkge1xuICAgICAgICByZXR1cm4gdGhpcy5wcm9wcy5leHRyYURhdGEuZXZlbnRzW2V2ZW50TmFtZV07XG4gICAgfVxuXG4gICAgX2dldEhUTUxDbGFzc2VzKCl7XG4gICAgICAgIGxldCBjbGFzc1N0cmluZyA9IHRoaXMucHJvcHMuZXh0cmFEYXRhLmNsYXNzZXMuam9pbihcIiBcIik7XG4gICAgICAgIC8vIHJlbWVtYmVyIHRvIHRyaW0gdGhlIGNsYXNzIHN0cmluZyBkdWUgdG8gYSBtYXF1ZXR0ZSBidWdcbiAgICAgICAgcmV0dXJuIGNsYXNzU3RyaW5nLnRyaW0oKTtcbiAgICB9XG59XG5cbmV4cG9ydCB7QnV0dG9uLCBCdXR0b24gYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIEJ1dHRvbkdyb3VwIENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBUaGlzIGNvbXBvbmVudCBoYXMgYSBzaW5nbGUgZW51bWVyYXRlZFxuICogcmVwbGFjZW1lbnQ6XG4gKiAqIGBidXR0b25gXG4gKi9cbmNsYXNzIEJ1dHRvbkdyb3VwIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkJ1dHRvbkdyb3VwXCIsXG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiYnRuLWdyb3VwXCIsXG4gICAgICAgICAgICAgICAgXCJyb2xlXCI6IFwiZ3JvdXBcIlxuICAgICAgICAgICAgfSwgdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRzRm9yKCdidXR0b24nKVxuICAgICAgICAgICAgIClcbiAgICAgICAgKTtcbiAgICB9XG5cbn1cblxuZXhwb3J0IHtCdXR0b25Hcm91cCwgQnV0dG9uR3JvdXAgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIENhcmQgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGNvbnRhaW5zIHR3b1xuICogcmVndWxhciByZXBsYWNlbWVudHM6XG4gKiAqIGBjb250ZW50c2BcbiAqICogYGhlYWRlcmBcbiAqL1xuY2xhc3MgQ2FyZCBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb21wb25lbnQgbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VCb2R5ID0gdGhpcy5tYWtlQm9keS5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLm1ha2VIZWFkZXIgPSB0aGlzLm1ha2VIZWFkZXIuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgbGV0IGJvZHlDbGFzcyA9ICdjYXJkLWJvZHknO1xuICAgICAgICBpZih0aGlzLnByb3BzLmV4dHJhRGF0YS5wYWRkaW5nKXtcbiAgICAgICAgICAgIGJvZHlDbGFzcyA9IGBjYXJkLWJvZHkgcC0ke3RoaXMucHJvcHMuZXh0cmFEYXRhLnBhZGRpbmd9YDtcbiAgICAgICAgfVxuICAgICAgICBsZXQgYm9keUFyZWEgPSBoKCdkaXYnLCB7XG4gICAgICAgICAgICBjbGFzczogYm9keUNsYXNzXG4gICAgICAgIH0sIFt0aGlzLm1ha2VCb2R5KCldKTtcbiAgICAgICAgbGV0IGhlYWRlciA9IHRoaXMubWFrZUhlYWRlcigpO1xuICAgICAgICBsZXQgaGVhZGVyQXJlYSA9IG51bGw7XG4gICAgICAgIGlmKGhlYWRlcil7XG4gICAgICAgICAgICBoZWFkZXJBcmVhID0gaCgnZGl2Jywge2NsYXNzOiBcImNhcmQtaGVhZGVyXCJ9LCBbaGVhZGVyXSk7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIGgoJ2RpdicsXG4gICAgICAgICAgICB7XG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbCBjYXJkXCIsXG4gICAgICAgICAgICAgICAgc3R5bGU6IHRoaXMucHJvcHMuZXh0cmFEYXRhLmRpdlN0eWxlLFxuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkNhcmRcIlxuICAgICAgICAgICAgfSwgW2hlYWRlckFyZWEsIGJvZHlBcmVhXSk7XG4gICAgfVxuXG4gICAgbWFrZUJvZHkoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY29udGVudHMnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkTmFtZWQoJ2NvbnRlbnRzJyk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBtYWtlSGVhZGVyKCl7XG4gICAgICAgIGlmKHRoaXMudXNlc1JlcGxhY2VtZW50cyl7XG4gICAgICAgICAgICBpZih0aGlzLnJlcGxhY2VtZW50cy5oYXNSZXBsYWNlbWVudCgnaGVhZGVyJykpe1xuICAgICAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignaGVhZGVyJyk7XG4gICAgICAgICAgICB9XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5yZW5kZXJDaGlsZE5hbWVkKCdoZWFkZXInKTtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gbnVsbDtcbiAgICB9XG59O1xuXG5jb25zb2xlLmxvZygnQ2FyZCBtb2R1bGUgbG9hZGVkJyk7XG5leHBvcnQge0NhcmQsIENhcmQgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIENhcmRUaXRsZSBDZWxsXG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyAgc2luZ2xlIHJlZ3VsYXJcbiAqIHJlcGxhY2VtZW50OlxuICogKiBgY29udGVudHNgXG4gKi9cbmNsYXNzIENhcmRUaXRsZSBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybihcbiAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBjbGFzczogXCJjZWxsXCIsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiQ2FyZFRpdGxlXCJcbiAgICAgICAgICAgIH0sIFtcbiAgICAgICAgICAgICAgICB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY29udGVudHMnKVxuICAgICAgICAgICAgXSlcbiAgICAgICAgKTtcbiAgICB9XG59XG5cbmV4cG9ydCB7Q2FyZFRpdGxlLCBDYXJkVGl0bGUgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIENpcmNsZUxvYWRlciBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuXG5jbGFzcyBDaXJjbGVMb2FkZXIgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkNpcmNsZUxvYWRlclwiLFxuICAgICAgICAgICAgICAgIGNsYXNzOiBcInNwaW5uZXItZ3Jvd1wiLFxuICAgICAgICAgICAgICAgIHJvbGU6IFwic3RhdHVzXCJcbiAgICAgICAgICAgIH0pXG4gICAgICAgICk7XG4gICAgfVxufVxuXG5leHBvcnQge0NpcmNsZUxvYWRlciwgQ2lyY2xlTG9hZGVyIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBDbGlja2FibGUgQ2VsbCBDb21wb25lbnRcbiAqL1xuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBUaGlzIGNvbXBvbmVudCBoYXMgYSBzaW5nbGVcbiAqIHJlZ3VsYXIgcmVwbGFjZW1lbnQ6XG4gKiAqIGBjb250ZW50c2BcbiAqL1xuY2xhc3MgQ2xpY2thYmxlIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICAvLyBCaW5kIGNvbnRleHQgdG8gbWV0aG9kc1xuICAgICAgICB0aGlzLl9nZXRFdmVudHMgPSB0aGlzLl9nZXRFdmVudC5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4oXG4gICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiQ2xpY2thYmxlXCIsXG4gICAgICAgICAgICAgICAgb25jbGljazogdGhpcy5fZ2V0RXZlbnQoJ29uY2xpY2snKSxcbiAgICAgICAgICAgICAgICBzdHlsZTogdGhpcy5wcm9wcy5leHRyYURhdGEuZGl2U3R5bGVcbiAgICAgICAgICAgIH0sIFtcbiAgICAgICAgICAgICAgICBoKCdkaXYnLCB7fSwgW3RoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdjb250ZW50cycpXSlcbiAgICAgICAgICAgIF1cbiAgICAgICAgICAgIClcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBfZ2V0RXZlbnQoZXZlbnROYW1lKSB7XG4gICAgICAgIHJldHVybiB0aGlzLnByb3BzLmV4dHJhRGF0YS5ldmVudHNbZXZlbnROYW1lXTtcbiAgICB9XG59XG5cbmV4cG9ydCB7Q2xpY2thYmxlLCBDbGlja2FibGUgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIENvZGUgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyBhIHNpbmdsZVxuICogcmVndWxhciByZXBsYWNlbWVudDpcbiAqICogYGNoaWxkYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgY29kZWAgKHNpbmdsZSkgLSBDb2RlIHRoYXQgd2lsbCBiZSByZW5kZXJlZCBpbnNpZGVcbiAqL1xuY2xhc3MgQ29kZSBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb21wb25lbnQgbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VDb2RlID0gdGhpcy5tYWtlQ29kZS5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gaCgncHJlJyxcbiAgICAgICAgICAgICAgICAge1xuICAgICAgICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbCBjb2RlXCIsXG4gICAgICAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJDb2RlXCJcbiAgICAgICAgICAgICAgICAgfSwgW1xuICAgICAgICAgICAgICAgICAgICAgaChcImNvZGVcIiwge30sIFt0aGlzLm1ha2VDb2RlKCldKVxuICAgICAgICAgICAgICAgICBdXG4gICAgICAgICAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlQ29kZSgpe1xuICAgICAgICBpZih0aGlzLnVzZXNSZXBsYWNlbWVudHMpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdjaGlsZCcpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMucmVuZGVyQ2hpbGROYW1lZCgnY29kZScpO1xuICAgICAgICB9XG4gICAgfVxufVxuXG5leHBvcnQge0NvZGUsIENvZGUgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIENvZGVFZGl0b3IgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbmNsYXNzIENvZGVFZGl0b3IgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuICAgICAgICB0aGlzLmVkaXRvciA9IG51bGw7XG4gICAgICAgIC8vIHVzZWQgdG8gc2NoZWR1bGUgcmVndWxhciBzZXJ2ZXIgdXBkYXRlc1xuICAgICAgICB0aGlzLlNFUlZFUl9VUERBVEVfREVMQVlfTVMgPSAxO1xuICAgICAgICB0aGlzLmVkaXRvclN0eWxlID0gJ3dpZHRoOjEwMCU7aGVpZ2h0OjEwMCU7bWFyZ2luOmF1dG87Ym9yZGVyOjFweCBzb2xpZCBsaWdodGdyYXk7JztcblxuICAgICAgICB0aGlzLnNldHVwRWRpdG9yID0gdGhpcy5zZXR1cEVkaXRvci5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLnNldHVwS2V5YmluZGluZ3MgPSB0aGlzLnNldHVwS2V5YmluZGluZ3MuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5jaGFuZ2VIYW5kbGVyID0gdGhpcy5jaGFuZ2VIYW5kbGVyLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgY29tcG9uZW50RGlkTG9hZCgpIHtcblxuICAgICAgICB0aGlzLnNldHVwRWRpdG9yKCk7XG5cbiAgICAgICAgaWYgKHRoaXMuZWRpdG9yID09PSBudWxsKSB7XG4gICAgICAgICAgICBjb25zb2xlLmxvZyhcImVkaXRvciBjb21wb25lbnQgbG9hZGVkIGJ1dCBmYWlsZWQgdG8gc2V0dXAgZWRpdG9yXCIpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgY29uc29sZS5sb2coXCJzZXR0aW5nIHVwIGVkaXRvclwiKTtcbiAgICAgICAgICAgIHRoaXMuZWRpdG9yLmxhc3RfZWRpdF9taWxsaXMgPSBEYXRlLm5vdygpO1xuXG4gICAgICAgICAgICB0aGlzLmVkaXRvci5zZXRUaGVtZShcImFjZS90aGVtZS90ZXh0bWF0ZVwiKTtcbiAgICAgICAgICAgIHRoaXMuZWRpdG9yLnNlc3Npb24uc2V0TW9kZShcImFjZS9tb2RlL3B5dGhvblwiKTtcbiAgICAgICAgICAgIHRoaXMuZWRpdG9yLnNldEF1dG9TY3JvbGxFZGl0b3JJbnRvVmlldyh0cnVlKTtcbiAgICAgICAgICAgIHRoaXMuZWRpdG9yLnNlc3Npb24uc2V0VXNlU29mdFRhYnModHJ1ZSk7XG4gICAgICAgICAgICB0aGlzLmVkaXRvci5zZXRWYWx1ZSh0aGlzLnByb3BzLmV4dHJhRGF0YS5pbml0aWFsVGV4dCk7XG5cbiAgICAgICAgICAgIGlmICh0aGlzLnByb3BzLmV4dHJhRGF0YS5hdXRvY29tcGxldGUpIHtcbiAgICAgICAgICAgICAgICB0aGlzLmVkaXRvci5zZXRPcHRpb25zKHtlbmFibGVCYXNpY0F1dG9jb21wbGV0aW9uOiB0cnVlfSk7XG4gICAgICAgICAgICAgICAgdGhpcy5lZGl0b3Iuc2V0T3B0aW9ucyh7ZW5hYmxlTGl2ZUF1dG9jb21wbGV0aW9uOiB0cnVlfSk7XG4gICAgICAgICAgICB9XG5cbiAgICAgICAgICAgIGlmICh0aGlzLnByb3BzLmV4dHJhRGF0YS5ub1Njcm9sbCkge1xuICAgICAgICAgICAgICAgIHRoaXMuZWRpdG9yLnNldE9wdGlvbihcIm1heExpbmVzXCIsIEluZmluaXR5KTtcbiAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgaWYgKHRoaXMucHJvcHMuZXh0cmFEYXRhLmZvbnRTaXplICE9PSB1bmRlZmluZWQpIHtcbiAgICAgICAgICAgICAgICB0aGlzLmVkaXRvci5zZXRPcHRpb24oXCJmb250U2l6ZVwiLCB0aGlzLnByb3BzLmV4dHJhRGF0YS5mb250U2l6ZSk7XG4gICAgICAgICAgICB9XG5cbiAgICAgICAgICAgIGlmICh0aGlzLnByb3BzLmV4dHJhRGF0YS5taW5MaW5lcyAhPT0gdW5kZWZpbmVkKSB7XG4gICAgICAgICAgICAgICAgdGhpcy5lZGl0b3Iuc2V0T3B0aW9uKFwibWluTGluZXNcIiwgdGhpcy5wcm9wcy5leHRyYURhdGEubWluTGluZXMpO1xuICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICB0aGlzLnNldHVwS2V5YmluZGluZ3MoKTtcblxuICAgICAgICAgICAgdGhpcy5jaGFuZ2VIYW5kbGVyKCk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIGgoJ2RpdicsXG4gICAgICAgICAgICB7XG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbFwiLFxuICAgICAgICAgICAgICAgIHN0eWxlOiB0aGlzLnByb3BzLmV4dHJhRGF0YS5kaXZTdHlsZSxcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJDb2RlRWRpdG9yXCJcbiAgICAgICAgICAgIH0sXG4gICAgICAgICAgICBbaCgnZGl2JywgeyBpZDogXCJlZGl0b3JcIiArIHRoaXMucHJvcHMuaWQsIHN0eWxlOiB0aGlzLmVkaXRvclN0eWxlIH0sIFtdKVxuICAgICAgICBdKTtcbiAgICB9XG5cbiAgICBzZXR1cEVkaXRvcigpe1xuICAgICAgICBsZXQgZWRpdG9ySWQgPSBcImVkaXRvclwiICsgdGhpcy5wcm9wcy5pZDtcbiAgICAgICAgLy8gVE9ETyBUaGVzZSBhcmUgZ2xvYmFsIHZhciBkZWZpbmVkIGluIHBhZ2UuaHRtbFxuICAgICAgICAvLyB3ZSBzaG91bGQgZG8gc29tZXRoaW5nIGFib3V0IHRoaXMuXG5cbiAgICAgICAgLy8gaGVyZSB3ZSBiaW5nIGFuZCBpbnNldCB0aGUgZWRpdG9yIGludG8gdGhlIGRpdiByZW5kZXJlZCBieVxuICAgICAgICAvLyB0aGlzLnJlbmRlcigpXG4gICAgICAgIHRoaXMuZWRpdG9yID0gYWNlLmVkaXQoZWRpdG9ySWQpO1xuICAgICAgICAvLyBUT0RPOiBkZWFsIHdpdGggdGhpcyBnbG9iYWwgZWRpdG9yIGxpc3RcbiAgICAgICAgYWNlRWRpdG9yc1tlZGl0b3JJZF0gPSB0aGlzLmVkaXRvcjtcbiAgICB9XG5cbiAgICBjaGFuZ2VIYW5kbGVyKCkge1xuXHR2YXIgZWRpdG9ySWQgPSB0aGlzLnByb3BzLmlkO1xuXHR2YXIgZWRpdG9yID0gdGhpcy5lZGl0b3I7XG5cdHZhciBTRVJWRVJfVVBEQVRFX0RFTEFZX01TID0gdGhpcy5TRVJWRVJfVVBEQVRFX0RFTEFZX01TO1xuICAgICAgICB0aGlzLmVkaXRvci5zZXNzaW9uLm9uKFxuICAgICAgICAgICAgXCJjaGFuZ2VcIixcbiAgICAgICAgICAgIGZ1bmN0aW9uKGRlbHRhKSB7XG4gICAgICAgICAgICAgICAgLy8gV1NcbiAgICAgICAgICAgICAgICBsZXQgcmVzcG9uc2VEYXRhID0ge1xuICAgICAgICAgICAgICAgICAgICBldmVudDogJ2VkaXRvcl9jaGFuZ2UnLFxuICAgICAgICAgICAgICAgICAgICAndGFyZ2V0X2NlbGwnOiBlZGl0b3JJZCxcbiAgICAgICAgICAgICAgICAgICAgZGF0YTogZGVsdGFcbiAgICAgICAgICAgICAgICB9O1xuICAgICAgICAgICAgICAgIGNlbGxTb2NrZXQuc2VuZFN0cmluZyhKU09OLnN0cmluZ2lmeShyZXNwb25zZURhdGEpKTtcbiAgICAgICAgICAgICAgICAvL3JlY29yZCB0aGF0IHdlIGp1c3QgZWRpdGVkXG4gICAgICAgICAgICAgICAgZWRpdG9yLmxhc3RfZWRpdF9taWxsaXMgPSBEYXRlLm5vdygpO1xuXG5cdFx0Ly9zY2hlZHVsZSBhIGZ1bmN0aW9uIHRvIHJ1biBpbiAnU0VSVkVSX1VQREFURV9ERUxBWV9NUydtc1xuXHRcdC8vdGhhdCB3aWxsIHVwZGF0ZSB0aGUgc2VydmVyLCBidXQgb25seSBpZiB0aGUgdXNlciBoYXMgc3RvcHBlZCB0eXBpbmcuXG5cdFx0Ly8gVE9ETyB1bmNsZWFyIGlmIHRoaXMgaXMgb3dya2luZyBwcm9wZXJseVxuXHRcdHdpbmRvdy5zZXRUaW1lb3V0KGZ1bmN0aW9uKCkge1xuXHRcdCAgICBpZiAoRGF0ZS5ub3coKSAtIGVkaXRvci5sYXN0X2VkaXRfbWlsbGlzID49IFNFUlZFUl9VUERBVEVfREVMQVlfTVMpIHtcblx0XHRcdC8vc2F2ZSBvdXIgY3VycmVudCBzdGF0ZSB0byB0aGUgcmVtb3RlIGJ1ZmZlclxuXHRcdFx0ZWRpdG9yLmN1cnJlbnRfaXRlcmF0aW9uICs9IDE7XG5cdFx0XHRlZGl0b3IubGFzdF9lZGl0X21pbGxpcyA9IERhdGUubm93KCk7XG5cdFx0XHRlZGl0b3IubGFzdF9lZGl0X3NlbnRfdGV4dCA9IGVkaXRvci5nZXRWYWx1ZSgpO1xuXHRcdFx0Ly8gV1Ncblx0XHRcdGxldCByZXNwb25zZURhdGEgPSB7XG5cdFx0XHQgICAgZXZlbnQ6ICdlZGl0aW5nJyxcblx0XHRcdCAgICAndGFyZ2V0X2NlbGwnOiBlZGl0b3JJZCxcblx0XHRcdCAgICBidWZmZXI6IGVkaXRvci5nZXRWYWx1ZSgpLFxuXHRcdFx0ICAgIHNlbGVjdGlvbjogZWRpdG9yLnNlbGVjdGlvbi5nZXRSYW5nZSgpLFxuXHRcdFx0ICAgIGl0ZXJhdGlvbjogZWRpdG9yLmN1cnJlbnRfaXRlcmF0aW9uXG5cdFx0XHR9O1xuXHRcdFx0Y2VsbFNvY2tldC5zZW5kU3RyaW5nKEpTT04uc3RyaW5naWZ5KHJlc3BvbnNlRGF0YSkpO1xuXHRcdCAgICB9XG5cdFx0fSwgU0VSVkVSX1VQREFURV9ERUxBWV9NUyArIDIpOyAvL25vdGUgdGhlIDJtcyBncmFjZSBwZXJpb2RcbiAgICAgICAgICAgIH1cbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBzZXR1cEtleWJpbmRpbmdzKCkge1xuICAgICAgICBjb25zb2xlLmxvZyhcInNldHRpbmcgdXAga2V5YmluZGluZ3NcIik7XG4gICAgICAgIHRoaXMucHJvcHMuZXh0cmFEYXRhLmtleWJpbmRpbmdzLm1hcCgoa2IpID0+IHtcbiAgICAgICAgICAgIHRoaXMuZWRpdG9yLmNvbW1hbmRzLmFkZENvbW1hbmQoXG4gICAgICAgICAgICAgICAge1xuICAgICAgICAgICAgICAgICAgICBuYW1lOiAnY21kJyArIGtiLFxuICAgICAgICAgICAgICAgICAgICBiaW5kS2V5OiB7d2luOiAnQ3RybC0nICsga2IsICBtYWM6ICdDb21tYW5kLScgKyBrYn0sXG4gICAgICAgICAgICAgICAgICAgIHJlYWRPbmx5OiB0cnVlLFxuICAgICAgICAgICAgICAgICAgICBleGVjOiAoKSA9PiB7XG4gICAgICAgICAgICAgICAgICAgICAgICB0aGlzLmVkaXRvci5jdXJyZW50X2l0ZXJhdGlvbiArPSAxO1xuICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy5lZGl0b3IubGFzdF9lZGl0X21pbGxpcyA9IERhdGUubm93KCk7XG4gICAgICAgICAgICAgICAgICAgICAgICB0aGlzLmVkaXRvci5sYXN0X2VkaXRfc2VudF90ZXh0ID0gdGhpcy5lZGl0b3IuZ2V0VmFsdWUoKTtcblxuICAgICAgICAgICAgICAgICAgICAgICAgLy8gV1NcbiAgICAgICAgICAgICAgICAgICAgICAgIGxldCByZXNwb25zZURhdGEgPSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgZXZlbnQ6ICdrZXliaW5kaW5nJyxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAndGFyZ2V0X2NlbGwnOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICdrZXknOiBrYixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAnYnVmZmVyJzogdGhpcy5lZGl0b3IuZ2V0VmFsdWUoKSxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAnc2VsZWN0aW9uJzogdGhpcy5lZGl0b3Iuc2VsZWN0aW9uLmdldFJhbmdlKCksXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgJ2l0ZXJhdGlvbic6IHRoaXMuZWRpdG9yLmN1cnJlbnRfaXRlcmF0aW9uXG4gICAgICAgICAgICAgICAgICAgICAgICB9O1xuICAgICAgICAgICAgICAgICAgICAgICAgY2VsbFNvY2tldC5zZW5kU3RyaW5nKEpTT04uc3RyaW5naWZ5KHJlc3BvbnNlRGF0YSkpO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgKTtcbiAgICAgICAgfSk7XG4gICAgfVxufVxuXG5leHBvcnQge0NvZGVFZGl0b3IsIENvZGVFZGl0b3IgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIENvbGxhcHNpYmxlUGFuZWwgQ2VsbCBDb21wb25lbnRcbiAqL1xuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50LmpzJztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBUaGlzIGNvbXBvbmVudCBoYXMgdHdvIHNpbmdsZSB0eXBlXG4gKiByZXBsYWNlbWVudHM6XG4gKiAqIGBjb250ZW50YFxuICogKiBgcGFuZWxgXG4gKiBOb3RlIHRoYXQgYHBhbmVsYCBpcyBvbmx5IHJlbmRlcmVkXG4gKiBpZiB0aGUgcGFuZWwgaXMgZXhwYW5kZWRcbiAqL1xuXG5jbGFzcyBDb2xsYXBzaWJsZVBhbmVsIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgaWYodGhpcy5wcm9wcy5leHRyYURhdGEuaXNFeHBhbmRlZCl7XG4gICAgICAgICAgICByZXR1cm4oXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgICAgICBjbGFzczogXCJjZWxsIGNvbnRhaW5lci1mbHVpZFwiLFxuICAgICAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiQ29sbGFwc2libGVQYW5lbFwiLFxuICAgICAgICAgICAgICAgICAgICBcImRhdGEtZXhwYW5kZWRcIjogdHJ1ZSxcbiAgICAgICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgICAgIHN0eWxlOiB0aGlzLnByb3BzLmV4dHJhRGF0YS5kaXZTdHlsZVxuICAgICAgICAgICAgICAgIH0sIFtcbiAgICAgICAgICAgICAgICAgICAgaCgnZGl2Jywge2NsYXNzOiBcInJvdyBmbGV4LW5vd3JhcCBuby1ndXR0ZXJzXCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgICAgICBoKCdkaXYnLCB7Y2xhc3M6IFwiY29sLW1kLWF1dG9cIn0sW1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdwYW5lbCcpXG4gICAgICAgICAgICAgICAgICAgICAgICBdKSxcbiAgICAgICAgICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtjbGFzczogXCJjb2wtc21cIn0sIFtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY29udGVudCcpXG4gICAgICAgICAgICAgICAgICAgICAgICBdKVxuICAgICAgICAgICAgICAgICAgICBdKVxuICAgICAgICAgICAgICAgIF0pXG4gICAgICAgICAgICApO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgICAgIGNsYXNzOiBcImNlbGwgY29udGFpbmVyLWZsdWlkXCIsXG4gICAgICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJDb2xsYXBzaWJsZVBhbmVsXCIsXG4gICAgICAgICAgICAgICAgICAgIFwiZGF0YS1leHBhbmRlZFwiOiBmYWxzZSxcbiAgICAgICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgICAgIHN0eWxlOiB0aGlzLnByb3BzLmV4dHJhRGF0YS5kaXZTdHlsZVxuICAgICAgICAgICAgICAgIH0sIFt0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY29udGVudCcpXSlcbiAgICAgICAgICAgICk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cblxuZXhwb3J0IHtDb2xsYXBzaWJsZVBhbmVsLCBDb2xsYXBzaWJsZVBhbmVsIGFzIGRlZmF1bHR9XG4iLCIvKipcbiAqIENvbHVtbnMgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyBvbmUgZW51bWVyYXRlZFxuICoga2luZCBvZiByZXBsYWNlbWVudDpcbiAqICogYGNgXG4gKi9cbmNsYXNzIENvbHVtbnMgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29udGV4dCB0byBtZXRob2RzXG4gICAgICAgIHRoaXMubWFrZUlubmVyQ2hpbGRyZW4gPSB0aGlzLm1ha2VJbm5lckNoaWxkcmVuLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbCBjb250YWluZXItZmx1aWRcIixcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJDb2x1bW5zXCIsXG4gICAgICAgICAgICAgICAgc3R5bGU6IHRoaXMucHJvcHMuZXh0cmFEYXRhLmRpdlN0eWxlXG4gICAgICAgICAgICB9LCBbXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge2NsYXNzOiBcInJvdyBmbGV4LW5vd3JhcFwifSwgdGhpcy5tYWtlSW5uZXJDaGlsZHJlbigpKVxuICAgICAgICAgICAgXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlSW5uZXJDaGlsZHJlbigpe1xuICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRzRm9yKCdjJykubWFwKHJlcGxFbGVtZW50ID0+IHtcbiAgICAgICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgICAgICBjbGFzczogXCJjb2wtc21cIlxuICAgICAgICAgICAgICAgIH0sIFtyZXBsRWxlbWVudF0pXG4gICAgICAgICAgICApO1xuICAgICAgICB9KTtcbiAgICB9XG59XG5cblxuZXhwb3J0IHtDb2x1bW5zLCBDb2x1bW5zIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBHZW5lcmljIGJhc2UgQ2VsbCBDb21wb25lbnQuXG4gKiBTaG91bGQgYmUgZXh0ZW5kZWQgYnkgb3RoZXJcbiAqIENlbGwgY2xhc3NlcyBvbiBKUyBzaWRlLlxuICovXG5pbXBvcnQge1JlcGxhY2VtZW50c0hhbmRsZXJ9IGZyb20gJy4vdXRpbC9SZXBsYWNlbWVudHNIYW5kbGVyJztcbmltcG9ydCB7UHJvcFR5cGVzfSBmcm9tICcuL3V0aWwvUHJvcGVydHlWYWxpZGF0b3InO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbmNsYXNzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMgPSB7fSwgcmVwbGFjZW1lbnRzID0gW10pe1xuICAgICAgICB0aGlzLl91cGRhdGVQcm9wcyhwcm9wcyk7XG4gICAgICAgIHRoaXMucmVwbGFjZW1lbnRzID0gbmV3IFJlcGxhY2VtZW50c0hhbmRsZXIocmVwbGFjZW1lbnRzKTtcbiAgICAgICAgdGhpcy51c2VzUmVwbGFjZW1lbnRzID0gKHJlcGxhY2VtZW50cy5sZW5ndGggPiAwKTtcblxuICAgICAgICAvLyBTZXR1cCBwYXJlbnQgcmVsYXRpb25zaGlwLCBpZlxuICAgICAgICAvLyBhbnkuIEluIHRoaXMgYWJzdHJhY3QgY2xhc3NcbiAgICAgICAgLy8gdGhlcmUgaXNuJ3Qgb25lIGJ5IGRlZmF1bHRcbiAgICAgICAgdGhpcy5wYXJlbnQgPSBudWxsO1xuICAgICAgICB0aGlzLl9zZXR1cENoaWxkUmVsYXRpb25zaGlwcygpO1xuXG4gICAgICAgIC8vIEVuc3VyZSB0aGF0IHdlIGhhdmUgcGFzc2VkIGluIGFuIGlkXG4gICAgICAgIC8vIHdpdGggdGhlIHByb3BzLiBTaG91bGQgZXJyb3Igb3RoZXJ3aXNlLlxuICAgICAgICBpZighdGhpcy5wcm9wcy5pZCB8fCB0aGlzLnByb3BzLmlkID09IHVuZGVmaW5lZCl7XG4gICAgICAgICAgICB0aHJvdyBFcnJvcignWW91IG11c3QgZGVmaW5lIGFuIGlkIGZvciBldmVyeSBjb21wb25lbnQgcHJvcHMhJyk7XG4gICAgICAgIH1cblxuICAgICAgICB0aGlzLnZhbGlkYXRlUHJvcHMoKTtcblxuICAgICAgICAvLyBCaW5kIGNvbnRleHQgdG8gbWV0aG9kc1xuICAgICAgICB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvciA9IHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50c0ZvciA9IHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50c0Zvci5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLmNvbXBvbmVudERpZExvYWQgPSB0aGlzLmNvbXBvbmVudERpZExvYWQuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5jaGlsZHJlbkRvID0gdGhpcy5jaGlsZHJlbkRvLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMubmFtZWRDaGlsZHJlbkRvID0gdGhpcy5uYW1lZENoaWxkcmVuRG8uYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5yZW5kZXJDaGlsZE5hbWVkID0gdGhpcy5yZW5kZXJDaGlsZE5hbWVkLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMucmVuZGVyQ2hpbGRyZW5OYW1lZCA9IHRoaXMucmVuZGVyQ2hpbGRyZW5OYW1lZC5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLl9zZXR1cENoaWxkUmVsYXRpb25zaGlwcyA9IHRoaXMuX3NldHVwQ2hpbGRSZWxhdGlvbnNoaXBzLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuX3VwZGF0ZVByb3BzID0gdGhpcy5fdXBkYXRlUHJvcHMuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5fcmVjdXJzaXZlbHlNYXBOYW1lZENoaWxkcmVuID0gdGhpcy5fcmVjdXJzaXZlbHlNYXBOYW1lZENoaWxkcmVuLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIC8vIE9iamVjdHMgdGhhdCBleHRlbmQgZnJvbVxuICAgICAgICAvLyBtZSBzaG91bGQgb3ZlcnJpZGUgdGhpc1xuICAgICAgICAvLyBtZXRob2QgaW4gb3JkZXIgdG8gZ2VuZXJhdGVcbiAgICAgICAgLy8gc29tZSBjb250ZW50IGZvciB0aGUgdmRvbVxuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ1lvdSBtdXN0IGltcGxlbWVudCBhIGByZW5kZXJgIG1ldGhvZCBvbiBDb21wb25lbnQgb2JqZWN0cyEnKTtcbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBPYmplY3QgdGhhdCBleHRlbmQgZnJvbSBtZSBjb3VsZCBvdmVyd3JpdGUgdGhpcyBtZXRob2QuXG4gICAgICogSXQgaXMgdG8gYmUgdXNlZCBmb3IgbGlmZWN5bGNlIG1hbmFnZW1lbnQgYW5kIGlzIHRvIGJlIGNhbGxlZFxuICAgICAqIGFmdGVyIHRoZSBjb21wb25lbnRzIGxvYWRzLlxuICAgICovXG4gICAgY29tcG9uZW50RGlkTG9hZCgpIHtcbiAgICAgICAgcmV0dXJuIG51bGw7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIFJlc3BvbmRzIHdpdGggYSBoeXBlcnNjcmlwdCBvYmplY3RcbiAgICAgKiB0aGF0IHJlcHJlc2VudHMgYSBkaXYgdGhhdCBpcyBmb3JtYXR0ZWRcbiAgICAgKiBhbHJlYWR5IGZvciB0aGUgcmVndWxhciByZXBsYWNlbWVudC5cbiAgICAgKiBUaGlzIG9ubHkgd29ya3MgZm9yIHJlZ3VsYXIgdHlwZSByZXBsYWNlbWVudHMuXG4gICAgICogRm9yIGVudW1lcmF0ZWQgcmVwbGFjZW1lbnRzLCB1c2VcbiAgICAgKiAjZ2V0UmVwbGFjZW1lbnRFbGVtZW50c0ZvcigpXG4gICAgICovXG4gICAgZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKHJlcGxhY2VtZW50TmFtZSl7XG4gICAgICAgIGxldCByZXBsYWNlbWVudCA9IHRoaXMucmVwbGFjZW1lbnRzLmdldFJlcGxhY2VtZW50Rm9yKHJlcGxhY2VtZW50TmFtZSk7XG4gICAgICAgIGlmKHJlcGxhY2VtZW50KXtcbiAgICAgICAgICAgIGxldCBuZXdJZCA9IGAke3RoaXMucHJvcHMuaWR9XyR7cmVwbGFjZW1lbnR9YDtcbiAgICAgICAgICAgIHJldHVybiBoKCdkaXYnLCB7aWQ6IG5ld0lkLCBrZXk6IG5ld0lkfSwgW10pO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiBudWxsO1xuICAgIH1cblxuICAgIC8qKlxuICAgICAqIFJlc3BvbmQgd2l0aCBhbiBhcnJheSBvZiBoeXBlcnNjcmlwdFxuICAgICAqIG9iamVjdHMgdGhhdCBhcmUgZGl2cyB3aXRoIGlkcyB0aGF0IG1hdGNoXG4gICAgICogcmVwbGFjZW1lbnQgc3RyaW5nIGlkcyBmb3IgdGhlIGtpbmQgb2ZcbiAgICAgKiByZXBsYWNlbWVudCBsaXN0IHRoYXQgaXMgZW51bWVyYXRlZCxcbiAgICAgKiBpZSBgX19fX2J1dHRvbl8xYCwgYF9fX19idXR0b25fMl9fYCBldGMuXG4gICAgICovXG4gICAgZ2V0UmVwbGFjZW1lbnRFbGVtZW50c0ZvcihyZXBsYWNlbWVudE5hbWUpe1xuICAgICAgICBpZighdGhpcy5yZXBsYWNlbWVudHMuaGFzUmVwbGFjZW1lbnQocmVwbGFjZW1lbnROYW1lKSl7XG4gICAgICAgICAgICByZXR1cm4gbnVsbDtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gdGhpcy5yZXBsYWNlbWVudHMubWFwUmVwbGFjZW1lbnRzRm9yKHJlcGxhY2VtZW50TmFtZSwgcmVwbGFjZW1lbnQgPT4ge1xuICAgICAgICAgICAgbGV0IG5ld0lkID0gYCR7dGhpcy5wcm9wcy5pZH1fJHtyZXBsYWNlbWVudH1gO1xuICAgICAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgICAgICBoKCdkaXYnLCB7aWQ6IG5ld0lkLCBrZXk6IG5ld0lkfSlcbiAgICAgICAgICAgICk7XG4gICAgICAgIH0pO1xuICAgIH1cblxuICAgIC8qKlxuICAgICAqIElmIHRoZXJlIGlzIGEgYHByb3BUeXBlc2Agb2JqZWN0IHByZXNlbnQgb25cbiAgICAgKiB0aGUgY29uc3RydWN0b3IgKGllIHRoZSBjb21wb25lbnQgY2xhc3MpLFxuICAgICAqIHRoZW4gcnVuIHRoZSBQcm9wVHlwZXMgdmFsaWRhdG9yIG9uIGl0LlxuICAgICAqL1xuICAgIHZhbGlkYXRlUHJvcHMoKXtcbiAgICAgICAgaWYodGhpcy5jb25zdHJ1Y3Rvci5wcm9wVHlwZXMpe1xuICAgICAgICAgICAgUHJvcFR5cGVzLnZhbGlkYXRlKFxuICAgICAgICAgICAgICAgIHRoaXMuY29uc3RydWN0b3IubmFtZSxcbiAgICAgICAgICAgICAgICB0aGlzLnByb3BzLFxuICAgICAgICAgICAgICAgIHRoaXMuY29uc3RydWN0b3IucHJvcFR5cGVzXG4gICAgICAgICAgICApO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogTG9va3MgdXAgdGhlIHBhc3NlZCBrZXkgaW4gbmFtZWRDaGlsZHJlbiBhbmRcbiAgICAgKiBpZiBmb3VuZCByZXNwb25kcyB3aXRoIHRoZSByZXN1bHQgb2YgY2FsbGluZ1xuICAgICAqIHJlbmRlciBvbiB0aGF0IGNoaWxkIGNvbXBvbmVudC4gUmV0dXJucyBudWxsXG4gICAgICogb3RoZXJ3aXNlLlxuICAgICAqL1xuICAgIHJlbmRlckNoaWxkTmFtZWQoa2V5KXtcbiAgICAgICAgbGV0IGZvdW5kQ2hpbGQgPSB0aGlzLnByb3BzLm5hbWVkQ2hpbGRyZW5ba2V5XTtcbiAgICAgICAgaWYoZm91bmRDaGlsZCl7XG4gICAgICAgICAgICByZXR1cm4gZm91bmRDaGlsZC5yZW5kZXIoKTtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gbnVsbDtcbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBMb29rcyB1cCB0aGUgcGFzc2VkIGtleSBpbiBuYW1lZENoaWxkcmVuXG4gICAgICogYW5kIGlmIGZvdW5kIC0tIGFuZCB0aGUgdmFsdWUgaXMgYW4gQXJyYXlcbiAgICAgKiBvciBBcnJheSBvZiBBcnJheXMsIHJlc3BvbmRzIHdpdGggYW5cbiAgICAgKiBpc29tb3JwaGljIHN0cnVjdHVyZSB0aGF0IGhhcyB0aGUgcmVuZGVyZWRcbiAgICAgKiB2YWx1ZXMgb2YgZWFjaCBjb21wb25lbnQuXG4gICAgICovXG4gICAgcmVuZGVyQ2hpbGRyZW5OYW1lZChrZXkpe1xuICAgICAgICBsZXQgZm91bmRDaGlsZHJlbiA9IHRoaXMucHJvcHMubmFtZWRDaGlsZHJlbltrZXldO1xuICAgICAgICBpZihmb3VuZENoaWxkcmVuKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLl9yZWN1cnNpdmVseU1hcE5hbWVkQ2hpbGRyZW4oZm91bmRDaGlsZHJlbiwgY2hpbGQgPT4ge1xuICAgICAgICAgICAgICAgIHJldHVybiBjaGlsZC5yZW5kZXIoKTtcbiAgICAgICAgICAgIH0pO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiBbXTtcbiAgICB9XG5cblxuXG4gICAgLyoqXG4gICAgICogR2V0dGVyIHRoYXQgd2lsbCByZXNwb25kIHdpdGggdGhlXG4gICAgICogY29uc3RydWN0b3IncyAoYWthIHRoZSAnY2xhc3MnKSBuYW1lXG4gICAgICovXG4gICAgZ2V0IG5hbWUoKXtcbiAgICAgICAgcmV0dXJuIHRoaXMuY29uc3RydWN0b3IubmFtZTtcbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBHZXR0ZXIgdGhhdCB3aWxsIHJlc3BvbmQgd2l0aCBhblxuICAgICAqIGFycmF5IG9mIHJlbmRlcmVkIChpZSBjb25maWd1cmVkXG4gICAgICogaHlwZXJzY3JpcHQpIG9iamVjdHMgdGhhdCByZXByZXNlbnRcbiAgICAgKiBlYWNoIGNoaWxkLiBOb3RlIHRoYXQgd2Ugd2lsbCBjcmVhdGUga2V5c1xuICAgICAqIGZvciB0aGVzZSBiYXNlZCBvbiB0aGUgSUQgb2YgdGhpcyBwYXJlbnRcbiAgICAgKiBjb21wb25lbnQuXG4gICAgICovXG4gICAgZ2V0IHJlbmRlcmVkQ2hpbGRyZW4oKXtcbiAgICAgICAgaWYodGhpcy5wcm9wcy5jaGlsZHJlbi5sZW5ndGggPT0gMCl7XG4gICAgICAgICAgICByZXR1cm4gW107XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIHRoaXMucHJvcHMuY2hpbGRyZW4ubWFwKGNoaWxkQ29tcG9uZW50ID0+IHtcbiAgICAgICAgICAgIGxldCByZW5kZXJlZENoaWxkID0gY2hpbGRDb21wb25lbnQucmVuZGVyKCk7XG4gICAgICAgICAgICByZW5kZXJlZENoaWxkLnByb3BlcnRpZXMua2V5ID0gYCR7dGhpcy5wcm9wcy5pZH0tY2hpbGQtJHtjaGlsZENvbXBvbmVudC5wcm9wcy5pZH1gO1xuICAgICAgICAgICAgcmV0dXJuIHJlbmRlcmVkQ2hpbGQ7XG4gICAgICAgIH0pO1xuICAgIH1cblxuICAgIC8qKiBQdWJsaWMgVXRpbCBNZXRob2RzICoqL1xuXG4gICAgLyoqXG4gICAgICogQ2FsbHMgdGhlIHByb3ZpZGVkIGNhbGxiYWNrIG9uIGVhY2hcbiAgICAgKiBhcnJheSBjaGlsZCBmb3IgdGhpcyBjb21wb25lbnQsIHdpdGhcbiAgICAgKiB0aGUgY2hpbGQgYXMgdGhlIHNvbGUgYXJnIHRvIHRoZVxuICAgICAqIGNhbGxiYWNrXG4gICAgICovXG4gICAgY2hpbGRyZW5EbyhjYWxsYmFjayl7XG4gICAgICAgIHRoaXMucHJvcHMuY2hpbGRyZW4uZm9yRWFjaChjaGlsZCA9PiB7XG4gICAgICAgICAgICBjYWxsYmFjayhjaGlsZCk7XG4gICAgICAgIH0pO1xuICAgIH1cblxuICAgIC8qKlxuICAgICAqIENhbGxzIHRoZSBwcm92aWRlZCBjYWxsYmFjayBvblxuICAgICAqIGVhY2ggbmFtZWQgY2hpbGQgd2l0aCBrZXksIGNoaWxkXG4gICAgICogYXMgdGhlIHR3byBhcmdzIHRvIHRoZSBjYWxsYmFjay5cbiAgICAgKi9cbiAgICBuYW1lZENoaWxkcmVuRG8oY2FsbGJhY2spe1xuICAgICAgICBPYmplY3Qua2V5cyh0aGlzLnByb3BzLm5hbWVkQ2hpbGRyZW4pLmZvckVhY2goa2V5ID0+IHtcbiAgICAgICAgICAgIGxldCBjaGlsZCA9IHRoaXMucHJvcHMubmFtZWRDaGlsZHJlbltrZXldO1xuICAgICAgICAgICAgY2FsbGJhY2soa2V5LCBjaGlsZCk7XG4gICAgICAgIH0pO1xuICAgIH1cblxuICAgIC8qKiBQcml2YXRlIFV0aWwgTWV0aG9kcyAqKi9cblxuICAgIC8qKlxuICAgICAqIFNldHMgdGhlIHBhcmVudCBhdHRyaWJ1dGUgb2YgYWxsIGluY29taW5nXG4gICAgICogYXJyYXkgYW5kL29yIG5hbWVkIGNoaWxkcmVuIHRvIHRoaXNcbiAgICAgKiBpbnN0YW5jZS5cbiAgICAgKi9cbiAgICBfc2V0dXBDaGlsZFJlbGF0aW9uc2hpcHMoKXtcbiAgICAgICAgLy8gTmFtZWQgY2hpbGRyZW4gZmlyc3RcbiAgICAgICAgT2JqZWN0LmtleXModGhpcy5wcm9wcy5uYW1lZENoaWxkcmVuKS5mb3JFYWNoKGtleSA9PiB7XG4gICAgICAgICAgICBsZXQgY2hpbGQgPSB0aGlzLnByb3BzLm5hbWVkQ2hpbGRyZW5ba2V5XTtcbiAgICAgICAgICAgIGNoaWxkLnBhcmVudCA9IHRoaXM7XG4gICAgICAgIH0pO1xuXG4gICAgICAgIC8vIE5vdyBhcnJheSBjaGlsZHJlblxuICAgICAgICB0aGlzLnByb3BzLmNoaWxkcmVuLmZvckVhY2goY2hpbGQgPT4ge1xuICAgICAgICAgICAgY2hpbGQucGFyZW50ID0gdGhpcztcbiAgICAgICAgfSk7XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogVXBkYXRlcyB0aGlzIGNvbXBvbmVudHMgcHJvcHMgb2JqZWN0XG4gICAgICogYmFzZWQgb24gYW4gaW5jb21pbmcgb2JqZWN0XG4gICAgICovXG4gICAgX3VwZGF0ZVByb3BzKGluY29taW5nUHJvcHMpe1xuICAgICAgICB0aGlzLnByb3BzID0gaW5jb21pbmdQcm9wcztcbiAgICAgICAgdGhpcy5wcm9wcy5jaGlsZHJlbiA9IGluY29taW5nUHJvcHMuY2hpbGRyZW4gfHwgW107XG4gICAgICAgIHRoaXMucHJvcHMubmFtZWRDaGlsZHJlbiA9IGluY29taW5nUHJvcHMubmFtZWRDaGlsZHJlbiB8fCB7fTtcbiAgICAgICAgdGhpcy5fc2V0dXBDaGlsZFJlbGF0aW9uc2hpcHMoKTtcbiAgICB9XG5cbiAgICAvKipcbiAgICAgKiBSZWN1cnNpdmVseSBtYXBzIGEgb25lIG9yIG11bHRpZGltZW5zaW9uYWxcbiAgICAgKiBuYW1lZCBjaGlsZHJlbiB2YWx1ZSB3aXRoIHRoZSBnaXZlbiBtYXBwaW5nXG4gICAgICogZnVuY3Rpb24uXG4gICAgICovXG4gICAgX3JlY3Vyc2l2ZWx5TWFwTmFtZWRDaGlsZHJlbihjb2xsZWN0aW9uLCBjYWxsYmFjayl7XG4gICAgICAgIHJldHVybiBjb2xsZWN0aW9uLm1hcChpdGVtID0+IHtcbiAgICAgICAgICAgIGlmKEFycmF5LmlzQXJyYXkoaXRlbSkpe1xuICAgICAgICAgICAgICAgIHJldHVybiB0aGlzLl9yZWN1cnNpdmVseU1hcE5hbWVkQ2hpbGRyZW4oaXRlbSwgY2FsbGJhY2spO1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICByZXR1cm4gY2FsbGJhY2soaXRlbSk7XG4gICAgICAgICAgICB9XG4gICAgICAgIH0pO1xuICAgIH1cbn07XG5cbmV4cG9ydCB7Q29tcG9uZW50LCBDb21wb25lbnQgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIENvbnRhaW5lciBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIGEgc2luZ2xlIHJlZ3VsYXJcbiAqIHJlcGxhY2VtZW50OlxuICogKiBgY2hpbGRgXG4gKi9cbmNsYXNzIENvbnRhaW5lciBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIGxldCBjaGlsZCA9IHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdjaGlsZCcpO1xuICAgICAgICBsZXQgc3R5bGUgPSBcIlwiO1xuICAgICAgICBpZighY2hpbGQpe1xuICAgICAgICAgICAgc3R5bGUgPSBcImRpc3BsYXk6bm9uZTtcIjtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkNvbnRhaW5lclwiLFxuICAgICAgICAgICAgICAgIGNsYXNzOiBcImNlbGxcIixcbiAgICAgICAgICAgICAgICBzdHlsZTogc3R5bGVcbiAgICAgICAgICAgIH0sIFtjaGlsZF0pXG4gICAgICAgICk7XG4gICAgfVxufVxuXG5leHBvcnQge0NvbnRhaW5lciwgQ29udGFpbmVyIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBDb250ZXh0dWFsRGlzcGxheSBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIGEgc2luZ2xlXG4gKiByZWd1bGFyIHJlcGxhY2VtZW50OlxuICogKiBgY2hpbGRgXG4gKi9cbmNsYXNzIENvbnRleHR1YWxEaXNwbGF5IGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIGgoJ2RpdicsXG4gICAgICAgICAgICB7XG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbCBjb250ZXh0dWFsRGlzcGxheVwiLFxuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkNvbnRleHR1YWxEaXNwbGF5XCJcbiAgICAgICAgICAgIH0sIFt0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY2hpbGQnKV1cbiAgICAgICAgKTtcbiAgICB9XG59XG5cbmV4cG9ydCB7Q29udGV4dHVhbERpc3BsYXksIENvbnRleHR1YWxEaXNwbGF5IGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBEcm9wZG93biBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIG9uZSByZWd1bGFyXG4gKiByZXBsYWNlbWVudDpcbiAqICogYHRpdGxlYFxuICogVGhpcyBjb21wb25lbnQgaGFzIG9uZVxuICogZW51bWVyYXRlZCByZXBsYWNlbWVudDpcbiAqICogYGNoaWxkYFxuICovXG5jbGFzcyBEcm9wZG93biBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb250ZXh0IHRvIG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlSXRlbXMgPSB0aGlzLm1ha2VJdGVtcy5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkRyb3Bkb3duXCIsXG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiYnRuLWdyb3VwXCJcbiAgICAgICAgICAgIH0sIFtcbiAgICAgICAgICAgICAgICBoKCdhJywge2NsYXNzOiBcImJ0biBidG4teHMgYnRuLW91dGxpbmUtc2Vjb25kYXJ5XCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCd0aXRsZScpXG4gICAgICAgICAgICAgICAgXSksXG4gICAgICAgICAgICAgICAgaCgnYnV0dG9uJywge1xuICAgICAgICAgICAgICAgICAgICBjbGFzczogXCJidG4gYnRuLXhzIGJ0bi1vdXRsaW5lLXNlY29uZGFyeSBkcm9wZG93bi10b2dnbGUgZHJvcGRvd24tdG9nZ2xlLXNwbGl0XCIsXG4gICAgICAgICAgICAgICAgICAgIHR5cGU6IFwiYnV0dG9uXCIsXG4gICAgICAgICAgICAgICAgICAgIGlkOiBgJHt0aGlzLnByb3BzLmV4dHJhRGF0YS50YXJnZXRJZGVudGl0eX0tZHJvcGRvd25NZW51QnV0dG9uYCxcbiAgICAgICAgICAgICAgICAgICAgXCJkYXRhLXRvZ2dsZVwiOiBcImRyb3Bkb3duXCJcbiAgICAgICAgICAgICAgICB9KSxcbiAgICAgICAgICAgICAgICBoKCdkaXYnLCB7Y2xhc3M6IFwiZHJvcGRvd24tbWVudVwifSwgdGhpcy5tYWtlSXRlbXMoKSlcbiAgICAgICAgICAgIF0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgbWFrZUl0ZW1zKCl7XG4gICAgICAgIC8vIEZvciBzb21lIHJlYXNvbiwgZHVlIGFnYWluIHRvIHRoZSBDZWxsIGltcGxlbWVudGF0aW9uLFxuICAgICAgICAvLyBzb21ldGltZXMgdGhlcmUgYXJlIG5vdCB0aGVzZSBjaGlsZCByZXBsYWNlbWVudHMuXG4gICAgICAgIGlmKCF0aGlzLnJlcGxhY2VtZW50cy5oYXNSZXBsYWNlbWVudCgnY2hpbGQnKSl7XG4gICAgICAgICAgICByZXR1cm4gW107XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50c0ZvcignY2hpbGQnKS5tYXAoKGVsZW1lbnQsIGlkeCkgPT4ge1xuICAgICAgICAgICAgcmV0dXJuIG5ldyBEcm9wZG93bkl0ZW0oe1xuICAgICAgICAgICAgICAgIGlkOiBgJHt0aGlzLnByb3BzLmlkfS1pdGVtLSR7aWR4fWAsXG4gICAgICAgICAgICAgICAgaW5kZXg6IGlkeCxcbiAgICAgICAgICAgICAgICBjaGlsZFN1YnN0aXR1dGU6IGVsZW1lbnQsXG4gICAgICAgICAgICAgICAgdGFyZ2V0SWRlbnRpdHk6IHRoaXMucHJvcHMuZXh0cmFEYXRhLnRhcmdldElkZW50aXR5LFxuICAgICAgICAgICAgICAgIGRyb3Bkb3duSXRlbUluZm86IHRoaXMucHJvcHMuZXh0cmFEYXRhLmRyb3Bkb3duSXRlbUluZm9cbiAgICAgICAgICAgIH0pLnJlbmRlcigpO1xuICAgICAgICB9KTtcbiAgICB9XG59XG5cblxuLyoqXG4gKiBBIHByaXZhdGUgc3ViY29tcG9uZW50IGZvciBlYWNoXG4gKiBEcm9wZG93biBtZW51IGl0ZW0uIFdlIG5lZWQgdGhpc1xuICogYmVjYXVzZSBvZiBob3cgY2FsbGJhY2tzIGFyZSBoYW5kbGVkXG4gKiBhbmQgYmVjYXVzZSB0aGUgQ2VsbHMgdmVyc2lvbiBkb2Vzbid0XG4gKiBhbHJlYWR5IGltcGxlbWVudCB0aGlzIGtpbmQgYXMgYSBzZXBhcmF0ZVxuICogZW50aXR5LlxuICovXG5jbGFzcyBEcm9wZG93bkl0ZW0gZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29udGV4dCB0byBtZXRob2RzXG4gICAgICAgIHRoaXMuY2xpY2tIYW5kbGVyID0gdGhpcy5jbGlja0hhbmRsZXIuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ2EnLCB7XG4gICAgICAgICAgICAgICAgY2xhc3M6IFwic3ViY2VsbCBjZWxsLWRyb3Bkb3duLWl0ZW0gZHJvcGRvd24taXRlbVwiLFxuICAgICAgICAgICAgICAgIGtleTogdGhpcy5wcm9wcy5pbmRleCxcbiAgICAgICAgICAgICAgICBvbmNsaWNrOiB0aGlzLmNsaWNrSGFuZGxlclxuICAgICAgICAgICAgfSwgW3RoaXMucHJvcHMuY2hpbGRTdWJzdGl0dXRlXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBjbGlja0hhbmRsZXIoZXZlbnQpe1xuICAgICAgICAvLyBUaGlzIGlzIHN1cGVyIGhhY2t5IGJlY2F1c2Ugb2YgdGhlXG4gICAgICAgIC8vIGN1cnJlbnQgQ2VsbCBpbXBsZW1lbnRhdGlvbi5cbiAgICAgICAgLy8gVGhpcyB3aG9sZSBjb21wb25lbnQgc3RydWN0dXJlIHNob3VsZCBiZSBoZWF2aWx5IHJlZmFjdG9yZWRcbiAgICAgICAgLy8gb25jZSB0aGUgQ2VsbHMgc2lkZSBvZiB0aGluZ3Mgc3RhcnRzIHRvIGNoYW5nZS5cbiAgICAgICAgbGV0IHdoYXRUb0RvID0gdGhpcy5wcm9wcy5kcm9wZG93bkl0ZW1JbmZvW3RoaXMucHJvcHMuaW5kZXgudG9TdHJpbmcoKV07XG4gICAgICAgIGlmKHdoYXRUb0RvID09ICdjYWxsYmFjaycpe1xuICAgICAgICAgICAgbGV0IHJlc3BvbnNlRGF0YSA9IHtcbiAgICAgICAgICAgICAgICBldmVudDogXCJtZW51XCIsXG4gICAgICAgICAgICAgICAgaXg6IHRoaXMucHJvcHMuaW5kZXgsXG4gICAgICAgICAgICAgICAgdGFyZ2V0X2NlbGw6IHRoaXMucHJvcHMudGFyZ2V0SWRlbnRpdHlcbiAgICAgICAgICAgIH07XG4gICAgICAgICAgICBjZWxsU29ja2V0LnNlbmRTdHJpbmcoSlNPTi5zdHJpbmdpZnkocmVzcG9uc2VEYXRhKSk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICB3aW5kb3cubG9jYXRpb24uaHJlZiA9IHdoYXRUb0RvO1xuICAgICAgICB9XG4gICAgfVxufVxuXG5leHBvcnQge0Ryb3Bkb3duLCBEcm9wZG93biBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogRXhwYW5kcyBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBUaGlzIGNvbXBvbmVudCBoYXMgdHdvXG4gKiByZWd1bGFyIHJlcGxhY2VtZW50czpcbiAqICogYGljb25gXG4gKiAqIGBjaGlsZGBcbiAqL1xuY2xhc3MgRXhwYW5kcyBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb250ZXh0IHRvIG1ldGhvZHNcbiAgICAgICAgdGhpcy5fZ2V0RXZlbnRzID0gdGhpcy5fZ2V0RXZlbnQuYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkV4cGFuZHNcIixcbiAgICAgICAgICAgICAgICBzdHlsZTogdGhpcy5wcm9wcy5leHRyYURhdGEuZGl2U3R5bGVcbiAgICAgICAgICAgIH0sXG4gICAgICAgICAgICAgICAgW1xuICAgICAgICAgICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgICAgICAgICBzdHlsZTogJ2Rpc3BsYXk6aW5saW5lLWJsb2NrO3ZlcnRpY2FsLWFsaWduOnRvcCcsXG4gICAgICAgICAgICAgICAgICAgICAgICBvbmNsaWNrOiB0aGlzLl9nZXRFdmVudCgnb25jbGljaycpXG4gICAgICAgICAgICAgICAgICAgIH0sXG4gICAgICAgICAgICAgICAgICAgICAgICBbdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ2ljb24nKV0pLFxuICAgICAgICAgICAgICAgICAgICBoKCdkaXYnLCB7c3R5bGU6J2Rpc3BsYXk6aW5saW5lLWJsb2NrJ30sXG4gICAgICAgICAgICAgICAgICAgICAgICBbdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ2NoaWxkJyldKSxcbiAgICAgICAgICAgICAgICBdXG4gICAgICAgICAgICApXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgX2dldEV2ZW50KGV2ZW50TmFtZSkge1xuICAgICAgICByZXR1cm4gdGhpcy5wcm9wcy5leHRyYURhdGEuZXZlbnRzW2V2ZW50TmFtZV07XG4gICAgfVxufVxuXG5leHBvcnQge0V4cGFuZHMsIEV4cGFuZHMgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIEdyaWQgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyAzIGVudW1lcmFibGVcbiAqIHJlcGxhY2VtZW50czpcbiAqICogYGhlYWRlcmBcbiAqICogYHJvd2xhYmVsYFxuICogKiBgY2hpbGRgXG4gKlxuICpcbiAqIE5PVEU6IENoaWxkIGlzIGEgMi1kaW1lbnNpb25hbFxuICogZW51bWVyYXRlZCByZXBsYWNlbWVudCFcbiAqL1xuY2xhc3MgR3JpZCBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb250ZXh0IHRvIG1ldGhvZHNcbiAgICAgICAgdGhpcy5fbWFrZUhlYWRlckVsZW1lbnRzID0gdGhpcy5fbWFrZUhlYWRlckVsZW1lbnRzLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuX21ha2VSb3dFbGVtZW50cyA9IHRoaXMuX21ha2VSb3dFbGVtZW50cy5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICBsZXQgdG9wVGFibGVIZWFkZXIgPSBudWxsO1xuICAgICAgICBpZih0aGlzLnByb3BzLmV4dHJhRGF0YS5oYXNUb3BIZWFkZXIpe1xuICAgICAgICAgICAgdG9wVGFibGVIZWFkZXIgPSBoKCd0aCcpO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCd0YWJsZScsIHtcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJHcmlkXCIsXG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbCB0YWJsZS1oc2Nyb2xsIHRhYmxlLXNtIHRhYmxlLXN0cmlwZWRcIlxuICAgICAgICAgICAgfSwgW1xuICAgICAgICAgICAgICAgIGgoJ3RoZWFkJywge30sIFtcbiAgICAgICAgICAgICAgICAgICAgaCgndHInLCB7fSwgW3RvcFRhYmxlSGVhZGVyLCAuLi50aGlzLl9tYWtlSGVhZGVyRWxlbWVudHMoKV0pXG4gICAgICAgICAgICAgICAgXSksXG4gICAgICAgICAgICAgICAgaCgndGJvZHknLCB7fSwgdGhpcy5fbWFrZVJvd0VsZW1lbnRzKCkpXG4gICAgICAgICAgICBdKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIF9tYWtlUm93RWxlbWVudHMoKXtcbiAgICAgICAgaWYgKHRoaXMucmVwbGFjZW1lbnRzLmhhc1JlcGxhY2VtZW50KCdjaGlsZCcpKSB7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRzRm9yKCdjaGlsZCcpLm1hcCgocm93LCByb3dJZHgpID0+IHtcbiAgICAgICAgICAgICAgICBsZXQgY29sdW1ucyA9IHJvdy5tYXAoKGNvbHVtbiwgY29sSWR4KSA9PiB7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICAgICAgICAgICAgICBoKCd0ZCcsIHtrZXk6IGAke3RoaXMucHJvcHMuaWR9LWdyaWQtY29sLSR7cm93SWR4fS0ke2NvbElkeH1gfSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNvbHVtblxuICAgICAgICAgICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgICAgICAgICAgKTtcbiAgICAgICAgICAgICAgICB9KTtcbiAgICAgICAgICAgICAgICBsZXQgcm93TGFiZWxFbCA9IG51bGw7XG4gICAgICAgICAgICAgICAgaWYodGhpcy5yZXBsYWNlbWVudHMuaGFzUmVwbGFjZW1lbnQoJ3Jvd2xhYmVsJykpe1xuICAgICAgICAgICAgICAgICAgICByb3dMYWJlbEVsID0gaCgndGgnLCB7a2V5OiBgJHt0aGlzLnByb3BzLmlkfS1ncmlkLXJvd2xibC0ke3Jvd0lkeH1gfSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRzRm9yKCdyb3dsYWJlbCcpW3Jvd0lkeF1cbiAgICAgICAgICAgICAgICAgICAgXSk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICAgICAgICAgIGgoJ3RyJywge2tleTogYCR7dGhpcy5wcm9wcy5pZH0tZ3JpZC1yb3ctJHtyb3dJZHh9YH0sIFtcbiAgICAgICAgICAgICAgICAgICAgICAgIHJvd0xhYmVsRWwsXG4gICAgICAgICAgICAgICAgICAgICAgICAuLi5jb2x1bW5zXG4gICAgICAgICAgICAgICAgICAgIF0pXG4gICAgICAgICAgICAgICAgKTtcbiAgICAgICAgICAgIH0pO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIFtdXG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBfbWFrZUhlYWRlckVsZW1lbnRzKCl7XG4gICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudHNGb3IoJ2hlYWRlcicpLm1hcCgoaGVhZGVyRWwsIGNvbElkeCkgPT4ge1xuICAgICAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgICAgICBoKCd0aCcsIHtrZXk6IGAke3RoaXMucHJvcHMuaWR9LWdyaWQtdGgtJHtjb2xJZHh9YH0sIFtcbiAgICAgICAgICAgICAgICAgICAgaGVhZGVyRWxcbiAgICAgICAgICAgICAgICBdKVxuICAgICAgICAgICAgKTtcbiAgICAgICAgfSk7XG4gICAgfVxufVxuXG5leHBvcnRcbntHcmlkLCBHcmlkIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBIZWFkZXJCYXIgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyB0aHJlZSBzZXBhcmF0ZVxuICogZW51bWVyYXRlZCByZXBsYWNlbWVudHM6XG4gKiAqIGBsZWZ0YFxuICogKiBgcmlnaHRgXG4gKiAqIGBjZW50ZXJgXG4gKi9cbmNsYXNzIEhlYWRlckJhciBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgLy8gQmluZCBjb250ZXh0IHRvIG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlRWxlbWVudHMgPSB0aGlzLm1ha2VFbGVtZW50cy5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLm1ha2VSaWdodCA9IHRoaXMubWFrZVJpZ2h0LmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMubWFrZUxlZnQgPSB0aGlzLm1ha2VMZWZ0LmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMubWFrZUNlbnRlciA9IHRoaXMubWFrZUNlbnRlci5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIGNsYXNzOiBcImNlbGwgcC0yIGJnLWxpZ2h0IGZsZXgtY29udGFpbmVyXCIsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiSGVhZGVyQmFyXCIsXG4gICAgICAgICAgICAgICAgc3R5bGU6ICdkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6YmFzZWxpbmU7J1xuICAgICAgICAgICAgfSwgW1xuICAgICAgICAgICAgICAgIHRoaXMubWFrZUxlZnQoKSxcbiAgICAgICAgICAgICAgICB0aGlzLm1ha2VDZW50ZXIoKSxcbiAgICAgICAgICAgICAgICB0aGlzLm1ha2VSaWdodCgpXG4gICAgICAgICAgICBdKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIG1ha2VMZWZ0KCl7XG4gICAgICAgIGxldCBpbm5lckVsZW1lbnRzID0gW107XG4gICAgICAgIGlmKHRoaXMucmVwbGFjZW1lbnRzLmhhc1JlcGxhY2VtZW50KCdsZWZ0Jykpe1xuICAgICAgICAgICAgaW5uZXJFbGVtZW50cyA9IHRoaXMubWFrZUVsZW1lbnRzKCdsZWZ0Jyk7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ2RpdicsIHtjbGFzczogXCJmbGV4LWl0ZW1cIiwgc3R5bGU6IFwiZmxleC1ncm93OjA7XCJ9LCBbXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgICAgICBjbGFzczogXCJmbGV4LWNvbnRhaW5lclwiLFxuICAgICAgICAgICAgICAgICAgICBzdHlsZTogJ2Rpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO2FsaWduLWl0ZW1zOmJhc2VsaW5lOydcbiAgICAgICAgICAgICAgICB9LCBpbm5lckVsZW1lbnRzKVxuICAgICAgICAgICAgXSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlQ2VudGVyKCl7XG4gICAgICAgIGxldCBpbm5lckVsZW1lbnRzID0gW107XG4gICAgICAgIGlmKHRoaXMucmVwbGFjZW1lbnRzLmhhc1JlcGxhY2VtZW50KCdjZW50ZXInKSl7XG4gICAgICAgICAgICBpbm5lckVsZW1lbnRzID0gdGhpcy5tYWtlRWxlbWVudHMoJ2NlbnRlcicpO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdkaXYnLCB7Y2xhc3M6IFwiZmxleC1pdGVtXCIsIHN0eWxlOiBcImZsZXgtZ3JvdzoxO1wifSwgW1xuICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICAgICAgY2xhc3M6IFwiZmxleC1jb250YWluZXJcIixcbiAgICAgICAgICAgICAgICAgICAgc3R5bGU6ICdkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OmNlbnRlcjthbGlnbi1pdGVtczpiYXNlbGluZTsnXG4gICAgICAgICAgICAgICAgfSwgaW5uZXJFbGVtZW50cylcbiAgICAgICAgICAgIF0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgbWFrZVJpZ2h0KCl7XG4gICAgICAgIGxldCBpbm5lckVsZW1lbnRzID0gW107XG4gICAgICAgIGlmKHRoaXMucmVwbGFjZW1lbnRzLmhhc1JlcGxhY2VtZW50KCdyaWdodCcpKXtcbiAgICAgICAgICAgIGlubmVyRWxlbWVudHMgPSB0aGlzLm1ha2VFbGVtZW50cygncmlnaHQnKTtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnZGl2Jywge2NsYXNzOiBcImZsZXgtaXRlbVwiLCBzdHlsZTogXCJmbGV4LWdyb3c6MDtcIn0sIFtcbiAgICAgICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgICAgIGNsYXNzOiBcImZsZXgtY29udGFpbmVyXCIsXG4gICAgICAgICAgICAgICAgICAgIHN0eWxlOiAnZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpjZW50ZXI7YWxpZ24taXRlbXM6YmFzZWxpbmU7J1xuICAgICAgICAgICAgICAgIH0sIGlubmVyRWxlbWVudHMpXG4gICAgICAgICAgICBdKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIG1ha2VFbGVtZW50cyhwb3NpdGlvbil7XG4gICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudHNGb3IocG9zaXRpb24pLm1hcChlbGVtZW50ID0+IHtcbiAgICAgICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICAgICAgaCgnc3BhbicsIHtjbGFzczogXCJmbGV4LWl0ZW0gcHgtM1wifSwgW2VsZW1lbnRdKVxuICAgICAgICAgICAgKTtcbiAgICAgICAgfSk7XG4gICAgfVxufVxuXG5leHBvcnQge0hlYWRlckJhciwgSGVhZGVyQmFyIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBMYXJnZVBlbmRpbmdEb3dubG9hZERpc3BsYXkgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbmNsYXNzIExhcmdlUGVuZGluZ0Rvd25sb2FkRGlzcGxheSBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgaWQ6ICdvYmplY3RfZGF0YWJhc2VfbGFyZ2VfcGVuZGluZ19kb3dubG9hZF90ZXh0JyxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJMYXJnZVBlbmRpbmdEb3dubG9hZERpc3BsYXlcIixcbiAgICAgICAgICAgICAgICBjbGFzczogXCJjZWxsXCJcbiAgICAgICAgICAgIH0pXG4gICAgICAgICk7XG4gICAgfVxufVxuXG5leHBvcnQge0xhcmdlUGVuZGluZ0Rvd25sb2FkRGlzcGxheSwgTGFyZ2VQZW5kaW5nRG93bmxvYWREaXNwbGF5IGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBMb2FkQ29udGVudHNGcm9tVXJsIENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG5jbGFzcyBMb2FkQ29udGVudHNGcm9tVXJsIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIkxvYWRDb250ZW50c0Zyb21VcmxcIixcbiAgICAgICAgICAgIH0sIFtoKCdkaXYnLCB7aWQ6IHRoaXMucHJvcHMuZXh0cmFEYXRhWydsb2FkVGFyZ2V0SWQnXX0sIFtdKV1cbiAgICAgICAgICAgIClcbiAgICAgICAgKTtcbiAgICB9XG5cbn1cblxuZXhwb3J0IHtMb2FkQ29udGVudHNGcm9tVXJsLCBMb2FkQ29udGVudHNGcm9tVXJsIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBNYWluIENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyBhIG9uZVxuICogcmVndWxhci1raW5kIHJlcGxhY2VtZW50OlxuICogKiBgY2hpbGRgXG4gKi9cbmNsYXNzIE1haW4gZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnbWFpbicsIHtcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBjbGFzczogXCJweS1tZC0yXCIsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiTWFpblwiXG4gICAgICAgICAgICB9LCBbXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge2NsYXNzOiBcImNvbnRhaW5lci1mbHVpZFwifSwgW1xuICAgICAgICAgICAgICAgICAgICB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY2hpbGQnKVxuICAgICAgICAgICAgICAgIF0pXG4gICAgICAgICAgICBdKVxuICAgICAgICApO1xuICAgIH1cbn1cblxuZXhwb3J0IHtNYWluLCBNYWluIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBNb2RhbCBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogTW9kYWwgaGFzIHRoZSBmb2xsb3dpbmcgc2luZ2xlIHJlcGxhY2VtZW50czpcbiAqICpgdGl0bGVgXG4gKiAqYG1lc3NhZ2VgXG4gKiBBbmQgaGFzIHRoZSBmb2xsb3dpbmcgZW51bWVyYXRlZFxuICogcmVwbGFjZW1lbnRzXG4gKiAqIGBidXR0b25gXG4gKi9cbmNsYXNzIE1vZGFsIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcbiAgICAgICAgdGhpcy5tYWluU3R5bGUgPSAnZGlzcGxheTpibG9jaztwYWRkaW5nLXJpZ2h0OjE1cHg7JztcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICBjbGFzczogXCJjZWxsIG1vZGFsIGZhZGUgc2hvd1wiLFxuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIk1vZGFsXCIsXG4gICAgICAgICAgICAgICAgcm9sZTogXCJkaWFsb2dcIixcbiAgICAgICAgICAgICAgICBzdHlsZTogbWFpblN0eWxlXG4gICAgICAgICAgICB9LCBbXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge3JvbGU6IFwiZG9jdW1lbnRcIiwgY2xhc3M6IFwibW9kYWwtZGlhbG9nXCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtjbGFzczogXCJtb2RhbC1jb250ZW50XCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgICAgICBoKCdkaXYnLCB7Y2xhc3M6IFwibW9kYWwtaGVhZGVyXCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgaCgnaDUnLCB7Y2xhc3M6IFwibW9kYWwtdGl0bGVcIn0sIFtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ3RpdGxlJylcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBdKVxuICAgICAgICAgICAgICAgICAgICAgICAgXSksXG4gICAgICAgICAgICAgICAgICAgICAgICBoKCdkaXYnLCB7Y2xhc3M6IFwibW9kYWwtYm9keVwifSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdtZXNzYWdlJylcbiAgICAgICAgICAgICAgICAgICAgICAgIF0pLFxuICAgICAgICAgICAgICAgICAgICAgICAgaCgnZGl2Jywge2NsYXNzOiBcIm1vZGFsLWZvb3RlclwifSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50c0ZvcignYnV0dG9uJylcbiAgICAgICAgICAgICAgICAgICAgICAgIF0pXG4gICAgICAgICAgICAgICAgICAgIF0pXG4gICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgIF0pXG4gICAgICAgICk7XG4gICAgfVxufVxuXG5leHBvcnQge01vZGFsLCBNb2RhbCBhcyBkZWZhdWx0fVxuIiwiLyoqXG4gKiBPY3RpY29uIENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG5jbGFzcyBPY3RpY29uIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICAvLyBCaW5kIGNvbnRleHQgdG8gbWV0aG9kc1xuICAgICAgICB0aGlzLl9nZXRIVE1MQ2xhc3NlcyA9IHRoaXMuX2dldEhUTUxDbGFzc2VzLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybihcbiAgICAgICAgICAgIGgoJ3NwYW4nLCB7XG4gICAgICAgICAgICAgICAgY2xhc3M6IHRoaXMuX2dldEhUTUxDbGFzc2VzKCksXG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiT2N0aWNvblwiLFxuICAgICAgICAgICAgICAgIFwiYXJpYS1oaWRkZW5cIjogdHJ1ZSxcbiAgICAgICAgICAgICAgICBzdHlsZTogdGhpcy5wcm9wcy5leHRyYURhdGEuZGl2U3R5bGVcbiAgICAgICAgICAgIH0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgX2dldEhUTUxDbGFzc2VzKCl7XG4gICAgICAgIGxldCBjbGFzc2VzID0gW1wiY2VsbFwiLCBcIm9jdGljb25cIl07XG4gICAgICAgIHRoaXMucHJvcHMuZXh0cmFEYXRhLm9jdGljb25DbGFzc2VzLmZvckVhY2gobmFtZSA9PiB7XG4gICAgICAgICAgICBjbGFzc2VzLnB1c2gobmFtZSk7XG4gICAgICAgIH0pO1xuICAgICAgICByZXR1cm4gY2xhc3Nlcy5qb2luKFwiIFwiKTtcbiAgICB9XG59XG5cbmV4cG9ydCB7T2N0aWNvbiwgT2N0aWNvbiBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogUGFkZGluZyBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuY2xhc3MgUGFkZGluZyBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdzcGFuJywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIlBhZGRpbmdcIixcbiAgICAgICAgICAgICAgICBjbGFzczogXCJweC0yXCJcbiAgICAgICAgICAgIH0sIFtcIiBcIl0pXG4gICAgICAgICk7XG4gICAgfVxufVxuXG5leHBvcnQge1BhZGRpbmcsIFBhZGRpbmcgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIFBsb3QgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGNvbnRhaW5zIHRoZSBmb2xsb3dpbmdcbiAqIHJlZ3VsYXIgcmVwbGFjZW1lbnRzOlxuICogKiBgY2hhcnQtdXBkYXRlcmBcbiAqICogYGVycm9yYFxuICovXG5jbGFzcyBQbG90IGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICB0aGlzLnNldHVwUGxvdCA9IHRoaXMuc2V0dXBQbG90LmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgY29tcG9uZW50RGlkTG9hZCgpIHtcbiAgICAgICAgdGhpcy5zZXR1cFBsb3QoKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJQbG90XCIsXG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbFwiXG4gICAgICAgICAgICB9LCBbXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge2lkOiBgcGxvdCR7dGhpcy5wcm9wcy5pZH1gLCBzdHlsZTogdGhpcy5wcm9wcy5leHRyYURhdGEuZGl2U3R5bGV9KSxcbiAgICAgICAgICAgICAgICB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY2hhcnQtdXBkYXRlcicpLFxuICAgICAgICAgICAgICAgIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdlcnJvcicpXG4gICAgICAgICAgICBdKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIHNldHVwUGxvdCgpe1xuICAgICAgICBjb25zb2xlLmxvZyhcIlNldHRpbmcgdXAgYSBuZXcgcGxvdGx5IGNoYXJ0LlwiKTtcbiAgICAgICAgLy8gVE9ETyBUaGVzZSBhcmUgZ2xvYmFsIHZhciBkZWZpbmVkIGluIHBhZ2UuaHRtbFxuICAgICAgICAvLyB3ZSBzaG91bGQgZG8gc29tZXRoaW5nIGFib3V0IHRoaXMuXG4gICAgICAgIHZhciBwbG90RGl2ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3Bsb3QnICsgdGhpcy5wcm9wcy5pZCk7XG4gICAgICAgIFBsb3RseS5wbG90KFxuICAgICAgICAgICAgcGxvdERpdixcbiAgICAgICAgICAgIFtdLFxuICAgICAgICAgICAge1xuICAgICAgICAgICAgICAgIG1hcmdpbjoge3QgOiAzMCwgbDogMzAsIHI6IDMwLCBiOjMwIH0sXG4gICAgICAgICAgICAgICAgeGF4aXM6IHtyYW5nZXNsaWRlcjoge3Zpc2libGU6IGZhbHNlfX1cbiAgICAgICAgICAgIH0sXG4gICAgICAgICAgICB7IHNjcm9sbFpvb206IHRydWUsIGRyYWdtb2RlOiAncGFuJywgZGlzcGxheWxvZ286IGZhbHNlLCBkaXNwbGF5TW9kZUJhcjogJ2hvdmVyJyxcbiAgICAgICAgICAgICAgICBtb2RlQmFyQnV0dG9uczogWyBbJ3BhbjJkJ10sIFsnem9vbTJkJ10sIFsnem9vbUluMmQnXSwgWyd6b29tT3V0MmQnXSBdIH1cbiAgICAgICAgKTtcbiAgICAgICAgcGxvdERpdi5vbigncGxvdGx5X3JlbGF5b3V0JyxcbiAgICAgICAgICAgIGZ1bmN0aW9uKGV2ZW50ZGF0YSl7XG4gICAgICAgICAgICAgICAgaWYgKHBsb3REaXYuaXNfc2VydmVyX2RlZmluZWRfbW92ZSA9PT0gdHJ1ZSkge1xuICAgICAgICAgICAgICAgICAgICByZXR1cm5cbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgLy9pZiB3ZSdyZSBzZW5kaW5nIGEgc3RyaW5nLCB0aGVuIGl0cyBhIGRhdGUgb2JqZWN0LCBhbmQgd2Ugd2FudCB0byBzZW5kXG4gICAgICAgICAgICAgICAgLy8gYSB0aW1lc3RhbXBcbiAgICAgICAgICAgICAgICBpZiAodHlwZW9mKGV2ZW50ZGF0YVsneGF4aXMucmFuZ2VbMF0nXSkgPT09ICdzdHJpbmcnKSB7XG4gICAgICAgICAgICAgICAgICAgIGV2ZW50ZGF0YSA9IE9iamVjdC5hc3NpZ24oe30sZXZlbnRkYXRhKTtcbiAgICAgICAgICAgICAgICAgICAgZXZlbnRkYXRhW1wieGF4aXMucmFuZ2VbMF1cIl0gPSBEYXRlLnBhcnNlKGV2ZW50ZGF0YVtcInhheGlzLnJhbmdlWzBdXCJdKSAvIDEwMDAuMDtcbiAgICAgICAgICAgICAgICAgICAgZXZlbnRkYXRhW1wieGF4aXMucmFuZ2VbMV1cIl0gPSBEYXRlLnBhcnNlKGV2ZW50ZGF0YVtcInhheGlzLnJhbmdlWzFdXCJdKSAvIDEwMDAuMDtcbiAgICAgICAgICAgICAgICB9XG5cbiAgICAgICAgICAgICAgICBsZXQgcmVzcG9uc2VEYXRhID0ge1xuICAgICAgICAgICAgICAgICAgICAnZXZlbnQnOidwbG90X2xheW91dCcsXG4gICAgICAgICAgICAgICAgICAgICd0YXJnZXRfY2VsbCc6ICdfX2lkZW50aXR5X18nLFxuICAgICAgICAgICAgICAgICAgICAnZGF0YSc6IGV2ZW50ZGF0YVxuICAgICAgICAgICAgICAgIH07XG4gICAgICAgICAgICAgICAgY2VsbFNvY2tldC5zZW5kU3RyaW5nKEpTT04uc3RyaW5naWZ5KHJlc3BvbnNlRGF0YSkpO1xuICAgICAgICAgICAgfSk7XG4gICAgfVxufVxuXG5leHBvcnQge1Bsb3QsIFBsb3QgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIFBvcG92ZXIgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiBUaGlzIGNvbXBvbmVudCBjb250YWlucyB0aGUgZm9sbG93aW5nXG4gKiByZWd1bGFyIHJlcGxhY2VtZW50czpcbiAqICogYHRpdGxlYFxuICogKiBgZGV0YWlsYFxuICogKiBgY29udGVudHNgXG4gKi9cbmNsYXNzIFBvcG92ZXIgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gaCgnZGl2JyxcbiAgICAgICAgICAgIHtcbiAgICAgICAgICAgICAgICBjbGFzczogXCJjZWxsXCIsXG4gICAgICAgICAgICAgICAgc3R5bGU6IHRoaXMucHJvcHMuZXh0cmFEYXRhLmRpdlN0eWxlLFxuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIlBvcG92ZXJcIlxuICAgICAgICAgICAgfSwgW1xuICAgICAgICAgICAgICAgIGgoJ2EnLFxuICAgICAgICAgICAgICAgICAgICB7XG4gICAgICAgICAgICAgICAgICAgICAgICBocmVmOiBcIiNwb3BtYWluX1wiICsgdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICAgICAgICAgIFwiZGF0YS10b2dnbGVcIjogXCJwb3BvdmVyXCIsXG4gICAgICAgICAgICAgICAgICAgICAgICBcImRhdGEtdHJpZ2dlclwiOiBcImZvY3VzXCIsXG4gICAgICAgICAgICAgICAgICAgICAgICBcImRhdGEtYmluZFwiOiBcIiNwb3BfXCIgKyB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgICAgICAgICAgXCJkYXRhLXBsYWNlbWVudFwiOiBcImJvdHRvbVwiLFxuICAgICAgICAgICAgICAgICAgICAgICAgcm9sZTogXCJidXR0b25cIixcbiAgICAgICAgICAgICAgICAgICAgICAgIGNsYXNzOiBcImJ0biBidG4teHNcIlxuICAgICAgICAgICAgICAgICAgICB9LFxuICAgICAgICAgICAgICAgICAgICBbdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ2NvbnRlbnRzJyldXG4gICAgICAgICAgICAgICAgKSxcbiAgICAgICAgICAgICAgICBoKCdkaXYnLCB7c3R5bGU6IFwiZGlzcGxheTpub25lXCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgIGgoXCJkaXZcIiwge2lkOiBcInBvcF9cIiArIHRoaXMucHJvcHMuaWR9LCBbXG4gICAgICAgICAgICAgICAgICAgICAgICBoKFwiZGl2XCIsIHtjbGFzczogXCJkYXRhLXRpdGxlXCJ9LCBbdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoXCJ0aXRsZVwiKV0pLFxuICAgICAgICAgICAgICAgICAgICAgICAgaChcImRpdlwiLCB7Y2xhc3M6IFwiZGF0YS1jb250ZW50XCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgaChcImRpdlwiLCB7c3R5bGU6IFwid2lkdGg6IFwiICsgdGhpcy5wcm9wcy53aWR0aCArIFwicHhcIn0sIFtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ2RldGFpbCcpXVxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIClcbiAgICAgICAgICAgICAgICAgICAgICAgIF0pXG4gICAgICAgICAgICAgICAgICAgIF0pXG4gICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgIF1cbiAgICAgICAgKTtcbiAgICB9XG59XG5cbmV4cG9ydCB7UG9wb3ZlciwgUG9wb3ZlciBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogUm9vdENlbGwgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIGEgb25lXG4gKiByZWd1bGFyLWtpbmQgcmVwbGFjZW1lbnQ6XG4gKiAqIGBjaGlsZGBcbiAqL1xuY2xhc3MgUm9vdENlbGwgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIlJvb3RDZWxsXCJcbiAgICAgICAgICAgIH0sIFt0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignYycpXSlcbiAgICAgICAgKTtcbiAgICB9XG59XG5cbmV4cG9ydCB7Um9vdENlbGwsIFJvb3RDZWxsIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBTY3JvbGxhYmxlICBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIGEgb25lXG4gKiByZWd1bGFyLWtpbmQgcmVwbGFjZW1lbnQ6XG4gKiAqIGBjaGlsZGBcbiAqL1xuY2xhc3MgU2Nyb2xsYWJsZSBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdkaXYnLCB7XG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiU2Nyb2xsYWJsZVwiXG4gICAgICAgICAgICB9LCBbdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ2NoaWxkJyldKVxuICAgICAgICApO1xuICAgIH1cbn1cblxuZXhwb3J0IHtTY3JvbGxhYmxlLCBTY3JvbGxhYmxlIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBTZXF1ZW5jZSBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogU2VxdWVuY2UgaGFzIHRoZSBmb2xsb3dpbmcgZW51bWVyYXRlZFxuICogcmVwbGFjZW1lbnQ6XG4gKiAqIGBjYFxuICovXG5cbi8qKlxuICogQWJvdXQgTmFtZWQgQ2hpbGRyZW5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBgZWxlbWVudHNgIChhcnJheSkgLSBBIGxpc3Qgb2YgQ2VsbHMgdGhhdCBhcmUgaW4gdGhlXG4gKiAgICBzZXF1ZW5jZS5cbiAqL1xuY2xhc3MgU2VxdWVuY2UgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29tcG9uZW50IG1ldGhvZHNcbiAgICAgICAgdGhpcy5tYWtlRWxlbWVudHMgPSB0aGlzLm1ha2VFbGVtZW50cy5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIGNsYXNzOiBcImNlbGxcIixcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJTZXF1ZW5jZVwiLFxuICAgICAgICAgICAgICAgIHN0eWxlOiB0aGlzLnByb3BzLmV4dHJhRGF0YS5kaXZTdHlsZVxuICAgICAgICAgICAgfSwgdGhpcy5tYWtlRWxlbWVudHMoKSlcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlRWxlbWVudHMoKXtcbiAgICAgICAgaWYodGhpcy51c2VzUmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudHNGb3IoJ2MnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnJlbmRlckNoaWxkcmVuTmFtZWQoJ2VsZW1lbnRzJyk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cbmV4cG9ydCB7U2VxdWVuY2UsIFNlcXVlbmNlIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBTaGVldCBDZWxsIENvbXBvbmVudFxuICogTk9URTogVGhpcyBpcyBpbiBwYXJ0IGEgd3JhcHBlclxuICogZm9yIGhhbmRzb250YWJsZXMuXG4gKi9cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogVGhpcyBjb21wb25lbnQgaGFzIG9uZSByZWd1bGFyXG4gKiByZXBsYWNlbWVudDpcbiAqICogYGVycm9yYFxuICovXG5jbGFzcyBTaGVldCBleHRlbmRzIENvbXBvbmVudCB7XG4gICAgY29uc3RydWN0b3IocHJvcHMsIC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlcihwcm9wcywgLi4uYXJncyk7XG5cbiAgICAgICAgdGhpcy5jdXJyZW50VGFibGUgPSBudWxsO1xuXG4gICAgICAgIC8vIEJpbmQgY29udGV4dCB0byBtZXRob2RzXG4gICAgICAgIHRoaXMuaW5pdGlhbGl6ZVRhYmxlID0gdGhpcy5pbml0aWFsaXplVGFibGUuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5pbml0aWFsaXplSG9va3MgPSB0aGlzLmluaXRpYWxpemVIb29rcy5iaW5kKHRoaXMpO1xuXG4gICAgICAgIC8qKlxuICAgICAgICAgKiBXQVJJTklORzogVGhlIENlbGwgdmVyc2lvbiBvZiBTaGVldCBpcyBzdGlsbCB1c2luZ1xuICAgICAgICAgKiBjZXJ0aWFuIHBvc3RzY3JpcHRzIGJlY2F1c2Ugd2UgaGF2ZSBub3QgeWV0IHJlZmFjdG9yZWRcbiAgICAgICAgICogdGhlIHNvY2tldCBwcm90b2NvbC5cbiAgICAgICAgICogUmVtb3ZlIHRoaXMgd2FybmluZyBhYm91dCBpdCBvbmNlIHRoYXQgaGFwcGVucyFcbiAgICAgICAgICoqL1xuICAgICAgICBjb25zb2xlLndhcm4oYFtUT0RPXSBTaGVldCBzdGlsbCB1c2VzIGNlcnRhaW4gcG9zdHNjcmlwdHMgaW4gaXRzIGludGVyYWN0aW9uLiBTZWUgY29tcG9uZW50IGNvbnN0cnVjdG9yIGNvbW1lbnRgKTtcbiAgICB9XG5cbiAgICBjb21wb25lbnREaWRMb2FkKCl7XG4gICAgICAgIGNvbnNvbGUubG9nKGAjY29tcG9uZW50RGlkTG9hZCBjYWxsZWQgZm9yIFNoZWV0ICR7dGhpcy5wcm9wcy5pZH1gKTtcbiAgICAgICAgY29uc29sZS5sb2coYFRoaXMgc2hlZXQgaGFzIHRoZSBmb2xsb3dpbmcgcmVwbGFjZW1lbnRzOmAsIHRoaXMucmVwbGFjZW1lbnRzKTtcbiAgICAgICAgdGhpcy5pbml0aWFsaXplVGFibGUoKTtcbiAgICAgICAgaWYodGhpcy5wcm9wcy5leHRyYURhdGFbJ2hhbmRsZXNEb3VibGVDbGljayddKXtcbiAgICAgICAgICAgIHRoaXMuaW5pdGlhbGl6ZUhvb2tzKCk7XG4gICAgICAgIH1cbiAgICAgICAgLy8gUmVxdWVzdCBpbml0aWFsIGRhdGE/XG4gICAgICAgIGNlbGxTb2NrZXQuc2VuZFN0cmluZyhKU09OLnN0cmluZ2lmeSh7XG4gICAgICAgICAgICBldmVudDogXCJzaGVldF9uZWVkc19kYXRhXCIsXG4gICAgICAgICAgICB0YXJnZXRfY2VsbDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgIGRhdGE6IDBcbiAgICAgICAgfSkpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICBjb25zb2xlLmxvZyhgUmVuZGVyaW5nIHNoZWV0ICR7dGhpcy5wcm9wcy5pZH1gKTtcbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJTaGVldFwiLFxuICAgICAgICAgICAgICAgIGNsYXNzOiBcImNlbGxcIlxuICAgICAgICAgICAgfSwgW1xuICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtcbiAgICAgICAgICAgICAgICAgICAgaWQ6IGBzaGVldCR7dGhpcy5wcm9wcy5pZH1gLFxuICAgICAgICAgICAgICAgICAgICBzdHlsZTogdGhpcy5wcm9wcy5leHRyYURhdGEuZGl2U3R5bGUsXG4gICAgICAgICAgICAgICAgICAgIGNsYXNzOiBcImhhbmRzb250YWJsZVwiXG4gICAgICAgICAgICAgICAgfSwgW3RoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdlcnJvcicpXSlcbiAgICAgICAgICAgIF0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgaW5pdGlhbGl6ZVRhYmxlKCl7XG4gICAgICAgIGNvbnNvbGUubG9nKGAjaW5pdGlhbGl6ZVRhYmxlIGNhbGxlZCBmb3IgU2hlZXQgJHt0aGlzLnByb3BzLmlkfWApO1xuICAgICAgICBsZXQgZ2V0UHJvcGVydHkgPSBmdW5jdGlvbihpbmRleCl7XG4gICAgICAgICAgICByZXR1cm4gZnVuY3Rpb24ocm93KXtcbiAgICAgICAgICAgICAgICByZXR1cm4gcm93W2luZGV4XTtcbiAgICAgICAgICAgIH07XG4gICAgICAgIH07XG4gICAgICAgIGxldCBlbXB0eVJvdyA9IFtdO1xuICAgICAgICBsZXQgZGF0YU5lZWRlZENhbGxiYWNrID0gZnVuY3Rpb24oZXZlbnRPYmplY3Qpe1xuICAgICAgICAgICAgZXZlbnRPYmplY3QudGFyZ2V0X2NlbGwgPSB0aGlzLnByb3BzLmlkO1xuICAgICAgICAgICAgY2VsbFNvY2tldC5zZW5kU3RyaW5nKEpTT04uc3RyaW5naWZ5KGV2ZW50T2JqZWN0KSk7XG4gICAgICAgIH0uYmluZCh0aGlzKTtcbiAgICAgICAgbGV0IGRhdGEgPSBuZXcgU3ludGhldGljSW50ZWdlckFycmF5KHRoaXMucHJvcHMuZXh0cmFEYXRhLnJvd0NvdW50LCBlbXB0eVJvdywgZGF0YU5lZWRlZENhbGxiYWNrKTtcbiAgICAgICAgbGV0IGNvbnRhaW5lciA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKGBzaGVldCR7dGhpcy5wcm9wcy5pZH1gKTtcbiAgICAgICAgbGV0IGNvbHVtbk5hbWVzID0gdGhpcy5wcm9wcy5leHRyYURhdGEuY29sdW1uTmFtZXM7XG4gICAgICAgIGxldCBjb2x1bW5zID0gY29sdW1uTmFtZXMubWFwKChuYW1lLCBpZHgpID0+IHtcbiAgICAgICAgICAgIGVtcHR5Um93LnB1c2goXCJcIik7XG4gICAgICAgICAgICByZXR1cm4ge2RhdGE6IGdldFByb3BlcnR5KGlkeCl9O1xuICAgICAgICB9KTtcblxuICAgICAgICB0aGlzLmN1cnJlbnRUYWJsZSA9IG5ldyBIYW5kc29udGFibGUoY29udGFpbmVyLCB7XG4gICAgICAgICAgICBkYXRhLFxuICAgICAgICAgICAgZGF0YVNjaGVtYTogZnVuY3Rpb24ob3B0cyl7cmV0dXJuIHt9O30sXG4gICAgICAgICAgICBjb2xIZWFkZXJzOiBjb2x1bW5OYW1lcyxcbiAgICAgICAgICAgIGNvbHVtbnMsXG4gICAgICAgICAgICByb3dIZWFkZXJzOnRydWUsXG4gICAgICAgICAgICByb3dIZWFkZXJXaWR0aDogMTAwLFxuICAgICAgICAgICAgdmlld3BvcnRSb3dSZW5kZXJpbmdPZmZzZXQ6IDEwMCxcbiAgICAgICAgICAgIGF1dG9Db2x1bW5TaXplOiBmYWxzZSxcbiAgICAgICAgICAgIGF1dG9Sb3dIZWlnaHQ6IGZhbHNlLFxuICAgICAgICAgICAgbWFudWFsQ29sdW1uUmVzaXplOiB0cnVlLFxuICAgICAgICAgICAgY29sV2lkdGhzOiB0aGlzLnByb3BzLmV4dHJhRGF0YS5jb2x1bW5XaWR0aCxcbiAgICAgICAgICAgIHJvd0hlaWdodHM6IDIzLFxuICAgICAgICAgICAgcmVhZE9ubHk6IHRydWUsXG4gICAgICAgICAgICBNYW51YWxSb3dNb3ZlOiBmYWxzZVxuICAgICAgICB9KTtcbiAgICAgICAgaGFuZHNPblRhYmxlc1t0aGlzLnByb3BzLmlkXSA9IHtcbiAgICAgICAgICAgIHRhYmxlOiB0aGlzLmN1cnJlbnRUYWJsZSxcbiAgICAgICAgICAgIGxhc3RDZWxsQ2xpY2tlZDoge3JvdzogLTEwMCwgY29sOiAtMTAwfSxcbiAgICAgICAgICAgIGRibENsaWNrZWQ6IHRydWVcbiAgICAgICAgfTtcbiAgICB9XG5cbiAgICBpbml0aWFsaXplSG9va3MoKXtcbiAgICAgICAgSGFuZHNvbnRhYmxlLmhvb2tzLmFkZChcImJlZm9yZU9uQ2VsbE1vdXNlRG93blwiLCAoZXZlbnQsIGRhdGEpID0+IHtcbiAgICAgICAgICAgIGxldCBoYW5kc09uT2JqID0gaGFuZHNPblRhYmxlc1t0aGlzLnByb3BzLmlkXTtcbiAgICAgICAgICAgIGxldCBsYXN0Um93ID0gaGFuZHNPbk9iai5sYXN0Q2VsbENsaWNrZWQucm93O1xuICAgICAgICAgICAgbGV0IGxhc3RDb2wgPSBoYW5kc09uT2JqLmxhc3RDZWxsQ2xpY2tlZC5jb2w7XG5cbiAgICAgICAgICAgIGlmKChsYXN0Um93ID09IGRhdGEucm93KSAmJiAobGFzdENvbCA9IGRhdGEuY29sKSl7XG4gICAgICAgICAgICAgICAgaGFuZHNPbk9iai5kYmxDbGlja2VkID0gdHJ1ZTtcbiAgICAgICAgICAgICAgICBzZXRUaW1lb3V0KCgpID0+IHtcbiAgICAgICAgICAgICAgICAgICAgaWYoaGFuZHNPbk9iai5kYmxDbGlja2VkKXtcbiAgICAgICAgICAgICAgICAgICAgICAgIGNlbGxTb2NrZXQuc2VuZFN0cmluZyhKU09OLnN0cmluZ2lmeSh7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgZXZlbnQ6ICdvbkNlbGxEYmxDbGljaycsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgdGFyZ2V0X2NlbGw6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgcm93OiBkYXRhLnJvdyxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBjb2w6IGRhdGEuY29sXG4gICAgICAgICAgICAgICAgICAgICAgICB9KSk7XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgaGFuZHNPbk9iai5sYXN0Q2VsbENsaWNrZWQgPSB7cm93OiAtMTAwLCBjb2w6IC0xMDB9O1xuICAgICAgICAgICAgICAgICAgICBoYW5kc09uT2JqLmRibENsaWNrZWQgPSBmYWxzZTtcbiAgICAgICAgICAgICAgICB9LCAyMDApO1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICBoYW5kc09uT2JqLmxhc3RDZWxsQ2xpY2tlZCA9IHtyb3c6IGRhdGEucm93LCBjb2w6IGRhdGEuY29sfTtcbiAgICAgICAgICAgICAgICBzZXRUaW1lb3V0KCgpID0+IHtcbiAgICAgICAgICAgICAgICAgICAgaGFuZHNPbk9iai5sYXN0Q2VsbENsaWNrZWQgPSB7cm93OiAtMTAwLCBjb2w6IC0xMDB9O1xuICAgICAgICAgICAgICAgICAgICBoYW5kc09uT2JqLmRibENsaWNrZWQgPSBmYWxzZTtcbiAgICAgICAgICAgICAgICB9LCA2MDApO1xuICAgICAgICAgICAgfVxuICAgICAgICB9LCB0aGlzLmN1cnJlbnRUYWJsZSk7XG5cbiAgICAgICAgSGFuZHNvbnRhYmxlLmhvb2tzLmFkZChcImJlZm9yZU9uQ2VsbENvbnRleHRNZW51XCIsIChldmVudCwgZGF0YSkgPT4ge1xuICAgICAgICAgICAgbGV0IGhhbmRzT25PYmogPSBoYW5kc09uVGFibGVzW3RoaXMucHJvcHMuaWRdO1xuICAgICAgICAgICAgaGFuZHNPbk9iai5kYmxDbGlja2VkID0gZmFsc2U7XG4gICAgICAgICAgICBoYW5kc09uT2JqLmxhc3RDZWxsQ2xpY2tlZCA9IHtyb3c6IC0xMDAsIGNvbDogLTEwMH07XG4gICAgICAgIH0sIHRoaXMuY3VycmVudFRhYmxlKTtcblxuICAgICAgICBIYW5kc29udGFibGUuaG9va3MuYWRkKFwiYmVmb3JlQ29udGV4dE1lbnVTaG93XCIsIChldmVudCwgZGF0YSkgPT4ge1xuICAgICAgICAgICAgbGV0IGhhbmRzT25PYmogPSBoYW5kc09uVGFibGVzW3RoaXMucHJvcHMuaWRdO1xuICAgICAgICAgICAgaGFuZHNPbk9iai5kYmxDbGlja2VkID0gZmFsc2U7XG4gICAgICAgICAgICBoYW5kc09uT2JqLmxhc3RDZWxsQ2xpY2tlZCA9IHtyb3c6IC0xMDAsIGNvbDogLTEwMH07XG4gICAgICAgIH0sIHRoaXMuY3VycmVudFRhYmxlKTtcbiAgICB9XG59XG5cbi8qKiBDb3BpZWQgb3ZlciBmcm9tIENlbGxzIGltcGxlbWVudGF0aW9uICoqL1xuY29uc3QgU3ludGhldGljSW50ZWdlckFycmF5ID0gZnVuY3Rpb24oc2l6ZSwgZW1wdHlSb3cgPSBbXSwgY2FsbGJhY2spe1xuICAgIHRoaXMubGVuZ3RoID0gc2l6ZTtcbiAgICB0aGlzLmNhY2hlID0ge307XG4gICAgdGhpcy5wdXNoID0gZnVuY3Rpb24oKXt9O1xuICAgIHRoaXMuc3BsaWNlID0gZnVuY3Rpb24oKXt9O1xuXG4gICAgdGhpcy5zbGljZSA9IGZ1bmN0aW9uKGxvdywgaGlnaCl7XG4gICAgICAgIGlmKGhpZ2ggPT09IHVuZGVmaW5lZCl7XG4gICAgICAgICAgICBoaWdoID0gdGhpcy5sZW5ndGg7XG4gICAgICAgIH1cblxuICAgICAgICBsZXQgcmVzID0gQXJyYXkoaGlnaCAtIGxvdyk7XG4gICAgICAgIGxldCBpbml0TG93ID0gbG93O1xuICAgICAgICB3aGlsZShsb3cgPCBoaWdoKXtcbiAgICAgICAgICAgIGxldCBvdXQgPSB0aGlzLmNhY2hlW2xvd107XG4gICAgICAgICAgICBpZihvdXQgPT09IHVuZGVmaW5lZCl7XG4gICAgICAgICAgICAgICAgaWYoY2FsbGJhY2spe1xuICAgICAgICAgICAgICAgICAgICBjYWxsYmFjayh7XG4gICAgICAgICAgICAgICAgICAgICAgICBldmVudDogJ3NoZWV0X25lZWRzX2RhdGEnLFxuICAgICAgICAgICAgICAgICAgICAgICAgZGF0YTogbG93XG4gICAgICAgICAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICBvdXQgPSBlbXB0eVJvdztcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIHJlc1tsb3cgLSBpbml0TG93XSA9IG91dDtcbiAgICAgICAgICAgIGxvdyArPSAxO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiByZXM7XG4gICAgfTtcbn07XG5cbmV4cG9ydCB7U2hlZXQsIFNoZWV0IGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBTaW5nbGVMaW5lVGV4dEJveCBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuY2xhc3MgU2luZ2xlTGluZVRleHRCb3ggZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29udGV4dCB0byBtZXRob2RzXG4gICAgICAgIHRoaXMuY2hhbmdlSGFuZGxlciA9IHRoaXMuY2hhbmdlSGFuZGxlci5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICBsZXQgYXR0cnMgPVxuICAgICAgICAgICAge1xuICAgICAgICAgICAgICAgIGNsYXNzOiBcImNlbGxcIixcbiAgICAgICAgICAgICAgICBpZDogXCJ0ZXh0X1wiICsgdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICB0eXBlOiBcInRleHRcIixcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJTaW5nbGVMaW5lVGV4dEJveFwiLFxuICAgICAgICAgICAgICAgIG9uY2hhbmdlOiAoZXZlbnQpID0+IHt0aGlzLmNoYW5nZUhhbmRsZXIoZXZlbnQudGFyZ2V0LnZhbHVlKTt9XG4gICAgICAgICAgICB9O1xuICAgICAgICBpZiAodGhpcy5wcm9wcy5leHRyYURhdGEuaW5wdXRWYWx1ZSAhPT0gdW5kZWZpbmVkKSB7XG4gICAgICAgICAgICBhdHRycy5wYXR0ZXJuID0gdGhpcy5wcm9wcy5leHRyYURhdGEuaW5wdXRWYWx1ZTtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gaCgnaW5wdXQnLCBhdHRycywgW10pO1xuICAgIH1cblxuICAgIGNoYW5nZUhhbmRsZXIodmFsKSB7XG4gICAgICAgIGNlbGxTb2NrZXQuc2VuZFN0cmluZyhcbiAgICAgICAgICAgIEpTT04uc3RyaW5naWZ5KFxuICAgICAgICAgICAgICAgIHtcbiAgICAgICAgICAgICAgICAgICAgXCJldmVudFwiOiBcImNsaWNrXCIsXG4gICAgICAgICAgICAgICAgICAgIFwidGFyZ2V0X2NlbGxcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICAgICAgXCJ0ZXh0XCI6IHZhbFxuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIClcbiAgICAgICAgKTtcbiAgICB9XG59XG5cbmV4cG9ydCB7U2luZ2xlTGluZVRleHRCb3gsIFNpbmdsZUxpbmVUZXh0Qm94IGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBTcGFuIENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG5cbmNsYXNzIFNwYW4gZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnc3BhbicsIHtcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC1pZFwiOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLXR5cGVcIjogXCJTcGFuXCIsXG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbFwiXG4gICAgICAgICAgICB9LCBbdGhpcy5wcm9wcy5leHRyYURhdGEudGV4dF0pXG4gICAgICAgICk7XG4gICAgfVxufVxuXG5leHBvcnQge1NwYW4sIFNwYW4gYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIFN1YnNjcmliZWQgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyBhIHNpbmdsZVxuICogcmVndWxhciByZXBsYWNlbWVudDpcbiAqICogYGNvbnRlbnRzYFxuICovXG5jbGFzcyBTdWJzY3JpYmVkIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcbiAgICAgICAgLy90aGlzLmFkZFJlcGxhY2VtZW50KCdjb250ZW50cycsICdfX19fX2NvbnRlbnRzX18nKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIGgoJ2RpdicsXG4gICAgICAgICAgICB7XG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbCBzdWJzY3JpYmVkXCIsXG4gICAgICAgICAgICAgICAgc3R5bGU6IHRoaXMucHJvcHMuZXh0cmFEYXRhLmRpdlN0eWxlLFxuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIlN1YnNjcmliZWRcIlxuICAgICAgICAgICAgfSwgW3RoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdjb250ZW50cycpXVxuICAgICAgICApO1xuICAgIH1cbn1cblxuZXhwb3J0IHtTdWJzY3JpYmVkLCBTdWJzY3JpYmVkIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBTdWJzY3JpYmVkU2VxdWVuY2UgQ2VsbCBDb21wb25lbnRcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhcyBhIHNpbmdsZVxuICogZW51bWVyYXRlZCByZXBsYWNlbWVudDpcbiAqICogYGNoaWxkYFxuICovXG5jbGFzcyBTdWJzY3JpYmVkU2VxdWVuY2UgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuICAgICAgICAvL3RoaXMuYWRkUmVwbGFjZW1lbnQoJ2NvbnRlbnRzJywgJ19fX19fY29udGVudHNfXycpO1xuICAgICAgICAvL1xuICAgICAgICAvLyBCaW5kIGNvbnRleHQgdG8gbWV0aG9kc1xuICAgICAgICB0aGlzLm1ha2VDbGFzcyA9IHRoaXMubWFrZUNsYXNzLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMubWFrZUNoaWxkcmVuID0gdGhpcy5tYWtlQ2hpbGRyZW4uYmluZCh0aGlzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuIGgoJ2RpdicsXG4gICAgICAgICAgICB7XG4gICAgICAgICAgICAgICAgY2xhc3M6IHRoaXMubWFrZUNsYXNzKCksXG4gICAgICAgICAgICAgICAgc3R5bGU6IHRoaXMucHJvcHMuZXh0cmFEYXRhLmRpdlN0eWxlLFxuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIlN1YnNjcmliZWRTZXF1ZW5jZVwiXG4gICAgICAgICAgICB9LCBbdGhpcy5tYWtlQ2hpbGRyZW4oKV1cbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBtYWtlQ2xhc3MoKSB7XG4gICAgICAgIGlmICh0aGlzLnByb3BzLmV4dHJhRGF0YS5hc0NvbHVtbnMpIHtcbiAgICAgICAgICAgIHJldHVybiBcImNlbGwgc3Vic2NyaWJlZFNlcXVlbmNlIGNvbnRhaW5lci1mbHVpZFwiO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiBcImNlbGwgc3Vic2NyaWJlZFNlcXVlbmNlXCI7XG4gICAgfVxuXG4gICAgbWFrZUNoaWxkcmVuKCl7XG4gICAgICAgIGlmKHRoaXMucHJvcHMuZXh0cmFEYXRhLmFzQ29sdW1ucyl7XG4gICAgICAgICAgICBsZXQgZm9ybWF0dGVkQ2hpbGRyZW4gPSB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudHNGb3IoJ2NoaWxkJykubWFwKGNoaWxkRWxlbWVudCA9PiB7XG4gICAgICAgICAgICAgICAgcmV0dXJuKFxuICAgICAgICAgICAgICAgICAgICBoKCdkaXYnLCB7Y2xhc3M6IFwiY29sLXNtXCIsIGtleTogY2hpbGRFbGVtZW50LmlkfSwgW1xuICAgICAgICAgICAgICAgICAgICAgICAgaCgnc3BhbicsIHt9LCBbY2hpbGRFbGVtZW50XSlcbiAgICAgICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgICAgICApO1xuICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtjbGFzczogXCJyb3cgZmxleC1ub3dyYXBcIiwga2V5OiBgJHt0aGlzLnByb3BzLmlkfS1zcGluZS13cmFwcGVyYH0sIGZvcm1hdHRlZENoaWxkcmVuKVxuICAgICAgICAgICAgKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge2tleTogYCR7dGhpcy5wcm9wcy5pZH0tc3BpbmUtd3JhcHBlcmB9LCB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudHNGb3IoJ2NoaWxkJykpXG4gICAgICAgICAgICApO1xuICAgICAgICB9XG4gICAgfVxufVxuXG5leHBvcnQge1N1YnNjcmliZWRTZXF1ZW5jZSwgU3Vic2NyaWJlZFNlcXVlbmNlIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBUYWJsZSBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiBUaGlzIGNvbXBvbmVudCBoYXMgMyByZWd1bGFyXG4gKiByZXBsYWNlbWVudHM6XG4gKiAqIGBwYWdlYFxuICogKiBgbGVmdGBcbiAqICogYHJpZ2h0YFxuICogVGhpcyBjb21wb25lbnQgaGFzIDIgZW51bWVyYXRlZFxuICogcmVwbGFjZW1lbnRzOlxuICogKiBgY2hpbGRgXG4gKiAqIGBoZWFkZXJgXG4gKiBOT1RFOiBgY2hpbGRgIGVudW1lcmF0ZWQgcmVwbGFjZW1lbnRzXG4gKiBhcmUgdHdvIGRpbWVuc2lvbmFsIGFycmF5cyFcbiAqL1xuY2xhc3MgVGFibGUgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuXG4gICAgICAgIC8vIEJpbmQgY29udGV4dCB0byBtZXRob2RzXG4gICAgICAgIHRoaXMuX21ha2VIZWFkZXJFbGVtZW50cyA9IHRoaXMuX21ha2VIZWFkZXJFbGVtZW50cy5iaW5kKHRoaXMpO1xuICAgICAgICB0aGlzLl9tYWtlUm93RWxlbWVudHMgPSB0aGlzLl9tYWtlUm93RWxlbWVudHMuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5fbWFrZUZpcnN0Um93RWxlbWVudCA9IHRoaXMuX21ha2VGaXJzdFJvd0VsZW1lbnQuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5fdGhlYWRTdHlsZSA9IHRoaXMuX3RoZWFkU3R5bGUuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5fZ2V0Um93RGlzcGxheUVsZW1lbnRzID0gdGhpcy5fZ2V0Um93RGlzcGxheUVsZW1lbnRzLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIHJldHVybihcbiAgICAgICAgICAgIGgoJ3RhYmxlJywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIlRhYmxlXCIsXG4gICAgICAgICAgICAgICAgY2xhc3M6IFwiY2VsbCB0YWJsZS1oc2Nyb2xsIHRhYmxlLXNtIHRhYmxlLXN0cmlwZWRcIlxuICAgICAgICAgICAgfSwgW1xuICAgICAgICAgICAgICAgIGgoJ3RoZWFkJywge3N0eWxlOiB0aGlzLl90aGVhZFN0eWxlKCl9LFtcbiAgICAgICAgICAgICAgICAgICAgdGhpcy5fbWFrZUZpcnN0Um93RWxlbWVudCgpXG4gICAgICAgICAgICAgICAgXSksXG4gICAgICAgICAgICAgICAgaCgndGJvZHknLCB7fSwgdGhpcy5fbWFrZVJvd0VsZW1lbnRzKCkpXG4gICAgICAgICAgICBdKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIF90aGVhZFN0eWxlKCl7XG4gICAgICAgIHJldHVybiBcImJvcmRlci1ib3R0b206IGJsYWNrO2JvcmRlci1ib3R0b20tc3R5bGU6c29saWQ7Ym9yZGVyLWJvdHRvbS13aWR0aDp0aGluO1wiO1xuICAgIH1cblxuICAgIF9tYWtlSGVhZGVyRWxlbWVudHMoKXtcbiAgICAgICAgcmV0dXJuIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50c0ZvcignaGVhZGVyJykubWFwKChyZXBsYWNlbWVudCwgaWR4KSA9PiB7XG4gICAgICAgICAgICByZXR1cm4gaCgndGgnLCB7XG4gICAgICAgICAgICAgICAgc3R5bGU6IFwidmVydGljYWwtYWxpZ246dG9wO1wiLFxuICAgICAgICAgICAgICAgIGtleTogYCR7dGhpcy5wcm9wcy5pZH0tdGFibGUtaGVhZGVyLSR7aWR4fWBcbiAgICAgICAgICAgIH0sIFtyZXBsYWNlbWVudF0pO1xuICAgICAgICB9KTtcbiAgICB9XG5cbiAgICBfbWFrZVJvd0VsZW1lbnRzKCl7XG4gICAgICAgIC8vIE5vdGU6IHJvd3MgYXJlIHRoZSAqZmlyc3QqIGRpbWVuc2lvblxuICAgICAgICAvLyBpbiB0aGUgMi1kaW1lbnNpb25hbCBhcnJheSByZXR1cm5lZFxuICAgICAgICAvLyBieSBnZXR0aW5nIHRoZSBgY2hpbGRgIHJlcGxhY2VtZW50IGVsZW1lbnRzLlxuICAgICAgICByZXR1cm4gdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRzRm9yKCdjaGlsZCcpLm1hcCgocm93LCByb3dJZHgpID0+IHtcbiAgICAgICAgICAgIGxldCBjb2x1bW5zID0gcm93Lm1hcCgoY2hpbGRFbGVtZW50LCBjb2xJZHgpID0+IHtcbiAgICAgICAgICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgICAgICAgICBoKCd0ZCcsIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIGtleTogYCR7dGhpcy5wcm9wcy5pZH0tdGQtJHtyb3dJZHh9LSR7Y29sSWR4fWBcbiAgICAgICAgICAgICAgICAgICAgfSwgW2NoaWxkRWxlbWVudF0pXG4gICAgICAgICAgICAgICAgKTtcbiAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgbGV0IGluZGV4RWxlbWVudCA9IGgoJ3RkJywge30sIFtgJHtyb3dJZHggKyAxfWBdKTtcbiAgICAgICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICAgICAgaCgndHInLCB7a2V5OiBgJHt0aGlzLnByb3BzLmlkfS10ci0ke3Jvd0lkeH1gfSwgW2luZGV4RWxlbWVudCwgLi4uY29sdW1uc10pXG4gICAgICAgICAgICApO1xuICAgICAgICB9KTtcbiAgICB9XG5cbiAgICBfbWFrZUZpcnN0Um93RWxlbWVudCgpe1xuICAgICAgICBsZXQgaGVhZGVyRWxlbWVudHMgPSB0aGlzLl9tYWtlSGVhZGVyRWxlbWVudHMoKTtcbiAgICAgICAgcmV0dXJuKFxuICAgICAgICAgICAgaCgndHInLCB7fSwgW1xuICAgICAgICAgICAgICAgIGgoJ3RoJywge3N0eWxlOiBcInZlcnRpY2FsLWFsaWduOnRvcDtcIn0sIFtcbiAgICAgICAgICAgICAgICAgICAgaCgnZGl2Jywge2NsYXNzOiBcImNhcmRcIn0sIFtcbiAgICAgICAgICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtjbGFzczogXCJjYXJkLWJvZHkgcC0xXCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgLi4udGhpcy5fZ2V0Um93RGlzcGxheUVsZW1lbnRzKClcbiAgICAgICAgICAgICAgICAgICAgICAgIF0pXG4gICAgICAgICAgICAgICAgICAgIF0pXG4gICAgICAgICAgICAgICAgXSksXG4gICAgICAgICAgICAgICAgLi4uaGVhZGVyRWxlbWVudHNcbiAgICAgICAgICAgIF0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgX2dldFJvd0Rpc3BsYXlFbGVtZW50cygpe1xuICAgICAgICByZXR1cm4gW1xuICAgICAgICAgICAgdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ2xlZnQnKSxcbiAgICAgICAgICAgIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdyaWdodCcpLFxuICAgICAgICAgICAgdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ3BhZ2UnKSxcbiAgICAgICAgXTtcbiAgICB9XG59XG5cbmV4cG9ydCB7VGFibGUsIFRhYmxlIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBUYWJzIENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG5cbi8qKlxuICogQWJvdXQgUmVwbGFjZW1lbnRzXG4gKiAtLS0tLS0tLS0tLS0tLS0tLS1cbiAqIFRoaXMgY29tcG9uZW50IGhhZCBhIHNpbmdsZVxuICogcmVndWxhciByZXBsYWNlbWVudDpcbiAqICogYGRpc3BsYXlgXG4gKiBUaGlzIGNvbXBvbmVudCBoYXMgYSBzaW5nbGVcbiAqIGVudW1lcmF0ZWQgcmVwbGFjZW1lbnQ6XG4gKiAqIGBoZWFkZXJgXG4gKi9cbmNsYXNzIFRhYnMgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIlRhYnNcIixcbiAgICAgICAgICAgICAgICBjbGFzczogXCJjb250YWluZXItZmx1aWQgbWItM1wiXG4gICAgICAgICAgICB9LCBbXG4gICAgICAgICAgICAgICAgaCgndWwnLCB7Y2xhc3M6IFwibmF2IG5hdi10YWJzXCIsIHJvbGU6IFwidGFibGlzdFwifSwgW1xuICAgICAgICAgICAgICAgICAgICB0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudHNGb3IoJ2hlYWRlcicpXG4gICAgICAgICAgICAgICAgXSksXG4gICAgICAgICAgICAgICAgaCgnZGl2Jywge2NsYXNzOiBcInRhYi1jb250ZW50XCJ9LCBbXG4gICAgICAgICAgICAgICAgICAgIGgoJ2RpdicsIHtjbGFzczogXCJ0YWItcGFuZSBmYWRlIHNob3cgYWN0aXZlXCIsIHJvbGU6IFwidGFicGFuZWxcIn0sIFtcbiAgICAgICAgICAgICAgICAgICAgICAgIHRoaXMuZ2V0UmVwbGFjZW1lbnRFbGVtZW50Rm9yKCdkaXNwbGF5JylcbiAgICAgICAgICAgICAgICAgICAgXSlcbiAgICAgICAgICAgICAgICBdKVxuICAgICAgICAgICAgXSlcbiAgICAgICAgKTtcbiAgICB9XG59XG5cblxuZXhwb3J0IHtUYWJzLCBUYWJzIGFzIGRlZmF1bHR9O1xuIiwiLyoqXG4gKiBUZXh0IENlbGwgQ29tcG9uZW50XG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG5jbGFzcyBUZXh0IGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcbiAgICB9XG5cbiAgICByZW5kZXIoKXtcbiAgICAgICAgcmV0dXJuKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGNsYXNzOiBcImNlbGxcIixcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBzdHlsZTogdGhpcy5wcm9wcy5leHRyYURhdGEuZGl2U3R5bGUsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtaWRcIjogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBcImRhdGEtY2VsbC10eXBlXCI6IFwiVGV4dFwiXG4gICAgICAgICAgICB9LCBbdGhpcy5wcm9wcy5leHRyYURhdGEucmF3VGV4dF0pXG4gICAgICAgICk7XG4gICAgfVxufVxuXG5leHBvcnQge1RleHQsIFRleHQgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIFRyYWNlYmFjayBDZWxsIENvbXBvbmVudFxuICovXG5cbmltcG9ydCB7Q29tcG9uZW50fSBmcm9tICcuL0NvbXBvbmVudCc7XG5pbXBvcnQge2h9IGZyb20gJ21hcXVldHRlJztcblxuLyoqXG4gKiBBYm91dCBSZXBsYWNlbWVudHNcbiAqIC0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIGEgc2luZ2xlIHJlZ3VsYXJcbiAqIHJlcGFsY2VtZW50OlxuICogKiBgY2hpbGRgXG4gKi9cbmNsYXNzICBUcmFjZWJhY2sgZXh0ZW5kcyBDb21wb25lbnQge1xuICAgIGNvbnN0cnVjdG9yKHByb3BzLCAuLi5hcmdzKXtcbiAgICAgICAgc3VwZXIocHJvcHMsIC4uLmFyZ3MpO1xuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgaCgnZGl2Jywge1xuICAgICAgICAgICAgICAgIGlkOiB0aGlzLnByb3BzLmlkLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIlRyYWNlYmFja1wiLFxuICAgICAgICAgICAgICAgIGNsYXNzOiBcImFsZXJ0IGFsZXJ0LXByaW1hcnlcIlxuICAgICAgICAgICAgfSwgW1xuICAgICAgICAgICAgICAgIGgoJ3ByZScsIHt9LCBbdGhpcy5nZXRSZXBsYWNlbWVudEVsZW1lbnRGb3IoJ2NoaWxkJyldKVxuICAgICAgICAgICAgXSlcbiAgICAgICAgKTtcbiAgICB9XG59XG5cblxuZXhwb3J0IHtUcmFjZWJhY2ssIFRyYWNlYmFjayBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogX05hdlRhYiBDZWxsIENvbXBvbmVudFxuICogTk9URTogVGhpcyBzaG91bGQgcHJvYmFibHkganVzdCBiZVxuICogcm9sbGVkIGludG8gdGhlIE5hdiBjb21wb25lbnQgc29tZWhvdyxcbiAqIG9yIGluY2x1ZGVkIGluIGl0cyBtb2R1bGUgYXMgYSBwcml2YXRlXG4gKiBzdWJjb21wb25lbnQuXG4gKi9cblxuaW1wb3J0IHtDb21wb25lbnR9IGZyb20gJy4vQ29tcG9uZW50JztcbmltcG9ydCB7aH0gZnJvbSAnbWFxdWV0dGUnO1xuXG4vKipcbiAqIEFib3V0IFJlcGxhY2VtZW50c1xuICogLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogVGhpcyBjb21wb25lbnQgaGFzIG9uZSByZWd1bGFyXG4gKiByZXBsYWNlbWVudDpcbiAqICogYGNoaWxkYFxuICovXG5jbGFzcyBfTmF2VGFiIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICAvLyBCaW5kIGNvbnRleHQgdG8gbWV0aG9kc1xuICAgICAgICB0aGlzLmNsaWNrSGFuZGxlciA9IHRoaXMuY2xpY2tIYW5kbGVyLmJpbmQodGhpcyk7XG4gICAgfVxuXG4gICAgcmVuZGVyKCl7XG4gICAgICAgIGxldCBpbm5lckNsYXNzID0gXCJuYXYtbGlua1wiO1xuICAgICAgICBpZih0aGlzLnByb3BzLmV4dHJhRGF0YS5pc0FjdGl2ZSl7XG4gICAgICAgICAgICBpbm5lckNsYXNzICs9IFwiIGFjdGl2ZVwiO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICBoKCdsaScsIHtcbiAgICAgICAgICAgICAgICBpZDogdGhpcy5wcm9wcy5pZCxcbiAgICAgICAgICAgICAgICBjbGFzczogXCJuYXYtaXRlbVwiLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIl9OYXZUYWJcIlxuICAgICAgICAgICAgfSwgW1xuICAgICAgICAgICAgICAgIGgoJ2EnLCB7XG4gICAgICAgICAgICAgICAgICAgIGNsYXNzOiBpbm5lckNsYXNzLFxuICAgICAgICAgICAgICAgICAgICByb2xlOiBcInRhYlwiLFxuICAgICAgICAgICAgICAgICAgICBvbmNsaWNrOiB0aGlzLmNsaWNrSGFuZGxlclxuICAgICAgICAgICAgICAgIH0sIFt0aGlzLmdldFJlcGxhY2VtZW50RWxlbWVudEZvcignY2hpbGQnKV0pXG4gICAgICAgICAgICBdKVxuICAgICAgICApO1xuICAgIH1cblxuICAgIGNsaWNrSGFuZGxlcihldmVudCl7XG4gICAgICAgIGNlbGxTb2NrZXQuc2VuZFN0cmluZyhcbiAgICAgICAgICAgIEpTT04uc3RyaW5naWZ5KHRoaXMucHJvcHMuZXh0cmFEYXRhLmNsaWNrRGF0YSwgbnVsbCwgNClcbiAgICAgICAgKTtcbiAgICB9XG59XG5cbmV4cG9ydCB7X05hdlRhYiwgX05hdlRhYiBhcyBkZWZhdWx0fTtcbiIsIi8qKlxuICogX1Bsb3RVcGRhdGVyIENlbGwgQ29tcG9uZW50XG4gKiBOT1RFOiBMYXRlciByZWZhY3RvcmluZ3Mgc2hvdWxkIHJlc3VsdCBpblxuICogdGhpcyBjb21wb25lbnQgYmVjb21pbmcgb2Jzb2xldGVcbiAqL1xuXG5pbXBvcnQge0NvbXBvbmVudH0gZnJvbSAnLi9Db21wb25lbnQnO1xuaW1wb3J0IHtofSBmcm9tICdtYXF1ZXR0ZSc7XG5cbmNvbnN0IE1BWF9JTlRFUlZBTFMgPSAyNTtcblxuY2xhc3MgX1Bsb3RVcGRhdGVyIGV4dGVuZHMgQ29tcG9uZW50IHtcbiAgICBjb25zdHJ1Y3Rvcihwcm9wcywgLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKHByb3BzLCAuLi5hcmdzKTtcblxuICAgICAgICB0aGlzLnJ1blVwZGF0ZSA9IHRoaXMucnVuVXBkYXRlLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMubGlzdGVuRm9yUGxvdCA9IHRoaXMubGlzdGVuRm9yUGxvdC5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIGNvbXBvbmVudERpZExvYWQoKSB7XG4gICAgICAgIC8vIElmIHdlIGNhbiBmaW5kIGEgbWF0Y2hpbmcgUGxvdCBlbGVtZW50XG4gICAgICAgIC8vIGF0IHRoaXMgcG9pbnQsIHdlIHNpbXBseSB1cGRhdGUgaXQuXG4gICAgICAgIC8vIE90aGVyd2lzZSB3ZSBuZWVkIHRvICdsaXN0ZW4nIGZvciB3aGVuXG4gICAgICAgIC8vIGl0IGZpbmFsbHkgY29tZXMgaW50byB0aGUgRE9NLlxuICAgICAgICBsZXQgaW5pdGlhbFBsb3REaXYgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZChgcGxvdCR7dGhpcy5wcm9wcy5leHRyYURhdGEucGxvdElkfWApO1xuICAgICAgICBpZihpbml0aWFsUGxvdERpdil7XG4gICAgICAgICAgICB0aGlzLnJ1blVwZGF0ZShpbml0aWFsUGxvdERpdik7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICB0aGlzLmxpc3RlbkZvclBsb3QoKTtcbiAgICAgICAgfVxuICAgIH1cblxuICAgIHJlbmRlcigpe1xuICAgICAgICByZXR1cm4gaCgnZGl2JyxcbiAgICAgICAgICAgIHtcbiAgICAgICAgICAgICAgICBjbGFzczogXCJjZWxsXCIsXG4gICAgICAgICAgICAgICAgaWQ6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgc3R5bGU6IFwiZGlzcGxheTogbm9uZVwiLFxuICAgICAgICAgICAgICAgIFwiZGF0YS1jZWxsLWlkXCI6IHRoaXMucHJvcHMuaWQsXG4gICAgICAgICAgICAgICAgXCJkYXRhLWNlbGwtdHlwZVwiOiBcIl9QbG90VXBkYXRlclwiXG4gICAgICAgICAgICB9LCBbXSk7XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogSW4gdGhlIGV2ZW50IHRoYXQgYSBgX1Bsb3RVcGRhdGVyYCBoYXMgY29tZVxuICAgICAqIG92ZXIgdGhlIHdpcmUgKmJlZm9yZSogaXRzIGNvcnJlc3BvbmRpbmdcbiAgICAgKiBQbG90IGhhcyBjb21lIG92ZXIgKHdoaWNoIGFwcGVhcnMgdG8gYmVcbiAgICAgKiBjb21tb24pLCB3ZSB3aWxsIHNldCBhbiBpbnRlcnZhbCBvZiA1MG1zXG4gICAgICogYW5kIGNoZWNrIGZvciB0aGUgbWF0Y2hpbmcgUGxvdCBpbiB0aGUgRE9NXG4gICAgICogTUFYX0lOVEVSVkFMUyB0aW1lcywgb25seSBjYWxsaW5nIGBydW5VcGRhdGVgXG4gICAgICogb25jZSB3ZSd2ZSBmb3VuZCBhIG1hdGNoLlxuICAgICAqL1xuICAgIGxpc3RlbkZvclBsb3QoKXtcbiAgICAgICAgbGV0IG51bUNoZWNrcyA9IDA7XG4gICAgICAgIGxldCBwbG90Q2hlY2tlciA9IHdpbmRvdy5zZXRJbnRlcnZhbCgoKSA9PiB7XG4gICAgICAgICAgICBpZihudW1DaGVja3MgPiBNQVhfSU5URVJWQUxTKXtcbiAgICAgICAgICAgICAgICB3aW5kb3cuY2xlYXJJbnRlcnZhbChwbG90Q2hlY2tlcik7XG4gICAgICAgICAgICAgICAgY29uc29sZS5lcnJvcihgQ291bGQgbm90IGZpbmQgbWF0Y2hpbmcgUGxvdCAke3RoaXMucHJvcHMuZXh0cmFEYXRhLnBsb3RJZH0gZm9yIF9QbG90VXBkYXRlciAke3RoaXMucHJvcHMuaWR9YCk7XG4gICAgICAgICAgICAgICAgcmV0dXJuO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgbGV0IHBsb3REaXYgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZChgcGxvdCR7dGhpcy5wcm9wcy5leHRyYURhdGEucGxvdElkfWApO1xuICAgICAgICAgICAgaWYocGxvdERpdil7XG4gICAgICAgICAgICAgICAgdGhpcy5ydW5VcGRhdGUocGxvdERpdik7XG4gICAgICAgICAgICAgICAgd2luZG93LmNsZWFySW50ZXJ2YWwocGxvdENoZWNrZXIpO1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICBudW1DaGVja3MgKz0gMTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfSwgNTApO1xuICAgIH1cblxuICAgIHJ1blVwZGF0ZShhRE9NRWxlbWVudCl7XG4gICAgICAgIGNvbnNvbGUubG9nKFwiVXBkYXRpbmcgcGxvdGx5IGNoYXJ0LlwiKTtcbiAgICAgICAgLy8gVE9ETyBUaGVzZSBhcmUgZ2xvYmFsIHZhciBkZWZpbmVkIGluIHBhZ2UuaHRtbFxuICAgICAgICAvLyB3ZSBzaG91bGQgZG8gc29tZXRoaW5nIGFib3V0IHRoaXMuXG4gICAgICAgIGlmICh0aGlzLnByb3BzLmV4dHJhRGF0YS5leGNlcHRpb25PY2N1cmVkKSB7XG4gICAgICAgICAgICBjb25zb2xlLmxvZyhcInBsb3QgZXhjZXB0aW9uIG9jY3VyZWRcIik7XG4gICAgICAgICAgICBQbG90bHkucHVyZ2UoYURPTUVsZW1lbnQpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgbGV0IGRhdGEgPSB0aGlzLnByb3BzLmV4dHJhRGF0YS5wbG90RGF0YS5tYXAobWFwUGxvdGx5RGF0YSk7XG4gICAgICAgICAgICBQbG90bHkucmVhY3QoYURPTUVsZW1lbnQsIGRhdGEsIGFET01FbGVtZW50LmxheW91dCk7XG4gICAgICAgIH1cbiAgICB9XG59XG5cbmV4cG9ydCB7X1Bsb3RVcGRhdGVyLCBfUGxvdFVwZGF0ZXIgYXMgZGVmYXVsdH07XG4iLCIvKipcbiAqIFRvb2wgZm9yIFZhbGlkYXRpbmcgQ29tcG9uZW50IFByb3BlcnRpZXNcbiAqL1xuXG5jb25zdCByZXBvcnQgPSAobWVzc2FnZSwgZXJyb3JNb2RlLCBzaWxlbnRNb2RlKSA9PiB7XG4gICAgaWYoZXJyb3JNb2RlID09IHRydWUgJiYgc2lsZW50TW9kZSA9PSBmYWxzZSl7XG4gICAgICAgIGNvbnNvbGUuZXJyb3IobWVzc2FnZSk7XG4gICAgfSBlbHNlIGlmKHNpbGVudE1vZGUgPT0gZmFsc2Upe1xuICAgICAgICBjb25zb2xlLndhcm4obWVzc2FnZSk7XG4gICAgfVxufTtcblxuY29uc3QgUHJvcFR5cGVzID0ge1xuICAgIGVycm9yTW9kZTogZmFsc2UsXG4gICAgc2lsZW50TW9kZTogZmFsc2UsXG4gICAgb25lT2Y6IGZ1bmN0aW9uKGFuQXJyYXkpe1xuICAgICAgICByZXR1cm4gZnVuY3Rpb24oY29tcG9uZW50TmFtZSwgcHJvcE5hbWUsIHByb3BWYWx1ZSwgaXNSZXF1aXJlZCl7XG4gICAgICAgICAgICBmb3IobGV0IGkgPSAwOyBpIDwgYW5BcnJheS5sZW5ndGg7IGkrKyl7XG4gICAgICAgICAgICAgICAgbGV0IHR5cGVDaGVja0l0ZW0gPSBhbkFycmF5W2ldO1xuICAgICAgICAgICAgICAgIGlmKHR5cGVvZih0eXBlQ2hlY2tJdGVtKSA9PSAnZnVuY3Rpb24nKXtcbiAgICAgICAgICAgICAgICAgICAgaWYodHlwZUNoZWNrSXRlbShjb21wb25lbnROYW1lLCBwcm9wTmFtZSwgcHJvcFZhbHVlLCBpc1JlcXVpcmVkLCB0cnVlKSl7XG4gICAgICAgICAgICAgICAgICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIH0gZWxzZSBpZih0eXBlQ2hlY2tJdGVtID09IHByb3BWYWx1ZSl7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIGxldCBtZXNzYWdlID0gYCR7Y29tcG9uZW50TmFtZX0gPj4gJHtwcm9wTmFtZX0gbXVzdCBiZSBvZiBvbmUgb2YgdGhlIGZvbGxvd2luZyB0eXBlczogJHthbkFycmF5fWA7XG4gICAgICAgICAgICByZXBvcnQobWVzc2FnZSwgdGhpcy5lcnJvck1vZGUsIHRoaXMuc2lsZW50TW9kZSk7XG4gICAgICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICAgIH0uYmluZCh0aGlzKTtcbiAgICB9LFxuXG4gICAgZ2V0VmFsaWRhdG9yRm9yVHlwZSh0eXBlU3RyKXtcbiAgICAgICAgcmV0dXJuIGZ1bmN0aW9uKGNvbXBvbmVudE5hbWUsIHByb3BOYW1lLCBwcm9wVmFsdWUsIGlzUmVxdWlyZWQsIGluQ29tcG91bmQgPSBmYWxzZSl7XG4gICAgICAgICAgICAvLyBXZSBhcmUgJ2luIGEgY29tcG91bmQgdmFsaWRhdGlvbicgd2hlbiB0aGUgaW5kaXZpZHVhbFxuICAgICAgICAgICAgLy8gUHJvcFR5cGUgY2hlY2tlcnMgKGllIGZ1bmMsIG51bWJlciwgc3RyaW5nLCBldGMpIGFyZVxuICAgICAgICAgICAgLy8gYmVpbmcgY2FsbGVkIHdpdGhpbiBhIGNvbXBvdW5kIHR5cGUgY2hlY2tlciBsaWtlIG9uZU9mLlxuICAgICAgICAgICAgLy8gSW4gdGhlc2UgY2FzZXMgd2Ugd2FudCB0byBwcmV2ZW50IHRoZSBjYWxsIHRvIHJlcG9ydCxcbiAgICAgICAgICAgIC8vIHdoaWNoIHRoZSBjb21wb3VuZCBjaGVjayB3aWxsIGhhbmRsZSBvbiBpdHMgb3duLlxuICAgICAgICAgICAgaWYoaW5Db21wb3VuZCA9PSBmYWxzZSl7XG4gICAgICAgICAgICAgICAgaWYodHlwZW9mKHByb3BWYWx1ZSkgPT0gdHlwZVN0cil7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgICAgICAgICAgIH0gZWxzZSBpZighaXNSZXF1aXJlZCAmJiAocHJvcFZhbHVlID09IHVuZGVmaW5lZCB8fCBwcm9wVmFsdWUgPT0gbnVsbCkpe1xuICAgICAgICAgICAgICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgICAgICAgICAgICB9IGVsc2UgaWYoaXNSZXF1aXJlZCl7XG4gICAgICAgICAgICAgICAgICAgIGxldCBtZXNzYWdlID0gYCR7Y29tcG9uZW50TmFtZX0gPj4gJHtwcm9wTmFtZX0gaXMgYSByZXF1aXJlZCBwcm9wLCBidXQgYXMgcGFzc2VkIGFzICR7cHJvcFZhbHVlfSFgO1xuICAgICAgICAgICAgICAgICAgICByZXBvcnQobWVzc2FnZSwgdGhpcy5lcnJvck1vZGUsIHRoaXMuc2lsZW50TW9kZSk7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgICAgICBsZXQgbWVzc2FnZSA9IGAke2NvbXBvbmVudE5hbWV9ID4+ICR7cHJvcE5hbWV9IG11c3QgYmUgb2YgdHlwZSAke3R5cGVTdHJ9IWA7XG4gICAgICAgICAgICAgICAgICAgIHJlcG9ydChtZXNzYWdlLCB0aGlzLmVycm9yTW9kZSwgdGhpcy5zaWxlbnRNb2RlKTtcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIC8vIE90aGVyd2lzZSB0aGlzIGlzIGEgc3RyYWlnaHRmb3J3YXJkIHR5cGUgY2hlY2tcbiAgICAgICAgICAgIC8vIGJhc2VkIG9uIHRoZSBnaXZlbiB0eXBlLiBXZSBjaGVjayBhcyB1c3VhbCBmb3IgdGhlIHJlcXVpcmVkXG4gICAgICAgICAgICAvLyBwcm9wZXJ0eSBhbmQgdGhlbiB0aGUgYWN0dWFsIHR5cGUgbWF0Y2ggaWYgbmVlZGVkLlxuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICBpZihpc1JlcXVpcmVkICYmIChwcm9wVmFsdWUgPT0gdW5kZWZpbmVkIHx8IHByb3BWYWx1ZSA9PSBudWxsKSl7XG4gICAgICAgICAgICAgICAgICAgIGxldCBtZXNzYWdlID0gYCR7Y29tcG9uZW50TmFtZX0gPj4gJHtwcm9wTmFtZX0gaXMgYSByZXF1aXJlZCBwcm9wLCBidXQgd2FzIHBhc3NlZCBhcyAke3Byb3BWYWx1ZX0hYDtcbiAgICAgICAgICAgICAgICAgICAgcmVwb3J0KG1lc3NhZ2UsIHRoaXMuZXJyb3JNb2RlLCB0aGlzLnNpbGVudE1vZGUpO1xuICAgICAgICAgICAgICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICAgICAgICAgICAgfSBlbHNlIGlmKCFpc1JlcXVpcmVkICYmIChwcm9wVmFsdWUgPT0gdW5kZWZpbmVkIHx8IHByb3BWYWx1ZSA9PSBudWxsKSl7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICByZXR1cm4gdHlwZW9mKHByb3BWYWx1ZSkgPT0gdHlwZVN0cjtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfTtcbiAgICB9LFxuXG4gICAgZ2V0IG51bWJlcigpe1xuICAgICAgICByZXR1cm4gdGhpcy5nZXRWYWxpZGF0b3JGb3JUeXBlKCdudW1iZXInKS5iaW5kKHRoaXMpO1xuICAgIH0sXG5cbiAgICBnZXQgYm9vbGVhbigpe1xuICAgICAgICByZXR1cm4gdGhpcy5nZXRWYWxpZGF0b3JGb3JUeXBlKCdib29sZWFuJykuYmluZCh0aGlzKTtcbiAgICB9LFxuXG4gICAgZ2V0IHN0cmluZygpe1xuICAgICAgICByZXR1cm4gdGhpcy5nZXRWYWxpZGF0b3JGb3JUeXBlKCdzdHJpbmcnKS5iaW5kKHRoaXMpO1xuICAgIH0sXG5cbiAgICBnZXQgb2JqZWN0KCl7XG4gICAgICAgIHJldHVybiB0aGlzLmdldFZhbGlkYXRvckZvclR5cGUoJ29iamVjdCcpLmJpbmQodGhpcyk7XG4gICAgfSxcblxuICAgIGdldCBmdW5jKCl7XG4gICAgICAgIHJldHVybiB0aGlzLmdldFZhbGlkYXRvckZvclR5cGUoJ2Z1bmN0aW9uJykuYmluZCh0aGlzKTtcbiAgICB9LFxuXG4gICAgdmFsaWRhdGU6IGZ1bmN0aW9uKGNvbXBvbmVudE5hbWUsIHByb3BzLCBwcm9wSW5mbyl7XG4gICAgICAgIGxldCBwcm9wTmFtZXMgPSBuZXcgU2V0KE9iamVjdC5rZXlzKHByb3BzKSk7XG4gICAgICAgIHByb3BOYW1lcy5kZWxldGUoJ2NoaWxkcmVuJyk7XG4gICAgICAgIHByb3BOYW1lcy5kZWxldGUoJ25hbWVkQ2hpbGRyZW4nKTtcbiAgICAgICAgcHJvcE5hbWVzLmRlbGV0ZSgnaWQnKTtcbiAgICAgICAgbGV0IHByb3BzVG9WYWxpZGF0ZSA9IEFycmF5LmZyb20ocHJvcE5hbWVzKTtcblxuICAgICAgICAvLyBQZXJmb3JtIGFsbCB0aGUgdmFsaWRhdGlvbnMgb24gZWFjaCBwcm9wZXJ0eVxuICAgICAgICAvLyBhY2NvcmRpbmcgdG8gaXRzIGRlc2NyaXB0aW9uLiBXZSBzdG9yZSB3aGV0aGVyXG4gICAgICAgIC8vIG9yIG5vdCB0aGUgZ2l2ZW4gcHJvcGVydHkgd2FzIGNvbXBsZXRlbHkgdmFsaWRcbiAgICAgICAgLy8gYW5kIHRoZW4gZXZhbHVhdGUgdGhlIHZhbGlkaXR5IG9mIGFsbCBhdCB0aGUgZW5kLlxuICAgICAgICBsZXQgdmFsaWRhdGlvblJlc3VsdHMgPSB7fTtcbiAgICAgICAgcHJvcHNUb1ZhbGlkYXRlLmZvckVhY2gocHJvcE5hbWUgPT4ge1xuICAgICAgICAgICAgbGV0IHByb3BWYWwgPSBwcm9wc1twcm9wTmFtZV07XG4gICAgICAgICAgICBsZXQgdmFsaWRhdGlvblRvQ2hlY2sgPSBwcm9wSW5mb1twcm9wTmFtZV07XG4gICAgICAgICAgICBpZih2YWxpZGF0aW9uVG9DaGVjayl7XG4gICAgICAgICAgICAgICAgbGV0IGhhc1ZhbGlkRGVzY3JpcHRpb24gPSB0aGlzLnZhbGlkYXRlRGVzY3JpcHRpb24oY29tcG9uZW50TmFtZSwgcHJvcE5hbWUsIHZhbGlkYXRpb25Ub0NoZWNrKTtcbiAgICAgICAgICAgICAgICBsZXQgaGFzVmFsaWRQcm9wVHlwZXMgPSB2YWxpZGF0aW9uVG9DaGVjay50eXBlKGNvbXBvbmVudE5hbWUsIHByb3BOYW1lLCBwcm9wVmFsLCB2YWxpZGF0aW9uVG9DaGVjay5yZXF1aXJlZCk7XG4gICAgICAgICAgICAgICAgaWYoaGFzVmFsaWREZXNjcmlwdGlvbiAmJiBoYXNWYWxpZFByb3BUeXBlcyl7XG4gICAgICAgICAgICAgICAgICAgIHZhbGlkYXRpb25SZXN1bHRzW3Byb3BOYW1lXSA9IHRydWU7XG4gICAgICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICAgICAgdmFsaWRhdGlvblJlc3VsdHNbcHJvcE5hbWVdID0gZmFsc2U7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICAvLyBJZiB3ZSBnZXQgaGVyZSwgdGhlIGNvbnN1bWVyIGhhcyBwYXNzZWQgaW4gYSBwcm9wXG4gICAgICAgICAgICAgICAgLy8gdGhhdCBpcyBub3QgcHJlc2VudCBpbiB0aGUgcHJvcFR5cGVzIGRlc2NyaXB0aW9uLlxuICAgICAgICAgICAgICAgIC8vIFdlIHJlcG9ydCB0byB0aGUgY29uc29sZSBhcyBuZWVkZWQgYW5kIHZhbGlkYXRlIGFzIGZhbHNlLlxuICAgICAgICAgICAgICAgIGxldCBtZXNzYWdlID0gYCR7Y29tcG9uZW50TmFtZX0gaGFzIGEgcHJvcCBjYWxsZWQgXCIke3Byb3BOYW1lfVwiIHRoYXQgaXMgbm90IGRlc2NyaWJlZCBpbiBwcm9wVHlwZXMhYDtcbiAgICAgICAgICAgICAgICByZXBvcnQobWVzc2FnZSwgdGhpcy5lcnJvck1vZGUsIHRoaXMuc2lsZW50TW9kZSk7XG4gICAgICAgICAgICAgICAgdmFsaWRhdGlvblJlc3VsdHNbcHJvcE5hbWVdID0gZmFsc2U7XG4gICAgICAgICAgICB9XG4gICAgICAgIH0pO1xuXG4gICAgICAgIC8vIElmIHRoZXJlIHdlcmUgYW55IHRoYXQgZGlkIG5vdCB2YWxpZGF0ZSwgcmV0dXJuXG4gICAgICAgIC8vIGZhbHNlIGFuZCByZXBvcnQgYXMgbXVjaC5cbiAgICAgICAgbGV0IGludmFsaWRzID0gW107XG4gICAgICAgIE9iamVjdC5rZXlzKHZhbGlkYXRpb25SZXN1bHRzKS5mb3JFYWNoKGtleSA9PiB7XG4gICAgICAgICAgICBpZih2YWxpZGF0aW9uUmVzdWx0c1trZXldID09IGZhbHNlKXtcbiAgICAgICAgICAgICAgICBpbnZhbGlkcy5wdXNoKGtleSk7XG4gICAgICAgICAgICB9XG4gICAgICAgIH0pO1xuICAgICAgICBpZihpbnZhbGlkcy5sZW5ndGggPiAwKXtcbiAgICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgICB9XG4gICAgfSxcblxuICAgIHZhbGlkYXRlUmVxdWlyZWQ6IGZ1bmN0aW9uKGNvbXBvbmVudE5hbWUsIHByb3BOYW1lLCBwcm9wVmFsLCBpc1JlcXVpcmVkKXtcbiAgICAgICAgaWYoaXNSZXF1aXJlZCA9PSB0cnVlKXtcbiAgICAgICAgICAgIGlmKHByb3BWYWwgPT0gbnVsbCB8fCBwcm9wVmFsID09IHVuZGVmaW5lZCl7XG4gICAgICAgICAgICAgICAgbGV0IG1lc3NhZ2UgPSBgJHtjb21wb25lbnROYW1lfSA+PiAke3Byb3BOYW1lfSByZXF1aXJlcyBhIHZhbHVlLCBidXQgJHtwcm9wVmFsfSB3YXMgcGFzc2VkIWA7XG4gICAgICAgICAgICAgICAgcmVwb3J0KG1lc3NhZ2UsIHRoaXMuZXJyb3JNb2RlLCB0aGlzLnNpbGVudE1vZGUpO1xuICAgICAgICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICB9LFxuXG4gICAgdmFsaWRhdGVEZXNjcmlwdGlvbjogZnVuY3Rpb24oY29tcG9uZW50TmFtZSwgcHJvcE5hbWUsIHByb3Ape1xuICAgICAgICBsZXQgZGVzYyA9IHByb3AuZGVzY3JpcHRpb247XG4gICAgICAgIGlmKGRlc2MgPT0gdW5kZWZpbmVkIHx8IGRlc2MgPT0gXCJcIiB8fCBkZXNjID09IG51bGwpe1xuICAgICAgICAgICAgbGV0IG1lc3NhZ2UgPSBgJHtjb21wb25lbnROYW1lfSA+PiAke3Byb3BOYW1lfSBoYXMgYW4gZW1wdHkgZGVzY3JpcHRpb24hYDtcbiAgICAgICAgICAgIHJlcG9ydChtZXNzYWdlLCB0aGlzLmVycm9yTW9kZSwgdGhpcy5zaWxlbnRNb2RlKTtcbiAgICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICB9XG59O1xuXG5leHBvcnQge1xuICAgIFByb3BUeXBlc1xufTtcblxuXG4vKioqXG5udW1iZXI6IGZ1bmN0aW9uKGNvbXBvbmVudE5hbWUsIHByb3BOYW1lLCBwcm9wVmFsdWUsIGluQ29tcG91bmQgPSBmYWxzZSl7XG4gICAgICAgIGlmKGluQ29tcG91bmQgPT0gZmFsc2Upe1xuICAgICAgICAgICAgaWYodHlwZW9mKHByb3BWYWx1ZSkgPT0gJ251bWJlcicpe1xuICAgICAgICAgICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICBsZXQgbWVzc2FnZSA9IGAke2NvbXBvbmVudE5hbWV9ID4+ICR7cHJvcE5hbWV9IG11c3QgYmUgb2YgdHlwZSBudW1iZXIhYDtcbiAgICAgICAgICAgICAgICByZXBvcnQobWVzc2FnZSwgdGhpcy5lcnJvck1vZGUsIHRoaXMuc2lsZW50TW9kZSk7XG4gICAgICAgICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICAgICAgfVxuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHR5cGVvZihwcm9wVmFsdWUpID09ICdudW1iZXInO1xuICAgICAgICB9XG5cbiAgICB9LmJpbmQodGhpcyksXG5cbiAgICBzdHJpbmc6IGZ1bmN0aW9uKGNvbXBvbmVudE5hbWUsIHByb3BOYW1lLCBwcm9wVmFsdWUsIGluQ29tcG91bmQgPSBmYWxzZSl7XG4gICAgICAgIGlmKGluQ29tcG91bmQgPT0gZmFsc2Upe1xuICAgICAgICAgICAgaWYodHlwZW9mKHByb3BWYWx1ZSkgPT0gJ3N0cmluZycpe1xuICAgICAgICAgICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICBsZXQgbWVzc2FnZSA9IGAke2NvbXBvbmVudE5hbWV9ID4+ICR7cHJvcE5hbWV9IG11c3QgYmUgb2YgdHlwZSBzdHJpbmchYDtcbiAgICAgICAgICAgICAgICByZXBvcnQobWVzc2FnZSwgdGhpcy5lcnJvck1vZGUsIHRoaXMuc2lsZW50TW9kZSk7XG4gICAgICAgICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICAgICAgfVxuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHR5cGVvZihwcm9wVmFsdWUpID09ICdzdHJpbmcnO1xuICAgICAgICB9XG4gICAgfS5iaW5kKHRoaXMpLFxuXG4gICAgYm9vbGVhbjogZnVuY3Rpb24oY29tcG9uZW50TmFtZSwgcHJvcE5hbWUsIHByb3BWYWx1ZSwgaW5Db21wb3VuZCA9IGZhbHNlKXtcbiAgICAgICAgaWYoaW5Db21wb3VuZCA9PSBmYWxzZSl7XG4gICAgICAgICAgICBpZih0eXBlb2YocHJvcFZhbHVlKSA9PSAnYm9vbGVhbicpe1xuICAgICAgICAgICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICBsZXQgbWVzc2FnZSA9IGAke2NvbXBvbmVudE5hbWV9ID4+ICR7cHJvcE5hbWV9IG11c3QgYmUgb2YgdHlwZSBib29sZWFuIWA7XG4gICAgICAgICAgICAgICAgcmVwb3J0KG1lc3NhZ2UsIHRoaXMuZXJyb3JNb2RlLCB0aGlzLnNpbGVudE1vZGUpO1xuICAgICAgICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiB0eXBlb2YocHJvcFZhbHVlKSA9PSAnYm9vbGVhbic7XG4gICAgICAgIH1cbiAgICB9LmJpbmQodGhpcyksXG5cbiAgICBvYmplY3Q6IGZ1bmN0aW9uKGNvbXBvbmVudE5hbWUsIHByb3BOYW1lLCBwcm9wVmFsdWUsIGluQ29tcG91bmQgPSBmYWxzZSl7XG4gICAgICAgIGlmKGluQ29tcG91bmQgPT0gZmFsc2Upe1xuICAgICAgICAgICAgaWYodHlwZW9mKHByb3BWYWx1ZSkgPT0gJ29iamVjdCcpe1xuICAgICAgICAgICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICBsZXQgbWVzc2FnZSA9IGAke2NvbXBvbmVudE5hbWV9ID4+ICR7cHJvcE5hbWV9IG11c3QgYmUgb2YgdHlwZSBvYmplY3QhYDtcbiAgICAgICAgICAgICAgICByZXBvcnQobWVzc2FnZSwgdGhpcy5lcnJvck1vZGUsIHRoaXMuc2lsZW50TW9kZSk7XG4gICAgICAgICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICAgICAgfVxuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHR5cGVvZihwcm9wVmFsdWUpID09ICdvYmplY3QnO1xuICAgICAgICB9XG4gICAgfS5iaW5kKHRoaXMpLFxuXG4gICAgZnVuYzogZnVuY3Rpb24oY29tcG9uZW50TmFtZSwgcHJvcE5hbWUsIHByb3BWYWx1ZSwgaW5Db21wb3VuZCA9IGZhbHNlKXtcbiAgICAgICAgaWYoaW5Db21wb3VuZCA9PSBmYWxzZSl7XG4gICAgICAgICAgICBpZih0eXBlb2YocHJvcFZhbHVlKSA9PSAnZnVuY3Rpb24nKXtcbiAgICAgICAgICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgbGV0IG1lc3NhZ2UgPSBgJHtjb21wb25lbnROYW1lfSA+PiAke3Byb3BOYW1lfSBtdXN0IGJlIG9mIHR5cGUgZnVuY3Rpb24hYDtcbiAgICAgICAgICAgICAgICByZXBvcnQobWVzc2FnZSwgdGhpcy5lcnJvck1vZGUsIHRoaXMuc2lsZW50TW9kZSk7XG4gICAgICAgICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICAgICAgfVxuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgcmV0dXJuIHR5cGVvZihwcm9wVmFsdWUpID09ICdmdW5jdGlvbic7XG4gICAgICAgIH1cbiAgICB9LmJpbmQodGhpcyksXG5cbioqKi9cbiIsImNsYXNzIFJlcGxhY2VtZW50c0hhbmRsZXIge1xuICAgIGNvbnN0cnVjdG9yKHJlcGxhY2VtZW50cyl7XG4gICAgICAgIHRoaXMucmVwbGFjZW1lbnRzID0gcmVwbGFjZW1lbnRzO1xuICAgICAgICB0aGlzLnJlZ3VsYXIgPSB7fTtcbiAgICAgICAgdGhpcy5lbnVtZXJhdGVkID0ge307XG5cbiAgICAgICAgaWYocmVwbGFjZW1lbnRzKXtcbiAgICAgICAgICAgIHRoaXMucHJvY2Vzc1JlcGxhY2VtZW50cygpO1xuICAgICAgICB9XG5cbiAgICAgICAgLy8gQmluZCBjb250ZXh0IHRvIG1ldGhvZHNcbiAgICAgICAgdGhpcy5wcm9jZXNzUmVwbGFjZW1lbnRzID0gdGhpcy5wcm9jZXNzUmVwbGFjZW1lbnRzLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMucHJvY2Vzc1JlZ3VsYXIgPSB0aGlzLnByb2Nlc3NSZWd1bGFyLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuaGFzUmVwbGFjZW1lbnQgPSB0aGlzLmhhc1JlcGxhY2VtZW50LmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuZ2V0UmVwbGFjZW1lbnRGb3IgPSB0aGlzLmdldFJlcGxhY2VtZW50Rm9yLmJpbmQodGhpcyk7XG4gICAgICAgIHRoaXMuZ2V0UmVwbGFjZW1lbnRzRm9yID0gdGhpcy5nZXRSZXBsYWNlbWVudHNGb3IuYmluZCh0aGlzKTtcbiAgICAgICAgdGhpcy5tYXBSZXBsYWNlbWVudHNGb3IgPSB0aGlzLm1hcFJlcGxhY2VtZW50c0Zvci5iaW5kKHRoaXMpO1xuICAgIH1cblxuICAgIHByb2Nlc3NSZXBsYWNlbWVudHMoKXtcbiAgICAgICAgdGhpcy5yZXBsYWNlbWVudHMuZm9yRWFjaChyZXBsYWNlbWVudCA9PiB7XG4gICAgICAgICAgICBsZXQgcmVwbGFjZW1lbnRJbmZvID0gdGhpcy5jb25zdHJ1Y3Rvci5yZWFkUmVwbGFjZW1lbnRTdHJpbmcocmVwbGFjZW1lbnQpO1xuICAgICAgICAgICAgaWYocmVwbGFjZW1lbnRJbmZvLmlzRW51bWVyYXRlZCl7XG4gICAgICAgICAgICAgICAgdGhpcy5wcm9jZXNzRW51bWVyYXRlZChyZXBsYWNlbWVudCwgcmVwbGFjZW1lbnRJbmZvKTtcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgdGhpcy5wcm9jZXNzUmVndWxhcihyZXBsYWNlbWVudCwgcmVwbGFjZW1lbnRJbmZvKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfSk7XG4gICAgICAgIC8vIE5vdyB3ZSB1cGRhdGUgdGhpcy5lbnVtZXJhdGVkIHRvIGhhdmUgaXQncyB0b3AgbGV2ZWxcbiAgICAgICAgLy8gdmFsdWVzIGFzIEFycmF5cyBpbnN0ZWFkIG9mIG5lc3RlZCBkaWN0cyBhbmQgd2Ugc29ydFxuICAgICAgICAvLyBiYXNlZCBvbiB0aGUgZXh0cmFjdGVkIGluZGljZXMgKHdoaWNoIGFyZSBhdCB0aGlzIHBvaW50XG4gICAgICAgIC8vIGp1c3Qga2V5cyBvbiBzdWJkaWN0cyBvciBtdWx0aWRpbWVuc2lvbmFsIGRpY3RzKVxuICAgICAgICBPYmplY3Qua2V5cyh0aGlzLmVudW1lcmF0ZWQpLmZvckVhY2goa2V5ID0+IHtcbiAgICAgICAgICAgIGxldCBlbnVtZXJhdGVkUmVwbGFjZW1lbnRzID0gdGhpcy5lbnVtZXJhdGVkW2tleV07XG4gICAgICAgICAgICB0aGlzLmVudW1lcmF0ZWRba2V5XSA9IHRoaXMuY29uc3RydWN0b3IuZW51bWVyYXRlZFZhbFRvU29ydGVkQXJyYXkoZW51bWVyYXRlZFJlcGxhY2VtZW50cyk7XG4gICAgICAgIH0pO1xuICAgIH1cblxuICAgIHByb2Nlc3NSZWd1bGFyKHJlcGxhY2VtZW50TmFtZSwgcmVwbGFjZW1lbnRJbmZvKXtcbiAgICAgICAgbGV0IHJlcGxhY2VtZW50S2V5ID0gdGhpcy5jb25zdHJ1Y3Rvci5rZXlGcm9tTmFtZVBhcnRzKHJlcGxhY2VtZW50SW5mby5uYW1lUGFydHMpO1xuICAgICAgICB0aGlzLnJlZ3VsYXJbcmVwbGFjZW1lbnRLZXldID0gcmVwbGFjZW1lbnROYW1lO1xuICAgIH1cblxuICAgIHByb2Nlc3NFbnVtZXJhdGVkKHJlcGxhY2VtZW50TmFtZSwgcmVwbGFjZW1lbnRJbmZvKXtcbiAgICAgICAgbGV0IHJlcGxhY2VtZW50S2V5ID0gdGhpcy5jb25zdHJ1Y3Rvci5rZXlGcm9tTmFtZVBhcnRzKHJlcGxhY2VtZW50SW5mby5uYW1lUGFydHMpO1xuICAgICAgICBsZXQgY3VycmVudEVudHJ5ID0gdGhpcy5lbnVtZXJhdGVkW3JlcGxhY2VtZW50S2V5XTtcblxuICAgICAgICAvLyBJZiBpdCdzIHVuZGVmaW5lZCwgdGhpcyBpcyB0aGUgZmlyc3RcbiAgICAgICAgLy8gb2YgdGhlIGVudW1lcmF0ZWQgcmVwbGFjZW1lbnRzIGZvciB0aGlzXG4gICAgICAgIC8vIGtleSwgaWUgc29tZXRoaW5nIGxpa2UgX19fX2NoaWxkXzBfX1xuICAgICAgICBpZihjdXJyZW50RW50cnkgPT0gdW5kZWZpbmVkKXtcbiAgICAgICAgICAgIHRoaXMuZW51bWVyYXRlZFtyZXBsYWNlbWVudEtleV0gPSB7fTtcbiAgICAgICAgICAgIGN1cnJlbnRFbnRyeSA9IHRoaXMuZW51bWVyYXRlZFtyZXBsYWNlbWVudEtleV07XG4gICAgICAgIH1cblxuICAgICAgICAvLyBXZSBhZGQgdGhlIGVudW1lcmF0ZWQgaW5kaWNlcyBhcyBrZXlzIHRvIGEgZGljdFxuICAgICAgICAvLyBhbmQgd2UgZG8gdGhpcyByZWN1cnNpdmVseSBhY3Jvc3MgZGltZW5zaW9ucyBhc1xuICAgICAgICAvLyBuZWVkZWQuXG4gICAgICAgIHRoaXMuY29uc3RydWN0b3IucHJvY2Vzc0RpbWVuc2lvbihyZXBsYWNlbWVudEluZm8uZW51bWVyYXRlZFZhbHVlcywgY3VycmVudEVudHJ5LCByZXBsYWNlbWVudE5hbWUpO1xuICAgIH1cblxuICAgIC8vIEFjY2Vzc2luZyBhbmQgb3RoZXIgQ29udmVuaWVuY2UgTWV0aG9kc1xuICAgIGhhc1JlcGxhY2VtZW50KGFSZXBsYWNlbWVudE5hbWUpe1xuICAgICAgICBpZih0aGlzLnJlZ3VsYXIuaGFzT3duUHJvcGVydHkoYVJlcGxhY2VtZW50TmFtZSkpe1xuICAgICAgICAgICAgcmV0dXJuIHRydWU7XG4gICAgICAgIH0gZWxzZSBpZih0aGlzLmVudW1lcmF0ZWQuaGFzT3duUHJvcGVydHkoYVJlcGxhY2VtZW50TmFtZSkpe1xuICAgICAgICAgICAgcmV0dXJuIHRydWU7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgIH1cblxuICAgIGdldFJlcGxhY2VtZW50Rm9yKGFSZXBsYWNlbWVudE5hbWUpe1xuICAgICAgICBsZXQgZm91bmQgPSB0aGlzLnJlZ3VsYXJbYVJlcGxhY2VtZW50TmFtZV07XG4gICAgICAgIGlmKGZvdW5kID09IHVuZGVmaW5lZCl7XG4gICAgICAgICAgICByZXR1cm4gbnVsbDtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gZm91bmQ7XG4gICAgfVxuXG4gICAgZ2V0UmVwbGFjZW1lbnRzRm9yKGFSZXBsYWNlbWVudE5hbWUpe1xuICAgICAgICBsZXQgZm91bmQgPSB0aGlzLmVudW1lcmF0ZWRbYVJlcGxhY2VtZW50TmFtZV07XG4gICAgICAgIGlmKGZvdW5kID09IHVuZGVmaW5lZCl7XG4gICAgICAgICAgICByZXR1cm4gbnVsbDtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gZm91bmQ7XG4gICAgfVxuXG4gICAgbWFwUmVwbGFjZW1lbnRzRm9yKGFSZXBsYWNlbWVudE5hbWUsIG1hcEZ1bmN0aW9uKXtcbiAgICAgICAgaWYoIXRoaXMuaGFzUmVwbGFjZW1lbnQoYVJlcGxhY2VtZW50TmFtZSkpe1xuICAgICAgICAgICAgdGhyb3cgbmV3IEVycm9yKGBJbnZhbGlkIHJlcGxhY2VtZW50IG5hbWU6ICR7YVJlcGxhY2VtZW50bmFtZX1gKTtcbiAgICAgICAgfVxuICAgICAgICBsZXQgcm9vdCA9IHRoaXMuZ2V0UmVwbGFjZW1lbnRzRm9yKGFSZXBsYWNlbWVudE5hbWUpO1xuICAgICAgICByZXR1cm4gdGhpcy5fcmVjdXJzaXZlbHlNYXAocm9vdCwgbWFwRnVuY3Rpb24pO1xuICAgIH1cblxuICAgIF9yZWN1cnNpdmVseU1hcChjdXJyZW50SXRlbSwgbWFwRnVuY3Rpb24pe1xuICAgICAgICBpZighQXJyYXkuaXNBcnJheShjdXJyZW50SXRlbSkpe1xuICAgICAgICAgICAgcmV0dXJuIG1hcEZ1bmN0aW9uKGN1cnJlbnRJdGVtKTtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gY3VycmVudEl0ZW0ubWFwKHN1Ykl0ZW0gPT4ge1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMuX3JlY3Vyc2l2ZWx5TWFwKHN1Ykl0ZW0sIG1hcEZ1bmN0aW9uKTtcbiAgICAgICAgfSk7XG4gICAgfVxuXG4gICAgLy8gU3RhdGljIGhlbHBlcnNcbiAgICBzdGF0aWMgcHJvY2Vzc0RpbWVuc2lvbihyZW1haW5pbmdWYWxzLCBpbkRpY3QsIHJlcGxhY2VtZW50TmFtZSl7XG4gICAgICAgIGlmKHJlbWFpbmluZ1ZhbHMubGVuZ3RoID09IDEpe1xuICAgICAgICAgICAgaW5EaWN0W3JlbWFpbmluZ1ZhbHNbMF1dID0gcmVwbGFjZW1lbnROYW1lO1xuICAgICAgICAgICAgcmV0dXJuO1xuICAgICAgICB9XG4gICAgICAgIGxldCBuZXh0S2V5ID0gcmVtYWluaW5nVmFsc1swXTtcbiAgICAgICAgbGV0IG5leHREaWN0ID0gaW5EaWN0W25leHRLZXldO1xuICAgICAgICBpZihuZXh0RGljdCA9PSB1bmRlZmluZWQpe1xuICAgICAgICAgICAgaW5EaWN0W25leHRLZXldID0ge307XG4gICAgICAgICAgICBuZXh0RGljdCA9IGluRGljdFtuZXh0S2V5XTtcbiAgICAgICAgfVxuICAgICAgICB0aGlzLnByb2Nlc3NEaW1lbnNpb24ocmVtYWluaW5nVmFscy5zbGljZSgxKSwgbmV4dERpY3QsIHJlcGxhY2VtZW50TmFtZSk7XG4gICAgfVxuXG4gICAgc3RhdGljIGVudW1lcmF0ZWRWYWxUb1NvcnRlZEFycmF5KGFEaWN0LCBhY2N1bXVsYXRlID0gW10pe1xuICAgICAgICBpZih0eXBlb2YgYURpY3QgIT09ICdvYmplY3QnKXtcbiAgICAgICAgICAgIHJldHVybiBhRGljdDtcbiAgICAgICAgfVxuICAgICAgICBsZXQgc29ydGVkS2V5cyA9IE9iamVjdC5rZXlzKGFEaWN0KS5zb3J0KChmaXJzdCwgc2Vjb25kKSA9PiB7XG4gICAgICAgICAgICByZXR1cm4gKHBhcnNlSW50KGZpcnN0KSAtIHBhcnNlSW50KHNlY29uZCkpO1xuICAgICAgICB9KTtcbiAgICAgICAgbGV0IHN1YkVudHJpZXMgPSBzb3J0ZWRLZXlzLm1hcChrZXkgPT4ge1xuICAgICAgICAgICAgbGV0IGVudHJ5ID0gYURpY3Rba2V5XTtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLmVudW1lcmF0ZWRWYWxUb1NvcnRlZEFycmF5KGVudHJ5KTtcbiAgICAgICAgfSk7XG4gICAgICAgIHJldHVybiBzdWJFbnRyaWVzO1xuICAgIH1cblxuICAgIHN0YXRpYyBrZXlGcm9tTmFtZVBhcnRzKG5hbWVQYXJ0cyl7XG4gICAgICAgIHJldHVybiBuYW1lUGFydHMuam9pbihcIi1cIik7XG4gICAgfVxuXG4gICAgc3RhdGljIHJlYWRSZXBsYWNlbWVudFN0cmluZyhyZXBsYWNlbWVudCl7XG4gICAgICAgIGxldCBuYW1lUGFydHMgPSBbXTtcbiAgICAgICAgbGV0IGlzRW51bWVyYXRlZCA9IGZhbHNlO1xuICAgICAgICBsZXQgZW51bWVyYXRlZFZhbHVlcyA9IFtdO1xuICAgICAgICBsZXQgcGllY2VzID0gcmVwbGFjZW1lbnQuc3BsaXQoJ18nKS5maWx0ZXIoaXRlbSA9PiB7XG4gICAgICAgICAgICByZXR1cm4gaXRlbSAhPSAnJztcbiAgICAgICAgfSk7XG4gICAgICAgIHBpZWNlcy5mb3JFYWNoKHBpZWNlID0+IHtcbiAgICAgICAgICAgIGxldCBudW0gPSBwYXJzZUludChwaWVjZSk7XG4gICAgICAgICAgICBpZihpc05hTihudW0pKXtcbiAgICAgICAgICAgICAgICBuYW1lUGFydHMucHVzaChwaWVjZSk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICBpc0VudW1lcmF0ZWQgPSB0cnVlO1xuICAgICAgICAgICAgZW51bWVyYXRlZFZhbHVlcy5wdXNoKG51bSk7XG4gICAgICAgIH1cbiAgICAgICAgfSk7XG4gICAgICAgIHJldHVybiB7XG4gICAgICAgICAgICBuYW1lUGFydHMsXG4gICAgICAgICAgICBpc0VudW1lcmF0ZWQsXG4gICAgICAgICAgICBlbnVtZXJhdGVkVmFsdWVzXG4gICAgICAgIH07XG4gICAgfVxufVxuXG5leHBvcnQge1xuICAgIFJlcGxhY2VtZW50c0hhbmRsZXIsXG4gICAgUmVwbGFjZW1lbnRzSGFuZGxlciBhcyBkZWZhdWx0XG59O1xuIiwiaW1wb3J0ICdtYXF1ZXR0ZSc7XG5jb25zdCBoID0gbWFxdWV0dGUuaDtcbi8vaW1wb3J0IHtsYW5nVG9vbHN9IGZyb20gJ2FjZS9leHQvbGFuZ3VhZ2VfdG9vbHMnO1xuaW1wb3J0IHtDZWxsSGFuZGxlcn0gZnJvbSAnLi9DZWxsSGFuZGxlcic7XG5pbXBvcnQge0NlbGxTb2NrZXR9IGZyb20gJy4vQ2VsbFNvY2tldCc7XG5pbXBvcnQge0NvbXBvbmVudFJlZ2lzdHJ5fSBmcm9tICcuL0NvbXBvbmVudFJlZ2lzdHJ5JztcblxuLyoqXG4gKiBHbG9iYWxzXG4gKiovXG53aW5kb3cubGFuZ1Rvb2xzID0gYWNlLnJlcXVpcmUoXCJhY2UvZXh0L2xhbmd1YWdlX3Rvb2xzXCIpO1xud2luZG93LmFjZUVkaXRvcnMgPSB7fTtcbndpbmRvdy5oYW5kc09uVGFibGVzID0ge307XG5cbi8qKlxuICogSW5pdGlhbCBSZW5kZXJcbiAqKi9cbmNvbnN0IGluaXRpYWxSZW5kZXIgPSBmdW5jdGlvbigpe1xuICAgIHJldHVybiBoKFwiZGl2XCIsIHt9LCBbXG4gICAgICAgICBoKFwiZGl2XCIsIHtpZDogXCJwYWdlX3Jvb3RcIn0sIFtcbiAgICAgICAgICAgICBoKFwiZGl2LmNvbnRhaW5lci1mbHVpZFwiLCB7fSwgW1xuICAgICAgICAgICAgICAgICBoKFwiZGl2LmNhcmRcIiwge2NsYXNzOiBcIm10LTVcIn0sIFtcbiAgICAgICAgICAgICAgICAgICAgIGgoXCJkaXYuY2FyZC1ib2R5XCIsIHt9LCBbXCJMb2FkaW5nLi4uXCJdKVxuICAgICAgICAgICAgICAgICBdKVxuICAgICAgICAgICAgIF0pXG4gICAgICAgICBdKSxcbiAgICAgICAgIGgoXCJkaXZcIiwge2lkOiBcImhvbGRpbmdfcGVuXCIsIHN0eWxlOiBcImRpc3BsYXk6bm9uZVwifSwgW10pXG4gICAgIF0pO1xufTtcblxuLyoqXG4gKiBDZWxsIFNvY2tldCBhbmQgSGFuZGxlclxuICoqL1xubGV0IHByb2plY3RvciA9IG1hcXVldHRlLmNyZWF0ZVByb2plY3RvcigpO1xuY29uc3QgY2VsbFNvY2tldCA9IG5ldyBDZWxsU29ja2V0KCk7XG5jb25zdCBjZWxsSGFuZGxlciA9IG5ldyBDZWxsSGFuZGxlcihoLCBwcm9qZWN0b3IsIENvbXBvbmVudFJlZ2lzdHJ5KTtcbmNlbGxTb2NrZXQub25Qb3N0c2NyaXB0cyhjZWxsSGFuZGxlci5oYW5kbGVQb3N0c2NyaXB0KTtcbmNlbGxTb2NrZXQub25NZXNzYWdlKGNlbGxIYW5kbGVyLmhhbmRsZU1lc3NhZ2UpO1xuY2VsbFNvY2tldC5vbkNsb3NlKGNlbGxIYW5kbGVyLnNob3dDb25uZWN0aW9uQ2xvc2VkKTtcbmNlbGxTb2NrZXQub25FcnJvcihlcnIgPT4ge1xuICAgIGNvbnNvbGUuZXJyb3IoXCJTT0NLRVQgRVJST1I6IFwiLCBlcnIpO1xufSk7XG5cbi8qKiBGb3Igbm93LCB3ZSBiaW5kIHRoZSBjdXJyZW50IHNvY2tldCBhbmQgaGFuZGxlciB0byB0aGUgZ2xvYmFsIHdpbmRvdyAqKi9cbndpbmRvdy5jZWxsU29ja2V0ID0gY2VsbFNvY2tldDtcbndpbmRvdy5jZWxsSGFuZGxlciA9IGNlbGxIYW5kbGVyO1xuXG4vKiogUmVuZGVyIHRvcCBsZXZlbCBjb21wb25lbnQgb25jZSBET00gaXMgcmVhZHkgKiovXG5kb2N1bWVudC5hZGRFdmVudExpc3RlbmVyKCdET01Db250ZW50TG9hZGVkJywgKCkgPT4ge1xuICAgIHByb2plY3Rvci5hcHBlbmQoZG9jdW1lbnQuYm9keSwgaW5pdGlhbFJlbmRlcik7XG4gICAgY2VsbFNvY2tldC5jb25uZWN0KCk7XG59KTtcblxuLy8gVEVTVElORzsgUkVNT1ZFXG5jb25zb2xlLmxvZygnTWFpbiBtb2R1bGUgbG9hZGVkJyk7XG4iLCIoZnVuY3Rpb24gKGdsb2JhbCwgZmFjdG9yeSkge1xuICAgIHR5cGVvZiBleHBvcnRzID09PSAnb2JqZWN0JyAmJiB0eXBlb2YgbW9kdWxlICE9PSAndW5kZWZpbmVkJyA/IGZhY3RvcnkoZXhwb3J0cykgOlxuICAgIHR5cGVvZiBkZWZpbmUgPT09ICdmdW5jdGlvbicgJiYgZGVmaW5lLmFtZCA/IGRlZmluZShbJ2V4cG9ydHMnXSwgZmFjdG9yeSkgOlxuICAgIChnbG9iYWwgPSBnbG9iYWwgfHwgc2VsZiwgZmFjdG9yeShnbG9iYWwubWFxdWV0dGUgPSB7fSkpO1xufSh0aGlzLCBmdW5jdGlvbiAoZXhwb3J0cykgeyAndXNlIHN0cmljdCc7XG5cbiAgICAvKiB0c2xpbnQ6ZGlzYWJsZSBuby1odHRwLXN0cmluZyAqL1xyXG4gICAgdmFyIE5BTUVTUEFDRV9XMyA9ICdodHRwOi8vd3d3LnczLm9yZy8nO1xyXG4gICAgLyogdHNsaW50OmVuYWJsZSBuby1odHRwLXN0cmluZyAqL1xyXG4gICAgdmFyIE5BTUVTUEFDRV9TVkcgPSBOQU1FU1BBQ0VfVzMgKyBcIjIwMDAvc3ZnXCI7XHJcbiAgICB2YXIgTkFNRVNQQUNFX1hMSU5LID0gTkFNRVNQQUNFX1czICsgXCIxOTk5L3hsaW5rXCI7XHJcbiAgICB2YXIgZW1wdHlBcnJheSA9IFtdO1xyXG4gICAgdmFyIGV4dGVuZCA9IGZ1bmN0aW9uIChiYXNlLCBvdmVycmlkZXMpIHtcclxuICAgICAgICB2YXIgcmVzdWx0ID0ge307XHJcbiAgICAgICAgT2JqZWN0LmtleXMoYmFzZSkuZm9yRWFjaChmdW5jdGlvbiAoa2V5KSB7XHJcbiAgICAgICAgICAgIHJlc3VsdFtrZXldID0gYmFzZVtrZXldO1xyXG4gICAgICAgIH0pO1xyXG4gICAgICAgIGlmIChvdmVycmlkZXMpIHtcclxuICAgICAgICAgICAgT2JqZWN0LmtleXMob3ZlcnJpZGVzKS5mb3JFYWNoKGZ1bmN0aW9uIChrZXkpIHtcclxuICAgICAgICAgICAgICAgIHJlc3VsdFtrZXldID0gb3ZlcnJpZGVzW2tleV07XHJcbiAgICAgICAgICAgIH0pO1xyXG4gICAgICAgIH1cclxuICAgICAgICByZXR1cm4gcmVzdWx0O1xyXG4gICAgfTtcclxuICAgIHZhciBzYW1lID0gZnVuY3Rpb24gKHZub2RlMSwgdm5vZGUyKSB7XHJcbiAgICAgICAgaWYgKHZub2RlMS52bm9kZVNlbGVjdG9yICE9PSB2bm9kZTIudm5vZGVTZWxlY3Rvcikge1xyXG4gICAgICAgICAgICByZXR1cm4gZmFsc2U7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGlmICh2bm9kZTEucHJvcGVydGllcyAmJiB2bm9kZTIucHJvcGVydGllcykge1xyXG4gICAgICAgICAgICBpZiAodm5vZGUxLnByb3BlcnRpZXMua2V5ICE9PSB2bm9kZTIucHJvcGVydGllcy5rZXkpIHtcclxuICAgICAgICAgICAgICAgIHJldHVybiBmYWxzZTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICByZXR1cm4gdm5vZGUxLnByb3BlcnRpZXMuYmluZCA9PT0gdm5vZGUyLnByb3BlcnRpZXMuYmluZDtcclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuICF2bm9kZTEucHJvcGVydGllcyAmJiAhdm5vZGUyLnByb3BlcnRpZXM7XHJcbiAgICB9O1xyXG4gICAgdmFyIGNoZWNrU3R5bGVWYWx1ZSA9IGZ1bmN0aW9uIChzdHlsZVZhbHVlKSB7XHJcbiAgICAgICAgaWYgKHR5cGVvZiBzdHlsZVZhbHVlICE9PSAnc3RyaW5nJykge1xyXG4gICAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ1N0eWxlIHZhbHVlcyBtdXN0IGJlIHN0cmluZ3MnKTtcclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgdmFyIGZpbmRJbmRleE9mQ2hpbGQgPSBmdW5jdGlvbiAoY2hpbGRyZW4sIHNhbWVBcywgc3RhcnQpIHtcclxuICAgICAgICBpZiAoc2FtZUFzLnZub2RlU2VsZWN0b3IgIT09ICcnKSB7XHJcbiAgICAgICAgICAgIC8vIE5ldmVyIHNjYW4gZm9yIHRleHQtbm9kZXNcclxuICAgICAgICAgICAgZm9yICh2YXIgaSA9IHN0YXJ0OyBpIDwgY2hpbGRyZW4ubGVuZ3RoOyBpKyspIHtcclxuICAgICAgICAgICAgICAgIGlmIChzYW1lKGNoaWxkcmVuW2ldLCBzYW1lQXMpKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIGk7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuIC0xO1xyXG4gICAgfTtcclxuICAgIHZhciBjaGVja0Rpc3Rpbmd1aXNoYWJsZSA9IGZ1bmN0aW9uIChjaGlsZE5vZGVzLCBpbmRleFRvQ2hlY2ssIHBhcmVudFZOb2RlLCBvcGVyYXRpb24pIHtcclxuICAgICAgICB2YXIgY2hpbGROb2RlID0gY2hpbGROb2Rlc1tpbmRleFRvQ2hlY2tdO1xyXG4gICAgICAgIGlmIChjaGlsZE5vZGUudm5vZGVTZWxlY3RvciA9PT0gJycpIHtcclxuICAgICAgICAgICAgcmV0dXJuOyAvLyBUZXh0IG5vZGVzIG5lZWQgbm90IGJlIGRpc3Rpbmd1aXNoYWJsZVxyXG4gICAgICAgIH1cclxuICAgICAgICB2YXIgcHJvcGVydGllcyA9IGNoaWxkTm9kZS5wcm9wZXJ0aWVzO1xyXG4gICAgICAgIHZhciBrZXkgPSBwcm9wZXJ0aWVzID8gKHByb3BlcnRpZXMua2V5ID09PSB1bmRlZmluZWQgPyBwcm9wZXJ0aWVzLmJpbmQgOiBwcm9wZXJ0aWVzLmtleSkgOiB1bmRlZmluZWQ7XHJcbiAgICAgICAgaWYgKCFrZXkpIHsgLy8gQSBrZXkgaXMganVzdCBhc3N1bWVkIHRvIGJlIHVuaXF1ZVxyXG4gICAgICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IGNoaWxkTm9kZXMubGVuZ3RoOyBpKyspIHtcclxuICAgICAgICAgICAgICAgIGlmIChpICE9PSBpbmRleFRvQ2hlY2spIHtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgbm9kZSA9IGNoaWxkTm9kZXNbaV07XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKHNhbWUobm9kZSwgY2hpbGROb2RlKSkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IocGFyZW50Vk5vZGUudm5vZGVTZWxlY3RvciArIFwiIGhhZCBhIFwiICsgY2hpbGROb2RlLnZub2RlU2VsZWN0b3IgKyBcIiBjaGlsZCBcIiArIChvcGVyYXRpb24gPT09ICdhZGRlZCcgPyBvcGVyYXRpb24gOiAncmVtb3ZlZCcpICsgXCIsIGJ1dCB0aGVyZSBpcyBub3cgbW9yZSB0aGFuIG9uZS4gWW91IG11c3QgYWRkIHVuaXF1ZSBrZXkgcHJvcGVydGllcyB0byBtYWtlIHRoZW0gZGlzdGluZ3Vpc2hhYmxlLlwiKTtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgdmFyIG5vZGVBZGRlZCA9IGZ1bmN0aW9uICh2Tm9kZSkge1xyXG4gICAgICAgIGlmICh2Tm9kZS5wcm9wZXJ0aWVzKSB7XHJcbiAgICAgICAgICAgIHZhciBlbnRlckFuaW1hdGlvbiA9IHZOb2RlLnByb3BlcnRpZXMuZW50ZXJBbmltYXRpb247XHJcbiAgICAgICAgICAgIGlmIChlbnRlckFuaW1hdGlvbikge1xyXG4gICAgICAgICAgICAgICAgZW50ZXJBbmltYXRpb24odk5vZGUuZG9tTm9kZSwgdk5vZGUucHJvcGVydGllcyk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgdmFyIHJlbW92ZWROb2RlcyA9IFtdO1xyXG4gICAgdmFyIHJlcXVlc3RlZElkbGVDYWxsYmFjayA9IGZhbHNlO1xyXG4gICAgdmFyIHZpc2l0UmVtb3ZlZE5vZGUgPSBmdW5jdGlvbiAobm9kZSkge1xyXG4gICAgICAgIChub2RlLmNoaWxkcmVuIHx8IFtdKS5mb3JFYWNoKHZpc2l0UmVtb3ZlZE5vZGUpO1xyXG4gICAgICAgIGlmIChub2RlLnByb3BlcnRpZXMgJiYgbm9kZS5wcm9wZXJ0aWVzLmFmdGVyUmVtb3ZlZCkge1xyXG4gICAgICAgICAgICBub2RlLnByb3BlcnRpZXMuYWZ0ZXJSZW1vdmVkLmFwcGx5KG5vZGUucHJvcGVydGllcy5iaW5kIHx8IG5vZGUucHJvcGVydGllcywgW25vZGUuZG9tTm9kZV0pO1xyXG4gICAgICAgIH1cclxuICAgIH07XHJcbiAgICB2YXIgcHJvY2Vzc1BlbmRpbmdOb2RlUmVtb3ZhbHMgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgcmVxdWVzdGVkSWRsZUNhbGxiYWNrID0gZmFsc2U7XHJcbiAgICAgICAgcmVtb3ZlZE5vZGVzLmZvckVhY2godmlzaXRSZW1vdmVkTm9kZSk7XHJcbiAgICAgICAgcmVtb3ZlZE5vZGVzLmxlbmd0aCA9IDA7XHJcbiAgICB9O1xyXG4gICAgdmFyIHNjaGVkdWxlTm9kZVJlbW92YWwgPSBmdW5jdGlvbiAodk5vZGUpIHtcclxuICAgICAgICByZW1vdmVkTm9kZXMucHVzaCh2Tm9kZSk7XHJcbiAgICAgICAgaWYgKCFyZXF1ZXN0ZWRJZGxlQ2FsbGJhY2spIHtcclxuICAgICAgICAgICAgcmVxdWVzdGVkSWRsZUNhbGxiYWNrID0gdHJ1ZTtcclxuICAgICAgICAgICAgaWYgKHR5cGVvZiB3aW5kb3cgIT09ICd1bmRlZmluZWQnICYmICdyZXF1ZXN0SWRsZUNhbGxiYWNrJyBpbiB3aW5kb3cpIHtcclxuICAgICAgICAgICAgICAgIHdpbmRvdy5yZXF1ZXN0SWRsZUNhbGxiYWNrKHByb2Nlc3NQZW5kaW5nTm9kZVJlbW92YWxzLCB7IHRpbWVvdXQ6IDE2IH0pO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgc2V0VGltZW91dChwcm9jZXNzUGVuZGluZ05vZGVSZW1vdmFscywgMTYpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIHZhciBub2RlVG9SZW1vdmUgPSBmdW5jdGlvbiAodk5vZGUpIHtcclxuICAgICAgICB2YXIgZG9tTm9kZSA9IHZOb2RlLmRvbU5vZGU7XHJcbiAgICAgICAgaWYgKHZOb2RlLnByb3BlcnRpZXMpIHtcclxuICAgICAgICAgICAgdmFyIGV4aXRBbmltYXRpb24gPSB2Tm9kZS5wcm9wZXJ0aWVzLmV4aXRBbmltYXRpb247XHJcbiAgICAgICAgICAgIGlmIChleGl0QW5pbWF0aW9uKSB7XHJcbiAgICAgICAgICAgICAgICBkb21Ob2RlLnN0eWxlLnBvaW50ZXJFdmVudHMgPSAnbm9uZSc7XHJcbiAgICAgICAgICAgICAgICB2YXIgcmVtb3ZlRG9tTm9kZSA9IGZ1bmN0aW9uICgpIHtcclxuICAgICAgICAgICAgICAgICAgICBpZiAoZG9tTm9kZS5wYXJlbnROb2RlKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGRvbU5vZGUucGFyZW50Tm9kZS5yZW1vdmVDaGlsZChkb21Ob2RlKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgc2NoZWR1bGVOb2RlUmVtb3ZhbCh2Tm9kZSk7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgfTtcclxuICAgICAgICAgICAgICAgIGV4aXRBbmltYXRpb24oZG9tTm9kZSwgcmVtb3ZlRG9tTm9kZSwgdk5vZGUucHJvcGVydGllcyk7XHJcbiAgICAgICAgICAgICAgICByZXR1cm47XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgaWYgKGRvbU5vZGUucGFyZW50Tm9kZSkge1xyXG4gICAgICAgICAgICBkb21Ob2RlLnBhcmVudE5vZGUucmVtb3ZlQ2hpbGQoZG9tTm9kZSk7XHJcbiAgICAgICAgICAgIHNjaGVkdWxlTm9kZVJlbW92YWwodk5vZGUpO1xyXG4gICAgICAgIH1cclxuICAgIH07XHJcbiAgICB2YXIgc2V0UHJvcGVydGllcyA9IGZ1bmN0aW9uIChkb21Ob2RlLCBwcm9wZXJ0aWVzLCBwcm9qZWN0aW9uT3B0aW9ucykge1xyXG4gICAgICAgIGlmICghcHJvcGVydGllcykge1xyXG4gICAgICAgICAgICByZXR1cm47XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHZhciBldmVudEhhbmRsZXJJbnRlcmNlcHRvciA9IHByb2plY3Rpb25PcHRpb25zLmV2ZW50SGFuZGxlckludGVyY2VwdG9yO1xyXG4gICAgICAgIHZhciBwcm9wTmFtZXMgPSBPYmplY3Qua2V5cyhwcm9wZXJ0aWVzKTtcclxuICAgICAgICB2YXIgcHJvcENvdW50ID0gcHJvcE5hbWVzLmxlbmd0aDtcclxuICAgICAgICB2YXIgX2xvb3BfMSA9IGZ1bmN0aW9uIChpKSB7XHJcbiAgICAgICAgICAgIHZhciBwcm9wTmFtZSA9IHByb3BOYW1lc1tpXTtcclxuICAgICAgICAgICAgdmFyIHByb3BWYWx1ZSA9IHByb3BlcnRpZXNbcHJvcE5hbWVdO1xyXG4gICAgICAgICAgICBpZiAocHJvcE5hbWUgPT09ICdjbGFzc05hbWUnKSB7XHJcbiAgICAgICAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ1Byb3BlcnR5IFwiY2xhc3NOYW1lXCIgaXMgbm90IHN1cHBvcnRlZCwgdXNlIFwiY2xhc3NcIi4nKTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBlbHNlIGlmIChwcm9wTmFtZSA9PT0gJ2NsYXNzJykge1xyXG4gICAgICAgICAgICAgICAgdG9nZ2xlQ2xhc3Nlcyhkb21Ob2RlLCBwcm9wVmFsdWUsIHRydWUpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGVsc2UgaWYgKHByb3BOYW1lID09PSAnY2xhc3NlcycpIHtcclxuICAgICAgICAgICAgICAgIC8vIG9iamVjdCB3aXRoIHN0cmluZyBrZXlzIGFuZCBib29sZWFuIHZhbHVlc1xyXG4gICAgICAgICAgICAgICAgdmFyIGNsYXNzTmFtZXMgPSBPYmplY3Qua2V5cyhwcm9wVmFsdWUpO1xyXG4gICAgICAgICAgICAgICAgdmFyIGNsYXNzTmFtZUNvdW50ID0gY2xhc3NOYW1lcy5sZW5ndGg7XHJcbiAgICAgICAgICAgICAgICBmb3IgKHZhciBqID0gMDsgaiA8IGNsYXNzTmFtZUNvdW50OyBqKyspIHtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgY2xhc3NOYW1lID0gY2xhc3NOYW1lc1tqXTtcclxuICAgICAgICAgICAgICAgICAgICBpZiAocHJvcFZhbHVlW2NsYXNzTmFtZV0pIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgZG9tTm9kZS5jbGFzc0xpc3QuYWRkKGNsYXNzTmFtZSk7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGVsc2UgaWYgKHByb3BOYW1lID09PSAnc3R5bGVzJykge1xyXG4gICAgICAgICAgICAgICAgLy8gb2JqZWN0IHdpdGggc3RyaW5nIGtleXMgYW5kIHN0cmluZyAoISkgdmFsdWVzXHJcbiAgICAgICAgICAgICAgICB2YXIgc3R5bGVOYW1lcyA9IE9iamVjdC5rZXlzKHByb3BWYWx1ZSk7XHJcbiAgICAgICAgICAgICAgICB2YXIgc3R5bGVDb3VudCA9IHN0eWxlTmFtZXMubGVuZ3RoO1xyXG4gICAgICAgICAgICAgICAgZm9yICh2YXIgaiA9IDA7IGogPCBzdHlsZUNvdW50OyBqKyspIHtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgc3R5bGVOYW1lID0gc3R5bGVOYW1lc1tqXTtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgc3R5bGVWYWx1ZSA9IHByb3BWYWx1ZVtzdHlsZU5hbWVdO1xyXG4gICAgICAgICAgICAgICAgICAgIGlmIChzdHlsZVZhbHVlKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGNoZWNrU3R5bGVWYWx1ZShzdHlsZVZhbHVlKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgcHJvamVjdGlvbk9wdGlvbnMuc3R5bGVBcHBseWVyKGRvbU5vZGUsIHN0eWxlTmFtZSwgc3R5bGVWYWx1ZSk7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGVsc2UgaWYgKHByb3BOYW1lICE9PSAna2V5JyAmJiBwcm9wVmFsdWUgIT09IG51bGwgJiYgcHJvcFZhbHVlICE9PSB1bmRlZmluZWQpIHtcclxuICAgICAgICAgICAgICAgIHZhciB0eXBlID0gdHlwZW9mIHByb3BWYWx1ZTtcclxuICAgICAgICAgICAgICAgIGlmICh0eXBlID09PSAnZnVuY3Rpb24nKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKHByb3BOYW1lLmxhc3RJbmRleE9mKCdvbicsIDApID09PSAwKSB7IC8vIGxhc3RJbmRleE9mKCwwKT09PTAgLT4gc3RhcnRzV2l0aFxyXG4gICAgICAgICAgICAgICAgICAgICAgICBpZiAoZXZlbnRIYW5kbGVySW50ZXJjZXB0b3IpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHByb3BWYWx1ZSA9IGV2ZW50SGFuZGxlckludGVyY2VwdG9yKHByb3BOYW1lLCBwcm9wVmFsdWUsIGRvbU5vZGUsIHByb3BlcnRpZXMpOyAvLyBpbnRlcmNlcHQgZXZlbnRoYW5kbGVyc1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGlmIChwcm9wTmFtZSA9PT0gJ29uaW5wdXQnKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAvKiB0c2xpbnQ6ZGlzYWJsZSBuby10aGlzLWtleXdvcmQgbm8taW52YWxpZC10aGlzIG9ubHktYXJyb3ctZnVuY3Rpb25zIG5vLXZvaWQtZXhwcmVzc2lvbiAqL1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgKGZ1bmN0aW9uICgpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAvLyByZWNvcmQgdGhlIGV2dC50YXJnZXQudmFsdWUsIGJlY2F1c2UgSUUgYW5kIEVkZ2Ugc29tZXRpbWVzIGRvIGEgcmVxdWVzdEFuaW1hdGlvbkZyYW1lIGJldHdlZW4gY2hhbmdpbmcgdmFsdWUgYW5kIHJ1bm5pbmcgb25pbnB1dFxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHZhciBvbGRQcm9wVmFsdWUgPSBwcm9wVmFsdWU7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcHJvcFZhbHVlID0gZnVuY3Rpb24gKGV2dCkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBvbGRQcm9wVmFsdWUuYXBwbHkodGhpcywgW2V2dF0pO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBldnQudGFyZ2V0WydvbmlucHV0LXZhbHVlJ10gPSBldnQudGFyZ2V0LnZhbHVlOyAvLyBtYXkgYmUgSFRNTFRleHRBcmVhRWxlbWVudCBhcyB3ZWxsXHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0oKSk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAvKiB0c2xpbnQ6ZW5hYmxlICovXHJcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICAgICAgZG9tTm9kZVtwcm9wTmFtZV0gPSBwcm9wVmFsdWU7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgZWxzZSBpZiAocHJvamVjdGlvbk9wdGlvbnMubmFtZXNwYWNlID09PSBOQU1FU1BBQ0VfU1ZHKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKHByb3BOYW1lID09PSAnaHJlZicpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgZG9tTm9kZS5zZXRBdHRyaWJ1dGVOUyhOQU1FU1BBQ0VfWExJTkssIHByb3BOYW1lLCBwcm9wVmFsdWUpO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgLy8gYWxsIFNWRyBhdHRyaWJ1dGVzIGFyZSByZWFkLW9ubHkgaW4gRE9NLCBzby4uLlxyXG4gICAgICAgICAgICAgICAgICAgICAgICBkb21Ob2RlLnNldEF0dHJpYnV0ZShwcm9wTmFtZSwgcHJvcFZhbHVlKTtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICBlbHNlIGlmICh0eXBlID09PSAnc3RyaW5nJyAmJiBwcm9wTmFtZSAhPT0gJ3ZhbHVlJyAmJiBwcm9wTmFtZSAhPT0gJ2lubmVySFRNTCcpIHtcclxuICAgICAgICAgICAgICAgICAgICBkb21Ob2RlLnNldEF0dHJpYnV0ZShwcm9wTmFtZSwgcHJvcFZhbHVlKTtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgICAgIGRvbU5vZGVbcHJvcE5hbWVdID0gcHJvcFZhbHVlO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfTtcclxuICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IHByb3BDb3VudDsgaSsrKSB7XHJcbiAgICAgICAgICAgIF9sb29wXzEoaSk7XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIHZhciBhZGRDaGlsZHJlbiA9IGZ1bmN0aW9uIChkb21Ob2RlLCBjaGlsZHJlbiwgcHJvamVjdGlvbk9wdGlvbnMpIHtcclxuICAgICAgICBpZiAoIWNoaWxkcmVuKSB7XHJcbiAgICAgICAgICAgIHJldHVybjtcclxuICAgICAgICB9XHJcbiAgICAgICAgZm9yICh2YXIgX2kgPSAwLCBjaGlsZHJlbl8xID0gY2hpbGRyZW47IF9pIDwgY2hpbGRyZW5fMS5sZW5ndGg7IF9pKyspIHtcclxuICAgICAgICAgICAgdmFyIGNoaWxkID0gY2hpbGRyZW5fMVtfaV07XHJcbiAgICAgICAgICAgIGNyZWF0ZURvbShjaGlsZCwgZG9tTm9kZSwgdW5kZWZpbmVkLCBwcm9qZWN0aW9uT3B0aW9ucyk7XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIHZhciBpbml0UHJvcGVydGllc0FuZENoaWxkcmVuID0gZnVuY3Rpb24gKGRvbU5vZGUsIHZub2RlLCBwcm9qZWN0aW9uT3B0aW9ucykge1xyXG4gICAgICAgIGFkZENoaWxkcmVuKGRvbU5vZGUsIHZub2RlLmNoaWxkcmVuLCBwcm9qZWN0aW9uT3B0aW9ucyk7IC8vIGNoaWxkcmVuIGJlZm9yZSBwcm9wZXJ0aWVzLCBuZWVkZWQgZm9yIHZhbHVlIHByb3BlcnR5IG9mIDxzZWxlY3Q+LlxyXG4gICAgICAgIGlmICh2bm9kZS50ZXh0KSB7XHJcbiAgICAgICAgICAgIGRvbU5vZGUudGV4dENvbnRlbnQgPSB2bm9kZS50ZXh0O1xyXG4gICAgICAgIH1cclxuICAgICAgICBzZXRQcm9wZXJ0aWVzKGRvbU5vZGUsIHZub2RlLnByb3BlcnRpZXMsIHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICBpZiAodm5vZGUucHJvcGVydGllcyAmJiB2bm9kZS5wcm9wZXJ0aWVzLmFmdGVyQ3JlYXRlKSB7XHJcbiAgICAgICAgICAgIHZub2RlLnByb3BlcnRpZXMuYWZ0ZXJDcmVhdGUuYXBwbHkodm5vZGUucHJvcGVydGllcy5iaW5kIHx8IHZub2RlLnByb3BlcnRpZXMsIFtkb21Ob2RlLCBwcm9qZWN0aW9uT3B0aW9ucywgdm5vZGUudm5vZGVTZWxlY3Rvciwgdm5vZGUucHJvcGVydGllcywgdm5vZGUuY2hpbGRyZW5dKTtcclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgdmFyIGNyZWF0ZURvbSA9IGZ1bmN0aW9uICh2bm9kZSwgcGFyZW50Tm9kZSwgaW5zZXJ0QmVmb3JlLCBwcm9qZWN0aW9uT3B0aW9ucykge1xyXG4gICAgICAgIHZhciBkb21Ob2RlO1xyXG4gICAgICAgIHZhciBzdGFydCA9IDA7XHJcbiAgICAgICAgdmFyIHZub2RlU2VsZWN0b3IgPSB2bm9kZS52bm9kZVNlbGVjdG9yO1xyXG4gICAgICAgIHZhciBkb2MgPSBwYXJlbnROb2RlLm93bmVyRG9jdW1lbnQ7XHJcbiAgICAgICAgaWYgKHZub2RlU2VsZWN0b3IgPT09ICcnKSB7XHJcbiAgICAgICAgICAgIGRvbU5vZGUgPSB2bm9kZS5kb21Ob2RlID0gZG9jLmNyZWF0ZVRleHROb2RlKHZub2RlLnRleHQpO1xyXG4gICAgICAgICAgICBpZiAoaW5zZXJ0QmVmb3JlICE9PSB1bmRlZmluZWQpIHtcclxuICAgICAgICAgICAgICAgIHBhcmVudE5vZGUuaW5zZXJ0QmVmb3JlKGRvbU5vZGUsIGluc2VydEJlZm9yZSk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICBwYXJlbnROb2RlLmFwcGVuZENoaWxkKGRvbU5vZGUpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8PSB2bm9kZVNlbGVjdG9yLmxlbmd0aDsgKytpKSB7XHJcbiAgICAgICAgICAgICAgICB2YXIgYyA9IHZub2RlU2VsZWN0b3IuY2hhckF0KGkpO1xyXG4gICAgICAgICAgICAgICAgaWYgKGkgPT09IHZub2RlU2VsZWN0b3IubGVuZ3RoIHx8IGMgPT09ICcuJyB8fCBjID09PSAnIycpIHtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgdHlwZSA9IHZub2RlU2VsZWN0b3IuY2hhckF0KHN0YXJ0IC0gMSk7XHJcbiAgICAgICAgICAgICAgICAgICAgdmFyIGZvdW5kID0gdm5vZGVTZWxlY3Rvci5zbGljZShzdGFydCwgaSk7XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKHR5cGUgPT09ICcuJykge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBkb21Ob2RlLmNsYXNzTGlzdC5hZGQoZm91bmQpO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICBlbHNlIGlmICh0eXBlID09PSAnIycpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgZG9tTm9kZS5pZCA9IGZvdW5kO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgaWYgKGZvdW5kID09PSAnc3ZnJykge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgcHJvamVjdGlvbk9wdGlvbnMgPSBleHRlbmQocHJvamVjdGlvbk9wdGlvbnMsIHsgbmFtZXNwYWNlOiBOQU1FU1BBQ0VfU1ZHIH0pO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGlmIChwcm9qZWN0aW9uT3B0aW9ucy5uYW1lc3BhY2UgIT09IHVuZGVmaW5lZCkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgZG9tTm9kZSA9IHZub2RlLmRvbU5vZGUgPSBkb2MuY3JlYXRlRWxlbWVudE5TKHByb2plY3Rpb25PcHRpb25zLm5hbWVzcGFjZSwgZm91bmQpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgZG9tTm9kZSA9IHZub2RlLmRvbU5vZGUgPSAodm5vZGUuZG9tTm9kZSB8fCBkb2MuY3JlYXRlRWxlbWVudChmb3VuZCkpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgaWYgKGZvdW5kID09PSAnaW5wdXQnICYmIHZub2RlLnByb3BlcnRpZXMgJiYgdm5vZGUucHJvcGVydGllcy50eXBlICE9PSB1bmRlZmluZWQpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAvLyBJRTggYW5kIG9sZGVyIGRvbid0IHN1cHBvcnQgc2V0dGluZyBpbnB1dCB0eXBlIGFmdGVyIHRoZSBET00gTm9kZSBoYXMgYmVlbiBhZGRlZCB0byB0aGUgZG9jdW1lbnRcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBkb21Ob2RlLnNldEF0dHJpYnV0ZSgndHlwZScsIHZub2RlLnByb3BlcnRpZXMudHlwZSk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICAgICAgaWYgKGluc2VydEJlZm9yZSAhPT0gdW5kZWZpbmVkKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBwYXJlbnROb2RlLmluc2VydEJlZm9yZShkb21Ob2RlLCBpbnNlcnRCZWZvcmUpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGVsc2UgaWYgKGRvbU5vZGUucGFyZW50Tm9kZSAhPT0gcGFyZW50Tm9kZSkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgcGFyZW50Tm9kZS5hcHBlbmRDaGlsZChkb21Ob2RlKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICBzdGFydCA9IGkgKyAxO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGluaXRQcm9wZXJ0aWVzQW5kQ2hpbGRyZW4oZG9tTm9kZSwgdm5vZGUsIHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgdmFyIHVwZGF0ZURvbTtcclxuICAgIC8qKlxyXG4gICAgICogQWRkcyBvciByZW1vdmVzIGNsYXNzZXMgZnJvbSBhbiBFbGVtZW50XHJcbiAgICAgKiBAcGFyYW0gZG9tTm9kZSB0aGUgZWxlbWVudFxyXG4gICAgICogQHBhcmFtIGNsYXNzZXMgYSBzdHJpbmcgc2VwYXJhdGVkIGxpc3Qgb2YgY2xhc3Nlc1xyXG4gICAgICogQHBhcmFtIG9uIHRydWUgbWVhbnMgYWRkIGNsYXNzZXMsIGZhbHNlIG1lYW5zIHJlbW92ZVxyXG4gICAgICovXHJcbiAgICB2YXIgdG9nZ2xlQ2xhc3NlcyA9IGZ1bmN0aW9uIChkb21Ob2RlLCBjbGFzc2VzLCBvbikge1xyXG4gICAgICAgIGlmICghY2xhc3Nlcykge1xyXG4gICAgICAgICAgICByZXR1cm47XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGNsYXNzZXMuc3BsaXQoJyAnKS5mb3JFYWNoKGZ1bmN0aW9uIChjbGFzc1RvVG9nZ2xlKSB7XHJcbiAgICAgICAgICAgIGlmIChjbGFzc1RvVG9nZ2xlKSB7XHJcbiAgICAgICAgICAgICAgICBkb21Ob2RlLmNsYXNzTGlzdC50b2dnbGUoY2xhc3NUb1RvZ2dsZSwgb24pO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfSk7XHJcbiAgICB9O1xyXG4gICAgdmFyIHVwZGF0ZVByb3BlcnRpZXMgPSBmdW5jdGlvbiAoZG9tTm9kZSwgcHJldmlvdXNQcm9wZXJ0aWVzLCBwcm9wZXJ0aWVzLCBwcm9qZWN0aW9uT3B0aW9ucykge1xyXG4gICAgICAgIGlmICghcHJvcGVydGllcykge1xyXG4gICAgICAgICAgICByZXR1cm47XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHZhciBwcm9wZXJ0aWVzVXBkYXRlZCA9IGZhbHNlO1xyXG4gICAgICAgIHZhciBwcm9wTmFtZXMgPSBPYmplY3Qua2V5cyhwcm9wZXJ0aWVzKTtcclxuICAgICAgICB2YXIgcHJvcENvdW50ID0gcHJvcE5hbWVzLmxlbmd0aDtcclxuICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IHByb3BDb3VudDsgaSsrKSB7XHJcbiAgICAgICAgICAgIHZhciBwcm9wTmFtZSA9IHByb3BOYW1lc1tpXTtcclxuICAgICAgICAgICAgLy8gYXNzdW1pbmcgdGhhdCBwcm9wZXJ0aWVzIHdpbGwgYmUgbnVsbGlmaWVkIGluc3RlYWQgb2YgbWlzc2luZyBpcyBieSBkZXNpZ25cclxuICAgICAgICAgICAgdmFyIHByb3BWYWx1ZSA9IHByb3BlcnRpZXNbcHJvcE5hbWVdO1xyXG4gICAgICAgICAgICB2YXIgcHJldmlvdXNWYWx1ZSA9IHByZXZpb3VzUHJvcGVydGllc1twcm9wTmFtZV07XHJcbiAgICAgICAgICAgIGlmIChwcm9wTmFtZSA9PT0gJ2NsYXNzJykge1xyXG4gICAgICAgICAgICAgICAgaWYgKHByZXZpb3VzVmFsdWUgIT09IHByb3BWYWx1ZSkge1xyXG4gICAgICAgICAgICAgICAgICAgIHRvZ2dsZUNsYXNzZXMoZG9tTm9kZSwgcHJldmlvdXNWYWx1ZSwgZmFsc2UpO1xyXG4gICAgICAgICAgICAgICAgICAgIHRvZ2dsZUNsYXNzZXMoZG9tTm9kZSwgcHJvcFZhbHVlLCB0cnVlKTtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBlbHNlIGlmIChwcm9wTmFtZSA9PT0gJ2NsYXNzZXMnKSB7XHJcbiAgICAgICAgICAgICAgICB2YXIgY2xhc3NMaXN0ID0gZG9tTm9kZS5jbGFzc0xpc3Q7XHJcbiAgICAgICAgICAgICAgICB2YXIgY2xhc3NOYW1lcyA9IE9iamVjdC5rZXlzKHByb3BWYWx1ZSk7XHJcbiAgICAgICAgICAgICAgICB2YXIgY2xhc3NOYW1lQ291bnQgPSBjbGFzc05hbWVzLmxlbmd0aDtcclxuICAgICAgICAgICAgICAgIGZvciAodmFyIGogPSAwOyBqIDwgY2xhc3NOYW1lQ291bnQ7IGorKykge1xyXG4gICAgICAgICAgICAgICAgICAgIHZhciBjbGFzc05hbWUgPSBjbGFzc05hbWVzW2pdO1xyXG4gICAgICAgICAgICAgICAgICAgIHZhciBvbiA9ICEhcHJvcFZhbHVlW2NsYXNzTmFtZV07XHJcbiAgICAgICAgICAgICAgICAgICAgdmFyIHByZXZpb3VzT24gPSAhIXByZXZpb3VzVmFsdWVbY2xhc3NOYW1lXTtcclxuICAgICAgICAgICAgICAgICAgICBpZiAob24gPT09IHByZXZpb3VzT24pIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgY29udGludWU7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgIHByb3BlcnRpZXNVcGRhdGVkID0gdHJ1ZTtcclxuICAgICAgICAgICAgICAgICAgICBpZiAob24pIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgY2xhc3NMaXN0LmFkZChjbGFzc05hbWUpO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgY2xhc3NMaXN0LnJlbW92ZShjbGFzc05hbWUpO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBlbHNlIGlmIChwcm9wTmFtZSA9PT0gJ3N0eWxlcycpIHtcclxuICAgICAgICAgICAgICAgIHZhciBzdHlsZU5hbWVzID0gT2JqZWN0LmtleXMocHJvcFZhbHVlKTtcclxuICAgICAgICAgICAgICAgIHZhciBzdHlsZUNvdW50ID0gc3R5bGVOYW1lcy5sZW5ndGg7XHJcbiAgICAgICAgICAgICAgICBmb3IgKHZhciBqID0gMDsgaiA8IHN0eWxlQ291bnQ7IGorKykge1xyXG4gICAgICAgICAgICAgICAgICAgIHZhciBzdHlsZU5hbWUgPSBzdHlsZU5hbWVzW2pdO1xyXG4gICAgICAgICAgICAgICAgICAgIHZhciBuZXdTdHlsZVZhbHVlID0gcHJvcFZhbHVlW3N0eWxlTmFtZV07XHJcbiAgICAgICAgICAgICAgICAgICAgdmFyIG9sZFN0eWxlVmFsdWUgPSBwcmV2aW91c1ZhbHVlW3N0eWxlTmFtZV07XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKG5ld1N0eWxlVmFsdWUgPT09IG9sZFN0eWxlVmFsdWUpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgY29udGludWU7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgIHByb3BlcnRpZXNVcGRhdGVkID0gdHJ1ZTtcclxuICAgICAgICAgICAgICAgICAgICBpZiAobmV3U3R5bGVWYWx1ZSkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBjaGVja1N0eWxlVmFsdWUobmV3U3R5bGVWYWx1ZSk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHByb2plY3Rpb25PcHRpb25zLnN0eWxlQXBwbHllcihkb21Ob2RlLCBzdHlsZU5hbWUsIG5ld1N0eWxlVmFsdWUpO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgcHJvamVjdGlvbk9wdGlvbnMuc3R5bGVBcHBseWVyKGRvbU5vZGUsIHN0eWxlTmFtZSwgJycpO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgIGlmICghcHJvcFZhbHVlICYmIHR5cGVvZiBwcmV2aW91c1ZhbHVlID09PSAnc3RyaW5nJykge1xyXG4gICAgICAgICAgICAgICAgICAgIHByb3BWYWx1ZSA9ICcnO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgaWYgKHByb3BOYW1lID09PSAndmFsdWUnKSB7IC8vIHZhbHVlIGNhbiBiZSBtYW5pcHVsYXRlZCBieSB0aGUgdXNlciBkaXJlY3RseSBhbmQgdXNpbmcgZXZlbnQucHJldmVudERlZmF1bHQoKSBpcyBub3QgYW4gb3B0aW9uXHJcbiAgICAgICAgICAgICAgICAgICAgdmFyIGRvbVZhbHVlID0gZG9tTm9kZVtwcm9wTmFtZV07XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKGRvbVZhbHVlICE9PSBwcm9wVmFsdWUgLy8gVGhlICd2YWx1ZScgaW4gdGhlIERPTSB0cmVlICE9PSBuZXdWYWx1ZVxyXG4gICAgICAgICAgICAgICAgICAgICAgICAmJiAoZG9tTm9kZVsnb25pbnB1dC12YWx1ZSddXHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICA/IGRvbVZhbHVlID09PSBkb21Ob2RlWydvbmlucHV0LXZhbHVlJ10gLy8gSWYgdGhlIGxhc3QgcmVwb3J0ZWQgdmFsdWUgdG8gJ29uaW5wdXQnIGRvZXMgbm90IG1hdGNoIGRvbVZhbHVlLCBkbyBub3RoaW5nIGFuZCB3YWl0IGZvciBvbmlucHV0XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICA6IHByb3BWYWx1ZSAhPT0gcHJldmlvdXNWYWx1ZSAvLyBPbmx5IHVwZGF0ZSB0aGUgdmFsdWUgaWYgdGhlIHZkb20gY2hhbmdlZFxyXG4gICAgICAgICAgICAgICAgICAgICAgICApKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIC8vIFRoZSBlZGdlIGNhc2VzIGFyZSBkZXNjcmliZWQgaW4gdGhlIHRlc3RzXHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGRvbU5vZGVbcHJvcE5hbWVdID0gcHJvcFZhbHVlOyAvLyBSZXNldCB0aGUgdmFsdWUsIGV2ZW4gaWYgdGhlIHZpcnR1YWwgRE9NIGRpZCBub3QgY2hhbmdlXHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGRvbU5vZGVbJ29uaW5wdXQtdmFsdWUnXSA9IHVuZGVmaW5lZDtcclxuICAgICAgICAgICAgICAgICAgICB9IC8vIGVsc2UgZG8gbm90IHVwZGF0ZSB0aGUgZG9tTm9kZSwgb3RoZXJ3aXNlIHRoZSBjdXJzb3IgcG9zaXRpb24gd291bGQgYmUgY2hhbmdlZFxyXG4gICAgICAgICAgICAgICAgICAgIGlmIChwcm9wVmFsdWUgIT09IHByZXZpb3VzVmFsdWUpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgcHJvcGVydGllc1VwZGF0ZWQgPSB0cnVlO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIGVsc2UgaWYgKHByb3BWYWx1ZSAhPT0gcHJldmlvdXNWYWx1ZSkge1xyXG4gICAgICAgICAgICAgICAgICAgIHZhciB0eXBlID0gdHlwZW9mIHByb3BWYWx1ZTtcclxuICAgICAgICAgICAgICAgICAgICBpZiAodHlwZSAhPT0gJ2Z1bmN0aW9uJyB8fCAhcHJvamVjdGlvbk9wdGlvbnMuZXZlbnRIYW5kbGVySW50ZXJjZXB0b3IpIHsgLy8gRnVuY3Rpb24gdXBkYXRlcyBhcmUgZXhwZWN0ZWQgdG8gYmUgaGFuZGxlZCBieSB0aGUgRXZlbnRIYW5kbGVySW50ZXJjZXB0b3JcclxuICAgICAgICAgICAgICAgICAgICAgICAgaWYgKHByb2plY3Rpb25PcHRpb25zLm5hbWVzcGFjZSA9PT0gTkFNRVNQQUNFX1NWRykge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgaWYgKHByb3BOYW1lID09PSAnaHJlZicpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBkb21Ob2RlLnNldEF0dHJpYnV0ZU5TKE5BTUVTUEFDRV9YTElOSywgcHJvcE5hbWUsIHByb3BWYWx1ZSk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAvLyBhbGwgU1ZHIGF0dHJpYnV0ZXMgYXJlIHJlYWQtb25seSBpbiBET00sIHNvLi4uXHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZG9tTm9kZS5zZXRBdHRyaWJ1dGUocHJvcE5hbWUsIHByb3BWYWx1ZSk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICAgICAgZWxzZSBpZiAodHlwZSA9PT0gJ3N0cmluZycgJiYgcHJvcE5hbWUgIT09ICdpbm5lckhUTUwnKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAocHJvcE5hbWUgPT09ICdyb2xlJyAmJiBwcm9wVmFsdWUgPT09ICcnKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZG9tTm9kZS5yZW1vdmVBdHRyaWJ1dGUocHJvcE5hbWUpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZG9tTm9kZS5zZXRBdHRyaWJ1dGUocHJvcE5hbWUsIHByb3BWYWx1ZSk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICAgICAgZWxzZSBpZiAoZG9tTm9kZVtwcm9wTmFtZV0gIT09IHByb3BWYWx1ZSkgeyAvLyBDb21wYXJpc29uIGlzIGhlcmUgZm9yIHNpZGUtZWZmZWN0cyBpbiBFZGdlIHdpdGggc2Nyb2xsTGVmdCBhbmQgc2Nyb2xsVG9wXHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBkb21Ob2RlW3Byb3BOYW1lXSA9IHByb3BWYWx1ZTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgICAgICBwcm9wZXJ0aWVzVXBkYXRlZCA9IHRydWU7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHJldHVybiBwcm9wZXJ0aWVzVXBkYXRlZDtcclxuICAgIH07XHJcbiAgICB2YXIgdXBkYXRlQ2hpbGRyZW4gPSBmdW5jdGlvbiAodm5vZGUsIGRvbU5vZGUsIG9sZENoaWxkcmVuLCBuZXdDaGlsZHJlbiwgcHJvamVjdGlvbk9wdGlvbnMpIHtcclxuICAgICAgICBpZiAob2xkQ2hpbGRyZW4gPT09IG5ld0NoaWxkcmVuKSB7XHJcbiAgICAgICAgICAgIHJldHVybiBmYWxzZTtcclxuICAgICAgICB9XHJcbiAgICAgICAgb2xkQ2hpbGRyZW4gPSBvbGRDaGlsZHJlbiB8fCBlbXB0eUFycmF5O1xyXG4gICAgICAgIG5ld0NoaWxkcmVuID0gbmV3Q2hpbGRyZW4gfHwgZW1wdHlBcnJheTtcclxuICAgICAgICB2YXIgb2xkQ2hpbGRyZW5MZW5ndGggPSBvbGRDaGlsZHJlbi5sZW5ndGg7XHJcbiAgICAgICAgdmFyIG5ld0NoaWxkcmVuTGVuZ3RoID0gbmV3Q2hpbGRyZW4ubGVuZ3RoO1xyXG4gICAgICAgIHZhciBvbGRJbmRleCA9IDA7XHJcbiAgICAgICAgdmFyIG5ld0luZGV4ID0gMDtcclxuICAgICAgICB2YXIgaTtcclxuICAgICAgICB2YXIgdGV4dFVwZGF0ZWQgPSBmYWxzZTtcclxuICAgICAgICB3aGlsZSAobmV3SW5kZXggPCBuZXdDaGlsZHJlbkxlbmd0aCkge1xyXG4gICAgICAgICAgICB2YXIgb2xkQ2hpbGQgPSAob2xkSW5kZXggPCBvbGRDaGlsZHJlbkxlbmd0aCkgPyBvbGRDaGlsZHJlbltvbGRJbmRleF0gOiB1bmRlZmluZWQ7XHJcbiAgICAgICAgICAgIHZhciBuZXdDaGlsZCA9IG5ld0NoaWxkcmVuW25ld0luZGV4XTtcclxuICAgICAgICAgICAgaWYgKG9sZENoaWxkICE9PSB1bmRlZmluZWQgJiYgc2FtZShvbGRDaGlsZCwgbmV3Q2hpbGQpKSB7XHJcbiAgICAgICAgICAgICAgICB0ZXh0VXBkYXRlZCA9IHVwZGF0ZURvbShvbGRDaGlsZCwgbmV3Q2hpbGQsIHByb2plY3Rpb25PcHRpb25zKSB8fCB0ZXh0VXBkYXRlZDtcclxuICAgICAgICAgICAgICAgIG9sZEluZGV4Kys7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICB2YXIgZmluZE9sZEluZGV4ID0gZmluZEluZGV4T2ZDaGlsZChvbGRDaGlsZHJlbiwgbmV3Q2hpbGQsIG9sZEluZGV4ICsgMSk7XHJcbiAgICAgICAgICAgICAgICBpZiAoZmluZE9sZEluZGV4ID49IDApIHtcclxuICAgICAgICAgICAgICAgICAgICAvLyBSZW1vdmUgcHJlY2VkaW5nIG1pc3NpbmcgY2hpbGRyZW5cclxuICAgICAgICAgICAgICAgICAgICBmb3IgKGkgPSBvbGRJbmRleDsgaSA8IGZpbmRPbGRJbmRleDsgaSsrKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIG5vZGVUb1JlbW92ZShvbGRDaGlsZHJlbltpXSk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGNoZWNrRGlzdGluZ3Vpc2hhYmxlKG9sZENoaWxkcmVuLCBpLCB2bm9kZSwgJ3JlbW92ZWQnKTtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgdGV4dFVwZGF0ZWQgPSB1cGRhdGVEb20ob2xkQ2hpbGRyZW5bZmluZE9sZEluZGV4XSwgbmV3Q2hpbGQsIHByb2plY3Rpb25PcHRpb25zKSB8fCB0ZXh0VXBkYXRlZDtcclxuICAgICAgICAgICAgICAgICAgICBvbGRJbmRleCA9IGZpbmRPbGRJbmRleCArIDE7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgICAvLyBOZXcgY2hpbGRcclxuICAgICAgICAgICAgICAgICAgICBjcmVhdGVEb20obmV3Q2hpbGQsIGRvbU5vZGUsIChvbGRJbmRleCA8IG9sZENoaWxkcmVuTGVuZ3RoKSA/IG9sZENoaWxkcmVuW29sZEluZGV4XS5kb21Ob2RlIDogdW5kZWZpbmVkLCBwcm9qZWN0aW9uT3B0aW9ucyk7XHJcbiAgICAgICAgICAgICAgICAgICAgbm9kZUFkZGVkKG5ld0NoaWxkKTtcclxuICAgICAgICAgICAgICAgICAgICBjaGVja0Rpc3Rpbmd1aXNoYWJsZShuZXdDaGlsZHJlbiwgbmV3SW5kZXgsIHZub2RlLCAnYWRkZWQnKTtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBuZXdJbmRleCsrO1xyXG4gICAgICAgIH1cclxuICAgICAgICBpZiAob2xkQ2hpbGRyZW5MZW5ndGggPiBvbGRJbmRleCkge1xyXG4gICAgICAgICAgICAvLyBSZW1vdmUgY2hpbGQgZnJhZ21lbnRzXHJcbiAgICAgICAgICAgIGZvciAoaSA9IG9sZEluZGV4OyBpIDwgb2xkQ2hpbGRyZW5MZW5ndGg7IGkrKykge1xyXG4gICAgICAgICAgICAgICAgbm9kZVRvUmVtb3ZlKG9sZENoaWxkcmVuW2ldKTtcclxuICAgICAgICAgICAgICAgIGNoZWNrRGlzdGluZ3Vpc2hhYmxlKG9sZENoaWxkcmVuLCBpLCB2bm9kZSwgJ3JlbW92ZWQnKTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgICAgICByZXR1cm4gdGV4dFVwZGF0ZWQ7XHJcbiAgICB9O1xyXG4gICAgdXBkYXRlRG9tID0gZnVuY3Rpb24gKHByZXZpb3VzLCB2bm9kZSwgcHJvamVjdGlvbk9wdGlvbnMpIHtcclxuICAgICAgICB2YXIgZG9tTm9kZSA9IHByZXZpb3VzLmRvbU5vZGU7XHJcbiAgICAgICAgdmFyIHRleHRVcGRhdGVkID0gZmFsc2U7XHJcbiAgICAgICAgaWYgKHByZXZpb3VzID09PSB2bm9kZSkge1xyXG4gICAgICAgICAgICByZXR1cm4gZmFsc2U7IC8vIEJ5IGNvbnRyYWN0LCBWTm9kZSBvYmplY3RzIG1heSBub3QgYmUgbW9kaWZpZWQgYW55bW9yZSBhZnRlciBwYXNzaW5nIHRoZW0gdG8gbWFxdWV0dGVcclxuICAgICAgICB9XHJcbiAgICAgICAgdmFyIHVwZGF0ZWQgPSBmYWxzZTtcclxuICAgICAgICBpZiAodm5vZGUudm5vZGVTZWxlY3RvciA9PT0gJycpIHtcclxuICAgICAgICAgICAgaWYgKHZub2RlLnRleHQgIT09IHByZXZpb3VzLnRleHQpIHtcclxuICAgICAgICAgICAgICAgIHZhciBuZXdUZXh0Tm9kZSA9IGRvbU5vZGUub3duZXJEb2N1bWVudC5jcmVhdGVUZXh0Tm9kZSh2bm9kZS50ZXh0KTtcclxuICAgICAgICAgICAgICAgIGRvbU5vZGUucGFyZW50Tm9kZS5yZXBsYWNlQ2hpbGQobmV3VGV4dE5vZGUsIGRvbU5vZGUpO1xyXG4gICAgICAgICAgICAgICAgdm5vZGUuZG9tTm9kZSA9IG5ld1RleHROb2RlO1xyXG4gICAgICAgICAgICAgICAgdGV4dFVwZGF0ZWQgPSB0cnVlO1xyXG4gICAgICAgICAgICAgICAgcmV0dXJuIHRleHRVcGRhdGVkO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIHZub2RlLmRvbU5vZGUgPSBkb21Ob2RlO1xyXG4gICAgICAgIH1cclxuICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgaWYgKHZub2RlLnZub2RlU2VsZWN0b3IubGFzdEluZGV4T2YoJ3N2ZycsIDApID09PSAwKSB7IC8vIGxhc3RJbmRleE9mKG5lZWRsZSwwKT09PTAgbWVhbnMgU3RhcnRzV2l0aFxyXG4gICAgICAgICAgICAgICAgcHJvamVjdGlvbk9wdGlvbnMgPSBleHRlbmQocHJvamVjdGlvbk9wdGlvbnMsIHsgbmFtZXNwYWNlOiBOQU1FU1BBQ0VfU1ZHIH0pO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGlmIChwcmV2aW91cy50ZXh0ICE9PSB2bm9kZS50ZXh0KSB7XHJcbiAgICAgICAgICAgICAgICB1cGRhdGVkID0gdHJ1ZTtcclxuICAgICAgICAgICAgICAgIGlmICh2bm9kZS50ZXh0ID09PSB1bmRlZmluZWQpIHtcclxuICAgICAgICAgICAgICAgICAgICBkb21Ob2RlLnJlbW92ZUNoaWxkKGRvbU5vZGUuZmlyc3RDaGlsZCk7IC8vIHRoZSBvbmx5IHRleHRub2RlIHByZXN1bWFibHlcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgICAgIGRvbU5vZGUudGV4dENvbnRlbnQgPSB2bm9kZS50ZXh0O1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIHZub2RlLmRvbU5vZGUgPSBkb21Ob2RlO1xyXG4gICAgICAgICAgICB1cGRhdGVkID0gdXBkYXRlQ2hpbGRyZW4odm5vZGUsIGRvbU5vZGUsIHByZXZpb3VzLmNoaWxkcmVuLCB2bm9kZS5jaGlsZHJlbiwgcHJvamVjdGlvbk9wdGlvbnMpIHx8IHVwZGF0ZWQ7XHJcbiAgICAgICAgICAgIHVwZGF0ZWQgPSB1cGRhdGVQcm9wZXJ0aWVzKGRvbU5vZGUsIHByZXZpb3VzLnByb3BlcnRpZXMsIHZub2RlLnByb3BlcnRpZXMsIHByb2plY3Rpb25PcHRpb25zKSB8fCB1cGRhdGVkO1xyXG4gICAgICAgICAgICBpZiAodm5vZGUucHJvcGVydGllcyAmJiB2bm9kZS5wcm9wZXJ0aWVzLmFmdGVyVXBkYXRlKSB7XHJcbiAgICAgICAgICAgICAgICB2bm9kZS5wcm9wZXJ0aWVzLmFmdGVyVXBkYXRlLmFwcGx5KHZub2RlLnByb3BlcnRpZXMuYmluZCB8fCB2bm9kZS5wcm9wZXJ0aWVzLCBbZG9tTm9kZSwgcHJvamVjdGlvbk9wdGlvbnMsIHZub2RlLnZub2RlU2VsZWN0b3IsIHZub2RlLnByb3BlcnRpZXMsIHZub2RlLmNoaWxkcmVuXSk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgaWYgKHVwZGF0ZWQgJiYgdm5vZGUucHJvcGVydGllcyAmJiB2bm9kZS5wcm9wZXJ0aWVzLnVwZGF0ZUFuaW1hdGlvbikge1xyXG4gICAgICAgICAgICB2bm9kZS5wcm9wZXJ0aWVzLnVwZGF0ZUFuaW1hdGlvbihkb21Ob2RlLCB2bm9kZS5wcm9wZXJ0aWVzLCBwcmV2aW91cy5wcm9wZXJ0aWVzKTtcclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuIHRleHRVcGRhdGVkO1xyXG4gICAgfTtcclxuICAgIHZhciBjcmVhdGVQcm9qZWN0aW9uID0gZnVuY3Rpb24gKHZub2RlLCBwcm9qZWN0aW9uT3B0aW9ucykge1xyXG4gICAgICAgIHJldHVybiB7XHJcbiAgICAgICAgICAgIGdldExhc3RSZW5kZXI6IGZ1bmN0aW9uICgpIHsgcmV0dXJuIHZub2RlOyB9LFxyXG4gICAgICAgICAgICB1cGRhdGU6IGZ1bmN0aW9uICh1cGRhdGVkVm5vZGUpIHtcclxuICAgICAgICAgICAgICAgIGlmICh2bm9kZS52bm9kZVNlbGVjdG9yICE9PSB1cGRhdGVkVm5vZGUudm5vZGVTZWxlY3Rvcikge1xyXG4gICAgICAgICAgICAgICAgICAgIHRocm93IG5ldyBFcnJvcignVGhlIHNlbGVjdG9yIGZvciB0aGUgcm9vdCBWTm9kZSBtYXkgbm90IGJlIGNoYW5nZWQuIChjb25zaWRlciB1c2luZyBkb20ubWVyZ2UgYW5kIGFkZCBvbmUgZXh0cmEgbGV2ZWwgdG8gdGhlIHZpcnR1YWwgRE9NKScpO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgdmFyIHByZXZpb3VzVk5vZGUgPSB2bm9kZTtcclxuICAgICAgICAgICAgICAgIHZub2RlID0gdXBkYXRlZFZub2RlO1xyXG4gICAgICAgICAgICAgICAgdXBkYXRlRG9tKHByZXZpb3VzVk5vZGUsIHVwZGF0ZWRWbm9kZSwgcHJvamVjdGlvbk9wdGlvbnMpO1xyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICBkb21Ob2RlOiB2bm9kZS5kb21Ob2RlXHJcbiAgICAgICAgfTtcclxuICAgIH07XG5cbiAgICB2YXIgREVGQVVMVF9QUk9KRUNUSU9OX09QVElPTlMgPSB7XHJcbiAgICAgICAgbmFtZXNwYWNlOiB1bmRlZmluZWQsXHJcbiAgICAgICAgcGVyZm9ybWFuY2VMb2dnZXI6IGZ1bmN0aW9uICgpIHsgcmV0dXJuIHVuZGVmaW5lZDsgfSxcclxuICAgICAgICBldmVudEhhbmRsZXJJbnRlcmNlcHRvcjogdW5kZWZpbmVkLFxyXG4gICAgICAgIHN0eWxlQXBwbHllcjogZnVuY3Rpb24gKGRvbU5vZGUsIHN0eWxlTmFtZSwgdmFsdWUpIHtcclxuICAgICAgICAgICAgLy8gUHJvdmlkZXMgYSBob29rIHRvIGFkZCB2ZW5kb3IgcHJlZml4ZXMgZm9yIGJyb3dzZXJzIHRoYXQgc3RpbGwgbmVlZCBpdC5cclxuICAgICAgICAgICAgZG9tTm9kZS5zdHlsZVtzdHlsZU5hbWVdID0gdmFsdWU7XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIHZhciBhcHBseURlZmF1bHRQcm9qZWN0aW9uT3B0aW9ucyA9IGZ1bmN0aW9uIChwcm9qZWN0b3JPcHRpb25zKSB7XHJcbiAgICAgICAgcmV0dXJuIGV4dGVuZChERUZBVUxUX1BST0pFQ1RJT05fT1BUSU9OUywgcHJvamVjdG9yT3B0aW9ucyk7XHJcbiAgICB9O1xyXG4gICAgdmFyIGRvbSA9IHtcclxuICAgICAgICAvKipcclxuICAgICAgICAgKiBDcmVhdGVzIGEgcmVhbCBET00gdHJlZSBmcm9tIGB2bm9kZWAuIFRoZSBbW1Byb2plY3Rpb25dXSBvYmplY3QgcmV0dXJuZWQgd2lsbCBjb250YWluIHRoZSByZXN1bHRpbmcgRE9NIE5vZGUgaW5cclxuICAgICAgICAgKiBpdHMgW1tQcm9qZWN0aW9uLmRvbU5vZGV8ZG9tTm9kZV1dIHByb3BlcnR5LlxyXG4gICAgICAgICAqIFRoaXMgaXMgYSBsb3ctbGV2ZWwgbWV0aG9kLiBVc2VycyB3aWxsIHR5cGljYWxseSB1c2UgYSBbW1Byb2plY3Rvcl1dIGluc3RlYWQuXHJcbiAgICAgICAgICogQHBhcmFtIHZub2RlIC0gVGhlIHJvb3Qgb2YgdGhlIHZpcnR1YWwgRE9NIHRyZWUgdGhhdCB3YXMgY3JlYXRlZCB1c2luZyB0aGUgW1toXV0gZnVuY3Rpb24uIE5PVEU6IFtbVk5vZGVdXVxyXG4gICAgICAgICAqIG9iamVjdHMgbWF5IG9ubHkgYmUgcmVuZGVyZWQgb25jZS5cclxuICAgICAgICAgKiBAcGFyYW0gcHJvamVjdGlvbk9wdGlvbnMgLSBPcHRpb25zIHRvIGJlIHVzZWQgdG8gY3JlYXRlIGFuZCB1cGRhdGUgdGhlIHByb2plY3Rpb24uXHJcbiAgICAgICAgICogQHJldHVybnMgVGhlIFtbUHJvamVjdGlvbl1dIHdoaWNoIGFsc28gY29udGFpbnMgdGhlIERPTSBOb2RlIHRoYXQgd2FzIGNyZWF0ZWQuXHJcbiAgICAgICAgICovXHJcbiAgICAgICAgY3JlYXRlOiBmdW5jdGlvbiAodm5vZGUsIHByb2plY3Rpb25PcHRpb25zKSB7XHJcbiAgICAgICAgICAgIHByb2plY3Rpb25PcHRpb25zID0gYXBwbHlEZWZhdWx0UHJvamVjdGlvbk9wdGlvbnMocHJvamVjdGlvbk9wdGlvbnMpO1xyXG4gICAgICAgICAgICBjcmVhdGVEb20odm5vZGUsIGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2RpdicpLCB1bmRlZmluZWQsIHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICAgICAgcmV0dXJuIGNyZWF0ZVByb2plY3Rpb24odm5vZGUsIHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICB9LFxyXG4gICAgICAgIC8qKlxyXG4gICAgICAgICAqIEFwcGVuZHMgYSBuZXcgY2hpbGQgbm9kZSB0byB0aGUgRE9NIHdoaWNoIGlzIGdlbmVyYXRlZCBmcm9tIGEgW1tWTm9kZV1dLlxyXG4gICAgICAgICAqIFRoaXMgaXMgYSBsb3ctbGV2ZWwgbWV0aG9kLiBVc2VycyB3aWxsIHR5cGljYWxseSB1c2UgYSBbW1Byb2plY3Rvcl1dIGluc3RlYWQuXHJcbiAgICAgICAgICogQHBhcmFtIHBhcmVudE5vZGUgLSBUaGUgcGFyZW50IG5vZGUgZm9yIHRoZSBuZXcgY2hpbGQgbm9kZS5cclxuICAgICAgICAgKiBAcGFyYW0gdm5vZGUgLSBUaGUgcm9vdCBvZiB0aGUgdmlydHVhbCBET00gdHJlZSB0aGF0IHdhcyBjcmVhdGVkIHVzaW5nIHRoZSBbW2hdXSBmdW5jdGlvbi4gTk9URTogW1tWTm9kZV1dXHJcbiAgICAgICAgICogb2JqZWN0cyBtYXkgb25seSBiZSByZW5kZXJlZCBvbmNlLlxyXG4gICAgICAgICAqIEBwYXJhbSBwcm9qZWN0aW9uT3B0aW9ucyAtIE9wdGlvbnMgdG8gYmUgdXNlZCB0byBjcmVhdGUgYW5kIHVwZGF0ZSB0aGUgW1tQcm9qZWN0aW9uXV0uXHJcbiAgICAgICAgICogQHJldHVybnMgVGhlIFtbUHJvamVjdGlvbl1dIHRoYXQgd2FzIGNyZWF0ZWQuXHJcbiAgICAgICAgICovXHJcbiAgICAgICAgYXBwZW5kOiBmdW5jdGlvbiAocGFyZW50Tm9kZSwgdm5vZGUsIHByb2plY3Rpb25PcHRpb25zKSB7XHJcbiAgICAgICAgICAgIHByb2plY3Rpb25PcHRpb25zID0gYXBwbHlEZWZhdWx0UHJvamVjdGlvbk9wdGlvbnMocHJvamVjdGlvbk9wdGlvbnMpO1xyXG4gICAgICAgICAgICBjcmVhdGVEb20odm5vZGUsIHBhcmVudE5vZGUsIHVuZGVmaW5lZCwgcHJvamVjdGlvbk9wdGlvbnMpO1xyXG4gICAgICAgICAgICByZXR1cm4gY3JlYXRlUHJvamVjdGlvbih2bm9kZSwgcHJvamVjdGlvbk9wdGlvbnMpO1xyXG4gICAgICAgIH0sXHJcbiAgICAgICAgLyoqXHJcbiAgICAgICAgICogSW5zZXJ0cyBhIG5ldyBET00gbm9kZSB3aGljaCBpcyBnZW5lcmF0ZWQgZnJvbSBhIFtbVk5vZGVdXS5cclxuICAgICAgICAgKiBUaGlzIGlzIGEgbG93LWxldmVsIG1ldGhvZC4gVXNlcnMgd2lsIHR5cGljYWxseSB1c2UgYSBbW1Byb2plY3Rvcl1dIGluc3RlYWQuXHJcbiAgICAgICAgICogQHBhcmFtIGJlZm9yZU5vZGUgLSBUaGUgbm9kZSB0aGF0IHRoZSBET00gTm9kZSBpcyBpbnNlcnRlZCBiZWZvcmUuXHJcbiAgICAgICAgICogQHBhcmFtIHZub2RlIC0gVGhlIHJvb3Qgb2YgdGhlIHZpcnR1YWwgRE9NIHRyZWUgdGhhdCB3YXMgY3JlYXRlZCB1c2luZyB0aGUgW1toXV0gZnVuY3Rpb24uXHJcbiAgICAgICAgICogTk9URTogW1tWTm9kZV1dIG9iamVjdHMgbWF5IG9ubHkgYmUgcmVuZGVyZWQgb25jZS5cclxuICAgICAgICAgKiBAcGFyYW0gcHJvamVjdGlvbk9wdGlvbnMgLSBPcHRpb25zIHRvIGJlIHVzZWQgdG8gY3JlYXRlIGFuZCB1cGRhdGUgdGhlIHByb2plY3Rpb24sIHNlZSBbW2NyZWF0ZVByb2plY3Rvcl1dLlxyXG4gICAgICAgICAqIEByZXR1cm5zIFRoZSBbW1Byb2plY3Rpb25dXSB0aGF0IHdhcyBjcmVhdGVkLlxyXG4gICAgICAgICAqL1xyXG4gICAgICAgIGluc2VydEJlZm9yZTogZnVuY3Rpb24gKGJlZm9yZU5vZGUsIHZub2RlLCBwcm9qZWN0aW9uT3B0aW9ucykge1xyXG4gICAgICAgICAgICBwcm9qZWN0aW9uT3B0aW9ucyA9IGFwcGx5RGVmYXVsdFByb2plY3Rpb25PcHRpb25zKHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICAgICAgY3JlYXRlRG9tKHZub2RlLCBiZWZvcmVOb2RlLnBhcmVudE5vZGUsIGJlZm9yZU5vZGUsIHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICAgICAgcmV0dXJuIGNyZWF0ZVByb2plY3Rpb24odm5vZGUsIHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICB9LFxyXG4gICAgICAgIC8qKlxyXG4gICAgICAgICAqIE1lcmdlcyBhIG5ldyBET00gbm9kZSB3aGljaCBpcyBnZW5lcmF0ZWQgZnJvbSBhIFtbVk5vZGVdXSB3aXRoIGFuIGV4aXN0aW5nIERPTSBOb2RlLlxyXG4gICAgICAgICAqIFRoaXMgbWVhbnMgdGhhdCB0aGUgdmlydHVhbCBET00gYW5kIHRoZSByZWFsIERPTSB3aWxsIGhhdmUgb25lIG92ZXJsYXBwaW5nIGVsZW1lbnQuXHJcbiAgICAgICAgICogVGhlcmVmb3JlIHRoZSBzZWxlY3RvciBmb3IgdGhlIHJvb3QgW1tWTm9kZV1dIHdpbGwgYmUgaWdub3JlZCwgYnV0IGl0cyBwcm9wZXJ0aWVzIGFuZCBjaGlsZHJlbiB3aWxsIGJlIGFwcGxpZWQgdG8gdGhlIEVsZW1lbnQgcHJvdmlkZWQuXHJcbiAgICAgICAgICogVGhpcyBpcyBhIGxvdy1sZXZlbCBtZXRob2QuIFVzZXJzIHdpbCB0eXBpY2FsbHkgdXNlIGEgW1tQcm9qZWN0b3JdXSBpbnN0ZWFkLlxyXG4gICAgICAgICAqIEBwYXJhbSBlbGVtZW50IC0gVGhlIGV4aXN0aW5nIGVsZW1lbnQgdG8gYWRvcHQgYXMgdGhlIHJvb3Qgb2YgdGhlIG5ldyB2aXJ0dWFsIERPTS4gRXhpc3RpbmcgYXR0cmlidXRlcyBhbmQgY2hpbGQgbm9kZXMgYXJlIHByZXNlcnZlZC5cclxuICAgICAgICAgKiBAcGFyYW0gdm5vZGUgLSBUaGUgcm9vdCBvZiB0aGUgdmlydHVhbCBET00gdHJlZSB0aGF0IHdhcyBjcmVhdGVkIHVzaW5nIHRoZSBbW2hdXSBmdW5jdGlvbi4gTk9URTogW1tWTm9kZV1dIG9iamVjdHNcclxuICAgICAgICAgKiBtYXkgb25seSBiZSByZW5kZXJlZCBvbmNlLlxyXG4gICAgICAgICAqIEBwYXJhbSBwcm9qZWN0aW9uT3B0aW9ucyAtIE9wdGlvbnMgdG8gYmUgdXNlZCB0byBjcmVhdGUgYW5kIHVwZGF0ZSB0aGUgcHJvamVjdGlvbiwgc2VlIFtbY3JlYXRlUHJvamVjdG9yXV0uXHJcbiAgICAgICAgICogQHJldHVybnMgVGhlIFtbUHJvamVjdGlvbl1dIHRoYXQgd2FzIGNyZWF0ZWQuXHJcbiAgICAgICAgICovXHJcbiAgICAgICAgbWVyZ2U6IGZ1bmN0aW9uIChlbGVtZW50LCB2bm9kZSwgcHJvamVjdGlvbk9wdGlvbnMpIHtcclxuICAgICAgICAgICAgcHJvamVjdGlvbk9wdGlvbnMgPSBhcHBseURlZmF1bHRQcm9qZWN0aW9uT3B0aW9ucyhwcm9qZWN0aW9uT3B0aW9ucyk7XHJcbiAgICAgICAgICAgIHZub2RlLmRvbU5vZGUgPSBlbGVtZW50O1xyXG4gICAgICAgICAgICBpbml0UHJvcGVydGllc0FuZENoaWxkcmVuKGVsZW1lbnQsIHZub2RlLCBwcm9qZWN0aW9uT3B0aW9ucyk7XHJcbiAgICAgICAgICAgIHJldHVybiBjcmVhdGVQcm9qZWN0aW9uKHZub2RlLCBwcm9qZWN0aW9uT3B0aW9ucyk7XHJcbiAgICAgICAgfSxcclxuICAgICAgICAvKipcclxuICAgICAgICAgKiBSZXBsYWNlcyBhbiBleGlzdGluZyBET00gbm9kZSB3aXRoIGEgbm9kZSBnZW5lcmF0ZWQgZnJvbSBhIFtbVk5vZGVdXS5cclxuICAgICAgICAgKiBUaGlzIGlzIGEgbG93LWxldmVsIG1ldGhvZC4gVXNlcnMgd2lsbCB0eXBpY2FsbHkgdXNlIGEgW1tQcm9qZWN0b3JdXSBpbnN0ZWFkLlxyXG4gICAgICAgICAqIEBwYXJhbSBlbGVtZW50IC0gVGhlIG5vZGUgZm9yIHRoZSBbW1ZOb2RlXV0gdG8gcmVwbGFjZS5cclxuICAgICAgICAgKiBAcGFyYW0gdm5vZGUgLSBUaGUgcm9vdCBvZiB0aGUgdmlydHVhbCBET00gdHJlZSB0aGF0IHdhcyBjcmVhdGVkIHVzaW5nIHRoZSBbW2hdXSBmdW5jdGlvbi4gTk9URTogW1tWTm9kZV1dXHJcbiAgICAgICAgICogb2JqZWN0cyBtYXkgb25seSBiZSByZW5kZXJlZCBvbmNlLlxyXG4gICAgICAgICAqIEBwYXJhbSBwcm9qZWN0aW9uT3B0aW9ucyAtIE9wdGlvbnMgdG8gYmUgdXNlZCB0byBjcmVhdGUgYW5kIHVwZGF0ZSB0aGUgW1tQcm9qZWN0aW9uXV0uXHJcbiAgICAgICAgICogQHJldHVybnMgVGhlIFtbUHJvamVjdGlvbl1dIHRoYXQgd2FzIGNyZWF0ZWQuXHJcbiAgICAgICAgICovXHJcbiAgICAgICAgcmVwbGFjZTogZnVuY3Rpb24gKGVsZW1lbnQsIHZub2RlLCBwcm9qZWN0aW9uT3B0aW9ucykge1xyXG4gICAgICAgICAgICBwcm9qZWN0aW9uT3B0aW9ucyA9IGFwcGx5RGVmYXVsdFByb2plY3Rpb25PcHRpb25zKHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICAgICAgY3JlYXRlRG9tKHZub2RlLCBlbGVtZW50LnBhcmVudE5vZGUsIGVsZW1lbnQsIHByb2plY3Rpb25PcHRpb25zKTtcclxuICAgICAgICAgICAgZWxlbWVudC5wYXJlbnROb2RlLnJlbW92ZUNoaWxkKGVsZW1lbnQpO1xyXG4gICAgICAgICAgICByZXR1cm4gY3JlYXRlUHJvamVjdGlvbih2bm9kZSwgcHJvamVjdGlvbk9wdGlvbnMpO1xyXG4gICAgICAgIH1cclxuICAgIH07XG5cbiAgICAvKiB0c2xpbnQ6ZGlzYWJsZSBmdW5jdGlvbi1uYW1lICovXHJcbiAgICB2YXIgdG9UZXh0Vk5vZGUgPSBmdW5jdGlvbiAoZGF0YSkge1xyXG4gICAgICAgIHJldHVybiB7XHJcbiAgICAgICAgICAgIHZub2RlU2VsZWN0b3I6ICcnLFxyXG4gICAgICAgICAgICBwcm9wZXJ0aWVzOiB1bmRlZmluZWQsXHJcbiAgICAgICAgICAgIGNoaWxkcmVuOiB1bmRlZmluZWQsXHJcbiAgICAgICAgICAgIHRleHQ6IGRhdGEudG9TdHJpbmcoKSxcclxuICAgICAgICAgICAgZG9tTm9kZTogbnVsbFxyXG4gICAgICAgIH07XHJcbiAgICB9O1xyXG4gICAgdmFyIGFwcGVuZENoaWxkcmVuID0gZnVuY3Rpb24gKHBhcmVudFNlbGVjdG9yLCBpbnNlcnRpb25zLCBtYWluKSB7XHJcbiAgICAgICAgZm9yICh2YXIgaSA9IDAsIGxlbmd0aF8xID0gaW5zZXJ0aW9ucy5sZW5ndGg7IGkgPCBsZW5ndGhfMTsgaSsrKSB7XHJcbiAgICAgICAgICAgIHZhciBpdGVtID0gaW5zZXJ0aW9uc1tpXTtcclxuICAgICAgICAgICAgaWYgKEFycmF5LmlzQXJyYXkoaXRlbSkpIHtcclxuICAgICAgICAgICAgICAgIGFwcGVuZENoaWxkcmVuKHBhcmVudFNlbGVjdG9yLCBpdGVtLCBtYWluKTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgIGlmIChpdGVtICE9PSBudWxsICYmIGl0ZW0gIT09IHVuZGVmaW5lZCAmJiBpdGVtICE9PSBmYWxzZSkge1xyXG4gICAgICAgICAgICAgICAgICAgIGlmICh0eXBlb2YgaXRlbSA9PT0gJ3N0cmluZycpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgaXRlbSA9IHRvVGV4dFZOb2RlKGl0ZW0pO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICBtYWluLnB1c2goaXRlbSk7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgZnVuY3Rpb24gaChzZWxlY3RvciwgcHJvcGVydGllcywgY2hpbGRyZW4pIHtcclxuICAgICAgICBpZiAoQXJyYXkuaXNBcnJheShwcm9wZXJ0aWVzKSkge1xyXG4gICAgICAgICAgICBjaGlsZHJlbiA9IHByb3BlcnRpZXM7XHJcbiAgICAgICAgICAgIHByb3BlcnRpZXMgPSB1bmRlZmluZWQ7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGVsc2UgaWYgKChwcm9wZXJ0aWVzICYmICh0eXBlb2YgcHJvcGVydGllcyA9PT0gJ3N0cmluZycgfHwgcHJvcGVydGllcy5oYXNPd25Qcm9wZXJ0eSgndm5vZGVTZWxlY3RvcicpKSkgfHxcclxuICAgICAgICAgICAgKGNoaWxkcmVuICYmICh0eXBlb2YgY2hpbGRyZW4gPT09ICdzdHJpbmcnIHx8IGNoaWxkcmVuLmhhc093blByb3BlcnR5KCd2bm9kZVNlbGVjdG9yJykpKSkge1xyXG4gICAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ2ggY2FsbGVkIHdpdGggaW52YWxpZCBhcmd1bWVudHMnKTtcclxuICAgICAgICB9XHJcbiAgICAgICAgdmFyIHRleHQ7XHJcbiAgICAgICAgdmFyIGZsYXR0ZW5lZENoaWxkcmVuO1xyXG4gICAgICAgIC8vIFJlY29nbml6ZSBhIGNvbW1vbiBzcGVjaWFsIGNhc2Ugd2hlcmUgdGhlcmUgaXMgb25seSBhIHNpbmdsZSB0ZXh0IG5vZGVcclxuICAgICAgICBpZiAoY2hpbGRyZW4gJiYgY2hpbGRyZW4ubGVuZ3RoID09PSAxICYmIHR5cGVvZiBjaGlsZHJlblswXSA9PT0gJ3N0cmluZycpIHtcclxuICAgICAgICAgICAgdGV4dCA9IGNoaWxkcmVuWzBdO1xyXG4gICAgICAgIH1cclxuICAgICAgICBlbHNlIGlmIChjaGlsZHJlbikge1xyXG4gICAgICAgICAgICBmbGF0dGVuZWRDaGlsZHJlbiA9IFtdO1xyXG4gICAgICAgICAgICBhcHBlbmRDaGlsZHJlbihzZWxlY3RvciwgY2hpbGRyZW4sIGZsYXR0ZW5lZENoaWxkcmVuKTtcclxuICAgICAgICAgICAgaWYgKGZsYXR0ZW5lZENoaWxkcmVuLmxlbmd0aCA9PT0gMCkge1xyXG4gICAgICAgICAgICAgICAgZmxhdHRlbmVkQ2hpbGRyZW4gPSB1bmRlZmluZWQ7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuIHtcclxuICAgICAgICAgICAgdm5vZGVTZWxlY3Rvcjogc2VsZWN0b3IsXHJcbiAgICAgICAgICAgIHByb3BlcnRpZXM6IHByb3BlcnRpZXMsXHJcbiAgICAgICAgICAgIGNoaWxkcmVuOiBmbGF0dGVuZWRDaGlsZHJlbixcclxuICAgICAgICAgICAgdGV4dDogKHRleHQgPT09ICcnKSA/IHVuZGVmaW5lZCA6IHRleHQsXHJcbiAgICAgICAgICAgIGRvbU5vZGU6IG51bGxcclxuICAgICAgICB9O1xyXG4gICAgfVxuXG4gICAgdmFyIGNyZWF0ZVBhcmVudE5vZGVQYXRoID0gZnVuY3Rpb24gKG5vZGUsIHJvb3ROb2RlKSB7XHJcbiAgICAgICAgdmFyIHBhcmVudE5vZGVQYXRoID0gW107XHJcbiAgICAgICAgd2hpbGUgKG5vZGUgIT09IHJvb3ROb2RlKSB7XHJcbiAgICAgICAgICAgIHBhcmVudE5vZGVQYXRoLnB1c2gobm9kZSk7XHJcbiAgICAgICAgICAgIG5vZGUgPSBub2RlLnBhcmVudE5vZGU7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHJldHVybiBwYXJlbnROb2RlUGF0aDtcclxuICAgIH07XHJcbiAgICB2YXIgZmluZDtcclxuICAgIGlmIChBcnJheS5wcm90b3R5cGUuZmluZCkge1xyXG4gICAgICAgIGZpbmQgPSBmdW5jdGlvbiAoaXRlbXMsIHByZWRpY2F0ZSkgeyByZXR1cm4gaXRlbXMuZmluZChwcmVkaWNhdGUpOyB9O1xyXG4gICAgfVxyXG4gICAgZWxzZSB7XHJcbiAgICAgICAgZmluZCA9IGZ1bmN0aW9uIChpdGVtcywgcHJlZGljYXRlKSB7IHJldHVybiBpdGVtcy5maWx0ZXIocHJlZGljYXRlKVswXTsgfTtcclxuICAgIH1cclxuICAgIHZhciBmaW5kVk5vZGVCeVBhcmVudE5vZGVQYXRoID0gZnVuY3Rpb24gKHZub2RlLCBwYXJlbnROb2RlUGF0aCkge1xyXG4gICAgICAgIHZhciByZXN1bHQgPSB2bm9kZTtcclxuICAgICAgICBwYXJlbnROb2RlUGF0aC5mb3JFYWNoKGZ1bmN0aW9uIChub2RlKSB7XHJcbiAgICAgICAgICAgIHJlc3VsdCA9IChyZXN1bHQgJiYgcmVzdWx0LmNoaWxkcmVuKSA/IGZpbmQocmVzdWx0LmNoaWxkcmVuLCBmdW5jdGlvbiAoY2hpbGQpIHsgcmV0dXJuIGNoaWxkLmRvbU5vZGUgPT09IG5vZGU7IH0pIDogdW5kZWZpbmVkO1xyXG4gICAgICAgIH0pO1xyXG4gICAgICAgIHJldHVybiByZXN1bHQ7XHJcbiAgICB9O1xyXG4gICAgdmFyIGNyZWF0ZUV2ZW50SGFuZGxlckludGVyY2VwdG9yID0gZnVuY3Rpb24gKHByb2plY3RvciwgZ2V0UHJvamVjdGlvbiwgcGVyZm9ybWFuY2VMb2dnZXIpIHtcclxuICAgICAgICB2YXIgbW9kaWZpZWRFdmVudEhhbmRsZXIgPSBmdW5jdGlvbiAoZXZ0KSB7XHJcbiAgICAgICAgICAgIHBlcmZvcm1hbmNlTG9nZ2VyKCdkb21FdmVudCcsIGV2dCk7XHJcbiAgICAgICAgICAgIHZhciBwcm9qZWN0aW9uID0gZ2V0UHJvamVjdGlvbigpO1xyXG4gICAgICAgICAgICB2YXIgcGFyZW50Tm9kZVBhdGggPSBjcmVhdGVQYXJlbnROb2RlUGF0aChldnQuY3VycmVudFRhcmdldCwgcHJvamVjdGlvbi5kb21Ob2RlKTtcclxuICAgICAgICAgICAgcGFyZW50Tm9kZVBhdGgucmV2ZXJzZSgpO1xyXG4gICAgICAgICAgICB2YXIgbWF0Y2hpbmdWTm9kZSA9IGZpbmRWTm9kZUJ5UGFyZW50Tm9kZVBhdGgocHJvamVjdGlvbi5nZXRMYXN0UmVuZGVyKCksIHBhcmVudE5vZGVQYXRoKTtcclxuICAgICAgICAgICAgcHJvamVjdG9yLnNjaGVkdWxlUmVuZGVyKCk7XHJcbiAgICAgICAgICAgIHZhciByZXN1bHQ7XHJcbiAgICAgICAgICAgIGlmIChtYXRjaGluZ1ZOb2RlKSB7XHJcbiAgICAgICAgICAgICAgICAvKiB0c2xpbnQ6ZGlzYWJsZSBuby1pbnZhbGlkLXRoaXMgKi9cclxuICAgICAgICAgICAgICAgIHJlc3VsdCA9IG1hdGNoaW5nVk5vZGUucHJvcGVydGllc1tcIm9uXCIgKyBldnQudHlwZV0uYXBwbHkobWF0Y2hpbmdWTm9kZS5wcm9wZXJ0aWVzLmJpbmQgfHwgdGhpcywgYXJndW1lbnRzKTtcclxuICAgICAgICAgICAgICAgIC8qIHRzbGludDplbmFibGUgbm8taW52YWxpZC10aGlzICovXHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgcGVyZm9ybWFuY2VMb2dnZXIoJ2RvbUV2ZW50UHJvY2Vzc2VkJywgZXZ0KTtcclxuICAgICAgICAgICAgcmV0dXJuIHJlc3VsdDtcclxuICAgICAgICB9O1xyXG4gICAgICAgIHJldHVybiBmdW5jdGlvbiAocHJvcGVydHlOYW1lLCBldmVudEhhbmRsZXIsIGRvbU5vZGUsIHByb3BlcnRpZXMpIHsgcmV0dXJuIG1vZGlmaWVkRXZlbnRIYW5kbGVyOyB9O1xyXG4gICAgfTtcclxuICAgIC8qKlxyXG4gICAgICogQ3JlYXRlcyBhIFtbUHJvamVjdG9yXV0gaW5zdGFuY2UgdXNpbmcgdGhlIHByb3ZpZGVkIHByb2plY3Rpb25PcHRpb25zLlxyXG4gICAgICpcclxuICAgICAqIEZvciBtb3JlIGluZm9ybWF0aW9uLCBzZWUgW1tQcm9qZWN0b3JdXS5cclxuICAgICAqXHJcbiAgICAgKiBAcGFyYW0gcHJvamVjdG9yT3B0aW9ucyAgIE9wdGlvbnMgdGhhdCBpbmZsdWVuY2UgaG93IHRoZSBET00gaXMgcmVuZGVyZWQgYW5kIHVwZGF0ZWQuXHJcbiAgICAgKi9cclxuICAgIHZhciBjcmVhdGVQcm9qZWN0b3IgPSBmdW5jdGlvbiAocHJvamVjdG9yT3B0aW9ucykge1xyXG4gICAgICAgIHZhciBwcm9qZWN0b3I7XHJcbiAgICAgICAgdmFyIHByb2plY3Rpb25PcHRpb25zID0gYXBwbHlEZWZhdWx0UHJvamVjdGlvbk9wdGlvbnMocHJvamVjdG9yT3B0aW9ucyk7XHJcbiAgICAgICAgdmFyIHBlcmZvcm1hbmNlTG9nZ2VyID0gcHJvamVjdGlvbk9wdGlvbnMucGVyZm9ybWFuY2VMb2dnZXI7XHJcbiAgICAgICAgdmFyIHJlbmRlckNvbXBsZXRlZCA9IHRydWU7XHJcbiAgICAgICAgdmFyIHNjaGVkdWxlZDtcclxuICAgICAgICB2YXIgc3RvcHBlZCA9IGZhbHNlO1xyXG4gICAgICAgIHZhciBwcm9qZWN0aW9ucyA9IFtdO1xyXG4gICAgICAgIHZhciByZW5kZXJGdW5jdGlvbnMgPSBbXTsgLy8gbWF0Y2hlcyB0aGUgcHJvamVjdGlvbnMgYXJyYXlcclxuICAgICAgICB2YXIgYWRkUHJvamVjdGlvbiA9IGZ1bmN0aW9uIChcclxuICAgICAgICAvKiBvbmUgb2Y6IGRvbS5hcHBlbmQsIGRvbS5pbnNlcnRCZWZvcmUsIGRvbS5yZXBsYWNlLCBkb20ubWVyZ2UgKi9cclxuICAgICAgICBkb21GdW5jdGlvbiwgXHJcbiAgICAgICAgLyogdGhlIHBhcmFtZXRlciBvZiB0aGUgZG9tRnVuY3Rpb24gKi9cclxuICAgICAgICBub2RlLCByZW5kZXJGdW5jdGlvbikge1xyXG4gICAgICAgICAgICB2YXIgcHJvamVjdGlvbjtcclxuICAgICAgICAgICAgdmFyIGdldFByb2plY3Rpb24gPSBmdW5jdGlvbiAoKSB7IHJldHVybiBwcm9qZWN0aW9uOyB9O1xyXG4gICAgICAgICAgICBwcm9qZWN0aW9uT3B0aW9ucy5ldmVudEhhbmRsZXJJbnRlcmNlcHRvciA9IGNyZWF0ZUV2ZW50SGFuZGxlckludGVyY2VwdG9yKHByb2plY3RvciwgZ2V0UHJvamVjdGlvbiwgcGVyZm9ybWFuY2VMb2dnZXIpO1xyXG4gICAgICAgICAgICBwcm9qZWN0aW9uID0gZG9tRnVuY3Rpb24obm9kZSwgcmVuZGVyRnVuY3Rpb24oKSwgcHJvamVjdGlvbk9wdGlvbnMpO1xyXG4gICAgICAgICAgICBwcm9qZWN0aW9ucy5wdXNoKHByb2plY3Rpb24pO1xyXG4gICAgICAgICAgICByZW5kZXJGdW5jdGlvbnMucHVzaChyZW5kZXJGdW5jdGlvbik7XHJcbiAgICAgICAgfTtcclxuICAgICAgICB2YXIgZG9SZW5kZXIgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgICAgIHNjaGVkdWxlZCA9IHVuZGVmaW5lZDtcclxuICAgICAgICAgICAgaWYgKCFyZW5kZXJDb21wbGV0ZWQpIHtcclxuICAgICAgICAgICAgICAgIHJldHVybjsgLy8gVGhlIGxhc3QgcmVuZGVyIHRocmV3IGFuIGVycm9yLCBpdCBzaG91bGQgaGF2ZSBiZWVuIGxvZ2dlZCBpbiB0aGUgYnJvd3NlciBjb25zb2xlLlxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIHJlbmRlckNvbXBsZXRlZCA9IGZhbHNlO1xyXG4gICAgICAgICAgICBwZXJmb3JtYW5jZUxvZ2dlcigncmVuZGVyU3RhcnQnLCB1bmRlZmluZWQpO1xyXG4gICAgICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IHByb2plY3Rpb25zLmxlbmd0aDsgaSsrKSB7XHJcbiAgICAgICAgICAgICAgICB2YXIgdXBkYXRlZFZub2RlID0gcmVuZGVyRnVuY3Rpb25zW2ldKCk7XHJcbiAgICAgICAgICAgICAgICBwZXJmb3JtYW5jZUxvZ2dlcigncmVuZGVyZWQnLCB1bmRlZmluZWQpO1xyXG4gICAgICAgICAgICAgICAgcHJvamVjdGlvbnNbaV0udXBkYXRlKHVwZGF0ZWRWbm9kZSk7XHJcbiAgICAgICAgICAgICAgICBwZXJmb3JtYW5jZUxvZ2dlcigncGF0Y2hlZCcsIHVuZGVmaW5lZCk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgcGVyZm9ybWFuY2VMb2dnZXIoJ3JlbmRlckRvbmUnLCB1bmRlZmluZWQpO1xyXG4gICAgICAgICAgICByZW5kZXJDb21wbGV0ZWQgPSB0cnVlO1xyXG4gICAgICAgIH07XHJcbiAgICAgICAgcHJvamVjdG9yID0ge1xyXG4gICAgICAgICAgICByZW5kZXJOb3c6IGRvUmVuZGVyLFxyXG4gICAgICAgICAgICBzY2hlZHVsZVJlbmRlcjogZnVuY3Rpb24gKCkge1xyXG4gICAgICAgICAgICAgICAgaWYgKCFzY2hlZHVsZWQgJiYgIXN0b3BwZWQpIHtcclxuICAgICAgICAgICAgICAgICAgICBzY2hlZHVsZWQgPSByZXF1ZXN0QW5pbWF0aW9uRnJhbWUoZG9SZW5kZXIpO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICBzdG9wOiBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgICAgICAgICBpZiAoc2NoZWR1bGVkKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgY2FuY2VsQW5pbWF0aW9uRnJhbWUoc2NoZWR1bGVkKTtcclxuICAgICAgICAgICAgICAgICAgICBzY2hlZHVsZWQgPSB1bmRlZmluZWQ7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICBzdG9wcGVkID0gdHJ1ZTtcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAgcmVzdW1lOiBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgICAgICAgICBzdG9wcGVkID0gZmFsc2U7XHJcbiAgICAgICAgICAgICAgICByZW5kZXJDb21wbGV0ZWQgPSB0cnVlO1xyXG4gICAgICAgICAgICAgICAgcHJvamVjdG9yLnNjaGVkdWxlUmVuZGVyKCk7XHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIGFwcGVuZDogZnVuY3Rpb24gKHBhcmVudE5vZGUsIHJlbmRlckZ1bmN0aW9uKSB7XHJcbiAgICAgICAgICAgICAgICBhZGRQcm9qZWN0aW9uKGRvbS5hcHBlbmQsIHBhcmVudE5vZGUsIHJlbmRlckZ1bmN0aW9uKTtcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAgaW5zZXJ0QmVmb3JlOiBmdW5jdGlvbiAoYmVmb3JlTm9kZSwgcmVuZGVyRnVuY3Rpb24pIHtcclxuICAgICAgICAgICAgICAgIGFkZFByb2plY3Rpb24oZG9tLmluc2VydEJlZm9yZSwgYmVmb3JlTm9kZSwgcmVuZGVyRnVuY3Rpb24pO1xyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICBtZXJnZTogZnVuY3Rpb24gKGRvbU5vZGUsIHJlbmRlckZ1bmN0aW9uKSB7XHJcbiAgICAgICAgICAgICAgICBhZGRQcm9qZWN0aW9uKGRvbS5tZXJnZSwgZG9tTm9kZSwgcmVuZGVyRnVuY3Rpb24pO1xyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICByZXBsYWNlOiBmdW5jdGlvbiAoZG9tTm9kZSwgcmVuZGVyRnVuY3Rpb24pIHtcclxuICAgICAgICAgICAgICAgIGFkZFByb2plY3Rpb24oZG9tLnJlcGxhY2UsIGRvbU5vZGUsIHJlbmRlckZ1bmN0aW9uKTtcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAgZGV0YWNoOiBmdW5jdGlvbiAocmVuZGVyRnVuY3Rpb24pIHtcclxuICAgICAgICAgICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgcmVuZGVyRnVuY3Rpb25zLmxlbmd0aDsgaSsrKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKHJlbmRlckZ1bmN0aW9uc1tpXSA9PT0gcmVuZGVyRnVuY3Rpb24pIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgcmVuZGVyRnVuY3Rpb25zLnNwbGljZShpLCAxKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgcmV0dXJuIHByb2plY3Rpb25zLnNwbGljZShpLCAxKVswXTtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ3JlbmRlckZ1bmN0aW9uIHdhcyBub3QgZm91bmQnKTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH07XHJcbiAgICAgICAgcmV0dXJuIHByb2plY3RvcjtcclxuICAgIH07XG5cbiAgICAvKipcclxuICAgICAqIENyZWF0ZXMgYSBbW0NhbGN1bGF0aW9uQ2FjaGVdXSBvYmplY3QsIHVzZWZ1bCBmb3IgY2FjaGluZyBbW1ZOb2RlXV0gdHJlZXMuXHJcbiAgICAgKiBJbiBwcmFjdGljZSwgY2FjaGluZyBvZiBbW1ZOb2RlXV0gdHJlZXMgaXMgbm90IG5lZWRlZCwgYmVjYXVzZSBhY2hpZXZpbmcgNjAgZnJhbWVzIHBlciBzZWNvbmQgaXMgYWxtb3N0IG5ldmVyIGEgcHJvYmxlbS5cclxuICAgICAqIEZvciBtb3JlIGluZm9ybWF0aW9uLCBzZWUgW1tDYWxjdWxhdGlvbkNhY2hlXV0uXHJcbiAgICAgKlxyXG4gICAgICogQHBhcmFtIDxSZXN1bHQ+IFRoZSB0eXBlIG9mIHRoZSB2YWx1ZSB0aGF0IGlzIGNhY2hlZC5cclxuICAgICAqL1xyXG4gICAgdmFyIGNyZWF0ZUNhY2hlID0gZnVuY3Rpb24gKCkge1xyXG4gICAgICAgIHZhciBjYWNoZWRJbnB1dHM7XHJcbiAgICAgICAgdmFyIGNhY2hlZE91dGNvbWU7XHJcbiAgICAgICAgcmV0dXJuIHtcclxuICAgICAgICAgICAgaW52YWxpZGF0ZTogZnVuY3Rpb24gKCkge1xyXG4gICAgICAgICAgICAgICAgY2FjaGVkT3V0Y29tZSA9IHVuZGVmaW5lZDtcclxuICAgICAgICAgICAgICAgIGNhY2hlZElucHV0cyA9IHVuZGVmaW5lZDtcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAgcmVzdWx0OiBmdW5jdGlvbiAoaW5wdXRzLCBjYWxjdWxhdGlvbikge1xyXG4gICAgICAgICAgICAgICAgaWYgKGNhY2hlZElucHV0cykge1xyXG4gICAgICAgICAgICAgICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgaW5wdXRzLmxlbmd0aDsgaSsrKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGlmIChjYWNoZWRJbnB1dHNbaV0gIT09IGlucHV0c1tpXSkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgY2FjaGVkT3V0Y29tZSA9IHVuZGVmaW5lZDtcclxuICAgICAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIGlmICghY2FjaGVkT3V0Y29tZSkge1xyXG4gICAgICAgICAgICAgICAgICAgIGNhY2hlZE91dGNvbWUgPSBjYWxjdWxhdGlvbigpO1xyXG4gICAgICAgICAgICAgICAgICAgIGNhY2hlZElucHV0cyA9IGlucHV0cztcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIHJldHVybiBjYWNoZWRPdXRjb21lO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfTtcclxuICAgIH07XG5cbiAgICAvKipcclxuICAgICAqIENyZWF0ZXMgYSB7QGxpbmsgTWFwcGluZ30gaW5zdGFuY2UgdGhhdCBrZWVwcyBhbiBhcnJheSBvZiByZXN1bHQgb2JqZWN0cyBzeW5jaHJvbml6ZWQgd2l0aCBhbiBhcnJheSBvZiBzb3VyY2Ugb2JqZWN0cy5cclxuICAgICAqIFNlZSB7QGxpbmsgaHR0cDovL21hcXVldHRlanMub3JnL2RvY3MvYXJyYXlzLmh0bWx8V29ya2luZyB3aXRoIGFycmF5c30uXHJcbiAgICAgKlxyXG4gICAgICogQHBhcmFtIDxTb3VyY2U+ICAgICAgIFRoZSB0eXBlIG9mIHNvdXJjZSBpdGVtcy4gQSBkYXRhYmFzZS1yZWNvcmQgZm9yIGluc3RhbmNlLlxyXG4gICAgICogQHBhcmFtIDxUYXJnZXQ+ICAgICAgIFRoZSB0eXBlIG9mIHRhcmdldCBpdGVtcy4gQSBbW01hcXVldHRlQ29tcG9uZW50XV0gZm9yIGluc3RhbmNlLlxyXG4gICAgICogQHBhcmFtIGdldFNvdXJjZUtleSAgIGBmdW5jdGlvbihzb3VyY2UpYCB0aGF0IG11c3QgcmV0dXJuIGEga2V5IHRvIGlkZW50aWZ5IGVhY2ggc291cmNlIG9iamVjdC4gVGhlIHJlc3VsdCBtdXN0IGVpdGhlciBiZSBhIHN0cmluZyBvciBhIG51bWJlci5cclxuICAgICAqIEBwYXJhbSBjcmVhdGVSZXN1bHQgICBgZnVuY3Rpb24oc291cmNlLCBpbmRleClgIHRoYXQgbXVzdCBjcmVhdGUgYSBuZXcgcmVzdWx0IG9iamVjdCBmcm9tIGEgZ2l2ZW4gc291cmNlLiBUaGlzIGZ1bmN0aW9uIGlzIGlkZW50aWNhbFxyXG4gICAgICogICAgICAgICAgICAgICAgICAgICAgIHRvIHRoZSBgY2FsbGJhY2tgIGFyZ3VtZW50IGluIGBBcnJheS5tYXAoY2FsbGJhY2spYC5cclxuICAgICAqIEBwYXJhbSB1cGRhdGVSZXN1bHQgICBgZnVuY3Rpb24oc291cmNlLCB0YXJnZXQsIGluZGV4KWAgdGhhdCB1cGRhdGVzIGEgcmVzdWx0IHRvIGFuIHVwZGF0ZWQgc291cmNlLlxyXG4gICAgICovXHJcbiAgICB2YXIgY3JlYXRlTWFwcGluZyA9IGZ1bmN0aW9uIChnZXRTb3VyY2VLZXksIGNyZWF0ZVJlc3VsdCwgdXBkYXRlUmVzdWx0KSB7XHJcbiAgICAgICAgdmFyIGtleXMgPSBbXTtcclxuICAgICAgICB2YXIgcmVzdWx0cyA9IFtdO1xyXG4gICAgICAgIHJldHVybiB7XHJcbiAgICAgICAgICAgIHJlc3VsdHM6IHJlc3VsdHMsXHJcbiAgICAgICAgICAgIG1hcDogZnVuY3Rpb24gKG5ld1NvdXJjZXMpIHtcclxuICAgICAgICAgICAgICAgIHZhciBuZXdLZXlzID0gbmV3U291cmNlcy5tYXAoZ2V0U291cmNlS2V5KTtcclxuICAgICAgICAgICAgICAgIHZhciBvbGRUYXJnZXRzID0gcmVzdWx0cy5zbGljZSgpO1xyXG4gICAgICAgICAgICAgICAgdmFyIG9sZEluZGV4ID0gMDtcclxuICAgICAgICAgICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgbmV3U291cmNlcy5sZW5ndGg7IGkrKykge1xyXG4gICAgICAgICAgICAgICAgICAgIHZhciBzb3VyY2UgPSBuZXdTb3VyY2VzW2ldO1xyXG4gICAgICAgICAgICAgICAgICAgIHZhciBzb3VyY2VLZXkgPSBuZXdLZXlzW2ldO1xyXG4gICAgICAgICAgICAgICAgICAgIGlmIChzb3VyY2VLZXkgPT09IGtleXNbb2xkSW5kZXhdKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHJlc3VsdHNbaV0gPSBvbGRUYXJnZXRzW29sZEluZGV4XTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgdXBkYXRlUmVzdWx0KHNvdXJjZSwgb2xkVGFyZ2V0c1tvbGRJbmRleF0sIGkpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBvbGRJbmRleCsrO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgdmFyIGZvdW5kID0gZmFsc2U7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGZvciAodmFyIGogPSAxOyBqIDwga2V5cy5sZW5ndGggKyAxOyBqKyspIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHZhciBzZWFyY2hJbmRleCA9IChvbGRJbmRleCArIGopICUga2V5cy5sZW5ndGg7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAoa2V5c1tzZWFyY2hJbmRleF0gPT09IHNvdXJjZUtleSkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHJlc3VsdHNbaV0gPSBvbGRUYXJnZXRzW3NlYXJjaEluZGV4XTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB1cGRhdGVSZXN1bHQobmV3U291cmNlc1tpXSwgb2xkVGFyZ2V0c1tzZWFyY2hJbmRleF0sIGkpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG9sZEluZGV4ID0gc2VhcmNoSW5kZXggKyAxO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGZvdW5kID0gdHJ1ZTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBicmVhaztcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgICAgICBpZiAoIWZvdW5kKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICByZXN1bHRzW2ldID0gY3JlYXRlUmVzdWx0KHNvdXJjZSwgaSk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICByZXN1bHRzLmxlbmd0aCA9IG5ld1NvdXJjZXMubGVuZ3RoO1xyXG4gICAgICAgICAgICAgICAga2V5cyA9IG5ld0tleXM7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9O1xyXG4gICAgfTtcblxuICAgIGV4cG9ydHMuY3JlYXRlQ2FjaGUgPSBjcmVhdGVDYWNoZTtcbiAgICBleHBvcnRzLmNyZWF0ZU1hcHBpbmcgPSBjcmVhdGVNYXBwaW5nO1xuICAgIGV4cG9ydHMuY3JlYXRlUHJvamVjdG9yID0gY3JlYXRlUHJvamVjdG9yO1xuICAgIGV4cG9ydHMuZG9tID0gZG9tO1xuICAgIGV4cG9ydHMuaCA9IGg7XG5cbiAgICBPYmplY3QuZGVmaW5lUHJvcGVydHkoZXhwb3J0cywgJ19fZXNNb2R1bGUnLCB7IHZhbHVlOiB0cnVlIH0pO1xuXG59KSk7XG4iXSwic291cmNlUm9vdCI6IiJ9