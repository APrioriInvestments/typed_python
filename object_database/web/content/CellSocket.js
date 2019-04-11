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

        if(update == 'postscripts'){
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
