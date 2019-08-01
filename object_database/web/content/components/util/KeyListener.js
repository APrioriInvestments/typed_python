/**
 * A Global Key Event Handler
 * and registry.
 * --------------------------
 * This module represents a set of classes whose
 * instances will comprise a global keypress
 * event registry.
 * `KeyListener` is the main combination and event
 * binding mechanism.
 * `KeyBinding` is a specific key combination plus
 * listener that will be managed by `KeyListener`
 * Here we can register different key combinations,
 * like 'Alt+i' 'Ctrl+x' 'Meta+D' etc.
 * `KeyBinding` listeners can be stored with
 * an optional priority level, and those listeners
 * will be fired before all others.
 * See class comments for `KeyListener` and
 * `KeyBinding` for more information.
 */

/**
 * A simple mapping from common
 * modifier key strings to the
 * respective keys on a keyup/down
 * event object.
 */
const modKeyMap = {
    'Shift': 'shiftKey',
    'Alt': 'altKey',
    'Meta': 'metaKey',
    'Control': 'ctrlKey',
    'Ctrl': 'ctrlKey'
};


/**
 * A class whose instances manage keypress
 * event listeners on the global window object.
 * Note that particular keycombo/listener combinations
 * are stored internally as instances of `KeyBinding`.
 * This class should be initialized only once on a given
 * page.
 */
class KeyListener {
    /**
     * Creates a new `KeyListener` instance.
     * @param {DOMElement} target - The target element
     * to which we will bind all key event listeners.
     * defaults to `window.document.body`
     * @param {CellSocket} cellSocket - An instance
     * of `CellSocket` so that we can create socket
     * event listeners. By default, we send the event data
     * over the `cellSocket` when a key event happens.
     */
    constructor(target, cellSocket){
        this._target = target | window.document.body;
        this.socket = cellSocket;
        this.listenersByPriority = {
            '1': [],
            '2': [],
            '3': [],
            '4': []
        };
        this.listenersByCellId = {};

        // Bind methods
        this.start = this.start.bind(this);
        this.pause = this.pause.bind(this);
        this.mainListener = this.mainListener.bind(this);
        this.privRegister = this.privRegister.bind(this);
        this.register = this.register.bind(this);
        this.deregister = this.deregister.bind(this);
        this.makeListenerFor = this.makeListenerFor.bind(this);
    }

    /**
     * Tells the instance to begin listening for keydown events.
     * This method will bind the instance's single main listener,
     * `mainListener` to the specified target object, either the
     * passed in value or the stored internal one from construction.
     * The idea here is that we can easily stop and start all of the
     * constituent listeners simply by adding / removing this single
     * listener.
     * @param {DOMElement} target - A target DOMElement to which
     * we will bind the global keydown listener. If not passed, will
     * use the target specified at instantiation.
     * @param {CellSocket} socket - A `CellSocket` instance that
     * will handle all constituent listener responses. Will use
     * the internal value from instantiation if nothing is passed.
     */
    start(target, socket){
        console.log('Starting global KeyListener');
        if(target){
            this._target = target;
        }
        if(socket){
            this.socket = socket;
        }
        this._target.addEventListener('keydown', this.mainListener);
    }

    /**
     * Stops global listening for keydown events.
     * In practice, this method simply removes the single
     * listener from the target DOMElement.
     * To resume, one can simply call `start()` again without
     * arguments.
     */
    pause(){
        console.log('keyListener paused');
        this._target.removeEventListener('keydown', this.mainListener);
    }

    /**
     * The main keydown event listener that is attached
     * to the target DOMElement when `start()` is called.
     * Cycles through the different listeners by priority level,
     * in order from 1 to 4. If at any point a keybinging
     * wants to stop propagation, this listener will return
     * from itself and stop calling the rest of the listeners.
     * @param {KeyEvent} event - A keydown event object.
     */
    mainListener(event){
        for(let i = 1; i < 5; i++){
            let level = i.toString();
            let bindings = this.listenersByPriority[level];
            for(var j = 0; j < bindings.length; j++){
                let currentBinding = bindings[j];
                let shouldStop = currentBinding.handle(event);
                if(shouldStop){
                    return;
                }
            }
        }
    }

    /**
     * Private method that will create a new `KeyBinding` and add
     * it to the appropriate priority level along with a function/listener
     * and other information.
     * @param {String} command - A key combination command like `Alt+i` or just `I`.
     * @param {Function} listener - A function that will be called as the listener
     * for this command.
     * @param {String|Number} cellId - The ID of the Cell object that is handling
     * this particular key combination.
     * @param {Number} priority - the priority level for the handler. Can be in range
     * 1 to 4.
     * @param {Boolean} stopsPropagation - Whether or not the listener, once triggered,
     * should stop further listeners of this key combination from firing.
     */
    privRegister(command, listener, cellId, priority=4, stopsPropagation=false){
        let binding = new KeyBinding(
            command,
            listener,
            priority,
            stopsPropagation
        );
        if(this.listenersByCellId[cellId]){
            this.listenersByCellId[cellId].push(binding);
        } else {
            this.listenersByCellId[cellId] = [binding];
        }
        this.listenersByPriority[priority].push(binding);
    }

    /**
     * Public method for registering a new kebinding and listener.
     * Uses `makeListener` internally to make a function that will
     * make an appropriate socket call with the `desiredInfo` attached.
     * Uses `privRegister` internally to set up priority levels etc.
     * @param {String} command - A key combination command like `Alt+i` or
     * `Meta+D` etc
     * @param {[String]} desiredInfo - An array of keys on the keydown event
     * object whose values should be sent back over the socket with the response
     * event data.
     * @param {String|Number} cellId - The ID of the Cell that owns this particular
     * key listener.
     * @param {Number} priority - The priority level of the listener. Can be in range
     * 1 to 4.
     * @param {Boolean} stopsPropagation - Whether or not the listener, once triggered,
     * should stop all other listeners on this command from also firing.
     */
    register(command, desiredInfo, cellId, priority=4, stopsPropagation=false){
        let listener = this.makeListenerFor(cellId, desiredInfo);
        this.privRegister(command, listener, cellId, priority, stopsPropagation);
    }

    /**
     * Removes all listeners from the registry
     * that match the provided Cell ID.
     * @param {String|Number} cellId - The ID of the cell whose
     * listener should be removed from the registry.
     */
    deregister(cellId){
        let found = this.listenersByCellId[cellId];
        if(found){
            Object.keys(this.listenersByPriority).forEach(priorityLevel => {
                let current = this.listenersByPriority[priorityLevel];
                let updated = current.filter(item => {
                    return !found.includes(item);
                });
                this.listenersByPriority[priorityLevel] = current;
            });
            delete this.listenersByCellId[cellId];
        }
    }

    /**
     * Creates a listener function that will send a correctly
     * formatted response over the `CellSocket`.
     * @param {String|Number} cellId - The ID of the Cell that owns
     * the listener function and keybinding
     * @param {[String]} desiredData - An array of string/keys on
     * the keydown event object whose values should be included
     * in the data sent back over the socket.
     */
    makeListenerFor(cellId, desiredData){
        if(this.socket){
            let currentSocket = this.socket;
            return function(event){
                let responseData = {};
                desiredData.forEach(key => {
                    responseData[key] = event[key];
                });

                currentSocket.sendString(JSON.stringify({
                    event: 'keypress',
                    target_cell: cellId,
                    data: responseData
                }));
            };
        } else {
            return function(event){
                console.warn(`Default keyhandler for Cell ${cellId}`);
            };
        }
    }

}

/**
 * A class whose instances represent a combination of key commands
 * (like `Alt+i`, `I`, `Meta+D` etc), listeners for keydown, priority
 * level, and whether or not the binding should stop other bindings with
 * the same command from being triggered.
 * It's primary consumer is `KeyListener`, whose instances register
 * all key events as `KeyBinding` objects.
 */
class KeyBinding {
    /**
      * @param {String} command - A key combo command string like `Alt+i`,
      * `X`, `Meta+D`, etc.
      * @param {Function} listener - A function that serves as the event
      * listener, which will be triggered when the command is pressed.
      * Will be passed the normal keydown event object.
      * @param {Number} priority - The priority level of the binding. Can
      * be in range 1 through 4.
      * @param {Boolean} stopsPropagation - Whether or not this keybinding, once
      * its listener has fired, should stop other keybindings with the same
      * command from firing.
      */
    constructor(command, listener, priority=4, stopsPropagation=false){
        this.command = command;
        this.listener = listener;
        this.priority = priority;
        this.stopsPropagation = stopsPropagation;
        this.commandKeys = this.command.split("+");

        // Bind instance methods
        this.handle = this.handle.bind(this);
        this.handleSingleKey = this.handleSingleKey.bind(this);
        this.handleComboKey = this.handleComboKey.bind(this);
    }

    /**
     * For a given keydown event, attempt to "handle" it
     * by calling this object's listener.
     * Note that if there is only one command key (ie the command
     * is a single character without a modifier key like `X`)
     * we call `handleSingleKey`. Otherwise calls `handleComboKey`.
     * @param {KeyEvent} event - A keydown event object
     * @returns {Boolean} - Will return true if the keybinding
     * both has its listener called and requires that propagation
     * stops. false in all other cases.
     */
    handle(event){
        if(this.commandKeys.length == 0){
            return false;
        } else if(this.commandKeys.length == 1){
            return this.handleSingleKey(event, this.commandKeys[0]);
        } else {
            return this.handleComboKey(event);
        }
    }

    /**
     * Determines if this KeyBinding's
     * single trigger key matches that of the
     * passed in event. If so, that means we have
     * a match and the listener should fire.
     * @param {KeyEvent} event - A keydown event
     * object that we will check for a match with.
     * @param {String} keyName - The name of this
     * instance's first `commandKey` value.
     * @returns {Boolean} - If there is a match,
     * we return true if this instance asks to
     * stop propagation. Returns false in all
     * other cases.
     */
    handleSingleKey(event, keyName){
        if(event.key == keyName){
            this.listener(event);
            return this.stopsPropagation;
        } else {
            return false;
        }
    }

    /**
     * Attempts to "handle" (ie call listener for)
     * cases where this instance uses a modifier key
     * and has a combo command like `Alt+i` or `Meta+D`.
     * Will attempts to match the internal `commandKey`
     * parts to the passed-in event object and, if there
     * is a match, will call the listener.
     * @param {KeyEvent} event - A keydown event object
     * @returns {Boolean} - Will return true only if
     * this binding is a match to the event and it is
     * also asking to stop propagation. False in all
     * other cases.
     */
    handleComboKey(event){
        let mappedModifier = modKeyMap[this.commandKeys[0]];
        let modKeyDown = event[mappedModifier];
        if(modKeyDown){
            return this.handleSingleKey(event, this.commandKeys[1]);
        } else {
            return false;
        }
    }
}

export {KeyListener, KeyListener as default};
