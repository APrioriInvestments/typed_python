/**
 * KeyAction Cell component
 * ------------------------
 * This component matches the unique
 * non-display Cell KeyAction, which
 * uses the Component class level
 * KeyListener object to globally
 * register certain key combinations
 * that should trigger at the global
 * level.
 */
import {Component} from './Component';

class KeyAction extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.registerKeyAction = this.registerKeyAction.bind(this);
    }

    componentDidLoad(){
        this.registerKeyAction();
    }

    render(){
        // This is a non-display cell
        // and does not add any elements
        // to the DOM.
        return null;
    }

    registerKeyAction(){
        if(this.constructor.keyListener){
            this.constructor.keyListener.register(
                this.props.extraData['keyCombo'],
                this.props.extraData['wantedEventKeys'],
                this.props.id,
                this.props.extraData['priority'],
                this.props.extraData['stopsPropagation']
            );
        } else {
            throw new Error(`KeyAction(${this.props.id}) attempted to register with the KeyListener but there was no contstructor instance found!`);
        }
    }
}

export {KeyAction, KeyAction as default};
