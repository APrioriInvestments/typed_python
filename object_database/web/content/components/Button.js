/**
 * Button Cell Component
 */
//import {Component} from './Component';


class Button extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
        this._getEvents = this._getEvent.bind(this);
        this._getHTMLClasses = this._getHTMLClasses.bind(this);
    }

    render(){
        return(
            h('button', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Button",
                class: this._getHTMLClasses(),
                onclick: this._getEvent('onclick')
            }, [this.getReplacementElementFor('contents')]
            ) 
        );
    }

    _getEvent(event_name) {
        return this.props.extraData.events[event_name];
    }

    _getHTMLClasses(){
        let classString = this.props.extraData.classes.join(" ");
        // remember to trim the class string due to a maquette bug
        return classString.trim()
    }
}

//export {Button, Button as default};
