/**
 * Expands Cell Component
 */

//import {Component} from './Component';
//import {h} from 'maquette';


/**
 * About Replacements
 * ------------------
 * This component has two
 * regular replacements:
 * * `icon`
 * * `child`
 */
class Expands extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
        this._getEvents = this._getEvent.bind(this);
    }

    render(){
        return(
            h('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Expands",
                style: this.props.extraData.divStyle
            },
                [
                    h('div', {
                        style: 'display:inline-block;vertical-align:top',
                        onclick: this._getEvent('onclick')
                    },
                        [this.getReplacementElementFor('icon')]),
                    h('div', {style:'display:inline-block'},
                        [this.getReplacementElementFor('child')]),
                ]
            )
        );
    }

    _getEvent(eventName) {
        return this.props.extraData.events[eventName];
    }
}

//export {Expands, Expands as default};
