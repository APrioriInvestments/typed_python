/**
 * Clickable Cell Component
 */
//import {Component} from './Component';
//import {h} from 'maquette';

/**
 * About Replacements
 * ------------------
 * This component has a single
 * regular replacement:
 * * `contents`
 */
class Clickable extends Component {
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
                "data-cell-type": "Clickable",
                onclick: this._getEvent('onclick'),
                style: this.props.extraData.divStyle
            }, [
                h('div', {}, [this.getReplacementElementFor('contents')])
            ]
            )
        );
    }

    _getEvent(eventName) {
        return this.props.extraData.events[eventName];
    }
}

//export {Clickable, Clickable as default};
