/**
 * Clickable Cell Component
 */
//import {Component} from './Component';


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

    _getEvent(event_name) {
        return this.props.extraData.events[event_name];
    }
}

//export {Clickable, Clickable as default};
