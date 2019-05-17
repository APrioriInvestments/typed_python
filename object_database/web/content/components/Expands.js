/**
 * Expands Cell Component
 */
//import {Component} from './Component';


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

    _getEvent(event_name) {
        return this.props.extraData.events[event_name];
    }
}

//export {Expands, Expands as default};
