/**
 * Clickable Cell Component
 */
import {Component} from './Component';
import {h} from 'maquette';

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
class Clickable extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
        this.makeContent = this.makeContent.bind(this);
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
                h('div', {}, [this.makeContent()])
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

export {Clickable, Clickable as default};
