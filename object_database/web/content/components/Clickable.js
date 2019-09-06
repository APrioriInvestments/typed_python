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
        this.getStyle = this.getStyle.bind(this);
        this._getEvent = this._getEvent.bind(this);
    }

    build(){
        return(
            h('div', {
                id: this.props.id,
                class: "cell clickable",
                "data-cell-id": this.props.id,
                "data-cell-type": "Clickable",
                onclick: this._getEvent('onclick'),
                style: this.getStyle()
            }, [
                h('div', {}, [this.makeContent()])
            ]
            )
        );
    }

    getStyle(){
        if(this.props.extraData.bold){
            return "cursor:pointer;*cursor:hand;font-weight:bold;";
        } else {
            return "";
        }
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
