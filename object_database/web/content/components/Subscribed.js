/**
 * Subscribed Cell Component
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
 * `content` (single) - The underlying Cell that is subscribed
 */
class Subscribed extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeContent = this.makeContent.bind(this);
    }

    render(){
        return h('div',
            {
                class: "cell subscribed",
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Subscribed"
            }, [this.makeContent()]
        );
    }

    makeContent(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('contents');
        } else {
            return this.renderChildNamed('content');
        }
    }
}

export {Subscribed, Subscribed as default};
