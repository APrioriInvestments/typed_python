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
class Subscribed extends Component {
    constructor(props, ...args){
        super(props, ...args);
        //this.addReplacement('contents', '_____contents__');
    }

    render(){
        return h('div',
            {
                class: "cell subscribed",
                style: this.props.extraData.divStyle,
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Subscribed"
            }, [this.getReplacementElementFor('contents')]
        );
    }
}

export {Subscribed, Subscribed as default};
