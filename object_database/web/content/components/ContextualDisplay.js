/**
 * ContextualDisplay Cell Component
 */

import {Component} from './Component';
import {h} from 'maquette';

/**
 * About Replacements
 * ------------------
 * This component has a single
 * regular replacement:
 * * `child`
 */
class ContextualDisplay extends Component {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return h('div',
            {
                class: "cell contextualDisplay",
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "ContextualDisplay"
            }, [this.getReplacementElementFor('child')]
        );
    }
}

export {ContextualDisplay, ContextualDisplay as default};
