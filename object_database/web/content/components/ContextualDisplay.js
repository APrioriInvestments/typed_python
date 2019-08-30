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

/**
 * About Named Children
 * --------------------
 * `child` (single) - A child cell to display in a context
 */
class ContextualDisplay extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeChild = this.makeChild.bind(this);
    }

    build(){
        return h('div',
            {
                class: "cell contextualDisplay",
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "ContextualDisplay"
            }, [this.makeChild()]
        );
    }

    makeChild(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('child');
        } else {
            return this.renderChildNamed('child');
        }
    }
}

export {ContextualDisplay, ContextualDisplay as default};
