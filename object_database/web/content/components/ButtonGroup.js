/**
 * ButtonGroup Cell Component
 */

import {Component} from './Component';
import {h} from 'maquette';

/**
 * About Replacements
 * ------------------
 * This component has a single enumerated
 * replacement:
 * * `button`
 */

/**
 * About Named Children
 * --------------------
 * `buttons` (array) - The constituent button cells
 */
class ButtonGroup extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeButtons = this.makeButtons.bind(this);
    }

    build(){
        return(
            h('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "ButtonGroup",
                class: "btn-group",
                "role": "group"
            }, this.makeButtons()
             )
        );
    }

    makeButtons(){
        if(this.usesReplacements){
            return this.getReplacementElementsFor('button');
        } else {
            return this.renderChildrenNamed('buttons');
        }
    }

}

export {ButtonGroup, ButtonGroup as default};
