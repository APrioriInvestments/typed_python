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
class ButtonGroup extends Component {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return(
            h('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "ButtonGroup",
                class: "btn-group",
                "role": "group"
            }, this.getReplacementElementsFor('button')
             )
        );
    }

}

export {ButtonGroup, ButtonGroup as default};
