/**
 * Scrollable  Component
 */

import {Component} from './Component';
import {h} from 'maquette';

/**
 * About Replacements
 * --------------------
 * This component has a one
 * regular-kind replacement:
 * * `child`
 */

/**
 * About Named Children
 * --------------------
 * `child` (single) - The cell/component this instance contains
 */
class Scrollable extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeChild = this.makeChild.bind(this);
    }

    render(){
        return (
            h('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Scrollable"
            }, [this.makeChild()])
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

export {Scrollable, Scrollable as default};
