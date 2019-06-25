/**
 * RootCell Cell Component
 */

import {Component} from './Component';
import {h} from 'maquette';

/**
 * About Replacements
 * --------------------
 * This component has a one
 * regular-kind replacement:
 * * `c`
 */

/**
 * About Named Children
 * --------------------
 * `child` (single) - The child cell this container contains
 */
class RootCell extends Component {
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
                "data-cell-type": "RootCell"
            }, [this.makeChild()])
        );
    }

    makeChild(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('c');
        } else {
            return this.renderChildNamed('child');
        }
    }
}

export {RootCell, RootCell as default};
