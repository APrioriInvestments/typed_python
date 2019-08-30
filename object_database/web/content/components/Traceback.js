/**
 * Traceback Cell Component
 */

import {Component} from './Component';
import {h} from 'maquette';

/**
 * About Replacements
 * ------------------
 * This component has a single regular
 * repalcement:
 * * `child`
 */

/**
 * About Named Children
 * `traceback` (single) - The cell containing the traceback text
 */
class  Traceback extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeTraceback = this.makeTraceback.bind(this);
    }

    build(){
        return (
            h('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Traceback",
                class: "alert alert-primary"
            }, [this.getReplacementElementFor('child')])
        );
    }

    makeTraceback(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('child');
        } else {
            return this.renderChildNamed('traceback');
        }
    }
}


export {Traceback, Traceback as default};
