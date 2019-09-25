/**
 * Traceback Cell Component
 */

import {Component} from './Component';
import {h} from 'maquette';

/**
 * About Named Children
 * `traceback` (single) - The cell containing the traceback text
 */
class  Traceback extends Component {
    constructor(props, ...args){
        super(props, ...args);
    }

    build(){
        return (
            h('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Traceback",
                class: "alert alert-primary traceback"
            }, [this.renderChildNamed('traceback')])
        );
    }
}


export {Traceback, Traceback as default};
