/**
 * Panel Cell Component
 */

import {Component} from './Component';
import {h} from 'maquette';

/**
 * About Replacements
 * ------------------
 * This component has one regular replacement:
 * `content`
 */

/**
 * About Named Children
 * --------------------
 * `content` (single) - The cell inside of the Panel
 */
class Panel extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.getClasses = this.getClasses.bind(this);
    }

    build(){
        return(
            h('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Panel",
                class: this.getClasses()
            }, [this.renderChildNamed('content')])
        );
    }

    getClasses(){
        let classes = ["cell", "cell-panel"];
        return classes.join(" ");
    }
}
