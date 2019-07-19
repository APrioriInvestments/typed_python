/**
 * HorizontalSequence Cell Components
 */

import {Component} from './Component';
import {PropTypes} from './util/PropertyValidator';
import {h} from 'maquette';

/**
 * About Replacements
 * ------------------
 * This component has the following
 * enumerated replacement:
 * * `c`
 */

/**
 * About Named Children
 * --------------------
 * `elements` (array) - A list of Cells that are in the
 *    sequence
 */
class HorizontalSequence extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeClasses = this.makeClasses.bind(this);
        this.makeElements = this.makeElements.bind(this);
    }

    render(){
        return (
            h('div', {
                id: this.props.id,
                class: this.makeClasses(),
                'data-cell-id': this.props.id,
                'data-cell-type': "HorizontalSequence"
            }, this.makeElements())
        );
    }

    makeElements(){
        if(this.usesReplacements){
            return this.getReplacementElementsFor('c');
        } else {
            return this.renderChildrenNamed('elements');
        }
    }

    makeClasses(){
        let classes = ["cell", "sequence", "sequence-horizonal"];
        if(this.props.overflow){
            classes.push("overflow");
        }
        return classes.join(" ");
    }
}

HorizontalSequence.propTypes = {
    overflow: {
        description: "If true, sets overflow of container to auto",
        type: PropTypes.boolean
    }
};

export {HorizontalSequence, HorizontalSequence as default};
