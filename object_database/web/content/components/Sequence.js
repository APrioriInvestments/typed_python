/**
 * Sequence Cell Component
 */

import {Component} from './Component';
import {PropTypes} from './util/PropertyValidator';
import {h} from 'maquette';

/**
 * About Replacements
 * ------------------
 * Sequence has the following enumerated
 * replacement:
 * * `c`
 */

/**
 * About Named Children
 * --------------------
 * `elements` (array) - A list of Cells that are in the
 *    sequence.
 */
class Sequence extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        //this.makeStyle = this.makeStyle.bind(this);
        this.makeClasses = this.makeClasses.bind(this);
        this.makeElements = this.makeElements.bind(this);
    }

    render(){
        return (
            h('div', {
                id: this.props.id,
                class: this.makeClasses(),
                "data-cell-id": this.props.id,
                "data-cell-type": "Sequence",
                //style: this.makeStyle()
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
        let classes = ["cell sequence sequence-vertical"];
        if(this.props.overflow){
            classes.push("overflow");
        }
        if (this.props.margin){
            classes.push(`child-margin-${this.props.margin}`);
        }
        return classes.join(" ");
    }

    /*makeStyle(){
        // Note: the server side uses "split" (axis) to denote the direction
        let direction = "row";
        if (this.props.split == "horizontal"){
            direction = "column";
        }
        let overflow = ""
        if (this.props.overflow) {
            overflow = "overflow:auto"
        }
        return `width:100%;height:100%;display:inline-flex;flex-direction:${direction};${overflow}`;
        }*/
}

Sequence.propTypes = {
    split: {
        description: "Horizontal/vertical layout of the children.",
        type: PropTypes.string
    },
    overflow: {
        description: "Overflow-auto.",
        type: PropTypes.boolean
    },
    margin: {
        description: "Bootstrap margin value for between element spacing",
        type: PropTypes.oneOf([PropTypes.number, PropTypes.string])
    }
};

export {Sequence, Sequence as default};
