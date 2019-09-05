/**
 * Sequence Cell Component
 */

import {Component, render} from './Component';
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

    build(){
        return (
            h('div', {
                id: this.props.id,
                class: this.makeClasses(),
                "data-cell-id": this.props.id,
                "data-cell-type": "Sequence"
            }, this.makeElements())
        );
    }

    makeElements(){
        if(this.usesReplacements){
            return this.getReplacementElementsFor('c');
        } else {
            let elements = this.props.namedChildren['elements'];
            return elements.map(childComponent => {
                let hyperscript = render(childComponent);
                if(childComponent.props.flexChild == true && this.props.flexParent){
                    hyperscript.properties.class += " flex-child";
                }
                return hyperscript;
            });
        }
    }

    makeClasses(){
        let classes = ["cell sequence sequence-vertical"];
        if(this.props.flexParent){
            classes.push("flex-parent");
        }
        if (this.props.margin){
            classes.push(`child-margin-${this.props.margin}`);
        }
        return classes.join(" ");
    }

}

Sequence.propTypes = {
    margin: {
        description: "Bootstrap margin value for between element spacing",
        type: PropTypes.oneOf([PropTypes.number, PropTypes.string])
    },
    flexParent: {
        description: "Whether or not the Sequence should display using Flexbox",
        type: PropTypes.boolean
    },
    flexChild: {
        description: "Whether or not this Sequence is a flexChild of some flexParent",
        type: PropTypes.boolean
    }
};

export {Sequence, Sequence as default};
