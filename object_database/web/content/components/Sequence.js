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
                "data-cell-type": "Sequence",
                //style: this.makeStyle()
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
                    console.log(`Sequence[${this.props.id}] has child [${childComponent.props.id}] that is specified as a flexChild!`);
                    hyperscript.properties.class += " flex-child";
                    console.log(hyperscript);
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
        if(this.props.overflow){
            classes.push("overflow");
        }
        if (this.props.margin){
            classes.push(`child-margin-${this.props.margin}`);
        }
        return classes.join(" ");
    }

}

Sequence.propTypes = {
    overflow: {
        description: "Overflow-auto.",
        type: PropTypes.boolean
    },
    margin: {
        description: "Bootstrap margin value for between element spacing",
        type: PropTypes.oneOf([PropTypes.number, PropTypes.string])
    },
    flexParent: {
        description: "Whether or not the Sequence should display using Flexbox",
        type: PropTypes.boolean
    }
};

export {Sequence, Sequence as default};
