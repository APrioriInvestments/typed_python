/**
 * HorizontalSequence Cell Components
 */

import {Component, render} from './Component';
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

    build(){
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
            //return this.renderChildrenNamed('elements');
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
        let classes = ["cell", "sequence", "sequence-horizontal"];
        if(this.props.flexParent){
            classes.push("flex-parent");
        }
        if (this.props.margin){
            classes.push(`child-margin-${this.props.margin}`);
        }
        if(this.props.wrap){
            classes.push('flex-wrap');
        }
        return classes.join(" ");
    }
}

HorizontalSequence.propTypes = {
    margin: {
        description: "Bootstrap margin value for between element spacing",
        type: PropTypes.oneOf([PropTypes.number, PropTypes.string])
    },
    flexParent: {
        description: "Whether or not the HorizontalSequence should display using Flexbox",
        type: PropTypes.boolean
    },
    wrap: {
        description: "Whether or not the HorizontalSequence should wrap on overflow",
        type: PropTypes.boolean
    }
};

export {HorizontalSequence, HorizontalSequence as default};
