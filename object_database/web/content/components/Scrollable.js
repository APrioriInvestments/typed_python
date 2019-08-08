/**
 * Scrollable  Component
 */

import {Component} from './Component';
import {PropTypes} from './util/PropertyValidator';
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
        let style = "";
        if (this.props.extraData.height){
            style = "height:" + this.props.extraData.height;
        }
        return (
            h('div', {
                id: this.props.id,
                class: "cell overflow",
                style: style,
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

Scrollable.propTypes = {
    height: {
        height: "Height of the Scrollable container.",
        type: PropTypes.oneOf([PropTypes.string])
    }
};

export {Scrollable, Scrollable as default};
