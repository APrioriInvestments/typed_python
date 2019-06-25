/**
 * Container Cell Component
 */

import {Component} from './Component';
import {h} from 'maquette';

/**
 * About Replacements
 * ------------------
 * This component has a single regular
 * replacement:
 * * `child`
 */

/**
 * About Named Children
 * --------------------
 * `child` (single) - The Cell that this component contains
 */
class Container extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeChild = this.makeChild.bind(this);
    }

    render(){
        let child = this.makeChild();
        let style = "";
        if(!child){
            style = "display:none;";
        }
        return (
            h('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Container",
                class: "cell",
                style: style
            }, [child])
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

export {Container, Container as default};
