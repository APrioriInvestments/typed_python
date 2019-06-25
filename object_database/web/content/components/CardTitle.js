/**
 * CardTitle Cell
 */

import {Component} from './Component';
import {h} from 'maquette';


/**
 * About Replacements
 * ------------------
 * This component has  single regular
 * replacement:
 * * `contents`
 */

/**
 * About Named Children
 * --------------------
 * `inner` (single) - The inner cell of the title component
 */
class CardTitle extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeInner = this.makeInner.bind(this);
    }

    render(){
        return(
            h('div', {
                id: this.props.id,
                class: "cell",
                "data-cell-id": this.props.id,
                "data-cell-type": "CardTitle"
            }, [
                this.makeInner()
            ])
        );
    }

    makeInner(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('contents');
        } else {
            return this.renderChildNamed('inner');
        }
    }
}

export {CardTitle, CardTitle as default};
