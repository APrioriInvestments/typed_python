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
class CardTitle extends Component {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return(
            h('div', {
                id: this.props.id,
                class: "cell",
                "data-cell-id": this.props.id,
                "data-cell-type": "CardTitle"
            }, [
                this.getReplacementElementFor('contents')
            ])
        );
    }
}

export {CardTitle, CardTitle as default};
