/**
 * Sequence Cell Component
 */

//import {Component} from './Component';
//import {h} from 'maquette';

/**
 * About Replacements
 * ------------------
 * Sequence has the following enumerated
 * replacement:
 * * `c`
 */
class Sequence extends Component {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return (
            h('div', {
                id: this.props.id,
                class: "cell",
                "data-cell-id": this.props.id,
                "data-cell-type": "Sequence",
                style: this.props.extraData.divStyle
            }, this.getReplacementElementsFor('c'))
        );
    }
}

//export {Sequence, Sequence as default};
