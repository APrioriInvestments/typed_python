/**
 * Scrollable  Component
 */

//import {Component} from './Component';
//import {h} from 'maquette';

/**
 * About Replacements
 * --------------------
 * This component has a one
 * regular-kind replacement:
 * * `child`
 */
class Scrollable extends Component {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return (
            h('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Scrollable"
            }, [this.getReplacementElementFor('child')])
        );
    }
}

//export {Scrollable, Scrollable as default};
