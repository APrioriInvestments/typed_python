/**
 * Main Cell Component
 */

import {Component} from './Component';
import {h} from 'maquette';

/**
 * About Replacements
 * --------------------
 * This component has a one
 * regular-kind replacement:
 * * `child`
 */
class Main extends Component {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return (
            h('main', {
                id: this.props.id,
                class: "py-md-2",
                "data-cell-id": this.props.id,
                "data-cell-type": "Main"
            }, [
                h('div', {class: "container-fluid"}, [
                    this.getReplacementElementFor('child')
                ])
            ])
        );
    }
}

export {Main, Main as default};
