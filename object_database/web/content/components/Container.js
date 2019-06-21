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
class Container extends Component {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        let child = this.getReplacementElementFor('child');
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
}

export {Container, Container as default};
