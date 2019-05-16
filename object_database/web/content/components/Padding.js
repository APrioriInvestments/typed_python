/**
 * Padding Cell Component
 */

//import {Component} from './Component';
//import {h} from 'maquette';

class Padding extends Component {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return (
            h('span', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Padding",
                class: "px-2"
            }, [" "])
        );
    }
}

//export {Padding, Padding as default};
