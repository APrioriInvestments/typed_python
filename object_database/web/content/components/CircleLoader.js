/**
 * CircleLoader Cell Component
 */

//import {Component} from './Component';
//import {h} from 'maquette';


class CircleLoader extends Component {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return (
            h('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "CircleLoader",
                class: "spinner-grow",
                role: "status"
            })
        );
    }
}

//export {CircleLoader, CircleLoader as default};
