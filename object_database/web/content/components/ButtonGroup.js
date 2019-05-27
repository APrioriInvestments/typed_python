/**
 * ButtonGroup Cell Component
 */

//import {Component} from './Component';
//import {h} from 'maquette';

class ButtonGroup extends Component {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return(
            h('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "ButtonGroup",
                class: "btn-group",
                "role": "group"
            }, [h("div", {}, [this.props.extraData.innerButtonsText])]
            )
        );
    }

}

//export {ButtonGroup, ButtonGroup as default};
