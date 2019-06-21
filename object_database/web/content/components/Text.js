/**
 * Text Cell Component
 */

import {Component} from './Component';
import {h} from 'maquette';

class Text extends Component {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return(
            h('div', {
                class: "cell",
                id: this.props.id,
                style: this.props.extraData.divStyle,
                "data-cell-id": this.props.id,
                "data-cell-type": "Text"
            }, [this.props.extraData.rawText])
        );
    }
}

export {Text, Text as default};
