/**
 * LoadContentsFromUrl Cell Component
 */

import {Component} from './Component';
import {h} from 'maquette';

class LoadContentsFromUrl extends Component {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return(
            h('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "LoadContentsFromUrl",
            }, [h('div', {id: this.props.extraData['loadTargetId']}, [])]
            )
        );
    }

}

export {LoadContentsFromUrl, LoadContentsFromUrl as default};
