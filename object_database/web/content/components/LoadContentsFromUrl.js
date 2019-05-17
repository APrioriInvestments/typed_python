/**
 * LoadContentsFromUrl Cell Component
 */
//import {Component} from './Component';


class LoadContentsFromUrl extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
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

//export {LoadContentsFromUrl, LoadContentsFromUrl as default};
