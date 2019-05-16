/**
 * CollapsiblePanel Cell Component
 */
//import {Component} from './Component.js';
//import {h} from 'maquette';

/**
 * About Replacements
 * ------------------
 * This component has two single type
 * replacements:
 * * `content`
 * * `panel`
 * Note that `panel` is only rendered
 * if the panel is expanded
 */

class CollapsiblePanel extends Component {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        if(this.extraData.isExpanded){
            return(
                h('div', {
                    class: "cell container-fluid",
                    "data-cell-id": this.props.id,
                    "data-cell-type": "CollapsiblePanel",
                    "data-expanded": true,
                    id: this.props.id,
                    style: this.extraData.divStyle
                }, [
                    h('div', {class: "row flex-nowrap no-gutters"}, [
                        h('div', {class: "col-md-auto"},[
                            this.getReplacementElementFor('panel')
                        ]),
                        h('div', {class: "col-sm"}, [
                            this.getReplacementElementFor('content')
                        ])
                    ])
                ])
            );
        } else {
            return (
                h('div', {
                    class: "cell container-fluid",
                    "data-cell-id": this.props.id,
                    "data-cell-type": "CollapsiblePanel",
                    "data-expanded": false,
                    id: this.props.id,
                    style: this.extraData.divStyle
                }, [this.getReplacementElementFor('content')])
            );
        }
    }
}


//export {CollapsiblePanel, CollapsiblePanel as default}
