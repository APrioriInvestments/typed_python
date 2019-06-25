/**
 * CollapsiblePanel Cell Component
 */
import {Component} from './Component.js';
import {h} from 'maquette';

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

/**
 * About Named Children
 * --------------------
 * `content` (single) - The current content Cell of the panel
 * `panel` (single) - The current (expanded) panel view
 */
class CollapsiblePanel extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makePanel = this.makePanel.bind(this);
        this.makeContent = this.makeContent.bind(this);
    }

    render(){
        if(this.props.extraData.isExpanded){
            return(
                h('div', {
                    class: "cell container-fluid",
                    "data-cell-id": this.props.id,
                    "data-cell-type": "CollapsiblePanel",
                    "data-expanded": true,
                    id: this.props.id,
                    style: this.props.extraData.divStyle
                }, [
                    h('div', {class: "row flex-nowrap no-gutters"}, [
                        h('div', {class: "col-md-auto"},[
                            this.makePanel()
                        ]),
                        h('div', {class: "col-sm"}, [
                            this.makeContent()
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
                    style: this.props.extraData.divStyle
                }, [this.makeContent()])
            );
        }
    }

    makeContent(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('content');
        } else {
            return this.renderChildNamed('content');
        }
    }

    makePanel(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('panel');
        } else {
            return this.renderChildNamed('panel');
        }
    }
}


export {CollapsiblePanel, CollapsiblePanel as default}
