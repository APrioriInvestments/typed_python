/**
 * CollapsiblePanel Cell Component
 */
import {Component} from './Component.js';
import {h} from 'maquette';
import {PropTypes} from './util/PropertyValidator.js';

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

    build(){
        let content = this.makeContent();
        let panel = null;
        if(this.props.isExpanded){
            panel = this.makePanel();
        }
        return (
            h('div', {
                class: "cell collapsible-panel",
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "CollapsiblePanel",
                "data-is-expanded": (this.props.isExpanded == true)
            }, [panel, content])
        );
    }

    makeContent(){
        let result = null;
        if(this.usesReplacements){
            result = this.getReplacementElementFor('content');
        } else {
            result = this.renderChildNamed('content');
        }
        return (
            h('div', {class: "collapsible-panel-content"}, [result])
        );
    }

    makePanel(){
        let result = null;
        if(this.usesReplacements){
            result = this.getReplacementElementFor('panel');
        } else {
            result = this.renderChildNamed('panel');
        }
        return (
            h('div', {class: "collapsible-panel-panel"}, [result])
        );
    }
}

CollapsiblePanel.propTypes = {
    isExpanded: {
        description: "Whether or not the Panel is expanded (showing)",
        type: PropTypes.boolean
    }
};


export {CollapsiblePanel, CollapsiblePanel as default}
