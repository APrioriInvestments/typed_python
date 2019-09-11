/**
 * SubscribedSequence Cell Component
 */

import {Component, render} from './Component';
import {h} from 'maquette';

/**
 * About Replacements
 * ------------------
 * This component has a single
 * enumerated replacement:
 * * `child`
 */

/**
 * About Named Replacements
 * ------------------------
 * `children` (array) - An array of Cells that are subscribed
 */
class SubscribedSequence extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
        this.makeClass = this.makeClass.bind(this);
        this.makeChildren = this.makeChildren.bind(this);
    }

    build(){
        return h('div',
            {
                class: this.makeClass(),
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "SubscribedSequence"
            }, this.makeChildren()
        );
    }

    makeChildren(){
        if(this.usesReplacements){
            return this.getReplacementElementsFor("child");
        } else {
            let elements = this.props.namedChildren["children"];
            return elements.map(childComponent => {
                let hyperscript = render(childComponent);
                if(childComponent.props.flexChild == true && this.props.flexParent){
                    hyperscrupt.properties.class += " flex-child";
                }
                return hyperscript;
            });
        }
    }

    makeClass() {
        let classes = [
            "cell",
            "sequence",
            "subscribed-sequence",
            "sequence-vertical"
        ];
        if(this.props.flexParent){
            classes.push("flex-parent");
        }
        return classes.join(" ");
    }
}

export {SubscribedSequence, SubscribedSequence as default};
