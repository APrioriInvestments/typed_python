/**
 * Columns Cell Component
 */

import {Component} from './Component';
import {h} from 'maquette';

/**
 * About Replacements
 * ------------------
 * This component has one enumerated
 * kind of replacement:
 * * `c`
 */

/**
 * About Named Children
 * --------------------
 * `elements` (array) - Cell column elements
 */
class Columns extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
        this.makeInnerChildren = this.makeInnerChildren.bind(this);
    }

    render(){
        return (
            h('div', {
                class: "cell container-fluid",
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Columns",
            }, [
                h('div', {class: "row flex-nowrap"}, this.makeInnerChildren())
            ])
        );
    }

    makeInnerChildren(){
        if(this.usesReplacements){
            return this.getReplacementElementsFor('c').map(replElement => {
                return (
                    h('div', {
                        class: "col-sm"
                    }, [replElement])
                );
            });
        } else {
            return this.renderChildrenNamed('elements');
        }
    }
}


export {Columns, Columns as default};
