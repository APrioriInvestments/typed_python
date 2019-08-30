/**
 * Tabs Cell Component
 */

import {Component} from './Component';
import {h} from 'maquette';


/**
 * About Replacements
 * ------------------
 * This component had a single
 * regular replacement:
 * * `display`
 * This component has a single
 * enumerated replacement:
 * * `header`
 */

/**
 * About Named Children
 * --------------------
 * `display` (single) - The Cell that gets displayed when
 *      the tabs are showing
 * `headers` (array) - An array of cells that serve as
 *     the tab headers
 */
class Tabs extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeHeaders = this.makeHeaders.bind(this);
        this.makeDisplay = this.makeDisplay.bind(this);
    }

    build(){
        return (
            h('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Tabs",
                class: "container-fluid mb-3"
            }, [
                h('ul', {class: "nav nav-tabs", role: "tablist"}, this.makeHeaders()),
                h('div', {class: "tab-content"}, [
                    h('div', {class: "tab-pane fade show active", role: "tabpanel"}, [
                        this.makeDisplay()
                    ])
                ])
            ])
        );
    }

    makeDisplay(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('display');
        } else {
            return this.renderChildNamed('display');
        }
    }

    makeHeaders(){
        if(this.usesReplacements){
            return this.getReplacementElementsFor('header');
        } else {
            return this.renderChildrenNamed('headers');
        }
    }
}


export {Tabs, Tabs as default};
