/**
 * Tabs Cell Component
 */

//import {Component} from './Component';
//import {h} from 'maquette';


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
class Tabs extends Component {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return (
            h('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Tabs",
                class: "container-fluid mb-3"
            }, [
                h('ul', {class: "nav nav-tabs", role: "tablist"}, [
                    this.getReplacementElementsFor('header')
                ]),
                h('div', {class: "tab-content"}, [
                    h('div', {class: "tab-pane fade show active", role: "tabpanel"}, [
                        this.getReplacementElementFor('display')
                    ])
                ])
            ])
        );
    }
}


//export {Tabs, Tabs as default};
