/**
 * Dropdown Cell Component
 */

//import {Component} from './Component';
//import {h} from 'maquette';

/**
 * About Replacements
 * ------------------
 * This component has one regular
 * replacement:
 * * `title`
 * This component has one
 * enumerated replacement:
 * * `child`
 */
class Dropdown extends Component {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return (
            h('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Dropdown",
                class: "btn-group"
            }, [
                h('a', {class: "btn btn-xs btn-outline-secondary"}, [
                    this.getReplacementElementFor('title')
                ]),
                h('button', {
                    class: "btn btn-xs btn-outline-secondary dropdown-toggle dropdown-toggle-split",
                    type: "button",
                    id: `${this.props.extraData.targetIdentity}-dropdownMenuButton`,
                    "data-toggle": "dropdown"
                }),
                h('div', {class: "dropdown-menu"}, this.makeItems())
            ])
        );
    }

    makeItems(){
        // For some reason, due again to the Cell implementation,
        // sometimes there are not these child replacements.
        if(!this.replacements.hasReplacement('child')){
            return [];
        }
        return this.getReplacementElementsFor('child').map((element, idx) => {
            return new DropdownItem({
                index: idx,
                childSubstitute: element,
                targetIdentity: this.props.extraData.targetIdentity,
                dropdownItemInfo: this.props.extraData.dropdownItemInfo
            }).render();
        });
    }
}


/**
 * A private subcomponent for each
 * Dropdown menu item. We need this
 * because of how callbacks are handled
 * and because the Cells version doesn't
 * already implement this kind as a separate
 * entity.
 */
class DropdownItem extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
        this.clickHandler = this.clickHandler.bind(this);
    }

    render(){
        return (
            h('a', {
                class: "subcell cell-dropdown-item dropdown-item",
                key: this.props.index,
                onclick: this.clickHandler
            }, [this.props.childSubstitute])
        );
    }

    clickHandler(event){
        // This is super hacky because of the
        // current Cell implementation.
        // This whole component structure should be heavily refactored
        // once the Cells side of things starts to change.
        let whatToDo = this.props.dropdownItemInfo[this.props.index.toString()];
        if(whatToDo == 'callback'){
            let responseData = {
                event: "menu",
                ix: this.props.index,
                target_cell: this.props.targetIdentity
            };
        } else {
            window.location.href = whatToDo;
        }
    }


}

//export {Dropdown, Dropdown as default};
