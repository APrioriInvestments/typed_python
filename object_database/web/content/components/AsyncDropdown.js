/**
 * AsyncDropdown Cell Component
 */

//import {Component} from './Component';
//import {h} from 'maquette';

/**
 * About Replacements
 * -------------------
 * This component has a single regular
 * replacement:
 * * `contents`
 *
 * NOTE: The Cells version of this child is
 * either a loading indicator, text, or a
 * AsyncDropdownContent cell.
 */
class AsyncDropdown extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
        this.addDropdownListener = this.addDropdownListener.bind(this);
    }

    render(){
        return (
            h('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "AsyncDropdown",
                class: "cell btn-group"
            }, [
                h('a', {class: "btn btn-xs btn-outline-secondary"}, [this.props.extraData.labelText]),
                h('button', {
                    class: "btn btn-xs btn-outline-secondary dropdown-toggle dropdown-toggle-split",
                    type: "button",
                    id: `${this.props.id}-dropdownMenuButton`,
                    "data-toggle": "dropdown",
                    afterCreate: this.addDropdownListener,
                    "data-firstclick": true
                }),
                h('div', {
                    id: `${this.props.id}-dropdownContentWrapper`,
                    class: "dropdown-menu"
                }, [this.getReplacementElementFor('contents')])
            ])
        );
    }

    addDropdownListener(element){
        let parentEl = element.parentElement;
        let firstTimeClicked = (element.dataset.firstclick == true);
        if(firstTimeClicked){
            $(parentEl).on('show.bs.dropdown', function(){
                cellSocket.sendString(JSON.stringify({
                    event:'dropdown',
                    target_cell: this.props.id,
                    isOpen: false
                }));
            });
            $(parentEl).on('hide.bs.dropdown', function(){
                cellSocket.sendString(JSON.stringify({
                    event: 'dropdown',
                    target_cell: this.props.id,
                    isOpen: true
                }));
            });
            element.dataset.firstclick = false;
        }
    }
}

/**
 * About Replacements
 * ------------------
 * This component has a single regular
 * replacement:
 * * `contents`
 */
class AsyncDropdownContent extends Component {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return (
            h('div', {
                id: `dropdownContent-${this.props.id}`,
                "data-cell-id": this.props.id,
                "data-cell-type": "AsyncDropdownContent"
            }, [this.getReplacementFor('contents')])
        );
    }
}


//export {AsyncDropdown, AsyncDropdown as default};
