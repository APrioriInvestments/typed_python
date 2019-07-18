/**
 * Modal Cell Component
 */

import {Component} from './Component';
import {h} from 'maquette';

/**
 * About Replacements
 * ------------------
 * Modal has the following single replacements:
 * *`title`
 * *`message`
 * And has the following enumerated
 * replacements
 * * `button`
 */

/**
 * About Named Children
 * --------------------
 * `title` (single) - A Cell containing the title
 * `message` (single) - A Cell contianing the body of the
 *     modal message
 * `buttons` (array) - An array of button cells
 */
class Modal extends Component {
    constructor(props, ...args){
        super(props, ...args);
        this.mainStyle = 'display:block;padding-right:15px;';
        this.modal = null;

        // Bind component methods
        this.makeTitle = this.makeTitle.bind(this);
        this.makeMessage = this.makeMessage.bind(this);
        this.makeButtons = this.makeButtons.bind(this);
        this.makeClasses = this.makeClasses.bind(this);
        this.makeStyle = this.makeStyle.bind(this);
    }

    componentDidLoad(){
        console.log(`Modal component ${this.props.id} loaded`);
    }

    componentDidUpdate(){
        console.log(`Modal component ${this.props.id} updated`);
    }

    render(){
        console.log('Modal render with show:');
        console.log(this.props.extraData.show.toString());
        return (
            h('div', {
                class: this.makeClasses(),
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Modal",
                role: "dialog",
                //tabindex: "-1",
                style: this.makeStyle()
            }, [
                h('div', {role: "document", class: "modal-dialog"}, [
                    h('div', {class: "modal-content"}, [
                        h('div', {class: "modal-header"}, [
                            h('h5', {class: "modal-title"}, [
                                this.makeTitle()
                            ])
                        ]),
                        h('div', {class: "modal-body"}, [
                            this.makeMessage()
                        ]),
                        h('div', {class: "modal-footer"}, this.makeButtons())
                    ])
                ])
            ])
        );
    }

    makeStyle(){
        // TODO: Move this into some
        // CSS classes and condition
        // using makeClasses()
        if(this.props.extraData.show == true){
            //return this.mainStyle;
            return "";
        } else {
            return "";
        }
    }

    makeClasses(){
        let classes = ["cell", "modal"];
        if(this.props.extraData.show == true){
            classes.push("show");
        }
        return classes.join(" ");
    }

    makeButtons(){
        if(this.usesReplacements){
            return this.getReplacementElementsFor('button');
        } else {
            return this.renderChildrenNamed('buttons');
        }
    }

    makeMessage(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('message');
        } else {
            return this.renderChildNamed('message');
        }
    }

    makeTitle(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('title');
        } else {
            return this.renderChildNamed('title');
        }
    }
}

export {Modal, Modal as default}
