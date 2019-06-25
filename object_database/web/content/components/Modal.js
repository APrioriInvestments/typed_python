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

        // Bind component methods
        this.makeTitle = this.makeTitle.bind(this);
        this.makeMessage = this.makeMessage.bind(this);
        this.makeButtons = this.makeButtons.bind(this);
    }

    render(){
        return (
            h('div', {
                class: "cell modal fade show",
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Modal",
                role: "dialog",
                style: mainStyle
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

    makeButtons(){
        if(this.usesReplacements){
            return this.getReplacementElementsFor('button');
        } else {
            return this.renderChildrenNamed('buttons')
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
