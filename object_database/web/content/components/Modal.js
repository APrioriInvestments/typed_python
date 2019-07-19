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

        // Bind component methods
        this.makeHeader = this.makeHeader.bind(this);
        this.makeBody = this.makeBody.bind(this);
        this.makeFooter = this.makeFooter.bind(this);
        this.makeClasses = this.makeClasses.bind(this);
    }

    render(){
        return (
            h('div', {
                id: this.props.id,
                'data-cell-id': this.props.id,
                'data-cell-type': "Modal",
                class: this.makeClasses(),
                tabindex: "-1",
                role: "dialog"
            }, [
                h('div', {class: "modal-dialog", role: "document"}, [
                    h('div', {class: "modal-content"}, [
                        h('div', {class: "modal-header"}, [this.makeHeader()]),
                        h('div', {class: "modal-body"}, [this.makeBody()]),
                        h('div', {class: "modal-footer"}, this.makeFooter())
                    ])
                ])
            ])
        );
    }

    makeClasses(){
        let classes = ["cell", "modal-cell"];
        if(this.props.extraData.show){
            classes.push("modal-cell-show");
        }
        return classes.join(" ");
    }

    makeFooter(){
        if(this.usesReplacements){
            return this.getReplacementElementsFor('button');
        } else {
            return this.renderChildrenNamed('buttons');
        }
    }

    makeBody(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('message');
        } else {
            return this.renderChildNamed('message');
        }
    }

    makeHeader(){
        var title = null;
        if(this.usesReplacements){
            title = this.getReplacementElementFor('title');
            if(title){
                return h('h5', {class: "modal-title", id: `${this.props.id}-modalTitle`}, [
                    title
                ]);
            }
        } else {
            title = this.renderChildNamed('title');
            if(title){
                return h('h5', {class: "modal-title", id: `${this.props.id}-modalTitle`}, [
                    title
                ]);
            }
        }
        return null;
    }
}

export {Modal, Modal as default}
