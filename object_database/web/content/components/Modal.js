/**
 * Modal Cell Component
 */
//import {Component} from './Component';

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
class Modal extends Component {
    constructor(props, ...args){
        super(props, ...args);
        this.mainStyle = 'display:block;padding-right:15px;';
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
                                this.getReplacementElementFor('title')
                            ])
                        ]),
                        h('div', {class: "modal-body"}, [
                            this.getReplacementElementFor('message')
                        ]),
                        h('div', {class: "modal-footer"}, [
                            this.getReplacementElementsFor('button')
                        ])
                    ])
                ])
            ])
        );
    }
}

//export {Modal, Modal as default}
