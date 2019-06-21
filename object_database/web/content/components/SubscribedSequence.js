/**
 * SubscribedSequence Cell Component
 */

import {Component} from './Component';
import {h} from 'maquette';

/**
 * About Replacements
 * ------------------
 * This component has a single
 * enumerated replacement:
 * * `child`
 */
class SubscribedSequence extends Component {
    constructor(props, ...args){
        super(props, ...args);
        //this.addReplacement('contents', '_____contents__');
        //
        // Bind context to methods
        this.makeClass = this.makeClass.bind(this);
        this.makeChildren = this.makeChildren.bind(this);
    }

    render(){
        return h('div',
            {
                class: this.makeClass(),
                style: this.props.extraData.divStyle,
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "SubscribedSequence"
            }, [this.makeChildren()]
        );
    }

    makeClass() {
        if (this.props.extraData.asColumns) {
            return "cell subscribedSequence container-fluid";
        }
        return "cell subscribedSequence";
    }

    makeChildren(){
        if(this.props.extraData.asColumns){
            let formattedChildren = this.getReplacementElementsFor('child').map(childElement => {
                return(
                    h('div', {class: "col-sm", key: childElement.id}, [
                        h('span', {}, [childElement])
                    ])
                );
            });
            return (
                h('div', {class: "row flex-nowrap", key: `${this.props.id}-spine-wrapper`}, formattedChildren)
            );
        } else {
            return (
                h('div', {key: `${this.props.id}-spine-wrapper`}, this.getReplacementElementsFor('child'))
            );
        }
    }
}

export {SubscribedSequence, SubscribedSequence as default};
