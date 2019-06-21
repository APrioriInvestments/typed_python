/**
 * Card Cell Component
 */

import {Component} from './Component';
import {h} from 'maquette';

/**
 * About Replacements
 * ------------------
 * This component contains two
 * regular replacements:
 * * `contents`
 * * `header`
 */
class Card extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeBody = this.makeBody.bind(this);
        this.makeHeader = this.makeHeader.bind(this);
    }

    render(){
        let bodyClass = 'card-body';
        if(this.props.extraData.padding){
            bodyClass = `card-body p-${this.props.extraData.padding}`;
        }
        let bodyArea = h('div', {
            class: bodyClass
        }, [this.makeBody()]);
        let header = this.makeHeader();
        let headerArea = null;
        if(header){
            headerArea = h('div', {class: "card-header"}, [header]);
        }
        return h('div',
            {
                class: "cell card",
                style: this.props.extraData.divStyle,
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Card"
            }, [headerArea, bodyArea]);
    }

    makeBody(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('contents');
        } else {
            return this.renderChildNamed('contents');
        }
    }

    makeHeader(){
        if(this.usesReplacements){
            if(this.replacements.hasReplacement('header')){
                return this.getReplacementElementFor('header');
            }
        } else {
            return this.renderChildNamed('header');
        }
        return null;
    }
};

console.log('Card module loaded');
export {Card, Card as default};
