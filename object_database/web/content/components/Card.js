/**
 * Card Cell Component
 */

import {Component} from './Component';
import {PropTypes} from './util/PropertyValidator';
import {h} from 'maquette';

/**
 * About Replacements
 * ------------------
 * This component contains two
 * regular replacements:
 * * `contents`
 * * `header`
 */

/**
 * About Named Children
 * `body` (single) - The cell to put in the body of the Card
 * `header` (single) - An optional header cell to put above
 *        body
 */
class Card extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeBody = this.makeBody.bind(this);
        this.makeHeader = this.makeHeader.bind(this);
    }

    render(){
        let bodyArea = h('div', {
            class: 'card-body p-${this.props.padding}'
        }, [this.makeBody()]);
        let header = this.makeHeader();
        let headerArea = null;
        if(header){
            headerArea = h('div', {class: "card-header"}, [header]);
        }
        return h('div',
            {
                class: "cell card",
                style: this.props.divStyle,
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Card"
            }, [headerArea, bodyArea]);
    }

    makeBody(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('contents');
        } else {
            return this.renderChildNamed('body');
        }
    }

    makeHeader(){
        if(this.usesReplacements){
            if(this.replacements.hasReplacement('header')){
                return this.getReplacementElementFor('header');
            }
            return null;
        } else {
            return this.renderChildNamed('header');
        }
    }
}

Card.propTypes = {
    padding: {
        description: "Padding weight as defined by Boostrap css classes.",
        type: PropTypes.oneOf([PropTypes.number, PropTypes.string])
    },
    divStyle: {
        description: "HTML style attribute string.",
        type: PropTypes.oneOf([PropTypes.string])
    }
};

export {Card, Card as default};
