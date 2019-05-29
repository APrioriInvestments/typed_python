/**
 * Card Cell Component
 */

//import {Component} from './Component';
//import {h} from 'maquette';

/**
 * About Replacements
 * ------------------
 * This component contains two
 * regular replacement:
 * * `contents`
 * * `header`
 */
class Card extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeHeader = this.makeHeader.bind(this);
    }

    render(){
        let bodyClass = "card-body";
        if(this.props.extraData.padding){
            bodyClass += ` p-${this.props.extraData.padding}`;
        }
        return h('div',
            {
                class: "cell card",
                style: this.props.extraData.divStyle,
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Card"
            },
                 [
                     this.makeHeader(),
                     h('div', { class: bodyClass }, [
                         this.getReplacementElementFor('contents')
                     ])
        ]);
    }

    makeHeader(){
        if(this.replacements.hasReplacement('header')){
            return h('div', {class: 'card-header'}, [
                this.getReplacementElementFor('header')
            ]);
        }
        return null;
    }
}

//export {Card, Card as default};
