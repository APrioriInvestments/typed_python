/**
 * Card Cell Component
 */

//import {Component} from './Component';
//import {h} from 'maquette';

/**
 * About Replacements
 * ------------------
 * This component contains a single
 * regular replacement:
 * * `contents`
 */
class Card extends Component {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return h('div',
            {
                class: "cell card",
                style: this.props.extraData.divStyle,
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Card"
            },
            [h('div', { class: "card-body p-1" }, [
                this.getReplacementElementFor('contents')
            ])
        ]);
    }
}

//export {Card, Card as default};
