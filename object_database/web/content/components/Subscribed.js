/**
 * Subscribed Cell Component
 */

import {Component} from './Component';
import {h} from 'maquette';

/**
 * About Replacements
 * ------------------
 * This component has a single
 * regular replacement:
 * * `contents`
 */

/**
 * About Named Children
 * --------------------
 * `content` (single) - The underlying Cell that is subscribed
 */
class Subscribed extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Responds true
        this.isSubscribed = true;
        this.previoudChildId = null;

        // Bind component methods
        this.makeContent = this.makeContent.bind(this);
        this.toString = this.toString.bind(this);
    }

    build(){
        let velement = this.makeContent();
        velement.properties['data-subscribed-to'] = this.props.id;
        return velement;
    }

    getDOMElement(){
        let el = document.querySelector(`[data-subscribed-to="${this.props.id}"]`);
        return el;
    }

    makeContent(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('contents');
        } else {
            return this.renderChildNamed('content');
        }
    }

    toString(){
        return `Sub[${this.props.id}]<${this.props.namedChildren.content}>`;
    }
}

export {Subscribed, Subscribed as default};
