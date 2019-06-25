/**
 * Code Cell Component
 */

import {Component} from './Component';
import {h} from 'maquette';

/**
 * About Replacements
 * ------------------
 * This component has a single
 * regular replacement:
 * * `child`
 */

/**
 * About Named Children
 * --------------------
 * `code` (single) - Code that will be rendered inside
 */
class Code extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeCode = this.makeCode.bind(this);
    }

    render(){
        return h('pre',
                 {
                     class: "cell code",
                     id: this.props.id,
                     "data-cell-type": "Code"
                 }, [
                     h("code", {}, [this.makeCode()])
                 ]
                );
    }

    makeCode(){
        if(this.usesReplacements){
            return this.getReplacementelementFor('child');
        } else {
            return this.renderChildNamed('code');
        }
    }
}

export {Code, Code as default};
