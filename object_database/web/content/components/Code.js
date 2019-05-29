/**
 * Code Cell Component
 */

//import {Component} from './Component';
//import {h} from 'maquette';

/**
 * About Replacements
 * ------------------
 * This component has a single
 * regular replacement:
 * * `child`
 */
class Code extends Component {
    constructor(props, ...args){
        super(props, ...args);
        //this.addReplacement('contents', '_____contents__');
    }

    render(){
        return h('pre',
                 {
                     class: "cell code",
                     id: this.props.id,
                     "data-cell-type": "Code"
                 }, [
                     h("code", {}, [this.getReplacementElementFor('child')])
                 ]
                );
    }
}

//export {Code, Code as default};
