/**
 * Traceback Cell Component
 */

//import {Component} from './Component';
//import {h} from 'maquette';

/**
 * About Replacements
 * ------------------
 * This component has a single regular
 * repalcement:
 * * `child`
 */
class  Traceback extends Component {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return (
            h('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Traceback",
                class: "alert alert-primary"
            }, [this.getReplacementElementFor('child')])
        );
    }
}


//export {Traceback, Traceback as default};
