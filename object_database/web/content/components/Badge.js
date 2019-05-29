/**
 * Badge Cell Component
 */
//import {Component} from './Component';
//import {h} from 'maquette';

/**
 * About Replacements
 * ------------------
 * Badge has a single replacement:
 * * `child`
 */
class Badge extends Component {
    constructor(props, ...args){
        super(...args);
    }

    render(){
        return(
            h('span', {
                class: `cell badge badge-${this.props.extraData.badgeStyle}`,
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Badge"
            }, [this.getReplacementElementFor('child')])
        );
    }
}

//export {Badge, Badge as default};
