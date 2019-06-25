/**
 * Badge Cell Component
 */
import {Component} from './Component';
import {h} from 'maquette';

/**
 * About Replacements
 * ------------------
 * Badge has a single replacement:
 * * `child`
 */

/**
 * About Named Children
 * --------------------
 * `inner` - The concent cell of the Badge
 */
class Badge extends Component {
    constructor(props, ...args){
        super(...args);

        // Bind component methods
        this.makeInner = this.makeInner.bind(this);
    }

    render(){
        return(
            h('span', {
                class: `cell badge badge-${this.props.extraData.badgeStyle}`,
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Badge"
            }, [this.makeContent()])
        );
    }

    makeInner(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('child');
        } else {
            return this.renderChildNamed('inner');
        }
    }
}

export {Badge, Badge as default};
