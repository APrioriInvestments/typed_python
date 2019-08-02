/**
 * SplitView Cell Component
 * -------------------------
 * A SplitView cell is a display container
 * that is either horizontal or vertical
 * that uses flexbox to divide up its child
 * components according to an array of proportions
 * that are passed in.
 */

/**
 * About Replacements
 * -----------------------
 * This component has a single enumerated
 * replacement:
 * * `element`
 */

/**
 * About Named Children
 * ---------------------
 * `elements` (array) - The contained component elements
 */
import {Component} from './Component';
import {h} from 'maquette';

class SplitView extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeStyle = this.makeStyle.bind(this);
        this.makeChildStyle = this.makeChildStyle.bind(this);
        this.makeChildElements = this.makeChildElements.bind(this);
    }

    render(){
        return (
            h('div', {
                id: this.props.id,
                class: 'cell split-view',
                'data-cell-id': this.props.id,
                'data-cell-type': "SplitView",
                style: this.makeStyle()
            }, this.makeChildElements())
        );
    }

    makeStyle(){
        // Note: the server side uses "split" (axis) to denote the direction
        let direction = "row";
        if(this.props.extraData.split == "horizontal"){
            direction = "column";
        }
        return `width:100%;height:100%;display:flex;flex-direction:${direction};`;
    }

    makeChildStyle(index){
        let proportion = this.props.extraData.proportions[index];
        if(typeof(proportion) == 'string'){
            return `width:${proportion};`;
        } else {
            return `flex: ${proportion}`;
        }
    }

    makeChildElements(){
        if(this.usesReplacements){
            return this.getReplacementElementsFor('element').map((child, idx) => {
                return h('div', {
                    style: this.makeChildStyle(idx)
                }, [child]);
            });
        } else {
            return this.renderChildrenNamed('elements').map((child, idx) => {
                return h('div', {
                    style: this.makeChildStyle(idx)
                }, [child]);
            });
        }
    }
}

export {SplitView, SplitView as default};
