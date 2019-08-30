/**
 * ResizablePanel Cell Component
 * --------------------------------
 * This component represents a flex view
 * of two children whereby the proportions
 * of each child in the parent can be updated
 * with a vertical/horizontal resizer.
 * NOTE: We are using the Splitjs
 * (https://github.com/nathancahill/split/tree/master/packages/splitjs#api)
 * library to deal with the complexity
 * of global event listeners etc associated
 * with resizing and dragging items.
 */

/**
 * About Replacements
 * ---------------------
 * This component has two regular replacements:
 * * `first` - The first cell in the two part splitview
 * * `second` - The second cell in the two part splitview
 */

/**
 * About Named Children
 * --------------------
 * `first` (single) - The first cell to show in the view
 * `second` (single) - The second cell to show in the view
 */
import {Component} from './Component';
import {h} from 'maquette';
import Split from 'split.js';

class ResizablePanel extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeFirstChild = this.makeFirstChild.bind(this);
        this.makeSecondChild = this.makeSecondChild.bind(this);
        this.afterCreate = this.afterCreate.bind(this);
        this.afterDestroyed = this.afterDestroyed.bind(this);
    }

    build(){
        let classString = "";
        if(this.props.extraData.split == 'vertical'){
            classString = " horizontal-panel";
        } else if(this.props.extraData.split == 'horizontal'){
            classString = " vertical-panel";
        }
        return (
            h('div', {
                id: this.props.id,
                class: `cell resizable-panel${classString}`,
                'data-cell-type': 'ResizablePanel',
                'data-cell-id': this.props.id,
                afterCreate: this.afterCreate,
                afterDestroyed: this.afterDestroyed
            }, [
                this.makeFirstChild(),
                this.makeSecondChild()
            ])
        );
    }

    makeFirstChild(){
        let inner = null;
        if(this.usesReplacements){
            inner = this.getReplacementElementFor('first');
        } else {
            inner = this.renderChildNamed('first');
        }
        return (
            h('div', {
                class: 'resizable-panel-item'
            }, [inner])
        );
    }

    makeSecondChild(){
        let inner = null;
        if(this.usesReplacements){
            inner = this.getReplacementElementFor('second');
        } else {
            inner = this.renderChildNamed('second');
        }
        return (
            h('div', {
                class: 'resizable-panel-item'
            }, [inner])
        );
    }

    afterCreate(element){
        // Sometimes maquette calls afterCreate
        // twice. This will end up creating two gutters
        // instead of one, so we need to check for the
        // splitter attached instance and return if it's
        // already there.
        if(element._splitter){
            return;
        }

        // Our Cell described directions are opposite those
        // required by the Splitjs library, so we need
        // to case them out and provide the opposite as
        // a Splitjs constructor option
        let reverseDirection = 'horizontal';
        if(this.props.extraData.split == 'horizontal'){
            reverseDirection = 'vertical';
        } else if(this.props.extraData.split == 'vertical'){
            reverseDirection = 'horizontal';
        }
        element._splitter = Split(
            element.querySelectorAll('.resizable-panel-item'),
            {
                direction: reverseDirection
            }
        );
    }

    afterDestroyed(element){
        if(element._splitter){
            element._splitter.destroy();
        }
    }
}

export {ResizablePanel, ResizablePanel as default};
