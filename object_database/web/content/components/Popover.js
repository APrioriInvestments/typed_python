/**
 * Popover Cell Component
 */

import {Component} from './Component';
import {h} from 'maquette';

/**
 * About Replacements
 * This component contains the following
 * regular replacements:
 * * `title`
 * * `detail`
 * * `contents`
 */
class Popover extends Component {
    constructor(props, ...args){
        super(props, ...args);
    }

    render(){
        return h('div',
            {
                class: "cell",
                style: this.props.extraData.divStyle,
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Popover"
            }, [
                h('a',
                    {
                        href: "#popmain_" + this.props.id,
                        "data-toggle": "popover",
                        "data-trigger": "focus",
                        "data-bind": "#pop_" + this.props.id,
                        "data-placement": "bottom",
                        role: "button",
                        class: "btn btn-xs"
                    },
                    [this.getReplacementElementFor('contents')]
                ),
                h('div', {style: "display:none"}, [
                    h("div", {id: "pop_" + this.props.id}, [
                        h("div", {class: "data-title"}, [this.getReplacementElementFor("title")]),
                        h("div", {class: "data-content"}, [
                            h("div", {style: "width: " + this.props.width + "px"}, [
                                this.getReplacementElementFor('detail')]
                            )
                        ])
                    ])
                ])
            ]
        );
    }
}

export {Popover, Popover as default};
