/**
 * Text Cell Component
 */

import {Component} from './Component';
import {PropTypes} from './util/PropertyValidator';
import {h} from 'maquette';

class Text extends Component {
    constructor(props, ...args){
        super(props, ...args);
        this.style = "color:" + this.props.textColor;
    }

    build(){
        return(
            h('div', {
                class: "cell",
                id: this.props.id,
                style: this.style,
                "data-cell-id": this.props.id,
                "data-cell-type": "Text"
            }, [this.props.rawText])
        );
    }
}

Text.propTypes = {
    textColor: {
        description: "Text color.",
        type: PropTypes.oneOf([PropTypes.string])
    },
    rawText: {
        description: "Basic display text.",
        type: PropTypes.oneOf([PropTypes.string])
    },
    escapedText: {
        description: "Escaped Text.", //TODO!
        type: PropTypes.oneOf([PropTypes.string])
    },
};

export {Text, Text as default};
