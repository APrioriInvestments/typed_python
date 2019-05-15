/**
 * Octicon Cell Component
 */
//import {Component} from './Component';


class Octicon extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
        this._getHTMLClasses = this._getHTMLClasses.bind(this);
    }

    render(){
        return(
            h('span', {
                class: this._getHTMLClasses(),
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Octicon",
                "aria-hidden": true,
                style: this.props.extraData.divStyle
            })
        );
    }

    _getHTMLClasses(){
        let classes = ["cell", "octicon"];
        this.props.extraData.octiconClasses.forEach(name => {
            classes.push(name);
        });
        return classes.join(" ");
    }
}

//export {Octicon, Octicon as default};
