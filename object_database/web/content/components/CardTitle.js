/**
 * CardTitle Cell
 */
// import {Component} from './Component';


class CardTitle extends Component {
    constructor(props, ...args){
        super(props, ...args);
        //this.addReplacement('contents', '____contents__');
    }

    render(){
        return(
            h('div', {
                id: this.props.id,
                class: "cell",
                "data-cell-id": this.props.id,
                "data-cell-type": "CardTitle"
            }, [
                this.getReplacementElementFor('contents')
            ])
        );
    }
}

//export {CardTitle, CardTitle as default};
