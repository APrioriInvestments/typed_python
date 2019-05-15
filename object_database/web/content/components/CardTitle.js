/**
 * CardTitle Cell
 */
// import {Component} from './Component';


class CardTitle extends Component {
    constructor(props){
        super(props);
        this.addReplacement('contents', '____contents__');
    }

    render(){
        return(
            h('div', {
                id: this.props.id,
                class: "cell",
                "data-cell-id": this.props.id,
                "data-cell-type": "CardTitle"
            }, [
                h('div', {id: this.getReplacement('contents')}, [])
            ])
        );
    }
}
