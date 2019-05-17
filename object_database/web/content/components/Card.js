// Creates a button component
/*function Card(id) {
    // TODO: this is some hacky nonsense, but 
    let contents_id = id + "_____contents__";

    var card = {
        render: function() {
            return h('div', {class: "card", id: id}, [
                h('div', { class: "card-body p-1" }, [
                    h('div', {id: contents_id}, [])
                ])
            ]);
        }
    };
    return card;
    }*/

class Card extends Component {
    constructor(props, ...args){
        super(props, ...args);
        //this.addReplacement('contents', '_____contents__');
    }

    render(){
        return h('div',
            { 
                class: "cell card",
                style: this.props.extraData.divStyle,
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Card"
            }, 
            [h('div', { class: "card-body p-1" }, [
                this.getReplacementElementFor('contents')
            ])
        ]);
    }
}

//export {Card, Card as default};
