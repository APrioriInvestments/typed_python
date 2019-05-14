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
    constructor(props){
        super(props);
        this.addReplacement('contents', '_____contents__');
    }

    render(){
        return h('div',
                 {
                     class: "card",
                     id: this.props.id,
                     "data-cell-id": this.props.id
                 }, [
            h('div', { class: "card-body p-1" }, [
                h('div', {id: this.getReplacement('contents')}, [])
            ])
        ]);
    }
}
