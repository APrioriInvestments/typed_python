// Creates a button component
/*function Code(id) {
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

class Code extends Component {
    constructor(props, ...args){
        super(props, ...args);
        //this.addReplacement('contents', '_____contents__');
    }

    render(){
        return h('pre',
            { 
                class: "cell code",
                id: this.props.id,
                "data-cell-type": "Code"
            }, [
                h("code", {}, [this.getReplacementElementFor('child')])
            ]
        )
    }
}

//export {Code, Code as default};
