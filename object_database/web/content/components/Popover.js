// Creates a button component
/*function Popover(id) {
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

class Popover extends Component {
    constructor(props, ...args){
        super(props, ...args);
        //this.addReplacement('contents', '_____contents__');
    }

    render(){
        return h('div',
            { 
                class: "cell popover",
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
        )
    }
}

//export {Popover, Popover as default};
