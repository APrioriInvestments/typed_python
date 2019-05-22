class SubscribedSequence extends Component {
    constructor(props, ...args){
        super(props, ...args);
        //this.addReplacement('contents', '_____contents__');
        //
        // Bind context to methods
        this.makeClass = this.makeClass.bind(this);
        this.makeChildren = this.makeChildren.bind(this);
    }

    render(){
        return h('div',
            { 
                class: this.makeClass(),
                style: this.props.extraData.divStyle,
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "SubscribedSequence"
            }, [this.makeChildren()]
        )
    }

    makeClass() {
        if (this.props.extraData.asColumns) {
            return "cell subscribedSequence container-fluid";
        }
        return "cell subscribedSequence";
    }

    makeChildren() {
        if (this.props.extraData.asColumns) {
            let spineChildren = [];
            for (var c = 0; c < this.props.extraData.numSpineChildren; c++){
                spineChildren.push(
                    h("div", {class: "col-sm"}, [
                        h("span", {}, ["____child_" + c + "__"]) // TODO: does this make sense?
                    ])
                )
            }
            return h("div", {class: "row flex-nowrap"}, [spineChildren]);
        }

        let spineChildren = "";
        for (var c = 0; c < this.props.extraData.numSpineChildren; c++){
            spineChildren += "____child_" + c + "__"; //TODO: does this make sense?
        }
        return h("div", {}, [spineChildren]);

    }
}

//export {SubscribedSequence, SubscribedSequence as default};
