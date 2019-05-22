class SingleLineTextBox extends Component {
    constructor(props, ...args){
        super(props, ...args);
        //this.addReplacement('contents', '_____contents__');
        //
        // Bind context to methods
        this.changeHandler = this.changeHandler.bind(this);
    }

    render(){
        let attrs = 
            { 
                class: "cell",
                id: "text_" + this.props.id,
                type: "text",
                "data-cell-id": this.props.id,
                "data-cell-type": "SingleLineTextBox",
                onchange: (event) => {this.changeHandler(event.target.value)}
            };
        if (this.props.extraData.inputValue !== undefined) {
            attrs.pattern = this.props.extraData.inputValue;
        }
        return h('input', attrs, []);
    }

    changeHandler(val) {
        cellSocket.sendString(
            JSON.stringify(
                {
                    "event": "click",
                    "target_cell": this.props.id,
                    "text": val
                }
            )
        );
    }
}

//export {SingleLineTextBox, SingleLineTextBox as default};
