
class ContextualDisplay extends Component {
    constructor(props, ...args){
        super(props, ...args);
        //this.addReplacement('contents', '_____contents__');
    }

    render(){
        return h('div',
            { 
                class: "cell contextualDisplay",
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "ContextualDisplay"
            }, [this.getReplacementElementFor('child')]
        );
    }
}

//export {ContextualDisplay, ContextualDisplay as default};
