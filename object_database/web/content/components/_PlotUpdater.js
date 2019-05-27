class _PlotUpdater extends Component {
    constructor(props, ...args){
        super(props, ...args);

        this.runUpdate = this.runUpdate.bind(this);
    }

    componentDidLoad() {
        let plotChecker = window.setInterval(() => {
            let plotDiv = document.getElementById(`plot${this.props.extraData.plotId}`);
            console.log('Checking for Plot element...');
            if(plotDiv){
                this.runUpdate(plotDiv);
                window.clearInterval(plotChecker);
                console.log('Found plot element and clearing interval....');
            }
        }, 50);
    }

    render(){
        return h('div',
            {
                class: "cell",
                id: this.props.id,
                style: "display: none",
                "data-cell-id": this.props.id,
                "data-cell-type": "_PlotUpdater"
            }, []);
    }

    runUpdate(aDOMElement){
        console.log("Updating plotly chart.");
        // TODO These are global var defined in page.html
        // we should do something about this.
        if(!aDOMElement){
            console.log(`Couldnot find plot ${this.props.extraData.plotId}`);
        }
        if (this.props.extraData.exceptionOccured) {
            console.log("plot exception occured");
            Plotly.purge(aDOMElement);
        } else {
            let data = this.props.extraData.plotData.map(mapPlotlyData);
            Plotly.react(aDOMElement, data, aDOMElement.layout);
        }
    }
}

//export {_PlotUpdater, _PlotUpdater as default};
