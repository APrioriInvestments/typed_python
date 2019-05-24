class _PlotUpdater extends Component {
    constructor(props, ...args){
        super(props, ...args);

        this.runUpdate = this.runUpdate.bind(this);
    }

    componentDidLoad() {
        this.runUpdate();
    }

    render(){
        return h('div',
            {
                class: "cell",
                id: "plot" + this.props.extraData.plotId,
                style: "display: none",
                "data-cell-id": this.props.id,
                "data-cell-type": "_PlotUpdater"
            }, []);
    }

    runUpdate(){
        console.log("Updating plotly chart.")
        // TODO These are global var defined in page.html
        // we should do something about this.
        var plotDiv = document.getElementById('plot' + this.props.extraData.plotId);
        if (this.props.extraData.exceptionOccured) {
            console.log("plot exception occured");
            Plotly.purge(plotDiv);
        } else {
            let data = this.props.extraData.plotData.map(mapPlotlyData);
            Plotly.react(plotDiv, data, plotDiv.layout);
        }
    }
}

//export {_PlotUpdater, _PlotUpdater as default};
