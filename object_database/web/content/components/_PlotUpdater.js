/**
 * _PlotUpdater Cell Component
 * NOTE: Later refactorings should result in
 * this component becoming obsolete
 */

//import {Component} from './Component';
//import {h} from 'maquette';

const MAX_INTERVALS = 25;

class _PlotUpdater extends Component {
    constructor(props, ...args){
        super(props, ...args);

        this.runUpdate = this.runUpdate.bind(this);
        this.listenForPlot = this.listenForPlot.bind(this);
    }

    componentDidLoad() {
        // If we can find a matching Plot element
        // at this point, we simply update it.
        // Otherwise we need to 'listen' for when
        // it finally comes into the DOM.
        let initialPlotDiv = document.getElementById(`plot${this.props.extraData.plotId}`);
        if(initialPlotDiv){
            this.runUpdate(initialPlotDiv);
        } else {
            this.listenForPlot();
        }
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

    /**
     * In the event that a `_PlotUpdater` has come
     * over the wire *before* its corresponding
     * Plot has come over (which appears to be
     * common), we will set an interval of 50ms
     * and check for the matching Plot in the DOM
     * MAX_INTERVALS times, only calling `runUpdate`
     * once we've found a match.
     */
    listenForPlot(){
        let numChecks = 0;
        let plotChecker = window.setInterval(() => {
            if(numChecks > MAX_INTERVALS){
                window.clearInterval(plotChecker);
                console.error(`Could not find matching Plot ${this.props.extraData.plotId} for _PlotUpdater ${this.props.id}`);
                return;
            }
            let plotDiv = document.getElementById(`plot${this.props.extraData.plotId}`);
            if(plotDiv){
                this.runUpdate(plotDiv);
                window.clearInterval(plotChecker);
            } else {
                numChecks += 1;
            }
        }, 50);
    }

    runUpdate(aDOMElement){
        console.log("Updating plotly chart.");
        // TODO These are global var defined in page.html
        // we should do something about this.
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
