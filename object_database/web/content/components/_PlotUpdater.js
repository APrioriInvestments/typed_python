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
                style: "display:none;",
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
        // TODO These are global var defined in page.html
        // we should do something about this.
        if (this.props.extraData.exceptionOccured) {
            console.log("plot exception occured");
            Plotly.purge(aDOMElement);
        } else {
            console.log('_PlotUpdater updating from component');
            let data = this.props.extraData.plotData.map(mapPlotlyData);
            Plotly.react(aDOMElement, data, aDOMElement.layout);
        }
    }
};

/** Helper Functions **/
function mapPlotlyData(d) {
    console.log('Calling mapPlotlyData helper...');
    if (d.timestamp !== undefined) {
        d.timestamp = unpackHexFloats(d.timestamp);
        d.x = Array.from(d.timestamp).map(ts => new Date(ts * 1000));
    } else {
        d.x = unpackHexFloats(d.x);
    }

    if (d.y !== undefined) {
        d.y = unpackHexFloats(d.y);
    }
    if (d.open !== undefined) {
        d.open = unpackHexFloats(d.open);
    }
    if (d.close !== undefined) {
        d.close = unpackHexFloats(d.close);
    }
    if (d.high !== undefined) {
        d.high = unpackHexFloats(d.high);
    }
    if (d.low !== undefined) {
        d.low = unpackHexFloats(d.low);
    }
    return d;
}

function hexcharToInt(x) {
    if (x>=97) return x - 97 + 10;
    return x - 48;
}

function unpackHexFloats(x) {
    if (typeof x != "string") {
        return x;
    }

    var buf = new ArrayBuffer(x.length/2);
    var bufView = new Uint8Array(buf);

    for (var i=0, strLen=x.length/2; i < strLen; i+=1) {
        bufView[i] = hexcharToInt(x.charCodeAt(i*2)) * 16 + hexcharToInt(x.charCodeAt(i*2+1));
    }

    return new Float64Array(buf);
}

//export {_PlotUpdater, _PlotUpdater as default};
