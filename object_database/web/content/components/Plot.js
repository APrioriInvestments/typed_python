/**
 * Plot Cell Component
 */

//import {Component} from './Component';
//import {h} from 'maquette';

/**
 * About Replacements
 * ------------------
 * This component contains the following
 * regular replacements:
 * * `chart-updater`
 * * `error`
 */
class Plot extends Component {
    constructor(props, ...args){
        super(props, ...args);

        this.setupPlot = this.setupPlot.bind(this);
    }

    componentDidLoad() {
        this.setupPlot();
    }

    render(){
        return (
            h('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Plot",
                class: "cell"
            }, [
                h('div', {id: `plot${this.props.id}`, style: this.props.extraData.divStyle}),
                this.getReplacementElementFor('chart-updater'),
                this.getReplacementElementFor('error')
            ])
        );
    }

    setupPlot(){
        console.log("Setting up a new plotly chart.");
        // TODO These are global var defined in page.html
        // we should do something about this.
        var plotDiv = document.getElementById('plot' + this.props.id);
        Plotly.plot(
            plotDiv,
            [],
            {
                margin: {t : 30, l: 30, r: 30, b:30 },
                xaxis: {rangeslider: {visible: false}}
            },
            { scrollZoom: true, dragmode: 'pan', displaylogo: false, displayModeBar: 'hover',
                modeBarButtons: [ ['pan2d'], ['zoom2d'], ['zoomIn2d'], ['zoomOut2d'] ] }
        );
        plotDiv.on('plotly_relayout',
            function(eventdata){
                if (plotDiv.is_server_defined_move === true) {
                    return
                }
                //if we're sending a string, then its a date object, and we want to send
                // a timestamp
                if (typeof(eventdata['xaxis.range[0]']) === 'string') {
                    eventdata = Object.assign({},eventdata);
                    eventdata["xaxis.range[0]"] = Date.parse(eventdata["xaxis.range[0]"]) / 1000.0;
                    eventdata["xaxis.range[1]"] = Date.parse(eventdata["xaxis.range[1]"]) / 1000.0;
                }

                let responseData = {
                    'event':'plot_layout',
                    'target_cell': '__identity__',
                    'data': eventdata
                };
                cellSocket.sendString(JSON.stringify(responseData));
            });
    }
}

//export {Plot, Plot as default};
