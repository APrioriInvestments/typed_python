/**
 * Table Cell Component
 */
//import {Component} from './Component';


class Table extends Component {
    constructor(props, ...args){
        super(props, ...args);
        this.theadStyle = 'border-bottom: black;border-bottom-style:solid;border-bottom-width:thin;'

        // Bind context to methods
        this._getEvents = this._getEvent.bind(this);
        this._makeHeaderRow = this._makeHeaderRow.bind(this);
        this._makeRows = this._makeRows.bind(this);
    }

    render(){
        return(
            h('table', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Table",
                class: "cell table-hscroll table-sm table-striped"
            },[
                h("thead", {style: this.theadStyle} [this._makeHeaderRow()]),
                h("tbody", {}, [this._makeRows()])
            ]
            )
        );
    }

    _makeHeaderRow() {
        // TODO: this really looks like it's just a basic Card component but
        // the inner display text is changed
        let row = (
            h('tr', {style: 'vertical-align:top;'}, [
                h("div", {class: "card"}, [
                    h("div", {class: "card-body p-1"}, [
                        this.props.extraData.rowDisplayText
                    ])
                    
                ])
            ])
        );
        //TODO: add the rest of the header elements to it
        return row
    }

    _makeRows(){
        // TODO
        let rows = null
        return rows
    }

    _getEvent(event_name) {
        return this.props.extraData.events[event_name];
    }
}

//export {Table, Table as default};
