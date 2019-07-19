/**
 * Sheet Cell Component
 * NOTE: This is in part a wrapper
 * for handsontables.
 */
import {Component} from './Component';
import {h} from 'maquette';


/**
 * About Replacements
 * This component has one regular
 * replacement:
 * * `error`
 */

/**
 * About Named Children
 * --------------------
 * `error` (single) - An error cell if present
 */
class Sheet extends Component {
    constructor(props, ...args){
        super(props, ...args);

        this.currentTable = null;

        // Bind context to methods
        this.initializeTable = this.initializeTable.bind(this);
        this.initializeHooks = this.initializeHooks.bind(this);
        this.makeError = this.makeError.bind(this);

        /**
         * WARNING: The Cell version of Sheet is still using certain
         * postscripts because we have not yet refactored the socket
         * protocol.
         * Remove this warning about it once that happens!
         */
        console.warn(`[TODO] Sheet still uses certain postsceripts in its interaction. See component constructor comment for more information`);
    }

    componentDidLoad(){
        console.log(`#componentDidLoad called for Sheet ${this.props.id}`);
        console.log(`This sheet has the following replacements:`, this.replacements);
        this.initializeTable();
        if(this.props.extraData['handlesDoubleClick']){
            this.initializeHooks();
        }
        // Request initial data?
        cellSocket.sendString(JSON.stringify({
            event: "sheet_needs_data",
            target_cell: this.props.id,
            data: 0
        }));
    }

    render(){
        console.log(`Rendering sheet ${this.props.id}`);
        return (
            h('div', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Sheet",
                class: "cell"
            }, [
                h('div', {
                    id: `sheet${this.props.id}`,
                    class: "handsontable"
                }, [this.makeError()])
            ])
        );
    }

    initializeTable(){
        console.log(`#initializeTable called for Sheet ${this.props.id}`);
        let getProperty = function(index){
            return function(row){
                return row[index];
            };
        };
        let emptyRow = [];
        let dataNeededCallback = function(eventObject){
            eventObject.target_cell = this.props.id;
            cellSocket.sendString(JSON.stringify(eventObject));
        }.bind(this);
        let data = new SyntheticIntegerArray(this.props.extraData.rowCount, emptyRow, dataNeededCallback);
        let container = document.getElementById(`sheet${this.props.id}`);
        let columnNames = this.props.extraData.columnNames;
        let columns = columnNames.map((name, idx) => {
            emptyRow.push("");
            return {data: getProperty(idx)};
        });

        this.currentTable = new Handsontable(container, {
            data,
            dataSchema: function(opts){return {};},
            colHeaders: columnNames,
            columns,
            rowHeaders:true,
            rowHeaderWidth: 100,
            viewportRowRenderingOffset: 100,
            autoColumnSize: false,
            autoRowHeight: false,
            manualColumnResize: true,
            colWidths: this.props.extraData.columnWidth,
            rowHeights: 23,
            readOnly: true,
            ManualRowMove: false
        });
        handsOnTables[this.props.id] = {
            table: this.currentTable,
            lastCellClicked: {row: -100, col: -100},
            dblClicked: true
        };
    }

    initializeHooks(){
        Handsontable.hooks.add("beforeOnCellMouseDown", (event, data) => {
            let handsOnObj = handsOnTables[this.props.id];
            let lastRow = handsOnObj.lastCellClicked.row;
            let lastCol = handsOnObj.lastCellClicked.col;

            if((lastRow == data.row) && (lastCol = data.col)){
                handsOnObj.dblClicked = true;
                setTimeout(() => {
                    if(handsOnObj.dblClicked){
                        cellSocket.sendString(JSON.stringify({
                            event: 'onCellDblClick',
                            target_cell: this.props.id,
                            row: data.row,
                            col: data.col
                        }));
                    }
                    handsOnObj.lastCellClicked = {row: -100, col: -100};
                    handsOnObj.dblClicked = false;
                }, 200);
            } else {
                handsOnObj.lastCellClicked = {row: data.row, col: data.col};
                setTimeout(() => {
                    handsOnObj.lastCellClicked = {row: -100, col: -100};
                    handsOnObj.dblClicked = false;
                }, 600);
            }
        }, this.currentTable);

        Handsontable.hooks.add("beforeOnCellContextMenu", (event, data) => {
            let handsOnObj = handsOnTables[this.props.id];
            handsOnObj.dblClicked = false;
            handsOnObj.lastCellClicked = {row: -100, col: -100};
        }, this.currentTable);

        Handsontable.hooks.add("beforeContextMenuShow", (event, data) => {
            let handsOnObj = handsOnTables[this.props.id];
            handsOnObj.dblClicked = false;
            handsOnObj.lastCellClicked = {row: -100, col: -100};
        }, this.currentTable);
    }

    makeError(){
        if(this.usesReplacements){
            return this.getReplacementElementFor('error');
        } else {
            return this.renderChildNamed('error');
        }
    }
}

/** Copied over from Cells implementation **/
const SyntheticIntegerArray = function(size, emptyRow = [], callback){
    this.length = size;
    this.cache = {};
    this.push = function(){};
    this.splice = function(){};

    this.slice = function(low, high){
        if(high === undefined){
            high = this.length;
        }

        let res = Array(high - low);
        let initLow = low;
        while(low < high){
            let out = this.cache[low];
            if(out === undefined){
                if(callback){
                    callback({
                        event: 'sheet_needs_data',
                        data: low
                    });
                }
                out = emptyRow;
            }
            res[low - initLow] = out;
            low += 1;
        }
        return res;
    };
};

export {Sheet, Sheet as default};
