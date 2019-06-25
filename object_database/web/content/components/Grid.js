/**
 * Grid Cell Component
 */

import {Component} from './Component';
import {h} from 'maquette';

/**
 * About Replacements
 * ------------------
 * This component has 3 enumerable
 * replacements:
 * * `header`
 * * `rowlabel`
 * * `child`
 *
 * NOTE: Child is a 2-dimensional
 * enumerated replacement!
 */

/**
 * About Named Children
 * --------------------
 * `headers` (array) - An array of table header cells
 * `rowLabels` (array) - An array of row label cells
 * `dataCells` (array-of-array) - A 2-dimensional array
 *     of cells that serve as table data, where rows
 *     are the outer array and columns are the inner
 *     array.
 */
class Grid extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
        this.makeHeaders = this.makeHeaders.bind(this);
        this.makeRows = this.makeRows.bind(this);
        this._makeReplacementHeaderElements = this._makeReplacementHeaderElements.bind(this);
        this._makeReplacementRowElements = this._makeReplacementRowElements.bind(this);
    }

    render(){
        let topTableHeader = null;
        if(this.props.extraData.hasTopHeader){
            topTableHeader = h('th');
        }
        return (
            h('table', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Grid",
                class: "cell table-hscroll table-sm table-striped"
            }, [
                h('thead', {}, [
                    h('tr', {}, [topTableHeader, ...this.makeHeaders()])
                ]),
                h('tbody', {}, this.makeRows())
            ])
        );
    }

    makeHeaders(){
        if(this.usesReplacements){
            return this._makeReplacementHeaderElements();
        } else {
            return this.renderChildrenNamed('headers').map((headerEl, colIdx) => {
                return (
                    h('th', {key: `${this.props.id}-grid-th-${colIdx}`}, [
                        headerEl
                    ])
                );
            });
        }
    }

    makeRows(){
        if(this.usesReplacements){
            return this._makeReplacementRowElements();
        } else {
            return this.renderChildrenNamed('dataCells').map((dataRow, rowIdx) => {
                let columns = dataRow.map((column, colIdx) => {
                    return(
                        h('td', {key: `${this.props.id}-grid-col-${rowIdx}-${colIdx}`}, [
                            column
                        ])
                    );
                });
                let rowLabelEl = null;
                if(this.props.namedChildren.rowLabels && this.props.namedChildren.rowLabels.length > 0){
                    rowLabelEl = h('th', {key: `${this.props.id}-grid-col-${rowIdx}-${colIdx}`}, [
                        this.props.namedChildren.rowLabels[rowIdx].render()
                    ]);
                }
                return (
                    h('tr', {key: `${this.props.id}-grid-row-${rowIdx}`}, [
                        rowLabelEl,
                        ...columns
                    ])
                );
            });
        }
    }

    _makeReplacementRowElements(){
        return this.getReplacementElementsFor('child').map((row, rowIdx) => {
            let columns = row.map((column, colIdx) => {
                return (
                    h('td', {key: `${this.props.id}-grid-col-${rowIdx}-${colIdx}`}, [
                        column
                    ])
                );
            });
            let rowLabelEl = null;
            if(this.replacements.hasReplacement('rowlabel')){
                rowLabelEl = h('th', {key: `${this.props.id}-grid-rowlbl-${rowIdx}`}, [
                    this.getReplacementElementsFor('rowlabel')[rowIdx]
                ]);
            }
            return (
                h('tr', {key: `${this.props.id}-grid-row-${rowIdx}`}, [
                    rowLabelEl,
                    ...columns
                ])
            );
        });
    }

    _makeReplacementHeaderElements(){
        return this.getReplacementElementsFor('header').map((headerEl, colIdx) => {
            return (
                h('th', {key: `${this.props.id}-grid-th-${colIdx}`}, [
                    headerEl
                ])
            );
        });
    }
}

export
{Grid, Grid as default};
