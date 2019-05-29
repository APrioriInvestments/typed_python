/**
 * Grid Cell Component
 */

//import {Component} from './Component';
//import {h} from 'maquette';

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
class Grid extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
        this._makeHeaderElements = this._makeHeaderElements.bind(this);
        this._makeRowElements = this._makeRowElements.bind(this);
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
                    h('tr', {}, [topTableHeader, ...this._makeHeaderElements()])
                ]),
                h('tbody', {}, this._makeRowElements())
            ])
        );
    }

    _makeRowElements(){
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

    _makeHeaderElements(){
        return this.getReplacementElementsFor('header').map((headerEl, colIdx) => {
            return (
                h('th', {key: `${this.props.id}-grid-th-${colIdx}`}, [
                    headerEl
                ])
            );
        });
    }
}

//export {Grid, Grid as default};
