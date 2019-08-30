/**
 * Table Cell Component
 */

import {Component} from './Component';
import {h} from 'maquette';


/**
 * About Replacements
 * ------------------
 * This component has 3 regular
 * replacements:
 * * `page`
 * * `left`
 * * `right`
 * This component has 2 enumerated
 * replacements:
 * * `child`
 * * `header`
 * NOTE: `child` enumerated replacements
 * are two dimensional arrays!
 */

/**
 * About Named Children
 * --------------------
 * `headers` (array) - An array of table header cells
 * `dataCells` (array-of-array) - A 2-dimensional array
 *    structures as rows by columns that contains the
 *    table data cells
 * `page` (single) - A cell that tells which page of the
 *     table we are looking at
 * `left` (single) - A cell that shows the number on the left
 * `right` (single) - A cell that show the number on the right
 */
class Table extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
        this.makeRows = this.makeRows.bind(this);
        this.makeFirstRow = this.makeFirstRow.bind(this);
        this._makeRowElements = this._makeRowElements.bind(this);
        this._theadStyle = this._theadStyle.bind(this);
        this._getRowDisplayElements = this._getRowDisplayElements.bind(this);
    }

    build(){
        return(
            h('table', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Table",
                class: "cell table-hscroll table-sm table-striped"
            }, [
                h('thead', {style: this._theadStyle()},[
                    this.makeFirstRow()
                ]),
                h('tbody', {}, this.makeRows())
            ])
        );
    }

    _theadStyle(){
        return "border-bottom: black;border-bottom-style:solid;border-bottom-width:thin;";
    }

    makeHeaderElements(){
        if(this.usesReplacements){
            return this.getReplacementElementsFor('header').map((replacement, idx) => {
                return h('th', {
                    style: "vertical-align:top;",
                    key: `${this.props.id}-table-header-${idx}`
                }, [replacement]);
            });
        } else {
            return this.renderChildrenNamed('headers').map((replacement, idx) => {
                return h('th', {
                    style: "vertical-align:top;",
                    key: `${this.props.id}-table-header-${idx}`
                }, [replacement]);
            });
        }
    }

    makeRows(){
        if(this.usesReplacements){
            return this._makeRowElements(this.getReplacementElementsFor('child'));
        } else {
            return this._makeRowElements(this.renderChildrenNamed('dataCells'));
        }
    }



    _makeRowElements(elements){
        // Note: rows are the *first* dimension
        // in the 2-dimensional array returned
        // by getting the `child` replacement elements.
        return elements.map((row, rowIdx) => {
            let columns = row.map((childElement, colIdx) => {
                return (
                    h('td', {
                        key: `${this.props.id}-td-${rowIdx}-${colIdx}`
                    }, [childElement])
                );
            });
            let indexElement = h('td', {}, [`${rowIdx + 1}`]);
            return (
                h('tr', {key: `${this.props.id}-tr-${rowIdx}`}, [indexElement, ...columns])
            );
        });
    }

    makeFirstRow(){
        let headerElements = this.makeHeaderElements();
        return(
            h('tr', {}, [
                h('th', {style: "vertical-align:top;"}, [
                    h('div', {class: "card"}, [
                        h('div', {class: "card-body p-1"}, [
                            ...this._getRowDisplayElements()
                        ])
                    ])
                ]),
                ...headerElements
            ])
        );
    }

    _getRowDisplayElements(){
        if(this.usesReplacements){
            return [
                this.getReplacementElementFor('left'),
                this.getReplacementElementFor('right'),
                this.getReplacementElementFor('page'),
            ];
        } else {
            return [
                this.renderChildNamed('left'),
                this.renderChildNamed('right'),
                this.renderChildNamed('page')
            ];
        }
    }
}

export {Table, Table as default};
