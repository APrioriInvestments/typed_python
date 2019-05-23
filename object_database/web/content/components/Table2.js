/**
 * Table Cell Component
 */

//import {Component} from './Component';
//import {h} from 'maquette';


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
class Table extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind context to methods
        this._makeHeaderElements = this._makeHeaderElements.bind(this);
        this._makeRowElements = this._makeRowElements.bind(this);
        this._makeFirstRowElement = this._makeFirstRowElement.bind(this);
        this._theadStyle = this._theadStyle.bind(this);
        this._getRowDisplayElements = this._getRowDisplayElements.bind(this);
    }

    render(){
        return(
            h('table', {
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Table",
                class: "cell table-hscroll table-sm table-striped"
            }, [
                h('thead', {style: this._theadStyle()},[
                    this._makeFirstRowElement()
                ]),
                h('tbody', {}, this._makeRowElements())
            ])
        );
    }

    _theadStyle(){
        return "border-bottom: black;border-bottom-style:solid;border-bottom-width:thin;";
    }

    _makeHeaderElements(){
        return this.getReplacementElementsFor('header').map((replacement, idx) => {
            return h('th', {
                style: "vertical-align:top;",
                key: `${this.props.id}-table-header-${idx}`
            }, [replacement]);
        });
    }

    _makeRowElements(){
        debugger;
        // Note: rows are the *first* dimension
        // in the 2-dimensional array returned
        // by getting the `child` replacement elements.
        return this.getReplacementElementsFor('child').map((row, rowIdx) => {
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

    _makeFirstRowElement(){
        let headerElements = this._makeHeaderElements();
        return(
            h('tr', {}, [
                h('th', {style: "vertical-align:top;"}, [
                    h('div', {class: "card"}, [
                        h('div', {class: "card-body p-1"}, [
                            ...this._getRowDisplayElements(),
                            ...headerElements
                        ])
                    ])
                ])
            ])
        );
    }

    _getRowDisplayElements(){
        return [
            this.getReplacementElementFor('left'),
            " ",
            this.getReplacementElementFor('right'),
            " Page ",
            this.getReplacementElementFor('page'),
            " of ",
            this.props.extraData.totalPages.toString()
        ];
    }
}

//export {Table, Table as default};
