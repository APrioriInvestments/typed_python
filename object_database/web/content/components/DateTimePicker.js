/**
 * DateTimePicker Cell Component
 */

import {Component} from './Component';
import {PropTypes} from './util/PropertyValidator';
import {h} from 'maquette';
import * as moment from 'moment';

/**
 * About Replacements
 * ------------------
 * This component contains no
 * regular replacements
 */

/**
 * About Named Children
 * --------------------
 *  This component has no children
 */
class DateTimePicker extends Component {
    constructor(props, ...args){
        super(props, ...args);
        this.timeformat = 'YYYY-MM-DThh:mm:ss'
        this.datetime = moment.unix(this.props.datetime).format(this.timeformat)

        // Bind component methods
        this.changeHandler = this.changeHandler.bind(this);
    }

    render(){
        return h('div',
            {
                class: "cell",
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "DateTimePicker"
            }, [
                h("input",
                    {
                        type: "datetime-local",
                        id: "datetimepicker-" + this.props.id,
                        value: moment.unix(this.props.datetime).format(this.timeformat),
                        onchange: this.changeHandler
                    },
                    []
                )
            ]);
    }

    changeHandler(event) {
        // don't send back NaN
        let unix_val = moment(event.target.value).unix()
        if (unix_val !== NaN) {
            cellSocket.sendString(
                JSON.stringify(
                    {
                        "event": "change",
                        "target_cell": this.props.id,
                        "value": unix_val
                    }
                )
            );
        }
        event.preventDefault();
    }
}

DateTimePicker.propTypes = {
    datetime: {
        description: "Start datetime in (unix) seconds from epoch.",
        type: PropTypes.oneOf([PropTypes.number])
    }
};

export {DateTimePicker, DateTimePicker as default};
