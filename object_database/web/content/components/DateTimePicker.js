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
 * This component contains two
 * regular replacements:
 * * `contents`
 * * `header`
 */

/**
 * About Named Children
 * `body` (single) - The cell to put in the body of the DateTimePicker
 * `header` (single) - An optional header cell to put above
 *        body
 *        <label for="meeting-time">Choose a time for your appointment:</label>

<input type="datetime-local" id="meeting-time"
       name="meeting-time" value="2018-06-12T19:30"
       min="2018-06-07T00:00" max="2018-06-14T00:00">
 */
class DateTimePicker extends Component {
    constructor(props, ...args){
        super(props, ...args);
        this.timeformat = 'YYYY-MM-DTh:mm:ss'
        this.datetime = moment.unix(this.props.datetime).format(this.timeformat)

        // Bind component methods
    }

    render(){
        return h('div',
            {
                class: "cell",
                style: this.props.divStyle,
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "DateTimePicker"
            }, [
                h("input",
                    {
                        type: "datetime-local",
                        id: "datetimepicker-" + this.props.id,
                        value: this.datetime
                    },
                    []
                )
            ]);
    }

}

DateTimePicker.propTypes = {
    datetime: {
        description: "Start datetime in (unix) seconds from epoch.",
        type: PropTypes.oneOf([PropTypes.number])
    },
    divStyle: {
        description: "HTML style attribute string.",
        type: PropTypes.oneOf([PropTypes.string])
    }
};

export {DateTimePicker, DateTimePicker as default};
