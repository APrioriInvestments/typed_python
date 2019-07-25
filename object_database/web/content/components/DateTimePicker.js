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
        // timeformat = 'YYYY-MM-DThh:mm:ss'
        this.datetime = moment.unix(this.props.datetime)
        this.year = this.datetime.format("YYYY");
        this.month = this.datetime.format("MM");
        this.date = this.datetime.format("D");
        this.hour = this.datetime.format("hh");
        this.minute = this.datetime.format("mm");
        this.second = this.datetime.format("ss");
        this.slider = null;

        // Bind component methods
        this.inputHandler = this.inputHandler.bind(this);
        this.inputElement = this.inputElement.bind(this);
        this.datetimePicker = this.datetimePicker.bind(this);
        this.updateInputValue = this.updateInputValue.bind(this);
        this.datetimeSlider = this.datetimeSlider.bind(this);
        this.showSlider = this.showSlider.bind(this);
        this.hideSlider = this.hideSlider.bind(this);
    }

    render(){
        return h('div',
            {
                class: "cell datetimepicker-wrapper",
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "DateTimePicker",
            }, [
                this.datetimePicker(),
                this.datetimeSlider()
            ]);
    }

    inputElement(type, value) {
        let size = 2;
        if (type == "year") {
            size = 4
        }

        // by convention and for styling we insist that all input is of length 2
        if (value.toString().length === 1) {
            value = "0" + value
        }

        return h(
            "input",
            {
                class: "datetimepicker",
                "data-cell-type": "DateTimePicker-" + type,
                id: "datetimepicker-" + type + "-" + this.props.id,
                type: "text",
                minsize: size,
                maxsize: size,
                size: size,
                value: value,
                oninput: (event) => {this.inputHandler(event, type, true)},
                ondblclick: (event) => {this.showSlider(event, type)}
            },
            []
        )

    }

    datetimePicker () {
        return h(
            "span",
            {
                id: "datetimepicker-" + this.props.id,
            },
            [
                this.inputElement("year", this.year),
                h("span", {}, ["/"]),
                this.inputElement("month", this.month),
                h("span", {}, ["/"]),
                this.inputElement("date", this.date),
                h("span", {}, [" "]),
                this.inputElement("hour", this.hour),
                h("span", {}, [":"]),
                this.inputElement("minute", this.minute),
                h("span", {}, [":"]),
                this.inputElement("second", this.second),
            ]
        )
    }

    inputHandler(event, type, callback) {
        let value = event.target.value;
        this.updateInputValue(value, type)
        if (callback) {
            // timeformat = 'YYYY-MM-DThh:mm:ss'
            let datetime = (this.year + "-" + this.month + "-" + this.date +
                "T" + this.hour + ":" + this.minute + ":" + this.second)
            let unix_val = moment(datetime).unix()
            // don't send back NaN since this can be caused by a "mid" or bad input
            if (!Number.isNaN(unix_val)) {
                cellSocket.sendString(
                    JSON.stringify(
                        {
                            "event": "input",
                            "target_cell": this.props.id,
                            "value": unix_val
                        }
                    )
                );
            }
        } else {
            // we need to update the corresponding input element manually
            let slider = event.target.closest("#datetimepicker-slider-" + this.props.id);
            let datetimepicker = slider.previousSibling
            let id = "datetimepicker-" + type + "-" + this.props.id
            let children = datetimepicker.childNodes
            for (var i = 0; i < children.length; i++) {
                if (children[i].id == id) {
                    let input = children[i]
                    input.value = value;
                    break;
                }
            }
        }
    }

    updateInputValue(value, type){
        // TODO: we need a validator here!
        if (type === "year") {
            this.year = value
        } else if (type === "month") {
            this.month = value
        } else if (type === "date") {
            this.date = value
        } else if (type === "hour") {
            this.hour = value
        } else if (type === "minute") {
            this.minute = value
        } else if (type === "second") {
            this.second = value
        }
    }

    /*Input Slider related helpers
     * ==========================*/

    datetimeSlider() {
        return h(
            "div",
            {
                id: "datetimepicker-slider-" + this.props.id,
                class: "invisible datetimeslider",
            },
            [
                h("input", {
                    type:"range",
                    min:"2001",
                    max:"2020",
                    value:"2010",
                    step:"1",
                    oninput: (event) => {this.inputHandler(event, this.slider, false)},
                    onchange: (event) => {this.inputHandler(event, this.slider, true)}
                }, []),
                h("span", {class: "octicon octicon-x", onclick: (event) => {this.hideSlider(event)}}, [])
            ]
        )
    }

    showSlider(event, type) {
        this.slider = type;
        let sibling = event.target.closest("#datetimepicker-" + this.props.id)
        let slider = sibling.nextSibling
        slider.classList.remove("invisible")
        console.log('going to show slider: ' + this.slider)
    }

    hideSlider(event) {
        this.slider = null;
        let slider = event.target.closest("#datetimepicker-slider-" + this.props.id)
        slider.classList.add("invisible")
    }
}

DateTimePicker.propTypes = {
    datetime: {
        description: "Start datetime in (unix) seconds from epoch.",
        type: PropTypes.oneOf([PropTypes.number])
    }
};

export {DateTimePicker, DateTimePicker as default};
