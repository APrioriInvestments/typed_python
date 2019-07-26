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
        this.setSliderMinMaxValue = this.setSliderMinMaxValue.bind(this);
        this._prepValue = this._prepValue.bind(this);
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

    /* Helper function that puts all the input elements, and their
     * respective separators into a nice datetime display
     */
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

    /* A basic input element
     * Note: the size is set here. For the moment we only have the year as size 4,
     * i.e. YYYY, and everything is size 2
     */
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

    /* This is the core input value change handler.
     * It handles two types of events: those requiring a WS call to the server
     * with the updated values (example: manual datetimeinput or releasing the slider)
     * and those requiring **only** a value update to the corresponding input element.
     * Note: we only send values back to the server that are valid dates. This prevents errors,
     * and also invalid dates occur naturally as a user manually updates the input value.
     */
    inputHandler(event, type, callback) {
        let value = event.target.value;
        value = this._prepValue(value);
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


    /* All our values are strings.
     * Months, date, hours, minutes, seconds are assumed to be two characters.
     * Example: "03" (not "3") for March
     */
    _prepValue(value) {
        value = value.toString()
        if (value.length === 1) {
            value = "0" + value;
        }
        return value;
    }
    /* Updates the corresponding class attribute with the given value
     */
    updateInputValue(value, type){
        switch (type) {
            case "year":
                this.year = value
                break;
            case "month":
                this.month = value
                break;
            case "date":
                this.date = value
                break;
            case "hour":
                this.hour = value
                break;
            case "minute":
                this.minute = value
                break;
            case "second":
                this.second = value
                break;
        }
    }

    /*Input Slider related helpers
     * ==========================*/

    /*Input Slider component
     * the input slider is made visible by a 'dblclick' event on any of the
     * datetime input elements and is hidden by a `click` event on its sibling
     * close icon.
     * Slider handles two types of events: `oninput` where the user is moving the slider
     * back and forth, i.e. mouse down, and on `onchange` where the user releases the slider,
     * i.e. mouse up. `oninput` triggers **only** an update to the input element, while
     * `onchange` triggers a WS send to the server. This minimizes unnecessary message chatter
     * and makes sure that the element is not re-rendered. Remember every callback to the server
     * updates the Slot object which in turns calls a cell.recalculate() causing a re-render.
     */
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
                    min: this.slider_min,
                    max: this.slider_max,
                    value: this.slider_value,
                    step:"1",
                    oninput: (event) => {this.inputHandler(event, this.slider, false)},
                    onchange: (event) => {this.inputHandler(event, this.slider, true)}
                }, []),
                h("span", {class: "octicon octicon-x", onclick: (event) => {this.hideSlider(event)}}, [])
            ]
        )
    }

    /* Sets the min, max and default value on the slider input element based on input type
     */
    setSliderMinMaxValue(type, slider){
        switch (type) {
            case "year":
                slider.min = 1970;
                slider.max = 2030;
                slider.value = 2019;
                break;
            case "month":
                slider.min = 1;
                slider.max = 12;
                slider.value = 6;
                break;
            case "date":
                slider.min = 1;
                slider.max = moment(this.year + "-" + this.month, "YYYY-MM").daysInMonth();
                slider.value = 15;
                break;
            case "hour":
                slider.min = 1;
                slider.max = 23;
                slider.value = 12;
                break;
            default:
                slider.min = 0;
                slider.max = 59;
                slider.value = 30;
                break;
        }
    }

    /* show and hide the slider
     * Note the use of node.closest() relying on the component element and id
     * structure staying the same
     */
    showSlider(event, type) {
        this.slider = type;
        let sibling = event.target.closest("#datetimepicker-" + this.props.id)
        let slider = sibling.nextSibling
        slider.classList.remove("invisible")
        this.setSliderMinMaxValue(type, slider.firstElementChild);
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
