/**
 * Timestamp Cell Component
 */

import {Component} from './Component';
import {PropTypes} from './util/PropertyValidator';
import {h} from 'maquette';
import * as moment from 'moment';

/**
 * About Replacements
 * ------------------
 * This component contains no replacements
 */

/**
 * About Named Children
 * Thi component contains no named children
 */
class Timestamp extends Component {
    constructor(props, ...args){
        super(props, ...args);

        this.timeformat = 'YYYY-MM-D h:mm:ss'
        this.timezone = this.getCurrentTimeZone()
        this.timestamp = moment.unix(this.props.timestamp)
        // make sure user knows to hover over
        this.style = "cursor: default"

        // Bind component methods
        this.handleMouseover = this.handleMouseover.bind(this);
    }

    render(){
        return h('span',
            {
                class: "cell d-flex justify-content-center",
                style: this.style,
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "Timestamp",
                onmouseover: this.handleMouseover
            }, [
        h('span', {}, [this.timestamp.format(this.timeformat)]), // Date + time
        h('span', {style: "font-weight: 150"}, ["." + this.timestamp.format('ms ')]), // Milliseconds in lighter font
        // h('span', {}, [this.timestamp.format(' A')]), // AM/PM
    ]);
    }

    getCurrentTimeZone(){
        let now = new Date()
        // ex format: "14:16:26 GMT-0400 (Eastern Daylight Time)"
        now = now.toTimeString()
        // ex format: "Eastern Daylight Time"
        let tz = now.split("(")[1].slice(0, -1)
        return tz
    }

    /**
     * Dynamically update the title attribute with every
     * hover/mouseover to display the current time and this.timestamp
     */
    handleMouseover(event) {
        let timediff = moment().diff(this.timestamp, "seconds")
        event.target.title = this._timediffString(timediff) + "(" + this.timezone + ")"
    }

    /**
     * Takes a time difference in seconds (!) and returns a user-friendly string
     */
    _timediffString(timediff) {
        if (timediff === 1){
            return timediff + " second ago "
        } else if(timediff < 60){
            return timediff + " seconds ago "
        } else if (timediff < 3600) {
            let minutediff = Math.round(timediff/60)
            if (minutediff === 1) {
                return minutediff + " minute ago "
            } else {
                return minutediff + " minutes ago "
            }
        } else {
            let hourdiff = Math.round(timediff/3600)
            if (hourdiff === 1) {
                return hourdiff + " hour ago "
            } else {
                return hourdiff + " hours ago "
            }
        }
    }
}

Timestamp.propTypes = {
    timestamp: {
        description: "Unix seconds from epoch float.",
        type: PropTypes.oneOf([PropTypes.number])
    }
};

export {Timestamp, Timestamp as default};
