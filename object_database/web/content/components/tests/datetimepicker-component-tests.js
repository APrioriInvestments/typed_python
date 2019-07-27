var Component = require('../Component.js').Component;
var DateTimePicker = require('../DateTimePicker.js').DateTimePicker;
var h = require('maquette').h;
var assert = require('chai').assert;
var moment = require('moment');

describe('Datetimepicker Component Tests', () => {
    describe('Date Handling', () => {
        let unix_val = moment("2019-01-01T13:10:10").unix()
        var datetimepicker = new DateTimePicker({
            id: 'datetime-id',
            namedChildren: {},
            datetime: unix_val
        })
        it('Setting proper year value ranges', () => {
            let slider = h("input", {type: "range"}, [])
            let type = "year";
            datetimepicker.setSliderMinMaxValue(type, slider)
            assert.equal(slider.min, 1970);
            assert.equal(slider.max, 2030);
            assert.equal(slider.value, 2019);
        });
        it('Setting proper month value ranges', () => {
            let slider = h("input", {type: "range"}, [])
            let type = "month";
            datetimepicker.setSliderMinMaxValue(type, slider)
            assert.equal(slider.min, 1);
            assert.equal(slider.max, 12);
            assert.equal(slider.value, 1);
        });
        xit('Setting proper date value ranges', () => {
            let slider = h("input", {type: "range"}, [])
            let type = "date";
            let maxDate = moment(datetimepicker.year + "-" + datetimePicker.month, "YYYY-MM").daysInMonth();
            datetimepicker.setSliderMinMaxValue(type, slider)
            assert.equal(slider.min, 1);
            assert.equal(slider.max, maxDate);
            assert.equal(slider.value, 15);
        });
        it('Setting proper hour value ranges', () => {
            let slider = h("input", {type: "range"}, [])
            let type = "hour";
            datetimepicker.setSliderMinMaxValue(type, slider)
            assert.equal(slider.min, 0);
            assert.equal(slider.max, 23);
            // we need to use moment's hour here to account for potential timezone differences
            assert.equal(slider.value, datetimepicker._prepSliderValue(datetimepicker.hour));
        });
        it('Setting proper minute value ranges', () => {
            let slider = h("input", {type: "range"}, [])
            let type = "minute";
            datetimepicker.setSliderMinMaxValue(type, slider)
            assert.equal(slider.min, 0);
            assert.equal(slider.max, 59);
            assert.equal(slider.value, 10);
        });
        it('Setting proper second value ranges', () => {
            let slider = h("input", {type: "range"}, [])
            let type = "second";
            datetimepicker.setSliderMinMaxValue(type, slider)
            assert.equal(slider.min, 0);
            assert.equal(slider.max, 59);
            assert.equal(slider.value, 10);
        });
        it('Prepping input values', () => {
            let v = 12
            let prepped_v =datetimepicker._prepValue(v)
            let test_v = "12"
            assert.equal(prepped_v, test_v);
            v = "12"
            prepped_v =datetimepicker._prepValue(v)
            test_v = "12"
            assert.equal(prepped_v, test_v);
            v = 1
            prepped_v =datetimepicker._prepValue(v)
            test_v = "01"
            assert.equal(prepped_v, test_v);
        });
        it('Prepping slider values', () => {
            let v = "12"
            let prepped_v =datetimepicker._prepSliderValue(v)
            let test_v = 12
            assert.equal(prepped_v, test_v);
            v = "01"
            prepped_v =datetimepicker._prepSliderValue(v)
            test_v = 1
            assert.equal(prepped_v, test_v);
        });
    });
});
