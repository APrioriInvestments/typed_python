/**
 * Tests for PropTypes and PropertyValidator
 */
var PropTypes = require('../util/PropertyValidator').PropTypes;
var assert = require('chai').assert;

PropTypes.silentMode = true; // To turn of console errors/warnings for testing

describe('Property Validation Tests', () => {
    describe('PropTypes tests', () => {
        it('#number validates only numbers', () => {
            assert.isTrue(PropTypes.number(null, null, 2));
            assert.isFalse(PropTypes.number(null, null, 'a string'));
            assert.isFalse(PropTypes.number(null, null, {some:'object'}));
            assert.isFalse(PropTypes.number(null, null, false));
            assert.isFalse(PropTypes.number(null, null, () =>  {}));
        });
        it('#string validates only strings', () => {
            assert.isTrue(PropTypes.string(null, null, "a string"));
            assert.isFalse(PropTypes.string(null, null, 2));
            assert.isFalse(PropTypes.string(null, null, false));
            assert.isFalse(PropTypes.string(null, null, {some: 'object'}));
            assert.isFalse(PropTypes.string(null, null, () =>  {}));
        });
        it('#boolean validates only booleans', () => {
            assert.isTrue(PropTypes.boolean(null, null, true));
            assert.isTrue(PropTypes.boolean(null, null, false));
            assert.isFalse(PropTypes.boolean(null, null, 0));
            assert.isFalse(PropTypes.boolean(null, null, 'false'));
            assert.isFalse(PropTypes.boolean(null, null, {some: 'object'}));
            assert.isFalse(PropTypes.boolean(null, null, () =>  {}));
        });
        it('#object validates only objects', () => {
            assert.isTrue(PropTypes.object(null, null, {}));
            assert.isFalse(PropTypes.object(null, null, true));
            assert.isFalse(PropTypes.object(null, null, 2));
            assert.isFalse(PropTypes.object(null, null, 'string'));
            assert.isFalse(PropTypes.object(null, null, () =>  {}));
        });
        it('#func validates only functions', () => {
            assert.isTrue(PropTypes.func(null, null, () => {}));
            assert.isFalse(PropTypes.func(null, null, 2));
            assert.isFalse(PropTypes.func(null, null, 'function'));
            assert.isFalse(PropTypes.func(null, null, false));
            assert.isFalse(PropTypes.func(null, null, {foo: () => {}}));
        });
        it('#oneOf validates PropType types correctly', () => {
            let exampleProp = {
                foo: PropTypes.oneOf([PropTypes.number, PropTypes.string])
            };
            assert.isTrue(exampleProp.foo('SomeComponent', 'foo', 2));
            assert.isTrue(exampleProp.foo('SomeComponent', 'foo', 'hello!'));
            assert.isFalse(exampleProp.foo('SomeComponent', 'foo', true));
            assert.isFalse(exampleProp.foo('SomeComponent', 'foo', {an: 'object'}));
        });
        it('#oneOf validates literal values correctly', () => {
            let exampleProp = {
                foo: PropTypes.oneOf(['literal', 2])
            };
            assert.isTrue(exampleProp.foo('SomeComponent', 'foo', 2));
            assert.isTrue(exampleProp.foo('SomeComponent', 'foo', 'literal'));
            assert.isFalse(exampleProp.foo('SomeComponent', 'foo', 'invalid'));
            assert.isFalse(exampleProp.foo('SomeComponent', 'foo', 1));
        });
        it('#oneOf validates mixed PropType and literal values correctly', () => {
            let exampleProp = {
                foo: PropTypes.oneOf([2, PropTypes.func])
            };
            assert.isTrue(exampleProp.foo('SomeComponent', 'foo', 2));
            assert.isTrue(exampleProp.foo('SomeComponent', 'foo', () => {}));
            assert.isFalse(exampleProp.foo('SomeComponent', 'foo', 'invalid'));
            assert.isFalse(exampleProp.foo('SomeComponent', 'foo', 0));
            assert.isFalse(exampleProp.foo('SomeComponent', 'foo', true));
            assert.isFalse(exampleProp.foo('SomeComponent', 'foo', {an: 'object'}));
        });
    });

    describe('Validation tests', () => {
        it('#validateRequired can validate a single prop based on requirement', () => {
            assert.isTrue(PropTypes.validateRequired('SomeComponent', 'foo', null, false));
            assert.isFalse(PropTypes.validateRequired('SomeComponent', 'foo', undefined, true));
        });
        it('#validateDescription correctly validates present and missing descriptions for a property', () => {
            let good = {description: "This is a description"};
            let bad = {description: ""};
            let empty = {};
            let nullDesc = {description: null};
            assert.isTrue(PropTypes.validateDescription('SomeComponent', 'foo', good));
            assert.isFalse(PropTypes.validateDescription('SomeComponent', 'foo', bad));
            assert.isFalse(PropTypes.validateDescription('SomeComponent', 'foo', nullDesc));
            assert.isFalse(PropTypes.validateDescription('SomeComponent', 'foo', {}));
        });
    });
    describe('Full scale validations', () => {
        it('Validates a correct structure', () => {
            let props = {
                padding: 2,
                someCallback: function(){},
                title: 'Some text here'
            };
            let propTypes = {
                padding: {
                    description: 'How much padding to add to the element',
                    required: true,
                    type: PropTypes.oneOf([PropTypes.number, PropTypes.string])
                },
                someCallback: {
                    description: "Callback for click events",
                    type: PropTypes.func
                },
                title: {
                    description: 'The title for the header',
                    required: false,
                    type: PropTypes.string
                }
            };
            assert.isTrue(PropTypes.validate('SomeComponent', props, propTypes));
        });
        it('Validates a non-required prop that is passed in as undefined', () => {
            let props = {
                something: undefined
            };
            let propTypes = {
                something: {
                    description: "Should validate",
                    required: false,
                    type: PropTypes.string
                }
            };
            assert.isTrue(PropTypes.validate('SomeComponent', props, propTypes));
        });
        it('Validates a non-required prop that is passed in as null', () => {
            let props = {
                something: null
            };
            let propTypes = {
                something: {
                    description: "Should validate",
                    required: false,
                    type: PropTypes.string
                }
            };
            assert.isTrue(PropTypes.validate('SomeComponent', props, propTypes));
        });
        it('Does not validate a required prop that is passed in as undefined', () => {
            let props = {
                something: undefined
            };
            let propTypes = {
                something: {
                    description: "Should validate",
                    required: true,
                    type: PropTypes.string
                }
            };
            assert.isFalse(PropTypes.validate('SomeComponent', props, propTypes));
        });
        it('Does not validate a required prop that is passed in as null', () => {
            let props = {
                something: null
            };
            let propTypes = {
                something: {
                    description: "Should validate",
                    required: true,
                    type: PropTypes.string
                }
            };
            assert.isFalse(PropTypes.validate('SomeComponent', props, propTypes));
        });
        it('Does not validate a prop with a missing description field', () => {
            let props = {
                something: undefined
            };
            let propTypes = {
                something: {
                    required: false,
                    type: PropTypes.string
                }
            };
            assert.isFalse(PropTypes.validate('SomeComponent', props, propTypes));
        });
        it('Does not validate a prop with an empty string for the description', () => {
            let props = {
                something: undefined
            };
            let propTypes = {
                something: {
                    description: "",
                    required: false,
                    type: PropTypes.string
                }
            };
            assert.isFalse(PropTypes.validate('SomeComponent', props, propTypes));
        });
        it('Does not validate when a prop is passed in that is not present on propTypes', () => {
            let props = {
                something: 200,
                anotherThing: "hello"
            };
            let propTypes = {
                yetAnotherThing: {
                    description: "This is described, but never passed in",
                    required: false,
                    type: PropTypes.string
                }
            };
            assert.isFalse(PropTypes.validate('SomeComponent', props, propTypes));
        });
    });
});
