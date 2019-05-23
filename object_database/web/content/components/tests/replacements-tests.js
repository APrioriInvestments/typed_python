var replacements = require('./replacement');
var ReplacementsHandler = replacements.ReplacementsHandler;
var chai = require('chai');
var assert = chai.assert;

describe('Smaller util functions', () => {
    it('Can make correct keys from name parts', () => {
        let parts = ['these', 'are', 'some', 'parts'];
        let result = ReplacementsHandler.keyFromNameParts(parts);
        assert.equal(result, 'these-are-some-parts');
    });
});

describe('Recursion #processDimension', () => {
    it('Can handle a 1 dimensional thing', () => {
        let name = '____child_3__';
        let remainingVals = [3];
        let rootDict = {};
        ReplacementsHandler.processDimension(remainingVals, rootDict, name);
        assert.equal(rootDict['3'], name);
    });
    it('Can handle a 2 dimensional thing', () => {
        let name = '____child_3_6__';
        let remainingVals = [3, 6];
        let rootDict = {};
        ReplacementsHandler.processDimension(remainingVals, rootDict, name);
        assert.isTrue(typeof rootDict['3'] == 'object');
        assert.equal(rootDict['3']['6'], name);
    });
    it('Can handle multiple 2 dimensional things', () => {
        let firstName = '____child_3_6__';
        let secondName = '____child_3_0__';
        let firstRemainingVals = [3, 6];
        let secondRemainingVals = [3, 0];
        let rootDict = {};
        ReplacementsHandler.processDimension(firstRemainingVals, rootDict, firstName);
        ReplacementsHandler.processDimension(secondRemainingVals, rootDict, secondName);
        assert.equal(rootDict['3']['6'], firstName);
        assert.equal(rootDict['3']['0'], secondName);
        assert.isTrue(typeof rootDict['3'] == 'object');
    });
});

describe('Sort enumerated dicts of varying dimensions #enumeratedValToSortedArray', () => {
    it('Can sort and Array-ify a 1-dimensional dict', () => {
        let example = {};
        for(let i = 0; i < 5; i++){
            example[i] = `____child_${i}__`;
        }
        let result = ReplacementsHandler.enumeratedValToSortedArray(example);
        assert.isTrue(Array.isArray(result));
        result.forEach((item, index) => {
            assert.equal(item, example[index]);
        });
    });
});

describe('Basic Replacements Tests #readReplacementString', () => {
    it('Can read a basic replacement', () => {
        let example = '____contents__';
        let result = ReplacementsHandler.readReplacementString(example);
        assert.isFalse(result.isEnumerated);
        assert.lengthOf(result.nameParts, 1);
        assert.lengthOf(result.enumeratedValues, 0);
        assert.equal(result.nameParts[0], 'contents');
    });
    it('Can read a list of enumerated replacements', () => {
        let example = '____child_5__';
        let result = ReplacementsHandler.readReplacementString(example);
        assert.isTrue(result.isEnumerated);
        assert.lengthOf(result.enumeratedValues, 1);
        assert.lengthOf(result.nameParts, 1);
        assert.equal(result.nameParts[0], 'child');
        assert.equal(result.enumeratedValues[0], 5);
    });
    it('Can read a multipart name', () => {
        let example = '____some_multipart_name__';
        let result = ReplacementsHandler.readReplacementString(example);
        assert.isFalse(result.isEnumerated);
        assert.lengthOf(result.nameParts, 3);
        assert.lengthOf(result.enumeratedValues, 0);
        assert.equal(result.nameParts[0], 'some');
        assert.equal(result.nameParts[1], 'multipart');
        assert.equal(result.nameParts[2], 'name');
    });
    it('Can read multidimensional enumerated values', () =>{
        let example = '____child_5_0__';
        let result = ReplacementsHandler.readReplacementString(example);
        assert.isTrue(result.isEnumerated);
        assert.lengthOf(result.enumeratedValues, 2);
        assert.equal(result.enumeratedValues[0], 5);
        assert.equal(result.enumeratedValues[1], 0);
    });
});

describe('ReplacementsHandler class tests', () => {
    describe('Basic Processing', () => {
        it('Can process regular replacements', () => {
            let examples = [
                '____header__',
                '____content__',
                '____left_bar__'
            ];
            let handler = new ReplacementsHandler(examples);
            assert.hasAllKeys(handler.regular, ['header', 'content', 'left-bar']);
            assert.equal(handler.regular['header'], examples[0]);
            assert.equal(handler.regular['content'], examples[1]);
            assert.equal(handler.regular['left-bar'], examples[2]);
        });

        it('Can process a 1-dimensional enumerated replacements', () => {
            let children = [];
            for(let i = 0; i < 3; i++){
                children.push(`____child_${i}__`);
            }
            let other = [];
            for(let i = 0; i < 5; i ++){
                other.push(`____other_${i}__`);
            }
            let examples = [...children, ...other];
            let handler = new ReplacementsHandler(examples);
            assert(handler.enumerated, ['child', 'other']);
            assert.isArray(handler.enumerated['child']);
            assert.isArray(handler.enumerated['other']);
            // Some random examples
            assert.equal(handler.enumerated.other[2], '____other_2__');
            assert.equal(handler.enumerated.child[0], '____child_0__');
        });

        it('Can process a single 2-dimensional correctly', () => {
            let examples = [];
            for(var i = 0; i < 3; i ++){
                for(var j = 0; j < 5; j++){
                    examples.push(`____child_${i}_${j}__`);
                }
            }
            let handler = new ReplacementsHandler(examples);
            assert.hasAllKeys(handler.enumerated, ['child']);
            assert.lengthOf(handler.enumerated.child, 3);
            assert.lengthOf(handler.enumerated.child[0], 5);
            // Random example from above
            assert.equal(handler.enumerated.child[2][2], '____child_2_2__');
        });
    });

    describe('Accessing methods', () => {
        it('#hasReplacement', () => {
            let examples = [
                '____child_0_0_0_0__',
                '____header__'
            ];
            let handler = new ReplacementsHandler(examples);
            let expectedTrue = handler.hasReplacement('child');
            let expectedFalse = handler.hasReplacement('should-not-be-there');
            assert.isTrue(expectedTrue);
            assert.isFalse(expectedFalse);
        });
        it('#getReplacementFor', () => {
            let examples = [
                '____header__',
                '____content__',
                '____child_0__',
                '____child_1__'
            ];
            let handler = new ReplacementsHandler(examples);
            let validResult = handler.getReplacementFor('header');
            let firstInvalidResult = handler.getReplacementFor('doesnt-exist');
            let secondInvalidResult = handler.getReplacementFor('child');
            assert.isNotNull(validResult);
            assert.equal(validResult, '____header__');
            assert.isNull(firstInvalidResult);
            assert.isNull(secondInvalidResult);
        });

        it('#getReplacementsFor', () => {
            let examples = ['____header__'];
            for(let i = 0; i < 5; i++){
                examples.push(`____child_${i}__`);
            }
            let handler = new ReplacementsHandler(examples);
            let validResult = handler.getReplacementsFor('child');
            let firstInvalidResult = handler.getReplacementsFor('header');
            let secondInvalidResult = handler.getReplacementsFor('doesnt-exist');
            assert.isNotNull(validResult);
            assert.isArray(validResult);
            assert.equal(validResult[1], '____child_1__');
            assert.isNull(firstInvalidResult);
            assert.isNull(secondInvalidResult);
        });

        it('#mapReplacementsFor: 1 dimensional', () => {
            let examples = [];
            for(var i = 0; i < 5; i++){
                examples.push(`____child_${i}__`);
            }
            let mapFunction = function(replacement){
                return replacement + '-modified';
            };
            let handler = new ReplacementsHandler(examples);
            let result = handler.mapReplacementsFor('child', mapFunction);
            assert.isArray(result);
            assert.lengthOf(result, 5);
            assert.equal(result[2], '____child_2__-modified');
        });
        it('#mapReplacementsFor: 2 dimensional', () => {
            let examples = [];
            for(let i = 0; i < 5; i++){
                for(let j = 0; j < 7; j++){
                    examples.push(`____child_${i}_${j}__`);
                }
            }
            let handler = new ReplacementsHandler(examples);
            let mapFunction = function(replacement){
                return replacement + '-modified';
            };
            let result = handler.mapReplacementsFor('child', mapFunction);
            assert.isArray(result);
            assert.isArray(result[0]);
            assert.equal(result[2][2], '____child_2_2__-modified');
        });
    });
});
