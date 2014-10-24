'use strict';

var SOM = require('../');

describe('SOM', function () {
    it('should init correctly', function () {
        var som = new SOM(4, 4);
        (function () {
            var som = new SOM(4);
        }).should.throw('x and y must be positive');
    });
    it('should work SOM 1', function () {
        var som = new SOM(40, 40, {
            fields: {
                r: [0, 255],
                g: [0, 255],
                b: [0, 255]
            },
            iterations: 1000,
            gridType: 'rect'
        });
        var data = [
            { r: 255, g: 0, b: 0 },
            { r: 0, g: 255, b: 0 },
            { r: 0, g: 0, b: 255},
            { r: 0, g: 0, b: 0},
            { r: 255, g: 255, b: 255},
            { r: 255, g: 255, b: 0 },
            { r: 255, g: 0, b: 255},
            { r: 0, g: 255, b: 255}
        ];
        som.train(data);
    });
    it('should work SOM 2', function () {
        var som = new SOM(40, 40, {
            fields: {
                r: [0, 255],
                g: [0, 255],
                b: [0, 255]
            },
            iterations: 10,
            gridType: 'rect'
        });
        var data = [
            { r: 255, g: 0, b: 0 },
            { r: 0, g: 255, b: 0 },
            { r: 0, g: 0, b: 255},
            { r: 0, g: 0, b: 0},
            { r: 255, g: 255, b: 255},
            { r: 255, g: 255, b: 0 },
            { r: 255, g: 0, b: 255},
            { r: 0, g: 255, b: 255}
        ];
        som.setTraining(data);
        while (som.trainOne()) {
        }
        var result = som.predict(som.trainingSet);
    });
    it('should work SOM 3', function () {
        var som = new SOM(20, 20, {
            fields: {
                r: [0, 255],
                g: [0, 255],
                b: [0, 255]
            },
            iterations: 10,
            method: 'traverse'
        });
        var data = getRandomData(1000);
        som.train(data);
    });
});

function getRandomData(qty) {
    var data = new Array(qty);
    for (var i = 0; i < qty; i++) {
        data[i] = getRandomColor();
    }
    return data;
}

function getRandomColor() {
    return {
        r: getRandomValue(),
        g: getRandomValue(),
        b: getRandomValue()
    };
}

function getRandomValue() {
    return Math.floor(Math.random() * 255);
}