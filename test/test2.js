'use strict';

var SOM = require('../');

function rnd(n) {
    return Math.floor(Math.random() * n);
}

function rndColors(n) {
    var i = 0,
        a = new Array(n),
        o;
    for(; i < n; i++) {
        o = {
            r: rnd(256),
            g: rnd(256),
            b: rnd(256)
        };
        a[i] = o;
    }
    return a;
}

function getChart(theSet, prediction) {
    var chart = {
        x: [],
        y: [],
        info: []
    };
    var result = {
        type: 'chart',
        value: {
            data: [chart]
        }
    };

    var i = 0,
        l = theSet.length;

    for (; i < l; i++) {
        chart.x[i] = prediction[i][0] + prediction[i][2][0];
        chart.y[i] = prediction[i][1] + prediction[i][2][1];
        chart.info[i] = theSet[i];
    }
    return result;
}



var trainingSet = rndColors(10000);
var testSet = rndColors(1000);

var som = new SOM(20, 20, {
    iterations: 10,
    fields: [{
        name: 'r',
        range: [0, 255]
    }, {
        name: 'g',
        range: [0, 255]
    }, {
        name: 'b',
        range: [0, 255]
    }]
});

som.train(trainingSet);

//var predTraining = som.predict(true);
var predTest = som.predict(testSet, true);

var model = som.export();

var fs = require('fs');

fs.writeFileSync('./model.json', JSON.stringify(model));

fs.writeFileSync('./test.json', JSON.stringify(getChart(testSet, predTest)));

fs.writeFileSync('./umatrix.json', JSON.stringify(som.getUMatrix()));
