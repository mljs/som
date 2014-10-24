'use strict';

var NodeSquare = require('./node-square'),
    NodeHexagonal = require('./node-hexagonal');

var defaultOptions = {
    fields: 3,
    randomizer: Math.random,
    distance: squareEuclidean,
    iterations: 10,
    learningRate: 0.1,
    gridType: 'rect',
    torus: true,
    method: 'random'
};

function SOM(x, y, options) {
    if (!(x > 0 && y > 0)) {
        throw new Error('x and y must be positive');
    }
    options = options || {};
    for (var i in defaultOptions) {
        if (!options.hasOwnProperty(i)) {
            options[i] = defaultOptions[i];
        }
    }
    this.x = x;
    this.y = y;
    this.numNodes = x * y;

    this.randomizer = options.randomizer;
    this.distance = options.distance;

    this.iterationCount = 0;
    this.iterations = options.iterations;

    this.startLearningRate = this.learningRate = options.learningRate;

    if (typeof options.fields === 'number') {
        this.numWeights = options.fields;
        this.extractor = null;
    } else {
        var fields = Object.keys(options.fields);
        this.numWeights = fields.length;
        var converters = getConverters(fields, options.fields);
        this.extractor = converters.extractor;
        this.creator = converters.creator;
    }

    this.mapRadius = Math.floor(Math.max(x, y) / 2);

    this.nodeType = options.gridType === 'rect' ? NodeSquare : NodeHexagonal;
    this.distanceMethod = options.torus ? 'getDistanceTorus' : 'getDistance';

    this.algorithmMethod = options.method;

    this._initNodes();

    this.done = false;
}

SOM.prototype._initNodes = function initNodes() {

    var SOMNode = this.nodeType;
    this.nodes = new Array(this.numNodes);
    var gridDim;
    if (this.nodeType === NodeSquare) {
        gridDim = {
            x: this.x,
            y: this.y
        };
    } else {
        var hx = this.x - Math.floor(this.y / 2);
        gridDim = {
            x: hx,
            y: this.y,
            z: -(0 - hx - this.y)
        }
    }
    for (var i = 0; i < this.x; i++) {
        for (var j = 0; j < this.y; j++) {
            var weights = new Array(this.numWeights);
            for (var k = 0; k < this.numWeights; k++) {
                weights[k] = this.randomizer();
            }
            this.nodes[i * this.x + j] = new SOMNode(i, j, weights, gridDim);
        }
    }

};

SOM.prototype.setTraining = function setTraining(trainingSet) {
    if (this.trainingSet) {
        throw new Error('training set has already been set');
    }
    var convertedSet = trainingSet;
    var i, l = trainingSet.length;
    if (this.extractor) {
        convertedSet = new Array(trainingSet);
        for (i = 0; i < l; i++) {
            convertedSet[i] = this.extractor(trainingSet[i]);
        }
    }
    this.numIterations = this.iterations * l;

    if (this.algorithmMethod === 'random') {
        this.timeConstant = this.numIterations / Math.log(this.mapRadius);
    } else {
        this.timeConstant = l / Math.log(this.mapRadius);
    }
    this.trainingSet = convertedSet;
};

SOM.prototype.trainOne = function trainOne() {
    if (this.done) {

        return false;

    } else if (this.numIterations-- > 0) {

        var neighbourhoodRadius,
            trainingValue,
            trainingSetFactor;

        if (this.algorithmMethod === 'random') { // Pick a random value of the training set at each step
            neighbourhoodRadius = this.mapRadius * Math.exp(-this.iterationCount / this.timeConstant);
            trainingValue = getRandomValue(this.trainingSet, this.randomizer);
            this._adjust(trainingValue, neighbourhoodRadius);
            this.learningRate = this.startLearningRate * Math.exp(-this.iterationCount / this.numIterations);
        } else { // Get next input vector
            trainingSetFactor = - Math.floor(this.iterationCount / this.trainingSet.length);
            neighbourhoodRadius = this.mapRadius * Math.exp(trainingSetFactor / this.timeConstant);
            trainingValue = this.trainingSet[this.iterationCount % this.trainingSet.length];
            this._adjust(trainingValue, neighbourhoodRadius);
            if (((this.iterationCount + 1) % this.trainingSet.length) === 0) {
                this.learningRate = this.startLearningRate * Math.exp(trainingSetFactor / Math.floor(this.numIterations / this.trainingSet.length));
            }
        }

        this.iterationCount++;

        return true;

    } else {

        this.done = true;
        return false;

    }
};

SOM.prototype._adjust = function adjust(trainingValue, neighbourhoodRadius) {
    var i, dist, influence;
    var bmu = this._findBestMatchingUnit(trainingValue);
    for (i = 0; i < this.nodes.length; i++) {
        dist = bmu[this.distanceMethod](this.nodes[i]);
        if (dist < neighbourhoodRadius) {
            influence = Math.exp(-dist / (2 * neighbourhoodRadius));
            this.nodes[i].adjustWeights(trainingValue, this.learningRate, influence);
        }
    }
};

SOM.prototype.train = function train(trainingSet) {
    if (!this.done) {
        this.setTraining(trainingSet);
        while (this.trainOne()) {
        }
    }
};

SOM.prototype.getConvertedNodes = function getConvertedNodes() {
    var nodes = this.nodes;
    var result = [];
    for (var i = 0; i < nodes.length; i++) {
        var node = nodes[i];
        if (!result[node.x]) {
            result[node.x] = [];
        }
        result[node.x][node.y] = this.creator ? this.creator(node.weights) : node.weights;
    }
    return result;
};

SOM.prototype._findBestMatchingUnit = function findBestMatchingUnit(candidate) {

    var bmu,
        lowest = Infinity,
        dist;

    for (var i = 0; i < this.numNodes; i++) {
        dist = this.distance(this.nodes[i].weights, candidate);
        if (dist < lowest) {
            lowest = dist;
            bmu = this.nodes[i];
        }
    }

    return bmu;

};

SOM.prototype.predict = function predict(data) {
    if (data instanceof Array) {
        var self = this;
        return data.map(function (element) {
            return self._predict(element);
        });
    } else {
        return this._predict(data);
    }
};

SOM.prototype._predict = function _predict(element) {
    if (!(element instanceof Array)) {
        element = this.extractor(element);
    }
    var bmu = this._findBestMatchingUnit(element);
    return [bmu.x, bmu.y];
};

// As seen in http://www.scholarpedia.org/article/Kohonen_network
SOM.prototype.getQuantizationError = function getQuantizationError() {
    var data = this.trainingSet,
        l = data.length,
        sum = 0,
        bmu;
    for (var i = 0; i < l; i++) {
        bmu = this._findBestMatchingUnit(data[i]);
        sum += Math.sqrt(squareEuclidean(data[i], bmu.weights));
    }
    return sum / l;
};

function getConverters(fields, fieldsOpt) {
    var l = fields.length,
        normalizers = new Array(l),
        denormalizers = new Array(l);
    for (var i = 0; i < l; i++) {
        normalizers[i] = getNormalizer(fieldsOpt[fields[i]]);
        denormalizers[i] = getDenormalizer(fieldsOpt[fields[i]]);
    }
    return {
        extractor: function extractor(value) {
            var result = new Array(l);
            for (var i = 0; i < l; i++) {
                result[i] = normalizers[i](value[fields[i]]);
            }
            return result;
        },
        creator: function creator(value) {
            var result = {};
            for (var i = 0; i < l; i++) {
                result[fields[i]] = denormalizers[i](value[i]);
            }
            return result;
        }
    };
}

function getNormalizer(minMax) {
    return function normalizer(value) {
        return (value - minMax[0]) / (minMax[1] - minMax[0]);
    };
}

function getDenormalizer(minMax) {
    return function denormalizer(value) {
        return (minMax[0] + value * (minMax[1] - minMax[0]));
    };
}

function squareEuclidean(a, b) {
    var d = 0;
    for (var i = 0, ii = a.length; i < ii; i++) {
        d += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return d;
}

function getRandomValue(arr, randomizer) {
    return arr[Math.floor(randomizer() * arr.length)];
}

module.exports = SOM;