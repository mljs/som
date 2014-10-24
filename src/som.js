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

function SOM(x, y, options, reload) {

    this.x = x;
    this.y = y;

    this.options = {};
    for (var i in defaultOptions) {
        if (options.hasOwnProperty(i)) {
            this.options[i] = options[i];
        } else {
            this.options[i] = defaultOptions[i];
        }
    }

    if (typeof this.options.fields === 'number') {
        this.numWeights = this.options.fields;
        this.extractor = null;
    } else {
        var fields = Object.keys(this.options.fields);
        this.numWeights = fields.length;
        var converters = getConverters(fields, this.options.fields);
        this.extractor = converters.extractor;
        this.creator = converters.creator;
    }

    this.nodeType = this.options.gridType === 'rect' ? NodeSquare : NodeHexagonal;
    this.distanceMethod = this.options.torus ? 'getDistanceTorus' : 'getDistance';

    this.distance = this.options.distance;

    if (reload === true) { // For model loading
        this.done = true;
        return;
    }
    if (!(x > 0 && y > 0)) {
        throw new Error('x and y must be positive');
    }

    this.randomizer = this.options.randomizer;

    this.iterationCount = 0;
    this.iterations = this.options.iterations;

    this.startLearningRate = this.learningRate = this.options.learningRate;

    this.mapRadius = Math.floor(Math.max(x, y) / 2);

    this.algorithmMethod = this.options.method;

    this._initNodes();

    this.done = false;
}

SOM.load = function loadModel(model, distance) {
    if(model.name === 'SOM') {
        var x = model.data.length,
            y = model.data[0].length;
        if(distance) {
            model.options.distance = distance;
        } else if(model.options.distance) {
            model.options.distance = eval('('+model.options.distance+')');
        }
        var som = new SOM(x, y, model.options, true);
        som.nodes = new Array(x);
        var gridDim = som._getGridDim();
        for (var i = 0; i < x; i++) {
            som.nodes[i] = new Array(y);
            for (var j = 0; j < y; j++) {
                som.nodes[i][j] = new som.nodeType(i, j, model.data[i][j], gridDim);
            }
        }
        return som;
    } else {
        throw new Error('expecting a SOM model');
    }
};

SOM.prototype.export = function exportModel(includeDistance) {
    if (!this.done) {
        throw new Error('model is not ready yet');
    }
    var model = {
        name: 'SOM'
    };
    model.options = {
        fields: this.options.fields,
        gridType: this.options.gridType,
        torus: this.options.torus
    };
    model.data = new Array(this.x);
    for (var i = 0; i < this.x; i++) {
        model.data[i] = new Array(this.y);
        for (var j = 0; j < this.y; j++) {
            model.data[i][j] = this.nodes[i][j].weights;
        }
    }
    if (includeDistance) {
        model.options.distance = this.distance.toString();
    }
    return model;
};

SOM.prototype._getGridDim = function getGridDim() {
    if (this.nodeType === NodeSquare) {
        return {
            x: this.x,
            y: this.y
        };
    } else {
        var hx = this.x - Math.floor(this.y / 2);
        return {
            x: hx,
            y: this.y,
            z: -(0 - hx - this.y)
        }
    }
};

SOM.prototype._initNodes = function initNodes() {
    var i, j, k;
    this.nodes = new Array(this.x);
    var gridDim = this._getGridDim();
    for (i = 0; i < this.x; i++) {
        this.nodes[i] = new Array(this.y);
        for (j = 0; j < this.y; j++) {
            var weights = new Array(this.numWeights);
            for (k = 0; k < this.numWeights; k++) {
                weights[k] = this.randomizer();
            }
            this.nodes[i][j] = new this.nodeType(i, j, weights, gridDim);
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
    var x, y, dist, influence;
    var bmu = this._findBestMatchingUnit(trainingValue);

    var radiusLimit = Math.floor(neighbourhoodRadius);
    var xMin = bmu.x - radiusLimit,
        xMax = bmu.x + radiusLimit,
        yMin = bmu.y - radiusLimit,
        yMax = bmu.y + radiusLimit;

    for (x = xMin; x <= xMax; x++) {
        var theX = x;
        if (x < 0) {
            theX += this.x;
        } else if (x >= this.x) {
            theX -= this.x;
        }
        for (y = yMin; y <= yMax; y++) {
            var theY = y;
            if (y < 0) {
                theY += this.y;
            } else if (y >= this.y) {
                theY -= this.y;
            }

            dist = bmu[this.distanceMethod](this.nodes[theX][theY]);

            if (dist < neighbourhoodRadius) {
                influence = Math.exp(-dist / (2 * neighbourhoodRadius));
                this.nodes[theX][theY].adjustWeights(trainingValue, this.learningRate, influence);
            }

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
    var result = new Array(this.x);
    for (var i = 0; i < this.x; i++) {
        result[i] = new Array(this.y);
        for (var j = 0; j < this.y; j++) {
            var node = this.nodes[i][j];
            result[i][j] = this.creator ? this.creator(node.weights) : node.weights;
        }
    }
    return result;
};

SOM.prototype._findBestMatchingUnit = function findBestMatchingUnit(candidate) {

    var bmu,
        lowest = Infinity,
        dist;

    for (var i = 0; i < this.x; i++) {
        for (var j = 0; j < this.y; j++) {
            dist = this.distance(this.nodes[i][j].weights, candidate);
            if (dist < lowest) {
                lowest = dist;
                bmu = this.nodes[i][j];
            }
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