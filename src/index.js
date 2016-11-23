'use strict';

var squareEuclidean = require('ml-euclidean-distance').squared;

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

    options = options || {};
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
    } else if (Array.isArray(this.options.fields)) {
        this.numWeights = this.options.fields.length;
        var converters = getConverters(this.options.fields);
        this.extractor = converters.extractor;
        this.creator = converters.creator;
    } else {
        throw new Error('Invalid fields definition');
    }

    if (this.options.gridType === 'rect') {
        this.NodeType = NodeSquare;
        this.gridDim = {
            x: x,
            y: y
        };
    } else {
        this.NodeType = NodeHexagonal;
        var hx = this.x - Math.floor(this.y / 2);
        this.gridDim = {
            x: hx,
            y: this.y,
            z: -(0 - hx - this.y)
        };
    }

    this.torus = this.options.torus;
    this.distanceMethod = this.torus ? 'getDistanceTorus' : 'getDistance';

    this.distance = this.options.distance;

    this.maxDistance = getMaxDistance(this.distance, this.numWeights);

    if (reload === true) { // For model loading
        this.done = true;
        return;
    }
    if (!(x > 0 && y > 0)) {
        throw new Error('x and y must be positive');
    }

    this.times = {
        findBMU: 0,
        adjust: 0
    };

    this.randomizer = this.options.randomizer;

    this.iterationCount = 0;
    this.iterations = this.options.iterations;

    this.startLearningRate = this.learningRate = this.options.learningRate;

    this.mapRadius = Math.floor(Math.max(x, y) / 2);

    this.algorithmMethod = this.options.method;

    this.initNodes();

    this.done = false;
}

SOM.load = function loadModel(model, distance) {
    if (model.name === 'SOM') {
        var x = model.data.length,
            y = model.data[0].length;
        if (distance) {
            model.options.distance = distance;
        }
        var som = new SOM(x, y, model.options, true);
        som.nodes = new Array(x);
        for (var i = 0; i < x; i++) {
            som.nodes[i] = new Array(y);
            for (var j = 0; j < y; j++) {
                som.nodes[i][j] = new som.NodeType(i, j, model.data[i][j], som);
            }
        }
        return som;
    } else {
        throw new Error('expecting a SOM model');
    }
};

SOM.prototype.export = function exportModel() {
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
    if (!this.done) {
        model.ready = false;
    }
    return model;
};

SOM.prototype.initNodes = function initNodes() {
    var now = Date.now(),
        i, j, k;
    this.nodes = new Array(this.x);
    for (i = 0; i < this.x; i++) {
        this.nodes[i] = new Array(this.y);
        for (j = 0; j < this.y; j++) {
            var weights = new Array(this.numWeights);
            for (k = 0; k < this.numWeights; k++) {
                weights[k] = this.randomizer();
            }
            this.nodes[i][j] = new this.NodeType(i, j, weights, this);
        }
    }
    this.times.initNodes = Date.now() - now;
};

SOM.prototype.setTraining = function setTraining(trainingSet) {
    if (this.trainingSet) {
        throw new Error('training set has already been set');
    }
    var now = Date.now();
    var convertedSet = trainingSet;
    var i, l = trainingSet.length;
    if (this.extractor) {
        convertedSet = new Array(l);
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
    this.times.setTraining = Date.now() - now;
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
            this.adjust(trainingValue, neighbourhoodRadius);
            this.learningRate = this.startLearningRate * Math.exp(-this.iterationCount / this.numIterations);
        } else { // Get next input vector
            trainingSetFactor = -Math.floor(this.iterationCount / this.trainingSet.length);
            neighbourhoodRadius = this.mapRadius * Math.exp(trainingSetFactor / this.timeConstant);
            trainingValue = this.trainingSet[this.iterationCount % this.trainingSet.length];
            this.adjust(trainingValue, neighbourhoodRadius);
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

SOM.prototype.adjust = function adjust(trainingValue, neighbourhoodRadius) {
    var now = Date.now(),
        x, y, dist, influence;

    var bmu = this.findBestMatchingUnit(trainingValue);

    var now2 = Date.now();
    this.times.findBMU += now2 - now;

    var radiusLimit = Math.floor(neighbourhoodRadius);
    var xMin = (bmu.x - radiusLimit ) % this.x,
        xMax = (bmu.x + radiusLimit) % this.x,
        yMin = (bmu.y - radiusLimit) % this.y,
        yMax = (bmu.y + radiusLimit) % this.y;
if (xMin<0) xMin+=this.x;
    // TODO in loop

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

    this.times.adjust += (Date.now() - now2);

};

SOM.prototype.train = function train(trainingSet) {
    if (!this.done) {
        this.setTraining(trainingSet);
        var needTrain = true;
        while (needTrain){
            needTrain = this.trainOne();
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

SOM.prototype.findBestMatchingUnit = function findBestMatchingUnit(candidate) {

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

SOM.prototype.predict = function predict(data, computePosition) {
    if (typeof data === 'boolean') {
        computePosition = data;
        data = null;
    }
    if (!data) {
        data = this.trainingSet;
    }
    if (Array.isArray(data) && (Array.isArray(data[0]) || (typeof data[0] === 'object'))) { // predict a dataset
        var self = this;
        return data.map(function (element) {
            return self.predictOne(element, computePosition);
        });
    } else { // predict a single element
        return this.predictOne(data, computePosition);
    }
};

SOM.prototype.predictOne = function predictOne(element, computePosition) {
    if (!Array.isArray(element)) {
        element = this.extractor(element);
    }
    var bmu = this.findBestMatchingUnit(element);
    var result = [bmu.x, bmu.y];
    if (computePosition) {
        result[2] = bmu.getPosition(element);
    }
    return result;
};

// As seen in http://www.scholarpedia.org/article/Kohonen_network
SOM.prototype.getQuantizationError = function getQuantizationError() {
    var fit = this.getFit(),
        l = fit.length,
        sum = 0;
    for (var i = 0; i < l; i++) {
        sum += fit[i];
    }
    return sum / l;
};

SOM.prototype.getFit = function getFit(dataset) {
    if (!dataset) {
        dataset = this.trainingSet;
    }
    var l = dataset.length,
        bmu,
        result = new Array(l);
    for (var i = 0; i < l; i++) {
        bmu = this.findBestMatchingUnit(dataset[i]);
        result[i] = Math.sqrt(this.distance(dataset[i], bmu.weights));
    }
    return result;
};

SOM.prototype.getUMatrix = function getUMatrix() {
    var matrix = new Array(this.x);
    for (var i = 0; i < this.x; i++) {
        matrix[i] = new Array(this.y);
        for (var j = 0; j < this.y; j++) {
            var node = this.nodes[i][j],
                nX = node.getNeighbors('x'),
                nY = node.getNeighbors('y');
            var sum = 0,
                total = 0,
                self = this;
            if(nX[0]) {
                total++;
                sum += self.distance(node.weights, nX[0].weights);
            }
            if(nX[1]) {
                total++;
                sum += self.distance(node.weights, nX[1].weights);
            }
            if(nY[0]) {
                total++;
                sum += self.distance(node.weights, nY[0].weights);
            }
            if(nY[1]) {
                total++;
                sum += self.distance(node.weights, nY[1].weights);
            }
            matrix[i][j] = sum / total;
        }
    }
    return matrix;
};

function getConverters(fields) {
    var l = fields.length,
        normalizers = new Array(l),
        denormalizers = new Array(l),
        range;
    for (var i = 0; i < l; i++) {
        range = fields[i].range;
        normalizers[i] = getNormalizer(range[0], range[1]);
        denormalizers[i] = getDenormalizer(range[0], range[1]);
    }
    return {
        extractor: function extractor(value) {
            var result = new Array(l);
            for (var j = 0; j < l; j++) {
                result[j] = normalizers[j](value[fields[j].name]);
            }
            return result;
        },
        creator: function creator(value) {
            var result = {};
            for (var j = 0; j < l; j++) {
                result[fields[j].name] = denormalizers[j](value[j]);
            }
            return result;
        }
    };
}

function getNormalizer(min, max) {
    return function normalizer(value) {
        return (value - min) / (max - min);
    };
}

function getDenormalizer(min, max) {
    return function denormalizer(value) {
        return (min + value * (max - min));
    };
}

function getRandomValue(arr, randomizer) {
    return arr[Math.floor(randomizer() * arr.length)];
}

function getMaxDistance(distance, numWeights) {
    var zero = new Array(numWeights),
        one = new Array(numWeights);
    for (var i = 0; i < numWeights; i++) {
        zero[i] = 0;
        one[i] = 1;
    }
    return distance(zero, one);
}

module.exports = SOM;
