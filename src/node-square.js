'use strict';

function NodeSquare(x, y, weights, som) {
    this.x = x;
    this.y = y;
    this.weights = weights;
    this.som = som;
    this.neighbors = {};
}

NodeSquare.prototype.adjustWeights = function adjustWeights(target, learningRate, influence) {
    for (var i = 0, ii = this.weights.length; i < ii; i++) {
        this.weights[i] += learningRate * influence * (target[i] - this.weights[i]);
    }
};

NodeSquare.prototype.getDistance = function getDistance(otherNode) {
    return Math.max(Math.abs(this.x - otherNode.x), Math.abs(this.y - otherNode.y));
};

NodeSquare.prototype.getDistanceTorus = function getDistanceTorus(otherNode) {
    var distX = Math.abs(this.x - otherNode.x),
        distY = Math.abs(this.y - otherNode.y);
    return Math.max(Math.min(distX, this.som.gridDim.x - distX), Math.min(distY, this.som.gridDim.y - distY));
};

NodeSquare.prototype.getNeighbors = function getNeighbors(xy) {
    if (!this.neighbors[xy]) {
        this.neighbors[xy] = new Array(2);

        // left or bottom neighbor
        var v;
        if (this[xy] > 0) {
            v = this[xy] - 1;
        } else if (this.som.torus) {
            v = this.som.gridDim[xy] - 1;
        }
        if (typeof v !== 'undefined') {
            var x, y;
            if (xy === 'x') {
                x = v;
                y = this.y;
            } else {
                x = this.x;
                y = v;
            }
            this.neighbors[xy][0] = this.som.nodes[x][y];
        }

        // top or right neighbor
        var w;
        if (this[xy] < (this.som.gridDim[xy] - 1)) {
            w = this[xy] + 1;
        } else if (this.som.torus) {
            w = 0;
        }
        if (typeof w !== 'undefined') {
            if (xy === 'x') {
                x = w;
                y = this.y;
            } else {
                x = this.x;
                y = w;
            }
            this.neighbors[xy][1] = this.som.nodes[x][y];
        }
    }
    return this.neighbors[xy];
};

NodeSquare.prototype.getPos = function getPos(xy, element) {
    var neighbors = this.getNeighbors(xy),
        distance = this.som.distance,
        bestNeighbor,
        direction;
    if(neighbors[0]) {
        if (neighbors[1]) {
            var dist1 = distance(element, neighbors[0].weights),
                dist2 = distance(element, neighbors[1].weights);
            if(dist1 < dist2) {
                bestNeighbor = neighbors[0];
                direction = -1;
            } else {
                bestNeighbor = neighbors[1];
                direction = 1;
            }
        } else {
            bestNeighbor = neighbors[0];
            direction = -1;
        }
    } else {
        bestNeighbor = neighbors[1];
        direction = 1;
    }
    var simA = 1 - distance(element, this.weights),
        simB = 1 - distance(element, bestNeighbor.weights);
    var factor = ((simA - simB) / (2 - simA - simB));
    return 0.5 + 0.5 * factor * direction;
};

NodeSquare.prototype.getPosition = function getPosition(element) {
    return [
        this.getPos('x', element),
        this.getPos('y', element)
    ];
};

module.exports = NodeSquare;
