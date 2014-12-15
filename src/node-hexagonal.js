'use strict';

var NodeSquare = require('./node-square');

function NodeHexagonal(x, y, weights, som) {

    NodeSquare.call(this, x, y, weights, som);

    this.hX = x - Math.floor(y / 2);
    this.z = 0 - this.hX - y;

}

NodeHexagonal.prototype = new NodeSquare();
NodeHexagonal.prototype.constructor = NodeHexagonal;

NodeHexagonal.prototype.getDistance = function getDistanceHexagonal(otherNode) {
    return Math.max(Math.abs(this.hX - otherNode.hX), Math.abs(this.y - otherNode.y), Math.abs(this.z - otherNode.z));
};

NodeHexagonal.prototype.getDistanceTorus = function getDistanceTorus(otherNode) {
    var distX = Math.abs(this.hX - otherNode.hX),
        distY = Math.abs(this.y - otherNode.y),
        distZ = Math.abs(this.z - otherNode.z);
    return Math.max(Math.min(distX, this.som.gridDim.x - distX), Math.min(distY, this.som.gridDim.y - distY), Math.min(distZ, this.som.gridDim.z - distZ));
};

NodeHexagonal.prototype.getPosition = function getPosition() {
    throw new Error('Unimplemented : cannot get position of the points for hexagonal grid');
};

module.exports = NodeHexagonal;
