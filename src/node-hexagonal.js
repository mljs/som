var NodeSquare = require('./node-square');

function NodeHexagonal(x, y, weights) {

    NodeSquare.call(this, x, y, weights);

    this.hX = x - Math.floor(y / 2);
    this.z = 0 - this.hX - y;

}

NodeHexagonal.prototype = new NodeSquare;
NodeHexagonal.prototype.constructor = NodeHexagonal;

NodeHexagonal.prototype.getDistance = function getDistanceHexagonal(otherNode) {
    return Math.max(Math.abs(this.hX - otherNode.hX), Math.abs(this.y - otherNode.y), Math.abs(this.z - otherNode.z));
};

module.exports = NodeHexagonal;