function NodeSquare(x, y, weights) {
    this.x = x;
    this.y = y;
    this.weights = weights;
}

NodeSquare.prototype.adjustWeights = function adjustWeights(target, learningRate, influence) {
    for (var i = 0, ii = this.weights.length; i < ii; i++) {
        this.weights[i] += learningRate * influence * (target[i] - this.weights[i]);
    }
};

NodeSquare.prototype.getDistance = function getDistance(otherNode) {
    return Math.max(Math.abs(this.x - otherNode.x), Math.abs(this.y - otherNode.y));
};

module.exports = NodeSquare;