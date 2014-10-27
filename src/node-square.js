function NodeSquare(x, y, weights, som) {
    this.x = x;
    this.y = y;
    this.weights = weights;
    this.som = som;
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

NodeSquare.prototype.getPosition = function getPosition(element) {
    var neighbors = this.neighbors || this.getNeighbors();
    var neighbor, dist;
    var position = [0.5, 0.5];
    for (var i = 0; i < neighbors.length; i++) {
        neighbor = neighbors[i];
        dist = 1 - this.som.distance(neighbor.weights, element) / this.som.maxDistance;
        var xFactor = (neighbor.x - this.x);
        if (xFactor > 1) {
            xFactor = -1;
        } else if (xFactor < -1) {
            xFactor = 1;
        }
        position[0] += xFactor * dist * 0.5;
        var yFactor = (neighbor.y - this.y);
        if (yFactor > 1) {
            yFactor = -1;
        } else if (yFactor < -1) {
            yFactor = 1;
        }
        position[1] += yFactor * dist * 0.5;
    }
    return position;
};

NodeSquare.prototype.getNeighbors = function getNeighbors() {
    var neighbors = [];
    for (var i = -1; i <= 1; i++) {
        var x = this.x + i;
        if (x < 0) {
            if (this.som.torus) {
                x = this.som.gridDim.x -1;
            } else {
                continue;
            }
        } else if (x === this.som.gridDim.x) {
            if (this.som.torus) {
                x = 0;
            } else {
                continue;
            }
        }
        for (var j = -1; j <= 1; j++) {
            if(i === 0 && j === 0) {
                continue;
            }
            var y = this.y + j;
            if (y < 0) {
                if (this.som.torus) {
                    y = this.som.gridDim.y -1;
                } else {
                    continue;
                }
            } else if (y === this.som.gridDim.y) {
                if (this.som.torus) {
                    y = 0;
                } else {
                    continue;
                }
            }
            neighbors.push(this.som.nodes[x][y]);
        }
    }
    return this.neighbors = neighbors;
};

module.exports = NodeSquare;