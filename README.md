# ml-som

self-organizing map (SOM) / Kohonen network

## Installation
```js
$ npm install ml-som
```

## Methods

### new SOM(x, y, [options])

Creates a new SOM instance with x * y dimensions.

__Arguments__

* `x` - Dimension of the x axis
* `y` - Dimension of the y axis
* `options` - Object with options for the algorithm

__Options__

* `fields` - Either a number (size of input vectors) or a map of field descriptions (to convert them to vectors)
* `iterations` - Number of iterations over the training set for the training phase (default: 10). The total number of training steps will be `iterations` * `trainingSet.length`
* `learningRate` - Multiplication coefficient for the learning algorithm (default: 0.1)
* `method` - Iteration method of the learning algorithm (default: random)
 *  `random` - Pick an object of the training set randomly
 *  `traverse` - Go sequentially through the training set
* `randomizer` - Function that must give numbers between 0 and 1 (default: Math.random)
* `distance` - Function that computes the distance between two vectors of the same length (default: square euclidean distance)
* `gridType` - Shape of the grid (default: rect)
 * `rect` - Rectangular grid
 * `hexa` - Hexagonal grid
* `torus` - Boolean indicating if the grid should be considered a torus for the selection of the neighbors (default: true)

__Example__

```js
var SOM = require('ml-som');
var options = {
  fields: {
    r: [0, 255],
    g: [0, 255],
    b: [0, 255]
  }
};

var som = new SOM(20, 20, options);
```

### train(trainingSet)

Train the SOM with the provided `trainingSet`.

__Arguments__

* `trainingSet` - Array of training elements. If the `fields` was a number, each array element must be a normalized vector. If it was an object, each array element must be an object with at least the described properties, within the described ranges

__Example__

```js
var trainingSet = [
  { r: 0, g: 0, b: 0 },
  { r: 255, g: 0, b: 0 },
  { r: 0, g: 255, b: 0 },
  { r: 0, g: 0, b: 255 },
  { r: 255, g: 255, b: 255 }
];

som.train(trainingSet);
```

### getConvertedNodes()

Returns a 2D array containing the nodes of the grid, in the structure described by the `fields` option.

### setTraining(trainingSet)

Set the training set for use with the next method

### trainOne()

Executes the next training iteration and returns true. Returns false if the training is over. Useful to draw the grid or compute some things after each learning step.

__Example__

```js
som.setTraining(trainingSet);
while(som.trainOne()) {
  var nodes = som.getConvertedNodes();
  // do something with the nodes
}
```

## License

  MIT
