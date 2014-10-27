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

### predict(data, [computePosition])

Returns for each data point the coordinates of the corresponding best matching unit (BMU) on the grid

__Arguments__

* `data` - Data point or array of data points.
* `computePosition` - True if you want to compute the position of the point in the cell, using the direct neighbors (default: false). This option is currently only implemented for rectangular grids.

__Example__

```js
// create and train the som
var result1 = som.predict({ r: 45, g: 209, b: 100 });
// result1 = [ 2, 26 ]
var result2 = som.predict([{ r: 45, g: 209, b: 100 }, { r: 155, g: 22, b: 12 }]);
// result2 = [ [ 2, 26 ], [ 33, 12 ] ]
```

### getQuantizationError()

Returns the mean of the euclidean distance between each point of the training set and its corresponding BMU. This number can be used to compare several runs of the same SOM.

### export([includeDistance])

Exports the model to a JSON object that can be written to disk and reloaded

__Arguments__

* `includeDistance` - Boolean indicating if the distance function should be included in the model as a String (not recommended). Note that there is no need to include the default function and that it cannot work if the function depends on variables that are out of its scope (default: false).

### SOM.load(model, [distanceFunction])

Returns a new SOM instance based on the `model`. If the model was created with a custom distance function, the `distance` argument should be this function.

__Arguments__

* `model` - JSON object generated with `som.export()`
* `distanceFunction` - Optionally provide the distance function used to create the model.

## License

  MIT
