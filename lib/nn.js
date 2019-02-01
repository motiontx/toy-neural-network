function newRandomMatrix(row, columns) {
  let array = [];
  for (let i = 0; i < row; i++) {
    let row = [];
    for (let j = 0; j < columns; j++) {
      row.push(math.random(-1, 1));
    }
    array.push(row);
  }
  return (math.matrix(array));
}

function InputFormat(input) {
  let array = [];
  for (var i = 0; i < input.length; i++) {
    array.push([input[i]]);
  }
  return (math.matrix(array));
}

function outputFormat(output) {
  let array = [];
  math.forEach(output, (x) => array.push(x));
  return array;
}

let sigmoid = x => 1 / (1 + math.exp(-x));

let dSigmoid = x => x * (1 - x);

class NeuralNetwork {
  constructor(i_nodes, h_nodes, o_nodes, l_rate) {
    this.input_nodes = i_nodes;
    this.hidden_nodes = h_nodes;
    this.output_nodes = o_nodes;

    this.learning_rate = l_rate;

    this.w_ih = newRandomMatrix(this.hidden_nodes, this.input_nodes);
    this.w_ho = newRandomMatrix(this.output_nodes, this.hidden_nodes);

    this.bias_h = newRandomMatrix(this.hidden_nodes, 1);
    this.bias_o = newRandomMatrix(this.output_nodes, 1);
  }

  query(input_array) {
    let inputs = InputFormat(input_array);
    let hidden = math.multiply(this.w_ih, inputs);
    hidden = math.add(hidden, this.bias_h);
    hidden = math.map(hidden, sigmoid);

    let outputs = math.multiply(this.w_ho, hidden);
    outputs = math.add(outputs, this.bias_o);
    outputs = math.map(outputs, sigmoid);

    return outputFormat(outputs);
  }

  train(input_array, target_array) {
    let targets = InputFormat(target_array);

    let inputs = InputFormat(input_array);
    let hidden = math.multiply(this.w_ih, inputs);
    hidden = math.add(hidden, this.bias_h);
    hidden = math.map(hidden, (x) => sigmoid(x));

    let outputs = math.multiply(this.w_ho, hidden);
    outputs = math.add(outputs, this.bias_o);
    outputs = math.map(outputs, (x) => sigmoid(x));

    // --------------------------------------------------------

    let output_errors = math.subtract(targets, outputs);
    let hidden_errors = math.multiply(math.transpose(this.w_ho), output_errors);

    let gradients = math.map(outputs, (x) => dSigmoid(x));
    gradients = math.dotMultiply(output_errors, gradients);
    gradients = math.multiply(this.learning_rate, gradients);
    let w_ho_deltas = math.multiply(gradients, math.transpose(hidden))

    var h_gradients = math.map(hidden, (x) => dSigmoid(x));
    h_gradients = math.dotMultiply(hidden_errors, h_gradients);
    h_gradients = math.multiply(this.learning_rate, h_gradients);
    let w_ih_deltas = math.multiply(h_gradients, math.transpose(inputs))

    this.w_ho = math.add(this.w_ho, w_ho_deltas);
    this.w_ih = math.add(this.w_ih, w_ih_deltas);

    this.bias_o = math.add(this.bias_o, gradients);
    this.bias_h = math.add(this.bias_h, h_gradients);
  }
}
