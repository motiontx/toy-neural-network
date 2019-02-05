# Toy-Neural-Network

Little example of a neural network programmed from scratch using [math.js](http://mathjs.org/)

## How to use it 🚀

```javascript
let brain = new NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate);
```
*Example*
```javascript

let brain = new NeuralNetwork(2, 16, 3, 0.1);
let inputs = [in_1, in_2];

//Make a prediction
let prediction = brain.query(inputs);

//Train
let expectedOutput = [out_1, out_2, out_3];
brain.train(inputs, expectedOutput);
```

### Inspired by  ✒️

**Rashid, T. Make your own neural network.** (http://makeyourownneuralnetwork.blogspot.com/)

**Daniel Shiffman [The Coding Train] - Session 4 - Neural Networks - Intelligence and Learning.** (https://www.youtube.com/playlist?list=PLRqwX-V7Uu6Y7MdSCaIfsxc561QI0U0Tb)


## License 📄

This project is under the MIT License - see the file [LICENSE.md](LICENSE) for details.

---
⌨️ With ❤️ by [motiontx](https://github.com/motiontx) 😊
