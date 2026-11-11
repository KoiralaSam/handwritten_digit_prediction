# Handwritten Digit Prediction

A neural network built from scratch with NumPy (sigmoid neurons, mini-batch SGD and
backpropagation) that recognizes handwritten digits from MNIST. It follows
[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/), chapter 1.

## Project layout

| Path | What it is |
|---|---|
| `exercises/exercise_chap1.md` | Written solutions to the chapter 1 exercises |
| `src/mnist_loader.py` | Loads MNIST (Python 3 compatible) |
| `src/network.ipynb` | The `Network` class, training and evaluation |
| `src/model.npz` | Trained weights and biases (about 95% test accuracy) |
| `src/predict.py` | Predicts a digit from a photo |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt pillow
```

Download the dataset into `data/`:

```bash
mkdir -p data
curl -L -o data/mnist.pkl.gz \
  https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz
```

## Train

Open `src/network.ipynb` and run the cells. To save the trained model for `predict.py`:

```python
np.savez("model.npz", *net.weights, *net.biases, sizes=net.sizes)
```

## Predict from a photo

```bash
python src/predict.py src/2.jpg
python src/predict.py src/2.jpg --show   # also shows what the network sees
```

For best results, write one digit large with a dark pen on plain paper, in even light.
