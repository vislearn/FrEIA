# FrEIA

Framework for Easily Invertible Architectures

## Dependencies

- [`torch>=0.4.1`](https://pytorch.org/)
- [`numpy>=1.15.0`](http://www.numpy.org/)

The examples additionally require:
- [`matplotlib>=2.0.0`](https://matplotlib.org/)
- [`tqdm`](https://tqdm.github.io/)


## Installation
Install `pytorch>=0.4.1` as described on their [website](https://pytorch.org/),
then install FrEIA using pip:
```
pip install git+[URL]
```
### Development
Clone the git repo
```
git clone [URL]
```
then install it in "Development Mode", so that changes don't require a
reinstall
```
cd FrEIA
python setup.py develop
```

## Documentation
For documentation consider the docstrings in the code and check out the [usage
example](./examples/toy_8-modes/toy_8-modes.ipynb).

## Cite Us
```
@article{ardizzone2018analyzing,
  title={Analyzing Inverse Problems with Invertible Neural Networks},
  author={Ardizzone, Lynton and Kruse, Jakob and Wirkert, Sebastian and Rahner, Daniel and Pellegrini, Eric W and Klessen, Ralf S and Maier-Hein, Lena and Rother, Carsten and K{\"o}the, Ullrich},
  journal={arXiv preprint arXiv:1808.04730},
  year={2018}
}
```
