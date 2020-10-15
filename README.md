# Almost Surely Stable Deep Dynamics
This repository contains the accompanying code for our NeurIPS 2020 paper [Almost Surely Stable Deep Dynamics](https://dais.chbe.ubc.ca/assets/preprints/2020C6_Lawrence_NeurIPS.pdf) by Nathan Lawrence, Philip Loewen, Michael Forbes, Johan Backstrom, Bhushan Gopaluni.

The focus of the paper is learning deep neural network based dynamic models with stability guarantees. Specifically, we consider stochastic discrete-time models. The method works by embedding a Lyapunov neural network into the dynamic model, thereby simulatenously learning the dynamics as well as a suitable Lyapunov function. We consider three cases of increasing difficulty:

  1. We propose a method for learning stable deterministic dynamics subject to a convex Lyapunov function;
  2. We generalize the first approach to the non-convex case through the use of an implicit layer designed to satisfy the stability criterion;
  3. We show how to extend these approaches to the stochastic setting by imposing stability on the mean and variance parameters of a mixture density network.

Here, we include code for the models described above along with instructions for training/testing and plotting.

Paper reference:
```
@article{lawrence2020almost,
  title = {Almost Surely Stable Deep Dynamics},
  author = {Lawrence, Nathan P and Loewen, Philip D and Forbes, Michael G and Backstrom, Johan U and Gopaluni, R Bhushan},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2020},
}
```

## Requirements
Install the necessary packages from requirements.txt via
```
pip install -r requirements.txt
```

## Training

## Testing and plotting

