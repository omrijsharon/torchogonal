# torchogonal
## Promise: 
Make your deep neural network Dynamically Isometric. Say farewell to vanishing and exploding gradients 

## What does this repo acctually do? 
- It purly makes your model weights 'w' tensor orthogonal (w.t()@w = eye).
This preserves the norm of the input tensor. Since the eigen-value of an orthogonal matrix is with norm equal to 1- gradients can easily seep even through a very deep network.
- Using SVD (**NOT** using QR Decomposition with Gram-Schmidt, that causes many issues).
This ensures that even non-square matrices (MxN) will be orthogonalize, and also that the orthogonalization will be as close as possible to the current weights.

## How to use?
- import it:
```
from torchogonal import orthogonlize_module
```

- When initializing the model run:
```
orthogonlize_model(model)
```
**IMPORTANT!** This is an *in-place* function.
- After each optimizer step, run this line again to re-orthogonalize the model:
```
orthogonlize_model(model)
```

## Tests and requirements
- Tested on Python 3.8 and Pytorch 1.10.
The functions used in this repo are basic. This repo should easily work even older versions.

## Inspired by the papers:
- Dynamical Isometry and a Mean Field Theory of CNNs: How to Train 10,000-Layer Vanilla Convolutional Neural Networks (https://arxiv.org/pdf/1806.05393.pdf)
- projUNN: efficient method for training deep networks with unitary matrices (https://arxiv.org/pdf/2203.05483.pdf)
