# torchogonal
Promise: \n
Make your deep neural network Dynamically Isometric. Say farewell to vanishing and exploding gradients \n

What does this repo acctually do? \n
It purly makes your model weights 'w' tensor orthogonal (w.t()@w = eye), using SVD (*NOT* using QR Decomposition with Gram-Schmidt, that causes many issues).
This preserves the norm of the input tensor. Since the eigen-value of an orthogonal matrix is with norm 1- gradients can easily seep even through a very deep network.



Tested on Python 3.8 and Pytorch 1.10.
The functions used are basic so this repo may work even older versions.

