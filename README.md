This module is a prototype for complete the implementation of the xnor kernel on CUDA. With a tensorflow interface.

Heavily inspired by the [original implementation](https://github.com/MatthieuCourbariaux/BinaryNet) in Theano by Matthieu Courbariaux

Major feature:

1. Supports arbitrary size matrices.
2. Comes with Tensorflow Binding

### Speed up
Generated with the ipython notebook that is also in this repo. benchmark ran with CUDA 7.5, cuDNN v4 on Titan Black, Intel core i7-5820K

![Speed Up comparison with cublas](matrix_benchmark.png?raw=true "Comparison")

Note: This code probably not the most optimized code, since it's my first CUDA program. Suggestions are welcome
