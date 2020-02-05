# Exponential Wasserstein Kernels

**Abstract** In the context of kernel methods, the similarity between data points is encoded by the kernel function which is often defined thanks to the Euclidean distance, a common example being the squared exponential kernel. Recently, other distances relying on optimal transport theory - such as the Wasserstein distance between probability distributions - have shown their practical relevance for different machine learning techniques. In this paper, we study the use of exponential kernels defined thanks to the regularized Wasserstein distance and discuss their positive definiteness. More specifically, we define Wasserstein feature maps and illustrate their interest for supervised learning problems involving shapes and images. Empirically, Wasserstein squared exponential kernels are shown to yield smaller classification errors on small training sets of shapes, compared to analogous classifiers using Euclidean distances.

### Notes on the MATLAB implementation
Just add everything to the path. Then run `mnist.m`, `quickdraw.m` or `usps.m`. You may want to comment/uncomment the computation of the distance matrices to avoid running it again at every run. You may also want to play with the parameters e.g. to run it on CPU or GPU (faster, but needs CUDA) or change the gridsearch bounds and resolution for the hyper-parameters tuning.

*Due to its size, the `quickdraw.mat` data-file is currently not uploaded. Before we finish to update it, please send us an email if you want to get your hands on it.*

### Packages needed
* [LS-SVMlab](https://www.esat.kuleuven.be/sista/lssvmlab/);
* [Sinkhorn Scaling for Optimal Transport](http://marcocuturi.net/SI.html).
