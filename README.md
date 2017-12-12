# tensor
Cythonized sparse tensor factorization that can handle N dimensions.

Uses the Tucker Composition with Stochastic Gradient Descent. Currently formulated to train on binary data and therefore to output probabilities — the loss function is binary crossentropy. Only l2 regularization is supported currently.

TO-DO:
- implement momentum and adaptive learning rate
- implement l1 regularization
- parallelize gradient updates for each input dimension
