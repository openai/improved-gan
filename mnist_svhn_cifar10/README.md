# MNIST/SVHN/CIFAR-10 experiments

This part of the code is built using Theano and Lasagne. Any recent version of these packages should work for running the code.

The experiments are run using the train*.py files. All experiments perform semi-supervised learning with a set of labeled examples and a set of unlabeled examples. There are two kinds of models: the "feature matching" models that achieve the best predictive performance, and the "minibatch discrimination" models that achieve the best image quality.

The provided train*.py files each train a single model for a single random labeled/unlabeled data split and a single random parameter initialization. To reproduce our results using ensembling / averaging over random seeds, you can run these files multiple times using different inputs for the "seed" and "seed_data" arguments, and then combine the results.

This code is still being developed and subject to change.

