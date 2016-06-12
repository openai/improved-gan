import sys
import argparse
import numpy as np
import theano as th
import theano.tensor as T
import lasagne
import lasagne.layers as ll
from lasagne.init import Normal
import time
import nn
import plotting
from theano.sandbox.rng_mrg import MRG_RandomStreams
import scipy.misc

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--seed_data', type=int, default=1)
parser.add_argument('--unlabeled_weight', type=float, default=1.)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--count', type=int, default=10)
args = parser.parse_args()
print(args)

# fixed random seeds
rng_data = np.random.RandomState(args.seed_data)
rng = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))

# specify generative model
print("Creating generator")
noise = theano_rng.uniform(size=(args.batch_size, 100))
gen_layers = [ll.InputLayer(shape=(args.batch_size, 100), input_var=noise)]
gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=500, nonlinearity=T.nnet.softplus), g=None))
gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=500, nonlinearity=T.nnet.softplus), g=None))
gen_layers.append(nn.weight_norm(ll.DenseLayer(gen_layers[-1], num_units=28**2, nonlinearity=T.nnet.sigmoid), train_g=True))
gen_dat = ll.get_output(gen_layers[-1], deterministic=False)

# specify supervised model
print("Creating supervised model")
disc_layers = [ll.InputLayer(shape=(None, 28**2))]
disc_layers.append(nn.GaussianNoiseLayer(disc_layers[-1], sigma=0.3))
disc_layers.append(nn.weight_norm(ll.DenseLayer(disc_layers[-1], num_units=1000, W=Normal(0.1), nonlinearity=nn.relu)))
disc_layers.append(nn.GaussianNoiseLayer(disc_layers[-1], sigma=0.5))
disc_layers.append(nn.weight_norm(ll.DenseLayer(disc_layers[-1], num_units=500, W=Normal(0.1), nonlinearity=nn.relu)))
disc_layers.append(nn.GaussianNoiseLayer(disc_layers[-1], sigma=0.5))
disc_layers.append(nn.weight_norm(ll.DenseLayer(disc_layers[-1], num_units=250, W=Normal(0.1), nonlinearity=nn.relu)))
disc_layers.append(nn.GaussianNoiseLayer(disc_layers[-1], sigma=0.5))
disc_layers.append(nn.weight_norm(ll.DenseLayer(disc_layers[-1], num_units=250, W=Normal(0.1), nonlinearity=nn.relu)))
disc_layers.append(nn.GaussianNoiseLayer(disc_layers[-1], sigma=0.5))
disc_layers.append(nn.weight_norm(ll.DenseLayer(disc_layers[-1], num_units=250, W=Normal(0.1), nonlinearity=nn.relu)))
disc_layers.append(nn.GaussianNoiseLayer(disc_layers[-1], sigma=0.5))
disc_layers.append(nn.weight_norm(ll.DenseLayer(disc_layers[-1], num_units=10, W=Normal(0.1), nonlinearity=None),train_g=True))

# costs
labels = T.ivector()
x_lab = T.matrix()
x_unl = T.matrix()

temp = ll.get_output(disc_layers[-1], x_lab, deterministic=False, init=True)
init_updates = [u for l in disc_layers for u in getattr(l,'init_updates',[])] # data based initialization

output_before_softmax_lab = ll.get_output(disc_layers[-1], x_lab, deterministic=False)
output_before_softmax_unl = ll.get_output(disc_layers[-1], x_unl, deterministic=False)
output_before_softmax_fake = ll.get_output(disc_layers[-1], gen_dat, deterministic=False)

# unsupervised loss
z_exp_lab = T.mean(nn.log_sum_exp(output_before_softmax_lab))
z_exp_unl = T.mean(nn.log_sum_exp(output_before_softmax_unl))
z_exp_fake = T.mean(nn.log_sum_exp(output_before_softmax_fake))
l_unl = nn.log_sum_exp(output_before_softmax_unl)
loss_unl = -0.5*T.mean(l_unl) + \
            0.5*T.mean(T.nnet.softplus(nn.log_sum_exp(output_before_softmax_unl))) + \
            0.5*T.mean(T.nnet.softplus(nn.log_sum_exp(output_before_softmax_fake)))

# supervised loss
l_lab = output_before_softmax_lab[T.arange(args.batch_size),labels]
loss_lab = -T.mean(l_lab) + T.mean(z_exp_lab)
train_err = T.mean(T.neq(T.argmax(output_before_softmax_lab,axis=1),labels))

# loss for training the generator
mom_gen = T.mean(ll.get_output(disc_layers[-3], gen_dat), axis=0)
mom_real = T.mean(ll.get_output(disc_layers[-3], x_unl), axis=0)
loss_gen = T.mean(T.square(mom_gen - mom_real))

# test error
output_before_softmax = ll.get_output(disc_layers[-1], x_lab, deterministic=True)
test_err = T.mean(T.neq(T.argmax(output_before_softmax,axis=1),labels))

# Theano functions for training and testing
lr = T.scalar()
disc_params = ll.get_all_params(disc_layers, trainable=True)
disc_param_updates = nn.adam_updates(disc_params, loss_lab + args.unlabeled_weight*loss_unl, lr=lr, mom1=0.5)
gen_params = ll.get_all_params(gen_layers[-1], trainable=True)
gen_param_updates = nn.adam_updates(gen_params, loss_gen, lr=lr, mom1=0.5)
init_param = th.function(inputs=[x_lab], outputs=None, updates=init_updates) # data based initialization
train_batch_disc = th.function(inputs=[x_lab,labels,x_unl,lr], outputs=[loss_lab, loss_unl, train_err], updates=disc_param_updates)
train_batch_gen = th.function(inputs=[x_unl,lr], outputs=[loss_gen], updates=gen_param_updates)
test_batch = th.function(inputs=[x_lab,labels], outputs=test_err)
samplefun = th.function(inputs=[], outputs=gen_dat)

print("Loading data")
# load MNIST data
data = np.load('mnist.npz')
trainx = np.concatenate([data['x_train'], data['x_valid']], axis=0).astype(th.config.floatX)
trainx_unl = trainx.copy()
trainx_unl2 = trainx.copy()
trainy = np.concatenate([data['y_train'], data['y_valid']]).astype(np.int32)
nr_batches_train = int(trainx.shape[0]/args.batch_size)
testx = data['x_test'].astype(th.config.floatX)
testy = data['y_test'].astype(np.int32)
nr_batches_test = int(testx.shape[0]/args.batch_size)

# select labeled data
inds = rng_data.permutation(trainx.shape[0])
trainx = trainx[inds]
trainy = trainy[inds]
txs = []
tys = []
for j in range(10):
    txs.append(trainx[trainy==j][:args.count])
    tys.append(trainy[trainy==j][:args.count])
txs = np.concatenate(txs, axis=0)
tys = np.concatenate(tys, axis=0)

init_param(trainx[:500]) # data based initialization

# //////////// perform training //////////////
start_lr = 0.001
for epoch in range(301):
    begin = time.time()
    lr = np.cast[th.config.floatX](start_lr * np.minimum(3. - epoch/100., 1.))

    # construct randomly permuted minibatches
    trainx = []
    trainy = []
    for t in range(trainx_unl.shape[0]/txs.shape[0]):
        inds = rng.permutation(txs.shape[0])
        trainx.append(txs[inds])
        trainy.append(tys[inds])
    trainx = np.concatenate(trainx, axis=0)
    trainy = np.concatenate(trainy, axis=0)
    trainx_unl = trainx_unl[rng.permutation(trainx_unl.shape[0])]
    trainx_unl2 = trainx_unl2[rng.permutation(trainx_unl2.shape[0])]

    # train
    loss_lab = 0.
    loss_unl = 0.
    train_err = 0.
    for t in range(nr_batches_train):
        ll, lu, te = train_batch_disc(trainx[t*args.batch_size:(t+1)*args.batch_size],trainy[t*args.batch_size:(t+1)*args.batch_size],
                                        trainx_unl[t*args.batch_size:(t+1)*args.batch_size],lr)
        loss_lab += ll
        loss_unl += lu
        train_err += te
        e = train_batch_gen(trainx_unl2[t*args.batch_size:(t+1)*args.batch_size],lr)

    loss_lab /= nr_batches_train
    loss_unl /= nr_batches_train
    train_err /= nr_batches_train

    # test
    test_err = 0.
    for t in range(nr_batches_test):
        test_err += test_batch(testx[t*args.batch_size:(t+1)*args.batch_size],testy[t*args.batch_size:(t+1)*args.batch_size])
    test_err /= nr_batches_test

    # report
    print("Iteration %d, time = %ds, loss_lab = %.4f, loss_unl = %.4f, train err = %.4f, test err = %.4f" % (epoch, time.time()-begin, loss_lab, loss_unl, train_err, test_err))
    sys.stdout.flush()

    # generate samples from the model
    sample_x = samplefun()
    img = plotting.tile_raster_images(X=sample_x, img_shape=(28, 28), tile_shape=(10, 10), tile_spacing=(1, 1))
    scipy.misc.imsave("mnist_sample_feature_match.png", img)

    # save params
    #np.savez('disc_params.npz', *[p.get_value() for p in disc_params])
    #np.savez('gen_params.npz', *[p.get_value() for p in gen_params])
