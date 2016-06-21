import os
import time
from glob import glob
import tensorflow as tf

from ops import *
from utils import *

filename = "/media/NAS_SHARED/imagenet/imagenet_train_128.tfrecords"

class DCGAN(object):
    def __init__(self, sess, image_size=108, is_crop=True,
                 batch_size=64, image_shape=[64, 64, 3],
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 d_label_smooth=.25,
                 generator_target_prob=1.,
                 checkpoint_dir=None, sample_dir='samples',
                 generator=None,
                 generator_func=None, train=None, train_func=None,
                 generator_cls = None,
                 discriminator_func=None,
                 encoder_func=None,
                 build_model=None,
                 build_model_func=None, config=None,
                 devices=None,
                 disable_vbn=False,
                 sample_size=64,
		 out_init_b=0.,
                 out_stddev=.15):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
        """
        self.disable_vbn = disable_vbn
        self.devices = devices
        self.d_label_smooth = d_label_smooth
	self.out_init_b = out_init_b
	self.out_stddev = out_stddev
        self.config = config
        self.generator_target_prob = generator_target_prob
        if generator is not None:
            generator.dcgan = self
        else:
            if generator_func is None:
                generator_func = default_generator
            if generator_cls is None:
                generator_cls = Generator
            generator = generator_cls(self, generator_func)
        self.generator = generator
        if discriminator_func is None:
            discriminator_func = default_discriminator
        self.discriminator = Discriminator(self, discriminator_func)
        if train is not None:
            self.train = train
            train.dcgan = self
        else:
            if train_func is None:
                train_func = default_train
            self.train = Train(self, train_func)
        if build_model is not None:
            assert build_model_func is None
            build_model.gan = self
            self.build_model = build_model
        else:
            if build_model_func is None:
                build_model_func = default_build_model
            self.build_model = BuildModel(self, build_model_func)
        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.image_shape = image_shape
        self.sample_dir = sample_dir

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = 3

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(batch_size, name='d_bn1')
        self.d_bn2 = batch_norm(batch_size, name='d_bn2')

        if not self.y_dim:
            self.d_bn3 = batch_norm(batch_size, name='d_bn3')

        self.g_bn0 = batch_norm(batch_size, name='g_bn0')
        self.g_bn1 = batch_norm(batch_size, name='g_bn1')
        self.g_bn2 = batch_norm(batch_size, name='g_bn2')

        if not self.y_dim:
            self.g_bn3 = batch_norm(batch_size, name='g_bn3')
        # Not used by all generators
        self.g_bn4 = batch_norm(batch_size, name='g_bn4')
        self.g_bn5 = batch_norm(batch_size, name='g_bn5')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def bn(self, tensor, name, batch_size=None):
        # the batch size argument is actually unused
        assert name.startswith('g_') or name.startswith('d_'), name
        if not hasattr(self, name):
            setattr(self, name, batch_norm(batch_size, name=name))
        bn = getattr(self, name)
        return bn(tensor)

    def bn2(self, tensor, name):
        assert name.startswith('g_') or name.startswith('d_'), name
        if not hasattr(self, name):
            setattr(self, name, batch_norm_second_half(name=name))
        bn = getattr(self, name)
        return bn(tensor)

    def bn1(self, tensor, name):
        assert name.startswith('g_') or name.startswith('d_'), name
        if not hasattr(self, name):
            setattr(self, name, batch_norm_first_half(name=name))
        bn = getattr(self, name)
        return bn(tensor)

    def bnx(self, tensor, name):
        assert name.startswith('g_') or name.startswith('d_'), name
        if not hasattr(self, name):
            setattr(self, name, batch_norm_cross(name=name))
        bn = getattr(self, name)
        return bn(tensor)

    def vbn(self, tensor, name, half=None):
        if self.disable_vbn:
            class Dummy(object):
                def __init__(self, tensor, ignored, half):
                    self.reference_output=tensor
                def __call__(self, x):
                    return x
            VBN_cls = Dummy
        else:
            VBN_cls = VBN
        if not hasattr(self, name):
            vbn = VBN_cls(tensor, name, half=half)
            setattr(self, name, vbn)
            return vbn.reference_output
        vbn = getattr(self, name)
        return vbn(tensor)

    def vbnl(self, tensor, name, half=None):
        if self.disable_vbn:
            class Dummy(object):
                def __init__(self, tensor, ignored, half):
                    self.reference_output=tensor
                def __call__(self, x):
                    return x
            VBN_cls = Dummy
        else:
            VBN_cls = VBNL
        if not hasattr(self, name):
            vbn = VBN_cls(tensor, name, half=half)
            setattr(self, name, vbn)
            return vbn.reference_output
        vbn = getattr(self, name)
        return vbn(tensor)

    def vbnlp(self, tensor, name, half=None):
        if self.disable_vbn:
            class Dummy(object):
                def __init__(self, tensor, ignored, half):
                    self.reference_output=tensor
                def __call__(self, x):
                    return x
            VBN_cls = Dummy
        else:
            VBN_cls = VBNLP
        if not hasattr(self, name):
            vbn = VBN_cls(tensor, name, half=half)
            setattr(self, name, vbn)
            return vbn.reference_output
        vbn = getattr(self, name)
        return vbn(tensor)

    def vbn1(self, tensor, name):
        return self.vbn(tensor, name, half=1)

    def vbn2(self, tensor, name):
        return self.vbn(tensor, name, half=2)


    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            print "Bad checkpoint: ", ckpt
            return False


class BuildModel(object):
    """
    A class that builds the generator forward prop when called.

    Parameters
    ----------
    dcgan: The DCGAN object to build within.
    func: The function to do it with.
    """
    def __init__(self, dcgan, func):
        self.dcgan = dcgan
        self.func = func

    def __call__(self):
        return self.func(self.dcgan)

class Generator(object):
    """
    A class that builds the generator forward prop when called.

    Parameters
    ----------
    dcgan: The DCGAN object to build the generator within.
    func: The function to do it with.
    """
    def __init__(self, dcgan, func):
        self.dcgan = dcgan
        self.func = func

    def __call__(self, z, y=None):
        return self.func(self.dcgan, z, y)


class Discriminator(object):
    """
    A class that builds the discriminator forward prop when called.

    Parameters
    ----------
    dcgan: The DCGAN object to build the discriminator within.
    func: The function to do it with.
    """
    def __init__(self, dcgan, func):
        self.dcgan = dcgan
        self.func = func

    def __call__(self, image, reuse=False, y=None, prefix=""):
        return self.func(self.dcgan, image, reuse, y, prefix)

class Train(object):
    """
    A class that runs the training loop when called.

    Parameters
    ----------
    dcgan: The DCGAN object to train.
    func: The function to do it with.
    """
    def __init__(self, dcgan, func):
        self.dcgan = dcgan
        self.func = func

    def __call__(self, config):
        return self.func(self.dcgan, config)

def get_vars(self):
    t_vars = tf.trainable_variables()
    self.d_vars = [var for var in t_vars if var.name.startswith('d_')]
    self.g_vars = [var for var in t_vars if var.name.startswith('g_')]
    for x in self.d_vars:
        assert x not in self.g_vars
    for x in self.g_vars:
        assert x not in self.d_vars
    for x in t_vars:
        assert x in  self.g_vars or x in self.d_vars, x.name
    self.all_vars = t_vars


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
            })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape(128 * 128 * 3)
    image = tf.reshape(image, [128, 128, 3])

    image = tf.cast(image, tf.float32) * (2. / 255) - 1.

    return image

def read_and_decode_with_labels(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label' : tf.FixedLenFeature([], tf.int64)
            })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape(128 * 128 * 3)
    image = tf.reshape(image, [128, 128, 3])

    image = tf.cast(image, tf.float32) * (2. / 255) - 1.

    label = tf.cast(features['label'], tf.int32)

    return image, label


def sigmoid_kl_with_logits(logits, targets):
    # broadcasts the same target value across the whole batch
    # this is implemented so awkwardly because tensorflow lacks an x log x op
    assert isinstance(targets, float)
    if targets in [0., 1.]:
        entropy = 0.
    else:
        entropy = - targets * np.log(targets) - (1. - targets) * np.log(1. - targets)
    return tf.nn.sigmoid_cross_entropy_with_logits(logits, tf.ones_like(logits) * targets) - entropy

class VBNL(object):
    """
    Virtual Batch Normalization, Log scale for the scale parameter
    """

    def __init__(self, x, name, epsilon=1e-5, half=None):
        """
        x is the reference batch
        """
        assert isinstance(epsilon, float)

        self.half = half
        shape = x.get_shape().as_list()
        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
            shape = x.get_shape().as_list()
        with tf.variable_scope(name) as scope:
            assert name.startswith("d_") or name.startswith("g_")
            self.epsilon = epsilon
            self.name = name
            if self.half is None:
                half = x
            elif self.half == 1:
                half = tf.slice(x, [0, 0, 0, 0],
                                      [shape[0] // 2, shape[1], shape[2], shape[3]])
            elif self.half == 2:
                half = tf.slice(x, [shape[0] // 2, 0, 0, 0],
                                      [shape[0] // 2, shape[1], shape[2], shape[3]])
            else:
                assert False
            self.mean = tf.reduce_mean(half, [0, 1, 2], keep_dims=True)
            self.mean_sq = tf.reduce_mean(tf.square(half), [0, 1, 2], keep_dims=True)
            self.batch_size = int(half.get_shape()[0])
            assert x is not None
            assert self.mean is not None
            assert self.mean_sq is not None
            out = self._normalize(x, self.mean, self.mean_sq, "reference")
            if needs_reshape:
                out = tf.reshape(out, orig_shape)
            self.reference_output = out

    def __call__(self, x):


        shape = x.get_shape().as_list()
        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
            shape = x.get_shape().as_list()
        with tf.variable_scope(self.name) as scope:
            new_coeff = 1. / (self.batch_size + 1.)
            old_coeff = 1. - new_coeff
            new_mean = tf.reduce_mean(x, [0, 1, 2], keep_dims=True)
            new_mean_sq = tf.reduce_mean(tf.square(x), [0, 1, 2], keep_dims=True)
            mean = new_coeff * new_mean + old_coeff * self.mean
            mean_sq = new_coeff * new_mean_sq + old_coeff * self.mean_sq
            out = self._normalize(x, mean, mean_sq, "live")
            if needs_reshape:
                out = tf.reshape(out, orig_shape)
            return out

    def _normalize(self, x, mean, mean_sq, message):
        # make sure this is called with a variable scope
        shape = x.get_shape().as_list()
        assert len(shape) == 4
        self.gamma_driver = tf.get_variable("gamma_driver", [shape[-1]],
                                initializer=tf.random_normal_initializer(0., 0.02))
        gamma = tf.exp(self.gamma_driver)
        gamma = tf.reshape(gamma, [1, 1, 1, -1])
        self.beta = tf.get_variable("beta", [shape[-1]],
                                initializer=tf.constant_initializer(0.))
        beta = tf.reshape(self.beta, [1, 1, 1, -1])
        assert self.epsilon is not None
        assert mean_sq is not None
        assert mean is not None
        std = tf.sqrt(self.epsilon + mean_sq - tf.square(mean))
        out = x - mean
        out = out / std
        # out = tf.Print(out, [tf.reduce_mean(out, [0, 1, 2]),
        #    tf.reduce_mean(tf.square(out - tf.reduce_mean(out, [0, 1, 2], keep_dims=True)), [0, 1, 2])],
        #    message, first_n=-1)
        out = out * gamma
        out = out + beta
        return out

class VBNLP(object):
    """
    Virtual Batch Normalization, Log scale for the scale parameter, per-Pixel normalization
    """

    def __init__(self, x, name, epsilon=1e-5, half=None):
        """
        x is the reference batch
        """
        assert isinstance(epsilon, float)

        self.half = half
        shape = x.get_shape().as_list()
        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
            shape = x.get_shape().as_list()
        with tf.variable_scope(name) as scope:
            assert name.startswith("d_") or name.startswith("g_")
            self.epsilon = epsilon
            self.name = name
            if self.half is None:
                half = x
            elif self.half == 1:
                half = tf.slice(x, [0, 0, 0, 0],
                                      [shape[0] // 2, shape[1], shape[2], shape[3]])
            elif self.half == 2:
                half = tf.slice(x, [shape[0] // 2, 0, 0, 0],
                                      [shape[0] // 2, shape[1], shape[2], shape[3]])
            else:
                assert False
            self.mean = tf.reduce_mean(half, [0], keep_dims=True)
            self.mean_sq = tf.reduce_mean(tf.square(half), [0], keep_dims=True)
            self.batch_size = int(half.get_shape()[0])
            assert x is not None
            assert self.mean is not None
            assert self.mean_sq is not None
            out = self._normalize(x, self.mean, self.mean_sq, "reference")
            if needs_reshape:
                out = tf.reshape(out, orig_shape)
            self.reference_output = out

    def __call__(self, x):

        shape = x.get_shape().as_list()
        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
            shape = x.get_shape().as_list()
        with tf.variable_scope(self.name) as scope:
            new_coeff = 1. / (self.batch_size + 1.)
            old_coeff = 1. - new_coeff
            new_mean = tf.reduce_mean(x, [0], keep_dims=True)
            new_mean_sq = tf.reduce_mean(tf.square(x), [0], keep_dims=True)
            mean = new_coeff * new_mean + old_coeff * self.mean
            mean_sq = new_coeff * new_mean_sq + old_coeff * self.mean_sq
            out = self._normalize(x, mean, mean_sq, "live")
            if needs_reshape:
                out = tf.reshape(out, orig_shape)
            return out

    def _normalize(self, x, mean, mean_sq, message):
        # make sure this is called with a variable scope
        shape = x.get_shape().as_list()
        assert len(shape) == 4
        self.gamma_driver = tf.get_variable("gamma_driver", shape[1:],
                                initializer=tf.random_normal_initializer(0., 0.02))
        gamma = tf.exp(self.gamma_driver)
        gamma = tf.expand_dims(gamma, 0)
        self.beta = tf.get_variable("beta", shape[1:],
                                initializer=tf.constant_initializer(0.))
        beta = tf.expand_dims(self.beta, 0)
        assert self.epsilon is not None
        assert mean_sq is not None
        assert mean is not None
        std = tf.sqrt(self.epsilon + mean_sq - tf.square(mean))
        out = x - mean
        out = out / std
        # out = tf.Print(out, [tf.reduce_mean(out, [0, 1, 2]),
        #    tf.reduce_mean(tf.square(out - tf.reduce_mean(out, [0, 1, 2], keep_dims=True)), [0, 1, 2])],
        #    message, first_n=-1)
        out = out * gamma
        out = out + beta
        return out

class VBN(object):
    """
    Virtual Batch Normalization
    """

    def __init__(self, x, name, epsilon=1e-5, half=None):
        """
        x is the reference batch
        """
        assert isinstance(epsilon, float)

        self.half = half
        shape = x.get_shape().as_list()
        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
            shape = x.get_shape().as_list()
        with tf.variable_scope(name) as scope:
            assert name.startswith("d_") or name.startswith("g_")
            self.epsilon = epsilon
            self.name = name
            if self.half is None:
                half = x
            elif self.half == 1:
                half = tf.slice(x, [0, 0, 0, 0],
                                      [shape[0] // 2, shape[1], shape[2], shape[3]])
            elif self.half == 2:
                half = tf.slice(x, [shape[0] // 2, 0, 0, 0],
                                      [shape[0] // 2, shape[1], shape[2], shape[3]])
            else:
                assert False
            self.mean = tf.reduce_mean(half, [0, 1, 2], keep_dims=True)
            self.mean_sq = tf.reduce_mean(tf.square(half), [0, 1, 2], keep_dims=True)
            self.batch_size = int(half.get_shape()[0])
            assert x is not None
            assert self.mean is not None
            assert self.mean_sq is not None
            out = self._normalize(x, self.mean, self.mean_sq, "reference")
            if needs_reshape:
                out = tf.reshape(out, orig_shape)
            self.reference_output = out

    def __call__(self, x):

        shape = x.get_shape().as_list()
        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
            shape = x.get_shape().as_list()
        with tf.variable_scope(self.name) as scope:
            new_coeff = 1. / (self.batch_size + 1.)
            old_coeff = 1. - new_coeff
            new_mean = tf.reduce_mean(x, [1, 2], keep_dims=True)
            new_mean_sq = tf.reduce_mean(tf.square(x), [0, 1, 2], keep_dims=True)
            mean = new_coeff * new_mean + old_coeff * self.mean
            mean_sq = new_coeff * new_mean_sq + old_coeff * self.mean_sq
            out = self._normalize(x, mean, mean_sq, "live")
            if needs_reshape:
                out = tf.reshape(out, orig_shape)
            return out

    def _normalize(self, x, mean, mean_sq, message):
        # make sure this is called with a variable scope
        shape = x.get_shape().as_list()
        assert len(shape) == 4
        self.gamma = tf.get_variable("gamma", [shape[-1]],
                                initializer=tf.random_normal_initializer(1., 0.02))
        gamma = tf.reshape(self.gamma, [1, 1, 1, -1])
        self.beta = tf.get_variable("beta", [shape[-1]],
                                initializer=tf.constant_initializer(0.))
        beta = tf.reshape(self.beta, [1, 1, 1, -1])
        assert self.epsilon is not None
        assert mean_sq is not None
        assert mean is not None
        std = tf.sqrt(self.epsilon + mean_sq - tf.square(mean))
        out = x - mean
        out = out / std
        # out = tf.Print(out, [tf.reduce_mean(out, [0, 1, 2]),
        #    tf.reduce_mean(tf.square(out - tf.reduce_mean(out, [0, 1, 2], keep_dims=True)), [0, 1, 2])],
        #    message, first_n=-1)
        out = out * gamma
        out = out + beta
        return out
