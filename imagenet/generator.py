from model import DCGAN
import tensorflow as tf
from ops import linear, deconv2d

class Generator(object):
    def __call__(self, is_ref):
        """
        Builds the graph propagating from z to x.
        On the first pass, should make variables.
        All variables with names beginning with "g_" will be used for the
        generator network.
        """
        dcgan = self.dcgan
        assert isinstance(dcgan, DCGAN)

        def make_z(shape, minval, maxval, name, dtype):
            assert dtype is tf.float32
            if is_ref:
                with tf.variable_scope(name) as scope:
                    z = tf.get_variable("z", shape,
                                initializer=tf.random_uniform_initializer(minval, maxval),
                                trainable=False)
                    if z.device != "/device:GPU:0":
                        print "z.device is " + str(z.device)
                        assert False
            else:
                z = tf.random_uniform(shape,
                                   minval=minval, maxval=maxval,
                                   name=name, dtype=tf.float32)
            return z


        z = make_z([dcgan.batch_size, dcgan.z_dim],
                                   minval=-1., maxval=1.,
                                   name='z', dtype=tf.float32)
        zs = [z]

        if hasattr(dcgan, 'generator_built'):
            tf.get_variable_scope().reuse_variables()
            make_vars = False
        else:
            make_vars = True


        def reuse_wrapper(packed, *args):
            """
            A wrapper that processes the output of TensorFlow calls differently
            based on whether we are reusing Variables or not.

            Parameters
            ----------
            packed: The output of the TensorFlow call
            args: List of names

            If make_vars is True, then `packed` will contain all the new Variables,
            and we need to assign them to dcgan.foo fields.
            If make_vars is False, then `packed` is just the output tensor, and we
            just return that.
            """
            if make_vars:
                assert len(packed) == len(args) + 1, len(packed)
                out = packed[0]
            else:
                out = packed
            return out

        assert not dcgan.y_dim
        # project `z` and reshape
        z_ = reuse_wrapper(linear(z, dcgan.gf_dim*8*4*4, 'g_h0_lin', with_w=make_vars), 'h0_w', 'h0_b')

        h0 = tf.reshape(z_, [-1, 4, 4, dcgan.gf_dim * 8])
        h0 = tf.nn.relu(dcgan.vbn(h0, "g_vbn_0"))
        h0z = make_z([dcgan.batch_size, 4, 4, dcgan.gf_dim],
                                   minval=-1., maxval=1.,
                                   name='h0z', dtype=tf.float32)
        zs.append(h0z)
        h0 = tf.concat(3, [h0, h0z])

        h1 = reuse_wrapper(deconv2d(h0,
            [dcgan.batch_size, 8, 8, dcgan.gf_dim*4], name='g_h1', with_w=make_vars),
            'h1_w', 'h1_b')
        h1 = tf.nn.relu(dcgan.vbn(h1, "g_vbn_1"))
        h1z = make_z([dcgan.batch_size, 8, 8, dcgan.gf_dim],
                                   minval=-1., maxval=1.,
                                   name='h1z', dtype=tf.float32)
        zs.append(h1z)
        h1 = tf.concat(3, [h1, h1z])


        h2 = reuse_wrapper(deconv2d(h1,
            [dcgan.batch_size, 16, 16, dcgan.gf_dim*2], name='g_h2', with_w=make_vars),
            'h2_w', 'h2_b')
        h2 = tf.nn.relu(dcgan.vbn(h2, "g_vbn_2"))
        half = dcgan.gf_dim // 2
        if half == 0:
            half = 1
        h2z = make_z([dcgan.batch_size, 16, 16, half],
                                   minval=-1., maxval=1.,
                                   name='h2z', dtype=tf.float32)
        zs.append(h2z)
        h2 = tf.concat(3, [h2, h2z])


        h3 = reuse_wrapper(deconv2d(h2,
            [dcgan.batch_size, 32, 32, dcgan.gf_dim*1], name='g_h3', with_w=make_vars),
            'h3_w', 'h3_b')
        if make_vars:
            h3_name = "h3_relu_first"
        else:
            h3_name = "h3_relu_reuse"
        h3 = tf.nn.relu(dcgan.vbn(h3, "g_vbn_3"), name=h3_name)
        print "h3 shape: ", h3.get_shape()

        quarter = dcgan.gf_dim // 4
        if quarter == 0:
            quarter = 1
        h3z = make_z([dcgan.batch_size, 32, 32, quarter],
                                   minval=-1., maxval=1.,
                                   name='h3z', dtype=tf.float32)
        zs.append(h3z)
        h3 = tf.concat(3, [h3, h3z])

        assert dcgan.image_shape[0] == 128

        h4 = reuse_wrapper(deconv2d(h3,
                [dcgan.batch_size, 64, 64, dcgan.gf_dim*1],
                name='g_h4', with_w=make_vars),
            'h4_w', 'h4_b')
        h4 = tf.nn.relu(dcgan.vbn(h4, "g_vbn_4"))
        print "h4 shape: ", h4.get_shape()

        eighth = dcgan.gf_dim // 8
        if eighth == 0:
            eighth = 1
        h4z = make_z([dcgan.batch_size, 64, 64, eighth],
                                   minval=-1., maxval=1.,
                                   name='h4z', dtype=tf.float32)
        zs.append(h4z)
        h4 = tf.concat(3, [h4, h4z])

        h5 = reuse_wrapper(deconv2d(h4,
                [dcgan.batch_size, 128, 128, dcgan.gf_dim * 1],
                name='g_h5', with_w=make_vars),
            'h5_w', 'h5_b')
        h5 = tf.nn.relu(dcgan.vbn(h5, "g_vbn_5"))
        print "h5 shape: ", h5.get_shape()

        sixteenth = dcgan.gf_dim // 16
        if sixteenth == 0:
            sixteenth = 1
        h5z = make_z([dcgan.batch_size, 128, 128, eighth],
                                   minval=-1., maxval=1.,
                                   name='h5z', dtype=tf.float32)
        zs.append(h5z)
        h5 = tf.concat(3, [h5, h5z])

        h6 = reuse_wrapper(deconv2d(h5,
                [dcgan.batch_size, 128, 128, 3],
                d_w = 1, d_h = 1,
                name='g_h6', with_w=make_vars,
                init_bias=dcgan.out_init_b,
                stddev=dcgan.out_stddev),
            'h6_w', 'h6_b')
        print 'h6 shape: ', h6.get_shape()

        out = tf.nn.tanh(h6)

        dcgan.generator_built = True
        return out, zs
