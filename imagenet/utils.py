"""
Some codes from https://github.com/Newmu/dcgan_code
"""
import math
import os
import errno
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
import tensorflow as tf

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

index = 0
def get_image(image_path, image_size, is_crop=True, resize_w=64):
    global index
    out = transform(imread(image_path), image_size, is_crop, resize_w)
    return out


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path):
    img = scipy.misc.imread(path)
    if len(img.shape) == 0:
        raise ValueError(path + " got loaded as a dimensionless array!")
    return img.astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def center_crop(x, crop_h, crop_w=None, resize_w=64):

    h, w = x.shape[:2]
    crop_h = min(h, w)  # we changed this to override the original DCGAN-TensorFlow behavior
                        # Just use as much of the image as possible while keeping it square

    if crop_w is None:
        crop_w = crop_h
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    cropped_image = center_crop(image, npx, resize_w=resize_w)
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
            if bn != None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

                lines += """
                    var layer_%s = {
                        "layer_type": "fc",
                        "sy": 1, "sx": 1,
                        "out_sx": 1, "out_sy": 1,
                        "stride": 1, "pad": 0,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
                    var layer_%s = {
                        "layer_type": "deconv",
                        "sy": 5, "sx": 5,
                        "out_sx": %s, "out_sy": %s,
                        "stride": 2, "pad": 1,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
                             W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

def visualize(sess, dcgan, config, option):
  option = 0
  if option == 0:
    all_samples = []
    for i in range(484):
      print(i)
      samples = sess.run(dcgan.G)
      all_samples.append(samples)
    samples = np.concatenate(all_samples, 0)
    n = int(np.sqrt(samples.shape[0]))
    m = samples.shape[0] // n
    save_images(samples, [m, n], './' + config.sample_dir + '/test.png')#_%s.png' % strftime("%Y-%m-%d %H:%M:%S", gmtime()))
  elif option == 5:
    counter = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    while counter < 1005:
        print(counter)
        samples, fake = sess.run([dcgan.G, dcgan.d_loss_class])
        fake = np.argsort(fake)
        print(np.sum(samples))
        print(fake)
        for i in range(samples.shape[0]):
            name = "%s%d.png" % (chr(ord('a') + counter % 10), counter)
            img = np.expand_dims(samples[fake[i]], 0)
            if counter >= 1000:
                save_images(img, [1, 1], './' + config.sample_dir + '/turk/fake%d.png' % (counter - 1000))
            else:
                save_images(img, [1, 1], './' + config.sample_dir + '/turk/%s' % (name))
            counter += 1
  elif option == 1:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      save_images(samples, [8, 8], './' + options.sample_dir + '/test_arange_%s.png' % (idx))
  elif option == 2:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in [random.randint(0, 99) for _ in xrange(100)]:
      print(" [*] %d" % idx)

      if hasattr(dcgan, z):
        z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
        z_sample = np.tile(z, (config.batch_size, 1))
      #z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      if hasattr(dcgan, "sampler"):
          sampler = dcgan.sampler
      else:
          sampler = dcgan.G
      samples = sess.run(sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './' + config.sample_dir + '/test_gif_%s.gif' % (idx))
  elif option == 3:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './' + config.sample_dir + '/test_gif_%s.gif' % (idx))
  elif option == 4:
    image_set = []
    values = np.arange(0, 1, 1./config.batch_size)

    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

      image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
      make_gif(image_set[-1], './' + config.sample_dir + '/test_gif_%s.gif' % (idx))

    new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
        for idx in range(64) + range(63, -1, -1)]
    make_gif(new_image_set, './' + config.sample_dir + '/test_gif_merged.gif', duration=8)


def colorize(img):
    if img.ndim == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)
        img = np.concatenate([img, img, img], axis=2)
    if img.shape[2] == 4:
        img = img[:, :, 0:3]
    return img


def mkdir_p(path):
    # Copied from http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
