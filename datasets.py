import tensorflow as tf
tf.config.set_visible_devices([], device_type='GPU')
import tensorflow_datasets as tfds
import jax.numpy as jnp

def get_batches(batch_size, split):
    # Get MNIST data as (image, label) pairs
    data_dir = '/tmp/tfds'
    ds = tfds.load(name='mnist', split=split, as_supervised=True, data_dir=data_dir)

    # Flatten images and convert labels to one-hot
    def preprocess(image, label):
        image = tf.reshape(image, [-1])
        image = tf.cast(image, tf.float32)
        label = tf.one_hot(label, depth=10)
        return image, label

    ds = ds.map(preprocess).batch(batch_size).prefetch(1)
    return tfds.as_numpy(ds)

def get_train_batches(batch_size):
    return get_batches(batch_size, 'train')

def get_test_batches(batch_size):
    return get_batches(batch_size, 'test')