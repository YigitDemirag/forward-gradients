import tensorflow as tf
tf.config.set_visible_devices([], device_type='GPU')
import tensorflow_datasets as tfds
import jax.numpy as jnp
from os import path
from jax import grad, jit, random, custom_jvp, custom_vjp, jvp, vjp
from jax.tree_util import tree_map
from jax.example_libraries import optimizers, stax
from jax.example_libraries.stax import Dense, LogSoftmax, Relu

def train(epochs, batch_size, lr, seed):

    key = random.PRNGKey(seed)

    def get_train_batches(batch_size):
        # Get MNIST train data as (image, label) pairs
        data_dir = '/tmp/tfds'
        ds = tfds.load(name='mnist', split='train', as_supervised=True, data_dir=data_dir)

        # Flatten images and convert labels to one-hot
        def preprocess(image, label):
            image = tf.reshape(image, [-1])
            label = tf.one_hot(label, depth=10)
            return image, label
    
        ds = ds.map(preprocess).batch(batch_size).prefetch(1)        
        return tfds.as_numpy(ds)

    def loss(params, batch):
        # Cross-entropy loss
        inputs, targets = batch
        preds = predict(params, inputs)
        return -jnp.mean(jnp.sum(preds * targets, axis=1))

    def accuracy(params, batch):
        # Calculate accuracy
        inputs, targets = batch
        target_class = jnp.argmax(targets, axis=1)
        predicted_class = jnp.argmax(predict(params, inputs), axis=1)
        return jnp.mean(predicted_class == target_class)

    # Initialize model
    init_params, predict = stax.serial(
        Dense(1024), Relu,
        Dense(1024), Relu,
        Dense(10), LogSoftmax)
    
    _, init_params = init_params(key, (-1, 28*28))

    opt_init, opt_update, get_params = optimizers.sgd(step_size=lr)
    opt_state = opt_init(init_params)

    @jit                 
    def update(i, opt_state, batch):
        # Update model via SGD
        params = get_params(opt_state)
        return opt_update(i, grad(loss)(params, batch), opt_state)

    for epoch in range(epochs):
        for i, batch in enumerate(get_train_batches(batch_size)):
            opt_state = update(i, opt_state, batch)
            params = get_params(opt_state)
            if i % 100 == 0:
                train_acc = accuracy(params, batch)
                wandb.log({'train_acc': train_acc})            
            

if __name__ == '__main__':
    import argparse
    import wandb
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=23)

    args = parser.parse_args()
    wandb.init(project='forward', config=args)
    wandb.config.update(args)

    train(args.epochs, 
          args.batch_size, 
          args.lr,
          args.seed)
