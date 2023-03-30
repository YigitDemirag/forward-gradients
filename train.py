'''
JAX implementation of "Gradients without Backpropagation"
Baydin, A. G. et al. ArXiv (2022). DOI: 10.48550/arxiv.2202.08587.

Author: Yigit Demirag, MILA, 2023
'''
import math
import optax
import jax.numpy as jnp
from jax import jit, random, jvp, vjp
from jax.lax import switch
from functools import partial
from jax.tree_util import tree_map, tree_flatten, tree_unflatten
from jax.example_libraries import optimizers, stax
from jax.example_libraries.stax import Dense, LogSoftmax, Relu
from datasets import get_train_batches, get_test_batches

def train(key, epochs, batch_size, lr, num_layers, ad_type, drct_der_clip):

    def loss(params, batch):
        """Cross-entropy loss function"""
        inputs, targets = batch
        preds = predict(params, inputs)
        return -jnp.mean(jnp.sum(preds * targets, axis=1))

    def accuracy(params, batch):
        """Compute accuracy of predictions"""
        inputs, targets = batch
        target_class = jnp.argmax(targets, axis=1)
        predicted_class = jnp.argmax(predict(params, inputs), axis=1)
        return jnp.mean(predicted_class == target_class)

    # Initialize model
    layers = [Dense(1024), Relu] * num_layers
    layers.append(Dense(10))
    layers.append(LogSoftmax)
    init_params, predict = stax.serial(*layers)
    _, init_rand_params = init_params(key, (-1, 28 * 28))

    # Exponential decay lr scheduler
    exponential_decay_scheduler = optax.exponential_decay(
        init_value=lr,
        transition_steps=-10000,
        decay_rate=math.e,
        transition_begin=0,
        staircase=False
    )
    opt_init, opt_update, get_params = optimizers.sgd(
        step_size=exponential_decay_scheduler
    )
    opt_state = opt_init(init_rand_params)

    @jit
    def create_v(key, params):
        """Create random vector for JVP"""
        flatten_params, tree_def = tree_flatten(params)
        keys = random.split(key, len(flatten_params))
        v_flat = list(map(lambda k, x: random.normal(k, x.shape),
                          keys, flatten_params))
        v = tree_unflatten(tree_def, v_flat)
        return v

    def loss_and_grads_with_vjp(key, params, batch):
        """Compute loss and gradients using VJP"""
        loss_value, f_vjp = vjp(loss, params, batch)
        grads, _ = f_vjp(jnp.ones_like(loss_value))
        return loss_value, grads, 0.0

    def loss_and_grads_with_jvp(key, params, batch):
        """Compute loss and gradients using JVP"""
        v = create_v(key, params)
        _loss = partial(loss, batch=batch)
        loss_value, drct_drv = jvp(_loss, (params,), (v,))
        grads = tree_map(lambda v_leaf: v_leaf * jnp.clip(
            drct_drv, -drct_der_clip, drct_der_clip), v)
        return loss_value, grads, drct_drv

    @jit
    def update(key, i, opt_state, batch, ad_type):
        params = get_params(opt_state)
        loss_value, grads, drct_drv = switch(
            ad_type,
            [loss_and_grads_with_jvp, loss_and_grads_with_vjp],
            key, params, batch
        )
        return opt_update(i, grads, opt_state), loss_value, drct_drv

    iter_cnt = 0
    for _ in range(epochs):
        for batch_id, batch in enumerate(get_train_batches(batch_size)):
            key, _ = random.split(key)
            opt_state, loss_value, drct_drv = update(
                key, iter_cnt, opt_state, batch, ad_type
            )
            iter_cnt += 1
            if batch_id % 10 == 0:
                params = get_params(opt_state)
                wandb.log({
                    'loss': loss_value,
                    'iter': iter_cnt,
                    'drct_drv': drct_drv
                })

        params = get_params(opt_state)
        train_acc = 100 * sum(
            accuracy(params, train_batch)
            for train_batch in get_train_batches(batch_size)
        ) / len(get_train_batches(batch_size))
        test_acc = 100 * sum(
            accuracy(params, test_batch)
            for test_batch in get_test_batches(batch_size)
        ) / len(get_test_batches(batch_size))
        wandb.log({'train_acc': train_acc, 'test_acc': test_acc})


if __name__ == '__main__':
    import argparse
    import wandb

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--seed', type=int, default=23, help='random seed')
    parser.add_argument('--num_layers', type=int, default=2, help='number of hidden layers for MLP')
    parser.add_argument('--ad', type=str, default='jvp', help='jvp or vjp')
    parser.add_argument('--drct_der_clip', type=float, default=5.0, help='directional derivative clipping')

    args = parser.parse_args()
    wandb.init(project='forward', config=args)
    wandb.config.update(args)

    ad_type = 0 if args.ad == 'jvp' else 1
    print('Training with {} mode'.format('jvp' if ad_type == 0 else 'vjp'))
    train(random.PRNGKey(args.seed),
          args.epochs,
          args.batch_size,
          args.lr,
          args.num_layers,
          ad_type,
          args.drct_der_clip)