import jax.numpy as jnp
from jax import jit, random, value_and_grad, jvp, vjp
from jax.tree_util import tree_map, tree_flatten, tree_unflatten, tree_leaves, tree_structure
from jax.example_libraries import optimizers, stax
from jax.example_libraries.stax import Dense, LogSoftmax, Relu
from datasets import get_train_batches, get_test_batches


def train(key, epochs, batch_size, lr):

    def loss(params, batch):
        inputs, targets = batch
        preds = predict(params, inputs)
        return -jnp.mean(jnp.sum(preds * targets, axis=1))

    def accuracy(params, batch):
        inputs, targets = batch
        target_class = jnp.argmax(targets, axis=1)
        predicted_class = jnp.argmax(predict(params, inputs), axis=1)
        return jnp.mean(predicted_class == target_class)

    init_params, predict = stax.serial(
        Dense(1024), Relu,
        Dense(1024), Relu,
        Dense(10), LogSoftmax)

    _, init_rand_params = init_params(key, (-1, 28 * 28))

    opt_init, opt_update, get_params = optimizers.sgd(step_size=lr)
    opt_state = opt_init(init_rand_params)

    @jit
    def create_v(key, params):
        flatten_params, tree_def = tree_flatten(params)
        keys = random.split(key, len(flatten_params))
        v_flat = list(map(lambda k, x: random.normal(k, x.shape), keys, flatten_params))
        v = tree_unflatten(tree_def, v_flat)
        return v

    @jit
    def update(key, i, opt_state, batch):
        params = get_params(opt_state)
        loss_value, grads = value_and_grad(loss)(params, batch)
        
        # Calculate value and loss via vjp
        #loss_value, f_vjp = vjp(loss, params, batch)
        #grads, _ = f_vjp(jnp.ones_like(loss_value))

        # Calculate value and loss via jvp
        v = create_v(key, params)
        loss_value, drct_grad = jvp(loss, (params, batch),
                                          (v, tree_map(lambda x: jnp.zeros_like(x), batch)))
        grads = tree_map(lambda v_leaf: v_leaf*jnp.clip(drct_grad,-1,1), v)
        return opt_update(i, grads, opt_state), loss_value, drct_grad

    iter_cnt = 0
    for epoch_id in range(epochs):
        for batch_id, batch in enumerate(get_train_batches(batch_size)):
            key, subkey = random.split(key)
            opt_state, loss_value, drct_grad = update(key, batch_id, opt_state, batch)
            iter_cnt += 1
            if batch_id % 10 == 0:
                params = get_params(opt_state)
                wandb.log({'loss': loss_value, 'iter': iter_cnt, 'drct_grad':drct_grad})

        params = get_params(opt_state)
        train_acc = 100 * sum(accuracy(params, train_batch) for train_batch in get_train_batches(batch_size)) / len(get_train_batches(batch_size))
        test_acc = 100 * sum(accuracy(params, test_batch) for test_batch in get_test_batches(batch_size)) / len(get_test_batches(batch_size))
        wandb.log({'train_acc': train_acc, 'test_acc': test_acc})


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

    train(random.PRNGKey(args.seed),
          args.epochs,
          args.batch_size,
          args.lr)
