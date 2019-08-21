# BMS Summerschool 2019 Tensorflow 2.0 basics

## Tutorial 1 contents

Tensorflow offers many ways to create models and train them:
- More high level methods conform to the Keras specification.
- Using lower level methods gives more control (write own model and training loop).

Combinations of these are quite interoperable

- Construct models of type `tf.keras.Model` via the `Sequential` API, the functional API, or writing your own subclass
- Choose loss function, optimizer, and metrics
- Supply data directly from numpy arrays or `tf.data.Dataset`
- Train via `model.fit()` or writing your own training loop, e.g.
```python
for i in range(n_epochs):
    for xs, ys in dataset:
        with tf.GradientTape() as tape:
            predictions = model(xs)
            loss = loss_object(ys, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```
- History of training metrics is automatically returned by `model.fit()`. In your own training loop you can do anything, e.g. fill a list of metrics yourself. (Tomorrow you'll learn about `tensorboard`)
- Saving a model means using checkpoints, which is done via 
    - `keras.ModelCheckpoint` and `model.fit(..., callbacks=...)`, or
    - write your own loop and use a `tf.train.Checkpoint` and `tf.train.CheckpointManager`
    
## Tutorial 2 contents
- `tf.function`: eager mode vs. graph mode
- visualization with Tensorboard
- writing your own variational autoencoder on MNIST
- writing your own timelagged variational autoencoder on swiss roll jump process
- using your own custom activation
- tracing graphs
- checkpointing with `CheckpointManager`
