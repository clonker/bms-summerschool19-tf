# BMS Summerschool 2019 Tensorflow 2.0 basics

## Installation instructions for tensorflow tutorial

We will be using the upcoming version 2.0 of tensorflow, which can be installed in several ways (https://www.tensorflow.org/install).
Additionally we will need a couple of standard python packages: numpy, matplotlib, jupyter

One possibility to get all requirements installed is miniconda (https://docs.conda.io/en/latest/miniconda.html):
- Get miniconda and install it according to your OS (https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
- The command `conda` should be available on the commandline
- Create an environment for the tutorial. Run `conda create -n tflow python=3.6`
- Activate the environment. Run `conda activate tflow`
- Install needed packages using conda. Run `conda install numpy matplotlib jupyter seaborn tqdm scikit-learn`
- Install tensorflow via pip of the `tflow` environment. Run `pip install tensorflow==2.0.0-beta1`


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
