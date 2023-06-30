#!/bin/env python3.8

"""
HW 1 Nicole Joseph
Based on example assignment by Professor Chris Curro
"""
import os
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from absl import app
from absl import flags
from tqdm import trange

from dataclasses import dataclass, field, InitVar

script_path = os.path.dirname(os.path.realpath(__file__))

# @dataclass
# class LinearModel:
# weights: np.ndarray
# bias: float
# sigma: np.ndarray
# mu: np.ndarray

# use @dataclass decorator to create a data class
# A class is a code template for creating objects. Objects have member variables and have behaviour associated with them.
@dataclass
class Data:
    # model: LinearModel
    rng: InitVar[
        np.random.Generator
    ]  # InitVar: Init only variable (this is a pseudo-field)
    num_features: int
    num_samples: int
    sigma: float
    # num_M: int
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)

    def __post_init__(self, rng):
        # self represents the instance of the class.
        # By using the “self” we can access the attributes and methods of the class in python.
        # It binds the attributes with the given arguments.
        self.index = np.arange(self.num_samples)
        self.x = rng.uniform(
            -4.0, 4.0, size=(self.num_samples, self.num_features)
        )  # num_samples = 50, num_features = 1
        # clean_y = tf.math.sin(2 * np.pi * self.x)
        clean_y = tf.math.sin(self.x)  # clean_y is clean sine wave
        self.y = rng.normal(loc=clean_y, scale=self.sigma)

    def get_batch(self, rng, batch_size):
        """
        Select random subset of examples for training batch
        """
        choices = rng.choice(self.index, size=batch_size)
        return self.x[choices], self.y[choices].flatten()


# we no longer need the weight and bias attributes in LinearModel class so I commented out the below section
# def compare_linear_models(a: LinearModel, b: LinearModel):
# for w_a, w_b in zip(a.weights, b.weights):
# print(f"{w_a:0.2f}, {w_b:0.2f}")
# print(f"{a.bias:0.2f}, {b.bias:0.2f}")

font = {
    # "family": "Adobe Caslon Pro",
    "size": 10,
}

matplotlib.style.use("classic")
matplotlib.rc("font", **font)

FLAGS = flags.FLAGS  # citation: https://abseil.io/docs/python/guides/flags
flags.DEFINE_integer("num_features", 1, "Number of features in record")
flags.DEFINE_integer("num_samples", 50, "Number of samples in dataset")
flags.DEFINE_integer("batch_size", 16, "Number of samples in batch")
flags.DEFINE_integer("num_iters", 300, "Number of SGD iterations")
flags.DEFINE_float("learning_rate", 0.1, "Learning rate / step size for SGD")
flags.DEFINE_integer("random_seed", 31415, "Random seed")
flags.DEFINE_float("sigma_noise", 0.1, "Standard deviation of noise random variable")
flags.DEFINE_bool("debug", False, "Set logging level to debug")
# add flag here for M
flags.DEFINE_integer("num_M", 4, "Number of Guassian basis functions")


class Model(tf.Module):
    def __init__(self, rng, num_features, num_M):

        # this does the estimation, add mu and sigma here because we are intitalizing what we need for estimation/y_hat
        # BROADCASTING
        self.num_M = num_M
        print(num_M)
        self.num_features = num_features
        self.w = tf.Variable(
            rng.normal(shape=[self.num_M, 1])
        )  # changed num_features to num_M
        self.b = tf.Variable(tf.zeros(shape=[1, 1]))  # constant bias term
        self.mu = tf.Variable(rng.normal(shape=[1, self.num_M]))
        self.sigma = tf.Variable(rng.normal(shape=[1, self.num_M]))
        # print(self.mu)
        # print(self.sigma)
        # print(self.num_M)

        # CHECK SHAPES
        # x ---> [num_samples, 1]
        # w ---> [M, 1]
        # b ---> [1, 1]
        # mu ---> [1, M]
        # sigma ---> [1, M]

    def __call__(self, x):
        # thought process:
        # replace x with function phi (guassian function)
        # matrix multiply phi and w
        # ensure it's in proper form w-transpose g(x) + b ---> citation: transpose notes from class

        phi = tf.transpose(tf.math.exp(-1 * ((x - self.mu) ** 2) / (self.sigma**2)))
        # print(self.sigma)
        return tf.squeeze(tf.transpose(self.w) @ phi + self.b)

    @property
    def model(self):
        # return LinearModel(
        # self.w.numpy().reshape([self.num_M]),
        # self.b.numpy().squeeze(),
        return self.mu.numpy().reshape([self.num_M]), self.sigma.numpy().reshape(
            [self.num_M]
        )


def main(a):
    logging.basicConfig()

    if FLAGS.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Safe np and tf PRNG
    seed_sequence = np.random.SeedSequence(FLAGS.random_seed)
    np_seed, tf_seed = seed_sequence.spawn(2)
    np_rng = np.random.default_rng(np_seed)
    tf_rng = tf.random.Generator.from_seed(tf_seed.entropy)

    # data_generating_model = LinearModel(weights=np_rng.integers(low=0, high=5, size=(FLAGS.num_features)), bias=2)
    # logging.debug(data_generating_model)
    # commented out data_generating_model lines because of AMAAN

    data = Data(
        # data_generating_model,
        np_rng,
        FLAGS.num_features,
        FLAGS.num_samples,
        FLAGS.sigma_noise,
    )

    model = Model(
        tf_rng,
        FLAGS.num_features,
        FLAGS.num_M,
    )
    logging.debug(model.model)

    optimizer = tf.optimizers.SGD(learning_rate=FLAGS.learning_rate)

    # GRADIENT TAPE --- for automatic differentiation --- tape context manager
    bar = trange(FLAGS.num_iters)
    for i in bar:
        with tf.GradientTape() as tape:
            x, y = data.get_batch(np_rng, FLAGS.batch_size)
            y_hat = model(x)
            loss = 0.5 * tf.reduce_mean(
                (y_hat - y) ** 2
            )  # objective function: mean squared error

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        bar.set_description(f"Loss @ {i} => {loss.numpy():0.6f}")
        bar.refresh()

    # logging.debug(model.model)
    # print out true values versus estimates
    # print("w,    w_hat")
    # compare_linear_models(data.model, model.model)

    if FLAGS.num_features > 1:
        # Only continue to plotting if x is a scalar
        exit(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), dpi=200)

    # sine wave
    ax1.set_title("Guassian Fit")
    ax1.set_xlabel("x")
    ax1.set_ylim(1.5 * np.amin(data.y), np.amax(data.y) * 1.5)
    h = ax1.set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    xs = np.linspace(-4, 4, 50)
    xs = xs[:, np.newaxis]
    ax1.plot(
        xs,
        np.squeeze(model(xs)),
        "r--",
        xs,
        tf.math.sin(xs),
        "b-",
        np.squeeze(data.x),
        data.y,
        "go",
    )

    # Basis Functions
    # citation: Lucia Rhode for showing me how to plot Guassian basis functions
    xs2 = np.linspace(-5, 5, 100)
    xs2 = xs2[:, np.newaxis]

    sigma, mu = model.model
    phi = tf.math.exp(-1 * ((xs2 - mu) ** 2) / (sigma**2))

    ax2.set_title("M Basis Functions")
    ax2.set_xlabel("x")
    ax2.set_ylim(0, np.amax(data.y))
    h = ax2.set_ylabel("y", labelpad=10)
    h.set_rotation(0)
    ax2.plot(xs2, np.squeeze(phi), "-")

    plt.tight_layout()
    plt.savefig(f"{script_path}/fit.pdf")


if __name__ == "__main__":
    app.run(main)
