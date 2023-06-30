#!/usr/bin/env python3

"""
HW 2 Nicole Joseph
    Setting the activation functions for the input layer and hidden layers as ReLU and SWISH, repectively,
worked better compared to setting ReLu for all layers. I didn't expect the combination of various activation functions
to produce smaller loss values but I guess it makes sense since different activation functions allow for different non-linearities 
(which I'm assuming worked better for solving this specific function). For the output layer I could have experimented with
softmax but opted with signmoid since I am more familiar with it.    
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
from absl import flags
from tqdm import trange
from dataclasses import dataclass, field, InitVar


@dataclass
class Data:
    rng: InitVar[np.random.Generator]
    num_samples: int
    # sigma: float
    x1: np.ndarray = field(init=False)
    x2: np.ndarray = field(init=False)
    y1: np.ndarray = field(init=False)
    y2: np.ndarray = field(init=False)
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    tclass: np.ndarray = field(init=False)

    def __post_init__(self, rng):
        # Data generation 
        self.index = np.arange(self.num_samples)
        r = rng.normal(loc=np.linspace(1, 14, self.num_samples // 2), scale=0.1)
        theta1 = rng.normal(loc=np.linspace(6, 2.5, self.num_samples // 2), scale=0.1)
        theta2 = rng.normal(loc=np.linspace(5, 1.75, self.num_samples // 2), scale=0.1)

        self.x1 = r * np.cos(np.pi * theta1).astype(np.float32)
        self.y1 = r * np.sin(np.pi * theta1).astype(np.float32)
        self.x2 = r * np.cos(np.pi * theta2).astype(np.float32)
        self.y2 = r * np.sin(np.pi * theta2).astype(np.float32)

        self.x = np.append(self.x1, self.x2).astype(np.float32)
        # print(tf.shape(self.x))
        self.y = np.append(self.y1, self.y2).astype(np.float32)
        # print(tf.shape(self.y))
        # in data vector:the classification label of 1 or 0
        self.tclass = np.append(
            (np.zeros(self.num_samples // 2)), (np.ones(self.num_samples // 2))
        ).astype(np.float32)
        # print(tf.shape(self.tclass))

    def get_batch(self, rng, batch_size):
        choices = rng.choice(self.index, size=batch_size)
        return self.x[choices], self.y[choices], self.tclass[choices]


font = {"family": "Adobe Caslon Pro", "size": 10}
matplotlib.style.use("classic")
matplotlib.rc("font", **font)

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_samples", 500, "Number of samples in dataset")
flags.DEFINE_integer("batch_size", 32, "Number of samples in batch")
flags.DEFINE_integer("num_iters", 3000, "Number of forward/backward pass iterations")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate/initial step size")
flags.DEFINE_integer("random_seed", 31415, "Random seed for reproducible results")
flags.DEFINE_float("sigma_noise", 0.5, "Standard deviation of noise random variable")
# flags.DEFINE_integer("sigma", 0.5, "Sigma")

# citation: https://www.tensorflow.org/guide/core/mlp_core#multilayer_perceptron_mlp_overview
# citation: To understand Xavier initialization better, I used this article https://www.deeplearning.ai/ai-notes/initialization/index.html
def xavier_init(shape):
    in_dim, out_dim = shape
    xavier_lim = tf.sqrt(6.0) / tf.sqrt(tf.cast(in_dim + out_dim, tf.float32))
    weight_vals = tf.random.uniform(
        shape=(in_dim, out_dim), minval=-xavier_lim, maxval=xavier_lim, seed=22
    )
    return weight_vals

# citation: https://www.tensorflow.org/guide/core/mlp_core#multilayer_perceptron_mlp_overview
class DenseLayer(tf.Module):
    def __init__(self, out_dim, weight_init=xavier_init, activation=tf.identity):
        self.out_dim = out_dim
        self.weight_init = weight_init
        self.activation = activation
        self.built = False

    # citation: https://www.tensorflow.org/guide/core/mlp_core#multilayer_perceptron_mlp_overview
    def __call__(self, x):
        if not self.built:
            self.in_dim = x.shape[1]
            self.w = tf.Variable(xavier_init(shape=(self.in_dim, self.out_dim)))
            self.b = tf.Variable(tf.zeros(shape=(self.out_dim,)))
            self.built = True
            print(x)
            # print(self.in_dim)
            # print(self.w)
            # print(self.b)

        z = tf.add(tf.matmul(x, self.w), self.b)
        return self.activation(z)

# citation: https://www.tensorflow.org/guide/core/mlp_core#multilayer_perceptron_mlp_overview
# class for the multilayer perceptron model that executes layers sequentially
class MLP(tf.Module):
    def __init__(self, layers):
        self.layers = layers

    @tf.function
    def __call__(self, x, preds=False):
        # for each layer initialized, make a layer
        for layer in self.layers:
            x = layer(x)
        return x


def main():
    # citation: from example assignment by Professor Chris Curro
    FLAGS(sys.argv)
    seed_sequence = np.random.SeedSequence(FLAGS.random_seed)
    np_seed, tf_seed = seed_sequence.spawn(2)
    np_rng = np.random.default_rng(np_seed)
    tf_rng = tf.random.Generator.from_seed(tf_seed.entropy)
    data = Data(np_rng, FLAGS.num_samples)

    # citation: https://www.tensorflow.org/guide/core/mlp_core#multilayer_perceptron_mlp_overview for this line
    model = MLP(
        [
            DenseLayer(256, activation=tf.nn.relu),
            DenseLayer(128, activation=tf.nn.swish),
            DenseLayer(1, activation=tf.math.sigmoid),
        ]
    )

    # Tensor("x:0", shape=(batch_size, 2), dtype=float32)
    # Tensor("Relu:0", shape=(batch_size, hiddenlayer1_size), dtype=float32)
    # Tensor("Relu_1:0", shape=(batch_seze, hiddenlayer2_size), dtype=float32)

    # optimizer = Adam instead of SGD this time
    optimizer = tf.optimizers.Adam(learning_rate=FLAGS.learning_rate)

    # citation: from example assignment by Chris Curro
    bar = trange(FLAGS.num_iters)
    for i in bar:
        with tf.GradientTape() as tape:
            x, y, tclass = data.get_batch(np_rng, FLAGS.batch_size)

            # citation: Joya Debi for helping make coordinates into tuple
            xycoord = (
                np.append(tf.squeeze(x), tf.squeeze(y)).reshape(2, FLAGS.batch_size).T
            )
            tclass = tf.squeeze(tclass)
            class_hat = tf.squeeze(model(xycoord, tf_rng))
            EPS = 1e-15
            temp_loss = tf.reduce_mean(
                (-tclass * tf.math.log(class_hat) + EPS)
                - ((1 - tclass) * tf.math.log(1 - class_hat) + EPS)
            )

            l2_reg_const = 0.001 * tf.reduce_mean(
                [tf.nn.l2_loss(v) for v in model.trainable_variables]
            )
            loss = temp_loss + l2_reg_const

            # print(loss)
            # Loss @ 2999 => 0.030478

        # https://stackoverflow.com/questions/67615051/implementing-binary-cross-entropy-loss-gives-different-answer-than-tensorflows
        # def BinaryCrossEntropy(y_true, y_pred):
        # y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
        # term_1 = y_true * np.log(y_pred + 1e-7)
        # return -np.mean(term_0+term_1, axis=0)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # bar.set_description(f"Loss @ {i} => {loss.numpy():0.6f}")
        bar.refresh()

    # PLOTTING
    axis = np.linspace(-15, 15, 1000).astype(np.float32)
    x_ax, y_ax = np.meshgrid(axis, axis)
    coords = np.vstack([x_ax.ravel(), y_ax.ravel()]).T
    y = model(coords, tf_rng)
    output = tf.squeeze(y)

    plt.figure()
    num_samples = FLAGS.num_samples
    plt.plot(data.x[: num_samples // 2], data.y[: num_samples // 2], "o", color="red")
    plt.plot(data.x[num_samples // 2 :], data.y[num_samples // 2 :], "o", color="blue")
    plt.contourf(
        x_ax,
        y_ax,
        output.numpy().reshape(1000, 1000),
        [0, 0.5, 1],
        colors=["paleturquoise", "lemonchiffon"],
    )
    plt.title("Spiral Data and 0.5% Prediction Boundary")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig("./fit.pdf")

if __name__ == "__main__":
    main()
