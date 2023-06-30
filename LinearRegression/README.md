# Linear Regression
Assignment:
Perform linear regression of a noisy sinewave using a set of gaussian basis functions with learned location and scale parameters. Model parameters are learned with stochastic
gradient descent. Use of automatic differentiation is required. Produce two plots. First, show the data-points, a noiseless sinewave, and the manifold produced by the regression model.
Second, show each of the M basis functions. Plots must be of suitable visual quality.

running instructions:
./tf.sh python hw1/main.py  --num_features 1 --num_samples 50 --batch_size 32 --num_iters 300 --random_seed 398729765279 --num_M 4
