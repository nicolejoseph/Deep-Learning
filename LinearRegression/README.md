# Linear Regression
Performed linear regression of a noisy sinewave using a set of gaussian basis functions with learned location and scale parameters. Model parameters were learned with stochastic
gradient descent, and automatic differentiation was required. Produced two plots. First plot shows the data-points, a noiseless sinewave, and the manifold produced by the regression model. The second plot shows each of the M basis functions. 

running instructions:
./tf.sh python hw1/main.py  --num_features 1 --num_samples 50 --batch_size 32 --num_iters 300 --random_seed 398729765279 --num_M 4

![Screenshot 2023-06-30 162248](https://github.com/nicolejoseph/Deep-Learning/assets/55464125/7e6b1085-52f4-4f8c-82fd-85588edc8a41)
