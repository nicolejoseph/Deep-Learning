# Adversarial Autoencoder
The goal of this project was to reproduce figures 2A and 2E from the paper “Adversarial Autoencoders” by Makhzani et al. Please refer to the [technical report](https://drive.google.com/file/d/1o2_ZZkRMbGfkUJhtw79ogLODh0BMazpc/view?usp=sharing) for a more comprehensive look at how this project was developed!
## Encoder-Decoder Structure
To implement the architecture, we created an adversarial autoencoder model using three basic model blocks: an encoder, decoder, 
and discriminator. The encoder takes in 28x28x1 images and compresses the input into the 2-D Gaussian latent space. The decoder 
transforms the output of the encoder, therefore reconstructing the original input images and simultaneously behaving as the generator. 
These two building blocks are the autoencoder component of the adversarial autoencoder. 
## Discriminator
The discriminator takes in real and fake input and learns to discriminate between them. The architecture involves
training and optimizing the discriminator to tell the difference between samples drawn from the prior distribution and samples
drawn from the encoder distribution (real vs fake input).
## Fitting to a Guassian Distribution
The functions we coded ensure that the encoder outputs to follow a known prior distribution (in this case, it's Guassian). Therefore, we get continuous data in the latent space; the code is evenly distributed over the prior distribution. To demonstrate this, we produced a scatter plot of the latent space distribution, shown below. We
also plotted the manifold of the adversarial autoencoder.



![figure2a_2](https://github.com/nicolejoseph/Adversarial-Autoencoder/assets/55464125/454f3cd2-d350-44d7-a540-984299605615)
