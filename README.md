# wavegan-asr
## WaveGAN for bandwidth extension problem
Adversarial Audio Synthesis paper introduced WaveGAN (https://arxiv.org/abs/1802.04208), which created high resolution audio form the noise. But, it is not a very suitable model for the bandwidth extension problem, which tries to generate high resolution audio given low resoltiton audio. As it appears, generating from the noise and generating from low resolution audio are two different problems. So, I found a project which solved similar problem called ASRWGAN, which uses Wasserstein GAN as a generator and ASRNET as a discriminator for their model https://cs230.stanford.edu/projects_spring_2018/posters/8285691.pdf. 

This code combines the best of two worlds: WaveGAN (WGAN) for the generator and ASRNET for the discriminator. It is trained on TIMIT and SC09 dataset, and report the loss results. 
