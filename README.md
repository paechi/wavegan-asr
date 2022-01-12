# WaveGAN-pytorch
PyTorch implementation of [Synthesizing Audio with Generative Adversarial Networks(Chris Donahue, Feb 2018)](https://arxiv.org/abs/1802.04208) and [ASRWGAN project](https://cs230.stanford.edu/projects_spring_2018/reports/8289876.pdf).

Befor running, make sure you have the `sc09` dataset, and put that dataset under your current filepath.

## Quick Start:
1. Installation
```
sudo apt-get install libav-tools
```

2. Download dataset
* `sc09`: [sc09 raw WAV files](http://deepyeti.ucsd.edu/cdonahue/sc09.tar.gz), utterances of spoken english words '0'-'9'

3. Run

```
$ python train.py
```

## Contributions
This repo is based on [mazzzystar's](https://github.com/mazzzystar/WaveGAN-pytorch) and https://cs230.stanford.edu/projects_spring_2018/reports/8289876.pdf  implementation.
