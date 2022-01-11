# GAN
A PyTorch implementation of Vanilla GAN.

## Related Papers

Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, 
Sherjil Ozair, Aaron Courville and Yoshua Bengio:
Generative Adversarial Networks.
https://arxiv.org/pdf/1406.2661.pdf

## Training

1. Clone this repository and move to the directory.

```shell
% clone https://github.com/mps-research/GAN.git
% cd GAN
```

2. At the repository root directory, build "gan" docker image and run the image inside of a container.

```shell
% docker build -t gan .
% ./train.sh
```

## Checking Training Results

```shell
% ./run_tensorboard.sh
```
