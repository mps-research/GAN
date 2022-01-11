# GAN
A PyTorch implementation of Vanilla GAN.

## Related Papers

Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, 
Sherjil Ozair, Aaron Courville and Yoshua Bengio:
Generative Adversarial Networks.
https://arxiv.org/pdf/1406.2661.pdf


## Training

```shell
% docker build -t gan .
% ./train.sh
```

## Checking Training Results

```shell
% ./run_tensorboard.sh
```