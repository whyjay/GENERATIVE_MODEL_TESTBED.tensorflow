# Generic GAN framework for experiments

## Key Files

#### GAN

* run_gan.py
* gan_models/config.py
* gan_models/model.py
* gan_models/generator.py
* gan_models/discriminator.py

#### VAE

* run_vae.py
* vae_models/config.py
* vae_models/model.py
* vae_models/encoder.py
* vae_models/decoder.py

## Getting Started with Base Model

* Choose dataset and set hyperparamters in `run_gan.py` or `run_vae.py`
```
MNIST, Affined MNIST, Fashion-MNIST and CIFAR10 are supported by default.
Our data loader automatically download dataset and offers batch sampling method `next_batch()`.
See '{DATASET_NAME}.py' scrips in project root folder for detail, or see `gan_models/train.py` and `vae_models/train.py` for usage example.
```

* Run it.

## Implementing Custom Network

#### GAN (Discriminator or Generator)
* Define your function in `gan_models/discriminator.py` or `gan_models/generator.py`
* Open `run_gan.py` and set `generator` or `discriminator` argument to the name of your new function.

#### VAE (Encoder or Decoder)
* Define your function in `vae_models/encoder.py` or `vae_models/decoder.py`
* Open `run_vae.py` and set `encoder` or `decoder` argument to the name of your new function.

## Implementing Whole-New Model

* Make a new model class that inherits `gan_models/model.py` or `vae_models/model.py` and place it in `gan_models` or `vae_models`.
* If you want, make new discriminator, generator, encoder or decoder as guided above.

## Evaluation (GAN only)
Supports MNIST, Affined MNIST, Fashion-MNIST and CIFAR10.
train classification model using

```
inception_score/train_{DATASET}_classifier.py
```

Evaluate inception score using

```
inception_score/eval_{DATASET}.py
```

Classification models are defined in

```
inception_score/model_{DATASET}.py
```
