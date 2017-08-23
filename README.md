# Generic GAN framework for experiments

## Key Files

* run.py
* models/config.py
* models/model.py
* models/generator.py
* models/discriminator.py

## Getting Started with DCGAN

* Choose dataset and set hyperparamters in `run.py`
```
MNIST, Affined MNIST and CIFAR10 are supported by default.
Our data loader automatically download dataset and offers batch sampling method `next_batch()`.
See '{DATASET_NAME}.py' scrips in project root folder for detail, or see `models/train.py` for usage example.
```

* Run it.

## Implementing Discriminator or Generator

* Define your function in `models/discriminator.py` or `models/generator.py`
* Set `generator` or `discriminator` argument to the name of your new function.

## Implementing Whole-New Model

* Make a new model class that inherits `models/model.py` and place it in `models`.
* If you want, make new discriminator or generator too as guided above.

## Evaluation
Supports MNIST, Affined MNIST and CIFAR10.
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
