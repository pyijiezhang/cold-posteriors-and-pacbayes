# CPE is a sign of underfitting 

This repository is the codebase to reproduce the experimental results in the paper CPE is a sign of underfitting. The experimental results can be divided into three parts.

## Bayesian linear regression on syntetic dataset using exact inference
Please check ./blm_regression_exact/blm_regression_eact.ipynb.

## Bayesian convolutional neural network on MNIST dataset using SGLD

Go to ./bnn_classification_sgld. Ensure Python modules under the `src` folder are importable as,
```
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
```
As an example, to run SGLD on MNIST, run:
```shell
python experiments/train_lik.py --likelihood_temp=1.0 \
                                --augment=True \
                                --seed=15 \
                                --dataset="mnist" \
                                --dirty_lik="lenet" \
                                --likelihood="softmax" \
                                --logits_temp=1.0 \
                                --prior-scale=1.0 \
                                --sgld-epochs=1000 \
                                --sgld-lr=1e-6 \
                                --momentum=0.99 \
                                --n-samples=50 \
                                --n_cycles=50 
```
## Bayesian convolutional neural network on MNIST dataset using MFVI

