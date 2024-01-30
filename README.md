# Artificial Neural Network Demo

A basic Artificial Neural Network for practice, without any third-party dependencies to train or predict.

## Showcase

1. Compile the c matrix library

```
cd matrix/CMatrix && make && cd -
```

2. Run demos

```
python3 baseline.py
python3 demo_curve_fitting.py
python3 demo_MNIST.py
```

## Optional dependencies

- `pytest`

    For debugging.

- `matplotlib`

    For loss curve visualization.

    > `matplotlib` suffers too many security issues, so I removed it from the `Pipfile`. To install it, run `pipenv install --dev matplotlib`
