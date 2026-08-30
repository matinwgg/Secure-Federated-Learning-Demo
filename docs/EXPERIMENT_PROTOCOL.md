# Experiment Protocol

Every experiment should record dataset/version, preprocessing, client count, participation rate, local epochs, optimizer, learning rate, clipping norm, noise multiplier, random seeds, number of rounds, hardware, software versions, and evaluation metrics.

## Metrics

Report utility alongside privacy and systems cost: accuracy/F1 or task-specific loss, convergence, communication volume, wall-clock time, memory, and privacy accounting (epsilon/delta where formally applicable).

## Baselines

Compare against centralized training and non-private federated training where applicable. Ablations should isolate clipping, noise, aggregation and participation effects.

## Reproducibility

Store experiment configuration separately from source code. Never imply a formal privacy guarantee without an explicit accountant and stated assumptions.
