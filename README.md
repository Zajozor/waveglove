# WaveGlove

All the used datasets are available on Zenodo:

Králik, Matej. (2020). Curated list of HAR datasets (Version 1) [Data set]. Zenodo. http://doi.org/10.5281/zenodo.3831958

This repository contains the implementation of various Classical Machine Learning (CML) and Deep Learing (DL) methods for
Human Activity recognition (HAR) classification.
The code uses the SciPy and PyTorch frameworks.

---

The original thesis, containing the full description of the work, as well as the **complete set of results**,
[is available in this repository](multi-sensor-accelerometer-based-gesture-recognition.pdf).

---

Structure of the code in the repository:

- [experiments](experiments) - helper scripts for smaller experiments
- [models](models) - model definitions and running code
- [plots](plots) - some of the output data from experiments
- [sanity](sanity) - sanity checking intermediate models used for testing
- [environment.yml](environment.yml) defines a conda environment used to run the application

Results from the original work
were produced using the respective models,
hyperparameter combinations
and datasets present in [models/run.py](models/run.py).

Running the code requires downloading the datasets and adjusting the relevant
configuration variables locally.
