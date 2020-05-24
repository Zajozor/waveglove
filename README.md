# WaveGlove

**This version of the repository is the stripped and cleaned version for ml2 - project 3**
The project was skimmed to be as slim as possible for the ml2 project, but still likely contains
some features which are unused in this subset of the original thesis version.

Repository structure:

- [models](models):
    - [utils](models/utils) various utility files containing constants and helper methods for splitting
      the datasets an creating the results.
    - [lightning-common.py](models/lightning_common.py)
    - [DeepConvLSTM model](models/m8_deepconvlstm.py) Implementation of DeepConvLSTM as proposed by
      Ordonez et al., reference implementation at https://github.com/sussexwearlab/DeepConvLSTM/blob/master/DeepConvLSTM.ipynb
    - [DeepConvLSTM with Self-attention model](models/m9_dclstm_att.py) - Implementation of DeepConvLSTM with
      Self-attention as proposed by Singh et al., reference implementation at 
      https://github.com/isukrit/encodingHumanActivity/blob/master/codes/model_proposed/model_with_self_attn.py
    - [run.py](models/run.py) - used to choose datasets and models to run (and also to actually run the code).
- [environment.yml](environment.yml) defines a conda environment used to run the application,
  see conda docs for instructions on how to use it.

## Installation

- Setup your Conda environment (recommended) or a python interpreter with the used packages is required.
  (mostly torch, torchvision, pytorch-lightning, pandas, scikit-learn, h5py, seaborn and matplotlib
  are required)

- Get some datasets. Datasets used for our evaluation are available at https://zenodo.org/record/3831958
  in the .h5 format. Place the corresponding h5 files in some directory.

## Configuration & Running

- in [models/utils/common.py](models/utils/common.py) you need to add your hostname as a key,
  and path as value to the `DATA_ROOTS` and `TENSORBOARD_ROOTS` dictionary. `DATA_ROOTS` path corresponds
  to the directory where you placed the h5 files and `TENSORBOARD_ROOTS` should be a directory, where
  pytorch tensorboard logs will be placed (if you won't use them, use something you will delete later).
  The `RUN_LOG_FILE` constant determines the log file to which the logs will be written (other than tensorboard).

- In [models/run.py](models/run.py) you can select the datasets (6 datasets, split to train/test in two different
  ways) and the models to run (m8_deepconvlstm refers to the work of Ordonez et al. and m9_dclstm_att refers
  to the work of Singh et al.).

- Use [models/run.py](models/run.py) to run the project. Two places can be inspected for results. First is
  the log file called `hparams-<model name>.log` usually create inside the current directory. It contains
  the accuracy/recall/f1 score for both the train and test sets.
  Secondly, tensorboard logs contain the corresponding confusion matrices.
  
Note 1: the `run.py` file expects the `PYTHONPATH` variable to contain the project root.
If this is not the case in your setup, you can use `waveglove-ml23$ PYTHONPATH=.:$PYTHONPATH python models/run.py`
(run in the project root).

Note 2: Running both models on all the datasets may require several tens of hours to complete.
However, the model using self-attention has significantly less parameters, therefore it runs a LOT
faster.
