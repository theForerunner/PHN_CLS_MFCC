# Realtime Phoneme Classfication with MFCC

Code base for project "Build a human-in-the-loop psychophysics speech synthesis simulator for a brain-computer interface to restore speech"

## How to run

* all configurations are done by editing `config/default.yaml` file, and here are some configuration entries that you need to edit before running
  * root_path: this should be the absolute path to parent folder of `data` & `dict` folder
  * run_name: the checkpoint and report will be saved / loaded with name like `model_${run_name}.pt`

## Prerequisites

* pytorch
* torchaudio
* scikit-learn
* hydra
* tqdm
* loguru
