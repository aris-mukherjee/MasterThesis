# MasterThesis
OOD-Robustness and Calibration Comparison U-Net vs. UNWT

## File Structure

This repo contains training & test files for prostate datasets (NCI = RUNMC, UCL, HK, BIDMC, BMC and USZ) and FeTS Brain tumor datasets (SD, TD1 and TD2). 

### General

* networks folder contains code related to network architectures (U-Net, U-Net Encoder, Vision Transformer, ...)
* data folder contains everything related to data preprocessing, both for the prostate and for brain tumor data (functions related to brain tumor data typically include FeTS in their name)
* bash scripts are included in the repo for all training and test files
* Logs capture indidividual patient Dice scores as well as average Dice scores over all patients

### Prostate

* train_prostate.py (parameters and main function) and trainer_prostate.py (dataloading & training loop)
* test_[dataset].py with dataset in [NCI, UCL, HK, BIDMC, BMC, USZ] files are the test files - evaluation and Dice score calculation happens in utils.py



### Brain Tumor (FeTS)

Organised in the same way as prostate files:
* train_FETS.py (parameters and main function) and trainer_FETS.py (dataloading & training loop)
* test_FETS_SD.py, test_FETS_TD1.py and test_FETS_TD2.py are the test files - evaluation and Dice score calculation happens in utils.py


## Prepare Data 

Preprocessed training and test data for prostate and brain tumor is stored as hdf5 files under /itet-stor/arismu/bmicdatasets_bmicnas01/Processed/Aris/FeTS
Note: FeTS training set is split up into 3 parts 

## Environment

Please prepare an environment with python=3.9, and then use the command "pip install -r packages.txt" for the dependencies.

## Train

Use the provided bash script or run: 

* Prostate:
`CUDA_VISIBLE_DEVICES=0 python -u train_prostate.py`

* FeTS:
`CUDA_VISIBLE_DEVICES=0 python -u train_FETS.py`

## Test
* Prostate: 
`python test_[dataset].py` where dataset is NCI, UCL, HK...
* FeTS:
`python test_FETS_[dataset].py` where dataset is SD, TD1 or TD2


